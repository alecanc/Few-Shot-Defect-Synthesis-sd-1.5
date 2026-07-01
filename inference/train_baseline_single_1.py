"""
train_baseline_single.py — Baseline: single-stage LoRA trained directly on defective images, without Stage 1.
It is used to isolate the value of separating the two concepts [V] and [D].


Output:
    checkpoints/baseline_single/{category}/{defect_type}/final/

Usage:
    python train/train_baseline_single.py \
        --config config.yaml \
        --category bottle \
        --defect_type broken_large \
        --n_shots 10
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL, DDPMScheduler,
    StableDiffusionPipeline, UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import json, random

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Baseline dataset

class SingleStageDataset(Dataset):
    """
    Dataset for single-stage baseline:
    Uses token [D] (xjy) but without token [V] (sks).
    Prompt: "a photo of a xjy broken_large defect on a bottle surface"
    Note: Includes product name in prompt (no special token), which is the key difference from Stage 2 of the full pipeline.
    """

    def __init__(self, split_path: str, defect_type: str, tokenizer,
                 token_D: str = "xjy", category: str = "bottle",
                 n_shots: int = None, image_size: int = 512,
                 augment: bool = True, seed: int = 42):

        with open(split_path) as f:
            split_data = json.load(f)

        all_paths = split_data["stage2"].get(defect_type, [])
        if n_shots is not None and n_shots < len(all_paths):
            rng = random.Random(seed)
            all_paths = rng.sample(all_paths, n_shots)

        self.image_paths = all_paths
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found for {category}/{defect_type}")

        tfms = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
        ]
        if augment:
            tfms.append(transforms.RandomHorizontalFlip())
        tfms += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.transform = transforms.Compose(tfms)

        # Prompt single-stage: product by name (not by special token)
        self.prompt = f"a photo of a {token_D} {defect_type} defect on a {category} surface"
        self.input_ids = tokenizer(
            self.prompt, padding="max_length", truncation=True,
            max_length=tokenizer.model_max_length, return_tensors="pt"
        ).input_ids[0]
        print(f"[SingleStage] {len(self.image_paths)} imgs | prompt: '{self.prompt}'")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return {"pixel_values": self.transform(img), "input_ids": self.input_ids}


# Training

def train(cfg: dict, category: str, defect_type: str, n_shots: int | None):
    splits_dir = Path(cfg["paths"]["splits_dir"])
    split_path = splits_dir / f"{category}_split.json"
    shots_tag  = f"shots{n_shots}" if n_shots is not None else "baseline"
    ckpt_dir   = (Path(cfg["paths"]["checkpoints"])
                  / "baseline_single" / category / defect_type)
    final_dir  = ckpt_dir / f"final_{shots_tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Use the same hyperparameters as Stage 2
    stage_cfg = cfg["stage2"]
    seed = cfg["seed"]

    accelerator = Accelerator(
        gradient_accumulation_steps=stage_cfg["gradient_accumulation_steps"],
        mixed_precision=stage_cfg["mixed_precision"],
        log_with="wandb" if cfg["wandb"]["project"] else None,
    )
    set_seed(seed)

    if accelerator.is_main_process:
        run_name = f"baseline_single_{category}_{defect_type}_{shots_tag}"
        accelerator.init_trackers(
            project_name=cfg["wandb"]["project"],
            config={"category": category, "defect_type": defect_type,
                    "n_shots": n_shots, "mode": "baseline_single", **stage_cfg},
            init_kwargs={"wandb": {"name": run_name}},
        )
        print(f"\n{'='*60}")
        print(f"  Baseline Single-Stage — {category}/{defect_type} [{shots_tag}]")
        print(f"{'='*60}\n")

    model_id = cfg["model_id"]
    tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    lora_config = LoraConfig(
        r=stage_cfg["lora_rank"], lora_alpha=stage_cfg["lora_alpha"],
        target_modules=stage_cfg["target_modules"], bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    if stage_cfg["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    dataset = SingleStageDataset(
        split_path=str(split_path), defect_type=defect_type,
        tokenizer=tokenizer, token_D=cfg["token_D"],
        category=category, n_shots=n_shots, seed=seed,
    )
    dataloader = DataLoader(dataset, batch_size=stage_cfg["train_batch_size"],
                            shuffle=True, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=stage_cfg["learning_rate"], betas=(0.9, 0.999),
        weight_decay=1e-2, eps=1e-8,
    )

    max_steps = stage_cfg["max_train_steps"]
    num_update_steps_per_epoch = math.ceil(len(dataloader) / stage_cfg["gradient_accumulation_steps"])
    num_epochs = math.ceil(max_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        stage_cfg["lr_scheduler"], optimizer=optimizer,
        num_warmup_steps=stage_cfg["lr_warmup_steps"],
        num_training_steps=max_steps,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler)

    weight_dtype = torch.float16 if stage_cfg["mixed_precision"] == "fp16" else torch.float32
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)

    global_step = 0
    progress_bar = tqdm(range(max_steps), desc="Baseline Single-Stage",
                        disable=not accelerator.is_main_process)

    for epoch in range(num_epochs):
        unet.train()
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz   = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    accelerator.log({"loss": loss.item(), "step": global_step}, step=global_step)
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

                if global_step % stage_cfg["save_steps"] == 0 and accelerator.is_main_process:
                    save_dir = ckpt_dir / f"step-{global_step:05d}_{shots_tag}"
                    accelerator.unwrap_model(unet).save_pretrained(save_dir)

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(unet).save_pretrained(final_dir)
        print(f"\nBaseline single-stage saved: {final_dir}")
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--category", required=True)
    parser.add_argument("--defect_type", required=True)
    parser.add_argument("--n_shots", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.category, args.defect_type, args.n_shots)


if __name__ == "__main__":
    main()
