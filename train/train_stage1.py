"""
Output:
    {checkpoints}/stage1/{category}/final/   LoRA adapter 
    {checkpoints}/stage1/{category}/step-*/  Intermediate checkpoints (UNet LoRA weights)
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset_stage1 import (
    DreamBoothDataset,
    Stage1InstanceDataset,
    Stage1PriorDataset,
    collate_fn,
)




def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def vram_usage() -> str:
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{used:.1f}/{total:.1f} GB"


def generate_validation_images(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    num_images: int,
    seed: int,
    output_dir: Path,
    step: int,
) -> list:
    """Generate validation images and save them to disk."""
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator("cuda").manual_seed(seed)

    images = pipeline(
        prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=generator,
    ).images

    val_dir = output_dir / "validation"
    val_dir.mkdir(exist_ok=True)

    for i, img in enumerate(images):
        img.save(val_dir / f"step{step:05d}_img{i:02d}.png")

    return images


# Training loop 
def train(cfg: dict, category: str):

    # Setup paths
    splits_dir = Path(cfg["paths"]["splits_dir"])
    split_path = splits_dir / f"{category}_split.json"
    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / "stage1" / category
    prior_dir = ckpt_dir / "prior_images"
    final_dir = ckpt_dir / "final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stage_cfg = cfg["stage1"]
    seed = cfg["seed"]

    # Accelerator 
    accelerator = Accelerator(
        gradient_accumulation_steps=stage_cfg["gradient_accumulation_steps"],
        mixed_precision=stage_cfg["mixed_precision"],
        log_with="wandb" if cfg["wandb"]["project"] else None,
    )
    set_seed(seed)

    # Init wandb 
    if accelerator.is_main_process:
        run_name = f"stage1_{category}_rank{stage_cfg['lora_rank']}"
        accelerator.init_trackers(
            project_name=cfg["wandb"]["project"],
            config={
                "category": category,
                "stage": 1,
                "token_V": cfg["token_V"],
                **stage_cfg,
            },
            init_kwargs={"wandb": {"name": run_name, "tags": cfg["wandb"]["tags"]}},
        )
        print(f"\n{'='*60}")
        print(f"  Stage 1 — {category.upper()} — token: [{cfg['token_V']}]")
        print(f"  Checkpoint output: {ckpt_dir}")
        print(f"  VRAM iniziale: {vram_usage()}")
        print(f"{'='*60}\n")

    # Load SD 1.5 
    model_id = cfg["model_id"]

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze VAE and text encoder 
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Add adapter LoRA to UNet 
    lora_config = LoraConfig(
        r=stage_cfg["lora_rank"],
        lora_alpha=stage_cfg["lora_alpha"],
        target_modules=stage_cfg["target_modules"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Gradient checkpointing
    if stage_cfg["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    # Dataset e DataLoader 
    instance_dataset = Stage1InstanceDataset(
        split_path=str(split_path),
        tokenizer=tokenizer,
        token_V=cfg["token_V"],
        category=category,
    )

    prior_dataset = None
    if stage_cfg["prior_preservation"] and prior_dir.exists():
        prior_dataset = Stage1PriorDataset(
            prior_images_dir=str(prior_dir),
            tokenizer=tokenizer,
            category=category,
        )
    elif stage_cfg["prior_preservation"]:
        print(
            f"WARN: prior_preservation=True but {prior_dir} not exists.\n"
            "      Execute: python train/generate_prior_images.py"
        )

    dreambooth_dataset = DreamBoothDataset(instance_dataset, prior_dataset)
    dataloader = DataLoader(
        dreambooth_dataset,
        batch_size=stage_cfg["train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer 
    # AdamW with weight decay standard for LoRA
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=stage_cfg["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # LR Scheduler 
    max_steps = stage_cfg["max_train_steps"]
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / stage_cfg["gradient_accumulation_steps"]
    )
    num_epochs = math.ceil(max_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        stage_cfg["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=stage_cfg["lr_warmup_steps"],
        num_training_steps=max_steps,
    )


    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # VAE and text_encoder on GPU
    weight_dtype = torch.float16 if stage_cfg["mixed_precision"] == "fp16" else torch.float32
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Training loop 
    global_step = 0
    progress_bar = tqdm(
        range(max_steps),
        desc="Training Stage 1",
        disable=not accelerator.is_main_process,
    )

    for epoch in range(num_epochs):
        unet.train()

        for batch in dataloader:
            with accelerator.accumulate(unet):

                # Encode images to latents
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise 
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to latents according to noise scheduler
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise with UNet (denoising)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                ).sample

                # Calculate Loss
                target = noise  

                if stage_cfg["prior_preservation"] and prior_dataset is not None:
                    # Split batch: first half = instance, second half = prior
                    half = model_pred.shape[0] // 2
                    model_pred_instance, model_pred_prior = (
                        model_pred[:half], model_pred[half:]
                    )
                    target_instance, target_prior = target[:half], target[half:]

                    # Loss instance
                    loss_instance = F.mse_loss(
                        model_pred_instance.float(),
                        target_instance.float(),
                        reduction="mean",
                    )
                    # Loss prior
                    loss_prior = F.mse_loss(
                        model_pred_prior.float(),
                        target_prior.float(),
                        reduction="mean",
                    )
                    loss = loss_instance + stage_cfg["prior_loss_weight"] * loss_prior
                else:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                    loss_instance = loss
                    loss_prior = torch.tensor(0.0)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logs and checkpoints 
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Logs on wandb
                if accelerator.is_main_process:
                    log_dict = {
                        "loss/total": loss.item(),
                        "loss/instance": loss_instance.item(),
                        "loss/prior": loss_prior.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "vram_gb": torch.cuda.memory_allocated() / 1e9,
                        "step": global_step,
                    }
                    accelerator.log(log_dict, step=global_step)
                    progress_bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        vram=vram_usage(),
                    )

                # Intermediate checkpoint
                if global_step % stage_cfg["save_steps"] == 0:
                    if accelerator.is_main_process:
                        save_dir = ckpt_dir / f"step-{global_step:05d}"
                        unwrapped = accelerator.unwrap_model(unet)
                        unwrapped.save_pretrained(save_dir)
                        print(f"\n  → Checkpoint salvato: {save_dir}")

                # Visual validation
                if global_step % stage_cfg["validation_steps"] == 0:
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            unwrapped_unet = accelerator.unwrap_model(unet)
                            if hasattr(unwrapped_unet, "basel_model"):
                                unwrapped_unet = unwrapped_unet.basemodel.eval()
                            val_pipeline = StableDiffusionPipeline.from_pretrained(
                                model_id,
                                unet=unwrapped_unet,
                                torch_dtype=weight_dtype,
                                safety_checker=None,
                            ).to(accelerator.device)

                            val_prompt = f"a photo of a {cfg['token_V']} {category}"
                            val_images = generate_validation_images(
                                pipeline=val_pipeline,
                                prompt=val_prompt,
                                num_images=stage_cfg["num_validation_images"],
                                seed=seed,
                                output_dir=ckpt_dir,
                                step=global_step,
                            )

                            # Send images to wandb 
                            wandb.log({
                                "validation": [
                                    wandb.Image(img, caption=val_prompt)
                                    for img in val_images
                                ],
                                "step": global_step,
                            })

                            del val_pipeline
                        torch.cuda.empty_cache()

            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    # Save final LoRA adapter
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(final_dir)
        print(f"\n{'='*60}")
        print(f"  Training completed")
        print(f"  Adapter LoRA saved in: {final_dir}")
        print(f"  VRAM final: {vram_usage()}")
        print(f"{'='*60}")

    accelerator.end_training()



def main():
    parser = argparse.ArgumentParser(description="Stage 1: DreamBooth+LoRA for [V]")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--category", required=True, help="Es: bottle")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, args.category)


if __name__ == "__main__":
    main()
