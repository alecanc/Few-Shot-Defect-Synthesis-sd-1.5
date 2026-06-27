"""
train_stage2.py - Stage 2: LoRA fine-tuning for the [D] defect-appearance token

Trains an independent LoRA adapter on top of vanilla SD 1.5
The [D] token binds defect appearance from a fixed set of defective images of a single defect type.

Characteristics:
  - No prior preservation loss, plain MSE on the full batch every step
  - Reads from cfg["stage2"] hyperparameters (lr=5e-5, max_steps=400)
  - Takes --defect_type in addition to --category
  - Checkpoint path: checkpoints/stage2/{category}/{defect_type}/
  - Validation prompt uses token_D

Output:
    {checkpoints}/stage2/{category}/{defect_type}/final/   LoRA adapter
    {checkpoints}/stage2/{category}/{defect_type}/step-*/  Intermediate checkpoints
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset_stage2 import Stage2Dataset, collate_fn_stage2


## Helpers

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def vram_usage() -> str:
    used  = torch.cuda.memory_allocated() / 1e9
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
    """
    Generate validation images with the current adapter state and save to disk
    """
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


# ======= Training loop ===================================================

def train(cfg: dict, category: str, defect_type: str):
    """
    Stage 2 training loop for one (category, defect_type) pair

    Design choices and rationale:
      - Base model: vanilla SD 1.5
        Following Kumari et al. (Multi-Concept Customization, CVPR 2023),
        the two adapters are kept fully independent and composed only at
        inference time. Training Stage 2 on top of Stage 1 weights would
        entangle [D] with [V] representations.

      - Loss: plain MSE on the full batch (no prior preservation).
        Stage 2 has no language drift risk because we are not overwriting
        a prior class concept — [D] is a novel token with no prior meaning

      - validation_steps / num_validation_images: these keys are absent from
        the stage2 section of config.yaml (Stage 1 had them, Stage 2 does not).
        We fall back to hardcoded defaults: validate every 100 steps, 4 images.
        OPTION: add these keys to config.yaml under stage2 to make them
        configurable without touching the code.

      - Validation prompt: "a photo of a {token_D} defect on a surface"
        This matches the training prompt exactly, so we see what [D] alone
        generates — useful to verify the token is learning defect appearance.
        OPTION: at inference you would use the composed prompt with [V]+[D],
        but that requires loading both adapters simultaneously, which is
        better handled in a dedicated inference/evaluation script.
    """

    ## Paths 
    splits_dir = Path(cfg["paths"]["splits_dir"])
    split_path = splits_dir / f"{category}_split.json"
    ckpt_dir   = Path(cfg["paths"]["checkpoints"]) / "stage2" / category / defect_type
    final_dir  = ckpt_dir / "final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stage_cfg = cfg["stage2"]
    seed      = cfg["seed"]

    # Validation config: fall back to safe defaults if keys missing in config.yaml  TODO
    # Add validation_steps and num_validation_images under stage2 in config.yaml
    # to override these defaults.
    validation_steps      = stage_cfg.get("validation_steps",      100)
    num_validation_images = stage_cfg.get("num_validation_images", 4)

    ##  Accelerator 
    accelerator = Accelerator(
        gradient_accumulation_steps=stage_cfg["gradient_accumulation_steps"],
        mixed_precision=stage_cfg["mixed_precision"],
        log_with="wandb" if cfg["wandb"]["project"] else None,
    )
    set_seed(seed)

    ##  Wandb
    if accelerator.is_main_process:
        run_name = f"stage2_{category}_{defect_type}_rank{stage_cfg['lora_rank']}"
        accelerator.init_trackers(
            project_name=cfg["wandb"]["project"],
            config={
                "category":    category,
                "defect_type": defect_type,
                "stage":       2,
                "token_D":     cfg["token_D"],
                **stage_cfg,
            },
            init_kwargs={"wandb": {"name": run_name, "tags": cfg["wandb"]["tags"]}},
        )
        print(f"\n{'='*60}")
        print(f"  Stage 2 - {category.upper()} / {defect_type}")
        print(f"  Token: [{cfg['token_D']}]")
        print(f"  Checkpoint output: {ckpt_dir}")
        print(f"  VRAM initial: {vram_usage()}")
        print(f"{'='*60}\n")

    ##  Load SD 1.5
    model_id = cfg["model_id"]

    tokenizer     = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder  = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae           = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet          = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    ##  Freeze VAE and text encoder 
    # Only UNet LoRA parameters are trained
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    ## ==== LoRA adapter on UNet =========================
    # Same rank/alpha/target_modules as Stage 1 ( 4/32/attention layers)
    # OPTION: increase lora_rank to 8 in config.yaml if [D] underfits
    # (defect texture not captured). Keep alpha=32 so alpha/rank scaling stays 8
    lora_config = LoraConfig(
        r=stage_cfg["lora_rank"],
        lora_alpha=stage_cfg["lora_alpha"],             # how strongly the LoRA update influences the output relative to the frozen base
        target_modules=stage_cfg["target_modules"],     # which layers to inject into
        lora_dropout=0.0,
        bias="none",
    )

    # wraps the UNet with LoRA adapters and freezes the original UNet parameters
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    if stage_cfg["gradient_checkpointing"]:         # to reduce VRAM usage
        unet.enable_gradient_checkpointing()

    ##  Dataset and DataLoader
    # Single-type mode: train on one defect type at a time.
    # The dataset reads image paths from the split JSON produced by splits.py.
    dataset = Stage2Dataset(
        split_path=str(split_path),
        tokenizer=tokenizer,
        token_D=cfg["token_D"],
        defect_type=defect_type,
        augment=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=stage_cfg["train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn_stage2,
        num_workers=0,
        pin_memory=True,
    )

    ##  Optimizer 
    # AdamW with lr=5e-5 
    # We use a lower lr for Stage 2 because defect appearance is a finer-grained signal than
    # product identity, and the smaller dataset makes larger steps risky
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),       # pass only trainable LoRA parameters to the optimizer
        lr=stage_cfg["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    ##  LR Scheduler 
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

    # Accelerator preparation
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    weight_dtype = (
        torch.float16 if stage_cfg["mixed_precision"] == "fp16" else torch.float32
    )
    vae          = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)

    ## ======== Training loop ============================================
    global_step  = 0
    progress_bar = tqdm(
        range(max_steps),
        desc=f"Stage 2 [{category}/{defect_type}]",
        disable=not accelerator.is_main_process,
    )

    for epoch in range(num_epochs):         
        unet.train()    # unet in training mode

        for batch in dataloader:
            with accelerator.accumulate(unet):

                # Encode images into latents with VAE
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise
                noise     = torch.randn_like(latents)
                bsz       = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Forward diffusion, add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text conditioning
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # UNet noise prediction
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                ).sample

                # Loss -> plain MSE
                # Every sample in the batch is a defect image, no batch splitting needed
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            ##  Logging and checkpoints (once per gradient sync)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    # Wandb scalar logs
                    accelerator.log(
                        {
                            "loss": loss.item(),
                            "lr":   lr_scheduler.get_last_lr()[0],
                            "vram_gb": torch.cuda.memory_allocated() / 1e9,
                            "step": global_step,
                        },
                        step=global_step,
                    )
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
                        print(f"\n  → Checkpoint saved: {save_dir}")

                # Validation images to wandb
                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            unwrapped_unet = accelerator.unwrap_model(unet)
                            val_pipeline = StableDiffusionPipeline.from_pretrained(
                                model_id,
                                unet=unwrapped_unet,
                                torch_dtype=weight_dtype,
                                safety_checker=None,
                            ).to(accelerator.device)

                            # Prompt matches training prompt: [D] alone, no [V]
                            # shows what the defect token has learned in isolation
                            val_prompt = f"a photo of a {cfg['token_D']} defect on a surface"

                            val_images = generate_validation_images(
                                pipeline=val_pipeline,
                                prompt=val_prompt,
                                num_images=num_validation_images,
                                seed=seed,
                                output_dir=ckpt_dir,
                                step=global_step,
                            )

                            wandb.log(
                                {
                                    "validation": [
                                        wandb.Image(img, caption=f"{val_prompt} | step {global_step}")
                                        for img in val_images
                                    ],
                                    "step": global_step,
                                }
                            )

                            del val_pipeline
                        torch.cuda.empty_cache()

            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    ##  Save final LoRA adapter 
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(final_dir)
        print(f"\n{'='*60}")
        print(f"  Training complete")
        print(f"  Category:    {category}")
        print(f"  Defect type: {defect_type}")
        print(f"  Adapter saved in: {final_dir}")
        print(f"  VRAM final: {vram_usage()}")
        print(f"{'='*60}")

    accelerator.end_training()


# ======== Entry point ========================================


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: LoRA fine-tuning for the [D] defect-appearance token"
    )
    parser.add_argument("--config",      default="config.yaml",   help="Path to config.yaml")
    parser.add_argument("--category",    required=True,            help="MVTec category, e.g. bottle")
    parser.add_argument("--defect_type", required=True,            help="Defect type, e.g. broken_large")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, args.category, args.defect_type)


if __name__ == "__main__":
    main()