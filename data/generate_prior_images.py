"""
generate_prior_images.py — Generate prior images for prior preservation loss.

Using the base SD 1.5 model (no LoRA), generate generic images of the category


Usage:
    python train/generate_prior_images.py --config config.yaml --category bottle
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_prior_images(
    model_id: str,
    category: str,
    output_dir: Path,
    num_images: int = 100,
    batch_size: int = 4,
    seed: int = 42,
):
    """
        Generate prior images for a given category using the base SD model.
    Args:
        model_id:   HuggingFace model ID (es. "runwayml/stable-diffusion-v1-5")
        category:   category MVTec (es. "bottle")
        output_dir: where to save the images
        num_images: how many images to generate (100 is the default DreamBooth)
        batch_size: images per inference (4 on P100 with fp16 is safe)
        seed:       seed for reproducibility
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(output_dir.glob("*.png")))
    if existing >= num_images:
        print(f"[Prior] {existing} images already present in {output_dir}. Skip.")
        return

    remaining = num_images - existing
    print(f"[Prior] Generating {remaining} prior images for '{category}'...")
    print(f"[Prior] Output: {output_dir}")

    # Load pipeline base model (no LoRA)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    prompt = f"a photo of a {category}"
    generator = torch.Generator("cuda").manual_seed(seed)

    n_batches = (remaining + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="Generating prior"):
        n_this_batch = min(batch_size, remaining - batch_idx * batch_size)

        images = pipe(
            prompt,
            num_images_per_prompt=n_this_batch,
            num_inference_steps=25,      
            guidance_scale=7.0,
            generator=generator,
        ).images

        for img in images:
            img_idx = existing + batch_idx * batch_size + images.index(img)
            img.save(output_dir / f"prior_{img_idx:04d}.png")

    print(f"[Prior] Completed. {num_images} images in {output_dir}")

    del pipe
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--category", required=True)
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Override num_class_images from config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_images = args.num_images or cfg["stage1"]["num_class_images"]

    output_dir = (
        Path(cfg["paths"]["checkpoints"])
        / "stage1"
        / args.category
        / "prior_images"
    )

    generate_prior_images(
        model_id=cfg["model_id"],
        category=args.category,
        output_dir=output_dir,
        num_images=num_images,
        seed=cfg["seed"],
    )


if __name__ == "__main__":
    main()
