"""
inference.py — Composed [V]+[D] inference for defect image synthesis.

Loads the Stage 1 adapter ([V], product identity) and the Stage 2 adapter
([D], defect appearance) simultaneously on SD 1.5 and generates images with
the composed prompt:

    "a photo of a {token_V} {category} with a {token_D} defect on the surface"

The two LoRA adapters are loaded as named PEFT adapters and composed via
linear delta addition with independent scaling weights (adapter_weight_V and
adapter_weight_D from config.yaml). This follows the multi-concept
composition approach of Kumari et al. (Multi-Concept Customization, CVPR 2023).

Usage:
    python inference/inference.py \
        --config      config.yaml \
        --category    bottle \
        --defect_type broken_large \
        --stage1_dir  /kaggle/working/checkpoints/stage1/bottle/final \
        --stage2_dir  /kaggle/working/checkpoints/stage2/bottle/broken_large/final \
        --weight_v    1.0 \
        --weight_d    0.8

    --stage1_dir and --stage2_dir override config.yaml paths entirely.
    --weight_v and --weight_d override config.yaml inference weights.
    All four are optional — if omitted, config.yaml values are used.

Output:
    {generated}/{category}_{defect_type}_composed_{n}images.png
    A 4-column grid of num_images_per_prompt generated images.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Adapter loading
# ─────────────────────────────────────────────────────────────────────────────

def load_composed_pipeline(
    model_id: str,
    stage1_dir: Path,
    stage2_dir: Path,
    weight_V: float,
    weight_D: float,
    device: str = "cuda",
) -> StableDiffusionPipeline:
    """
    Load SD 1.5 and attach both LoRA adapters as named PEFT adapters.

    Adapter composition strategy:
        The UNet weight update is the linear sum of both adapter deltas,
        each scaled by its own weight:
            delta_W = weight_V * delta_W_V + weight_D * delta_W_D

        weight_V=1.0, weight_D=0.8 follows the config defaults, which
        slightly down-weights the defect signal relative to product identity
        to prevent the defect from overriding the product geometry.

    The VAE is kept in float32 for decode stability (diffusers 0.29 behaviour).
    The UNet and text encoder run in float16.

    Args:
        model_id:   HuggingFace model ID for SD 1.5
        stage1_dir: path to Stage 1 adapter final/ directory
        stage2_dir: path to Stage 2 adapter final/ directory
        weight_V:   scaling factor for the [V] adapter
        weight_D:   scaling factor for the [D] adapter
        device:     torch device string

    Returns:
        StableDiffusionPipeline with both adapters active and weighted
    """
    print(f"Loading base model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)

    # Load Stage 1 adapter as "identity" — product appearance [V]
    print(f"Loading Stage 1 adapter ([V]): {stage1_dir}")
    pipe.unet = PeftModel.from_pretrained(
        pipe.unet,
        str(stage1_dir),
        adapter_name="identity",
    )

    # Load Stage 2 adapter as "defect" — defect appearance [D]
    # load_adapter attaches a second named adapter without replacing the first
    print(f"Loading Stage 2 adapter ([D]): {stage2_dir}")
    pipe.unet.load_adapter(str(stage2_dir), adapter_name="defect")

    # Activate both adapters simultaneously with independent scaling weights.
    # PEFT's set_adapter with a list enables multi-adapter mode.
    pipe.unet.set_adapter(["identity", "defect"])

    # Apply the scaling weights.
    # set_scale multiplies each adapter's LoRA output by the given factor
    # before the linear combination with the other adapter.
    for name, weight in [("identity", weight_V), ("defect", weight_D)]:
        for module in pipe.unet.modules():
            if hasattr(module, "set_scale"):
                module.set_scale(name, weight)

    # Cast UNet to fp16 explicitly after PEFT operations
    pipe.unet = pipe.unet.to(dtype=torch.float16)

    # VAE stays in float32 — diffusers 0.29 upcasts latents before decode
    pipe.vae = pipe.vae.to(dtype=torch.float32)

    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Image grid
# ─────────────────────────────────────────────────────────────────────────────

def make_grid(images: list, n_cols: int = 4) -> Image.Image:
    """Assemble a list of PIL images into a rectangular grid."""
    w, h   = images[0].size
    n_rows = (len(images) + n_cols - 1) // n_cols
    grid   = Image.new("RGB", (w * n_cols, h * n_rows))
    for i, img in enumerate(images):
        grid.paste(img, ((i % n_cols) * w, (i // n_cols) * h))
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Main generation
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    cfg: dict,
    category: str,
    defect_type: str,
    stage1_dir: Path,
    stage2_dir: Path,
    weight_V: float,
    weight_D: float,
):
    inf_cfg   = cfg["inference"]
    paths_cfg = cfg["paths"]
    out_dir   = Path(paths_cfg["generated"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate adapter paths before loading the model
    for label, path in [("Stage 1", stage1_dir), ("Stage 2", stage2_dir)]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} adapter not found: {path}\n"
                f"Check that the path is correct and the adapter was saved."
            )
        if not next(path.glob("*.safetensors"), None):
            raise FileNotFoundError(
                f"{label} adapter directory exists but contains no "
                f"*.safetensors file: {path}"
            )

    # Composed prompt — both tokens in a single natural-language description.
    # Template from project proposal:
    # "a photo of a {token_V} {category} with a {token_D} defect on the surface"
    token_V = cfg["token_V"]
    token_D = cfg["token_D"]
    prompt  = (
        f"a photo of a {token_V} {category} "
        f"with a {token_D} defect on the surface"
    )

    print(f"\n{'='*60}")
    print(f"  Inference: {category} / {defect_type}")
    print(f"  Prompt   : {prompt}")
    print(f"  Weights  : [V]={weight_V}  [D]={weight_D}")
    print(f"  Steps    : {inf_cfg['num_inference_steps']}")
    print(f"  CFG scale: {inf_cfg['guidance_scale']}")
    print(f"  N images : {inf_cfg['num_images_per_prompt']}")
    print(f"  Stage 1  : {stage1_dir}")
    print(f"  Stage 2  : {stage2_dir}")
    print(f"{'='*60}\n")

    # Load pipeline with composed adapters
    pipe = load_composed_pipeline(
        model_id=cfg["model_id"],
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        weight_V=weight_V,
        weight_D=weight_D,
    )

    # Generate
    generator = torch.Generator("cuda").manual_seed(cfg["seed"])
    n_images  = inf_cfg["num_images_per_prompt"]

    print("Generating images...")
    images = pipe(
        prompt,
        num_images_per_prompt=n_images,
        num_inference_steps=inf_cfg["num_inference_steps"],
        guidance_scale=inf_cfg["guidance_scale"],
        generator=generator,
    ).images

    # Save grid
    grid     = make_grid(images, n_cols=4)
    out_name = f"{category}_{defect_type}_composed_{n_images}imgs.png"
    out_path = out_dir / out_name
    grid.save(out_path)
    print(f"\nGrid saved: {out_path}")

    # Save individual images for evaluation metrics (FID/KID/LPIPS/DINO).
    # Commented out 
    # indiv_dir = out_dir / f"{category}_{defect_type}"
    # indiv_dir.mkdir(exist_ok=True)
    # for i, img in enumerate(images):
    #     img.save(indiv_dir / f"{i:04d}.png")
    # print(f"Individual images saved: {indiv_dir}/ ({len(images)} files)")

    # VRAM cleanup
    del pipe
    torch.cuda.empty_cache()

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Composed [V]+[D] inference for defect image synthesis"
    )
    parser.add_argument("--config",      default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--category",    required=True,
                        help="MVTec category, e.g. bottle")
    parser.add_argument("--defect_type", required=True,
                        help="Defect type, e.g. broken_large")

    # Adapter paths - override config.yaml entirely.
    # Must be set in the notebook since adapters are not committed to the repo.
    parser.add_argument("--stage1_dir",  default=None,
                        help="Path to Stage 1 adapter final/ directory. "
                             "If omitted, falls back to config.yaml checkpoints path.")
    parser.add_argument("--stage2_dir",  default=None,
                        help="Path to Stage 2 adapter final/ directory. "
                             "If omitted, falls back to config.yaml checkpoints path.")

    # Adapter weights - override config.yaml inference section.
    parser.add_argument("--weight_v",    type=float, default=None,
                        help="[V] adapter scaling weight (default: config inference.adapter_weight_V)")
    parser.add_argument("--weight_d",    type=float, default=None,
                        help="[D] adapter scaling weight (default: config inference.adapter_weight_D)")

    args = parser.parse_args()
    cfg  = load_config(args.config)

    # Resolve adapter paths: CLI args take priority over config.yaml
    ckpt_root  = Path(cfg["paths"]["checkpoints"])
    stage1_dir = (
        Path(args.stage1_dir)
        if args.stage1_dir
        else ckpt_root / "stage1" / args.category / "final"
    )
    stage2_dir = (
        Path(args.stage2_dir)
        if args.stage2_dir
        else ckpt_root / "stage2" / args.category / args.defect_type / "final"
    )

    # Resolve weights: CLI args take priority over config.yaml
    weight_V = args.weight_v if args.weight_v is not None else cfg["inference"]["adapter_weight_V"]
    weight_D = args.weight_d if args.weight_d is not None else cfg["inference"]["adapter_weight_D"]

    generate(
        cfg=cfg,
        category=args.category,
        defect_type=args.defect_type,
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        weight_V=weight_V,
        weight_D=weight_D,
    )


if __name__ == "__main__":
    main()
