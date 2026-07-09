"""
inference.py — Generation of synthetic images for all category × defect
               combinations using the two LoRA adapters (Stage 1 + Stage 2).

Usage:
    # All categories and defects (baseline n_shots=10):
    python inference.py --config config.yaml

    # A single category:
    python inference.py --config config.yaml --category bottle

    # A specific combination:
    python inference.py --config config.yaml --category bottle --defect_type broken_large

    # Ablation: specify which Stage 2 checkpoint to use (for different shot counts):
    python inference.py --config config.yaml --category bottle --defect_type broken_large --n_shots 5

    # Stage 1 only (baseline: product without defect / clean):
    python inference.py --config config.yaml --category bottle --stage1_only

    # Zero-shot baseline (no LoRA applied):
    python inference.py --config config.yaml --category bottle --defect_type broken_large --zero_shot
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def vram_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def make_grid(images: list[Image.Image], ncols: int = 4, padding: int = 8,
              bg_color: tuple = (30, 30, 46)) -> Image.Image:
    """Assemble a grid of PIL images with padding and a dark background."""
    assert len(images) > 0
    w, h = images[0].size
    nrows = (len(images) + ncols - 1) // ncols

    grid_w = ncols * w + (ncols + 1) * padding
    grid_h = nrows * h + (nrows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), bg_color)

    for idx, img in enumerate(images):
        row, col = divmod(idx, ncols)
        x = padding + col * (w + padding)
        y = padding + row * (h + padding)
        grid.paste(img, (x, y))

    return grid


def add_caption(image: Image.Image, text: str,
                text_color: tuple = (205, 214, 244),
                bg_color: tuple = (30, 30, 46),
                font_size: int = 18) -> Image.Image:
    """Add a caption bar above the image """
    bar_h = font_size + 14
    new_img = Image.new("RGB", (image.width, image.height + bar_h), bg_color)
    new_img.paste(image, (0, bar_h))
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 7), text, fill=text_color, font=font)
    return new_img


def build_two_stage_pipeline(cfg: dict, category: str, defect_type: str,
                              n_shots: int | None = None) -> StableDiffusionPipeline:
    """Load SD 1.5 with Stage 1 + Stage 2 adapters."""
    ckpt_base = Path(cfg["paths"]["checkpoints"])

    stage1_path = ckpt_base / "stage1" / category / "final"
    
    if n_shots is not None:
        stage2_path = ckpt_base / "stage2" / category / defect_type / f"final_shots{n_shots}"
    else:
        stage2_path = ckpt_base / "stage2" / category / defect_type / "final"

    if not stage1_path.exists():
        raise FileNotFoundError(f"Stage 1 adapter not found: {stage1_path}")
    if not stage2_path.exists():
        raise FileNotFoundError(f"Stage 2 adapter not found: {stage2_path}")

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_id"], torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")

    pipe.load_lora_weights(str(stage1_path), adapter_name="V")
    pipe.load_lora_weights(str(stage2_path), adapter_name="D")
    pipe.set_adapters(
        ["V", "D"],
        adapter_weights=[
            cfg["inference"]["adapter_weight_V"],
            cfg["inference"]["adapter_weight_D"],
        ],
    )
    return pipe


def build_stage1_only_pipeline(cfg: dict, category: str) -> StableDiffusionPipeline:
    """Only Stage 1 (baseline: product without defect)."""
    stage1_path = Path(cfg["paths"]["checkpoints"]) / "stage1" / category / "final"
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_id"], torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    pipe.load_lora_weights(str(stage1_path), adapter_name="V")
    pipe.set_adapters(["V"], adapter_weights=[cfg["inference"]["adapter_weight_V"]])
    return pipe


def build_zero_shot_pipeline(cfg: dict) -> StableDiffusionPipeline:
    """SD 1.5 base without LoRA — baseline zero-shot."""
    return StableDiffusionPipeline.from_pretrained(
        cfg["model_id"], torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")


def make_prompt(cfg: dict, category: str, defect_type: str, mode: str) -> str:
    V = cfg["token_V"]   # sks
    D = cfg["token_D"]   # xjy
    prompts = {
        "two_stage":    f"a photo of a {V} {category} with a {D} {defect_type} defect",
        "stage1_only":  f"a photo of a {V} {category}",
        "zero_shot":    f"a photo of a {category} with a {defect_type} defect on the surface",
    }
    return prompts[mode]



def generate_run(
    pipe: StableDiffusionPipeline,
    prompt: str,
    output_dir: Path,
    cfg: dict,
    seed: int | None = None,
    num_images: int | None = None,
    ncols_grid: int = 4,
) -> dict:
    """
    Generates num_images images, saves single PNGs, the grid and returns a statistic
    dictionary to add in metadata.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = seed if seed is not None else cfg["seed"]
    num_images = num_images if num_images is not None else cfg["inference"]["num_images_per_prompt"]

    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator("cuda").manual_seed(seed)

    t0 = time.time()
    images = pipe(
        prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=cfg["inference"]["num_inference_steps"],
        guidance_scale=cfg["inference"]["guidance_scale"],
        generator=generator,
    ).images
    elapsed = time.time() - t0

    saved_paths = []
    for i, img in enumerate(images):
        p = output_dir / f"gen_{i:03d}.png"
        img.save(p)
        saved_paths.append(str(p))

   
    grid = make_grid(images, ncols=ncols_grid)
    caption_text = f"{prompt[:90]}{'...' if len(prompt) > 90 else ''}"
    grid_captioned = add_caption(grid, caption_text)
    grid_path = output_dir / "grid.png"
    grid_captioned.save(grid_path)

    return {
        "prompt": prompt,
        "seed": seed,
        "num_images": num_images,
        "num_inference_steps": cfg["inference"]["num_inference_steps"],
        "guidance_scale": cfg["inference"]["guidance_scale"],
        "generation_time_s": round(elapsed, 2),
        "saved_images": saved_paths,
        "grid_path": str(grid_path),
        "vram_gb_after": round(vram_gb(), 2),
    }



def run_inference(
    cfg: dict,
    category: str,
    defect_type: str | None = None,
    n_shots: int | None = None,
    mode: str = "two_stage",   # "two_stage" | "stage1_only" | "zero_shot"
) -> None:
    """Execute a single inference run and saves output + metadata."""

    generated_root = Path(cfg["paths"]["generated"])
    shots_tag = f"shots{n_shots}" if n_shots is not None else "baseline"

    if mode == "stage1_only":
        out_dir = generated_root / mode / category
        prompt = make_prompt(cfg, category, "", mode)
    else:
        assert defect_type is not None
        out_dir = generated_root / mode / category / defect_type / shots_tag
        prompt = make_prompt(cfg, category, defect_type, mode)

    print(f"\n[{mode.upper()}] {category} / {defect_type or 'no_defect'} / {shots_tag}")
    print(f"  Prompt : {prompt}")
    print(f"  Output : {out_dir}")

    
    try:
        if mode == "two_stage":
            pipe = build_two_stage_pipeline(cfg, category, defect_type, n_shots)
        elif mode == "stage1_only":
            pipe = build_stage1_only_pipeline(cfg, category)
        elif mode == "zero_shot":
            pipe = build_zero_shot_pipeline(cfg)
        else:
            raise ValueError(f"{mode}")
    except FileNotFoundError as e:
        print(f"  SKIP — {e}")
        return

   
    stats = generate_run(pipe, prompt, out_dir, cfg)

    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "category": category,
        "defect_type": defect_type,
        "n_shots": n_shots,
        "shots_tag": shots_tag,
        "adapter_weight_V": cfg["inference"].get("adapter_weight_V"),
        "adapter_weight_D": cfg["inference"].get("adapter_weight_D"),
        **stats,
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"{stats['num_images']} images in {stats['generation_time_s']:.1f}s {out_dir}")

    
    del pipe
    torch.cuda.empty_cache()



def get_defect_types(cfg: dict, category: str, split_dir: Path) -> list[str]:
    """
    Reads defect types from  Split JSON,
    else it inserts them in MVTec direcotry.
    """
    split_file = split_dir / f"{category}_split.json"
    if split_file.exists():
        with open(split_file) as f:
            split = json.load(f)
        return list(split["stage2"].keys())

    
    test_dir = Path(cfg["paths"]["mvtec_root"]) / category / "test"
    return sorted(d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "good")



def main():
    parser = argparse.ArgumentParser(description="Inference — generates synthetic images")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--category", default=None, help="Es: bottle.")
    parser.add_argument("--defect_type", default=None, help="Es: broken_large")
    parser.add_argument("--n_shots", type=int, default=None,
                        help="Number of shot for Stage 2. None=baseline.")
    parser.add_argument("--ablation", action="store_true",
                        help="Executes for all values of n_shots defined in config.")
    parser.add_argument("--mode", default="two_stage",
                        choices=["two_stage", "stage1_only", "zero_shot"],
                        help="Generation mode.")
    parser.add_argument("--all_modes", action="store_true",
                        help="Executes all modes.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_dir = Path(cfg["paths"]["splits_dir"])

    categories = [args.category] if args.category else cfg["categories"]
    modes = (["two_stage", "stage1_only", "zero_shot"]
             if args.all_modes else [args.mode])

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode.upper()}")
        print(f"{'='*60}")

        for category in categories:
            if mode == "stage1_only":
                run_inference(cfg, category, mode=mode)
                continue

            defect_types = ([args.defect_type] if args.defect_type
                            else get_defect_types(cfg, category, split_dir))

            shots_list = (cfg["splits"]["stage2_ablation_shots"]
                          if args.ablation else [args.n_shots])

            for defect_type in tqdm(defect_types, desc=f"{category}"):
                for n_shots in shots_list:
                    run_inference(cfg, category, defect_type, n_shots, mode)

    print("\nInference completed.")


if __name__ == "__main__":
    main()
