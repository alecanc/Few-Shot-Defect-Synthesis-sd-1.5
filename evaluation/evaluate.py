"""
evaluate.py — Quantitative evaluation for defect synthesis conditions.

Compares three generation conditions against real held-out images.

Metrics:
  - FID   (clean-fid, folder vs folder)
  - KID   (torchmetrics KernelInceptionDistance, mean ± std)
  - LPIPS diversity (mean pairwise LPIPS within the GENERATED set only —
    higher = more diverse samples, not a similarity-to-real metric)
  - DINO score (mean cosine similarity between real and generated CLS
    embeddings from facebook/dino-vits8 — a proxy for perceptual/identity
    fidelity that is less texture-biased than Inception features)

Usage:
    python evaluation/evaluate.py --config defect-synthesis/config.yaml \
        --category bottle --defect_type broken_large \
        --conditions two_stage single_stage stage1_only \
        --n_images 20

    python evaluation/evaluate.py --config defect-synthesis/config.yaml \
        --conditions two_stage single_stage \
        --output results/eval_summary.csv

Output:
    Prints a results table and writes a CSV (and JSON) with one row per
    (category, defect_type, condition, metric).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont




IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXCLUDED_IMAGE_NAMES = {"grid.png"}


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists() and path.name == "config.yaml":
        fallback = Path("defect-synthesis") / "config.yaml"
        if fallback.exists():
            path = fallback
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Pass --config defect-synthesis/config.yaml from the repository root."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def load_split(splits_dir: Path, category: str) -> dict:
    split_path = splits_dir / f"{category}_split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}\nRun data/splits.py first."
        )
    with open(split_path) as f:
        return json.load(f)


def list_images(folder: Path, n_images: int = None) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and p.name not in EXCLUDED_IMAGE_NAMES
    )
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found in {folder}")
    if n_images is not None:
        paths = paths[:n_images]
    return paths



# FID  


def compute_fid(real_dir: Path, fake_dir: Path, device: str = "cuda") -> float:
    from cleanfid import fid as cleanfid

    return cleanfid.compute_fid(
        str(real_dir), str(fake_dir), mode="clean", device=device, verbose=False
    )


# KID  

def _load_as_uint8_tensor(paths: list, size: int = 299) -> torch.Tensor:
    tfm_imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
        arr = np.array(img).transpose(2, 0, 1)  
        tfm_imgs.append(torch.from_numpy(arr))
    return torch.stack(tfm_imgs).to(torch.uint8)  


def compute_kid(
    real_paths: list, fake_paths: list, device: str = "cuda"
) -> tuple:
    from torchmetrics.image.kid import KernelInceptionDistance

    n = min(len(real_paths), len(fake_paths))
    subset_size = min(50, n)  
    if n < 2:
        raise ValueError("KID needs at least 2 images per set.")

    kid = KernelInceptionDistance(subset_size=subset_size, normalize=False).to(device)

    real_t = _load_as_uint8_tensor(real_paths).to(device)
    fake_t = _load_as_uint8_tensor(fake_paths).to(device)

    kid.update(real_t, real=True)
    kid.update(fake_t, real=False)
    mean, std = kid.compute()
    return float(mean), float(std)



# LPIPS diversity 

_lpips_model = None


def _get_lpips_model(device: str = "cuda"):
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net="alex").to(device)
    return _lpips_model


def _load_lpips_tensor(path: Path, size: int = 256, device: str = "cuda") -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return t


def compute_lpips_diversity(fake_paths: list, device: str = "cuda") -> float:
    model = _get_lpips_model(device)
    tensors = [_load_lpips_tensor(p, device=device) for p in fake_paths]

    dists = []
    with torch.no_grad():
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                d = model(tensors[i], tensors[j]).item()
                dists.append(d)

    if not dists:
        return float("nan")
    return float(np.mean(dists))


# DINO score 

_dino_model = None
_dino_transform = None


def _get_dino_model(device: str = "cuda"):
    global _dino_model, _dino_transform
    if _dino_model is None:
        from torchvision import transforms

        _dino_model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        _dino_model.eval().to(device)
        _dino_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return _dino_model, _dino_transform


def _dino_embed(paths: list, device: str = "cuda") -> torch.Tensor:
    model, tfm = _get_dino_model(device)
    embeds = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            feat = model(x) 
            feat = torch.nn.functional.normalize(feat, dim=-1)
            embeds.append(feat.cpu())
    return torch.cat(embeds, dim=0)  


def compute_dino_score(real_paths: list, fake_paths: list, device: str = "cuda") -> float:
    real_emb = _dino_embed(real_paths, device)
    fake_emb = _dino_embed(fake_paths, device)
    sim_matrix = real_emb @ fake_emb.T  # (N_real, N_fake)
    return float(sim_matrix.mean())


# Visual Evaluation 

def save_diff_heatmap(img_path: Path, ref_path: Path, output_path: Path):
    """
    Computes absolute pixel difference between a generated image and a reference clean image,
    creating a red heatmap overlay to visualize the synthesized defect region.
    """
    img = Image.open(img_path).convert("RGB").resize((256, 256), Image.BICUBIC)
    ref = Image.open(ref_path).convert("RGB").resize((256, 256), Image.BICUBIC)
    
    img_arr = np.array(img).astype(np.float32)
    ref_arr = np.array(ref).astype(np.float32)
    
    # Compute absolute difference and take mean across channels
    diff = np.abs(img_arr - ref_arr)
    diff_gray = np.mean(diff, axis=2)
    
    # Normalize to [0, 255]
    diff_min, diff_max = diff_gray.min(), diff_gray.max()
    if diff_max > diff_min:
        diff_norm = (diff_gray - diff_min) / (diff_max - diff_min) * 255.0
    else:
        diff_norm = diff_gray
    diff_norm = diff_norm.astype(np.uint8)
    
    # Create red-highlight heatmap
    heatmap = np.zeros_like(img_arr, dtype=np.uint8)
    heatmap[..., 0] = diff_norm  # Red channel represents intensity of difference
    
    blended = (img_arr * 0.7 + heatmap * 0.3).astype(np.uint8)
    
  
    combined = Image.new("RGB", (256 * 3, 256))
    combined.paste(ref, (0, 0))
    combined.paste(img, (256, 0))
    combined.paste(Image.fromarray(blended), (512, 0))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)


def generate_comparison_grid(
    real_paths: list[Path],
    fake_paths: list[Path],
    output_path: Path,
    title: str,
    max_imgs: int = 8,
):
    """
    Create a clean side-by-side comparison grid of Real images vs Generated images.
    """
    n = min(max_imgs, len(real_paths), len(fake_paths))
    if n == 0:
        return
        
    size = 256
    padding = 10
    label_height = 40
    
    grid_w = n * size + (n + 1) * padding
    grid_h = 2 * size + 3 * padding + 2 * label_height
    
    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 46))
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        

    draw.text((padding, padding), f"REAL IMAGES - {title}", fill=(205, 214, 244), font=font)
    for idx in range(n):
        img = Image.open(real_paths[idx]).convert("RGB").resize((size, size), Image.BICUBIC)
        x = padding + idx * (size + padding)
        y = padding + label_height + padding
        grid.paste(img, (x, y))
        

    y_label = padding + label_height + padding + size + padding
    draw.text((padding, y_label), f"GENERATED IMAGES - {title}", fill=(205, 214, 244), font=font)
    for idx in range(n):
        img = Image.open(fake_paths[idx]).convert("RGB").resize((size, size), Image.BICUBIC)
        x = padding + idx * (size + padding)
        y = y_label + label_height + padding
        grid.paste(img, (x, y))
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"  Saved comparison grid: {output_path}")


def generate_multi_condition_grid(
    conditions_paths: dict[str, list[Path]],
    output_path: Path,
    title: str,
    max_imgs: int = 6,
):
    """
    Create a grid showing multiple evaluated conditions as separate rows.
    """
    size = 256
    padding = 10
    label_height = 40
    
    valid_conditions = {k: v for k, v in conditions_paths.items() if len(v) > 0}
    if not valid_conditions:
        return
        
    n_conds = len(valid_conditions)
    n_imgs = min(max_imgs, *(len(v) for v in valid_conditions.values()))
    if n_imgs == 0:
        return
        
    grid_w = n_imgs * size + (n_imgs + 1) * padding
    grid_h = n_conds * size + (n_conds + 1) * padding + n_conds * label_height
    
    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 46))
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        
    for cond_idx, (cond_name, paths) in enumerate(valid_conditions.items()):
        y_label = padding + cond_idx * (size + padding + label_height)
        draw.text((padding, y_label), f"CONDITION: {cond_name.upper()} - {title}", fill=(205, 214, 244), font=font)
        
        for img_idx in range(n_imgs):
            img = Image.open(paths[img_idx]).convert("RGB").resize((size, size), Image.BICUBIC)
            x = padding + img_idx * (size + padding)
            y = y_label + label_height + padding
            grid.paste(img, (x, y))
            
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"  Saved multi-condition grid: {output_path}")



def resolve_condition_dirs(
    generated_root: Path,
    mvtec_root: Path,
    category: str,
    defect_type: str,
    split: dict,
    condition: str,
    shots_tag: str,
) -> tuple:
    """
    Returns (real_dir_or_paths, fake_dir) for a given condition.
    real is returned as a list of Paths since eval images may need to be
    staged into a temp folder for clean-fid (which wants a folder path).
    """
    # Normalize condition names to check both with and without underscores/hyphens
    condition_variants = [condition]
    if "_" in condition:
        condition_variants.append(condition.replace("_", ""))
        condition_variants.append(condition.replace("_", "-"))
    elif "-" in condition:
        condition_variants.append(condition.replace("-", ""))
        condition_variants.append(condition.replace("-", "_"))
    else:
        if condition == "singlestage":
            condition_variants.extend(["single_stage", "single-stage"])
        elif condition == "singlestagefull":
            condition_variants.extend(["single_stage_full", "singlestage_full", "single-stage-full"])
        elif condition == "singlestage1":
            condition_variants.extend(["single_stage1", "single-stage1"])
        elif condition == "twostage":
            condition_variants.extend(["two_stage", "two-stage"])
        elif condition == "stageunified":
            condition_variants.extend(["stage-unified", "stage_unified"])

    fake_dir = None
    for cond_var in condition_variants:
        fake_base = generated_root / cond_var / category / defect_type
        if not fake_base.exists() and cond_var == "stage1_only":
            fake_base = generated_root / cond_var / category

        # Fallbacks for baseline_single or single1
        if not fake_base.exists() and cond_var in ("single1", "single_stage1", "singlestage1"):
            fb_alt = generated_root / "baseline_single" / category / defect_type
            if fb_alt.exists():
                fake_base = fb_alt
            else:
                fb_alt = generated_root / "singlestage" / category / defect_type
                if fb_alt.exists():
                    fake_base = fb_alt

        trial_dir = fake_base / shots_tag
        if not trial_dir.exists() and shots_tag == "baseline" and fake_base.exists():
            if (fake_base / "validation-imgs").exists():
                trial_dir = fake_base / "validation-imgs"
            elif (fake_base / "validation-img").exists():
                trial_dir = fake_base / "validation-img"
            else:
                trial_dir = fake_base
            
        if trial_dir.exists():
            fake_dir = trial_dir
            break

    # If none of the variants exist, just default to the original structure
    if fake_dir is None:
        fake_base = generated_root / condition / category / defect_type
        if condition == "stage1_only":
            fake_dir = generated_root / condition / category
        else:
            if (fake_base / "validation-imgs").exists():
                fake_dir = fake_base / "validation-imgs"
            elif (fake_base / "validation-img").exists():
                fake_dir = fake_base / "validation-img"
            else:
                fake_dir = fake_base / shots_tag if (fake_base / shots_tag).exists() else fake_base

    if condition == "stage1_only":
        good_dir = mvtec_root / category / "test" / "good"
        real_paths = list_images(good_dir)
        return real_paths, fake_dir
    else:
        real_paths = [Path(p) for p in split["eval"].get(defect_type, [])]
        if not real_paths:
            raise ValueError(
                f"No real eval images for {category}/{defect_type}."
            )
        return real_paths, fake_dir


def stage_real_folder(real_paths: list, staging_root: Path, tag: str) -> Path:
    staged = staging_root / tag
    staged.mkdir(parents=True, exist_ok=True)
    for old in staged.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()
    for idx, p in enumerate(real_paths):
        target = staged / f"{idx:04d}_{p.name}"
        try:
            target.symlink_to(p.resolve())
        except (OSError, NotImplementedError):
            import shutil

            shutil.copy(p, target)
    return staged


def stage_image_folder(image_paths: list[Path], staging_root: Path, tag: str) -> Path:
    """Create a filtered folder for metrics that require directory inputs."""
    return stage_real_folder(image_paths, staging_root, tag)



def evaluate_one(
    category: str,
    defect_type: str,
    condition: str,
    generated_root: Path,
    mvtec_root: Path,
    split: dict,
    staging_root: Path,
    n_images: int,
    device: str,
    metrics: list,
    shots_tag: str,
) -> dict:
    real_paths, fake_dir = resolve_condition_dirs(
        generated_root, mvtec_root, category, defect_type, split, condition, shots_tag
    )
    fake_paths = list_images(fake_dir, n_images=n_images)
    real_paths = real_paths[:n_images] if n_images else real_paths

    tag = f"{category}_{defect_type}_{condition}" if condition != "stage1_only" else f"{category}_{condition}"
    real_dir = stage_image_folder(real_paths, staging_root, f"{tag}_real")
    fake_metric_dir = stage_image_folder(fake_paths, staging_root, f"{tag}_fake")

    row = {
        "category": category,
        "defect_type": defect_type if condition != "stage1_only" else "-",
        "condition": condition,
        "n_real": len(real_paths),
        "n_fake": len(fake_paths),
    }

    if "fid" in metrics:
        try:
            row["fid"] = compute_fid(real_dir, fake_metric_dir, device=device)
        except Exception as e:
            print(f"   FID failed for {tag}: {e}")
            row["fid"] = None

    if "kid" in metrics:
        try:
            kid_mean, kid_std = compute_kid(real_paths, fake_paths, device=device)
            row["kid_mean"] = kid_mean
            row["kid_std"] = kid_std
        except Exception as e:
            print(f"   KID failed for {tag}: {e}")
            row["kid_mean"], row["kid_std"] = None, None

    if "lpips" in metrics:
        try:
            row["lpips_diversity"] = compute_lpips_diversity(fake_paths, device=device)
        except Exception as e:
            print(f"  LPIPS failed for {tag}: {e}")
            row["lpips_diversity"] = None

    if "dino" in metrics:
        try:
            row["dino_score"] = compute_dino_score(real_paths, fake_paths, device=device)
        except Exception as e:
            print(f"   DINO failed for {tag}: {e}")
            row["dino_score"] = None

    return row


def print_table(rows: list) -> None:
    if not rows:
        print("No results.")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def save_results(rows: list, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {output}")
    print(f"Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate defect synthesis conditions")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--category", action="append", default=None,
                         help="Category to evaluate. Repeatable. Default: all in config.")
    parser.add_argument("--defect_type", action="append", default=None,
                         help="Defect type to evaluate. Repeatable. Default: all with eval images.")
    parser.add_argument("--conditions", nargs="+",
                         default=["two_stage", "single_stage", "stage1_only", "single1", "single_stage1", "single_stage_full", "zero_shot", "sweep_weight", "sweep_prompt", "singlestage", "singlestage_full", "stage-unified", "stage_unified"],
                         choices=["two_stage", "single_stage", "stage1_only", "single1", "single_stage1", "single_stage_full", "zero_shot", "sweep_weight", "sweep_prompt", "singlestage", "singlestage_full", "stage-unified", "stage_unified"])
    parser.add_argument("--metrics", nargs="+",
                         default=["fid", "kid", "lpips", "dino"],
                         choices=["fid", "kid", "lpips", "dino"])
    parser.add_argument("--n_images", type=int, default=None,
                         help="Cap on images per set (real and fake). Default: use all available.")
    parser.add_argument("--shots_tag", default="baseline",
                         help="Generated subfolder to evaluate, e.g. baseline, shots5, shots10.")
    parser.add_argument("--device", default=None,
                         help="cuda, cpu, or omitted for automatic selection.")
    parser.add_argument("--generated_root", default=None,
                         help="Override generated path from config.yaml.")
    parser.add_argument("--output", default="results/eval_summary.csv")
    parser.add_argument("--visualize", action="store_true",
                         help="Generate comparison grids and difference heatmaps.")
    parser.add_argument("--visuals_dir", default="results/visuals",
                         help="Directory where visual evaluation plots are saved.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generated_root = Path(args.generated_root) if args.generated_root else Path(cfg["paths"]["generated"])
    mvtec_root = Path(cfg["paths"]["mvtec_root"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    output_path = Path(args.output)
    staging_root = output_path.parent / "_eval_staging"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    visuals_root = Path(args.visuals_dir)

    categories = args.category or cfg["categories"]

    rows = []
    for category in categories:
        split = load_split(splits_dir, category)

        if "stage1_only" in args.conditions:
            try:
                res_row = evaluate_one(
                    category, None, "stage1_only",
                    generated_root, mvtec_root, split, staging_root,
                    args.n_images, device, args.metrics, args.shots_tag,
                )
                rows.append(res_row)
                
                if args.visualize:
                    good_dir = mvtec_root / category / "test" / "good"
                    real_paths = list_images(good_dir)
                    fake_dir = generated_root / "stage1_only" / category
                    fake_paths = list_images(fake_dir, n_images=args.n_images)
                    grid_name = f"{category}_stage1_only_comparison.png"
                    generate_comparison_grid(
                        real_paths, fake_paths, visuals_root / grid_name, f"{category} (stage1_only)"
                    )
            except Exception as e:
                print(f"[SKIP] {category}/stage1_only evaluation/visuals: {e}")

        defect_types = args.defect_type or sorted(split["eval"].keys())
        for defect_type in defect_types:
            cond_paths_dict = {}
            try:
                real_paths = [Path(p) for p in split["eval"].get(defect_type, [])]
                real_paths = real_paths[:args.n_images] if args.n_images else real_paths
                cond_paths_dict["real"] = real_paths
            except Exception:
                real_paths = []

            for condition in args.conditions:
                if condition == "stage1_only":
                    continue
                try:
                    res_row = evaluate_one(
                        category, defect_type, condition,
                        generated_root, mvtec_root, split, staging_root,
                        args.n_images, device, args.metrics, args.shots_tag,
                    )
                    rows.append(res_row)

                    if args.visualize:
                        _, fake_dir = resolve_condition_dirs(
                            generated_root, mvtec_root, category, defect_type, split, condition, args.shots_tag
                        )
                        fake_paths = list_images(fake_dir, n_images=args.n_images)
                        cond_paths_dict[condition] = fake_paths

                        grid_name = f"{category}_{defect_type}_{condition}_comparison.png"
                        generate_comparison_grid(
                            real_paths, fake_paths, visuals_root / grid_name, f"{category} / {defect_type} ({condition})"
                        )

                        if len(fake_paths) > 0 and len(real_paths) > 0:
                            diff_name = f"{category}_{defect_type}_{condition}_diff_heatmap.png"
                            save_diff_heatmap(
                                fake_paths[0], real_paths[0], visuals_root / diff_name
                            )
                except Exception as e:
                    print(f"[SKIP] {category}/{defect_type}/{condition} evaluation/visuals: {e}")

           
            if args.visualize and len(cond_paths_dict) > 1:
                try:
                    multi_grid_name = f"{category}_{defect_type}_multi_condition.png"
                    generate_multi_condition_grid(
                        cond_paths_dict, visuals_root / multi_grid_name, f"{category} / {defect_type}"
                    )
                except Exception as e:
                    print(f"   Multi-condition grid failed for {category}/{defect_type}: {e}")

    print_table(rows)
    if rows:
        save_results(rows, output_path)


if __name__ == "__main__":
    main()
