"""
splits.py — Generate and save the train/test splits for MVTec AD.

Run once with a fixed seed. The produced JSON files
are then read by dataset_stage1.py and dataset_stage2.py.

Usage:
    python data/splits.py --config config.yaml
    python data/splits.py --config config.yaml --dry-run   # Print stats without saving
"""

import argparse
import json
import random
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_split(
    category: str,
    mvtec_root: Path,
    n_stage1: int = 25,
    n_stage2_per_type: int = 10,
    seed: int = 42,
) -> dict:
    """
    Produce the split for a single MVTec category.

    Returns:
        {
          "stage1": ["path/to/img.png", ...],          # n_stage1 clean images
          "stage2": {"defect_type": ["path", ...], …}, # n_stage2_per_type per type
          "eval":   {"defect_type": ["path", ...], …}, # remaining for evaluation
          "stats":  {...}                               # control statistics
        }
    """
    rng = random.Random(seed)
    cat_dir = mvtec_root / category

    if not cat_dir.exists():
        raise FileNotFoundError(f"Category not found: {cat_dir}")

    # Stage 1: clean images
    good_dir = cat_dir / "train" / "good"
    good_imgs = sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg"))

    if len(good_imgs) < n_stage1:
        raise ValueError(
            f"{category}: found only {len(good_imgs)} clean images "
            f"but {n_stage1} are required."
        )

    rng.shuffle(good_imgs)
    stage1_imgs = [str(p) for p in good_imgs[:n_stage1]]

    # Stage 2 + Eval: defective images per type 
    test_dir = cat_dir / "test"
    stage2, eval_set = {}, {}

    defect_types = sorted(
        d.name for d in test_dir.iterdir()
        if d.is_dir() and d.name != "good"
    )

    for defect_type in defect_types:
        defect_dir = test_dir / defect_type
        defect_imgs = sorted(defect_dir.glob("*.png")) + sorted(defect_dir.glob("*.jpg"))

        if len(defect_imgs) == 0:
            print(f"  WARN: no images found in {defect_dir}, skipping.")
            continue

        rng.shuffle(defect_imgs)

        # If there are fewer images than expected, take all for stage2
        n_train = min(n_stage2_per_type, len(defect_imgs))
        stage2[defect_type] = [str(p) for p in defect_imgs[:n_train]]
        eval_set[defect_type] = [str(p) for p in defect_imgs[n_train:]]

    # Control Statistics 
    stats = {
        "total_clean_available": len(good_imgs),
        "stage1_used": len(stage1_imgs),
        "stage2": {k: len(v) for k, v in stage2.items()},
        "eval":   {k: len(v) for k, v in eval_set.items()},
        "defect_types": defect_types,
    }

    return {
        "category": category,
        "seed": seed,
        "stage1": stage1_imgs,
        "stage2": stage2,
        "eval": eval_set,
        "stats": stats,
    }


def print_split_summary(split: dict) -> None:
    cat = split["category"]
    stats = split["stats"]
    print(f"\n{'─'*50}")
    print(f"  {cat.upper()}")
    print(f"{'─'*50}")
    print(f"  Clean available : {stats['total_clean_available']}")
    print(f"  Stage 1           : {stats['stage1_used']} images")
    print(f"  Stage 2 per type  :")
    for dtype, n in stats["stage2"].items():
        n_eval = stats["eval"].get(dtype, 0)
        print(f"    {dtype:<20} train={n}  eval={n_eval}")


def main():
    parser = argparse.ArgumentParser(description="Generate split MVTec AD")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print only the statistics without saving",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    mvtec_root = Path(cfg["paths"]["mvtec_root"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    n_stage1 = cfg["splits"]["stage1_n_images"]
    n_stage2 = cfg["splits"]["stage2_n_per_type"]

    if not args.dry_run:
        splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"MVTec root : {mvtec_root}")
    print(f"Seed       : {seed}")
    print(f"Stage 1    : {n_stage1} clean images per category")
    print(f"Stage 2    : {n_stage2} defective images per type")

    for category in cfg["categories"]:
        try:
            split = make_split(
                category=category,
                mvtec_root=mvtec_root,
                n_stage1=n_stage1,
                n_stage2_per_type=n_stage2,
                seed=seed,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"\nERROR [{category}]: {e}")
            continue

        print_split_summary(split)

        if not args.dry_run:
            out_path = splits_dir / f"{category}_split.json"
            with open(out_path, "w") as f:
                json.dump(split, f, indent=2)
            print(f"  Saved in: {out_path}")

    if args.dry_run:
        print("\n[DRY RUN] No files saved.")
    else:
        print("\nSplit generated correctly.")


if __name__ == "__main__":
    main()
