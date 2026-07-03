"""
dataset_stage2.py - PyTorch Dataset for unified Stage 2 ([D] token, defect concept)

The prompt is f"a photo of a {token_V} {self.category} with a {token_D} defect"

The dataset supports three operating modes, all needed by the project:

  1. SINGLE TYPE  : one defect type at a time (standard training per type)
  2. ALL TYPES    : all defect images of a category pooled together (alternative)

"a photo of a [D] defect on a surface", we keep the prompt fixed and identical for every item in the batch 
as the [D] token must absorb defect appearance from the images alone.

Augmentation
------------
Stage 2 uses more aggressive augmentation than Stage 1, because of
  - far fewer images (15 per type vs 25 clean).
  - defects are local features and global flips and crops do not destroy them.
  - Color jitter is included only for structural/geometric defect types where
    chromatic shifts would not be semantically misleading. For chromatic defect
    types (es. "color") we disable color jitter so the model learns the actual
    hue shift, not a random one on top of it
"""

import json
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Defect types where appearance is primarily a colour change.
# For these we skip colour jitter: we do not want to augment away the very signal
# the model is supposed to learn.
CHROMATIC_DEFECT_TYPES = {"color", "colour", "contamination"}


# ──────────────────────────────────────────────────────────────────────────────
# Transform builders
# ──────────────────────────────────────────────────────────────────────────────

def build_transform_stage2(
    size: int = 512,
    augment: bool = True,
    chromatic: bool = False,
) -> transforms.Compose:
    """
    Build the image preprocessing pipeline for Stage 2

    Stage 1 build_transform() with additions:
      - RandomResizedCrop: simulates different zoom levels / framing.
        scale=(0.85, 1.0) keeps most of the image so not to crop defects out
      - ColorJitter (only when augment=True AND chromatic=False): small brightness
        and contrast perturbations add diversity without destroying colour-defects.

    Args:
        size:      target spatial resolution (512 for SD 1.5)
        augment:   whether to add stochastic augmentations
        chromatic: if True, skip ColorJitter (defect is a colour change)
    """
    tfms = []

    if augment:
        tfms += [
            # RandomResizedCrop instead of CenterCrop to vary the composition slightly
            transforms.RandomResizedCrop(
                size,
                scale=(0.85, 1.0),        # stay close to full image
                ratio=(0.95, 1.05),        # near-square crops only
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if not chromatic:
            # Small brightness/contrast jitter is safe for structural/geometric defects
            tfms.append(
                transforms.ColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                    saturation=0.0,   # no saturation change
                    hue=0.0,          # no hue change
                )
            )
    else:
        # resize + centre crop, same as Stage 1 validation
        tfms += [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
        ]

    tfms += [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # [0,1] → [-1,1]
    ]
    return transforms.Compose(tfms)


# ──────────────────────────────────────────────────────────────────────────────
# Main Dataset
# ──────────────────────────────────────────────────────────────────────────────

class Stage2Dataset(Dataset):
    """
    Dataset for Stage 2: binds defect appearance to the [D] token via LoRA.

    Operating modes (controlled by `defect_type`):

      Single-type mode (standard):
        Stage2Dataset(split_path, tokenizer, defect_type="broken_large")
        - trains on `splits["stage2"]["broken_large"]` images

      All-types mode (pooled):
        Stage2Dataset(split_path, tokenizer, defect_type=None)
        - pools every defect type in stage2 into one flat list


    Every __getitem__ returns:
        {
          "pixel_values": Tensor [3, 512, 512],    # normalized image in [-1,1]
          "input_ids":    Tensor [77],             # tokenized prompt
          "defect_type":  str,                     
        }

    Args:
        split_path:   path to the category JSON produced by splits.py
        tokenizer:    CLIPTokenizer from SD 1.5
        token_D:      special defect token  ("xjy" from config.yaml)
        defect_type:  one defect type name, or None to pool all types
        image_size:   spatial resolution (512 for SD 1.5)
        augment:      enable stochastic augmentation
    """

    def __init__(
        self,
        split_path: str,
        tokenizer,
        token_D: str = "xjy",
        token_V: str = "sks",
        defect_type: Optional[str] = None,
        image_size: int = 512,
        augment: bool = True,
        include_clean: bool = False,
        n_clean: int = 8,
    ):
        with open(split_path) as f:
            split_data = json.load(f)

        self.category = split_data["category"]
        self.token_D = token_D
        self.tokenizer = tokenizer
        self.image_size = image_size

        #  Collect image paths and associated defect-type labels
        self.image_paths: List[Path] = []
        self.defect_labels: List[str] = []   # parallel list, same length

        stage2_dict: dict = split_data["stage2"]  # {defect_type: [str paths]}

        if defect_type is not None:
            # Single-type mode
            if defect_type not in stage2_dict:
                available = list(stage2_dict.keys())
                raise ValueError(
                    f"Defect type '{defect_type}' not found in split for "
                    f"category '{self.category}'. "
                    f"Available types: {available}"
                )
            paths = [Path(p) for p in stage2_dict[defect_type]]
            self.image_paths = paths
            self.defect_labels = [defect_type] * len(paths)
        else:
            # All-types mode
            for dtype, paths_str in stage2_dict.items():
                for p in paths_str:
                    self.image_paths.append(Path(p))
                    self.defect_labels.append(dtype)

        if len(self.image_paths) == 0:
            raise ValueError(
                f"Stage2Dataset for category='{self.category}', "
                f"defect_type='{defect_type}' "
                f"resulted in 0 images. Check your split JSON."
            )

        # Flags which items are clean product images. Only the fallback sets any True.
        self.is_clean: List[bool] = [False] * len(self.image_paths)

        # Optional identity reinforcement, off by default. Enable only if the
        # single-stage adapter overfits to the defect scenes and loses object diversity.
        # Pulls a few clean images from the stage1 split and tags them "good".
        if include_clean:
            clean_all = [Path(p) for p in split_data.get("stage1", [])]
            clean_sel = clean_all[:n_clean] if n_clean is not None else clean_all
            self.image_paths += clean_sel
            self.defect_labels += ["good"] * len(clean_sel)
            self.is_clean += [True] * len(clean_sel)

        # Verify all paths exist
        missing = [p for p in self.image_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image(s) listed in the split do not exist on disk.\n"
                f"First missing: {missing[0]}\n"
                f"Check that MVTec is extracted at the path specified in config.yaml."
            )

        ## Prompt construction

        # Combined single-stage prompt: both tokens co-occur on images that
        # contain the product and the defect together. This is the supervision
        # the two-stage compose-at-inference design never provided.
        self.prompt = f"a photo of a {token_V} {self.category} with a {token_D} defect"

        # Pre-tokenize once
        # every defect item shares the same prompt, so we tokenize once and store the tensor
        self.input_ids: torch.Tensor = tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]   # squeeze batch dim → [77]

        # Identity-only prompt for the optional clean images (fallback path)
        self.clean_input_ids: Optional[torch.Tensor] = None
        if include_clean:
            clean_prompt = f"a photo of a {token_V} {self.category}"
            self.clean_input_ids = tokenizer(
                clean_prompt,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

        ## Augmentation 
        
        # Detect if this dataset is chromatic-only (all images are colour-defects)
        unique_types = set(self.defect_labels)
        is_all_chromatic = unique_types.issubset(CHROMATIC_DEFECT_TYPES)

        # If we have a mix of types (all-types mode) we build two transforms and select per-item in __getitem__ 
        # if single-type, one transform is enough
        self._mixed_types = (defect_type is None) and not is_all_chromatic

        if self._mixed_types:
            # Per-item transform selection
            self._transform_structural = build_transform_stage2(
                image_size, augment=augment, chromatic=False
            )
            self._transform_chromatic = build_transform_stage2(
                image_size, augment=augment, chromatic=True
            )
        else:
            chromatic_flag = is_all_chromatic or (
                defect_type is not None and defect_type in CHROMATIC_DEFECT_TYPES
            )
            self.transform = build_transform_stage2(
                image_size, augment=augment, chromatic=chromatic_flag
            )

        # Clean images (fallback) always use the structural transform: no colour
        # jitter concern, we just want pose and framing variety on the product.
        self._transform_clean = None
        if include_clean:
            self._transform_clean = build_transform_stage2(
                image_size, augment=augment, chromatic=False
            )

        print(
            f"[Stage2Dataset] category={self.category!r} | "
            f"defect_type={defect_type!r} | "
            f"n_images={len(self.image_paths)} | "
            f"augment={augment} | "
            f"prompt: '{self.prompt}'"
        )

    ## Length & item access 

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        dtype    = self.defect_labels[idx]

        image = Image.open(img_path).convert("RGB")

        # Select transform and prompt. Clean items (fallback) get the identity
        # prompt, defect items get the combined prompt.
        if self.is_clean[idx]:
            tfm = self._transform_clean
            input_ids = self.clean_input_ids
        elif self._mixed_types:
            tfm = (
                self._transform_chromatic
                if dtype in CHROMATIC_DEFECT_TYPES
                else self._transform_structural
            )
            input_ids = self.input_ids
        else:
            tfm = self.transform
            input_ids = self.input_ids

        pixel_values = tfm(image)

        return {
            "pixel_values": pixel_values,   # [3, 512, 512] float, in [-1,1]
            "input_ids":    input_ids,       # [77] long
            "defect_type":  dtype,           
        }

    # helpers

    def available_defect_types(self) -> List[str]:
        """Return the sorted unique defect types present in this dataset"""
        return sorted(set(self.defect_labels))

    def shots_per_type(self) -> dict:
        """Return a {defect_type: count} summary, useful for sanity checks"""
        counts: dict = {}
        for dtype in self.defect_labels:
            counts[dtype] = counts.get(dtype, 0) + 1
        return counts


# ──────────────────────────────────────────────────────────────────────────────
# Collation function
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn_stage2(examples: list) -> dict:
    """
    Collation function for the Stage 2 DataLoader.

    Stacks pixel_values and input_ids into batched tensors.
    Passes defect_type labels through as a plain list of strings
    (not consumed by the model, only for info)

    The batch is always a flat [B, ...] tensor
    """
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    input_ids    = torch.stack([e["input_ids"]    for e in examples])
    defect_types = [e["defect_type"] for e in examples]

    return {
        # contiguous_format ensures the tensor layout is optimal for CUDA kernels
        "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(),
        "input_ids":    input_ids,
        "defect_types": defect_types,   # list[str], length = batch_size
    }