# Few-Shot Personalization of a Diffusion Model for Industrial Defect Synthesis

**Authors:** Alessia Cancemi (s347156) - Elia Francesco Vigè (s339126)  
**Course:** Generative AI for Graphics and Multimedia

## Overview

Two-Stage pipeline which trains two separated adapters LoRa on SD 1.5:
- **Stage 1** (`[V]` token): binds the visual identity of the product
- **Stage 2** (`[D]` token): binds the appearance of the defect
- **Inference**: composition of `[V] + [D]` to generate defective variants of the specific product


## Setup Kaggle

### 1. Prerequisites 

- Account HuggingFace with license  SD 1.5 
- Kaggle Secrets:
  - `HF_TOKEN` -> HuggingFace Token 
  - `WANDB_API_KEY` → API key wandb 

### 2. Upload MVTec with Kaggle Dataset

```bash
# Download locally:
# https://www.mvtec.com/company/research/datasets/mvtec-ad
# then: Kaggle -> Datasets -> New Dataset -> upload bottle.tar.xz, metal_nut.tar.xz, leather.tar.xz
```

### 3. Clone repo 

```python
!git clone https://github.com/USERNAME/defect-synthesis.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -r requirements.txt
```

### 4. Execution

```bash

python data/splits.py --config config.yaml

python train/generate_prior_images.py --config config.yaml --category bottle

python train/train_stage1.py --config config.yaml --category bottle

python train/train_stage2.py --config config.yaml --category bottle --defect_type broken_large
```

## Hyperparameters

| Hyperparameter | Stage 1 | Stage 2 | Note |
|-----------|---------|---------|------|
| `lora_rank` | 4 | 4 | increase to 8 if underfitting |
| `lora_alpha` | 32 | 32 | alpha/rank = 8  |
| `max_train_steps` | 800 | 400 | - |
| `learning_rate` | 1e-4 | 5e-5 | -|
| `prior_preservation` | True | False | Only Stage 1 |

## Monitoring

Training is monitoring on [wandb.ai](https://wandb.ai) — project `defect-synthesis`.  
Every run is named `stage1_{category}_rank{r}` o `stage2_{category}_{defect_type}`.

## Category MVTec 

| Category | Defect Type | N defects |
|-----------|-------------|-----------------|
| Bottle | broken_large, broken_small, contamination | ~90 |
| Metal Nut | bent, color, flip, scratch | ~78 |
| Leather | color, cut, fold, glue, poke | ~60 |

## Token 

- `[V]` = `"sks"` -> product identity (Stage 1)
- `[D]` = `"xjy"` -> defect appearance (Stage 2)
- Inference: `"a photo of a sks bottle with a xjy scratch on the surface"`

## References

1. Bergmann et al. — MVTec AD (CVPR 2019)
2. Rombach et al. — Latent Diffusion Models (CVPR 2022)
3. Hu et al. — LoRA (ICLR 2022)
4. Ruiz et al. — DreamBooth (CVPR 2023)
5. Kumari et al. — Multi-Concept Customization (CVPR 2023)
