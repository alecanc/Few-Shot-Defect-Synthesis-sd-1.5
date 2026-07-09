# Few-Shot Personalization of a Diffusion Model for Industrial Defect Synthesis

University project for the course **Generative AI for Graphics and Multimedia**, Politecnico di Torino.

**Authors:** Alessia Cancemi, Elia Vigè 

---

## Overview

Real defect images are rare in industrial quality control, which makes it hard to train
anomaly detection models. This project tests whether Stable Diffusion 1.5 can be
personalized with few images to generate new, realistic defect images of a specific product.

We compare two designs:

**Two-stage (the proposed pipeline).** Two LoRA adapters are trained separately on the
frozen backbone and combined only at inference:

- Stage 1, token `sks`: learns the product identity from 25 clean images (DreamBooth with
  prior preservation)
- Stage 2, token `xjy`: learns the defect appearance from 15 defective images, using a
  prompt that names no product
- Inference: the two weight updates are added, `dW = w_V * dW_V + w_D * dW_D`

**Single-stage.** One LoRA adapter is trained directly on the 15 defective images with the
prompt `"a photo of a sks {category} with a xjy defect"`. Product and defect are learned
together, so there is no composition step.

## Main result

Adding the two adapter updates causes interference. Raising the product weight restores the
object but hides the defect, and raising the defect weight does the opposite. Almost no
weight combination gives a clear product together with a clear defect at the same time.

The single-stage adapter avoids this and gives better results on most defect types. On
`metal_nut/flip`, the only defect we evaluated under both conditions, it is better on all
four metrics. Two failures remain in every configuration:

- `metal_nut`: the hexagonal shape is often lost, because that geometry is rare in the
  training distribution of Stable Diffusion 1.5
- `leather/color`: the hue shift is too subtle to produce a useful training signal

Full analysis is in the report.

## Repository structure
```
defect-synthesis/
  config.yaml                      hyperparameters, tokens, paths
  requirements.txt                 project dependencies

evaluation/
  evaluate.py                    evaluation pipeline

inference/
  inference.py                   baseline zero shot inference pipeline
  inference_new.py               updated inference pipeline

train/
  train_stage1.py                product identity adapter (DreamBooth)
  train_stage2.py                defect appearance adapter

single-stage-approach/           single adapter trained with both tokens in the same prompt (code based on stage 2)
  dataset_stage2.py              defect dataset for single-stage training
  train_stage2.py                single-stage training
  01-stage2-training-singlestage.ipynb

notebook/                        Kaggle notebooks
  01_stage1_training.ipynb
  02_stage2_training_all.ipynb
  03_inference.ipynb
  04_inference_new.ipynb
  05_evaluation.ipynb

README.md                        project documentation
```
## Setup

Trained on Kaggle with a single 16GB GPU, fp16 mixed precision and gradient checkpointing.

You need a Hugging Face account that has accepted the Stable Diffusion 1.5 license, and
these Kaggle secrets:

- `HF_TOKEN`
- `WANDB_API_KEY`

Download MVTec AD from the link in the References section, then upload `bottle.tar.xz`,
`metal_nut.tar.xz` and `leather.tar.xz` as a Kaggle dataset. Or add a public MVTec AD dataset from kaggle

### Library versions

Follow the requirements or the notebook cell that specify the versions.

Do not add a second `pip install --upgrade` cell. A newer PEFT writes extra fields into
`adapter_config.json`, and adapters saved that way cannot be loaded by the pinned version.

## Running

```bash
# 1. build the splits (run once)
python data/splits.py --config config.yaml

# 2. two-stage pipeline
python train/generate_prior_images.py --config config.yaml --category bottle
python train/train_stage1.py --config config.yaml --category bottle
python train/train_stage2.py --config config.yaml --category bottle --defect_type broken_large

# 3. single-stage
python single-stage-approach/train_stage2.py --config config.yaml \
    --category bottle --defect_type broken_large
```

## Hyperparameters

| | Stage 1 | Stage 2 | Single-stage |
|---|---|---|---|
| LoRA rank | 4 | 4 to 8 | 8 |
| LoRA alpha | 32 | 32 | 32 |
| Train steps | 800 | 300 to 400 | 400 to 500 |
| Learning rate | 1e-4 | 5e-5 | 5e-5 to 7e-5 |
| Prior preservation | yes | no | no |
| Effective batch size | 4 | 4 | 4 |

LoRA is applied to the query, key, value and output projection layers of the UNet attention
blocks. The VAE and the text encoder stay frozen.

Two practical notes. The training loss is noisy and does not track visual quality, so pick
the checkpoint by looking at the validation images, not at the loss curve. Single-stage
adapters can start showing bright artifacts between step 400 and 500, so the earlier
checkpoint is sometimes better.

## Monitoring

Runs are logged to Weights and Biases, project `defect-synthesis`:

- `stage1_{category}_rank{r}`
- `stage2_{category}_{defect}_rank{r}`
- `stage2joint_{category}_{defect}_rank{r}` (single-stage, tagged `single-stage-joint`)

## Dataset

We use three categories of MVTec AD: `bottle`, `metal_nut` and `leather`. For each category
we take 25 clean images for Stage 1, and 15 defective images per defect type for Stage 2 and
for the single-stage runs. The remaining defective images are held out for evaluation. All
splits use seed 42, and images are resized to 512x512.

MVTec AD is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License (CC BY-NC-SA 4.0)**. It cannot be used for commercial purposes. The
dataset is not included in this repository and must be downloaded from the official page.
If you use it, cite the two papers below.

Stable Diffusion 1.5 is distributed under the CreativeML OpenRAIL-M license. Its use
restrictions apply to any images generated with this code.

This repository contains code only, written for a university course project. It is not
intended for commercial use.

## References

Dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad

1. Bergmann, P., Fauser, M., Sattlegger, D., Steger, C. *MVTec AD: A Comprehensive
   Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019.
2. Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., Steger, C. *The MVTec Anomaly
   Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly
   Detection.* IJCV 129(4), 2021.
3. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. *High-Resolution Image
   Synthesis with Latent Diffusion Models.* CVPR 2022. arXiv:2112.10752
4. Hu, E. J., et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
   arXiv:2106.09685
5. Ruiz, N., et al. *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for
   Subject-Driven Generation.* CVPR 2023. arXiv:2208.12242
6. Kumari, N., et al. *Multi-Concept Customization of Text-to-Image Diffusion.* CVPR 2023.
   arXiv:2212.04488
7. Hu, T., et al. *AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion
   Model.* AAAI 2024. arXiv:2312.05767
8. Shi, Q., Wei, J., Shen, F., Zhang, Z. *Few-shot Defect Image Generation based on
   Consistency Modeling.* ECCV 2024. arXiv:2408.00372