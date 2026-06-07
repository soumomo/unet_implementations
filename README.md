# Nuclei Segmentation on TNBC Histopathology — A Comparative Study of U-Net Variants

This repository contains implementations of five deep learning architectures for nuclei segmentation, benchmarked on the TNBC (Triple Negative Breast Cancer) histopathology dataset. Each model is implemented as a self-contained Jupyter notebook with training, evaluation, and visualization.

## Dataset

All models are trained and evaluated on the **MoNuSeg / TNBC** nuclei segmentation dataset, available on Kaggle:

**[TNBC Dataset on Kaggle](https://www.kaggle.com/datasets/momoketchum/monuseg-2018-splitted)**

| Split | Samples |
|-------|---------|
| Train | 880 |
| Validation | 165 |
| Test | 55 |

The images are H&E-stained histopathology slides with corresponding binary segmentation masks. The task is binary pixel-level classification: nucleus vs. background.

---

## Results

### Test Set Performance

| Model | Dice (F1) | IoU | Precision | Recall | HD95 (px) | Epochs Trained |
|:------|:---------:|:---:|:---------:|:------:|:---------:|:--------------:|
| **Attention U-Net** | **0.8652** | **0.7876** | 0.8502 | **0.8877** | **2.88** | 50 |
| ResNet34 Attention U-Net | 0.8559 | 0.7574 | **0.8581** | 0.8554 | 3.48 | 10 |
| CBAM U-Net | 0.7639 | 0.6387 | 0.7590 | 0.7883 | 23.54 | 15 |
| SA U-Net | 0.7586 | 0.6415 | 0.7544 | 0.7836 | 29.75 | 10 |
| EViT-UNet | 0.7040 | 0.6072 | 0.6782 | 0.7451 | 28.35 | 25 |

**Note on training duration:** The models were not trained for an equal number of epochs. The Attention U-Net (50 epochs) had the most training time, while CBAM U-Net (15), SA U-Net (10), and ResNet34 Attention U-Net (10) were stopped early. All models with fewer epochs were still showing improvement in their validation curves at the time training was halted — their reported metrics represent a lower bound on what these architectures can achieve. A fair comparison would require training all models to convergence with the same epoch budget and scheduler.

---

## Implemented Architectures

### 1. Attention U-Net (`unet_attention.ipynb`)

Standard U-Net encoder-decoder with Attention Gates on skip connections. The attention gates learn to suppress irrelevant background regions in the encoder features before concatenation with decoder features, improving focus on target structures.

- **Skip mechanism:** Attention Gate (spatial gating)
- **Parameters:** ~34M
- **Input resolution:** 572 x 572
- **Reference:** Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018)

### 2. CBAM U-Net (`cbam-unet.ipynb`)

U-Net augmented with Convolutional Block Attention Modules (CBAM) applied after each encoder and decoder block. CBAM applies sequential channel attention and spatial attention to refine intermediate feature maps.

- **Skip mechanism:** Standard concatenation (CBAM applied within blocks)
- **Parameters:** ~35M
- **Input resolution:** 572 x 572
- **Reference:** Woo et al., "CBAM: Convolutional Block Attention Module" (2018)

### 3. SA U-Net (`unet-spatial-attention.ipynb`)

U-Net with spatial attention blocks and DropBlock regularization. Uses a reverse attention mechanism where spatial attention maps are derived from feature statistics to guide the network's focus.

- **Skip mechanism:** Standard concatenation
- **Parameters:** ~31M
- **Input resolution:** 572 x 572
- **Reference:** Guo et al., "SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation" (2021)

### 4. ResNet34 Attention U-Net (`unet-resnet-attention-notebook.ipynb`)

U-Net with a ResNet34 backbone pretrained on ImageNet as the encoder, combined with Attention Gates on skip connections. The pretrained encoder provides strong initial feature representations, leading to faster convergence.

- **Skip mechanism:** Attention Gate (same as Attention U-Net)
- **Parameters:** ~24M
- **Input resolution:** 572 x 572
- **Pretrained:** ResNet34 (ImageNet)
- **Reference:** He et al. (2016); Oktay et al. (2018)

### 5. EViT-UNet (`evit-unet.ipynb`)

Hybrid CNN-Transformer architecture based on EfficientFormer. The encoder uses convolutional FFN blocks in early (high-resolution) stages and self-attention blocks in deeper (low-resolution) stages. Skip connections use Channel Cross Attention (CCA) instead of standard concatenation or spatial attention gates.

- **Skip mechanism:** CCA (Channel Cross Attention)
- **Parameters:** ~55M
- **Input resolution:** 224 x 224 (fixed, due to pre-registered attention bias buffers)
- **Multi-GPU:** Trained with DataParallel across 2x T4 GPUs
- **Reference:** Li et al., "EViT-UNet: U-Net Like Efficient Vision Transformer for Medical Image Segmentation on Mobile and Edge Devices" (2024)

---

## Training Setup

All models share the following configuration unless noted otherwise:

| Parameter | Value |
|-----------|-------|
| Loss function | BCEDiceLoss (0.5 x BCEWithLogitsLoss + 0.5 x Dice Loss) |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 2 (except EViT-UNet: 8 across 2 GPUs) |
| Hardware | Kaggle T4 GPU |

| Model-specific | Scheduler | Epochs | Crop | GPU |
|----------------|-----------|--------|------|-----|
| Attention U-Net | ReduceLROnPlateau | 50 | 572 | 1x T4 |
| CBAM U-Net | ReduceLROnPlateau | 15 | 572 | 1x T4 |
| SA U-Net | ReduceLROnPlateau | 10 | 572 | 1x T4 |
| ResNet34 Attn U-Net | ReduceLROnPlateau | 10 | 572 | 1x T4 |
| EViT-UNet | CosineAnnealingLR | 25 | 224 | 2x T4 |

---

## Observations

### Convergence behavior

The Attention U-Net, trained for 50 epochs, reached a validation Dice of 0.8960 with train and validation losses converging, indicating near-complete training. The ResNet34 Attention U-Net achieved 0.8675 validation Dice in only 10 epochs — the fastest convergence among all models — due to its pretrained backbone. CBAM U-Net, SA U-Net, and EViT-UNet were all still on an upward trajectory when training was stopped, and would benefit substantially from extended training.

### Boundary precision

The most notable performance gap is in boundary quality (HD95). The Attention U-Net (2.88 px) and ResNet34 Attention U-Net (3.48 px) produce boundaries that are nearly pixel-accurate, while the remaining models show HD95 values of 23-30 px. This suggests that Attention Gates on skip connections are particularly effective at preserving fine boundary information during the decoder upsampling path.

### EViT-UNet

The hybrid CNN-Transformer architecture was trained from scratch without pretrained weights, which is a significant disadvantage for a 55M-parameter model. Additionally, its fixed 224x224 input resolution provides less spatial context per sample compared to the 572x572 crops used by the other models. Despite this, the model showed consistent improvement through all 25 epochs (val Dice: 0.6281 to 0.8222) and would likely improve further with extended training and/or pretrained encoder initialization.

### Attention mechanism comparison

| Mechanism | Operates on | Applied at |
|-----------|-------------|------------|
| Attention Gate | Spatial regions | Skip connections |
| CBAM | Channel + spatial | Within encoder/decoder blocks |
| Spatial Attention | Spatial features | Within blocks |
| CCA | Channel cross-correlation | Skip connections |
| Self-Attention (4D) | Global pixel relationships | Deep encoder/decoder stages |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Dice Coefficient (F1) | Harmonic mean of precision and recall over segmented regions. Range [0, 1], higher is better. |
| IoU (Jaccard Index) | Intersection over union of predicted and ground truth masks. Stricter than Dice. |
| Precision (PPV) | Fraction of predicted positive pixels that are truly positive. |
| Recall (Sensitivity) | Fraction of actual positive pixels that are correctly predicted. |
| HD95 | 95th percentile Hausdorff Distance between prediction and ground truth boundaries, in pixels. Lower is better. |

---

## Repository Structure

```
.
├── unet_attention.ipynb                  # Attention U-Net
├── cbam-unet.ipynb                       # CBAM U-Net
├── unet-spatial-attention.ipynb          # SA U-Net
├── unet-resnet-attention-notebook.ipynb  # ResNet34 Attention U-Net
├── evit-unet.ipynb                       # EViT-UNet
└── README.md
```

Each notebook is self-contained and includes: model architecture definition, dataset loading, training loop, metric evaluation, and visualizations (training curves, violin/box plots, radar charts, boundary contour overlays).

---

## References

1. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI, 2015.
2. O. Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," MIDL, 2018.
3. S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," ECCV, 2018.
4. C. Guo et al., "SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation," ICPR, 2021.
5. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2016.
6. X. Li, W. Zhu, X. Dong, O. M. Dumitrascu, and Y. Wang, "EViT-UNet: U-Net Like Efficient Vision Transformer for Medical Image Segmentation on Mobile and Edge Devices," arXiv:2410.15036, 2024.
7. Y. Li et al., "EfficientFormerV2: Rethinking Vision Transformers for MobileNet Size and Speed," ICCV, 2023.
