# Nuclei Segmentation on TNBC Histopathology — A Comparative Study of U-Net Variants

This repository contains implementations of nine deep learning architectures for nuclei segmentation, benchmarked on the TNBC (Triple Negative Breast Cancer) histopathology dataset. Each model is implemented as a self-contained Jupyter notebook with training, evaluation, and visualization.

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

### Work 2: Architectures

| Model | Dice (F1) | IoU | Precision | Recall | HD (px) | Params | Epochs Trained |
|:------|:---------:|:---:|:---------:|:------:|:-------:|:------:|:--------------:|
| SE-UNet | **0.8350** | **0.7257** | 0.8482 | 0.8268 | **53.67** | 75M | 12 |
| TransRes-UNet | 0.8337 | 0.7225 | **0.8559** | 0.8179 | 80.94 | 110M | 12 (10+2) |
| Base U-Net | 0.8303 | 0.7195 | 0.8391 | **0.8284** | 70.90 | 31M | 10 |
| DA-UNet | 0.7614 | 0.6222 | 0.8311 | 0.7108 | 77.54 | **21M** | 10 |

**Note on HD metric:** The Hausdorff Distance reported above is the full (max) directed Hausdorff distance in pixels, computed using `scipy.spatial.distance.directed_hausdorff`. This differs from the HD95 (95th percentile) reported in the table above. The values are not directly comparable across the two tables. All four models were trained on 640×640 crops with batch size 6 (TransRes-UNet: batch size 2 with gradient accumulation × 4, simulating effective batch 8). TransRes-UNet used a 10-epoch frozen backbone phase followed by 2 epochs of full fine-tuning at 10× reduced learning rate.

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

## Work 2: Architectures

### 6. Base U-Net (`unet-base.ipynb`)

Vanilla U-Net reimplemented from scratch with a 4-level encoder-decoder and transposed convolution upsampling. Serves as the baseline for comparing all attention and hybrid variants in this work. Each encoder block consists of two Conv-BN-ReLU operations followed by MaxPooling; the decoder mirrors this with skip concatenation.

- **Skip mechanism:** Standard concatenation
- **Parameters:** ~31M
- **Input resolution:** 640 x 640
- **Reference:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI, 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

### 7. SE-UNet (`se-unet.ipynb`)

U-Net with Squeeze-and-Excitation (SE) blocks embedded into both the encoder and decoder paths. After each convolutional block, global average pooling produces a channel descriptor, which is recalibrated through a two-layer FC bottleneck (reduction ratio applied) and multiplied back as channel-wise attention weights. A dilated convolution bottleneck replaces the standard bridge. SE blocks are also applied in the decoder upsampling path (`UNetDec`).

- **Skip mechanism:** Standard concatenation (SE recalibration within encoder/decoder blocks)
- **Parameters:** ~75M
- **Input resolution:** 640 x 640
- **Reference:** Iyer et al., "Squeeze Excitation Embedded Attention UNet for Brain Tumor Segmentation," arXiv:2305.07850, 2023. [arXiv:2305.07850](https://arxiv.org/abs/2305.07850)

### 8. DA-UNet (`daunet.ipynb`)

A lightweight U-Net variant combining two novel components: **Deformable V2 Convolutions** in the bottleneck (using a learned offset field + modulator to dynamically adapt the sampling grid to irregular nucleus shapes) and **SimAM** (Simple, Parameter-Free Attention Module) applied to all four skip connections. SimAM computes a 3D neuron importance score from feature statistics without adding any learnable parameters. The bottleneck uses a Conv→Deformable Conv→Conv structure with channel bottleneck compression.

- **Skip mechanism:** SimAM attention on skip connections (parameter-free)
- **Bottleneck:** Deformable V2 Conv (3×3 kernel) with channel compression (÷4)
- **Parameters:** ~21M (lightest model in this study)
- **Input resolution:** 640 x 640
- **Reference:** Ghosh et al., "DAUNet: A Lightweight UNet Variant with Deformable Convolutions and Parameter-Free Attention for Medical Image Segmentation," arXiv:2512.07051, 2024. [arXiv:2512.07051](https://arxiv.org/abs/2512.07051)

### 9. TransRes-UNet (`transres-unet.ipynb`)

Hybrid CNN-Transformer architecture combining a ResNet-50 backbone and ViT-B/16 encoder with a residual U-Net decoder. The ResNet-50 extracts multi-scale feature maps that serve as skip connections, while the final feature map is tokenized into patches and processed by a 12-layer Transformer (hidden size 768, 12 heads). The decoder uses `ResDecoderBlock`s (transposed conv + residual double conv) to progressively upsample and fuse features. The pretrained `R50+ViT-B_16.npz` weights were loaded for the encoder. Training used a **hybrid frozen/unfrozen strategy**: the backbone was frozen for 10 epochs (decoder only), then fully unfrozen for 2 epochs at 10× reduced LR. Mixed Precision (AMP) and gradient accumulation (×4 steps, effective batch 8) were used to manage the ~110M parameter model on 2× T4 GPUs.

- **Skip mechanism:** Residual decoder with ResNet skip connections
- **Parameters:** ~110M
- **Input resolution:** 640 x 640
- **Pretrained:** R50+ViT-B/16 (ImageNet-21k)
- **Multi-GPU:** DataParallel across 2x T4 GPUs; AMP + gradient accumulation (steps=4)
- **Reference:** Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation," arXiv:2102.04306, 2021. [arXiv:2102.04306](https://arxiv.org/abs/2102.04306)

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
| Base U-Net | ReduceLROnPlateau | 10 | 640 | 1x T4 |
| SE-UNet | ReduceLROnPlateau | 12 | 640 | 1x T4 |
| DA-UNet | ReduceLROnPlateau | 10 | 640 | 1x T4 |
| TransRes-UNet | ReduceLROnPlateau | 12 (10 frozen + 2 unfrozen) | 640 | 2x T4 + AMP |

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
| SE Block | Channel recalibration | Within encoder/decoder blocks |
| SimAM | Parameter-free 3D neuron score | Skip connections |
| Deformable Conv | Adaptive spatial sampling | Bottleneck |

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
├── unet-base.ipynb                       # Base U-Net (this work)
├── se-unet.ipynb                         # SE-UNet (this work)
├── daunet.ipynb                          # DA-UNet (this work)
├── transres-unet.ipynb                   # TransRes-UNet (this work)
└── README.md
```

Each notebook is self-contained and includes: model architecture definition, dataset loading, training loop, metric evaluation, and visualizations (training curves, violin/box plots, radar charts, boundary contour overlays).

---

## References

1. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI, 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
2. O. Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," MIDL, 2018.
3. S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," ECCV, 2018.
4. C. Guo et al., "SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation," ICPR, 2021.
5. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2016.
6. X. Li, W. Zhu, X. Dong, O. M. Dumitrascu, and Y. Wang, "EViT-UNet: U-Net Like Efficient Vision Transformer for Medical Image Segmentation on Mobile and Edge Devices," arXiv:2410.15036, 2024.
7. Y. Li et al., "EfficientFormerV2: Rethinking Vision Transformers for MobileNet Size and Speed," ICCV, 2023.
8. S. Iyer et al., "Squeeze Excitation Embedded Attention UNet for Brain Tumor Segmentation," arXiv:2305.07850, 2023. [arXiv:2305.07850](https://arxiv.org/abs/2305.07850)
9. S. Ghosh et al., "DAUNet: A Lightweight UNet Variant with Deformable Convolutions and Parameter-Free Attention for Medical Image Segmentation," arXiv:2512.07051, 2024. [arXiv:2512.07051](https://arxiv.org/abs/2512.07051)
10. J. Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation," arXiv:2102.04306, 2021. [arXiv:2102.04306](https://arxiv.org/abs/2102.04306)
