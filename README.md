# Deepfake Detection using Fine-Tuned ResNet50

A CNN-based classifier that distinguishes real human face photographs from AI-generated (StyleGAN) fake faces, with Grad-CAM explainability to visualize what the model actually learned.

## Overview

This project fine-tunes a pretrained ResNet50 on the [140k Real and Fake Faces dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) to classify face images as real or AI-generated. Beyond raw accuracy, Grad-CAM is used to verify the model is attending to genuine facial regions (eyes, skin texture, blending boundaries) rather than exploiting shortcut artifacts like compression differences between real and synthetic image sources.

## Dataset

- **Source:** 140k Real and Fake Faces (real photos from Flickr-Faces-HQ, fakes generated via StyleGAN)
- **Split:** 100,000 train / 20,000 validation / 20,000 test, perfectly balanced (50/50 real-fake in each split)
- **Note on scope:** this dataset represents *AI-generated face detection* (StyleGAN vs real photo) rather than *face-swap video deepfake detection* — a narrower, related problem to the classic "deepfake" use case. This distinction matters for interpreting the results below.

## Approach

**Architecture:** ResNet50 (ImageNet-pretrained), adapted for binary classification via a custom linear output head.

**Two-phase fine-tuning:**
1. **Phase 1 — Feature extraction:** Backbone frozen, only the new classification head trained. Reached ~89.7% validation accuracy.
2. **Phase 2 — Fine-tuning:** Unfroze the final residual block (`layer4`) and the head, trained further at a lower learning rate (1e-5) to let high-level features adapt specifically to GAN-artifact patterns.

**Preprocessing:** Images resized to 224×224, normalized with ImageNet mean/std. Horizontal flip augmentation applied during training only.

**Loss/Optimizer:** `BCEWithLogitsLoss`, Adam optimizer.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **99.02%** |
| Test Loss | 0.0281 |

Test set was held out entirely from training and hyperparameter decisions.

## Explainability — Grad-CAM

To check whether the model learned genuine facial cues rather than shortcut artifacts, Grad-CAM was applied to the final convolutional block (`layer4`) to visualize which image regions most influenced each prediction.

![Grad-CAM Example 1](GRAD%20CAM.png)
![Grad-CAM Example 2](GRAD%20CAM%20!.png)

Across multiple samples spanning both classes, attention consistently concentrated on facial regions — eyes, nose bridge, cheek/skin texture areas — rather than background or image borders, suggesting the model is picking up on plausible content-relevant signals rather than a positional or compression-based shortcut.

## Honest Limitations

- This model detects **StyleGAN-generated faces**, not face-swap or reenactment-based deepfakes (e.g., FaceForensics++-style manipulations) — a different and generally harder problem.
- High accuracy on this dataset partly reflects the fact that real (photographed) and fake (GAN-generated) images can differ in subtle low-level statistics beyond semantic "fakeness," which Grad-CAM helps but cannot fully rule out.
- Frame/image-level only — no temporal analysis, since this is a static-image classification task, not video.

## Tech Stack

- PyTorch, torchvision (ResNet50, ImageFolder, DataLoader)
- Grad-CAM (custom implementation via forward/backward hooks)
- Trained on Google Colab (T4 GPU)

## Future Work

- Extend to face-swap deepfake datasets (FaceForensics++, Celeb-DF) for broader manipulation-type coverage
- Streamlit demo app for interactive upload → prediction + Grad-CAM overlay
- Compare against XceptionNet (the architecture used in the original FaceForensics++ paper)
