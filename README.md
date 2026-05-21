---
title: Wrist Fracture Detector
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
---

# 🩻 Wrist Fracture Detector

An AI-powered wrist X-ray classifier using **DenseNet121 Transfer Learning** with **CBAM Attention** mechanism.

## Features
- **Binary classification**: Fracture vs Normal
- **Grad-CAM explainability**: Visual heatmap showing which region the model focused on
- **Confidence scores**: Fracture and Normal probability bars
- **Custom preprocessing**: CLAHE enhancement + Sobel edge detection

## Model Architecture
- Base: DenseNet121 (pretrained on ImageNet)
- Attention: CBAM (Channel + Spatial Attention)
- Loss: Focal Loss (gamma=2, alpha=0.60)
- Input: 320×320 RGB (3-channel preprocessed X-ray)
- Trained on: MURA v1.1 dataset (Wrist subset)

## ⚠️ Disclaimer
This tool is for **research and educational purposes only**.  
It is not a substitute for professional medical diagnosis.  
Always consult a qualified radiologist or physician.
