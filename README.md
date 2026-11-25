---
title: Vision Mamba Image Captioning
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Vision Mamba Image Captioning

Image caption generation using a Vision Mamba Tiny encoder
and a Mamba-based text decoder.

## Architecture

- Encoder: Vision Mamba Tiny (24 bidirectional Mamba blocks)
- Visual adapter: 192 → 256
- Decoder: 4 Mamba blocks
- Dataset: Flickr8k
- Inference: CPU-only PyTorch

## Features

- Upload an image and generate an English caption
- No GPU required
- Pure PyTorch Mamba inference
- No mamba_ssm or Triton dependency

## Limitations

The model was trained on Flickr8k and may generate
inaccurate captions for unfamiliar images.
