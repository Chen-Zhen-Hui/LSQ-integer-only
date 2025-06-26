**Read this in other languages: [中文](README_zh.md).**
# LSQ-Integer-Only

A PyTorch-based LSQ (Learned Step Size Quantization) quantization training and integer inference framework that supports flexible bit-width quantization for weights and activations.

## Project Overview

This project implements the LSQ quantization algorithm and supports:
- **Quantization-Aware Training (QAT)**: Low-bit quantization training using LSQ algorithm
- **Integer Inference**: Pure integer arithmetic model inference
- **Model Architectures**: ResNet/VGG
- **Datasets**: CIFAR/ImageNet

## Key Features

### 🎯 Core Functionality
- **LSQ Quantization Algorithm**: Implementation of Learned Step Size Quantization
- **W/A Quantization**: Flexible bit-width selection for weight and activation quantization
- **Integer Inference**: Pure integer arithmetic model inference (fixed-point environment)
- **BatchNorm Fusion**: Automatic BatchNorm layer fusion for improved inference efficiency

## Project Structure

```
LSQ-integer-only/
├── main.py              # Floating-point model training script
├── q_main.py            # Quantization-aware training script
├── q_inference.py       # Integer inference script
├── models/              # Model definitions
│   ├── module.py        # LSQ quantization module implementation
│   ├── resnet.py        # ResNet models
│   ├── resnet_cifar10.py # CIFAR-10 specific ResNet
│   └── tiny_vgg.py      # TinyVGG model
├── logs/                # Floating-point training logs
├── qat_logs/            # Quantization training logs
└── README.md            # Project documentation
```

## Installation Requirements

### Environment Requirements
- Python 3.7+
- PyTorch 1.8+
- CUDA 11.0+

### Dependency Installation
```bash
pip install torch torchvision
pip install tensorboard
pip install tqdm
```

## Usage

### 1. Floating-Point Model Training

First, train a floating-point model as the starting point for quantization training:

```bash
# Train ResNet18 on CIFAR-10
python main.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.1 \
    --cuda 0 \
    --amp
```

### 2. Quantization-Aware Training

Perform quantization-aware training using the LSQ algorithm:

```bash
# W4A4 quantization training
python q_main.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-3 \
    --w-num-bits 4 \
    --a-num-bits 4 \
    --cuda 0
```

### 3. Integer Inference

Perform integer inference using the trained quantized model:

```bash
# Integer inference
python q_inference.py \
    --model resnet18 \
    --dataset cifar10 \
    --batch-size 128 \
    --w-num-bits 4 \
    --a-num-bits 4 \
    --cuda 0
```

## Parameter Description

### Training Parameters
- `--model`: Model type (resnet18, resnet34, resnet_cifar10, tiny_vgg)
- `--dataset`: Dataset (cifar10)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--lr`: Learning rate
- `--cuda`: GPU device ID (-1 for CPU)

### Quantization Parameters
- `--w-num-bits`: Weight quantization bit-width (default: 4)
- `--a-num-bits`: Activation quantization bit-width (default: 4)
- `--resume`: Resume training from checkpoint

## LSQ Algorithm Principle

LSQ (Learned Step Size Quantization) is a quantization method that learns quantization step sizes:

1. **Quantization Step Size Learning**: Treat quantization step size as a learnable parameter
2. **Gradient Scaling**: Adjust gradients based on quantization bit-width
3. **STE Approximation**: Use Straight-Through Estimator for gradient propagation

### Quantization Formula
```
q = round(clamp(x/α, qmin, qmax))
x_q = α * q
```

Where:
- `α` is the learnable quantization step size
- `qmin, qmax` are the quantization range
- `q` is the quantized integer

## Performance Optimization

### 1. Mixed Precision Training
Using AMP can significantly reduce memory usage and accelerate training:
```bash
python main.py --amp
```

### 2. BatchNorm Fusion
Automatic BatchNorm layer fusion during quantized inference to reduce computational overhead.

### 3. Integer Inference
Support for pure integer arithmetic, suitable for deployment on resource-constrained devices.

## Experimental Results

Floating-point model training for 200 epochs, floating-point model accuracy: 58.27%

Quantization-aware training for 100 epochs

Results on ImageNet dataset:

| W | A | Accuracy |
|---|---|----------|
| 4 | 4 | 52.22% |
| 4 | 6 | 58.38% |
| 6 | 4 | 53.31% |
| 8 | 8 | 60.23% | 