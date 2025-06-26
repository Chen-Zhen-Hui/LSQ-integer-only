# LSQ-Integer-Only

一个基于PyTorch的LSQ（Learned Step Size Quantization）量化训练和整数推理框架，支持自由旋转权重、激活值低比特量化。

## 项目简介

本项目实现了LSQ量化算法，支持：
- **量化感知训练（QAT）**：使用LSQ算法进行低比特量化训练
- **整数推理**：支持纯整数运算的模型推理
- **模型架构**：ResNet/VGG
- **数据集**：CIFAR/ImageNet

## 主要特性

### 🎯 核心功能
- **LSQ量化算法**：实现Learned Step Size Quantization
- **W/A量化**：权重和激活的量化比特可自由选择
- **整数推理**：支持纯整数运算的模型推理（纯定点环境）
- **BatchNorm融合**：自动融合BatchNorm层以提高推理效率


## 项目结构

```
LSQ-integer-only/
├── main.py              # 浮点模型训练脚本
├── q_main.py            # 量化感知训练脚本
├── q_inference.py       # 整数推理脚本
├── models/              # 模型定义
│   ├── module.py        # LSQ量化模块实现
│   ├── resnet.py        # ResNet模型
│   ├── resnet_cifar10.py # CIFAR-10专用ResNet
│   └── tiny_vgg.py      # TinyVGG模型
├── logs/                # 浮点训练日志
├── qat_logs/            # 量化训练日志
└── README.md            # 项目说明文档
```

## 安装要求

### 环境要求
- Python 3.7+
- PyTorch 1.8+
- CUDA 11.0+

### 依赖安装
```bash
pip install torch torchvision
pip install tensorboard
pip install tqdm
```

## 使用方法

### 1. 浮点模型训练

首先训练一个浮点模型作为量化训练的起点：

```bash
# 训练ResNet18在CIFAR-10上
python main.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.1 \
    --cuda 0 \
    --amp
```

### 2. 量化感知训练

使用LSQ算法进行量化感知训练：

```bash
# W4A4量化训练
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

### 3. 整数推理

使用训练好的量化模型进行整数推理：

```bash
# 整数推理
python q_inference.py \
    --model resnet18 \
    --dataset cifar10 \
    --batch-size 128 \
    --w-num-bits 4 \
    --a-num-bits 4 \
    --cuda 0
```

## 参数说明

### 训练参数
- `--model`: 模型类型 (resnet18, resnet34, resnet_cifar10, tiny_vgg)
- `--dataset`: 数据集 (cifar10)
- `--epochs`: 训练轮数
- `--batch-size`: 训练批次大小
- `--lr`: 学习率
- `--cuda`: GPU设备ID (-1表示使用CPU)

### 量化参数
- `--w-num-bits`: 权重量化位数 (默认4)
- `--a-num-bits`: 激活量化位数 (默认4)
- `--resume`: 从检查点恢复训练

## LSQ算法原理

LSQ（Learned Step Size Quantization）是一种学习量化步长的量化方法：

1. **量化步长学习**：将量化步长作为可学习参数
2. **梯度缩放**：根据量化位宽调整梯度
3. **STE近似**：使用Straight-Through Estimator进行梯度传播

### 量化公式
```
q = round(clamp(x/α, qmin, qmax))
x_q = α * q
```

其中：
- `α` 是可学习的量化步长
- `qmin, qmax` 是量化范围
- `q` 是量化后的整数

## 性能优化

### 1. 混合精度训练
使用AMP可以显著减少显存使用并加速训练：
```bash
python main.py --amp
```

### 2. BatchNorm融合
量化推理时自动融合BatchNorm层，减少计算开销。

### 3. 整数推理
支持纯整数运算，适合在资源受限的设备上部署。

## 实验结果
浮点模型训练200个 epoch，浮点模型精度：58.27%

量化感知训练100个 epoch

在ImageNet数据集上的结果：

| W | A | 准确率 |
|---|---|--------|
| 4 | 4 | 52.22% |
| 4 | 6 | 58.38% |
| 6 | 4 | 53.31% |
| 8 | 8 | 60.23% |
