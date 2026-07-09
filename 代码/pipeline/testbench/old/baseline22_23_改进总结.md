# Baseline22 & Baseline23 改进总结

## 任务背景

模型用于 **海面溢油检测**，核心挑战：
| 挑战 | 描述 | 对模型设计的影响 |
|------|------|------------------|
| 二分类 | 海水(0) vs 溢油(1) | 输出通道=2 |
| 极端不平衡 | 海水:溢油 ≈ **99:1** | 需要聚焦于少数类（溢油）的机制 |
| 长尾分布 | 溢油在图中分布不均匀，少量大油膜、大量小油斑 | 需要多尺度特征提取 |

---

## Baseline22 — 基于 Baseline20 的改进

**Baseline20 基线**: UNet3Layer + SnakeConvUnit + CoordAtt + 注意力先验(均值) + base=24 + 每层不同头数

### 改进点

| 模块 | 原版 (baseline20) | 改进版 (baseline22) | 针对问题的动机 |
|------|-------------------|---------------------|----------------|
| **SpectralWeightedPrior** | `torch.mean(x, dim=1, keepdim=True)` — 简单全局平均，丢失光谱差异 | 可学习 1×1 卷积融合多光谱通道生成先验图 | 油膜和水体在不同光谱波段有显著差异，learnable weighted sum 保留光谱判别信息 |
| **SEBlock** | 无 | 在 SnakeConvUnit 输出后插入 Squeeze-and-Excitation 通道注意力 (reduction=16) | 通道维度上增强溢油相关特征、抑制背景噪声，缓解类别不平衡 |
| **FocalGate** | 标准 Attention (v = v × priori) | Value 调制后再经 FocalGate: `v = FocalGate(v × priori)`，即 `sigmoid(linear(v)) × v` | 让模型聚焦于难分样本（溢油区域），缓解 99:1 不平衡 |
| **Base channels** | base=24 | base=26 | 适度增加模型容量，学习更精细的油膜特征 |

### 参数量变化

| 模型 | 参数量 | 增量 |
|------|--------|------|
| baseline20 | ~1.39M | — |
| **baseline22** | **~1.83M** | **+0.44M (+31.7%)** |

主要增量来自 SpectralWeightedPrior (增加的 Conv2d) + SEBlock (fc 层) + FocalGate (linear) + base 从 24 提至 26 带来的通道级联增长。

---

## Baseline23 — 基于 Baseline21 的改进

**Baseline21 基线**: UNet3Layer + SnakeConvUnit + CoordAtt + SobelPrior (边缘先验) + base=24 + 每层不同头数

### 改进点

| 模块 | 原版 (baseline21) | 改进版 (baseline23) | 针对问题的动机 |
|------|-------------------|---------------------|----------------|
| **MultiScaleSobelPrior** | 单尺度 Sobel (3×3)，仅提取单一尺度边缘 | 3×3、5×5、7×7 三种 Sobel 并行提取边缘，1×1 conv 融合 | 大油膜产 coarse 边缘(大尺度 Sobel 响应强)，小油斑产 fine 边缘(小尺度 Sobel 响应强)；多尺度适配长尾分布 |
| **DilatedSnakeConvUnit** | SnakeConvUnit (普通卷积，感受野=3) | 空洞蛇形卷积，支持 dilation 参数；encoder 越深 dilation 越大 (1→1→2→2) | 扩大感受野捕获溢油上下文，改善稀疏小油斑的检测 |
| **GatedSkipFusion** | 直接 `torch.cat([dec, enc], dim=1)` 后过 DoubleConv | 在每层解码器输出后插入门控融合: `gate = sigmoid(W_enc×enc + W_dec×dec)`; `output = enc×gate + dec×(1-gate)` | 让解码器自适应选择编码器有用特征，抑制背景噪声传递 |
| **SnakeCoordAttViTBlock** | 使用 SnakeConvUnit | 替换为 DilatedSnakeConvUnit (渐进式 dilation) | 深层获取更大感受野，匹配油膜多尺度分布 |

### 参数量变化

| 模型 | 参数量 | 增量 |
|------|--------|------|
| baseline21 | ~1.39M | — |
| **baseline23** | **~1.57M** | **+0.18M (+12.9%)** |

增量来自 MultiScaleSobelPrior (fusion conv) + GatedSkipFusion (3× 2×Conv2d 1×1)，DilatedSnakeConvUnit 仅增加 dilation 几乎不增参数量。

---

## 改进汇总对比

```
Baseline20 ───→ Baseline22
  │                  │
  │ priori=mean      │ SpectralWeightedPrior (learnable)
  │ base=24          │ base=26
  │ standard attn    │ + FocalGate (hard-sample focusing)
  │ no SE            │ + SEBlock (channel recalibration)
  │                   │
  │ ~1.39M params    │ ~1.83M params
  │                   │
  v                  v

Baseline21 ───→ Baseline23
  │                  │
  │ SobelPrior(3×3)  │ MultiScaleSobelPrior(3×3+5×5+7×7)
  │ SnakeConvUnit    │ DilatedSnakeConvUnit (progressive dilation)
  │ cat skip         │ + GatedSkipFusion (selective feature passing)
  │                   │
  │ ~1.39M params    │ ~1.57M params
  │                   │
  v                  v
```

### 使用方式

在 `IP_train.py` 的 `TRAIN_MODELS` 列表中指定：

```python
TRAIN_MODELS = [
    "baseline22",   # 基于 baseline20 的改进（光谱加权先验 + 聚焦门控）
    "baseline23",   # 基于 baseline21 的改进（多尺度 Sobel + 空洞蛇形卷积）
]
```

运行时确保模块可导入（`testbench.baseline22.build_tri_branch_net` 和 `testbench.baseline23.build_tri_branch_net` 会被 `resolve_tri_branch_builder` 自动发现）。
