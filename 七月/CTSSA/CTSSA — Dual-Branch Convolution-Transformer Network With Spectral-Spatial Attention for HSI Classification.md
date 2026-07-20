# CTSSA: Dual-Branch Convolution–Transformer Network With Spectral–Spatial Attention for Hyperspectral Image Classification

## 核心信息

- **论文**: Dual-Branch Convolution–Transformer Network With Spectral–Spatial Attention for Hyperspectral Image Classification
- **作者**: Yao Lu, Yongshan Zhang (通讯), Xinwei Jiang, Xiaobo Liu, Zhihua Cai (通讯)
- **机构**: 中国地质大学（武汉）计算机学院 / 自动化学院
- **期刊**: IEEE Transactions on Geoscience and Remote Sensing, Vol. 63, 2025
- **DOI**: 10.1109/TGRS.2025.3584668
- **关键词**: CNN, Transformer, 高光谱图像分类 (HSI), 金字塔光谱注意力 (PSAM), 中心 Transformer 编码器 (CenterTE)

## 原文摘要翻译

高光谱图像分类是遥感领域的关键任务，旨在利用高光谱图像中的光谱和空间信息为每个像素分配类别标签。近年来，许多深度学习方法（如 CNN 和 Transformer）已被应用于该任务并取得了显著成果。然而，现有的大多数基于图像块的深度学习方法往往忽略了中心像素与其周围像素之间的潜在关系。此外，高光谱图像独特的光谱特性——如相邻波段之间的高相关性和远距离波段之间的低依赖性——也需要特别关注。基于此，我们提出了一种新颖的双分支卷积-Transformer 网络，称为 CTSSA（Convolution–Transformer with Spectral–Spatial Attention），能够有效聚合局部和全局的光谱-空间特征。具体来说，CTSSA 包含两个核心模块：金字塔光谱注意力模块（PSAM）和中心 Transformer 编码器（CenterTE）。前者通过分层多尺度注意力机制提取高判别性的光谱特征，捕捉相邻光谱波段之间的细微差异；后者通过引入中心注意力机制改进原始 Transformer 编码器，建模中心像素与其周围像素之间的全局关系，从而提高分类精度并降低计算复杂度。在四个公开数据集（Salinas、Pavia University、Houston 2013 和 WHU-Hi-LongKou）上的实验结果表明，与九个其他网络相比，CTSSA 以更少的参数量和较高的效率取得了令人满意的性能。

## 创新点

1. **金字塔光谱注意力模块 (PSAM)**：提出了一种分层多尺度光谱分组策略 $H = \{1, 2, 3, 6\}$，将高光谱波段按不同粒度分组后独立计算注意力，再跨组融合。粗粒度分组抗噪性强、适合捕捉稳定趋势，细粒度分组能检测细微光谱变化，解决了现有方法无法有效建模波段间非均匀相关性的问题。

2. **中心引导的稀疏注意力机制 (CenterTE/CAM)**：不同于标准 Transformer 对所有 token 做全局自注意力的做法，CenterTE 显式以中心像素为 Query，仅计算其与周围像素的相似度（余弦相似度），将注意力从 $O(N^2)$ 降至 $O(N)$。这种中心先验的设计更贴合 HSI 分类中"中心像素是分类目标"的物理直觉，在降低计算量的同时提升了精度。

3. **双分支互补架构 (CenterTE + 3D-CNN)**：CenterTE 分支建模全局空间上下文，3D-CNN 分支捕获局部光谱-空间细节，两分支通过通道拼接和逐点卷积融合。消融实验证明这种互补设计比纯 CNN、纯 Transformer 或同质双分支效果更好。

4. **轻量化设计**：在保持高分类精度的同时，CTSSA 的参数量和 FLOPs 显著低于大多数对比方法，特别是在低训练样本条件下优势明显——在 LK 数据集仅用 0.1% 训练样本即达到 98.88% OA。

## 一句话总结

CTSSA 通过"金字塔光谱分组注意力 + 中心引导稀疏 Transformer + 3D-CNN 局部分支"的三合一设计，在四个 HSI 数据集上以更少参数超越了 9 种对比方法，核心洞察是：让 Transformer 只关注中心像素的全局关系，比让它对所有像素做全量自注意力更高效且更准确。

---

## 研究问题

高光谱图像（HSI）由数百个连续光谱波段组成，每个像素是一个高维光谱向量。HSI 分类的核心挑战在于：

- **光谱维度的非均匀相关性**：相邻波段高度相关，远距离波段弱相关。现有的全局自注意力（Transformer）或固定尺度 CNN 无法有效建模这种非均匀结构。
- **中心像素的角色被忽视**：大多数基于图像块（patch）的方法将块内所有像素平等对待，但实际上分类目标是中心像素，周围像素只是上下文。CNN 受限于局部感受野，Transformer 的全局自注意力又缺乏对中心像素的显式偏好。
- **小样本问题**：HSI 标注成本极高，训练样本通常仅占总像素的 0.1%~5%。

![Fig. 1](七月/CTSSA/images/page_003_fig_fig_1.png)

## 数据与任务定义

### 数据集

论文在四个公开 HSI 数据集上评估：

| 数据集 | 传感器 | 光谱波段 | 空间分辨率 | 类别数 | 标注像素 | 训练比例 |
|--------|--------|----------|------------|--------|----------|----------|
| Salinas (SA) | AVIRIS | 204 (224→去水汽) | 3.7m | 16 | 54,129 | 1% |
| Pavia University (PU) | ROSIS | 103 (115→去噪声) | 1.3m | 9 | 42,776 | 0.7% |
| Houston 2013 (HS) | CASI | 144 | 2.5m | 15 | 15,029 | 5% |
| WHU-Hi-LongKou (LK) | Headwall Nano | 270 | ~0.5m | 9 | 204,542 | 0.1% |

### 对比方法

9 种方法：**纯 CNN**（SSRN, FDSSC, DBDA）、**纯 Transformer**（SpectralFormer, morphFormer）、**CNN-Transformer 混合**（SSFTT, GSC-ViT, DCTN, DBCTNet）。

### 评价指标

**精度指标**：OA（总体精度）、AA（平均精度）、Kappa 系数；**效率指标**：参数量、FLOPs、推理时间。

### 训练配置

- 优化器: Adam，权重衰减 0.001
- 学习率: SA/PU/LK = 0.001, HS = 0.003，配合 cosine warmup 策略
- 损失函数: Focal Loss，$\gamma$ 按数据集分别设为 4.2/0.5/1.9/5.0
- Epochs: SA/PU/LK = 400, HS = 600
- Patch 大小: SA/PU = 13×13, HS = 15×15, LK = 9×9
- Batch size: 128

## 方法主线

### 整体架构

CTSSA 由三个核心组件构成，整体流程为：输入 HSI patch → PSAM（多尺度光谱特征提取）→ 双分支并行（CenterTE 全局建模 + 3D-CNN 局部提取）→ 特征融合 → 分类。

![Fig. 2](七月/CTSSA/images/page_004_fig_fig_2.png)

### 机制流程

1. **金字塔光谱分组与注意力 (PSAM)**：输入 HSI patch $X \in \mathbb{R}^{p \times p \times B}$（$B$ 为波段数）
- **Operation**: 按层次参数 $H = \{1, 2, 3, 6\}$ 将 $B$ 个波段分组。$H=1$ 不分组的全波段自注意力，$H=6$ 将波段分成 6 组分别做组内注意力。每组内通过 SAB 模块（见 Fig. 3）计算注意力权重：Query 直接使用原始输入（避免线性变换损失信息），Key 和 Value 通过 $1\times1$ 卷积得到，用余弦相似度替代点积计算注意力分数。多尺度输出通过 CAM 模块（见 Fig. 4）动态加权融合。
- **Output**: 多尺度光谱特征 $Y_{\text{PSAM}}$

![Fig. 3](七月/CTSSA/images/page_004_fig_fig_3.png)

![Fig. 4](七月/CTSSA/images/page_005_fig_fig_4.png)

2. **双分支并行处理**
   - **CenterTE 分支 (全局上下文)**: 以中心像素为 Query，仅计算中心像素与 patch 内所有像素的余弦相似度 → 稀疏注意力矩阵 → 加权聚合空间上下文。相比标准自注意力的 $O(N^2)$ 复杂度，CenterTE 为 $O(N)$。
- **3D-CNN 分支 (局部细节)**: $1\times1\times7$ Conv3D (通道调整+光谱降维) → $3\times3\times3$ DWConv3D (分组卷积提取局部空间-光谱特征) → Conv3D 整合 + 残差连接。
- **Output**: $Z_{\text{CenterTE}}$ 和 $Z_{\text{cnn}}$，形状均为 $(16, p \times p \times 1)$。

3. **特征融合与分类** 两分支输出沿通道维拼接 → $1\times1\times1$ Conv3D 融合 → 全局平均池化 (GAP) → 全连接层。
- **Output**: $d$ 类分类结果（$d$ 为类别数）。

### 关键公式

**SAB 注意力权重**:
$$A^n_i = \text{Softmax}(W_\theta \cdot \text{Concat}(K^n_i, Q^n_i)) \tag{1}$$

其中 $Q^n_i = X^n_i$（原始输入），$K^n_i = W_K * X^n_i$，$V^n_i = W_V * X^n_i$。Query 直接使用原始输入避免了特征变换造成的信息损失。

**Focal Loss**:
$$\mathcal{L}_{\text{focal}} = -\alpha_y (1 - p_y)^\gamma \log(p_y) \tag{12}$$

用于缓解 HSI 分类中的类别不平衡问题，$\alpha_y$ 按各类别样本比例倒数设定，$\gamma$ 聚焦难分样本。

## 关键结果

### 主要对比实验

| 数据集 | 指标 | CTSSA | 次优方法 | 提升 |
|--------|------|-------|----------|------|
| SA (1%) | OA | **98.93%** | SSFTT 97.96% | +0.97% |
| | AA | **99.41%** | — | — |
| | Kappa | **98.80%** | — | — |
| PU (0.7%) | OA | **99.25%** | DBCTNet 98.67% | +0.58% |
| | AA | **99.07%** | DBCTNet 98.10% | +0.97% |
| | Kappa | **99.00%** | DBCTNet 98.23% | +0.77% |
| HS (5%) | OA | **98.85%** | DBCTNet 98.03% | +0.82% |
| | AA | **98.63%** | DBCTNet 98.20% | +0.43% |
| | Kappa | **98.76%** | DBCTNet 97.87% | +0.89% |
| LK (0.1%) | OA | **98.88%** | DBCTNet 97.52% | +1.36% |
| | AA | **97.59%** | — | — |
| | Kappa | **98.53%** | — | — |

CTSSA 在所有四个数据集上全面领先。在 LK 仅用 0.1% 训练样本即达到 98.88% OA，优势最为显著（+1.36%）。

### 不同训练样本量下的表现

![Fig. 5](七月/CTSSA/images/page_009_fig_fig_5.png)

CTSSA 在所有训练比例下均保持最优。DBCTNet 和 DCTN 在充足训练样本时接近 CTSSA，但在低样本条件下精度下降明显——CTSSA 的小样本鲁棒性突出。

### 分类图可视化

![Fig. 6](七月/CTSSA/images/page_010_fig_fig_6.png)

![Fig. 7](七月/CTSSA/images/page_010_fig_fig_7.png)

![Fig. 8](七月/CTSSA/images/page_011_fig_fig_8.png)

![Fig. 9](七月/CTSSA/images/page_011_fig_fig_9.png)

CTSSA 的分类图噪声最少、边界最清晰。SpectralFormer（纯 Transformer）由于缺乏归纳偏置和训练样本不足，产生了大量噪声区域。CNN 方法（SSRN 等）虽精度不低，但因感受野受限，分类图区域连续性较差。DBCTNet 在 Salinas 和 LK 上产生了边缘像素误分类，论文归因于其对局部特征的主导和对中心像素关注不足。

### 计算效率

CTSSA 在精度和效率之间取得了良好平衡。虽然推理时间略高于纯 CNN 方法（光谱注意力机制引入额外开销），但参数量和 FLOPs 显著低于大多数混合方法（如 DCTN）。在 LK 数据集上，CTSSA 以 165.980M FLOPs 的代价换来了 1.36% OA 的提升，投入产出比合理。

## 深度分析

### 消融实验：各模块贡献

**PSAM 模块**:

![Fig. 10](七月/CTSSA/images/page_012_fig_fig_10.png)

移除 PSAM 后，四个数据集平均 OA 下降 1.61%、AA 下降 2.16%、Kappa 下降 1.48%。PSAM 的分层光谱分组策略既能保留全局特征，又能有效提取局部判别性特征。

**CenterTE 的 CAM 机制**: 将 CAM 替换为空间池化、标准自注意力 (SA)、多头自注意力 (MHSA) 后，CenterTE 在所有数据集上精度最高。相比次优的 SA 机制，OA 分别提升 4.01%/1.86%/3.25%/4.23%（SA/PU/HS/LK）。同时 CenterTE 参数量显著低于 SA 和 MHSA——稀疏注意力在降本增效方面效果明确。

**相似度函数选择**:

![Fig. 11](七月/CTSSA/images/page_013_fig_fig_11.png)

将 CAM 中的余弦相似度替换为欧氏距离后，所有数据集精度一致下降。特别是 LK 数据集的 AA 从 97.59% 骤降至 93.72%。余弦相似度对光谱曲线的形状相似性更敏感，且天然具有尺度不变性——这对受光照影响的高光谱数据至关重要。

**双分支结构**:

![Fig. 12](七月/CTSSA/images/page_013_fig_fig_12.png)

- 移除 CNN 分支：OA 平均 -2.36%
- 移除 CenterTE 分支：OA 平均 -1.83%
- 替换为双 CNN 或双 CenterTE：OA 分别 -0.98% 和 -1.53%

两个分支缺一不可，且**异构双分支（CNN + Transformer）优于同质双分支**——证明了局部/全局互补设计的有效性。

### 超参数敏感性

**层次尺度数 H**: 从单层到 8 层测试，$H = \{1, 2, 3, 6\}$（4 层）效果最优。层数过多（5~8 层）导致各尺度特征通道数减少，表示能力下降，精度反而回落。

**Patch 大小**:

![Fig. 13](七月/CTSSA/images/page_014_fig_fig_13.png)

最优 patch 大小因数据集而异：SA/PU = 13×13, HS = 15×15, LK = 9×9。这反映了不同数据集在空间分辨率、噪声水平和地物类型上的差异，需要灵活调整。

## 局限

1. **推理效率仍有提升空间**: CTSSA 的推理时间略高于纯 CNN 方法，光谱注意力机制的前向开销在实时场景下可能成为瓶颈。
2. **未探索自监督/迁移学习**: 论文仅在监督设定下验证，对于 HSI 标注稀缺的现实场景，自监督预训练或跨场景迁移是自然的下一步。
3. **双分支交互机制较简单**: 当前融合方式为通道拼接 + 逐点卷积，论文自述后续将探索更深层的特征交互机制。
4. **数据集偏小偏简单**: 四个数据集均为经典基准，缺乏更具挑战性的大规模、高异质性场景测试。
5. **缺少与最新大模型方法的对比**: 对比方法集中在 2023 年前的 CNN/Transformer 混合模型，未包含 2024-2025 年涌现的视觉基础模型（如 SpectralGPT 等）。

## 我的笔记

CTSSA 是一篇工程扎实的 HSI 分类方法论文，核心贡献在于两点洞察：(1) 高光谱波段间的非均匀相关性需要分层多尺度建模（PSAM），(2) 分类任务中中心像素的特殊地位可以通过稀疏注意力机制显式编码（CenterTE）。这两点都有充分的消融实验支撑。方法设计简洁、可复现性强，性能在四个数据集上全面超越当时的 SOTA。

对我而言最可借鉴的是 CenterTE 的设计理念——"让 Transformer 做减法"：不是把所有 token 两两算注意力，而是让 Query 只来自中心像素，这既是领域知识的注入（中心像素是分类目标），也是计算效率的优化（$O(N)$ 替代 $O(N^2)$）。这种"用任务先验约束注意力范围"的思路在遥感之外的密集预测任务中同样值得尝试。

## 引用

- 论文原始 PDF 见同目录下 `CTSSA.pdf`
- 对比方法引用参见原文 [34]-[55]
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
