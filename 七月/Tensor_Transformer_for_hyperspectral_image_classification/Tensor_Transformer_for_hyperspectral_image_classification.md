# Tensor Transformer for Hyperspectral Image Classification

**论文**: Tensor Transformer for hyperspectral image classification  
**作者**: Wei-Tao Zhang, Yv Bai, Sheng-Di Zheng, Jian Cui, Zhen-zhen Huang  
**机构**: 西安电子科技大学 信息力学与感知工程学院  
**期刊**: Pattern Recognition, 163 (2025) 111470  
**DOI**: [10.1016/j.patcog.2025.111470](https://doi.org/10.1016/j.patcog.2025.111470)  
**关键词**: 高光谱图像分类、张量自注意力机制、Tensor Transformer (TT)、长程空间-光谱特征

---

## 原文摘要翻译

高光谱图像（HSI）因包含数百个连续波段的丰富空间和光谱特征，被广泛用于实际分类任务。近年来，基于深度学习的 HSI 分类方法（如卷积神经网络和 Transformer）在分类任务中取得了良好表现。事实上，Transformer 类神经网络因其出色的长程特征提取能力，在 HSI 分类场景中通常优于卷积神经网络类方法。然而，基于 Transformer 的方法始终需要对原始三维 HSI 数据做序列化处理，这可能破坏空间-光谱结构特征。这一缺陷降低了分类精度。本文提出了一种用于 HSI 分类的张量 Transformer（Tensor Transformer，简称 TT）框架。TT 模型是一个端到端网络，直接以原始 HSI 张量数据作为输入样本，无需原始数据序列化。该框架的核心组件是张量自注意力机制（Tensor Self-Attention Mechanism，简称 TSAM），它使网络能够在保持样本内部固有结构关系的前提下，高效提取长程空间-光谱结构特征。通过在四个广泛使用的 HSI 数据集上进行大量实验，所提出的 TT 模型在区分光谱相似的地物类别方面，展现出优于当前最先进方法的分类性能。

## 创新点

1. **避免序列化的张量输入范式**：传统 Transformer 必须将三维 HSI 数据序列化为一维序列，导致三维空间-光谱结构信息丢失。TT 沿光谱维度切分为子张量块，直接以张量形式输入，完整保留了空间结构信息。

2. **张量自注意力机制（TSAM）**：用 Tucker 分解替代传统的 $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$ 矩阵乘法。TSAM 从张量的多个维度同时执行线性变换，将每个维度上的大参数矩阵分解为三个小因子矩阵，在保留多维结构特征的同时大幅减少参数量。

3. **参数效率显著**：TT 仅需 0.1M–0.3M 参数即可达到或超越参数量数倍乃至数十倍于己的对比方法（SpectralFormer 3.6M–7.6M, HiT 7.7M–16.3M），同时 FLOPs 和预测时间也处于合理范围。

## 一句话总结

> 用 Tucker 分解替代传统自注意力的矩阵乘法，让 Transformer 直接处理三维高光谱张量，避免了序列化带来的结构信息损失，在四个基准数据集上以更少参数实现了最优分类精度。

---

## 1. 问题背景

### 1.1 高光谱图像分类的挑战

高光谱图像（HSI）包含数百个连续光谱波段，每个像素实质上是一条高维光谱曲线。不同地物在不同波段上表现出差异化特征，这为精确地物分类创造了条件。HSI 分类广泛应用于生态科学、精准农业、森林监测和气候预测等领域。

### 1.2 现有方法的局限

- **传统机器学习**（SVM、随机森林、KNN）：无法提取深层次抽象特征，分类精度有限。
- **CNN 类方法**（3D-CNN、RSSAN、A2S2K-ResNet）：擅长捕获局部空间上下文，但对非相邻光谱波段之间的远距离依赖关系建模能力不足。
- **RNN 类方法**（MSLAN）：依赖光谱波段序列顺序，学习长程依赖困难，存在性能瓶颈。
- **Transformer 类方法**（SpectralFormer、HiT、morphFormer、SSFTT）：能捕获全局依赖，但**必须将三维 HSI 数据序列化为一维序列**，这会破坏三维空间中固有的空间-光谱结构特征，限制了分类精度的进一步提升。

**核心矛盾**：Transformer 擅长长程依赖建模，但序列化操作恰好破坏了它最需要利用的三维结构信息。

---

## 2. 方法

### 2.1 张量自注意力机制（TSAM）

TSAM 是 TT 的核心创新。与标准自注意力不同，TSAM 不要求输入数据序列化，而是直接在三维张量上操作。

**输入表示**：每个训练样本为三维张量 $\mathcal{X} = \{x_{i_1 i_2 i_3}\} \in \mathbb{R}^{I_1 \times I_2 \times I_3}$，其中 $I_1 \times I_2$ 为像素邻域空间尺寸，$I_3$ 为光谱波段数。

**光谱维切分**：沿光谱维度以重叠采样方式切分为 $N$ 个子张量 $\mathcal{Y}_1 \dots \mathcal{Y}_N \in \mathbb{R}^{I_1 \times I_2 \times J}$，每个子张量保留完整的空间信息和局部光谱信息。

**Tucker 变换替代矩阵乘法**：

传统自注意力需要构建 $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$ 三个大矩阵。TSAM 改用三组因子矩阵：

- $\mathbf{M}^q_1, \mathbf{M}^q_2, \mathbf{M}^q_3$ 构建 Query 张量组 $\mathcal{Q}_1 \dots \mathcal{Q}_N$
- $\mathbf{M}^k_1, \mathbf{M}^k_2, \mathbf{M}^k_3$ 构建 Key 张量组 $\mathcal{K}_1 \dots \mathcal{K}_N$
- $\mathbf{M}^v_1, \mathbf{M}^v_2, \mathbf{M}^v_3$ 构建 Value 张量组 $\mathcal{V}_1 \dots \mathcal{V}_N$

以 Query 为例，输出张量通过 Tucker 模式的模乘计算：

$$\mathcal{Q}_i = \mathcal{Y}_i \times_1 \mathbf{M}^q_1 \times_2 \mathbf{M}^q_2 \times_3 \mathbf{M}^q_3$$

其中 $\times_n$ 表示沿第 $n$ 维的模 $n$ 乘积，计算顺序不影响最终结果。

**注意力计算**：通过张量内积计算 Query 和 Key 之间的相关性：

$$\alpha_{nc} = \langle \mathcal{Q}_n, \mathcal{K}_c \rangle = \sum_{i_1=1}^{I_1} \sum_{i_2=1}^{I_2} \sum_{j=1}^{J} q_{i_1 i_2 j} \cdot k_{i_1 i_2 j}$$

得到相关系数矩阵 $\mathbf{U} \in \mathbb{R}^{N \times N}$，经 softmax 归一化后与 Value 张量组加权求和，得到 TSAM 最终输出。

**参数效率分析**：标准自注意力的 $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$ 参数量随光谱维度 $I_3$ 呈二次增长。TSAM 将每个大矩阵分解为三个小因子矩阵，参数量从 $O(I_3^2)$ 降至 $O(I_1^2 + I_2^2 + J^2)$，在光谱维度较高时优势显著。
![Fig. 1](七月/Tensor_Transformer_for_hyperspectral_image_classification/images/page_003_fig_fig_1.png))

### 2.2 张量编码器层（TEL）

TEL 模块基于 TSAM 构建，结构类似于标准 Transformer 的编码器层：
![Fig. 2](七月/Tensor_Transformer_for_hyperspectral_image_classification/images/page_003_fig_fig_2.png))

流程：输入张量 → LayerNorm → TSAM → 残差加法 → LayerNorm → 残差加法 → 输出。

关键特性：输入和输出维度完全一致，TEL 可堆叠多层以完成深层特征编码。

### 2.3 TT 网络整体架构
![Fig. 3](七月/Tensor_Transformer_for_hyperspectral_image_classification/images/page_004_fig_fig_3.png))

TT 网络由多个 TEL 模块堆叠后接全连接分类层组成。输入为原始三维 HSI 张量样本，经过多层 TEL 编码后，将特征表示送入分类层，输出类别预测。整个流程端到端训练，无需 PCA 降维或序列化预处理。

---

## 3. 实验

### 3.1 数据集

| 数据集 | 传感器 | 光谱波段 | 空间分辨率 | 类别数 | 特点 |
|--------|--------|----------|------------|--------|------|
| Salinas (SA) | AVIRIS | 224 (去除水汽吸收后 204) | 3.7 m | 16 | 农业区域 |
| Pavia University (PU) | ROSIS | 103 | 1.3 m | 9 | 城市区域 |
| Longkou (LK) | 无人机高光谱 | 9 | — | 9 | 小样本场景 |
| Houston (Hou) | ITRES-CASI 1500 | 144 | 2.5 m | 15 | 复杂城市区域 |
![Table 1](七月/Tensor_Transformer_for_hyperspectral_image_classification/images/page_007_fig_table_1.png))
![Table 2](七月/Tensor_Transformer_for_hyperspectral_image_classification/images/page_007_fig_table_2.png))

### 3.2 主要结果

**SA 数据集**：TT 取得 OA 99.13%，优于 SSFTT (98.38%) 和 A2S2K-ResNet (99.08%)。

**PU 数据集**：TT 在 9×9 空间窗口 + 12 层编码器配置下取得 OA 97.12%，优于 SSFTT (96.52%) 和 morphFormer (95.60%)。

**LK 小样本数据集**：每类仅 20 个训练样本。TT 取得 OA 95.42%，与擅长小样本的 A2S2K-ResNet (95.45%) 持平，且 Kappa 更高 (94.03% vs 93.84%)。这表明 TT 即使在小样本条件下也具备竞争力。
![Fig. 11](七月/Tensor_Transformer_for_hyperspectral_image_classification/images/page_009_fig_fig_11.png))

**Hou 数据集**：TT 取得 OA 98.44%，超越 SSFTT (98.11%) 和 A2S2K-ResNet (97.11%)，在 15 个类别中多类达到 100% 精度。

### 3.3 t-SNE 特征可视化
![Fig. 10](七月/Tensor_Transformer_for_hyperspectral_image_classification/images/page_008_fig_fig_10.png))

在 PU 数据集上，TT 的特征聚类效果显著优于对比方法：类间方差最大、类内方差最小。相比之下，SAE-CNN 和 RSSAN 无法有效区分不同类别特征，A2S2K-ResNet 和 morphFormer 在 "Bricks" 与 "Gravel" 两个类别上仍存在混淆。

### 3.4 计算效率分析

| 方法 | 参数量 | FLOPs (PU) | 预测时间 (PU) | 类型 |
|------|--------|-----------|--------------|------|
| SAE-CNN | ~0.1M | 3.6M | 58.5s | CNN |
| A2S2K-ResNet | ~0.3M–0.5M | 74M–141M | 10.6–144.9s | Attention-CNN |
| SpectralFormer | 3.6M–7.6M | 155M–359M | 15.2–182.8s | Transformer |
| HiT | 7.7M–16.3M | 257M–624M | 15.7–605.6s | Transformer |
| SSFTT | ~0.2M | 4.2M–18.5M | 7.3–96.3s | Transformer |
| **TT** | **~0.1M–0.3M** | **22M–54M** | **9.2–148.3s** | **Tensor Transformer** |

TT 以最少或接近最少的参数量，在大多数数据集上取得最高分类精度。SSFTT 虽然参数更少且更快，但它依赖 PCA 预降维，破坏了空间-光谱特征结构；TT 保留了原始数据结构，因此分类精度更高。TT 的 FLOPs 和预测时间高于 SSFTT 是因为输入样本保留了完整的三维空间-光谱结构。

### 3.5 超参数分析

**学习率**：在四个数据集上实验表明，lr = 0.001 时 TT 分类效果最佳。lr 过小（0.0001）导致收敛不足，lr 过大导致训练振荡。

**输入光谱尺寸**：以 Hou 数据集为例，当光谱长度 = 54、采样步长 = 30 时性能最优（OA 96.15%）。光谱长度过短导致子块过多、计算量增大；光谱长度过长则无法捕获细微光谱差异。
![Fig. 14](七月/Tensor_Transformer_for_hyperspectral_image_classification/images/page_011_fig_fig_14.png))

**空间窗口与编码器层数**：以 PU 数据集为例，9×9 空间窗口 + 12 层 TEL 是最优配置（OA 97.12%, 0.3M 参数, 110.2M FLOPs）。窗口太小丢失空间细节，太大引入噪声。编码器层数超过 12 层后收益递减，计算代价却显著上升。
![Table 5](七月/Tensor_Transformer_for_hyperspectral_image_classification/images/page_011_fig_table_5.png))

### 3.6 消融实验

将 TSAM 分别替换为仅关注空间注意力的 Spatial Transformer 和仅关注光谱注意力的 Spectral Transformer：

| 方法 | SA | PU | LK | Hou |
|------|-----|-----|-----|------|
| Spatial Transformer | 96.82 | 93.76 | 96.27 | 94.43 |
| Spectral Transformer | 95.34 | 91.25 | 93.42 | 93.71 |
| **TT** | **99.13** | **97.12** | **98.20** | **96.15** |
![Fig. 15](七月/Tensor_Transformer_for_hyperspectral_image_classification/images/page_012_fig_fig_15.png))

t-SNE 可视化进一步证实：TT 的聚类效果显著优于仅使用空间或光谱注意力的变体，验证了 TSAM 联合提取空间-光谱长程结构特征的有效性。
![Table 6](七月/Tensor_Transformer_for_hyperspectral_image_classification/images/page_011_fig_table_6.png))

---

## 4. 深度分析

### 4.1 TSAM 的数学本质

TSAM 可以理解为**在张量 Tucker 分解框架下重新定义的自注意力机制**。传统自注意力的 $\mathbf{Q} = \mathbf{X} \mathbf{W}_q$ 是矩阵空间中的线性变换，而 TSAM 的 Tucker 模乘是在张量空间中的多线性变换。这种变换天然保持了张量各维度之间的结构性关联——这正是 HSI 数据中空间邻域关系与光谱连续性的数学表达。

### 4.2 为什么避免序列化如此重要

HSI 的三维结构并非冗余——邻域像素在空间上形成纹理和形状信息，相邻波段在光谱上形成物质吸收/反射的连续特征。序列化操作将 $(I_1, I_2, I_3)$ 的三维结构强行展平为 $(I_1 \cdot I_2 \cdot I_3)$ 的一维序列，意味着原本相邻的空间位置和相邻的光谱波段被随意打散。TSAM 通过沿光谱维度切分（保留每个子块内的空间完整性）和 Tucker 变换（多维度同时编码），在数学上保证了结构信息的不丢失。

### 4.3 与现有方法的本质区别

- **vs SpectralFormer/SSFTT**：两者都需要序列化或 PCA 降维，TT 不需要。
- **vs HiT**：HiT 使用 3D 卷积投影 + 深度可分离卷积编码局部特征，本质上是 CNN+Transformer 的混合方案，而 TT 是纯 Transformer 架构，对长程依赖的建模更彻底。
- **vs A2S2K-ResNet**：A2S2K 以 CNN 为骨架仅在特定模块引入注意力，TT 以 TSAM 为核心构造编码器，更充分利用了 Transformer 的长程建模优势。

### 4.4 小样本性能的启示

TT 在 LK 数据集上（每类仅 20 个训练样本）与 AES2K-ResNet 持平，这很不寻常——Transformer 通常被认为是"数据饥渴"的。可能的原因是 TSAM 的 Tucker 分解天然带有低秩正则化效果，因子矩阵结构起到了隐式参数共享的作用，减轻了小样本条件下的过拟合风险。

---

## 5. 局限与展望

### 5.1 论文明确指出的局限

- **输入光谱尺寸需手动调参**：不同数据集的最优光谱长度和采样步长不同，需要逐数据集实验确定，缺乏自适应能力。作者计划开发自适应光谱尺寸模块。
- **计算效率可进一步优化**：TT 的 FLOPs 和预测时间高于 SSFTT（因为保留了完整三维结构），需要优化参数数量和计算效率。

### 5.2 未明确讨论但值得关注的问题

- **光谱切分策略的通用性**：当前采用固定长度 + 固定步长的重叠采样，这种均匀切分是否能适应光谱分辨率差异较大的数据集（如 9 波段的 LK vs 224 波段的 SA）值得进一步研究。
- **TSAM 中的因子矩阵秩选择**：论文未讨论 Tucker 分解中因子矩阵 $\mathbf{M}_1, \mathbf{M}_2, \mathbf{M}_3$ 的秩如何确定，是否均使用满秩？低秩近似的理论依据可以进一步挖掘。
- **多模态扩展**：TT 的设计天然适合多维张量输入（如 SAR+HSI 融合、时序 HSI），但论文未探讨这一方向。

---

## 6. 总结

这篇论文提出的 Tensor Transformer (TT) 解决了 HSI 分类中 Transformer 需要序列化输入导致结构信息损失的根本问题。核心贡献是 TSAM——用 Tucker 分解替代传统自注意力的矩阵乘法，使得网络可以直接处理三维张量输入。实验验证全面，在四个数据集上均取得最优或接近最优结果，消融实验清晰证明了 TSAM 联合空间-光谱建模的有效性。TT 以 0.1M–0.3M 的参数规模超越了参数量数倍于己的 SpectralFormer 和 HiT，具有很好的实用价值。

**适合复现的场景**：有 HSI 数据且需要高精度分类（尤其是光谱相似地物区分），对参数效率有要求的任务。

**不适合的场景**：对推理速度极度敏感且可接受一定精度损失的部署场景（此时 SSFTT 可能更合适）。

---

## 参考文献

核心引用：

- [18] Vaswani et al., "Attention Is All You Need", NeurIPS 2017 — 标准 Transformer
- [27] Dosovitskiy et al., "An Image is Worth 16x16 Words", 2020 — ViT
- [29] Hong et al., "SpectralFormer", IEEE TGRS 2021 — HSI Transformer 先驱
- [30] Yang et al., "HiT", IEEE TGRS 2022 — 3D 卷积 + Transformer 混合方案
- [32] Sun et al., "SSFTT", IEEE TGRS 2022 — 光谱-空间特征标记化 Transformer
- [26] Roy et al., "A2S2K-ResNet", IEEE TGRS 2020 — 注意力增强 ResNet
- [38] Zhang et al., "A Full Tensor Network for HSI Classification", IEEE GRSL 2022 — 张量网络前作
