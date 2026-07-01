---
tags:
  - 引入物理特性
aliases: PINNs for fluid mechanics review
---

# Physics-informed neural networks (PINNs) for fluid mechanics: A review

## 核心信息

- 标题: Physics-informed neural networks (PINNs) for fluid mechanics: A review
- 作者: Shengze Cai, Zhiping Mao, Zhicheng Wang, Minglang Yin, George Em Karniadakis
- 机构: Brown University, Xiamen University, Dalian University of Technology
- 年份: 2021
- 期刊: Acta Mechanica Sinica
- DOI: 10.1007/s10409-021-0xxxx-x
- arXiv: 2105.09506
- 类型: 综述 + 新结果
- 代码: 未提供

## 摘要

本文是由 Karniadakis 课题组撰写的一篇综述，系统回顾了物理信息神经网络在流体力学中的应用。文章从传统计算流体力学（CFD）的三个核心瓶颈出发——数据融合困难、网格生成复杂、反问题代价高昂——引出物理信息神经网络作为互补求解范式。论文首先介绍基本原理，随后展开不可压缩流、可压缩流和生物医学流动的建模方法，重点讨论了域分解策略（子域网络耦合方法）在处理复杂几何中的作用。最后通过三维圆柱尾流、超声速钝体绕流和血栓变形三个案例展示了该方法在反问题中的独特价值。

## 原文摘要翻译

尽管过去五十年在使用 Navier-Stokes 方程（NSE）数值离散化模拟流动问题方面取得了重大进展，我们仍然无法将噪声数据无缝融入现有算法，网格生成复杂，且无法处理由参数化 NSE 控制的高维问题。此外，求解反问题往往代价极其高昂，需要复杂昂贵的公式和新代码。本文回顾了流动物理信息学习，将数据与数学模型无缝集成，并通过物理信息神经网络（PINNs）加以实现。我们展示了 PINNs 在与三维尾流、超声速流和生物医学流动相关的反问题中的有效性。

## 创新点

1. **统一的流动反问题求解框架**：PINNs 的正问题和反问题公式完全一致，无需传统 CFD 中昂贵的数据同化方案。这是该综述强调的核心优势——当存在稀疏时空测量数据时，PINNs 在反问题上的精度和效率远超 CFD。
2. **域分解扩展（cPINN / XPINN）**：针对复杂几何和多尺度问题，引入子域 PINN 方法，用多个神经网络分别处理不同子域，在界面处施加连续条件。这篇综述整合了多篇前序工作，展示了域分解在工业复杂性问题上的适用性。
3. **跨流态统一建模**：覆盖不可压缩流、可压缩流以及生物医学流动三种截然不同的流态，在同一 PINN 框架下展示一致性，而非为每类问题重新设计求解器。
4. **无缝融合实验数据与物理约束**：通过将测量数据直接作为损失函数项，PINNs 将多保真度/多模态实验数据整合到 NSE 求解中，避免了传统方法中网格生成和数据同化的双重瓶颈。

## 一句话总结

这是一篇由 Karniadakis 组撰写的 PINNs 在流体力学中的综述，系统梳理了 PINN 基本原理、不可压缩/可压缩/生物医学流动的建模方法以及域分解扩展，并通过三维尾流、超声速流和血栓变形三个应用案例展示了 PINNs 在稀疏数据反问题上的独特优势。

## 研究问题

传统 CFD 在解决流体力学问题上面临三个核心瓶颈：（1）无法将噪声或稀疏的测量数据无缝融入求解过程；（2）对工业复杂几何的网格生成耗时且依赖经验；（3）反问题（如推断未知边界条件、材料参数）需要重新构建公式和代码，计算代价极高。PINNs 本质上不旨在取代 CFD，而是在上述场景中提供一种互补范式——尤其是在有部分时空测量数据可用的情况下，PINNs 的反问题求解精度和效率显著优于 CFD。

## 数据与任务定义

本文并非基于固定数据集的实证研究，而是一篇综述性论文，在各应用案例中使用了不同的数据配置：

- **三维圆柱尾流**：使用二维截面上的速度场测量数据（2D2C 观测），反演三维流场。流场由 DNS 模拟生成作为参考。
- **超声速钝体绕流**：二维可压缩 NSE，在部分壁面或激波区域给出稀疏的密度、压力或速度测量。
- **血栓变形**：生物医学流动中，利用流场观测推断血栓的材料参数（如弹性模量）。

核心任务定义：给定控制方程（NSE 的不同形式）和稀疏的时空观测数据，求解流场（正向推断）或反演未知参数/边界条件（反向推断）。

## 方法主线

### 基础 PINN 框架

考虑参数化 PDE 系统：

$$f(\mathbf{x}, t, \hat{u}, \partial_{\mathbf{x}} \hat{u}, \partial_t \hat{u}, \ldots; \boldsymbol{\lambda}) = 0, \quad \mathbf{x} \in \Omega, t \in [0, T]$$

网络模型是一个全连接前馈神经网络，以时空坐标作为输入，通过以下映射逼近 PDE 的解：

$$\hat{u} = \mathcal{N}(\mathbf{x}, t; \boldsymbol{\theta})$$

各隐藏层的变换关系为：

$$z_0 = (\mathbf{x}, t), \quad z_k = \sigma(W_k z_{k-1} + b_k), \quad 1 \leq k \leq L-1, \quad z_L = W_L z_{L-1} + b_L$$

核心损失函数为四项加权和：

$$\mathcal{L} = \omega_1 \mathcal{L}_{\text{PDE}} + \omega_2 \mathcal{L}_{\text{data}} + \omega_3 \mathcal{L}_{\text{IC}} + \omega_4 \mathcal{L}_{\text{BC}}$$

其中 $\mathcal{L}_{\text{PDE}}$ 惩罚控制方程的残差（通过自动微分计算微分算子），其余项分别约束数据匹配、初始条件和边界条件。训练通常先用 Adam 优化器进行粗调，再用 L-BFGS 进行精调。

> [!figure] Fig. 1 PINN 架构示意图
> 建议位置：方法主线 - 基础 PINN 框架
> 放置原因：展示 PINN 的核心架构——全连接神经网络以时空坐标为输入、输出 PDE 解，损失函数包含 PDE 残差、数据、初边值条件四项加权项。
> 当前状态：占位符（候选图片质量不足，保留占位符）

### 机制流程

PINN 求解流动问题的完整执行流程可以归纳为四个阶段：

1. **网络构建与微分算子定义**：将时空坐标 $(x, t)$ 作为输入传入网络，构建隐藏层数为 $L$（通常 $L=4$ 至 $8$）、每层神经元数为 20 至 50 的全连接网络，激活函数通常取 $\tanh$（因其在自动微分下各阶导数均连续）。利用自动微分框架一次性计算并提取 $\partial_x$、$\partial_t$、$\nabla^2$ 等所有微分算子，这些算子用于拼接装配控制方程的残差项。
2. **配点采样与多源损失组装**：在内部域 $\Omega$ 随机采样 $N_f$ 个配点（通常数万量级）用于计算 PDE 残差；在边界 $\partial\Omega$ 上采样 $N_b$ 个配点生成边界条件约束；在初值时刻 $t=0$ 上采样 $N_0$ 个配点生成初值条件约束；在实验测量位置使用 $N_d$ 个数据点生成数据匹配约束。四种损失以权重 $\omega_1$ 至 $\omega_4$ 加权求和融合为总损失，权重选择是训练中最关键的调参决策。
3. **两阶段梯度优化训练**：首先使用 Adam 优化器进行数千至数万步随机梯度下降以逃离局部极小区域，起始学习率设为 $10^{-3}$ 至 $10^{-4}$，配合阶梯衰减或余弦退火调度。随后切换到 L-BFGS 优化器精调——L-BFGS 利用近似二阶曲率信息压缩收敛步数，可在几百步内显著提升精度，但代价是每次迭代需要完整的全批量梯度。
4. **推理与参数扫描**：训练完成后，网络在任意时空坐标直接输出预测解，无需网格插值。对于参数化 PDE，改变参数 $\lambda$ 时直接在新的 $\lambda$ 值上评估网络即可得到对应解，无需重新训练——这一特性使 PINN 在反问题中天然支持"一次训练、多参数推断"。

### 不可压缩流动的 PINN 建模

对于不可压缩 NSE：

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}, \quad \nabla \cdot \mathbf{u} = 0$$

模型以速度-压力场作为输出变量：

$$(u, v, w, p)$$

也可采用涡量-速度公式作为替代方案。不可压缩条件已在前述方程中以散度为零的形式纳入 PDE 残差损失。由于 NSE 固有的多尺度特性，单网络学习整个时空域存在困难——特别是在高雷诺数下，边界层和尾流区域的多尺度特征对网络表达能力提出了很高要求。

### 可压缩流动与域分解

可压缩 NSE 引入了密度 $\rho$、能量 $E$ 作为额外的场变量，方程体系更为复杂。关键技术在于损失函数中质量、动量、能量三种守恒方程残差项之间的权重平衡。对于复杂几何问题，论文引入了域分解策略（cPINN、XPINN）：

> [!figure] Fig. 2 域分解示意图
> 建议位置：方法主线 - 可压缩流动与域分解
> 放置原因：展示 cPINN 将计算域分解为多个子域，每个子域由独立网络处理，界面处通过通量连续性条件耦合。
> 当前状态：占位符（候选图片质量不足，保留占位符）

cPINN 将全局域划分为多个子域，每个子域分配一个独立神经网络，子域间的界面处施加解的连续性和通量连续性条件。这一方法类似于传统 CFD 中的域分解策略，极大提升了 PINNs 处理工业级复杂几何的能力。

### 生物医学流动

对血管内的血流建模，PINNs 需要额外整合本构关系（如血液的非牛顿特性）和组织力学参数。在这一领域中，反问题尤为重要——医生可以通过有限的速度场测量反演出病人的个性化血管壁弹性和血栓参数，而传统 CFD 在缺乏完整边界条件的情况下几乎无法完成这类推断。

## 关键结果

### 三维圆柱尾流反问题

> [!figure] Fig. 3 三维圆柱尾流 PINN 结果
> 建议位置：关键结果 - 三维圆柱尾流反问题
> 放置原因：展示利用二维截面速度数据（2D2C 观测）反演三维尾流流场的结果，包括涡量等值面和速度剖面对比。
> 当前状态：占位符

利用二维截面上的速度测量数据，PINNs 成功反演了圆柱后的三维尾流结构。这是一个典型的反问题设置——已知部分二维测量，推断完整三维流场。结果显示 PINNs 能够准确捕捉尾流中的涡结构和大尺度流动特征，与 DNS 参考解高度吻合。

### 超声速钝体绕流

> [!figure] Fig. 4 超声速钝体绕流结果
> 建议位置：关键结果 - 超声速钝体绕流
> 放置原因：展示可压缩 PINN 在处理含激波的超声速流场中的表现，包括密度和马赫数分布。
> 当前状态：占位符

在二维超声速钝体绕流问题中，PINNs 成功捕捉了弓形激波结构。对于部分壁面压力或密度测量数据已知的反问题场景，PINNs 能够以较高精度重建全场密度、速度和压力分布。这一结果表明 PINNs 在可压缩流反问题中有实际应用潜力。

### 血栓变形参数推断

在生物医学流动应用中，PINNs 被用于从血流速度场观测推断血栓的弹性材料参数（杨氏模量等）。这是一个典型的参数反演问题：已知本构关系形式（如超弹性模型），但材料常数未知。PINNs 将材料参数作为可训练变量，与流场同时优化。结果表明即使在测量噪声存在的情况下，PINNs 仍能以合理精度反演出材料参数。

> [!figure] Table 1 不可压缩流 PINNs 案例研究
> 建议位置：关键结果
> 放置原因：汇总了各不可压缩流问题的配置、数据量和精度指标，是结果定位的核心表格。
> 当前状态：占位符

## 深度分析

这篇综述的真正价值不在于提出全新算法，而在于**系统梳理了 PINNs 在流体力学反问题中的定位优势**。传统 CFD 经过 50 年发展已经非常成熟，但它的架构决定了三个不可逾越的瓶颈：（1）网格生成；（2）数据同化；（3）反问题公式变更。PINNs 以自动微分替代网格、以统一损失函数替代分步同化、以参数化为变量替代代码修改，在这三个维度上正好击中了 CFD 的弱点。

然而需要注意：**作者反复强调 PINNs 并非 CFD 的替代品**。在当前阶段，PINNs 在标准正向问题上的精度和效率都不及高阶 CFD 求解器。PINNs 的真正价值区间是反问题——特别是那些有稀疏实验数据可用、CFD 又难以处理的场景。

另一个值得注意的点是**域分解的引入使 PINNs 具备了处理复杂几何的潜力**。单网络在整个流场上求解 NSE 会遇到表达能力和可训练性的瓶颈，而 cPINN/XPINN 分解策略提供了可扩展的解决方案。这一方向的发展可能使 PINNs 逐步逼近工业应用的可行性边界。

## 局限

1. **正向问题精度不足**：PINNs 在无数据辅助的纯正向问题上的精度和效率无法与高阶 CFD 相比，这限制了其在设计迭代等场景中的应用。
2. **高维非凸优化的训练困难**：PINNs 的损失函数是高维非凸函数，训练难度大——不同损失项之间的权重需要仔细调参，训练收敛性没有理论保证。
3. **缺乏泛化到新几何/边界条件的能力**：PINNs 是对单一构型进行训练，每次改变几何或边界条件需要从头训练新的网络，而 CFD 在改变边界条件时只需要修改输入文件。
4. **案例类型有限**：综述展示的三个案例（圆柱尾流、超声速钝体、血栓）都是相对简单的几何，真正的工业级复杂几何（如整机气动优化）尚未验证。
5. **实验噪声敏感性未充分研究**：虽然展示了部分噪声情况下的表现，但缺乏系统性的噪声鲁棒性分析。

## 引用

1. Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 2019, 378: 686-707.
2. Karniadakis G E, Kevrekidis I G, Lu L, et al. Physics-informed machine learning. Nature Reviews Physics, 2021, 3(6): 422-440.
3. Jagtap A D, Karniadakis G E. Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations. Communications in Computational Physics, 2020, 28(5): 2002-2041.
4. Mao Z, Jagtap A D, Karniadakis G E. Physics-informed neural networks for high-speed flows. Computer Methods in Applied Mechanics and Engineering, 2020, 360: 112789.
5. Kissas G, Yang Y, Hwuang E, et al. Machine learning in cardiovascular flows modeling: Predicting arterial blood pressure from non-invasive 4D flow MRI data using physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering, 2020, 358: 112623.
6. Lu L, Jin P, Pang G, et al. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 2021, 3(3): 218-229.

## 我的笔记

- 这篇综述的入口价值极高：如果你在 CFD 领域工作但对 PINNs 不熟悉，读这一篇就能建立起完整的认知框架——PINNs 的核心思想、数学原理、在不同流态下的建模方式、以及实际效果边界。
- 关键理解：**PINNs 的优势不是精度而是灵活性**。它的核心武器是：能用同一套代码解决正向和反向问题，能自然融合实验数据，不需要网格。如果你遇到的问题不符合这三个条件中的任何一个，CFD 可能仍然是更好的选择。
- 域分解方向值得跟踪：cPINN/XPINN 本质上是用工程手段解决深度学习模型容量的物理上限问题，思路朴素但有效。未来与算子学习（如 DeepONet）的结合可能会进一步突破可扩展性瓶颈。
- 缺少与数据驱动方法（如基于 CNN/Transformer 的流场预测）的系统对比，这在 2021 年之后已成为重要的竞争范式。
- 需要关注后续工作：Karniadakis 组的 DeepONet（2021）、基于注意力的流体 PINN（2022-2023）都是这一脉络的重要延伸。
