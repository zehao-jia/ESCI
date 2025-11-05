# Abstract
- 提出的模型:轻量级并行设计,称为轻量级双流mamba卷积网络(DualMamba)
- 具体来说,开发了并行轻量级mamba和cnn块来提取全局和局部谱空间特征

## 交叉注意谱空间Mamba模块(CAS 2 MM)
- 在该模块中,设计了动态位置嵌入(DPE)增强视觉序列的空间位置信息
## 轻量级的谱空Mamba块
- 1.有效的扫描策略
- 2.轻量级的Mamba设计
- 3.交叉谱空融合(CAS 2F)
## 自适应全局-局部融合方法

# Introduction
## 存在的问题
- 1.现有的HSI分类方法难以在全局-局部谱-空间建模中实现有效和高效
- 2.现有方法往往采用级联方式顺序学习全局-局部关系,而[[关于模型架构#级联结构|级联结构]]在局部建模过程中往往忽略上下文,且在建模光谱-空间关系式无法解耦全局和局部的特征
- 
## 解决方法
- 采取分治思想考虑分别捕获全局和局部光谱-空间特征的并行结构
- 使用mamba替换transformer,解决时间和空间上的低效
## 仍然存在的问题

# 方法论(METHOD)

## 整体架构

![[高光谱/DualMamba_一种用于高光谱图像分类的轻量级光谱空间mamba卷积网络/屏幕截图 2025-08-28 155315.png]]
## A.准备工作
### 1.SSM
受SSM的启发,提出了结构化状态空间序列模型(S4),该模型通过将一维序列$x_t \in \mathbb{R}$通过隐藏状态$H_t \in \mathbb{R}$映射到$y_t \in \mathbb{R}$,它的整体流程可以用如下的微分方程表示:
$$
h'(t) = Ah(t) + Bx(t)
$$
$$
y(t) = Ch(t)
$$
- 其中,$A\in\mathbb{R}^{N\times N}$表示状态转移矩阵,$B\in\mathbb{R}^{N\times1}$和$C\in\mathbb{R}^{1\times N}$分别表示将输入映射到状态和将状态映射到输出的矩阵
- 考虑到图像和文本都是离散信号,S4模型需要使用 [[离散#零阶保持器ZOH|零阶保持器ZOH]]进行离散化,即将连续的时间参数A,B转化为时标参数$\Delta$的对应参数$\overline{A},\overline{B}$,表示如下

$$
\overline{A} = \exp(\Delta A)$$
$$\overline{B} = (\Delta A)^{-1}
$$
- 综上,原先的s4模型可以表达为:
$$
h_t = \overline{A}h_{t-1} + \overline{B}x_t
$$$$
y_t = Ch_t
$$
为了提高s4模型的计算效率,可以将(3)中的迭代过程表示为卷积:
$$
\overline{K}=(C\overline{B},C\overline{AB},C\overline{A^{L-1}B})
$$
- 其中,L表示序列x的长度,$K\in \mathbb{R}^L$表示S4卷积核
## Mamba
S4模型具有线性时间复杂度,但由于其[[复杂度相关#时不变参数化(time-invariant parameterization)|时不变参数化]],在表达序列上下文时收到约束,Mamba提出了选择性扫描S4模型(S6),具体来说，Mamba简单地使几个参数$B，C，\Delta$成为输入$x ∈ R^{B×L×D}$的函数:
$$
B,C,\Delta=Linear(x)
$$
- 其中,$B,C,\Delta\in\mathbb{R}^{B\times L\times N}$
- 此外,Mamba引入了一种硬件感知的高效训练算法,我们的方法利用了Mamba的S6模型，利用其计算效率和能力来有效地建模全局长范围依赖性
## B.交叉注意频谱-Mamba模块
### CAS 2 MM
- 作用:引入 DPE 为扫描序列提供空间位置信息,并执行交叉注意光谱空间融合%% (CAS 2F) %%将空间特征和轻量级空间光谱Mamba块提取的光谱特征相结合
- 1. 动态位置嵌入(Dynamic Positional Embedding)
由于Mamba只对序列数据进行编码,所以对HSI数据进行位置编码,DPE具有输入依赖性,可以动态的适应不同类型输入的位置信息,这里采用深度卷积作为DPE方法
$$
DPE(X) = DWConv_{3\times 3}(X)
$$
- %% 其中,$X\in\mathbb{R}^{P\times P\times D}$是像素嵌入特征,p表示块大小,D表示嵌入维数 %%
### 2. 轻型空间Mamba块
![[高光谱/DualMamba_一种用于高光谱图像分类的轻量级光谱空间mamba卷积网络/屏幕截图 2025-08-29 123507.png]]与vanilla vim块相比,简化了Mamba块的结构,消除了门控MLP和深度卷积,首先,使用DPE对像素嵌入特征执行层归一化,然后使用一个线性层将特征投影到S6模型需要的维度
$$
X_{pos}=X+DPE(X)
$$
$$
X_{spa}=Linear_1(LN(X_{pos}))
$$
- 其中$X_{spa}\in\mathbb{R}^{P\times P\times D_{ssm}}, D_{ssm}=ssm\_ratio\times D,linear_1()\in\mathbb{R}^{D\times D_{ssm}}$,$LN$表示层归一化.
- 扫描上,使用空间单向扫描策略有效的提取全局空间特征,如图生成空间视觉序列$S_{spa}$然后将序列馈送到S6模型中
$$
h_t^{spa}=\overline{A}_{spa}h_{t-1}^{spa}+\overline{B}_{spa}x_t^{spa}
$$
$$
y_t^{spa}=C_{spa}h_t^{spa}
$$
- 其中,$A_{spa}\in\mathbb{R}^{D_{ssm}\times N},\overline{B}_{spa}\in\mathbb{R}^{B\times P^2\times N},C_{spa}\in\mathbb{R}^{B\times P^2\times N}$,是S6的训练模型,N表示Mamba的隐状态维数
- Mamba处理后,获得输出序列,然后对原始嵌入维度执行层归一化和线性投影最后通过来自原始输入的残差链接获得全局空间Mamba特征$G_spa\in\mathbb{R}^{P\times P\times D}$如下所示
$$G_{spa}=X_{pos}+Linear_2(LN(Y_{spa}))$$
其中$Linear\in\mathbb{R}^{D_{ssm}\times D}$
### 轻型光谱Mamba块
这个光谱Mamba块类似于空间Mamba块,不同之处在于它将hsi贴片中的中心像素的特征输入到块,其原因有两个
> 1)中心像素位置对应的特征直接反映目标表面特征的光谱特性,受光谱混合的影响较小
> 2)该方法大大降低了模型的参数量和计算量
- 首先,从$X_{pos}$的中心像素位置提取特征,并将其整形为(D,1)大小,对中心特征进行层归一化,再投影到S6模型要求的维度上
$$X_{spe}=Linear_3(LN(Center(X_{pos})))$$
- 其中$X_{spe}\in \mathbb{R}^{D\times ssm\_ratio},Linear_3 \in \mathbb{R}^{1\times ssm\_ratio}$
- 对于光谱序列特征,考虑到光谱序列的不对称性,从正反两个方向获取全局光谱背景信息将获取的两个序列分别馈送到两个S6模型中用于如下的空间状态演化:
$$ 
h_{t,j}^{spe}=\overline{A}^j_{spe}h_{t-1,j}^{spe}+\overline{B}^j_{spe}x_{t,j}^{spe}
$$
$$
y_{t,j}^{spe}=C_{spe}^jh_{t,j}^{spe}
$$
经过Mamba处理,得到两个输出序列,然后将他们合并为一个谱序列特征:
$$
Merge(Y_{spe})=Y_{spe}^0+Flip(Y_{spe}^1)
$$
我们对原始嵌入维度进行层归一化和线性投影最后通过原始输入的残差链接获取全局光谱Mamba特征$G_{spe}\in\mathbb{R}^{1\times 1\times D}$
$$
G_{spe}=X_{pos}+Linear_4(LN(Merge(Y_{spe})))
$$
### 交叉注意谱空融合(CAS2f)
- 目的:建模复杂的光谱-空间关系,充分利用光谱和空间的互补性
- 具体地,先将全局空间Mamba特征$G_{spa}$和全局光谱Mamba特征$G_{spe}$进行归一化,归一化意在平滑softmax函数的输出,运用softmax计算光谱注意力和空间注意力权重:
$$
A_{spe}=Softmax(Norm(G_{spe}))
$$
$$
A_{spa}=Softmax(AP(Norm(G_{spa})))
$$
- 其中,AP表示平均池化
- 将获得的光谱注意力权重和空间注意力权重分别与空间特征和光谱特征交叉相乘,最后将交叉注意力求和以融合成综合的全局光谱-空间特征G
$$
G = A_{spe}G{spa}+A_{spa}G{spe}+X_{pos}
$$
## C.轻量化频谱-空间残差卷积模块
- 目的:利用CNN的局部特征提取能力补充Mamba,文中设置了两个轻量级并行分支:光谱分支和空间分支
#### 光谱分支
- 一个$1\times1\times3$的3d卷积算子,将卷积主要聚合在光谱维度上,在聚合光谱信息的同时保留空间信息:
$$
L_{spe}=δ(BN(Conv3d(X)))
$$
- 其中Conv3d（·）表示具有1 × 1 × 3内核的3D卷积，BN（·）表示批归一化（BN），δ（·）表示SiLU激活函数
#### 空间分支
- $3\times 3\times D$卷积核提取局部空间特征:
  $$
L_{spa}=δ(BN(DWConv(X)))
$$

其中DWConv（·）表示具有3 × 3内核的深度卷积，δ（·）表示SiLU激活函数
- 最终,将提取的局部光谱特征$L_{spe}$与局部空间特征$L_{spa}$连接起来,然后用逐点卷积融合这些特征将特征通道减少到原始通道:
$$
L=δ(BN(PWConv([L_{spe},L_{spa}])))
$$
- 其中PWConv（·）表示逐点卷积，[·]表示级联，δ（·）表示SiLU激活函数。
## D.自适应融合机制
![[高光谱/DualMamba_一种用于高光谱图像分类的轻量级光谱空间mamba卷积网络/屏幕截图 2025-08-29 155612.png]]
- 目的:自适应地融合CAS2MM的全局光谱空间特征和来自轻量级光谱空间残差卷积模块的局部光谱空间特征
- 具体地说，我们首先通过逐元素求和和平均池化操作融合全局谱空间特征G和局部谱空间特征L，然后将融合特征馈送到简单多层感知（MLP）中，以获得具有更少通道的紧凑特征$z ∈ R^{B×1}$，以提高效率
$$
z=W_1(δ(BN(W_2\cdot AP(G+L))))
$$
- 其中δ（·）是ReLU函数，BN（·）是BN，W1 ∈ RK×Kr，W2 ∈ RKr×1，Kr = K/2，AP（·）表示表I土地覆盖类型和印第安松数据集的标记训练样本和测试样本的数量平均池化操作。全局特征权重Wg和局部特征权重Wl通过Sigmoid函数计算如下：$$W_g = Sigmoid（z）$$$$W_l = 1-W_g$$为了稳定训练过程，我们从先前融合的特征中添加额外的跳过连接。最后，我们通过乘以全局-局部权重$F = G + L + W_gG + W_l L$来获得全面的全局-局部频谱-空间表示F.（21）最后，采用线性分类器对HSI分类器进行标签预测，并采用交叉熵损失函数对该方法进行优化。
# 结论
对三个HSI数据集的广泛评估表明,dualmamba显著优于最先进方法,以最少的参数和FLOP实现上级分类精度
