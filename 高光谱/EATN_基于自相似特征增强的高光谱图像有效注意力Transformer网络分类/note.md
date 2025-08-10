# Abstract

## 问题:

- 高光谱图像的高维特性使计算成本增加
- 基于变换的HSIC方法引入不相关的光谱信息
> - hyperspectral image classification(HSIC):高光谱图像分类
## 创新
1. 提出两种基于HSI patch的自相似描述子来增强空间特征表示
- 中心自相似描述子(center self-similarity descriptor):<br>
强调与中心像素相似的像素
- 邻域自相似描述子(neighborhood self-similarity discriptor):<br>
探索patch内每个像素和与其相邻像素的相关性

# Introduction
1. 早期,SVM,KNN在HSIC领域取得了相当好的性能
2. 这些方法容易受到噪声影响,无法达到有希望的分类效果
3. 深度学习应用于HSIC:通过CNN卷积核提取光谱和空间信息
4. CNN利用其内置的归纳偏差实现强大的分类性能,但难以建立全局依赖性,这是由于感受野的限制
## 论文的主要贡献
1. 提出深度框架EATN,通过一种特征增强策略将CNN和Transformer相结合,以对HSIC的每个像素进行高效准确的分类
2. 设计了SSFE模块,在模块内构造两个描述子,将描述子与原始块融合,从而充分利用原始快的空间位置信息
3. 为了降低自注意力的计算复杂度,提出SIT和SCA模块,用于提取HSI的局部-全局空间-频谱信息,避免了自注意力机制过多的计算开销

# related work

## A.CNN相关HSIC
## B.注意力机制模型
## C.Transformer相关HSI

# methodology(方法论)

## A.EATN的总体框架

- 关键组件:SSFE,SCA,SIT<br>
1. 假设输入HSI可以表示为$P\in\mathbb{R} ^{H\times W\times C}$其中,HW分别表示长度和宽度,C表示波段数,

2. 将选定的像素和其相邻像素作为patch输入网络,补丁可以表示为$P\in\mathbb{R}^{S\times S\times C}$其中$S\times S$表示patch的大小,SSFE模块生成两个大小相同的自相似性描述符,然后将他们与原始补丁融合,特征图的大小保持为$S\times S\times C$

3. 将特征图输入到空间和光谱特征提取模块:
- 对于光谱分支,使用$1\times1$的2D卷积来降低特征图的维度,随后,将其输入到提出的[SIT模块](#SIT)
- 对于空间分支,采用核尺寸为$3\times3$的二维卷积对特征图进行降维,随后将其送入一系列[SCA模块](#SCA)中
4. 降维后空间分支和光谱分支的维数为$S\times S\times D$,此外,初始化一个可训练的加权参数$\lambda $,并对特征图进行加权求和

5. 最后,通过结合一个dropout层和一个全局平均池化构造一个线性分类器,得出最终的分类结果

## B.自相似特征增强
- 问题:自注意不能有效挖掘局部相关性,基于变换的模型难以感知像素之间的位置信息
### SSFE模块处理原始HSI块
![alt text](<屏幕截图 2025-08-07 173139.png>)
- [自相似性](#self-similarity)描述符增强HSI的特征信息
#### 1. 对原始patch进行归一化,以探索像素之间的关联
#### 2. 两种自相似性描述块
##### 邻域自相似性描述子
-  邻域自相似性描述子:保留原始图像块中像素的局部位置信息,将自相似性引入卷积计算 
- 简单来讲,每个像素为中心取得一个patch随后以像素为基准求余弦自相似性
$$
R_{Neighborhood}=\frac{P(x,c)\cdot P(x+z,c)}{\left\lVert P(x,c) \right\rVert \left\lVert P(x+z,c)\right\rVert }
$$
- 其中$z\in[-du,du]\times[-dv,dv]$对应邻域窗口内的相对位置,$U=2du+1,V=2dv+1$,x表示该像素的空间位置c的范围从1到c,表示像素中的[谱带索引](#sprctral_band_index)

- 为了尽可能地保留原图像的语义信息，我们采用通道计算的方法生成RN邻域，并对得到的自相关张量进行非负处理。

- 为了将自相关张量从[高维描述符](#high-dimensional_descriptors)编码为[紧凑描述符](#compact_descriptors)，我们将获得的自相关张量馈送到相似性编码器块(self-similarity encoder)中，该相似性编码器块由3-D卷积层、批归一化（BN）层和ReLU层组成。相似性编码器块的数量与U和V的值有关。我们将自相关张量的空间维度从U × V降低到1 × 1，从而确保原始输入补丁大小的一致性。

- 因此，邻域自相似性描述子DN能有效地保留局部像素关系信息，并通过与原始图像块的融合增强基本特征表示

##### 中心自相似性描述子
- 中心自相似性描述子:突出中心像素,根据输入的中心像素,定义所有其他像素与中心像素的余弦自相似性:
$$
R_{center}=\frac{P(x_c)\cdot P(x_c+z
)}{\left\lVert P(x_c)\right\lVert\left\lVert P(x_c+z)\right\lVert
}
$$
- 其中,有$z\in[-ds,ds]\times[-ds,ds]$且$S=2ds+1$ 计算后所得大小为$P_{center}\in\mathbb{R}^{S\times S\times1} $,将其与原始patch相乘
- 这个模块有效突出有益于模块分类的信息,同时避免了不相关信息的干扰
#### 3. 将两种描述子的结果添加到原始图像块,送入特征融合模块
- 特征融合模块:由两个2D卷积层和一个relu层组成
计算结果F可表示为如下:
$$
F=max(0,(P+D_N+D_C)w_1+b_1)w_2+b_2
$$
- 其中,$D_N,D_C$分别是邻域自相似性描述子和中心自相似性描述子的计算结果

我们从SSFE模块得到特征图F,将其输送到空间-频谱特征提取阶段,与原始patch相比,生成的特征图增强了特征的表示并显示了有价值的信息,从而促进了特征提取任务

### SIT
#### 存在的问题
- 正常用Transformer建模谱带之间的长程相关性会由于HSIs的高维性消耗计算成本和存储空间
- 高光谱数据存在连续性和局部相关性,Transformer独立处理序列中每个位置的特征,不能充分利用光谱信息的局部相关性
#### 解决-SIT模块
![alt text](<屏幕截图 2025-08-10 134107.png>)
通过引入GISSA模块开发一个SIT模块
##### GISSA数据流动
- 1. 对于输入的特征图$F\in\mathbb{R}^{S\times S\times D}$,我们将其分为N组,每组长度相等,得到N组$F'\in\mathbb{R}^{S\times S\times\frac{D}{N}}$
- 2. 对于每个$F_i$,将其通道视为一个token进行自注意力操作,具体来讲使用$1\times 1$卷积将输入图像映射为QKV矩阵,随后计算注意力图$M_i$
$$
M_i=\frac{Q_iK_i^T}{\sqrt{d_k}}
$$
- 3. 对得到的$M_i$进行带符号平方根运算,经过softmax后再乘以$V_i$矩阵得到第一批自注意力输出$F_i'$
$$
F_i'=softmax(sign(M_i),\sqrt{\vert M_i \vert+\delta })
$$
- 其中,$M_i\in \mathbb{R}^{d\times n},n=S\times S$
- 此外,引入残差链接来融合$F_i'$和$F$,对输出特征经过BN和RELU激活后进行第二次注意力计算,为了实现光谱的信息交互,将每个$F_i$中具有相同索引的谱段组合为新的$F_i''$,如果将$F_ij$表示为第i个F的第j个通道,新输入可以表示为:
$$
F_i''=\left\{F_{1i},F_{2i}\dots F_{ni}\right\} ,i=\left\{1,2,\dots n\right\}
$$
- 4.使用与第一次计算相同的方法得出计算结果记为$F_i'''$
$$
F_i'''=F_{1i}''',F_{2i}''',\dots F_{ni}'''
$$
- 将得出的结果再还原回原先的格式得到$F_i''''$:
$$
F_i''''=\left\{F_{1i}''',F_{2i}'''\dots F_{di}'''\right\},i=\left\{ 1,2,\dots,N\right\}
$$
- 5.最终,将恢复顺序的特征图链接起来作为输出,可表示为
$$
Attention(F)=Contact(F_1'''',\dots F_n'''')
$$
##### MLP部分
- 在注意力层和MLP层之间添加了一个[线性归一化](#LinerNormalization)层,并添加了残差连接<br>
![alt text](<屏幕截图 2025-08-10 152053.png>)
- MLP表示如下:
$$
MLP(X)=FC_2(GLEU(FC_1(X)))
$$

- 综上,SIT模块可表示为:
$$
A=Attention(LayerNorm(X))+X
$$
$$
O=MLP(LayerNorm(X))+X
$$
其中AO分别表示Attention层和MLP层的输出特征,
#### SIT的优点

- 通过采用两步交互式自关注计算,有效的捕获了局部细粒度的光谱信息,同时促进了全局信息的交互,不增加额外的计算成本

### SCA
![alt text](<屏幕截图 2025-08-10 155813.png>)
#### 1. 作用
- 有效的提取HSI的判别性局部-全局空间特征
#### 2.组成
##### 2.1 卷积序列
- 组成:$2\times 3$ 的卷积层,BN层和RelU层
- 旨在通过卷积中内置的归纳偏置探索本地信息

##### 2.2 self-attention层
![alt text](<屏幕截图 2025-08-10 161210.png>)
- 与传统的自注意力相比,使用两个独立的 $1\times 1$ 卷积层生成QV矩阵,设置Q=K[这消除了每层中的一个卷积](#huh1)
- 与广泛使用的LayerNorm将自我注意力的计算划分为众多元素级操作相比，我们在每次卷积后采用BN来确保稳定的训练。BN被集成到卷积运算中，而不引入
- 引入共享权重自我关注(SWSA),缓解自我注意力中的计算负担
> - 具体来说,在计算出第一个注意力图之后,在随后的SCA模块中重用他,而不会影响模型性能

- 此外,在卷积序列和自我关注层之间添加了残差链接,通过交替使用卷积序列和自关注层，该模块输出包含局部-全局空间信息的特征图Fspatial
### 输出

- 我们通过动态集成将从SIT模块中提取的特征图Fspectral与特征图Fspatial结合起来,如下:
$$
F=(\lambda\times F_{spatial}+(1-\lambda)\times F_{spectral})
$$
其中 $\lambda$ 是可学习的参数

<font color="green">
    <a id="SIT">
    关于什么是SIT模块:<br>
    </a>
    <a id='SCA'>
    关于什么是SCA:<br>
    </a>    
    <a id='self-similarity'>
    关于什么是自相似性:<br>
    &emsp;自相似性指的是图像中特定区域与整个区域之间的相似性或可重复性<br>
    </a>
    <a id='sprctral_band_index'>
    关于什么是谱带索引:<br>
    &emsp;高光谱图像包含的不同波段的光谱信息通道称为谱带;<br>
    &emsp;谱带索引是用于标识不同光谱波段的索引值,高光谱图像包含多个不同的波段,谱带索引即用于区分这些波段的索引值,文中以符号c表示
    </a>
    <a id='high-dimensional_descriptors'>
    关于什么是高维描述符:<br>
    &emsp;高维描述符指维度较高的描述符,这类张量包含了多个像素在不同谱带和空间位置上的相关信息,例如与邻域窗口U\times V相关的维度(空间维度)和c(光谱维度)<br>
    </a>
    <a id=compact_descriptors>
    关于什么是紧凑描述符:<br>
    &emsp;对高位描述符进行降维处理后得到的低维度特征描述符,文中通过相似性编码器块将空间维度从uv降低到1得到的描述符为紧凑型描述符
    </a>
    <a id=LinerNormalization>
    关于什么是线性归一化:<br>
    </a>
    <a id=huh1>
    消除卷积:Q=k意味着两个矩阵通过一次卷积得出,这消除了一次卷积运算
    </a>
</font>

