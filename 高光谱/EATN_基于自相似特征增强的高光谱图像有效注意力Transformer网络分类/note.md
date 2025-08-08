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
1. 对原始patch进行归一化,以探索像素之间的关联
2. 两种自相似性描述块
#### 邻域自相似性描述子
-  邻域自相似性描述子:保留原始图像块中像素的局部位置信息,将自相似性引入卷积计算 
- 简单来讲,每个像素为中心取得一个patch随后以像素为基准求余弦自相似性
$$
R_{Neighborhood}=\frac{P(x,c)\cdot P(x+z,c)}{\left\lVert P(x,c) \right\rVert \left\lVert P(x+z,c)\right\rVert }
$$
- 其中$z\in[-du,du]\times[-dv,dv]$对应邻域窗口内的相对位置,$U=2du+1,V=2dv+1$,c的范围从1到c,表示像素中的[谱带索引](#sprctral_band_index)

#### 中心自相似性描述子
- 中心自相似性描述子
3. 将两种描述子的结果添加到原始图像块,送入特征融合模块
- 特征融合模块:由两个2D卷积层和一个relu层组成


<font color="red">
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
    &emsp;谱带索引(band index)通过特定的波段组合,来增强或提取目标的量化指标
    </a>
</font>

