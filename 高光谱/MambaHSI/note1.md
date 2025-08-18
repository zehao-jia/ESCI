# 摘要abstract
- transformer的有效,计算的复杂性
- mamba的有效性,长距离建模能力并保持线性计算复杂性
- 提出高光谱图像分类模型mambahsi,可以对整个图像的远程交互进行建模,并以自适应的方法整合光谱和空间信息

- 空间mamba块,在像素层面对整个图像的长距离交互进行建模
- 频谱mamba块,将光谱向量分成多个组,挖掘不同光谱组之间的关系,提取光谱特征
- 空间光谱融合模块

- 对四种数据集的测试结果证明类模型的有效性和优越性


# 方法论
## 动机
- cnn和Transformer固有局限性促使开发一个新的模型利用线性复杂性对长程依赖关系进行建模,从而实现更精细的功能
- 引入mamba作为远程依赖性建模的基本单元

- 从序列的角度重新思考HSI图像分类,提出一个光谱mamba利用光谱的序列特性

## overview

![alt text](<屏幕截图 2025-08-15 162034.png>)

- 主要组件:嵌入层(embedding layer),编码器骨干(encoder backbone),分割头(segmentation head)

### embedding layer
- 嵌入层将光谱矢量投影到嵌入空间中
- 通过提取每个像素的嵌入获得更细粒度的像素嵌入
- 具体来说,细粒度像素嵌入$E\in R^{H\times W\times D}$可以从高光谱图像$I\in R^{H\times W\times C}$中获得
$$
E = Embedding(I)\\
= SiLU(GN(Conv(I)))
$$
- 其中,conv是核为$1\times 1$的卷积层,GN是组归一化,SiLU是激活函数

### encoder backbone
编码器骨干由一个提取空间特征的mamba块,一个提取光谱特征的mamba块和一个融合两个特征的融合模块组成,可表示如下:
$$
H=Encoder(E)
$$
其中,H表示提取的隐藏特征,Encoder表示编码器骨干

### 分类头
分类头通过一个$1\times 1$的卷积层获得一个logits $l$作为最终输出

## 空间mamba块
- 像素级分类任务要求:
> - 细化表示,反应像素之间差异$\rightarrow$以像素级方式提取嵌入
> - 表示应具有区分性$\rightarrow$要求模块具有强大的远程建模能力
- Transformer具有二次计算复杂性,不能用于在像素级别建立长距离依赖关系 

### 公式化模块
${HF}_{spa}=Flatten(H^i)$<br>
${HR}_{spa}=SiLU(GN(Mamba(HF_{spa})))$<br>
$H_{spa}^o=Reshape(HR_{spa})+R^i$<br>

其中,$H^i \in \mathbb{R}^{B\times W\times H\times D}$,$H_{spa}^O \in \mathbb{R}^{B\times W\times H\times D}$ 分别表示批次大小,图像高度,宽度,嵌入尺寸
- ${HF}_{spa} \in \mathbb{R}^{B\times L_1 \times D}$表示展平输出
- ${HR}_{spa} \in \mathbb{R}^{B\times L_1 \times D}$表示学习到的残差空间特征
- $L_1$等于$H\times W$,$Mamba$是标准mamba块
- 组范数和残差连接的设计有助于$SpaMB$学习
## 光谱mamba块
- 设计了一个频谱mamba块$(SpeMB)$,实现对光谱之间的关系进行建模
> 具体来说,将光谱特征化为G组,对不同的光谱组之间的关系进行建模,根据挖掘的光谱间关系更新光谱特征
- 提取的$H_o$可如下获得
- $HG_{spe}=SplitSpectralGroup(H^i)$
- $HF_{spe}=Flatten(HG_{spe})$
- $HR_{spe}=SiLU(GN(Mamba(HF_{spe})))$
- $H_{spe}^o=Reshape(HR_{spe})+H^i$
其中:
- $HG_{spe}\in\mathbb{R}^{B\times H\times W\times G\times M}$:分割的光谱组特征
- $HF_{spe}\in\mathbb{R}^{N\times G\times M}$:平坦的光谱组
- $RN_{spe}\in\mathbb{R}^{N\times G\times M}$:残差光谱组
- $H_{spe}^{o} \in \mathbb{R}^{B\times H\times W\times D}$:输出光谱特征
## 空间光谱特征融合模块(SSFM)
- SSFM会自适应地估计空间和频谱的重要性以进行指导融合
- 公式化:$H_{fus}=H_i+\omega_{spa}\times H_{spa}^o+\omega_{spe}\times H_{spe}^o$,其中,$\omega_{spa}$和$\omega_{spe}$被随机初始化,这些权重通过反向传播进行更新,以确定最终的融合权重
