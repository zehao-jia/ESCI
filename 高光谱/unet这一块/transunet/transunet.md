![[高光谱/unet这一块/transunet/overview.png]]
# transformer作为encoder
## 图像序列化
- 首先,将输入x重塑为一系列扁平化的patch$N =\frac{HW}{P^2}$为补丁数量,得到的patch为$\{x_{p}^i\in\mathbb{R}^{P^2\cdot C}\}$,即将每个PPC的张量拉平为1d向量
## patch embeding
- 通过transformer将$x_{p}$映射到潜在的D维空间,对于先前得到的序列有:
$$
z_{0} = \{x_{p}^1E; \cdots x_{p}^NE\}+E_{pos}
$$
- 其中E表示一个投影矩阵$E \in\mathbb{R}^{(P^2\cdot C\times D)}$,这会将原图变为一个$1\times D$的向量,拼接起来就是$D\times N$的向量
# 上采样
- 多个上采样步骤,用于解码隐藏特征从而输出最终的特征其中每个块有上采样算子,卷积层和Relu层组成

### 问题1:输入transformer块的内容
- cnn特征图经patch分割,线性投影嵌入,位置编码后得到的1d序列
## 问题2:上采样的创新
- 首先,将transformer的结果重塑为空间特征图
- 然后与分辨率最低特征图做concat,经过一个上采样算子,一个33卷积层,一个Relu激活函数得到上采样结果