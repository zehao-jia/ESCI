**** 
#差分注意力
****

- 创新点:
***1.差分多头自注意力机制
2.SWiGLU激活函数的集成
3.基于类token的统一光谱-空间表示
4.高效的基于块的光谱-空间token策略

# introduction
## 差分多头自注意力
- ***目的:突出局部区域内token间的细微变化
- DiffFormer的核心创新在于DMHSA模块,通过引入差分操作捕捉**相邻token**的相对变化
- 这个相邻的意思是在进行过QKV运算后得到的token有编码,按编码顺序的相邻
$$
S = \frac{QK^T}{\sqrt{ d_{head} }}
$$
- 这里S矩阵是一个$N\times N$的矩阵
$$
S_{diff} = S[:,1:]-S[:,:-1]
$$
- 这个公式有些抽象,意思是说,在得到S后,每一列对应的是一个token的分数,假设有个token编号是a,那就让他减去编号为a-1的token整合为矩阵就是S的第二列到最后一列减去第一列到倒数第二列,最终生成一个$N\times (N-1)$的矩阵,经过Softmax后得到注意力权重矩阵A,再用A对全局矩阵V进行加权求和,得到的矩阵表示为$$Z=AV
$$
其中,$z\in \mathbb{R}^{N\times M}$,而$V\in \mathbb{R}^{M\times d_{head}}$


## Transformer编码器块
- transformer编码器包含DMHSA模块和SWiGLU激活增强的前馈层
$$
SWiGLU(x,g) = x\cdot\sigma(g)+x
$$
- 其中,$\sigma$表示sigmoid函数,层归一化和残差链接确保了训练的稳定性