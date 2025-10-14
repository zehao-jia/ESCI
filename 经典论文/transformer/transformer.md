# Attention is all you need
## 摘要
- 主流的序列转录模型:编码器(encoder) 解码器(decoder) 中间加attention
- 这篇文章:提出简单的架构,依赖于注意力机制,不依赖卷积或循环这样做的优点是:并行性好,训练时间少且效果好
- 成果:英德翻译降了2个bleu score

## 结论
- 将之前的循环层全部换成了multi-head self-attention
- 可以用在video或其他上使工作不时序化
## 导言
- 主流模型:语言模型的编码器解码器架构
- rnn:将当前词的隐藏状态$h^t$输出,隐藏状态由当前词h和$h^{t-1}$决定.存在并行性差,长期依赖问题
- RNN中的attention机制:把encoder中的内容有效转给decoder
- 本文只使用attention机制而不使用循环神经层,因为可变形,并行度较高

## 相关工作
- 1.使用卷积神经网络替换循环神经网络
- 卷积神经网络通过多输出通道识别不同的特征<br>
$\quad$优点:多个通道输出<br>
$\quad$缺点:相隔远的像素难以关联
<br>$\rightarrow$通过multi-head attention机制模拟卷积神经网络的效果
- 2.自注意力机制(self-attention)
## 模型架构(model architecture)
### 1.encoder-decoder是干什么的
- 给定一个序列$(x_1,x_2, \dots ,x_n)$输入到编码器中,编码器给出一个序列 $(z_1,\dots,z_n)$ <br>将新序列输入进解码器,解码器给出结果$(y_1,\dots ,y_m)$ <br>
$\quad$可见,生成序列的序列长度可能与输入不同

### 2.transformer的encoder
- 六个完全一致的层,每个层包含两个子层:1.多头自注意力机制(multi-head self attention)2.MLP
- 最终:样本标准化(layer normalization)
<font color="green">
标准化:
常规的标准化方法有两种:<br>
样本标准化(layer norm),特征标准化(batch norm)<br>
不使用特征标准化是为了防止样本长度影响均值方差
</font>
### transformer的decoder
- 自回归:当前的输入来自之前的输出
- 带有掩码的多头注意力机制
- 计算:多个qkv向量分别组合形成矩阵<br>
<font color="blue">
Q:除以根号下dk的原因
A:梯度的问题,防止难以更新梯度
</font>

- scaled dot product attention<br>
以$q^t$为例,我们不想使用t之后的k向量,就给他们赋值为负无穷<br>在经过softmax后对应位置变为0<br>
<img src="屏幕截图 2025-07-01 220934.png" width="200px"><br>

- 多头注意力机制

- 目的:学习到不同的内容,应对不同语境,提升灵活性<br>
<img src="屏幕截图 2025-07-01 222135.png" width="200px"><br>
<img src="屏幕截图 2025-07-01 222632.png" width="200px"><br>
- 自注意力机制(self-attention):自己作qkv
#### 三个attention分别干什么
- 左下:自注意力机制,找出各个词之间的相似度
- 右下:同样的自注意力,添加mask,忽略t后的内容
- 右上:不是自注意力,根据解码器输入的向量查询相关度高的向量
#### feed-forward层干什么
$$FFN(x)=max(0,xW_1+b_1)W_2+b_2$$
- 在运算时,这个层现将输入的矩阵通过$W_1$映射到高维空间<br>
    借助非线性激活函数学习复杂的模式在映射回原始维度
- 作用:通过非线性变换增强模型感知能力<br>
在保持序列位置信息的同时,对位置的特征进行丰富
- mlp(多层感知机)
#### embedding层
- 给出一个词,学习一个向量表示它
- 由于维度越大的向量归一化后其单个值相应的也就越小,因此,在输出权重时,给这些权重乘上$\sqrt{d_{model}}$
#### 位置编码(positional encoding)
使用一组函数,将词的位置信息加入词向量,
- 位置编码的数值和token处于同一量级,两者不会存在融合,覆盖的问题
- 模型会自行学习区分原始特征和位置信息
### 为什么使用self-attention
![alt text](<屏幕截图 2025-07-01 231831.png>)<br>
- 算法复杂度
- 执行顺序(进行当前运算需要等待的步骤)
- 最大长度(信号从一个数据点到另一个数据点要走多远/信息从一个输入到输出需要经过的层数)
# 总结
- 解决的问题:通过attention机制提升模型并行性