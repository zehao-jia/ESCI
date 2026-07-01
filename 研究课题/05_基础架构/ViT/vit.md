# 第一遍
## 核心贡献
- 将纯transformer应用于图像分类任务,不依赖于卷积神经网络
- 图像分块:将图像分割为大小固定的块(16*16),线性嵌入后作为序列输入transformer
- 性能超过cnn
- 需要计算资源少 
- 卷积的两个归纳偏置:局部性(locality)和平移不变性(translation euqvariance)
## 结论
- 标准transformer解决cv问题
- 问题:vit做分割和检测
# 第二遍
## 网络架构
<img src="屏幕截图 2025-07-04 135511.png" width="500px"><br>
### 1.patch embedding
- eg:224 * 224 * 3的图片,分割为16 * 16 * 3的小图变为196 * 768的输入
- 全链接层768*768(后面的768可变)
- 最终得到一个196词汇768投影的词嵌入
### 2.class token的使用
- 借鉴<font color="red">bert的cls</font>,添加一个(1,768)的向量
- 得到一个197*768的词嵌入
### position embedding
- 位置编码用于保留各个图像块之间的位置信息,这主要是由于自注意力的扰动不变性
### transformer encoder
- 输入特征大小：197 × 768
- Multi-Head Attention：假设有N个头，K、Q、V：197 × (768 / N)，拼接后仍然输出197×768
- MLP：197×768 → 197 × 3072 → 197 × 768
- L层输出特征大小：197 × 768
## 对比实验
- 是否需要2-d aware position embedding? -> 消融实验证明区别不大
- 图像<br>
<img src="屏幕截图 2025-07-04 151003.png" width="600px"><br>
如图,在小量级的数据中,vit的效果是不如resnet的,当数据集增大,vit的效果获得了显著提升<br>
- 解释:缺少归纳偏置和约束方法如<font color="red">weight-decay label-smoothing</font>
# 总结
- 做了什么:transformer应用于cv领域,并证明其可行性