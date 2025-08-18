# 总体框架
![alt text](<屏幕截图 2025-08-07 173139-1.png>)

## 预处理:SSFE(self-similarity feature enhancement)


### 邻域自相似性描述子

![alt text](<屏幕截图 2025-08-13 134622.png>)<br>
每个像素取邻域,求余弦相似度,经过相似度编码器重整为一个特征图1

### 中心自相似性描述子

![alt text](<屏幕截图 2025-08-13 134646.png>)<br>
中心像素为基准,计算余弦相似度,经过dotproduct获得一个特征图2

## 训练

### SCA

![alt text](<屏幕截图 2025-08-10 155813-1.png>)
- 作用:提取HSI的判别性局部-全局空间特征

#### 卷积层
- 通过卷积内置的归纳偏置获取本地信息

#### SWSA(selfattention的改进)

![alt text](<屏幕截图 2025-08-10 161210.png>)

- 1. 第一张特征图的复用
- 2. 卷积层和BN代替线性层

### SIT

#### GISSA
![alt text](<屏幕截图 2025-08-10 134107.png>)

- 解决的问题:
- 1. 正常用Transformer建模谱带之间的长程相关性会由于HSIs的高维性消耗计算成本和存储空间
- 2. 高光谱数据存在连续性和局部相关性,Transformer独立处理序列中每个位置的特征,不能充分利用光谱信息的局部相关性

$分组\rightarrow分组光谱自注意力\rightarrow数据重排\rightarrow 分组光谱自注意力\rightarrow recombine\rightarrow contact$

### 可训练参数
对于两个模块的输出,初始化一个 $\lambda$ 参数,将两个结果加权求和

## 输出

结合dropout和全局池化层构造一个线性分类器,输出结果