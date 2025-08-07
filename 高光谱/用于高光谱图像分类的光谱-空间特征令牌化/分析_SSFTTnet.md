## 权重初始化函数

```python
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)
```

- 首先,函数接受参数m,m代表模型中的一个层例如卷积层之类的
- 通过m.__class__.__name__获取层的类名
- 如果该层为全链接层或3d卷积层,则使用kaiming正则化

> 1. isinstance(对象,类名):如果实例是类或类的子类,返回True<br>
> 2. kaiming正则化:一种初始化方法,用于缓解梯度消失或梯度爆炸的问题,根据上一层的神经元数量,以正态分布或均匀分布的方式初始化权重,使得每一层的输出方差保持稳定
> 3. 代码中使用正态分布初始化

## 残差链接模块

```python
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
```

- 定义残差链接模块
- 所有模块的初始化先继承父类初始化函数nn.moudule
- 向实例中添加函数**fn**
- 前向传播:残差连接的前向传播,输入加输出

## 预归一化模块
```python
# 等于 PreNorm
class LayerNormalize(nn.Module):
    """预归一化模块,在运行子模块之前对输入做层归一化,稳定训练过程,提高收敛速度"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)#层归一化,对最后一维(dim)做层归一化
        self.fn = fn#传入的子模块

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)#先归一化输入在传入子模块
```
- **作用:在运行子模块之前对输入做层归一化**
- *传入:dim:要进行归一化操作的张量的最后一维的长度*
```python
def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)#层归一化,对最后一维(dim)做层归一化
        self.fn = fn#传入的子模块
```
> 1. 初始化函数:利用父类初始化,添加层归一化函数,传入子模块
>> - 什么是层归一化:在特征维度上进行归一化,不依赖于批次大小,使得每个样本特征均值为0,方差为1
>> - 例如,一个实例的特征为身高,体重,年龄,层归一化就对这三个内容进行操作
>> - 公式:$\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon } }$
>> - 其中,$\mu$为实例各个特征的均值,$\sigma$为各个参数的标准差,$\epsilon$用以防止分母为零报错<br>
<font color='red'>nn.LayerNorm(dim):对输入的样本进行层归一化,其中传入的dim是最后输入样本的最后一维大小</font>
```python
def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)#先归一化输入在传入子模块
```
> 2. 前向传播
先对参数进行层归一化,在执行预定的函数fn

## MLP模块(多层感知机)
```python
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(   #序列化模块
            nn.Linear(dim, hidden_dim), #线性层:输入dim->hidden_dim
            nn.GELU(),#GELU激活函数
            nn.Dropout(dropout),    #drpoout函数防止过拟合
            nn.Linear(hidden_dim, dim), #线性层,hidden_dim->dim
            nn.Dropout(dropout) #再次dropout
        )

    def forward(self, x):
        return self.net(x)#输入通过MLP网络输出
```
- **作用:非线性变换和信息融合**
> 1. 线性层特征升维
> 2. GELU激活函数增强非线性表达能力
> 3. dropout防止过拟合
> 4. 线性层降维会=回原始特征维度
> 5. dropout

## 自注意力模块
```python
class Attention(nn.Module):
    """自注意力模块,实现多头自注意力机制,让每个token关注其他token的信息"""
```
> - **作用:实现多头自注意力机制,让每个token关注其他token的信息**
### 1.初始化函数
```python
def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads  #初始化注意力头数=8
        self.scale = dim ** -0.5  # 缩放因子,防止注意力分数过大

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # 线性层,生成qkv矩阵

        self.nn1 = nn.Linear(dim, dim)#注意力输出后的线性层

        self.do1 = nn.Dropout(dropout)#dropout
```
1. 继承父类初始化函数
2. 初始化注意力头数
3. 缩放因子scale=$\frac{1}{\sqrt{dim}}$其中dim是特征维度
> - 使用缩放因子的目的是防止点积数值结果过大
4. 线性层生成qkv矩阵
5. 线性层输出注意力输出后的线性层
6. dropout,概率设置为0.1<br>
> - <font color="red">nn.Linear(in_features, out_features, bias):in,out分别代表输入和输出特征的维度,都是标量,bias是偏置项,其赋值为布尔值</font>

### 前向传播
```python
def forward(self, x, mask=None):

    b, n, _, h = *x.shape, self.heads   # b=批次大小,n=token数量,h=头数
    qkv = self.to_qkv(x).chunk(3, dim = -1)  # 将输出按照最后一维分成qkv三个张量
    
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

    #计算注意力分数:(Q K^T)/((dim/h)**0.5)
    dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
    mask_value = -torch.finfo(dots.dtype).max

    attn = dots.softmax(dim=-1)  # 注意力权重(按最后一维softmax)
    out = torch.einsum('bhij,bhjd->bhid', attn, v)  # 注意力输出: 权重V,形状(b,h,n,d)
    out = rearrange(out, 'b h n d -> b n (h d)')  # 拼接多头输出
    out = self.nn1(out) #线性层变换
    out = self.do1(out) #dropout
    return out
```
- **函数作用:**
- *函数输入:3维张量x,设置掩膜处理为None*
1. 将输入的张量解包为批次大小b,token数量n和头数h
2. qkv矩阵:通过对输入x进行线性变换在切割为3等份获取qkv矩阵
> - <font color="red">.chunk(3,dim=-1):将张亮沿最后一个维度平均切分为三份</font>
3. 将qkv矩阵的最后一个维度平分给多个头<br>

> - <font color="red">lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h)将最后张量的最后一维hd拆解为h和d并将原本形状为b,n,(hd)的三维张量重塑为b,h,n,d的四维张量</font>
> - 解释:有b个样本,每个样本有h个注意力头,每个头有n个token,每个token有d维特征

4. 计算注意力分数,随后线性变换,dropout

## Transformer类的实现
```python


#分类类别共有16个
```
### 1.初始化函数
```python
def __init__(self, dim, depth, heads, mlp_dim, dropout):
    super().__init__()
    self.layers = nn.ModuleList([])#存储transformer层
    for _ in range(depth):  #depth=transformer层数
        self.layers.append(nn.ModuleList([
            #注意力块:残差连接+预归一化+注意力
            Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
            #MLP块:残差连接+预归一化+MLP
            Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
        ]))
```

#### *函数输入:*
- *dim: 输入特征维度*
- *depth: transformer层数*
- *heads: 多头注意力头数*
- *mlp_dim: mlp模块中隐藏层维度*
- *dropout: dropout层的丢弃概率*

#### 逻辑:
1.  初始化nn.Modukelist容器self.layers存储所有transformer层
2.  循环depth次，每次添加一个由多头注意力机制和MLP组成的层

#### 函数:
<font color='660033'>nn.ModuleList():存储nn.Module子类的容器</font>

### 2.前向传播
```python
def forward(self, x, mask=None):
    for attention, mlp in self.layers:#逐层处理
        x = attention(x, mask=mask)  #先通过注意力层
        x = mlp(x)  # 在通过mlp层
    return x
```
#### 数据流动
- 遍历每一层,先将待处理张量传入attention层,再传入mlp层

# SSFTT

## 1.初始化函数

```python
NUM_CLASS = 16#分类类别共有16个
def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):

    super(SSFTTnet, self).__init__()
    self.L = num_tokens #   Token数量
    self.cT = dim   #   Token维度
    """3d卷积模块"""
    self.conv3d_features = nn.Sequential(
        nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
        nn.BatchNorm3d(8),
        nn.ReLU(),
    )
    """2d卷积模块"""
    self.conv2d_features = nn.Sequential(
        nn.Conv2d(in_channels=8*28, out_channels=64, kernel_size=(3, 3)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )

    # Tokenization机制,这是模型的核心创新点之一
    """WA:用于生成注意力权重,决定如何聚合原始特征"""
    self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),
                                    requires_grad=True)  # Tokenization parameters
    torch.nn.init.xavier_normal_(self.token_wA)
    """WV:用于将原始特征映射到注意力空间"""
    self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                    requires_grad=True)  # Tokenization parameters
    torch.nn.init.xavier_normal_(self.token_wV)

    """位置编码与分类Token
    位置编码:为Token添加位置信息
    分类token:类似bert的[cls]token,用于整体分类决策
    dropout:防止过拟合,应用于嵌入层"""
    self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
    torch.nn.init.normal_(self.pos_embedding, std=.02)

    self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
    self.dropout = nn.Dropout(emb_dropout)

    #transformer编码器:处理token序列,捕获全局依赖关系
    self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
    #分类头:基于cls token进行最终分类预测
    self.to_cls_token = nn.Identity()
    #使用xavier和小标准差的正态分布初始化偏置
    self.nn1 = nn.Linear(dim, num_classes)
    torch.nn.init.xavier_uniform_(self.nn1.weight)
    torch.nn.init.normal_(self.nn1.bias, std=1e-6)
```

### 数据输入:

- in_channels：输入数据的通道数（默认 1，单模态数据）
- num_classes：分类类别数（默认 16）
- num_tokens：生成的语义 Token 数量（默认 4，模型核心创新点之一）
- dim：Token 的维度（默认 64）
- depth：Transformer 编码器的层数（默认 1）
- heads：多头注意力的头数（默认 8）


### 模块:
1. 3d卷积模块:3d卷积后批归一化在进行ReLU
2. 2d卷积模块:2d卷积后批归一化在进行ReLU
3. tokenization机制:
> - 注意力权重矩阵:计算原始特征到目标Token的注意力权重,采用Xavier初始化
> - 特征投影矩阵:用于将原始特征投影到注意力矩阵,采用Xavier初始化
4. 位置编码与分类 Token
> - 位置编码：为 Token 添加位置信息，采用标准差 0.02 的正态分布初始化
> - 分类 Token：类似 BERT 的[CLS]Token，用于最终分类决策，初始化为全零
5. Transformer编码器
6. 分类头:基于分类token输出最终分类结果
7. 线性层:将token维度映射到类别数
```python
#SSFTT的前向传播
def forward(self, x, mask=None):
    """特征提取"""
    x = self.conv3d_features(x)
    #3d卷积(B,1,30,13,13->B,8,28,11,11)
    x = rearrange(x, 'b c h w y -> b (c h) w y')
    #维度重排,合并光谱和通道维数(B,8,28,11,11->B,224,11,11)
    x = self.conv2d_features(x)
    #   2d卷积提取空间特征(B,224,11,11->B,64,9,9)
    x = rearrange(x,'b c h w -> b (h w) c')
    #   维度展平:将空间维度转换为token序列(B,81,64)
    """注意力权重计算"""
    wa = rearrange(self.token_wA, 'b h w -> b w h')
    # 转置注意力权重(1,L,64->1,64,L)
    A = torch.einsum('bij,bjk->bik', x, wa)
    #   通过矩阵乘法计算每个原始token(81)对目标token(L,现在是4)的权重
    A = rearrange(A, 'b h w -> b w h')  # 转置
    A = A.softmax(dim=-1)
    # 归一化,转置并softmax后,A表示每个目标token对原始token的注意力分布
    """特征投影与聚合"""
    VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
    #   特征投影到token空间 Wv前文提到过,是将生成的原始token组成的序列映射到token空间而生成的矩阵
    T = torch.einsum('bij,bjk->bik', A, VV)
    #   加权聚合生成token

    cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)#扩展分类token到批次大小
    x = torch.cat((cls_tokens, T), dim=1)   #   拼接分类token和语义token
    x += self.pos_embedding #   添加位置编码
    x = self.dropout(x) #   dropout防止过拟合
    x = self.transformer(x, mask)  #通过transformer处理序列
    x = self.to_cls_token(x[:, 0])  #   提取分类token
    x = self.nn1(x) #   通过线性层输出分类结果

    return x
```
## 前向传播函数
### 特征提取:
1. 先用 3D 卷积处理输入的时空数据，提取初步特征
2. 通过维度重排合并相关维度，再用 2D 卷积进一步提取空间特征
3. 最终将空间维度展平为序列形式，为 Tokenization 做准备
### Tokenization机制
1. 通过注意力权重矩阵（token_wA）计算原始特征对目标 Token 的关注度
2. 利用特征投影矩阵（token_wV）将原始特征映射到 Token 空间
3. 加权聚合生成语义 Token，实现从原始特征到紧凑 Token 序列的转换
### transformer处理
1. 引入分类 Token（类似 BERT 的 [CLS]）用于最终分类决策
2. 添加位置编码以保留序列位置信息
3. 通过 Transformer 编码器捕获 Token 间的全局依赖关系
### 分类输出
1. 提取分类 Token 的特征
2. 经线性层映射到分类类别数，得到最终预测结果