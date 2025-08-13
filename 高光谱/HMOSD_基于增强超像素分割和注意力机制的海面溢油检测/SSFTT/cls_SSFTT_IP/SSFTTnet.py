import torch
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from torch import nn


def _weights_init(m):
    """权重初始化函数,这个函数首先获取模块的类名,如果是3d卷积层或线性层,则使用kaiming正则化"""
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    """残差链接模块"""
    def __init__(self, fn):
        super().__init__()#调用父类的初始化函数
        self.fn = fn#传入的子模块(如注意力层或MLP)

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x#子模块输出+原始输入

# 等于 PreNorm
class LayerNormalize(nn.Module):
    """预归一化模块,在运行子模块之前对输入做层归一化,稳定训练过程,提高收敛速度"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)#层归一化,对最后一维(dim)做层归一化
        self.fn = fn#传入的子模块

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)#先归一化输入在传入子模块

# 等于 FeedForward
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


class Attention(nn.Module):
    """自注意力模块,实现多头自注意力机制,让每个token关注其他token的信息"""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads  #初始化注意力头数=8
        self.scale = dim ** -0.5  # 缩放因子,防止注意力分数过大

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # 线性层,生成qkv矩阵
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)#注意力输出后的线性层
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)#dropout

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads   # b=批次大小,n=token数量,h=头数
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 将输出按照最后一维分成qkv三个张量
        """将qkv矩阵的最后一个维度平分给多个头形状变为b,h,n,dim/h"""
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        #计算注意力分数:(Q K^T)/((dim/h)**0.5)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        #掩码处理,代码实际未使用
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # 注意力权重(按最后一维softmax)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # 注意力输出: 权重V,形状(b,h,n,d)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 拼接多头输出
        out = self.nn1(out) #线性层变换
        out = self.do1(out) #dropout
        return out


class Transformer(nn.Module):
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

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:#逐层处理
            x = attention(x, mask=mask)  #先通过注意力层
            x = mlp(x)  # 在通过mlp层
        return x

NUM_CLASS = 2#分类类别共有2个(背景/溢油)

class SSFTTnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        """模型的主要参数包括:
        in_channels:输入通道数(表示单模态)   num_classes:分类类别数   num_tokens:生成的Token数量(默认为4)
        dim:Token的维数    depth:transformer的层数    heads:注意力头数
        """
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
            nn.Conv2d(in_channels=8*13, out_channels=64, kernel_size=(3, 3)),
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


if __name__ == '__main__':
    model = SSFTTnet()
    input = torch.randn(16, 1, 15, 5, 5)
    y = model(input)#触发forward函数
    print(y.size())
