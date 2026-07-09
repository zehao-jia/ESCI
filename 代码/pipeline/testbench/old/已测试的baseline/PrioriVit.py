import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 预处理:将输入图像切分成 patch，线性映射到 ViT 的维度，并添加位置编码和 CLS token。

class pre_proces(nn.Module):
    def __init__(self, image_size, patch_size, patch_dim, dim):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.patch_num = (image_size//patch_size)**2
        self.linear_embedding = nn.Linear(patch_dim, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, self.patch_num+1, self.dim))  # 使用广播
        self.CLS_token = nn.Parameter(torch.randn(1, 1, self.dim))  # 别忘了维度要和 (B,L,C) 对齐

    def forward(self, x):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)  # (B,L,C)
        x = self.linear_embedding(x)
        b, l, c = x.shape   # 获取 token 的形状 (B,L,c)
        CLS_token = repeat(self.CLS_token, '1 1 d -> b 1 d', b=b)  # 位置编码复制 B 份
        x = torch.concat((CLS_token, x), dim=1)
        x = x+self.position_embedding
        return x
    
class Multihead_self_attention(nn.Module):
    def __init__(self, heads, head_dim, dim):
        super().__init__()
        self.head_dim = head_dim    # 每一个注意力头的维度
        self.heads = heads  # 注意力头个数
        self.inner_dim = self.heads*self.head_dim  # 多头自注意力最后的输出维度
        self.scale = self.head_dim**-0.5   # 正则化系数
        self.to_qkv = nn.Linear(dim, self.inner_dim*3)  # 生成 qkv，每一个矩阵的维度和由自注意力头的维度以及头的个数决定
        self.to_output = nn.Linear(self.inner_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.prior_proj = (nn.Linear(dim, self.inner_dim) if self.inner_dim != dim
                           else nn.Identity())

    def forward(self, x, prior=None):
        x = self.norm(x)    # PreNorm
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 划分 QKV，返回一个列表，其中就包含了 QKV
        Q, K, V = map(lambda t: rearrange(t, 'b l (h dim) -> b h l dim', dim=self.head_dim), qkv)
        if prior is not None:
            prior = self.prior_proj(prior)
            prior_heads = rearrange(
                prior, 'b l (h d) -> b h l d', h=self.heads, d=self.head_dim
            )
            V = V * prior_heads  # 先验与 V 矩阵逐元素相乘
        K_T = K.transpose(-1, -2)
        att_score = Q@K_T*self.scale
        att = self.softmax(att_score)
        out = att@V   # (B,H,L,dim)
        out = rearrange(out, 'b h l dim -> b l (h dim)')  # 拼接
        output = self.to_output(out)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Transformer_block(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim):
        super().__init__()
        self.MHA = Multihead_self_attention(heads=heads, head_dim=head_dim, dim=dim)
        self.FeedForward = FeedForward(dim=dim, mlp_dim=mlp_dim)

    def forward(self, x, prior=None):
        x = self.MHA(x, prior)+x
        x = self.FeedForward(x)+x
        return x
    
# ==================== AI 补全: Transformer 类（堆叠多个 Transformer_block）====================
class Transformer(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Transformer_block(dim=dim, heads=heads, head_dim=head_dim, mlp_dim=mlp_dim))

    def forward(self, x, prior=None):
        for layer in self.layers:
            x = layer(x, prior)
        return x
# # ==================== AI 补全结束 ====================

# class ViT(nn.Module):
#     def __init__(self, image_size, channels, patch_size, dim, heads, head_dim, mlp_dim, depth, num_class):
#         super().__init__()
#         self.dim = dim
#         self.patch_size = patch_size
#         self.to_patch_embedding = pre_proces(image_size=image_size, patch_size=patch_size, patch_dim=channels*patch_size**2, dim=dim)
#         self.grayscale_prior = nn.Linear(patch_size**2, dim)
#         # ==================== AI 修复: 使用 Transformer 代替 Transformer_block ====================
#         self.transformer = Transformer(dim=dim, heads=heads, head_dim=head_dim, mlp_dim=mlp_dim, depth=depth)
#         # ==================== AI 修复结束 ====================
#         self.MLP_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_class)
#         )
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         token = self.to_patch_embedding(x)
#         b = token.shape[0]
#         # 构造灰度先验 token 并补齐 CLS 位置
#         gray = x.mean(dim=1, keepdim=True)
#         gray_patches = rearrange(
#             gray, 'b 1 (h p1) (w p2) -> b (h w) (p1 p2)',
#             p1=self.patch_size, p2=self.patch_size
#         )
#         prior_tokens = self.grayscale_prior(gray_patches)
#         cls_prior = torch.zeros(b, 1, self.dim, device=prior_tokens.device, dtype=prior_tokens.dtype)
#         prior_tokens = torch.cat([cls_prior, prior_tokens], dim=1)
#         output = self.transformer(token, prior=prior_tokens)
#         CLS_token = output[:, 0, :] # 提取出 CLS Token
#         out = self.softmax(self.MLP_head(CLS_token))
#         return out
