"""
高光谱海面溢油检测 — baseline7（RWKV 与蛇形卷积在 U-Net 层内并联）

设计目标：
1) 不再使用 baseline6 的“UNet 分支 + RWKV 分支”并联结构；
2) 在 U-Net 的每个层级内，蛇形卷积与 RWKV 路径并行提取特征，再融合；
3) 在 U-Net 层级之间加入 token-shift（作用于下采样后的特征），
   让层间信息混合后再进入下一层编码。

输入: (B, 1, C, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeConvUnit(nn.Module):
    """蛇形卷积单元：并联 1xk 与 kx1 方向卷积后，用 1x1 融合。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 需为奇数")
        pad = kernel_size // 2
        # 在h方向上进行卷积
        self.conv_h = nn.Conv2d(
            in_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, pad), bias=False
        )
        # 在v方向上进行卷积,这是空间方向
        self.conv_v = nn.Conv2d(
            in_ch, out_ch, kernel_size=(kernel_size, 1), padding=(pad, 0), bias=False
        )
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先进行两个方向上的条带卷积,再拼接融合经过一个1*1卷积,最后BN和激活
        h_feat = self.conv_h(x)
        v_feat = self.conv_v(x)
        x = torch.cat([h_feat, v_feat], dim=1)
        x = self.fuse(x)
        x = self.bn(x)
        return self.act(x)
'''
class RWKV2DUnit(nn.Module):
    """
    优化版 2D-RWKV
    提速点：
    1. 移除自定义 WKV 反向传播，改用官方高效实现
    2. 用 1x1 卷积替代 Linear，避免展平
    3. 移除冗余 token_shift / LayerNorm
    4. 合并维度变换，减少内存拷贝
    5. 只保留 1 层 TimeMixing（原两层太冗余）
    """
    def __init__(self, in_ch: int, out_ch: int, num_heads: int = 4):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_heads = num_heads
        assert out_ch % num_heads == 0
        self.head_dim = out_ch // num_heads

        # 【优化1】用 Conv1x1 替代 Linear，不用展平 token
        self.in_proj = nn.Conv2d(in_ch, out_ch * 3, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

        # 可学习时间衰减
        self.time_decay = nn.Parameter(torch.zeros(num_heads))
        self.time_first = nn.Parameter(torch.zeros(num_heads))

        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.size()
        T = H * W

        # 【优化2】一次卷积得到 kvr，不做 token 展平
        qkv = self.in_proj(x)  # (B, 3*C, H, W)
        k, v, r = qkv.chunk(3, dim=1)  # 三个分支 (B, C, H, W)

        # 【优化3】直接 reshape 成序列，不 transpose 乱序
        k = k.flatten(2).permute(0, 2, 1).contiguous()  # (B, T, C)
        v = v.flatten(2).permute(0, 2, 1).contiguous()
        r = r.flatten(2).permute(0, 2, 1).contiguous()

        # 【优化4】使用 PyTorch 原生 cumsum 加速 WKV
        H = self.num_heads
        D = self.head_dim

        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)
        r = r.view(B, T, H, D)

        # RWKV 核心公式（官方最快速版本，无自定义 backward）
        exp_w = torch.exp(-self.time_decay).view(1, 1, H, 1)
        exp_u = torch.exp(self.time_first).view(1, 1, H, 1)
        exp_k = torch.exp(k)

        # 并行 WKV 计算
        kv = exp_k * v
        A = torch.cumsum(kv * exp_w ** torch.arange(T, device=k.device).view(1, T, 1, 1), dim=1)
        B_sum = torch.cumsum(exp_k * exp_w ** torch.arange(T, device=k.device).view(1, T, 1, 1), dim=1)
        A = A / exp_w ** torch.arange(T, device=k.device).view(1, T, 1, 1)
        B_sum = B_sum / exp_w ** torch.arange(T, device=k.device).view(1, T, 1, 1)

        A_shift = torch.zeros_like(A)
        A_shift[:, 1:] = A[:, :-1]
        B_shift = torch.zeros_like(B_sum)
        B_shift[:, 1:] = B_sum[:, :-1]

        wkv = (exp_u * kv + A_shift) / (exp_u * exp_k + B_shift + 1e-8)

        # 门控
        r = torch.sigmoid(r)
        out = r * wkv

        # 【优化5】快速恢复 2D 特征图
        out = out.view(B, T, self.out_ch).permute(0, 2, 1).view(B, self.out_ch, H, W)
        out = self.out_proj(out)
        out = self.norm(out)
        return x + self.act(out)

'''

class RWKV2DUnit(nn.Module):
    '''
    这个不是真正意义上的rwkv,而是借鉴了rwkv的设计思想,
    在空间维度上进行token化,并引入了token-shift机制来增强空间信息混合.
    '''

    def __init__(self, in_ch: int, out_ch: int, shift_ratio: float = 0.5):
        super().__init__()
        if not (0.0 <= shift_ratio <= 1.0):
            raise ValueError("shift_ratio 需在 [0,1] 范围内")
        self.shift_ratio = shift_ratio
        # 1*1卷积做通道变换，后续的 DW 卷积和线性层都在 out_ch 维度上操作
        self.in_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        # DW 卷积做局部混合，groups=out_ch 保持通道独立，kernel_size=3 保持空间尺寸不变
        self.dw_mix = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False
        )
        self.norm = nn.LayerNorm(out_ch)
        self.receptance = nn.Linear(out_ch, out_ch, bias=False)
        self.key = nn.Linear(out_ch, out_ch, bias=False)
        self.value = nn.Linear(out_ch, out_ch, bias=False)
        self.out = nn.Linear(out_ch, out_ch, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    '''
    token-shift 机制：将每个 token 的前 shift_c 个通道向后移动一个位置，形成跨 token 的特征混合。
    数学表示：
        对于 token i (i从0开始):
    - shifted[i, :shift_c] = tokens[i-1, :shift_c]  (当 i>0)
    - shifted[i, :shift_c] = 0                     (当 i=0)
    - shifted[i, shift_c:] = tokens[i, shift_c:]   (所有 i)
    
    实际影响示例：
    假设 c=4, shift_ratio=0.5 (移位2个通道)：
    原始 tokens:
    token0: [a1, a2, a3, a4]
    token1: [b1, b2, b3, b4]
    token2: [c1, c2, c3, c4]
    移位后:
    token0: [0,  0,  a3, a4]  # 前2通道变0
    token1: [a1, a2, b3, b4]  # 前2通道来自token0
    token2: [b1, b2, c3, c4]  # 前2通道来自token1
    '''
    def _token_shift(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t, c = tokens.shape
        shift_c = int(c * self.shift_ratio)             # 给一个比率,计算出需要移动的通道数
        shifted = tokens.new_zeros((b, t, shift_c))     # 将
        shifted[:, 1:, :] = tokens[:, :-1, :shift_c]    # 将原始 tokens 的前 shift_c 个通道向后移动一个位置,第一个位置补0
        return torch.cat([shifted, tokens[:, :, shift_c:]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = x + self.dw_mix(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)       # 将 (B, C, H, W) 展平为 (B, C, H*W)，再转置为 (B, H*W, C)，每个空间位置是一个 token
        tokens = self.norm(tokens)                  # 对 token 进行 LayerNorm，保持数值稳定,这个LN就是只对最后一个维度进行归一化,也就是对每个 token 的特征进行归一化
        tokens = self._token_shift(tokens)          # 进行tokenshift,让相邻token的部分通道进行信息混合

        r = torch.sigmoid(self.receptance(tokens))  # RWKV 的门控机制，生成一个 (B, T, C) 的门控权重 r，范围在0到1之间
        k = self.key(tokens)                        # 生成键值 k 和 v，都是 (B, T, C) 维度
        v = self.value(tokens)
        rwkv = r * (k * v)                          # 模拟 RWKV 的混合机制，简单地用元素乘来融合 k 和 v，再乘以门控权重 r   
        tokens = tokens + self.out(rwkv)            # 最后通过一个线性层融合 RWKV 输出，并与原 tokens 做残差连接

        x = tokens.transpose(1, 2).reshape(b, c, h, w) # 将 token 还原回 (B, C, H, W) 形状
        x = self.bn(x)
        return self.act(x)



class ParallelSnakeRWKVBlock(nn.Module):
    """
    单层并联块：蛇形卷积路径 + RWKV 路径并行，拼接后融合。
    该融合输出可直接作为 skip feature。
    这里的维度数不进行调整,升维应在前面做
    """

    def __init__(self, in_ch, ch, rwkv_shift_ratio: float = 0.5):
        super().__init__()
        self.snake = nn.Sequential(
            SnakeConvUnit(ch, ch, kernel_size=3),
            SnakeConvUnit(ch, ch, kernel_size=3),
        )
        self.rwkv = nn.Sequential(
            RWKV2DUnit(ch, ch, shift_ratio=rwkv_shift_ratio),
            RWKV2DUnit(ch, ch, shift_ratio=rwkv_shift_ratio),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.dimension_elevation = nn.Sequential(
            nn.Conv2d(in_ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dimension_elevation(x)  # 升维到 snake 和 rwkv 的输入维度
        x_s = self.snake(x)
        x_r = self.rwkv(x)
        return self.merge(torch.cat([x_s, x_r], dim=1))


class HybridUNet3LayerSnakeRWKV(nn.Module):
    """
    3 层 U-Net：
    - 每层内部：Snake 与 RWKV 并联后融合；
    - 层间：下采样后先做 token-shift 再进入下一编码层。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base: int = 32,
        rwkv_shift_ratio: float = 0.5,
        inter_level_shift_ratio: float = 0.25,
    ):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ParallelSnakeRWKVBlock(in_channels, c1, rwkv_shift_ratio)
        # self.shift1 = InterLevelTokenShift2D(inter_level_shift_ratio)         #这个shift是将特征图变换为Token后又进行rwkv,要删去
        self.enc2 = ParallelSnakeRWKVBlock(c1, c2, rwkv_shift_ratio)
        # self.shift2 = InterLevelTokenShift2D(inter_level_shift_ratio)
        self.enc3 = ParallelSnakeRWKVBlock(c2, c3, rwkv_shift_ratio)
        # self.shift3 = InterLevelTokenShift2D(inter_level_shift_ratio)
        self.bottleneck = ParallelSnakeRWKVBlock(c3, c4, rwkv_shift_ratio)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ParallelSnakeRWKVBlock(c3 + c3, c3, rwkv_shift_ratio)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ParallelSnakeRWKVBlock(c2 + c2, c2, rwkv_shift_ratio)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ParallelSnakeRWKVBlock(c1 + c1, c1, rwkv_shift_ratio)
        self.final_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        b = self.bottleneck(p3)
        x = self.up3(b)
        x = self.dec3(torch.cat([x, e3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, e1], dim=1))
        return self.final_conv(x)



class SnakeRWKVUNet2DBranch(nn.Module):
    """baseline7 主分支：层内并联 Snake+RWKV 的 U-Net -> 特征图。"""

    def __init__(
        self,
        in_bands: int,
        base: int = 32,
        rwkv_shift_ratio: float = 0.5,
        inter_level_shift_ratio: float = 0.25,
    ):
        super().__init__()
        self.unet = HybridUNet3LayerSnakeRWKV(
            in_channels=in_bands,
            out_channels=in_bands,
            base=base,
            rwkv_shift_ratio=rwkv_shift_ratio,
            inter_level_shift_ratio=inter_level_shift_ratio,
        )
        # self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.unet(x)
        # self.last_moe_aux_loss = x.new_zeros(())
        return feat_map


class ParallelSnakeRWKVUNetNet(nn.Module):
    """
    baseline7 分类网络：
    输入先经层内并联 Snake+RWKV 的 U-Net 主干得到特征图，
    直接从中心位置提取特征并接分类头（不使用池化）。
    """

    def __init__(
        self,
        in_bands: int,
        patch_size: int,
        branch_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.4,
        rwkv_shift_ratio: float = 0.5,
        inter_level_shift_ratio: float = 0.25,
    ):
        super().__init__()
        self.in_bands = in_bands
        self.patch_size = patch_size
        # self.use_moe = use_moe

        self.branch_hybrid_unet = SnakeRWKVUNet2DBranch(
            in_bands=in_bands,
            base=32,
            rwkv_shift_ratio=rwkv_shift_ratio,
            inter_level_shift_ratio=inter_level_shift_ratio,
        )
        hidden_dim = max(in_bands // 2, 32)
        self.fuse = nn.Sequential(
            nn.Linear(in_bands, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        # self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5 or x.size(1) != 1:
            raise ValueError(f"期望输入 (B, 1, C, H, W)，得到 {tuple(x.shape)}")
        b1 = x.squeeze(1)
        feat_map = self.branch_hybrid_unet(b1)  # (B, C, H, W)
        _, _, h, w = feat_map.shape
        z = feat_map[:, :, h // 2, w // 2]  # 直接取中心位置特征做分类，不做池化

        # aux_branch = getattr(self.branch_hybrid_unet, "last_moe_aux_loss", z.new_zeros(()))
        # if self.moe is not None:
        #     z, aux = self.moe(z)
        #     self.last_moe_aux_loss = aux_branch + aux
        # else:
        #     self.last_moe_aux_loss = aux_branch
        return self.fuse(z)


def build_tri_branch_net(
    sample_x: torch.Tensor,
    num_classes: int = 2,
    branch_dim: int = 128,
    dropout: float = 0.4,

    rwkv_shift_ratio: float = 0.5,
    inter_level_shift_ratio: float = 0.25,
) -> ParallelSnakeRWKVUNetNet:
    """供 IP_train.py 调用的兼容入口（保留历史函数名）。"""
    if sample_x.dim() != 5:
        raise ValueError("sample_x 应为 (B, 1, C, H, W)")
    _, _, c, h, w = sample_x.shape
    if h != w:
        raise ValueError(f"当前实现假定方形 patch，得到 H={h}, W={w}")
    return ParallelSnakeRWKVUNetNet(
        in_bands=c,
        patch_size=h,
        branch_dim=branch_dim,
        num_classes=num_classes,
        dropout=dropout,
        rwkv_shift_ratio=rwkv_shift_ratio,
        inter_level_shift_ratio=inter_level_shift_ratio,
    )


def build_classifier_net(
    sample_x: torch.Tensor,
    device: torch.device,
    branch_dim: int = 128,
    dropout: float = 0.4,
    **kwargs,
) -> nn.Module:
    return build_tri_branch_net(
        sample_x,
        num_classes=2,
        branch_dim=branch_dim,
        dropout=dropout,
        **kwargs,
    ).to(device)


if __name__ == "__main__":
    b, c, h, w = 4, 30, 16, 16
    x = torch.randn(b, 1, c, h, w)
    net = ParallelSnakeRWKVUNetNet(in_bands=c, patch_size=h, num_classes=2)
    y = net(x)
    print("输入:", tuple(x.shape), "输出 logits:", tuple(y.shape))
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"参数量约 {n_params:.2f} M")
