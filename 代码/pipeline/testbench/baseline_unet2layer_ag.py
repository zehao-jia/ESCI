import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .old.PrioriVit import Multihead_self_attention

'''
baseline_unet2layer_ag：在 baseline_unet2layer 基础上加入 Attention Gate。
改动：
  ① UNet 编码层 3→2（bottleneck 在 4×4）
  ② 跳跃连接中加入加法注意力门（Attention Gate），用解码器特征过滤编码器噪声
  ③ 移除未使用的 dec3/up3

Attention Gate 来源：Oktay et al. "Attention U-Net" (2018)
  - 加法注意力（W_x·x + W_g·g），比乘法注意力收敛更好
  - 小样本下 DSC 提升 +2.6%，参数量仅增 ~8%
  - 反向传播时背景区域梯度被自动降权，浅层专注前景
'''


# ===================== 基础组件（与 baseline 相同） =====================

class SnakeConvUnit(nn.Module):
    """蛇形卷积单元：并联 1xk 与 kx1 方向卷积后，用 1x1 融合。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 需为奇数")
        pad = kernel_size // 2
        self.conv_h = nn.Conv2d(
            in_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, pad), bias=False
        )
        self.conv_v = nn.Conv2d(
            in_ch, out_ch, kernel_size=(kernel_size, 1), padding=(pad, 0), bias=False
        )
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_feat = self.conv_h(x)
        v_feat = self.conv_v(x)
        x = torch.cat([h_feat, v_feat], dim=1)
        x = self.fuse(x)
        x = self.bn(x)
        return self.act(x)


class CoordAtt(nn.Module):
    """
    坐标注意力 (Coordinate Attention, CVPR 2021)
    对 H 和 W 方向分别池化，编码位置信息后作为注意力权重。
    """

    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid = max(8, in_ch // reduction)
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv_h = nn.Conv2d(mid, in_ch, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, in_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.relu(y, inplace=True)
        x_h, x_w = y.split([h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w


class SpatialViTBlock(nn.Module):
    """用于空间 token 序列的单个 ViT Block（Pre-LN），已支持先验 gating。"""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        head_dim = dim // num_heads
        self.attn = Multihead_self_attention(
            heads=num_heads,
            head_dim=head_dim,
            dim=dim,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, prior=prior)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class CoordAttViTBlock(nn.Module):
    """
    CoordAtt → token化 → N个 ViT Block → 还原特征图。
    输入形状: (B, C, H, W)   输出形状: (B, C, H, W)
    """

    def __init__(self, ch: int, reduction: int = 16, num_heads: int = 4,
                 mlp_ratio: float = 2.0,
                 num_blocks: int = 1, dropout: float = 0.1):
        super().__init__()
        self.coord_att = CoordAtt(ch, reduction=reduction)
        self.norm = nn.LayerNorm(ch)
        self.prior_proj = nn.Linear(1, ch)
        self.blocks = nn.ModuleList([
            SpatialViTBlock(ch, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coord_att(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        gray = x.mean(dim=1, keepdim=True)
        prior = gray.flatten(2).transpose(1, 2)
        prior = self.prior_proj(prior)
        for block in self.blocks:
            tokens = block(tokens, prior)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x


class SnakeCoordAttViTBlock(nn.Module):
    """
    蛇形卷积与 CoordAtt→ViT 的并联加残差块。
    结构：shortcut + snake(x) + coord_att_vit(x)
    """

    def __init__(self, ch: int, snake_kernel_size: int = 3, reduction: int = 16,
                 vit_num_heads: int = 4, vit_mlp_ratio: float = 2.0,
                 vit_num_blocks: int = 1, vit_dropout: float = 0.1):
        super().__init__()
        self.snake = SnakeConvUnit(ch, ch, kernel_size=snake_kernel_size)
        self.coord_att_vit = CoordAttViTBlock(
            ch, reduction=reduction, num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            num_blocks=vit_num_blocks, dropout=vit_dropout
        )
        self.conv_1d = nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        snake_out = self.snake(x)
        vit_out = self.coord_att_vit(x)
        combined = torch.cat([snake_out, vit_out], dim=1)
        return x + self.conv_1d(combined)


def _double_conv(in_ch: int, out_ch: int, head_num: int,
                 mlp_ratio: float = 2.0,
                 num_blocks: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        SnakeCoordAttViTBlock(out_ch, vit_num_heads=head_num,
                              vit_mlp_ratio=mlp_ratio,
                              vit_num_blocks=num_blocks),
    )


# ===================== 新增：Attention Gate =====================

class AttentionGate(nn.Module):
    """
    加法注意力门 (Additive Attention Gate)，来自 Oktay et al. "Attention U-Net" (2018).

    用解码器的粗尺度上下文信息 g（门控信号）过滤编码器跳跃连接特征 x，
    输出 x * α，其中 α ∈ [0,1] 是空间注意力系数。

    公式:
        q = ψ^T · ReLU(W_x·x + W_g·g + b_g) + b_ψ
        α = sigmoid(q)
        output = x · α

    Args:
        F_g: 门控信号 g 的通道数（来自解码器/上采样）
        F_l: 跳跃连接特征 x 的通道数（来自编码器）
        F_int: 中间特征通道数（默认 F_l // 2）
    """

    def __init__(self, F_g: int, F_l: int, F_int: Optional[int] = None):
        super().__init__()
        if F_int is None:
            F_int = max(1, F_l // 2)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, bias=True)

        # 初始化：偏置使初始 α ≈ 0.5，所有位置都能通过特征（训练初期无偏好）
        nn.init.constant_(self.psi.bias, 0.0)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 编码器跳跃连接特征 (B, F_l, H, W)
            g: 解码器门控信号     (B, F_g, H_g, W_g)
        Returns:
            门控后的特征 (B, F_l, H, W)，值域与 x 相同
        """
        # 若空间尺寸不一致，将 g 上采样到 x 的尺寸
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)

        g1 = self.W_g(g)                        # (B, F_int, H, W)
        x1 = self.W_x(x)                        # (B, F_int, H, W)
        q = F.relu(g1 + x1, inplace=True)       # 加法融合
        q = self.psi(q)                          # (B, 1, H, W)
        alpha = torch.sigmoid(q)                 # α ∈ [0, 1]
        return x * alpha


# ===================== 2 层 UNet + Attention Gate =====================

class UNet2LayerAG(nn.Module):
    """
    2 层 UNet + Attention Gate：适用于 9×9 输入。

    空间尺寸变化:
      enc1:         (B, C, 9, 9) → (B, c1, 9, 9)
        → pool → (B, c1, 4, 4)
      enc2:         (B, c1, 4, 4) → (B, c2, 4, 4)   ← bottleneck
      up1:          (B, c2, 4, 4) → (B, c1, 9, 9)   ← 门控信号 g
      attn_gate:    x=e1, g=up1  → (B, c1, 9, 9)   ← 过滤后的跳跃连接
      dec1:         concat(up1, attn_gate(e1)) → (B, c1, 9, 9)
      final_conv:   (B, c1, 9, 9) → (B, 2, 9, 9)
    """

    def __init__(self, in_channels: int, base: int = 24):
        super().__init__()
        c1, c2 = base, base * 2  # 24, 48
        self.pool = nn.MaxPool2d(2)

        # 编码器
        self.enc1 = _double_conv(in_channels, c1, head_num=4)         # 9×9, 81 tokens
        self.enc2 = _double_conv(c1, c2, head_num=8)                  # 4×4, 16 tokens

        # 解码器
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2,
                                       output_padding=1)               # 4→9
        # Attention Gate: 用 up1 的门控信号过滤 enc1 的跳跃连接
        self.attn_gate = AttentionGate(F_g=c1, F_l=c1)

        self.dec1 = _double_conv(c1 + c1, c1, head_num=4)             # concat 后 2*c1 通道
        self.final_conv = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                                              # (B, c1, 9, 9)
        e2 = self.enc2(self.pool(e1))                                  # (B, c2, 4, 4)
        d1 = self.up1(e2)                                              # (B, c1, 9, 9)
        e1_gated = self.attn_gate(e1, d1)                              # 门控过滤跳跃连接
        x = self.dec1(torch.cat([d1, e1_gated], dim=1))               # (B, c1, 9, 9)
        return self.final_conv(x)                                      # (B, 2, 9, 9)


class UNetClassifier(nn.Module):
    def __init__(self, in_bands: int, num_classes: int = 2):
        super().__init__()
        self.unet = UNet2LayerAG(in_channels=in_bands, base=24)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, num_classes)
        self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = x.squeeze(1)
        seg_logits = self.unet(b1)
        logits = self.fc(self.gap(seg_logits).flatten(1))
        self.last_moe_aux_loss = x.new_zeros(())
        return logits


def build_tri_branch_net(
    sample_x: torch.Tensor,
    num_classes: int = 2,
    branch_dim: int = 128,
    dropout: float = 0.4,
    **kwargs,
) -> UNetClassifier:
    if sample_x.dim() != 5:
        raise ValueError("sample_x 应为 (B, 1, C, H, W)")
    _, _, c, h, w = sample_x.shape
    return UNetClassifier(in_bands=c, num_classes=num_classes)


if __name__ == "__main__":
    B, C, H, W = 4, 30, 9, 9
    x = torch.randn(B, 1, C, H, W)
    net = UNetClassifier(in_bands=C, num_classes=2)
    y = net(x)
    print("输入:", tuple(x.shape), "输出 logits:", tuple(y.shape))
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"参数量约 {n_params:.2f} M")
