import torch
import torch.nn as nn
import torch.nn.functional as F

'''
baseline22：基于 baseline20 的改进，针对海面溢油检测任务优化
任务特点：
  ① 二分类（海水 vs 溢油）
  ② 标签严重不平衡（海水:溢油 ≈ 99:1）
  ③ 溢油分布不均匀，存在长尾分布

改进点：
  1. SpectralWeightedPrior: 用可学习的 1×1 卷积融合多光谱通道生成先验图，
     替代简单的全局平均，保留油膜与水体的光谱差异信息。
  2. SEBlock: 在 SnakeCoordAttViTBlock 中引入通道注意力，
     增强溢油相关特征通道的表达，抑制背景噪声。
  3. FocalGate: 在注意力 Value 调制后加入门控机制，
     让模型聚焦于难分样本（溢油区域），缓解类别不平衡。
  4. base=26: 适度增加模型容量以学习更精细的油膜特征。
'''

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid = max(8, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpectralWeightedPrior(nn.Module):
    """
    可学习的光谱加权先验。
    使用 1×1 卷积学习每个光谱波段的权重，生成保留光谱差异信息的先验图。
    油膜和水体在不同光谱波段有显著差异 —> 用 learnable weighted sum 替代 uniform mean。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Conv2d(in_channels, max(8, in_channels // 4), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(8, in_channels // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_channels // 4), 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight_net(x)


class FocalGate(nn.Module):
    """
    聚焦门控：对注意力调制后的 Value 应用可学习的门控，
    突出硬样本（溢油区域）的梯度信号，缓解 99:1 的不平衡问题。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, v_modulated: torch.Tensor) -> torch.Tensor:
        gate = self.sigmoid(self.gate_proj(v_modulated))
        return v_modulated * gate


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, use_focal_gate: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_focal_gate = use_focal_gate

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        if use_focal_gate:
            self.focal_gate = FocalGate(self.head_dim)

    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
        is_4d = x.dim() == 4
        if is_4d:
            B, C, H, W = x.shape
            N = H * W
            x = x.view(B, C, N).transpose(1, 2)
        else:
            B, N, C = x.shape
            H, W = priori.shape[2:]

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        priori = priori.view(B, -1, N).transpose(1, 2).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v * priori

        if self.use_focal_gate:
            v = self.focal_gate(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        if is_4d:
            return x.transpose(1, 2).reshape(B, C, H, W)
        return x


class attention_with_priori(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, use_focal_gate: bool = True):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout, use_focal_gate=use_focal_gate)
        self.conv_priori = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if priori.shape[2:] != x.shape[2:]:
                priori = F.adaptive_avg_pool2d(priori, output_size=x.shape[2:])
        else:
            N = x.shape[1]
            H = W = int(N ** 0.5)
            if priori.shape[2:] != (H, W):
                priori = F.adaptive_avg_pool2d(priori, output_size=(H, W))
        priori = self.conv_priori(priori)
        return self.attn(x, priori)


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
    """用于空间 token 序列的单个 ViT Block（Pre-LN）。"""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1,
                 use_focal_gate: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attention_with_priori(dim, num_heads=num_heads, dropout=dropout, use_focal_gate=use_focal_gate)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        x = x + self.attn(attn_in, priori)
        return x + self.mlp(self.norm2(x))


class CoordAttViTBlock(nn.Module):
    """
    坐标注意力作用:降噪,增强空间特征表达。
    CoordAtt → token化 → 1个 ViT Block → 还原特征图。

    输入形状: (B, C, H, W)
    输出形状: (B, C, H, W)
    """

    def __init__(self, ch: int, reduction: int = 16, num_heads: int = 4, dropout: float = 0.1,
                 use_focal_gate: bool = True):
        super().__init__()
        self.coord_att = CoordAtt(ch, reduction=reduction)
        self.norm = nn.LayerNorm(ch)
        self.block = SpatialViTBlock(ch, num_heads=num_heads, mlp_ratio=2.0, dropout=dropout,
                                     use_focal_gate=use_focal_gate)

    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
        x = self.coord_att(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        tokens = self.block(tokens, priori)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x


class SnakeCoordAttViTBlock(nn.Module):
    """
    蛇形卷积与 CoordAtt→ViT 的并联加残差块。
    结构：shortcut + snake(x) + coord_att_vit(x) + SE recalibration
    """

    def __init__(self, ch: int, snake_kernel_size: int = 3, reduction: int = 16,
                 vit_num_heads: int = 4, vit_dropout: float = 0.1, use_focal_gate: bool = True):
        super().__init__()
        self.snake = SnakeConvUnit(ch, ch, kernel_size=snake_kernel_size)
        self.se = SEBlock(ch, reduction=reduction)
        self.coord_att_vit = CoordAttViTBlock(
            ch, reduction=reduction, num_heads=vit_num_heads, dropout=vit_dropout,
            use_focal_gate=use_focal_gate
        )

    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
        snake_out = self.snake(x)
        snake_out = self.se(snake_out)
        return x + snake_out + self.coord_att_vit(x, priori)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, num_heads=4, use_focal_gate: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.block = SnakeCoordAttViTBlock(out_ch, vit_num_heads=num_heads, use_focal_gate=use_focal_gate)

    def forward(self, x, priori):
        x = self.relu(self.bn(self.conv(x)))
        return self.block(x, priori)


class UNet3Layer(nn.Module):
    def __init__(self, in_channels: int, base: int = 26):
        super().__init__()
        self.spectral_prior = SpectralWeightedPrior(in_channels)
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_channels, c1, num_heads=1)
        self.enc2 = DoubleConv(c1, c2, num_heads=2)
        self.enc3 = DoubleConv(c2, c3, num_heads=4)
        self.enc4 = DoubleConv(c3, c4, num_heads=8)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3, num_heads=4)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2, num_heads=2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1, num_heads=1)
        self.final_conv = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pri = self.spectral_prior(x)  # (B, 1, H, W) — learnable spectral weighted prior
        e1 = self.enc1(x, pri)
        e2 = self.enc2(self.pool(e1), pri)
        e3 = self.enc3(self.pool(e2), pri)
        e4 = self.enc4(self.pool(e3), pri)
        x = self.up3(e4)
        x = self.dec3(torch.cat([x, e3], dim=1), pri)
        x = self.up2(x)
        x = self.dec2(torch.cat([x, e2], dim=1), pri)
        x = self.up1(x)
        x = self.dec1(torch.cat([x, e1], dim=1), pri)
        return self.final_conv(x)


class UNetClassifier(nn.Module):
    def __init__(self, in_bands: int, num_classes: int = 2):
        super().__init__()
        self.unet = UNet3Layer(in_channels=in_bands, base=26)
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
    B, C, H, W = 4, 30, 16, 16
    x = torch.randn(B, 1, C, H, W)
    net = UNetClassifier(in_bands=C, num_classes=2)
    y = net(x)
    print("输入:", tuple(x.shape), "输出 logits:", tuple(y.shape))
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"参数量约 {n_params:.2f} M")
