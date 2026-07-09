import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .priori15 import TopKAttention

'''
baseline13：基于 baseline10 的两项优化：
① base=32 → 24（降低各层通道数）
② ViT mlp_ratio=4.0 → 2.0（缩小 MLP 隐层）
baseline14:
1.将处理快中输出的结果进行concat后送入1*1cbr后再与原图进行一个残差链接
2.将高层的unet中Vit的头数增加
3.加入原图先验       
baseline15:加入top k(25%)   
'''


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
        self.attn = TopKAttention(
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
    坐标注意力作用:降噪,增强空间特征表达。
    CoordAtt → token化 → N个 ViT Block → 还原特征图。

    输入形状: (B, C, H, W)
    输出形状: (B, C, H, W)
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
        # 生成沿通道平均后的灰度先验并升维
        gray = x.mean(dim=1, keepdim=True)
        prior = gray.flatten(2).transpose(1, 2)  # 将原图先验展平为 (B, H*W, 1)
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
    #双模块concat后过一个1*1后和原图相加
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

class UNet3Layer(nn.Module):
    def __init__(self, in_channels: int, base: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(2)
        self.enc1 = _double_conv(in_channels, c1, head_num=4)
        self.enc2 = _double_conv(c1, c2, head_num=8)
        self.enc3 = _double_conv(c2, c3, head_num=32, mlp_ratio=4.0)
        self.up3 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2)
        self.dec3 = _double_conv(c3 + c3, c3, head_num=32, mlp_ratio=4.0)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _double_conv(c2 + c2, c2, head_num=8)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _double_conv(c1 + c1, c1, head_num=4)
        self.final_conv = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        x = self.up2(e3)
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, e1], dim=1))
        return self.final_conv(x)


class UNetClassifier(nn.Module):
    def __init__(self, in_bands: int, num_classes: int = 2):
        super().__init__()
        self.unet = UNet3Layer(in_channels=in_bands, base=24)
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
