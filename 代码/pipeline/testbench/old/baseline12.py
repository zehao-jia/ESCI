import torch
import torch.nn as nn
import torch.nn.functional as F

'''
baseline12：空洞卷积 + 蛇形卷积的多尺度 MSCA（SegNeXtMultiScaleMSCA）：
并联空洞卷积分支 (dilation=1,2,3) 与蛇形卷积，聚合后门控相乘。
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


class SegNeXtMultiScaleMSCA(nn.Module):
    """
    多尺度 MSCA：1×1 投影 → GELU
    → 并联空洞卷积分支 (dilation=1,2,3) + 蛇形卷积分支 → Concat → 1×1 聚合
    → 与投影门控相乘 → 1×1 输出。
    """

    def __init__(self, channels: int, snake_kernel: int = 3):
        super().__init__()
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act = nn.GELU()

        self.dilated_d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dilated_d2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.snake_conv = SnakeConvUnit(channels, channels, kernel_size=snake_kernel)
        self.agg_conv = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.act(self.proj_in(x))
        attn = torch.cat([
            self.dilated_d1(x_proj),
            self.dilated_d2(x_proj),
            self.snake_conv(x_proj),
        ], dim=1)
        attn = self.agg_conv(attn)
        return self.proj_out(attn * x_proj)


class SegNeXtEncoderBlock(nn.Module):
    """SegNeXt 风格编码块：占位 MSCA + ConvFFN（均带残差）。"""

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)

        self.norm1 = nn.BatchNorm2d(channels)
        self.attn = SegNeXtMultiScaleMSCA(channels)

        self.norm2 = nn.BatchNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


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

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class CoordAttViTBlock(nn.Module):
    """
    坐标注意力作用:降噪,增强空间特征表达。
    CoordAtt → token化 → 1个 ViT Block → 还原特征图。

    输入形状: (B, C, H, W)
    输出形状: (B, C, H, W)
    """

    def __init__(self, ch: int, reduction: int = 16, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.coord_att = CoordAtt(ch, reduction=reduction)
        self.norm = nn.LayerNorm(ch)
        self.blocks = nn.Sequential(
            SpatialViTBlock(ch, num_heads=num_heads, mlp_ratio=4.0, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coord_att(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        tokens = self.blocks(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x


class ConvCoordAttViTBlock(nn.Module):
    """
    并联：占位 2D SegNeXt 编码块 + CoordAtt→ViT。
    结构：shortcut + local_encoder(x) + coord_att_vit(x)
    """

    def __init__(self, ch: int, reduction: int = 16,
                 vit_num_heads: int = 4, vit_dropout: float = 0.1):
        super().__init__()
        self.local_encoder = SegNeXtEncoderBlock(ch)
        self.coord_att_vit = CoordAttViTBlock(
            ch, reduction=reduction, num_heads=vit_num_heads, dropout=vit_dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.local_encoder(x) + self.coord_att_vit(x)


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        ConvCoordAttViTBlock(out_ch),
    )

class UNet3Layer(nn.Module):
    """
    对称 U-Net：三次下采样（四档分辨率），三次上采样与跳跃连接一一对应。
    例如输入 16×16 → e1:16 → e2:8 → e3:4 → e4:2（瓶颈）→ 逐级解码回 16×16。
    """

    def __init__(self, in_channels: int, base: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(2)
        self.enc1 = _double_conv(in_channels, c1)
        self.enc2 = _double_conv(c1, c2)
        self.enc3 = _double_conv(c2, c3)
        self.enc4 = _double_conv(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _double_conv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _double_conv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _double_conv(c1 + c1, c1)
        self.final_conv = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        x = self.up3(e4)
        x = self.dec3(torch.cat([x, e3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, e1], dim=1))
        return self.final_conv(x)


class UNetClassifier(nn.Module):
    def __init__(self, in_bands: int, num_classes: int = 2):
        super().__init__()
        self.unet = UNet3Layer(in_channels=in_bands, base=32)
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
