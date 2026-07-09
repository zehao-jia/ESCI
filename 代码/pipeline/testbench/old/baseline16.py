import torch
import torch.nn as nn
import torch.nn.functional as F

'''
baseline13：基于 baseline10 的两项优化：
① base=32 → 24（降低各层通道数）
② ViT mlp_ratio=4.0 → 2.0（缩小 MLP 隐层）
'''

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

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

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        if is_4d:
            return x.transpose(1, 2).reshape(B, C, H, W)
        return x

class attention_with_priori(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
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

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attention_with_priori(dim, num_heads=num_heads, dropout=dropout)
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

    def __init__(self, ch: int, reduction: int = 16, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.coord_att = CoordAtt(ch, reduction=reduction)
        self.norm = nn.LayerNorm(ch)
        self.block = SpatialViTBlock(ch, num_heads=num_heads, mlp_ratio=2.0, dropout=dropout)

    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
        x = self.coord_att(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        tokens = self.block(tokens, priori)  # 传给 blocks
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x


class SnakeCoordAttViTBlock(nn.Module):
    """
    蛇形卷积与 CoordAtt→ViT 的并联加残差块。
    结构：shortcut + snake(x) + coord_att_vit(x)
    """

    def __init__(self, ch: int, snake_kernel_size: int = 3, reduction: int = 16,
                 vit_num_heads: int = 4, vit_dropout: float = 0.1):
        super().__init__()
        self.snake = SnakeConvUnit(ch, ch, kernel_size=snake_kernel_size)
        self.coord_att_vit = CoordAttViTBlock(
            ch, reduction=reduction, num_heads=vit_num_heads, dropout=vit_dropout
        )
    #双模块concat后过一个1*1后和原图相加
    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
        return x + self.snake(x) + self.coord_att_vit(x, priori)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.block = SnakeCoordAttViTBlock(out_ch)
    def forward(self, x, priori):
        x = self.relu(self.bn(self.conv(x)))
        return self.block(x, priori)

class UNet3Layer(nn.Module):
    def __init__(self, in_channels: int, base: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.final_conv = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pri = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        e1 = self.enc1(x, pri)
        e2 = self.enc2(self.pool(e1), pri)
        e3 = self.enc3(self.pool(e2), pri)
        e4 = self.enc4(self.pool(e3), pri)
        # print("e3 shape:", e3.shape, "pri shape:", pri.shape)
        # print("e4 shape:", e4.shape, "pri shape:", pri.shape)
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
