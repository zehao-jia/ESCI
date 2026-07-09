import torch
import torch.nn as nn
import torch.nn.functional as F

'''
baseline23：基于 baseline21 的改进，针对海面溢油检测任务优化
任务特点：
  ① 二分类（海水 vs 溢油）
  ② 标签严重不平衡（海水:溢油 ≈ 99:1）
  ③ 溢油分布不均匀，存在长尾分布

改进点：
  1. MultiScaleSobelPrior: 多尺度 Sobel 算子 (3×3, 5×5, 7×7) 提取不同尺度的边缘结构信息，
     适配不同大小的油膜区域（小油斑需细粒度边缘，大油膜需粗粒度边界）。
  2. DilatedSnakeConvUnit: 在蛇形卷积基础上引入空洞卷积变体，扩大感受野以捕获溢油上下文，
     改善长尾分布下稀疏油斑的检测。
  3. GatedSkipFusion: 在跳跃连接处引入门控特征融合，让解码器自适应地选择来自编码器的有用特征，
     抑制背景噪声传递，缓解类别不平衡。
'''


class MultiScaleSobelPrior(nn.Module):
    """
    多尺度 Sobel 先验。
    使用 3×3、5×5、7×7 三种尺度的 Sobel 算子并行提取边缘结构，
    用 1×1 卷积融合多尺度边缘响应。
    大油膜产生粗边缘（大尺度 Sobel 响应强），小油斑产生细边缘（小尺度 Sobel 响应强）。
    """
    def __init__(self):
        super().__init__()
        # 3x3 Sobel
        sobel_x_3 = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_3 = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
        # 5x5 Sobel (approximation)
        sobel_x_5 = torch.tensor([
            [-1, -2, 0, 2, 1],
            [-2, -3, 0, 3, 2],
            [-3, -5, 0, 5, 3],
            [-2, -3, 0, 3, 2],
            [-1, -2, 0, 2, 1],
        ], dtype=torch.float32).view(1, 1, 5, 5) / 4.0
        sobel_y_5 = sobel_x_5.transpose(2, 3).contiguous()
        # 7x7 Sobel (approximation)
        sobel_x_7 = torch.tensor([
            [-1, -2, -3, 0, 3, 2, 1],
            [-2, -3, -4, 0, 4, 3, 2],
            [-3, -5, -6, 0, 6, 5, 3],
            [-4, -6, -8, 0, 8, 6, 4],
            [-3, -5, -6, 0, 6, 5, 3],
            [-2, -3, -4, 0, 4, 3, 2],
            [-1, -2, -3, 0, 3, 2, 1],
        ], dtype=torch.float32).view(1, 1, 7, 7) / 8.0
        sobel_y_7 = sobel_x_7.transpose(2, 3).contiguous()

        self.register_buffer('sobel_x_3', sobel_x_3)
        self.register_buffer('sobel_y_3', sobel_y_3)
        self.register_buffer('sobel_x_5', sobel_x_5)
        self.register_buffer('sobel_y_5', sobel_y_5)
        self.register_buffer('sobel_x_7', sobel_x_7)
        self.register_buffer('sobel_y_7', sobel_y_7)

        self.fusion = nn.Conv2d(3, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.mean(dim=1, keepdim=True)

        # 3x3
        gx3 = F.conv2d(x, self.sobel_x_3, padding=1)
        gy3 = F.conv2d(x, self.sobel_y_3, padding=1)
        edge3 = torch.sqrt(gx3 ** 2 + gy3 ** 2 + 1e-8)

        # 5x5
        gx5 = F.conv2d(x, self.sobel_x_5, padding=2)
        gy5 = F.conv2d(x, self.sobel_y_5, padding=2)
        edge5 = torch.sqrt(gx5 ** 2 + gy5 ** 2 + 1e-8)

        # 7x7
        gx7 = F.conv2d(x, self.sobel_x_7, padding=3)
        gy7 = F.conv2d(x, self.sobel_y_7, padding=3)
        edge7 = torch.sqrt(gx7 ** 2 + gy7 ** 2 + 1e-8)

        multi_edge = torch.cat([edge3, edge5, edge7], dim=1)
        return self.fusion(multi_edge)


class DilatedSnakeConvUnit(nn.Module):
    """
    空洞蛇形卷积单元。
    在 SnakeConvUnit 的基础上引入空洞卷积，扩大感受野，
    在不增加参数量过多的前提下捕获多尺度上下文信息。
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 需为奇数")
        pad = kernel_size // 2 * dilation
        self.conv_h = nn.Conv2d(
            in_ch, out_ch, kernel_size=(1, kernel_size),
            padding=(0, pad), dilation=(1, dilation), bias=False
        )
        self.conv_v = nn.Conv2d(
            in_ch, out_ch, kernel_size=(kernel_size, 1),
            padding=(pad, 0), dilation=(dilation, 1), bias=False
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


class GatedSkipFusion(nn.Module):
    """
    门控跳跃连接融合。
    对编码器特征和解码器特征分别生成门控权重，自适应融合。
    帮助模型抑制背景噪声传递，突出溢油区域特征。
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.gate_enc = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.gate_dec = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        gate = self.sigmoid(self.gate_enc(enc_feat) + self.gate_dec(dec_feat))
        return enc_feat * gate + dec_feat * (1 - gate)


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
        tokens = self.block(tokens, priori)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x


class DilatedSnakeCoordAttViTBlock(nn.Module):
    """
    空洞蛇形卷积 + CoordAtt→ViT 并联加残差块。
    用空洞蛇形卷积替代普通蛇形卷积，扩大感受野。
    """
    def __init__(self, ch: int, snake_kernel_size: int = 3, dilation: int = 1, reduction: int = 16,
                 vit_num_heads: int = 4, vit_dropout: float = 0.1):
        super().__init__()
        self.snake = DilatedSnakeConvUnit(ch, ch, kernel_size=snake_kernel_size, dilation=dilation)
        self.coord_att_vit = CoordAttViTBlock(
            ch, reduction=reduction, num_heads=vit_num_heads, dropout=vit_dropout
        )

    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
        return x + self.snake(x) + self.coord_att_vit(x, priori)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, num_heads=4, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.block = DilatedSnakeCoordAttViTBlock(out_ch, vit_num_heads=num_heads, dilation=dilation)

    def forward(self, x, priori):
        x = self.relu(self.bn(self.conv(x)))
        return self.block(x, priori)


class UNet3Layer(nn.Module):
    def __init__(self, in_channels: int, base: int = 24):
        super().__init__()
        self.ms_sobel_prior = MultiScaleSobelPrior()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(2)
        # Encoder with progressive dilation
        self.enc1 = DoubleConv(in_channels, c1, num_heads=1, dilation=1)
        self.enc2 = DoubleConv(c1, c2, num_heads=2, dilation=1)
        self.enc3 = DoubleConv(c2, c3, num_heads=4, dilation=2)
        self.enc4 = DoubleConv(c3, c4, num_heads=8, dilation=2)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3, num_heads=4, dilation=2)
        self.gate3 = GatedSkipFusion(c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2, num_heads=2, dilation=1)
        self.gate2 = GatedSkipFusion(c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1, num_heads=1, dilation=1)
        self.gate1 = GatedSkipFusion(c1)
        self.final_conv = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pri = self.ms_sobel_prior(x)
        e1 = self.enc1(x, pri)
        e2 = self.enc2(self.pool(e1), pri)
        e3 = self.enc3(self.pool(e2), pri)
        e4 = self.enc4(self.pool(e3), pri)
        x = self.up3(e4)
        x = self.dec3(torch.cat([x, e3], dim=1), pri)
        x = self.gate3(e3, x)
        x = self.up2(x)
        x = self.dec2(torch.cat([x, e2], dim=1), pri)
        x = self.gate2(e2, x)
        x = self.up1(x)
        x = self.dec1(torch.cat([x, e1], dim=1), pri)
        x = self.gate1(e1, x)
        return self.final_conv(x)


class UNetClassifier(nn.Module):
    def __init__(self, in_bands: int, num_classes: int = 2):
        super().__init__()
        self.unet = UNet3Layer(in_channels=in_bands, base=24)
        self.gap = nn.AdaptiveMaxPool2d(1)
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
