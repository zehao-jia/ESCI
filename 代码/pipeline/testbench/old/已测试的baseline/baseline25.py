import torch
import torch.nn as nn
import torch.nn.functional as F

"""
baseline25: based on baseline22, enhanced for sea surface oil spill detection
Task features:
  ① Binary classification (seawater vs oil spill)
  ② Severe label imbalance (seawater:oil spill ~ 99:1)
  ③ Non-uniform oil spill distribution, long-tail distribution

Improvements (5 items):
  1. DualPriorFusion: dual prior fusion gate
     - Spectral branch captures band discriminative info
     - Edge branch captures multi-scale structural boundaries
     - Learnable gate alpha(x) = sigmoid(Conv(spectral, edge)) dynamic fusion

  2. MultiScaleChannelGate: multi-scale channel gate (replaces SEBlock)
     - 3 parallel pooling paths (1x1, 3x3, 5x5) -> independent FC -> weighted sum -> Sigmoid

  3. FocalGatePlus: enhanced focal gate (replaces FocalGate)
     - 2-layer MLP gate: Linear -> ReLU -> Linear -> Sigmoid
     - Residual connection: output = v * gate + v * 0.1

  4. FeaturePyramidFusion: feature pyramid fusion
     - Decoder 3-level features (dec1/dec2/dec3) upsampled and fused
     - 1x1 conv fusion, enhances multi-scale oil slick detection

  5. base=28: from base=26 to 28, moderate capacity increase
"""


class LightEdgePrior(nn.Module):
    """Lightweight multi-scale edge prior using 3x3 and 7x7 Sobel operators."""
    def __init__(self):
        super().__init__()
        sobel_x_3 = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_3 = sobel_x_3.transpose(2, 3).contiguous()
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
        self.register_buffer("sobel_x_3", sobel_x_3)
        self.register_buffer("sobel_y_3", sobel_y_3)
        self.register_buffer("sobel_x_7", sobel_x_7)
        self.register_buffer("sobel_y_7", sobel_y_7)
        self.fusion = nn.Conv2d(2, 1, kernel_size=1, bias=False)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        gx3 = F.conv2d(gray, self.sobel_x_3, padding=1)
        gy3 = F.conv2d(gray, self.sobel_y_3, padding=1)
        edge3 = torch.sqrt(gx3 ** 2 + gy3 ** 2 + 1e-8)
        gx7 = F.conv2d(gray, self.sobel_x_7, padding=3)
        gy7 = F.conv2d(gray, self.sobel_y_7, padding=3)
        edge7 = torch.sqrt(gx7 ** 2 + gy7 ** 2 + 1e-8)
        return self.fusion(torch.cat([edge3, edge7], dim=1))


class DualPriorFusion(nn.Module):
    """Dual prior fusion gate: spectral + edge, learnable gate fusion."""
    def __init__(self, in_channels):
        super().__init__()
        self.spectral_branch = nn.Sequential(
            nn.Conv2d(in_channels, max(8, in_channels // 4), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(8, in_channels // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_channels // 4), 1, kernel_size=1, bias=True),
        )
        self.edge_branch = LightEdgePrior()
        self.gate_net = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        spec_prior = self.spectral_branch(x)
        edge_prior = self.edge_branch(x)
        alpha = self.gate_net(torch.cat([spec_prior, edge_prior], dim=1))
        return alpha * spec_prior + (1.0 - alpha) * edge_prior


class MultiScaleChannelGate(nn.Module):
    """Multi-scale channel gate: 3 pooling scales -> independent FC -> weighted sum -> Sigmoid."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        mid = max(8, channel // reduction)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Linear(channel, mid, bias=False), nn.ReLU(inplace=True), nn.Linear(mid, channel, bias=False))
        self.pool2 = nn.AdaptiveAvgPool2d(3)
        self.fc2 = nn.Sequential(nn.Linear(channel * 9, mid, bias=False), nn.ReLU(inplace=True), nn.Linear(mid, channel, bias=False))
        self.pool3 = nn.AdaptiveAvgPool2d(5)
        self.fc3 = nn.Sequential(nn.Linear(channel * 25, mid, bias=False), nn.ReLU(inplace=True), nn.Linear(mid, channel, bias=False))
        self.scale_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        y1 = self.fc1(self.pool1(x).view(b, c))
        y2 = self.fc2(self.pool2(x).view(b, -1))
        y3 = self.fc3(self.pool3(x).view(b, -1))
        w = torch.softmax(self.scale_weights, dim=0)
        gate = torch.sigmoid(w[0] * y1 + w[1] * y2 + w[2] * y3).view(b, c, 1, 1)
        return x * gate


class FocalGatePlus(nn.Module):
    """Enhanced focal gate: 2-layer MLP + residual connection."""
    def __init__(self, dim, hidden_ratio=2.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dim), nn.Sigmoid(),
        )

    def forward(self, v_modulated):
        gate = self.gate_mlp(v_modulated)
        return v_modulated * gate + v_modulated * 0.1


class Attention(nn.Module):
    """Multi-head attention with prior modulation + FocalGatePlus."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.focal_gate = FocalGatePlus(self.head_dim)

    def forward(self, x, priori):
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
        v = self.focal_gate(v * priori)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = self.proj((attn @ v).transpose(1, 2).reshape(B, N, -1))
        if is_4d:
            return x.transpose(1, 2).reshape(B, C, H, W)
        return x


class attention_with_priori(nn.Module):
    """Attention wrapper: projects 1-channel prior to dim then feeds to Attention."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.conv_priori = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
        )

    def forward(self, x, priori):
        if x.dim() == 4:
            if priori.shape[2:] != x.shape[2:]:
                priori = F.adaptive_avg_pool2d(priori, output_size=x.shape[2:])
        else:
            N = x.shape[1]
            H = W = int(N ** 0.5)
            if priori.shape[2:] != (H, W):
                priori = F.adaptive_avg_pool2d(priori, output_size=(H, W))
        return self.attn(x, self.conv_priori(priori))


class SnakeConvUnit(nn.Module):
    """Snake convolution: parallel 1xk + kx1 directional convs, 1x1 fusion."""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        pad = kernel_size // 2
        self.conv_h = nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, pad), bias=False)
        self.conv_v = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), padding=(pad, 0), bias=False)
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h_feat = self.conv_h(x)
        v_feat = self.conv_v(x)
        x = torch.cat([h_feat, v_feat], dim=1)
        return self.act(self.bn(self.fuse(x)))


class CoordAtt(nn.Module):
    """Coordinate Attention (CVPR 2021)."""
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid = max(8, in_ch // reduction)
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv_h = nn.Conv2d(mid, in_ch, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, in_ch, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.relu(y, inplace=True)
        x_h, x_w = y.split([h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))


class SpatialViTBlock(nn.Module):
    """Single ViT block (Pre-LN) for spatial tokens."""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attention_with_priori(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x, priori):
        x = x + self.attn(self.norm1(x), priori)
        return x + self.mlp(self.norm2(x))


class CoordAttViTBlock(nn.Module):
    """CoordAtt -> tokenize -> ViT -> detokenize."""
    def __init__(self, ch, reduction=16, num_heads=4, dropout=0.1):
        super().__init__()
        self.coord_att = CoordAtt(ch, reduction=reduction)
        self.norm = nn.LayerNorm(ch)
        self.block = SpatialViTBlock(ch, num_heads=num_heads, mlp_ratio=2.0, dropout=dropout)

    def forward(self, x, priori):
        x = self.coord_att(x)
        b, c, h, w = x.shape
        tokens = self.norm(x.flatten(2).transpose(1, 2))
        tokens = self.block(tokens, priori)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class SnakeCoordAttViTBlock(nn.Module):
    """Snake conv + CoordAtt-ViT parallel with residual + MultiScaleChannelGate."""
    def __init__(self, ch, snake_kernel_size=3, reduction=16, vit_num_heads=4, vit_dropout=0.1):
        super().__init__()
        self.snake = SnakeConvUnit(ch, ch, kernel_size=snake_kernel_size)
        self.channel_gate = MultiScaleChannelGate(ch, reduction=reduction)
        self.coord_att_vit = CoordAttViTBlock(ch, reduction=reduction, num_heads=vit_num_heads, dropout=vit_dropout)

    def forward(self, x, priori):
        snake_out = self.channel_gate(self.snake(x))
        return x + snake_out + self.coord_att_vit(x, priori)


class DoubleConv(nn.Module):
    """Conv3x3 -> BN -> ReLU -> SnakeCoordAttViTBlock."""
    def __init__(self, in_ch, out_ch, num_heads=4):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.block = SnakeCoordAttViTBlock(out_ch, vit_num_heads=num_heads)

    def forward(self, x, priori):
        return self.block(self.relu(self.bn(self.conv(x))), priori)


class FeaturePyramidFusion(nn.Module):
    """Feature pyramid fusion: upsample decoder level 2 and 3, concat with level 1, fuse via 1x1 conv."""
    def __init__(self, c1, c2, c3):
        super().__init__()
        self.proj2 = nn.Conv2d(c2, c1, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(c3, c1, kernel_size=1, bias=False)
        self.fuse = nn.Conv2d(c1 * 3, c1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, d1, d2, d3):
        h, w = d1.shape[2:]
        d2_up = F.interpolate(self.proj2(d2), size=(h, w), mode="bilinear", align_corners=False)
        d3_up = F.interpolate(self.proj3(d3), size=(h, w), mode="bilinear", align_corners=False)
        return self.relu(self.bn(self.fuse(torch.cat([d1, d2_up, d3_up], dim=1))))


class UNet3Layer(nn.Module):
    """3-layer U-Net + DualPriorFusion + FeaturePyramidFusion. base=28."""
    def __init__(self, in_channels, base=28):
        super().__init__()
        self.dual_prior = DualPriorFusion(in_channels)
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
        self.pyramid_fusion = FeaturePyramidFusion(c1, c2, c3)
        self.final_conv = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x):
        pri = self.dual_prior(x)
        e1 = self.enc1(x, pri)
        e2 = self.enc2(self.pool(e1), pri)
        e3 = self.enc3(self.pool(e2), pri)
        e4 = self.enc4(self.pool(e3), pri)
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1), pri)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), pri)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), pri)
        return self.final_conv(self.pyramid_fusion(d1, d2, d3))


class UNetClassifier(nn.Module):
    """Classifier wrapper: U-Net -> GAP -> FC."""
    def __init__(self, in_bands, num_classes=2):
        super().__init__()
        self.unet = UNet3Layer(in_channels=in_bands, base=28)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, num_classes)
        self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x):
        b1 = x.squeeze(1)
        seg_logits = self.unet(b1)
        logits = self.fc(self.gap(seg_logits).flatten(1))
        self.last_moe_aux_loss = x.new_zeros(())
        return logits


def build_tri_branch_net(sample_x, num_classes=2, branch_dim=128, dropout=0.4, **kwargs):
    """Build baseline25 model. Compatible with IP_train.py interface."""
    if sample_x.dim() != 5:
        raise ValueError("sample_x must be (B, 1, C, H, W)")
    _, _, c, h, w = sample_x.shape
    return UNetClassifier(in_bands=c, num_classes=num_classes)


if __name__ == "__main__":
    B, C, H, W = 4, 30, 16, 16
    x = torch.randn(B, 1, C, H, W)
    net = UNetClassifier(in_bands=C, num_classes=2)
    y = net(x)
    print("Input:", tuple(x.shape), "Output logits:", tuple(y.shape))
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f} M")