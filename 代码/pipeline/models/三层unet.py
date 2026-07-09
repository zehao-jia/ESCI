import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def create_gaussian_weight_matrix(size=13, sigma=None):
    """
    创建二维高斯权重矩阵
    """
    if sigma is None:
        sigma = size / 6.0  # 默认sigma，使得边缘值约为中心的1%
    center = (size - 1) / 2.0
    x = np.arange(size) - center
    y = np.arange(size) - center
    X, Y = np.meshgrid(x, y)
    gaussian_matrix = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    gaussian_matrix = gaussian_matrix / gaussian_matrix.max()
    return gaussian_matrix


class FullAttentionResidual(nn.Module):
    def __init__(self, total_layers, hidden_dim):
        super(FullAttentionResidual, self).__init__()
        self.total_layers = total_layers
        self.hidden_dim = hidden_dim
        self.pseudo_queries = nn.Parameter(torch.zeros(total_layers, hidden_dim))
        self.layers = nn.ModuleList([self.create_layer() for _ in range(total_layers)])

    def create_layer(self):
        return nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        layer_outputs = []

        # 正确使用张量而不是调用
        for l in range(self.total_layers):
            output = self.layers[l](x)
            layer_outputs.append(output)
            attention_weights = self.compute_attention_weights(l, layer_outputs)
            x = sum(attention_weights[i] * layer_outputs[i] for i in range(l + 1))
        return x

    def compute_attention_weights(self, current_layer, layer_outputs):
        current_query = self.pseudo_queries[current_layer]
        weights = [torch.exp(current_query @ output) for output in layer_outputs[:current_layer + 1]]
        total_weight = sum(weights)
        return [w / total_weight for w in weights]


class MultiScaleConvModule(nn.Module):
    """
    多尺度卷积模块 (MetaFormer结构)：
    输入特征图 → LayerNorm → 多尺度卷积Token Mixer → 残差连接 → 
    LayerNorm → Channel Mixer (MLP) → 残差连接
    """

    def __init__(self, in_channels):
        super(MultiScaleConvModule, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.dilated_conv = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=3, groups=in_channels, dilation=3, bias=False),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=5, groups=in_channels, dilation=5, bias=False),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.output_conv = nn.Conv2d(in_channels * 3, in_channels, 1, bias=False)

        self.norm2 = nn.LayerNorm(in_channels)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 4, in_channels, 1, bias=False)
        )

    def forward(self, x):
        residual1 = x
        out = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = self.norm1(out)
        out = out.permute(0, 3, 1, 2)

        out = self.dilated_conv(out)
        out = self.relu(out)
        out1 = self.scale1(out)
        out2 = self.scale2(out)
        out3 = self.scale3(out)

        combined = torch.cat([out1, out2, out3], dim=1)
        out = self.output_conv(combined)

        out = out + residual1

        residual2 = out
        out = out.permute(0, 2, 3, 1)
        out = self.norm2(out)
        out = out.permute(0, 3, 1, 2)

        out = self.mlp(out)
        out = out + residual2

        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_adjust = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else None
        self.norm1 = nn.LayerNorm(out_channels)
        self.multi_scale = MultiScaleConvModule(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False)
        )

    def forward(self, x):
        if self.channel_adjust is not None:
            x = self.channel_adjust(x)

        residual_input = x
        residual1 = x
        out1 = x.permute(0, 2, 3, 1)
        out1 = self.norm1(out1)
        out1 = out1.permute(0, 3, 1, 2)
        out1 = self.multi_scale(out1)

        x = residual1 + out1
        residual2 = x
        out2 = x.permute(0, 2, 3, 1)
        out2 = self.norm2(out2)
        out2 = out2.permute(0, 3, 1, 2)
        out2 = self.mlp(out2)

        x = residual2 + out2
        return x


class SimpleSkipConnection(nn.Module):
    def __init__(self, in_channels):
        super(SimpleSkipConnection, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
        self.conv1d_1 = nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1), bias=False)
        self.conv1d_2 = nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0), bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, enc_feat, dec_feat):
        combined = torch.cat([enc_feat, dec_feat], dim=1)
        out = self.conv1x1(combined)
        out1 = self.conv1d_1(out)
        out2 = self.conv1d_2(out)
        out = out1 + out2
        out = self.relu(out)
        return out


class UNet_Deep(nn.Module):
    """
    3层UNet：3层编码器 + 3层解码器
    下采样使用SkipConnectionModule + 池化下采样
    跳跃连接改为简单拼接 + 1DCNN
    """

    def __init__(self, in_channels=25, out_channels=25, num_filters=64):
        super(UNet_Deep, self).__init__()

        # 编码器（3层）
        self.enc1 = DoubleConv(in_channels, num_filters)
        self.down1 = nn.MaxPool2d(2)  # 池化下采样
        
        self.enc2 = DoubleConv(num_filters, num_filters * 2)
        self.down2 = nn.MaxPool2d(2)  # 池化下采样
        
        self.enc3 = DoubleConv(num_filters * 2, num_filters * 4)
        self.down3 = nn.MaxPool2d(2)  # 池化下采样
        
        # 瓶颈层
        self.bottleneck = DoubleConv(num_filters * 4, num_filters * 8)
        
        # 解码器（3层）
        self.up3 = nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 2, stride=2)
        self.dec3 = DoubleConv(num_filters * 4, num_filters * 4)  # 输入通道数与跳跃连接输出一致
        
        self.up2 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 2, stride=2)
        self.dec2 = DoubleConv(num_filters * 2, num_filters * 2)  # 输入通道数与跳跃连接输出一致
        
        self.up1 = nn.ConvTranspose2d(num_filters * 2, num_filters, 2, stride=2)
        self.dec1 = DoubleConv(num_filters, num_filters)  # 输入通道数与跳跃连接输出一致
        
        # 输出层
        self.final_conv = nn.Conv2d(num_filters, out_channels, 1)
        
        # 跳跃连接模块（简单拼接 + 1DCNN）
        self.skip3 = SimpleSkipConnection(num_filters * 4)
        self.skip2 = SimpleSkipConnection(num_filters * 2)
        self.skip1 = SimpleSkipConnection(num_filters)

    def forward(self, x):
        # 编码器路径（3层）
        enc1 = self.enc1(x)
        down1 = self.down1(enc1)
        
        enc2 = self.enc2(down1)
        down2 = self.down2(enc2)
        
        enc3 = self.enc3(down2)
        down3 = self.down3(enc3)
        
        # 瓶颈层
        bottleneck = self.bottleneck(down3)
        
        # 解码器路径（3层）
        dec3 = self.up3(bottleneck)
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        # 使用简单跳跃连接融合编码器和解码器特征
        fused3 = self.skip3(enc3, dec3)
        dec3 = self.dec3(fused3)
        
        dec2 = self.up2(dec3)
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        # 使用简单跳跃连接融合编码器和解码器特征
        fused2 = self.skip2(enc2, dec2)
        dec2 = self.dec2(fused2)
        
        dec1 = self.up1(dec2)
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        # 使用简单跳跃连接融合编码器和解码器特征
        fused1 = self.skip1(enc1, dec1)
        dec1 = self.dec1(fused1)
        
        # 输出层
        out = self.final_conv(dec1)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out
