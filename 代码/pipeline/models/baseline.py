import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DWT(nn.Module):
    """
    离散小波变换模块：将输入特征图分解为四个小波子带 (LL, LH, HL, HH)
    使用用户指定的滤波器：
    - 低通滤波器 L=[1/2, 1/2]
    - 高通滤波器 H=[-1/2, 1/2]
    """

    def __init__(self):
        super(DWT, self).__init__()
        # 用户指定的滤波器
        L = [1 / 2, 1 / 2]  # 低通滤波器
        H = [-1 / 2, 1 / 2]  # 高通滤波器

        # 构建四个卷积核 (LL, LH, HL, HH)
        # LL: L行卷积 + L列卷积
        # LH: L行卷积 + H列卷积
        # HL: H行卷积 + L列卷积
        # HH: H行卷积 + H列卷积
        ll_kernel = np.outer(L, L).reshape(1, 1, 2, 2)
        lh_kernel = np.outer(L, H).reshape(1, 1, 2, 2)
        hl_kernel = np.outer(H, L).reshape(1, 1, 2, 2)
        hh_kernel = np.outer(H, H).reshape(1, 1, 2, 2)

        # 堆叠四个卷积核
        kernel_array = np.stack([ll_kernel, lh_kernel, hl_kernel, hh_kernel], axis=0).squeeze()
        self.weight = nn.Parameter(torch.from_numpy(kernel_array).float().view(4, 1, 2, 2), requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        # 对每个通道单独应用DWT
        x = x.view(b * c, 1, h, w)
        # 步长为2的卷积，得到四个子带
        out = F.conv2d(x, self.weight, stride=2)
        # 重塑为 (b, c*4, h//2, w//2)
        out = out.view(b, c * 4, h // 2, w // 2)
        return out


class MultiScaleConvModule(nn.Module):
    """
    多尺度卷积模块 (MetaFormer结构)：
    输入特征图 → LayerNorm → 多尺度卷积Token Mixer → 残差连接 → 
    LayerNorm → Channel Mixer (MLP) → 残差连接
    """

    def __init__(self, in_channels):
        super(MultiScaleConvModule, self).__init__()
        
        # LayerNorm 1 (用于Token Mixer前)
        self.norm1 = nn.LayerNorm(in_channels)
        
        # Token Mixer: 多尺度卷积模块
        # 初始空洞卷积
        self.dilated_conv = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # 三个不同尺度的带状深度分离卷积（不同膨胀率）
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),  # 膨胀率1
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=3, groups=in_channels, dilation=3, bias=False),  # 膨胀率3
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=5, groups=in_channels, dilation=5, bias=False),  # 膨胀率5
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        # 输出卷积
        self.output_conv = nn.Conv2d(in_channels * 3, in_channels, 1, bias=False)
        
        # LayerNorm 2 (用于Channel Mixer前)
        self.norm2 = nn.LayerNorm(in_channels)
        
        # Channel Mixer: MLP结构
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, 1, bias=False),  # 扩展通道数
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 4, in_channels, 1, bias=False)  # 压缩回原始通道数
        )

    def forward(self, x):
        # 残差连接1的起点
        residual1 = x
        
        # LayerNorm 1
        out = x.permute(0, 2, 3, 1)  # (B, H, W, C) for LayerNorm
        out = self.norm1(out)
        out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Token Mixer: 多尺度卷积
        # 初始空洞卷积
        out = self.dilated_conv(out)
        out = self.relu(out)

        # 三个不同尺度的带状深度分离卷积并联
        out1 = self.scale1(out)
        out2 = self.scale2(out)
        out3 = self.scale3(out)

        # 拼接三个尺度的输出
        combined = torch.cat([out1, out2, out3], dim=1)

        # 输出卷积降维
        out = self.output_conv(combined)
        
        # 残差连接1
        out = out + residual1
        
        # 残差连接2的起点
        residual2 = out
        
        # LayerNorm 2
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C) for LayerNorm
        out = self.norm2(out)
        out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Channel Mixer: MLP
        out = self.mlp(out)
        
        # 残差连接2
        out = out + residual2

        return out


class DoubleConv(nn.Module):
    """
    新的DoubleConv模块：按照用户描述的架构设计
    - 输入 → 第一个残差加法节点（起点）
    - 路径1：LN + 多尺度CNN → 回到第一个残差加法节点
    - 路径2：LN + MLP → 第二个残差加法节点
    - top_k attention → 输出
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        # 保存通道数信息
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 如果输入输出通道不同，先调整通道数
        self.channel_adjust = nn.Conv2d(in_channels, out_channels, 1,
                                        bias=False) if in_channels != out_channels else None

        # 第一个残差路径：LN + 多尺度CNN（基于输出通道数）
        self.norm1 = nn.LayerNorm(out_channels)
        self.multi_scale = MultiScaleConvModule(out_channels)

        # 第二个残差路径：LN + MLP（基于输出通道数）
        self.norm2 = nn.LayerNorm(out_channels)
        # MLP：两个线性层，中间有激活函数
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 4, 1, bias=False),  # 扩展通道数
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False)  # 压缩回原始通道数
        )

        # top_k attention（基于输出通道数）
        self.topk_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, out_channels, 1, bias=False)
        )
        self.topk_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.topk_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 如果输入输出通道不同，先调整通道数
        if self.channel_adjust is not None:
            x = self.channel_adjust(x)

        # 保存调整后的输入用于最终的残差连接
        residual_input = x

        # 第一个残差连接路径：LN + 多尺度CNN
        # 第一个残差加法节点（起点）
        residual1 = x

        # LN + 多尺度CNN路径
        out1 = x.permute(0, 2, 3, 1)  # (B, H, W, C) for LayerNorm
        out1 = self.norm1(out1)
        out1 = out1.permute(0, 3, 1, 2)  # (B, C, H, W)
        out1 = self.multi_scale(out1)

        # 回到第一个残差加法节点
        x = residual1 + out1

        # 第二个残差连接路径：LN + MLP
        residual2 = x

        # LN + MLP路径
        out2 = x.permute(0, 2, 3, 1)  # (B, H, W, C) for LayerNorm
        out2 = self.norm2(out2)
        out2 = out2.permute(0, 3, 1, 2)  # (B, C, H, W)
        out2 = self.mlp(out2)

        # 回到第二个残差加法节点
        x = residual2 + out2

        # top_k attention
        # 简化的注意力机制（模拟top_k效果，选择重要特征）
        attention_weights = self.topk_attention(x)
        attention_weights = torch.sigmoid(attention_weights)  # 归一化到0-1
        x = x * attention_weights  # 应用注意力权重

        # 归一化和激活
        if x.shape[2] > 1 and x.shape[3] > 1:
            x = self.topk_norm(x)
        x = self.topk_relu(x)

        # 最终的残差连接
        x = x + residual_input

        return x


class AdaptiveDownsampleModule(nn.Module):
    """
    自适应下采样模块：
    - 大特征图 (>8x8)：使用完整的小波下采样
    - 中等特征图 (2x2到8x8)：使用步长为2的深度可分离卷积
    - 小特征图 (<2x2)：使用自适应平均池化
    """

    def __init__(self, in_channels):
        super(AdaptiveDownsampleModule, self).__init__()
        self.in_channels = in_channels
        self.d = in_channels  # 通道数

        # 离散小波变换
        self.dwt = DWT()

        # 1×1点卷积（通道压缩，从4D→2D）
        self.compression = nn.Conv2d(self.d * 4, self.d * 2, 1, bias=False)

        # 3×3深度卷积（特征编码）
        self.depth_conv1 = nn.Conv2d(self.d, self.d, 3, padding=1, groups=self.d, bias=False)
        self.depth_conv2 = nn.Conv2d(self.d, self.d, 3, padding=1, groups=self.d, bias=False)

        # 1×1点卷积（调整通道数，从D→2D）
        self.point_conv = nn.Conv2d(self.d, self.d * 2, 1, bias=False)

        # sigmoid激活（生成注意力图）
        self.sigmoid = nn.Sigmoid()

        # 输出层
        self.output_conv = nn.Conv2d(self.d * 2, self.d, 1, bias=False)
        self.norm = nn.InstanceNorm2d(self.d, affine=True)
        self.relu = nn.ReLU(inplace=True)

        # 中等特征图的下采样：步长为2的深度可分离卷积
        self.medium_downsample = nn.Sequential(
            nn.Conv2d(self.d, self.d, 2, stride=2, groups=self.d, bias=False),  # 深度卷积，步长2
            nn.Conv2d(self.d, self.d, 1, bias=False)  # 逐点卷积
        )

    def forward(self, x):
        b, c, h, w = x.shape
        d = self.d

        # 确定特征图大小阈值，区分大小特征图
        large_threshold = 8  # 特征图大于8x8时使用小波形式
        medium_threshold = 2  # 特征图2x2到8x8时使用其他方法

        if h > large_threshold and w > large_threshold:
            # 情况1：大特征图 (>8x8)，使用完整的小波下采样
            # 步骤1：小波分解，拆分频率子带
            dwt_out = self.dwt(x)  # (b, 4d, h//2, w//2)

            # 拆分四个子带
            ll = dwt_out[:, :d, :, :]  # 低频子带 (b, d, h//2, w//2)
            lh = dwt_out[:, d:2 * d, :, :]  # 水平高频子带 (b, d, h//2, w//2)
            hl = dwt_out[:, 2 * d:3 * d, :, :]  # 垂直高频子带 (b, d, h//2, w//2)

            # 步骤2：子带拼接与通道压缩
            combined = dwt_out  # (b, 4d, h//2, w//2)
            compressed = self.compression(combined)  # (b, 2d, h//2, w//2)

            # 步骤3：小波注意力机制
            sum1 = ll + lh  # (b, d, h//2, w//2)
            sum2 = ll + hl  # (b, d, h//2, w//2)
            encoded1 = self.depth_conv1(sum1)  # (b, d, h//2, w//2)
            encoded2 = self.depth_conv2(sum2)  # (b, d, h//2, w//2)
            encoded = encoded1 + encoded2  # (b, d, h//2, w//2)
            encoded = self.point_conv(encoded)  # (b, 2d, h//2, w//2)
            attention_map = self.sigmoid(encoded)  # (b, 2d, h//2, w//2)

            # 步骤4：特征加权与输出
            attended = compressed * attention_map  # (b, 2d, h//2, w//2)
            out = attended + compressed  # (b, 2d, h//2, w//2)
            out = self.output_conv(out)  # (b, d, h//2, w//2)

            # 归一化和激活
            out_h, out_w = out.shape[2], out.shape[3]
            if out_h > 1 and out_w > 1:
                out = self.norm(out)
            out = self.relu(out)

        elif h >= medium_threshold and w >= medium_threshold:
            # 情况2：中等特征图 (2x2到8x8)，使用步长为2的深度可分离卷积
            out = self.medium_downsample(x)
            out = self.relu(out)

        else:
            # 情况3：小特征图 (<2x2)，使用自适应平均池化
            out = F.adaptive_avg_pool2d(x, (max(1, h // 2), max(1, w // 2)))

        return out


class SkipConnectionModule(nn.Module):
    """
    跳跃连接模块：接收编码器特征图和解码器特征图，输出融合后的特征图
    结构：卷积 + LayerNorm + ReLU + 残差连接
    """

    def __init__(self, in_channels):
        super(SkipConnectionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False)
        self.norm = nn.LayerNorm(in_channels)  # 通道维度的LayerNorm，作用于最后一个维度
        self.relu = nn.ReLU(inplace=True)

    def forward(self, enc_feat, dec_feat):
        # 拼接编码器和解码器特征
        combined = torch.cat([enc_feat, dec_feat], dim=1)
        # 卷积降维
        out = self.conv(combined)
        # LayerNorm - 需要调整维度顺序，因为LayerNorm默认作用于最后一个维度
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
        # ReLU激活
        out = self.relu(out)
        # 残差连接：使用编码器特征作为残差
        out = out + enc_feat
        return out


class UNet_Deep(nn.Module):
    """
    5层深层UNet：5层编码器 + 5层解码器
    """

    def __init__(self, in_channels=25, out_channels=25, num_filters=64):
        super(UNet_Deep, self).__init__()

        # 编码器（5层）
        self.enc1 = DoubleConv(in_channels, num_filters)
        self.down1 = AdaptiveDownsampleModule(num_filters)
        self.enc2 = DoubleConv(num_filters, num_filters * 2)
        self.down2 = AdaptiveDownsampleModule(num_filters * 2)
        self.enc3 = DoubleConv(num_filters * 2, num_filters * 4)
        self.down3 = AdaptiveDownsampleModule(num_filters * 4)
        self.enc4 = DoubleConv(num_filters * 4, num_filters * 8)
        self.down4 = AdaptiveDownsampleModule(num_filters * 8)
        self.enc5 = DoubleConv(num_filters * 8, num_filters * 16)
        self.down5 = AdaptiveDownsampleModule(num_filters * 16)

        # 瓶颈层
        self.bottleneck = DoubleConv(num_filters * 16, num_filters * 32)

        # 解码器（5层）
        self.up5 = nn.ConvTranspose2d(num_filters * 32, num_filters * 16, 2, stride=2)
        self.dec5 = DoubleConv(num_filters * 16, num_filters * 16)  # 输入通道数改为num_filters*16，因为skip连接后只有一个特征图

        self.up4 = nn.ConvTranspose2d(num_filters * 16, num_filters * 8, 2, stride=2)
        self.dec4 = DoubleConv(num_filters * 8, num_filters * 8)

        self.up3 = nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 2, stride=2)
        self.dec3 = DoubleConv(num_filters * 4, num_filters * 4)

        self.up2 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 2, stride=2)
        self.dec2 = DoubleConv(num_filters * 2, num_filters * 2)

        self.up1 = nn.ConvTranspose2d(num_filters * 2, num_filters, 2, stride=2)
        self.dec1 = DoubleConv(num_filters, num_filters)

        # 输出层
        self.final_conv = nn.Conv2d(num_filters, out_channels, 1)

        # 跳跃连接模块
        self.skip5 = SkipConnectionModule(num_filters * 16)
        self.skip4 = SkipConnectionModule(num_filters * 8)
        self.skip3 = SkipConnectionModule(num_filters * 4)
        self.skip2 = SkipConnectionModule(num_filters * 2)
        self.skip1 = SkipConnectionModule(num_filters)

        # DoubleConv模块（替换Top-K Token Attention模块）
        self.topk_attention5 = DoubleConv(num_filters * 16, num_filters * 16)
        self.topk_attention4 = DoubleConv(num_filters * 8, num_filters * 8)
        self.topk_attention3 = DoubleConv(num_filters * 4, num_filters * 4)
        self.topk_attention2 = DoubleConv(num_filters * 2, num_filters * 2)
        self.topk_attention1 = DoubleConv(num_filters, num_filters)

    def forward(self, x):
        # 编码器路径
        enc1 = self.enc1(x)
        down1 = self.down1(enc1)
        enc2 = self.enc2(down1)
        down2 = self.down2(enc2)
        enc3 = self.enc3(down2)
        down3 = self.down3(enc3)
        enc4 = self.enc4(down3)
        down4 = self.down4(enc4)
        enc5 = self.enc5(down4)
        down5 = self.down5(enc5)

        # 瓶颈层
        bottleneck = self.bottleneck(down5)

        # 解码器路径（带新的跳跃连接机制）
        dec5 = self.up5(bottleneck)
        if dec5.shape[2:] != enc5.shape[2:]:
            dec5 = F.interpolate(dec5, size=enc5.shape[2:], mode='bilinear', align_corners=False)
        # 使用跳跃连接模块融合编码器和解码器特征
        fused5 = self.skip5(enc5, dec5)
        # 使用DoubleConv模块（替换Top-K Token Attention）
        fused5 = self.topk_attention5(fused5)
        dec5 = self.dec5(fused5)

        dec4 = self.up4(dec5)
        if dec4.shape[2:] != enc4.shape[2:]:
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        fused4 = self.skip4(enc4, dec4)
        # 使用DoubleConv模块（替换Top-K Token Attention）
        fused4 = self.topk_attention4(fused4)
        dec4 = self.dec4(fused4)

        dec3 = self.up3(dec4)
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        fused3 = self.skip3(enc3, dec3)
        # 使用DoubleConv模块（替换Top-K Token Attention）
        fused3 = self.topk_attention3(fused3)
        dec3 = self.dec3(fused3)

        dec2 = self.up2(dec3)
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        fused2 = self.skip2(enc2, dec2)
        # 使用DoubleConv模块（替换Top-K Token Attention）
        fused2 = self.topk_attention2(fused2)
        dec2 = self.dec2(fused2)

        dec1 = self.up1(dec2)
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        fused1 = self.skip1(enc1, dec1)
        # 使用DoubleConv模块（替换Top-K Token Attention）
        fused1 = self.topk_attention1(fused1)
        dec1 = self.dec1(fused1)

        # 输出层
        out = self.final_conv(dec1)

        # 如果输出尺寸与输入不匹配，进行插值
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out


def create_gaussian_weight_matrix(size=13, sigma=None):
    """
    创建二维高斯权重矩阵

    Args:
        size: 矩阵大小（默认13×13）
        sigma: 高斯函数的标准差，如果为None则使用size/6作为默认值

    Returns:
        gaussian_matrix: (size, size)的numpy数组，中心值最大，边缘值最小
    """
    if sigma is None:
        sigma = size / 6.0  # 默认sigma，使得边缘值约为中心的1%

    # 创建坐标网格
    center = (size - 1) / 2.0
    x = np.arange(size) - center
    y = np.arange(size) - center
    X, Y = np.meshgrid(x, y)

    # 计算二维高斯函数
    gaussian_matrix = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

    # 归一化，使得中心值为1
    gaussian_matrix = gaussian_matrix / gaussian_matrix.max()

    return gaussian_matrix


class UNet(nn.Module):
    """
    5层深层UNet
    在输入前应用高斯权重
    """

    def __init__(self, in_channels=25, out_channels=25, num_filters=64, patch_size=13):
        super(UNet, self).__init__()

        # 深层UNet（5层）
        self.deep_unet = UNet_Deep(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=num_filters
        )

        # 创建高斯权重矩阵（注册为buffer，不参与训练但会随模型保存）
        gaussian_weight = create_gaussian_weight_matrix(size=patch_size)
        # 转换为torch tensor并注册为buffer
        self.register_buffer('gaussian_weight', torch.FloatTensor(gaussian_weight))

        # 保存输出通道数用于兼容性
        self.out_channels = out_channels

        # 为了兼容性，保留stage_out属性
        class StageOutWrapper(nn.Module):
            def __init__(self, out_channels):
                super().__init__()
                self.out_channels = out_channels

            def forward(self, x):
                return x

        self.stage_out = StageOutWrapper(out_channels)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            out: 输出特征图 (B, out_channels, H, W)
        """
        # 应用高斯权重：对每个通道都乘以高斯权重矩阵
        # x形状: (B, C, H, W)
        # gaussian_weight形状: (H, W) 或 (patch_size, patch_size)
        # 使用广播机制：(B, C, H, W) * (1, 1, H, W) -> (B, C, H, W)
        gaussian_weight = self.gaussian_weight.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        x_weighted = x * gaussian_weight

        # 深层UNet特征提取
        out = self.deep_unet(x_weighted)

        return out
