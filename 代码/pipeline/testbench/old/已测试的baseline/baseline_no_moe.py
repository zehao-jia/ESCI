"""
高光谱海面溢油检测 — 1D / 2D / 3D 三分支融合网络

输入与常见 patch 管线一致: (B, 1, C, H, W)，C 为光谱维，H/W 为空间 patch。
- 1D 分支：空间全局池化后沿光谱维做 1D 卷积，刻画像元级光谱形状。
- 2D 分支：三层编码-解码 U-Net（3 次下采样），在 (H, W) 上建模多尺度空间上下文。
- 3D 分支：3D 卷积同时在光谱与空间上建模联合谱空特征。

三分支特征拼接后经融合层得到二分类 logits（可改 num_classes）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Spectral1DBranch(nn.Module):
    """沿光谱维度的 1D 卷积分支（输入经空间 GAP 后为每条 patch 一条光谱曲线）。"""

    def __init__(self, in_bands: int, hidden: int = 64, out_dim: int = 128):
        super().__init__()
        # (B, 1, C) -> 在长度 C 上卷积
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x = F.adaptive_avg_pool2d(x, 1).view(b, c)  # (B, C)
        x = x.unsqueeze(1)  # (B, 1, C)
        x = self.conv(x)
        x = self.pool(x).flatten(1)  # (B, out_dim)
        return x


class _DoubleConv2d(nn.Module):
    """UNet 常用双卷积块。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WaveletDownsample(nn.Module):
    """
    2D Haar 小波下采样，提取低频分量 (LL)。
    输入: (B, C, H, W)
    输出: (B, C, H//2, W//2) 的 LL 低频特征
    """

    def __init__(self):
        super().__init__()
        # Haar 小波的低通滤波器系数 (归一化)
        ll_filter = torch.tensor([[0.25, 0.25],
                                    [0.25, 0.25]], dtype=torch.float32)
        # 注册为不可学习的参数
        self.register_buffer('ll_filter', ll_filter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        # 确保尺寸是偶数
        if h % 2 != 0 or w % 2 != 0:
            # 填充到偶数尺寸
            pad_h = 1 if h % 2 != 0 else 0
            pad_w = 1 if w % 2 != 0 else 0
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            h, w = x.shape[2], x.shape[3]

        # 将滤波器扩展到输入通道数 (C, 1, 2, 2)
        weight = self.ll_filter.view(1, 1, 2, 2).expand(c, 1, 2, 2)

        # 使用 stride=2 的 depthwise 卷积实现小波下采样
        # groups=c 表示每个通道独立卷积
        ll = F.conv2d(x, weight, stride=2, groups=c, padding=0)

        return ll


class Spatial2DBranch(nn.Module):
    """
    2D 分支：3 层 U-Net（3 次小波下采样 + 对称上采样与跳跃连接）。
    输入以光谱为通道 (B, C, H, W)，要求 H、W 至少为 8，以便 2^3 倍下采样后尺寸有效。
    下采样使用 Haar 小波变换提取低频分量 (LL)。
    """

    def __init__(self, in_bands: int, base: int = 32, out_dim: int = 128):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        # 使用小波下采样替代 MaxPool
        self.pool = WaveletDownsample()

        self.enc1 = _DoubleConv2d(in_bands, c1)
        self.enc2 = _DoubleConv2d(c1, c2)
        self.enc3 = _DoubleConv2d(c2, c3)
        self.bottleneck = _DoubleConv2d(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv2d(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv2d(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv2d(c1 + c1, c1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c1, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))   # 小波下采样，取 LL
        e3 = self.enc3(self.pool(e2))   # 小波下采样，取 LL
        b = self.bottleneck(self.pool(e3))  # 小波下采样，取 LL

        u = self.up3(b)
        if u.shape[2:] != e3.shape[2:]:
            u = F.interpolate(u, size=e3.shape[2:], mode="bilinear", align_corners=False)
        u = self.dec3(torch.cat([u, e3], dim=1))

        u = self.up2(u)
        if u.shape[2:] != e2.shape[2:]:
            u = F.interpolate(u, size=e2.shape[2:], mode="bilinear", align_corners=False)
        u = self.dec2(torch.cat([u, e2], dim=1))

        u = self.up1(u)
        if u.shape[2:] != e1.shape[2:]:
            u = F.interpolate(u, size=e1.shape[2:], mode="bilinear", align_corners=False)
        u = self.dec1(torch.cat([u, e1], dim=1))

        u = self.gap(u).flatten(1)
        return self.fc(u)


class SpectralSpatial3DBranch(nn.Module):
    """3D 卷积分支：联合光谱 C 与空间 H、W（Conv3d 的 D-H-W 对应 C-H-W）。"""

    def __init__(self, in_bands: int, patch_h: int, patch_w: int, hidden: int = 24, out_dim: int = 128):
        super().__init__()
        # 输入 (B, 1, C, H, W)；部分小波 patch 上 C 或 H/W 较小，用小核与 padding
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden * 2, hidden * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(hidden * 2),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(hidden * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, H, W)
        x = self.conv3d(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TriBranchOilSpillNet(nn.Module):
    """
    1D + 2D + 3D 三分支溢油检测网络。

    Args:
        in_bands: 光谱维长度 C（如 PCA 后 30）
        patch_size: 空间边长（假定方形 patch，H=W）
        branch_dim: 各分支输出特征维度（拼接前）
        num_classes: 分类类别数，溢油二分类设为 2
        dropout: 融合层 dropout
    """

    def __init__(
        self,
        in_bands: int,
        patch_size: int,
        branch_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.in_bands = in_bands
        self.patch_size = patch_size

        self.branch_1d = Spectral1DBranch(in_bands, hidden=64, out_dim=branch_dim)
        self.branch_2d = Spatial2DBranch(in_bands, base=32, out_dim=branch_dim)
        self.branch_3d = SpectralSpatial3DBranch(
            in_bands, patch_size, patch_size, hidden=24, out_dim=branch_dim
        )

        fused = branch_dim * 3
        self.fuse = nn.Sequential(
            nn.Linear(fused, fused // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        if x.dim() != 5 or x.size(1) != 1:
            raise ValueError(f"期望输入 (B, 1, C, H, W)，得到 {tuple(x.shape)}")

        b1 = x.squeeze(1)  # (B, C, H, W)
        f1 = self.branch_1d(b1)
        f2 = self.branch_2d(b1)
        f3 = self.branch_3d(x)
        z = torch.cat([f1, f2, f3], dim=1)
        return self.fuse(z)


def build_tri_branch_net(
    sample_x: torch.Tensor,
    num_classes: int = 2,
    branch_dim: int = 128,
    dropout: float = 0.4,
) -> TriBranchOilSpillNet:
    """根据一个 batch 样本张量自动推断 C、patch 尺寸并构建网络。"""
    if sample_x.dim() != 5:
        raise ValueError("sample_x 应为 (B, 1, C, H, W)")
    _, _, c, h, w = sample_x.shape
    if h != w:
        raise ValueError(f"当前实现假定方形 patch，得到 H={h}, W={w}")
    return TriBranchOilSpillNet(
        in_bands=c,
        patch_size=h,
        branch_dim=branch_dim,
        num_classes=num_classes,
        dropout=dropout,
    )


if __name__ == "__main__":
    B, C, H, W = 4, 30, 16, 16
    x = torch.randn(B, 1, C, H, W)
    net = TriBranchOilSpillNet(in_bands=C, patch_size=H, num_classes=2)
    y = net(x)
    print("输入:", tuple(x.shape), "输出 logits:", tuple(y.shape))
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"参数量约 {n_params:.2f} M")
