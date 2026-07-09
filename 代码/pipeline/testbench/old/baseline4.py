"""
高光谱海面溢油检测 — baseline4（仅 2D 分支）

相对 baseline3 的改动：
1) 去掉 1D 与 3D 分支，仅保留 2D U-Net 分支。
2) 在 U-Net 每层使用并联块：蛇形卷积路径 + RWKV-like 路径，再做融合。

输入: (B, 1, C, H, W)
输出: logits (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SnakeConvUnit(nn.Module):
    """蛇形卷积单元：并联 1xk 与 kx1 方向卷积，再 1x1 融合。"""

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


class RWKV2DUnit(nn.Module):
    """
    轻量 RWKV-like 2D 模块：
    - 先把 (H,W) 展平为 token 序列
    - 通过 receptance/key/value 门控进行 token 交互
    - 再映射回 2D 特征图
    """

    def __init__(self, in_ch: int, out_ch: int, shift_ratio: float = 0.5):
        super().__init__()
        if not (0.0 <= shift_ratio <= 1.0):
            raise ValueError("shift_ratio 需在 [0,1] 范围内")
        self.in_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.dw_mix = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False
        )
        self.shift_ratio = shift_ratio
        self.norm = nn.LayerNorm(out_ch)
        self.receptance = nn.Linear(out_ch, out_ch, bias=False)
        self.key = nn.Linear(out_ch, out_ch, bias=False)
        self.value = nn.Linear(out_ch, out_ch, bias=False)
        self.out = nn.Linear(out_ch, out_ch, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def _token_shift(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        对部分通道做 1-step token shift（RWKV 常用技巧）：
        前半通道使用上一 token 信息，后半通道保持当前 token。
        """
        b, t, c = tokens.shape
        shift_c = int(c * self.shift_ratio)
        if shift_c <= 0:
            return tokens
        if shift_c >= c:
            shift_c = c - 1

        shifted = tokens.new_zeros((b, t, shift_c))
        shifted[:, 1:, :] = tokens[:, :-1, :shift_c]
        return torch.cat([shifted, tokens[:, :, shift_c:]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.in_proj(x)
        x = x + self.dw_mix(x)
        b, c, h, w = x.shape
        t = h * w

        tokens = x.flatten(2).transpose(1, 2)  # (B, T, C)
        tokens = self.norm(tokens)
        tokens = self._token_shift(tokens)

        r = torch.sigmoid(self.receptance(tokens))
        k = self.key(tokens)
        v = self.value(tokens)
        rwkv = r * (k * v)
        tokens = tokens + self.out(rwkv)

        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        x = self.bn(x)
        return self.act(x)


class ParallelSnakeRWKVBlock(nn.Module):
    """蛇形卷积与 RWKV-like 并联，再 1x1 融合。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.snake = SnakeConvUnit(in_ch, out_ch, kernel_size=kernel_size)
        self.rwkv = RWKV2DUnit(in_ch, out_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.snake(x)
        r = self.rwkv(x)
        return self.fuse(torch.cat([s, r], dim=1))


def _double_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        ParallelSnakeRWKVBlock(in_ch, out_ch, kernel_size=3),
        ParallelSnakeRWKVBlock(out_ch, out_ch, kernel_size=3),
    )


class SimpleUNet3Layer(nn.Module):
    """三层 U-Net，每层使用并联 Snake+RWKV 块。"""

    def __init__(self, in_channels: int, out_channels: int, base: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(2)

        self.enc1 = _double_block(in_channels, c1)
        self.enc2 = _double_block(c1, c2)
        self.enc3 = _double_block(c2, c3)
        self.bottleneck = _double_block(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _double_block(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _double_block(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _double_block(c1 + c1, c1)
        self.final_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        x = self.up3(b)
        x = self.dec3(torch.cat([x, e3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, e1], dim=1))
        return self.final_conv(x)


class Spatial2DBranch(nn.Module):
    """仅 2D 分支：UNet -> GAP -> FC。"""

    def __init__(self, in_bands: int, patch_size: int, base: int = 32, out_dim: int = 128):
        super().__init__()
        self.unet = SimpleUNet3Layer(
            in_channels=in_bands,
            out_channels=in_bands,
            base=base,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_bands, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.unet(x)
        return self.fc(self.gap(feat_map).flatten(1))


class TriBranchOilSpillNet(nn.Module):
    """
    保持类名兼容旧训练脚本，但实际为单分支（2D）网络。
    """

    def __init__(
        self,
        in_bands: int,
        patch_size: int,
        branch_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.4,
        use_moe: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_expert_hidden: int | None = None,
        moe_load_balance_coef: float = 0.01,
        moe_residual: bool = True,
    ):
        super().__init__()
        self.in_bands = in_bands
        self.patch_size = patch_size
        self.use_moe = use_moe

        self.branch_2d = Spatial2DBranch(
            in_bands=in_bands,
            patch_size=patch_size,
            base=32,
            out_dim=branch_dim,
        )
        self.fuse = nn.Sequential(
            nn.Linear(branch_dim, branch_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(branch_dim // 2, num_classes),
        )
        self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5 or x.size(1) != 1:
            raise ValueError(f"期望输入 (B, 1, C, H, W)，得到 {tuple(x.shape)}")
        b1 = x.squeeze(1)  # (B, C, H, W)
        f2 = self.branch_2d(b1)
        self.last_moe_aux_loss = b1.new_zeros(())
        return self.fuse(f2)


def build_tri_branch_net(
    sample_x: torch.Tensor,
    num_classes: int = 2,
    branch_dim: int = 128,
    dropout: float = 0.4,
    use_moe: bool = False,
    moe_num_experts: int = 4,
    moe_top_k: int = 2,
    moe_expert_hidden: int | None = None,
    moe_load_balance_coef: float = 0.01,
    moe_residual: bool = True,
) -> TriBranchOilSpillNet:
    """根据样本张量自动推断 C 与 patch 尺寸并构建 baseline4。"""
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
        use_moe=use_moe,
        moe_num_experts=moe_num_experts,
        moe_top_k=moe_top_k,
        moe_expert_hidden=moe_expert_hidden,
        moe_load_balance_coef=moe_load_balance_coef,
        moe_residual=moe_residual,
    )


if __name__ == "__main__":
    B, C, H, W = 4, 30, 16, 16
    x = torch.randn(B, 1, C, H, W)
    net = TriBranchOilSpillNet(in_bands=C, patch_size=H, num_classes=2)
    y = net(x)
    print("输入:", tuple(x.shape), "输出 logits:", tuple(y.shape))
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"参数量约 {n_params:.2f} M")
