"""
方案 B — 可变形卷积 MoE：4 个可变形卷积专家 + 方向偏置初始化 + 空间自适应路由。

设计要点:
  - Expert 0: offset 偏置 → 水平拉伸 (0°)
  - Expert 1: offset 偏置 → 45° 对角拉伸
  - Expert 2: offset 偏置 → 垂直拉伸 (90°)
  - Expert 3: offset 偏置 → 135° 反对角拉伸
  - 每个 expert 内部: offset_conv 预测采样偏移 → DeformConv2d 在偏移位置采样
  - offset_conv 权重初始化为 ~0，偏置编码方向——训练初期行为接近方案 A
  - Router: 逐像素 1×1 卷积 → 4 通道 softmax
  - 加权融合 + 1×1 卷积混合 + residual
  - Load balancing 辅助 loss 防止路由坍塌

与方案 A 的关键区别:
  - 方案 A: 卷积核固定，仅权重可学习
  - 方案 B: 采样位置可学习（offset），方向可从初始化偏置进一步微调
  - 方案 B 需要 torchvision >= 0.4.0

基于: baseline_unet2layer (baseline2.py)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .old.PrioriVit import Multihead_self_attention

# 可选依赖
try:
    from torchvision.ops import DeformConv2d
    HAS_DEFORM = True
except ImportError:
    HAS_DEFORM = False


# ===================== 可变形卷积专家 =====================

class DeformableExpert(nn.Module):
    """
    带方向偏置初始化的可变形卷积专家。

    结构: offset_conv (预测偏移) → DeformConv2d (在偏移位置采样)
    offset_conv 的偏置编码方向信息，权重初始化趋于零，
    使训练初期近似固定方向卷积，随后可学习微调。
    """

    _DIRECTION_ANGLES_DEG = [0, 45, 90, 135]  # 4 个方向的角度

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 direction_idx: int = 0, offset_scale: float = 0.3):
        super().__init__()
        if not HAS_DEFORM:
            raise ImportError(
                "方案 B 需要 torchvision.ops.DeformConv2d。"
                "请安装 torchvision >= 0.4.0 或回退到方案 A。"
            )
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 需为奇数")

        self.kernel_size = kernel_size
        self.direction_idx = direction_idx

        # 偏移预测网络: 输入 → 2*K*K 偏移量
        self.offset_conv = nn.Conv2d(
            in_ch, 2 * kernel_size * kernel_size,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=True
        )

        # 可变形卷积
        self.deform_conv = DeformConv2d(
            in_ch, out_ch, kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2, bias=False
        )

        # 方向偏置初始化
        self._init_directional_offset_bias(offset_scale)

    def _init_directional_offset_bias(self, scale: float):
        """
        初始化 offset_conv 的偏置，使初始偏移编码指定方向。

        对 3×3 卷积核的 9 个采样位置 (gx, gy):
          offset_x = scale * cos(θ) * gx
          offset_y = scale * sin(θ) * gy

        例如: θ=0° → offset_y=0, offset_x 沿水平方向拉伸
        """
        angle_deg = self._DIRECTION_ANGLES_DEG[self.direction_idx]
        theta = math.radians(angle_deg)

        ks = self.kernel_size
        # 基核网格位置
        grid_y, grid_x = torch.meshgrid(
            torch.arange(-(ks // 2), ks // 2 + 1),
            torch.arange(-(ks // 2), ks // 2 + 1),
            indexing='ij',
        )
        grid_x = grid_x.float().flatten()  # (K*K,)
        grid_y = grid_y.float().flatten()  # (K*K,)

        # 18 维偏置: 前 9 个为 Δx，后 9 个为 Δy
        bias = torch.zeros(2 * ks * ks)
        bias[:ks * ks] = scale * math.cos(theta) * grid_x
        bias[ks * ks:] = scale * math.sin(theta) * grid_y

        with torch.no_grad():
            self.offset_conv.bias.data.copy_(bias)
            # 权重初始化为极小值，使初始行为由偏置主导
            nn.init.normal_(self.offset_conv.weight, mean=0.0, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)            # (B, 2*K*K, H, W)
        return self.deform_conv(x, offset)       # (B, out_ch, H, W)


# ===================== 可变形方向 MoE 模块（替代 SnakeConvUnit） =====================

class DeformableMoE(nn.Module):
    """
    可变形方向 Mixture of Experts，替代 SnakeConvUnit。

    路由: 1×1 conv → 4 通道 softmax → 空间自适应加权
    专家: 4 个 DeformableExpert (0°, 45°, 90°, 135°)
    融合: 加权和 → 1×1 conv → BN → ReLU
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 num_experts: int = 4, router_hidden_ratio: int = 4,
                 offset_scale: float = 0.3):
        super().__init__()
        self.num_experts = num_experts

        # 可变形专家
        self.experts = nn.ModuleList([
            DeformableExpert(in_ch, out_ch, kernel_size, direction_idx=i,
                             offset_scale=offset_scale)
            for i in range(num_experts)
        ])

        # 逐像素路由器
        router_hidden = max(4, in_ch // router_hidden_ratio)
        self.router = nn.Sequential(
            nn.Conv2d(in_ch, router_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(router_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(router_hidden, num_experts, kernel_size=1, bias=True),
        )

        # 融合层
        self.fuse = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        # 路由统计
        self.last_balance_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ---- 路由 ----
        router_logits = self.router(x)
        router_weights = router_logits.softmax(dim=1)     # (B, num_experts, H, W)

        # ---- 专家计算 ----
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        # ---- 加权融合 ----
        out = torch.zeros(B, self.fuse.in_channels, H, W,
                          device=x.device, dtype=x.dtype)
        for i, e_out in enumerate(expert_outputs):
            w = router_weights[:, i:i + 1, :, :]
            out = out + w * e_out

        # ---- Load balancing loss ----
        with torch.no_grad():
            avg_prob = router_weights.mean(dim=[0, 2, 3])
        target = 1.0 / self.num_experts
        self.last_balance_loss = torch.sum((avg_prob - target) ** 2)

        out = self.fuse(out)
        out = self.bn(out)
        return self.act(out)


# ===================== 基础组件（与 baseline 相同） =====================

class CoordAtt(nn.Module):
    """坐标注意力 (Coordinate Attention, CVPR 2021)"""

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
    """用于空间 token 序列的单个 ViT Block（Pre-LN）"""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        head_dim = dim // num_heads
        self.attn = Multihead_self_attention(
            heads=num_heads, head_dim=head_dim, dim=dim,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, prior=prior)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class CoordAttViTBlock(nn.Module):
    """CoordAtt → token化 → N个 ViT Block → 还原特征图"""

    def __init__(self, ch: int, reduction: int = 16, num_heads: int = 4,
                 mlp_ratio: float = 2.0, num_blocks: int = 1, dropout: float = 0.1):
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
        gray = x.mean(dim=1, keepdim=True)
        prior = gray.flatten(2).transpose(1, 2)
        prior = self.prior_proj(prior)
        for block in self.blocks:
            tokens = block(tokens, prior)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x


# ===================== 可变形 MoE + ViT 并联块 =====================

class DeformMoECoordAttViTBlock(nn.Module):
    """
    DeformableMoE + CoordAttViT 的并联加残差块。
    结构：shortcut + DeformMoE(x) + coord_att_vit(x)
    """

    def __init__(self, ch: int, num_experts: int = 4,
                 vit_num_heads: int = 4, vit_mlp_ratio: float = 2.0,
                 vit_num_blocks: int = 1, vit_dropout: float = 0.1,
                 offset_scale: float = 0.3):
        super().__init__()
        self.moe = DeformableMoE(ch, ch, num_experts=num_experts,
                                  offset_scale=offset_scale)
        self.coord_att_vit = CoordAttViTBlock(
            ch, num_heads=vit_num_heads, mlp_ratio=vit_mlp_ratio,
            num_blocks=vit_num_blocks, dropout=vit_dropout
        )
        self.conv_1d = nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moe_out = self.moe(x)
        vit_out = self.coord_att_vit(x)
        combined = torch.cat([moe_out, vit_out], dim=1)
        return x + self.conv_1d(combined)


def _double_conv(in_ch: int, out_ch: int, head_num: int,
                 mlp_ratio: float = 2.0, num_blocks: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        DeformMoECoordAttViTBlock(out_ch, vit_num_heads=head_num,
                                   vit_mlp_ratio=mlp_ratio,
                                   vit_num_blocks=num_blocks),
    )


# ===================== 2 层 UNet =====================

class UNet2Layer(nn.Module):
    """2 层 UNet + 可变形方向 MoE：适用于 9×9 输入"""

    def __init__(self, in_channels: int, base: int = 24):
        super().__init__()
        c1, c2 = base, base * 2
        self.pool = nn.MaxPool2d(2)

        self.enc1 = _double_conv(in_channels, c1, head_num=4)
        self.enc2 = _double_conv(c1, c2, head_num=8)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2,
                                       output_padding=1)
        self.dec1 = _double_conv(c1 + c1, c1, head_num=4)
        self.final_conv = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        x = self.up1(e2)
        x = self.dec1(torch.cat([x, e1], dim=1))
        return self.final_conv(x)


class UNetClassifier(nn.Module):
    def __init__(self, in_bands: int, num_classes: int = 2):
        super().__init__()
        self.unet = UNet2Layer(in_channels=in_bands, base=24)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, num_classes)
        self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = x.squeeze(1)
        seg_logits = self.unet(b1)
        logits = self.fc(self.gap(seg_logits).flatten(1))

        # 收集所有 DeformMoECoordAttViTBlock 中 MoE 的 balance loss
        balance_losses = []
        for module in self.unet.modules():
            if isinstance(module, DeformMoECoordAttViTBlock):
                balance_losses.append(module.moe.last_balance_loss)
        self.last_moe_aux_loss = (sum(balance_losses) / len(balance_losses)
                                   if balance_losses else x.new_zeros(()))
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
    if not HAS_DEFORM:
        print("⚠ 方案 B 需要 torchvision >= 0.4.0，当前环境未安装。")
        print("  安装: pip install torchvision")
        print("  回退方案: 使用 baseline_unet2layer_moe_dir (方案 A)")
        exit(1)

    B, C, H, W = 4, 30, 9, 9
    x = torch.randn(B, 1, C, H, W)
    net = UNetClassifier(in_bands=C, num_classes=2)
    y = net(x)
    print("方案 B — 可变形卷积 MoE")
    print("输入:", tuple(x.shape), "输出 logits:", tuple(y.shape))
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"参数量约 {n_params:.2f} M")
    print(f"Load balance loss: {net.last_moe_aux_loss.item():.4f}")

    # 验证方向偏置初始化
    for i, module in enumerate(net.unet.modules()):
        if isinstance(module, DeformableMoE):
            print(f"\n  DeformableMoE #{i}:")
            angles = ['0°(水平)', '45°(对角)', '90°(垂直)', '135°(反对角)']
            for j, expert in enumerate(module.experts):
                bias = expert.offset_conv.bias.data
                dx = bias[:9].numpy()
                dy = bias[9:].numpy()
                print(f"    Expert {j} ({angles[j]}):")
                print(f"      offset_x = {[f'{v:.2f}' for v in dx]}")
                print(f"      offset_y = {[f'{v:.2f}' for v in dy]}")
            break
