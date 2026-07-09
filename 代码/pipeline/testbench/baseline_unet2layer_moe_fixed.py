"""
方案 A — 固定方向卷积 MoE：4 个不同方向偏置的卷积专家 + 空间自适应路由。

设计要点:
  - Expert 0: 水平方向偏置 (0°)   — 初始化为水平 Sobel 核
  - Expert 1: 45° 对角偏置       — 初始化为 45° 对角核
  - Expert 2: 垂直方向偏置 (90°)  — 初始化为垂直 Sobel 核
  - Expert 3: 135° 反对角偏置    — 初始化为 135° 对角核
  - Router: 逐像素 1×1 卷积预测 4 通道软权重 → softmax 归一化
  - 加权融合后 1×1 卷积混合 + residual
  - Load balancing 辅助 loss 防止路由坍塌

背景: 替代 SnakeConvUnit（原仅 1×k 水平 + k×1 垂直，2 个固定方向）

基于: baseline_unet2layer (baseline2.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .old.PrioriVit import Multihead_self_attention


# ===================== 方向卷积专家 =====================

class DirectionalConvExpert(nn.Module):
    """
    带方向初始化偏置的 3×3 卷积专家。
    权重初始化为对应方向的边缘检测核 + 小扰动，使专家从预设方向起步。
    """

    # 4 个方向的 3×3 核模板
    _DIRECTIONAL_KERNELS = {
        0: torch.tensor([  # 水平 (0°): 增强水平连续性
            [-1., -1., -1.],
            [ 2.,  2.,  2.],
            [-1., -1., -1.],
        ]),
        1: torch.tensor([  # 45° 对角
            [ 2.,  1., -1.],
            [ 1.,  0., -1.],
            [-1., -1., -2.],
        ]),
        2: torch.tensor([  # 垂直 (90°)
            [-1.,  2., -1.],
            [-1.,  2., -1.],
            [-1.,  2., -1.],
        ]),
        3: torch.tensor([  # 135° 反对角
            [-1.,  1.,  2.],
            [-1.,  0.,  1.],
            [-2., -1., -1.],
        ]),
    }

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, direction_idx: int = 0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 需为奇数")
        assert direction_idx in self._DIRECTIONAL_KERNELS, f"direction_idx 需在 {list(self._DIRECTIONAL_KERNELS.keys())}"

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False)
        self._init_directional_bias(direction_idx)

    def _init_directional_bias(self, direction_idx: int):
        """用方向模板初始化权重，叠加小噪声让不同 in/out 通道组合略有差异。"""
        template = self._DIRECTIONAL_KERNELS[direction_idx]  # (3, 3)
        template = template / template.abs().sum()  # 归一化到单位量级

        with torch.no_grad():
            out_ch, in_ch, k, k = self.conv.weight.shape
            for oc in range(out_ch):
                for ic in range(in_ch):
                    noise = torch.randn(k, k) * 0.02
                    self.conv.weight[oc, ic] = template + noise.to(self.conv.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ===================== 方向感知 MoE 模块（替代 SnakeConvUnit） =====================

class DirectionalMoE(nn.Module):
    """
    方向感知 Mixture of Experts，替代 SnakeConvUnit。

    路由: 1×1 conv → 4 通道 softmax → 空间自适应加权
    专家: 4 个 DirectionalConvExpert (0°, 45°, 90°, 135°)
    融合: 加权和 → 1×1 conv → BN → ReLU
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 num_experts: int = 4, router_hidden_ratio: int = 4):
        super().__init__()
        self.num_experts = num_experts

        # 专家
        self.experts = nn.ModuleList([
            DirectionalConvExpert(in_ch, out_ch, kernel_size, direction_idx=i)
            for i in range(num_experts)
        ])

        # 逐像素路由器: 轻量 1×1 卷积
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

        # 路由统计（用于 load balancing loss）
        self.last_balance_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ---- 路由 ----
        router_logits = self.router(x)           # (B, num_experts, H, W)
        router_weights = router_logits.softmax(dim=1)  # 逐像素 softmax

        # ---- 专家计算 ----
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))     # each (B, out_ch, H, W)

        # ---- 加权融合 ----
        out = torch.zeros(B, self.fuse.in_channels, H, W,
                          device=x.device, dtype=x.dtype)
        for i, e_out in enumerate(expert_outputs):
            w = router_weights[:, i:i + 1, :, :]  # (B, 1, H, W)
            out = out + w * e_out

        # ---- Load balancing loss ----
        # 鼓励路由器均匀使用各专家，防止坍塌
        with torch.no_grad():
            avg_prob = router_weights.mean(dim=[0, 2, 3])  # (num_experts,)
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


# ===================== 方向 MoE + ViT 并联块（替代 SnakeCoordAttViTBlock） =====================

class DirMoECoordAttViTBlock(nn.Module):
    """
    DirectionalMoE + CoordAttViT 的并联加残差块。
    结构：shortcut + MoE_dir(x) + coord_att_vit(x)
    """

    def __init__(self, ch: int, num_experts: int = 4,
                 vit_num_heads: int = 4, vit_mlp_ratio: float = 2.0,
                 vit_num_blocks: int = 1, vit_dropout: float = 0.1):
        super().__init__()
        self.moe = DirectionalMoE(ch, ch, num_experts=num_experts)
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
        DirMoECoordAttViTBlock(out_ch, vit_num_heads=head_num,
                                vit_mlp_ratio=mlp_ratio,
                                vit_num_blocks=num_blocks),
    )


# ===================== 2 层 UNet =====================

class UNet2Layer(nn.Module):
    """2 层 UNet + 方向 MoE：适用于 9×9 输入"""

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

        # 收集所有 DirMoECoordAttViTBlock 中 MoE 的 balance loss
        balance_losses = []
        for module in self.unet.modules():
            if isinstance(module, DirMoECoordAttViTBlock):
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
    B, C, H, W = 4, 30, 9, 9
    x = torch.randn(B, 1, C, H, W)
    net = UNetClassifier(in_bands=C, num_classes=2)
    y = net(x)
    print("方案 A — 固定方向 MoE")
    print("输入:", tuple(x.shape), "输出 logits:", tuple(y.shape))
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"参数量约 {n_params:.2f} M")
    print(f"Load balance loss: {net.last_moe_aux_loss.item():.4f}")

    # 验证方向初始化
    for i, module in enumerate(net.unet.modules()):
        if isinstance(module, DirectionalMoE):
            print(f"\n  DirectionalMoE #{i}:")
            for j, expert in enumerate(module.experts):
                w = expert.conv.weight[0, 0].detach()
                angles = ['0°(水平)', '45°(对角)', '90°(垂直)', '135°(反对角)']
                print(f"    Expert {j} ({angles[j]}): mean={w.mean():.4f}, std={w.std():.4f}")
            break
