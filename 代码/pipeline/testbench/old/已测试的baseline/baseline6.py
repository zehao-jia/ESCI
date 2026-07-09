"""
高光谱海面溢油检测 — baseline6（在 baseline5 上将空间 ViT 换为 RWKV 形式）

与 baseline5 相同：蛇形 U-Net 支路 + 空间级联支路并联，拼接融合。
差异：原 4 个 SpatialViTBlock 改为 4 个基于 baseline4 中 RWKV2DUnit 的空间块
（token 展成 (B, D, H, W) 做 RWKV-like 混合后再回到 token 序列）。

输入: (B, 1, C, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _double_conv_snake(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        SnakeConvUnit(in_ch, out_ch, kernel_size=3),
        SnakeConvUnit(out_ch, out_ch, kernel_size=3),
    )


class SimpleUNet3LayerSnake(nn.Module):
    """轻量 2D U-Net（蛇形双卷积块），与 baseline3 / baseline5 一致。"""

    def __init__(self, in_channels: int, out_channels: int, base: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(2)
        self.enc1 = _double_conv_snake(in_channels, c1)
        self.enc2 = _double_conv_snake(c1, c2)
        self.enc3 = _double_conv_snake(c2, c3)
        self.bottleneck = _double_conv_snake(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _double_conv_snake(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _double_conv_snake(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _double_conv_snake(c1 + c1, c1)
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


class VectorSparseMoE(nn.Module):
    """对 (B, D) 向量特征做 Top-K 稀疏混合专家。"""

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        expert_hidden: int | None = None,
        top_k: int = 2,
        load_balance_coef: float = 0.01,
        residual: bool = True,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts 至少为 1")
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.load_balance_coef = load_balance_coef
        self.residual = residual
        hidden = expert_hidden if expert_hidden is not None else max(dim // 2, 64)
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, dim),
            )
            for _ in range(num_experts)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, d = x.shape
        gate_logits = self.gate(x)
        probs = F.softmax(gate_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)
        topk_gate = F.softmax(topk_vals, dim=-1)

        expert_stack = torch.stack([expert(x) for expert in self.experts], dim=1)
        chosen = expert_stack.gather(1, topk_idx.unsqueeze(-1).expand(b, self.top_k, d))
        moe_out = (chosen * topk_gate.unsqueeze(-1)).sum(dim=1)
        y = x + moe_out if self.residual else moe_out

        hard = torch.zeros(b, self.num_experts, device=x.device, dtype=x.dtype)
        hard.scatter_(1, topk_idx, 1.0)
        f = hard.sum(dim=0) / (b * self.top_k + 1e-8)
        p_mean = probs.mean(dim=0)
        balance_loss = self.num_experts * (f.detach() * p_mean).sum()
        aux = self.load_balance_coef * balance_loss

        return y, aux


class SnakeUNet2DBranch(nn.Module):
    """蛇形 U-Net → GAP → 向量。"""

    def __init__(
        self,
        in_bands: int,
        base: int = 32,
        out_dim: int = 128,
    ):
        super().__init__()
        self.unet = SimpleUNet3LayerSnake(
            in_channels=in_bands,
            out_channels=in_bands,
            base=base,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_bands, out_dim)
        self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.unet(x)
        self.last_moe_aux_loss = x.new_zeros(())
        return self.fc(self.gap(feat_map).flatten(1))


class RWKV2DUnit(nn.Module):
    """
    轻量 RWKV-like 2D 模块（与 baseline4 一致）：
    (B,C,H,W) 展平为 token，经 receptance/key/value 与 token-shift，再还原为特征图。
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
        x = self.in_proj(x)
        x = x + self.dw_mix(x)
        b, c, h, w = x.shape

        tokens = x.flatten(2).transpose(1, 2)
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


class SpatialRWKVBlock(nn.Module):
    """对 (B, T, D) 空间 token 序列应用 RWKV2DUnit（T=H*W，假定方形 patch）。"""

    def __init__(self, dim: int, patch_size: int, shift_ratio: float = 0.5):
        super().__init__()
        self.patch_size = patch_size
        self.rwkv = RWKV2DUnit(dim, dim, shift_ratio=shift_ratio)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t, d = tokens.shape
        ps = self.patch_size
        if t != ps * ps:
            raise ValueError(f"SpatialRWKVBlock 期望 T={ps * ps}，得到 T={t}")
        x = tokens.transpose(1, 2).reshape(b, d, ps, ps)
        x = self.rwkv(x)
        return x.flatten(2).transpose(1, 2)


class SpatialRWKVCascade2DBranch(nn.Module):
    """4 个 RWKV 形式空间块级联（替换 baseline5 的 ViT 级联分支）。"""

    def __init__(
        self,
        in_bands: int,
        patch_size: int,
        base: int = 32,
        out_dim: int = 128,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_expert_hidden: int | None = None,
        moe_load_balance_coef: float = 0.01,
        moe_residual: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        embed_dim = max(base * 2, 64)
        num_tokens = patch_size * patch_size
        self.token_proj = nn.Linear(in_bands, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.blocks = nn.Sequential(
            SpatialRWKVBlock(embed_dim, patch_size, shift_ratio=0.5),
            SpatialRWKVBlock(embed_dim, patch_size, shift_ratio=0.5),
            SpatialRWKVBlock(embed_dim, patch_size, shift_ratio=0.5),
            SpatialRWKVBlock(embed_dim, patch_size, shift_ratio=0.5),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, out_dim)
        self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h * w != self.pos_embed.size(1):
            raise ValueError(
                f"SpatialRWKVCascade2DBranch 期望 token 数为 {self.pos_embed.size(1)}，得到 {h * w}。"
            )
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        tokens = self.token_proj(tokens) + self.pos_embed
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        feat = tokens.mean(dim=1)
        self.last_moe_aux_loss = x.new_zeros(())
        return self.fc(feat)


class ParallelUNetRWKV2DNet(nn.Module):
    """蛇形 U-Net 与 4 级联 RWKV 空间分支并联，拼接后融合分类。"""

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

        self.branch_snake_unet = SnakeUNet2DBranch(
            in_bands=in_bands,
            base=32,
            out_dim=branch_dim,
        )
        self.branch_rwkv = SpatialRWKVCascade2DBranch(
            in_bands=in_bands,
            patch_size=patch_size,
            base=32,
            out_dim=branch_dim,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_expert_hidden=moe_expert_hidden,
            moe_load_balance_coef=moe_load_balance_coef,
            moe_residual=moe_residual,
        )

        fused = branch_dim * 2
        self.moe: VectorSparseMoE | None
        if use_moe:
            self.moe = VectorSparseMoE(
                dim=fused,
                num_experts=moe_num_experts,
                expert_hidden=moe_expert_hidden,
                top_k=moe_top_k,
                load_balance_coef=moe_load_balance_coef,
                residual=moe_residual,
            )
        else:
            self.moe = None

        self.fuse = nn.Sequential(
            nn.Linear(fused, fused // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused // 2, num_classes),
        )
        self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5 or x.size(1) != 1:
            raise ValueError(f"期望输入 (B, 1, C, H, W)，得到 {tuple(x.shape)}")

        b1 = x.squeeze(1)
        f_u = self.branch_snake_unet(b1)
        f_r = self.branch_rwkv(b1)
        z = torch.cat([f_u, f_r], dim=1)

        aux_rwkv = getattr(self.branch_rwkv, "last_moe_aux_loss", z.new_zeros(()))
        aux_unet = getattr(self.branch_snake_unet, "last_moe_aux_loss", z.new_zeros(()))
        if self.moe is not None:
            z, aux = self.moe(z)
            self.last_moe_aux_loss = aux_rwkv + aux_unet + aux
        else:
            self.last_moe_aux_loss = aux_rwkv + aux_unet
        return self.fuse(z)


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
) -> ParallelUNetRWKV2DNet:
    """供 IP_train.py ``--model baseline6`` 调用。"""
    if sample_x.dim() != 5:
        raise ValueError("sample_x 应为 (B, 1, C, H, W)")
    _, _, c, h, w = sample_x.shape
    if h != w:
        raise ValueError(f"当前实现假定方形 patch，得到 H={h}, W={w}")
    return ParallelUNetRWKV2DNet(
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


def build_classifier_net(
    sample_x: torch.Tensor,
    device: torch.device,
    branch_dim: int = 128,
    dropout: float = 0.4,
    **moe_kwargs,
) -> nn.Module:
    return build_tri_branch_net(
        sample_x,
        num_classes=2,
        branch_dim=branch_dim,
        dropout=dropout,
        **moe_kwargs,
    ).to(device)


if __name__ == "__main__":
    B, C, H, W = 4, 30, 16, 16
    x = torch.randn(B, 1, C, H, W)
    net = ParallelUNetRWKV2DNet(in_bands=C, patch_size=H, num_classes=2, moe_top_k=2)
    y = net(x)
    print("输入:", tuple(x.shape), "输出 logits:", tuple(y.shape))
    print("MoE 辅助损失:", float(net.last_moe_aux_loss))
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"参数量约 {n_params:.2f} M")
