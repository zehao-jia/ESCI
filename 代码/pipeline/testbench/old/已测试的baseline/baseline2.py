# baseline2 实验变体：可与 baseline 并行迭代（如计划将 2D 换为 ViT、在 IP_train 使用 FocalLoss 等）。
"""
高光谱海面溢油检测 — 1D / 2D / 3D 三分支融合网络（testbench/baseline2）

输入与常见 patch 管线一致: (B, 1, C, H, W)，C 为光谱维，H/W 为空间 patch。
- 1D 分支：空间金字塔池化（SPP 风格）：对 (H,W) 做多尺度自适应平均池化与自适应最大池化并联
  （默认尺度 1×1、2×2、4×4），拼接后经各自 MLP 投影并以可学习系数融合，得到谱–空摘要向量。
- 2D 分支：轻量三层 U-Net（3 次 MaxPool 下采样 + 转置卷积上采样与跳跃连接），在 (H, W) 上建模多尺度空间上下文。
- 3D 分支：3D 卷积同时在光谱与空间上建模联合谱空特征。

三分支特征拼接后经融合层得到二分类 logits（可改 num_classes）。
默认不启用拼接后稀疏 MoE（如需可手动 use_moe=True 开启）；通过 top_k 控制每次前向激活的专家数；
开启时负载均衡项写入 last_moe_aux_loss，训练时需并入总损失。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SimpleUNet3Layer(nn.Module):
    """
    轻量 2D U-Net：编码端 3 次下采样，解码端对称上采样并与跳跃特征拼接。
    输入/输出空间尺寸一致，通道数由 in_channels / out_channels 指定。
    """

    def __init__(self, in_channels: int, out_channels: int, base: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(2)
        self.enc1 = _double_conv(in_channels, c1)
        self.enc2 = _double_conv(c1, c2)
        self.enc3 = _double_conv(c2, c3)
        self.bottleneck = _double_conv(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _double_conv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _double_conv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _double_conv(c1 + c1, c1)
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
    """
    对 (B, D) 向量特征做 Top-K 稀疏混合专家。

    - **容量**：`top_k` 越小，每次前向只汇合越少专家的输出，等价于更低的计算/表达容量，
      门控 softmax 会在专家之间竞争，更倾向把样本分给「更对口」的专家。
    - **负载均衡**：辅助损失鼓励各专家在 batch 内被使用得相对均匀，避免专家坍塌；
      系数由 `load_balance_coef` 控制，训练时加到主损失上（见 `TriBranchOilSpillNet.last_moe_aux_loss`）。
    """

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
        # x: (B, D)
        b, d = x.shape
        gate_logits = self.gate(x)
        probs = F.softmax(gate_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)
        topk_gate = F.softmax(topk_vals, dim=-1)

        expert_stack = torch.stack([expert(x) for expert in self.experts], dim=1)
        chosen = expert_stack.gather(
            1, topk_idx.unsqueeze(-1).expand(b, self.top_k, d)
        )
        moe_out = (chosen * topk_gate.unsqueeze(-1)).sum(dim=1)
        if self.residual:
            y = x + moe_out
        else:
            y = moe_out

        # 负载均衡（Switch 风格）：f 为各专家被选中的比例（detach），P 为门控 softmax 在 batch 上的均值（可导），
        # 使辅助损失对 gate 有梯度，同时避免仅用 detach 路由导致 aux 不参与反传。
        hard = torch.zeros(b, self.num_experts, device=x.device, dtype=x.dtype)
        hard.scatter_(1, topk_idx, 1.0)
        f = hard.sum(dim=0) / (b * self.top_k + 1e-8)
        p_mean = probs.mean(dim=0)
        balance_loss = self.num_experts * (f.detach() * p_mean).sum()
        aux = self.load_balance_coef * balance_loss

        return y, aux


class Spectral1DBranch(nn.Module):
    """
    1D 引导分支（改造）：
    两个并联金字塔池化分支（Avg/Max），经投影后通过可学习参数做逐元素加法，
    输出向量用于指导 2D 分支输出。
    """

    def __init__(
        self,
        in_bands: int,
        hidden: int = 64,
        out_dim: int = 128,
        pyramid_scales: tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        self.in_bands = in_bands
        self.out_dim = out_dim
        self.pyramid_scales = pyramid_scales
        pyramid_bins = sum(s * s for s in pyramid_scales)
        in_dim = in_bands * pyramid_bins

        self.avg_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )
        self.max_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )
        # 可学习逐元素融合系数：fused = avg_feat + lambda * max_feat
        self.mix_lambda = nn.Parameter(torch.ones(out_dim))

    def _pyramid_pool_flatten(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        pooled_feats = []
        for s in self.pyramid_scales:
            if mode == "avg":
                p = F.adaptive_avg_pool2d(x, output_size=(s, s))
            elif mode == "max":
                p = F.adaptive_max_pool2d(x, output_size=(s, s))
            else:
                raise ValueError(f"不支持的池化模式: {mode}")
            pooled_feats.append(p.flatten(1))
        return torch.cat(pooled_feats, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_feat = self.avg_proj(self._pyramid_pool_flatten(x, mode="avg"))
        max_feat = self.max_proj(self._pyramid_pool_flatten(x, mode="max"))
        return avg_feat + self.mix_lambda.unsqueeze(0) * max_feat


class SpatialViTBlock(nn.Module):
    """用于空间 token 序列的单个 ViT Block（Pre-LN）。"""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class Spatial2DBranch(nn.Module):
    """
    2D 分支（baseline2 变体）：
    将 (H,W) 展平为空间 token 序列，经过 4 个 ViT Block 级联后全局池化并投影为向量。
    patch_size 用于固定长度位置编码；若干 moe_* 参数仅保留接口兼容性。
    """

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
        embed_dim = max(base * 2, 64)
        num_tokens = patch_size * patch_size
        self.token_proj = nn.Linear(in_bands, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.blocks = nn.Sequential(
            SpatialViTBlock(embed_dim, num_heads=4, mlp_ratio=4.0, dropout=0.1),
            SpatialViTBlock(embed_dim, num_heads=4, mlp_ratio=4.0, dropout=0.1),
            SpatialViTBlock(embed_dim, num_heads=4, mlp_ratio=4.0, dropout=0.1),
            SpatialViTBlock(embed_dim, num_heads=4, mlp_ratio=4.0, dropout=0.1),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, out_dim)
        self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        if h * w != self.pos_embed.size(1):
            raise ValueError(
                f"Spatial2DBranch 期望 token 数为 {self.pos_embed.size(1)}，得到 {h * w}。"
            )
        # (B, C, H, W) -> (B, H*W, C)
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        tokens = self.token_proj(tokens) + self.pos_embed
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        feat = tokens.mean(dim=1)
        self.last_moe_aux_loss = x.new_zeros(())
        return self.fc(feat)


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
        use_moe: 是否在拼接后使用 MoE，默认 False
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

        self.branch_1d = Spectral1DBranch(in_bands, hidden=64, out_dim=branch_dim)
        self.branch_2d = Spatial2DBranch(
            in_bands,
            patch_size=patch_size,
            base=32,
            out_dim=branch_dim,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_expert_hidden=moe_expert_hidden,
            moe_load_balance_coef=moe_load_balance_coef,
            moe_residual=moe_residual,
        )
        self.branch_3d = SpectralSpatial3DBranch(
            in_bands, patch_size, patch_size, hidden=24, out_dim=branch_dim
        )

        fused = branch_dim * 3
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
        # 1D 分支输出用于指导 2D 分支（逐元素加法）
        f2 = f2 + f1
        f3 = self.branch_3d(x)
        z = torch.cat([f1, f2, f3], dim=1)
        aux_2d = getattr(self.branch_2d, "last_moe_aux_loss", z.new_zeros(()))
        if self.moe is not None:
            z, aux = self.moe(z)
            self.last_moe_aux_loss = aux_2d + aux
        else:
            self.last_moe_aux_loss = aux_2d
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
    """供 IP_train.py ``--model baseline2`` 调用，与 testbench/baseline 入口一致。"""
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
    net = TriBranchOilSpillNet(in_bands=C, patch_size=H, num_classes=2, moe_top_k=2)
    y = net(x)
    print("输入:", tuple(x.shape), "输出 logits:", tuple(y.shape))
    print("MoE 辅助损失:", float(net.last_moe_aux_loss))
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"参数量约 {n_params:.2f} M")
