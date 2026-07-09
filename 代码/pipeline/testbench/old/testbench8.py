"""
高光谱海面溢油检测 — baseline7（RWKV 与蛇形卷积在 U-Net 层内并联）

设计目标：
1) 不再使用 baseline6 的“UNet 分支 + RWKV 分支”并联结构；
2) 在 U-Net 的每个层级内，蛇形卷积与 RWKV 路径并行提取特征，再融合；
3) 在 U-Net 层级之间加入 token-shift（作用于下采样后的特征），
   让层间信息混合后再进入下一层编码。

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
        # 在h方向上进行卷积
        self.conv_h = nn.Conv2d(
            in_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, pad), bias=False
        )
        # 在v方向上进行卷积,这是空间方向
        self.conv_v = nn.Conv2d(
            in_ch, out_ch, kernel_size=(kernel_size, 1), padding=(pad, 0), bias=False
        )
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先进行两个方向上的条带卷积,再拼接融合经过一个1*1卷积,最后BN和激活
        h_feat = self.conv_h(x)
        v_feat = self.conv_v(x)
        x = torch.cat([h_feat, v_feat], dim=1)
        x = self.fuse(x)
        x = self.bn(x)
        return self.act(x)


class WKVFunction(torch.autograd.Function):
    """
    WKV 递归前向/反向计算（8 头独立）。
    公式：
        wkv[t] = (sum_{i=1}^{t-1} exp(-(t-1-i)*w + k[i]) * v[i] + exp(u + k[t]) * v[t])
               / (sum_{i=1}^{t-1} exp(-(t-1-i)*w + k[i]) + exp(u + k[t]))
    
    其中 w = time_decay, u = time_first。
    """
    @staticmethod
    def forward(ctx, w, u, k, v):
        B, T, H, C = k.shape  # (batch, seq_len, heads, head_dim)
        dtype = k.dtype
        w = w.float().contiguous()  # (H,)
        u = u.float().contiguous()              # (H,)
        k = k.float().contiguous()
        v = v.float().contiguous()
        
        # 输出初始化
        y = torch.zeros((B, T, H, C), dtype=dtype, device=k.device)
        
        # 为反向传播保存中间状态
        sa = torch.zeros((B, H, C), device=k.device)
        sb = torch.zeros((B, H, C), device=k.device)
        # 累积衰减的 a / b
        a = torch.zeros((B, H, C), device=k.device)
        b = torch.zeros((B, H, C), device=k.device)
        
        # 前向：沿序列递推
        for t in range(T):
            k_t = k[:, t]  # (B, H, C)
            v_t = v[:, t]
            
            # num = exp(u + k[t]) * v[t] + sum_i exp(-(t-1-i)*w + k[i]) * v[i]
            # den = exp(u + k[t]) + sum_i exp(-(t-1-i)*w + k[i])
            # 递推：a = a * exp(-w) + exp(k[t]) * v[t],  b = b * exp(-w) + exp(k[t])
            # wkv[t] = (exp(u) * a + (exp(u + k[t]) - exp(u) * exp(k[t])) * v[t]) / (exp(u) * b + (exp(u + k[t])))
            # 简化版本（等价于标准公式）：
            # num = exp(u + k[t]) * v[t] + a_decayed
            # den = exp(u + k[t]) + b_decayed
            # 其中 a_decayed = a * exp(-w), b_decayed = b * exp(-w)
            # 然后 a = a_decayed + exp(k[t]) * v[t], b = b_decayed + exp(k[t])
            
            # 实际使用 RWKV 官方简化实现：
            exp_k = torch.exp(k_t)                       # (B, H, C)
            exp_u_plus_k = torch.exp(u.view(1, H, 1) + k_t)            # (B, H, C)
            
            num = exp_u_plus_k * v_t + a                 # a 已经累积了衰减前的 sum
            den = exp_u_plus_k + b                       # b 同理
            
            y[:, t] = num / (den + 1e-8)
            
            # 更新 a, b：衰减 + 加入当前步
            decay = torch.exp(-w)                         # (H,) -> (1, H, 1) 广播
            a = a * decay.view(1, H, 1) + exp_k * v_t
            b = b * decay.view(1, H, 1) + exp_k
        
        ctx.save_for_backward(w, u, k, v, y)
        return y.to(dtype)
    @staticmethod
    def backward(ctx, gy):
        w, u, k, v, y = ctx.saved_tensors
        B, T, H, C = k.shape
        # 简化的反向传播实现
        # 实际生产代码需要完整的 autograd 实现
        # 这里使用 PyTorch 自动微分作为 fallback（效率较低但正确）
        with torch.enable_grad():
            w_f = w.detach().requires_grad_(True)
            u_f = u.detach().requires_grad_(True)
            k_f = k.detach().requires_grad_(True)
            v_f = v.detach().requires_grad_(True)
            
            # 重新执行前向（带梯度）
            y_f = torch.zeros_like(y)
            a = torch.zeros((B, H, C), device=k.device)
            b = torch.zeros((B, H, C), device=k.device)
            decay = torch.exp(-w_f)
            for t in range(T):
                k_t = k_f[:, t]
                v_t = v_f[:, t]
                exp_k = torch.exp(k_t)
                exp_u_plus_k = torch.exp(u_f.view(1, H, 1) + k_t)
                num = exp_u_plus_k * v_t + a
                den = exp_u_plus_k + b
                y_f[:, t] = num / (den + 1e-8)
                a = a * decay.view(1, H, 1) + exp_k * v_t
                b = b * decay.view(1, H, 1) + exp_k
            
            grad_w, grad_u, grad_k, grad_v = torch.autograd.grad(
                y_f, (w_f, u_f, k_f, v_f), gy
            )
        return grad_w, grad_u, grad_k, grad_v
class RWKV2DUnit(nn.Module):
    """
    完整 RWKV-v4 Block（Time Mixing + Channel Mixing），应用于 2D 空间 token 序列。
    
    结构：
        输入 (B, C, H, W)
          → in_proj (1x1 Conv): (B, out_ch, H, W)
          → flatten → tokens: (B, T, out_ch), T=H*W
          
          → [Time Mixing（8 头 WKV 递归）→ 残差] : (B, T, out_ch)
          → [Channel Mixing（扩展 4x 的 MLP）→ 残差] : (B, T, out_ch)
          
          → reshape → (B, out_ch, H, W)
          → BN + ReLU
    
    外部接口不变：__init__(in_ch, out_ch, shift_ratio=0.5)
    """
    def __init__(self, in_ch: int, out_ch: int, shift_ratio: float = 0.5, num_heads: int = 4):
        super().__init__()
        if not (0.0 <= shift_ratio <= 1.0):
            raise ValueError("shift_ratio 需在 [0,1] 范围内")
        self.shift_ratio = shift_ratio
        self.out_ch = out_ch
        self.num_heads = num_heads
        assert out_ch % num_heads == 0, f"out_ch({out_ch}) 必须能被 num_heads({num_heads}) 整除"
        self.head_dim = out_ch // num_heads
        
        # 输入投影
        self.in_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        
        # ─── Time Mixing 子层 ───
        # 可学习的 token-shift 混合系数 (out_ch,)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, out_ch))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, out_ch))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, out_ch))
        
        self.ln_tm = nn.LayerNorm(out_ch)
        
        # 线性投影到 8 头
        self.key = nn.Linear(out_ch, out_ch, bias=False)
        self.value = nn.Linear(out_ch, out_ch, bias=False)
        self.receptance = nn.Linear(out_ch, out_ch, bias=False)
        
        # WKV 递归参数：每头独立
        self.time_decay = nn.Parameter(torch.zeros(num_heads))       # w
        self.time_first = nn.Parameter(torch.zeros(num_heads))       # u
        
        # Time Mixing 输出投影
        self.tm_output = nn.Linear(out_ch, out_ch, bias=False)
        
        # ─── Channel Mixing 子层 ───
        self.time_mix_k_cm = nn.Parameter(torch.ones(1, 1, out_ch))
        self.time_mix_r_cm = nn.Parameter(torch.ones(1, 1, out_ch))
        
        self.ln_cm = nn.LayerNorm(out_ch)
        
        # 中间扩展 4 倍（标准 RWKV 做法）
        hidden_ratio = 4
        hidden_dim = out_ch * hidden_ratio
        self.cm_key = nn.Linear(out_ch, hidden_dim, bias=False)
        self.cm_receptance = nn.Linear(out_ch, out_ch, bias=False)
        self.cm_value = nn.Linear(hidden_dim, out_ch, bias=False)
        
        # ─── 输出层 ───
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def _token_shift(self, tokens: torch.Tensor, ratio: float) -> torch.Tensor:
        b, t, c = tokens.shape
        shift_c = max(1, int(c * ratio))
        shifted = tokens.new_zeros((b, t, shift_c))
        shifted[:, 1:, :] = tokens[:, :-1, :shift_c]
        return torch.cat([shifted, tokens[:, :, shift_c:]], dim=-1)
    def _time_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Time Mixing 子层：
        - token-shift 混合（带可学习 time_mix 系数）
        - 8 头线性投影
        - WKV 递归（每头独立）
        - 门控输出 sigmoid(r) * wkv
        """
        B, T, C = x.shape
        
        # token-shift 混合
        x_shifted = self._token_shift(x, self.shift_ratio)
        xk = x * self.time_mix_k + x_shifted * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_shifted * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_shifted * (1 - self.time_mix_r)
        
        # LayerNorm
        xk = self.ln_tm(xk)
        xv = self.ln_tm(xv)
        xr = self.ln_tm(xr)
        
        # 线性投影 → 8 头
        k = self.key(xk).view(B, T, self.num_heads, self.head_dim)  # (B, T, H, C//H)
        v = self.value(xv).view(B, T, self.num_heads, self.head_dim)
        r = self.receptance(xr).view(B, T, self.num_heads, self.head_dim)
        
        # WKV 递归
        wkv = WKVFunction.apply(self.time_decay, self.time_first, k, v)  # (B, T, H, C//H)
        
        # 门控
        r_gate = torch.sigmoid(r)
        out = r_gate * wkv  # (B, T, H, C//H)
        
        # 合并多头
        out = out.reshape(B, T, C)
        
        # 输出投影 + 残差
        return x + self.tm_output(out)
    def _channel_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Channel Mixing 子层：
        - token-shift 混合（带可学习 time_mix 系数）
        - relu(k) @ W_k (扩展 4x)
        - sigmoid(r) @ W_r (门控)
        - 输出投影 + 残差
        """
        B, T, C = x.shape
        
        # token-shift 混合
        x_shifted = self._token_shift(x, self.shift_ratio)
        xk = x * self.time_mix_k_cm + x_shifted * (1 - self.time_mix_k_cm)
        xr = x * self.time_mix_r_cm + x_shifted * (1 - self.time_mix_r_cm)
        
        # LayerNorm
        xk = self.ln_cm(xk)
        xr = self.ln_cm(xr)
        
        # 通道混合
        k = F.relu(self.cm_key(xk))       # (B, T, 4*C)
        r = torch.sigmoid(self.cm_receptance(xr))  # (B, T, C)
        v = self.cm_value(k)              # (B, T, C)
        
        return x + r * v
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        B, C, H, W = x.shape
        T = H * W
        
        # 展平为 token 序列
        tokens = x.flatten(2).transpose(1, 2)  # (B, T, C)
        
        # Time Mixing → 残差
        tokens = self._time_mixing(tokens)
        
        # Channel Mixing → 残差
        tokens = self._channel_mixing(tokens)
        
        # 还原为 2D 特征图
        x = tokens.transpose(1, 2).reshape(B, C, H, W)
        
        x = self.bn(x)
        return self.act(x)


class ParallelSnakeRWKVBlock(nn.Module):
    """
    单层并联块：蛇形卷积路径 + RWKV 路径并行，拼接后融合。
    该融合输出可直接作为 skip feature。
    这里的维度数不进行调整,升维应在前面做
    """

    def __init__(self, in_ch, ch, rwkv_shift_ratio: float = 0.5):
        super().__init__()
        self.snake = nn.Sequential(
            SnakeConvUnit(ch, ch, kernel_size=3),
            SnakeConvUnit(ch, ch, kernel_size=3),
        )
        self.rwkv = nn.Sequential(
            RWKV2DUnit(ch, ch, shift_ratio=rwkv_shift_ratio),
            RWKV2DUnit(ch, ch, shift_ratio=rwkv_shift_ratio),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.dimension_elevation = nn.Sequential(
            nn.Conv2d(in_ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dimension_elevation(x)  # 升维到 snake 和 rwkv 的输入维度
        x_s = self.snake(x)
        x_r = self.rwkv(x)
        return self.merge(torch.cat([x_s, x_r], dim=1))


class HybridUNet3LayerSnakeRWKV(nn.Module):
    """
    3 层 U-Net：
    - 每层内部：Snake 与 RWKV 并联后融合；
    - 层间：下采样后先做 token-shift 再进入下一编码层。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base: int = 32,
        rwkv_shift_ratio: float = 0.5,
        inter_level_shift_ratio: float = 0.25,
    ):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ParallelSnakeRWKVBlock(in_channels, c1, rwkv_shift_ratio)
        # self.shift1 = InterLevelTokenShift2D(inter_level_shift_ratio)         #这个shift是将特征图变换为Token后又进行rwkv,要删去
        self.enc2 = ParallelSnakeRWKVBlock(c1, c2, rwkv_shift_ratio)
        # self.shift2 = InterLevelTokenShift2D(inter_level_shift_ratio)
        self.enc3 = ParallelSnakeRWKVBlock(c2, c3, rwkv_shift_ratio)
        # self.shift3 = InterLevelTokenShift2D(inter_level_shift_ratio)
        self.bottleneck = ParallelSnakeRWKVBlock(c3, c4, rwkv_shift_ratio)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ParallelSnakeRWKVBlock(c3 + c3, c3, rwkv_shift_ratio)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ParallelSnakeRWKVBlock(c2 + c2, c2, rwkv_shift_ratio)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ParallelSnakeRWKVBlock(c1 + c1, c1, rwkv_shift_ratio)
        self.final_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        b = self.bottleneck(p3)
        x = self.up3(b)
        x = self.dec3(torch.cat([x, e3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, e1], dim=1))
        return self.final_conv(x)



class SnakeRWKVUNet2DBranch(nn.Module):
    """baseline7 主分支：层内并联 Snake+RWKV 的 U-Net -> 特征图。"""

    def __init__(
        self,
        in_bands: int,
        base: int = 32,
        rwkv_shift_ratio: float = 0.5,
        inter_level_shift_ratio: float = 0.25,
    ):
        super().__init__()
        self.unet = HybridUNet3LayerSnakeRWKV(
            in_channels=in_bands,
            out_channels=in_bands,
            base=base,
            rwkv_shift_ratio=rwkv_shift_ratio,
            inter_level_shift_ratio=inter_level_shift_ratio,
        )
        # self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.unet(x)
        # self.last_moe_aux_loss = x.new_zeros(())
        return feat_map


class ParallelSnakeRWKVUNetNet(nn.Module):
    """
    baseline7 分类网络：
    输入先经层内并联 Snake+RWKV 的 U-Net 主干得到特征图，
    直接从中心位置提取特征并接分类头（不使用池化）。
    """

    def __init__(
        self,
        in_bands: int,
        patch_size: int,
        branch_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.4,
        rwkv_shift_ratio: float = 0.5,
        inter_level_shift_ratio: float = 0.25,
    ):
        super().__init__()
        self.in_bands = in_bands
        self.patch_size = patch_size
        # self.use_moe = use_moe

        self.branch_hybrid_unet = SnakeRWKVUNet2DBranch(
            in_bands=in_bands,
            base=32,
            rwkv_shift_ratio=rwkv_shift_ratio,
            inter_level_shift_ratio=inter_level_shift_ratio,
        )
        hidden_dim = max(in_bands // 2, 32)
        self.fuse = nn.Sequential(
            nn.Linear(in_bands, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        # self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5 or x.size(1) != 1:
            raise ValueError(f"期望输入 (B, 1, C, H, W)，得到 {tuple(x.shape)}")
        b1 = x.squeeze(1)
        feat_map = self.branch_hybrid_unet(b1)  # (B, C, H, W)
        _, _, h, w = feat_map.shape
        z = feat_map[:, :, h // 2, w // 2]  # 直接取中心位置特征做分类，不做池化

        # aux_branch = getattr(self.branch_hybrid_unet, "last_moe_aux_loss", z.new_zeros(()))
        # if self.moe is not None:
        #     z, aux = self.moe(z)
        #     self.last_moe_aux_loss = aux_branch + aux
        # else:
        #     self.last_moe_aux_loss = aux_branch
        return self.fuse(z)


def build_tri_branch_net(
    sample_x: torch.Tensor,
    num_classes: int = 2,
    branch_dim: int = 128,
    dropout: float = 0.4,

    rwkv_shift_ratio: float = 0.5,
    inter_level_shift_ratio: float = 0.25,
) -> ParallelSnakeRWKVUNetNet:
    """供 IP_train.py 调用的兼容入口（保留历史函数名）。"""
    if sample_x.dim() != 5:
        raise ValueError("sample_x 应为 (B, 1, C, H, W)")
    _, _, c, h, w = sample_x.shape
    if h != w:
        raise ValueError(f"当前实现假定方形 patch，得到 H={h}, W={w}")
    return ParallelSnakeRWKVUNetNet(
        in_bands=c,
        patch_size=h,
        branch_dim=branch_dim,
        num_classes=num_classes,
        dropout=dropout,
        rwkv_shift_ratio=rwkv_shift_ratio,
        inter_level_shift_ratio=inter_level_shift_ratio,
    )


def build_classifier_net(
    sample_x: torch.Tensor,
    device: torch.device,
    branch_dim: int = 128,
    dropout: float = 0.4,
    **kwargs,
) -> nn.Module:
    return build_tri_branch_net(
        sample_x,
        num_classes=2,
        branch_dim=branch_dim,
        dropout=dropout,
        **kwargs,
    ).to(device)


if __name__ == "__main__":
    b, c, h, w = 4, 30, 16, 16
    x = torch.randn(b, 1, c, h, w)
    net = ParallelSnakeRWKVUNetNet(in_bands=c, patch_size=h, num_classes=2)
    y = net(x)
    print("输入:", tuple(x.shape), "输出 logits:", tuple(y.shape))
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"参数量约 {n_params:.2f} M")
