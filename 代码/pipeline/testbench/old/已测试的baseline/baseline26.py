import torch
import torch.nn as nn
import torch.nn.functional as F

'''
baseline26：基于 baseline22 的优化，针对海面溢油检测任务提升 AUC/Recall
任务特点：
  ① 二分类（海水 vs 溢油）
  ② 标签严重不平衡（海水:溢油 ≈ 99:1）
  ③ 溢油分布不均匀，存在长尾分布
  ④ 训练数据极度稀缺（TEST_RATIO=0.99，仅~1%用于训练）

改进点（5项，对标 hmosd 最优模型）：
  1. Spectral3DEncoder: 3D 卷积前端处理光谱-空间联合特征，
     将 30 波段作为深度维度，用 Conv3d 捕获跨波段相关性（hmosd MAM-SSFEN 核心思想）。
  2. SpectralReweight: 轻量 1D 卷积跨通道注意力，替代 SEBlock，
     在通道间学习相关性权重（对标 hmosd 光谱注意力）。
  3. MultiScaleClassHead: 多尺度特征聚合分类头，
     融合最深编码器特征(e4)+最浅解码器特征(d1)+分割输出(seg)，
     替代原 GAP(2)→FC(2,2) 的极度简化分类头。
  4. 熵正则化辅助损失: 通过 last_moe_aux_loss 注入，
     鼓励模型做更确定的预测，间接缓解类别不平衡。
  5. base=32 + DropPath: 适度扩容 + 随机深度正则化防过拟合。

保留 baseline22 已验证有效的组件: SnakeConv、CoordAtt、FocalGate、SpectralWeightedPrior。
不采用 baseline25 的 DualPriorFusion / MultiScaleChannelGate / FeaturePyramidFusion（已验证更差）。
'''


# ============================================================
# 基础组件
# ============================================================

class DropPath(nn.Module):
    """Stochastic Depth (DropPath) 正则化，参考 timm 实现"""
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        if self.scale_by_keep:
            x = x.div(keep_prob) * random_tensor
        else:
            x = x * random_tensor
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力（保留但不再用于主路径，供对比）"""
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid = max(8, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpectralWeightedPrior(nn.Module):
    """
    可学习的光谱加权先验。
    使用 1×1 卷积学习每个光谱波段的权重，生成保留光谱差异信息的先验图。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Conv2d(in_channels, max(8, in_channels // 4), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(8, in_channels // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_channels // 4), 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight_net(x)


class FocalGate(nn.Module):
    """
    聚焦门控：对注意力调制后的 Value 应用可学习的门控，
    突出硬样本（溢油区域）的梯度信号，缓解 99:1 的不平衡问题。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, v_modulated: torch.Tensor) -> torch.Tensor:
        gate = self.sigmoid(self.gate_proj(v_modulated))
        return v_modulated * gate


# ============================================================
# 新增模块 1: Spectral3DEncoder — 3D 卷积光谱-空间前端
# ============================================================

class Spectral3DEncoder(nn.Module):
    """
    3D 卷积前端：将光谱维度作为深度维度，Conv3d 联合处理 (D, H, W)。
    对标 hmosd 的 MAM-SSFEN 3D conv 设计。
    
    输入: (B, C=30, H=16, W=16) — squeezed from (B,1,30,16,16)
    输出: (B, out_channels=32, H=16, W=16) — 与 UNet 第一层通道数对齐
    
    设计: 两层 Conv3d(kernel=(5,3,3)) 覆盖 5 个光谱波段 + 3×3 空间邻域，
         保留空间分辨率不变，最后沿光谱维度均值池化。
    """
    def __init__(self, in_bands: int = 30, out_channels: int = 32):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv3d_2 = nn.Conv3d(16, out_channels, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) = (B, 30, 16, 16)
        x = x.unsqueeze(1)                              # (B, 1, 30, 16, 16)
        x = F.relu(self.bn1(self.conv3d_1(x)), inplace=True)  # (B, 16, 30, 16, 16)
        x = F.relu(self.bn2(self.conv3d_2(x)), inplace=True)  # (B, 32, 30, 16, 16)
        x = x.mean(dim=2)                               # (B, 32, 16, 16)
        return x


# ============================================================
# 新增模块 2: SpectralReweight — 轻量跨通道注意力（替代 SEBlock）
# ============================================================

class SpectralReweight(nn.Module):
    """
    轻量跨通道重标定：1D 卷积沿通道维度学习通道间相关性权重。
    替代 SEBlock，对标 hmosd 的光谱注意力思想，但适配 2D UNet 通道维度。
    
    输入: (B, C, H, W) — 任意通道数
    输出: (B, C, H, W) — 重标定后的特征
    """
    def __init__(self, num_channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 全局空间池化 → per-channel descriptor
        desc = x.mean(dim=[2, 3])               # (B, C)
        desc = desc.unsqueeze(1)                 # (B, 1, C)
        weights = self.conv1d(desc)              # (B, 1, C)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, 1, C, 1, 1)
        return x * weights.squeeze(1)            # (B, C, H, W)


# ============================================================
# 新增模块 3: MultiScaleClassHead — 多尺度特征聚合分类头
# ============================================================

class MultiScaleClassHead(nn.Module):
    """
    多尺度分类头：融合最深编码器(e4)、最浅解码器(d1)、分割输出(seg) 三类特征。
    替代 baseline22 的 GAP(2)→FC(2,2)，极大增强分类判别能力。
    
    输入:
      e4: (B, c4, H/8, W/8) — 最深编码器特征，含高层语义
      d1: (B, c1, H, W)     — 最浅解码器特征，含细节信息
      seg: (B, 2, H, W)     — UNet 分割输出 logits
    输出: (B, 2) 分类 logits
    """
    def __init__(self, c1: int, c2: int, c3: int, c4: int,
                 num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.e4_pool = nn.AdaptiveAvgPool2d(1)
        self.d1_pool = nn.AdaptiveAvgPool2d(1)
        self.seg_pool = nn.AdaptiveAvgPool2d(1)

        total_dim = c4 + c1 + 2  # e.g., 256 + 32 + 2 = 290 (base=32)

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, e4: torch.Tensor, d1: torch.Tensor,
                seg_logits: torch.Tensor) -> torch.Tensor:
        f_e4 = self.e4_pool(e4).flatten(1)          # (B, c4)
        f_d1 = self.d1_pool(d1).flatten(1)          # (B, c1)
        f_seg = self.seg_pool(seg_logits).flatten(1)  # (B, 2)
        fused = torch.cat([f_e4, f_d1, f_seg], dim=1) # (B, c4+c1+2)
        return self.classifier(fused)


# ============================================================
# 注意力模块（来自 baseline22，FocalGate 保留）
# ============================================================

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1,
                 use_focal_gate: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_focal_gate = use_focal_gate

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        if use_focal_gate:
            self.focal_gate = FocalGate(self.head_dim)

    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
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
        v = v * priori

        if self.use_focal_gate:
            v = self.focal_gate(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        if is_4d:
            return x.transpose(1, 2).reshape(B, C, H, W)
        return x


class attention_with_priori(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1,
                 use_focal_gate: bool = True):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout,
                              use_focal_gate=use_focal_gate)
        self.conv_priori = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if priori.shape[2:] != x.shape[2:]:
                priori = F.adaptive_avg_pool2d(priori, output_size=x.shape[2:])
        else:
            N = x.shape[1]
            H = W = int(N ** 0.5)
            if priori.shape[2:] != (H, W):
                priori = F.adaptive_avg_pool2d(priori, output_size=(H, W))
        priori = self.conv_priori(priori)
        return self.attn(x, priori)


# ============================================================
# 蛇形卷积 + 坐标注意力 + ViT 块（来自 baseline22，升级版）
# ============================================================

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


class CoordAtt(nn.Module):
    """
    坐标注意力 (Coordinate Attention, CVPR 2021)
    对 H 和 W 方向分别池化，编码位置信息后作为注意力权重。
    """
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
    """用于空间 token 序列的单个 ViT Block（Pre-LN）+ DropPath 正则化。"""
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1, use_focal_gate: bool = True,
                 drop_path: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attention_with_priori(dim, num_heads=num_heads, dropout=dropout,
                                          use_focal_gate=use_focal_gate)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), priori))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CoordAttViTBlock(nn.Module):
    """
    坐标注意力 → token化 → 1个 ViT Block → 还原特征图。
    输入形状: (B, C, H, W)，输出形状: (B, C, H, W)
    """
    def __init__(self, ch: int, reduction: int = 16, num_heads: int = 4,
                 dropout: float = 0.1, use_focal_gate: bool = True,
                 drop_path: float = 0.1):
        super().__init__()
        self.coord_att = CoordAtt(ch, reduction=reduction)
        self.norm = nn.LayerNorm(ch)
        self.block = SpatialViTBlock(ch, num_heads=num_heads, mlp_ratio=2.0,
                                     dropout=dropout, use_focal_gate=use_focal_gate,
                                     drop_path=drop_path)

    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
        x = self.coord_att(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        tokens = self.block(tokens, priori)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x


class SnakeCoordAttViTBlock(nn.Module):
    """
    蛇形卷积与 CoordAtt→ViT 的并联加残差块（升级版）。
    结构：shortcut + SpectralReweight(snake(x)) + coord_att_vit(x) + DropPath
    改进：SEBlock → SpectralReweight，添加 DropPath 正则化。
    """
    def __init__(self, ch: int, snake_kernel_size: int = 3, reduction: int = 16,
                 vit_num_heads: int = 4, vit_dropout: float = 0.1,
                 use_focal_gate: bool = True, drop_path: float = 0.1):
        super().__init__()
        self.snake = SnakeConvUnit(ch, ch, kernel_size=snake_kernel_size)
        self.spec_reweight = SpectralReweight(ch)       # 替代 SEBlock
        self.coord_att_vit = CoordAttViTBlock(
            ch, reduction=reduction, num_heads=vit_num_heads, dropout=vit_dropout,
            use_focal_gate=use_focal_gate, drop_path=drop_path
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, priori: torch.Tensor) -> torch.Tensor:
        snake_out = self.spec_reweight(self.snake(x))
        return x + self.drop_path(snake_out + self.coord_att_vit(x, priori))


# ============================================================
# UNet 组件
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, num_heads=4, use_focal_gate: bool = True,
                 drop_path: float = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.block = SnakeCoordAttViTBlock(out_ch, vit_num_heads=num_heads,
                                           use_focal_gate=use_focal_gate,
                                           drop_path=drop_path)

    def forward(self, x, priori):
        x = self.relu(self.bn(self.conv(x)))
        return self.block(x, priori)


# ============================================================
# 升级版 UNet3Layer: 3D 前端 + base=32 + SpectralReweight + DropPath
# ============================================================

class UNet3Layer(nn.Module):
    def __init__(self, in_channels: int, base: int = 32):
        super().__init__()
        # ── 新增：3D 卷积光谱前端 + 光谱加权先验 ──
        self.spectral_encoder = Spectral3DEncoder(in_bands=in_channels, out_channels=base)
        self.spectral_prior = SpectralWeightedPrior(in_channels)

        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(base, c1, num_heads=1, drop_path=0.0)        # 第一层不用 DropPath
        self.enc2 = DoubleConv(c1, c2, num_heads=2, drop_path=0.05)
        self.enc3 = DoubleConv(c2, c3, num_heads=4, drop_path=0.1)
        self.enc4 = DoubleConv(c3, c4, num_heads=8, drop_path=0.1)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3, num_heads=4, drop_path=0.1)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2, num_heads=2, drop_path=0.05)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1, num_heads=1, drop_path=0.0)    # 最后一层不用 DropPath

        self.final_conv = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        输入: (B, C, H, W) = (B, 30, 16, 16) — 原始 PCA 降维后的光谱 patch
        输出: (seg_logits, e4, d1)
          seg_logits: (B, 2, H, W)
          e4: (B, c4, H/8, W/8)  — 最深编码器特征
          d1: (B, c1, H, W)      — 最浅解码器特征
        """
        # 光谱先验：从原始 30 波段生成
        pri = self.spectral_prior(x)                   # (B, 1, H, W)
        # 3D 光谱-空间编码
        feat = self.spectral_encoder(x)                 # (B, base, H, W)

        # 编码器
        e1 = self.enc1(feat, pri)                       # (B, c1, H, W)
        e2 = self.enc2(self.pool(e1), pri)              # (B, c2, H/2, W/2)
        e3 = self.enc3(self.pool(e2), pri)              # (B, c3, H/4, W/4)
        e4 = self.enc4(self.pool(e3), pri)              # (B, c4, H/8, W/8)

        # 解码器
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1), pri)  # (B, c3, H/4, W/4)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), pri)  # (B, c2, H/2, W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), pri)  # (B, c1, H, W)

        seg_logits = self.final_conv(d1)                # (B, 2, H, W)
        return seg_logits, e4, d1


# ============================================================
# 升级版 UNetClassifier: MultiScaleClassHead + 熵辅助损失
# ============================================================

class UNetClassifier(nn.Module):
    def __init__(self, in_bands: int, num_classes: int = 2, base: int = 32):
        super().__init__()
        self.unet = UNet3Layer(in_channels=in_bands, base=base)

        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.class_head = MultiScaleClassHead(c1, c2, c3, c4,
                                               num_classes=num_classes, dropout=0.5)
        self.last_moe_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (B, 1, C, H, W) — 与 IP_train.py DataLoader 格式一致
        输出: (B, num_classes) 分类 logits
        """
        b1 = x.squeeze(1)                                  # (B, C, H, W)
        seg_logits, e4, d1 = self.unet(b1)                 # seg: (B,2,H,W), e4: (B,c4,H/8,W/8), d1: (B,c1,H,W)
        logits = self.class_head(e4, d1, seg_logits)        # (B, num_classes)

        # ── 熵正则化辅助损失：鼓励模型做更确定的预测 ──
        # 高熵(不确定) → 大损失 → 梯度推动模型降低不确定性
        # 间接帮助模型对少数类（溢油）做出更果断的判断
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        self.last_moe_aux_loss = 0.05 * entropy

        return logits


# ============================================================
# 模型构建入口（兼容 IP_train.py 接口）
# ============================================================

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
    return UNetClassifier(in_bands=c, num_classes=num_classes, base=32)


# ============================================================
# 本地测试
# ============================================================

if __name__ == "__main__":
    B, C, H, W = 4, 30, 16, 16
    x = torch.randn(B, 1, C, H, W)
    net = UNetClassifier(in_bands=C, num_classes=2)
    y = net(x)
    print("输入:", tuple(x.shape), "输出 logits:", tuple(y.shape))
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"参数量约 {n_params:.2f} M")
    print(f"辅助损失: {net.last_moe_aux_loss.item():.6f}")
