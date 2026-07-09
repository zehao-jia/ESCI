import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic
from scipy.linalg import eigh


# ==================== 1. 超像素图构建与拉普拉斯位置编码模块 ====================
class SuperpixelGraphBuilder(nn.Module):
    """
    基于SLIC的超像素图构建与拉普拉斯位置编码生成器
    输入: 高光谱图像 (H, W, C)
    输出: 超像素节点特征、拉普拉斯位置编码、像素-超像素映射矩阵
    """

    def __init__(self, n_segments=100, compactness=10, sigma=1, laplacian_dim=32):
        super().__init__()
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.laplacian_dim = laplacian_dim  # 拉普拉斯位置编码维度m

    def forward(self, hsi_data):
        """
        Args:
            hsi_data: (H, W, C) 高光谱图像numpy数组
        Returns:
            node_features: (z, C) 超像素节点特征，z为超像素数量
            laplacian_pe: (z, m) 拉普拉斯位置编码
            pixel_to_superpixel: (H, W) 像素到超像素的索引映射
        """
        H, W, C = hsi_data.shape

        # 1. SLIC超像素分割
        segments = slic(
            hsi_data,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            channel_axis=-1,
            start_label=0
        )
        z = segments.max() + 1  # 超像素总数

        # 2. 计算超像素节点特征（区域均值）
        node_features = np.zeros((z, C), dtype=np.float32)
        for seg_id in range(z):
            mask = segments == seg_id
            node_features[seg_id] = hsi_data[mask].mean(axis=0)

        # 3. 构建超像素邻接矩阵W（8邻域空间邻接 + 光谱相似度）
        adjacency = np.zeros((z, z), dtype=np.float32)
        # 先找空间邻接的超像素对
        for i in range(H):
            for j in range(W):
                current_seg = segments[i, j]
                # 检查8邻域
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            neighbor_seg = segments[ni, nj]
                            if current_seg != neighbor_seg:
                                adjacency[current_seg, neighbor_seg] = 1
        # 用高斯核函数计算光谱相似度作为边权重
        for i in range(z):
            for j in range(z):
                if adjacency[i, j] == 1:
                    dist = np.linalg.norm(node_features[i] - node_features[j])
                    adjacency[i, j] = np.exp(-dist ** 2 / (2 * 1.0 ** 2))  # sigma=1.0

        # 4. 计算归一化图拉普拉斯矩阵 L = I - D^(-1/2) W D^(-1/2)
        degree = np.sum(adjacency, axis=1)
        degree_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))  # 防止除零
        normalized_laplacian = np.eye(z) - degree_inv_sqrt @ adjacency @ degree_inv_sqrt

        # 5. 拉普拉斯特征分解，取前m个最小特征值对应的特征向量作为位置编码
        eigenvalues, eigenvectors = eigh(normalized_laplacian)
        # 取第2到第m+1个特征向量（跳过第一个特征值为0的平凡解）
        laplacian_pe = eigenvectors[:, 1:self.laplacian_dim + 1]

        return (
            torch.FloatTensor(node_features),
            torch.FloatTensor(laplacian_pe),
            segments
        )


# ==================== 2. 结构感知多头自注意力(SPMHSA)模块 ====================
class StructurePerceptionMHSA(nn.Module):
    """
    论文核心创新：结构感知多头自注意力
    将k-hop局部拓扑结构作为注意力偏置融入计算
    """

    def __init__(self, d_model, n_heads, k_hop=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.k_hop = k_hop

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def build_k_hop_adjacency(self, adjacency):
        """
        构建多尺度k-hop邻接矩阵（简化版：邻接矩阵幂次）
        Args:
            adjacency: (z, z) 1-hop邻接矩阵
        Returns:
            k_hop_struct: (z, z) 多尺度结构偏置矩阵
        """
        z = adjacency.shape[0]
        k_hop_struct = torch.zeros_like(adjacency)

        # 计算1到k-hop的邻接矩阵并累加
        current_adj = adjacency.clone()
        for k in range(1, self.k_hop + 1):
            k_hop_struct += current_adj
            # 计算k+1-hop邻接矩阵（幂次）
            current_adj = torch.matmul(current_adj, adjacency)
            # 二值化，只保留连接关系
            current_adj = (current_adj > 0).float()

        # 归一化到[-1, 0]作为注意力偏置（抑制非局部连接）
        k_hop_struct = - (k_hop_struct / (k_hop_struct.max() + 1e-8))
        return k_hop_struct

    def forward(self, x, adjacency):
        """
        Args:
            x: (z, d_model) 输入特征（节点特征或位置编码）
            adjacency: (z, z) 1-hop邻接矩阵
        Returns:
            out: (z, d_model) 输出特征
        """
        residual = x
        z = x.shape[0]

        # 1. 线性投影 + 分头
        q = self.w_q(x).view(z, self.n_heads, self.d_k).transpose(0, 1)  # (n_heads, z, d_k)
        k = self.w_k(x).view(z, self.n_heads, self.d_k).transpose(0, 1)
        v = self.w_v(x).view(z, self.n_heads, self.d_k).transpose(0, 1)

        # 2. 构建k-hop结构偏置
        k_hop_struct = self.build_k_hop_adjacency(adjacency)  # (z, z)
        k_hop_struct = k_hop_struct.unsqueeze(0).repeat(self.n_heads, 1, 1)  # (n_heads, z, z)

        # 3. 计算注意力分数 + 结构偏置
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)  # (n_heads, z, z)
        scores = scores + k_hop_struct  # 融入结构感知偏置
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 4. 聚合特征
        out = torch.matmul(attn, v)  # (n_heads, z, d_k)
        out = out.transpose(0, 1).contiguous().view(z, self.d_model)  # (z, d_model)
        out = self.w_o(out)

        # 5. 残差连接 + LayerNorm
        out = self.layer_norm(out + residual)
        return out


# ==================== 3. 双向交叉注意力模块(BCAM) ====================
class BidirectionalCrossAttention(nn.Module):
    """
    双向交叉注意力：实现NFT与SPPT分支的双向信息交互
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 光谱→结构 交叉注意力
        self.cross_attn_spec_to_struct = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        # 结构→光谱 交叉注意力
        self.cross_attn_struct_to_spec = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)

        # 卷积瓶颈层（降维+升维）
        self.conv_bottleneck_spec = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.conv_bottleneck_struct = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.layer_norm_spec = nn.LayerNorm(d_model)
        self.layer_norm_struct = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat_spec, feat_struct):
        """
        Args:
            feat_spec: (z, d_model) NFT分支的光谱特征
            feat_struct: (z, d_model) SPPT分支的结构特征
        Returns:
            enhanced_spec: (z, d_model) 增强后的光谱特征
            enhanced_struct: (z, d_model) 增强后的结构特征
        """
        # 1. 卷积瓶颈预处理
        feat_spec_bottleneck = self.conv_bottleneck_spec(feat_spec)
        feat_struct_bottleneck = self.conv_bottleneck_struct(feat_struct)

        # 2. 光谱→结构 交叉注意力（结构特征作为Query，光谱特征作为Key/Value）
        attn_struct, _ = self.cross_attn_spec_to_struct(
            query=feat_struct_bottleneck,
            key=feat_spec_bottleneck,
            value=feat_spec_bottleneck
        )
        enhanced_struct = self.layer_norm_struct(feat_struct + self.dropout(attn_struct))

        # 3. 结构→光谱 交叉注意力（光谱特征作为Query，结构特征作为Key/Value）
        attn_spec, _ = self.cross_attn_struct_to_spec(
            query=feat_spec_bottleneck,
            key=feat_struct_bottleneck,
            value=feat_struct_bottleneck
        )
        enhanced_spec = self.layer_norm_spec(feat_spec + self.dropout(attn_spec))

        return enhanced_spec, enhanced_struct


# ==================== 4. 双分支Transformer层(DBIT) ====================
class DualBranchInteractiveTransformerLayer(nn.Module):
    """
    完整的双分支交互Transformer层
    """

    def __init__(self, d_model, n_heads, k_hop=3, dropout=0.1):
        super().__init__()
        # NFT分支：标准多头自注意力
        self.nft_mhsa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.nft_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.nft_norm1 = nn.LayerNorm(d_model)
        self.nft_norm2 = nn.LayerNorm(d_model)

        # SPPT分支：结构感知多头自注意力
        self.sppt_spmhsa = StructurePerceptionMHSA(d_model, n_heads, k_hop, dropout)
        self.sppt_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.sppt_norm1 = nn.LayerNorm(d_model)
        self.sppt_norm2 = nn.LayerNorm(d_model)

        # 双向交叉注意力
        self.bcam = BidirectionalCrossAttention(d_model, n_heads, dropout)

    def forward(self, feat_spec, feat_struct, adjacency):
        """
        Args:
            feat_spec: (z, d_model) 光谱特征
            feat_struct: (z, d_model) 结构特征
            adjacency: (z, z) 邻接矩阵
        Returns:
            feat_spec_out: (z, d_model) 输出光谱特征
            feat_struct_out: (z, d_model) 输出结构特征
        """
        # ---------------- NFT分支前向 ----------------
        residual_spec = feat_spec
        # 标准MHSA
        attn_spec, _ = self.nft_mhsa(feat_spec, feat_spec, feat_spec)
        feat_spec = self.nft_norm1(feat_spec + attn_spec)
        # MLP
        feat_spec = self.nft_norm2(feat_spec + self.nft_mlp(feat_spec))

        # ---------------- SPPT分支前向 ----------------
        residual_struct = feat_struct
        # 结构感知MHSA
        feat_struct = self.sppt_spmhsa(feat_struct, adjacency)
        # MLP
        feat_struct = self.sppt_norm2(feat_struct + self.sppt_mlp(feat_struct))

        # ---------------- 双向交叉注意力交互 ----------------
        feat_spec, feat_struct = self.bcam(feat_spec, feat_struct)

        return feat_spec, feat_struct


# ==================== 5. 完整SPGFormer特征提取器（简化版） ====================
class SPGFormerFeatureExtractor(nn.Module):
    """
    简化版SPGFormer：用于嵌入你现有UNet/Mamba框架的特征提取器
    输入：高光谱图像patch
    输出：增强后的空间-光谱联合特征
    """

    def __init__(self, in_channels=30, d_model=64, n_layers=3, n_heads=8,
                 n_segments=100, laplacian_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model

        # 超像素图构建器
        self.graph_builder = SuperpixelGraphBuilder(
            n_segments=n_segments,
            laplacian_dim=laplacian_dim
        )

        # 输入投影层：光谱特征和位置编码投影到d_model
        self.spec_proj = nn.Linear(in_channels, d_model)
        self.struct_proj = nn.Linear(laplacian_dim, d_model)

        # 双分支Transformer层堆叠
        self.transformer_layers = nn.ModuleList([
            DualBranchInteractiveTransformerLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])

        # 像素级卷积分支（深度可分离卷积）
        self.pixel_cnn = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )

        # 自适应融合参数
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def superpixel_to_pixel(self, superpixel_feat, segments):
        """
        将超像素级特征映射回像素级
        Args:
            superpixel_feat: (z, d_model) 超像素特征
            segments: (H, W) 像素-超像素索引映射
        Returns:
            pixel_feat: (H, W, d_model) 像素级特征
        """
        H, W = segments.shape
        pixel_feat = np.zeros((H, W, self.d_model), dtype=np.float32)
        for seg_id in range(superpixel_feat.shape[0]):
            mask = segments == seg_id
            pixel_feat[mask] = superpixel_feat[seg_id].detach().cpu().numpy()
        return torch.FloatTensor(pixel_feat)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 高光谱图像patch
        Returns:
            fused_feat: (B, d_model, H, W) 融合后的特征
        """
        B, C, H, W = x.shape
        fused_feats = []

        # 对batch中的每个样本单独处理（图结构处理通常为单样本）
        for i in range(B):
            # 1. 提取单样本高光谱数据 (H, W, C)
            hsi_data = x[i].permute(1, 2, 0).detach().cpu().numpy()

            # 2. 构建超像素图，获取节点特征和位置编码
            node_spec, node_struct, segments = self.graph_builder(hsi_data)
            node_spec = node_spec.to(x.device)
            node_struct = node_struct.to(x.device)

            # 3. 输入投影
            feat_spec = self.spec_proj(node_spec)  # (z, d_model)
            feat_struct = self.struct_proj(node_struct)  # (z, d_model)

            # 4. 构建邻接矩阵（简化版：基于空间邻接的二值矩阵）
            z = feat_spec.shape[0]
            adjacency = torch.zeros((z, z), device=x.device)
            # 这里简化处理，实际可复用graph_builder中的邻接矩阵
            # 为演示代码简洁，这里用随机邻接矩阵替代，实际请使用真实邻接
            adjacency = torch.eye(z, device=x.device)  # 临时用单位矩阵

            # 5. 通过双分支Transformer层
            for layer in self.transformer_layers:
                feat_spec, feat_struct = layer(feat_spec, feat_struct, adjacency)

            # 6. 双分支特征融合
            trs_feat = feat_spec + feat_struct  # (z, d_model)

            # 7. 超像素特征映射回像素级
            trs_pixel_feat = self.superpixel_to_pixel(trs_feat, segments).to(x.device)  # (H, W, d_model)
            trs_pixel_feat = trs_pixel_feat.permute(2, 0, 1).unsqueeze(0)  # (1, d_model, H, W)

            # 8. 像素级卷积分支
            pixel_cnn_feat = self.pixel_cnn(x[i:i + 1])  # (1, d_model, H, W)

            # 9. 自适应加权融合
            alpha = torch.sigmoid(self.alpha)
            fused_feat = alpha * trs_pixel_feat + (1 - alpha) * pixel_cnn_feat
            fused_feats.append(fused_feat)

        # 拼接batch
        return torch.cat(fused_feats, dim=0)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 模拟高光谱海面溢油检测输入 (Batch=2, 30个波段, 16x16 patch)
    dummy_input = torch.randn(2, 30, 16, 16)

    # 初始化SPGFormer特征提取器
    model = SPGFormerFeatureExtractor(
        in_channels=30,
        d_model=64,
        n_layers=3,
        n_heads=8,
        n_segments=50,  # 小patch用较少超像素
        laplacian_dim=32
    )

    # 前向传播
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出特征形状: {output.shape}")  # 期望输出: (2, 64, 16, 16)