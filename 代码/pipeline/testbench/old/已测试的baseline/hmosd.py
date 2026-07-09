import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from skimage.segmentation import slic
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple


# ==============================================
# 1. 论文核心：ISLIC改进超像素分割模块
# 对应论文II.A节：Superpixel Segmentation Based on Improved SLIC
# ==============================================
class ISLICProcessor:
    """
    改进的SLIC超像素分割算法，包含：
    1. 高斯滤波(GF)平滑去噪
    2. SLIC超像素分割
    3. 像素强度平滑技术(PIST)超像素内均值平滑
    """

    def __init__(
            self,
            n_segments: int = 100,  # 论文参数：超像素分割数100
            gaussian_sigma: float = 5,  # 论文参数：高斯滤波标准差5
            slic_compactness: float = 10,
            slic_max_iter: int = 10
    ):
        self.n_segments = n_segments
        self.gaussian_sigma = gaussian_sigma
        self.slic_compactness = slic_compactness
        self.slic_max_iter = slic_max_iter

    def gaussian_filtering(self, hsi_data: np.ndarray) -> np.ndarray:
        """高斯滤波平滑，对应论文公式(7)(8)"""
        # 输入形状：(H, W, Bands)，对每个波段单独做高斯滤波
        smoothed_data = np.zeros_like(hsi_data)
        for band in range(hsi_data.shape[-1]):
            smoothed_data[..., band] = gaussian_filter(
                hsi_data[..., band], sigma=self.gaussian_sigma
            )
        return smoothed_data

    def pist_smoothing(self, hsi_data: np.ndarray, segments: np.ndarray) -> np.ndarray:
        """像素强度平滑技术PIST，对应论文公式(9)(10)(11)"""
        # 输入形状：(H, W, Bands)，segments：(H, W)超像素标签
        n_segments = segments.max() + 1
        smoothed_data = np.zeros_like(hsi_data)

        # 对每个超像素，用均值替换内部所有像素值
        for seg_id in range(n_segments):
            mask = segments == seg_id
            if mask.sum() == 0:
                continue
            # 计算超像素内每个波段的均值
            seg_mean = hsi_data[mask].mean(axis=0)
            smoothed_data[mask] = seg_mean
        return smoothed_data

    def __call__(self, hsi_pca_data: np.ndarray) -> np.ndarray:
        """
        ISLIC完整流程
        :param hsi_pca_data: PCA降维后的高光谱数据，形状(H, W, Bands)
        :return: ISLIC处理后的空间特征，形状(H, W, Bands)
        """
        # 步骤1：高斯滤波平滑
        smoothed_hsi = self.gaussian_filtering(hsi_pca_data)

        # 步骤2：SLIC超像素分割
        # 归一化到0-1，适配SLIC输入要求
        norm_hsi = (smoothed_hsi - smoothed_hsi.min()) / (smoothed_hsi.max() - smoothed_hsi.min() + 1e-8)
        segments = slic(
            norm_hsi,
            n_segments=self.n_segments,
            compactness=self.slic_compactness,
            max_num_iter=self.slic_max_iter,
            channel_axis=-1,
            start_label=0
        )

        # 步骤3：PIST像素强度平滑
        pist_smoothed = self.pist_smoothing(hsi_pca_data, segments)
        return pist_smoothed


# ==============================================
# 2. 论文核心：混合注意力模块MAM
# 对应论文II.B.2节：Mixed Attention Mechanism
# ==============================================
class SpectralAttention(nn.Module):
    """光谱注意力模块，对应论文公式(20)(22)"""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        # 瓶颈维度过小时 in_channels // reduction_ratio 可能为 0（如 C=12、ratio=16），会导致 Linear(12,0) 与退化输出
        hidden = max(1, in_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：3D卷积特征，形状(B, C, D, H, W)
        C: 通道数(卷积核数)，D: 光谱维度，H/W: 空间维度
        """
        B, C, D, H, W = x.shape
        # 全局平均池化和最大池化，在光谱+空间维度聚合
        avg_pool = F.adaptive_avg_pool3d(x, output_size=(1, 1, 1)).view(B, C)
        max_pool = F.adaptive_max_pool3d(x, output_size=(1, 1, 1)).view(B, C)

        # 共享MLP学习权重
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        # 生成光谱注意力权重
        spectral_att = self.sigmoid(avg_out + max_out).view(B, C, 1, 1, 1)
        # 注意力加权
        return x * spectral_att


class SpatialAttention(nn.Module):
    """空间注意力模块，对应论文公式(21)(23)"""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        # 论文参数：卷积核大小5×5×5
        self.conv = nn.Conv3d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：光谱注意力加权后的特征，形状(B, C, D, H, W)
        """
        # 在光谱通道维度做全局平均池化和最大池化
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接池化结果
        concat = torch.cat([avg_pool, max_pool], dim=1)
        # 卷积生成空间注意力权重
        spatial_att = self.sigmoid(self.conv(concat))
        # 注意力加权
        return x * spatial_att


class MAM(nn.Module):
    """
    混合注意力模块MAM，串行结构：先光谱注意力，后空间注意力
    对应论文Fig2、Fig3、Fig4，论文验证该顺序性能更优
    """

    def __init__(self, in_channels: int, spa_kernel_size: int = 5, reduction_ratio: int = 16):
        super().__init__()
        self.spectral_att = SpectralAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spa_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectral_att(x)
        x = self.spatial_att(x)
        return x


# ==============================================
# 3. 论文核心：MAM-SSFEN光谱-空间特征提取网络
# 对应论文II.B.1节：Spectral–Spatial Feature Extraction Network
# ==============================================
class MAM_SSFEN(nn.Module):
    """
    带混合注意力的光谱-空间特征提取网络
    论文参数：
    - 3层3D卷积，filters分别为12、24、48
    - 卷积核分别为(3,3,3)、(1,1,3)、(1,1,3)
    - 步长分别为(1,1,1)、(1,1,1)、(1,1,2)
    - MAM加在第一个卷积层之后
    """

    def __init__(self, in_bands: int = 25, dropout_rate: float = 0.5):
        super().__init__()
        # 第一层3D卷积 + MAM
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=12,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.mam = MAM(in_channels=12, spa_kernel_size=5)
        self.act = nn.ReLU(inplace=True)

        # 第二层3D卷积
        self.conv2 = nn.Conv3d(
            in_channels=12,
            out_channels=24,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 1),
            padding=(0, 0, 1)
        )

        # 第三层3D卷积
        self.conv3 = nn.Conv3d(
            in_channels=24,
            out_channels=48,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            padding=(0, 0, 1)
        )

        # Dropout防止过拟合，论文参数0.5
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：高光谱 patch，形状 (B, 1, Bands, H_patch, W_patch)，与 Conv3d(N,C,D,H,W) 一致。
        输出：提取后的深度特征（flatten 后向量）
        """
        # 第一层卷积 + MAM
        x = self.act(self.conv1(x))
        x = self.mam(x)

        # 后续卷积层
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))

        # 展平 + Dropout
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        return x


# ==============================================
# 4. 完整HMOSDN端到端模型
# 对应论文Fig1整体流程图
# ==============================================
class HMOSDN(nn.Module):
    """
    高光谱海洋溢油检测网络HMOSDN完整实现
    整体流程：
    1. 输入高光谱数据 → PCA降维
    2. ISLIC超像素分割提取空间特征
    3. 光谱特征+空间特征融合
    4. MAM-SSFEN深度特征提取
    5. 全连接层二分类输出
    """

    def __init__(
            self,
            in_bands: int = 25,  # PCA降维后的波段数
            patch_size: int = 5,  # 论文patch size 5
            dropout_rate: float = 0.5,
            num_classes: int = 1  # 二分类：溢油/海水
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_bands = in_bands

        # 核心特征提取主干
        self.feature_extractor = MAM_SSFEN(in_bands=in_bands, dropout_rate=dropout_rate)

        # 计算全连接层输入维度（自动适配）
        # 与 PyTorch Conv3d 一致：(N, C, D, H, W)，D 为光谱维，H/W 为空间 patch
        with torch.no_grad():
            dummy = torch.randn(1, 1, in_bands, patch_size, patch_size)
            feat_dim = self.feature_extractor(dummy).shape[-1]

        # 分类头：num_classes==1 时用 Sigmoid + BCE；多类时用 logits，配合 CrossEntropyLoss
        if num_classes == 1:
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_classes),
                nn.Sigmoid(),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_classes),
            )

        # ISLIC处理器
        self.islic_processor = ISLICProcessor()
        # PCA降维器
        self.pca = PCA(n_components=in_bands, random_state=42)

    def preprocess(
            self,
            hsi_data: np.ndarray,
            fit_pca: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理：PCA降维 + ISLIC空间特征提取 + 特征融合
        :param hsi_data: 原始高光谱数据，形状(H, W, 原始波段数)
        :param fit_pca: 是否训练PCA（训练集设为True，测试集设为False）
        :return: 融合后的特征(H, W, 25)，标签掩码(H, W)（如果有）
        """
        H, W, B = hsi_data.shape
        # 步骤1：PCA降维
        flatten_data = hsi_data.reshape(-1, B)
        if fit_pca:
            pca_data = self.pca.fit_transform(flatten_data)
        else:
            pca_data = self.pca.transform(flatten_data)
        pca_data = pca_data.reshape(H, W, self.in_bands)

        # 步骤2：ISLIC提取空间特征
        spatial_feature = self.islic_processor(pca_data)

        # 步骤3：光谱特征+空间特征融合（论文中直接叠加输入网络）
        fused_feature = pca_data + spatial_feature
        return fused_feature, pca_data

    def extract_patches(
            self,
            fused_feature: np.ndarray,
            labels: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        从融合特征中提取patch块，适配网络输入
        :param fused_feature: 融合后的特征，形状(H, W, Bands)
        :param labels: 标签掩码，形状(H, W)，1=溢油，0=海水
        :return: patch张量(B, 1, 5, 5, 25)，标签张量(B, 1)
        """
        H, W, B = fused_feature.shape
        pad_size = self.patch_size // 2
        # 镜像填充边界
        padded_feature = np.pad(fused_feature, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="symmetric")

        patches = []
        patch_labels = []

        for i in range(H):
            for j in range(W):
                # 提取当前像素的邻域patch
                patch = padded_feature[i:i + self.patch_size, j:j + self.patch_size, :]
                patches.append(patch)
                if labels is not None:
                    patch_labels.append(labels[i, j])

        # 转换为张量：(N, H, W, B) -> (N, 1, B, H, W)，与 MAM_SSFEN / Conv3d 一致
        patches_tensor = torch.from_numpy(np.array(patches)).float().permute(0, 3, 1, 2).unsqueeze(1)
        if labels is not None:
            labels_tensor = torch.from_numpy(np.array(patch_labels)).float().unsqueeze(1)
            return patches_tensor, labels_tensor
        return patches_tensor, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x: 输入 patch，形状 (B, 1, in_bands, patch_h, patch_w)
        :return: 分类 logits（num_classes=2 时为 (B,2)；num_classes=1 时为 (B,1) 且含 Sigmoid）
        """
        feat = self.feature_extractor(x)
        out = self.classifier(feat)
        return out


def build_tri_branch_net(
    sample_x: torch.Tensor,
    num_classes: int = 2,
    branch_dim: int = 128,
    dropout: float = 0.4,
    **kwargs,
) -> HMOSDN:
    """训练脚本入口：从样本张量推断输入维度并构建 HMOSDN 模型。"""
    if sample_x.dim() != 5:
        raise ValueError("sample_x 应为 (B, 1, C, H, W)")
    _, _, c, h, w = sample_x.shape
    return HMOSDN(in_bands=c, patch_size=h, dropout_rate=dropout, num_classes=num_classes)


# ==============================================
# 5. 训练/推理配置与测试代码
# ==============================================
if __name__ == "__main__":
    # --------------------------
    # 1. 模型初始化
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HMOSDN(in_bands=25, patch_size=5).to(device)
    print("HMOSDN模型初始化完成！")
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # --------------------------
    # 2. 测试前向传播
    # --------------------------
    # 模拟论文HOSD数据集的输入：单张高光谱图像，160×160，224个原始波段
    dummy_hsi = np.random.randn(160, 160, 224)
    dummy_label = np.random.randint(0, 2, size=(160, 160))

    # 预处理
    fused_feat, pca_feat = model.preprocess(dummy_hsi, fit_pca=True)
    print(f"PCA降维后形状：{pca_feat.shape}")
    print(f"ISLIC融合后特征形状：{fused_feat.shape}")

    # 提取patch
    patches, labels = model.extract_patches(fused_feat, dummy_label)
    patches = patches.to(device)
    labels = labels.to(device)
    print(f"输入patch形状：{patches.shape}")
    print(f"标签形状：{labels.shape}")

    # 前向推理
    model.eval()
    with torch.no_grad():
        pred = model(patches[:64])  # 取一个batch测试
    print(f"模型输出形状：{pred.shape}")
    print("模型前向传播测试通过！")

    # --------------------------
    # 3. 训练配置（完全对齐论文）
    # --------------------------
    # 论文参数：
    # - 优化器：Adam
    # - 学习率：0.001
    # - 损失函数：二元交叉熵BCE
    # - batch size：64
    # - epoch：300
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    epochs = 300
    batch_size = 64

    print("\n===== 论文训练配置 =====")
    print(f"优化器：Adam，学习率：0.001")
    print(f"损失函数：二元交叉熵BCE")
    print(f"Batch Size：{batch_size}，Epoch：{epochs}")