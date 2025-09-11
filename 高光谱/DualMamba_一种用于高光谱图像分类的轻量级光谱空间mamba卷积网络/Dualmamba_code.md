- **dualmamba的主要贡献之一就是提供了一个轻量级的mamba块**
 ```python
 class LightweightSpatialMambaBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, kernel_size=3, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 2
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # 线性变换层
        self.linear1 = nn.Linear(dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, dim)
        
        # 空间卷积（Depthwise Conv）
        self.spatial_conv = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, 
            kernel_size=kernel_size, 
            padding=kernel_size//2,
            groups=self.hidden_dim  # Depthwise卷积
        )
        # 激活函数
        self.act = nn.SiLU()        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 保存残差连接
        residual = x
        # 确保输入格式为 [B, H, W, C]
        if x.dim() == 4 and x.shape[1] != self.dim:
            # 假设是 [B, C, H, W] 格式，转换为 [B, H, W, C]
            x = x.permute(0, 2, 3, 1)
        # 第一层归一化和线性变换
        x = self.norm1(x)
        x = self.linear1(x)
        # 转换为卷积需要的格式 [B, C, H, W]
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        # 空间卷积操作
        x = self.spatial_conv(x)
        # 激活函数
        x = self.act(x)
        # 转换回 [B, H, W, C] 格式
        x = x.permute(0, 2, 3, 1)
        # Dropout
        x = self.dropout(x)
        # 第二层线性变换和归一化
        x = self.linear2(x)
        x = self.norm2(x)
        # 残差连接
        x = x + residual
        # 如果需要，转换回原始格式
        if residual.dim() == 4 and residual.shape[1] == self.dim:
            x = x.permute(0, 3, 1, 2)
        
        return x
 ```