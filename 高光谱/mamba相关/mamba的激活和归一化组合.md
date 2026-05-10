# 核心激活函数:SiLU(Swish)
$$
Swish(x) = x\cdot \sigma(x) = \frac{x}{1+e^{-x}}
$$
- 光滑性:处处可微,更适配mamba的连续状态空间模型的微分特性
- 梯度流动:梯度衰减相比于ReLU更平缓
- 工程验证
# 核心归一化方式:RMSNorm
$$
RMSNorm(x) = \frac{x}{\sqrt{ \frac{1}{d}\sum_{i=1}^d x_{i^2}+\epsilon}}\cdot \gamma
$$
- 计算效率高:去掉均值中心化步骤,计算量减少约50%
- 长序列稳定性:在超长序列中,均值中心化容易受到极端值影响,rmsnorm归一化方差
- 对齐ssm的无偏置线性变化的数学结构
# 变体
- 低资源:RELU
- 小批量:LN
- 高算力:GELU
- 分布式训练:DeepNorm
# 避免
- ReLU+LN:梯度消失
- sigmoid/Tanh:饱和区使得ssm更新失效
- BN:序列长度不固定导致批次统计混乱
