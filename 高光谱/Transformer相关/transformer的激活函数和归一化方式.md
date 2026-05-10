# 激活函数:GeLU(高斯误差线性单元)

$GELU(x)=x⋅Φ(x)≈0.5x(1+tanh\sqrt{(\frac{2}{\pi}​​(x+0.044715x^3))})$
- 光滑性:连续可微,梯度流动平滑
- 随机性:对噪声有更强的鲁棒性
- 工程优化:$quickGeLU:QuickGELU(x) = x\cdot\sigma(1.702x)$
# 归一化方式:LN+残差
$LN(x)=\gamma\cdot \frac{x-\mu}{\sqrt{ \sigma^2+\epsilon }}+\beta$
