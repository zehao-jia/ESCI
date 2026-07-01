![[研究课题/01_高光谱分类/Transformer系/ASSA/overview.png]]
# ASSA:自适应稀疏注意力
- 设计思路:利用[[ReLU|relu函数的稀疏性]]过滤负相关特征,利用softmax的稠密性兜底信息流
## 数据流:
### 稀疏分支(SSA)
- 首先,特征图经过处理生成qkv矩阵,使用平方relu激活函数代替softmax,relu将负分数的权重置零,从而切断无关token的交互
$$
SSA = ReLU^2\left( \frac{QK^T}{\sqrt{ d }}+B \right)
$$
- 效果:显式稀疏化,避免topk带来的排序开销
### 稠密分支(DSA)
- 目的:防止ReLU导致的过度稀疏,引起信息断层
$$
DSA = Softmax\left( \frac{QK^T}{\sqrt{ d }} +B\right)
$$
### 自适应融合
- 最终的注意力图A由两者加权获得
$$
A = (\omega_{1}*SSA+\omega_{2}*DSA)V
$$
