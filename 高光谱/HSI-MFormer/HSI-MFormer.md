# 前备知识
## [[transformer|transformer]]

# 网络架构
![[屏幕截图 2025-10-15 204044.png]]
## 1.MTG(多尺度令牌生成模块)
- 使用不同内核尺寸的3d卷积,这些卷积随后被传输到mamba和transformer中进行短程和远程空间光谱特征提取
- 如图MTG由三个卷积组构成,这个生成HSI块的过程可表述为
$$
F_{s} = ReLU(BN(3DConv_{s\times s\times s(x)}))
$$
$$
F_{m} = ReLU(BN(3DConv_{m\times m\times m(x)}))
$$
$$
F_{l} = ReLU(BN(3DConv_{l\times l\times l(x)}))
$$
- 三个尺度的滤波器个数都是32
- 在每个卷积组后,MTG进一步设计一个用于维度转换的线性嵌入层,以s为例
$$
T_{s} = Linear(f_s)
$$
- 就这样,生成了三个频谱-空间标记组$T_s,T_m,T_l$,嵌入操作对每个组独立进行,没有参数共享
## 2.ITE(内尺度transformer专家)
- ITE将三个尺度的token传入三个MHSA中进行短程依赖建模随后配合残差连接,以s组为例,可表示为
$$
F_{s} = T_s + MHSA_{s}(LN(T_{s}))
$$
- 之后,将所有组得到的特征链接起来全面利用获得的信息
$$
F_{ITE} = Concat(F_{s},F_m,F_l)
$$
- 随后,通过一个FFN模块(两个线性层和一个激活函数组成的前馈网络)进行非线性变化,增强模型的表示能力
$$F_{ITE} = F_{ITE}+FFN(F_{ITE})$$
## 3.跨尺度Mamba专家(CME)
- 首先,CME将所有标记组按分辨率顺序集成并展平为一个长序列,考虑正向和反向扫描方向以增强对空间的感知能力,以MTG生成的标记组为输入,展平过程可表示为:
$$
S_{forward} = \left[T_{S}^1\cdots T_{S}^{S^2},T_{M}^1\cdots T_{M}^{M^2},T_{L}^1\cdots T_{L}^{L^2} \right]
$$
$$S_{backward}为S_{forward}的反向
$$
- 随后,展平的序列被传输进双向mamba块进行独立的远程建模,对于每个方向,序列首先经过归一化,随后传入两个并行的线性嵌入层