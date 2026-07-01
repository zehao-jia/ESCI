# 引言
 MK-UNet 的核心是多内核深度卷积块 (MKDC)，
 - 我们的 MK-UNet 网络具有仅 0.316M 个参数和 0.314G FLOP 的适度计算量
 - 在六个二进制医学成像基准上提供了比最先进 (SOTA) 方法更高的准确性。 具体来说，MKUNet 在 DICE 分数方面优于 TransUNet，参数和 FLOP 分别减少了近 333 倍和 123 倍。 
 - 同样，与 UNeXt 相比，MK-UNet 表现出卓越的分割性能，将 DICE 分数提高了 6.7%，同时 Params 减少了 4.7 倍
 - 我们的 MK-UNet 还优于其他最近的轻量级网络，例如 MedT、CMUNeXt、EGEUNet 和 Rolling-UNet，且计算资源要低得多。 
# 模型内容

![[屏幕截图 2026-05-02 143352.png]]
$$
SA(x) = \sigma(LKC([Ch_{max}(x),Ch_{avg}(x)]))
$$
- 其中,LKC是一个$7\times7$的卷积核
$$
CA(x) = \sigma(PWC_{2}(R(PWC_{1}(AMP(x))))+PWC_{1}(R(PWC_{1}(MMP(x))))
$$
- 输入经过两种池化后一个逐点卷积,激活函数relu后逐点卷积在相加加一个sigmoid函数

