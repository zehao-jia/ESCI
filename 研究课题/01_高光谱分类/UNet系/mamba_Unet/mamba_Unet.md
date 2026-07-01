本文首次介绍了一种纯SSM（Selective Sequential Modeling）模型用于医学图像分割的方法，并提出了基于VSS（Variational Selective Sequential）块的VM-UNet作为基准模型。通过在VM-UNet中使用VSS块并使用预训练的VMamba-S初始化权重，充分发挥了SSM模型的能力。
![[屏幕截图 2026-01-06 225849.png]]
# vm-unet
## 结构:
- patch embedding
- vss_block
- patch merging
- patch expanding,
### patch embedding
- 将输入图像截取为$4\times 4$大小的patch,随后将图像的维度映射到C(默认96)
- 随后,使用LN归一化,将其输入encoder
### encoder
- 编码器由四个阶段组成,前三个阶段的末尾用patch_merging升维
### decoder
- 解码器是类似的操作,在最后三个阶段的开始,利用patch_expanding减少特征通道并增加宽度和长度
### skip_connection
- 采用简单的加法操作,没有引入额外的操作