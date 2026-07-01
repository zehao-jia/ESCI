# 第一遍
## 摘要
- 深的神经网络难以训练
- 使用残差学习,使训练容易很多
- 152层,<font color="red">8倍于vgg但复杂度更低</font>
## 结论(无)
## 关键图表

- <img src="屏幕截图 2025-07-03 215047.png" width="300px"><br>
- 正常情况下,层数过高并不代表模型效果就更好(不是过拟合,在训练和验证上表现都不好)<br>
<img src="屏幕截图 2025-07-03 215552.png" width="300px"><br>
# 第二遍
## introduction
- 堆叠更多层模型就更好了吗?->不是,层数堆叠导致训练误差变高(不是过拟合)

- 下层和比较浅的网络一致,上层变为恒等映射(identity mapping)

### deep residual learning network
- 学习的内容:<br>
- 假设我想学的是$\mathcal{H} (x)$
- 1.:一部分层学习内容和比较浅的网络一致,试图输出一个x
- 2.:另一部分学习的内容是输出值和真实值的残差,记为$\mathcal{F} (x)=\mathcal{H} (x)-x$<br>
<img src="屏幕截图 2025-07-03 221549.png" width="300px"><br>
- 方法:residual connection
## related work
### 输入输出形状不同
- 添加额外的0
- <font color="red">1 * 1卷积调整通道数</font>
- 没有全链接层->没有dropout
- <font color="red">bottle neck设计</font>
## 实验
- map精度
- cnn的主干模型换成resnet
- resnet如何避免梯度消失:<br>

<img src="屏幕截图 2025-07-04 124411.png" width="300px">
# 总结
- 解决的问题:神经网络堆叠层数模型性能未见显著提升,分别训练残差和预想输出