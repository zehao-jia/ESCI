# SSFTTnet.py

## 权重初始化函数

```python
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)
```

- 首先,函数接受参数m,m代表模型中的一个层例如卷积层之类的
- 通过m.__class__.__name__获取层的类名
- 如果该层为全链接层或3d卷积层,则使用kaiming正则化

> 1. isinstance(对象,类名):如果实例是类或类的子类,返回True<br>
> 2. kaiming正则化:一种初始化方法,用于缓解梯度消失或梯度爆炸的问题,根据上一层的神经元数量,以正态分布或均匀分布的方式初始化权重,使得每一层的输出方差保持稳定
> 3. 代码中使用正态分布初始化

## 残差链接模块

```python
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
```

- 定义残差链接模块
- 所有模块的初始化先继承父类初始化函数nn.moudule
- 向实例中添加函数**fn**
- 前向传播:残差连接的前向传播,输入加输出


