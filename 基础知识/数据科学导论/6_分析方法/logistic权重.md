# Logistic回归：最大似然法求解权重（一般情况）

Logistic回归是二分类任务的线性模型，通过Sigmoid函数将线性输出映射为概率，权重参数$\boldsymbol{w}$和偏置$b$的求解核心是**最大化似然函数**（等价于最小化负对数似然损失）。以下是一般情况的详细推导过程：

## 1. 模型定义与符号说明
设数据集为 $D = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^n$，其中：
- $\boldsymbol{x}_i \in \mathbb{R}^d$：第$i$个样本的$d$维特征向量；
- $y_i \in \{0, 1\}$：第$i$个样本的二分类标签（0为负类，1为正类）；
- $\boldsymbol{w} \in \mathbb{R}^d$：特征权重向量；
- $b \in \mathbb{R}$：偏置项。

### 1.1 Sigmoid函数与概率模型
引入Sigmoid函数将线性输出映射到$[0,1]$区间（表示概率）：
$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \boldsymbol{w}^T \boldsymbol{x}_i + b
$$
Sigmoid函数的关键导数性质（后续推导核心）：
$$
\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))
$$

正类（$y=1$）与负类（$y=0$）的概率可统一表示为：
$$
P(y_i \mid \boldsymbol{x}_i; \boldsymbol{w}, b) = \sigma\left(\boldsymbol{w}^T \boldsymbol{x}_i + b\right)^{y_i} \cdot \left[1 - \sigma\left(\boldsymbol{w}^T \boldsymbol{x}_i + b\right)\right]^{1-y_i}
$$

## 2. 似然函数与对数似然函数
由于样本独立同分布，**似然函数**为所有样本概率的乘积（衡量参数对观测数据的拟合程度）：
$$
\mathcal{L}(\boldsymbol{w}, b) = \prod_{i=1}^n P(y_i \mid \boldsymbol{x}_i; \boldsymbol{w}, b)
$$
代入概率表达式，得：
$$
\mathcal{L}(\boldsymbol{w}, b) = \prod_{i=1}^n \left\{ \sigma\left(\boldsymbol{w}^T \boldsymbol{x}_i + b\right)^{y_i} \cdot \left[1 - \sigma\left(\boldsymbol{w}^T \boldsymbol{x}_i + b\right)\right]^{1-y_i} \right\}
$$

为简化计算（将乘积转为求和，降低数值难度），对似然函数取自然对数，得到**对数似然函数**：
$$
\ln \mathcal{L}(\boldsymbol{w}, b) = \sum_{i=1}^n \left\{ y_i \cdot \ln \sigma\left(\boldsymbol{w}^T \boldsymbol{x}_i + b\right) + (1-y_i) \cdot \ln \left[1 - \sigma\left(\boldsymbol{w}^T \boldsymbol{x}_i + b\right)\right] \right\}
$$

## 3. 目标函数：最小化负对数似然
最大似然估计（MLE）的目标是找到使$\ln \mathcal{L}(\boldsymbol{w}, b)$最大的参数$(\boldsymbol{w}, b)$，等价于最小化**负对数似然函数**（即Logistic回归的损失函数$J(\boldsymbol{w}, b)$）：
$$
J(\boldsymbol{w}, b) = -\frac{1}{n} \ln \mathcal{L}(\boldsymbol{w}, b)
$$
代入对数似然表达式，得：
$$
J(\boldsymbol{w}, b) = -\frac{1}{n} \sum_{i=1}^n \left\{ y_i \cdot \ln \hat{p}_i + (1-y_i) \cdot \ln (1 - \hat{p}_i) \right\}
$$
其中$\hat{p}_i = \sigma\left(\boldsymbol{w}^T \boldsymbol{x}_i + b\right)$为第$i$个样本的正类预测概率。

## 4. 梯度计算（核心步骤）
通过梯度下降法最小化$J(\boldsymbol{w}, b)$，需先计算损失函数对$\boldsymbol{w}$和$b$的梯度。

### 4.1 单个样本的梯度
令$z_i = \boldsymbol{w}^T \boldsymbol{x}_i + b$，则$\hat{p}_i = \sigma(z_i)$，且$\frac{\partial z_i}{\partial \boldsymbol{w}} = \boldsymbol{x}_i$、$\frac{\partial z_i}{\partial b} = 1$。

单个样本损失$j_i = - \left[ y_i \ln \hat{p}_i + (1-y_i) \ln (1 - \hat{p}_i) \right]$的梯度，由链式法则推导：
1. 对$\hat{p}_i$的偏导：
$$
\frac{\partial j_i}{\partial \hat{p}_i} = - \left( \frac{y_i}{\hat{p}_i} - \frac{1-y_i}{1 - \hat{p}_i} \right)
$$
2. 对$z_i$的偏导（利用Sigmoid导数性质）：
$$
\frac{\partial \hat{p}_i}{\partial z_i} = \hat{p}_i (1 - \hat{p}_i)
$$
3. 对$\boldsymbol{w}$和$b$的偏导：
$$
\frac{\partial j_i}{\partial \boldsymbol{w}} = \frac{\partial j_i}{\partial \hat{p}_i} \cdot \frac{\partial \hat{p}_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial \boldsymbol{w}} = (\hat{p}_i - y_i) \cdot \boldsymbol{x}_i
$$
$$
\frac{\partial j_i}{\partial b} = \frac{\partial j_i}{\partial \hat{p}_i} \cdot \frac{\partial \hat{p}_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial b} = (\hat{p}_i - y_i)
$$

### 4.2 整体损失的梯度
对所有样本的梯度取平均，得到整体损失的梯度：
$$
\nabla_{\boldsymbol{w}} J(\boldsymbol{w}, b) = \frac{1}{n} \sum_{i=1}^n (\hat{p}_i - y_i) \cdot \boldsymbol{x}_i
$$
$$
\nabla_{b} J(\boldsymbol{w}, b) = \frac{1}{n} \sum_{i=1}^n (\hat{p}_i - y_i)
$$

## 5. 参数更新规则
梯度下降法通过迭代更新参数，使损失函数逐步减小。设$\eta > 0$为学习率（控制步长），参数更新公式为：
$$
\boldsymbol{w} \leftarrow \boldsymbol{w} - \eta \cdot \nabla_{\boldsymbol{w}} J(\boldsymbol{w}, b) = \boldsymbol{w} - \frac{\eta}{n} \sum_{i=1}^n (\hat{p}_i - y_i) \cdot \boldsymbol{x}_i
$$
$$
b \leftarrow b - \eta \cdot \nabla_{b} J(\boldsymbol{w}, b) = b - \frac{\eta}{n} \sum_{i=1}^n (\hat{p}_i - y_i)
$$

## 6. 结论
通过最大似然估计，Logistic回归的权重$\boldsymbol{w}$和偏置$b$可通过梯度下降法迭代求解。核心是利用Sigmoid函数的导数性质简化梯度计算，最终通过最小化负对数似然损失得到最优参数，实现对二分类概率的准确预测。