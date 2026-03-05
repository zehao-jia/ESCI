## 缩放点积注意力
$$
A = Softmax(\frac{Q\cdot K^T}{\lambda })\cdot V
$$
-  $Q \in \mathbb{R}^{N \times d}$（查询矩阵）、$K \in \mathbb{R}^{N \times d}$（键矩阵）、$V \in \mathbb{R}^{N \times d}$（值矩阵），N为 Token 总数，d为每个 Token 的特征维度；