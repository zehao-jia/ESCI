@misc{kimiteam2026attentionresiduals,
      title={Attention Residuals}, 
      author={Kimi Team and Guangyu Chen and Yu Zhang and Jianlin Su and Weixin Xu and Siyuan Pan and Yaoyu Wang and Yucheng Wang and Guanduo Chen and Bohong Yin and Yutian Chen and Junjie Yan and Ming Wei and Y. Zhang and Fanqing Meng and Chao Hong and Xiaotong Xie and Shaowei Liu and Enzhe Lu and Yunpeng Tai and Yanru Chen and Xin Men and Haiqing Guo and Y. Charles and Haoyu Lu and Lin Sui and Jinguo Zhu and Zaida Zhou and Weiran He and Weixiao Huang and Xinran Xu and Yuzhi Wang and Guokun Lai and Yulun Du and Yuxin Wu and Zhilin Yang and Xinyu Zhou},
      year={2026},
      eprint={2603.15031},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.15031}, 
}
# 一句话总结
- 使用softmax函数让残差链接有选择地学习内容,避免范数增长和内容稀释
# 引入
## 残差连接的缺陷
- 缺乏选择性访问:权重固定,早期层的内容被稀释,后层的贡献变小
- 不可逆损失:后续层无法恢复早期的特定表示
- 输出增长:随着层数增加,残差流的范数不断增长
### 深层机制:prenorm稀释效应
- 由于范数增长,深层网络不得不放大输出,导致许多层实际上是冗余的
# 核心创新 attention residual机制
- 将固定的权重累加改为softmax函数
$$
h_{l} = \sum^{l-1}_{i=0}\alpha_{l,i}\cdot RMSNorm(h_{i})
$$
- 其中,$\alpha$为注意力权重
$$
\alpha_{l,i} = softmax_{i}\left( \frac{q_{l}^Tk_{i}}{\sqrt{ d }} \right)
$$
- q:生成的可学习参数
- k:历史层归一化
- v与键相同
# 两种架构
## full attnres
- 在具体实现中,kv矩阵在深度上被扩展为存储所有历史层的RMSNorm输出,q矩阵则是当前层的未查询向量
- 细节一:无条件化的Q矩阵:Q与输入无关,只与当前的位置索引有关,类似于位置编码在深度维度上的优化
- 细节二:对KV添加RMSNorm,这可以防止量级较大的层输出一家独大
- 细节三:零初始化伪查询向量:训练起始时所有等价于标准的均匀平均,避免随机初始化


- 深度累积和序列循环是对偶的

# attnres
