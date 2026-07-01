![[overall.png]]

# 总体架构

- 给定一个LR图像,经过一个$3\times3$卷积,再送入多个CGA块的级联,最终输出经过一个$3\times3$卷积与原图残差链接得到输出
- loss使用L1loss(从经验上看，L1 在恢复方面也优于 L2，因为它对异常值具有鲁棒性并减少了过度平滑)
-  在局部，LCGA 执行窗口注意力来稳定聚合 [35]，同时通过曲率调制 logits 显式增强每个窗口内的脊/曲线连续性。 在全球范围内，CGTA 遵循内容自适应路由策略 [27]、[28]，该策略选择由曲率感知显着性引导的紧凑 Top-k 令牌集，并仅在选定的键/值上应用混合交叉注意力。 该设计针对 SR 的两个关键需求：局部保留精细曲线结构并保持沿延伸脊的远程一致性，同时将注意力成本从 O(N2) 降低到 O(Nk)，即固定 k/N 的接近线性
  ![[LCGA.png]]

# LCGA

- 如上图所示,LCGA通过将标准的窗口注意力与曲率调制变体混合实现在山脊邻域强调聚合,同时在山外保持稳定
- 输入图经过$3\times3$卷积后经过一个跨通道平均和一个LN层得到每个窗口位置的标准化标量代理
- 直观上，深度 3 × 3 滤波器充当可学习的局部类微分算子库，在特征场改变方向时强烈激活。
- 高响应表明相对曲率显着性而不是原始激活幅度。 如图 3 所示，X~c 突出了 LCGA 旨在保留的脊线/曲线模式，同时保持轻量级和端到端学习。
- ! 与手工算子区别
- 与手工制作的曲率算子（例如固定且对比例敏感的 LoG/拉普拉斯算子）不同，CGA 中的曲率引导是从 SR 监督中端到端学习的，从而使代理能够适应 RS 图像统计数据。

# CCGA

## 曲率感知令牌选择

- 用LCGA代理生成$\hat{X_{c}}\in R^{W*H*1}$,同时有一个保留门生成标量可靠性图$X_{gs}\in[0,1]$,可靠性图有助于抑制不稳定的激活并防止噪声干扰,两个x最终合为score

$$
Score=\frac{1}{2}(|\hat{{X_{c}}}|+X_{gs})
$$

- 随后使用大小感知的Ktokens策略来确定tokens预算k

$$
\text{time\_level} = \max\left(\text{int}\left(\log\frac{H}{r} \right),\; \text{int}\left(\log\frac{W}{r}\right)\right)
$$

- 其中,r表示reduction factor(论文未给出,反推得r=2)
- 得到路由密度$\rho$

$$
\rho=4^{\text{time\_level}}
$$

- 根据路由密度,我们可以得到token预算k(选中的token数量)

$$
k = \left\lfloor \frac{H}{\rho} \right\rfloor \times \left\lfloor \frac{W}{\rho} \right\rfloor
$$

![[屏幕截图 2026-05-02 161150.png]]

## 混合交叉注意力



1. Query 投影：对所有 $N$ 个位置，用线性投影将通道压缩至 $c = C/2$，得到 $Q \in \mathbb{R}^{N \times c}$
2. Key/Value 投影：对选中的 $k$ 个 token，先用 $1 \times 1$ 卷积压缩通道，再分别线性投影：
   - $K \in \mathbb{R}^{k \times c}$（内积维度 $c$，降低计算成本）
   - $V \in \mathbb{R}^{k \times C}$（保持全通道 $C$ 以保留合成能力）
3. Value 门控：用选择置信度调制 value：$\tilde{V} = V \odot S_k$，$S_k$是第一阶段的score弱选中的 token 对聚合贡献更小
4. 标准 logits：$L_{Std} = \frac{Q \otimes K^T}{\sqrt{c}}$
5. 曲率调制 logits：$L_{Curv} = L_{Std} \odot (1 + \alpha \odot \tilde{X}_k)$，沿 key 维度广播
6. 双路 Softmax：
   - $A_{Std} = \text{Softmax}(L_{Std})$
   - $A_{Curv} = \text{Softmax}(L_{Curv})$
7. 凸混合：$A_{Blend} = \beta \odot A_{Std} + (1 - \beta) \odot A_{Curv}$，$\beta \in [0,1]$ 为可学习参数
8. 最终聚合：$Output = A_{Blend} \otimes \tilde{V}$，将结果 reshape 回原特征图布局

### gather


Gather 是一个索引取值操作。给定一个源张量和一组索引，它按索引位置从源张量中提取对应的元素/行。
用 PyTorch 的 torch.gather 来理解最直观。假设 Score 是 4×4 的图像，$\tilde{S}_k = \{5, 7, 10, 14\}$ 是 Top-4 的索引（展平后的位置）：
```python
# Score: 4x4
# 索引 0  1  2  3
# 4  5  6  7
# 8  9  10 11
# 12 13 14 15
indices = torch.tensor([5, 7, 10, 14])
Xk = Score.flatten()[indices]  # 取出第5,7,10,14个位置的值
```
在 CGTA 中，$\tilde{S}_k$ 是 $k$ 个索引值（展平的空间位置），对三个不同的源张量分别 gather：

- $X_k$ = Gather($X_{in}$, $\tilde{S}_k$) — 按空间索引取特征向量（$C$ 维）
- $\tilde{X}_k$ = Gather($|\tilde{X}_c|$, $\tilde{S}_k$) — 按空间索引取曲率显著性标量
- $S_k$ = Gather(Score, $\tilde{S}_k$) — 按空间索引取综合评分标量
  三个 gather 共用同一组索引 $\tilde{S}_k$，只是从不同源张量的相同空间位置取出对应的值。论文中用 $H_{\text{Gather}}(\cdot)$ 表示这个操作。
