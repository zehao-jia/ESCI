# abstract
- 特点:bidirectional(双向的),即通过输入的内容的前文和下文一起进行训练
- 写论文如何写:绝对上数值,相对上比别人高多少
# introduction
- ELMo:双向RNN
- GPT:单向transformer
- MLM:带有掩码的语言模型
- 下一个句子的预测:给两个句子,判断是否相邻
## 贡献
- 1.双向信息的重要性:之前的模型是简单的两个单向模型的合并
- 2.在句子层面和单词层面都进行训练
# 方法论
- 步骤:预训练和微调
- 输入:一个序列,可以是多个句子(方法:world piece)
- 序列一$[cls]$,表示分类,$[sep]$表示断句
- embedding:三层,分别是token的,断句的和位置的
## 预训练
- %15的掩盖,除了关键位置
- 0.8,0.1,0.1的区分my dog is hairy,my dog is [mask],my dog is apple