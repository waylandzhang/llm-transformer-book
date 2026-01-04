# 附录 C：常见问题 FAQ

> **本附录收集了读者在学习 Transformer 过程中最常遇到的问题和困惑，提供简明扼要的解答。**

---

## C.1 基础概念

### Q1: Transformer 和 GPT 有什么关系？

**答**：GPT（Generative Pre-trained Transformer）是基于 Transformer 架构的一种具体模型。

- **Transformer** 是一种通用的神经网络架构，2017 年由 Google 提出
- **GPT** 是 OpenAI 用 Transformer 的**解码器部分**构建的语言模型

类比：Transformer 是"发动机设计图"，GPT 是"用这个发动机造的某款车"。

### Q2: 为什么 Transformer 比 RNN 好？

**答**：主要三点优势：

1. **并行计算**：RNN 必须按顺序处理，Transformer 可以并行处理所有位置
2. **长距离依赖**：RNN 的信息需要逐步传递，容易丢失；Transformer 通过 Attention 直接连接任意位置
3. **训练效率**：并行化使得 Transformer 训练速度快很多

### Q3: Encoder 和 Decoder 有什么区别？

**答**：

| | Encoder | Decoder |
|---|---------|---------|
| **Attention** | 双向（看全部） | 单向（只看前面） |
| **用途** | 理解输入 | 生成输出 |
| **代表模型** | BERT | GPT |
| **典型任务** | 分类、NER | 文本生成 |

GPT 系列只用 Decoder，因为生成任务只能看到已生成的内容。

### Q4: 为什么 GPT 用 Decoder-only，不用 Encoder？

**答**：

1. **任务特性**：文本生成是自回归的，只能依赖已生成的 token
2. **简单有效**：实践证明 Decoder-only 在 scaling up 时表现更好
3. **统一架构**：Encoder-Decoder 需要处理输入输出对齐，Decoder-only 更简洁

---

## C.2 Attention 机制

### Q5: 为什么 Attention 用点积而不是其他相似度？

**答**：

1. **计算高效**：点积可以用矩阵乘法批量计算，GPU 友好
2. **数学性质好**：点积就是向量夹角的余弦乘以长度，直接反映相似度
3. **可学习**：通过 Q、K 投影矩阵，可以学习到合适的相似度度量

其他相似度（如 L2 距离）也有人尝试，但点积仍是主流。

### Q6: Q、K、V 分别代表什么意义？

**答**：

- **Q（Query）**：当前位置的"提问"——"我在找什么信息？"
- **K（Key）**：其他位置的"标签"——"我有什么信息？"
- **V（Value）**：其他位置的"内容"——"如果你需要，这是我的信息"

流程：Q 和 K 计算相似度，用相似度作为权重对 V 加权求和。

### Q7: 为什么要除以 sqrt(d_k)？

**答**：防止点积结果过大导致 softmax 梯度消失。

当维度 d 很大时，点积的方差会变成 d（假设 Q、K 的元素是标准正态分布）。除以 sqrt(d) 可以把方差稳定在 1，让 softmax 的输入在合理范围内。

### Q8: Multi-Head Attention 为什么有效？

**答**：

1. **多视角**：不同的 head 可以学习不同类型的关系（语法、语义、位置等）
2. **低秩近似**：多个小矩阵可以近似一个大矩阵，参数更高效
3. **稳定性**：多个 head 的结果平均，减少单一 head 的噪声

### Q9: Head 数量怎么选？

**答**：

常见配置：head_dim = 64 或 128，head_num = d_model / head_dim

| 模型 | d_model | num_heads | head_dim |
|------|---------|-----------|----------|
| GPT-2 Small | 768 | 12 | 64 |
| LLaMA-7B | 4096 | 32 | 128 |
| GPT-3 | 12288 | 96 | 128 |

经验上，head_dim = 64-128 效果较好。

---

## C.3 训练相关

### Q10: 为什么用 Cross-Entropy Loss？

**答**：

Cross-Entropy Loss 衡量预测概率分布与真实分布的差距：

```
Loss = -log(P(正确token))
```

本质上是让模型给正确 token 更高的概率。这是分类问题（从词表中选一个词）的标准做法。

### Q11: 为什么需要 Learning Rate Warmup？

**答**：

训练初期，模型参数是随机的，梯度可能很大且方向不稳定。直接用大学习率容易发散。

Warmup 让学习率从小到大逐渐增加，给模型一个"热身"过程，等梯度稳定后再用正常学习率。

### Q12: 为什么训练时用 Teacher Forcing？

**答**：

Teacher Forcing = 训练时给模型看正确答案，而不是模型自己的预测。

好处：
- 训练稳定：错误不会累积
- 并行计算：可以一次性计算所有位置的损失

坏处：
- 训练和推理不一致（exposure bias）

实践中好处远大于坏处。

### Q13: Batch Size 应该选多大？

**答**：

越大越好（在内存允许范围内），但需要配合调整学习率。

经验规则：
- 批量越大，学习率可以越大
- 线性缩放：batch size 翻倍，学习率也翻倍
- 太大可能需要更多 warmup steps

常见配置：
- 小模型（<1B）：batch_size = 512-2048
- 大模型（>1B）：batch_size = 1M-4M tokens（通过梯度累积）

---

## C.4 推理相关

### Q14: 为什么推理要一个 token 一个 token 生成？

**答**：

因为语言模型是**自回归**的：下一个 token 的概率依赖于所有之前的 token。

```
P(x_t | x_1, x_2, ..., x_{t-1})
```

在生成 x_t 之前，必须先知道 x_1 到 x_{t-1}。所以只能一个一个生成。

### Q15: 什么是 KV Cache？

**答**：

KV Cache 缓存已计算的 Key 和 Value，避免重复计算。

没有 KV Cache：生成第 100 个 token 时，要重新算前 99 个 token 的 K、V。

有 KV Cache：前 99 个 token 的 K、V 已经存着了，只算新 token 的。

加速比：从 O(n²) 降到 O(n)。

### Q16: Temperature 怎么影响输出？

**答**：

Temperature 调节概率分布的"锐度"：

- T=0：等于 Greedy，选最高概率
- T<1：分布更尖锐，输出更确定
- T=1：原始分布
- T>1：分布更平缓，输出更随机

代码：`probs = softmax(logits / temperature)`

### Q17: 什么时候用 Beam Search，什么时候用 Sampling？

**答**：

| 方法 | 适用场景 |
|------|---------|
| Beam Search | 翻译、摘要（需要准确性）|
| Sampling + Top-P | 对话、创作（需要多样性）|
| Greedy (T=0) | 代码、数学（需要确定性）|

现代 LLM（ChatGPT、Claude）主要用 Sampling + Top-P。

---

## C.5 架构细节

### Q18: 为什么用 LayerNorm 而不是 BatchNorm？

**答**：

1. **序列长度变化**：BatchNorm 需要固定的统计量，但 NLP 序列长度不固定
2. **小 batch**：大模型训练时 batch size 相对较小，BatchNorm 统计量不稳定
3. **推理一致**：LayerNorm 在训练和推理时行为一致

### Q19: Pre-Norm 和 Post-Norm 有什么区别？

**答**：

```
Post-Norm: x + LayerNorm(SubLayer(x))  # 原始 Transformer
Pre-Norm:  x + SubLayer(LayerNorm(x))  # 现代常用
```

Pre-Norm 更容易训练深层网络，是现在的主流选择。

### Q20: 为什么 FFN 的中间层是 4 倍大？

**答**：

原始论文的设计，经验上效果好。

FFN 的作用是"存储知识"，更大的中间层意味着更大的存储容量。4x 是经验值，有些模型用 8/3 x 或其他比例。

### Q21: Residual Connection 有什么作用？

**答**：

1. **梯度流动**：提供"高速公路"让梯度直接传回浅层
2. **特征保留**：每层可以专注于学习"增量"，而不是全部特征
3. **训练稳定**：让深层网络更容易训练

没有残差连接，100 层的 Transformer 几乎无法训练。

---

## C.6 实践问题

### Q22: 微调时应该冻结哪些层？

**答**：

常见策略：

| 策略 | 冻结 | 训练 | 适用场景 |
|------|------|------|---------|
| 全参数微调 | 无 | 全部 | 资源充足 |
| LoRA | 原始权重 | 低秩增量 | 资源有限 |
| 只训练最后几层 | 底层 | 顶层 | 快速实验 |
| 只训练 LM Head | 全部 Transformer | 输出层 | 简单适配 |

推荐从 LoRA 开始，效果好、成本低。

### Q23: 怎么判断模型是否过拟合？

**答**：

看训练 loss 和验证 loss 的差距：

- **健康**：两者都在下降，差距小
- **过拟合**：训练 loss 下降，验证 loss 上升或停滞
- **欠拟合**：两者都很高，下降缓慢

解决过拟合：增加数据、Dropout、减小模型、早停。

### Q24: 怎么选择模型大小？

**答**：

根据任务复杂度和资源选择：

| 任务 | 推荐参数量 |
|------|-----------|
| 简单分类 | 100M-1B |
| 通用问答 | 7B-13B |
| 复杂推理 | 30B-70B |
| 多模态/代理 | 70B+ |

**Chinchilla 法则**：数据量 ≈ 20 × 参数量（tokens）

### Q25: 怎么处理长文本？

**答**：

| 方法 | 原理 | 适用场景 |
|------|------|---------|
| 截断 | 只用前/后 N tokens | 简单任务 |
| 滑动窗口 | 分块处理 | 长文档 |
| Sparse Attention | 稀疏注意力 | 训练时 |
| RoPE/ALiBi | 可外推位置编码 | 推理时 |

现代模型（Claude 100k+，GPT-4 128k）通过优化支持更长上下文。

---

## C.7 常见错误

### Q26: 为什么模型输出乱码？

**答**：可能原因：

1. **没训练好**：loss 还很高
2. **temperature 太高**：采样到低概率 token
3. **tokenizer 不匹配**：编解码不一致
4. **量化精度太低**：损失太多信息

检查方法：先用 T=0（Greedy）测试，排除采样问题。

### Q27: 为什么模型不断重复？

**答**：可能原因：

1. **temperature 太低**：总是选同一个高概率 token
2. **没有重复惩罚**：增加 repetition_penalty
3. **训练数据有重复**：数据质量问题

解决：增加 temperature，加入 repetition_penalty (1.1-1.2)。

### Q28: 为什么显存不够？

**答**：

显存占用 = 模型参数 + 梯度 + 优化器状态 + 激活值

优化方法：
- 减小 batch size
- 使用梯度检查点（用时间换空间）
- 使用混合精度（FP16/BF16）
- 使用 LoRA（冻结大部分参数）
- 使用量化（4-bit 加载）

---

## C.8 概念辨析

### Q29: Token 和 Word 有什么区别？

**答**：

- **Word**：自然语言的词（用空格分隔）
- **Token**：模型实际处理的单位（由 tokenizer 决定）

例子：
```
Word: "unbelievable"（1 个词）
Token: ["un", "believ", "able"]（3 个 token）
```

中文通常每个字是 1-2 个 token。

### Q30: Embedding 和 Encoding 有什么区别？

**答**：

- **Embedding**：将离散 token 映射到连续向量（查表操作）
- **Encoding**：将输入转换为某种表示（更广义）

在 Transformer 中：
- Token Embedding：token ID → 向量
- Positional Encoding：位置 → 向量
- 最终 Encoding = Token Embedding + Positional Encoding

### Q31: Self-Attention 和 Cross-Attention 有什么区别？

**答**：

| | Self-Attention | Cross-Attention |
|---|---------------|-----------------|
| Q 来源 | 当前序列 | 当前序列 |
| K/V 来源 | 当前序列 | 另一个序列 |
| 用途 | 理解上下文 | 融合两个序列 |
| 例子 | GPT 每一层 | 翻译中 decoder 看 encoder |

GPT 只用 Self-Attention，Encoder-Decoder 模型才用 Cross-Attention。

---

## C.9 进阶问题

### Q32: 为什么大模型能涌现新能力？

**答**：

"涌现"（Emergence）指的是小模型没有、大模型突然出现的能力（如思维链推理）。

可能原因：
1. **量变到质变**：足够多的参数才能学会复杂模式
2. **数据多样性**：大模型训练数据更丰富
3. **评估方式**：某些能力在小模型上表现太差，看不出变化

这仍是活跃研究领域。

### Q33: Prompt Engineering 为什么有效？

**答**：

大模型在预训练时见过各种格式的文本。好的 prompt 可以：

1. **激活相关知识**：让模型"想起"训练时见过的类似内容
2. **设定上下文**：告诉模型用什么"角色"回答
3. **提供示例**：Few-shot 让模型理解任务格式

本质上是利用模型的 in-context learning 能力。

### Q34: RLHF 是怎么工作的？

**答**：

RLHF（Reinforcement Learning from Human Feedback）三步走：

1. **SFT**：用人工标注数据微调基础模型
2. **训练 Reward Model**：让模型学习人类偏好（哪个回答更好）
3. **PPO 强化学习**：用 Reward Model 的打分作为奖励，优化模型

RLHF 让模型更"对齐"人类意图，是 ChatGPT 成功的关键。

---

## C.10 学习建议

### Q35: 学 Transformer 应该先学什么？

**答**：

推荐顺序：
1. Python 基础
2. NumPy 矩阵操作
3. PyTorch 基础
4. 线性代数直觉（不需要太深）
5. 本书/视频

不需要先学 RNN/LSTM，直接学 Transformer 就好。

### Q36: 应该看论文还是看教程？

**答**：

建议顺序：
1. **先看教程**：建立直觉，理解大图景
2. **再看论文**：补充细节，了解设计动机
3. **最后看代码**：真正理解实现

直接看论文容易迷失在细节里，先有直觉再补细节更高效。

### Q37: 怎么跟进最新进展？

**答**：

推荐资源：
- **Twitter/X**：关注 AI 研究者
- **Hugging Face Blog**：新模型/新技术解读
- **arXiv**：原始论文（搜 cs.CL, cs.LG）
- **YouTube**：Yannic Kilcher 等人的论文解读

不必追每一篇论文，关注重要的里程碑即可。

---

## 本附录要点

本 FAQ 覆盖了从基础概念到进阶问题的 37 个常见问题。

**核心建议**：
1. 先建立直觉，再补细节
2. 动手实现是最好的学习方式
3. 不必追求完美理解每个细节
4. 实践中遇到问题再深入研究

如果你的问题没有在这里找到答案，欢迎在视频评论区或 GitHub Issues 提问！
