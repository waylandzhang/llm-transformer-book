# Transformer 架构：从直觉到实现
## 完整目录

---

### 前言
- [前言 - 为什么写这本书](./前言.md)

---

## Part 1：建立直觉
> *用最通俗的语言理解 GPT 和 Transformer 是什么*

| 章节 | 标题 | 核心内容 |
|------|------|----------|
| 第 1 章 | [GPT是什么 - LLM发展简史与核心思想](./Part1-建立直觉/第1章：GPT是什么%20-%20LLM发展简史与核心思想.md) | 2015-2024 发展历程、训练成本、核心人物 |
| 第 2 章 | [大模型的本质 - 就是两个文件](./Part1-建立直觉/第2章：大模型的本质%20-%20就是两个文件.md) | parameters.bin + run.c、史努比铺铁轨类比 |
| 第 3 章 | [Transformer全景图](./Part1-建立直觉/第3章：Transformer全景图.md) | 架构总览、信息流、各组件作用 |

---

## Part 2：核心组件
> *拆解 Transformer 的基础积木*

| 章节 | 标题 | 核心内容 |
|------|------|----------|
| 第 4 章 | [Tokenization - 文字如何变成数字](./Part2-核心组件/第4章：Tokenization%20-%20文字如何变成数字.md) | BPE算法、词表构建、subword 分词 |
| 第 5 章 | [Positional Encoding - 给文字加位置](./Part2-核心组件/第5章：Positional%20Encoding%20-%20给文字加位置.md) | 正弦余弦编码、为什么用三角函数 |
| 第 6 章 | [LayerNorm与Softmax - 数字的缩放与概率化](./Part2-核心组件/第6章：LayerNorm与Softmax%20-%20数字的缩放与概率化.md) | 归一化直觉、Softmax 温度、数值稳定性 |
| 第 7 章 | [神经网络层 - 不需要懂也能理解Transformer](./Part2-核心组件/第7章：神经网络层%20-%20不需要懂也能理解Transformer.md) | FFN 的作用、知识存储、激活函数 |

---

## Part 3：Attention 机制
> *本书核心，彻底搞懂注意力*

| 章节 | 标题 | 核心内容 |
|------|------|----------|
| 第 8 章 | [线性变换的几何意义 - 矩阵乘法的本质](./Part3-Attention机制/第8章：线性变换的几何意义%20-%20矩阵乘法的本质.md) | 旋转、缩放、投影的几何直觉 |
| 第 9 章 | [Attention的几何逻辑 - 为什么是点积](./Part3-Attention机制/第9章：Attention的几何逻辑%20-%20为什么是点积.md) | 相似度测量、点积 vs 余弦、缩放因子 |
| 第 10 章 | [QKV到底是什么 - Attention的三个主角](./Part3-Attention机制/第10章：QKV到底是什么%20-%20Attention的三个主角.md) | Query/Key/Value 的直觉解释 |
| 第 11 章 | [Multi-Head Attention - 多视角理解](./Part3-Attention机制/第11章：Multi-Head%20Attention%20-%20多视角理解.md) | 多头的意义、head 数量、concat + 投影 |
| 第 12 章 | [QKV输出的本质](./Part3-Attention机制/第12章：QKV输出的本质.md) | Attention 输出的几何意义、信息聚合 |

---

## Part 4：完整架构
> *把所有组件串起来*

| 章节 | 标题 | 核心内容 |
|------|------|----------|
| 第 13 章 | [残差连接与Dropout - 训练稳定的秘密](./Part4-完整架构/第13章：残差连接与Dropout%20-%20训练稳定的秘密.md) | 梯度流动、skip connection、正则化 |
| 第 14 章 | [词嵌入与位置信息的深层逻辑 - 为什么相加而不是拼接](./Part4-完整架构/第14章：词嵌入与位置信息的深层逻辑%20-%20为什么相加而不是拼接.md) | 加法 vs 拼接、维度利用率 |
| 第 15 章 | [Transformer完整前向传播 - 从输入到输出](./Part4-完整架构/第15章：Transformer完整前向传播%20-%20从输入到输出.md) | 完整数据流、参数量计算、维度变化 |
| 第 16 章 | [训练与推理的异同 - 为什么推理要一个字一个字生成](./Part4-完整架构/第16章：训练与推理的异同%20-%20为什么推理要一个字一个字生成.md) | Teacher forcing、自回归生成、并行 vs 串行 |
| 第 17 章 | [学习率的理解 - 训练稳定的关键](./Part4-完整架构/第17章：学习率的理解%20-%20训练稳定的关键.md) | Warmup、Cosine decay、Adam 优化器 |

---

## Part 5：代码实现
> *从零手写，不调库*

| 章节 | 标题 | 核心内容 |
|------|------|----------|
| 第 18 章 | [手写Model.py - 模型定义](./Part5-代码实现/第18章：手写Model.py%20-%20模型定义.md) | 完整模型代码、逐行解释、PyTorch 实现 |
| 第 19 章 | [手写Train.py - 训练循环](./Part5-代码实现/第19章：手写Train.py%20-%20训练循环.md) | 训练循环、损失函数、梯度更新 |
| 第 20 章 | [手写Inference.py - 推理逻辑](./Part5-代码实现/第20章：手写Inference.py%20-%20推理逻辑.md) | 自回归生成、采样策略、代码实现 |

---

## Part 6：生产优化
> *真正跑起来需要什么*

| 章节 | 标题 | 核心内容 |
|------|------|----------|
| 第 21 章 | [Flash Attention - 内存优化原理](./Part6-生产优化/第21章：Flash%20Attention%20-%20内存优化原理.md) | IO-aware、分块计算、在线 Softmax |
| 第 22 章 | [KV Cache - 推理加速](./Part6-生产优化/第22章：KV%20Cache%20-%20推理加速.md) | 缓存机制、内存占用计算、实现细节 |

---

## Part 7：架构变体
> *Transformer 的进化*

| 章节 | 标题 | 核心内容 |
|------|------|----------|
| 第 23 章 | [MHA到MQA到GQA演进](./Part7-架构变体/第23章：MHA到MQA到GQA演进.md) | Multi-Query、Grouped-Query、KV 压缩 |
| 第 24 章 | [Sparse与Infinite Attention](./Part7-架构变体/第24章：Sparse与Infinite%20Attention.md) | 稀疏注意力模式、长序列处理、压缩记忆 |
| 第 25 章 | [位置编码演进 - Sinusoidal到RoPE到ALiBi](./Part7-架构变体/第25章：位置编码演进%20-%20Sinusoidal到RoPE到ALiBi.md) | 相对位置、旋转编码、外推能力 |

---

## Part 8：部署与微调
> *落地实战*

| 章节 | 标题 | 核心内容 |
|------|------|----------|
| 第 26 章 | [LoRA与QLoRA - 高效微调](./Part8-部署与微调/第26章：LoRA与QLoRA%20-%20高效微调.md) | 低秩分解、NF4 量化、PEFT 框架 |
| 第 27 章 | [模型量化 - GPTQ, AWQ, GGUF](./Part8-部署与微调/第27章：模型量化%20-%20GPTQ,%20AWQ,%20GGUF.md) | 量化原理、各格式对比、实战代码 |

---

## Part 9：前沿进展（2024-2025）
> *从 ChatGPT 到 o1/R1：理解大模型最新突破*

| 章节 | 标题 | 核心内容 |
|------|------|----------|
| 第 28 章 | [Prompt Engineering - 提示工程实战](./Part9-前沿进展/第28章：Prompt%20Engineering%20-%20提示工程实战.md) | CoT、Few-shot、Self-Consistency、ToT |
| 第 29 章 | [RLHF与偏好学习 - 让模型对齐人类](./Part9-前沿进展/第29章：RLHF与偏好学习%20-%20让模型对齐人类.md) | RLHF 流程、奖励模型、PPO、DPO |
| 第 30 章 | [Mixture of Experts - 稀疏激活的秘密](./Part9-前沿进展/第30章：Mixture%20of%20Experts%20-%20稀疏激活的秘密.md) | MoE 架构、Mixtral、DeepSeek-V3 |
| 第 31 章 | [推理模型革命 - 从o1到R1](./Part9-前沿进展/第31章：推理模型革命%20-%20从o1到R1.md) | Test-Time Compute、o1/o3、DeepSeek-R1、K1.5 |
| 第 32 章 | [后Transformer架构 - Mamba与混合模型](./Part9-前沿进展/第32章：后Transformer架构%20-%20Mamba与混合模型.md) | SSM、Mamba、Jamba 混合架构 |

---

## 附录

| 附录 | 标题 | 核心内容 |
|------|------|----------|
| 附录 A | [Scaling Law与计算量估算](./附录/附录A：Scaling%20Law与计算量估算.md) | N-D-C 幂律、Chinchilla Optimal、成本估算 |
| 附录 B | [解码策略详解](./附录/附录B：解码策略详解.md) | Greedy、Top-K、Top-P、Beam Search、Temperature |
| 附录 C | [常见问题FAQ](./附录/附录C：常见问题FAQ.md) | 37 个高频问题解答 |

---

## 统计信息

| 指标 | 数值 |
|------|------|
| 总章节数 | 32 章 |
| 附录数 | 3 篇 |
| 预估总字数 | ~208,000 字 |
| Part 数量 | 9 个主题 |

---

## 学习路径建议

### 快速入门（1-2 天）
Part 1 全部 → 第 3 章 → 第 10 章 → 第 15 章

### 深入理解（1 周）
Part 1-4 顺序阅读

### 完整学习（2-3 周）
全书顺序阅读，配合代码实践

### 生产部署（按需）
Part 6-8 + 附录 A

### 前沿追踪（2024-2025）
Part 9：第 28 章（Prompt）→ 第 29 章（RLHF）→ 第 31 章（推理模型）
