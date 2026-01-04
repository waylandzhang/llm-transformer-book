# 附录 A：Scaling Law 与计算量估算

> **一句话总结**：Scaling Law 揭示了一个惊人的规律——模型性能与参数量、数据量、计算量之间存在可预测的幂律关系。掌握这些公式，你就能估算训练一个大模型需要多少钱。

---

## A.1 什么是 Scaling Law？

### A.1.1 一个反直觉的发现

2020 年，OpenAI 发表了著名的 Scaling Law 论文，揭示了一个令人惊讶的规律：

> **语言模型的性能（用困惑度/loss 衡量）与三个因素呈幂律关系**：
> - 模型参数量 N
> - 训练数据量 D
> - 计算量 C

这意味着：**只要你有足够的资源，模型性能是可以预测的**。

### A.1.2 幂律关系

Scaling Law 的核心公式：

```
L(N) = (N_c / N)^α_N    # 参数量 scaling
L(D) = (D_c / D)^α_D    # 数据量 scaling
L(C) = (C_c / C)^α_C    # 计算量 scaling
```

其中：
- L 是模型的 loss（越低越好）
- N, D, C 分别是参数量、数据量、计算量
- N_c, D_c, C_c 是常数
- α 是幂律指数

OpenAI 的实验发现：
- α_N ≈ 0.076
- α_D ≈ 0.095
- α_C ≈ 0.050

### A.1.3 图解 Scaling Law

```
Loss
 ↑
 │  ╲
 │   ╲
 │    ╲
 │     ╲____
 │          ╲____
 │               ╲____
 └─────────────────────→ log(参数量/数据量/计算量)
```

在双对数坐标下，这些关系近似为直线。

---

## A.2 参数量估算

### A.2.1 Transformer 参数量公式

一个标准的 Transformer 模型，主要参数来自：

**1. Embedding 层**
```
词嵌入：vocab_size × d_model
位置嵌入：max_seq_len × d_model（如果是学习式）
```

**2. 每个 Transformer Block**
```
Multi-Head Attention:
- W_Q: d_model × d_model
- W_K: d_model × d_model
- W_V: d_model × d_model
- W_O: d_model × d_model
小计：4 × d_model²

Feed-Forward Network:
- W_1: d_model × d_ff (通常 d_ff = 4 × d_model)
- W_2: d_ff × d_model
小计：2 × d_model × d_ff = 8 × d_model²

LayerNorm (×2):
- γ, β: 2 × 2 × d_model
小计：4 × d_model（可忽略）

每个 Block 总计：≈ 12 × d_model²
```

**3. 输出层**
```
LM Head：d_model × vocab_size
（通常与词嵌入共享权重）
```

### A.2.2 简化估算公式

对于 L 层、d_model 维度的 Transformer：

```
N ≈ 12 × L × d_model²
```

**验证**：
- GPT-3 175B：L=96, d_model=12288
- 估算：12 × 96 × 12288² ≈ 173B ✓

### A.2.3 常见模型参数量

| 模型 | 层数 L | 维度 d_model | 参数量 |
|------|--------|-------------|--------|
| GPT-2 Small | 12 | 768 | 117M |
| GPT-2 Medium | 24 | 1024 | 345M |
| GPT-2 Large | 36 | 1280 | 762M |
| GPT-2 XL | 48 | 1600 | 1.5B |
| LLaMA-7B | 32 | 4096 | 6.7B |
| LLaMA-13B | 40 | 5120 | 13B |
| LLaMA-70B | 80 | 8192 | 70B |
| GPT-3 | 96 | 12288 | 175B |

---

## A.3 计算量估算

### A.3.1 FLOPs 是什么？

**FLOPs**（Floating Point Operations）= 浮点运算次数

- 1 FLOP = 一次加法或乘法
- 矩阵乘法 A(m×n) @ B(n×p) 需要 2mnp FLOPs

### A.3.2 训练计算量公式

训练一个模型的总计算量：

```
C ≈ 6 × N × D
```

其中：
- N = 模型参数量
- D = 训练 token 数
- 6 = 前向 + 反向传播的系数（前向 2N，反向 4N）

**例子**：训练 LLaMA-7B（1T tokens）
```
C = 6 × 7B × 1T = 42 × 10²¹ FLOPs = 42 ZFLOPs
```

### A.3.3 推理计算量

推理时只需要前向传播：

```
C_inference ≈ 2 × N × tokens_generated
```

**例子**：LLaMA-7B 生成 100 tokens
```
C = 2 × 7B × 100 = 1.4 × 10¹² FLOPs = 1.4 TFLOPs
```

### A.3.4 GPU 算力换算

常见 GPU 的算力（Tensor Core FP16）：

| GPU | FP16 Tensor 算力 | FP32 算力 | 显存 |
|-----|-----------------|----------|------|
| RTX 3090 | 142 TFLOPS | 35 TFLOPS | 24GB |
| RTX 4090 | 330 TFLOPS | 83 TFLOPS | 24GB |
| A100 40GB | 312 TFLOPS | 156 TFLOPS | 40GB |
| A100 80GB | 312 TFLOPS | 156 TFLOPS | 80GB |
| H100 | 989 TFLOPS | 495 TFLOPS | 80GB |

> **注意**：Tensor Core FP16 是混合精度训练的实际吞吐量，FP32 仅供参考。

**训练时间估算**：

```
训练时间 = C / (GPU算力 × GPU数量 × 利用率)
```

利用率通常为 30%-50%（受通信、数据加载等影响）。

---

## A.4 训练成本估算

### A.4.1 GPU 小时

```
GPU 小时 = 训练时间(小时) × GPU 数量
```

### A.4.2 云服务价格

AWS/Azure/GCP 的 GPU 实例价格（2024）：

| GPU | 按需价格 ($/小时) | Spot 价格 ($/小时) |
|-----|-----------------|------------------|
| A100 40GB | ~$3.0 | ~$1.0 |
| A100 80GB | ~$4.0 | ~$1.3 |
| H100 | ~$5.0 | ~$2.0 |

### A.4.3 实际案例

**训练 LLaMA-7B（假设 1T tokens）**

1. 计算量：C = 6 × 7B × 1T = 42 ZFLOPs

2. 使用 1000 块 A100（80GB）：
   - 算力：312 × 1000 = 312 PFLOPS
   - 利用率 40%：实际 125 PFLOPS
   - 时间：42 × 10²¹ / (125 × 10¹⁵) = 336,000 秒 ≈ 93 小时

3. 成本（Spot 价格）：
   - GPU 小时：93 × 1000 = 93,000
   - 费用：93,000 × $1.3 ≈ **$120,000**

**训练 GPT-3 175B（300B tokens）**

1. 计算量：C = 6 × 175B × 300B = 315 ZFLOPs

2. 使用 10000 块 V100：
   - 时间：约 34 天
   - 成本：约 **$4.6M**（OpenAI 2020 年估计）

### A.4.4 Chinchilla Optimal

2022 年，DeepMind 的 Chinchilla 论文修正了 Scaling Law：

> **最优训练**：参数量和数据量应该同等 scaling

Chinchilla Optimal 公式：

```
D_optimal ≈ 20 × N
```

即：**训练 token 数应该是参数量的 20 倍**。

| 参数量 | Chinchilla Optimal 数据量 |
|--------|-------------------------|
| 1B | 20B tokens |
| 7B | 140B tokens |
| 70B | 1.4T tokens |
| 175B | 3.5T tokens |

GPT-3 训练了 300B tokens，按 Chinchilla Optimal 标准是**欠训练**的。

---

## A.5 Scaling Law 的实践意义

### A.5.1 资源规划

给定预算，如何分配资源？

**传统观点**：模型越大越好
**Chinchilla 观点**：模型和数据要平衡

```python
# 伪代码：资源规划
def plan_training(budget_flops):
    # Chinchilla Optimal
    N = (budget_flops / 6 / 20) ** 0.5  # 参数量
    D = 20 * N                           # 数据量

    return N, D
```

### A.5.2 预测模型性能

在决定训练之前，可以预测最终性能：

```
L(C) ≈ 1.69 × C^(-0.048)  # OpenAI 的经验公式
```

### A.5.3 小模型 vs 大模型

| 策略 | 优点 | 缺点 |
|------|------|------|
| 小模型多数据 | 训练快，推理便宜 | 能力上限低 |
| 大模型少数据 | 能力上限高 | 训练贵，欠训练 |
| Chinchilla Optimal | 效率最优 | 需要大量数据 |

---

## A.6 计算量速查表

### A.6.1 常见操作的 FLOPs

| 操作 | FLOPs |
|------|-------|
| 矩阵乘法 A(m×n) @ B(n×p) | 2mnp |
| 向量点积 (长度 n) | 2n |
| Softmax (长度 n) | ~5n |
| LayerNorm (维度 d) | ~5d |
| GELU 激活 | ~10 per element |

### A.6.2 Transformer Block FLOPs

对于序列长度 s、维度 d、FFN 维度 4d 的一个 Block：

```
Attention QKV 投影：6sd²
Attention 矩阵乘法：4s²d
FFN：16sd²
总计：≈ 24sd² + 4s²d
```

当 s << d 时（短序列），FFN 主导。
当 s >> d 时（长序列），Attention 主导。

---

## A.7 本附录要点

1. **Scaling Law**：性能与 N、D、C 呈幂律关系

2. **参数量估算**：N ≈ 12 × L × d_model²

3. **计算量估算**：C_train ≈ 6ND，C_inference ≈ 2N × tokens

4. **Chinchilla Optimal**：D_optimal ≈ 20N

5. **成本估算**：
   - GPU 小时 = C / (算力 × 利用率)
   - 费用 = GPU 小时 × 单价

---

## 延伸阅读

- **Scaling Laws for Neural Language Models** (OpenAI, 2020)
- **Training Compute-Optimal Large Language Models** (Chinchilla, DeepMind, 2022)
- **Scaling Laws for Autoregressive Generative Modeling** (OpenAI, 2020)
