# 附录 B：解码策略详解

> **一句话总结**：解码策略决定了模型如何从概率分布中选择下一个 token。Greedy 选最高概率，Sampling 随机采样，Top-K/Top-P 限制候选范围，Temperature 调节分布锐度。不同策略适合不同场景。

---

## B.1 为什么需要解码策略？

### B.1.1 从 Logits 到 Token

模型前向传播的输出是一个 **logits 向量**，维度等于词表大小：

```
logits = [2.1, -0.5, 1.3, 0.8, ..., -1.2]  # 长度 = vocab_size
```

每个值代表对应 token 的"得分"。要选出下一个 token，需要：

1. 将 logits 转换为概率分布（通过 softmax）
2. 从概率分布中选择一个 token

**第一步是确定的，第二步有多种选择**——这就是"解码策略"。

### B.1.2 不同策略的权衡

| 策略 | 确定性 | 多样性 | 质量风险 |
|------|--------|--------|---------|
| Greedy | 高 | 低 | 重复、单调 |
| Sampling | 低 | 高 | 可能输出乱码 |
| Top-K | 中 | 中 | 较平衡 |
| Top-P | 中 | 中 | 较平衡 |
| Beam Search | 高 | 低 | 计算成本高 |

---

## B.2 Greedy Decoding（贪心解码）

### B.2.1 原理

最简单的策略：**每一步都选概率最高的 token**。

```python
def greedy_decode(logits):
    probs = softmax(logits)
    return argmax(probs)
```

### B.2.2 例子

```
Prompt: "今天天气"
logits → softmax → [很: 0.4, 不: 0.3, 真: 0.15, ...]
选择: "很"（概率最高）

当前序列: "今天天气很"
logits → softmax → [好: 0.5, 热: 0.2, 冷: 0.15, ...]
选择: "好"

最终输出: "今天天气很好"
```

### B.2.3 优缺点

**优点**：
- 实现简单
- 输出确定（同样输入总是同样输出）
- 计算快

**缺点**：
- **容易重复**：高概率路径可能陷入循环
- **缺乏多样性**：无法生成创意内容
- **可能错过更好的整体序列**：局部最优不等于全局最优

### B.2.4 适用场景

- 事实性问答
- 代码补全
- 需要确定性输出的场景

---

## B.3 Random Sampling（随机采样）

### B.3.1 原理

按照概率分布**随机采样**一个 token。

```python
def random_sample(logits):
    probs = softmax(logits)
    return multinomial(probs, num_samples=1)
```

### B.3.2 例子

```
probs = [很: 0.4, 不: 0.3, 真: 0.15, 太: 0.1, 挺: 0.05]

采样结果可能是：
- "很"（40% 概率）
- "不"（30% 概率）
- "真"（15% 概率）
- ...
```

每次运行可能得到不同结果。

### B.3.3 优缺点

**优点**：
- 多样性高
- 可以生成创意内容
- 避免重复

**缺点**：
- **可能采样到低概率 token**，导致输出不连贯
- 输出不确定，难以复现

---

## B.4 Temperature（温度）

### B.4.1 原理

Temperature 调节概率分布的"锐度"：

```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return softmax(scaled_logits)
```

### B.4.2 Temperature 的效果

假设原始 logits = [2.0, 1.0, 0.5]

| Temperature | Softmax 结果（近似） | 特点 |
|-------------|-------------|------|
| 0.1 | [1.00, 0.00, 0.00] | 极端确定 |
| 0.5 | [0.84, 0.11, 0.04] | 较确定 |
| 1.0 | [0.63, 0.23, 0.14] | 原始分布 |
| 2.0 | [0.48, 0.29, 0.23] | 较平缓 |
| 10.0 | [0.36, 0.33, 0.31] | 接近均匀 |

> **注**：数值经 softmax(logits/T) 计算，四舍五入到两位小数。

**直觉理解**：
- T < 1：概率分布变尖锐，高概率更高，低概率更低
- T = 1：保持原始分布
- T > 1：概率分布变平缓，趋向均匀分布

### B.4.3 Temperature 选择

| Temperature | 效果 | 适用场景 |
|-------------|------|---------|
| 0.1 - 0.3 | 非常确定 | 事实问答、代码 |
| 0.5 - 0.7 | 较确定 | 通用对话 |
| 0.8 - 1.0 | 平衡 | 创意写作 |
| 1.0 - 1.5 | 较随机 | 头脑风暴 |
| > 1.5 | 非常随机 | 实验性（可能乱码）|

### B.4.4 Temperature = 0

当 T → 0 时，softmax 退化为 argmax，等价于 Greedy Decoding。

很多 API 中 `temperature=0` 就是 Greedy。

---

## B.5 Top-K Sampling

### B.5.1 原理

只保留概率最高的 K 个 token，在它们之间采样：

```python
def top_k_sample(logits, k, temperature=1.0):
    # 1. 找到 top-k 的索引
    top_k_logits, top_k_indices = topk(logits, k)

    # 2. 对 top-k 重新计算概率（归一化）
    top_k_probs = softmax(top_k_logits / temperature)

    # 3. 采样
    sampled_index = multinomial(top_k_probs, 1)

    return top_k_indices[sampled_index]
```

### B.5.2 例子

```
原始分布: [A: 0.4, B: 0.3, C: 0.15, D: 0.08, E: 0.05, F: 0.02]

Top-K=3:
候选: [A: 0.4, B: 0.3, C: 0.15]
归一化: [A: 0.47, B: 0.35, C: 0.18]

只在 A、B、C 之间采样，排除了 D、E、F
```

### B.5.3 K 值的选择

| K 值 | 效果 |
|------|------|
| 1 | 等于 Greedy |
| 10-50 | 常用范围 |
| 100+ | 接近 Random Sampling |

**经验值**：K=50 是常用的默认值。

### B.5.4 Top-K 的问题

K 是固定的，但不同位置的概率分布差异很大：

**情况 1**：分布很确定
```
[A: 0.95, B: 0.03, C: 0.01, D: 0.005, ...]
K=50 会包含很多几乎不可能的 token
```

**情况 2**：分布很平缓
```
[A: 0.1, B: 0.09, C: 0.08, D: 0.08, E: 0.07, ...]
K=50 可能还不够，排除了合理的选项
```

这个问题催生了 Top-P。

---

## B.6 Top-P (Nucleus) Sampling

### B.6.1 原理

Top-P 选择**累积概率**达到阈值 P 的最小 token 集合：

```python
def top_p_sample(logits, p, temperature=1.0):
    # 1. 计算概率并排序
    probs = softmax(logits / temperature)
    sorted_probs, sorted_indices = sort(probs, descending=True)

    # 2. 计算累积概率
    cumulative_probs = cumsum(sorted_probs)

    # 3. 找到累积概率 >= p 的位置
    cutoff_index = first_index_where(cumulative_probs >= p)

    # 4. 保留前 cutoff_index 个 token
    nucleus_probs = sorted_probs[:cutoff_index + 1]
    nucleus_indices = sorted_indices[:cutoff_index + 1]

    # 5. 归一化并采样
    nucleus_probs = nucleus_probs / sum(nucleus_probs)
    sampled_index = multinomial(nucleus_probs, 1)

    return nucleus_indices[sampled_index]
```

### B.6.2 例子

```
排序后分布: [A: 0.4, B: 0.3, C: 0.15, D: 0.08, E: 0.05, F: 0.02]
累积概率:   [0.4,   0.7,   0.85,   0.93,   0.98,   1.0]

Top-P=0.9:
保留 A, B, C, D（累积概率 0.93 >= 0.9）
候选集大小 = 4
```

### B.6.3 Top-P 的优势

Top-P **自适应**调整候选集大小：

**确定分布**：
```
[A: 0.95, B: 0.03, ...]
Top-P=0.9 → 只保留 A
```

**平缓分布**：
```
[A: 0.1, B: 0.09, C: 0.08, ...]
Top-P=0.9 → 保留约 15 个 token
```

### B.6.4 P 值的选择

| P 值 | 效果 |
|------|------|
| 0.1 - 0.5 | 较确定 |
| 0.8 - 0.95 | 常用范围 |
| 1.0 | 等于 Random Sampling |

**经验值**：P=0.9 或 P=0.95 是常用的默认值。

---

## B.7 Beam Search（束搜索）

### B.7.1 原理

Beam Search 同时维护 **B 个最优候选序列**：

```
beam_width = 3

Step 0: [""]
Step 1: ["今", "我", "你"]  # 保留 top-3
Step 2: ["今天", "今日", "我们"]  # 每个扩展，保留总体 top-3
...
```

### B.7.2 算法流程

```python
def beam_search(model, prompt, beam_width, max_length):
    # 初始化：一个候选序列
    beams = [(prompt, 0.0)]  # (序列, 累积 log 概率)

    for step in range(max_length):
        all_candidates = []

        for seq, score in beams:
            # 获取下一个 token 的概率
            logits = model(seq)
            log_probs = log_softmax(logits)

            # 扩展每个候选
            for token_id in range(vocab_size):
                new_seq = seq + token_id
                new_score = score + log_probs[token_id]
                all_candidates.append((new_seq, new_score))

        # 保留 top-B 个候选
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]

    # 返回得分最高的序列
    return beams[0][0]
```

### B.7.3 Beam Search 的特点

**优点**：
- 考虑更长的上下文依赖
- 输出质量通常更高（尤其是翻译任务）
- 可以生成多个候选答案

**缺点**：
- **计算成本高**：每步计算量是 Greedy 的 B 倍
- **输出单调**：倾向于选择"安全"的输出
- **长度偏好**：倾向于生成较短的序列（需要长度惩罚）

### B.7.4 长度归一化

为了避免 Beam Search 偏好短序列，通常加入长度惩罚：

```
score_normalized = score / (length ^ alpha)
```

alpha 通常取 0.6 - 1.0。

### B.7.5 适用场景

- 机器翻译
- 摘要生成
- 需要高质量、确定性输出的场景

**注意**：现代 LLM（ChatGPT 等）通常**不用** Beam Search，而用 Sampling + Top-P/K。

---

## B.8 组合策略

### B.8.1 常见组合

实践中通常组合使用多种策略：

```python
# 常见组合 1：Top-P + Temperature
def decode(logits, p=0.9, temperature=0.7):
    scaled_logits = logits / temperature
    return top_p_sample(scaled_logits, p)

# 常见组合 2：Top-K + Top-P
def decode(logits, k=50, p=0.9):
    # 先 Top-K
    top_k_logits = top_k_filter(logits, k)
    # 再 Top-P
    return top_p_sample(top_k_logits, p)
```

### B.8.2 各大模型的默认配置

| 模型/API | 默认配置 |
|---------|---------|
| OpenAI GPT | temperature=1.0, top_p=1.0 |
| Claude | temperature=1.0 |
| LLaMA | temperature=0.6, top_p=0.9 |
| Mistral | temperature=0.7, top_k=50 |

### B.8.3 场景推荐

| 场景 | 推荐配置 |
|------|---------|
| 代码生成 | temperature=0.2, top_p=0.95 |
| 事实问答 | temperature=0, 或 temperature=0.3 |
| 创意写作 | temperature=0.8-1.0, top_p=0.9 |
| 对话聊天 | temperature=0.7, top_p=0.9 |
| 翻译 | beam_search, beam_width=4 |

---

## B.9 重复惩罚

### B.9.1 问题：生成重复内容

LLM 容易陷入重复：

```
输出："我喜欢吃苹果。我喜欢吃苹果。我喜欢吃苹果。..."
```

### B.9.2 解决方案

**1. Repetition Penalty**

降低已生成 token 的概率：

```python
def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
    for token in set(generated_tokens):
        if logits[token] > 0:
            logits[token] /= penalty
        else:
            logits[token] *= penalty
    return logits
```

**2. Presence Penalty**

对出现过的 token 施加固定惩罚：

```
logits[token] -= presence_penalty  # 如果 token 出现过
```

**3. Frequency Penalty**

惩罚与出现次数成正比：

```
logits[token] -= frequency_penalty * count[token]
```

### B.9.3 参数选择

| 参数 | 范围 | 效果 |
|------|------|------|
| repetition_penalty | 1.0-1.5 | 1.0=不惩罚，1.2=常用 |
| presence_penalty | 0-2.0 | 0=不惩罚，0.6=常用 |
| frequency_penalty | 0-2.0 | 0=不惩罚，0.5=常用 |

---

## B.10 本附录要点

1. **Greedy**：选最高概率，确定但单调

2. **Sampling**：随机采样，多样但可能乱码

3. **Temperature**：调节分布锐度，T<1 更确定，T>1 更随机

4. **Top-K**：限制候选数量，K=50 常用

5. **Top-P**：限制累积概率，自适应候选数量，P=0.9 常用

6. **Beam Search**：维护多个候选，质量高但计算贵

7. **组合使用**：Top-P + Temperature 是现代 LLM 的主流

8. **重复惩罚**：避免生成重复内容

---

## 延伸阅读

- **The Curious Case of Neural Text Degeneration** (Holtzman et al., 2019) - Top-P 的原始论文
- **Hierarchical Neural Story Generation** (Fan et al., 2018) - Top-K 的应用
- **CTRL: A Conditional Transformer Language Model** (Keskar et al., 2019) - 关于 repetition penalty
