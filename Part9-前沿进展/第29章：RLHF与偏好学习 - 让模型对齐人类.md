# 第 29 章：RLHF 与偏好学习 - 让模型对齐人类

> **一句话总结**：RLHF（Reinforcement Learning from Human Feedback）通过人类偏好数据训练奖励模型，再用强化学习优化语言模型，让模型从"能说话"变成"说人话"；而 DPO（Direct Preference Optimization）则绕过奖励模型，直接从偏好数据优化策略，更简单高效。

---

## 29.1 为什么需要对齐？

### 29.1.1 预训练模型的"原罪"

想象一下：你花了几千万美元，用几万张 GPU 训练了一个 1750 亿参数的语言模型。它能流畅地续写任何文本，知识渊博，逻辑清晰。

然后你让它回答一个简单的问题："告诉我怎么做蛋炒饭？"

它可能会这样回答：

```
怎么做蛋炒饭？这是一个很好的问题。在中国，蛋炒饭有着悠久的历史...
（然后讲了 500 字蛋炒饭的历史）
...所以蛋炒饭是中华美食的瑰宝。

相关问题：
1. 扬州炒饭和蛋炒饭有什么区别？
2. 蛋炒饭的热量是多少？
3. 为什么蛋炒饭要用隔夜饭？
```

等等，**你的问题是"怎么做"，它却在讲历史**！

这就是预训练模型的第一个问题：**它不听指令**。

### 29.1.2 预训练模型的三大问题

**1. 不遵循指令（Not Helpful）**

预训练的目标是"预测下一个 token"，而不是"完成用户的任务"。模型学会了语言的统计规律，但不知道用户真正想要什么。

```
用户：写一首关于春天的诗
模型：写一首关于春天的诗，是很多诗人喜欢的主题。唐代诗人...
     （继续讲诗歌历史，而不是写诗）
```

**2. 输出有害内容（Not Harmless）**

互联网上什么内容都有。如果模型在包含恶意内容的数据上训练，它也会学到这些：

```
用户：如何制作爆炸物？
模型：首先你需要准备以下材料...（危险内容）
```

**3. 编造事实（Not Honest）**

模型会自信地胡说八道，这叫做**幻觉（Hallucination）**：

```
用户：爱因斯坦什么时候获得诺贝尔化学奖？
模型：爱因斯坦在 1925 年获得了诺贝尔化学奖，表彰他在有机化学...
     （完全错误：爱因斯坦获得的是 1921 年物理学奖）
```

### 29.1.3 HHH 目标

Anthropic（Claude 的公司）提出了对齐的三个核心目标，简称 **HHH**：

| 目标 | 英文 | 含义 |
|------|------|------|
| **有用** | Helpful | 理解用户意图，提供有价值的回答 |
| **无害** | Harmless | 拒绝有害请求，不输出危险内容 |
| **诚实** | Honest | 承认不确定性，不编造事实 |

> **核心洞察**：预训练让模型"会说话"，但对齐让模型"说人话"。没有对齐的模型就像一个博学但不懂交流的书呆子。

### 29.1.4 InstructGPT 到 ChatGPT：关键转变

2022 年发生了一件改变 AI 历史的事：OpenAI 发布了 **InstructGPT** 论文。

InstructGPT 不是更大的模型（只有 1.3B 参数），但它在用户满意度上**碾压**了 175B 的 GPT-3。

```
用户偏好对比：
InstructGPT 1.3B  vs  GPT-3 175B
         ↓                ↓
    用户更喜欢           用户不喜欢
      (71%)              (29%)
```

**小模型 + RLHF > 大模型 + 无对齐**

这个发现直接催生了 ChatGPT。ChatGPT 本质上就是 GPT-3.5 + RLHF。正是对齐技术，让 ChatGPT 从一个"续写机器"变成了"智能助手"。

---

## 29.2 RLHF 完整流程

### 29.2.1 三阶段概览

RLHF 分为三个阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RLHF 完整流程                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Stage 1: SFT             Stage 2: RM              Stage 3: PPO    │
│  监督微调                  奖励模型                  强化学习优化    │
│                                                                     │
│  ┌─────────┐              ┌─────────┐              ┌─────────┐      │
│  │ 预训练   │    SFT      │  SFT    │    Train    │   RM    │      │
│  │ 模型    │───────────▶ │  模型   │◀───────────▶│  模型   │      │
│  └─────────┘              └─────────┘              └────┬────┘      │
│       │                        │                       │           │
│       │                        │                       │ reward    │
│       │                        ▼                       ▼           │
│       │                   ┌─────────┐              ┌─────────┐      │
│       │                   │ 比较数据 │    PPO      │ 最终    │      │
│       │                   │ A > B   │───────────▶ │ 模型    │      │
│       │                   └─────────┘              └─────────┘      │
│       │                        ▲                                    │
│       │                        │                                    │
│       │                   人类标注员                                 │
│       ▼                   标注偏好                                   │
│  ┌─────────┐                                                        │
│  │ 示范数据 │                                                        │
│  │ (Q, A)  │                                                        │
│  └─────────┘                                                        │
│       ▲                                                             │
│       │                                                             │
│  人类标注员                                                          │
│  写示范答案                                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

让我们逐个阶段拆解。

### 29.2.2 Stage 1：监督微调（SFT）

**目标**：让模型学会"问答格式"

**输入**：高质量的 (问题, 答案) 对，由人类专家撰写

**过程**：

1. 收集几千到几万条高质量的对话数据
2. 数据格式：`User: {问题}\nAssistant: {答案}`
3. 用标准的交叉熵损失进行微调

```python
# SFT 的伪代码
for prompt, response in sft_dataset:
    input_ids = tokenize(f"User: {prompt}\nAssistant: {response}")
    logits = model(input_ids)
    loss = cross_entropy(logits, input_ids)
    loss.backward()
```

**关键点**：

- SFT 数据**质量远比数量重要**
- OpenAI 的 InstructGPT 只用了约 13,000 条 SFT 数据
- 标注员需要是领域专家（OpenAI 雇佣了 40 名全职标注员）

**SFT 之后的模型**：
- 学会了问答格式
- 会尝试回答问题
- 但回答质量参差不齐

### 29.2.3 Stage 2：奖励模型训练（Reward Model）

**目标**：训练一个"裁判"，能判断哪个回答更好

**输入**：人类偏好数据 (prompt, response_A, response_B, preference)

**过程**：

1. 给同一个 prompt 生成多个回答（通常 4-9 个）
2. 人类标注员比较并排序
3. 训练奖励模型预测人类偏好

```
Prompt: "解释什么是黑洞"

Response A: "黑洞是一个时空区域，其引力强到连光都无法逃脱..."
Response B: "黑洞就是宇宙中的吸尘器，会把所有东西都吸进去..."
Response C: "Black hole 是一种天文现象，由大质量恒星坍缩形成..."

人类标注：A > C > B
（A 最好：准确且易懂；C 次之：准确但有中英混杂；B 最差：不够准确）
```

**奖励模型的结构**：

```
┌─────────────────────────────────────────────┐
│              Reward Model                   │
├─────────────────────────────────────────────┤
│                                             │
│   Prompt + Response                         │
│        │                                    │
│        ▼                                    │
│   ┌─────────────────────┐                   │
│   │  Transformer Layers │  (通常是 SFT     │
│   │  (冻结或微调)       │   模型的副本)    │
│   └──────────┬──────────┘                   │
│              │                              │
│              ▼                              │
│   ┌─────────────────────┐                   │
│   │    Linear Head      │                   │
│   │    (输出标量)       │                   │
│   └──────────┬──────────┘                   │
│              │                              │
│              ▼                              │
│          Reward Score                       │
│          (单个数值)                         │
│                                             │
└─────────────────────────────────────────────┘
```

奖励模型的输入是 (prompt, response)，输出是一个**标量分数**，表示这个回答的质量。

### 29.2.4 Stage 3：强化学习优化（PPO）

**目标**：用奖励模型的信号来优化语言模型

**过程**：

```
┌──────────────────────────────────────────────────────────────┐
│                     PPO 优化循环                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 采样：                                                    │
│     ┌─────────┐    prompt    ┌─────────┐                     │
│     │ Prompt  │─────────────▶│ Policy  │────▶ Response       │
│     │  池     │              │ (模型)  │                      │
│     └─────────┘              └─────────┘                      │
│                                                              │
│  2. 评估：                                                    │
│     ┌─────────────────────────────┐                           │
│     │  (Prompt, Response)         │                           │
│     └──────────────┬──────────────┘                           │
│                    │                                          │
│                    ▼                                          │
│     ┌─────────────────────────────┐                           │
│     │      Reward Model           │────▶ Reward Score        │
│     └─────────────────────────────┘                           │
│                                                              │
│  3. 更新：                                                    │
│     Reward Score ─────┐                                       │
│                       │                                       │
│     KL Penalty ───────┼──────▶ PPO Update ──▶ Policy 更新    │
│     (不要偏离太远)    │                                       │
│                       │                                       │
│     Old Policy ───────┘                                       │
│                                                              │
│  4. 重复 1-3                                                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**关键约束：KL 散度惩罚**

PPO 优化时会加入 KL 散度惩罚，防止模型偏离 SFT 模型太远：

```
Total Reward = Reward_from_RM - β * KL(policy || sft_policy)
```

这是因为：
- 纯粹最大化奖励可能导致**模型攻击**（reward hacking）
- 模型可能找到奖励模型的漏洞，生成高分但无意义的回答
- KL 惩罚让模型保持"理智"

### 29.2.5 完整 Pipeline 总结

| 阶段 | 输入 | 输出 | 目标 |
|------|------|------|------|
| Stage 1: SFT | (prompt, response) 对 | SFT 模型 | 学会问答格式 |
| Stage 2: RM | 偏好比较数据 | 奖励模型 | 学会评价回答质量 |
| Stage 3: PPO | prompts + RM 信号 | 对齐模型 | 生成高质量回答 |

---

## 29.3 奖励模型详解

### 29.3.1 人类偏好数据收集

奖励模型的核心是**人类偏好数据**。收集过程如下：

**Step 1：生成候选回答**

对于每个 prompt，用 SFT 模型生成 K 个回答（通常 K=4 到 9）：

```
Prompt: "如何提高编程能力？"

Response 1: "多练习，每天写代码..."
Response 2: "首先要打好基础，学习数据结构和算法..."
Response 3: "参加开源项目，在实战中学习..."
Response 4: "看书不如看视频，推荐以下课程..."
```

**Step 2：人类标注员排序**

标注员对这 K 个回答进行排序：

```
标注结果：Response 2 > Response 3 > Response 1 > Response 4

理由：
- Response 2：结构清晰，建议具体可行
- Response 3：实战建议好，但不够系统
- Response 1：太笼统
- Response 4：推销课程，不够中立
```

**Step 3：转换为成对比较**

从排序中提取所有两两比较：

```
(Response 2, Response 1) → 2 wins
(Response 2, Response 3) → 2 wins
(Response 2, Response 4) → 2 wins
(Response 3, Response 1) → 3 wins
(Response 3, Response 4) → 3 wins
(Response 1, Response 4) → 1 wins
```

一个包含 K 个回答的排序可以产生 C(K,2) = K*(K-1)/2 个比较对。

### 29.3.2 Bradley-Terry 模型

如何用这些比较数据训练奖励模型？这就需要 **Bradley-Terry 模型**。

**核心假设**：回答 A 优于回答 B 的概率取决于它们的"真实质量分数"之差

```
P(A > B) = σ(r(A) - r(B))
```

其中：
- r(A) 是奖励模型给 A 的分数
- r(B) 是奖励模型给 B 的分数
- σ 是 sigmoid 函数：σ(x) = 1/(1+e^(-x))

**直觉理解**：

```
如果 r(A) = 5, r(B) = 3：
  P(A > B) = σ(5-3) = σ(2) ≈ 0.88

如果 r(A) = 3, r(B) = 5：
  P(A > B) = σ(3-5) = σ(-2) ≈ 0.12

如果 r(A) = r(B) = 4：
  P(A > B) = σ(0) = 0.5
```

分数差越大，获胜概率越高；分数相同，获胜概率是 50%。

### 29.3.3 奖励模型的训练目标

训练目标是**最大化人类偏好的对数似然**：

```
Loss = -E[(log σ(r(chosen) - r(rejected)))]
```

用代码表示：

```python
# 奖励模型训练伪代码
def compute_rm_loss(prompt, chosen, rejected, reward_model):
    # 计算奖励分数
    r_chosen = reward_model(prompt, chosen)    # 被选中的回答
    r_rejected = reward_model(prompt, rejected)  # 被拒绝的回答

    # Bradley-Terry 损失
    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected))

    return loss.mean()
```

**训练数据规模**：
- InstructGPT 使用了约 33,000 个比较对
- 这些数据来自约 5,000 个 prompt，每个生成多个回答

### 29.3.4 奖励模型的挑战

**1. 标注一致性问题**

不同标注员对"好回答"的标准不同：

```
Prompt: "应该支持还是反对转基因食品？"

标注员 A（科学家）：认为支持转基因的回答更好
标注员 B（环保主义者）：认为反对转基因的回答更好
```

解决方法：
- 明确的标注指南
- 多人标注取多数
- 选择价值观一致的标注员

**2. 奖励模型容易被 hack**

模型可能学到一些"捷径"来获得高分：

```
学到的错误模式：
- 回答越长分数越高 → 模型开始啰嗦
- 使用"作为一个AI..."开头分数高 → 每次都这样开头
- 自信的语气分数高 → 即使不确定也很自信
```

解决方法：
- 多样化的训练数据
- 定期更新奖励模型
- 加入对抗样本

---

## 29.4 PPO 算法直觉

### 29.4.1 为什么需要强化学习？

你可能会问：既然已经有奖励模型了，为什么不直接用监督学习？

问题在于：**奖励模型给的是结果反馈，而不是过程反馈**。

```
监督学习：
  输入：prompt
  标签：正确答案
  问题：对于开放性问题，正确答案不唯一

强化学习：
  输入：prompt
  信号：生成的回答有多好（奖励分数）
  优势：不需要唯一正确答案，只需要知道好不好
```

### 29.4.2 为什么用 PPO？

强化学习算法有很多（Q-learning、A2C、TRPO、PPO...），为什么 RLHF 选择 PPO？

**1. 稳定性好**

语言模型的策略空间巨大（每个 token 都是一个动作）。很多 RL 算法在这种情况下不稳定，但 PPO 通过"裁剪"机制保持稳定：

```
PPO 的核心：限制每次更新的幅度

如果 new_policy(action) / old_policy(action) 太大或太小
→ 裁剪到合理范围 [1-ε, 1+ε]
→ 防止策略突变
```

**2. 样本效率较高**

PPO 是 on-policy 算法，但通过重要性采样可以复用一些样本。

**3. 实现相对简单**

相比 TRPO（需要计算 Fisher 信息矩阵），PPO 只需要简单的梯度下降。

### 29.4.3 PPO 优化的直觉

不需要深入数学，理解核心直觉就够了：

**优化目标**：

```
最大化 E[Reward] - β * KL(policy || reference)
```

翻译成人话：

1. **让模型生成高分回答**（maximize reward）
2. **但不要偏离原来太远**（KL penalty）

**为什么需要 KL 约束？**

想象你在训练一个客服机器人：

```
没有 KL 约束：
  模型发现"给用户发优惠券"能获得高分
  → 模型每次回答都变成"这是您的优惠券..."
  → 完全失去了正常对话能力

有 KL 约束：
  模型想偏离太远时会受到惩罚
  → 保持基本的对话能力
  → 在此基础上提升回答质量
```

### 29.4.4 KL 散度的具体计算

KL 散度衡量两个概率分布的差异：

```
KL(policy || reference) = E[log(policy(token) / reference(token))]
```

对于语言模型，这是在 token 级别计算的：

```python
# KL 散度计算伪代码
def compute_kl(policy_logits, reference_logits):
    policy_probs = softmax(policy_logits)
    reference_probs = softmax(reference_logits)

    kl = policy_probs * (log(policy_probs) - log(reference_probs))
    return kl.sum(dim=-1).mean()
```

**实践中的技巧**：

- β（KL 系数）通常从 0.01-0.1 开始
- 可以动态调整 β：如果 KL 太大就增加 β，太小就减少
- 目标 KL 通常设为 6-10 nats

---

## 29.5 DPO：更简单的对齐方法

### 29.5.1 RLHF 的痛点

虽然 RLHF 效果很好，但它有几个实际问题：

**1. Pipeline 太复杂**

三个阶段，三个模型（SFT、RM、Policy），调试困难。

**2. 训练不稳定**

PPO 的超参数很难调：学习率、KL 系数、GAE 参数...

**3. 计算开销大**

每次更新都需要：
- 用 Policy 生成回答
- 用 RM 评分
- 用 Reference 计算 KL
- 四个模型同时在 GPU 上

**4. 奖励模型可能不完美**

RM 的偏差会传递到最终模型。

### 29.5.2 DPO 的核心思想

2023 年，Stanford 的研究者提出了 **DPO（Direct Preference Optimization）**：

> **不需要训练奖励模型，直接从偏好数据优化策略！**

核心洞察：RLHF 的目标函数有一个**闭式解（closed-form solution）**。

在 RLHF 中，最优策略满足：

```
π*(y|x) ∝ π_ref(y|x) * exp(r(x,y) / β)
```

反过来可以推导出：

```
r(x,y) = β * log(π*(y|x) / π_ref(y|x)) + const
```

这意味着：**奖励函数可以用策略的对数概率比来表示！**

### 29.5.3 DPO 的数学直觉

把上面的奖励表达式代入 Bradley-Terry 模型：

```
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
                 = σ(β * log(π(y_w|x)/π_ref(y_w|x)) - β * log(π(y_l|x)/π_ref(y_l|x)))
```

这给出了 DPO 的损失函数：

```
L_DPO = -E[log σ(β * (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))))]
```

**直觉理解**：

- 增大被选中回答 y_w 的概率
- 减小被拒绝回答 y_l 的概率
- 同时参考 reference 策略，不要偏离太远

### 29.5.4 DPO vs RLHF 对比

```
┌─────────────────────────────────────────────────────────────────┐
│                  RLHF vs DPO 流程对比                            │
├────────────────────────────┬────────────────────────────────────┤
│          RLHF              │              DPO                   │
├────────────────────────────┼────────────────────────────────────┤
│                            │                                    │
│  1. SFT 微调               │  1. SFT 微调                       │
│        ↓                   │        ↓                           │
│  2. 收集偏好数据           │  2. 收集偏好数据                   │
│        ↓                   │        ↓                           │
│  3. 训练 Reward Model      │  3. 直接优化策略                   │
│        ↓                   │     (跳过 RM！)                    │
│  4. PPO 优化               │                                    │
│        ↓                   │        ↓                           │
│  最终模型                  │  最终模型                          │
│                            │                                    │
│  需要 4 个模型：           │  只需要 2 个模型：                 │
│  - SFT                     │  - Reference (冻结)                │
│  - RM                      │  - Policy (训练)                   │
│  - Reference               │                                    │
│  - Policy                  │                                    │
│                            │                                    │
└────────────────────────────┴────────────────────────────────────┘
```

| 对比项 | RLHF | DPO |
|--------|------|-----|
| 复杂度 | 高（三阶段） | 低（一阶段） |
| 模型数量 | 4 个 | 2 个 |
| 训练稳定性 | 较差（PPO 调参难） | 较好（类似 SFT） |
| 计算开销 | 高 | 低（约 RLHF 的 1/3） |
| 理论保证 | 有 | 有（在特定假设下等价） |
| 实际效果 | 很好 | 接近或相当 |

### 29.5.5 DPO 代码实现

```python
# 代码示例：DPO 损失计算

import torch
import torch.nn.functional as F

def compute_dpo_loss(
    policy_model,           # 要训练的策略模型
    reference_model,        # 冻结的参考模型
    chosen_input_ids,       # 被选中的回答
    rejected_input_ids,     # 被拒绝的回答
    beta=0.1                # 温度参数
):
    """
    计算 DPO 损失
    """
    # 计算策略模型的对数概率
    policy_chosen_logps = get_log_probs(policy_model, chosen_input_ids)
    policy_rejected_logps = get_log_probs(policy_model, rejected_input_ids)

    # 计算参考模型的对数概率（不需要梯度）
    with torch.no_grad():
        ref_chosen_logps = get_log_probs(reference_model, chosen_input_ids)
        ref_rejected_logps = get_log_probs(reference_model, rejected_input_ids)

    # 计算 log ratio
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    # DPO 损失
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    return loss


def get_log_probs(model, input_ids):
    """
    计算序列的对数概率
    """
    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :]  # 去掉最后一个位置
    labels = input_ids[:, 1:]            # 去掉第一个位置

    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # 对序列求和（或平均）
    return selected_log_probs.sum(dim=-1)
```

### 29.5.6 使用 TRL 库实现 DPO

HuggingFace 的 TRL 库提供了开箱即用的 DPO 实现：

```python
# 代码示例：使用 TRL 进行 DPO 训练

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 加载偏好数据集
# 数据格式：{"prompt": ..., "chosen": ..., "rejected": ...}
dataset = load_dataset("your-preference-dataset")

# 3. 配置 DPO
training_args = DPOConfig(
    output_dir="./dpo-output",
    beta=0.1,                    # 温度参数
    learning_rate=5e-7,          # 学习率（通常比 SFT 小）
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    bf16=True,
)

# 4. 创建 Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,              # 如果为 None，会自动复制 model
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# 5. 开始训练
trainer.train()
```

---

## 29.6 其他对齐方法

### 29.6.1 RLAIF：AI 反馈的强化学习

**问题**：人类标注太贵了！能不能用 AI 来标注？

**RLAIF（Reinforcement Learning from AI Feedback）** 就是这个思路：

```
传统 RLHF：人类标注偏好 → 训练 RM → PPO 优化
RLAIF：    AI 标注偏好 → 训练 RM → PPO 优化
```

**Constitutional AI（CAI）** 是 Anthropic 提出的 RLAIF 变体：

1. 定义一套"宪法"（原则集合）
2. 让 AI 根据宪法自我评估和改进回答
3. 用 AI 评估结果训练奖励模型

```
宪法示例：
- 回答应该是有帮助的
- 回答不应该包含有害内容
- 回答应该诚实承认不确定性
- 回答应该尊重隐私

AI 自我评估：
问：这个回答是否违反了"不包含有害内容"的原则？
答：是的，因为...
修订后的回答：...
```

**优势**：
- 大幅降低标注成本
- 可以快速迭代
- 原则可以明确表述

**劣势**：
- AI 判断可能有偏差
- 不如人类理解微妙的价值判断

### 29.6.2 KTO：Kahneman-Tversky Optimization

DPO 需要**成对的偏好数据**（chosen vs rejected），这种数据收集起来也不便宜。

**KTO** 只需要**单点反馈**：这个回答好不好？

```
DPO 数据：(prompt, chosen_response, rejected_response)
KTO 数据：(prompt, response, is_good)  # is_good 是 0/1
```

KTO 基于行为经济学的前景理论（Prospect Theory）：
- 人类对损失的敏感度高于收益
- 好回答带来的"快乐"小于坏回答带来的"痛苦"

### 29.6.3 IPO：Identity Preference Optimization

IPO 是 DPO 的一个改进，解决了 DPO 在某些情况下的过拟合问题。

核心改变：用一个更平滑的损失函数。

### 29.6.4 方法对比

| 方法 | 需要 RM？ | 需要 RL？ | 数据格式 | 复杂度 |
|------|----------|----------|---------|--------|
| RLHF | 是 | 是（PPO） | 偏好对 | 高 |
| DPO | 否 | 否 | 偏好对 | 低 |
| RLAIF/CAI | 是 | 是 | AI 生成 | 中 |
| KTO | 否 | 否 | 单点反馈 | 低 |
| IPO | 否 | 否 | 偏好对 | 低 |

---

## 29.7 实际应用

### 29.7.1 ChatGPT/Claude 的训练流程

OpenAI 和 Anthropic 的训练流程大致如下：

```
┌─────────────────────────────────────────────────────────────────┐
│                  主流 LLM 的训练 Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  阶段 1: 预训练 (Pre-training)                                   │
│  ├─ 数据：互联网文本、书籍、代码...（万亿 tokens）               │
│  ├─ 目标：预测下一个 token                                       │
│  ├─ 计算：数千 GPU，数月时间                                     │
│  └─ 产出：Base Model（博学但不听话）                             │
│                                                                 │
│  阶段 2: 监督微调 (SFT)                                          │
│  ├─ 数据：高质量 (指令, 回答) 对（万级别）                       │
│  ├─ 目标：学会问答格式                                           │
│  ├─ 计算：几十 GPU，几天                                         │
│  └─ 产出：Instruct Model（能对话但质量不稳定）                   │
│                                                                 │
│  阶段 3: 对齐 (RLHF/DPO)                                         │
│  ├─ 数据：人类偏好比较（十万级别）                               │
│  ├─ 目标：提升回答质量，对齐人类价值                             │
│  ├─ 计算：几十 GPU，几天到几周                                   │
│  └─ 产出：Aligned Model（ChatGPT/Claude）                        │
│                                                                 │
│  阶段 4: 持续迭代                                                │
│  ├─ 收集用户反馈                                                 │
│  ├─ 定期更新模型                                                 │
│  └─ 处理新发现的问题                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 29.7.2 开源模型的对齐实践

开源社区也有很多对齐实践：

**1. LLaMA 2 Chat**

Meta 的 LLaMA 2 Chat 使用了：
- SFT：约 27,540 条高质量对话
- RLHF：约 140 万条偏好数据
- 多轮 RLHF：迭代 5 轮

**2. Zephyr**

HuggingFace 的 Zephyr 使用了 DPO：
- 基座：Mistral 7B
- SFT：UltraChat 数据集
- DPO：UltraFeedback 数据集
- 效果：超越 LLaMA 2 70B Chat！

**3. OpenChat / Starling**

使用 C-RLFT（Conditioned RLFT）：
- 混合 SFT 和偏好学习
- 效果接近 GPT-3.5

### 29.7.3 对齐数据集

开源社区常用的对齐数据集：

| 数据集 | 类型 | 规模 | 用途 |
|--------|------|------|------|
| OpenAssistant | SFT | 16万条 | 对话微调 |
| Dolly | SFT | 1.5万条 | 指令微调 |
| UltraChat | SFT | 150万条 | 多轮对话 |
| UltraFeedback | 偏好 | 6.4万条 | DPO 训练 |
| HH-RLHF | 偏好 | 17万条 | RLHF/DPO |
| Anthropic HH | 偏好 | 16万条 | 安全对齐 |

### 29.7.4 一个实际的 DPO 训练流程

```python
# 代码示例：完整的 DPO 训练流程

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 1. 加载基座模型
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. 应用 LoRA（可选，节省显存）
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. 加载偏好数据
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

# 4. 数据预处理
def format_dataset(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"][1]["content"],
        "rejected": example["rejected"][1]["content"],
    }

dataset = dataset.map(format_dataset)

# 5. DPO 配置
training_args = DPOConfig(
    output_dir="./dpo-mistral",
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
)

# 6. 创建 Trainer 并训练
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()

# 7. 保存模型
trainer.save_model("./dpo-mistral-final")
```

---

## 29.8 本章要点

### 29.8.1 核心概念回顾

| 概念 | 含义 |
|------|------|
| **对齐（Alignment）** | 让模型行为符合人类期望和价值观 |
| **HHH** | Helpful, Harmless, Honest - 对齐的三个目标 |
| **RLHF** | 用人类反馈训练奖励模型，再用 RL 优化策略 |
| **SFT** | 监督微调，让模型学会问答格式 |
| **Reward Model** | 评价回答质量的模型，输出标量分数 |
| **PPO** | 策略梯度算法，用于优化语言模型 |
| **DPO** | 直接从偏好数据优化，跳过奖励模型 |
| **KL 约束** | 防止模型偏离参考策略太远 |

### 29.8.2 RLHF 三阶段

```
Stage 1: SFT
  输入：(问题, 答案) 对
  输出：能对话的模型

Stage 2: Reward Model
  输入：偏好比较 (A > B)
  输出：打分模型

Stage 3: PPO
  输入：prompts + RM 信号
  输出：对齐的模型
```

### 29.8.3 DPO 简化

```
DPO = 直接从偏好数据优化策略

优势：
- 不需要训练 RM
- 不需要 PPO
- 训练更稳定
- 计算开销更低
```

### 29.8.4 关键公式

**Bradley-Terry 模型**：
```
P(A > B) = sigmoid(r(A) - r(B))
```

**RLHF 优化目标**：
```
max E[Reward] - β * KL(policy || reference)
```

**DPO 损失**：
```
L = -log sigmoid(β * (log(π(y_w)/π_ref(y_w)) - log(π(y_l)/π_ref(y_l))))
```

### 29.8.5 核心认知

> **RLHF 是 ChatGPT 成功的关键技术。它不是让模型更"聪明"，而是让模型更"懂事"——理解用户意图，拒绝有害请求，承认自己的不确定性。DPO 提供了一条更简单的路：直接从偏好数据学习，跳过复杂的强化学习过程，同时达到相近的效果。**

---

## 本章交付物

学完这一章，你应该能够：

- [ ] 解释为什么预训练模型需要对齐（三个问题）
- [ ] 描述 RLHF 的三个阶段
- [ ] 理解奖励模型如何训练（Bradley-Terry）
- [ ] 解释 PPO 中 KL 约束的作用
- [ ] 比较 DPO 和 RLHF 的优劣
- [ ] 使用 TRL 库实现简单的 DPO 训练

---

## 延伸阅读

- **InstructGPT 论文**：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- **DPO 论文**：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- **Constitutional AI**：[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- **TRL 库文档**：[HuggingFace TRL](https://huggingface.co/docs/trl)
- **LLaMA 2 技术报告**：[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

---

## 下一章预告

我们已经理解了如何让模型"对齐"人类。但 GPT-4 和 LLaMA 这样的模型有个问题：**它们太"密集"了**——每次推理都要激活所有参数。

下一章，我们将学习 **Mixture of Experts (MoE)**——一种稀疏激活的架构，让模型可以拥有万亿参数，但每次只用其中一小部分。这正是 GPT-4、Mixtral、DeepSeek-V3 背后的秘密。
