# Part 9：前沿进展（2024-2025）

> **从 ChatGPT 到 o1/R1：理解大模型最新突破**

---

## 本部分定位

2024-2025 年是大模型发展的关键转折点：

- **对齐技术成熟**：RLHF/DPO 让模型"听话"
- **推理能力飞跃**：o1/R1 在数学/编程上超越人类专家
- **架构持续演进**：MoE、Mamba 挑战传统 Transformer
- **效率大幅提升**：相同性能，成本降低 10-100 倍

本部分不再是"理论讲解"，而是**追踪最新技术动态**，帮你理解：
- 为什么 DeepSeek-R1 能用 1/10 成本对标 o1？
- 为什么 Mixtral 8x7B 能打 70B 模型？
- 为什么 Mamba 被称为"后 Transformer"？

---

## 章节导航

| 章节 | 主题 | 你将学会 |
|------|------|----------|
| [第 28 章](./第28章：Prompt%20Engineering%20-%20提示工程实战.md) | Prompt Engineering | CoT、Few-shot、让模型输出更准确 |
| [第 29 章](./第29章：RLHF与偏好学习%20-%20让模型对齐人类.md) | RLHF 与 DPO | 理解 ChatGPT 为什么"听话" |
| [第 30 章](./第30章：Mixture%20of%20Experts%20-%20稀疏激活的秘密.md) | Mixture of Experts | 理解 Mixtral、DeepSeek-V3 架构 |
| [第 31 章](./第31章：推理模型革命%20-%20从o1到R1.md) | 推理模型 | 理解 o1、R1、K1.5 的突破 |
| [第 32 章](./第32章：后Transformer架构%20-%20Mamba与混合模型.md) | Mamba 与 SSM | 理解 O(N) 复杂度的新架构 |

---

## 关键论文索引

| 论文 | 组织 | 年份 | 对应章节 |
|------|------|------|----------|
| Training language models to follow instructions (InstructGPT) | OpenAI | 2022 | 第 29 章 |
| Direct Preference Optimization (DPO) | Stanford | 2023 | 第 29 章 |
| Mixtral of Experts | Mistral AI | 2024 | 第 30 章 |
| DeepSeek-V3 Technical Report | DeepSeek | 2024 | 第 30 章 |
| Learning to Reason with LLMs (o1) | OpenAI | 2024 | 第 31 章 |
| DeepSeek-R1: Incentivizing Reasoning via RL | DeepSeek | 2025 | 第 31 章 |
| Kimi K1.5: Scaling RL with LLMs | Moonshot | 2025 | 第 31 章 |
| Mamba: Linear-Time Sequence Modeling | CMU/Princeton | 2023 | 第 32 章 |
| Jamba: Hybrid Transformer-Mamba | AI21 | 2024 | 第 32 章 |

---

## 学习路径建议

### 如果你是应用开发者
优先阅读：第 28 章（Prompt）→ 第 31 章（推理模型）

### 如果你是 ML 工程师
完整阅读：第 29 章（RLHF）→ 第 30 章（MoE）→ 第 31 章 → 第 32 章

### 如果你想了解行业动态
快速浏览：第 31 章（推理模型革命）是 2024-2025 最大突破

---

## 前置知识

阅读本部分前，建议已完成：
- Part 1-4：理解 Transformer 基础架构
- Part 5：理解训练和推理代码
- 第 26 章：理解 LoRA 微调（对理解 RLHF 有帮助）

---

> **本部分核心观点**：2024-2025 的突破不在于"更大的模型"，而在于**更聪明的训练方法**（RL）和**更高效的架构**（MoE/Mamba）。
