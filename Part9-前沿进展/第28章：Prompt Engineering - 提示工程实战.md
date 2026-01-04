# 第 28 章：Prompt Engineering - 提示工程实战

> **一句话总结**：Prompt Engineering 不是玄学，而是理解模型如何"思考"的科学——通过 Few-shot 给模型示范，用 Chain-of-Thought 激发推理能力，借助 Self-Consistency 提高可靠性，这些技巧能让同一个模型的表现提升 20%-50%。

---

## 28.1 为什么需要 Prompt Engineering

### 28.1.1 同一个模型，不同的表现

假设你想让 GPT-4 帮你做一道数学题：

**提示 A（直接问）**：
```
小明有 5 个苹果，小红给了他 3 个，然后他吃了 2 个。现在小明有几个苹果？
```

**提示 B（引导思考）**：
```
小明有 5 个苹果，小红给了他 3 个，然后他吃了 2 个。现在小明有几个苹果？

请一步一步思考：
1. 首先，小明最初有多少苹果？
2. 小红给了他多少？加上之后是多少？
3. 他吃了多少？减去之后是多少？
```

对于简单问题，两种方式可能都对。但对于复杂问题，**提示 B 的正确率可能比 A 高 30% 以上**。

这就是 Prompt Engineering 的价值：**同样的模型，不同的问法，效果天差地别**。

### 28.1.2 模型不是"理解"，而是"续写"

要理解 Prompt Engineering，首先要理解大模型的本质：

> **核心认知**：大模型的本质是**条件概率分布**——给定前文，预测下一个最可能的 token。

模型不是"理解"你的问题然后"思考"出答案，而是根据你给的 prompt，**续写最可能出现的文本**。

这意味着：
- **好的 prompt = 好的上下文 = 更容易续写出正确答案**
- **差的 prompt = 模糊的上下文 = 可能续写出任何内容**

### 28.1.3 Prompt Engineering 的三个层次

| 层次 | 技巧 | 效果提升 |
|------|------|----------|
| **基础层** | 清晰表达、格式约束 | 减少歧义，提高一致性 |
| **进阶层** | Few-shot、CoT、角色扮演 | 显著提升复杂任务表现 |
| **高级层** | Self-Consistency、ToT、Agent 设计 | 接近或超越人类专家水平 |

本章会从基础讲到高级，每个技巧都配有实战代码。

---

## 28.2 Zero-shot vs Few-shot Prompting

### 28.2.1 三种基本范式

根据是否提供示例，Prompt 可以分为三类：

| 类型 | 示例数量 | 特点 |
|------|----------|------|
| **Zero-shot** | 0 个 | 直接问，不给示例 |
| **One-shot** | 1 个 | 给一个示例 |
| **Few-shot** | 2-10 个 | 给多个示例 |

### 28.2.2 Zero-shot：直接问

Zero-shot 是最简单的方式，直接描述任务，不提供任何示例。

```python
# 代码示例：Zero-shot Prompting

prompt = """
请判断以下评论的情感是正面还是负面：

评论：这家餐厅的服务态度太差了，等了一个小时才上菜。
情感：
"""

# 模型输出：负面
```

**优点**：
- 简单快速
- 不需要准备示例
- 适合模型已经"理解"的任务

**缺点**：
- 对于复杂任务效果可能不好
- 输出格式不稳定

### 28.2.3 One-shot：给一个例子

One-shot 提供一个示例，让模型"模仿"：

```python
# 代码示例：One-shot Prompting

prompt = """
请判断以下评论的情感是正面还是负面。

示例：
评论：食物很新鲜，价格也实惠，下次还会来！
情感：正面

现在请判断：
评论：这家餐厅的服务态度太差了，等了一个小时才上菜。
情感：
"""

# 模型输出：负面
```

**关键点**：示例展示了期望的输入输出格式，模型会模仿这种格式。

### 28.2.4 Few-shot：给多个例子

Few-shot 提供多个示例，覆盖不同情况：

```python
# 代码示例：Few-shot Prompting

prompt = """
请判断以下评论的情感是正面、负面还是中性。

示例 1：
评论：食物很新鲜，价格也实惠，下次还会来！
情感：正面

示例 2：
评论：这家餐厅的服务态度太差了，等了一个小时才上菜。
情感：负面

示例 3：
评论：菜品一般，环境还行，中规中矩吧。
情感：中性

示例 4：
评论：虽然价格贵了点，但味道确实不错，值得尝试。
情感：正面

现在请判断：
评论：装修很有特色，但菜量太少了，性价比不高。
情感：
"""

# 模型输出：中性（或负面，取决于模型判断）
```

**Few-shot 的关键设计**：

1. **示例数量**：通常 3-8 个，太少不够代表性，太多浪费 context
2. **示例多样性**：覆盖不同类别，避免偏向某一类
3. **示例质量**：确保示例本身是正确的
4. **顺序影响**：有研究表明，最后几个示例的影响更大

### 28.2.5 三种方式的对比

```python
# 代码示例：对比测试

import openai

def test_prompt(prompt, test_cases):
    """测试 prompt 在多个用例上的表现"""
    correct = 0
    for text, expected in test_cases:
        full_prompt = prompt + f"\n评论：{text}\n情感："
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip()
        if expected in answer:
            correct += 1
    return correct / len(test_cases)

# 测试用例
test_cases = [
    ("太棒了，强烈推荐！", "正面"),
    ("简直是浪费钱", "负面"),
    ("还可以吧，一般般", "中性"),
    # ... 更多测试用例
]

# 对比结果（示意）
# Zero-shot 准确率：75%
# One-shot 准确率：82%
# Few-shot 准确率：91%
```

### 28.2.6 什么时候用哪种？

| 场景 | 推荐方式 | 理由 |
|------|----------|------|
| 简单任务（翻译、摘要） | Zero-shot | 模型已经"理解"，不需要示例 |
| 格式要求严格 | One-shot/Few-shot | 示例展示期望格式 |
| 分类任务 | Few-shot | 每个类别至少一个示例 |
| 复杂推理 | Few-shot + CoT | 需要展示推理过程 |
| Context 有限 | Zero-shot/One-shot | 节省 token |

---

## 28.3 Chain-of-Thought (CoT) 思维链

### 28.3.1 一个神奇的发现

2022 年，Google 的研究者发现了一个惊人的现象：

> 只需要在 prompt 里加上 **"Let's think step by step"（让我们一步一步思考）**，模型在数学和逻辑推理任务上的准确率就能提高 20%-50%。

这就是 **Chain-of-Thought (CoT)** 的起源。

### 28.3.2 为什么 "Let's think step by step" 有效？

回想大模型的本质：**给定前文，预测下一个 token**。

**没有 CoT 时**：
```
问题：小明有 17 个苹果，他分给 3 个朋友，每人 4 个，还剩多少？
答案：
```

模型需要直接从问题跳到答案。对于复杂问题，这种"一步到位"的推理很容易出错。

**有 CoT 时**：
```
问题：小明有 17 个苹果，他分给 3 个朋友，每人 4 个，还剩多少？
让我们一步一步思考：
- 首先，小明有 17 个苹果
- 他分给 3 个朋友，每人 4 个
- 所以一共分出去：3 × 4 = 12 个
- 还剩：17 - 12 = 5 个
答案：5 个
```

**关键洞察**：

1. **中间步骤提供上下文**：每一步的输出成为下一步的输入
2. **减少单步跨度**：把大问题分解成小问题
3. **利用模型的续写能力**：模型擅长基于前文续写

> **类比**：就像人类做数学题时写草稿一样，CoT 让模型"展示工作过程"，而不是直接给答案。

### 28.3.3 Zero-shot CoT

最简单的 CoT 是在 prompt 后面加一句话：

```python
# 代码示例：Zero-shot CoT

# 没有 CoT
prompt_no_cot = """
问题：一个停车场有 5 排车位，每排 8 个车位，已经停了 23 辆车，还能停多少辆？
答案：
"""

# 有 CoT
prompt_with_cot = """
问题：一个停车场有 5 排车位，每排 8 个车位，已经停了 23 辆车，还能停多少辆？
让我们一步一步思考：
"""

# 模型输出（有 CoT）：
# 1. 首先计算总车位数：5 排 × 8 个/排 = 40 个车位
# 2. 已经停了 23 辆车
# 3. 还能停：40 - 23 = 17 辆
# 答案：17 辆
```

**常用的 Zero-shot CoT 触发语**：
- "Let's think step by step"
- "让我们一步一步思考"
- "请详细分析这个问题"
- "First, ... Then, ... Finally, ..."

### 28.3.4 Few-shot CoT

更强大的方式是 Few-shot + CoT：在示例中展示推理过程。

```python
# 代码示例：Few-shot CoT

prompt = """
问题：小明有 5 个苹果，小红给了他 3 个，他吃了 2 个，还剩多少？
思考过程：
1. 小明最初有 5 个苹果
2. 小红给了 3 个，现在有：5 + 3 = 8 个
3. 吃了 2 个，剩下：8 - 2 = 6 个
答案：6 个

问题：一个书架有 4 层，每层放 7 本书，借走了 12 本，还剩多少？
思考过程：
1. 总共有：4 × 7 = 28 本书
2. 借走了 12 本
3. 剩下：28 - 12 = 16 本
答案：16 本

问题：小李买了 3 盒巧克力，每盒 8 块，他吃了 5 块，送人 10 块，还剩多少？
思考过程：
"""

# 模型会模仿这种格式，展示推理过程
```

### 28.3.5 Zero-shot CoT vs Few-shot CoT

| 特性 | Zero-shot CoT | Few-shot CoT |
|------|---------------|--------------|
| **示例需求** | 无 | 需要 2-8 个示例 |
| **准备工作** | 几乎不需要 | 需要准备高质量示例 |
| **效果** | 较好 | 更好 |
| **适用场景** | 快速测试、简单任务 | 生产环境、复杂任务 |
| **Context 消耗** | 少 | 多 |

### 28.3.6 实际例子：数学题

```python
# 代码示例：数学推理

import openai

def solve_math_with_cot(problem):
    """使用 CoT 解决数学问题"""
    prompt = f"""
你是一个数学老师，请解决以下问题，展示完整的推理过程。

问题：{problem}

解题步骤：
"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # 降低随机性，提高一致性
    )

    return response.choices[0].message.content

# 测试
problem = """
一个水箱原来有 120 升水。第一天用了总量的 1/4，
第二天用了剩余的 1/3，第三天又加入 25 升。
现在水箱里有多少升水？
"""

print(solve_math_with_cot(problem))

# 输出示例：
# 解题步骤：
# 1. 原来有 120 升水
# 2. 第一天用了 1/4：120 × 1/4 = 30 升
#    剩余：120 - 30 = 90 升
# 3. 第二天用了剩余的 1/3：90 × 1/3 = 30 升
#    剩余：90 - 30 = 60 升
# 4. 第三天加入 25 升：60 + 25 = 85 升
# 答案：现在水箱里有 85 升水
```

### 28.3.7 实际例子：逻辑推理

```python
# 代码示例：逻辑推理

prompt = """
请根据以下线索推理出答案，展示你的思考过程。

线索：
- 小明、小红、小华三人分别住在 1 楼、2 楼、3 楼
- 小明不住最高层
- 小红住在小华楼上
- 住 2 楼的人不是小明

问题：每个人分别住在几楼？

让我们一步一步分析：
"""

# 模型输出：
# 1. 从"小明不住最高层"得知：小明住 1 楼或 2 楼
# 2. 从"住 2 楼的人不是小明"得知：小明不住 2 楼
# 3. 结合 1 和 2：小明住 1 楼
# 4. 从"小红住在小华楼上"得知：小红比小华高
# 5. 剩下 2 楼和 3 楼给小红和小华
# 6. 由于小红比小华高：小红住 3 楼，小华住 2 楼
#
# 答案：
# - 小明：1 楼
# - 小华：2 楼
# - 小红：3 楼
```

---

## 28.4 Self-Consistency 自洽性

### 28.4.1 为什么需要 Self-Consistency？

CoT 虽然有效，但有一个问题：**同一个问题，模型可能给出不同的推理路径，有些对，有些错**。

这是因为语言模型的采样过程具有随机性（temperature > 0）。

**Self-Consistency 的核心思想**：**多次采样，取多数票**。

### 28.4.2 工作原理

```
                    ┌─────────────────┐
                    │    同一个问题    │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │ 推理路径1 │      │ 推理路径2 │      │ 推理路径3 │
    │ 答案: 17 │      │ 答案: 17 │      │ 答案: 15 │
    └──────────┘      └──────────┘      └──────────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  多数票: 17 ✓   │
                    └─────────────────┘
```

步骤：
1. 用 CoT 生成多个回答（比如 5-10 个）
2. 提取每个回答的最终答案
3. 对答案进行投票，选择出现次数最多的

### 28.4.3 代码实现

```python
# 代码示例：Self-Consistency

import openai
from collections import Counter
import re

def self_consistency(problem, num_samples=5, temperature=0.7):
    """
    使用 Self-Consistency 方法解决问题

    Args:
        problem: 问题描述
        num_samples: 采样次数
        temperature: 采样温度（越高越随机）

    Returns:
        最一致的答案
    """
    prompt = f"""
请解决以下问题，展示你的推理过程，最后用"答案：X"的格式给出答案。

问题：{problem}

推理过程：
"""

    answers = []
    reasoning_paths = []

    for i in range(num_samples):
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500
        )

        text = response.choices[0].message.content
        reasoning_paths.append(text)

        # 提取答案（假设格式是"答案：X"）
        match = re.search(r'答案[：:]\s*(\d+)', text)
        if match:
            answers.append(int(match.group(1)))

    # 投票
    if answers:
        counter = Counter(answers)
        most_common_answer, count = counter.most_common(1)[0]
        confidence = count / len(answers)

        return {
            "answer": most_common_answer,
            "confidence": confidence,
            "all_answers": answers,
            "reasoning_paths": reasoning_paths
        }

    return None

# 使用示例
problem = "一个农场有 15 只鸡和 12 只兔子，一共有多少条腿？"
result = self_consistency(problem, num_samples=5)

print(f"答案: {result['answer']}")
print(f"置信度: {result['confidence']:.0%}")
print(f"所有采样答案: {result['all_answers']}")

# 输出示例：
# 答案: 78
# 置信度: 100%
# 所有采样答案: [78, 78, 78, 78, 78]
```

### 28.4.4 Self-Consistency 的效果

根据原论文的实验结果，Self-Consistency 在多个基准测试上的提升：

| 任务 | CoT 单次 | CoT + Self-Consistency | 提升 |
|------|----------|------------------------|------|
| GSM8K（数学） | 56.5% | 74.4% | +17.9% |
| SVAMP（数学） | 68.9% | 81.6% | +12.7% |
| AQuA（数学） | 48.3% | 57.9% | +9.6% |
| StrategyQA（推理） | 73.4% | 81.3% | +7.9% |

### 28.4.5 参数选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **采样次数** | 5-10 | 太少不够稳定，太多成本高 |
| **Temperature** | 0.5-0.8 | 太低缺乏多样性，太高太随机 |
| **提取方式** | 正则匹配 | 确保答案格式统一 |

### 28.4.6 成本考虑

Self-Consistency 的主要缺点是**成本增加**：

- 5 次采样 = 5 倍 API 调用费用
- 10 次采样 = 10 倍 API 调用费用

**何时使用**：
- 准确性要求高的场景
- 答案可以明确验证的任务（如数学题）
- 成本不是主要考虑因素

**优化策略**：
- 先用单次 CoT 快速筛选
- 对不确定的结果再用 Self-Consistency
- 动态调整采样次数

---

## 28.5 其他高级技巧

### 28.5.1 Role/Persona Prompting（角色扮演）

让模型扮演特定角色，可以激活相关的知识和表达方式。

```python
# 代码示例：角色扮演

# 基础提问
prompt_basic = """
如何提高 Python 代码的执行效率？
"""

# 角色扮演
prompt_role = """
你是一位有 20 年经验的 Python 核心开发者，曾参与过 CPython 的优化工作。
现在一位初级开发者问你：如何提高 Python 代码的执行效率？

请从底层原理出发，给出专业而实用的建议。
"""

# 角色扮演通常会产生更专业、更深入的回答
```

**常用角色**：

| 任务 | 推荐角色 |
|------|----------|
| 代码审查 | "资深软件工程师" |
| 学术写作 | "某领域教授" |
| 法律问题 | "执业律师" |
| 创意写作 | "获奖小说家" |
| 数据分析 | "数据科学家" |

**注意**：角色扮演不会让模型真正"变成"专家，但可以激活相关的训练数据和表达模式。

### 28.5.2 Tree-of-Thought (ToT)

ToT 是 CoT 的升级版，不是线性推理，而是**树状探索**。

```
                    问题
                      │
           ┌─────────┼─────────┐
           │         │         │
         思路A     思路B     思路C
           │         │         │
        ┌──┴──┐   ┌──┴──┐   ┌──┴──┐
        │     │   │     │   │     │
      评估  评估 评估  评估 评估  评估
        │           │
      继续         继续        放弃
        │           │
       ...         ...
```

**核心思想**：
1. 生成多个可能的思路
2. 评估每个思路的可行性
3. 选择最有希望的继续探索
4. 遇到死路回溯

```python
# 代码示例：Tree-of-Thought（简化版）

def tree_of_thought(problem, depth=3, branches=3):
    """
    Tree-of-Thought 推理
    """

    def generate_thoughts(context, num_thoughts):
        """生成多个思路"""
        prompt = f"""
{context}

请提出 {num_thoughts} 个不同的下一步思路，每个思路用一行描述。
格式：
1. 思路一
2. 思路二
3. 思路三
"""
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        return parse_thoughts(response.choices[0].message.content)

    def evaluate_thought(context, thought):
        """评估思路的可行性"""
        prompt = f"""
问题：{problem}
当前进展：{context}
候选思路：{thought}

请评估这个思路的可行性，给出 1-10 分，并简要说明理由。
格式：分数：X，理由：...
"""
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return parse_score(response.choices[0].message.content)

    # 主循环
    context = f"问题：{problem}\n"

    for d in range(depth):
        thoughts = generate_thoughts(context, branches)
        scored_thoughts = [(t, evaluate_thought(context, t)) for t in thoughts]
        best_thought = max(scored_thoughts, key=lambda x: x[1])[0]
        context += f"\n第 {d+1} 步：{best_thought}"

    return context

# 适用于复杂的规划、游戏策略、创意任务等
```

### 28.5.3 Prompt Chaining（提示链）

把复杂任务分解成多个简单步骤，串联执行。

```python
# 代码示例：Prompt Chaining

def analyze_article(article):
    """多步骤分析文章"""

    # 第一步：提取关键信息
    step1_prompt = f"""
请从以下文章中提取：
1. 主题
2. 主要论点（3-5 个）
3. 结论

文章：
{article}
"""
    step1_result = call_llm(step1_prompt)

    # 第二步：评估论证质量
    step2_prompt = f"""
基于以下文章分析：
{step1_result}

请评估：
1. 论证的逻辑性（1-10 分）
2. 证据的充分性（1-10 分）
3. 存在的逻辑漏洞
"""
    step2_result = call_llm(step2_prompt)

    # 第三步：生成综合报告
    step3_prompt = f"""
基于以下分析：

文章信息：
{step1_result}

质量评估：
{step2_result}

请生成一份简洁的综合评价报告。
"""
    final_report = call_llm(step3_prompt)

    return final_report
```

**优点**：
- 每一步专注于单一任务
- 便于调试和优化
- 可以在中间步骤加入人工检查

**缺点**：
- 多次 API 调用，成本和延迟增加
- 需要设计好步骤之间的衔接

### 28.5.4 格式约束技巧

让模型输出结构化格式，便于程序解析：

```python
# 代码示例：格式约束

# JSON 格式约束
prompt_json = """
请分析以下产品评论，以 JSON 格式输出：

评论：这款手机拍照效果很好，但电池续航差，价格偏贵。

请输出格式：
{
  "优点": ["..."],
  "缺点": ["..."],
  "情感": "正面/负面/中性",
  "推荐度": 1-5
}
"""

# XML 格式约束
prompt_xml = """
请将以下信息结构化：

原文：John Smith, 35 岁，软件工程师，住在纽约

输出格式：
<person>
  <name>...</name>
  <age>...</age>
  <occupation>...</occupation>
  <location>...</location>
</person>
"""

# Markdown 表格格式
prompt_table = """
请比较 Python 和 JavaScript，以表格形式输出：

| 特性 | Python | JavaScript |
|------|--------|------------|
| ... | ... | ... |
"""
```

**格式约束最佳实践**：

1. **明确给出格式模板**：不要只说"用 JSON"，要给出具体结构
2. **使用代码块**：用 ``` 包裹期望的格式
3. **提供示例**：展示正确的输出格式
4. **添加验证提示**：如"确保 JSON 格式正确，可以被解析"

---

## 28.6 实战案例

### 28.6.1 文本分类

```python
# 代码示例：多标签文本分类

def classify_text(text, categories):
    """
    多标签文本分类

    Args:
        text: 待分类文本
        categories: 可能的类别列表

    Returns:
        分类结果
    """

    categories_str = "\n".join([f"- {c}" for c in categories])

    prompt = f"""
你是一个文本分类专家。请将以下文本分类到合适的类别。

可选类别：
{categories_str}

文本：
{text}

要求：
1. 可以选择多个类别
2. 只选择确实相关的类别
3. 以 JSON 数组格式输出
4. 给出分类的理由

输出格式：
{{
  "categories": ["类别1", "类别2"],
  "reasoning": "分类理由..."
}}
"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}  # 强制 JSON 输出
    )

    return json.loads(response.choices[0].message.content)

# 使用示例
text = """
OpenAI 今日发布了 GPT-4 Turbo，上下文窗口扩展到 128K，
价格下降 3 倍，还支持函数调用和 JSON 模式。
开发者社区反响热烈，股价应声上涨。
"""

categories = ["科技", "财经", "体育", "娱乐", "政治", "AI/机器学习"]

result = classify_text(text, categories)
print(result)

# 输出：
# {
#   "categories": ["科技", "AI/机器学习", "财经"],
#   "reasoning": "文章主要讨论 OpenAI 发布新模型，属于科技和 AI 领域；
#                提到股价和价格，涉及财经内容"
# }
```

### 28.6.2 代码生成

```python
# 代码示例：代码生成

def generate_code(task_description, language="Python", context=None):
    """
    生成代码

    Args:
        task_description: 任务描述
        language: 编程语言
        context: 上下文代码（可选）
    """

    context_part = ""
    if context:
        context_part = f"""
现有代码：
```{language.lower()}
{context}
```
"""

    prompt = f"""
你是一位资深 {language} 开发者。请根据以下需求编写代码。

{context_part}

需求：
{task_description}

要求：
1. 代码要简洁、高效、可读
2. 添加必要的注释
3. 包含错误处理
4. 如果需要，提供使用示例

请用以下格式输出：

```{language.lower()}
# 你的代码
```

**使用说明**：
...

**注意事项**：
...
"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # 代码生成用较低温度
    )

    return response.choices[0].message.content

# 使用示例
task = """
实现一个 LRU (Least Recently Used) 缓存类，支持：
1. get(key) - 获取值，不存在返回 -1
2. put(key, value) - 插入或更新值
3. 容量限制，超出时删除最久未使用的项
"""

code = generate_code(task, "Python")
print(code)
```

### 28.6.3 数学问题求解

```python
# 代码示例：数学问题求解（结合多种技术）

def solve_math_problem(problem, use_self_consistency=True):
    """
    解决数学问题，结合 CoT 和 Self-Consistency
    """

    few_shot_examples = """
问题：小明有 15 元，买了 3 支笔，每支 2.5 元，还剩多少钱？
解题过程：
1. 计算总花费：3 × 2.5 = 7.5 元
2. 计算剩余：15 - 7.5 = 7.5 元
答案：7.5 元

问题：一个水池有两个进水管，单独开 A 管 6 小时能注满，单独开 B 管 4 小时能注满。同时开两管，多久能注满？
解题过程：
1. A 管每小时注入：1/6 池
2. B 管每小时注入：1/4 池
3. 同时开，每小时注入：1/6 + 1/4 = 2/12 + 3/12 = 5/12 池
4. 注满需要时间：1 ÷ (5/12) = 12/5 = 2.4 小时
答案：2.4 小时（或 2 小时 24 分钟）
"""

    prompt = f"""
你是一位数学老师，请认真解决以下问题。

参考示例：
{few_shot_examples}

现在请解决：
问题：{problem}

解题过程：
"""

    if use_self_consistency:
        # 多次采样
        answers = []
        for _ in range(5):
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            text = response.choices[0].message.content
            # 提取答案
            answer = extract_answer(text)
            if answer:
                answers.append(answer)

        # 投票
        if answers:
            from collections import Counter
            most_common = Counter(answers).most_common(1)[0]
            return {
                "answer": most_common[0],
                "confidence": most_common[1] / len(answers),
                "all_answers": answers
            }
    else:
        # 单次采样
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return {"answer": response.choices[0].message.content}

# 使用
problem = """
甲乙两地相距 360 公里。一辆汽车从甲地出发，每小时行驶 60 公里。
2 小时后，一辆摩托车从乙地出发迎面驶来，每小时行驶 80 公里。
问：摩托车出发后多久两车相遇？
"""

result = solve_math_problem(problem)
print(f"答案：{result['answer']}")
print(f"置信度：{result.get('confidence', 'N/A')}")
```

---

## 28.7 常见误区与最佳实践

### 28.7.1 常见误区

| 误区 | 问题 | 正确做法 |
|------|------|----------|
| **越长越好** | 过长的 prompt 浪费 token，可能引入噪声 | 精简、清晰、直达要点 |
| **堆砌示例** | 示例太多，超出 context 限制 | 3-8 个高质量示例即可 |
| **忽略格式** | 输出格式不一致，难以解析 | 明确指定输出格式 |
| **万能 prompt** | 想用一个 prompt 解决所有问题 | 针对不同任务定制 |
| **盲目 CoT** | 简单任务也用 CoT | 简单任务直接问更高效 |
| **忽略温度** | 不调整 temperature 参数 | 创意任务高温，确定性任务低温 |

### 28.7.2 最佳实践清单

**Prompt 结构设计**：

```python
# 代码示例：推荐的 Prompt 结构

def build_prompt(
    role: str,
    context: str,
    task: str,
    examples: list,
    constraints: list,
    output_format: str
):
    """构建结构化 Prompt"""

    prompt = f"""
# 角色设定
{role}

# 背景信息
{context}

# 任务描述
{task}

# 示例
"""

    for i, ex in enumerate(examples, 1):
        prompt += f"\n示例 {i}：\n输入：{ex['input']}\n输出：{ex['output']}\n"

    prompt += f"""
# 约束条件
"""
    for c in constraints:
        prompt += f"- {c}\n"

    prompt += f"""
# 输出格式
{output_format}

# 现在请处理：
"""

    return prompt

# 使用示例
prompt = build_prompt(
    role="你是一位资深的技术文档工程师",
    context="我们正在编写一个 Python 库的 API 文档",
    task="为给定的函数生成规范的 docstring",
    examples=[
        {
            "input": "def add(a, b): return a + b",
            "output": '"""Add two numbers..."""'
        }
    ],
    constraints=[
        "使用 Google 风格的 docstring",
        "包含参数说明和返回值说明",
        "如果可能，包含使用示例"
    ],
    output_format="直接输出 docstring，用三引号包裹"
)
```

### 28.7.3 调试技巧

当 prompt 效果不好时：

1. **检查歧义**：prompt 是否有多种理解方式？
2. **添加示例**：模型是否理解你要什么？
3. **分解任务**：任务是否太复杂？
4. **调整温度**：输出是否太随机或太死板？
5. **换模型**：当前模型是否适合这个任务？

```python
# 代码示例：Prompt 调试工具

def debug_prompt(prompt, num_runs=3):
    """
    调试 prompt：多次运行观察一致性
    """
    results = []

    for i in range(num_runs):
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        results.append(response.choices[0].message.content)

    # 分析一致性
    print("=" * 50)
    print(f"Prompt:\n{prompt[:200]}...")
    print("=" * 50)

    for i, r in enumerate(results, 1):
        print(f"\n--- 运行 {i} ---")
        print(r[:300] + "..." if len(r) > 300 else r)

    # 简单的一致性检查
    if len(set(results)) == 1:
        print("\n[结论] 输出完全一致")
    elif len(set(results)) < num_runs:
        print("\n[结论] 输出部分一致")
    else:
        print("\n[结论] 输出不一致，考虑降低 temperature 或改进 prompt")

# 使用
debug_prompt("请用一句话介绍 Python 语言")
```

### 28.7.4 性能优化

```python
# 代码示例：批量处理优化

import asyncio
import aiohttp

async def batch_process(prompts, model="gpt-4", max_concurrent=5):
    """
    批量处理 prompts，使用异步并发
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(prompt):
        async with semaphore:
            # 使用异步 API 调用
            response = await async_openai_call(prompt, model)
            return response

    tasks = [process_one(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

# 使用
prompts = [f"将以下数字翻译成英文：{i}" for i in range(100)]
results = asyncio.run(batch_process(prompts))
```

---

## 28.8 本章要点

### 28.8.1 核心技术总结

| 技术 | 核心思想 | 适用场景 | 效果提升 |
|------|----------|----------|----------|
| **Zero-shot** | 直接问，不给示例 | 简单任务，快速测试 | 基准 |
| **Few-shot** | 提供示例让模型模仿 | 格式要求、分类任务 | +10-20% |
| **Chain-of-Thought** | 展示推理过程 | 数学、逻辑推理 | +20-40% |
| **Self-Consistency** | 多次采样取多数 | 需要高可靠性 | +10-20% |
| **Role Prompting** | 角色扮演激活知识 | 专业领域任务 | 视情况 |
| **Tree-of-Thought** | 树状探索多种思路 | 复杂规划、创意 | +10-30% |

### 28.8.2 技术选择流程图

```
                    任务类型
                       │
        ┌──────────────┼──────────────┐
        │              │              │
     简单任务       中等任务       复杂任务
        │              │              │
        ▼              ▼              ▼
    Zero-shot      Few-shot       Few-shot
        │          + CoT         + CoT
        │              │         + Self-Consistency
        │              │              │
        ▼              ▼              ▼
    效果满意？     效果满意？     效果满意？
        │              │              │
    是 → 完成      是 → 完成    是 → 完成
        │              │              │
    否 → 加示例    否 → ToT      否 → 换模型/
                       或              人工介入
                   Self-Consistency
```

### 28.8.3 关键参数参考

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **Temperature** | 0-0.3（确定性）/ 0.7-1.0（创意） | 控制随机性 |
| **Few-shot 示例数** | 3-8 | 平衡效果和 context 消耗 |
| **Self-Consistency 采样数** | 5-10 | 平衡准确性和成本 |
| **Max tokens** | 按需设置 | 避免截断或浪费 |

### 28.8.4 核心认知

> **Prompt Engineering 的本质**：大模型是"续写机器"，好的 prompt 就是为模型创造一个"容易续写出正确答案"的上下文。Few-shot 通过示例展示格式和逻辑，CoT 通过中间步骤降低推理难度，Self-Consistency 通过多次采样提高可靠性——这些技术的共同点是**帮助模型更好地"续写"**。

---

## 本章交付物

学完这一章，你应该能够：

- [ ] 解释 Zero-shot、One-shot、Few-shot 的区别和适用场景
- [ ] 使用 Chain-of-Thought 提升模型推理能力
- [ ] 实现 Self-Consistency 提高答案可靠性
- [ ] 运用角色扮演、格式约束等技巧
- [ ] 为不同任务设计合适的 prompt 策略
- [ ] 调试和优化 prompt 效果

---

## 延伸阅读

- **Chain-of-Thought 原论文**：[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- **Self-Consistency 原论文**：[Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)
- **Tree-of-Thought 原论文**：[Tree of Thoughts: Deliberate Problem Solving with LLMs](https://arxiv.org/abs/2305.10601)
- **Prompt Engineering Guide**：[Prompt Engineering Guide](https://www.promptingguide.ai/)

---

## 下一章预告

Prompt Engineering 是让模型"更好用"的技术，但如果我们想让模型"更听话"呢？

下一章，我们将深入 **RLHF（Reinforcement Learning from Human Feedback）** 和 **DPO（Direct Preference Optimization）**，理解 ChatGPT 为什么能够如此"对齐"人类意图。
