# 第 19 章：手写 Train.py - 训练循环

> **一句话总结**：训练循环就是：准备数据 → 前向传播 → 计算损失 → 反向传播 → 更新参数，不断重复。代码不到 100 行，但让模型从"一无所知"变成能预测下一个词。

> 📦 **完整代码仓库**：[github.com/waylandzhang/Transformer-from-scratch](https://github.com/waylandzhang/Transformer-from-scratch)

---

## 19.1 训练的本质

### 19.1.1 模型初始时是什么状态？

刚创建的模型，所有参数都是**随机初始化**的。让它预测下一个词，输出基本是乱猜。

```python
# 随机初始化的模型
model = Model(h_params)

# 输入 "农夫山泉"
input_ids = tokenizer.encode("农夫山泉")

# 模型输出：可能是任何乱七八糟的字符
output = model.generate(input_ids)
# 可能输出："农夫山泉睡觉月亮飞机汽车..."  # 完全随机
```

### 19.1.2 训练的目标

通过大量的"输入-目标"对，让模型**学会预测下一个词**。

```
输入：农 夫 山 泉
目标：夫 山 泉 天

模型需要学会：
- 看到"农"，预测"夫"
- 看到"农夫"，预测"山"
- 看到"农夫山"，预测"泉"
- 看到"农夫山泉"，预测"天"（或其他合理的续写）
```

### 19.1.3 训练循环的四步

```
1. 前向传播：输入数据，得到预测
2. 计算损失：预测 vs 目标，差多少？
3. 反向传播：损失对每个参数求梯度
4. 更新参数：朝着减少损失的方向调整
```

重复这四步，损失会逐渐下降，模型预测越来越准。

---

## 19.2 超参数配置

### 19.2.1 超参数字典

```python
# 超参数配置
h_params = {
    # 模型架构
    "d_model": 80,           # 嵌入维度（小模型用小值）
    "num_blocks": 6,         # Transformer 块数量
    "num_heads": 4,          # 注意力头数

    # 训练配置
    "batch_size": 2,         # 每次训练多少个样本
    "context_length": 128,   # 上下文长度（序列长度）
    "max_iters": 500,        # 训练多少步
    "learning_rate": 1e-3,   # 学习率

    # 正则化
    "dropout": 0.1,          # Dropout 概率

    # 评估配置
    "eval_interval": 50,     # 每多少步评估一次
    "eval_iters": 10,        # 评估时用多少个 batch

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 随机种子（可复现）
    "TORCH_SEED": 1337
}
```

### 19.2.2 关键超参数解释

| 超参数 | 作用 | 典型值 |
|-------|------|--------|
| `batch_size` | 每次训练的样本数 | 2-32（取决于显存） |
| `context_length` | 模型能"看到"多长的上下文 | 128-2048 |
| `learning_rate` | 参数更新的步长 | 1e-3 到 1e-5 |
| `max_iters` | 总共训练多少步 | 数百到数百万 |
| `dropout` | 随机丢弃的比例 | 0.1-0.3 |

---

## 19.3 数据准备

### 19.3.1 加载原始文本

```python
# 加载训练数据
with open('data/订单商品名称.csv', 'r', encoding="utf-8") as file:
    text = file.read()

print(f"文本长度：{len(text):,} 字符")
# 输出：文本长度：324,523 字符
```

### 19.3.2 Tokenization

```python
# 使用 TikToken 分词
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")
tokenized_text = tokenizer.encode(text)

print(f"Token 数量：{len(tokenized_text):,}")
# 输出：Token 数量：77,919
```

### 19.3.3 转为 Tensor 并分割数据集

```python
# 转换为 PyTorch Tensor
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=h_params['device'])

# 90% 训练，10% 验证
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

print(f"训练集：{len(train_data):,} tokens")
print(f"验证集：{len(val_data):,} tokens")
```

### 19.3.4 获取 Batch

```python
# 随机获取一个 batch
def get_batch(split: str):
    """
    获取一个 batch 的训练数据

    Args:
        split: 'train' 或 'valid'

    Returns:
        x: 输入 [batch_size, context_length]
        y: 目标 [batch_size, context_length]（右移一位）
    """
    data = train_data if split == 'train' else val_data

    # 随机选择起始位置
    idxs = torch.randint(
        low=0,
        high=len(data) - h_params['context_length'],
        size=(h_params['batch_size'],)
    )

    # 构建输入和目标
    x = torch.stack([data[idx:idx + h_params['context_length']] for idx in idxs])
    y = torch.stack([data[idx + 1:idx + h_params['context_length'] + 1] for idx in idxs])

    return x.to(h_params['device']), y.to(h_params['device'])
```

### 19.3.5 理解 x 和 y 的关系

```
假设 context_length = 8

原始数据：[农, 夫, 山, 泉, 天, 然, 水, 甜, 蜂, 蜜, ...]
              ↓
x（输入）：[农, 夫, 山, 泉, 天, 然, 水, 甜]
y（目标）：[夫, 山, 泉, 天, 然, 水, 甜, 蜂]

y 就是 x 右移一位。模型需要学会：x[i] → y[i]
```

---

## 19.4 损失函数

### 19.4.1 交叉熵损失

模型输出的是每个位置对词表中每个词的概率分布。我们用**交叉熵损失**来衡量预测和真实的差距。

```python
# 计算损失
loss = F.cross_entropy(
    input=logits_reshaped,    # 模型预测 [batch*seq, vocab_size]
    target=targets_reshaped   # 真实目标 [batch*seq]
)
```

### 19.4.2 损失越低越好

- **随机初始化**：损失约 10-11（接近 ln(vocab_size)）
- **训练后**：损失可以降到 2-4
- **过拟合**：训练损失很低，验证损失很高

---

## 19.5 评估函数

### 19.5.1 为什么需要评估？

训练损失下降不代表模型真的学好了——可能只是"背答案"（过拟合）。

我们需要在**验证集**上评估，看模型对没见过的数据表现如何。

### 19.5.2 评估代码

```python
# 评估函数
@torch.no_grad()  # 不计算梯度，节省内存
def estimate_loss():
    out = {}
    model.eval()  # 切换到评估模式（关闭 Dropout）

    for split in ['train', 'valid']:
        losses = torch.zeros(h_params['eval_iters'])

        for k in range(h_params['eval_iters']):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()  # 切换回训练模式
    return out
```

### 19.5.3 `model.train()` vs `model.eval()`

| 模式 | Dropout | BatchNorm |
|------|---------|-----------|
| `model.train()` | 随机丢弃 | 使用 batch 统计量 |
| `model.eval()` | 不丢弃 | 使用全局统计量 |

评估时必须用 `model.eval()`，否则结果会有随机性。

---

## 19.6 优化器

### 19.6.1 AdamW 优化器

```python
# 创建优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=h_params['learning_rate']
)
```

AdamW 是目前最常用的优化器，结合了：
- **Momentum**：考虑历史梯度方向
- **自适应学习率**：每个参数有自己的学习率
- **Weight Decay**：L2 正则化，防止过拟合

### 19.6.2 为什么选 AdamW？

| 优化器 | 优点 | 缺点 |
|-------|------|------|
| SGD | 简单，泛化好 | 收敛慢 |
| Adam | 收敛快 | 可能泛化不好 |
| **AdamW** | 收敛快 + 泛化好 | 略复杂 |

现代大模型训练几乎都用 AdamW。

---

## 19.7 训练循环

### 19.7.1 完整训练循环

```python
# 训练循环
for step in range(h_params['max_iters']):

    # 定期评估
    if step % h_params['eval_interval'] == 0 or step == h_params['max_iters'] - 1:
        losses = estimate_loss()
        print(f'Step: {step}, '
              f'Training Loss: {losses["train"]:.3f}, '
              f'Validation Loss: {losses["valid"]:.3f}')

    # 1. 获取一个 batch
    xb, yb = get_batch('train')

    # 2. 前向传播
    logits, loss = model(xb, yb)

    # 3. 反向传播
    optimizer.zero_grad(set_to_none=True)  # 清零梯度
    loss.backward()                         # 计算梯度

    # 4. 更新参数
    optimizer.step()
```

### 19.7.2 每一步详解

**`optimizer.zero_grad()`**：清除上一步的梯度。

PyTorch 默认会**累加**梯度，所以每步开始前要清零。

**`loss.backward()`**：反向传播，计算每个参数的梯度。

这是 PyTorch 自动微分的魔法——它会自动追踪所有计算，然后求导。

**`optimizer.step()`**：根据梯度更新参数。

```
参数_new = 参数_old - learning_rate × 梯度
```

---

## 19.8 训练输出示例

```
Step: 0, Training Loss: 10.847, Validation Loss: 10.852
Step: 50, Training Loss: 7.234, Validation Loss: 7.198
Step: 100, Training Loss: 5.421, Validation Loss: 5.456
Step: 150, Training Loss: 4.312, Validation Loss: 4.387
Step: 200, Training Loss: 3.876, Validation Loss: 3.921
Step: 250, Training Loss: 3.542, Validation Loss: 3.678
Step: 300, Training Loss: 3.298, Validation Loss: 3.512
Step: 350, Training Loss: 3.112, Validation Loss: 3.398
Step: 400, Training Loss: 2.987, Validation Loss: 3.287
Step: 450, Training Loss: 2.876, Validation Loss: 3.198
Step: 499, Training Loss: 2.798, Validation Loss: 3.145
```

可以看到：
- 损失从 ~10.8 下降到 ~2.8
- 验证损失略高于训练损失（正常，因为是没见过的数据）
- 如果验证损失开始上升，说明过拟合了

---

## 19.9 保存模型

### 19.9.1 保存检查点

```python
# 保存模型
import os

if not os.path.exists('model/'):
    os.makedirs('model/')

torch.save({
    'model_state_dict': model.state_dict(),
    'h_params': h_params
}, 'model/model.ckpt')

print("模型已保存到 model/model.ckpt")
```

### 19.9.2 保存什么？

| 内容 | 作用 |
|-----|------|
| `model.state_dict()` | 所有模型参数 |
| `h_params` | 超参数（加载时需要） |

保存超参数是为了之后加载时能用**相同的配置**重建模型。

---

## 19.10 完整 train.py 代码

```python
"""
Train a Transformer model
"""
import os
import torch
import tiktoken
from model import Model

# GPU 内存配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()

# 超参数
h_params = {
    "d_model": 80,
    "batch_size": 2,
    "context_length": 128,
    "num_blocks": 6,
    "num_heads": 4,
    "dropout": 0.1,
    "max_iters": 500,
    "learning_rate": 1e-3,
    "eval_interval": 50,
    "eval_iters": 10,
    "device": "cuda" if torch.cuda.is_available() else
              ("mps" if torch.backends.mps.is_available() else "cpu"),
    "TORCH_SEED": 1337
}
torch.manual_seed(h_params["TORCH_SEED"])

# 加载数据
with open('data/订单商品名称.csv', 'r', encoding="utf-8") as file:
    text = file.read()

# 分词
tokenizer = tiktoken.get_encoding("cl100k_base")
tokenized_text = tokenizer.encode(text)
max_token_value = max(tokenized_text) + 1
h_params['max_token_value'] = max_token_value
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=h_params['device'])

print(f"Total: {len(tokenized_text):,} tokens")

# 分割数据
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

# 初始化模型
model = Model(h_params).to(h_params['device'])


def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - h_params['context_length'],
                         size=(h_params['batch_size'],))
    x = torch.stack([data[idx:idx + h_params['context_length']] for idx in idxs])
    y = torch.stack([data[idx + 1:idx + h_params['context_length'] + 1] for idx in idxs])
    return x.to(h_params['device']), y.to(h_params['device'])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(h_params['eval_iters'])
        for k in range(h_params['eval_iters']):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=h_params['learning_rate'])

for step in range(h_params['max_iters']):
    if step % h_params['eval_interval'] == 0 or step == h_params['max_iters'] - 1:
        losses = estimate_loss()
        print(f'Step: {step}, Training Loss: {losses["train"]:.3f}, '
              f'Validation Loss: {losses["valid"]:.3f}')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 保存模型
if not os.path.exists('model/'):
    os.makedirs('model/')

torch.save({
    'model_state_dict': model.state_dict(),
    'h_params': h_params
}, 'model/model.ckpt')

print("Training complete. Model saved to model/model.ckpt")
```

---

## 19.11 可选：WandB 训练追踪

### 19.11.1 什么是 WandB？

[Weights & Biases](https://wandb.ai/) 是一个训练追踪工具，可以：
- 可视化损失曲线
- 记录超参数
- 对比不同实验

### 19.11.2 集成代码

```python
# WandB 集成（可选）
import wandb

# 初始化
run = wandb.init(
    project="LLMZhang_lesson_2",
    config={
        "d_model": h_params["d_model"],
        "batch_size": h_params["batch_size"],
        "context_length": h_params["context_length"],
        "max_iters": h_params["max_iters"],
        "learning_rate": h_params["learning_rate"],
    },
)

# 在训练循环中记录
for step in range(h_params['max_iters']):
    ...
    wandb.log({
        "train_loss": losses['train'].item(),
        "valid_loss": losses['valid'].item()
    })
```

---

## 19.12 本章总结

### 19.12.1 训练流程

```
1. 加载数据 → 分词 → 转 Tensor → 分割 train/val

2. 训练循环：
   for step in range(max_iters):
       x, y = get_batch('train')     # 获取数据
       logits, loss = model(x, y)    # 前向传播
       optimizer.zero_grad()         # 清零梯度
       loss.backward()               # 反向传播
       optimizer.step()              # 更新参数

3. 保存模型 → torch.save()
```

### 19.12.2 关键函数

| 函数 | 作用 |
|------|------|
| `get_batch()` | 随机获取一个 batch |
| `estimate_loss()` | 在 train/val 上评估损失 |
| `model.train()` | 切换到训练模式 |
| `model.eval()` | 切换到评估模式 |
| `loss.backward()` | 反向传播 |
| `optimizer.step()` | 更新参数 |

### 19.12.3 核心认知

> **train.py 不到 100 行代码，但实现了完整的训练流程。核心就是四步循环：前向传播 → 计算损失 → 反向传播 → 更新参数。PyTorch 的自动微分让我们只需要定义前向传播，反向传播自动完成。**

---

## 本章交付物

学完这一章，你应该能够：

- [ ] 理解训练循环的四个步骤
- [ ] 知道 x 和 y 的关系（右移一位）
- [ ] 理解 `model.train()` 和 `model.eval()` 的区别
- [ ] 能独立写出一个简单的训练脚本

---

## 完整代码

本章代码对应的完整实现可在 GitHub 获取：

> 📦 **[github.com/waylandzhang/Transformer-from-scratch](https://github.com/waylandzhang/Transformer-from-scratch)**

包含 `model.py`、`train.py`、`inference.py` 以及 step-by-step Jupyter notebook。

---

## 下一章预告

模型训练好了，参数已经保存。现在我们要用它来**生成文本**！

下一章，我们来写 **inference.py**：加载模型、输入 prompt、让模型自回归生成。看看它学到了什么！
