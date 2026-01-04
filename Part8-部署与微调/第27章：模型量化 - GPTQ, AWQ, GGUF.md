# 第 27 章：模型量化 - GPTQ, AWQ, GGUF

> **一句话总结**：量化就是用更少的比特数表示权重，把 FP16 的 7B 模型从 14GB 压缩到 4-bit 的 3.5GB，让你在消费级硬件上也能跑大模型。GPTQ 精度高但量化慢，AWQ 保护重要权重效果更好，GGUF 是 CPU 推理的事实标准。

---

## 27.1 为什么需要量化？

### 27.1.1 大模型的内存困境

让我们算一笔账：

**LLaMA-7B 模型的内存需求**：
- 参数量：70 亿
- FP32 精度：7B × 4 bytes = **28GB**
- FP16 精度：7B × 2 bytes = **14GB**
- INT8 精度：7B × 1 byte = **7GB**
- INT4 精度：7B × 0.5 bytes = **3.5GB**

从 28GB 到 3.5GB，整整压缩了 **8 倍**！

**更大的模型更需要量化**：

| 模型 | FP16 显存 | INT4 显存 | 压缩比 |
|------|----------|----------|--------|
| LLaMA-7B | 14GB | 3.5GB | 4x |
| LLaMA-13B | 26GB | 6.5GB | 4x |
| LLaMA-70B | 140GB | 35GB | 4x |
| Mixtral-8x7B | 90GB | 22GB | 4x |

一块 RTX 4090（24GB）：
- 跑不了 FP16 的 13B 模型
- 可以轻松跑 INT4 的 70B 模型（如果用 CPU offloading）

### 27.1.2 量化不只是省内存

量化还能**加速推理**：

1. **内存带宽**：模型越小，从显存/内存读取越快
2. **计算效率**：INT8/INT4 运算比 FP16 更快（在支持的硬件上）
3. **缓存友好**：更小的模型更容易放进 CPU/GPU 缓存

实测数据（LLaMA-7B，RTX 3090）：

| 精度 | 显存占用 | 生成速度 (tokens/s) |
|------|---------|-------------------|
| FP16 | 14GB | 25 |
| INT8 | 7GB | 35 |
| INT4 | 4GB | 45 |

量化后速度反而**更快**，因为内存带宽是 LLM 推理的主要瓶颈。

### 27.1.3 量化的代价

天下没有免费的午餐。量化会带来**精度损失**：

- 原始权重：`0.12345678`（32-bit float，可表示约 7 位有效数字）
- 量化后：`0.125`（可能只有 2-3 位有效数字）

精度损失会累积，可能导致：
- 输出质量下降
- 某些任务性能退化
- 极端情况下输出乱码

好消息是：**现代量化技术（GPTQ、AWQ）的精度损失很小**，在大多数任务上几乎察觉不到。

---

## 27.2 量化基础概念

### 27.2.1 什么是量化？

量化（Quantization）是将**连续的浮点数**映射到**离散的整数**的过程。

```
原始权重（FP16）：-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, ...
量化后（INT4）：  -8,  0,   2,    4,   6,    7,   ...
```

INT4 只有 16 个可能的值（-8 到 7 或 0 到 15），而 FP16 有 65536 个可能的值。

### 27.2.2 线性量化

最简单的量化方式是**线性量化**：

```
量化值 = round((原始值 - 零点) / 缩放因子)
反量化 = 量化值 × 缩放因子 + 零点
```

**例子**：将 [-1.0, 1.0] 的权重量化到 INT8 [-128, 127]

```python
# 量化参数
scale = 2.0 / 255  # (max - min) / (2^8 - 1)
zero_point = 0

# 量化
original = 0.5
quantized = round(original / scale) = round(0.5 / 0.00784) = 64

# 反量化
dequantized = 64 * 0.00784 = 0.50176  # 有误差！
```

### 27.2.3 对称量化 vs 非对称量化

**对称量化**（Symmetric）：
- 零点固定为 0
- 公式：`q = round(x / scale)`
- 适合权重分布对称的情况

**非对称量化**（Asymmetric）：
- 零点可以是任意值
- 公式：`q = round(x / scale) + zero_point`
- 更灵活，精度可能更高

### 27.2.4 量化粒度

量化的**粒度**（granularity）影响精度和效率的权衡：

**1. Per-Tensor 量化**（每张量一个 scale）
```
整个权重矩阵共享一个 scale 和 zero_point
优点：简单高效
缺点：精度较低
```

**2. Per-Channel 量化**（每通道一个 scale）
```
每个输出通道有自己的 scale 和 zero_point
优点：精度更高
缺点：需要更多存储
```

**3. Per-Group 量化**（每组一个 scale）
```
把权重分成小组（如每 128 个权重一组）
每组有自己的 scale 和 zero_point
优点：精度和效率的平衡
缺点：实现更复杂
```

GPTQ 和 AWQ 通常使用 **Per-Group 量化**，group size 常见值是 128。

### 27.2.5 常见量化位数

| 位数 | 取值范围 | 压缩比 | 精度 | 用途 |
|------|---------|--------|------|------|
| INT8 | -128 ~ 127 | 2x | 高 | 服务端推理 |
| INT4 | -8 ~ 7 | 4x | 中 | 消费级推理 |
| INT3 | -4 ~ 3 | 5.3x | 低 | 极端压缩 |
| INT2 | -2 ~ 1 | 8x | 很低 | 实验性 |

**实践建议**：
- **INT8**：追求高精度，显存充足
- **INT4**：最常用，精度和大小的最佳平衡
- **INT3/INT2**：仅在极端内存限制下使用

---

## 27.3 GPTQ：后训练量化的标杆

### 27.3.1 GPTQ 是什么？

GPTQ（GPT Quantization）是 2022 年提出的一种**后训练量化**（Post-Training Quantization, PTQ）方法。

核心思想：**用少量校准数据，在量化时补偿精度损失**。

与简单的"直接量化"不同，GPTQ 会：
1. 分析权重的重要性
2. 按顺序量化每一列
3. 量化一列后，调整剩余列来补偿误差

### 27.3.2 OBQ 算法

GPTQ 基于 **OBQ**（Optimal Brain Quantization）算法：

```
目标：最小化量化前后输出的差异

argmin_W_q ||WX - W_q X||^2

其中：
- W 是原始权重
- W_q 是量化后权重
- X 是输入激活值（来自校准数据）
```

OBQ 的关键步骤：

1. **计算 Hessian 矩阵**：`H = 2 X X^T`
   - 这个矩阵描述了每个权重对输出的影响

2. **贪心量化**：
   - 找到当前"最容易量化"的权重（Hessian 对角线最大）
   - 量化这个权重
   - 更新其他权重来补偿误差

3. **迭代**直到所有权重都被量化

### 27.3.3 GPTQ 的加速技巧

原始 OBQ 对每个权重单独处理，太慢了。GPTQ 的创新：

**1. 按列批量处理**

不是一个一个权重量化，而是一次量化一整列（128 个权重）。

**2. 懒惰批更新**（Lazy Batch Updates）

不是每量化一个就更新 Hessian，而是累积多次更新后批量处理。

**3. Cholesky 分解**

用 Cholesky 分解加速 Hessian 逆矩阵的计算。

这些优化让 GPTQ 可以在**几小时内量化 175B 的模型**。

### 27.3.4 GPTQ 的量化过程

```
输入：FP16 模型 + 校准数据集（几百条样本）
输出：INT4 量化模型

步骤：
1. 加载模型到 GPU
2. 用校准数据计算每层的激活值
3. 对每一层：
   a. 计算 Hessian 矩阵 H = X @ X.T
   b. 对 Hessian 做 Cholesky 分解
   c. 按列顺序量化权重，同时更新未量化列
4. 保存量化后的权重和量化参数（scale, zero_point）
```

### 27.3.5 使用 AutoGPTQ

```python
# 代码示例：使用 AutoGPTQ 量化模型

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 1. 准备校准数据
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
calibration_data = [
    tokenizer("Hello, how are you?", return_tensors="pt"),
    tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt"),
    # ... 更多样本（通常需要 128-512 条）
]

# 2. 配置量化参数
quantize_config = BaseQuantizeConfig(
    bits=4,                  # 量化到 4-bit
    group_size=128,          # 每 128 个权重共享一个 scale
    desc_act=True,           # 激活值降序排列（提高精度）
    sym=False,               # 非对称量化
)

# 3. 加载模型并量化
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config,
)
model.quantize(calibration_data)

# 4. 保存量化模型
model.save_quantized("./llama-7b-gptq-4bit")
tokenizer.save_pretrained("./llama-7b-gptq-4bit")
```

### 27.3.6 加载 GPTQ 模型

```python
# 代码示例：加载 GPTQ 模型进行推理

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

# 加载量化模型
model = AutoGPTQForCausalLM.from_quantized(
    "./llama-7b-gptq-4bit",
    device="cuda:0",
    use_safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained("./llama-7b-gptq-4bit")

# 推理
inputs = tokenizer("Hello, I am", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 27.3.7 GPTQ 的优缺点

**优点**：
- 精度高（接近原始模型 99%+）
- 量化后推理快（尤其配合 ExLlama）
- 生态成熟，模型多

**缺点**：
- 量化过程慢（几小时）
- 需要 GPU 来量化
- 需要校准数据

---

## 27.4 AWQ：激活感知量化

### 27.4.1 AWQ 的核心洞察

AWQ（Activation-aware Weight Quantization）的核心发现：

> **不是所有权重都同等重要**。一小部分权重（约 1%）对模型输出影响巨大，保护这些权重可以大幅提高量化精度。

怎么找到"重要"的权重？看**激活值**！

如果某个权重对应的输入激活值很大，那这个权重的量化误差会被放大。

### 27.4.2 保护重要权重

AWQ 的策略：**不直接量化重要权重，而是缩放它们**。

```
原始：W @ X
AWQ：(W × s) @ (X / s)

其中 s 是缩放因子
```

对于重要权重，用更大的 s：
- W × s 变大，量化时相对误差变小
- X / s 变小，但这个缩放可以融入前一层的权重

### 27.4.3 自动搜索最优缩放因子

AWQ 用网格搜索找最优的缩放因子：

```python
# 伪代码：AWQ 缩放因子搜索
def find_best_scale(W, X, n_bits):
    best_scale = 1.0
    best_loss = float('inf')

    for alpha in [0.1, 0.2, ..., 0.9, 1.0]:
        # 计算缩放因子
        scale = X.abs().mean() ** alpha

        # 量化
        W_scaled = W * scale
        W_quant = quantize(W_scaled, n_bits)
        W_dequant = dequantize(W_quant) / scale

        # 计算损失
        loss = ||W @ X - W_dequant @ X||^2

        if loss < best_loss:
            best_loss = loss
            best_scale = scale

    return best_scale
```

### 27.4.4 使用 AutoAWQ

```python
# 代码示例：使用 AutoAWQ 量化模型

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 1. 加载模型
model_path = "meta-llama/Llama-2-7b-hf"
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. 配置量化参数
quant_config = {
    "zero_point": True,      # 使用零点（非对称量化）
    "q_group_size": 128,     # 组大小
    "w_bit": 4,              # 4-bit 量化
    "version": "GEMM",       # 使用 GEMM 后端
}

# 3. 量化
model.quantize(tokenizer, quant_config=quant_config)

# 4. 保存
model.save_quantized("./llama-7b-awq-4bit")
tokenizer.save_pretrained("./llama-7b-awq-4bit")
```

### 27.4.5 AWQ vs GPTQ

| 特性 | GPTQ | AWQ |
|------|------|-----|
| 量化速度 | 慢（几小时） | 快（几十分钟） |
| 精度 | 很高 | 很高（某些任务更好） |
| 推理速度 | 快 | 快 |
| 校准数据需求 | 需要 | 需要（但更少） |
| 实现复杂度 | 复杂 | 中等 |

**实践建议**：
- 追求最高精度：先试 AWQ，再试 GPTQ，选效果好的
- 追求量化速度：AWQ
- 需要兼容性：GPTQ（生态更成熟）

---

## 27.5 GGUF：CPU 推理的事实标准

### 27.5.1 GGUF 是什么？

GGUF（GPT-Generated Unified Format）是 **llama.cpp** 项目使用的模型格式。

它不是一种量化算法，而是一种**模型文件格式**，专为 CPU 推理优化。

GGUF 的前身是 GGML，后来升级为 GGUF 以支持更多功能。

### 27.5.2 GGUF 的特点

**1. CPU 优先**
- 针对 CPU 推理优化（也支持 GPU 加速）
- 使用 SIMD 指令加速计算
- 支持内存映射（mmap），减少加载时间

**2. 多种量化级别**

GGUF 支持多种量化类型：

| 类型 | 位数 | 说明 | 推荐场景 |
|------|------|------|---------|
| Q2_K | ~2.5 | 极度压缩 | 极端内存限制 |
| Q3_K_S | ~3.0 | 小型量化 | 低内存 |
| Q3_K_M | ~3.3 | 中型量化 | 低内存 |
| Q4_0 | 4.0 | 基础 4-bit | 通用 |
| Q4_K_S | ~4.5 | 小型 K-量化 | 通用 |
| Q4_K_M | ~4.8 | 中型 K-量化 | 推荐 |
| Q5_0 | 5.0 | 基础 5-bit | 高精度 |
| Q5_K_S | ~5.5 | 小型 K-量化 | 高精度 |
| Q5_K_M | ~5.8 | 中型 K-量化 | 推荐 |
| Q6_K | 6.0 | 6-bit | 最高精度 |
| Q8_0 | 8.0 | 8-bit | 接近原始 |
| F16 | 16.0 | 半精度 | 无损 |

**K-量化**（K-quant）是一种改进的量化方式，对不同层使用不同的量化策略，精度更高。

**3. 单文件格式**

所有内容（权重、分词器、元数据）都在一个 `.gguf` 文件中，方便分发和管理。

### 27.5.3 量化类型详解

让我们深入理解几种常用的量化类型：

**Q4_0**：最基础的 4-bit 量化
```
每 32 个权重共享一个 scale（FP16）
存储：32 × 4 bits + 16 bits = 144 bits
平均每权重：4.5 bits
```

**Q4_K_M**：改进的 4-bit K-量化
```
不同类型的层使用不同量化：
- 重要层（如 attention 的 Q、K）：更高精度
- 不重要层（如 FFN 中间层）：更低精度
整体平均约 4.8 bits
```

**Q5_K_M**：推荐的高精度选择
```
类似 Q4_K_M，但基础是 5-bit
精度接近 FP16，文件大小约为 FP16 的 35%
```

### 27.5.4 使用 llama.cpp 转换模型

```bash
# 1. 克隆 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 2. 编译
make -j

# 3. 转换 HuggingFace 模型到 GGUF
python convert.py /path/to/llama-7b --outfile llama-7b-f16.gguf --outtype f16

# 4. 量化
./quantize llama-7b-f16.gguf llama-7b-q4_k_m.gguf Q4_K_M
```

### 27.5.5 使用 llama.cpp 推理

```bash
# 命令行推理
./main -m llama-7b-q4_k_m.gguf \
       -p "Hello, I am" \
       -n 128 \
       --temp 0.7

# 启动 API 服务器
./server -m llama-7b-q4_k_m.gguf --host 0.0.0.0 --port 8080
```

### 27.5.6 使用 llama-cpp-python

```python
# 代码示例：使用 llama-cpp-python

from llama_cpp import Llama

# 加载模型
llm = Llama(
    model_path="./llama-7b-q4_k_m.gguf",
    n_ctx=4096,           # 上下文长度
    n_gpu_layers=35,      # GPU 加速层数（0 = 纯 CPU）
    n_threads=8,          # CPU 线程数
)

# 生成
output = llm(
    "Hello, I am",
    max_tokens=128,
    temperature=0.7,
    stop=["</s>"],
)

print(output["choices"][0]["text"])
```

### 27.5.7 GGUF 的优缺点

**优点**：
- CPU 推理性能优秀
- 支持 GPU 加速（CUDA, Metal, Vulkan）
- 文件格式简洁，易于分发
- 社区活跃，更新快
- 支持多种量化级别

**缺点**：
- 生态相对独立（不是 HuggingFace 原生格式）
- 某些高级功能（如 LoRA）支持有限
- 量化精度可能略低于 GPTQ/AWQ

---

## 27.6 其他量化方法

### 27.6.1 bitsandbytes (BNB)

HuggingFace 集成的量化方案，支持 **INT8** 和 **NF4**（4-bit）。

```python
# 代码示例：使用 bitsandbytes 加载 4-bit 模型

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,   # 双重量化
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

**NF4**（NormalFloat4）是专门为神经网络权重设计的 4-bit 数据类型，假设权重服从正态分布。

### 27.6.2 SmoothQuant

针对 **INT8** 量化的优化方法。

核心思想：将量化难度从激活值"平滑"到权重。

```
原始：Y = X @ W
SmoothQuant：Y = (X / s) @ (W × s)
```

通过选择合适的 s，让 X/s 和 W×s 都更容易量化。

### 27.6.3 EETQ

高效 INT8 量化，针对推理速度优化。

```python
# 使用 EETQ
from transformers import AutoModelForCausalLM, EetqConfig

eetq_config = EetqConfig("int8")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=eetq_config,
    device_map="auto",
)
```

### 27.6.4 HQQ

无需校准数据的快速量化方法。

```python
# 使用 HQQ
from transformers import AutoModelForCausalLM, HqqConfig

hqq_config = HqqConfig(nbits=4, group_size=128)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=hqq_config,
    device_map="auto",
)
```

---

## 27.7 量化方法对比

### 27.7.1 全面对比

| 方法 | 位数 | 量化速度 | 推理速度 | 精度 | 需要校准数据 | CPU 支持 |
|------|------|---------|---------|------|-------------|---------|
| GPTQ | 4/3/2 | 慢 | 快 | 高 | 是 | 差 |
| AWQ | 4 | 中 | 快 | 高 | 是 | 差 |
| GGUF | 2-8 | 快 | 中 | 中-高 | 否 | 优秀 |
| BNB | 8/4 | 即时 | 中 | 中 | 否 | 差 |
| HQQ | 4/3/2 | 快 | 中 | 中 | 否 | 中 |

### 27.7.2 精度对比（Perplexity）

以 LLaMA-7B 在 WikiText-2 上的困惑度为例：

| 方法 | FP16 | INT8 | INT4 |
|------|------|------|------|
| 原始 | 5.68 | - | - |
| GPTQ | - | 5.70 | 5.85 |
| AWQ | - | 5.69 | 5.78 |
| GGUF Q4_K_M | - | - | 5.92 |
| BNB NF4 | - | 5.72 | 6.05 |

**困惑度越低越好**。可以看到 AWQ 和 GPTQ 的精度损失很小。

### 27.7.3 选择建议

**你的场景 → 推荐方案**：

| 场景 | 推荐 | 理由 |
|------|------|------|
| GPU 推理，追求精度 | AWQ/GPTQ | 精度最高 |
| GPU 推理，快速部署 | BNB (NF4) | 无需预先量化 |
| CPU 推理 | GGUF (Q4_K_M) | CPU 优化 |
| Apple Silicon | GGUF + Metal | llama.cpp 支持 Metal |
| 极端内存限制 | GGUF Q2_K/Q3_K | 压缩比最高 |
| 服务端高吞吐 | GPTQ + ExLlama | 推理速度最快 |

---

## 27.8 实践指南

### 27.8.1 量化前的检查清单

1. **确定目标硬件**
   - GPU：选 GPTQ/AWQ
   - CPU：选 GGUF
   - Apple Silicon：GGUF + Metal

2. **确定精度需求**
   - 高精度：Q5_K_M 或 AWQ
   - 平衡：Q4_K_M 或 GPTQ-4bit
   - 极限压缩：Q2_K/Q3_K

3. **准备校准数据**（GPTQ/AWQ）
   - 128-512 条代表性样本
   - 覆盖目标任务的典型输入

### 27.8.2 量化后的验证

```python
# 代码示例：量化后验证

def evaluate_quantized_model(original_model, quantized_model, test_prompts):
    """比较原始模型和量化模型的输出"""
    results = []

    for prompt in test_prompts:
        # 原始模型输出
        original_output = original_model.generate(prompt, max_new_tokens=100)

        # 量化模型输出
        quantized_output = quantized_model.generate(prompt, max_new_tokens=100)

        # 比较
        results.append({
            "prompt": prompt,
            "original": original_output,
            "quantized": quantized_output,
            "match": original_output == quantized_output,
        })

    return results

# 验证要点：
# 1. 输出是否合理（不是乱码）
# 2. 关键任务的准确率
# 3. 生成文本的质量
```

### 27.8.3 常见问题解决

**问题 1：量化后输出乱码**

可能原因：
- 量化位数太低
- 校准数据不合适
- 模型不适合量化

解决：
- 尝试更高位数（Q5 而不是 Q4）
- 使用更多/更好的校准数据
- 尝试不同的量化方法

**问题 2：量化后速度反而变慢**

可能原因：
- 硬件不支持高效的低精度计算
- 没有使用正确的推理后端

解决：
- GPTQ：使用 ExLlama/ExLlamaV2
- AWQ：使用 AutoAWQ 的 GEMM 后端
- GGUF：编译时启用 CUDA/Metal 支持

**问题 3：显存占用没有减少**

可能原因：
- 模型是"假量化"（权重量化但计算时反量化）
- 使用了错误的加载方式

解决：
- 确保使用正确的加载 API
- 检查是否真正加载了量化权重

---

## 27.9 本章总结

### 27.9.1 核心概念

| 概念 | 说明 |
|------|------|
| **量化** | 用更少的比特表示权重 |
| **GPTQ** | 后训练量化，用校准数据补偿误差 |
| **AWQ** | 激活感知量化，保护重要权重 |
| **GGUF** | llama.cpp 的模型格式，CPU 友好 |
| **K-量化** | 对不同层使用不同量化策略 |

### 27.9.2 压缩比速查

| 原始精度 | 量化精度 | 压缩比 | 7B 模型大小 |
|---------|---------|--------|------------|
| FP16 | INT8 | 2x | 7GB |
| FP16 | INT4 | 4x | 3.5GB |
| FP16 | INT3 | 5.3x | 2.6GB |
| FP16 | INT2 | 8x | 1.75GB |

### 27.9.3 选择决策树

```
你的硬件是什么？
├── NVIDIA GPU
│   └── 追求精度还是速度？
│       ├── 精度：AWQ
│       └── 速度：GPTQ + ExLlama
├── CPU
│   └── GGUF (Q4_K_M 或 Q5_K_M)
├── Apple Silicon
│   └── GGUF + Metal
└── 混合（GPU + CPU offload）
    └── GGUF (n_gpu_layers 调节)
```

### 27.9.4 核心认知

> **量化是让大模型"平民化"的关键技术。通过将 FP16 权重压缩到 INT4，我们可以在消费级显卡甚至 CPU 上运行曾经只有数据中心才能负担的模型。GPTQ 和 AWQ 代表了 GPU 量化的最高水平，而 GGUF 则是 CPU 推理的事实标准。选择合适的量化方案，让大模型为每个人所用。**

---

## 本章交付物

学完这一章，你应该能够：

- [ ] 解释量化的基本原理（为什么能压缩，代价是什么）
- [ ] 区分对称/非对称量化、不同粒度的量化
- [ ] 理解 GPTQ 的 OBQ 算法思想
- [ ] 理解 AWQ 的激活感知策略
- [ ] 使用 GGUF 格式进行 CPU 推理
- [ ] 为不同场景选择合适的量化方案

---

## 延伸阅读

- **GPTQ 论文**：[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- **AWQ 论文**：[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- **llama.cpp**：[GitHub - ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **AutoGPTQ**：[GitHub - AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)
- **AutoAWQ**：[GitHub - AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

---

## Part 8 总结

恭喜！你已经完成了**部署与微调**部分：

| 章节 | 内容 | 核心技术 |
|------|------|---------|
| 第 26 章 | LoRA 与 QLoRA | 低秩适应，高效微调 |
| 第 27 章 | 模型量化 | GPTQ、AWQ、GGUF |

这两章的技术让大模型真正**落地可用**：
- LoRA/QLoRA：让你在消费级显卡上微调大模型
- 量化：让你在普通硬件上运行大模型

掌握了这些技术，你就拥有了完整的大模型部署工具箱。

---

## 下一步

主体内容到此完成！接下来是附录部分：
- 附录 A：Scaling Law 与计算量估算
- 附录 B：解码策略详解
- 附录 C：常见问题 FAQ
