#!/bin/bash
# Build script for "Transformer 架构：从直觉到实现"
# Generates PDF using pandoc + typst

set -e

BOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BOOK_DIR"

OUTPUT_DIR="$BOOK_DIR/output"
mkdir -p "$OUTPUT_DIR"

echo "=== Building: Transformer 架构：从直觉到实现 ==="
echo "Book directory: $BOOK_DIR"

# Check dependencies
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc not found. Install with: brew install pandoc"
    exit 1
fi

if ! command -v typst &> /dev/null; then
    echo "Error: typst not found. Install with: brew install typst"
    exit 1
fi

# Create metadata file for pandoc
cat > "$OUTPUT_DIR/metadata.yaml" << 'EOF'
---
title: "Transformer 架构：从直觉到实现"
subtitle: "从直觉到代码，彻底搞懂 GPT"
author: "Wayland Zhang（张老师）"
date: "2025年1月"
lang: zh-CN
documentclass: book
mainfont: "Heiti SC"
sansfont: "Heiti SC"
monofont: "SF Mono"
CJKmainfont: "Heiti SC"
toc: true
toc-depth: 3
number-sections: true
colorlinks: false
geometry:
  - margin=2.5cm
  - a4paper
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
---
EOF

# List of all chapters in order
CHAPTERS=(
    "前言.md"
    "Part1-建立直觉/第1章：GPT是什么 - LLM发展简史与核心思想.md"
    "Part1-建立直觉/第2章：大模型的本质 - 就是两个文件.md"
    "Part1-建立直觉/第3章：Transformer全景图.md"
    "Part2-核心组件/第4章：Tokenization - 文字如何变成数字.md"
    "Part2-核心组件/第5章：Positional Encoding - 给文字加位置.md"
    "Part2-核心组件/第6章：LayerNorm与Softmax - 数字的缩放与概率化.md"
    "Part2-核心组件/第7章：神经网络层 - 不需要懂也能理解Transformer.md"
    "Part3-Attention机制/第8章：线性变换的几何意义 - 矩阵乘法的本质.md"
    "Part3-Attention机制/第9章：Attention的几何逻辑 - 为什么是点积.md"
    "Part3-Attention机制/第10章：QKV到底是什么 - Attention的三个主角.md"
    "Part3-Attention机制/第11章：Multi-Head Attention - 多视角理解.md"
    "Part3-Attention机制/第12章：QKV输出的本质.md"
    "Part4-完整架构/第13章：残差连接与Dropout - 训练稳定的秘密.md"
    "Part4-完整架构/第14章：词嵌入与位置信息的深层逻辑 - 为什么相加而不是拼接.md"
    "Part4-完整架构/第15章：Transformer完整前向传播 - 从输入到输出.md"
    "Part4-完整架构/第16章：训练与推理的异同 - 为什么推理要一个字一个字生成.md"
    "Part4-完整架构/第17章：学习率的理解 - 训练稳定的关键.md"
    "Part5-代码实现/第18章：手写Model.py - 模型定义.md"
    "Part5-代码实现/第19章：手写Train.py - 训练循环.md"
    "Part5-代码实现/第20章：手写Inference.py - 推理逻辑.md"
    "Part6-生产优化/第21章：Flash Attention - 内存优化原理.md"
    "Part6-生产优化/第22章：KV Cache - 推理加速.md"
    "Part7-架构变体/第23章：MHA到MQA到GQA演进.md"
    "Part7-架构变体/第24章：Sparse与Infinite Attention.md"
    "Part7-架构变体/第25章：位置编码演进 - Sinusoidal到RoPE到ALiBi.md"
    "Part8-部署与微调/第26章：LoRA与QLoRA - 高效微调.md"
    "Part8-部署与微调/第27章：模型量化 - GPTQ, AWQ, GGUF.md"
    "Part9-前沿进展/第28章：Prompt Engineering - 提示工程实战.md"
    "Part9-前沿进展/第29章：RLHF与偏好学习 - 让模型对齐人类.md"
    "Part9-前沿进展/第30章：Mixture of Experts - 稀疏激活的秘密.md"
    "Part9-前沿进展/第31章：推理模型革命 - 从o1到R1.md"
    "Part9-前沿进展/第32章：后Transformer架构 - Mamba与混合模型.md"
    "附录/附录A：Scaling Law与计算量估算.md"
    "附录/附录B：解码策略详解.md"
    "附录/附录C：常见问题FAQ.md"
)

# Verify all files exist
echo ""
echo "Verifying chapter files..."
MISSING=0
for chapter in "${CHAPTERS[@]}"; do
    if [[ ! -f "$BOOK_DIR/$chapter" ]]; then
        echo "  [MISSING] $chapter"
        MISSING=$((MISSING + 1))
    fi
done

if [[ $MISSING -gt 0 ]]; then
    echo "Error: $MISSING chapter(s) missing!"
    exit 1
fi
echo "  All ${#CHAPTERS[@]} chapters found."

# Build file list for pandoc
FILE_LIST=()
for chapter in "${CHAPTERS[@]}"; do
    FILE_LIST+=("$BOOK_DIR/$chapter")
done

# Generate PDF with typst
echo ""
echo "Generating PDF with typst..."
OUTPUT_PDF="$OUTPUT_DIR/Transformer架构从直觉到实现.pdf"

pandoc \
    --from=markdown-citations \
    --pdf-engine=typst \
    --lua-filter="$OUTPUT_DIR/image-filter.lua" \
    --toc \
    --toc-depth=2 \
    --number-sections \
    --metadata-file="$OUTPUT_DIR/metadata.yaml" \
    --resource-path="$BOOK_DIR:$BOOK_DIR/Part1-建立直觉:$BOOK_DIR/Part2-核心组件:$BOOK_DIR/Part3-Attention机制:$BOOK_DIR/Part4-完整架构:$BOOK_DIR/Part5-代码实现:$BOOK_DIR/Part6-生产优化:$BOOK_DIR/Part7-架构变体:$BOOK_DIR/Part8-部署与微调:$BOOK_DIR/Part9-前沿进展:$BOOK_DIR/附录" \
    -V mainfont="Heiti SC" \
    -V sansfont="Heiti SC" \
    -V monofont="SF Mono" \
    -o "$OUTPUT_PDF" \
    "${FILE_LIST[@]}"

if [[ -f "$OUTPUT_PDF" ]]; then
    PDF_SIZE=$(du -h "$OUTPUT_PDF" | cut -f1)
    echo ""
    echo "=== Build Complete ==="
    echo "Output: $OUTPUT_PDF"
    echo "Size: $PDF_SIZE"
    echo ""
    echo "To open: open \"$OUTPUT_PDF\""
else
    echo "Error: PDF generation failed!"
    exit 1
fi
