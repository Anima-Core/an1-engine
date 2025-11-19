#!/bin/bash
# Benchmark script for H100 GPU
# Typical configuration for large-scale LLM inference

echo "AN1 LLM Benchmark - H100 Configuration"
echo "========================================"
echo ""
echo "Running benchmark with H100-optimized parameters:"
echo "  batch=16, seq_len=4096, n_heads=32, head_dim=128, dtype=bf16"
echo ""

# Check if license key is set
if [ -z "$AN1_LICENSE_KEY" ]; then
    echo "Warning: AN1_LICENSE_KEY not set. Will use baseline PyTorch implementation."
    echo ""
fi

python bench/llm_bench.py \
    --batch-size 16 \
    --seq-len 4096 \
    --n-heads 32 \
    --head-dim 128 \
    --dtype bf16 \
    --warmup 20 \
    --iters 200 \
    --op attention

echo ""
echo "Benchmark completed. Check bench_logs/llm_bench_results.csv for results."

