"""
Benchmark script for AN1 LLM operations.

Measures real GPU performance for attention and matmul operations,
comparing AN1-accelerated backend against baseline PyTorch.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import numpy as np
from tabulate import tabulate

from an1_cache import public_api, reference, gpu_backend


def create_attention_tensors(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random Q, K, V tensors for attention."""
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    return q, k, v


def benchmark_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: str,
    warmup: int = 10,
    iters: int = 100,
    causal: bool = True,
) -> Optional[Dict[str, float]]:
    """
    Benchmark attention operation.
    
    Returns:
        Dictionary with 'latency_ms', 'tokens_per_sec', and 'tflops' keys, or None if backend unavailable.
    """
    batch_size, n_heads, seq_len, head_dim = q.shape
    
    # Warmup
    try:
        for _ in range(warmup):
            if backend == "baseline":
                _ = reference.llm_attention(q, k, v, causal=causal)
            else:
                _ = public_api.llm_attention(q, k, v, backend="an1", causal=causal)
            torch.cuda.synchronize()
    except Exception as e:
        print(f"  Warning: Backend '{backend}' failed during warmup: {e}")
        return None
    
    # Timed runs
    events = []
    for _ in range(iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        try:
            start_event.record()
            if backend == "baseline":
                _ = reference.llm_attention(q, k, v, causal=causal)
            else:
                _ = public_api.llm_attention(q, k, v, backend="an1", causal=causal)
            end_event.record()
            
            torch.cuda.synchronize()
            events.append((start_event, end_event))
        except Exception as e:
            print(f"  Warning: Backend '{backend}' failed during timing: {e}")
            return None
    
    if not events:
        return None
    
    # Compute statistics
    times_ms = [start.elapsed_time(end) for start, end in events]
    avg_latency_ms = np.mean(times_ms)
    std_latency_ms = np.std(times_ms)
    min_latency_ms = np.min(times_ms)
    max_latency_ms = np.max(times_ms)
    
    # Compute tokens per second
    total_tokens = batch_size * seq_len
    tokens_per_sec = (total_tokens / (avg_latency_ms / 1000.0))
    
    # Compute TFLOPs (approximate for attention)
    # Attention: 2 * batch * n_heads * seq_len^2 * head_dim (QK^T)
    #          + 2 * batch * n_heads * seq_len^2 * head_dim (softmax * V)
    flops = 4 * batch_size * n_heads * (seq_len * seq_len) * head_dim
    tflops = (flops / 1e12) / (avg_latency_ms / 1000.0)
    
    return {
        "latency_ms": avg_latency_ms,
        "latency_std_ms": std_latency_ms,
        "latency_min_ms": min_latency_ms,
        "latency_max_ms": max_latency_ms,
        "tokens_per_sec": tokens_per_sec,
        "tflops": tflops,
    }


def save_results(results: Dict[str, Any], output_dir: Path):
    """Save benchmark results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "llm_bench_results.csv"
    
    file_exists = csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark AN1 LLM operations")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--n-heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                        help="Data type")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Timing iterations")
    parser.add_argument("--backend-baseline", action="store_true",
                        help="Run baseline PyTorch benchmark")
    parser.add_argument("--backend-an1", action="store_true",
                        help="Run AN1 accelerated benchmark (if available)")
    parser.add_argument("--output-dir", type=str, default="bench_logs",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # If no backend specified, run both
    if not args.backend_baseline and not args.backend_an1:
        args.backend_baseline = True
        args.backend_an1 = True
    
    # Setup
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires GPU.")
        return
    
    device = "cuda"
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    print("=" * 70)
    print("AN1 LLM Benchmark")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Configuration:")
    print(f"  batch_size={args.batch_size}, seq_len={args.seq_len}")
    print(f"  n_heads={args.n_heads}, head_dim={args.head_dim}, dtype={args.dtype}")
    print(f"  warmup={args.warmup}, iterations={args.iters}")
    print()
    
    # Check backend availability
    an1_available = gpu_backend.is_available()
    license_key_set = os.getenv("AN1_LICENSE_KEY") is not None
    print(f"Backend Status:")
    print(f"  AN1 backend available: {an1_available}")
    print(f"  License key set: {license_key_set}")
    print()
    
    # Create tensors
    q, k, v = create_attention_tensors(
        args.batch_size, args.n_heads, args.seq_len, args.head_dim, dtype, device
    )
    
    results = {}
    baseline_results = None
    an1_results = None
    
    # Benchmark baseline
    if args.backend_baseline:
        print("Benchmarking baseline (PyTorch)...")
        baseline_results = benchmark_attention(q, k, v, "baseline", args.warmup, args.iters)
        if baseline_results:
            print(f"  ✓ Latency: {baseline_results['latency_ms']:.3f} ms")
            print(f"  ✓ Tokens/sec: {baseline_results['tokens_per_sec']:.0f}")
            print(f"  ✓ TFLOPs: {baseline_results['tflops']:.2f}")
            results.update({
                "baseline_latency_ms": baseline_results['latency_ms'],
                "baseline_tflops": baseline_results['tflops'],
            })
        else:
            print("  ✗ Baseline benchmark failed")
        print()
    
    # Benchmark AN1
    if args.backend_an1:
        print("Benchmarking AN1 accelerated...")
        an1_results = benchmark_attention(q, k, v, "an1", args.warmup, args.iters)
        if an1_results:
            print(f"  ✓ Latency: {an1_results['latency_ms']:.3f} ms")
            print(f"  ✓ Tokens/sec: {an1_results['tokens_per_sec']:.0f}")
            print(f"  ✓ TFLOPs: {an1_results['tflops']:.2f}")
            results.update({
                "an1_latency_ms": an1_results['latency_ms'],
                "an1_tflops": an1_results['tflops'],
            })
        else:
            print("  ✗ AN1 benchmark failed (backend not available or license invalid)")
        print()
    
    # Print summary table
    if baseline_results and an1_results:
        speedup = baseline_results['latency_ms'] / an1_results['latency_ms']
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        
        table_data = [
            ["Backend", "Latency (ms)", "Tokens/sec", "TFLOPs"],
            ["Baseline", f"{baseline_results['latency_ms']:.3f}", 
             f"{baseline_results['tokens_per_sec']:.0f}", 
             f"{baseline_results['tflops']:.2f}"],
            ["AN1", f"{an1_results['latency_ms']:.3f}", 
             f"{an1_results['tokens_per_sec']:.0f}", 
             f"{an1_results['tflops']:.2f}"],
            ["Speedup", f"{speedup:.2f}x", "-", "-"],
        ]
        
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
        print()
        
        results.update({
            "speedup": speedup,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_heads": args.n_heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
        })
        
        # Save results
        save_results(results, Path(args.output_dir))
        print(f"Results saved to {args.output_dir}/llm_bench_results.csv")
    elif baseline_results:
        print("=" * 70)
        print("Summary (Baseline only)")
        print("=" * 70)
        print(f"Latency: {baseline_results['latency_ms']:.3f} ms")
        print(f"Tokens/sec: {baseline_results['tokens_per_sec']:.0f}")
        print(f"TFLOPs: {baseline_results['tflops']:.2f}")
    elif an1_results:
        print("=" * 70)
        print("Summary (AN1 only)")
        print("=" * 70)
        print(f"Latency: {an1_results['latency_ms']:.3f} ms")
        print(f"Tokens/sec: {an1_results['tokens_per_sec']:.0f}")
        print(f"TFLOPs: {an1_results['tflops']:.2f}")


if __name__ == "__main__":
    main()
