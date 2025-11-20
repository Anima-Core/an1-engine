# AN1 Engine: High-Performance LLM Operations

AN1 Engine is a public API and benchmark suite for the AN1 accelerated attention layer, delivering **validated 7.21×+ speedup** and **peak 11.3× acceleration** on NVIDIA H100 NVL GPUs. It provides accelerated implementations of attention and matrix multiplication operations with automatic fallback to baseline PyTorch implementations.

### AN1 backend

This repo contains:

- A readable PyTorch baseline  
- The public `an1_cache` API  
- Benchmarks and tests  

The actual acceleration comes from **AN1**, a closed binary backend (`an1_core_gpu`) that is **not** included here.

This repository does not contain the AN1 kernel source code or scheduling logic. All low-level implementations live in a separate closed-source package (`an1_core_gpu`).

When `an1_core_gpu` is installed and `AN1_LICENSE_KEY` is set, the same API automatically routes to the accelerated backend.

For pilot access or evaluation on your own H100 / A100 / L40S hardware, contact: **Ryan@animacore.ai** or **305-505-5268**.

---

## Features

- **Accelerated Attention**: Optimized scaled dot-product attention for transformer models  
- **Accelerated Matrix Multiplication**: High-performance matmul operations  
- **Automatic Backend Selection**: Seamlessly falls back to PyTorch when acceleration is unavailable  
- **License-Based Access**: Fast backend requires valid license key for enterprise partners  
- **Real GPU Performance**: All benchmarks use actual GPU execution with proper timing  
- **Production Ready**: Tested on H100, A100, L40S, and other modern GPUs  

---

## Installation

```bash
pip install -e .
```

For development with tests:

```bash
pip install -e ".[dev]"
```

---

## Quick Start

```python
import torch
from an1_cache import llm_attention

# Create input tensors: (batch, n_heads, seq_len, head_dim)
q = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)

# Compute attention (automatically uses AN1 backend if available and licensed)
output = llm_attention(q, k, v, causal=True)
```

---

## API Reference

### `llm_attention(q, k, v, *, backend="auto", causal=True, scale=None)`

Compute scaled dot-product attention.

**Arguments:**
- `q`: Query tensor, shape `(batch, n_heads, seq_len, head_dim)`
- `k`: Key tensor, shape `(batch, n_heads, seq_len, head_dim)`
- `v`: Value tensor, shape `(batch, n_heads, seq_len, head_dim)`
- `backend`: Backend to use (`"an1"`, `"baseline"`, or `"auto"`)
- `causal`: Apply causal masking if `True`
- `scale`: Optional attention scale. If `None`, uses `1/sqrt(head_dim)` for baseline only.

**Returns:**
- Output tensor of shape `(batch, n_heads, seq_len, head_dim)`

### `matmul(a, b, *, backend="auto")`

Matrix multiplication.

**Arguments:**
- `a`: Left tensor, shape `(..., m, k)`
- `b`: Right tensor, shape `(..., k, n)`
- `backend`: Backend to use (`"an1"`, `"baseline"`, or `"auto"`)

**Returns:**
- Output tensor of shape `(..., m, n)`

---

## Backend Selection

The library automatically selects the best available backend:

1. If `an1_core_gpu` is installed, `AN1_LICENSE_KEY` is set with a valid key, and `backend="auto"` (or `AN1_BACKEND=an1`), it uses the accelerated AN1 backend  
2. Otherwise, it falls back to the reference PyTorch implementation

You may explicitly control the backend:

```python
# Force AN1 backend (requires valid license key)
output = llm_attention(q, k, v, backend="an1")

# Force baseline PyTorch
output = llm_attention(q, k, v, backend="baseline")

# Automatic selection (default)
output = llm_attention(q, k, v, backend="auto")
```

---

## License Key Setup

To use the accelerated AN1 backend, you need:

1. Install the private `an1_core_gpu` wheel package  
2. Set the `AN1_LICENSE_KEY` environment variable with a valid license key  

```bash
export AN1_LICENSE_KEY="your-license-key-here"
python your_script.py
```

On Windows:

```cmd
set AN1_LICENSE_KEY=your-license-key-here
python your_script.py
```

If the license key is missing or invalid, the library automatically falls back to the baseline PyTorch implementation.

---

## Performance (H100 NVL example)

This repository ships a clear PyTorch baseline and a plug-in socket for the closed AN1 backend.

The numbers below come from the benchmark harness in `bench/llm_bench.py`, run on a **single NVIDIA H100 NVL** with PyTorch 2.8 and CUDA 12.9.

**Configuration**

- Device: NVIDIA H100 NVL  
- Batch size: 8  
- Sequence length: 2048  
- Attention heads: 32  
- Head dimension: 128  
- Dtype: fp16  
- Warmup: 10 iterations  
- Measured: 100 iterations  

**Results**

| Backend | Latency (ms) | Tokens / sec | TFLOPs | Speedup vs baseline |
|---------|-------------:|-------------:|-------:|--------------------:|
| Baseline (PyTorch) | 9.701 | 1,688,951 | 56.67 | 1.00× |
| **AN1 backend (private)** | **1.346** | **12,171,685** | **408.41** | **7.21×+ validated** |

> Additional benchmarking runs show **peak acceleration of 11.3×** on the same hardware under alternative scheduling configurations.

---

### 🌱 Energy Efficiency: H100 NVL Power + Speed Results

AN1 was profiled using NVIDIA NVML (`pynvml`) to measure power draw during attention workloads.  
A second benchmark pass on H100 NVL captured both **latency** and **power draw**, using the same public API and a valid AN1 license key.

**Configuration (same as baseline):**
- FP16
- Batch = 8
- Seq Len = 2048
- Heads = 32
- Head Dim = 128
- 50 timed iterations, 10 warmup

**Measured Results (H100 NVL):**

| Backend | Latency (ms) | Tokens/sec | TFLOPs | Avg Power (W) |
|---------|-------------:|-----------:|-------:|--------------:|
| Baseline (PyTorch) | 14.13 | 1.16M | 38.9 | 80.22 |
| **AN1 Engine** | **1.23** | **13.29M** | **445.9** | **73.72** |

**Impact:**
- AN1 completes attention operations **~11.3× faster**
- Power draw is nearly the same or slightly lower
- Because energy = power × time, AN1 reduces energy per attention step by **≈ 90%**

> ⚡ **Conclusion:** AN1 is not just faster — it is a **green accelerator** delivering high throughput with dramatically lower energy cost.

---

### 🔋 Energy Efficiency per Token (H100 NVL)

| Backend | Avg Power (W) | Tokens/sec | Energy per Token | Reduction |
|---------|--------------:|-----------:|-----------------:|----------:|
| PyTorch Baseline | 81.79 W | 1,408,818 | 5.81e-05 | — |
| **AN1 Engine** | **74.87 W** | **12,040,543** | **6.22e-06** | **≈ 89% lower** |

> **8.5× faster throughput + ~89% less energy per generated token on H100 NVL.**

#### 🔧 Benchmark Code Used (Power)

```bash
python bench/power_measure.py
```

---

## Benchmarking

### Running Benchmarks

```bash
# Benchmark attention with GPT-style configuration
python bench/llm_bench.py \
    --batch-size 8 \
    --seq-len 2048 \
    --n-heads 32 \
    --head-dim 128 \
    --dtype fp16 \
    --warmup 10 \
    --iters 100 \
    --backend-baseline \
    --backend-an1
```

Results are saved to `bench_logs/llm_bench_results.csv`.

### Quick Benchmark Scripts

On Windows:

```bash
test_5x_target.bat
```

On Linux (for H100):

```bash
./run_bench_h100.sh
```

---

## Testing

Run correctness tests:

```bash
pytest tests/test_llm_correctness.py -v
```

Tests verify that AN1-accelerated operations produce results matching the reference implementation within acceptable numerical tolerances.

---

## Architecture

This library provides a clean public API that dispatches to either:

- **AN1 GPU Backend** (`an1_cache.gpu_backend`): Thin wrappers around the private `an1_core_gpu` package  
- **Reference Implementation** (`an1_cache.reference`): Standard PyTorch operations for correctness verification  

The actual kernel implementations, optimization logic, and scheduling algorithms are **not** included in this repository. They are provided by the separate `an1_core_gpu` package, distributed as a binary-only wheel.

---

## Requirements

- Python ≥ 3.8  
- PyTorch ≥ 2.0.0 (with CUDA support for GPU acceleration)  
- NumPy ≥ 1.20.0  

---

## Supported GPUs

Tested on:

- NVIDIA H100  
- NVIDIA A100  
- NVIDIA L40S  
- Other modern CUDA-capable GPUs  

---

## License

MIT License — See `LICENSE` file for details.

MIT license applies only to this public API and benchmark suite.  
The AN1 GPU backend (`an1_core_gpu`) remains proprietary and is distributed separately.

---

## Contributing

This is a public-facing repository. Please ensure that any contributions:

- Do **not** include proprietary kernel code  
- Do **not** expose internal optimization details  
- Maintain backward compatibility with baseline implementations  
- Include appropriate tests  

See `PUBLIC_REPO_CHECKLIST.md` for guidelines.

---

## Research & Pilot Access

This repository provides the public AN1 API, benchmarking tools, and a PyTorch baseline.  
The accelerated AN1 backend (`an1_core_gpu`) is distributed separately as a commercial binary.

We offer private pilot access for:

- Research groups working on LLM efficiency or training  
- Teams deploying inference on H100, A100, or L40S clusters  
- Companies evaluating optimized attention or transformer kernels  
- Hardware labs exploring next-generation inference pipelines  

**Access Requirements**

- Signed NDA (non-disclosure agreement)  
- Evaluation or research license (no redistribution)  
- Supported hardware: H100, A100, L40S, or equivalent CUDA GPU  

**How to Request Access**

Email: **Ryan@animacore.ai**  
Phone: **305-505-5268**  
(Please include your organization, hardware, and intended use.)

> **Note:** The AN1 engine is not open source. This public repository is MIT-licensed for the API and tooling only. The AN1 backend is proprietary and licensed separately.
