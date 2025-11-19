# Repository Structure

This document describes the organization of the AN1 Cache public repository.

## Package Layout

```
an1_cache/
├── __init__.py          # Public API exports (llm_attention, matmul)
├── public_api.py        # High-level API with automatic backend selection
├── reference.py          # Baseline PyTorch implementations
├── gpu_backend.py        # Thin loader for private an1_core_gpu package
├── cache.py             # (Internal utility, not part of public API)
├── director.py           # (Internal utility, not part of public API)
├── energy.py             # (Internal utility, not part of public API)
└── utils.py              # (Internal utility, not part of public API)

bench/
├── __init__.py
└── llm_bench.py          # Main benchmark script

tests/
└── test_llm_correctness.py  # Correctness tests

examples/
└── basic_usage.py        # Usage examples
```

## Public API

The public API is exposed through `an1_cache/__init__.py`:

- `llm_attention(q, k, v, *, backend="auto", causal=True, scale=None)`
- `matmul(a, b, *, backend="auto")`

These functions automatically dispatch to:
- **AN1 backend** (if `an1_core_gpu` is installed and licensed)
- **Reference implementation** (standard PyTorch, always available)

## Backend Architecture

```
User Code
    ↓
public_api.py (shape checks, dispatch logic)
    ↓
    ├─→ gpu_backend.py → an1_core_gpu (private package, not in repo)
    │                    └─→ License validation (delegated to backend)
    └─→ reference.py (PyTorch baseline)
```

## License Key Handling

License keys are handled as follows:

1. User sets `AN1_LICENSE_KEY` environment variable
2. `gpu_backend.py` reads the env var and passes it to `an1_core_gpu.validate_license()`
3. The private backend performs actual validation
4. This repository contains NO license validation logic, only pass-through

## Key Files

### Public API Files (Safe for Public Repo)
- `an1_cache/public_api.py`: High-level wrappers, no kernel code
- `an1_cache/reference.py`: Baseline implementations, fully open
- `an1_cache/gpu_backend.py`: Thin wrappers only, no implementation, license pass-through only
- `bench/llm_bench.py`: Benchmarking utilities
- `tests/test_llm_correctness.py`: Correctness verification

### Internal Files (Not Part of Public API)
- `an1_cache/cache.py`: Cache management utilities
- `an1_cache/director.py`: Internal coordination logic
- `an1_cache/energy.py`: Energy monitoring utilities
- `an1_cache/utils.py`: Internal helpers

These internal files are not imported by `__init__.py` and are not part of the public API.

## Installation

The package can be installed with:
```bash
pip install -e .
```

This installs the `an1_cache` package with the public API accessible via:
```python
from an1_cache import llm_attention, matmul
```

## Private Backend

The actual GPU kernels are provided by a separate private package `an1_core_gpu` that:
- Is NOT included in this repository
- Must be installed separately as a binary-only wheel
- Provides the actual accelerated implementations
- Exposes a minimal API: `llm_attention()`, `matmul()`, and `validate_license()`
- Performs all license validation internally

When `an1_core_gpu` is not available or not licensed, the library automatically falls back to PyTorch reference implementations.

