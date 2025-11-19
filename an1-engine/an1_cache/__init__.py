"""
AN1 Cache: High-performance LLM operations library.

This package provides accelerated implementations of common LLM operations
such as attention and matrix multiplication. The library automatically
dispatches to optimized GPU kernels when available and licensed, falling back to
reference PyTorch implementations otherwise.

Public API:
    - llm_attention: Scaled dot-product attention
    - matmul: Matrix multiplication

Example:
    >>> import torch
    >>> from an1_cache import llm_attention
    >>> 
    >>> q = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)
    >>> k = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)
    >>> v = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)
    >>> 
    >>> output = llm_attention(q, k, v, causal=True)
"""

from .public_api import llm_attention, matmul

__all__ = ["llm_attention", "matmul"]

__version__ = "0.1.0"
