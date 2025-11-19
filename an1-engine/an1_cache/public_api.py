"""
Public API for AN1-accelerated LLM operations.

This module provides high-level functions that automatically dispatch
to either the AN1 GPU backend (if available and licensed) or the reference implementation.
"""

import os
import torch
from typing import Optional, Literal

from . import reference
from . import gpu_backend


def _get_backend(backend: Optional[str] = None) -> str:
    """
    Determine which backend to use based on argument and environment.
    
    Args:
        backend: Explicit backend choice ("an1", "baseline", or None for "auto")
    
    Returns:
        Backend name: "an1" or "baseline"
    """
    if backend is None:
        backend = os.getenv("AN1_BACKEND", "auto")
    
    if backend == "auto":
        if gpu_backend.is_available():
            return "an1"
        else:
            return "baseline"
    elif backend == "an1":
        return "an1"
    elif backend == "baseline":
        return "baseline"
    else:
        raise ValueError(f"Unknown backend: {backend}. Must be 'an1', 'baseline', or 'auto'")


def llm_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: Optional[Literal["an1", "baseline", "auto"]] = "auto",
    causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    LLM attention computation with automatic backend selection.
    
    This function dispatches to either the AN1-accelerated backend (if available
    and licensed) or the reference PyTorch implementation. Use the backend parameter
    or AN1_BACKEND environment variable to control selection.
    
    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, n_heads, seq_len, head_dim)
        v: Value tensor of shape (batch, n_heads, seq_len, head_dim)
        backend: Backend to use ("an1", "baseline", or "auto" for automatic)
        causal: If True, apply causal masking
        scale: Attention scale factor. If None, uses 1/sqrt(head_dim).
               Only used with baseline backend.
    
    Returns:
        Output tensor of shape (batch, n_heads, seq_len, head_dim)
    
    Raises:
        ValueError: If tensor shapes are invalid
        FastBackendUnavailable: If backend="an1" but an1_core_gpu is not available or license is invalid
    """
    # Shape validation
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v must have same shape. Got q={q.shape}, k={k.shape}, v={v.shape}")
    
    if len(q.shape) != 4:
        raise ValueError(f"Expected 4D tensors (batch, n_heads, seq_len, head_dim), got shape {q.shape}")
    
    # Dispatch to appropriate backend
    selected_backend = _get_backend(backend)
    
    if selected_backend == "an1":
        try:
            return gpu_backend.llm_attention(q, k, v, causal=causal)
        except gpu_backend.FastBackendUnavailable:
            # Fallback to baseline if AN1 backend fails
            return reference.llm_attention(q, k, v, causal=causal, scale=scale)
    else:
        return reference.llm_attention(q, k, v, causal=causal, scale=scale)


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    backend: Optional[Literal["an1", "baseline", "auto"]] = "auto",
) -> torch.Tensor:
    """
    Matrix multiplication with automatic backend selection.
    
    This function dispatches to either the AN1-accelerated backend (if available
    and licensed) or the reference PyTorch implementation.
    
    Args:
        a: Left tensor, shape (..., m, k)
        b: Right tensor, shape (..., k, n)
        backend: Backend to use ("an1", "baseline", or "auto" for automatic)
    
    Returns:
        Output tensor of shape (..., m, n)
    
    Raises:
        ValueError: If tensor shapes are incompatible
        FastBackendUnavailable: If backend="an1" but an1_core_gpu is not available or license is invalid
    """
    # Basic shape validation
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError("Tensors must be at least 2D for matrix multiplication")
    
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Incompatible shapes for matmul: a.shape[-1]={a.shape[-1]} != b.shape[-2]={b.shape[-2]}"
        )
    
    # Dispatch to appropriate backend
    selected_backend = _get_backend(backend)
    
    if selected_backend == "an1":
        try:
            return gpu_backend.matmul(a, b)
        except gpu_backend.FastBackendUnavailable:
            # Fallback to baseline if AN1 backend fails
            return reference.matmul(a, b)
    else:
        return reference.matmul(a, b)

