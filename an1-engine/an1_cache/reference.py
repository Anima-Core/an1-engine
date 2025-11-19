"""
Reference implementations using standard PyTorch operations.

These are the baseline implementations that demonstrate correctness
and serve as the comparison point for AN1-accelerated operations.
All implementations use standard PyTorch operations with no proprietary logic.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def llm_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Standard scaled dot-product attention implementation.
    
    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, n_heads, seq_len, head_dim)
        v: Value tensor of shape (batch, n_heads, seq_len, head_dim)
        causal: If True, apply causal masking
        scale: Attention scale factor. If None, uses 1/sqrt(head_dim)
    
    Returns:
        Output tensor of shape (batch, n_heads, seq_len, head_dim)
    """
    batch, n_heads, seq_len, head_dim = q.shape
    
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    # Compute attention scores: (batch, n_heads, seq_len, seq_len)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Apply causal mask if requested
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=q.dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply to values
    output = torch.matmul(attn_weights, v)
    
    return output


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Standard matrix multiplication.
    
    Args:
        a: Left tensor, shape (..., m, k)
        b: Right tensor, shape (..., k, n)
    
    Returns:
        Output tensor of shape (..., m, n)
    """
    return torch.matmul(a, b)

