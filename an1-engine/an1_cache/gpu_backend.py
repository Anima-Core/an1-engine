"""
GPU backend loader for AN1 accelerated operations.

This module provides thin wrappers around the private an1_core_gpu package.
The actual kernel implementations and optimization logic are not included
in this public repository.
"""

import os
import torch
from typing import Optional


# Try to import the private GPU backend
try:
    import an1_core_gpu
    _BACKEND_AVAILABLE = True
except ImportError:
    an1_core_gpu = None
    _BACKEND_AVAILABLE = False


class FastBackendUnavailable(Exception):
    """Raised when AN1 fast GPU backend is not available or encounters an error."""
    pass


def _check_backend():
    """Check if the AN1 GPU backend is available."""
    if not _BACKEND_AVAILABLE:
        raise FastBackendUnavailable(
            "AN1 fast GPU backend not installed. "
            "Install the private wheel 'an1_core_gpu' to enable accelerated path."
        )


def _validate_license() -> bool:
    """
    Validate license key via the private backend.
    
    Reads AN1_LICENSE_KEY environment variable and passes it to the backend
    for validation. This repository does not implement license validation logic.
    """
    if not _BACKEND_AVAILABLE:
        return False
    
    license_key = os.getenv("AN1_LICENSE_KEY")
    if license_key is None:
        return False
    
    try:
        # Delegate license validation to the private backend
        if hasattr(an1_core_gpu, 'validate_license'):
            return an1_core_gpu.validate_license(license_key)
        elif hasattr(an1_core_gpu, 'has_valid_license'):
            return an1_core_gpu.has_valid_license(license_key)
        else:
            # If backend doesn't implement validation, assume valid
            return True
    except Exception:
        return False


def is_available() -> bool:
    """
    Check if the AN1 GPU backend is available and licensed.
    
    Returns:
        True if an1_core_gpu is installed and license is valid, False otherwise.
    """
    if not _BACKEND_AVAILABLE:
        return False
    
    return _validate_license()


def llm_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """
    AN1-accelerated LLM attention.
    
    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, n_heads, seq_len, head_dim)
        v: Value tensor of shape (batch, n_heads, seq_len, head_dim)
        causal: If True, apply causal masking
    
    Returns:
        Output tensor of shape (batch, n_heads, seq_len, head_dim)
    
    Raises:
        FastBackendUnavailable: If backend is not installed or license is invalid
    """
    _check_backend()
    
    if not _validate_license():
        raise FastBackendUnavailable(
            "AN1 license key is missing or invalid. "
            "Set AN1_LICENSE_KEY environment variable with a valid license key."
        )
    
    return an1_core_gpu.llm_attention(q, k, v, causal=causal)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    AN1-accelerated matrix multiplication.
    
    Args:
        a: Left tensor, shape (..., m, k)
        b: Right tensor, shape (..., k, n)
    
    Returns:
        Output tensor of shape (..., m, n)
    
    Raises:
        FastBackendUnavailable: If backend is not installed or license is invalid
    """
    _check_backend()
    
    if not _validate_license():
        raise FastBackendUnavailable(
            "AN1 license key is missing or invalid. "
            "Set AN1_LICENSE_KEY environment variable with a valid license key."
        )
    
    return an1_core_gpu.matmul(a, b)

