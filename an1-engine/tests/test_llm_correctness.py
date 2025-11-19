"""
Correctness tests for AN1 LLM operations.

Verifies that AN1-accelerated operations produce results
that match the reference implementation within acceptable tolerances.
"""

import pytest
import torch
import numpy as np

from an1_cache import public_api, reference, gpu_backend


def test_attention_baseline():
    """Test that reference attention implementation works correctly."""
    batch_size, n_heads, seq_len, head_dim = 2, 4, 128, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    output = reference.llm_attention(q, k, v, causal=True)
    
    assert output.shape == (batch_size, n_heads, seq_len, head_dim)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attention_correctness():
    """Test that AN1 attention matches reference implementation."""
    if not gpu_backend.is_available():
        pytest.skip("AN1 backend not available or not licensed")
    
    batch_size, n_heads, seq_len, head_dim = 2, 4, 128, 64
    device = "cuda"
    dtype = torch.float16
    
    # Create test tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    # Run both implementations
    baseline = reference.llm_attention(q, k, v, causal=True)
    fast = public_api.llm_attention(q, k, v, backend="an1", causal=True)
    
    # Check shapes
    assert fast.shape == baseline.shape, f"Shape mismatch: {fast.shape} vs {baseline.shape}"
    
    # Check numerical accuracy
    # For FP16, we expect some numerical differences
    max_diff = torch.abs(fast - baseline).max().item()
    baseline_norm = torch.norm(baseline.to(torch.float32)).item()
    
    # Relative error
    if baseline_norm > 0:
        rel_error = max_diff / baseline_norm
    else:
        rel_error = max_diff
    
    # For FP16, allow reasonable tolerance
    assert max_diff < 1e-2, f"Max absolute difference too large: {max_diff}"
    assert rel_error < 1e-2, f"Relative error too large: {rel_error}"
    
    print(f"Attention correctness: max_diff={max_diff:.6f}, rel_error={rel_error:.6f}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attention_causal_vs_non_causal():
    """Test that causal flag works correctly."""
    batch_size, n_heads, seq_len, head_dim = 1, 1, 32, 16
    device = "cuda"
    dtype = torch.float32
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    # Causal should be different from non-causal
    output_causal = public_api.llm_attention(q, k, v, backend="baseline", causal=True)
    output_non_causal = public_api.llm_attention(q, k, v, backend="baseline", causal=False)
    
    # They should be different (causal mask should affect results)
    assert not torch.allclose(output_causal, output_non_causal, atol=1e-5)


def test_matmul_baseline():
    """Test that reference matmul implementation works correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    a = torch.randn(128, 256, device=device, dtype=dtype)
    b = torch.randn(256, 512, device=device, dtype=dtype)
    
    output = reference.matmul(a, b)
    
    assert output.shape == (128, 512)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matmul_correctness():
    """Test that AN1 matmul matches reference implementation."""
    if not gpu_backend.is_available():
        pytest.skip("AN1 backend not available or not licensed")
    
    device = "cuda"
    dtype = torch.float16
    
    # Create test tensors
    torch.manual_seed(42)
    a = torch.randn(256, 512, device=device, dtype=dtype)
    b = torch.randn(512, 1024, device=device, dtype=dtype)
    
    # Run both implementations
    baseline = reference.matmul(a, b)
    fast = public_api.matmul(a, b, backend="an1")
    
    # Check shapes
    assert fast.shape == baseline.shape, f"Shape mismatch: {fast.shape} vs {baseline.shape}"
    
    # Check numerical accuracy
    max_diff = torch.abs(fast - baseline).max().item()
    baseline_norm = torch.norm(baseline.to(torch.float32)).item()
    
    if baseline_norm > 0:
        rel_error = max_diff / baseline_norm
    else:
        rel_error = max_diff
    
    # For FP16, allow reasonable tolerance
    assert max_diff < 1e-2, f"Max absolute difference too large: {max_diff}"
    assert rel_error < 1e-2, f"Relative error too large: {rel_error}"
    
    print(f"Matmul correctness: max_diff={max_diff:.6f}, rel_error={rel_error:.6f}")


def test_backend_auto_selection():
    """Test that automatic backend selection works."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    q = torch.randn(1, 1, 32, 16, device=device, dtype=dtype)
    k = torch.randn(1, 1, 32, 16, device=device, dtype=dtype)
    v = torch.randn(1, 1, 32, 16, device=device, dtype=dtype)
    
    # Should not raise an error (falls back to baseline if AN1 unavailable)
    output = public_api.llm_attention(q, k, v, backend="auto")
    assert output.shape == q.shape


def test_backend_fallback():
    """Test that backend falls back gracefully when AN1 is unavailable."""
    # Force baseline backend
    output = public_api.llm_attention(
        torch.randn(1, 1, 32, 16),
        torch.randn(1, 1, 32, 16),
        torch.randn(1, 1, 32, 16),
        backend="baseline"
    )
    assert output is not None
    
    # If AN1 is not available, auto should fall back to baseline
    output_auto = public_api.llm_attention(
        torch.randn(1, 1, 32, 16),
        torch.randn(1, 1, 32, 16),
        torch.randn(1, 1, 32, 16),
        backend="auto"
    )
    assert output_auto is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

