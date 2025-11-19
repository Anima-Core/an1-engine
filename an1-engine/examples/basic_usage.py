"""
Basic usage example for AN1 Cache library.

This demonstrates how to use the public API for LLM attention and matmul operations.
"""

import os
import torch
from an1_cache import llm_attention, matmul


def example_attention():
    """Example: Using llm_attention."""
    print("Example: LLM Attention")
    print("-" * 50)
    
    # Create input tensors
    batch_size, n_heads, seq_len, head_dim = 2, 8, 512, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    
    # Compute attention (automatically selects best backend)
    output = llm_attention(q, k, v, causal=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
    print()


def example_matmul():
    """Example: Using matmul."""
    print("Example: Matrix Multiplication")
    print("-" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Create matrices
    m, k, n = 1024, 512, 2048
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    
    print(f"Matrix shapes: a={a.shape}, b={b.shape}")
    
    # Compute matmul
    output = matmul(a, b)
    
    print(f"Output shape: {output.shape}")
    print(f"Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
    print()


def example_backend_selection():
    """Example: Explicit backend selection."""
    print("Example: Backend Selection")
    print("-" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    q = torch.randn(1, 1, 32, 16, device=device, dtype=dtype)
    k = torch.randn(1, 1, 32, 16, device=device, dtype=dtype)
    v = torch.randn(1, 1, 32, 16, device=device, dtype=dtype)
    
    # Automatic backend selection (default)
    output_auto = llm_attention(q, k, v, backend="auto")
    print("✓ Automatic backend selection")
    
    # Force baseline backend
    output_baseline = llm_attention(q, k, v, backend="baseline")
    print("✓ Baseline backend (PyTorch)")
    
    # Try AN1 backend (may fail if not installed or not licensed)
    try:
        output_an1 = llm_attention(q, k, v, backend="an1")
        print("✓ AN1 backend (accelerated)")
    except Exception as e:
        print(f"✗ AN1 backend not available: {e}")
    
    print()


def example_license_setup():
    """Example: License key setup."""
    print("Example: License Key Setup")
    print("-" * 50)
    
    license_key = os.getenv("AN1_LICENSE_KEY")
    if license_key:
        print(f"License key is set (length: {len(license_key)})")
        print("AN1 backend will be used if an1_core_gpu is installed")
    else:
        print("No license key found in AN1_LICENSE_KEY environment variable")
        print("Library will use baseline PyTorch implementation")
        print()
        print("To enable AN1 backend:")
        print("  export AN1_LICENSE_KEY='your-license-key'  # Linux/Mac")
        print("  set AN1_LICENSE_KEY=your-license-key       # Windows")
    print()


if __name__ == "__main__":
    print("AN1 Cache - Basic Usage Examples")
    print("=" * 50)
    print()
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (GPU not available)")
    print()
    
    example_license_setup()
    example_attention()
    example_matmul()
    example_backend_selection()
    
    print("Examples completed!")

