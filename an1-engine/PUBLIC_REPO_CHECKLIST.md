# Public Repository Checklist

This document helps ensure no proprietary code or trade secrets are accidentally included in the public repository.

## Files That Should NOT Be in Public Repo

- **CUDA kernel files**: `.cu`, `.ptx`, `.cubin`, `.fatbin`
- **Private backend package**: `an1_core_gpu/` directory or any compiled extensions
- **Planning/scheduling logic**: Any files that reveal how AN1 finds speedups
- **Internal optimization details**: Comments or code that expose kernel tricks, tiling parameters, cache layouts
- **Debug prints**: Any print statements that expose internal schedules or planning decisions
- **License validation implementation**: Only pass-through to backend, no actual validation logic
- **License keys**: Never commit actual license keys or `.key`, `.license` files

## Files That ARE Safe for Public Repo

- ✅ `an1_cache/public_api.py` - High-level API wrappers
- ✅ `an1_cache/reference.py` - Baseline PyTorch implementations
- ✅ `an1_cache/gpu_backend.py` - Thin loader for private package (no kernel code, only license pass-through)
- ✅ `bench/llm_bench.py` - Benchmarking utilities
- ✅ `tests/test_llm_correctness.py` - Correctness tests
- ✅ `README.md`, `setup.py`, `pyproject.toml` - Documentation and packaging
- ✅ `examples/` - Usage examples

## Verification Steps

Before pushing to public repo:

1. Search for `.cu` files: `find . -name "*.cu"`
2. Search for `.ptx` files: `find . -name "*.ptx"`
3. Search for `.cubin` files: `find . -name "*.cubin"`
4. Check for debug prints that expose internals: `grep -r "print.*schedule\|print.*tile\|print.*cache" --include="*.py"`
5. Review all comments for proprietary information
6. Ensure `an1_core_gpu` is not in the repo (check `.gitignore`)
7. Verify no license validation logic (only pass-through to backend)
8. Check for any hardcoded license keys or secrets

## Code Review Checklist

- [ ] No CUDA kernel code in any `.py` files
- [ ] No comments describing kernel optimization tricks
- [ ] No debug prints exposing planning decisions
- [ ] `gpu_backend.py` only contains thin wrappers, no implementation
- [ ] License validation only passes env var to backend, no crypto logic
- [ ] All reported speedups come from actual GPU execution (no mocks)
- [ ] Reference implementations are clear and readable (they're the baseline)
- [ ] No `.whl` files for `an1_core_gpu` in the repo

