@echo off
echo AN1 LLM Benchmark - GPT-style configuration
echo.
echo Running benchmark with realistic GPT parameters:
echo   batch=8, seq_len=2048, n_heads=32, head_dim=128, dtype=fp16
echo.

python bench\llm_bench.py --batch-size 8 --seq-len 2048 --n-heads 32 --head-dim 128 --dtype fp16 --warmup 10 --iters 100 --op attention

echo.
echo Benchmark completed. Check bench_logs\llm_bench_results.csv for results.
pause
