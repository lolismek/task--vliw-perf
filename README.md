# VLIW SIMD Performance Optimization

Optimize a tree-traversal kernel for a custom VLIW SIMD processor simulator. Minimize cycle count from a baseline of 147,734 cycles.

Based on [Anthropic's original performance take-home](https://github.com/anthropics/original_performance_takehome).

## Quickstart

```bash
bash prepare.sh          # verify environment
bash eval/eval.sh        # run evaluation (baseline: 147734 cycles)
```

## What to Modify

Only `perf_takehome.py` (the `KernelBuilder` class). Read `program.md` for full instructions.

## Scoring

- **Metric:** speedup over baseline (higher is better)
- `speedup = 147734 / cycles`
- See `program.md` for thresholds

## Validation

```bash
python tests/submission_tests.py
```
