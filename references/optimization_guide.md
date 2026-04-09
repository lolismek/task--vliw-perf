# Optimization Guide

Research findings from public solutions and discussions. Use this as a starting point for ideas.

## Performance Progression (what's possible)

This progression comes from a documented 19-step optimization journey that reached 1,338 cycles:

| Stage | Cycles | Speedup | Key Technique |
|-------|--------|---------|---------------|
| Baseline | 147,734 | 1.0x | One operation per cycle |
| SIMD vectorization | 13,456 | 11.0x | Process 8 walkers with valu/vload/vstore |
| + Bitwise ops | 12,944 | 11.4x | Replace modulo with AND, multiply with shift |
| + Group processing (N=24) | 5,600 | 26.4x | Process groups through phases; saturate engines |
| + Phase overlapping | 4,128 | 35.8x | Overlap load and compute phases across groups |
| + Software pipelining v2 | 3,744 | 39.4x | Cleaner pipeline boundaries |
| + Round fusion | 3,056 | 48.3x | Round-first processing, values stay in scratch |
| + Static scheduler | 2,472 | 59.7x | Global dependency-aware instruction packing |
| + multiply_add fusion | 2,349 | 62.9x | Collapse hash stages using fused ops |
| + Tree node preloading | 1,338 | 110.4x | Preload levels 0-3, use vselect instead of loads |

Other reported results:
- 1,137 cycles (human + AI, 1 hour): pipelined vectorized hash with speculation
- 1,112 cycles (human, week of evenings): best known public result

## Key Bottleneck Analysis

### Load slots are the primary bottleneck
Only 2 loads per cycle. Each walker needs to load a tree node value every step. With 256 walkers and 16 rounds, that's 4,096 loads. At 2 per cycle, the absolute minimum is 2,048 cycles just for loads — unless you eliminate loads via preloading.

### VALU slots are the computation bottleneck
6 valu slots per cycle. Each hash has 6 stages x 3 ops = 18 vector operations. For 256 walkers (32 groups of 8) x 16 rounds: 32 x 16 x 18 = 9,216 valu operations. At 6 per cycle: minimum 1,536 cycles. Hash fusion with multiply_add reduces this.

### Flow slot is scarce
Only 1 flow slot per cycle. Both `select`/`vselect` and `cond_jump` compete for it. Prefer arithmetic equivalents where possible (e.g., use alu for conditionals).

## Detailed Technique Notes

### SIMD Vectorization

The batch size (256) divides evenly by VLEN (8), giving 32 groups. For each group:
1. `vload` indices and values (2 load slots)
2. Compute addresses: `valu("+", addr, forest_base, indices)`
3. Load node values (this is the hard part — `vload` loads contiguous memory, but tree nodes at arbitrary indices are NOT contiguous)
4. `valu("^", ...)` to XOR
5. Vectorized hash (6 stages, each with vbroadcast + valu ops)
6. `vstore` results back

The gather problem (step 3): You can't `vload` non-contiguous addresses. Options:
- 8 individual scalar `load` ops (uses 4 cycles at 2 loads/cycle)
- Preload tree nodes into scratch to avoid memory loads entirely

### VLIW Instruction Packing

Build a dependency graph:
- Track which scratch addresses each slot reads and writes
- Two slots can go in the same cycle if they don't have write-after-write or read-after-write conflicts (remember: writes happen at end of cycle, so write-after-read in the same cycle is OK)
- Greedy algorithm: for each slot, find the earliest cycle where it fits within slot limits and dependency constraints

### Hash Fusion with multiply_add

Consider hash stage 0: `a = (a + 0x7ED55D16) + (a << 12)`
- This equals `a = a + 0x7ED55D16 + a * 4096`
- Which is `a = a * 4097 + 0x7ED55D16`
- Can be computed as: broadcast 4097 to a vector, broadcast the constant, then `multiply_add(dest, a_vec, const_4097_vec, const_hex_vec)`

Not all stages fuse this cleanly (XOR stages don't distribute over addition), but stages 0, 2, 4 (which use `+` and `<<`) can benefit.

### Tree Node Preloading

The tree has 2,047 nodes across 11 levels (0-10). Level L has 2^L nodes.

- Level 0: 1 node (root) — same for all walkers, broadcast once
- Level 1: 2 nodes — select with 1 vselect based on index bit
- Level 2: 4 nodes — select with 2 vselects
- Level 3: 8 nodes — select with 3 vselects

Total: 15 nodes = 120 scratch words. This replaces 4 levels of memory loads with scratch reads and vselect chains. Since ~40% of tree visits hit the top 4 levels (walkers wrap to root frequently), this is a massive win.

For deeper levels (4-10), fall back to memory loads.

### Software Pipelining

Structure the inner loop as overlapping stages:
```
Cycle N:     [Group A: hash stage 3-4] [Group B: load nodes] [Group C: compute addresses]
Cycle N+1:   [Group A: hash stage 5-6] [Group B: XOR + hash 1-2] [Group C: load nodes]
```

The key insight: while VALU slots are busy with hash computation for group A, the load slots are idle — use them to load data for group B.

### Parity Check Without Modulo

Instead of `% 2` (which uses an ALU modulo), use `& 1` (bitwise AND):
```
("&", result, value, one_const)    # is value even? result = value & 1
```
Then use the result with `select` or `vselect` to choose left/right child.

Similarly, replace `* 2` with `<< 1`:
```
("<<", result, idx, one_const)     # idx * 2 = idx << 1
```

## Common Pitfalls

1. **Modifying tests** — The eval script detects this and reverts. Your solution will be scored on the unmodified frozen simulator.

2. **Exceeding SCRATCH_SIZE** — 1,536 words is the hard limit. Vector temporaries eat 8 words each. Budget carefully: with 32 groups x 2 vectors (indices + values) = 512 words, plus hash constants, loop vars, and temporaries, you can hit the limit.

3. **Ignoring data dependencies** — If slot B reads a scratch address that slot A writes, they CANNOT be in the same cycle (the write happens at end of cycle, so B would read the old value). Track dependencies carefully in your scheduler.

4. **Write-after-read is OK** — Within the same cycle, reads happen before writes. So if A reads address X and B writes address X, they CAN coexist in the same cycle.

5. **Gather is expensive** — The tree node load is a gather (non-contiguous addresses). `vload` only works for contiguous memory. You need 8 scalar loads (4 cycles) or preloading.

6. **Flow slot contention** — `vselect`, `select`, `cond_jump`, and `add_imm` all compete for the single flow slot. If your inner loop needs multiple selects per cycle, you'll bottleneck here. Consider arithmetic alternatives.

7. **Forgetting to handle tree wrap** — When `idx >= n_nodes`, the walker wraps to index 0. This must be computed correctly for every walker every step.

## Theoretical Minimum

Lower bounds (not necessarily achievable simultaneously):
- Load-bound: 4,096 loads / 2 per cycle = 2,048 cycles (without preloading)
- Store-bound: 4,096 stores / 2 per cycle = 2,048 cycles (can be reduced with scratch residency)
- VALU-bound: ~9,216 valu ops / 6 per cycle = 1,536 cycles (without fusion)
- With fusion + preloading: the true minimum is likely in the 800-1,000 cycle range

The best known human result (1,112 cycles) suggests the practical floor is near 1,000 cycles.
