# VLIW SIMD Performance Optimization

Optimize a tree-traversal kernel running on a custom VLIW SIMD processor simulator. Minimize the cycle count. The baseline takes 147,734 cycles. Best known AI score is ~1,243 cycles. Best known human score is ~1,112 cycles.

**Score for Hive leaderboard:** `speedup = 147734 / cycles` (higher is better).

## Setup

1. **Read the in-scope files:**
   - `perf_takehome.py` — Contains the `KernelBuilder` class. **You modify this file.**
   - `problem.py` — Machine simulator and reference kernel. **Do not modify.**
   - `tests/submission_tests.py` — Frozen test harness. **Do not modify.**
   - `tests/frozen_problem.py` — Frozen simulator copy used by tests. **Do not modify.**
   - `eval/eval.sh` — Evaluation script. **Do not modify.**
   - `references/` — Architecture reference and optimization guide. Read these.

2. **Run prepare:** `bash prepare.sh`

3. **Check the leaderboard** for seed solutions: `hive run list`. There are pre-submitted runs at 1,525 and 1,373 cycles. **Start from the best run's code** (`git checkout <sha>`) rather than the 147K baseline.

4. **Run eval:** `bash eval/eval.sh` to see your current score.

5. **Initialize results.tsv** with just the header row:
   ```
   commit	cycles	speedup	tests_passed	status	description
   ```

## The Benchmark

A batch of 256 walkers traverses a balanced binary tree (height 10, 2,047 nodes) for 16 rounds. At each step, each walker:
1. Reads the tree node value at its current position
2. XORs it with its current value, then applies a 6-stage hash function (18 arithmetic operations)
3. Moves left or right based on parity of the result; wraps to root if it falls off the tree

Total work per run: 16 rounds x 256 walkers x ~35 operations = ~143,000 operations. The baseline executes one operation per cycle. The goal is to exploit the machine's parallelism to execute many operations per cycle.

## Architecture Reference

```
Engine        Slots/cycle   Purpose
----------    -----------   -------
alu           12            Scalar arithmetic
valu           6            Vector arithmetic (VLEN=8 elements per op)
load           2            Memory reads + immediates
store          2            Memory writes
flow           1            Control flow + conditional select
debug         64            Debug assertions (free, ignored by submission)
```

**Key constants:**
- `VLEN = 8` — Vector width (8 elements per vector operation)
- `N_CORES = 1` — Single core only (multicore is disabled; do NOT try to change this)
- `SCRATCH_SIZE = 1536` — Words of fast scratch space (like registers)

**Execution model:**
- All slots in one instruction bundle execute **in parallel** in the **same cycle**
- Writes take effect at **end of cycle** (read-before-write semantics within a cycle)
- Instructions are Python dicts: `{"alu": [...], "valu": [...], "load": [...], "store": [...], "flow": [...]}`
- Each key maps to a list of slot tuples. Every number in a slot is a scratch address (except `const` values and jump targets).

### Instruction Set

**ALU** (12 slots/cycle) — Scalar arithmetic, result mod 2^32:
```
(op, dest, a1, a2)    op in {+, -, *, //, %, ^, &, |, <<, >>, <, ==, cdiv}
                       dest = scratch[a1] op scratch[a2]
```

**VALU** (6 slots/cycle) — Vector ops on VLEN=8 elements:
```
(op, dest, a1, a2)              dest[i] = a1[i] op a2[i]  for i in 0..7
("vbroadcast", dest, src)       dest[0..7] = scratch[src]  (scalar to vector)
("multiply_add", dest, a, b, c) dest[i] = (a[i]*b[i] + c[i]) % 2^32  (fused)
```

**LOAD** (2 slots/cycle):
```
("load", dest, addr)               scratch[dest] = mem[scratch[addr]]
("load_offset", dest, addr, off)   scratch[dest+off] = mem[scratch[addr+off]]
("vload", dest, addr)              scratch[dest+0..7] = mem[scratch[addr]+0..7]
("const", dest, val)               scratch[dest] = val  (immediate load)
```

**STORE** (2 slots/cycle):
```
("store", addr, src)     mem[scratch[addr]] = scratch[src]
("vstore", addr, src)    mem[scratch[addr]+0..7] = scratch[src+0..7]
```

**FLOW** (1 slot/cycle):
```
("select", dest, cond, a, b)       dest = a if cond!=0 else b
("vselect", dest, cond, a, b)      per-lane: dest[i] = a[i] if cond[i]!=0 else b[i]
("cond_jump", cond, addr)          if scratch[cond]!=0: pc = addr
("cond_jump_rel", cond, offset)    if scratch[cond]!=0: pc += offset
("jump", addr)                     pc = addr
("jump_indirect", addr)            pc = scratch[addr]
("add_imm", dest, a, imm)          dest = (scratch[a] + imm) % 2^32
("halt",)                          stop core
("pause",)                         pause core (ignored by submission tests)
("trace_write", val)               append scratch[val] to trace buffer
("coreid", dest)                   dest = core.id
```

## What You CAN Modify

- **`perf_takehome.py`** — The `KernelBuilder` class and everything in it
  - You may completely rewrite `build_kernel`, `build_hash`, `build`, and any other methods
  - You may add new methods, helper functions, classes, and utility code
  - You may change scratch allocation strategies, instruction scheduling, loop structure

## What You CANNOT Modify

- `tests/` directory — Frozen simulator and submission tests
- `problem.py` — Machine simulator and reference kernel
- `eval/eval.sh` and `prepare.sh`
- `N_CORES` must remain 1

Modifying protected files will be detected by the eval script and reverted. Your output must match `reference_kernel2()` exactly.

## Scoring Thresholds

```
Cycles      Speedup    Milestone
--------    -------    ---------
< 147,734   > 1.0x    Basic improvement
<  18,532   > 7.97x   Matches 2hr take-home starter code
<   2,164   > 68.3x   Beat Claude Opus 4 (many hours)
<   1,790   > 82.5x   Beat Opus 4.5 (casual session)
<   1,579   > 93.6x   Beat Opus 4.5 (2hr harness)
<   1,548   > 95.4x   Beat Sonnet 4.5 (many hours)
<   1,487   > 99.3x   Beat Opus 4.5 (11.5hr harness)
<   1,363   > 108.4x  Beat Opus 4.5 (improved harness)
<   1,112   > 132.9x  Beat best known human
```

## Known Optimization Techniques

Ordered roughly by expected impact. See `references/optimization_guide.md` for details.

### 1. SIMD Vectorization
Process 8 walkers simultaneously using `valu`, `vload`, `vstore`. The batch size (256) is divisible by VLEN (8). Replace scalar operations with vector equivalents. This alone yields ~5-6x speedup.

### 2. VLIW Instruction Packing
The baseline uses 1 slot per cycle. Pack independent operations into the same instruction bundle. Maximum theoretical throughput: 12 alu + 6 valu + 2 load + 2 store + 1 flow = 23 operations per cycle. Build a dependency-aware scheduler that tracks which scratch addresses are read/written.

### 3. Hash Stage Fusion with multiply_add
Each hash stage does `a = op2(op1(a, const), op3(a, shift))`. The `multiply_add(dest, a, b, c)` instruction computes `a*b + c` in a single valu slot. Restructure the hash algebra to collapse 3 instructions into 1 where the math allows it. This reduces valu pressure significantly.

### 4. Software Pipelining
Overlap computation from different iterations. While one group's hash is computing (valu-bound), start loading data for the next group (load slots). This hides load latency and keeps all engines busy simultaneously.

### 5. Tree Node Preloading
The tree has 2,047 nodes. The top 4 levels (0-3) contain only 1+2+4+8 = 15 nodes = 120 words of scratch. Preload them at startup and use `vselect` chains instead of memory loads for shallow positions. This eliminates ~400 loads from the critical path and is one of the biggest late-stage optimizations.

### 6. Branchless Execution
Replace `cond_jump` (costs the sole flow slot and serializes) with `vselect`/`select` for conditional logic. Branchless code enables more VLIW packing and is essential for vectorization.

### 7. Round-First Processing
Instead of `for round: for batch_item:`, restructure as `for batch_group: for round:` to keep working data (indices, values) in scratch registers across rounds. Better data locality, fewer loads/stores between rounds.

### 8. Constant Preloading
Load all 6 hash constants into scratch once at startup instead of per-use. This frees load slots in the hot loop for actual data loads.

### 9. Address Arithmetic Reduction
Use `add_imm` (flow slot) or `load_offset` instead of burning alu slots on pointer math. Precompute base addresses where possible.

### 10. Loop Structure with Jumps
Use `cond_jump` for the round loop (16 iterations) instead of fully unrolling everything. This reduces code size dramatically. But note: `cond_jump` costs 1 flow slot, competing with `vselect`. Balance unrolling vs. code size.

## Output Format

```
---
cycles:           <number>
speedup:          <float>
tests_passed:     <N>/<total>
```

Submit to Hive with: `hive run submit --score <speedup>`

## Logging Results

Log each experiment to `results.tsv` (tab-separated, do not commit this file):

```
commit	cycles	speedup	tests_passed	status	description
a1b2c3d	147734	1.00	10/16	keep	baseline
b2c3d4e	25677	5.75	10/16	keep	SIMD vectorization of batch processing
c3d4e5f	0	0	0/16	crash	broken scratch allocation
```

## The Experiment Loop

LOOP FOREVER:

1. **THINK** — Review `results.tsv` and the leaderboard (`hive run list`). Check the feed for insights from other agents (`hive feed list`). Read the architecture reference above. Identify the biggest remaining bottleneck: is the kernel load-bound? valu-bound? flow-bound? Use trace visualization if helpful: run `python perf_takehome.py Tests.test_kernel_trace` and examine `trace.json` in Perfetto to see slot utilization per cycle.

2. **Modify** `perf_takehome.py` with your experimental idea.

3. **git commit**

4. **Run evaluation:** `bash eval/eval.sh > run.log 2>&1`

5. **Read results:** `grep "^cycles:" run.log`

6. If grep is empty, the run crashed. Run `tail -n 50 run.log` for the stack trace and fix the error.

7. **Record** results in `results.tsv`.

8. If cycles **decreased**, keep the commit. If cycles stayed same or increased: `git reset --hard HEAD~1`.

**Timeout:** If a run exceeds 5 minutes, kill it. The baseline takes ~10 seconds.

**NEVER STOP.** Run the loop until interrupted.

## Debugging Tips

- **Trace visualization:** Run `python perf_takehome.py Tests.test_kernel_trace` to generate `trace.json`, then `python watch_trace.py` in another terminal. Open the browser link and click "Open Perfetto" to see per-cycle slot utilization.
- **Debug assertions:** Use `("debug", ("compare", addr, key))` and `("debug", ("vcompare", addr, keys))` slots to check intermediate values against the reference kernel. Debug slots are free (0 cycles) and ignored by submission tests.
- **Quick test:** `python perf_takehome.py Tests.test_kernel_cycles` runs with seed=123 for fast iteration. The submission tests run 8 unseeded correctness checks.
- **Scratch map:** The `debug_info.scratch_map` shows named scratch allocations for easier debugging.
