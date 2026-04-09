# Architecture Quick Reference

## Machine Overview

Custom VLIW SIMD processor simulator. Single-core, in-order, no branch prediction. All parallelism is explicit in the instruction bundles.

## Execution Model

- Each instruction is a Python dict mapping engine names to lists of slot tuples
- All slots in one instruction execute **in parallel in the same cycle**
- Writes take effect at **end of cycle** (read-before-write within a cycle)
- An instruction containing only `debug` slots costs 0 cycles
- Cores execute instructions sequentially by program counter (pc)

## Engine Slot Limits

| Engine | Slots/cycle | Purpose |
|--------|-------------|---------|
| `alu`  | 12 | Scalar 32-bit arithmetic |
| `valu` | 6 | Vector arithmetic (VLEN=8) |
| `load` | 2 | Memory reads + immediates |
| `store`| 2 | Memory writes |
| `flow` | 1 | Control flow + conditional select |
| `debug`| 64 | Free debug assertions |

## Constants

| Name | Value | Notes |
|------|-------|-------|
| VLEN | 8 | Vector width |
| N_CORES | 1 | Multicore disabled |
| SCRATCH_SIZE | 1536 | Words of scratch space |

## Memory Model

- **Main memory**: Flat array of 32-bit words. Addressed by index.
- **Scratch space**: 1536 words of fast storage. All operands in instructions are scratch addresses (except const values and jump targets).
- Reading scratch is free (same cycle). Loading from main memory uses a load slot.

## Complete Instruction Set

### ALU (12 slots/cycle)

All operations: `result = (a1 op a2) % 2^32`

```
("+", dest, a1, a2)     addition
("-", dest, a1, a2)     subtraction
("*", dest, a1, a2)     multiplication
("//", dest, a1, a2)    integer division
("%", dest, a1, a2)     modulo
("^", dest, a1, a2)     XOR
("&", dest, a1, a2)     AND
("|", dest, a1, a2)     OR
("<<", dest, a1, a2)    left shift
(">>", dest, a1, a2)    right shift
("<", dest, a1, a2)     less-than (returns 0 or 1)
("==", dest, a1, a2)    equality (returns 0 or 1)
("cdiv", dest, a1, a2)  ceiling division: (a1 + a2 - 1) // a2
```

All operands are scratch addresses. `dest` receives the result.

### VALU (6 slots/cycle)

Element-wise vector operations on VLEN=8 contiguous scratch words:

```
(op, dest, a1, a2)                 dest[i] = a1[i] op a2[i]  for i in 0..7
                                   Same ops as ALU: +, -, *, //, %, ^, &, |, <<, >>, <, ==

("vbroadcast", dest, src)          dest[0..7] = scratch[src]  (scalar -> vector)

("multiply_add", dest, a, b, c)   dest[i] = (a[i] * b[i] + c[i]) % 2^32
                                   Fused multiply-add in a single slot
```

Vector operands: `dest`, `a1`, `a2`, `a`, `b`, `c` are base scratch addresses. The operation acts on addresses `base+0` through `base+7`.

### LOAD (2 slots/cycle)

```
("load", dest, addr)               scratch[dest] = mem[scratch[addr]]
("load_offset", dest, addr, off)   scratch[dest+off] = mem[scratch[addr+off]]
("vload", dest, addr)              scratch[dest+i] = mem[scratch[addr]+i]  for i in 0..7
("const", dest, val)               scratch[dest] = val  (immediate, val is a literal)
```

Note: `vload` address is a scalar (scratch[addr] gives the base memory address). `load_offset` is useful for treating vector dest/addr as a block.

### STORE (2 slots/cycle)

```
("store", addr, src)     mem[scratch[addr]] = scratch[src]
("vstore", addr, src)    mem[scratch[addr]+i] = scratch[src+i]  for i in 0..7
```

Note: `vstore` address is a scalar. Both read scratch[addr] for the memory address.

### FLOW (1 slot/cycle)

```
("select", dest, cond, a, b)       scratch[dest] = scratch[a] if scratch[cond]!=0 else scratch[b]
("vselect", dest, cond, a, b)      per-lane select: dest[i] = a[i] if cond[i]!=0 else b[i]
("add_imm", dest, a, imm)          scratch[dest] = (scratch[a] + imm) % 2^32
("cond_jump", cond, addr)          if scratch[cond]!=0: pc = addr (absolute)
("cond_jump_rel", cond, offset)    if scratch[cond]!=0: pc += offset (relative)
("jump", addr)                     pc = addr
("jump_indirect", addr)            pc = scratch[addr]
("halt",)                          stop core
("pause",)                         pause core (ignored by submission tests)
("trace_write", val)               append scratch[val] to trace buffer
("coreid", dest)                   scratch[dest] = core.id
```

### DEBUG (64 slots/cycle, 0 cycle cost)

```
("compare", loc, key)              assert scratch[loc] == value_trace[key]
("vcompare", loc, keys)            assert scratch[loc:loc+8] == [value_trace[k] for k in keys]
("comment", text)                  no-op annotation
```

Debug slots are completely free and ignored by submission tests (`machine.enable_debug = False`).

## Memory Layout

The `build_mem_image` function creates this layout:

```
mem[0] = rounds           (16)
mem[1] = n_nodes           (2047)
mem[2] = batch_size        (256)
mem[3] = forest_height     (10)
mem[4] = forest_values_p   (pointer to tree values)
mem[5] = inp_indices_p     (pointer to walker indices)
mem[6] = inp_values_p      (pointer to walker values)
mem[7] = extra_room_p      (pointer to extra workspace)
mem[8..8+2046] = tree node values
mem[8+2047..] = walker indices, then walker values, then extra room
```

## Hash Function

The `myhash` function applies 6 stages. Each stage:
```
a = op2(op1(a, const), op3(a, shift_amount))
```

Stages:
```
Stage 0: a = (a + 0x7ED55D16) + (a << 12)      ops: +, +, <<
Stage 1: a = (a ^ 0xC761C23C) ^ (a >> 19)      ops: ^, ^, >>
Stage 2: a = (a + 0x165667B1) + (a << 5)       ops: +, +, <<
Stage 3: a = (a + 0xD3A2646C) ^ (a << 9)       ops: +, ^, <<
Stage 4: a = (a + 0xFD7046C5) + (a << 3)       ops: +, +, <<
Stage 5: a = (a ^ 0xB55A4F09) ^ (a >> 16)      ops: ^, ^, >>
```

Each stage requires 3 ALU/VALU operations in the naive implementation, but stages with `+` and `<<` can potentially be fused using `multiply_add`.
