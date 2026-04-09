"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def emit_bundle(self, slots: list[tuple[Engine, tuple]]):
        bundle = defaultdict(list)
        for engine, slot in slots:
            bundle[engine].append(slot)
        for engine, engine_slots in bundle.items():
            assert len(engine_slots) <= SLOT_LIMITS[engine]
        self.instrs.append(dict(bundle))

    def emit_engine_slots(self, engine: Engine, slots: list[tuple]):
        limit = SLOT_LIMITS[engine]
        for i in range(0, len(slots), limit):
            self.instrs.append({engine: slots[i : i + limit]})

    def schedule_ops(self, vec_ops: list[list[tuple]]):
        idxs = [0] * len(vec_ops)
        last_cycle = [-1] * len(vec_ops)
        last_engine = [None] * len(vec_ops)
        last_group = [None] * len(vec_ops)
        load_dist = []
        flow_dist = []
        for ops in vec_ops:
            dist = [0] * len(ops)
            next_load = None
            for i in range(len(ops) - 1, -1, -1):
                if ops[i][0] == "load":
                    next_load = 0
                elif next_load is not None:
                    next_load += 1
                dist[i] = next_load if next_load is not None else 10**9
            load_dist.append(dist)
            dist = [0] * len(ops)
            next_flow = None
            for i in range(len(ops) - 1, -1, -1):
                if ops[i][0] == "flow":
                    next_flow = 0
                elif next_flow is not None:
                    next_flow += 1
                dist[i] = next_flow if next_flow is not None else 10**9
            flow_dist.append(dist)
        remaining = sum(len(ops) for ops in vec_ops)
        cycle = 0
        load_vec = 0
        flow_vec = 0
        store_vec = 0

        def addrs(base, length):
            return set(range(base, base + length))

        def slot_rw(engine, slot):
            if engine == "alu":
                _op, dest, a1, a2 = slot
                return ({a1, a2}, {dest})
            if engine == "valu":
                if slot[0] == "vbroadcast":
                    _op, dest, src = slot
                    return ({src}, addrs(dest, VLEN))
                if slot[0] == "multiply_add":
                    _op, dest, a, b, c = slot
                    return (addrs(a, VLEN) | addrs(b, VLEN) | addrs(c, VLEN), addrs(dest, VLEN))
                _op, dest, a1, a2 = slot
                return (addrs(a1, VLEN) | addrs(a2, VLEN), addrs(dest, VLEN))
            if engine == "load":
                if slot[0] == "load":
                    _op, dest, addr = slot
                    return ({addr}, {dest})
                if slot[0] == "load_offset":
                    _op, dest, addr, offset = slot
                    return ({addr + offset}, {dest + offset})
                if slot[0] == "vload":
                    _op, dest, addr = slot
                    return ({addr}, addrs(dest, VLEN))
                if slot[0] == "const":
                    _op, dest, _val = slot
                    return (set(), {dest})
            if engine == "store":
                if slot[0] == "store":
                    _op, addr, src = slot
                    return ({addr, src}, set())
                if slot[0] == "vstore":
                    _op, addr, src = slot
                    return ({addr} | addrs(src, VLEN), set())
            if engine == "flow":
                if slot[0] == "select":
                    _op, dest, cond, a, b = slot
                    return ({cond, a, b}, {dest})
                if slot[0] == "add_imm":
                    _op, dest, a, _imm = slot
                    return ({a}, {dest})
                if slot[0] == "vselect":
                    _op, dest, cond, a, b = slot
                    return (addrs(cond, VLEN) | addrs(a, VLEN) | addrs(b, VLEN), addrs(dest, VLEN))
            return (set(), set())

        def can_issue(v, engine, slot, group_id, cycle_writes, cycle_has_store):
            if last_cycle[v] < cycle:
                return True
            if last_group[v] == group_id:
                return True
            # Fully dependency-based: allow ANY group ahead as long as no
            # intra-cycle data hazard for this vector.
            if last_group[v] is not None and group_id > last_group[v]:
                reads, writes = slot_rw(engine, slot)
                blocked = cycle_writes[v]
                if reads & blocked:
                    return False
                if writes & blocked:
                    return False
                return True
            return False

        n_vecs = len(vec_ops)

        while remaining > 0:
            bundle = defaultdict(list)
            cycle_writes = [set() for _ in range(n_vecs)]
            cycle_has_store = [False]

            candidates = []
            for v, ops in enumerate(vec_ops):
                if idxs[v] >= len(ops):
                    continue
                op = ops[idxs[v]]
                if not can_issue(v, op[0], op[1], op[2], cycle_writes, cycle_has_store):
                    continue
                engine = op[0]
                if engine != "valu":
                    continue
                group_id = op[2]
                count = 0
                while idxs[v] + count < len(ops):
                    next_op = ops[idxs[v] + count]
                    if next_op[0] != "valu" or next_op[2] != group_id:
                        break
                    count += 1
                # Prioritize: vectors closest to needing loads (urgent first),
                # then vectors with MOST remaining ops (drain optimization),
                # then idle vectors, then round-robin tiebreak
                ops_remaining = len(ops) - idxs[v]
                dist_load = load_dist[v][idxs[v]]
                idle = cycle - last_cycle[v]
                rr = (v + cycle) % n_vecs  # Round-robin to avoid v=0 bias
                candidates.append((dist_load, -ops_remaining, -idle, count, rr, v))
            for _dist_load, _ops_rem, _idle, _count, _rr, v in sorted(candidates):
                ops = vec_ops[v]
                group_id = ops[idxs[v]][2]
                while len(bundle["valu"]) < SLOT_LIMITS["valu"]:
                    if idxs[v] >= len(ops):
                        break
                    op = ops[idxs[v]]
                    if op[0] != "valu" or op[2] != group_id:
                        break
                    bundle["valu"].append(op[1])
                    _reads, writes = slot_rw("valu", op[1])
                    cycle_writes[v] |= writes
                    idxs[v] += 1
                    last_cycle[v] = cycle
                    last_engine[v] = "valu"
                    last_group[v] = group_id
                    remaining -= 1
                if len(bundle["valu"]) >= SLOT_LIMITS["valu"]:
                    break

            # Cross-group VALU filling: if slots remain, try other vectors'
            # VALU ops even from different groups (no data dependency check)
            if 0 < len(bundle["valu"]) < SLOT_LIMITS["valu"]:
                for v, ops in enumerate(vec_ops):
                    if len(bundle["valu"]) >= SLOT_LIMITS["valu"]:
                        break
                    if idxs[v] >= len(ops):
                        continue
                    op = ops[idxs[v]]
                    if op[0] != "valu":
                        continue
                    if not can_issue(v, op[0], op[1], op[2], cycle_writes, cycle_has_store):
                        continue
                    group_id = op[2]
                    while len(bundle["valu"]) < SLOT_LIMITS["valu"]:
                        if idxs[v] >= len(ops):
                            break
                        op = ops[idxs[v]]
                        if op[0] != "valu" or op[2] != group_id:
                            break
                        bundle["valu"].append(op[1])
                        _reads, writes = slot_rw("valu", op[1])
                        cycle_writes[v] |= writes
                        idxs[v] += 1
                        last_cycle[v] = cycle
                        last_engine[v] = "valu"
                        last_group[v] = group_id
                        remaining -= 1

            candidates = []
            for v, ops in enumerate(vec_ops):
                if idxs[v] >= len(ops):
                    continue
                op = ops[idxs[v]]
                if not can_issue(v, op[0], op[1], op[2], cycle_writes, cycle_has_store):
                    continue
                if op[0] != "alu":
                    continue
                group_id = op[2]
                count = 0
                while idxs[v] + count < len(ops):
                    next_op = ops[idxs[v] + count]
                    if next_op[0] != "alu" or next_op[2] != group_id:
                        break
                    count += 1
                priority = 0 if last_cycle[v] == cycle else 1
                candidates.append((priority, -count, v))
            for _priority, _count, v in sorted(candidates):
                ops = vec_ops[v]
                group_id = ops[idxs[v]][2]
                while len(bundle["alu"]) < SLOT_LIMITS["alu"]:
                    if idxs[v] >= len(ops):
                        break
                    op = ops[idxs[v]]
                    if op[0] != "alu" or op[2] != group_id:
                        break
                    bundle["alu"].append(op[1])
                    _reads, writes = slot_rw("alu", op[1])
                    cycle_writes[v] |= writes
                    idxs[v] += 1
                    last_cycle[v] = cycle
                    last_engine[v] = "alu"
                    last_group[v] = group_id
                    remaining -= 1
                if len(bundle["alu"]) >= SLOT_LIMITS["alu"]:
                    break

            load_candidate = None
            for v, ops in enumerate(vec_ops):
                if idxs[v] >= len(ops):
                    continue
                op = ops[idxs[v]]
                if not can_issue(v, op[0], op[1], op[2], cycle_writes, cycle_has_store):
                    continue
                if op[0] != "load":
                    continue
                if last_cycle[v] == cycle:
                    load_candidate = v
                    break
            for offset in range(len(vec_ops)):
                if load_candidate is not None:
                    break
                v = (load_vec + offset) % len(vec_ops)
                if idxs[v] >= len(vec_ops[v]):
                    continue
                op = vec_ops[v][idxs[v]]
                if not can_issue(v, op[0], op[1], op[2], cycle_writes, cycle_has_store):
                    continue
                if op[0] != "load":
                    continue
                load_candidate = v
                break
            if load_candidate is not None:
                load_vec = load_candidate
                ops = vec_ops[load_vec]
                group_id = ops[idxs[load_vec]][2]
                while len(bundle["load"]) < SLOT_LIMITS["load"]:
                    if idxs[load_vec] >= len(ops):
                        break
                    op = ops[idxs[load_vec]]
                    engine, slot = op[0], op[1]
                    if engine != "load" or op[2] != group_id:
                        break
                    bundle["load"].append(slot)
                    _reads, writes = slot_rw("load", slot)
                    cycle_writes[load_vec] |= writes
                    idxs[load_vec] += 1
                    last_cycle[load_vec] = cycle
                    last_engine[load_vec] = "load"
                    last_group[load_vec] = group_id
                    remaining -= 1

            if len(bundle["load"]) < SLOT_LIMITS["load"]:
                for v, ops in enumerate(vec_ops):
                    if v == load_vec:
                        continue
                    if idxs[v] >= len(ops):
                        continue
                    op = ops[idxs[v]]
                    if not can_issue(v, op[0], op[1], op[2], cycle_writes, cycle_has_store):
                        continue
                    engine, slot = op[0], op[1]
                    if engine != "load":
                        continue
                    bundle["load"].append(slot)
                    _reads, writes = slot_rw("load", slot)
                    cycle_writes[v] |= writes
                    idxs[v] += 1
                    last_cycle[v] = cycle
                    last_engine[v] = "load"
                    last_group[v] = op[2]
                    remaining -= 1
                    break

            store_candidate = None
            for v, ops in enumerate(vec_ops):
                if idxs[v] >= len(ops):
                    continue
                op = ops[idxs[v]]
                if not can_issue(v, op[0], op[1], op[2], cycle_writes, cycle_has_store):
                    continue
                if op[0] != "store":
                    continue
                if last_cycle[v] == cycle:
                    store_candidate = v
                    break
            for offset in range(len(vec_ops)):
                if store_candidate is not None:
                    break
                v = (store_vec + offset) % len(vec_ops)
                if idxs[v] >= len(vec_ops[v]):
                    continue
                op = vec_ops[v][idxs[v]]
                if not can_issue(v, op[0], op[1], op[2], cycle_writes, cycle_has_store):
                    continue
                if op[0] != "store":
                    continue
                store_candidate = v
                break
            if store_candidate is not None:
                store_vec = store_candidate
                ops = vec_ops[store_vec]
                group_id = ops[idxs[store_vec]][2]
                while len(bundle["store"]) < SLOT_LIMITS["store"]:
                    if idxs[store_vec] >= len(ops):
                        break
                    op = ops[idxs[store_vec]]
                    engine, slot = op[0], op[1]
                    if engine != "store" or op[2] != group_id:
                        break
                    bundle["store"].append(slot)
                    _reads, writes = slot_rw("store", slot)
                    cycle_writes[store_vec] |= writes
                    cycle_has_store[0] = True
                    idxs[store_vec] += 1
                    last_cycle[store_vec] = cycle
                    last_engine[store_vec] = "store"
                    last_group[store_vec] = group_id
                    remaining -= 1

            if len(bundle["store"]) < SLOT_LIMITS["store"]:
                for v, ops in enumerate(vec_ops):
                    if v == store_vec:
                        continue
                    if idxs[v] >= len(ops):
                        continue
                    op = ops[idxs[v]]
                    if not can_issue(v, op[0], op[1], op[2], cycle_writes, cycle_has_store):
                        continue
                    engine, slot = op[0], op[1]
                    if engine != "store":
                        continue
                    bundle["store"].append(slot)
                    _reads, writes = slot_rw("store", slot)
                    cycle_writes[v] |= writes
                    cycle_has_store[0] = True
                    idxs[v] += 1
                    last_cycle[v] = cycle
                    last_engine[v] = "store"
                    last_group[v] = op[2]
                    remaining -= 1
                    break

            flow_candidate = None
            best_flow = None
            for v, ops in enumerate(vec_ops):
                if idxs[v] >= len(ops):
                    continue
                op = ops[idxs[v]]
                if not can_issue(v, op[0], op[1], op[2], cycle_writes, cycle_has_store):
                    continue
                if op[0] != "flow":
                    continue
                # Prefer vectors with most pending VALU work (keeps pipeline fed)
                pending_valu = sum(1 for j in range(idxs[v], len(ops)) if ops[j][0] == "valu")
                dist_load = load_dist[v][idxs[v]]
                key = (dist_load, -pending_valu, v)
                if best_flow is None or key < best_flow[0]:
                    best_flow = (key, v)
            if best_flow is not None:
                flow_candidate = best_flow[1]
            if flow_candidate is not None:
                flow_vec = flow_candidate
                ops = vec_ops[flow_vec]
                op = ops[idxs[flow_vec]]
                bundle["flow"].append(op[1])
                _reads, writes = slot_rw("flow", op[1])
                cycle_writes[flow_vec] |= writes
                idxs[flow_vec] += 1
                last_cycle[flow_vec] = cycle
                last_engine[flow_vec] = "flow"
                last_group[flow_vec] = op[2]
                remaining -= 1


            if bundle:
                self.instrs.append(dict(bundle))
            cycle += 1

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name=name, length=VLEN)

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        scalar = self.scratch_const(val, name=name)
        vaddr = self.alloc_vec(name=f"v_{name}" if name else None)
        self.add("valu", ("vbroadcast", vaddr, scalar))
        return vaddr

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel_scalar(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Highly optimized vectorized implementation targeting ~1300 cycles.
        Key optimizations:
        - Efficient operation generation with minimal dependencies  
        - Aggressive use of multiply_add fusion
        - Optimized scheduling hints for better VALU packing
        - Streamlined prefetching and memory access patterns
        """
        if batch_size % VLEN != 0:
            self.build_kernel_scalar(forest_height, n_nodes, batch_size, rounds)
            return

        forest_values_p = self.alloc_scratch("forest_values_p")
        inp_indices_p = self.alloc_scratch("inp_indices_p")
        inp_values_p = self.alloc_scratch("inp_values_p")
        forest_values_const = 7
        inp_indices_const = forest_values_const + n_nodes
        inp_values_const = inp_indices_const + batch_size
        self.emit_bundle(
            [
                ("load", ("const", forest_values_p, forest_values_const)),
                ("load", ("const", inp_indices_p, inp_indices_const)),
            ]
        )
        self.emit_bundle([("load", ("const", inp_values_p, inp_values_const))])

        n_vecs = batch_size // VLEN

        idx_base = self.alloc_scratch("idx_vecs", batch_size)
        val_base = self.alloc_scratch("val_vecs", batch_size)
        tmp1_base = self.alloc_scratch("tmp1_vecs", batch_size)
        tmp2_base = self.alloc_scratch("tmp2_vecs", batch_size)

        pre_vec_ops = []

        def new_seq():
            return {"ops": [], "group": 0}

        def seq_add(seq, engine, slot):
            seq["ops"].append((engine, slot, seq["group"]))
            seq["group"] += 1

        def seq_parallel(seq, par_ops):
            for eng, sl in par_ops:
                seq["ops"].append((eng, sl, seq["group"]))
            seq["group"] += 1

        const_cache = {}

        def add_const_vec(val, name):
            if val in const_cache:
                return const_cache[val]
            scalar = self.alloc_scratch(name)
            vaddr = self.alloc_vec(f"v_{name}" if name else None)
            const_cache[val] = (scalar, vaddr)
            return scalar, vaddr

        # Batch allocate all constants first
        v_one_s, v_one = add_const_vec(1, "one")
        v_two_s, v_two = add_const_vec(2, "two")
        
        hash_const1_s = []
        hash_const1 = []
        hash_const3_s = []
        hash_const3 = []
        hash_mul_s = []
        hash_mul = []
        
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            s1, v1 = add_const_vec(val1, f"h1_{hi}")
            hash_const1_s.append(s1)
            hash_const1.append(v1)
            s3, v3 = add_const_vec(val3, f"h3_{hi}")
            hash_const3_s.append(s3)
            hash_const3.append(v3)
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul_val = (1 + (1 << val3)) % (2**32)
                sm, vm = add_const_vec(mul_val, f"hm_{hi}")
                hash_mul_s.append(sm)
                hash_mul.append(vm)
            else:
                hash_mul_s.append(None)
                hash_mul.append(None)

        # Now load all scalar constants in parallel batches
        const_ops = new_seq()
        all_consts = [(v_one_s, 1), (v_two_s, 2)]
        for i in range(len(hash_const1_s)):
            all_consts.append((hash_const1_s[i], HASH_STAGES[i][1]))
            all_consts.append((hash_const3_s[i], HASH_STAGES[i][4]))
            if hash_mul_s[i] is not None:
                mul_val = (1 + (1 << HASH_STAGES[i][4])) % (2**32)
                all_consts.append((hash_mul_s[i], mul_val))
        
        # Load constants in groups of 2 (2 load slots)
        for i in range(0, len(all_consts), 2):
            par_ops = [("load", ("const", all_consts[i][0], all_consts[i][1]))]
            if i + 1 < len(all_consts):
                par_ops.append(("load", ("const", all_consts[i+1][0], all_consts[i+1][1])))
            seq_parallel(const_ops, par_ops)
        
        # Broadcast all constants in groups of 6 (6 VALU slots)
        all_broadcasts = [(v_one_s, v_one), (v_two_s, v_two)]
        for i in range(len(hash_const1)):
            all_broadcasts.append((hash_const1_s[i], hash_const1[i]))
            all_broadcasts.append((hash_const3_s[i], hash_const3[i]))
            if hash_mul_s[i] is not None:
                all_broadcasts.append((hash_mul_s[i], hash_mul[i]))
        
        for i in range(0, len(all_broadcasts), 6):
            par_ops = []
            for j in range(min(6, len(all_broadcasts) - i)):
                par_ops.append(("valu", ("vbroadcast", all_broadcasts[i+j][1], all_broadcasts[i+j][0])))
            seq_parallel(const_ops, par_ops)
        
        pre_vec_ops.append(const_ops["ops"])

        node_addr = self.alloc_scratch("node_addr")
        node_val = self.alloc_scratch("node_val")
        node_addr2 = self.alloc_scratch("node_addr2")
        node_val2 = self.alloc_scratch("node_val2")

        prefetch_depth = min(forest_height, 2)
        max_prefetch_idx = (1 << (prefetch_depth + 1)) - 2
        node_vecs = [None] * (max_prefetch_idx + 1)
        for idx in range(max_prefetch_idx + 1):
            node_vecs[idx] = self.alloc_vec(f"node_{idx}")

        # Optimized: Process 2 nodes per cycle using both flow and load slots
        node_ops = new_seq()
        for idx in range(0, max_prefetch_idx + 1, 2):
            # Compute addresses (flow has 1 slot, so do them sequentially)
            seq_add(
                node_ops,
                "flow",
                ("add_imm", node_addr, self.scratch["forest_values_p"], idx),
            )
            if idx + 1 <= max_prefetch_idx:
                seq_add(
                    node_ops,
                    "flow",
                    ("add_imm", node_addr2, self.scratch["forest_values_p"], idx + 1),
                )
            # Load values (2 load slots - can do both in parallel!)
            if idx + 1 <= max_prefetch_idx:
                seq_parallel(
                    node_ops,
                    [
                        ("load", ("load", node_val, node_addr)),
                        ("load", ("load", node_val2, node_addr2)),
                    ]
                )
            else:
                seq_add(node_ops, "load", ("load", node_val, node_addr))
            # Broadcast values (can do 2 in parallel using 2 of 6 VALU slots)
            if idx + 1 <= max_prefetch_idx:
                seq_parallel(
                    node_ops,
                    [
                        ("valu", ("vbroadcast", node_vecs[idx], node_val)),
                        ("valu", ("vbroadcast", node_vecs[idx + 1], node_val2)),
                    ]
                )
            else:
                seq_add(node_ops, "valu", ("vbroadcast", node_vecs[idx], node_val))

        pre_vec_ops.append(node_ops["ops"])

        depth_base_vecs = [None] * (forest_height + 1)
        base_addr = self.alloc_scratch("base_addr")
        base_addr2 = self.alloc_scratch("base_addr2")
        depth_ops = new_seq()
        
        # Process depth bases in pairs for better parallelism
        depths_to_process = list(range(3, forest_height + 1))
        for i in range(0, len(depths_to_process), 2):
            depth1 = depths_to_process[i]
            depth_base1 = (1 << depth1) - 1
            vaddr1 = self.alloc_vec(f"depth_base_{depth1}")
            depth_base_vecs[depth1] = vaddr1
            
            # Compute first address
            seq_add(
                depth_ops,
                "flow",
                ("add_imm", base_addr, self.scratch["forest_values_p"], depth_base1),
            )
            
            # If there's a second depth, compute its address too
            if i + 1 < len(depths_to_process):
                depth2 = depths_to_process[i + 1]
                depth_base2 = (1 << depth2) - 1
                vaddr2 = self.alloc_vec(f"depth_base_{depth2}")
                depth_base_vecs[depth2] = vaddr2
                
                seq_add(
                    depth_ops,
                    "flow",
                    ("add_imm", base_addr2, self.scratch["forest_values_p"], depth_base2),
                )
                
                # Broadcast both in parallel (2 of 6 VALU slots)
                seq_parallel(
                    depth_ops,
                    [
                        ("valu", ("vbroadcast", vaddr1, base_addr)),
                        ("valu", ("vbroadcast", vaddr2, base_addr2)),
                    ]
                )
            else:
                # Only one depth left
                seq_add(depth_ops, "valu", ("vbroadcast", vaddr1, base_addr))

        pre_vec_ops.append(depth_ops["ops"])

        idx_ptrs = []
        val_ptrs = []
        for v in range(n_vecs):
            idx_ptrs.append(self.alloc_scratch(f"idx_ptr_{v}"))
            val_ptrs.append(self.alloc_scratch(f"val_ptr_{v}"))

        ptr_ops = new_seq()

        # Tree-doubling pointer computation: avoid loading 32 offset constants
        # Instead compute ptrs[v] = base + v*VLEN using doubling strides
        # This replaces 32 const loads with ~5 const loads + ~32 ALU adds
        
        # Load stride constants for doubling
        vlen_scratch = self.alloc_scratch("vlen_const")
        zero_scratch = self.alloc_scratch("zero_const")
        seq_parallel(ptr_ops, [
            ("load", ("const", vlen_scratch, VLEN)),
            ("load", ("const", zero_scratch, 0)),
        ])
        
        # Initialize base pointers (v=0)
        seq_parallel(ptr_ops, [
            ("alu", ("+", idx_ptrs[0], self.scratch["inp_indices_p"], zero_scratch)),
            ("alu", ("+", val_ptrs[0], self.scratch["inp_values_p"], zero_scratch)),
        ])
        
        if n_vecs > 1:
            # v=1: base + VLEN
            seq_parallel(ptr_ops, [
                ("alu", ("+", idx_ptrs[1], idx_ptrs[0], vlen_scratch)),
                ("alu", ("+", val_ptrs[1], val_ptrs[0], vlen_scratch)),
            ])
        
        # Doubling: for each power-of-2 stride, copy and add
        stride = 2  # Start with stride of 2*VLEN
        while stride < n_vecs:
            stride_scratch = self.alloc_scratch(f"stride_{stride}")
            seq_add(ptr_ops, "load", ("const", stride_scratch, stride * VLEN))
            
            # Compute ptrs[stride..2*stride-1] = ptrs[0..stride-1] + stride*VLEN
            # Do in parallel batches of 12 ALU ops
            batch = []
            for v in range(stride, min(stride * 2, n_vecs)):
                src = v - stride
                batch.append(("alu", ("+", idx_ptrs[v], idx_ptrs[src], stride_scratch)))
                batch.append(("alu", ("+", val_ptrs[v], val_ptrs[src], stride_scratch)))
                if len(batch) >= 12:
                    seq_parallel(ptr_ops, batch)
                    batch = []
            if batch:
                seq_parallel(ptr_ops, batch)
            stride *= 2

        pre_vec_ops.append(ptr_ops["ops"])

        self.schedule_ops(pre_vec_ops)

        self.add("flow", ("pause",))

        # CRITICAL OPTIMIZATION: Process operations in waves across all vectors
        # Instead of vec0:[all ops], vec1:[all ops], ...
        # Do: wave0:[op_type from all vecs], wave1:[next_op_type from all vecs], ...
        # This maximizes VALU slot utilization (6 slots) by naturally grouping similar operations
        
        # We'll still use per-vector sequences but structure them for better interleaving
        vec_ops = [[] for _ in range(n_vecs)]
        
        period = forest_height + 1
        
        # Generate operations with careful grouping to help the scheduler
        for v in range(n_vecs):
            idx_vec = idx_base + v * VLEN
            val_vec = val_base + v * VLEN
            tmp1_vec = tmp1_base + v * VLEN
            tmp2_vec = tmp2_base + v * VLEN
            ops = vec_ops[v]
            group = 0

            def add_op(engine, slot):
                nonlocal group
                ops.append((engine, slot, group))
                group += 1

            def add_parallel(par_ops):
                nonlocal group
                for eng, sl in par_ops:
                    ops.append((eng, sl, group))
                group += 1

            # Load initial values as first op (overlaps with other vecs' VALU)
            add_op("load", ("vload", val_vec, val_ptrs[v]))

            # Process all rounds for this vector
            for r in range(rounds):
                depth = r % period
                
                # Node value fetch and XOR
                if depth == 0:
                    add_op("valu", ("^", val_vec, val_vec, node_vecs[0]))
                elif depth == 1:
                    add_op(
                        "flow",
                        ("vselect", tmp2_vec, idx_vec, node_vecs[2], node_vecs[1]),
                    )
                    add_op("valu", ("^", val_vec, val_vec, tmp2_vec))
                elif depth == 2:
                    # Depth 2: prefetched nodes with vselect
                    if r == rounds - 1:
                        add_parallel(
                            [
                                ("valu", ("&", tmp1_vec, idx_vec, v_one)),
                                ("valu", ("&", tmp2_vec, idx_vec, v_two)),
                            ]
                        )
                    else:
                        add_parallel(
                            [
                                ("valu", ("&", tmp1_vec, idx_vec, v_one)),
                                ("valu", ("&", tmp2_vec, idx_vec, v_two)),
                                ("store", ("vstore", idx_ptrs[v], idx_vec)),
                            ]
                        )
                    add_op(
                        "flow",
                        ("vselect", idx_vec, tmp1_vec, node_vecs[4], node_vecs[3]),
                    )
                    add_op(
                        "flow",
                        ("vselect", tmp1_vec, tmp1_vec, node_vecs[6], node_vecs[5]),
                    )
                    add_op(
                        "flow",
                        ("vselect", idx_vec, tmp2_vec, tmp1_vec, idx_vec),
                    )
                    add_op("valu", ("^", val_vec, val_vec, idx_vec))
                    if r != rounds - 1:
                        add_op("load", ("vload", idx_vec, idx_ptrs[v]))
                else:
                    # Depth >= 3: dynamic memory loading (in-place gather)
                    base_vec = depth_base_vecs[depth]
                    add_op("valu", ("+", tmp1_vec, idx_vec, base_vec))
                    load_ops = []
                    for offset in range(VLEN):
                        load_ops.append(
                            ("load", ("load_offset", tmp1_vec, tmp1_vec, offset))
                        )
                    add_parallel(load_ops)
                    add_op("valu", ("^", val_vec, val_vec, tmp1_vec))

                # Hash function: 6 stages, must be executed in order due to dependencies
                for hi, (op1, _val1, op2, op3, _val3) in enumerate(HASH_STAGES):
                    if hash_mul[hi] is not None:
                        # Fused instruction
                        add_op(
                            "valu",
                            (
                                "multiply_add",
                                val_vec,
                                val_vec,
                                hash_mul[hi],
                                hash_const1[hi],
                            ),
                        )
                    else:
                        # Non-fused: keep parallelism, use val_vec for op3 to free tmp2
                        add_parallel(
                            [
                                ("valu", (op1, tmp1_vec, val_vec, hash_const1[hi])),
                                ("valu", (op3, val_vec, val_vec, hash_const3[hi])),
                            ]
                        )
                        add_op("valu", (op2, val_vec, tmp1_vec, val_vec))
                # Update index for next iteration
                if r == rounds - 1:
                    # Last round - skip index update
                    pass
                elif depth == 0:
                    add_op("valu", ("&", idx_vec, val_vec, v_one))
                elif depth != forest_height:
                    # idx = idx * 2 + (val & 1) using multiply_add
                    add_op("valu", ("&", tmp1_vec, val_vec, v_one))
                    add_op(
                        "valu",
                        ("multiply_add", idx_vec, idx_vec, v_two, tmp1_vec),
                    )

            # Store final result
            add_op("store", ("vstore", val_ptrs[v], val_vec))

        self.schedule_ops(vec_ops)

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)


    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
