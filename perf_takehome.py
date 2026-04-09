"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

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

class AutoScheduler:
    def __init__(self):
        self.bundles = defaultdict(lambda: defaultdict(list))
        self.ready_at = defaultdict(int)
        self.last_read_at = defaultdict(int)
        self.max_cycle = 0

    def add(self, engine, slot, inputs=(), outputs=(), min_cycle=0):
        def flatten(items):
            if isinstance(items, (list, range, tuple)): return list(items)
            return [items]
        
        in_list = flatten(inputs)
        out_list = flatten(outputs)
        
        # RAW: Wait for inputs to be ready
        min_raw = max([0] + [self.ready_at[i] for i in in_list])
        
        # WAR: Wait for previous reads to finish (Write can happen in same cycle as Read)
        min_war = max([0] + [self.last_read_at[o] for o in out_list])
        
        # WAW: Wait for previous writes availability
        min_waw = max([0] + [self.ready_at[o] for o in out_list])
        
        start_cycle = max(min_cycle, min_raw, min_war, min_waw)
        
        cycle = start_cycle
        while len(self.bundles[cycle][engine]) >= SLOT_LIMITS[engine]:
            cycle += 1
        
        self.bundles[cycle][engine].append(slot)
        self.max_cycle = max(self.max_cycle, cycle)
        
        # Update State
        for i in in_list:
            self.last_read_at[i] = max(self.last_read_at[i], cycle)
        for o in out_list:
            self.ready_at[o] = cycle + 1
            
        return cycle

    def get_instrs(self):
        return [dict(self.bundles[c]) for c in range(self.max_cycle + 1)]

class KernelBuilder:
    def __init__(self):
        self.scratch_ptr = 0
        self.scratch_debug = {}
        self.const_map = {}
        self.sched = AutoScheduler()

    def alloc(self, name=None, length=1):
        addr = self.scratch_ptr
        if name: self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space: {self.scratch_ptr}"
        return addr

    def get_const(self, val):
        if val in self.const_map: return self.const_map[val]
        addr = self.alloc(f"c_{val}")
        self.sched.add("load", ("const", addr, val), [], [addr])
        self.const_map[val] = addr
        return addr

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        num_chunks = batch_size // VLEN

        v_idx = [self.alloc(f"v_idx_{i}", VLEN) for i in range(num_chunks)]
        v_val = [self.alloc(f"v_val_{i}", VLEN) for i in range(num_chunks)]
        v_t1 = [self.alloc(f"v_t1_{i}", VLEN) for i in range(num_chunks)]
        v_t2 = [self.alloc(f"v_t2_{i}", VLEN) for i in range(num_chunks)]
        sc_tmps = [[self.alloc(f"sc_t_{i}_{vi}") for vi in range(2)] for i in range(num_chunks)]

        v_zero = self.alloc("v_0", VLEN)
        v_one = self.alloc("v_1", VLEN)
        v_two = self.alloc("v_2", VLEN)
        v_n_nodes = self.alloc("v_nn", VLEN)
        sc_meta = {k: self.alloc(k) for k in ["idx_p", "val_p", "tree_p", "nn"]}
        sc_m_tmp = self.alloc("sc_m_tmp")

        c0 = self.get_const(0); c1 = self.get_const(1); c2 = self.get_const(2)
        for addr, dest in zip([5, 6, 4, 1], sc_meta.values()):
            self.sched.add("load", ("load", dest, self.get_const(addr)), [self.get_const(addr)], [dest])

        self.sched.add("valu", ("vbroadcast", v_zero, c0), [c0], range(v_zero, v_zero+VLEN))
        self.sched.add("valu", ("vbroadcast", v_one, c1), [c1], range(v_one, v_one+VLEN))
        self.sched.add("valu", ("vbroadcast", v_two, c2), [c2], range(v_two, v_two+VLEN))
        self.sched.add("valu", ("vbroadcast", v_n_nodes, sc_meta["nn"]), [sc_meta["nn"]], range(v_n_nodes, v_n_nodes+VLEN))

        c3 = self.get_const(3)
        v_base_3 = self.alloc("v_base_3", VLEN)
        self.sched.add("valu", ("vbroadcast", v_base_3, c3), [c3], range(v_base_3, v_base_3+VLEN))

        v_h_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            vc1 = self.alloc(f"vc1_{hi}", VLEN); vc3 = self.alloc(f"vc3_{hi}", VLEN)
            self.sched.add("valu", ("vbroadcast", vc1, self.get_const(val1)), [self.get_const(val1)], range(vc1, vc1+VLEN))
            self.sched.add("valu", ("vbroadcast", vc3, self.get_const(val3)), [self.get_const(val3)], range(vc3, vc3+VLEN))
            vmul = None
            if op1 == "+" and op2 == "+" and op3 == "<<":
                vmul = self.alloc(f"vmul_{hi}", VLEN)
                self.sched.add("valu", ("vbroadcast", vmul, self.get_const((1 << val3) + 1)), [self.get_const((1 << val3) + 1)], range(vmul, vmul+VLEN))
            v_h_consts.append((vc1, vc3, vmul))

        mux_levels = 3
        n_mux_tmps = 1
        v_mux_tmps = [self.alloc(f"v_mux_tmp_{j}", VLEN) for j in range(n_mux_tmps)]
        mux_vals = []
        for d in range(mux_levels):
            level_vals = []
            start_node = (1 << d) - 1
            for k in range(1 << d):
                node_idx = start_node + k
                c_off = self.get_const(node_idx)
                self.sched.add("alu", ("+", sc_m_tmp, sc_meta["tree_p"], c_off), [sc_meta["tree_p"], c_off], [sc_m_tmp])
                v_node = self.alloc(f"v_mux_{node_idx}", VLEN)
                self.sched.add("load", ("load", sc_m_tmp, sc_m_tmp), [sc_m_tmp], [sc_m_tmp])
                self.sched.add("valu", ("vbroadcast", v_node, sc_m_tmp), [sc_m_tmp], range(v_node, v_node+VLEN))
                level_vals.append(v_node)
            mux_vals.append(level_vals)

        def emit_hash_index_wrap(i, xor_src, needs_wrap=True):
            curr_v = v_val[i]; curr_idx = v_idx[i]
            v_in = list(range(curr_v, curr_v+VLEN))
            idx_in = list(range(curr_idx, curr_idx+VLEN))
            t1 = list(range(v_t1[i], v_t1[i]+VLEN))
            t2 = list(range(v_t2[i], v_t2[i]+VLEN))
            src_in = list(range(xor_src, xor_src+VLEN))

            self.sched.add("valu", ("^", curr_v, curr_v, xor_src), v_in + src_in, v_in)
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                vc1, vc3, vmul = v_h_consts[hi]
                if vmul:
                    self.sched.add("valu", ("multiply_add", curr_v, curr_v, vmul, vc1), v_in + [vmul, vc1], v_in)
                else:
                    self.sched.add("valu", (op1, v_t1[i], curr_v, vc1), v_in + [vc1], t1)
                    self.sched.add("valu", (op3, v_t2[i], curr_v, vc3), v_in + [vc3], t2)
                    self.sched.add("valu", (op2, curr_v, v_t1[i], v_t2[i]), t1 + t2, v_in)

            for vi in range(VLEN):
                self.sched.add("alu", ("&", v_t1[i]+vi, curr_v+vi, c1), [curr_v+vi, c1], [v_t1[i]+vi])
            self.sched.add("valu", ("multiply_add", curr_idx, curr_idx, v_two, v_one), idx_in + [v_two, v_one], idx_in)
            for vi in range(VLEN):
                self.sched.add("alu", ("+", curr_idx+vi, curr_idx+vi, v_t1[i]+vi), [curr_idx+vi, v_t1[i]+vi], [curr_idx+vi])
            if needs_wrap:
                self.sched.add("valu", ("<", v_t1[i], curr_idx, v_n_nodes), idx_in + [v_n_nodes], t1)
                self.sched.add("flow", ("vselect", curr_idx, v_t1[i], curr_idx, v_zero), t1 + idx_in + [v_zero], idx_in)

        def emit_round(i, round_idx):
            if round_idx == 0:
                ic = self.get_const(i * VLEN)
                st = sc_tmps[i][0]
                self.sched.add("alu", ("+", st, sc_meta["idx_p"], ic), [sc_meta["idx_p"], ic], [st])
                self.sched.add("load", ("vload", v_idx[i], st), [st], range(v_idx[i], v_idx[i]+VLEN))
                self.sched.add("alu", ("+", st, sc_meta["val_p"], ic), [sc_meta["val_p"], ic], [st])
                self.sched.add("load", ("vload", v_val[i], st), [st], range(v_val[i], v_val[i]+VLEN))

            current_depth = round_idx % (forest_height + 1)
            use_mux = current_depth < mux_levels
            needs_wrap = (current_depth == forest_height)

            idx_in = list(range(v_idx[i], v_idx[i]+VLEN))
            t1 = list(range(v_t1[i], v_t1[i]+VLEN))
            t2 = list(range(v_t2[i], v_t2[i]+VLEN))

            if use_mux:
                if current_depth == 0:
                    emit_hash_index_wrap(i, mux_vals[0][0], needs_wrap)
                elif current_depth == 1:
                    self.sched.add("valu", ("&", v_t1[i], v_idx[i], v_one),
                                   idx_in + list(range(v_one, v_one+VLEN)), t1)
                    v_node1 = mux_vals[1][0]; v_node2 = mux_vals[1][1]
                    self.sched.add("flow", ("vselect", v_t1[i], v_t1[i], v_node1, v_node2),
                                   t1 + list(range(v_node1, v_node1+VLEN)) + list(range(v_node2, v_node2+VLEN)), t1)
                    emit_hash_index_wrap(i, v_t1[i], needs_wrap)
                elif current_depth == 2:
                    one_in = list(range(v_one, v_one+VLEN))
                    b3_in = list(range(v_base_3, v_base_3+VLEN))
                    self.sched.add("valu", ("-", v_t2[i], v_idx[i], v_base_3), idx_in + b3_in, t2)
                    self.sched.add("valu", ("&", v_t1[i], v_t2[i], v_one), t2 + one_in, t1)
                    self.sched.add("valu", (">>", v_t2[i], v_t2[i], v_one), t2 + one_in, t2)
                    n0 = mux_vals[2][0]; n1 = mux_vals[2][1]
                    n2 = mux_vals[2][2]; n3 = mux_vals[2][3]
                    v_mt = v_mux_tmps[i % n_mux_tmps]
                    mt_range = list(range(v_mt, v_mt+VLEN))
                    self.sched.add("flow", ("vselect", v_mt, v_t1[i], n1, n0),
                                   t1 + list(range(n1, n1+VLEN)) + list(range(n0, n0+VLEN)), mt_range)
                    self.sched.add("flow", ("vselect", v_t1[i], v_t1[i], n3, n2),
                                   t1 + list(range(n3, n3+VLEN)) + list(range(n2, n2+VLEN)), t1)
                    self.sched.add("flow", ("vselect", v_t1[i], v_t2[i], v_t1[i], v_mt),
                                   t2 + t1 + mt_range, t1)
                    emit_hash_index_wrap(i, v_t1[i], needs_wrap)
                else:
                    curr_nv = v_t1[i]
                    nv_in = list(range(curr_nv, curr_nv+VLEN))
                    c_base = self.get_const((1 << current_depth) - 1)
                    self.sched.add("valu", ("vbroadcast", v_t2[i], c_base), [c_base], t2)
                    self.sched.add("valu", ("-", v_idx[i], v_idx[i], v_t2[i]), idx_in + t2, idx_in)
                    layer_input = mux_vals[current_depth]
                    v_mux_pool = [v_t2[k] for k in range(8)]
                    for bit in range(current_depth):
                        c_bit = self.get_const(1 << bit)
                        self.sched.add("valu", ("vbroadcast", v_t1[i], c_bit), [c_bit], t1)
                        self.sched.add("valu", ("&", v_t1[i], v_idx[i], v_t1[i]), idx_in + t1, t1)
                        next_layer = []
                        for j in range(0, len(layer_input), 2):
                            v_res = v_mux_pool[j // 2]
                            in_regs = t1 + list(range(layer_input[j+1], layer_input[j+1]+VLEN)) + list(range(layer_input[j], layer_input[j]+VLEN))
                            self.sched.add("flow", ("vselect", v_res, v_t1[i], layer_input[j+1], layer_input[j]),
                                           in_regs, range(v_res, v_res+VLEN))
                            next_layer.append(v_res)
                        layer_input = next_layer
                    self.sched.add("valu", ("+", curr_nv, layer_input[0], v_zero),
                                   list(range(layer_input[0], layer_input[0]+VLEN)) + list(range(v_zero, v_zero+VLEN)), nv_in)
                    self.sched.add("valu", ("vbroadcast", v_t2[i], c_base), [c_base], t2)
                    self.sched.add("valu", ("+", v_idx[i], v_idx[i], v_t2[i]), idx_in + t2, idx_in)
                    emit_hash_index_wrap(i, curr_nv, needs_wrap)
            else:
                curr_idx = v_idx[i]; curr_nv = v_t1[i]
                for vi in range(VLEN):
                    sc_t = sc_tmps[i][vi % 2]
                    self.sched.add("alu", ("+", sc_t, sc_meta["tree_p"], curr_idx + vi), [sc_meta["tree_p"], curr_idx + vi], [sc_t])
                    self.sched.add("load", ("load", curr_nv + vi, sc_t), [sc_t], [curr_nv + vi])
                emit_hash_index_wrap(i, v_t1[i], needs_wrap)

        c_0 = self.get_const(0)
        for i in range(1, num_chunks):
            val = i * VLEN
            if val not in self.const_map:
                addr = self.alloc(f"c_{val}")
                self.sched.add("flow", ("add_imm", addr, c_0, val), [c_0], [addr])
                self.const_map[val] = addr

        n_groups = 3
        group_size = num_chunks // n_groups
        remainder = num_chunks % n_groups
        chunk_offset = {}
        pos = 0
        for g in range(n_groups):
            gs = group_size + (1 if g < remainder else 0)
            if g % 2 == 0:
                for k in range(gs):
                    chunk_offset[pos + k] = k
            else:
                for k in range(gs):
                    chunk_offset[pos + k] = gs - 1 - k
            pos += gs
        max_off = max(chunk_offset.values())
        total_waves = max_off + rounds
        fh1 = forest_height + 1
        for wave in range(total_waves):
            for i in range(num_chunks):
                round_idx = wave - chunk_offset[i]
                if 0 <= round_idx < rounds:
                    emit_round(i, round_idx)

        for i in range(num_chunks):
            ic = self.get_const(i * VLEN)
            st = sc_tmps[i][0]
            self.sched.add("alu", ("+", st, sc_meta["idx_p"], ic), [sc_meta["idx_p"], ic], [st])
            self.sched.add("store", ("vstore", st, v_idx[i]), [st] + list(range(v_idx[i], v_idx[i]+VLEN)), [])
            self.sched.add("alu", ("+", st, sc_meta["val_p"], ic), [sc_meta["val_p"], ic], [st])
            self.sched.add("store", ("vstore", st, v_val[i]), [st] + list(range(v_val[i], v_val[i]+VLEN)), [])

        self.sched.add("flow", ("pause",), [], [], min_cycle=self.sched.max_cycle)
        self.instrs = self.sched.get_instrs()

BASELINE = 147734

def do_kernel_test(forest_height: int, rounds: int, batch_size: int, seed: int = 123):
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    
    # Run reference FIRST to populate trace (use COPY to avoid polluting mem)
    value_trace = {}
    for ref_mem in reference_kernel2(list(mem), value_trace): pass

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace)
    machine.enable_pause = False
    machine.run()
    
    inp_values_p = ref_mem[6]
    res_val = machine.mem[inp_values_p : inp_values_p + len(inp.values)]
    ref_val = ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    
    inp_indices_p = ref_mem[5]
    res_idx = machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)]
    ref_idx = ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)]

    if res_val != ref_val:
        for j in range(min(8, len(res_val))):
            if res_val[j] != ref_val[j]:
                print(f"  val[{j}] got={res_val[j]} ref={ref_val[j]}")
    if res_idx != ref_idx:
        for j in range(min(8, len(res_idx))):
            if res_idx[j] != ref_idx[j]:
                print(f"  idx[{j}] got={res_idx[j]} ref={ref_idx[j]}")
    
    assert res_val == ref_val, "Incorrect final values"
    assert res_idx == ref_idx, "Incorrect final indices"

    print(f"CYCLES: {machine.cycle} (Speedup: {BASELINE/machine.cycle:.2f}x)")
    return machine.cycle

class Tests(unittest.TestCase):
    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)

if __name__ == "__main__":
    unittest.main()
