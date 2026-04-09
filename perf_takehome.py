import random
import unittest

from problem import (
    DebugInfo,
    Engine,
    HASH_STAGES,
    Input,
    Machine,
    N_CORES,
    SCRATCH_SIZE,
    SLOT_LIMITS,
    Tree,
    VLEN,
    build_mem_image,
    reference_kernel,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.enable_vdebug = False

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple], vliw: bool = False):
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_packed(self, bundle):
        self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(
                ("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))
            )
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

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

        tmp_addrs = [tmp1, tmp2, tmp1, tmp2, tmp1, tmp2, tmp1]

        for i in range(0, len(init_vars), 2):
            if i + 1 < len(init_vars):
                self.add_packed(
                    {
                        "load": [
                            ("const", tmp_addrs[i], i),
                            ("const", tmp_addrs[i + 1], i + 1),
                        ]
                    }
                )
                self.add_packed(
                    {
                        "load": [
                            ("load", self.scratch[init_vars[i]], tmp_addrs[i]),
                            ("load", self.scratch[init_vars[i + 1]], tmp_addrs[i + 1]),
                        ]
                    }
                )
            else:
                odd_idx = i
                odd_addr = tmp_addrs[i]
                odd_var = init_vars[i]

        zero_const = self.alloc_scratch("zero_const")
        one_const = self.alloc_scratch("one_const")
        two_const = self.alloc_scratch("two_const")
        self.const_map[0] = zero_const
        self.const_map[1] = one_const
        self.const_map[2] = two_const

        tree0_scalar = self.alloc_scratch("tree0_scalar")
        tree0_v = self.alloc_scratch("tree0_v", VLEN)

        self.add_packed(
            {"load": [("const", odd_addr, odd_idx), ("const", zero_const, 0)]}
        )
        self.add_packed(
            {
                "load": [
                    ("load", self.scratch[odd_var], odd_addr),
                    ("const", one_const, 1),
                ]
            }
        )
        self.add_packed(
            {
                "load": [
                    ("const", two_const, 2),
                    ("load", tree0_scalar, self.scratch["forest_values_p"]),
                ]
            }
        )

        zero_v = self.alloc_scratch("zero_v", VLEN)
        one_v = self.alloc_scratch("one_v", VLEN)
        two_v = self.alloc_scratch("two_v", VLEN)
        n_nodes_v = self.alloc_scratch("n_nodes_v", VLEN)
        forest_base_v = self.alloc_scratch("forest_base_v", VLEN)

        vector_batch_early = (batch_size // VLEN) * VLEN
        block_offset_values_early = list(range(0, vector_batch_early, VLEN))

        block_off_addrs = []
        for i in range(len(block_offset_values_early)):
            addr = self.alloc_scratch(f"block_off_{i}")
            block_off_addrs.append(addr)

        base_indices = list(range(0, len(block_offset_values_early), 4))

        eight_const = self.alloc_scratch("eight_const")
        sixteen_const = self.alloc_scratch("sixteen_const")
        twentyfour_const = self.alloc_scratch("twentyfour_const")

        early_block_loads = []
        for i in base_indices:
            early_block_loads.append((block_off_addrs[i], block_offset_values_early[i]))
        early_block_loads.append((eight_const, 8))
        early_block_loads.append((sixteen_const, 16))
        early_block_loads.append((twentyfour_const, 24))

        early_load_idx = 0

        vb_bundle = {
            "valu": [
                ("vbroadcast", zero_v, zero_const),
                ("vbroadcast", one_v, one_const),
                ("vbroadcast", two_v, two_const),
                ("vbroadcast", n_nodes_v, self.scratch["n_nodes"]),
                ("vbroadcast", forest_base_v, self.scratch["forest_values_p"]),
                ("vbroadcast", tree0_v, tree0_scalar),
            ]
        }
        if early_load_idx + 1 < len(early_block_loads):
            vb_bundle["load"] = [
                (
                    "const",
                    early_block_loads[early_load_idx][0],
                    early_block_loads[early_load_idx][1],
                ),
                (
                    "const",
                    early_block_loads[early_load_idx + 1][0],
                    early_block_loads[early_load_idx + 1][1],
                ),
            ]
            early_load_idx += 2
        self.add_packed(vb_bundle)

        tree1_scalar = self.alloc_scratch("tree1_scalar")
        tree2_scalar = self.alloc_scratch("tree2_scalar")
        tree1_v = self.alloc_scratch("tree1_v", VLEN)
        tree2_v = self.alloc_scratch("tree2_v", VLEN)
        diff_1_2_v = self.alloc_scratch("diff_1_2_v", VLEN)

        three_const = self.alloc_scratch("three_const")
        four_const = self.alloc_scratch("four_const")
        five_const = self.alloc_scratch("five_const")
        six_const = self.alloc_scratch("six_const")
        self.const_map[3] = three_const
        self.const_map[4] = four_const
        self.const_map[5] = five_const
        self.const_map[6] = six_const

        self.add_packed(
            {
                "alu": [
                    ("+", tree1_scalar, self.scratch["forest_values_p"], one_const),
                    ("+", tree2_scalar, self.scratch["forest_values_p"], two_const),
                ],
                "load": [("const", three_const, 3), ("const", four_const, 4)],
            }
        )

        self.add_packed(
            {
                "load": [
                    ("load", tree1_scalar, tree1_scalar),
                    ("load", tree2_scalar, tree2_scalar),
                ]
            }
        )

        self.add_packed(
            {
                "load": [("const", five_const, 5), ("const", six_const, 6)],
                "valu": [
                    ("vbroadcast", tree1_v, tree1_scalar),
                    ("vbroadcast", tree2_v, tree2_scalar),
                ],
            }
        )

        three_v = self.alloc_scratch("three_v", VLEN)
        tree3_scalar = self.alloc_scratch("tree3_scalar")
        tree4_scalar = self.alloc_scratch("tree4_scalar")
        tree5_scalar = self.alloc_scratch("tree5_scalar")
        tree6_scalar = self.alloc_scratch("tree6_scalar")
        tree3_v = self.alloc_scratch("tree3_v", VLEN)
        tree4_v = self.alloc_scratch("tree4_v", VLEN)
        tree5_v = self.alloc_scratch("tree5_v", VLEN)
        tree6_v = self.alloc_scratch("tree6_v", VLEN)
        diff_3_4_v = self.alloc_scratch("diff_3_4_v", VLEN)
        diff_5_6_v = self.alloc_scratch("diff_5_6_v", VLEN)

        tree_alu_bundle = {
            "valu": [
                ("-", diff_1_2_v, tree2_v, tree1_v),
                ("vbroadcast", three_v, three_const),
            ],
            "alu": [
                ("+", tree3_scalar, self.scratch["forest_values_p"], three_const),
                ("+", tree4_scalar, self.scratch["forest_values_p"], four_const),
                ("+", tree5_scalar, self.scratch["forest_values_p"], five_const),
                ("+", tree6_scalar, self.scratch["forest_values_p"], six_const),
            ],
        }
        if early_load_idx + 1 < len(early_block_loads):
            tree_alu_bundle["load"] = [
                (
                    "const",
                    early_block_loads[early_load_idx][0],
                    early_block_loads[early_load_idx][1],
                ),
                (
                    "const",
                    early_block_loads[early_load_idx + 1][0],
                    early_block_loads[early_load_idx + 1][1],
                ),
            ]
            early_load_idx += 2
        self.add_packed(tree_alu_bundle)

        self.add_packed(
            {
                "load": [
                    ("load", tree3_scalar, tree3_scalar),
                    ("load", tree4_scalar, tree4_scalar),
                ]
            }
        )

        self.add_packed(
            {
                "load": [
                    ("load", tree5_scalar, tree5_scalar),
                    ("load", tree6_scalar, tree6_scalar),
                ],
                "valu": [
                    ("vbroadcast", tree3_v, tree3_scalar),
                    ("vbroadcast", tree4_v, tree4_scalar),
                ],
            }
        )

        tree56_bundle = {
            "valu": [
                ("vbroadcast", tree5_v, tree5_scalar),
                ("vbroadcast", tree6_v, tree6_scalar),
                ("-", diff_3_4_v, tree4_v, tree3_v),
            ]
        }
        if early_load_idx + 1 < len(early_block_loads):
            tree56_bundle["load"] = [
                (
                    "const",
                    early_block_loads[early_load_idx][0],
                    early_block_loads[early_load_idx][1],
                ),
                (
                    "const",
                    early_block_loads[early_load_idx + 1][0],
                    early_block_loads[early_load_idx + 1][1],
                ),
            ]
            early_load_idx += 2
        self.add_packed(tree56_bundle)

        deferred_diff_5_6 = ("-", diff_5_6_v, tree6_v, tree5_v)

        hash_c1_v = []
        hash_c3_v = []
        hash_c3_s = []
        hash_mul_v = []

        c1_scalars = []
        c3_scalars = []
        mul_scalars = []

        const_loads = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if val1 not in self.const_map:
                addr = self.alloc_scratch(f"hash_c1_s_{hi}")
                self.const_map[val1] = addr
                const_loads.append((addr, val1))
            c1_scalars.append(self.const_map[val1])

            if val3 not in self.const_map:
                addr = self.alloc_scratch(f"hash_c3_s_{hi}")
                self.const_map[val3] = addr
                const_loads.append((addr, val3))
            c3_scalars.append(self.const_map[val3])
            hash_c3_s.append(self.const_map[val3])

            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul = (1 + (1 << val3)) % (2**32)
                if mul not in self.const_map:
                    addr = self.alloc_scratch(f"hash_mul_s_{hi}")
                    self.const_map[mul] = addr
                    const_loads.append((addr, mul))
                mul_scalars.append((hi, self.const_map[mul]))
            else:
                mul_scalars.append((hi, None))

        for i in range(0, len(const_loads), 2):
            if i + 1 < len(const_loads):
                bundle = {
                    "load": [
                        ("const", const_loads[i][0], const_loads[i][1]),
                        ("const", const_loads[i + 1][0], const_loads[i + 1][1]),
                    ]
                }
                if i == 0 and deferred_diff_5_6:
                    bundle["valu"] = [deferred_diff_5_6]
                self.add_packed(bundle)
            else:
                self.add("load", ("const", const_loads[i][0], const_loads[i][1]))

        for hi in range(len(HASH_STAGES)):
            c1_v = self.alloc_scratch(f"hash_c1_v_{hi}", VLEN)
            c3_v = self.alloc_scratch(f"hash_c3_v_{hi}", VLEN)
            hash_c1_v.append(c1_v)
            hash_c3_v.append(c3_v)

        for hi, mul_scalar in mul_scalars:
            if mul_scalar is not None:
                mul_v = self.alloc_scratch(f"hash_mul_v_{hi}", VLEN)
                hash_mul_v.append(mul_v)
            else:
                hash_mul_v.append(None)

        remaining_block_loads = early_block_loads[early_load_idx:]

        all_broadcasts = []
        for i in range(len(HASH_STAGES)):
            all_broadcasts.append(("vbroadcast", hash_c1_v[i], c1_scalars[i]))
        for i in range(len(HASH_STAGES)):
            all_broadcasts.append(("vbroadcast", hash_c3_v[i], c3_scalars[i]))
        for hi in range(len(HASH_STAGES)):
            if hash_mul_v[hi] is not None:
                all_broadcasts.append(
                    ("vbroadcast", hash_mul_v[hi], mul_scalars[hi][1])
                )

        bc_idx = 0
        rem_idx = 0
        while bc_idx < len(all_broadcasts) or rem_idx < len(remaining_block_loads):
            bundle = {}

            valu_ops = []
            while len(valu_ops) < 6 and bc_idx < len(all_broadcasts):
                valu_ops.append(all_broadcasts[bc_idx])
                bc_idx += 1
            if valu_ops:
                bundle["valu"] = valu_ops

            load_ops = []
            while len(load_ops) < 2 and rem_idx < len(remaining_block_loads):
                addr, val = remaining_block_loads[rem_idx]
                load_ops.append(("const", addr, val))
                rem_idx += 1
            if load_ops:
                bundle["load"] = load_ops

            if bundle:
                self.add_packed(bundle)

        offset_alu_ops = []
        last_base_idx = base_indices[-1] if base_indices else None
        for base_idx in base_indices:
            base_addr = block_off_addrs[base_idx]
            if base_idx + 1 < len(block_off_addrs):
                offset_alu_ops.append(
                    ("+", block_off_addrs[base_idx + 1], base_addr, eight_const)
                )
            if base_idx + 2 < len(block_off_addrs):
                offset_alu_ops.append(
                    ("+", block_off_addrs[base_idx + 2], base_addr, sixteen_const)
                )
            if base_idx + 3 < len(block_off_addrs):
                offset_alu_ops.append(
                    ("+", block_off_addrs[base_idx + 3], base_addr, twentyfour_const)
                )
            if len(offset_alu_ops) == SLOT_LIMITS["alu"]:
                if base_idx == last_base_idx:
                    self.add_packed({"alu": offset_alu_ops, "flow": [("pause",)]})
                else:
                    self.add_packed({"alu": offset_alu_ops})
                offset_alu_ops = []
        if offset_alu_ops:
            self.add_packed({"alu": offset_alu_ops, "flow": [("pause",)]})

        body_instrs = []
        buffers = []

        vector_batch = (batch_size // VLEN) * VLEN
        vector_blocks = vector_batch // VLEN

        pipe_buffers = min(13, vector_blocks)

        for bi in range(pipe_buffers):
            buffers.append(
                {
                    "idx": self.alloc_scratch(f"idx_v{bi}", VLEN),
                    "val": self.alloc_scratch(f"val_v{bi}", VLEN),
                    "node": self.alloc_scratch(f"node_val_v{bi}", VLEN),
                    "addr": self.alloc_scratch(f"addr_v{bi}", VLEN),
                    "tmp1": self.alloc_scratch(f"tmp1_v{bi}", VLEN),
                    "tmp2": self.alloc_scratch(f"tmp2_v{bi}", VLEN),
                    "cond": self.alloc_scratch(f"cond_v{bi}", VLEN),
                    "val_addr": self.alloc_scratch(f"val_addr{bi}"),
                }
            )

        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        block_offsets = block_off_addrs

        wrap_threshold = forest_height

        def schedule_all_rounds():
            if vector_blocks == 0:
                return []

            instrs = []
            active = []
            free_bufs = list(range(pipe_buffers))
            next_block = 0

            def start_block():
                nonlocal next_block
                if next_block >= vector_blocks or not free_bufs:
                    return False
                buf_idx = free_bufs.pop(0)
                active.append(
                    {
                        "block": next_block,
                        "buf_idx": buf_idx,
                        "buf": buffers[buf_idx],
                        "offset": block_offsets[next_block],
                        "phase": "init_addr",
                        "round": 0,
                        "stage": 0,
                        "gather": 0,
                    }
                )
                next_block += 1
                return True

            while free_bufs and next_block < vector_blocks:
                start_block()

            while active or next_block < vector_blocks:
                while free_bufs and next_block < vector_blocks:
                    start_block()

                alu_ops = []
                load_ops = []
                valu_ops = []
                store_ops = []
                flow_ops = []

                alu_slots = SLOT_LIMITS["alu"]
                load_slots = SLOT_LIMITS["load"]
                valu_slots = SLOT_LIMITS["valu"]
                store_slots = SLOT_LIMITS["store"]

                scheduled_this_cycle = set()

                def next_round_phase(current_round):
                    next_r = current_round + 1
                    if next_r >= rounds:
                        return "store_val"

                    if next_r <= wrap_threshold:
                        depth = next_r
                    else:
                        depth = next_r - wrap_threshold - 1

                    if depth == 0:
                        return "round0_xor"
                    elif depth == 1:
                        return "round1_select"
                    elif depth == 2:
                        return "round2_select1"
                    else:
                        return "addr"

                for block in active:
                    if store_slots == 0:
                        break
                    if (
                        block["phase"] == "store_val"
                        and block["block"] not in scheduled_this_cycle
                    ):
                        buf = block["buf"]
                        store_ops.append(("vstore", buf["val_addr"], buf["val"]))
                        block["next_phase"] = "done"
                        scheduled_this_cycle.add(block["block"])
                        store_slots -= 1

                for block in active:
                    if load_slots < 1:
                        break
                    if (
                        block["phase"] == "vload"
                        and block["block"] not in scheduled_this_cycle
                    ):
                        buf = block["buf"]
                        load_ops.append(("vload", buf["val"], buf["val_addr"]))

                        if valu_slots >= 1:
                            valu_ops.append(("+", buf["idx"], zero_v, zero_v))
                            valu_slots -= 1

                        if block["round"] == 0:
                            block["next_phase"] = "round0_xor"
                        else:
                            block["next_phase"] = "addr"
                        scheduled_this_cycle.add(block["block"])
                        load_slots -= 1

                for block in active:
                    if load_slots == 0:
                        break
                    if block["phase"] == "gather":
                        if block["round"] == 0:
                            block["next_phase"] = "round0_xor"
                            scheduled_this_cycle.add(block["block"])
                        else:
                            buf = block["buf"]
                            while load_slots > 0 and block["gather"] < VLEN:
                                lane = block["gather"]
                                load_ops.append(
                                    ("load_offset", buf["node"], buf["addr"], lane)
                                )
                                block["gather"] += 1
                                load_slots -= 1

                            if block["gather"] >= VLEN:
                                block["next_phase"] = "xor"
                                scheduled_this_cycle.add(block["block"])

                valu_tasks = []
                for block in active:
                    if block["block"] in scheduled_this_cycle:
                        continue
                    phase = block["phase"]

                    if phase == "wrap_reset":
                        valu_tasks.append((0, 1, block, "wrap_reset"))
                    elif phase == "update2":
                        valu_tasks.append((1, 1, block, "update2"))
                    elif phase == "update1":
                        valu_tasks.append((2, 2, block, "update1"))
                    elif phase == "hash_op2":
                        valu_tasks.append((6, 1, block, "hash_op2"))
                    elif phase == "hash_mul":
                        valu_tasks.append((5, 1, block, "hash_mul"))
                    elif phase == "hash_op1":
                        valu_tasks.append((4, 1, block, "hash_op1"))
                    elif phase == "xor":
                        valu_tasks.append((7, 1, block, "xor"))
                    elif phase == "round0_xor":
                        valu_tasks.append((7, 1, block, "round0_xor"))
                    elif phase == "round1_select":
                        valu_tasks.append((7, 1, block, "round1_select"))
                    elif phase == "round2_select1":
                        valu_tasks.append((7, 1, block, "round2_select1"))
                    elif phase == "round2_select2":
                        valu_tasks.append((4, 1, block, "round2_select2"))
                    elif phase == "round2_select3":
                        valu_tasks.append((6, 2, block, "round2_select3"))
                    elif phase == "round2_select4":
                        valu_tasks.append((7, 1, block, "round2_select4"))
                    elif phase == "round2_select5":
                        valu_tasks.append((6, 1, block, "round2_select5"))
                    elif phase == "addr":
                        valu_tasks.append((4, 1, block, "addr"))

                valu_tasks.sort(key=lambda x: x[0])

                for _, cost, block, phase in valu_tasks:
                    if block["block"] in scheduled_this_cycle:
                        continue
                    buf = block["buf"]

                    if phase == "hash_op1":
                        hi = block["stage"]
                        op1 = HASH_STAGES[hi][0]
                        op3 = HASH_STAGES[hi][3]
                        if alu_slots >= VLEN and valu_slots >= 1:
                            valu_ops.append(
                                (op1, buf["tmp1"], buf["val"], hash_c1_v[hi])
                            )
                            for lane in range(VLEN):
                                alu_ops.append(
                                    (
                                        op3,
                                        buf["tmp2"] + lane,
                                        buf["val"] + lane,
                                        hash_c3_s[hi],
                                    )
                                )
                            alu_slots -= VLEN
                            valu_slots -= 1
                            block["next_phase"] = "hash_op2"
                            scheduled_this_cycle.add(block["block"])
                            continue

                        if valu_slots >= 2:
                            valu_ops.append(
                                (op1, buf["tmp1"], buf["val"], hash_c1_v[hi])
                            )
                            valu_ops.append(
                                (op3, buf["tmp2"], buf["val"], hash_c3_v[hi])
                            )
                            valu_slots -= 2
                            block["next_phase"] = "hash_op2"
                            scheduled_this_cycle.add(block["block"])
                            continue
                        continue

                    if valu_slots < cost:
                        continue

                    if phase == "wrap_reset":
                        valu_ops.append(("+", buf["idx"], zero_v, zero_v))
                        block["round"] += 1
                        block["stage"] = 0
                        block["gather"] = 0
                        block["next_phase"] = next_round_phase(block["round"] - 1)
                    elif phase == "update2":
                        valu_ops.append(("+", buf["idx"], buf["idx"], buf["tmp1"]))
                        block["round"] += 1
                        block["stage"] = 0
                        block["gather"] = 0
                        block["next_phase"] = next_round_phase(block["round"] - 1)
                    elif phase == "update1":
                        if alu_slots >= VLEN and valu_slots >= 1:
                            for lane in range(VLEN):
                                alu_ops.append(
                                    (
                                        "&",
                                        buf["tmp1"] + lane,
                                        buf["val"] + lane,
                                        one_const,
                                    )
                                )
                            valu_ops.append(
                                ("multiply_add", buf["idx"], buf["idx"], two_v, one_v)
                            )
                            alu_slots -= VLEN
                            valu_slots -= 1
                            block["next_phase"] = "update2"
                            scheduled_this_cycle.add(block["block"])
                            continue
                        valu_ops.append(("&", buf["tmp1"], buf["val"], one_v))
                        valu_ops.append(
                            ("multiply_add", buf["idx"], buf["idx"], two_v, one_v)
                        )
                        block["next_phase"] = "update2"
                    elif phase == "hash_op2":
                        hi = block["stage"]
                        op2 = HASH_STAGES[hi][2]
                        valu_ops.append((op2, buf["val"], buf["tmp1"], buf["tmp2"]))
                        if hi + 1 == len(HASH_STAGES):
                            if block["round"] == rounds - 1:
                                block["next_phase"] = "store_val"
                            elif block["round"] == wrap_threshold:
                                block["next_phase"] = "wrap_reset"
                            else:
                                block["next_phase"] = "update1"
                        else:
                            block["stage"] = hi + 1
                            block["next_phase"] = (
                                "hash_mul"
                                if hash_mul_v[hi + 1] is not None
                                else "hash_op1"
                            )
                    elif phase == "hash_mul":
                        hi = block["stage"]
                        mul_v = hash_mul_v[hi]
                        valu_ops.append(
                            (
                                "multiply_add",
                                buf["val"],
                                buf["val"],
                                mul_v,
                                hash_c1_v[hi],
                            )
                        )
                        if hi + 1 == len(HASH_STAGES):
                            if block["round"] == rounds - 1:
                                block["next_phase"] = "store_val"
                            elif block["round"] == wrap_threshold:
                                block["next_phase"] = "wrap_reset"
                            else:
                                block["next_phase"] = "update1"
                        else:
                            block["stage"] = hi + 1
                            block["next_phase"] = (
                                "hash_mul"
                                if hash_mul_v[hi + 1] is not None
                                else "hash_op1"
                            )
                    elif phase == "xor":
                        valu_ops.append(("^", buf["val"], buf["val"], buf["node"]))
                        block["next_phase"] = (
                            "hash_mul" if hash_mul_v[0] is not None else "hash_op1"
                        )
                    elif phase == "round0_xor":
                        valu_ops.append(("^", buf["val"], buf["val"], tree0_v))
                        block["next_phase"] = (
                            "hash_mul" if hash_mul_v[0] is not None else "hash_op1"
                        )
                    elif phase == "round1_select":
                        valu_ops.append(
                            (
                                "multiply_add",
                                buf["node"],
                                diff_1_2_v,
                                buf["tmp1"],
                                tree1_v,
                            )
                        )
                        block["next_phase"] = "xor"
                    elif phase == "round2_select1":
                        valu_ops.append(("-", buf["tmp2"], buf["idx"], three_v))
                        block["next_phase"] = "round2_select2"
                    elif phase == "round2_select2":
                        valu_ops.append((">>", buf["cond"], buf["tmp2"], one_v))
                        block["next_phase"] = "round2_select3"
                    elif phase == "round2_select3":
                        valu_ops.append(
                            (
                                "multiply_add",
                                buf["tmp2"],
                                diff_3_4_v,
                                buf["tmp1"],
                                tree3_v,
                            )
                        )
                        valu_ops.append(
                            (
                                "multiply_add",
                                buf["node"],
                                diff_5_6_v,
                                buf["tmp1"],
                                tree5_v,
                            )
                        )
                        block["next_phase"] = "round2_select4"
                    elif phase == "round2_select4":
                        valu_ops.append(("-", buf["node"], buf["node"], buf["tmp2"]))
                        block["next_phase"] = "round2_select5"
                    elif phase == "round2_select5":
                        valu_ops.append(
                            (
                                "multiply_add",
                                buf["node"],
                                buf["node"],
                                buf["cond"],
                                buf["tmp2"],
                            )
                        )
                        block["next_phase"] = "xor"
                    elif phase == "addr":
                        valu_ops.append(("+", buf["addr"], buf["idx"], forest_base_v))
                        block["next_phase"] = "gather"

                    scheduled_this_cycle.add(block["block"])
                    valu_slots -= cost

                for block in active:
                    if alu_slots < 1:
                        break
                    if (
                        block["phase"] == "init_addr"
                        and block["block"] not in scheduled_this_cycle
                    ):
                        buf = block["buf"]
                        alu_ops.append(
                            (
                                "+",
                                buf["val_addr"],
                                self.scratch["inp_values_p"],
                                block["offset"],
                            )
                        )
                        block["next_phase"] = "vload"
                        scheduled_this_cycle.add(block["block"])
                        alu_slots -= 1

                if not (alu_ops or load_ops or valu_ops or store_ops or flow_ops):
                    stuck = False
                    for block in active:
                        if block["phase"] == "gather" and block["gather"] < VLEN:
                            stuck = True
                            break
                    if not stuck:
                        raise RuntimeError("scheduler made no progress")
                    continue

                instr = {}
                if alu_ops:
                    instr["alu"] = alu_ops
                if load_ops:
                    instr["load"] = load_ops
                if valu_ops:
                    instr["valu"] = valu_ops
                if store_ops:
                    instr["store"] = store_ops
                if flow_ops:
                    instr["flow"] = flow_ops
                instrs.append(instr)

                new_active = []
                for block in active:
                    next_phase = block.pop("next_phase", None)
                    if next_phase:
                        block["phase"] = next_phase
                    if block["phase"] == "done":
                        free_bufs.append(block["buf_idx"])
                    else:
                        new_active.append(block)
                active = new_active

            return instrs

        body_instrs.extend(schedule_all_rounds())

        tail_idx_addrs = []
        for i in range(vector_batch, batch_size):
            tail_idx_addrs.append(self.alloc_scratch(f"tail_idx_{i}"))

        if tail_idx_addrs:
            init_ops = []
            for addr in tail_idx_addrs:
                init_ops.append(("+", addr, zero_const, zero_const))
                if len(init_ops) == SLOT_LIMITS["alu"]:
                    body_instrs.append({"alu": init_ops})
                    init_ops = []
            if init_ops:
                body_instrs.append({"alu": init_ops})

        for round_i in range(rounds):
            for i in range(vector_batch, batch_size):
                tail_slots = []
                i_const = self.scratch_const(i)

                tail_slots.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                )
                tail_slots.append(("load", ("load", tmp_val, tmp_addr)))
                idx_addr = tail_idx_addrs[i - vector_batch]
                tail_slots.append(
                    ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], idx_addr))
                )
                tail_slots.append(("load", ("load", tmp_node_val, tmp_addr)))
                tail_slots.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                tail_slots.extend(self.build_hash(tmp_val, tmp1, tmp2, round_i, i))

                if round_i != rounds - 1:
                    if round_i == wrap_threshold:
                        tail_slots.append(
                            ("alu", ("+", idx_addr, zero_const, zero_const))
                        )
                    else:
                        tail_slots.append(("alu", ("%", tmp1, tmp_val, two_const)))
                        tail_slots.append(("alu", ("==", tmp1, tmp1, zero_const)))
                        tail_slots.append(
                            ("flow", ("select", tmp3, tmp1, one_const, two_const))
                        )
                        tail_slots.append(("alu", ("*", idx_addr, idx_addr, two_const)))
                        tail_slots.append(("alu", ("+", idx_addr, idx_addr, tmp3)))
                        tail_slots.append(
                            ("alu", ("<", tmp1, idx_addr, self.scratch["n_nodes"]))
                        )
                        tail_slots.append(
                            ("flow", ("select", idx_addr, tmp1, idx_addr, zero_const))
                        )

                tail_slots.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                )
                tail_slots.append(("store", ("store", tmp_addr, tmp_val)))

                body_instrs.extend(self.build(tail_slots))

        self.instrs.extend(body_instrs)


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
    # print(kb.instrs)

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
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

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

    def test_kernel_correctness(self):
        for batch in range(1, 3):
            for forest_height in range(3):
                do_kernel_test(
                    forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
                )

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