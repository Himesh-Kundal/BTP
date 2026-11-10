"""Compiled-Artifact Software Interpreter (Task UB.7).

Executes the generated sequencer microprogram directly on packed ROM images
(`weights.hex`, `requant_params.hex`, `tables.hex`) and address maps, simulating
the exact hardware engine and non-linear unit operations.

Strict Layering (Golden Rule 6):
Only pure Python and NumPy are used. Never import torch or transformers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from compiler.schedule import Opcode
from numerics.ops import requant, rshift_rnd, sat


class CompiledArtifactInterpreter:
    """Interprets sequencer microprogram on packed ROM arrays and dynamic RAM buffers."""

    def __init__(self, rom_dir: str | Path):
        self.rom_dir = Path(rom_dir)

        # 1. Load Address Map
        with open(self.rom_dir / "address_map.json", "r", encoding="utf-8") as f:
            self.addr_map = json.load(f)

        # 2. Load Microprogram Schedule
        with open(self.rom_dir / "schedule.json", "r", encoding="utf-8") as f:
            self.schedule = json.load(f)

        # 3. Load Format Table if available
        fmt_path = self.rom_dir / "format_table.json"
        if fmt_path.exists():
            with open(fmt_path, "r", encoding="utf-8") as f:
                self.fmt_table = json.load(f)
        else:
            self.fmt_table = {}

        # 4. Load ROM Hex Files into Memory Arrays
        self.weight_rom = self._load_hex_file(self.rom_dir / "weights.hex", 8, signed=True)
        self.requant_rom = self._load_param_hex_file(self.rom_dir / "requant_params.hex")
        self.table_rom = self._load_hex_file(self.rom_dir / "tables.hex", 16, signed=True)

        # Engine config
        self.tile_r = self.addr_map["engine_config"]["tile_r"]
        self.tile_c = self.addr_map["engine_config"]["tile_c"]

        # 5. Extract Lookup Tables from Table ROM
        self._setup_tables()

        # 6. Initialize State: Buffers & KV Cache
        self.buffers: dict[int, np.ndarray] = {}
        self.kv_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # layer -> (K, V)
        self.traces: dict[str, np.ndarray] = {}

    def _load_hex_file(self, path: Path, bits: int, signed: bool = True) -> np.ndarray:
        """Load hex lines into a NumPy integer array."""
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        values: list[int] = []
        mask = (1 << bits) - 1
        half = 1 << (bits - 1)
        for line in lines:
            if not line.strip():
                continue
            val = int(line.strip(), 16) & mask
            if signed and val >= half:
                val -= (1 << bits)
            values.append(val)
        dtype = np.int8 if bits <= 8 else (np.int16 if bits <= 16 else np.int32)
        return np.array(values, dtype=dtype)

    def _load_param_hex_file(self, path: Path) -> list[tuple[int, int]]:
        """Load 64-bit {M[31:0], shift[31:0]} pairs."""
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        params: list[tuple[int, int]] = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            m_hex = line_str[:8]
            s_hex = line_str[8:16]
            M_val = int(m_hex, 16)
            s_val = int(s_hex, 16)
            params.append((M_val, s_val))
        return params

    def _setup_tables(self) -> None:
        """Slice lookup tables from Table ROM according to address_map."""
        tbl_info = self.addr_map["tables"]

        # RoPE tables
        cos_info = tbl_info["rope_cos"]
        sin_info = tbl_info["rope_sin"]
        self.rope_cos = self.table_rom[
            cos_info["base_addr"] : cos_info["base_addr"] + cos_info["length"]
        ].reshape(cos_info["shape"])
        self.rope_sin = self.table_rom[
            sin_info["base_addr"] : sin_info["base_addr"] + sin_info["length"]
        ].reshape(sin_info["shape"])

        # rsqrt mantissa LUT
        r_info = tbl_info["rsqrt_mantissa_lut"]
        self.rsqrt_lut = self.table_rom[r_info["base_addr"] : r_info["base_addr"] + r_info["length"]]

        # exp2 LUT
        e_info = tbl_info["exp2_lut"]
        self.exp2_lut = self.table_rom[e_info["base_addr"] : e_info["base_addr"] + e_info["length"]]

        # reciprocal LUT
        rec_info = tbl_info["reciprocal_lut"]
        self.recip_lut = self.table_rom[
            rec_info["base_addr"] : rec_info["base_addr"] + rec_info["length"]
        ]

        # SiLU LUT
        silu_info = tbl_info["silu_lut"]
        self.silu_lut = self.table_rom[
            silu_info["base_addr"] : silu_info["base_addr"] + silu_info["length"]
        ]

    def reset(self) -> None:
        """Reset internal buffers, traces, and KV cache."""
        self.buffers.clear()
        self.kv_cache.clear()
        self.traces.clear()

    def _get_linear_weight_matrix(self, node_id: str) -> np.ndarray:
        """Reconstruct 2D weight matrix from tiled Weight ROM layout."""
        info = self.addr_map["weights"][node_id]
        base = info["base_addr"]
        n_pad, k_pad = info["padded_shape"]
        num_tr, num_tc = info["num_tiles"]

        raw_tile_words = self.weight_rom[base : base + info["length"]]

        # Reconstruct padded matrix from tile-ordered elements
        w_reconstructed = np.zeros((n_pad, k_pad), dtype=np.int8)
        idx = 0
        for tr in range(num_tr):
            for tc in range(num_tc):
                for r in range(self.tile_r):
                    for c in range(self.tile_c):
                        w_reconstructed[tr * self.tile_r + r, tc * self.tile_c + c] = raw_tile_words[idx]
                        idx += 1

        # Unpad to original shape
        orig_n, orig_k = info["orig_shape"]
        return w_reconstructed[:orig_n, :orig_k]

    def decode_step(self, token_id: int, pos: int) -> tuple[np.ndarray, int]:
        """Execute one complete decode step using the microprogram."""
        num_layers = self.schedule["num_layers"]
        instructions = self.schedule["instructions"]
        loop_start = self.schedule["layer_loop_start"]
        loop_end = self.schedule["layer_loop_end"]
        layer_bases = self.schedule["layer_rom_bases"]

        pc = 0

        # 1. Execute Prologue (e.g. Embed)
        while pc < loop_start:
            ins = instructions[pc]
            self._execute_instruction(ins, layer_idx=0, pos=pos, token_id=token_id)
            pc += 1

        # 2. Execute Layer Loop
        for l_idx in range(num_layers):
            loop_pc = loop_start
            while loop_pc <= loop_end:
                ins = instructions[loop_pc]
                self._execute_instruction(
                    ins, layer_idx=l_idx, pos=pos, layer_bases=layer_bases[l_idx]
                )
                loop_pc += 1

        # 3. Execute Epilogue (e.g. Final Norm, LM Head, ArgMax)
        pc = loop_end + 1
        while pc < len(instructions):
            ins = instructions[pc]
            if ins["opcode_val"] == Opcode.HALT:
                break
            self._execute_instruction(ins, layer_idx=0, pos=pos)
            pc += 1

        logits = self.buffers.get(9, np.zeros(1, dtype=np.int32))
        next_tok = int(self.buffers.get(0, np.array([0]))[0])
        return logits, next_tok

    def _execute_instruction(
        self,
        ins: dict[str, Any],
        layer_idx: int,
        pos: int,
        token_id: int = 0,
        layer_bases: dict[str, int] | None = None,
    ) -> None:
        """Execute a single microprogram instruction."""
        opcode = ins["opcode_val"]
        name = ins["name"]
        src = ins["src_buf"]
        dst = ins["dst_buf"]
        dims = ins["dims"]

        if opcode == Opcode.EMBED:
            # Gather row from embed_tokens weight matrix
            w = self._get_linear_weight_matrix("embed_tokens")
            row = w[token_id].astype(np.int64)
            p_info = self.addr_map["requant_params"]["embed_tokens"]
            p_base = p_info["base_addr"] + token_id
            M, s = self.requant_rom[p_base]
            res = rshift_rnd(row * M, s)
            self.buffers[dst] = sat(res, -8388608, 8388607)
            self.traces[f"embed_tokens_tok{token_id}"] = self.buffers[dst]

        elif opcode == Opcode.RMSNORM:
            x = self.buffers[src].astype(np.int64)
            d = dims[0]
            ss = int(np.sum(x * x))
            mean = ss // d
            if mean <= 0:
                self.buffers[dst] = np.zeros(d, dtype=np.int8)
                return

            lzc = 64 - mean.bit_length()
            shift = (64 - lzc) & ~1
            mantissa = (mean >> max(0, shift - 8)) & 0xFF
            rsqrt_val = int(self.rsqrt_lut[mantissa])
            scaled_x = rshift_rnd(x * rsqrt_val, 16)

            # Get gamma from requant ROM
            node_key = f"attn_norm_{layer_idx}" if "attn" in name else (
                f"mlp_norm_{layer_idx}" if "mlp" in name else "final_norm"
            )
            p_info = self.addr_map["requant_params"].get(node_key)
            if p_info:
                p_base = p_info["base_addr"]
                params = self.requant_rom[p_base : p_base + d]
                m_arr = np.array([p[0] for p in params], dtype=np.int64)
                s_val = params[0][1] if params else 16
                out = requant(scaled_x * m_arr, 1, s_val, -127, 127)
            else:
                out = sat(scaled_x, -127, 127)

            self.buffers[dst] = out.astype(np.int8)
            self.traces[f"{name}_L{layer_idx}"] = self.buffers[dst]

        elif opcode == Opcode.LINEAR:
            # Linear engine: x @ W.T with requant
            x = self.buffers[src].astype(np.int64)
            node_key = (
                f"{name[6:]}_{layer_idx}" if name.startswith("layer_") else name
            )
            w = self._get_linear_weight_matrix(node_key)
            acc = x @ w.T  # INT32 or INT48

            p_info = self.addr_map["requant_params"].get(node_key)
            if p_info:
                p_base = p_info["base_addr"]
                n_ch = w.shape[0]
                params = self.requant_rom[p_base : p_base + n_ch]
                m_arr = np.array([p[0] for p in params], dtype=np.int64)
                s_val = params[0][1] if params else 16
                # Requantize
                out_min = -8388608 if dst in (0, 9) else -127
                out_max = 8388607 if dst in (0, 9) else 127
                res = requant(acc, m_arr, s_val, out_min, out_max)
            else:
                res = acc

            self.buffers[dst] = res
            self.traces[f"{name}_L{layer_idx}"] = self.buffers[dst]

        elif opcode == Opcode.ROPE:
            x = self.buffers[src]
            num_heads = dims[0]
            hd = dims[1] if len(dims) > 1 else (len(x) // num_heads)
            half = hd // 2
            x_reshaped = x.reshape(num_heads, hd)
            x1 = x_reshaped[:, :half].astype(np.int64)
            x2 = x_reshaped[:, half:].astype(np.int64)

            cos_m = self.rope_cos[pos, :half].astype(np.int64)
            sin_m = self.rope_sin[pos, :half].astype(np.int64)

            out1 = rshift_rnd(x1 * cos_m - x2 * sin_m, 14)
            out2 = rshift_rnd(x2 * cos_m + x1 * sin_m, 14)

            out = np.concatenate([out1, out2], axis=-1)
            self.buffers[dst] = sat(out.reshape(num_heads * hd), -127, 127).astype(np.int8)

        elif opcode == Opcode.ATTENTION:
            num_q_heads, num_kv_heads, hd = dims[0], dims[1], dims[2]
            q = self.buffers[2].reshape(num_q_heads, hd)
            k = self.buffers[3].reshape(num_kv_heads, hd)
            v = self.buffers[4].reshape(num_kv_heads, hd)

            # Store into KV cache
            if layer_idx not in self.kv_cache:
                self.kv_cache[layer_idx] = (
                    np.zeros((num_kv_heads, 256, hd), dtype=np.int8),
                    np.zeros((num_kv_heads, 256, hd), dtype=np.int8),
                )
            k_cache, v_cache = self.kv_cache[layer_idx]
            k_cache[:, pos, :] = k
            v_cache[:, pos, :] = v

            gqa_group = num_q_heads // num_kv_heads
            attn_out = np.zeros((num_q_heads, hd), dtype=np.int64)

            for q_idx in range(num_q_heads):
                kv_idx = q_idx // gqa_group
                k_head = k_cache[kv_idx, : pos + 1, :].astype(np.int64)
                v_head = v_cache[kv_idx, : pos + 1, :].astype(np.int64)
                q_head = q[q_idx].astype(np.int64)

                scores = (q_head @ k_head.T) >> 2  # INT32
                max_score = int(np.max(scores))
                delta = scores - max_score  # <= 0

                int_part = -delta >> 6
                frac_part = ((-delta) & 0x3F) << 2
                exp_vals = np.array(
                    [
                        (int(self.exp2_lut[f]) >> min(16, int(i)))
                        for i, f in zip(int_part, frac_part, strict=False)
                    ],
                    dtype=np.int64,
                )

                sum_exp = max(1, int(np.sum(exp_vals)))
                lzc = 64 - sum_exp.bit_length()
                shift = max(0, 64 - lzc - 8)
                mantissa = (sum_exp >> shift) & 0xFF
                recip = int(self.recip_lut[mantissa])
                probs = rshift_rnd(exp_vals * recip, 16)
                ctx = rshift_rnd(probs @ v_head, 15)
                attn_out[q_idx] = ctx

            self.buffers[dst] = sat(attn_out.reshape(num_q_heads * hd), -127, 127).astype(np.int8)
            self.traces[f"{name}_L{layer_idx}"] = self.buffers[dst]

        elif opcode == Opcode.ADD:
            x_src = self.buffers[src].astype(np.int64)
            x_dst = self.buffers[dst].astype(np.int64)
            self.buffers[dst] = sat(x_src + x_dst, -8388608, 8388607)
            self.traces[f"{name}_L{layer_idx}"] = self.buffers[dst]

        elif opcode == Opcode.SWIGLU:
            gate = self.buffers[src]
            up = self.buffers[7].astype(np.int64)
            silu_out = np.array([self.silu_lut[int(g) + 256] for g in gate], dtype=np.int64)
            prod = silu_out * up
            self.buffers[dst] = sat(rshift_rnd(prod, 7), -127, 127).astype(np.int8)
            self.traces[f"{name}_L{layer_idx}"] = self.buffers[dst]

        elif opcode == Opcode.ARGMAX:
            logits = self.buffers[src]
            best_tok = int(np.argmax(logits))
            self.buffers[dst] = np.array([best_tok], dtype=np.int64)
            self.traces[f"{name}_tok"] = self.buffers[dst]
