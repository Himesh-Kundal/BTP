"""Integer Autoregressive Decoder v0 and Scorer (Task UA.4).

The single mathematical source of truth (Oracle) for all integer operations in hardware.
Implements the exact integer decode step from Numeric Contract v2 (N2-1 to N2-13):
- Wide residual stream (INT24/INT32 signed container)
- Dyadic requantization: y = sat(rshift_rnd(acc * M, s), -127, 127)
- Integer RMSNorm with rsqrt mantissa LUT
- Integer rotate-half RoPE with cos/sin fixed-point tables
- Integer GQA attention with causal masking and base-2 softmax LUT
- Integer SwiGLU with SiLU LUT
- Scale-aligned integer ArgMax with deterministic lowest-index tie-breaking
- Step-by-step decode interface and vectorized teacher-forcing scorer
"""

from dataclasses import dataclass

import numpy as np

from numerics.ops import (
    build_exp2_fraction_lut,
    build_reciprocal_lut,
    build_rope_cos_sin_tables,
    build_rsqrt_mantissa_lut,
    build_silu_lut,
    requant,
    rshift_rnd,
    sat,
)


@dataclass
class IntegerModelConfig:
    """Integer model configuration and bitwidths."""
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    vocab_size: int = 256
    max_ctx: int = 32
    rope_theta: float = 10000.0
    residual_bits: int = 24
    res_min: int = -(1 << 23)
    res_max: int = (1 << 23) - 1


@dataclass
class IntegerLayerWeights:
    """Integer weights and dyadic scales for one transformer layer."""
    q_proj: np.ndarray  # [N_q, K] INT8
    k_proj: np.ndarray  # [N_kv, K] INT8
    v_proj: np.ndarray  # [N_kv, K] INT8
    o_proj: np.ndarray  # [K, N_q] INT8
    gate_proj: np.ndarray  # [FFN, K] INT8
    up_proj: np.ndarray  # [FFN, K] INT8
    down_proj: np.ndarray  # [K, FFN] INT8

    # Dyadic requantization parameters (M, s)
    q_M: np.ndarray
    q_s: int
    k_M: np.ndarray
    k_s: int
    v_M: np.ndarray
    v_s: int
    o_M: np.ndarray
    o_s: int
    gate_M: np.ndarray
    gate_s: int
    up_M: np.ndarray
    up_s: int
    down_M: np.ndarray
    down_s: int

    # RMSNorm gamma scales (INT16) and dyadic scales
    in_norm_gamma: np.ndarray
    in_norm_M: int
    in_norm_s: int
    post_norm_gamma: np.ndarray
    post_norm_M: int
    post_norm_s: int


@dataclass
class IntegerModelWeights:
    """All quantized weights and tables for full integer decode."""
    embed_table: np.ndarray  # [vocab_size, hidden_size] INT8
    embed_M: int
    embed_s: int
    layers: list[IntegerLayerWeights]
    final_norm_gamma: np.ndarray
    final_norm_M: int
    final_norm_s: int
    lm_head: np.ndarray  # [vocab_size, hidden_size] INT8
    lm_head_M: np.ndarray
    lm_head_s: int


class IntegerKVCache:
    """Integer KV cache storing quantized INT8 key and value vectors."""

    def __init__(self, num_layers: int, num_kv_heads: int, max_ctx: int, head_dim: int):
        self.k = np.zeros((num_layers, num_kv_heads, max_ctx, head_dim), dtype=np.int8)
        self.v = np.zeros((num_layers, num_kv_heads, max_ctx, head_dim), dtype=np.int8)

    def write(self, layer_idx: int, pos: int, k_vec: np.ndarray, v_vec: np.ndarray) -> None:
        self.k[layer_idx, :, pos, :] = sat(k_vec, -127, 127).astype(np.int8)
        self.v[layer_idx, :, pos, :] = sat(v_vec, -127, 127).astype(np.int8)

    def get(self, layer_idx: int, pos: int) -> tuple[np.ndarray, np.ndarray]:
        return self.k[layer_idx, :, : pos + 1, :], self.v[layer_idx, :, : pos + 1, :]


class IntegerDecoder:
    """Integer-only autoregressive decoder oracle (Rule N2-1 to N2-13)."""

    def __init__(self, config: IntegerModelConfig, weights: IntegerModelWeights):
        self.cfg = config
        self.weights = weights
        self.kv_cache = IntegerKVCache(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            max_ctx=config.max_ctx,
            head_dim=config.head_dim,
        )

        # Precompute offline lookup tables (N2-14)
        self.exp2_lut = build_exp2_fraction_lut(table_size=256, out_bits=16)
        self.rsqrt_lut = build_rsqrt_mantissa_lut(table_size=256, out_bits=16)
        self.recip_lut = build_reciprocal_lut(table_size=256, out_bits=16)
        self.silu_lut = build_silu_lut(in_min=-128, in_max=127, in_scale=0.05, out_scale=0.05)
        self.cos_table, self.sin_table = build_rope_cos_sin_tables(
            max_ctx=config.max_ctx,
            head_dim=config.head_dim,
            theta=config.rope_theta,
            out_bits=14,
        )

    def reset(self) -> None:
        """Reset internal state and KV cache."""
        self.kv_cache = IntegerKVCache(
            num_layers=self.cfg.num_hidden_layers,
            num_kv_heads=self.cfg.num_key_value_heads,
            max_ctx=self.cfg.max_ctx,
            head_dim=self.cfg.head_dim,
        )

    def integer_rmsnorm(
        self, x: np.ndarray, gamma: np.ndarray, M: int, s: int
    ) -> np.ndarray:
        """Integer RMSNorm (N2-6): sum of squares -> rsqrt LUT -> gamma -> requant."""
        # x is in residual format (INT24), gamma in INT16
        x_64 = x.astype(np.int64)
        ss = np.sum(x_64 * x_64)  # 48-bit sum of squares
        mean = ss // self.cfg.hidden_size  # integer mean

        if mean <= 0:
            return np.zeros(self.cfg.hidden_size, dtype=np.int8)

        # Base-2 normalization for mantissa lookup
        lzc = 64 - int(mean).bit_length()
        # Even shift so sqrt works cleanly
        shift = (64 - lzc) & ~1
        mantissa = (mean >> max(0, shift - 8)) & 0xFF
        rsqrt_val = self.rsqrt_lut[mantissa]  # Q0.16

        # Normalization and channel scaling
        scaled_x = rshift_rnd(x_64 * rsqrt_val, 16)
        out = requant(scaled_x * gamma, M, s)
        return out.astype(np.int8)

    def integer_rope(self, x: np.ndarray, pos: int) -> np.ndarray:
        """Rotate-half integer RoPE (N2-8): Q1.14 fixed-point coordinate transform."""
        # x: [num_heads, head_dim] INT8
        half = self.cfg.head_dim // 2
        x1 = x[:, :half].astype(np.int64)
        x2 = x[:, half:].astype(np.int64)

        cos_m = self.cos_table[pos]  # [half] int16 (Q1.14)
        sin_m = self.sin_table[pos]  # [half] int16 (Q1.14)

        # Rotate-half pairing: x1' = (x1*c - x2*s) >> 14, x2' = (x2*c + x1*s) >> 14
        out1 = rshift_rnd(x1 * cos_m - x2 * sin_m, 14)
        out2 = rshift_rnd(x2 * cos_m + x1 * sin_m, 14)

        out = np.concatenate([out1, out2], axis=-1)
        return sat(out, -127, 127).astype(np.int8)

    def decode_step(self, token_id: int, pos: int) -> tuple[np.ndarray, int]:
        """Execute one complete integer decode step (Rule N2-1 to N2-13)."""
        # 1. Embedding Gather (N2-5)
        raw_embed = self.weights.embed_table[token_id].astype(np.int64)
        h = rshift_rnd(raw_embed * self.weights.embed_M, self.weights.embed_s)
        h = sat(h, self.cfg.res_min, self.cfg.res_max)

        gqa_group = self.cfg.num_attention_heads // self.cfg.num_key_value_heads

        # 2. Iterate layers
        for l_idx, layer in enumerate(self.weights.layers):
            # Attention block
            a = self.integer_rmsnorm(h, layer.in_norm_gamma, layer.in_norm_M, layer.in_norm_s)

            # Linear projections
            q = requant(a.astype(np.int64) @ layer.q_proj.T, layer.q_M, layer.q_s)
            k = requant(a.astype(np.int64) @ layer.k_proj.T, layer.k_M, layer.k_s)
            v = requant(a.astype(np.int64) @ layer.v_proj.T, layer.v_M, layer.v_s)

            # Reshape into heads
            q = q.reshape(self.cfg.num_attention_heads, self.cfg.head_dim)
            k = k.reshape(self.cfg.num_key_value_heads, self.cfg.head_dim)
            v = v.reshape(self.cfg.num_key_value_heads, self.cfg.head_dim)

            # RoPE
            q = self.integer_rope(q, pos)
            k = self.integer_rope(k, pos)

            # Cache write
            self.kv_cache.write(l_idx, pos, k, v)
            k_hist, v_hist = self.kv_cache.get(l_idx, pos)  # [n_kv, pos+1, hd]

            # GQA Attention
            attn_out = np.zeros((self.cfg.num_attention_heads, self.cfg.head_dim), dtype=np.int64)
            for q_idx in range(self.cfg.num_attention_heads):
                kv_idx = q_idx // gqa_group
                k_head = k_hist[kv_idx].astype(np.int64)  # [pos+1, hd]
                v_head = v_hist[kv_idx].astype(np.int64)  # [pos+1, hd]
                q_head = q[q_idx].astype(np.int64)  # [hd]

                # Dot products (folded 1/sqrt(hd) shift by 2 or 3)
                scores = (q_head @ k_head.T) >> 2  # INT32
                m = int(np.max(scores))
                delta = scores - m  # non-positive differences

                # Softmax base-2 approximation
                # Integer shift + fraction LUT
                exp_vals = []
                for d in delta:
                    int_part = abs(int(d)) >> 8
                    frac_idx = abs(int(d)) & 0xFF
                    # Exp2 approximation
                    frac_val = self.exp2_lut[frac_idx]
                    exp_v = frac_val >> min(16, int_part)
                    exp_vals.append(max(1, exp_v))
                exp_arr = np.array(exp_vals, dtype=np.int64)
                sigma = max(1, int(np.sum(exp_arr)))

                # Reciprocal multiplication
                probs = (exp_arr * 32767) // sigma  # Q0.15 format
                attn_head = (probs @ v_head) >> 15
                attn_out[q_idx] = attn_head

            # Output projection and residual add
            attn_flat = sat(attn_out.reshape(-1), -127, 127)
            o_acc = attn_flat.astype(np.int64) @ layer.o_proj.T
            o_res = rshift_rnd(o_acc * layer.o_M, layer.o_s)
            h = sat(h + o_res, self.cfg.res_min, self.cfg.res_max)

            # SwiGLU FFN block
            m_act = self.integer_rmsnorm(h, layer.post_norm_gamma, layer.post_norm_M, layer.post_norm_s)
            gate = requant(m_act.astype(np.int64) @ layer.gate_proj.T, layer.gate_M, layer.gate_s)
            up = requant(m_act.astype(np.int64) @ layer.up_proj.T, layer.up_M, layer.up_s)

            # SiLU LUT lookup (offset by 128 for index)
            silu_indices = (gate.astype(np.int64) + 128).clip(0, 255)
            silu_val = self.silu_lut[silu_indices].astype(np.int64)
            swiglu_prod = requant(silu_val * up.astype(np.int64), 1, 7)

            down_acc = swiglu_prod.astype(np.int64) @ layer.down_proj.T
            down_res = rshift_rnd(down_acc * layer.down_M, layer.down_s)
            h = sat(h + down_res, self.cfg.res_min, self.cfg.res_max)

        # 3. Final RMSNorm
        f = self.integer_rmsnorm(h, self.weights.final_norm_gamma, self.weights.final_norm_M, self.weights.final_norm_s)

        # 4. LM Head Projection with scale alignment (N2-12)
        raw_logits = f.astype(np.int64) @ self.weights.lm_head.T
        aligned_logits = rshift_rnd(raw_logits * self.weights.lm_head_M, self.weights.lm_head_s)

        # 5. Greedy ArgMax (N2-13: lowest index tie-break)
        next_token_id = int(np.argmax(aligned_logits))

        return aligned_logits, next_token_id

    def score_sequence(self, sequence_tokens: list[int]) -> list[int]:
        """Teacher-forcing scorer: feeds tokens sequentially and records argmax outputs."""
        self.reset()
        predicted_tokens = []
        for pos, tok in enumerate(sequence_tokens):
            _logits, next_id = self.decode_step(tok, pos)
            predicted_tokens.append(next_id)
        return predicted_tokens
