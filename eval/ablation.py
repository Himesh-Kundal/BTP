"""Per-Op Numerical Ablation Experiment EN1 (Task UA.5).

Swaps individual operations between FP32 and Integer to attribute quality loss:
1. All FP32 baseline
2. Integer RMSNorm only
3. Integer RoPE only
4. Integer Softmax only
5. Integer SwiGLU only
6. Integer Linear Projections only
7. All Integer (Full Model)

Logs results, metrics (PPL, Delta PPL, KL, Top-1), and saturation counts to:
results/part2/ua5_ablation.csv
"""

import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from eval.fp32.decoder import (
    DecoderConfig,
    ModelWeights,
    NumPyFP32Decoder,
    apply_rope,
    load_model_weights_from_hf,
    rms_norm,
    silu,
)
from eval.harness import (
    compute_cross_entropy_and_perplexity,
    compute_kl_divergence,
    compute_top1_agreement,
)
from frontend.tiny_model import generate_tiny_model
from numerics.ops import (
    build_exp2_fraction_lut,
    build_reciprocal_lut,
    build_rope_cos_sin_tables,
    build_rsqrt_mantissa_lut,
    build_silu_lut,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_ROOT / "results" / "part2" / "ua5_ablation.csv"


class HybridAblationDecoder(NumPyFP32Decoder):
    """Configurable decoder allowing individual operators to run in integer mode."""

    def __init__(
        self,
        config: DecoderConfig,
        weights: ModelWeights,
        int_rmsnorm: bool = False,
        int_rope: bool = False,
        int_softmax: bool = False,
        int_swiglu: bool = False,
        int_linear: bool = False,
    ):
        super().__init__(config, weights)
        self.int_rmsnorm = int_rmsnorm
        self.int_rope = int_rope
        self.int_softmax = int_softmax
        self.int_swiglu = int_swiglu
        self.int_linear = int_linear

        # Precompute LUTs
        self.exp2_lut = build_exp2_fraction_lut(256, 16)
        self.rsqrt_lut = build_rsqrt_mantissa_lut(256, 16)
        self.recip_lut = build_reciprocal_lut(256, 16)
        self.silu_lut = build_silu_lut(-128, 127)
        self.cos_tab, self.sin_tab = build_rope_cos_sin_tables(
            self.max_ctx, config.head_dim, config.rope_theta, 14
        )

        # Statistics
        self.saturation_count = 0

    def decode_step(self, token_id: int, pos: int) -> tuple[np.ndarray, int]:
        h = self.weights.embed_tokens[token_id].astype(np.float32).copy()
        gqa_group = self.cfg.num_attention_heads // self.cfg.num_key_value_heads
        scale = 1.0 / np.sqrt(self.cfg.head_dim)

        for l_idx, layer in enumerate(self.weights.layers):
            # Attention RMSNorm
            if self.int_rmsnorm:
                # Quantized norm
                var = np.mean(h * h)
                idx = min(255, int(abs(var * 256.0)) & 0xFF)
                r_val = self.rsqrt_lut[idx] / 65536.0
                norm_a = h * r_val * layer.input_layernorm
                if np.any(np.abs(norm_a) > 127):
                    self.saturation_count += 1
                norm_a = np.clip(norm_a, -127, 127)
            else:
                norm_a = layer.input_layernorm * (
                    h / np.sqrt(np.mean(h * h) + self.cfg.rms_norm_eps)
                )

            # Linear projections
            if self.int_linear:
                q = np.clip(norm_a @ layer.q_proj.T, -127, 127)
                k = np.clip(norm_a @ layer.k_proj.T, -127, 127)
                v = np.clip(norm_a @ layer.v_proj.T, -127, 127)
            else:
                q = norm_a @ layer.q_proj.T
                k = norm_a @ layer.k_proj.T
                v = norm_a @ layer.v_proj.T

            q = q.reshape(self.cfg.num_attention_heads, self.cfg.head_dim)
            k = k.reshape(self.cfg.num_key_value_heads, self.cfg.head_dim)
            v = v.reshape(self.cfg.num_key_value_heads, self.cfg.head_dim)

            # RoPE
            if self.int_rope:
                half = self.cfg.head_dim // 2
                c = self.cos_tab[pos] / 16384.0
                s = self.sin_tab[pos] / 16384.0
                q = np.concatenate([q[:, :half] * c - q[:, half:] * s, q[:, half:] * c + q[:, :half] * s], axis=-1)
                k = np.concatenate([k[:, :half] * c - k[:, half:] * s, k[:, half:] * c + k[:, :half] * s], axis=-1)
            else:
                q = apply_rope(q, pos, self.inv_freq)
                k = apply_rope(k, pos, self.inv_freq)

            self.kv_cache.write(l_idx, pos, k, v)
            k_hist, v_hist = self.kv_cache.get(l_idx, pos)

            # GQA Attention
            attn_out = np.zeros((self.cfg.num_attention_heads, self.cfg.head_dim), dtype=np.float32)
            for q_idx in range(self.cfg.num_attention_heads):
                kv_idx = q_idx // gqa_group
                k_head = k_hist[kv_idx]
                v_head = v_hist[kv_idx]
                q_head = q[q_idx]

                scores = (q_head @ k_head.T) * scale

                if self.int_softmax:
                    m = np.max(scores)
                    diff = scores - m
                    exp_v = np.exp2(diff * 1.44269504)  # log2(e)
                    probs = exp_v / np.sum(exp_v)
                else:
                    m = np.max(scores)
                    exp_s = np.exp(scores - m)
                    probs = exp_s / np.sum(exp_s)

                attn_out[q_idx] = probs @ v_head

            attn_flat = attn_out.reshape(-1)
            o = attn_flat @ layer.o_proj.T
            h = h + o

            # MLP block
            if self.int_rmsnorm:
                var = np.mean(h * h)
                idx = min(255, int(abs(var * 256.0)) & 0xFF)
                r_val = self.rsqrt_lut[idx] / 65536.0
                norm_m = h * r_val * layer.post_attention_layernorm
            else:
                norm_m = layer.post_attention_layernorm * (
                    h / np.sqrt(np.mean(h * h) + self.cfg.rms_norm_eps)
                )

            gate = norm_m @ layer.gate_proj.T
            up = norm_m @ layer.up_proj.T

            if self.int_swiglu:
                indices = np.clip(np.round(gate * 20.0) + 128, 0, 255).astype(int)
                silu_act = self.silu_lut[indices] / 20.0
                swiglu_out = silu_act * up
            else:
                swiglu_out = silu(gate) * up

            down = swiglu_out @ layer.down_proj.T
            h = h + down

        # Final norm & head
        f = rms_norm(h, self.weights.final_norm, eps=self.cfg.rms_norm_eps)
        logits = f @ self.weights.lm_head.T
        next_tok = int(np.argmax(logits))
        return logits, next_tok


def run_ablation_study() -> list[dict]:
    """Execute per-op ablation study on synthetic test sequences."""
    print("[*] Running Per-Op Numerical Ablation (Task UA.5)...")
    hf_model = generate_tiny_model(seed=42)
    cfg, weights = load_model_weights_from_hf(hf_model)

    test_seqs = [
        [10, 20, 30, 40, 50],
        [5, 15, 25, 35, 45, 55],
        [1, 2, 4, 8, 16],
        [99, 101, 103, 107],
        [12, 34, 56, 78],
    ]

    configs = [
        ("FP32_Baseline", {}),
        ("Int_RMSNorm", {"int_rmsnorm": True}),
        ("Int_RoPE", {"int_rope": True}),
        ("Int_Softmax", {"int_softmax": True}),
        ("Int_SwiGLU", {"int_swiglu": True}),
        ("Int_Linear", {"int_linear": True}),
        ("All_Integer", {
            "int_rmsnorm": True,
            "int_rope": True,
            "int_softmax": True,
            "int_swiglu": True,
            "int_linear": True,
        }),
    ]

    # Baseline FP32 run
    base_dec = HybridAblationDecoder(cfg, weights)
    base_logits_all = []
    targets_all = []
    for seq in test_seqs:
        base_dec.reset()
        for pos, tok in enumerate(seq[:-1]):
            logits, _ = base_dec.decode_step(tok, pos)
            base_logits_all.append(logits)
            targets_all.append(seq[pos + 1])

    base_logits_arr = np.array(base_logits_all)
    _base_loss, base_ppl = compute_cross_entropy_and_perplexity(base_logits_arr, targets_all)

    results = []
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    for name, kwargs in configs:
        t0 = time.time()
        dec = HybridAblationDecoder(cfg, weights, **kwargs)
        cand_logits_all = []
        for seq in test_seqs:
            dec.reset()
            for pos, tok in enumerate(seq[:-1]):
                logits, _ = dec.decode_step(tok, pos)
                cand_logits_all.append(logits)

        cand_logits_arr = np.array(cand_logits_all)
        _loss, ppl = compute_cross_entropy_and_perplexity(cand_logits_arr, targets_all)
        delta_ppl = ppl - base_ppl
        kl = compute_kl_divergence(base_logits_arr, cand_logits_arr)
        top1 = compute_top1_agreement(base_logits_arr, cand_logits_arr)
        elapsed = round(time.time() - t0, 3)

        row = {
            "config": name,
            "ppl": round(ppl, 4),
            "delta_ppl": round(delta_ppl, 4),
            "kl_div": round(kl, 6),
            "top1_agr_pct": round(top1, 2),
            "saturations": dec.saturation_count,
            "runtime_sec": elapsed,
        }
        results.append(row)
        print(f"  [{name:<15}] PPL={row['ppl']:<8} ΔPPL={row['delta_ppl']:<8} KL={row['kl_div']:<10} Top1={row['top1_agr_pct']}% Sats={row['saturations']}")

    # Write CSV
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"[+] Ablation results saved to: {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    run_ablation_study()
