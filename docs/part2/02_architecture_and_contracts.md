# Part II / 02 — Architecture and Contracts (frozen spec, after gate HA)

Part I's contracts (`docs/02`) remain in force for the linear op (Y = X·Wᵀ + b, INT8 symmetric, per-channel weight scales, INT32 accumulation) **except** that requantization and dequantization are now in RTL (see §3 N2-3). Numeric *defaults* in §3 are starting points that gate HA validates and freezes by ADR; the *structure* (rules, interfaces, immutability) is fixed now.

## 1. System boundary
```
Host (Python/testbench):  text ──tokenizer──► token ids          next id ──► detokenizer ──► text
                                         │                              ▲
                                         ▼                              │
                        ┌──────────── Model accelerator (generated RTL) ─────────────┐
                        │ token step interface  →  sequencer ─► units ─► buffers      │
                        │ ROMs (immutable): weights, tables, requant params, program  │
                        │ RAMs (dynamic):   KV cache, activation buffers              │
                        └─────────────────────────────────────────────────────────────┘
```
**Step interface (top level):** `clk, rst_n, seq_reset, step_valid, step_ready, tok_in[15:0], out_valid, tok_out[15:0]`. The host presents one token id per step; the accelerator runs one decode step at its internal position counter `pos` (reset by `seq_reset`) and pulses `out_valid` with the argmax token. During prefill the host ignores `tok_out` and feeds the next prompt token; during generation it feeds back `tok_out`. Vocabulary (49152 expected) fits in 16 bits.

## 2. Model facts (verify in U0.3 and write `docs/model_facts.md`; nothing below is assumed correct)
Expected: vocab 49152 (tied embedding/LM head), d = 576, L = 30, 9 query heads, 3 KV heads (group size 3), head_dim 64, FFN 1536, SwiGLU with SiLU, RMSNorm with config eps, RoPE (rotate-half pairing `(i, i+hd/2)`, theta from config), no linear biases.

| Quantity | Value (expected) |
|---|---|
| Layer weights | 3.54M × 30 = 106.2M |
| Embedding / LM head (tied) | 28.3M |
| MACs per decoded token | ≈134.5M + attention `34,560 × ctx` |
| Share of per-token MACs in the 7 projections (short ctx) | ≈79% (LM head ≈21%) |
| KV cache | 11,520 bytes per position at INT8 (2 · 30 · 3 · 64) |
| Weight ROM at INT8 | ≈1.08 Gbit |

## 3. Numeric contract v2
Defaults marked (D) are validated and frozen at HA. Everything is integer; no float in RTL or in the reference path.

| # | Rule |
|---|---|
| N2-1 | Every edge in the IR carries `(bits, signed, scale, fixed-point format)`. Hardware never sees a float. |
| N2-2 | `rshift_rnd(x, s) = (x + (1 << (s-1))) >>> s` (arithmetic shift, round half up toward +∞); `s = 0` is identity. Saturation: INT8 clips to [−127, 127] (symmetric), other widths clip to their full range unless stated. One shared implementation in `numerics/ops.py` and one in `rtl/lib/numerics.vh`; both are tested against each other exhaustively on small widths. |
| N2-3 | Requantization (dyadic): `y = sat(rshift_rnd(acc · M, s))`, `M` positive integer, `s ∈ [0, 62]`, constants in ROM, per output channel for linear outputs. The compiler derives `(M, s)` offline from real-valued scales in float64. Widths of `M` and of the product are set by UA (cost proxy in UA.9). |
| N2-4 | Residual stream `h`: signed integer with a power-of-two scale chosen by calibration (D: 24-bit container, saturating adds; UA sweeps {16, 20, 24, 32}). Layer outputs are requantized into `h`'s format before the add. |
| N2-5 | Embedding gather: row `t` of the tied table (INT8, per-row scale) → residual format via per-row `(M, s)` ROM. |
| N2-6 | RMSNorm: `ss = Σ h²` in a wide accumulator; `mean = rshift_rnd(ss · INV_D, k)` with a constant reciprocal of d; `rsqrt` by normalization (leading-zero count, even shift) + mantissa LUT (D: 8–10 bits, optional Newton step); `y_i = requant(h_i · r · γ_i)` with `γ_i` folded into per-channel `(M_i, s_i)` ROM; output INT8 with a static scale. LUT bits, widths and Newton iterations are UA parameters. |
| N2-7 | Linear: Part I contract with requant. Inputs INT8 (D; UA may select INT16 for specific layers, e.g. `down_proj` input), weights INT8 per-channel, accumulator width from the bound analysis, output requantized per N2-3. |
| N2-8 | RoPE: cos/sin tables in ROM for positions `0..MAX_CTX-1` and `hd/2` frequencies (D: Q1.14, int16). `x1' = rshift_rnd(x1·c − x2·s, 14)`, `x2' = rshift_rnd(x2·c + x1·s, 14)` (rotate-half pairing). Applied to q and k; K is stored **after** RoPE. |
| N2-9 | Attention (GQA, causal by construction): q-head `i` uses kv-head `i // 3`. `score_j = Σ_d q_d · k_{j,d}` (INT32); `1/√64 = 2⁻³` is folded into the score scale; positions `j ∈ [0, pos]` only. |
| N2-10 | Softmax: scores converted to a fixed-point `z`; subtract the running max; `exp` via base-2 decomposition (integer part = shift, fractional part = LUT, D: 256 entries, Q0.16); sum in a wide accumulator; reciprocal by normalization + LUT; probabilities `p_j` unsigned Q0.PB (D: PB = 15). Output `Σ p_j · v_j` requantized to INT8 for `o_proj`. Multi-pass over cached scores (scores buffer of `MAX_CTX` entries). |
| N2-11 | SwiGLU: `silu` by LUT over the INT8 gate output (D: 256 entries, exhaustively testable; UA may widen the input to INT16); product with `up`; requantized to INT8 for `down_proj`. |
| N2-12 | Final RMSNorm → INT8 → LM head (tied table). **Trap:** weights are per-row quantized, so raw INT32 accumulators are *not comparable across vocabulary entries*. Apply the per-row `(M, s)` into a common fixed-point logit format before argmax, or quantize the LM head per-tensor (UA decides; either is frozen at HA). |
| N2-13 | ArgMax: greedy; ties resolve to the **lowest token id**; scan order and tie-break are part of the spec. |
| N2-14 | LUT and table contents are generated by `numerics/` from closed-form definitions in float64 with explicit rounding; hashes recorded in the manifest; RTL loads the same files. |
| N2-15 | Two claims. **C1′:** RTL equals the integer reference exactly: every traced tensor and every output token. **C2′:** integer reference vs fp32, reported (perplexity, KL, top-1 agreement, greedy match length); thresholds fixed at HA from measurements. |

## 4. Decode step dataflow (per token, position `pos`)
```
h   = Embed[tok]                                    (N2-5)
for l in 0..L-1:
    a = RMSNorm_attn[l](h)             → INT8
    q = Linear_q[l](a)  → RoPE(pos)    ; k = Linear_k[l](a) → RoPE(pos) → Kcache[l][·][pos]
    v = Linear_v[l](a)                 → Vcache[l][·][pos]
    c = Attention(q, Kcache[l][0..pos], Vcache[l][0..pos])   (GQA, softmax)
    h = h + requant(Linear_o[l](c))                          (saturating)
    m = RMSNorm_mlp[l](h)              → INT8
    g = Linear_gate[l](m) ; u = Linear_up[l](m)
    s = SwiGLU(g, u)                   → INT8
    h = h + requant(Linear_down[l](s))
f = RMSNorm_final(h) → INT8 ; logits = LMHead(f) ; tok_out = ArgMax(logits)
```

## 5. IR v2 (graph)
Dataclasses in `ir/`; no HF objects, no Verilog; JSON + `.npy`/hex references with sha256.

| Element | Content |
|---|---|
| Node ops | `Embed, RMSNorm, Linear, RoPE, Attention, SiLU, EltMul, Add, ArgMax` |
| Tensor kinds | `CONSTANT` (weights, tables, requant params), `DYNAMIC` (activations), `STATE` (KV cache) |
| Edge annotation | `bits, signed, scale, format, producer, consumers` |
| Node attributes | dims, head layout (n_heads, n_kv, hd), ROM refs, requant param refs, MAX_CTX |
| Provenance | strings only (model id, revision, module path) |

Invariants (each has a test): I1 shapes consistent · I2 value ranges fit declared formats · I3 CONSTANT tensors never feed a write port · I4 round-trip lossless · I5 no torch/transformers/Verilog imports · I6 every edge has a format · I7 STATE tensors accessed only by Attention/KV ops.

## 6. Hardware architecture
**Principles:** weights, tables, requant parameters and the program are ROM; activations and KV cache are RAM; everything is driven by a generated static schedule.

- **Sequencer:** executes a **microprogram** held in a program ROM. Instruction fields: `op, dims, src_buf, dst_buf, rom_base_index, flags`. One program covers a layer; a layer counter selects per-layer ROM bases from a constant table (so the program is tens of instructions, not thousands).
- **Linear engine:** R×C MAC lanes. Per cycle it consumes C input elements (from the input buffer) and R×C weights (from ROM in the order produced by the Part I tiling pass), accumulates R output rows, then streams requantized outputs. Cycles for an N×K matrix ≈ `ceil(N/R)·ceil(K/C)` plus fill/drain; padding follows Part I's tiling contract (zero-padded).
- **Non-linear units:** RMSNorm, RoPE, softmax, SwiGLU, residual add, requant, embedding gather, argmax. Each exposes the **unit protocol**: `start, busy, done`, parameters from the sequencer, one read port on its source buffer and one write port on its destination buffer.
- **KV cache:** layout `[layer][kv_head][pos][hd]` INT8, generated behavioral synchronous RAM sized by `MAX_CTX` (default set at U0.4/UB.6).
- **Buffers:** activation buffers generated by the memory planner; simple dual-port (1R1W).
- **ROMs:** `module rom_<name>(input clk, input [A-1:0] addr, output reg [W-1:0] data)`; initial contents via `$readmemh` from compiler-generated `.hex`; no write port, no write enable. Large images are loaded from files, never emitted as giant `case` statements.
- **Immutable-weights property:** no module outside ROMs stores weights; no path writes a ROM; audited by VR2-5.
- **Part I spatial design** is not used in the full-model top level. It is used in EA4 (block-level comparison).

## 7. RTL conventions (extend Part I `docs/02` §6)
1. Verilog-2001/2005 subset for generated and library RTL (`$clog2`, `generate` allowed). Testbenches plain Verilog.
2. Signed discipline, explicit widths, explicit sign extension; no implicit width-dependent behaviour; shared `numerics.vh` macros/functions for `rshift_rnd`, `sat`, `requant`.
3. Every unit has parameters for widths; no magic numbers outside `numerics.vh` and generated config files.
4. No latches, tri-states, `#` delays or `$display` in RTL. Trace taps are named wires/regs listed in the manifest and read hierarchically by the testbench (Verilator: `--public-flat-rw`).
5. Generated files carry the manifest hash, no timestamps or absolute paths.
6. Lint gates: `iverilog -g2005 -Wall`, `verilator --lint-only -Wall`, Yosys `hierarchy -check; proc; check -assert`.

## 8. Trace and golden formats
- **Trace record:** `(seq_id, token_idx, layer, op, tensor) → <name>.hex`, one element per line, two's complement hex of the declared width. Directory layout `build/<run>/trace/seq<id>/tok<idx>/L<layer>/<op>_<tensor>.hex`.
- **Golden:** produced by `numerics/` for the tiny model (all tensors, all tokens) and for SmolLM2 (layer 0 all tokens; sampled layers and tokens elsewhere, plus every output token). Hashes in `verification/golden/MANIFEST.json`. Frozen after HA.
- **`verification/trace_diff.py`:** compares two trace directories and reports the first diverging `(token, layer, op, tensor, element)`.
- Testbench prints one summary line: `TB2_RESULT TOKENS=<n> TOK_MISMATCH=<m> TRACE_CHECKED=<k> TRACE_MISMATCH=<j> TIMING_ERR=<t>`; the runner passes only if counts equal expectations and mismatches are zero.

## 9. Manifest v2 additions
Model id/revision; `numerics` config hash; format table hash; ROM list with sizes and sha256; address map hash; program ROM hash; engine config `(R, C)`; `MAX_CTX`; unit parameters; cycle estimate per op and per token; tool versions; git commit.

## 10. Repository additions
`numerics/` (ops, LUT builders, integer decoder, compiled-artifact interpreter) · `eval/` (fp32 reference, datasets, metrics) · `compiler/` (format assignment, ROM packing, schedule, memory plan) · `backend/` (top generator, ROM generator) · `rtl/lib/` (units, `numerics.vh`) · `sim/` (testbench generators, Verilator harness) · `verification/` (`trace_diff.py`, `mutate2.py`, `weight_audit2.py`, golden) · `results/part2/`.
