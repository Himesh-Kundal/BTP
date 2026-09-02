# 02 — Architecture and Contracts (frozen spec)

Anything here is changed only through an ADR (`docs/03`). Code that disagrees with this file is wrong until an ADR says otherwise.

## 1. Dataflow and module boundaries

```
HF model ──frontend──► artifacts/<model>@<rev>/<layer>/{weight.npy, bias.npy?, meta.json}
                              │  (+ calib/x_sample.npy, calib.json from forward hooks)
                              ▼
                     IR (ir/) ◄── compiler passes (compiler/):
                              quantize → constant-weight marking → tiling
                              ▼
                     backend (backend/): Verilog F, Verilog P, manifest.json
                              ▼
                     verification/: vectors (.hex) + testbench → iverilog/verilator → report.json
                              ▼
                     scripts/synth: Yosys → results/*.json
```

Handoffs are files, not Python objects, so each stage is testable alone. Only `frontend/` touches torch/transformers. The compiler consumes `weight.npy` + `meta.json` and nothing model-specific.

### Expected SmolLM2-135M linear shapes (**unverified; T2.4 must confirm and write `docs/model_facts.md`**)
hidden 576, intermediate 1536, 30 layers, 9 heads, 3 KV heads, head_dim 64, no linear biases, tied embeddings.

| Projection | N (out) | K (in) |
|---|---|---|
| q_proj, o_proj | 576 | 576 |
| k_proj, v_proj | 192 | 576 |
| gate_proj, up_proj | 1536 | 576 |
| down_proj | 576 | 1536 |

`lm_head` and embeddings are excluded. `gate_proj` is included because SwiGLU needs it; the compiler is shape-generic anyway.

## 2. Numeric contract
| # | Rule |
|---|---|
| N1 | Op: `Y = X·Wᵀ + b_q`. `X[M,K]` int8, `W[N,K]` int8, `b_q[N]` int32, `Y[M,N]` int32. Hardware processes one row of X per transaction; M is time. |
| N2 | Weight quant: per-output-channel symmetric. `s_w[n] = max|W[n,:]| / 127` (computed in float64). If the row is all zero, `s_w[n] = 1.0` and the row is zero. `W_q = clip(rint(W / s_w), -127, 127)`. `rint` = round-half-to-even. |
| N3 | Activation quant: per-tensor symmetric **static** scale from calibration. `s_x = stat / 127`, `stat` = absmax over the calibration set (default) or a recorded percentile. `X_q = clip(rint(X / s_x), -127, 127)`. |
| N4 | Accumulation is exact integer: int64 in the reference, int32 in RTL. The compiler computes the worst case `K·128·127 + |b_q|max` and **rejects** the layer if ≥ 2³¹. It also records the minimal safe accumulator width in the IR. |
| N5 | Bias: `b_q[n] = clip(rint(b[n] / (s_x·s_w[n])), int32)`. SmolLM2 has no linear biases, so bias paths are tested with synthetic bias. |
| N6 | Dequantization is **not in RTL** (Part I): `y_float ≈ s_x · s_w[n] · Y[m,n]`. |
| N7 | RTL must be correct for the full int8 input range including −128, even though the quantizer never emits it. Weights are in [−127, 127]. |
| N8 | Overflow behaviour is undefined and unreachable by N4. Tests must never depend on wraparound. |
| N9 | Two separate claims. **C1:** RTL == integer reference exactly. **C2:** integer reference ≈ float reference, reported with max-abs error, relative L2 error, cosine similarity, SQNR (dB). C2 thresholds are set from the T3.6 measurements, then frozen. They are not invented in advance. |
| N10 | For fair comparison, F and P use the same widths: x 8, w 8, product 16, accumulator width as in the IR. |

## 3. IR specification
Dataclasses in `ir/`. No HF objects, no Verilog syntax, JSON-serializable (arrays stored beside the JSON as `.npy` and referenced by path + sha256).

`LinearOp`
| Field | Meaning |
|---|---|
| `name`, `op="linear"` | identity |
| `in_features` (K), `out_features` (N) | shape |
| `input_shape` | `(M, K)`; M may be symbolic/1 |
| `weight` | TensorRef: shape `[N,K]`, dtype int8, `kind=CONSTANT`, path, sha256 |
| `bias` | optional TensorRef int32 `[N]` |
| `weight_scale` | float64 `[N]` |
| `act_scale`, `act_dtype=int8` | per-tensor |
| `acc_dtype=int32`, `acc_bits_min` | from N4 |
| `quant` | scheme, rounding mode, calibration id and statistic |
| `hw` | `tile_n`, `tile_k`, `padded_n`, `padded_k`, `arch ∈ {fixed, programmable}`, tile manifest |
| `source` | provenance strings only (model id, revision, module path) |

IR invariants (each has a test): I1 shapes consistent · I2 `W_q ⊂ [−127,127]` · I3 `kind=CONSTANT` ⇒ backend F emits no weight port · I4 serialization round-trip is lossless · I5 no torch/transformers/Verilog imports in `ir/`.

## 4. Tiling semantics
Tile size `(Nt, Kt)` (default 8×16; the evaluation grid varies it).
- `N' = ceil(N/Nt)·Nt`, `K' = ceil(K/Kt)·Kt`. Pad `W` and `X` with zeros. Padded output rows are generated but not exported.
- Tile `(i,j)` holds `W[i·Nt:(i+1)·Nt, j·Kt:(j+1)·Kt]`.
- Output tile `i`: `Y_i = Σ_j T_ij(x_j) + b_i`.
- Tile manifest lists, for every tile: indices, module name, nonzero-weight count, sha256 of its weights.
- Property tests: tiles cover the padded matrix exactly once; padding is all zero; unpadded region equals the original.

## 5. Hardware designs
**Design F (fixed-weight, spatial; MUST).** Per tile, a combinational core with constant coefficients (zero-weight terms omitted; ±1 weights pass through; others use `*` with a constant and are left to synthesis). All tiles instantiated; per output tile an adder tree over `j`; bias added as a constant. A wrapper registers inputs/outputs. No weight ports, no weight storage.

**Design P (programmable baseline; MUST).** Same tile topology, same wrapper, same widths. Weights are runtime data: a register array `W_reg[N'][K']` written via `w_we / w_addr / w_data`. Generic signed multipliers. A second variant `P_bus` takes weights as a primary-input bus (logic-only accounting, no storage). Both are used in the evaluation.

**Design R (coefficient ROM, time-multiplexed; STRETCH S2).** Compute unit reused across input tiles; coefficients selected from a `case`-based constant table. Behind `--arch rom`. Not part of the core comparison.

## 6. RTL interface and coding conventions
**Core:** `module <name>_core (input [8*K'-1:0] x_flat, output [32*N'-1:0] y_flat);` purely combinational.
**Wrapper F:** `(input clk, input rst_n, input valid_in, input [8*K'-1:0] x_flat, output valid_out, output [32*N'-1:0] y_flat)`. One transaction per cycle, no back-pressure, fixed latency `L` (default 1) recorded in the manifest.
**Wrapper P:** same plus `w_we, w_addr, w_data`. The testbench loads weights, then streams the same vectors as F.

Packing: `x_flat[8*i +: 8] = x[i]`, `y_flat[32*n +: 32] = y[n]` (element 0 in the least significant bits).

Conventions:
1. Verilog-2001 subset only. No SystemVerilog in generated RTL. Testbenches are plain Verilog too (Icarus `-g2005`).
2. Everything signed. Unpack `x_flat` into explicitly named `wire signed [7:0] x_i`. Constants are written `8'sd3` / `-8'sd2`. Products get explicit `wire signed [15:0]`. Accumulation uses explicit sign extension into 32 bits. **Never rely on implicit context-width extension and never mix signed with unsigned operands.**
3. Active-low synchronous reset on the valid pipeline only. Data registers need no reset.
4. No latches, no tri-states, no `initial` or `$display` in RTL, no `#` delays.
5. Names: `llmhw_<layer>_<arch>_n<N>k<K>_t<Nt>x<Kt>[_core]`, sanitized to `[a-z0-9_]`.
6. Generated files have a header comment with the manifest hash and **no timestamps or absolute paths**.
7. Lint gates: `iverilog -g2005 -Wall`, `verilator --lint-only -Wall`, Yosys `hierarchy -check; proc; check -assert`.

## 7. Vector and testbench formats
- `x.hex`: `M·K` lines, row-major, 2 hex digits per int8 (two's complement). `y_exp.hex`: `M·N` lines, 8 hex digits per int32. For P, `w.hex`: `N'·K'` lines.
- Padding lanes are driven to 0 by the testbench.
- The testbench packs per-element lines into buses, checks `valid_out` timing, compares, and prints exactly one summary line: `TB_RESULT CHECKED=<n> MISMATCH=<m> TIMING_ERR=<t>`.
- The runner passes only if `CHECKED == expected_count`, `MISMATCH == 0`, `TIMING_ERR == 0`. (A testbench that checks nothing must not pass.)
- Optional `+dump` plusarg writes a VCD for GTKWave.

## 8. Manifest (`build/<run>/manifest.json`)
Model id and revision; layer name; shapes; N, K, N', K', Nt, Kt; arch; latency; scales hash; weight sha256 (raw and quantized); calibration id; tool versions; git commit; generator version; list of generated files with sha256; tile list.

## 9. Repository layout
The plan's layout plus: `docs/`, `state/`, `tasks/`, `results/` (gate reports, experiment JSON/CSV, env), `scripts/` (synth, env-check, demo), `tests/` (unit, layering, property), `verification/golden/` (frozen), `verification/mutate.py`, `verification/weight_audit.py`, `artifacts/` and `build/` (gitignored).
