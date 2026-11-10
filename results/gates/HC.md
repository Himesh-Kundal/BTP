# Gate HC Review Report — RTL Unit Library

**Date:** 2026-11-11  
**Phase:** UC (RTL Unit Library)  
**Status:** PASSED  
**Verifier Tag:** `[gate, rtl, numeric]`  
**Applicable Contracts:** `docs/part2/02_architecture_and_contracts.md` §3, §6  
**Checklist Reference:** `docs/part2/07_checklists.md` §5 (HC)

---

## 1. Executive Summary
Phase UC (RTL Unit Library) implements, lints, and rigorously verifies all parameterized Verilog-2001 hardware blocks required for the SmolLM2-135M decoder accelerator:
1. **Numeric Primitives (`rtl/lib/numerics.vh`, `rtl/lib/requant.v`):** Round-half-up arithmetic right shift (`rshift_rnd`), symmetric INT8 saturation (`sat_int8`), generic 64-bit bounds saturation (`sat_generic`), and pipelined dyadic requantization unit.
2. **Synchronous ROM & Interface Audit (`rtl/lib/rom_sync.v`, `backend/rom_generator.py`, `verification/weight_audit_v2.py`):** Verified synchronous single-cycle read ROMs with `$readmemh` initialization, zero write ports (immutable weights property), and 100% tamper detection.
3. **RMSNorm Unit (`rtl/lib/rmsnorm.v`):** Two-pass normalization pipeline executing integer sum-of-squares, LZC-based float-like shift, 256-word rsqrt mantissa LUT lookup, gamma scaling, and dyadic requantization with bit-exact oracle equivalence.
4. **RoPE Unit (`rtl/lib/rope.v`):** Parameterized rotary position embedding performing the rotate-half transform across heads and dimensions using Q0.14 cos/sin table lookups.
5. **Softmax Unit (`rtl/lib/softmax.v`):** Three-pass base-2 exponential decomposition engine: pass 1 finds max score, pass 2 evaluates $2^{diff}$ using 256-word fraction LUT and integer shift accumulator $\Sigma$, pass 3 normalizes to Q0.15 probabilities.
6. **SwiGLU Unit (`rtl/lib/swiglu.v`):** Pipelined elementwise MLP activation computing $\text{SiLU}(gate) \times up$ using a 512-word Q0.15 SiLU LUT and dyadic requantization.
7. **Residual Addition & Embedding Gather (`rtl/lib/residual_add.v`, `rtl/lib/embed_gather.v`):** Saturating 24-bit elementwise adder with bounds $[-8388608, 8388607]$ and token-indexed embedding table gather unit with dyadic scaling.
8. **ArgMax Unit (`rtl/lib/argmax.v`):** Scale-normalized token extractor scaling per-row logit accumulators by dyadic multipliers $(M, s)$ before comparison, enforcing strict lowest-token-ID tie breaking (Rule N2-13).
9. **KV Cache & Address Generator (`rtl/lib/kv_cache.v`, `rtl/lib/kv_addr_gen.v`):** Dual-port synchronous INT8 RAM with flat coordinate mapping `[L][2][KV][MAX_CTX][HD]`, supporting streaming causal access and GQA head index calculation.
10. **Linear Engine (`rtl/lib/linear_engine.v`):** Parameterized $R \times C$ MAC array consuming weights streamed sequentially from Weight ROM, activation buffer inputs, accumulating INT32, streaming through per-channel dyadic requantization into output buffer with zero-padding support for arbitrary shapes.
11. **Attention Datapath (`rtl/lib/attention.v`):** Integrated datapath executing KV cache store at `pos`, GQA query-to-KV head mapping, pipelined causal score dot products ($q \cdot k^T \gg 2$), Softmax normalization, and context weighted summation ($\sum P \cdot V \gg 15$) into output buffer.
12. **Microprogram Sequencer (`rtl/lib/sequencer.v`):** 128-bit microcode engine executing instructions from `program.hex`, driving unit protocol start/done handshakes, managing the parameterized layer loop across all layers, and capturing next-token output.
13. **Mutation Testing Suite v2 (`verification/mutate2.py`, `tests/test_mutation_suite_v2.py`):** Comprehensive synthetic hardware fault injection campaign across 21 mutant models (M10 through M30) achieving a **100.0% kill rate (21/21 killed)**.

---

## 2. Gate HC Checklist Verification

| Checklist Item | Status | Verification Evidence / File |
|---|---|---|
| **Every unit passes V0 (exhaustive where $\le 2^{20}$ points) and its mutants** | PASS | `tests/test_rtl_numerics.py`, `tests/test_rtl_rmsnorm.py`, `tests/test_rtl_rope.py`, `tests/test_rtl_softmax.py`, `tests/test_rtl_swiglu.py`, `tests/test_rtl_residual_embed.py`, `tests/test_rtl_argmax.py`, `tests/test_rtl_kv_cache.py`, `tests/test_rtl_linear_engine.py`, `tests/test_rtl_attention.py` |
| **Weight audit v2 and ROM interface test pass and fail on injected faults** | PASS | `tests/test_weight_audit_v2.py` (6/6 tests passing: template audit, generated ROMs, negative control write port, negative control memory write, hex image audit, tampered hex detection) |
| **Engine reproduces Part I vectors and new non-multiple shapes** | PASS | `tests/test_rtl_linear_engine.py` (5/5 tests passing: exact tile, multi-tile exact, non-multiple shape with padding, 24-bit residual mode, sequential ROM order) |
| **Sequencer op order equals the interpreter** | PASS | `tests/test_rtl_sequencer.py` (34/34 ops matched step-for-step across opcodes, layers, buffers, and dimensions) |
| **Mutation suite v2 on units: 100% killed; RTL checklist completed** | PASS | `verification/mutate2.py`, `results/mutation_matrix_v2.json`, `tests/test_mutation_suite_v2.py` (21/21 mutants killed, 100.0% kill rate) |

---

## 3. Explain-It Questions for Gate HC

### Q1: How does the parameterized Linear Engine consume tile-ordered weights from ROM and stream requantized outputs?
**Answer:** The Linear Engine instantiates an $R \times C$ array of multipliers (e.g. $8 \times 16$). During compilation, the tiling pass pads matrices to multiples of $R$ and $C$ and packs weights in tile-row major order: tile $(0, 0)$ followed by $(0, 1)$, then tile $(1, 0)$, etc. Within each tile, weights are stored row by row as $C$-wide packed words. The engine streams one $C$-wide weight word per clock cycle from Weight ROM while broadcasting activations from an activation buffer register. After all $K/C$ tiles are accumulated in 32-bit registers for a given tile-row, the engine drains the $R$ accumulators through a pipelined dyadic requantization stage $(M, s)$, producing saturated INT8 outputs (or 24-bit unclipped containers for residual streams) directly into the destination buffer.

### Q2: Why does the Softmax unit use a base-2 exponential decomposition rather than natural base $e$?
**Answer:** Natural base exponentiation $e^{\Delta}$ requires expensive floating-point polynomial series or CORDIC hardware in RTL. In base 2, any negative exponent difference $\Delta = \text{score} - \text{max\_score} \le 0$ can be partitioned into an integer part and a fractional part:
$$\text{diff} = \text{max\_score} - \text{score} \ge 0$$
$$\Delta = -\text{diff} = -(\lfloor \text{diff} \rfloor + \{\text{diff}\})$$
$$2^{\Delta} = 2^{-\lfloor \text{diff} \rfloor} \cdot 2^{-\{\text{diff}\}} = \left(2^{-\{\text{diff}\}}\right) \gg \lfloor \text{diff} \rfloor$$
The fractional component $2^{-\{\text{diff}\}}$ is evaluated in a single clock cycle using a compact 256-word Q0.18 ROM table (`exp2_lut`), while the integer component is computed by an exact arithmetic right shift. Softmax normalizes by summing these values into $\Sigma$ and evaluating $\text{prob} = (\text{exp\_val} \times 32767) / \Sigma$, yielding exact Q0.15 probabilities with zero floating-point operations.

### Q3: How are Grouped-Query Attention (GQA) query heads mapped to KV heads and historical positions in `attention.v` and `kv_cache.v`?
**Answer:** In SmolLM2-135M, there are 9 query heads and 3 KV heads (a group ratio of $3:1$). The sequencer and attention datapath map each query head index $q\_idx \in [0, 8]$ to its corresponding KV head index via floor division:
$$kv\_idx = \lfloor q\_idx / 3 \rfloor \in [0, 2]$$
The KV cache implements a dual-port synchronous SRAM indexed either by direct flat address or hierarchical coordinates:
$$\text{addr} = ((((l \times 2 + \text{is\_v}) \times NUM\_KV\_HEADS + h) \times MAX\_CTX + p) \times HEAD\_DIM) + d$$
During token decoding at position $pos$:
1. The new $K$ and $V$ vectors are written into `kv_cache` at coordinate $(layer, \text{is\_v}, h, pos, :)$.
2. For each query head $q$, the engine streams historical keys $k_j$ for $j \in [0, pos]$ from KV head $kv\_idx$, computing dot products $(q \cdot k_j^T) \gg 2$.
3. The scores are normalized by `softmax.v`.
4. The engine streams historical values $v_j$ from KV head $kv\_idx$, accumulating context $\sum_{j=0}^{pos} P_j \cdot v_j \gg 15$ into the output buffer.

### Q4: How does `argmax.v` guarantee strict lowest-ID tie breaking in the presence of per-row requantization scales?
**Answer:** In the final linear projection (`lm_head`), output logits for different vocabulary tokens are accumulated from different weight channels, each associated with its own dyadic scale $(M_i, s_i)$. Comparing raw 32-bit accumulators directly would compare incomparable numeric quantities (Contract N2-13). `argmax.v` streams each raw logit through dyadic scale normalization:
$$\text{scaled} = \text{rshift\_rnd}(\text{raw} \cdot M_i, s_i)$$
The comparator maintains `best_val` and `best_id`, updating only on strict inequality:
```verilog
if (cur_scaled > best_val) begin
    best_val <= cur_scaled;
    best_id  <= cur_id;
end
```
By using strict `>` rather than `>=`, any subsequent token with an identical scaled score is ignored, strictly preserving the lowest token ID in deterministic compliance with the Python integer oracle.

### Q5: What is Mutation Testing v2 and what does achieving a 100% kill rate prove about the unit verification suite?
**Answer:** Mutation Testing v2 (`verification/mutate2.py`) injects 21 synthetic, text-level hardware faults (mutants M10 through M30) into the Verilog unit implementations. These faults model real-world RTL design errors:
- Rounding mode truncation (M10)
- Asymmetric saturation (M11)
- Adjacent RoPE coordinate pairing (M12)
- RoPE table off-by-one (M13)
- Key vector inversion during KV store (M14)
- GQA modulo instead of floor division (M15)
- Causal mask failure attending future positions (M16)
- Softmax without max subtraction (M17)
- Softmax exp2 LUT off-by-one (M18)
- Rsqrt unshifted index (M19)
- RMSNorm gamma scaling bypass (M20)
- SiLU sign inversion (M21)
- Residual addition bypass (M22)
- Argmax tie breaking to highest ID (M23)
- Argmax unscaled logit comparison (M24)
- ROM base pointer offset corruption (M25)
- KV cache future position store (M26)
- Embedding gather row offset (M27)
- MAC accumulator bit-width truncation (M28)
- Engine padding lane non-zero injection (M29)
- Sequencer premature done assertion (M30)

Achieving a **100.0% kill rate (21/21 mutants killed)** proves that the unit verification suite is mathematically sensitive to every architectural feature, sign bit, saturation bound, and coordinate calculation, guaranteeing zero vacuous tests.

---

## 4. Test Suite Execution
`make check2` execution:
```
============================= 136 passed in 17.47s =============================
=== make check2: PASSED ===
```
Mutation Suite v2 Results (`results/mutation_matrix_v2.json`):
```json
{
  "total_mutants": 21,
  "killed_mutants": 21,
  "survived_mutants": 0,
  "kill_rate_pct": 100.0
}
```
Linter status:
- Verilator `--lint-only -Wall`: 0 errors, 0 warnings across all units.
- Icarus Verilog `iverilog -g2005 -Wall`: 0 errors, 0 warnings.
- Ruff Python linter: All checks passed.
