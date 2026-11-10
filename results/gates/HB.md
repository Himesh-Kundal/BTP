# Gate HB Review Report — Compiler Extension

**Date:** 2026-11-11  
**Phase:** UB (Compiler Extension)  
**Status:** PASSED  
**Verifier Tag:** `[gate, numeric]`  
**Applicable Contracts:** `docs/part2/02_architecture_and_contracts.md` §3, §5, §6  
**Checklist Reference:** `docs/part2/07_checklists.md` §5 (HB)

---

## 1. Executive Summary
Phase UB (Compiler Extension) extends the Part I compiler from isolated linear tiles to the complete autoregressive transformer decoder pipeline:
1. **Graph IR v2 (`ir/graph_v2.py`):** Full decoder graph representation with node ops, tensor kinds (`CONSTANT`, `DYNAMIC`, `STATE`), edge formats, node attributes, provenance strings, and validation of Invariants I1–I7 with negative controls.
2. **Frontend Importer v2 (`frontend/importer_v2.py`):** Complete model extractor and artifact manager with SHA256 integrity verification, supporting both the deterministic tiny model and SmolLM2-135M.
3. **Format & Dyadic Scale Assignment (`compiler/format_assignment.py`):** Automatically maps edges to frozen numerics formats, folding RMSNorm $\gamma$, attention $1/\sqrt{d}$, and per-channel/per-row weights into dyadic scaling parameters $(M, s)$ with $0 \le s \le 62$.
4. **ROM Packing Pass (`compiler/rom_packing.py`):** Formats immutable parameters into `$readmemh`-compatible `.hex` images (`weights.hex`, `requant_params.hex`, `tables.hex`) and generates an exhaustive `address_map.json` with cryptographic SHA256 hashes. Weights are tile-ordered and zero-padded to engine dimensions $(R \times C = 8 \times 16)$.
5. **Sequencer Schedule & Microprogram Generator (`compiler/schedule.py`):** Compresses the whole decoder step into a static microprogram (< 25 instructions) leveraging a parameterized layer loop and per-layer ROM base table. Emits `program.hex` and `schedule.json` with cycle estimates.
6. **Memory Planner (`compiler/memory_planner.py`):** Conducts activation buffer liveness analysis and sizes synchronous KV cache RAM (`[L][2][KV][MAX_CTX][HD]` INT8). Confirms SmolLM2 KV cache requires exactly 2,949,120 bytes (2.8125 MB) at `MAX_CTX=256`.
7. **Compiled-Artifact Software Interpreter (`compiler/interpreter.py`):** Directly interprets the microprogram on packed ROM hex images and memory buffers, executing decode steps and recording trace tensors bit-identically.

---

## 2. Gate HB Checklist Verification

| Checklist Item | Status | Verification Evidence / File |
|---|---|---|
| **Graph IR invariants I1–I7 tested; round-trip lossless** | PASS | `tests/test_ir_v2.py` (9 tests passing with negative controls for I1, I2, I3, I5, I6, I7) |
| **Format assignment covers every edge; calibration ranges respected** | PASS | `compiler/format_assignment.py`, `tests/test_format_assignment.py` (all edges assigned, valid $M \ge 0, 0 \le s \le 62$) |
| **ROM packing: images, address map, sha256; padding tests green** | PASS | `compiler/rom_packing.py`, `tests/test_rom_packing.py` (`weights.hex`, `requant_params.hex`, `tables.hex`, `address_map.json`) |
| **Schedule/microprogram and memory plan generated for tiny and full model** | PASS | `compiler/schedule.py`, `compiler/memory_planner.py`, `results/part2/memory_report_tiny.json`, `results/part2/memory_report_full.json` |
| **Interpreter equals reference decoder bit-exactly** | PASS | `compiler/interpreter.py`, `tests/test_interpreter.py` (deterministic decode steps and full trace capture) |

---

## 3. Explain-It Questions for Gate HB

### Q1: Why is the sequencer microprogram so compact (< 25 instructions) for a 30-layer model?
**Answer:** Because the model architecture possesses structural regularity across layers. Rather than unrolling 30 separate copies of identical layer instructions in program memory, the sequencer uses a **parameterized layer loop**. The instruction specifies the operation type (e.g. `LINEAR(q_proj)`, `RMSNORM`, `ATTENTION`), buffer indices, and engine dimensions once. An internal layer loop counter indexes into the `layer_rom_bases` table to offset Weight ROM and Requant Parameter ROM pointers for each layer iteration. This reduces program ROM storage from thousands of words to tens of words.

### Q2: How does the ROM packing pass ensure zero write-hazard and data integrity?
**Answer:** All weights, lookup tables, and requantization parameters are packed into contiguous, read-only memory spaces with pre-computed offsets in `address_map.json`. The generated Verilog modules for ROMs expose only read ports (`clk, addr, data`) and load directly via `$readmemh` from `.hex` files with zero write enable signals (`we`) or write data ports. Every packed file is hashed with SHA256 during compiler generation, and reloaded with checksum verification.

### Q3: How is dynamic memory separated from immutable memory in the memory plan?
**Answer:**
- **Immutable Memory (ROM):** Holds weights (129.28 MB for full model), lookup tables (RoPE, rsqrt, exp2, reciprocal, SiLU), requantization constants $(M, s)$, and sequencer program instructions.
- **Dynamic Memory (RAM):** Exclusively holds activation buffers (`BUF_RES`, `BUF_ACT`, `BUF_Q`, `BUF_K`, `BUF_V`, `BUF_ATTN_OUT`, `BUF_MLP_GATE`, `BUF_MLP_UP`, `BUF_MLP_SWIGLU`, `BUF_LOGITS`) and the autoregressive KV cache (`[L][2][KV][MAX_CTX][HD]`). At `MAX_CTX=256`, the KV cache requires exactly 2.8125 MB of SRAM.

### Q4: Why is requantization dyadic $(M, s)$ rather than floating-point scaling?
**Answer:** Real-valued scale multipliers $S = (s_{in} \cdot s_w) / s_{out}$ cannot be evaluated in fixed-point RTL without costly floating-point units. Dyadic scaling expresses $S \approx M \cdot 2^{-s}$ where $M$ is a 32-bit positive integer and $s$ is an arithmetic right shift amount with round-half-up toward $+\infty$ (`rshift_rnd`). The compiler derives $(M, s)$ offline in float64, ensuring the RTL datapath only executes integer multiplication and arithmetic right shifts.

### Q5: What is the role of the Compiled-Artifact Interpreter in the verification ladder?
**Answer:** The Compiled-Artifact Interpreter bridges the high-level Python oracle (`numerics/decoder.py`) and the cycle-accurate RTL. While the oracle evaluates tensor math directly on Python arrays, the interpreter executes the exact binary microprogram instructions reading from the packed `.hex` ROM images that the Verilog simulator will load. Achieving bit-exact trace and token equality between the interpreter and the oracle guarantees that all compiler transformations (tiling, padding, format assignment, ROM packing, and sequencer scheduling) preserve numeric semantics with zero divergence.

---

## 4. Test Suite Execution
`make check2` execution:
```
============================= 101 passed in 4.46s ==============================
=== make check2: PASSED ===
```
Architectural layering check:
```
tests/test_layering.py::test_layering_in_codebase PASSED                 [ 33%]
tests/test_layering.py::test_layering_negative_control_import PASSED     [ 66%]
tests/test_layering.py::test_layering_negative_control_model_strings PASSED [100%]
============================== 3 passed in 0.15s ===============================
```
Ruff linting:
```
All checks passed!
```
