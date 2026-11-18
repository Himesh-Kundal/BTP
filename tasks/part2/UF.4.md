# Task UF.4 — Spatial (Part I) vs ROM-Sequential (Part II) Comparative Study (EA4)

## Objective
Evaluate a flat unrolled constant-weight spatial datapath (Part I Design F) against the time-multiplexed ROM-sequential linear engine (Part II) for an identical linear block at a synthesizable dimension ($32 \times 32 = 1,024$ INT8 weights). Measure cell counts, flip-flops, LUT4s, latency (cycles), critical path depth (`ltp`), and throughput. Extrapolate to the full SmolLM2-135M model ($134.48\text{M}$ weights) with explicitly stated limits of extrapolation. Emits `results/part2/EA4.csv` and `results/part2/EA4.json`.

## Implementation (`scripts/uf4_spatial_vs_sequential.py`)
- Synthesized the flat spatial constant-coefficient linear module in Yosys 0.52 for both standard generic gates and iCE40 FPGA LUT-4 cell mapping proxy.
- Modeled the identical $32 \times 32$ computation on the parameterized $8 \times 16$ systolic linear engine (`linear_engine.v`).
- Computed cycles per block:
  $$T_n = \lceil 32/8 \rceil = 4, \quad T_k = \lceil 32/16 \rceil = 2 \implies \text{Cycles} = 4 \times (2 \times 12 + 10) + 2 = 138 \text{ cycles}$$
- Extrapolated spatial gate and LUT scaling to full-model scale ($134.48\text{M}$ weights) and contrasted against measured ROM-sequential full-chip logic.

## Results (`results/part2/EA4.csv`, `results/part2/EA4.json`)
- **$32 \times 32$ Block Comparison ($1,024$ weights):**
  - **Spatial (Part I):**
    - Generic: **213,022 cells** (212,020 combinational, 993 flip-flops, `ltp = 66`). $208.03\text{ gates/weight}$.
    - iCE40: **78,166 cells** (72,958 LUT4, 1,025 flip-flops, `ltp = 47`). $76.33\text{ cells/weight}$.
    - Latency: **1 cycle**. Throughput: **$100\text{M}$ inferences/sec** (@ 100 MHz).
  - **ROM-Sequential (Part II):**
    - Generic: **21,387 cells** (691 flip-flops, `ltp = 587`). $20.89\text{ gates/weight}$.
    - iCE40: **10,597 cells** (8,961 LUT4, 692 flip-flops, `ltp = 531`). $10.35\text{ cells/weight}$.
    - Latency: **138 cycles**. Throughput: **$724,638$ inferences/sec** (@ 100 MHz).
  - **Block Takeaway:** Spatial datapath requires **10.0× more generic gates** and **7.4× more FPGA cells** than the ROM-sequential engine for just a $32 \times 32$ matrix slice.

- **Full-Model Extrapolation (SmolLM2-135M, $134.48\text{M}$ weights):**
  - **Spatial (extrapolated):**
    - Generic gates: **$27.98\text{ Billion gates}$** ($>19,500\,\text{mm}^2$ in 28nm, $>23$ reticle limits).
    - iCE40 LUT4: **$9.58\text{ Billion LUT4}$**.
  - **ROM-Sequential (measured):**
    - Logic gates: **$191,387\text{ gates}$** ($0.134\,\text{mm}^2$ in 28nm).
    - Mask ROM area: $105.73\,\text{mm}^2$ (Mid). Total die area: $147.95\,\text{mm}^2$.
    - Latency: $14,100,262$ cycles/token ($28.37\text{ tok/s}$ @ 400 MHz).

- **Limits of Extrapolation:**
  Extrapolation from $32 \times 32$ spatial blocks assumes linear scaling of constant multiplier trees without placement/routing congestion. In physical silicon, routing 134.48M unrolled weights would incur catastrophic interconnect congestion, clock skew, and timing failure. The ROM-sequential engine avoids this entirely by decoupling arithmetic from storage.

## Status
Status: complete
