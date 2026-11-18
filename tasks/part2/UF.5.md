# Task UF.5 — ASIC Projection and Sensitivity Analysis (EA5)

## Objective
Provide a pre-layout ASIC area, frequency, and throughput projection for SmolLM2-135M in TSMC 28nm HPC+ across low/mid/high parameter ranges with complete sources and sensitivity analysis. Explicitly label all figures as estimates. Emit `results/part2/EA5.json`.

## Implementation (`scripts/uf5_asic_projection.py`)
- Standardized analytical area projection equation:
  $$A_{\text{die}} = \text{overhead\_factor} \times \left( N_{\text{rom\_bits}} \cdot a_{\text{rom}} + N_{\text{sram\_bits}} \cdot a_{\text{sram}} + N_{\text{gates}} \cdot a_{\text{gate}} \right)$$
- Pinned literature assumptions for TSMC 28nm HPC+ from memory compiler datasheets, standard cell libraries, and ISSCC/IEDM surveys:
  - $a_{\text{rom}} \in [0.05, 0.08, 0.12] \;\mu\text{m}^2/\text{bit}$ (via/diffusion mask ROM).
  - $a_{\text{sram}} \in [0.12, 0.15, 0.22] \;\mu\text{m}^2/\text{bit}$ (high-density 6T dual-port macro).
  - $a_{\text{gate}} \in [0.55, 0.70, 0.90] \;\mu\text{m}^2/\text{GE}$ (9-track standard cell library).
  - $\text{overhead} \in [1.20, 1.35, 1.50]$ (floorplan utilization factor).
  - $\text{freq} \in [200, 400, 600] \;\text{MHz}$.
- Evaluated scaling across grid configurations (16 to 512 MAC lanes).
- Performed sensitivity analysis with $\pm 10\%$ parameter perturbation to extract area elasticity:
  $$\text{Elasticity} = \frac{\Delta A / A}{\Delta P / P}$$

## Results (`results/part2/EA5.json`)
- **Baseline (128 MAC lanes, 8x16 grid):**
  - Raw core area: $109.59 \;\text{mm}^2$ (Mid). Breakdown: **96.5% ROM** ($105.73 \;\text{mm}^2$), **3.4% SRAM** ($3.73 \;\text{mm}^2$), **0.12% Logic** ($0.134 \;\text{mm}^2$).
  - Total die area estimate: $83.00 \;\text{mm}^2$ (Low), $147.95 \;\text{mm}^2$ (Mid), $246.35 \;\text{mm}^2$ (High).
  - Throughput: $14.18 \;\text{tok/s}$ (@ 200 MHz), $28.37 \;\text{tok/s}$ (@ 400 MHz), $42.55 \;\text{tok/s}$ (@ 600 MHz).
- **Lanes Scaling (@ 400 MHz mid-frequency):**
  - 16 lanes: $5.85 \;\text{tok/s}$, $147.94 \;\text{mm}^2$.
  - 32 lanes: $7.76 \;\text{tok/s}$, $147.94 \;\text{mm}^2$.
  - 64 lanes: $15.21 \;\text{tok/s}$, $147.94 \;\text{mm}^2$.
  - 128 lanes (baseline): $28.37 \;\text{tok/s}$, $147.95 \;\text{mm}^2$.
  - 256 lanes: $34.59 \;\text{tok/s}$, $147.95 \;\text{mm}^2$.
  - 512 lanes: $63.37 \;\text{tok/s}$, $147.96 \;\text{mm}^2$.
- **Sensitivity & Elasticity:**
  - $a_{\text{rom}}$ elasticity: **0.965** (High sensitivity: a 10% change in ROM bitcell area shifts total die area by 9.65%).
  - Floorplan overhead elasticity: **1.000** (High sensitivity).
  - $a_{\text{sram}}$ elasticity: **0.034** (Low sensitivity).
  - $a_{\text{gate}}$ elasticity: **0.001** (Negligible sensitivity: expanding MAC lanes by 32x increases die area by $<0.02\%$).

## Status
Status: complete
