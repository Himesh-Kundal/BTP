# Task UF.2 — Engine Grid Design Exploration (EA2)

## Objective
Explore the parameterized matrix engine grid across $(R, C) \in \{(4,4), (8,4), (8,8), (8,16), (16,16), (16,32)\}$ (16 to 512 MAC lanes), measure synthesis metrics (cells, LUT4, flip-flops, critical path depth), compute analytical cycle estimates per token, and validate against the measured UE.4 simulation baseline (`14,100,262` cycles/token).

## Implementation (`scripts/uf2_engine_grid.py`)
- Synthesized `linear_engine` across all 6 grid configurations for both `generic` gates and `ice40` FPGA LUT-4 cell mapping.
- Formulated an exact analytical cycle model accounting for tile streaming, fill/drain latency, requantization, and non-linear op execution.
- Emitted `results/part2/EA2.csv` and `results/part2/EA2.json`.

## Results
- **Baseline Configuration (8, 16) [128 MAC lanes]:**
  - Generic: `21,387` cells (`691` flip-flops), critical path depth `ltp = 587`.
  - iCE40: `10,597` cells (`8,961` LUT4, `692` flip-flops), critical path depth `ltp = 531`.
  - Analytical cycles/token: `13,691,648`.
  - Actual measured UE.4 simulated cycles/token: `14,100,262`.
  - Model error: **-2.90%** (validated within 3%).
- **Scaling Range:**
  - 16 lanes: `68,375,264` cycles/token (`13,070` generic cells / `5,306` LUT4).
  - 512 lanes: `6,311,696` cycles/token (`31,306` generic cells / `12,447` LUT4).

## Status
Status: complete
