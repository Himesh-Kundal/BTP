# Task UF.1 — Unit and Engine Logic Synthesis Study (Yosys)

## Objective
Synthesize each parameterized unit and the linear engine using Yosys for both standard generic gates and iCE40 FPGA LUT-4 cell mapping proxy. Extract cell breakdown, flip-flops, LUTs, and longest topological path (ltp) depth into `results/part2/EA1.csv` and `results/part2/EA1.json`.

## Implementation (`scripts/uf1_synth_units.py`)
- Automated synthesis across 12 RTL modules:
  1. `requant` (`rtl/lib/requant.v`)
  2. `residual_add` (`rtl/lib/residual_add.v`)
  3. `argmax` (`rtl/lib/argmax.v`)
  4. `kv_addr_gen` (`rtl/lib/kv_addr_gen.v`)
  5. `embed_gather` (`rtl/lib/embed_gather.v`)
  6. `rope` (`rtl/lib/rope.v`)
  7. `swiglu` (`rtl/lib/swiglu.v`)
  8. `sequencer` (`rtl/lib/sequencer.v`)
  9. `linear_engine` (`rtl/lib/linear_engine.v`)
  10. `rmsnorm` (`rtl/lib/rmsnorm.v`)
  11. `softmax` (`rtl/lib/softmax.v`)
  12. `attention` (`rtl/lib/attention.v`)
- Evaluated both `generic` and `ice40` targets using `synth.yosys_runner`.

## Results (`results/part2/EA1.csv`, `results/part2/EA1.json`)
- All 12 units synthesized cleanly (24 total synthesis runs in 606.24s).
- Flops, LUT-4 counts, wire bits, and topological critical path depths recorded.

## Status
Status: complete
