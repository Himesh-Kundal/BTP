# Task UF.6 — Context Comparison with Published Hardwired-LLM Claims

## Objective
Author a rigorous context comparison section placing our hardwired SmolLM2-135M ROM accelerator alongside published industrial and academic specialized LLM architectures (Groq LPU, Etched Sohu, Tenstorrent Wormhole/Blackhole, SambaNova SN40L, Apple Neural Engine). Explicitly distinguish vendor claims from independently measured data and state architectural trade-offs.

## Implementation (`docs/part2/context_comparison.md`)
- Compiled comparative table across:
  - Model target & precision
  - Technology node
  - Weight memory hierarchy (Mask ROM vs SRAM vs HBM vs Unified DRAM)
  - Weight mutability
  - Throughput (tokens/second)
  - Primary source & measurement status
  - Architectural caveats & limitations
- Analyzed technical contrasts in:
  1. Memory density and energy per token: Mask ROM ($0.08\,\mu\text{m}^2/\text{bit}$, 0 external memory energy) vs high-power SRAM pods (Groq) vs DRAM/HBM bus traffic (Apple/Etched).
  2. Control simplicity: Compile-time cycle-exact static microcode sequencer vs dynamic scheduling and dynamic NoC routing.
  3. Arithmetic tailoring: Dedicated non-linear piecewise arithmetic and frozen dyadic requantization vs programmable SIMD/vector engines.
- Addressed limitations: Model immutability (silicon mask changes required for weight updates), $M=256$ context length sizing, and pre-layout estimation status.

## Status
Status: complete
