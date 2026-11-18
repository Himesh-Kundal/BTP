# Task UE.2 — Verilator Build & Benchmark of Full SmolLM2-135M Top

## Objective
Compile the full SmolLM2-135M `top_accel.v` with Verilator (`--binary --timing -j 0`) and execute a 2-step decode simulation, recording build time, peak build/sim RAM, binary size, exact cycle count, and simulation throughput (cycles/s).

## Bugs Fixed During Bring-Up
1. **SwiGLU LUT Size (`rtl/lib/top_accel.v`):** Updated `.LUT_SIZE(256)` to `.LUT_SIZE(512)` on `swiglu_inst` to match the 512-entry `rom_silu_inst`.
2. **RMSNorm rsqrt LUT 16-bit Packing (`compiler/rom_packing.py`):** Clamped `rsqrt_lut[0]` (`65536` = $1.0$ in Q0.16) to `65535` (`0xFFFF`) so 16-bit ROM packing never wraps to `0x0000`.
3. **Interpreter Alignment (`compiler/interpreter.py`):**
   - Vectorized 8-bit `weights.hex` loading for 162.8M-element ROMs.
   - Fixed RMSNorm per-channel shift `s`, SiLU LUT 8-bit sign extension, and 24-bit residual bounds for `o_proj`/`down_proj`.
4. **Cycle Instrumentation (`sim/tb_generator.py`):** Added `TOTAL_CYCLES` reporting in `tb_top_accel.v` and `model_config` propagation.

## Verification & Measurements (`results/part2/ue2_verilator_build.json`)
- **Build time:** `3.187s` (Verilator `--binary --timing -j 0`, peak RSS `0.261 GB`, binary `0.20 MB`)
- **2-step Simulation (`prompt=[1], num_generate=1`):**
  - `tokens_out`: `[1029, 11121]`
  - `total_cycles`: `27,220,724` (`13,610,362` cycles/token)
  - `sim_wall_time_sec`: `32.987s` (peak RSS `0.392 GB`)
  - `sim_speed`: **`825,197.1 cycles/s`** (`825.2 kcycles/s`)

## Status
Status: complete
Commit: UE.2: Verilator build and benchmark of full SmolLM2-135M top
