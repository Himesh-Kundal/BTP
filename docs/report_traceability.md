# LLMHW Whole-Project Numerical Traceability Table (Part I & Part II)

This table maps every single numerical claim, benchmark measurement, architectural parameter, and evaluation metric in the whole-project thesis report (`docs/whole_project_report.md` and `docs/report.md`) to its authoritative artifact source in `results/` or `numerics/`, the command/script that produced it, and the git commit in which it was generated.

Per project Golden Rule 1 and Gate HG checklist §7: **Zero unsourced numbers are permitted in the report. Every extrapolation or pre-layout projection is explicitly labeled as `[estimate]`.**

---

## Part I — Spatial Constant-Weight Linear Layer Compiler

| Item ID | Report Claim / Parameter | Value | Source File | Producing Command / Script | Generating Commit |
|---|---|---|---|---|---|
| **M-01** | SmolLM2-135M Transformer Layers | `30` | `results/model.json` | `python -m frontend.model_loader` | `775f407` |
| **M-02** | SmolLM2 Hidden Dimension ($d_{\text{model}}$) | `576` | `results/model.json` | `python -m frontend.model_loader` | `775f407` |
| **M-03** | SmolLM2 Intermediate Dimension ($d_{\text{ffn}}$) | `1536` | `results/model.json` | `python -m frontend.model_loader` | `775f407` |
| **M-04** | SmolLM2 Total Parameters (with tied vocab & norms) | `134,515,008` (~134.5M) | `docs/model_facts.md` | `python -m frontend.model_loader` | `bf06427` |
| **M-05** | Total Dense Linear Projections in 30 Layers | `210` ($30 \times 7$) | `results/model.json` | `python -m frontend.linear_extractor` | `a82dddb` |
| **M-06** | Per-Token 7-Projection Linear MACs (30 Layers) | `106,168,320` (~106.17M MACs) | `results/part2/part1_audit.md` | Analytical re-derivation in `part1_audit.md` §5.4 | `bf06427` |
| **M-07** | Per-Token Tied LM Head MACs ($49,152 \times 576$) | `28,311,552` (~28.31M MACs) | `results/part2/part1_audit.md` | Analytical re-derivation in `part1_audit.md` §5.4 | `bf06427` |
| **M-08** | Arithmetic Share of 7 Projections (Decode w/ LM Head vs Prefill) | `78.9%` of linear MACs (`74.1%` total decode); `>90%` prefill | `results/part2/part1_audit.md` | Analytical re-derivation in `part1_audit.md` §5.4 | `bf06427` |
| **N-01** | Quantized Weight & Activation Dynamic Range | $[-127, 127]$ (symmetric INT8) | `numerics/config_frozen.json` | `pytest tests/test_numerics_ops.py` | `a25c348` |
| **N-02** | Intermediate Product & Accumulator Bitwidths | `16` bits product, `32` bits accumulator | `rtl/lib/linear_engine.v` | `pytest tests/test_rtl_linear_engine.py` | `a25c348` |
| **C2-01** | Part I Linear Layer Cosine Similarity (Min / Mean / Max) | `0.9984` / `0.9996` / `0.9999` | `results/quant_quality.json` | `python -m compiler.quant_quality` | `4c7679a` |
| **C2-02** | Part I Linear Layer Relative Frobenius Error (Min / Mean / Max) | `2.10%` / `3.81%` / `5.60%` | `results/quant_quality.json` | `python -m compiler.quant_quality` | `4c7679a` |
| **V-01** | Part I Regression Suite Total Checks & Mismatches | `124` checks, `853` vectors, `0` mismatches | `results/regression.json` | `make regression` | `11a37b6` |
| **MUT-01** | Part I Mutation Suite (M1–M9) Kill Rate | `100.0%` (`9/9` killed) | `results/mutation_matrix.json` | `pytest tests/test_mutation_hardening.py` | `be622c6` |
| **E1-01** | Spatial Area: $(4 \times 16)$ Design F vs P Cells & Flops | `5,690` vs `21,399` cells (`3.76×`); `114` vs `641` DFFs (`5.62×`) | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-02** | Spatial Area: $(8 \times 16)$ Design F vs P Cells & Flops | `22,764` vs `85,283` cells (`3.75×`); `226` vs `1,293` DFFs (`5.72×`) | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-03** | Spatial Area: $(16 \times 16)$ Design F vs P Cells & Flops | `38,912` vs `149,768` cells (`3.85×`); `442` vs `2,561` DFFs (`5.79×`) | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-04** | Spatial Area: $(16 \times 64)$ Design F vs P Cells & Flops | `155,648` vs `599,085` cells (`3.85×`); `491` vs `8,705` DFFs (`17.73×`) | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-05** | Spatial Area: $(32 \times 64)$ Design F vs P Cells & Flops | `311,296` vs `1,198,170` cells (`3.85×`); `981` vs `17,409` DFFs (`17.75×`) | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-06** | FPGA $(8 \times 16)$ iCE40 LUT-4 Count Design F vs P | `1,602` vs `3,745` LUT4 (`2.34×`) | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E2-01** | Weight Distribution Effect on $(8 \times 16)$ Tile | Real `22,764` vs Random `28,006 ± 48` (`-23.0%`); Shuffled `22,810`; Zero `3` | `results/E2.csv` | `python -m synth.experiments` | `1a3d3c2` |
| **E3-01** | Critical Path Depth (`ltp`) Design F vs P | $(4,16)$: `41` vs `59` (`0.69`); $(8,16)$: `57` vs `77` (`0.74`); $(16,64)$: `89` vs `111` (`0.80`) | `results/E3.csv` | `python -m synth.experiments` | `02d2b17` |
| **E5-01** | Multi-Tile ($16 \times 32$) Direct vs Extrapolated Cells | `82,160` direct vs `91,056` estimate (`10.82%` error) | `results/E5.json` | `python -m synth.extrapolation` | `9dd8781` |
| **E5-02** | Layer 0 (3,456 tiles) Extrapolated Cells F vs P | `570,642,816` vs `2,075,123,088` cells (`3.64×`) [estimate] | `results/E5.json` | `python -m synth.extrapolation` | `9dd8781` |

---

## Part II — Full SmolLM2-135M Decoder Hardware Accelerator

| Item ID | Report Claim / Parameter | Value | Source File | Producing Command / Script | Generating Commit |
|---|---|---|---|---|---|
| **P2-M01** | SmolLM2 GQA Heads & Head Dimension | `9` Q heads, `3` KV heads (3:1 GQA), `head_dim = 64` | `numerics/config_frozen.json` | `python scripts/ue1_generate_full.py` | `a25c348` |
| **P2-M02** | SmolLM2 Vocabulary Size & RoPE Base $\theta$ | `vocab_size = 49,152`, `rope_theta = 100,000.0` | `numerics/config_frozen.json` | `python scripts/ue1_generate_full.py` | `a25c348` |
| **P2-M03** | Hardware Maximum Context Length (`MAX_CTX`) | `256` tokens | `numerics/config_frozen.json` | `python scripts/ue1_generate_full.py` | `a25c348` |
| **P2-N01** | Residual Stream & RMSNorm Accumulator Widths | `24`-bit signed residual (`[-8388608, 8388607]`), `48`-bit RMSNorm acc | `numerics/config_frozen.json` | `pytest tests/test_numerics_ops.py` | `a25c348` |
| **P2-N02** | RMSNorm `INV_D` Fixed-Point Constants ($d=576$) | `INV_D_M = 7282`, `INV_D_S = 22` ($7282 / 2^{22} \approx 1/576$) | `numerics/config_frozen.json` | `pytest tests/test_rtl_rmsnorm.py` | `a25c348` |
| **P2-N03** | Non-Linear Lookup Table Geometries | `rsqrt`: $256 \times 16\text{b}$; `exp2`: $256 \times 16\text{b}$; `silu`: $512 \times 8\text{b}$; `rope`: $256 \times 32 \times 16\text{b}$ Q1.14 | `numerics/config_frozen.json` | `pytest tests/test_numerics_ops.py` | `a25c348` |
| **UE1-01** | Full SmolLM2 Hardware Bundle Generation Time & RAM | `12.44` s wall time (`8.103` s extract/calib, `3.588` s emit), `2.557` GB peak RSS | `results/part2/ue1_generation.json` | `python scripts/ue1_generate_full.py` | `a25c348` |
| **UE1-02** | Full Bundle ROM Word Counts & Disk Size | `162,791,424` weight bytes (`10,174,464` 128b words), `288,990` requant words, `799.1` MB disk | `results/part2/ue1_generation.json` | `python scripts/ue1_generate_full.py` | `a25c348` |
| **UE2-01** | Full Top Verilator Build Time & Simulation Speed | `4.45` s build time, `2,415,740` cycles/s (`2.42` MHz) | `results/part2/ue2_verilator_build.json` | `python scripts/ue3_layer0_traces.py` | `a25c348` |
| **UE3-01** | Layer-0 Multi-Position Trace Verification | `12` positions, `216` files, `728,064` elements, `0` divergences (`165,511,344` cycles) | `results/part2/ue3_layer0_traces.json` | `python scripts/ue3_layer0_traces.py` | `a25c348` |
| **UE4-01** | Full 30-Layer Multi-Prompt Decode Verification | `10/10` prompts $\times 16$ new tokens (`285` decode steps) **100% token-exact** | `results/part2/full_decode.json` | `python scripts/ue4_full_decode.py` | `a25c348` |
| **UE4-02** | Sampled-Layer (`L0`, `L14`, `L29`) Trace Equality | `1,344` files, `2,279,424` elements checked, `0` divergences (`sampled_trace_identical: true`) | `results/part2/full_decode.json` | `python scripts/ue4_full_decode.py` | `a25c348` |
| **UE4-03** | Measured Simulation Cycles per Token | `14,100,262` mean cycles/token (`4,018,574,895` total cycles across `285` steps) | `results/part2/full_decode.json` | `python scripts/ue4_full_decode.py` | `a25c348` |
| **UE5-01** | Long-Context (`MAX_CTX = 256`) RTL Verification | `256` positions (`240` prompt + `16` gen), `4,669,315,072` cycles, `90` files / `303,360` elements **100% bit-exact** | `results/part2/ue5_long_context.json` | `python scripts/ue5_long_context.py` | `a25c348` |
| **UE6-01** | Host Demo Greedy Match Statistics vs FP32 (10 Prompts) | Mean greedy match length = `4.5` tokens, Median = `1.5`, Max = `16/16` (`p01`), Mean token agreement = `32.5%` | `results/part2/ue6_host_demo.json` | `python scripts/ue6_host_demo.py` | `a25c348` |
| **UE7-01** | WikiText-2 Quality Evaluation (10 Windows, 630 Tokens) | FP32 PPL = `89.6749` (CE `4.4962`), INT8 PPL = `107.0031` (CE `4.6729`), $\Delta \text{PPL} = \mathbf{+17.3282}$ ($\le 25.0$) | `results/part2/quality_final.json` | `python scripts/ue7_quality_final.py` | `a25c348` |
| **UE7-02** | WikiText-2 Top-1 ArgMax Agreement & Mean KL Divergence | Top-1 Agreement = **`70.16%`** ($\ge 35.0\%$), Mean $\text{KL}(P_{\text{fp32}} \parallel P_{\text{int}}) = \mathbf{0.2280}$ ($\le 0.25$) | `results/part2/quality_final.json` | `python scripts/ue7_quality_final.py` | `a25c348` |
| **MUT-02** | Part II Unit Mutation Suite (M10–M30) Kill Rate | `100.0%` (`21/21` mutants killed) | `results/mutation_matrix_v2.json` | `pytest tests/test_mutation_suite_v2.py` | `a25c348` |
| **MUT-03** | Part II Top-Level System Mutation Suite (TM1–TM7) Kill Rate | `100.0%` (`7/7` mutants killed) | `results/top_mutation_matrix.json` | `pytest tests/test_top_mutation.py` | `a25c348` |
| **EA1-01** | Full Part II Datapath Synthesis (All 12 Units, Generic Gates) | `78,510` total cells (`74,849` combinational, `3,661` DFFs) | `results/part2/EA1.csv` | `python scripts/uf1_synth_units.py` | `a25c348` |
| **EA1-02** | Full Part II Datapath Synthesis (All 12 Units, iCE40 Proxy) | `80,762` total cells (`31,908` LUT-4s, `3,660` DFFs) | `results/part2/EA1.csv` | `python scripts/uf1_synth_units.py` | `a25c348` |
| **EA1-03** | Baseline $8 \times 16$ `linear_engine.v` Synthesis | Generic: `21,387` cells (`503` DFFs, LTP `96`); iCE40: `8,376` cells (`7,260` LUT-4, `503` DFFs) | `results/part2/EA1.csv` | `python scripts/uf1_synth_units.py` | `a25c348` |
| **EA1-04** | Non-Linear Unit Synthesis (`rmsnorm`, `softmax`, `rope`, `attention`) | `rmsnorm`: `51,656` cells; `softmax`: `2,690` cells; `rope`: `892` cells; `attention`: `3,389` cells | `results/part2/EA1.csv` | `python scripts/uf1_synth_units.py` | `a25c348` |
| **EA2-01** | Analytical vs Simulated Cycle Model Validation ($8 \times 16$ Grid) | Analytical `13,691,648` vs Measured `14,100,262` cycles/token (**`-2.90%` relative error**) | `results/part2/EA2.json` | `python scripts/uf2_engine_grid.py` | `a25c348` |
| **EA2-02** | Engine Grid Scaling (`16` to `512` MAC Lanes) | `4x4` (`16` lanes): `2,477` cells, `89.38M` cyc; `8x16` (`128` lanes): `21,387` cells, `13.69M` cyc; `16x32` (`512` lanes): `81,969` cells, `6.13M` cyc | `results/part2/EA2.csv` | `python scripts/uf2_engine_grid.py` | `a25c348` |
| **EA3-01** | On-Chip Memory Accounting (`MAX_CTX = 256`) | Total ROM: `157.5435` MB (`165,196,496` B, **`98.15%`**); Total SRAM: `2.9718` MB (`3,116,160` B, **`1.85%`**) | `results/part2/EA3.json` | `python scripts/uf3_memory_accounting.py` | `a25c348` |
| **EA3-02** | ROM & SRAM Sub-Component Breakdown | Weights ROM: `155.25` MB; Requant ROM: `2.20` MB; KV Cache SRAM: `2.8125` MB (`2,949,120` B); Activation Buffers: `0.1593` MB (`167,040` B) | `results/part2/EA3.json` | `python scripts/uf3_memory_accounting.py` | `a25c348` |
| **EA4-01** | Spatial vs ROM-Sequential ($32 \times 32$ Block, $1,024$ Weights) | Spatial: `213,022` generic cells (`72,958` LUT4, `1` cycle) vs ROM-Sequential: `21,387` generic cells (`7,260` LUT4, `138` cycles) (**`9.96×` fewer gates**) | `results/part2/EA4.json` | `python scripts/uf4_spatial_vs_sequential.py` | `a25c348` |
| **EA4-02** | Full-Model Spatial Extrapolation ($134.5\text{M}$ Weights) | `27.98 Billion` generic gates ($\approx 19,698\,\text{mm}^2$ silicon) [estimate] — physically unfeasible on single die | `results/part2/EA4.json` | `python scripts/uf4_spatial_vs_sequential.py` | `a25c348` |
| **EA5-01** | TSMC 28nm HPC+ Pre-Layout Die Area ($8 \times 16$ Baseline) | Low: `115.65` mm², **Mid: `147.95` mm²**, High: `191.50` mm² [estimate] (ROM = `142.84` mm², SRAM = `3.68` mm², Logic = `0.06` mm²) | `results/part2/EA5.json` | `python scripts/uf5_asic_projection.py` | `a25c348` |
| **EA5-02** | TSMC 28nm HPC+ Projected Throughput ($8 \times 16$ Baseline) | Low (`250` MHz): `17.73` tok/s, **Mid (`400` MHz): `28.37` tok/s** (`35.25` ms/tok), High (`550` MHz): `39.01` tok/s [estimate] | `results/part2/EA5.json` | `python scripts/uf5_asic_projection.py` | `a25c348` |
| **EA5-03** | TSMC 28nm HPC+ Scaling at `512` MAC Lanes ($16 \times 32$) | Mid Die Area: `148.10` mm² (`+0.10%`), Mid Throughput (`400` MHz): **`63.37` tok/s** (`15.78` ms/tok) [estimate] | `results/part2/EA5.json` | `python scripts/uf5_asic_projection.py` | `a25c348` |
| **EA5-04** | Die Area Sensitivity Elasticities ($\epsilon$) | ROM density: `0.9655` (`96.55%`); Routing overhead: `0.0846`; SRAM density: `0.0249`; Logic density: `0.0010` [estimate] | `results/part2/EA5.json` | `python scripts/uf5_asic_projection.py` | `a25c348` |
| **UG2-01** | Turnkey Demo Runtimes (`make demo2-fast` & `make demo2`) | `make demo2-fast`: `11.45` s (`115,227` cycles); `make demo2`: `106.16` s (`221,848,192` cycles, `4/4` token match `' Berlin. The capital'`) | `results/part2/demo2_fast.json`, `demo2.json` | `make demo2-fast && make demo2` | `a25c348` |
| **UG3-01** | Clean-Directory Reproduction Verification | `env-check` (`1.95` s), `demo2-fast` (`13.11` s), `synth2` (`39.80` s), Total `54.86` s (`status: PASS`) | `results/part2/ug3_clean_repro.json` | `python scripts/ug3_clean_repro.py` | `a25c348` |
