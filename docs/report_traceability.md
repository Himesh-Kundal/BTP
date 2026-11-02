# LLMHW Report Numerical Traceability Table

This table maps every single numerical claim, benchmark measurement, architectural parameter, and evaluation metric in the thesis report (`docs/report.md`) to its authoritative artifact source in `results/` or `artifacts/`, the command that produced it, and the git commit in which it was generated.

Per project Golden Rule 1 and Gate G8 checklist §7: **Zero unsourced numbers are permitted in the report. Any extrapolation is explicitly labeled as an estimate.**

---

| Item ID | Report Claim / Parameter | Value | Source File | Producing Command / Script | Generating Commit |
|---|---|---|---|---|---|
| **M-01** | SmolLM2-135M Transformer Layers | `30` | `results/model.json` | `python -m frontend.model_loader` | `775f407` |
| **M-02** | SmolLM2 Hidden Dimension ($d_{\text{model}}$) | `576` | `results/model.json` | `python -m frontend.model_loader` | `775f407` |
| **M-03** | SmolLM2 Intermediate Dimension ($d_{\text{ffn}}$) | `1536` | `results/model.json` | `python -m frontend.model_loader` | `775f407` |
| **M-04** | SmolLM2 Total Parameters | `134,516,352` (~134.5M) | `results/model.json` | `python -m frontend.model_loader` | `775f407` |
| **M-05** | Total Dense Linear Projections in Model | `210` ($30 \times 7$) | `results/model.json` | `python -m frontend.linear_extractor` | `a82dddb` |
| **M-06** | Total Projections in Layer 0 | `7` | `results/model.json` | `python -m frontend.linear_extractor` | `a82dddb` |
| **N-01** | Quantized Weight Dynamic Range | $[-127, 127]$ | `compiler/quantization.py` | `pytest tests/test_quantization.py` | `2f20b12` |
| **N-02** | Quantized Activation Range | $[-127, 127]$ | `compiler/calibration.py` | `pytest tests/test_calibration.py` | `1cb6000` |
| **N-03** | Intermediate Product Bitwidth | `16` bits signed | `backend/verilog_codegen.py` | `pytest tests/test_verilog_codegen.py` | `8745851` |
| **N-04** | Accumulator Bitwidth | `32` bits signed | `compiler/accumulator.py` | `pytest tests/test_accumulator.py` | `3ebee29` |
| **N-05** | Pipeline Latency ($L$) | `1` cycle | `backend/module_generator.py` | `pytest tests/test_module_generator.py` | `c252d12` |
| **C2-01** | Minimum Cosine Similarity (Layer 0–29) | `0.9984` | `results/quant_quality.json` | `python -m compiler.quant_quality` | `4c7679a` |
| **C2-02** | Mean Cosine Similarity (Layer 0–29) | `0.9996` | `results/quant_quality.json` | `python -m compiler.quant_quality` | `4c7679a` |
| **C2-03** | Maximum Relative Frobenius Error | `0.0560` (5.60%) | `results/quant_quality.json` | `python -m compiler.quant_quality` | `4c7679a` |
| **C2-04** | Mean Relative Frobenius Error | `0.0381` (3.81%) | `results/quant_quality.json` | `python -m compiler.quant_quality` | `4c7679a` |
| **C2-05** | Frozen Cosine Quality Threshold (ADR-0001) | $\ge 0.990$ | `docs/adr/ADR-0001-quantization-thresholds.md` | Formal frozen policy | `4c7679a` |
| **C2-06** | Frozen Relative Error Threshold (ADR-0001) | $\le 0.080$ (8.0%) | `docs/adr/ADR-0001-quantization-thresholds.md` | Formal frozen policy | `4c7679a` |
| **V-01** | Level L1 Test Matrix Combinations | `1,080` conditions | `verification/test_matrix.py` | `pytest tests/test_test_matrix.py` | `bef2695` |
| **V-02** | Level L2 Real Layer Slices Checked | `21` configs ($7 \times 3$) | `results/regression.json` | `make regression` | `11a37b6` |
| **V-03** | Level L2 Mismatches against Reference | `0` (100% bit-exact) | `results/regression.json` | `make regression` | `11a37b6` |
| **V-04** | Level L3 Multi-Tile Mismatches | `0` (100% bit-exact) | `results/regression.json` | `make regression` | `11a37b6` |
| **V-05** | Regression Suite Execution Runtime | `22.2` seconds | `results/regression.json` | `make regression` | `11a37b6` |
| **V-06** | Regression Total Checks | `124` checks | `results/regression.json` | `make regression` | `11a37b6` |
| **MUT-01** | Mutation Testing Total Mutants (M1–M9) | `9` mutants | `results/mutation_matrix.json` | `pytest tests/test_mutation_hardening.py` | `be622c6` |
| **MUT-02** | Mutation Testing Kill Rate | `100.0%` (9/9 killed) | `results/mutation_matrix.json` | `pytest tests/test_mutation_hardening.py` | `be622c6` |
| **I3-01** | Design F Runtime Weight Input Ports | `0` bits | `verification/weight_audit.py` | `pytest tests/test_weight_audit.py` | `9971c80` |
| **I3-02** | Design F Internal Weight Signals | `0` signals | `verification/weight_audit.py` | `pytest tests/test_weight_audit.py` | `9971c80` |
| **I3-03** | Design F Writable Memory Arrays | `0` arrays | `verification/weight_audit.py` | `pytest tests/test_weight_audit.py` | `9971c80` |
| **E1-01** | Area: (4, 16) Design F vs P Cells | `5,690` vs `21,399` | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-02** | Area: (4, 16) Cell Reduction Ratio | `3.76×` ($0.266$) | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-03** | Area: (8, 16) Design F vs P Cells | `22,764` vs `85,283` | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-04** | Area: (8, 16) Cell Reduction Ratio | `3.75×` ($0.267$) | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-05** | Area: (16, 16) Design F vs P Cells | `38,912` vs `149,768` | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-06** | Area: (16, 16) Cell Reduction Ratio | `3.85×` ($0.260$) | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-07** | Area: (16, 64) Design F vs P Cells | `155,648` vs `599,085` | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-08** | Area: (16, 64) Cell Reduction Ratio | `3.85×` ($0.260$) | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-09** | Area: (32, 64) Design F vs P Cells | `311,296` vs `1,198,170` | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-10** | Area: (32, 64) Cell Reduction Ratio | `3.85×` ($0.260$) | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-11** | Storage: (8, 16) Sequential Flops F vs P | `226` vs `1,293` | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-12** | Storage: (8, 16) Flop Reduction Ratio | `5.72×` | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-13** | Storage: (16, 64) Sequential Flops F vs P | `491` vs `8,705` | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-14** | Storage: (16, 64) Flop Reduction Ratio | `17.73×` | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-15** | FPGA: (8, 16) iCE40 LUT4 Count F vs P | `1,602` vs `3,745` | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E1-16** | FPGA: (8, 16) LUT4 Reduction Ratio | `2.34×` | `results/E1.csv` | `python -m synth.experiments` | `02d2b17` |
| **E2-01** | Weight Effect: Real SmolLM2 Weights Cells | `22,764` | `results/E2.csv` | `python -m synth.experiments` | `1a3d3c2` |
| **E2-02** | Weight Effect: Uniform Random INT8 Mean Cells | `28,006` $\pm 48$ | `results/E2.csv` | `python -m synth.experiments` | `1a3d3c2` |
| **E2-03** | Weight Effect: Real vs Random Difference | `-23.0%` (Real is smaller) | `results/E2.csv` | `python -m synth.experiments` | `1a3d3c2` |
| **E2-04** | Weight Effect: Shuffled Real Weights Cells | `22,810` ($<0.3\%$ var) | `results/E2.csv` | `python -m synth.experiments` | `1a3d3c2` |
| **E2-05** | Weight Effect: All-Zero Sparse Weight Cells | `3` (99.99% pruned) | `results/E2.csv` | `python -m synth.experiments` | `1a3d3c2` |
| **E3-01** | Depth: (4, 16) LTP Depth F vs P | `41` vs `59` | `results/E3.csv` | `python -m synth.experiments` | `02d2b17` |
| **E3-02** | Depth: (4, 16) LTP Ratio | `0.69` (31% reduction) | `results/E3.csv` | `python -m synth.experiments` | `02d2b17` |
| **E3-03** | Depth: (8, 16) LTP Depth F vs P | `57` vs `77` | `results/E3.csv` | `python -m synth.experiments` | `02d2b17` |
| **E3-04** | Depth: (8, 16) LTP Ratio | `0.74` (26% reduction) | `results/E3.csv` | `python -m synth.experiments` | `02d2b17` |
| **E3-05** | Depth: (16, 64) LTP Depth F vs P | `89` vs `111` | `results/E3.csv` | `python -m synth.experiments` | `02d2b17` |
| **E3-06** | Depth: (16, 64) LTP Ratio | `0.80` (20% reduction) | `results/E3.csv` | `python -m synth.experiments` | `02d2b17` |
| **E5-01** | Extrapolation Validation Point Multi-Tile Shape | $16 \times 32$ (4 tiles of $8 \times 16$) | `results/E5.json` | `python -m synth.extrapolation` | `9dd8781` |
| **E5-02** | Extrapolation Direct Synthesis Cells (Design F) | `82,160` | `results/E5.json` | `python -m synth.extrapolation` | `9dd8781` |
| **E5-03** | Extrapolation Tile-Sum Estimate Cells (Design F) | `91,056` [estimate] | `results/E5.json` | `python -m synth.extrapolation` | `9dd8781` |
| **E5-04** | Extrapolation Relative Error ($\epsilon_{\text{rel}}$) | `10.82%` (overestimate) | `results/E5.json` | `python -m synth.extrapolation` | `9dd8781` |
| **E5-05** | Total Tiles in SmolLM2 Layer 0 | `3,456` tiles of $8 \times 16$ | `results/E5.json` | `python -m synth.extrapolation` | `9dd8781` |
| **E5-06** | Extrapolated Full Layer 0 Design F Cells | `570,642,816` [estimate] | `results/E5.json` | `python -m synth.extrapolation` | `9dd8781` |
| **E5-07** | Extrapolated Full Layer 0 Design P Cells | `2,075,123,088` [estimate] | `results/E5.json` | `python -m synth.extrapolation` | `9dd8781` |
| **E5-08** | Extrapolated Layer 0 Overall Cell Ratio (F/P) | `0.275` (3.64× reduction) [estimate] | `results/E5.json` | `python -m synth.extrapolation` | `9dd8781` |
| **D-01** | Full Layer ($576 \times 576$) Generation Time | `0.54` seconds | `results/demo.json` | `python scripts/run_varying_shape_demo.py` | `17317e9` |
| **D-02** | Full Layer ($576 \times 576$) Verilog Lines | `548,223` lines | `results/demo.json` | `python scripts/run_varying_shape_demo.py` | `17317e9` |
| **D-03** | Full Layer ($576 \times 576$) Verilog File Size | `30.02` MB | `results/demo.json` | `python scripts/run_varying_shape_demo.py` | `17317e9` |
| **D-04** | Full Layer Constant Audit Total Weights | `331,776` weights | `results/demo.json` | `python scripts/run_varying_shape_demo.py` | `17317e9` |
| **D-05** | Full Layer Constant Audit Execution Time | `0.85` seconds | `results/demo.json` | `python scripts/run_varying_shape_demo.py` | `17317e9` |
| **D-06** | Multi-Layer Flagship `make demo` Runtime | `3.58` seconds | `results/demo_layer0.json` | `make demo` | `7911a1c` |
| **D-07** | Fresh Clone `make demo` Runtime | `9.23` seconds | `tasks/T8.4.md` | `make demo` in clean checkout | `4c46e53` |
| **T-01** | Total Automated Regression Test Cases | `226` tests | `tests/` | `make check` | `dec4a51` |
| **T-02** | Complete Test Suite Runtime | `142.2` seconds | `state/LOG.md` | `make check` | `dec4a51` |
