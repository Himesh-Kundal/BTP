# LLMHW Viva Preparation & Defense Rehearsal Guide

This guide prepares the candidate for comprehensive questioning by the BTP evaluation committee. It addresses all mandatory "explain-it" criteria from `docs/07_checklists.md` §8 with exact technical justifications, file references, and architectural decisions.

---

## Question 1: Explain Every Pipeline Stage and Its File Handoff

### **Examiner:** "Walk us through the entire compilation pipeline from raw PyTorch model to synthesizable Verilog. What files are passed between stages?"

### **Answer:**
The LLMHW compiler operates across 5 decoupled stages with strict file handoffs:

1. **Frontend Stage (`frontend/`)**:
   - `model_loader.py` loads `HuggingFaceTB/SmolLM2-135M` at pinned commit `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` in evaluation mode.
   - `linear_extractor.py` scans the model graph, categorizing all 210 projections into the 7 canonical types (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - `activation_capture.py` uses forward hooks on fixed prompts with fixed seed 42 to capture input activations.
   - **File Handoff**: Dumps `artifacts/<model>@<rev>/layer_<idx>/<proj>/` containing:
     * `weight.npy`: Raw FP32/BF16 weights.
     * `meta.json`: Module dimensions, paths, and SHA256 checksums.
     * `calib/x_sample.npy`: Captured activation tensor.

2. **Intermediate Representation Stage (`ir/`)**:
   - `LinearOp` and `TensorRef` ingest raw NumPy arrays.
   - Invariants **I1** (frozen dimensions), **I2** (datatype validation), and **I5** (lossless serialization) are enforced.
   - **File Handoff**: Memory objects or serialized JSON ASTs.

3. **Compiler Stage (`compiler/`)**:
   - `calibration.py`: Computes static activation scale $s_x = \text{absmax}(X) / 127$ (Rule **N3**).
   - `quantization.py`: Computes per-output-channel weight scales $s_w[n] = \max_k |W[n,k]| / 127$ and quantizes $W$ into $[-127, 127]$ with round-half-to-even (Rule **N2**).
   - `accumulator.py`: Performs analytical bound check $K \times 127 \times 127 + |b_q|_{\max} < 2^{31}$ (Rule **N4**).
   - Specialization pass: Marks weights as `TensorKind.CONSTANT`.
   - **File Handoff**: Fully specialized quantized `LinearOp`.

4. **Backend Code Generation (`backend/`)**:
   - `tiling.py`: Divides matrix into $N_t \times K_t$ tiles ($8 \times 16$), pads boundaries with zeros, and produces `TileManifest` with manifest SHA256.
   - `weight_encoder.py`: Emits Verilog literals (`8'sd...`), prunes zero-product terms (zero-skipping), and replaces $\pm 1$ with wire assignments.
   - `verilog_codegen.py`: Synthesizes combinational tile cores.
   - `module_generator.py`: Generates column reduction adder trees, bias adders, and the synchronous $L=1$ registered wrapper.
   - **File Handoff**: `build/rtl/llmhw_<name>.v` and `llmhw_<name>.manifest.json`.

5. **Verification & Evaluation Stage (`verification/` & `synth/`)**:
   - `verification/reference.py`: Computes golden reference vectors in INT64.
   - `verification/simulator.py`: Executes Icarus Verilog (`vvp`) or Verilator, enforcing `TB_RESULT CHECKED=n MISMATCH=0`.
   - `synth/yosys_runner.py`: Executes automated synthesis script and parses `stat -json` and `ltp`.

---

## Question 2: Explain the Numeric Contract and Why Each Choice Was Made

### **Examiner:** "Why did you choose symmetric INT8 quantization over asymmetric? Why is the range $[-127, 127]$ instead of $[-128, 127]$? How do you prevent accumulator overflow?"

### **Answer:**

1. **Symmetric vs Asymmetric Quantization (Rule N2, N3)**:
   - In asymmetric quantization, a non-zero integer zero-point $z$ is required: $X_q = \text{round}(X/s) + z$.
   - Matrix multiplication becomes:
     $$Y = \sum_k (X_{q,k} - z_x)(W_{q,k} - z_w) = \sum_k X_{q,k} W_{q,k} - z_w \sum_k X_{q,k} - z_x \sum_k W_{q,k} + K z_x z_w$$
   - This introduces dynamic zero-point correction offsets that require runtime accumulators and extra multipliers.
   - **Design Decision**: Symmetric quantization sets $z_x = 0$ and $z_w = 0$, completely eliminating all offset correction hardware.

2. **Exclusion of $-128$ (Range $[-127, 127]$, Rule N2, ADR-0002)**:
   - In two's complement 8-bit arithmetic, the range is $[-128, 127]$.
   - Negating $-128$ (`-(-128)`) produces an arithmetic overflow because $+128$ cannot be represented in 8 signed bits.
   - In constant-coefficient hardware, negative weights require negating terms: $x \times (-c) = -(x \times c)$. If $c = -128$, negation overflows.
   - **Design Decision**: Clamping weights and activations strictly to $[-127, 127]$ guarantees that every number can be negated without bit extension, preventing boundary overflow.

3. **Accumulator Bounding & Bitwidth (Rule N4, N8)**:
   - Products are $8 \times 8 \to 16$ bits signed ($[-127, 127] \times [-127, 127] \in [-16129, 16129]$).
   - In a matrix row of length $K$, maximum possible accumulation is $K \times 127 \times 127$.
   - For SmolLM2 ($K \le 1536$), maximum accumulation is $1536 \times 16129 = 24,774,144$.
   - A 32-bit signed integer represents values up to $2^{31} - 1 = 2,147,483,647$.
   - The headroom factor is $\frac{2.14 \times 10^9}{2.47 \times 10^7} \approx 86.6\times$.
   - **Design Decision**: A 32-bit signed accumulator provably guarantees **zero** possibility of arithmetic overflow across all transformer operations.

---

## Question 3: Why Can Your Verification Results Be Trusted?

### **Examiner:** "How do we know your testbenches aren't circular? What prevents the generated hardware and testbench from sharing the same bug?"

### **Answer:**
Our verification methodology enforces project **Golden Rule 2**:

1. **Independent Oracle**:
   - `verification/reference.py` is written purely in NumPy using 64-bit signed integers.
   - Expected values are **never** computed from generated Verilog, from the compiler backend, or inside the testbench itself.
   - Vector serialization writes static `x.hex` and `y_exp.hex` text files before simulation begins.

2. **The "CHECKED Rule" (docs/02 §7)**:
   - Every testbench must print a single summary line: `TB_RESULT CHECKED=<n> MISMATCH=<m> TIMING_ERR=<t>`.
   - The simulator runner asserts $n > 0$, $n == \text{expected}$, $m == 0$, and $t == 0$.
   - If a testbench crashes early, hangs, or checks 0 vectors, it is automatically marked as **FAIL**.

3. **Mutation Testing Kill Rate (100.0%, M1–M9)**:
   - We subjected our test harness to 9 deliberate structural mutations in generated RTL:
     * Sign flips (`8'sd` $\to$ `-8'sd`)
     * Weight perturbations ($w \pm 1$)
     * Dropped terms in reduction adder trees
     * Bit-slice off-by-one errors
     * Accumulator truncation
   - **Result**: **100.0% of mutants were killed** (`results/mutation_matrix.json`). If the testbench were loose or circular, mutants would survive.

4. **Static Interface and Weight Audits (`verification/weight_audit.py`)**:
   - AST regex parsing parses every constant literal from the generated Verilog source code.
   - It reconstructs the 2D array and asserts a bit-exact match against $W_q$.
   - Injected faults (perturbing a single literal, changing a sign) are detected and fail the audit immediately.

5. **Cross-Simulator Verification**:
   - Simulations run under both **Icarus Verilog** and **Verilator**.
   - Outputs match bit-for-bit across completely independent simulation engines.

---

## Question 4: Explain the Fairness Protocol and What the Results Do NOT Show

### **Examiner:** "How did you ensure the comparison between Design F and Design P was fair? And what are the limitations of your synthesis metrics?"

### **Answer:**

1. **Fairness Protocol (docs/06 §B2, Rule N10)**:
   - **Identical Bitwidths**: Input $x$ is 8 bits, weight $w$ is 8 bits, intermediate product is 16 bits, accumulator is 32 bits in both designs.
   - **Identical Clocking & Latency**: Both designs use a synchronous active-low reset wrapper with fixed pipeline latency $L=1$.
   - **Identical Synthesis Flow**: Both designs are synthesized with the identical Yosys script (`synth -top <top>; flatten; stat -json; ltp`).
   - **Dual Reporting (P vs P_bus)**: We report both:
     * `Design P`: Includes $W_{\text{reg}}$ flip-flops and write-address decoding logic (full system view).
     * `Design P_bus`: Strips $W_{\text{reg}}$ registers and takes weights as primary inputs, evaluating pure datapath multiplier logic.

2. **What the Results DO Show**:
   - In logic synthesis, hardcoding weights as constants reduces cell count by **3.75× to 3.85×** and eliminates up to **17.7× flip-flops**.
   - Topological critical path depth is reduced by **20% to 32%** because constant shift-add eliminates partial-product compression stages.
   - Real LLM weights are **23% smaller** than uniform random INT8 because trained weights cluster near zero.

3. **What the Results Do NOT Show (Pre-Layout Limitations)**:
   - **Not Post-Place-and-Route**: Synthesis metrics reflect standard-cell logic mapping. Interconnect wire delays, clock tree insertion delays, and routing congestion must be validated with physical PDKs (SkyWater 130nm, TSMC 28nm).
   - **Not Full Transformers**: Part I covers dense linear projections only. Normalization (RMSNorm) and non-linearities (Softmax, SiLU) are not included on-chip in Part I.

---

## Question 5: What Would Part II Add and What Would It Change in the IR and Backend?

### **Examiner:** "If you were to continue this project into Part II, what would you add and what changes would be required in the codebase?"

### **Answer:**
Part II transitions from linear layer specialization to full transformer block execution:

1. **Pipelined Adder Trees ($L > 1$)**:
   - For wide inner dimensions ($K = 1536$ in `down_proj`), combinational reduction trees have high topological depth.
   - In Part II, we would introduce pipeline register stages into the reduction tree, parameterizing latency $L = 1 + \lceil \log_2(\text{num\_tiles}) \rceil$.
   - **IR Change**: Annotate `LinearOp.hw["latency"] = L` and generate multi-stage valid shift registers.

2. **Hardware Accelerators for Non-Linear Operators**:
   - Implement dedicated fixed-point units for:
     * Fused RMSNorm: reciprocal square-root polynomial approximation.
     * SwiGLU: fixed-point SiLU elementwise approximation ($x \cdot \sigma(x)$) multiplied by `up_proj` output.
   - **Backend Change**: Add `backend/rmsnorm_codegen.py` and `backend/activation_codegen.py`.

3. **On-Chip Activation Double Buffering (Ping-Pong BRAM)**:
   - Stream token activations from on-chip BRAM/URAM blocks rather than primary input buses.
   - **Backend Change**: Wrap cores with AXI-Stream memory interfaces.

---

## Question 6: Waveform Inspection Walkthrough

### **Examiner:** "Can you open a simulation waveform and explain a single inference transaction?"

### **Answer:**
Referring to our waveform walkthrough in `docs/walkthroughs/waveforms.md`:

1. **Reset Sequence**:
   - `rst_n` asserted LOW for 2 clock cycles.
   - On the rising edge of `clk`, `valid_out` resets to `0` and internal accumulator registers clear.
   - `rst_n` deasserted HIGH to enable normal operation.

2. **Inference Transaction**:
   - Host drives `valid_in = 1` and packs $K$ 8-bit inputs into `x_flat[8*K-1:0]`.
   - On Clock Edge $T_1$: Input is latched into the combinational core.
   - Combinational core computes all tile products and adder tree reductions within the clock period.
   - On Clock Edge $T_2$ (Latency $L=1$ cycle later):
     * `valid_out` transitions from `0` to `1`.
     * `y_flat[32*N-1:0]` is registered and stable on the output bus.
   - Host samples `y_flat` on $T_2$, completing the transaction.

3. **Verification of Fixed Weights in Waveform**:
   - In Design F, there is no `w_data`, `w_we`, or `w_addr` signal toggling before inference. The core is instantly ready to compute on cycle 1.
   - In Design P, an initial weight-load phase takes $N \times K$ clock cycles to write weights into $W_{\text{reg}}$ before `valid_in` can be asserted.
