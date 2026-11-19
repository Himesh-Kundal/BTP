# LLMHW System Limitations and Architectural Boundaries (Part I & Part II)

This document formalizes the explicit scope boundaries, hardware assumptions, numeric trade-offs, and physical design caveats across Part I (spatial constant-weight linear layers) and Part II (complete ROM-backed SmolLM2-135M transformer decoder accelerator).

---

## 1. Architectural & Functional Boundaries

1. **Autoregressive Greedy Decoding (`batch_size = 1`):**
   - The Part II hardware accelerator (`top_accel.v`) processes one token per step (`tok_in` $\to$ `tok_out`) using deterministic greedy decoding (`argmax.v` with lowest-token-ID tie-breaking).
   - Stochastic sampling (temperature scaling, top-$k$, top-$p$ nucleus sampling) and batched beam search are excluded from the core hardware pipeline to guarantee deterministic bit-for-bit verification (Claim C1′).
2. **Host-Side Tokenization:**
   - Byte-Pair Encoding (BPE) string tokenization (`text` $\to$ `uint16` token IDs) and detokenization (`uint16` token IDs $\to$ `text`) run on the host CPU using the pinned HuggingFace `AutoTokenizer`. The hardware interface operates strictly on 16-bit integer token IDs (`[0, 49151]`).
3. **Bounded Context Window (`MAX_CTX = 256`):**
   - While SmolLM2-135M supports up to $8,192$ context positions in software, the synthesized hardware accelerator parameterizes the on-chip KV cache SRAM (`kv_cache.v`) and RoPE tables to `MAX_CTX = 256` positions ($2.81\text{ MB}$ KV SRAM) to keep Verilator simulation and on-chip SRAM capacity tractable.
   - Because `MAX_CTX` is a top-level Verilog parameter, extending to $2,048$ or $8,192$ positions requires zero datapath logic changes—only linear scaling of the KV cache SRAM macro ($22.5\text{ MB}$ at $2,048$; $90.0\text{ MB}$ at $8,192$).
4. **Sequential Prefill Execution:**
   - Prompt prefill reuses the single-token decode engine sequentially across prompt positions `pos = 0 .. L_prompt - 1`. Dedicated multi-token parallel prefill systolic arrays are left to future work.
5. **Immutable Model-Specific ROM Weights:**
   - In Part I (Design F), weights are hardwired into combinational shift-add networks.
   - In Part II (`top_accel.v`), all $162.79\text{M}$ weights, $288,990$ dyadic requantization parameters, lookup tables, and microprogram instructions reside in read-only synchronous ROMs (`rom_sync.v`) with **zero write ports**. Updating model weights requires regenerating the ROM initialization images (or mask-ROM / metal-layer respin in an ASIC).

---

## 2. Numeric & Quantization Limitations

1. **Static W8A8 Linear Arithmetic with Offline SmoothQuant Migration:**
   - All linear projections use symmetric INT8 weights ($[-127, 127]$) and symmetric INT8 input activations ($[-127, 127]$), while the residual stream uses saturating 24-bit signed integers (`INT24`).
   - Because SmolLM2-135M exhibits extreme activation outliers in specific layers (`L02`, `L11`, `L28`, `L29` reaching magnitudes up to $2,630$ at `pos = 0`), static per-tensor INT8 activation scaling without offline channel migration causes severe clipping or underflow (`BLOCKER-UE7-01`).
   - Our `ADR-2-2` compile-time SmoothQuant migration folds per-channel scale balancing into the preceding/succeeding weight matrices (`weights.hex`) and dyadic multipliers (`requant_params.hex`) at compile time with zero runtime hardware overhead, achieving **70.16% Top-1 ArgMax agreement** and **$\Delta \text{PPL} = +17.33$** ($89.67 \to 107.00$) on WikiText-2.
2. **Multi-Step Autoregressive Trajectory Divergence:**
   - Even with 70.16% single-token Top-1 agreement and $\text{KL} = 0.2280$, autoregressive greedy decoding compounds small logit differences: once a single token diverges from the FP32 greedy choice at step $t$, subsequent tokens condition on a different prefix. Across the 10-prompt evaluation battery (`UE.6`), the mean exact greedy prefix match length before first divergence is **4.5 tokens** (ranging up to 16/16 tokens on factual completions), after which the integer hardware continues generating grammatical, coherent text along an alternative trajectory.

---

## 3. Synthesis, Simulation, and ASIC Projection Caveats

1. **Simulation vs. Silicon:**
   - Cycle counts ($14,100,262$ cycles/token at $8 \times 16$ MAC lanes) and bit-exact equivalence are proven via cycle-accurate RTL simulation in Icarus Verilog and Verilator, not fabricated silicon.
2. **Pre-Layout Synthesis Proxies:**
   - Logic cell counts (`EA1`: $78,510$ generic cells; $80,762$ iCE40 cells) and topological critical path depths (`ltp`) come from Yosys pre-layout synthesis without physical placement, routing congestion, clock-tree synthesis (CTS), or wire parasitics.
3. **Analytical 28nm ASIC Area & Throughput Estimates (`EA5`):**
   - All die area ($147.95\,\text{mm}^2$ Mid estimate in TSMC 28nm HPC+), clock frequency ($250\text{–}550\text{ MHz}$), and throughput ($28.37\text{ tokens/s}$ @ 400 MHz for 128 lanes) figures in `results/part2/EA5.json` are **clearly labeled analytical estimates** derived from published 28nm high-density via-programmable ROM ($0.105\,\mu\text{m}^2/\text{bit}$), 6T SRAM ($0.155\,\mu\text{m}^2/\text{bit}$), and standard-cell gate density ($1.42\times 10^6\,\text{gates/mm}^2$) literature.
   - Die area is $96.5\%$ dominated by the $157.54\text{ MB}$ weight/parameter ROM macros ($\epsilon_{\text{ROM}} = 0.965$); a production tapeout would require commercial Foundry memory compilers, multi-banked ROM macro floorplanning, timing closure across macro-to-logic buses, and power/IR-drop signoff.
