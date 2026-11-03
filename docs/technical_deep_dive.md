# Technical Deep-Dive: Engineering, Mathematics, RTL & VLSI Architecture
**Project:** *Model-Specific Hardware Compilation of Dense Transformer Projections into Synthesizable Constant-Weight Verilog*

> **Purpose of this document:** Now that you know the fundamentals from [`docs/complete_beginner_guide.md`](file:///home/himesh/BTP/docs/complete_beginner_guide.md), this guide takes you into the **deep engineering, bit-level two's complement mathematics, VLSI multiplier architecture, exact Verilog-2001 RTL code, and Yosys synthesis internals** of your project. This prepares you for the toughest technical questions from VLSI and Signal Processing professors (such as Prof. Mrityunjoy Chakraborty).

---

## Chapter 1: Bit-Level Two's Complement Math & Bit-Growth Proofs

### 1.1 Signed Two's Complement Representation (Rule N1)
In digital hardware, negative numbers are represented using **Two's Complement**.
For an $N$-bit signed binary number $b_{N-1} b_{N-2} \dots b_1 b_0$, the most significant bit (MSB, $b_{N-1}$) has a **negative weight**, while all other bits have positive weights:
$$\text{Value} = -b_{N-1} \cdot 2^{N-1} + \sum_{i=0}^{N-2} b_i \cdot 2^i$$

For our **8-bit signed integer (`INT8`)**:
* Weight of bit 7 (MSB) $= -128$, weights of bits 6..0 $= +64, +32, +16, +8, +4, +2, +1$.
* Example 1: `+42` $= \texttt{0010\_1010}_2 = 0 + 32 + 8 + 2 = +42$.
* Example 2: `-42` (invert bits of `+42` and add 1):
  $$\texttt{0010\_1010}_2 \xrightarrow{\text{invert}} \texttt{1101\_0101}_2 \xrightarrow{+1} \texttt{1101\_0110}_2 = -128 + 64 + 16 + 4 + 2 = -42$$

### 1.2 Mathematical Proof: Why Exclude `-128` (`1000_0000`)? (Rule N2 & ADR-0002)
An 8-bit two's complement wire can represent `[-128, +127]`. However, our quantization contract strictly clips both weights and activations to **`[-127, +127]`** (`8'sd127` down to `-8'sd127`, forbidding `-128`). Why?

1. **Unary Negation Overflow in Constant Optimization:**
   When a weight is $w = -1$, our compiler optimizes the multiplication $x \times (-1)$ into a hardware negation: `-x`.
   If $x = -128$ (`1000_0000`), what happens when an 8-bit subtractor computes $0 - (-128)$?
   $$\text{Invert } \texttt{1000\_0000}_2 \to \texttt{0111\_1111}_2 \xrightarrow{+1} \texttt{1000\_0000}_2 = -128 \neq +128!$$
   Because $+128$ cannot be represented in 8-bit signed two's complement, negating $-128$ wraps around to $-128$ (a catastrophic sign-inversion error!). By restricting $x, w \in [-127, +127]$, $-x$ is always strictly inside $[-127, +127]$.
2. **Product Range Symmetry:**
   When multiplying $x \in [-127, 127]$ and $w \in [-127, 127]$, the product $p = x \cdot w$ lies strictly in:
   $$p \in [-16{,}129, \; +16{,}129]$$
   Notice that $16{,}129 < 2^{14} = 16{,}384$. If $-128$ were allowed, $(-128) \times (-128) = +16{,}384$, requiring an extra magnitude bit in the partial product tree.
3. **Zero-Point Symmetry:**
   In symmetric quantization ($q = \text{round}(r / s)$), the floating-point range $[-M, +M]$ is symmetric around $0.0$. Mapping $[-M, +M]$ to $[-127, +127]$ ensures that $+M \mapsto +127$, $-M \mapsto -127$, and $0.0 \mapsto 0$ with zero DC offset.

### 1.3 Bit-Growth Derivation for Dot-Product Accumulators (Rules N4 & N10)
**Professor Question:** *"How many bits are strictly required to accumulate a dot product of length $K$ without overflow, and why did you choose 32 bits?"*

**Exact Derivation:**
1. Each input $x_j \in [-127, 127]$ and weight $w_{i,j} \in [-127, 127]$ is bounded by $|x_j| \le 2^7 - 1$ and $|w_{i,j}| \le 2^7 - 1$.
2. Each product $p_{i,j} = x_j \cdot w_{i,j}$ is bounded by:
   $$|p_{i,j}| \le 127 \times 127 = 16{,}129 < 2^{14}$$
   With 1 sign bit, a single product requires at most **15 bits** (we allocate standard byte-aligned **16-bit signed** wires, `[-32768, +32767]`).
3. When summing $K$ products in an inner product $y_i = \sum_{j=0}^{K-1} p_{i,j}$, every doubling of $K$ adds **1 guard bit** to the maximum magnitude. Therefore, for accumulation length $K$, the required signed bitwidth $B_{\text{req}}$ is:
   $$B_{\text{req}} = 1 + \left\lceil \log_2\left( K \times 16{,}129 \right) \right\rceil$$
4. Let's plug in the exact dimensions of **SmolLM2-135M**:
   * For Attention & MLP input projections ($K = 576$):
     $$|y_{\max}| = 576 \times 16{,}129 = 9{,}290{,}304 < 2^{24} = 16{,}777{,}216 \implies B_{\text{req}} = \mathbf{25\text{ bits}}$$
   * For MLP `down_proj` ($K = 1536$):
     $$|y_{\max}| = 1536 \times 16{,}129 = 24{,}774{,}144 < 2^{25} = 33{,}554{,}432 \implies B_{\text{req}} = \mathbf{26\text{ bits}}$$
5. By selecting a standard **32-bit signed accumulator (`[31:0]`, max value $2^{31}-1 = 2{,}147{,}483{,}647$)**, we have **6 to 7 bits of guaranteed overflow margin** even under the worst-case adversarial input where every single element is $\pm 127$ aligned with the sign of $w_{i,j}$.

### 1.4 Dead-Bit Pruning in Design F (Why Design F Needs Fewer Output Flip-Flops!)
Here is a subtle VLSI insight from Experiment E1:
In **Design P**, the weights $w_{i,j}$ are unknown at compile time. Therefore, Yosys must assume every $w_{i,j}$ *could* be $\pm 127$, so it must synthesize full 32-bit adders and all 32 flip-flops per output channel.

In **Design F**, the weights $w_{i,j}$ are **known constants**! For a specific row $i$ of an $8 \times 16$ tile, the worst-case sum is bounded by the $L_1$ norm of that row's constants:
$$|y_i|_{\max} = 127 \times \sum_{j=0}^{15} |w_{i,j}|$$
In real SmolLM2-135M tiles, most weights are small ($\approx 5\text{ to }20$), so $\sum_{j=0}^{15} |w_{i,j}| \approx 180$, meaning $|y_i|_{\max} \approx 22{,}860 < 2^{15}$.
Yosys's constant-propagation pass (`opt_expr`) proves statically that the upper 16 bits (`y[31:16]`) are **identical copies of the sign bit `y[15]`**! Consequently, Yosys **prunes the upper 16 carry-chain stages and flip-flops**, tying those wires directly to the sign bit `y[15]` at zero gate cost!

---

## Chapter 2: Quantization Mathematics & Round-Half-to-Even (`frontend/` & `verification/reference.py`)

### 2.1 Per-Channel Weight vs. Per-Tensor Activation Quantization
Why do we use **per-output-channel** scaling for weights (Rule N2) and **per-tensor** scaling for activations (Rule N3)?
Look at the matrix multiplication equation for output channel $i$:
$$y_{\text{fp32}, i} = \sum_{j=0}^{K-1} x_{\text{fp32}, j} \cdot W_{\text{fp32}, i, j}$$
Substitute $x_{\text{fp32}, j} \approx s_x \cdot x_{q, j}$ and $W_{\text{fp32}, i, j} \approx s_{w, i} \cdot w_{q, i, j}$:
$$y_{\text{fp32}, i} \approx \sum_{j=0}^{K-1} (s_x \cdot x_{q, j}) \cdot (s_{w, i} \cdot w_{q, i, j}) = \underbrace{(s_x \cdot s_{w, i})}_{\text{Combined Scale } S_i} \cdot \underbrace{\left( \sum_{j=0}^{K-1} x_{q, j} \cdot w_{q, i, j} \right)}_{\text{Pure Integer Hardware Core } y_{q, i}}$$

* **Why $s_{w,i}$ can vary per row $i$:** Because $s_{w,i}$ depends only on row $i$, it factors **outside** the summation $\sum_{j=0}^{K-1}$! Thus, inside the inner loop (the hardware dot-product engine), we do **100% pure integer math** with zero floating-point scaling! Once the 32-bit integer sum $y_{q,i}$ exits the core, a single multiplication by $S_i = s_x \cdot s_{w,i}$ restores the floating-point scale.
* **Why $s_x$ cannot vary per row $i$:** Activations $\mathbf{x}$ are shared across all rows. If $s_x$ varied by column $j$, it could *not* be factored out of the summation $\sum_{j=0}^{K-1}$!

### 2.2 Why Round-Half-to-Even (`np.round`) Instead of Round-Half-Up?
In [`verification/reference.py`](file:///home/himesh/BTP/verification/reference.py), quantization uses **Round-Half-to-Even (Banker's Rounding)**:
* In standard "round-half-up", midpoint values ($+0.5, +1.5, +2.5$) always round in the positive direction. When summing $K = 576$ terms, this introduces a systematic positive statistical bias ($\mathbb{E}[\epsilon] > 0$).
* In **Round-Half-to-Even**, $0.5 \to 0$, $1.5 \to 2$, $2.5 \to 2$, $3.5 \to 4$. Half of the midpoints round up and half round down, guaranteeing a **zero-mean quantization error ($\mathbb{E}[\epsilon] = 0$)**.
* Across all 7 projections of SmolLM2-135M (verified in [`results/quant_quality.json`](file:///home/himesh/BTP/results/quant_quality.json)), our INT8 quantization achieves a **Cosine Similarity $> 0.9994$** (threshold $\ge 0.990$) and **Relative $L_2$ Error $< 0.032$** (threshold $\le 0.080$).

---

## Chapter 3: Inside the Generated Verilog-2001 RTL (`backend/`)

Let's examine the actual Verilog-2001 code generated by your compiler in [`backend/verilog_codegen.py`](file:///home/himesh/BTP/backend/verilog_codegen.py), [`backend/module_gen.py`](file:///home/himesh/BTP/backend/module_gen.py), and [`backend/baseline_p.py`](file:///home/himesh/BTP/backend/baseline_p.py).

### 3.1 How Vector Ports Work in Verilog-2001 (Bus Flattening)
Standard Verilog-2001 (IEEE 1364-2001) **forbids 2D arrays in module input/output ports** (you cannot write `input [7:0] x_in [0:15]` in a Verilog-2001 port list; that is a SystemVerilog feature).
To maintain strict IEEE 1364-2001 compliance across all synthesis tools, we **flatten** vector ports into contiguous 1D buses:
* For $K = 16$ inputs of 8 bits each: `input wire [127:0] x_in` ($16 \times 8 = 128$ bits).
* Element $j \in [0, 15]$ is packed at bit slice `[8*j + 7 : 8*j]`:
  * `x[0]` is at `x_in[7:0]`
  * `x[1]` is at `x_in[15:8]`
  * `x[15]` is at `x_in[127:120]`
* For $N = 8$ outputs of 32 bits each: `output reg [255:0] y_out` ($8 \times 32 = 256$ bits).
  * `y[i]` is packed at `y_out[32*i + 31 : 32*i]`.

### 3.2 Actual Generated Verilog for a Design F Tile (`backend/verilog_codegen.py`)
Here is what our compiler generates for a tile `tile_r0_c0` in **Design F**:

```verilog
module tile_r0_c0 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [127:0] x_in,
    output reg          valid_out,
    output reg  [255:0] y_out
);
    // 1. Unpack 128-bit input bus into 16 signed 8-bit wires
    wire signed [7:0] x_col_0  = x_in[7:0];
    wire signed [7:0] x_col_1  = x_in[15:8];
    wire signed [7:0] x_col_2  = x_in[23:16];
    // ... up to x_col_15

    // 2. Combinational constant-weight dot products (32-bit signed)
    // Notice: w[0,0] = +12, w[0,1] = 0 (SKIPPED!), w[0,2] = -1 (NEGATED!), w[0,3] = +1 (PASSED!)
    wire signed [31:0] acc_0 = ($signed(x_col_0) * 8'sd12)
                             + (-$signed(x_col_2))
                             + $signed(x_col_3);
    // ... acc_1 to acc_7

    // 3. Synchronous Registered Output Stage (Latency L = 1 cycle)
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            y_out     <= 256'd0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                y_out[31:0]   <= acc_0;
                y_out[63:32]  <= acc_1;
                // ... up to acc_7
            end
        end
    end
endmodule
```

### 3.3 Look Closely at the 4 Compiler Optimizations in `acc_0` (`compiler/weight_encoder.py`):
In [`compiler/weight_encoder.py`](file:///home/himesh/BTP/compiler/weight_encoder.py), function `encode_product_term(col_idx, weight_val)` emits specialized Verilog for each weight value:
1. **`weight_val == 0`:** Returns `None`. The term is **completely omitted** from the sum expression! If an entire row is zero, it emits `32'sd0`.
2. **`weight_val == 1`:** Emits `$signed(x_col_j)`. No multiplication operator `*` is emitted.
3. **`weight_val == -1`:** Emits `-$signed(x_col_j)`. Pure unary negation, no multiplier.
4. **General `weight_val` (e.g., `-42`):** Emits `$signed(x_col_j) * (-8'sd42)`.
   * *Technical Detail:* Why write `(-8'sd42)` instead of `8'sd-42`? Because in IEEE 1364-2001 Verilog syntax, the sign `-` must precede the bitwidth literal `8'sd42`, and wrapping it in parentheses `(-8'sd42)` prevents a parser ambiguity between `* -` and binary subtraction.

### 3.4 How Design P and Design P\_bus Differ in RTL (`backend/baseline_p.py`)
In [`backend/baseline_p.py`](file:///home/himesh/BTP/backend/baseline_p.py):
* **Design P** adds weight loading ports `input wire w_we`, `input wire [ADDR_W-1:0] w_addr`, `input wire signed [7:0] w_din`, and an internal flip-flop array:
  ```verilog
  reg signed [7:0] w_mem [0:127];
  always @(posedge clk) begin
      if (w_we) w_mem[w_addr] <= w_din;
  end
  ```
  And its dot product multiplies two runtime variables:
  ```verilog
  wire signed [15:0] prod_0_0 = x_col_0 * w_mem[0]; // Forces a full 8x8 hardware multiplier!
  ```
* **Design P\_bus** has no `w_mem` flip-flops; instead, it has a primary input bus `input wire [1023:0] w_in` ($128 \times 8$ bits) and computes:
  ```verilog
  wire signed [15:0] prod_0_0 = x_col_0 * w_0_0;    // Still forces a full 8x8 hardware multiplier!
  ```

---

## Chapter 4: VLSI Multiplier Architecture (Why Design F Shrinks Area by 3.85× & Critical Path by 32%)

If Prof. Chakraborty asks: *"From a VLSI Digital Signal Processing perspective, what happens inside the logic synthesizer when `x * w_var` (Design P) vs. `x * 8'sd12` (Design F) is synthesized?"* — give this exact explanation:

### 4.1 Cost of a Generic $8 \times 8$ Signed Multiplier (`Design P` / `Design P_bus`)
To multiply an 8-bit signed $x$ by an 8-bit signed variable $w$, Yosys must instantiate a **Baugh-Wooley Two's Complement Array Multiplier**:
1. **Partial Product Generation:** Every bit $x_a$ ($a \in [0..7]$) must be ANDed with every bit $w_b$ ($b \in [0..7]$). That immediately requires **$8 \times 8 = 64$ two-input AND/NAND gates** just to form the partial products.
2. **Reduction Tree (Carry-Save Adder Array):** Adding 8 rows of shifted partial products requires approximately **48 Full Adders (FAs) and 8 Half Adders (HAs)**. Since each Full Adder (`sum = a ^ b ^ cin`, `cout = (a&b) | (cin&(a^b))`) costs **5 to 9 primitive logic gates**, the reduction tree + final 16-bit Carry-Propagate Adder (CPA) consumes **~350 to 450 gates per multiplier**.
3. **Critical Path Delay:** Signal propagation must ripple through the partial product gate, through $\approx 6$ stages of Full Adders in the multiplier tree, and then through the 16-bit final carry chain—adding **25 to 35 gate levels** of delay *before* the row accumulation tree even starts!

### 4.2 Cost of Constant Shift-and-Add Multiplication (`Design F`)
When the multiplier is a compile-time constant (e.g., $w = 12 = \texttt{0000\_1100}_2 = 8 + 4$):
1. **Zero Partial-Product AND Gates:** Because the bits of $w$ are known `0`s and `1`s:
   * Where a bit of $w$ is `0` ($b \in \{0, 1, 4, 5, 6, 7\}$): $x_a \land 0 = 0$. The entire partial product row disappears!
   * Where a bit of $w$ is `1` ($b \in \{2, 3\}$): $x_a \land 1 = x_a$. No AND gate is needed—it is just the wire $x$ shifted left by 2 bits ($4x$) and 3 bits ($8x$)!
2. **Canonical Signed Digit (CSD) Recoding:**
   What if $w = 31$ ($\texttt{0001\_1111}_2$, five `1` bits)? Instead of adding $16x + 8x + 4x + 2x + x$ (4 adders), constant multiplication rewrites $31 = 32 - 1$:
   $$31 \cdot x = (x \ll 5) - x$$
   That requires **only ONE subtractor** instead of a 64-product Baugh-Wooley multiplier!
   In general, for any 8-bit constant in $[-127, 127]$, CSD recoding guarantees **at most 4 non-zero digits**, and on average for Gaussian-distributed LLM weights ($\mu \approx 0, \sigma \ll 127$), the average number of adders per weight is **$< 1.2$ adders**!
3. **Why Critical Path Shrinks by 20% to 32% (Experiment E3):**
   Because each constant product has only 0, 1, or 2 adder stages instead of a full $8 \times 8$ multiplier array, the **Longest Topological Path (`ltp`)** measured by Yosys drops from **56 gate levels in Design P down to 38 gate levels in Design F** (ratio **0.68** at $16 \times 64$).

---

## Chapter 5: Multi-Tile Hierarchy & Inter-Tile Reduction Trees (`compiler/tiling.py` & `backend/module_gen.py`)

### 5.1 How a Multi-Tile Core is Wired (`generate_full_module`)
Suppose we compile a matrix of shape $(N, K) = (16, 32)$ using tiles of size $(T_R, T_C) = (8, 16)$.
1. **Grid Dimensions:**
   $$R_{\text{tiles}} = \left\lceil \frac{16}{8} \right\rceil = 2 \text{ tile rows}, \quad C_{\text{tiles}} = \left\lceil \frac{32}{16} \right\rceil = 2 \text{ tile columns} \implies 4 \text{ tiles total}$$
2. **Combinational Tile Cores + Top-Level Registering (`Latency L = 1`):**
   Look at [`backend/module_gen.py`](file:///home/himesh/BTP/backend/module_gen.py). If each tile had its own flip-flop AND the top-level adder tree also had a flip-flop, the total latency would become $L = 2$ cycles, violating Claim C2 ($L=1$ parity with Design P).
   Instead, in multi-tile mode:
   * Each tile (`tile_r{r}_c{c}`) computes its **combinational** partial sum over its $T_C = 16$ columns:
     $$s_{r, c}[i] = \sum_{j=0}^{15} x_{16c + j} \cdot W_{8r + i, \; 16c + j}$$
   * At the top-level module, the **Inter-Tile Adder Reduction Tree** sums the partial outputs across all $C_{\text{tiles}}$ column tiles in the same row $r$ and adds the 32-bit bias $b_{8r+i}$:
     $$\text{total\_acc}_{8r+i} = \sum_{c=0}^{C_{\text{tiles}}-1} s_{r, c}[i] + b_{8r+i}$$
   * Finally, a **single synchronous register stage** latches `y_out <= total_acc` and `valid_out <= valid_in` on `posedge clk`, guaranteeing a strict **1-cycle pipeline latency ($L = 1$)** regardless of how many tiles are in the grid!

---

## Chapter 6: Inside Yosys Synthesis (`synth/yosys_runner.py` & `synth/fairness.py`)

### 6.1 What Yosys Executes Under the Hood
In [`synth/yosys_runner.py`](file:///home/himesh/BTP/synth/yosys_runner.py), we invoke Yosys 0.52 with:
```bash
yosys -p "read_verilog <file.v>; synth -top <top_module> -flatten; stat -json; ltp"
```
Here is what each pass inside `synth -flatten` does:
1. **`proc`:** Converts behavioral `always @(posedge clk)` blocks into hardware multiplexers (`$mux`) and D-flip-flops (`$dff`).
2. **`flatten`:** Inlines all `tile_r{r}_c{c}` submodules into the top module so constant propagation can optimize across tile boundaries and into the inter-tile reduction tree.
3. **`opt_expr` & `opt_clean`:** Performs constant folding ($0 \times x \to 0$, $x + 0 \to x$) and deletes unconnected wires and dead upper accumulator bits.
4. **`alumacc` & `share`:** Extracts arithmetic Carry-Save / ALU trees and shares common subexpressions.
5. **`techmap` & `abc`:** Maps wide adders and multipliers down to single-bit primitive logic gates (`$_AND_`, `$_NAND_`, `$_OR_`, `$_NOR_`, `$_XOR_`, `$_XNOR_`, `$_NOT_`, `$_MUX_`) and runs the Berkeley **ABC** engine to minimize the And-Inverter Graph (AIG).
6. **`stat -json`:** Outputs a machine-readable JSON dictionary of every primitive cell type. Our parser separates:
   * **Sequential Flip-Flops (`flops`):** Cells matching `$_DFF*` or `$_SDFF*` (e.g., `$_SDFF_PN0_` = synchronous D-flip-flop with positive clock, negative reset to 0).
   * **Combinational Logic (`comb_cells`):** `total_cells - flops`.
7. **`ltp` (Longest Topological Path):** Performs a breadth-first DAG traversal between flip-flop boundaries and prints:
   `Longest topological path in <top> (length = 38): ...`
   which our regex parser records as `ltp_depth = 38`.

### 6.2 How the Extrapolation Model Works (Experiment E5, `synth/extrapolation.py`)
In [`synth/extrapolation.py`](file:///home/himesh/BTP/synth/extrapolation.py), why can't we just multiply a single $(8 \times 16)$ tile's cell count by the number of tiles to get the full layer area?
Because a multi-tile grid also needs the **Inter-Tile 32-Bit Adder Reduction Tree**!
* For a grid of $R_{\text{tiles}} \times C_{\text{tiles}}$ tiles, each of the $N$ output rows must add $C_{\text{tiles}}$ partial sums together.
* Adding $C_{\text{tiles}}$ numbers requires a binary tree of **$C_{\text{tiles}} - 1$ thirty-two-bit adders** per output row:
  $$\text{Total Inter-Tile Adders} = N \times (C_{\text{tiles}} - 1)$$
* Therefore, our analytical extrapolation formula is:
  $$\text{Area}_{\text{est}}(N, K) = \left( R_{\text{tiles}} \cdot C_{\text{tiles}} \cdot \text{Area}_{\text{tile}} \right) + N \cdot (C_{\text{tiles}} - 1) \cdot \text{Area}_{\text{adder32}}$$
* When we tested this formula on a $16 \times 32$ core ($2 \times 2 = 4$ tiles) and synthesized the actual 4-tile core in Yosys, the formula was within **10.82% for Design F** and **1.37% for Design P**! (In fact, actual Design F synthesis was 10.82% *smaller* than the formula predicted because Yosys `flatten` optimized across the tile boundaries and pruned upper bits of the inter-tile adders too!).

---

## Chapter 7: Verification Internals — Hex Vectors, Static AST/Regex Audit, & Mutants

### 7.1 How the Verilog Testbench Works (`backend/testbench.py`)
When `llmhw verify` or `make demo` runs:
1. Python generates $V$ test activation vectors $\mathbf{X} \in [-127, 127]^{V \times K}$ and computes the golden output $\mathbf{Y}_{\text{ref}} \in \mathbb{Z}^{V \times N}$ using the 64-bit NumPy oracle.
2. Python packs each row of $\mathbf{X}$ into a hexadecimal string in `inputs.hex` and each row of $\mathbf{Y}_{\text{ref}}$ into `expected.hex`.
   * Example for $K=3$ with inputs `[10, -1, 127]`:
     `10` $= \texttt{0a}$, `-1` $= \texttt{ff}$ (in 8-bit two's complement: $256 - 1 = 255 = \texttt{0xff}$), `127` $= \texttt{7f}$ $\implies \texttt{7fff0a}$.
3. The generated Verilog testbench loads `inputs.hex` and `expected.hex` using `$readmemh`, drives one input vector per clock cycle with `valid_in = 1`, waits for `valid_out == 1`, and compares every 32-bit output word against `expected_mem[vec_idx]`.
4. At the end of simulation, the testbench prints:
   `PASS: checked=10 mismatch_count=0`
   To prevent false passes (e.g., if `valid_out` never went high and 0 vectors were checked!), our Python runner in [`verification/simulator.py`](file:///home/himesh/BTP/verification/simulator.py) asserts:
   * `mismatch_count == 0` **AND** `checked == num_vectors` ($\ge 1$).

### 7.2 How the Static Weight Constant Auditor Works (`verification/weight_audit.py`)
Even without running simulation, how can we prove that a 548,223-line Verilog file for a full $576 \times 576$ layer contains all 331,776 weights accurately and has zero hidden weight ports?
[`verification/weight_audit.py`](file:///home/himesh/BTP/verification/weight_audit.py) performs two static checks:
1. **Interface Port Audit (`verify_design_f_interface_rtl`):**
   Parses the module header and checks that:
   * Only the 6 allowed ports (`clk`, `rst_n`, `valid_in`, `x_in`, `valid_out`, `y_out`) exist.
   * Zero identifiers matching `w_*`, `weight*`, or memory write arrays exist (enforcing Invariant I3).
2. **Constant Reconstruction Audit (`parse_rtl_weights`):**
   Uses compiled regular expressions to parse every `wire signed [31:0] acc_{r} = ...;` statement across all tile modules, extracts the constant multiplier attached to each `x_col_{c}` term (recovering `+1` for bare `$signed(x_col_c)`, `-1` for `-$signed(x_col_c)`, `0` for omitted columns, and $\pm W$ for `* 8'sdW`), reassembles the full $N \times K$ matrix, and asserts exact element-by-element equality with the original quantized tensor in **0.85 seconds**!

---

## Chapter 8: Complete Codebase Map & Strict Layering (Golden Rule 5)

If an examiner asks you to open your code and show how it is organized, use this map:

```text
/home/himesh/BTP/
├── frontend/                   # ONLY place allowed to import PyTorch / HuggingFace!
│   ├── model_loader.py         # Loads HuggingFaceTB/SmolLM2-135M (pinned revision)
│   ├── linear_extractor.py     # Extracts 7 projections (q,k,v,o,gate,up,down) & quantizes to INT8
│   └── activation_capture.py   # Runs calibration prompts through model to capture real x vectors
├── ir/                         # Model-Agnostic Intermediate Representation
│   ├── layer_ir.py             # LinearLayerIR dataclass (N, K, weights, bias, scales)
│   └── manifest.py             # TileManifest JSON serializer/deserializer
├── compiler/                   # Hardware Specialization Passes (Zero PyTorch imports!)
│   ├── constant_weights.py     # Enforces Invariant I3 & validates INT8 [-127, 127] bounds
│   ├── tiling.py               # Pads arbitrary (N,K) matrices & slices into (Tr, Tc) tiles
│   └── weight_encoder.py       # Encodes weights into Verilog (0-skip, ±1 wire, shift-add literals)
├── backend/                    # Synthesizable Verilog-2001 Generators
│   ├── verilog_codegen.py      # Generates single-tile Design F module
│   ├── module_gen.py           # Generates multi-tile hierarchy + inter-tile adder reduction tree
│   ├── baseline_p.py           # Generates programmable baselines (Design P and Design P_bus)
│   └── testbench.py            # Generates self-checking Verilog testbenches & .hex vectors
├── verification/               # Independent Verification & Mutation Framework
│   ├── reference.py            # Pure NumPy INT64 mathematical reference oracle (Claim C1)
│   ├── simulator.py            # Runs Icarus Verilog (iverilog/vvp) and Verilator simulations
│   ├── weight_audit.py         # Static regex/AST auditor verifying Invariant I3 & all constants
│   └── mutate.py               # Injects 9 hardware fault mutants (M1–M9) for 100% kill test
├── synth/                      # Yosys Synthesis & Evaluation Pipeline
│   ├── yosys_runner.py         # Runs Yosys 0.52 (generic gates & iCE40 LUT4), parses cells/flops/ltp
│   ├── fairness.py             # Audits Rule N10 width parity, L=1 latency parity, and flags
│   ├── experiments.py          # Parallel runner for Experiments E1, E2, E3 (5 seeds)
│   └── extrapolation.py        # Analytical tile-scaling model & E5 multi-tile validation
└── cli.py / llmhw              # Unified CLI: `llmhw extract | compile | verify | synth`
```

### Why "Strict Layering" (Golden Rule 5) Matters:
In [`tests/test_layering.py`](file:///home/himesh/BTP/tests/test_layering.py), an automated test parses the Python Abstract Syntax Tree (`ast`) of every file in `ir/`, `compiler/`, `backend/`, and `verification/` to guarantee that **none of them ever `import torch` or mention `"SmolLM"`**.
* **Why is this good engineering?** Because our compiler core only cares about a generic 2D INT8 matrix in `TileManifest`! Tomorrow, if you want to compile **LLaMA-3**, **Mistral**, or a **FIR Digital Filter** for Prof. Chakraborty, you don't touch a single line of the compiler, Verilog generator, or verifier!
