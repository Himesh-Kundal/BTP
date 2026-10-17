# Waveform Walkthrough: Pipeline Timing, Transactions, and Fault Signatures

**Scope:** Visual cycle-by-cycle breakdown of Design F registered execution, stimulus-response timing contracts, and fault detection mechanisms.  
**Generated Artifacts:**
- [`artifacts/waveforms/pass.vcd`](file:///home/himesh/BTP/artifacts/waveforms/pass.vcd): Bit-exact passing transaction ($L=1$ cycle latency).
- [`artifacts/waveforms/fail_m1_sign.vcd`](file:///home/himesh/BTP/artifacts/waveforms/fail_m1_sign.vcd): Arithmetic mismatch signature (M1 sign-flip fault).
- [`artifacts/waveforms/fail_m9_timing.vcd`](file:///home/himesh/BTP/artifacts/waveforms/fail_m9_timing.vcd): Protocol violation signature (M9 latency fault).
- Reproduction command: `PYTHONPATH=. uv run python scripts/generate_waveforms.py`

---

## 1. Architectural Timing Contract ($L=1$)

Design F implements a registered wrapper around the combinational fixed-weight multiplier-reduction core:

```
                +-------------------------------------------------------------+
                | llmhw_<layer> (Registered Wrapper, Latency L=1)            |
                |                                                             |
   x_flat[31:0] ------> [ Combinational Core: Multipliers & Adder Tree ]      |
                |                            |                                |
                |                         y_comb[63:0]                        |
                |                            |                                |
  valid_in -----------> [ 1-bit Reg ]        v                                |
                |              |        [ 64-bit Reg ]                        |
                |              v             |                                |
                |          valid_out         v                                |
                |                       y_flat[63:0]                          |
                +-------------------------------------------------------------+
```

### Signal Specifications:
| Signal | Dir | Width | Signedness | Semantics |
|---|---|---|---|---|
| `clk` | In | 1 bit | N/A | Master system clock (10 ns period, 100 MHz). |
| `rst_n` | In | 1 bit | Active-low | Synchronous/asynchronous active-low reset. Clears `valid_out` and `y_flat`. |
| `valid_in` | In | 1 bit | Active-high | Indicates `x_flat` contains a valid input activation vector on this cycle. |
| `x_flat` | In | $8 \times K$ bits | Packed INT8 | Little-endian vector packing: lane $k$ is at `[8*k+7 : 8*k]`. |
| `valid_out`| Out| 1 bit | Active-high | Asserts exactly 1 cycle after `valid_in` to qualify `y_flat`. |
| `y_flat` | Out| $32 \times N$ bits| Packed INT32| Packed output activations: row $n$ is at `[32*n+31 : 32*n]`. |

---

## 2. Passing Inference Transaction (`pass.vcd`)

### 2.1 Test Matrix Point
A $2 \times 4$ projection ($N=2, K=4$) structured as a $2 \times 2$ grid of $2 \times 2$ tiles:
$$
W = \begin{pmatrix} 12 & -7 & 0 & 15 \\ -3 & 8 & 22 & -1 \end{pmatrix}
$$

Stimulus vectors $X \in \mathbb{Z}^{3 \times 4}$:
1. **Vector 0 (General signed):** $X_0 = (10, 20, 30, 40)$  
   - Row 0: $10 \times 12 + 20 \times (-7) + 30 \times 0 + 40 \times 15 = 120 - 140 + 0 + 600 = \mathbf{580}$ (`32'h00000244`)
   - Row 1: $10 \times (-3) + 20 \times 8 + 30 \times 22 + 40 \times (-1) = -30 + 160 + 660 - 40 = \mathbf{750}$ (`32'h000002EE`)
2. **Vector 1 (Extremes & bounds):** $X_1 = (127, -128, 0, 1)$  
   - Row 0: $127 \times 12 + (-128) \times (-7) + 0 \times 0 + 1 \times 15 = 1524 + 896 + 0 + 15 = \mathbf{2435}$ (`32'h00000983`)
   - Row 1: $127 \times (-3) + (-128) \times 8 + 0 \times 22 + 1 \times (-1) = -381 - 1024 + 0 - 1 = \mathbf{-1406}$ (`32'hFFFFFA82`)
3. **Vector 2 (All zero):** $X_2 = (0, 0, 0, 0)$  
   - Row 0: $\mathbf{0}$ (`32'h00000000`)
   - Row 1: $\mathbf{0}$ (`32'h00000000`)

---

### 2.2 Cycle-by-Cycle Timing Diagram (ASCII)

```
Time (ps):       0         15000     25000     35000     45000     55000     65000     75000
Cycle:           C0         C1        C2        C3        C4        C5        C6        C7
                 |          |         |         |         |         |         |         |
clk          ____/¯¯¯¯\____/¯¯¯¯\____/¯¯¯¯\____/¯¯¯¯\____/¯¯¯¯\____/¯¯¯¯\____/¯¯¯¯\____/¯¯¯¯
                 |          |         |         |         |         |         |         |
rst_n        ______________________________/¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
                 |          |         |         |         |         |         |         |
valid_in     ___________________________________/¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\__________
                 |          |         |         |         |         |         |         |
x_flat       <==================================X== V0 ===X== V1 ===X== V2 ===X==== 0 =====>
             (00000000)                         (281E140A)(0100807F)(00000000)
                 |          |         |         |         |         |         |         |
valid_out    _____________________________________________//¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\_____
                 |          |         |         |    ^    |         |         |         |
y_flat       <=======================================X====== Y0 =====X=== Y1 ===X=== Y2 ===>
             (0000000000000000)                      (02EE_0244)(FA82_0983)(0000_0000)
                                                     |<- L=1 ->|
```

---

### 2.3 Detailed Signal Log from `pass.vcd`

| Time (ps) | Clock Edge | Active Signals | Bus Hex Values | Transaction Meaning |
|---|---|---|---|---|
| **0 – 20,000** | C0 – C2 | `clk`, `rst_n=0` | `x_flat=0`, `y_flat=0` | Power-on reset held low for 2 full clock cycles. Registers cleared. |
| **26,000** | Between C2-C3| `rst_n -> 1` | `rst_n=1` | Reset deasserted 1 ns after posedge C2, setup time fully met. |
| **36,000** | Post C3 | `valid_in -> 1` | `x_flat = 32'h281e140a` | **Vector 0 presented:** Lane 0=`0a` (10), Lane 1=`14` (20), Lane 2=`1e` (30), Lane 3=`28` (40). |
| **36,000** | Combinational | `y_comb` settles | `y_comb = 64'h000002ee_00000244` | Adder tree computes Row 0 (580) and Row 1 (750) combinatorially. |
| **45,000** | **Posedge C4**| `valid_out -> 1` | `y_flat = 64'h000002ee_00000244` | **L=1 Delay Met:** Output register latches $Y_0$. Monitor compares `act == exp` -> **PASS (0 mismatches)**. |
| **46,000** | Post C4 | `valid_in = 1` | `x_flat = 32'h0100807f` | **Vector 1 presented:** Lane 0=`7f` (127), Lane 1=`80` (-128), Lane 2=`00` (0), Lane 3=`01` (1). |
| **55,000** | **Posedge C5**| `valid_out = 1` | `y_flat = 64'hfffffa82_00000983` | Output register latches $Y_1$: Row 0 = 2435, Row 1 = -1406. Monitor checks -> **PASS**. |
| **56,000** | Post C5 | `valid_in = 1` | `x_flat = 32'h00000000` | **Vector 2 presented:** All zero lanes. |
| **65,000** | **Posedge C6**| `valid_out = 1` | `y_flat = 64'h00000000_00000000` | Output register latches $Y_2$: Row 0 = 0, Row 1 = 0. Monitor checks -> **PASS**. |
| **66,000** | Post C6 | `valid_in -> 0` | `x_flat = 0` | Stimulus stream exhausted; input strobe deactivated. |
| **75,000** | **Posedge C7**| `valid_out -> 0` | `y_flat = 0` | Pipeline flushed. `valid_out` returns to 0. Total vectors checked = 3. |

---

## 3. Deliberately Failing Transactions

### 3.1 Arithmetic Fault: M1 Sign Flip (`fail_m1_sign.vcd`)
- **Fault Description:** Mutant M1 flips the sign of weight constant $W_{0,0}$ from $+12$ (`8'sd12`) to $-12$ (`-8'sd12`).
- **Effect on Vector 0:**
  $$Y_{0, \text{exp}} = 10 \times 12 + 20 \times (-7) + 40 \times 15 = 580 \quad (\text{0x00000244})$$
  $$Y_{0, \text{act}} = 10 \times (-12) + 20 \times (-7) + 40 \times 15 = 340 \quad (\text{0x00000154})$$
- **Waveform Signature at $t = 45,000$ ps (C4):**
  ```
  Time:              45000 ps (Posedge C4)
  valid_out:         1
  y_flat[31:0]:      32'h00000154   <-- INCORRECT (Expected 32'h00000244)
  y_flat[63:32]:     32'h000002ee   <-- CORRECT
  Testbench Output:  MISMATCH at vec 0, row 0: exp=00000244 act=00000154
  ```
- **Conclusion:** The testbench monitors each row slice independently and catches arithmetic corruption in the exact cycle it emerges from the pipeline.

---

### 3.2 Protocol Fault: M9 Latency Violation (`fail_m9_timing.vcd`)
- **Fault Description:** Mutant M9 violates the latency contract by inserting an extra flip-flop stage ($L=2$ instead of $L=1$).
- **Effect on Vector 0:**
  - Expected output cycle: C4 ($t = 45,000$ ps).
  - Actual output cycle: C5 ($t = 55,000$ ps).
- **Waveform Signature at $t = 45,000$ ps (C4):**
  ```
  Time:              45000 ps (Posedge C4)
  valid_in (delayed): 0 (still stuck in extra stage)
  valid_out:         0   <-- TIMING VIOLATION (Expected valid_out == 1)
  Testbench Monitor: TIMING ERROR at vec 0: valid_out not asserted at expected cycle L=1
  ```
- **Conclusion:** The testbench contains strict cycle-accurate timing checkers (`TIMING_ERR`), ensuring Design F adheres rigidly to the single-cycle registered contract required by the host system.

---

## 4. Viva Preparation: Questions and Concrete Answers

### Q1: Why does Design F use registered outputs ($L=1$) rather than purely combinational logic ($L=0$)?
**Answer:** While the internal core is combinational, raw combinational outputs would make the layer's critical path depend on the external interconnect and upstream drivers. Registering the outputs decouples the layer's internal tree latency from external routing, provides clean timing closure at 100 MHz, and establishes a deterministic single-cycle synchronous handshake (`valid_in` $\to$ `valid_out`).

### Q2: How does the testbench verify multiple rows without race conditions?
**Answer:** The testbench uses non-blocking synchronous checking at `posedge clk`. Inputs are driven on `#1` (1 ns after clock edge) to satisfy flip-flop setup times without race conditions. Outputs are sampled on the clock edge, compared against expected arrays in `y_exp.hex`, and evaluated independently per row slice `[32*n +: 32]`.

### Q3: How are signed INT8 activations safely extended into INT32 accumulators?
**Answer:** The generator emits explicit signed wires with sign extension:
```verilog
wire signed [7:0] x_0 = x_flat[7:0];
wire signed [15:0] prod_0_0 = x_0 * 8'sd12;
wire signed [31:0] ext_prod_0_0 = {{16{prod_0_0[15]}}, prod_0_0};
```
By replicating the sign bit `prod_0_0[15]` across the upper 16 bits, negative values (such as $-1406$) retain two's-complement integrity across the adder tree without unsigned overflow.

### Q4: What happens to weights that are zero?
**Answer:** Weights with $W_{n,k} = 0$ are completely omitted from the RTL codegen (Rule N7). No multiplier, no sign-extension wire, and no adder port is instantiated for zero weights. In our $2 \times 4$ example, $W_{0,2} = 0$, so `ext_prod_0_2` does not exist and `y_0` is a 3-input adder tree instead of a 4-input tree, saving silicon area and dynamic power.
