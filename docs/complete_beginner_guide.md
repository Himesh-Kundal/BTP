# The Complete From-Zero Guide to Your BTP Report
**Project Title:** *Model-Specific Hardware Compilation of Dense Transformer Projections into Synthesizable Constant-Weight Verilog*

> **How to use this guide:** This document assumes **zero prior knowledge** of Large Language Models (LLMs), Verilog, FPGAs, hardware design, or compilers. Read it from top to bottom. By the time you finish, you will understand every single term, equation, figure, table, and design decision in your 14-page BTP report.

---

## Part 1: The Absolute Basics — Software vs. Hardware

### 1.1 How Normal Software Works (CPU / GPU)
When you write a Python or C++ program, you are writing **instructions** for a processor (like an Intel CPU or an NVIDIA GPU) that already physically exists.
* Think of a CPU/GPU like a **general-purpose kitchen with a human chef**.
* The **Pantry (RAM / Memory)** holds the ingredients (numbers/data).
* The **Chef (Arithmetic Logic Unit / Multiplier)** fetches two numbers from the pantry, multiplies them on a cutting board, and puts the result back into the pantry.
* Because the chef has to be ready to cook *any* recipe in the world, the chef spends most of their time walking back and forth between the pantry and the cutting board.

### 1.2 What is Digital Hardware (Chips, Gates, Flip-Flops)?
Inside every computer chip, everything is stored as binary bits: **`0`** (low voltage, 0V) and **`1`** (high voltage, e.g., 1V).
A hardware chip is built out of two basic physical building blocks:

1. **Logic Gates (Combinational Logic — The "Math" Parts):**
   Tiny electronic switches that compute outputs instantly when inputs change.
   * **AND gate:** Outputs `1` only if both inputs are `1`.
   * **XOR gate:** Adds two bits together.
   * By wiring thousands of AND/OR/XOR gates together, we build **Adders** (which add two numbers) and **Multipliers** (which multiply two numbers).
   * *In your report, these are called **Combinational Cells**.*

2. **Flip-Flops / Registers (Sequential Logic — The "Memory" Parts):**
   Logic gates have no memory—if you change the input wire, the output changes immediately. To *remember* a number from one step to the next, we use a **Flip-Flop** (also called a **Register**).
   * One Flip-Flop stores **1 bit** (`0` or `1`).
   * To store an 8-bit number (like `42`), you need **8 Flip-Flops** side by side.
   * Flip-Flops are triggered by a heartbeat signal called the **Clock (`clk`)**. Every time the clock ticks, the flip-flop locks in its new value.
   * *In your report, these are called **Storage Flops / Flip-Flops**.*

### 1.3 What is Verilog?
**Verilog** (specifically **Verilog-2001** used in your project) is a **Hardware Description Language (HDL)**.
* **Crucial distinction:** Python tells a processor *what steps to execute one after another*. **Verilog does NOT run on a processor.** Verilog is a **blueprint** that describes *how to wire logic gates and flip-flops together* to build a brand-new physical microchip!
* When you write `assign y = a + b;` in Verilog, you are literally saying: *"Place a physical hardware adder circuit on the silicon chip, solder wire `a` and wire `b` to its inputs, and solder wire `y` to its output."*

### 1.4 What is Simulation vs. Logic Synthesis?
Once you write a Verilog blueprint, what do you do with it? Two things:

1. **Simulation (Icarus Verilog & Verilator):**
   Before spending millions of dollars manufacturing a physical chip in a factory, we test the Verilog blueprint on a computer. A **Simulator** pretends to be the electricity flowing through your wires, feeding test numbers in (`x`) and checking if the output wires (`y`) produce the right math answer.
   * In your project, we used two famous open-source simulators: **Icarus Verilog (`iverilog`)** and **Verilator** to prove our chip works.

2. **Logic Synthesis (Yosys):**
   How do we know how big or fast our chip will be? We run a **Logic Synthesizer** (like a compiler for hardware). In your project, we used **Yosys 0.52**.
   * Yosys reads your human-readable Verilog code, optimizes the math, and converts it into a **Gate-Level Netlist**—an exact count of how many physical logic gates (Cells) and memory boxes (Flip-Flops) are required to build the chip.

### 1.5 What is an FPGA vs. an ASIC?
When turning Verilog into real hardware, there are two types of chips:
* **FPGA (Field-Programmable Gate Array):** A chip full of pre-made programmable Lookup Tables (**LUTs**—little truth tables, like 4-input **LUT-4** on the Lattice iCE40 FPGA mentioned in your report) that can be rewired electronically. Like building a castle out of **LEGO bricks**.
* **ASIC (Application-Specific Integrated Circuit):** A chip permanently etched onto silicon in a semiconductor foundry (like TSMC or SkyWater). Like **injection-molding a custom statue**. You can never change it after it is made, but it is way smaller, faster, and uses much less electricity.

---

## Part 2: What is an LLM and What Happens Inside It?

### 2.1 What is an LLM (Large Language Model)?
An LLM (like ChatGPT, LLaMA, or **SmolLM2-135M**) is an Artificial Intelligence model that takes text as input and predicts the next word.
* Despite sounding magical, inside the computer, an LLM is nothing more than a **giant series of matrix multiplications**.
* **SmolLM2-135M** is a compact, modern LLM made by HuggingFace. "135M" means it has **135 Million parameters (weights)**. It has **30 stacked layers** (Layer 0, Layer 1, ..., Layer 29), all having the exact same structure.

### 2.2 Weights vs. Activations (The Most Important Concept!)
Every calculation inside an LLM involves two kinds of numbers:

1. **Activations ($\mathbf{x}$):**
   These represent the **user's current input text** (e.g., *"What is the capital of India?"*) converted into a list of numbers.
   * **Activations change every millisecond** as new words are typed or generated.

2. **Weights ($\mathbf{W}$):**
   These represent the **brain/knowledge of the AI** that was learned when HuggingFace trained the model on terabytes of internet text for weeks on supercomputers.
   * **Crucial Fact:** Once training is finished and you download `SmolLM2-135M`, **the weights NEVER change again!** They are 100% frozen constants.

### 2.3 What is a "Dense Linear Projection" (Matrix Multiplication)?
Over 90% of the work inside an LLM happens inside **Dense Linear Layers** (also called **Projections**).
Mathematically, a linear layer does this simple equation:
$$\mathbf{y} = \mathbf{x}\mathbf{W}^T + \mathbf{b}$$

Let's do a tiny toy example with real numbers so you see how simple it is!
Suppose we have **3 input activations** ($\mathbf{x}$) and we want to compute **2 outputs** ($\mathbf{y}$):
* Input vector $\mathbf{x} = [10, \quad 20, \quad 5]$
* Weight matrix $\mathbf{W}$ (2 rows, 3 columns):
  $$\mathbf{W} = \begin{bmatrix} 2 & 0 & -1 \\ 3 & 1 & 4 \end{bmatrix}$$
* Bias vector $\mathbf{b} = [0, \quad 0]$

How do we calculate the outputs $y_0$ and $y_1$? We just multiply each input by its matching weight and add them up (this is called a **Dot Product**):
* **Row 0 ($y_0$):** $(10 \times 2) + (20 \times 0) + (5 \times -1) = 20 + 0 - 5 = \mathbf{15}$
* **Row 1 ($y_1$):** $(10 \times 3) + (20 \times 1) + (5 \times 4) = 30 + 20 + 20 = \mathbf{70}$

That's it! Every single "Linear Projection" in SmolLM2-135M is just doing this exact dot-product math, except instead of $2 \times 3$, the matrix size ($N \times K$) is $576 \times 576$ or $1536 \times 576$.

### 2.4 What are the 7 Projections in SmolLM2-135M?
In Section 2.1 of your report, we mention the **7 projection types** inside each of the 30 Transformer layers. Each layer has two parts:
1. **Attention Block (helps words look at other words in a sentence):**
   * `q_proj` (Query projection): size $576 \times 576$
   * `k_proj` (Key projection): size $192 \times 576$
   * `v_proj` (Value projection): size $192 \times 576$
   * `o_proj` (Output projection): size $576 \times 576$
2. **SwiGLU MLP / Feed-Forward Network (processes information within each word):**
   * `gate_proj` (Gate projection): size $1536 \times 576$
   * `up_proj` (Up projection): size $1536 \times 576$
   * `down_proj` (Down projection): size $576 \times 1536$

---

## Part 3: The Problem Statement — Why Current Chips Struggle & Your Core Idea

### 3.1 The Problem with Normal AI Chips ("Design P" — Programmable Baseline)
Suppose you want to put SmolLM2-135M into a small device (like a smart glasses chip, a drone, or a medical sensor).
How does a normal AI accelerator (like a Google TPU or generic NPU, which we call **Design P** in your report) run $\mathbf{y} = \mathbf{x}\mathbf{W}^T$?
1. Because a normal chip is **programmable** (it doesn't know what AI model you will load tomorrow), it must build **generic $8\text{-bit} \times 8\text{-bit}$ multipliers** that can multiply *any* number $x$ by *any* number $w$. A generic $8 \times 8$ multiplier takes **hundreds of logic gates**.
2. It must build **Flip-Flop registers (memory boxes)** on the chip to store all the weights $w$, plus **input ports/wires (`w_data`)** to load those weights in.
3. Moving weights around and keeping clock signals ticking on millions of weight registers wastes massive chip area and electricity (this is famous in computer architecture as the **Memory Wall / Von Neumann Bottleneck**).

### 3.2 Your BTP's Solution ("Design F" — Fixed Constant-Weight Hardware)
Your BTP asks a bold question (The **Hardware Specialization Hypothesis**):
> *"If the weights $\mathbf{W}$ of SmolLM2-135M are completely frozen after training, why are we building memory registers to store them and generic multipliers to multiply them? Why don't we write a **Compiler (`LLMHW`)** that bakes the exact weight numbers directly into the hardware wires as permanent constants?"*

We call this specialized hardware **Design F (Fixed-Weight Design)**.

---

## Part 4: How Baking Weights into Wires Works (The Magic of Constant Multiplication)

Why is **Design F** so much smaller and faster than **Design P**? Let's look at what happens inside hardware when a weight $w$ is a **known constant** at compile time vs. an unknown variable:

Suppose the input activation wire is $x$ (unknown at compile time, depends on what the user types).
In **Design P**, $w$ is also an unknown wire, so you MUST build a full $8 \times 8$ hardware multiplier circuit for every single weight.

In **Design F**, our compiler looks at the actual number $w$ from SmolLM2-135M and replaces the multiplier with ultra-simple circuits:

* **Case 1: The weight is $w = 0$ (Zero-Skipping)**
  * What is $x \times 0$? It is **always $0$**, no matter what $x$ is!
  * **Hardware needed:** **0 gates!** We completely delete the multiplier and don't even connect the wire.
* **Case 2: The weight is $w = +1$ or $w = -1$**
  * $x \times 1 = x$. We just connect the input wire directly to the adder! **0 multiplier gates!**
* **Case 3: The weight is a power of 2 ($w = 2, 4, 8, 16, 32, 64$)**
  * In normal decimal math, how do you multiply `45` by `10`? You don't calculate—you just put a `0` at the end (`450`).
  * In binary hardware, how do you multiply $x$ by $2$? You just shift the wires left by 1 bit ($x \ll 1$) and connect the bottom wire to ground (`0`)!
  * **Hardware needed:** **0 logic gates!** Just wire routing!
* **Case 4: Any other constant (e.g., $w = 3$ or $w = 5$)**
  * Since $3 = 2 + 1$, then $3 \times x = (2 \times x) + x = (x \ll 1) + x$.
  * Instead of a huge 64-gate multiplier, Yosys builds **just 1 simple adder**!
  * This technique is called **Shift-and-Add / Canonical Signed Digit (CSD) Optimization** (Equation 4 in your report).

### What is "Invariant I3" in Your Report?
Throughout your report, you will see **"Invariant I3 (Zero Weight Ports)"**.
* "Invariant" means an unbreakable rule of our architecture.
* **Invariant I3** states that a **Design F** Verilog module has **ZERO weight input ports (`w_data`)**, **ZERO weight write signals (`w_we`)**, and **ZERO weight flip-flop registers**. The weights exist *purely* as hardwired shift-and-add logic gates!

---

## Part 5: Numbers in Hardware — What is INT8 Quantization?

### 5.1 Why Quantize from FP32 to INT8?
When HuggingFace trains SmolLM2-135M on GPUs, the weights are stored as **32-bit Floating-Point numbers (`FP32`)**—scientific notation decimals like `0.0418293` or `-1.203948`.
* Building hardware circuits that multiply 32-bit floating-point decimals is huge and slow.
* So, in Section 2.2 of your report, we convert (**Quantize**) both the weights and activations into **8-bit Signed Integers (`INT8`)**—whole numbers between **`-127` and `+127`**.

### 5.2 How Does Symmetric INT8 Quantization Work (Rules N1–N3)?
It's just scaling! Suppose a row of weights has a maximum absolute value of $0.254$.
1. We find the **Scale Factor ($s_w$)**:
   $$s_w = \frac{\max(|\mathbf{W}|)}{127} = \frac{0.254}{127} = 0.002$$
2. We divide every decimal weight by $s_w$ and round to the nearest whole number (using round-half-to-even, `\lfloor \cdot \rceil`):
   * Weight `0.254` $\to 0.254 / 0.002 = \mathbf{+127}$
   * Weight `-0.006` $\to -0.006 / 0.002 = \mathbf{-3}$
   * Weight `0.0001` $\to 0.0001 / 0.002 = 0.05 \to \mathbf{0}$
3. Now all weights and activations are simple whole numbers in $[-127, +127]$!

### 5.3 Viva Favorite Question: Why $[-127, +127]$ instead of $[-128, +127]$?
An 8-bit signed binary number (Two's Complement) can technically hold values from `-128` to `+127`. Why did our Rule N2 clip the range symmetrically to **`[-127, +127]`** and forbid `-128`?
1. **Symmetry:** With `[-127, +127]`, every negative number has an exact positive twin (`-127` and `+127`), so zero (`0.0`) maps cleanly with zero bias drift.
2. **Preventing Hardware Sign-Flip Overflow:** In 8-bit two's complement math, if you negate `-128` ($-(-128) = +128$), `+128` cannot fit in 8 bits (which maxes out at `+127`), causing a catastrophic sign-inversion bug where $+128$ wraps around to $-128$! Also, $(-128) \times (-128) = +16,384$, whereas $(-127) \times (-127) = +16,129$, keeping the product cleanly inside signed 16-bit bounds without edge-case overflow.

### 5.4 Why 32-Bit Accumulators (Rules N4 & N10)?
Look at **Rule N10 (Width Conformance)** in Section 2.2:
* **Inputs ($x$):** 8-bit signed (`[-127, 127]`)
* **Weights ($w$):** 8-bit signed (`[-127, 127]`)
* **Single Product ($x \cdot w$):** 16-bit signed (since $127 \times 127 = 16,129$, which easily fits in 16 bits, where max is $32,767$).
* **Accumulator ($y = \sum x_j w_{i,j}$):** **32-bit signed**.
  Why 32 bits? Because in SmolLM2-135M, a row adds up to $K = 576$ products (or $1536$ in `down_proj`). Even if every single input is `+127` and every single weight is `+127`:
  $$\text{Max Sum} = 576 \times (127 \times 127) = 9,290,304$$
  A 16-bit integer only holds up to $32,767$ (so it would overflow!), whereas a **32-bit signed integer** holds up to **$2,147,483,647$** ($2^{31}-1$). Thus, a 32-bit accumulator is **100% mathematically guaranteed never to overflow**!

---

## Part 6: System Architecture — Tiling & The Three Designs (`F`, `P`, `P_bus`)

### 6.1 What is Spatial Tiling (Section 3.1)?
A full `q_proj` matrix in SmolLM2-135M has size $576 \times 576 = 331,776$ weights.
Trying to synthesize 331,776 multipliers as one giant flat block would take hours for EDA tools to process.
Instead, our compiler divides the big $(N \times K)$ matrix into smaller rectangular building blocks called **Tiles** of size $(T_R \times T_C)$—for example, **$8 \times 16$** (8 output rows, 16 input columns = 128 weights per tile).

```text
Big Matrix (e.g., 16 x 32) broken into four (8 x 16) Tiles:

         16 cols (x[0..15])      16 cols (x[16..31])
       +----------------------+----------------------+
8 rows |     Tile (0, 0)      |     Tile (0, 1)      | ---> Adder Tree ---> y[0..7]
       +----------------------+----------------------+
8 rows |     Tile (1, 0)      |     Tile (1, 1)      | ---> Adder Tree ---> y[8..15]
       +----------------------+----------------------+
```
* Each tile computes its partial sum for its 16 inputs in **1 clock cycle** ($L = 1$).
* Then, an **Inter-Tile Adder Reduction Tree** adds the results of `Tile (0,0)` and `Tile (0,1)` together to produce the final 32-bit outputs!
* What if a matrix size isn't an exact multiple of the tile size (like $19 \times 37$ divided into $8 \times 16$ tiles)? Our compiler automatically **pads the edges with zeros** to $24 \times 48$ ($3 \times 3 = 9$ tiles) and slices off the extra outputs!

### 6.2 Why Do We Compare Three Designs (`Design F`, `Design P`, `Design P_bus`)?
In Section 3.3 of your report, we compare **Design F** against **two** baselines. An examiner will definitely ask why! Here is the crystal-clear reason:

1. **Design F (Your Fixed-Weight Specialized Design):**
   * Weights are baked in as constants. Zero weight registers, zero weight input ports.
2. **Design P (Programmable Baseline with Weight Registers):**
   * Has $8 \times 8$ generic multipliers AND internal flip-flop registers to store the weights on-chip.
   * *What a skeptical examiner might say:* *"Hold on! Of course Design F is smaller than Design P—you deleted all the flip-flop registers that store the weights! Did you actually make the math multipliers smaller, or did you just delete the memory?"*
3. **Design P\_bus (Logic-Only Programmable Baseline — No Weight Registers!):**
   * To answer that skeptical examiner, we ALSO built **Design P\_bus**.
   * `Design P_bus` has **ZERO weight storage flip-flops**—it feeds the weights directly from external input wires (`w_data`) straight into the generic $8 \times 8$ multipliers.
   * **The Proof:** Even when compared against `Design P_bus` (which has zero weight registers), **Design F is STILL 3.14× to 3.79× smaller in logic cells!** This proves beyond a shadow of a doubt that constant specialization shrinks the *arithmetic multiplier circuits themselves*, not just the storage registers!

---

## Part 7: How We Proved Our Hardware is 100% Bug-Free (Section 5)

In hardware design, a single wrong wire ruins a million-dollar chip. How did we prove our compiler generates 100% mathematically perfect Verilog?

### 7.1 The Independent Reference Oracle (`verification/reference.py`)
We wrote a short, crystal-clear Python function using **NumPy 64-bit integers (`INT64`)** that computes $\mathbf{Y} = \mathbf{X}\mathbf{W}^T + \mathbf{b}$ directly from the original extracted weights (Listing 1 in your report).
* It shares **zero code** with the Verilog generator (no circular bugs).
* For every test, we feed the same input vectors $\mathbf{X}$ into both the **NumPy Reference Oracle** and the **Verilog Hardware Simulator** (Icarus Verilog & Verilator) and check that every single output matches **bit-for-bit (0 mismatches)**!

### 7.2 Multi-Tiered Verification (Levels L0, L1, L2, L3)
We tested our hardware at 4 increasing levels of difficulty:
* **Level L0 (Corner Cases):** Hand-calculated tiny tests ($1 \times 3$) with extreme numbers: all `0`s, all `+127`s, all `-127`s, and alternating signs.
* **Level L1 (1,080 Automated Sweep Cases):** Testing 1,080 combinations of tile sizes, odd matrix shapes, and weight distributions.
* **Level L2 (Real SmolLM2 Single Tiles):** Real weights from Layers 0, 14, and 29 of SmolLM2-135M across all 7 projections.
* **Level L3 (Multi-Tile Real Cores):** Full multi-tile grids ($16 \times 32$) using real SmolLM2-135M weights and real captured text activations—**0 mismatches across all 7 projections**!

### 7.3 Mutation Testing (M1 to M9 — 100% Kill Rate)
*Question: How do you know your testbench actually checks the outputs and isn't passing by accident?*
We used **Mutation Testing** (Section 5.3). We wrote a script that deliberately injects **9 different types of hardware bugs ("Mutants" M1–M9)** into the generated Verilog:
* **M1:** Add $+1$ to a single weight constant ($w_{i,j} \gets w_{i,j} + 1$).
* **M2:** Flip the $+/-$ sign of a weight.
* **M3:** Delete one product term from an adder tree.
* **M4:** Chop off the upper bits of the 32-bit accumulator.
* **M5:** Swap two input bit wires.
* **M6:** Delay the `valid_out` signal by 1 clock cycle.
* **M7:** Trigger reset early.
* **M8:** Forget to add the bias $b_i$.
* **M9:** Corrupt the output clamping logic.
**Result:** Our verification suite caught and failed ("killed") **9 out of 9 mutants (100.0% Mutation Kill Rate)** within a single simulation cycle!

---

## Part 8: Understanding All 4 Experiments, Figures & Tables (Section 6)

Let's walk through the 4 experiments in Section 6 so you can point at any graph or table in your report and explain it effortlessly.

### 8.1 Fairness Rules (Section 6.1)
Before comparing Design F and Design P in Yosys, we enforced strict fairness:
1. **Same Yosys commands:** Both used `synth -top <name> -flatten`.
2. **Same bitwidths (Rule N10):** 8-bit inputs, 8-bit weights, 16-bit products, 32-bit accumulators.
3. **Same Latency ($L = 1$ clock cycle):** Both have identical input/output registers.

---

### 8.2 Experiment E1: Area Scaling Across Tile Sizes (Figure 1 & Table 1)
* **What we did:** We synthesized tiles of increasing size—from $4 \times 16$ (64 weights) up to $32 \times 64$ (2,048 weights)—for `Design F`, `Design P`, and `Design P_bus`.
* **What Table 1 & Figure 1 show:**
  * At $8 \times 16$ (128 weights): `Design F` uses **22,764 cells**, while `Design P` uses **72,501 cells** (**3.18× smaller**).
  * At $16 \times 64$ (1,024 weights): `Design F` uses **155,648 cells**, while `Design P` uses **599,085 cells** (**3.85× smaller**).
  * **Flip-Flop Reduction:** At $16 \times 64$, `Design F` uses only **491 flip-flops** (just for the inputs/outputs!), whereas `Design P` needs **8,705 flip-flops** to store the 1,024 weights (**17.73× fewer flip-flops**!).
* **Why does even the output register shrink in Design F?** Because in Design F, Yosys knows the exact constant weights, so it calculates the maximum possible sum for each row and automatically deletes unused upper bits of the 32-bit accumulator ("Dead-Bit Pruning")!

---

### 8.3 Experiment E2: Weight Distribution Ablation (Figure 2)
* **The Question:** Does it matter that we used *real* SmolLM2-135M weights instead of random numbers?
* **What we did:** We took the exact same $8 \times 16$ tile (128 weights) and synthesized it with 6 different types of numbers:
  1. `all_zero` (all weights = 0): **3 cells total!** (99.99% of the chip disappears!).
  2. `one_hot` (mostly 0s, one 1 per row): **115 cells** (93.8% smaller).
  3. `real` (actual SmolLM2 `q_proj` weights): **22,764 cells**.
  4. `shuffled_real` (same real weights, randomly rearranged in position): **22,918 cells** (almost identical to `real`—proving position doesn't matter).
  5. `magnitude_matched` (same small sizes as real weights, random signs): **23,254 cells**.
  6. `uniform_random` (random integers spread evenly from `-127` to `+127`): **28,006 cells** (**23% larger than real weights!**).
* **The Big Insight (Claim C3):** Why are real AI weights **23% smaller** in hardware than random numbers?
  Because trained neural network weights follow a **Bell Curve (Gaussian distribution) centered at zero**! Most real weights are tiny numbers like `0, +1, -1, +2, -2, +3`, which have very few `1` bits in binary (low **Hamming Weight**) and need almost zero logic gates! Random numbers pick huge values like `119` (`01110111` in binary, six `1` bits) that require many adders.

---

### 8.4 Experiment E3: Critical Path Depth / Speed (Figure 3)
* **What is "Critical Path" (`ltp` — Longest Topological Path)?**
  In a hardware chip, electricity has to travel from the input flip-flop, through a chain of logic gates, to the output flip-flop before the next clock tick. The **longest chain of gates** on the chip is called the **Critical Path**.
  * **Shorter critical path = electricity finishes faster = you can run the chip at a higher clock frequency (GHz)!**
* **What Figure 3 shows:**
  Across all tile sizes, **Design F has a 20% to 32% shorter critical path** than Design P (e.g., 38 gate levels vs. 56 gate levels at $16 \times 64$, a ratio of **0.68**).
* **Why?** Because a generic $8 \times 8$ multiplier in Design P has a deep tree of partial-product adders, whereas constant shift-and-add in Design F only needs 1 or 2 adder levels!

---

### 8.5 Experiment E5: Full-Layer Extrapolation & Validation (Figure 4)
* **The Question:** How many logic gates would it take to build all 7 projections of an **entire SmolLM2-135M layer** (3,456 tiles)?
* **How we checked our formula first (Left panel of Figure 4):**
  Before estimating the whole layer, we tested our estimation formula on a 4-tile ($16 \times 32$) core and compared the formula's prediction against actual Yosys synthesis of the 4-tile core:
  * Our formula predicted within **10.82% error** for Design F (and **1.37% error** for Design P), proving our scaling math is trustworthy!
* **Full Layer 0 Result (Right panel of Figure 4):**
  * **Design F Full Layer 0:** **570.6 Million cells** (and 1.68M flip-flops).
  * **Design P Full Layer 0:** **2,075.2 Million cells** (and 30.08M flip-flops).
  * **Overall Full-Layer Savings:** **3.64× fewer logic cells** and **17.9× fewer flip-flops**!
* **Compiler Speed:** Our compiler generated the entire $576 \times 576$ `q_proj` layer (**548,223 lines of Verilog**, 30.02 MB, 331,776 weights) in just **0.54 seconds**, and audited every single constant in **0.85 seconds**!

---

## Part 9: Limitations & Part II Roadmap (Section 8)

Examiners love when a student honestly states the limitations of Part I and knows the exact plan for Part II:

1. **Pre-Layout Logic Synthesis Caveat:**
   In Part I, Yosys gave us the **exact count of logic gates and topological path depth** (Pre-Layout Synthesis). However, we have not yet placed those gates onto a physical 2D silicon floorplan or routed the physical metal wires between them (Place-and-Route).
2. **Linear Layers Only:**
   Part I compiles the 7 dense matrix multiplication projections (`q`, `k`, `v`, `o`, `gate`, `up`, `down`), which make up >90% of the compute. Non-linear operations like Softmax, RMSNorm, and RoPE positional embeddings are handled outside the core.
3. **Part II Roadmap:**
   In BTP Part II, we will feed our generated Verilog into the **OpenROAD ASIC physical design flow** using the open-source **SkyWater 130nm PDK** (Process Design Kit) to perform floorplanning, clock-tree synthesis, wire routing, and generate a physical **GDSII chip layout** with real post-layout area ($\text{mm}^2$), timing ($\text{MHz}$), and power ($\text{mW}$) measurements!

---

## Part 10: Quick-Reference Viva Cheat Sheet (Top 5 Questions)

| Examiner Question | Your Confident 30-Second Answer |
| :--- | :--- |
| **1. "What is the main contribution of your BTP in one sentence?"** | *"We built an automated hardware compiler (`LLMHW`) that compiles dense linear layers of SmolLM2-135M directly into synthesizable Verilog-2001 with weights hardcoded as compile-time constants, achieving a 3.0× to 3.85× logic cell reduction, 17.7× flip-flop reduction, and 20–32% critical path reduction with bit-exact verification."* |
| **2. "Did you just save area by deleting weight registers?"** | *"No! To prove that, we created `Design P_bus`, a baseline that has zero weight registers and takes weights as input wires. Even against `Design P_bus`, our `Design F` is 3.14× to 3.79× smaller because constant weights turn complex $8 \times 8$ multipliers into simple wire-shifts and 1-stage adders."* |
| **3. "Why do real LLM weights synthesize into 23% fewer gates than random INT8 numbers?"** | *"Because trained neural network weights follow a zero-centered Gaussian distribution. Most weights are small numbers near zero (`0, ±1, ±2`), which have low Hamming weight (very few active `1` bits in binary) and simplify to zero gates or simple bit-shifts, unlike uniform random numbers."* |
| **4. "How do you know your generated Verilog has no bugs or overflow?"** | *"First, our accumulators are 32-bit signed integers, and the worst-case dot product for $K=576$ is $576 \times 127^2 \approx 9.29 \times 10^6$, which is over 200× smaller than the 32-bit limit ($2.14 \times 10^9$). Second, we verified all 7 projections against an independent 64-bit NumPy oracle with 0 mismatches, and achieved a 100% kill rate across 9 hardware fault mutants."* |
| **5. "When would you use Design F in the real world vs. a normal GPU?"** | *"You use a GPU in data centers where models change daily. You use Design F in edge/embedded ASICs or FPGAs—like drones, medical wearables, or smart sensors—where a single compact model is frozen and battery power and silicon area are tightly constrained."* |
