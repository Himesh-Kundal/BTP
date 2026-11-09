# Model Facts: SmolLM2-135M (Part II Specification v2)

**Model ID:** `HuggingFaceTB/SmolLM2-135M`  
**Pinned Revision:** `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`  
**Reference Class:** `LlamaForCausalLM`  
**Contract Version:** Part II Architecture Contract v2 (`docs/part2/02_architecture_and_contracts.md`)

---

## 1. Global Architectural Hyperparameters

| Hyperparameter | Symbol | Empirical Value | Hardware Implications |
|---|---|---|---|
| **Transformer Layers** | $L$ | **30** | Sequencer loops 30 iterations per decode step |
| **Hidden Dimension** | $D$ | **576** | Width of residual stream and linear layer activations |
| **Intermediate FFN Dimension** | $D_{\text{ff}}$ | **1536** | Dimension of SwiGLU `gate_proj`, `up_proj`, and `down_proj` |
| **Query Attention Heads** | $n_q$ | **9** | 9 parallel query projections of dimension $d_h = 64$ |
| **Key / Value Attention Heads** | $n_{kv}$ | **3** | Grouped Query Attention (GQA) with 3:1 query-to-KV ratio |
| **Head Dimension** | $d_h$ | **64** | $576 / 9 = 64$; head dimension for RoPE and dot-products |
| **Vocabulary Size** | $V$ | **49,152** | Number of rows in embedding table and LM head |
| **Tied Word Embeddings** | `tie_word_embeddings` | **True** | `embed_tokens.weight` $\equiv$ `lm_head.weight` ($49152 \times 576$) |
| **Linear Biases** | `bias` | **False** | All 210 projections have 0 additive bias |
| **Normalization Type** | — | **RMSNorm** | 2 norms per layer + 1 final norm = 61 instances |
| **RMSNorm Epsilon** | $\epsilon$ | **$1.0 \times 10^{-5}$** | Added to variance before reciprocal square root |
| **Activation Function** | — | **SwiGLU (SiLU)** | Elementwise $\text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)$ |
| **RoPE Base Theta** | $\theta$ | **100,000.0** (`1e5`) | Base for inverse frequency geometric progression |
| **RoPE Convention** | — | **Rotate-Half** | $R(x) = [-x_{32:63}, x_{0:31}]$ |
| **Maximum Context Window** | — | 8,192 (model) | Hardware synthesis target: **$\text{MAX\_CTX} = 256$** |

---

## 2. Complete Parameter Count & Memory Breakdown

SmolLM2-135M contains exactly **134,515,008 parameters** (~134.5M):

```
SmolLM2-135M Parameter Hierarchy:
├── Tied Word Embedding / LM Head: 49,152 × 576 ............. 28,311,552 params
└── 30 Decoder Transformer Layers (3,538,944 params each) ... 106,168,320 params
    ├── Self-Attention Block (884,736 params):
    │   ├── q_proj: 576 × 576 .................................. 331,776 params
    │   ├── k_proj: 192 × 576 .................................. 110,592 params
    │   ├── v_proj: 192 × 576 .................................. 110,592 params
    │   └── o_proj: 576 × 576 .................................. 331,776 params
    ├── Feed-Forward Network / SwiGLU Block (2,654,208 params):
    │   ├── gate_proj: 1536 × 576 .............................. 884,736 params
    │   ├── up_proj: 1536 × 576 ................................ 884,736 params
    │   └── down_proj: 576 × 1536 .............................. 884,736 params
    └── RMSNorm Weights (61 instances × 576 params) ............... 35,136 params
        ├── input_layernorm: 30 × 576 ........................... 17,280 params
        ├── post_attention_layernorm: 30 × 576 .................. 17,280 params
        └── final_layernorm: 1 × 576 ............................... 576 params
```

At INT8 precision (1 byte per parameter), the immutable weight ROM budget is:
- **Tied Embedding / LM Head ROM:** $28.31\text{ MB}$
- **30 Transformer Layers ROM:** $106.17\text{ MB}$
- **RMSNorm Scales ROM ($\gamma \in \text{INT16}$):** $70.27\text{ KB}$
- **Total Weight ROM Storage:** $\approx \mathbf{134.55\text{ MB}}$

---

## 3. Operator Specifications & Hardware Contracts

### 3.1 RMSNorm (Root Mean Square Layer Normalization)
- **Input:** $x \in \mathbb{Z}^{D}$ (residual stream, wide integer representation)
- **Mathematical Form:**
  $$\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=0}^{D-1} x_i^2 + \epsilon}$$
  $$\bar{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$$
- **Hardware Realization:**
  1. Mean of squares: accumulate $x_i^2$ into a 48-bit register, right-shift by 9 (divide by 576 approximated via dyadic multiplication by $M_{\text{inv\_d}} = 7282, s = 22$).
  2. Add $\epsilon_{\text{int}}$.
  3. Lookup reciprocal square root in `rsqrt` LUT (mantissa indexed, exponent normalized).
  4. Multiply by learned per-channel scale $\gamma_i$ and requantize to INT8.

### 3.2 Rotary Position Embedding (RoPE)
- **Input:** Query vectors $Q \in \mathbb{Z}^{n_q \times d_h}$, Key vectors $K \in \mathbb{Z}^{n_{kv} \times d_h}$ where $d_h = 64$.
- **Frequencies:**
  $$\theta_k = \theta^{-2k / d_h} = 100000^{-2k / 64} \quad \text{for } k \in [0, 31]$$
- **Rotation Rule (Rotate-Half):**
  For token position $m \in [0, \text{MAX\_CTX}-1]$:
  $$\begin{bmatrix} x_{\text{out}}[k] \\ x_{\text{out}}[k + 32] \end{bmatrix} = \begin{bmatrix} \cos(m \theta_k) & -\sin(m \theta_k) \\ \sin(m \theta_k) & \cos(m \theta_k) \end{bmatrix} \begin{bmatrix} x_{\text{in}}[k] \\ x_{\text{in}}[k + 32] \end{bmatrix}$$
- **Hardware Realization:** Precomputed $\cos(m \theta_k)$ and $\sin(m \theta_k)$ tables stored in a $256 \times 32$ ROM in Q1.15 fixed-point format.

### 3.3 Grouped Query Attention (GQA) & KV Cache
- **Query / Key Heads:** $n_q = 9, n_{kv} = 3$.
- **Mapping:** Query head $q \in [0, 8]$ pairs with Key/Value head $k = \lfloor q / 3 \rfloor$.
  - Heads 0, 1, 2 $\rightarrow$ KV head 0
  - Heads 3, 4, 5 $\rightarrow$ KV head 1
  - Heads 6, 7, 8 $\rightarrow$ KV head 2
- **Score Scaling:** Pre-multiplication by $\frac{1}{\sqrt{d_h}} = \frac{1}{\sqrt{64}} = \frac{1}{8} = 0.125$ (arithmetic right shift by 3).
- **KV Cache Memory:**
  - Shape: $[L=30][n_{kv}=3][\text{MAX\_CTX}=256][d_h=64]$ for Keys and Values.
  - Sized at INT8: $30 \times 3 \times 256 \times 64 \times 1\text{ byte} = 1,474,560\text{ bytes}$ ($\approx 1.41\text{ MB}$ each for K and V; total $\mathbf{2.81\text{ MB}}$ dual-port SRAM).

### 3.4 Softmax
- **Input:** Scaled attention logits $S \in \mathbb{Z}^{1 \times (\text{pos}+1)}$.
- **Algorithm (Base-2 Decomposition):**
  1. Find maximum logit: $m = \max_{0 \le j \le \text{pos}} S_j$.
  2. Max-subtracted difference: $\Delta_j = S_j - m \le 0$.
  3. Base-2 exponential approximation: $e^{\Delta_j} = 2^{\Delta_j \cdot \log_2 e} = 2^{\text{integer}} \times 2^{\text{fraction}}$.
  4. Fraction evaluated via a 256-entry lookup table.
  5. Sum normalizer: $\Sigma = \sum_j 2^{\Delta_j \cdot \log_2 e}$.
  6. Multiply by reciprocal table $\frac{1}{\Sigma}$ and requantize to 8-bit attention probabilities.

### 3.5 SwiGLU Activation
- **Input:** $x \in \mathbb{Z}^{D_{\text{ff}}}$ from `gate_proj`, $u \in \mathbb{Z}^{D_{\text{ff}}}$ from `up_proj`.
- **Form:** $y = \text{SiLU}(x) \odot u$ where $\text{SiLU}(x) = x \cdot \sigma(x)$.
- **Hardware Realization:** Symmetric 512-entry lookup table for $\text{SiLU}(x)$, followed by 16-bit $\times$ 8-bit multiplier and dyadic requantization to INT8 for `down_proj`.

### 3.6 ArgMax
- **Input:** Unscaled logit stream from LM head projection ($49,152$ entries).
- **Scale Alignment:** Per Numeric Rule N2-13, per-row scale factor applied before comparisons:
  $$\tilde{z}_v = \text{rshift\_rnd}(z_v \cdot M_{\text{scale}}, s_{\text{scale}})$$
- **Comparator:** Maintains current running maximum $\tilde{z}_{\max}$ and token index $t_{\max} \in [0, 49151]$. Strict greater-than ($>$) ensures deterministic tie-breaking to the lowest token ID.
