# Model Facts: SmolLM2-135M

This document records the empirical inspection and verification of `HuggingFaceTB/SmolLM2-135M` at pinned revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

## 1. Global Model Parameters

| Parameter | Inspected Value | Expected in `docs/02` §1 | Reconciled Status |
| :--- | :--- | :--- | :--- |
| **Model ID** | `HuggingFaceTB/SmolLM2-135M` | `SmolLM2-135M` | **MATCH** |
| **Revision** | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | pinned | **MATCH** |
| **Architecture** | `LlamaForCausalLM` | Decoder-only Transformer | **MATCH** |
| **Hidden Size ($D$)** | `576` | `576` | **MATCH** |
| **Intermediate Size** | `1536` | `1536` | **MATCH** |
| **Number of Layers** | `30` | `30` | **MATCH** |
| **Attention Heads** | `9` | `9` | **MATCH** |
| **Key/Value Heads** | `3` (Grouped Query Attn) | `3` | **MATCH** |
| **Head Dimension** | `64` ($576 / 9$) | `64` | **MATCH** |
| **Linear Biases** | `False` (0 bias vectors) | None | **MATCH** |
| **Tied Embeddings** | `True` (`lm_head` == `embed_tokens`) | Tied | **MATCH** |
| **Vocabulary Size** | `49152` | N/A (excluded) | **MATCH** |

---

## 2. Projection Types and Shapes

Each of the 30 decoder layers contains exactly 7 dense linear projections:

| Projection | Category | $N$ (out_features) | $K$ (in_features) | Bias | Parameters per layer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `q_proj` | Self-Attention | 576 | 576 | None | 331,776 |
| `k_proj` | Self-Attention | 192 | 576 | None | 110,592 |
| `v_proj` | Self-Attention | 192 | 576 | None | 110,592 |
| `o_proj` | Self-Attention | 576 | 576 | None | 331,776 |
| `gate_proj` | MLP / SwiGLU | 1536 | 576 | None | 884,736 |
| `up_proj` | MLP / SwiGLU | 1536 | 576 | None | 884,736 |
| `down_proj` | MLP / SwiGLU | 576 | 1536 | None | 884,736 |

- **Parameters per layer**: 3,538,944
- **Total Linear Parameters across 30 layers**: $3,538,944 \times 30 = 106,168,320$ (~106.17 M)
- **Total Linear Modules**: $30 \times 7 = 210$ modules

---

## 3. Reconciliation with `docs/02` §1

All shapes, dimensions, head counts, and bias absence exactly match the initial architectural expectations in [`docs/02_architecture_and_contracts.md`](file:///home/himesh/BTP/docs/02_architecture_and_contracts.md) §1.
**No deviations or discrepancies exist; no ADR is required.**
