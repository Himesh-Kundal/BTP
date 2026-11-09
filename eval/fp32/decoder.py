"""NumPy fp32 Reference Decoder for SmolLM2 and Tiny Model (Task UA.1).

Defines the mathematical real-valued reference for autoregressive transformer decoding:
- RMSNorm
- Rotary Position Embeddings (RoPE, rotate-half, theta=100000.0 or configured)
- Grouped Query Attention (GQA) with dynamic KV cache
- SwiGLU activation (SiLU * Up)
- Residual stream additions
- Tied LM Head projection and ArgMax
- Validated to match HuggingFace PyTorch logits within <= 1e-3 max abs diff.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DecoderConfig:
    """Hyperparameters for autoregressive transformer decoder."""
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    vocab_size: int = 49152
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    head_dim: int = 64

    @classmethod
    def from_dict(cls, d: dict) -> "DecoderConfig":
        hd = d.get("head_dim") or (d["hidden_size"] // d["num_attention_heads"])
        return cls(
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            vocab_size=d["vocab_size"],
            max_position_embeddings=d.get("max_position_embeddings", 8192),
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-5)),
            rope_theta=float(d.get("rope_theta", 100000.0)),
            head_dim=hd,
        )


@dataclass
class LayerWeights:
    """Weights for a single transformer decoder layer."""
    q_proj: np.ndarray  # [num_attention_heads * head_dim, hidden_size]
    k_proj: np.ndarray  # [num_key_value_heads * head_dim, hidden_size]
    v_proj: np.ndarray  # [num_key_value_heads * head_dim, hidden_size]
    o_proj: np.ndarray  # [hidden_size, num_attention_heads * head_dim]
    gate_proj: np.ndarray  # [intermediate_size, hidden_size]
    up_proj: np.ndarray  # [intermediate_size, hidden_size]
    down_proj: np.ndarray  # [hidden_size, intermediate_size]
    input_layernorm: np.ndarray  # [hidden_size]
    post_attention_layernorm: np.ndarray  # [hidden_size]


@dataclass
class ModelWeights:
    """All weights required for full autoregressive decode."""
    embed_tokens: np.ndarray  # [vocab_size, hidden_size]
    layers: list[LayerWeights]
    final_norm: np.ndarray  # [hidden_size]
    lm_head: np.ndarray  # [vocab_size, hidden_size] (tied or separate)


# ==============================================================================
# Primitive Operations (Pure NumPy)
# ==============================================================================


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute Root Mean Square Layer Normalization (RMSNorm)."""
    # x: [..., hidden_size], weight: [hidden_size]
    var = np.mean(x * x, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(var + eps)
    return x * inv_rms * weight


def compute_rope_freqs(head_dim: int, theta: float) -> np.ndarray:
    """Compute inverse frequency vector for RoPE."""
    # head_dim must be even
    idx = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = 1.0 / (theta ** (idx / head_dim))
    return inv_freq


def apply_rope(
    x: np.ndarray, pos: int, inv_freq: np.ndarray
) -> np.ndarray:
    """Apply rotate-half rotary position embedding to query or key head vectors.
    
    x: shape [num_heads, head_dim]
    """
    _num_heads, head_dim = x.shape
    half = head_dim // 2

    # Rotate-half convention: [-x2, x1]
    x1 = x[:, :half]
    x2 = x[:, half:]

    # Angles for position pos: [half]
    angles = pos * inv_freq
    cos_m = np.cos(angles)[np.newaxis, :]
    sin_m = np.sin(angles)[np.newaxis, :]

    out1 = x1 * cos_m - x2 * sin_m
    out2 = x2 * cos_m + x1 * sin_m
    return np.concatenate([out1, out2], axis=-1)


def silu(x: np.ndarray) -> np.ndarray:
    """Sigmoid Linear Unit activation: x / (1 + exp(-x))."""
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


# ==============================================================================
# Autoregressive Decoder State & Step Execution
# ==============================================================================


class KVCache:
    """Dynamic Key-Value cache storing states across decode steps."""

    def __init__(self, num_layers: int, num_kv_heads: int, max_ctx: int, head_dim: int):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_ctx = max_ctx
        self.head_dim = head_dim
        # Shape: [num_layers, num_kv_heads, max_ctx, head_dim]
        self.k = np.zeros((num_layers, num_kv_heads, max_ctx, head_dim), dtype=np.float32)
        self.v = np.zeros((num_layers, num_kv_heads, max_ctx, head_dim), dtype=np.float32)

    def write(self, layer_idx: int, pos: int, k_vec: np.ndarray, v_vec: np.ndarray) -> None:
        """Store k and v head vectors for current position."""
        self.k[layer_idx, :, pos, :] = k_vec
        self.v[layer_idx, :, pos, :] = v_vec

    def get(self, layer_idx: int, pos: int) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve historical keys and values up to position pos (inclusive)."""
        # Returns: [num_kv_heads, pos + 1, head_dim]
        return self.k[layer_idx, :, : pos + 1, :], self.v[layer_idx, :, : pos + 1, :]


class NumPyFP32Decoder:
    """Autoregressive transformer decoder implemented in NumPy."""

    def __init__(self, config: DecoderConfig, weights: ModelWeights, max_ctx: int = 256):
        self.cfg = config
        self.weights = weights
        self.max_ctx = max_ctx
        self.inv_freq = compute_rope_freqs(config.head_dim, config.rope_theta)
        self.kv_cache = KVCache(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            max_ctx=max_ctx,
            head_dim=config.head_dim,
        )

    def reset(self) -> None:
        """Reset KV cache and internal state."""
        self.kv_cache = KVCache(
            num_layers=self.cfg.num_hidden_layers,
            num_kv_heads=self.cfg.num_key_value_heads,
            max_ctx=self.max_ctx,
            head_dim=self.cfg.head_dim,
        )

    def decode_step(self, token_id: int, pos: int) -> tuple[np.ndarray, int]:
        """Execute a single token decode step at position pos.
        
        Returns:
            logits: [vocab_size] float32 array
            next_token_id: int greedy argmax
        """
        # 1. Embedding Gather
        h = self.weights.embed_tokens[token_id].astype(np.float32).copy()

        gqa_group = self.cfg.num_attention_heads // self.cfg.num_key_value_heads
        scale = 1.0 / np.sqrt(self.cfg.head_dim)

        # 2. Iterate through 30 transformer layers
        for l_idx, layer in enumerate(self.weights.layers):
            # Attention block
            norm_a = rms_norm(h, layer.input_layernorm, eps=self.cfg.rms_norm_eps)

            # Projections
            q = norm_a @ layer.q_proj.T  # [num_attention_heads * head_dim]
            k = norm_a @ layer.k_proj.T  # [num_key_value_heads * head_dim]
            v = norm_a @ layer.v_proj.T  # [num_key_value_heads * head_dim]

            # Reshape into heads
            q = q.reshape(self.cfg.num_attention_heads, self.cfg.head_dim)
            k = k.reshape(self.cfg.num_key_value_heads, self.cfg.head_dim)
            v = v.reshape(self.cfg.num_key_value_heads, self.cfg.head_dim)

            # Apply RoPE
            q = apply_rope(q, pos, self.inv_freq)
            k = apply_rope(k, pos, self.inv_freq)

            # Store in KV cache
            self.kv_cache.write(l_idx, pos, k, v)
            k_hist, v_hist = self.kv_cache.get(l_idx, pos)  # [n_kv, pos+1, hd]

            # Grouped Query Attention
            attn_out = np.zeros((self.cfg.num_attention_heads, self.cfg.head_dim), dtype=np.float32)
            for q_idx in range(self.cfg.num_attention_heads):
                kv_idx = q_idx // gqa_group
                k_head = k_hist[kv_idx]  # [pos+1, hd]
                v_head = v_hist[kv_idx]  # [pos+1, hd]
                q_head = q[q_idx]  # [hd]

                # Dot-product scores: [pos+1]
                scores = (q_head @ k_head.T) * scale
                probs = softmax(scores)
                attn_out[q_idx] = probs @ v_head

            # Flatten and project output
            attn_flat = attn_out.reshape(-1)
            o = attn_flat @ layer.o_proj.T
            h = h + o  # Residual addition

            # Feed-Forward / SwiGLU block
            norm_m = rms_norm(h, layer.post_attention_layernorm, eps=self.cfg.rms_norm_eps)
            gate = norm_m @ layer.gate_proj.T
            up = norm_m @ layer.up_proj.T
            swiglu_out = silu(gate) * up
            down = swiglu_out @ layer.down_proj.T
            h = h + down  # Residual addition

        # 3. Final RMSNorm
        f = rms_norm(h, self.weights.final_norm, eps=self.cfg.rms_norm_eps)

        # 4. LM Head (Tied or separate)
        logits = f @ self.weights.lm_head.T

        # 5. Greedy ArgMax (lowest index tie-break)
        next_token_id = int(np.argmax(logits))

        return logits, next_token_id

    def generate(self, prompt_tokens: list[int], max_new_tokens: int = 16) -> list[int]:
        """Autoregressively generate new tokens starting from prompt_tokens."""
        self.reset()
        # Prefill prompt tokens through step interface
        for pos, tok in enumerate(prompt_tokens):
            _logits, next_tok = self.decode_step(tok, pos)

        output_tokens = []
        curr_tok = next_tok
        curr_pos = len(prompt_tokens)

        for _ in range(max_new_tokens):
            output_tokens.append(curr_tok)
            _logits, curr_tok = self.decode_step(curr_tok, curr_pos)
            curr_pos += 1

        return output_tokens


def load_model_weights_from_hf(model) -> tuple[DecoderConfig, ModelWeights]:
    """Extract NumPy ModelWeights and DecoderConfig from Hugging Face model instance."""
    cfg_dict = model.config.to_dict()
    if hasattr(model.config, "rope_parameters") and "rope_theta" in model.config.rope_parameters:
        cfg_dict["rope_theta"] = model.config.rope_parameters["rope_theta"]
    elif getattr(model.config, "rope_theta", None) is not None:
        cfg_dict["rope_theta"] = model.config.rope_theta

    cfg = DecoderConfig.from_dict(cfg_dict)

    embed_w = model.model.embed_tokens.weight.detach().float().cpu().numpy()
    lm_head_w = model.lm_head.weight.detach().float().cpu().numpy()
    final_norm_w = model.model.norm.weight.detach().float().cpu().numpy()

    layers = []
    for l in model.model.layers:
        layers.append(
            LayerWeights(
                q_proj=l.self_attn.q_proj.weight.detach().float().cpu().numpy(),
                k_proj=l.self_attn.k_proj.weight.detach().float().cpu().numpy(),
                v_proj=l.self_attn.v_proj.weight.detach().float().cpu().numpy(),
                o_proj=l.self_attn.o_proj.weight.detach().float().cpu().numpy(),
                gate_proj=l.mlp.gate_proj.weight.detach().float().cpu().numpy(),
                up_proj=l.mlp.up_proj.weight.detach().float().cpu().numpy(),
                down_proj=l.mlp.down_proj.weight.detach().float().cpu().numpy(),
                input_layernorm=l.input_layernorm.weight.detach().float().cpu().numpy(),
                post_attention_layernorm=l.post_attention_layernorm.weight.detach().float().cpu().numpy(),
            )
        )

    weights = ModelWeights(
        embed_tokens=embed_w,
        layers=layers,
        final_norm=final_norm_w,
        lm_head=lm_head_w,
    )

    return cfg, weights
