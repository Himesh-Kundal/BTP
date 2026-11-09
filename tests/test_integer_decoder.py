"""Verification tests for Integer Decoder Oracle v0 and Scorer (Task UA.4)."""

import numpy as np
import pytest

from numerics.decoder import (
    IntegerDecoder,
    IntegerLayerWeights,
    IntegerModelConfig,
    IntegerModelWeights,
)


@pytest.fixture(scope="module")
def synthetic_integer_model():
    """Create a fully quantized, deterministic tiny integer model for oracle testing."""
    cfg = IntegerModelConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_ctx=32,
        rope_theta=10000.0,
    )

    rng = np.random.default_rng(42)

    # Helper to generate INT8 matrices and dyadic scales
    def rand_int8(shape):
        return rng.integers(-120, 120, size=shape, dtype=np.int8)

    layers = []
    for _ in range(cfg.num_hidden_layers):
        layers.append(
            IntegerLayerWeights(
                q_proj=rand_int8((cfg.hidden_size, cfg.hidden_size)),
                k_proj=rand_int8((cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)),
                v_proj=rand_int8((cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)),
                o_proj=rand_int8((cfg.hidden_size, cfg.hidden_size)),
                gate_proj=rand_int8((cfg.intermediate_size, cfg.hidden_size)),
                up_proj=rand_int8((cfg.intermediate_size, cfg.hidden_size)),
                down_proj=rand_int8((cfg.hidden_size, cfg.intermediate_size)),
                q_M=rng.integers(1000, 5000, size=cfg.hidden_size),
                q_s=12,
                k_M=rng.integers(1000, 5000, size=cfg.num_key_value_heads * cfg.head_dim),
                k_s=12,
                v_M=rng.integers(1000, 5000, size=cfg.num_key_value_heads * cfg.head_dim),
                v_s=12,
                o_M=rng.integers(1000, 5000, size=cfg.hidden_size),
                o_s=12,
                gate_M=rng.integers(1000, 5000, size=cfg.intermediate_size),
                gate_s=12,
                up_M=rng.integers(1000, 5000, size=cfg.intermediate_size),
                up_s=12,
                down_M=rng.integers(1000, 5000, size=cfg.hidden_size),
                down_s=12,
                in_norm_gamma=rng.integers(100, 300, size=cfg.hidden_size),
                in_norm_M=1024,
                in_norm_s=10,
                post_norm_gamma=rng.integers(100, 300, size=cfg.hidden_size),
                post_norm_M=1024,
                post_norm_s=10,
            )
        )

    weights = IntegerModelWeights(
        embed_table=rand_int8((cfg.vocab_size, cfg.hidden_size)),
        embed_M=1024,
        embed_s=8,
        layers=layers,
        final_norm_gamma=rng.integers(100, 300, size=cfg.hidden_size),
        final_norm_M=1024,
        final_norm_s=10,
        lm_head=rand_int8((cfg.vocab_size, cfg.hidden_size)),
        lm_head_M=rng.integers(1000, 5000, size=cfg.vocab_size),
        lm_head_s=12,
    )

    decoder = IntegerDecoder(cfg, weights)
    return decoder


def test_integer_decoder_step_vs_scorer_exact_50_sequences(synthetic_integer_model):
    """Verify that teacher-forcing scorer matches step-by-step decoder bit-exactly on >= 50 sequences."""
    decoder = synthetic_integer_model
    rng = np.random.default_rng(12345)

    # Generate 50 diverse sequences with lengths from 2 to 16
    for seq_idx in range(50):
        seq_len = rng.integers(2, 16)
        seq_tokens = rng.integers(0, decoder.cfg.vocab_size, size=seq_len).tolist()

        # Method 1: Scorer
        scored_tokens = decoder.score_sequence(seq_tokens)

        # Method 2: Step-by-step
        decoder.reset()
        stepped_tokens = []
        for pos, tok in enumerate(seq_tokens):
            _logits, next_id = decoder.decode_step(tok, pos)
            stepped_tokens.append(next_id)

        # Assert 100% bit-exact equivalence
        assert scored_tokens == stepped_tokens, (
            f"Discrepancy on sequence {seq_idx} (len {seq_len}):\n"
            f"Scored : {scored_tokens}\n"
            f"Stepped: {stepped_tokens}"
        )


def test_integer_decoder_determinism(synthetic_integer_model):
    """Verify that identical inputs produce 100% byte-identical logits across decoder resets."""
    decoder = synthetic_integer_model
    prompt = [5, 12, 23, 42]

    decoder.reset()
    logits1 = []
    for pos, tok in enumerate(prompt):
        l, _ = decoder.decode_step(tok, pos)
        logits1.append(l)

    decoder.reset()
    logits2 = []
    for pos, tok in enumerate(prompt):
        l, _ = decoder.decode_step(tok, pos)
        logits2.append(l)

    for l1, l2 in zip(logits1, logits2, strict=True):
        np.testing.assert_array_equal(l1, l2)
