"""Verification tests for NumPy FP32 reference decoder against Hugging Face (Task UA.1)."""

import numpy as np
import pytest
import torch

from eval.fp32.decoder import NumPyFP32Decoder, load_model_weights_from_hf
from frontend.tiny_model import generate_tiny_model


@pytest.fixture(scope="module")
def tiny_hf_and_numpy():
    """Create shared tiny model instance and NumPy decoder."""
    hf_model = generate_tiny_model(seed=42)
    cfg, weights = load_model_weights_from_hf(hf_model)
    decoder = NumPyFP32Decoder(cfg, weights)
    return hf_model, decoder


# 20 diverse prompts covering short, long, single token, and boundary values
PROMPTS_20 = [
    [0],
    [255],
    [1, 2, 3],
    [42, 100, 200],
    [10, 20, 30, 40],
    [5, 5, 5],
    [128, 64, 32, 16],
    [7, 13, 29, 31, 37],
    [99, 101, 103, 107],
    [1, 10, 100],
    [15, 88, 142],
    [50, 150, 250],
    [2, 4, 8, 16, 32],
    [11, 22, 33, 44, 55],
    [3, 9, 27, 81],
    [100, 200, 50, 25],
    [12, 34, 56, 78],
    [254, 253, 252],
    [1, 255, 1, 255],
    [17, 19, 23, 29, 31, 37, 41, 43],
]


@pytest.mark.parametrize("prompt", PROMPTS_20)
def test_fp32_decoder_tiny_model_against_hf(tiny_hf_and_numpy, prompt):
    """Verify NumPy FP32 decoder matches Hugging Face within <= 1e-3 and argmax on >= 20 prompts."""
    hf_model, decoder = tiny_hf_and_numpy
    decoder.reset()

    # Step-by-step decode
    for pos, tok in enumerate(prompt):
        np_logits, np_next = decoder.decode_step(tok, pos)

    # Hugging Face reference
    with torch.no_grad():
        t_in = torch.tensor([prompt], dtype=torch.long)
        hf_out = hf_model(t_in)
        hf_logits = hf_out.logits[0, -1].cpu().numpy()
        hf_next = int(torch.argmax(hf_out.logits[0, -1]).item())

    max_diff = float(np.max(np.abs(np_logits - hf_logits)))
    assert max_diff <= 1e-3, f"Max diff {max_diff} exceeded 1e-3 on prompt {prompt}"
    assert np_next == hf_next, f"ArgMax mismatch on prompt {prompt}: np={np_next}, hf={hf_next}"


def test_fp32_decoder_autoregressive_generation(tiny_hf_and_numpy):
    """Verify multi-step autoregressive generation matches Hugging Face greedy decoding."""
    hf_model, decoder = tiny_hf_and_numpy
    prompt = [10, 42, 99]
    max_new_tokens = 8

    # NumPy decoder generation
    np_gen = decoder.generate(prompt, max_new_tokens=max_new_tokens)

    # Hugging Face generation
    with torch.no_grad():
        t_in = torch.tensor([prompt], dtype=torch.long)
        hf_gen_tensor = hf_model.generate(
            t_in,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=None,
        )
        hf_gen = hf_gen_tensor[0, len(prompt):].cpu().tolist()

    assert np_gen == hf_gen, f"Generated tokens differ: np={np_gen}, hf={hf_gen}"
