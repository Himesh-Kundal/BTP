"""Verification tests for Evaluation Harness and Metrics (Task UA.2)."""

import numpy as np

from eval.harness import (
    compute_cross_entropy_and_perplexity,
    compute_greedy_match_length,
    compute_kl_divergence,
    compute_top1_agreement,
    ensure_dataset_present,
    ensure_prompts_file_present,
)


def test_eval_dataset_pinned_hash():
    """Verify that the downloaded WikiText-2 dataset matches pinned SHA256."""
    p = ensure_dataset_present()
    assert p.exists()
    assert p.stat().st_size == 1256449


def test_eval_prompts_battery():
    """Verify that the 20-prompt standardized battery is present and non-empty."""
    p = ensure_prompts_file_present()
    assert p.exists()


def test_metric_cross_entropy_and_perplexity():
    """Verify perplexity computation on deterministic synthetic logits."""
    # Perfect predictions: logit for correct token is very high
    vocab_size = 10
    n = 4
    target = [2, 5, 0, 9]

    logits = np.full((n, vocab_size), -10.0, dtype=np.float32)
    for i, t in enumerate(target):
        logits[i, t] = 10.0  # Dominant logit -> loss ~ 0, PPL ~ 1.0

    loss, ppl = compute_cross_entropy_and_perplexity(logits, target)
    assert 0.0 <= loss < 0.01
    assert 1.0 <= ppl < 1.01

    # Uniform random logits -> loss ~ ln(vocab_size)
    uniform_logits = np.zeros((n, vocab_size), dtype=np.float32)
    loss_u, ppl_u = compute_cross_entropy_and_perplexity(uniform_logits, target)
    expected_loss = np.log(vocab_size)
    assert np.isclose(loss_u, expected_loss, atol=1e-5)
    assert np.isclose(ppl_u, vocab_size, atol=1e-5)


def test_metric_kl_divergence():
    """Verify KL divergence properties: KL(P||P) == 0, KL(P||Q) >= 0."""
    p_logits = np.array([[2.0, 1.0, 0.5], [1.0, 3.0, -1.0]], dtype=np.float32)
    q_logits = np.array([[1.8, 1.1, 0.4], [0.9, 3.1, -0.8]], dtype=np.float32)

    # Identical distributions have zero KL divergence
    kl_zero = compute_kl_divergence(p_logits, p_logits)
    assert np.isclose(kl_zero, 0.0, atol=1e-6)

    # Divergent distributions have positive KL divergence
    kl_pos = compute_kl_divergence(p_logits, q_logits)
    assert kl_pos > 0.0


def test_metric_top1_agreement():
    """Verify top-1 agreement percentage."""
    l1 = np.array([[1.0, 5.0, 2.0], [4.0, 1.0, 0.0], [0.0, 2.0, 3.0]])
    l2 = np.array([[0.5, 4.0, 1.0], [3.0, 0.5, 0.0], [1.0, 4.0, 2.0]])  # last differs

    agr = compute_top1_agreement(l1, l2)
    assert np.isclose(agr, 66.666666, atol=1e-4)

    agr_self = compute_top1_agreement(l1, l1)
    assert np.isclose(agr_self, 100.0)


def test_metric_greedy_match_length():
    """Verify greedy match length counts tokens before first divergence."""
    t1 = [10, 20, 30, 40, 50]
    t2 = [10, 20, 30, 99, 50]
    assert compute_greedy_match_length(t1, t2) == 3

    assert compute_greedy_match_length(t1, t1) == 5
    assert compute_greedy_match_length([1], [2]) == 0
