//! Pure-Rust FP32 Reference Decoder & Quality Evaluation Metrics (Cross-Entropy, Perplexity, KL Divergence, Top-1 Agreement, Greedy Match Length) (Phase R7).

use rayon::prelude::*;

/// Compute mean cross-entropy loss and perplexity `exp(CE)` over `[N, vocab_size]` logits and `target_tokens` of length `N`.
pub fn compute_cross_entropy_and_perplexity(
    logits: &[Vec<f64>],
    target_tokens: &[usize],
) -> (f64, f64) {
    assert_eq!(logits.len(), target_tokens.len());
    let n = target_tokens.len();
    if n == 0 {
        return (0.0, 1.0);
    }

    let total_nll: f64 = logits
        .par_iter()
        .zip(target_tokens.par_iter())
        .map(|(row, &target)| {
            let max_v = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = row.iter().map(|&x| (x - max_v).exp()).sum();
            let log_sum_exp = sum_exp.ln();
            let log_prob = (row[target] - max_v) - log_sum_exp;
            -log_prob
        })
        .sum();

    let mean_loss = total_nll / (n as f64);
    let ppl = mean_loss.exp();
    (mean_loss, ppl)
}

/// Compute mean KL divergence `KL(P_fp32 || Q_int)` across token positions.
pub fn compute_kl_divergence(p_logits: &[Vec<f64>], q_logits: &[Vec<f64>]) -> f64 {
    assert_eq!(p_logits.len(), q_logits.len());
    let n = p_logits.len();
    if n == 0 {
        return 0.0;
    }

    let eps = 1e-12f64;
    let total_kl: f64 = p_logits
        .par_iter()
        .zip(q_logits.par_iter())
        .map(|(p_row, q_row)| {
            let p_max = p_row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let q_max = q_row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let p_sum: f64 = p_row.iter().map(|&x| (x - p_max).exp()).sum();
            let q_sum: f64 = q_row.iter().map(|&x| (x - q_max).exp()).sum();

            let mut kl_pos = 0.0f64;
            for (&pv, &qv) in p_row.iter().zip(q_row.iter()) {
                let p_prob = ((pv - p_max).exp() / p_sum).clamp(eps, 1.0);
                let q_prob = ((qv - q_max).exp() / q_sum).clamp(eps, 1.0);
                kl_pos += p_prob * (p_prob / q_prob).ln();
            }
            kl_pos
        })
        .sum();

    total_kl / (n as f64)
}

/// Compute percentage of positions where argmax matches exactly (`0.0 ..= 100.0`).
pub fn compute_top1_agreement(logits1: &[Vec<f64>], logits2: &[Vec<f64>]) -> f64 {
    assert_eq!(logits1.len(), logits2.len());
    let n = logits1.len();
    if n == 0 {
        return 100.0;
    }
    let argmax = |row: &[f64]| -> usize {
        let mut best_idx = 0usize;
        let mut best_val = row[0];
        for (i, &v) in row.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        best_idx
    };
    let matches = logits1
        .iter()
        .zip(logits2.iter())
        .filter(|(r1, r2)| argmax(r1) == argmax(r2))
        .count();
    (matches as f64) * 100.0 / (n as f64)
}

/// Compute number of consecutive generated tokens matching before first divergence.
pub fn compute_greedy_match_length(tokens1: &[usize], tokens2: &[usize]) -> usize {
    tokens1
        .iter()
        .zip(tokens2.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_metrics_basic() {
        let p = vec![vec![2.0, 1.0, 0.1], vec![0.1, 3.0, 0.2]];
        let q = vec![vec![2.0, 1.0, 0.1], vec![0.1, 3.0, 0.2]];
        let targets = vec![0usize, 1usize];

        let (ce, ppl) = compute_cross_entropy_and_perplexity(&p, &targets);
        assert!(ce > 0.0 && ce < 1.0);
        assert!((ppl - ce.exp()).abs() < 1e-12);

        let kl = compute_kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-12);

        let top1 = compute_top1_agreement(&p, &q);
        assert!((top1 - 100.0).abs() < 1e-12);

        assert_eq!(compute_greedy_match_length(&[1, 2, 3, 4], &[1, 2, 9, 4]), 2);
    }
}
