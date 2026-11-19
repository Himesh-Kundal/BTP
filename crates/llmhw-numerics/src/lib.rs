//! Pure Integer Operations Library, LUT Builders, and Integer Autoregressive Decoder Oracle (Phases R0 & R1).
//!
//! Strict Layering (Golden Rule 6):
//! Only Rust stdlib and serde/serde_json are used. Zero ML framework dependencies.

use serde::{Deserialize, Serialize};

// ==============================================================================
// 1. Core Dyadic Arithmetic & Saturation (N2-2, N2-3)
// ==============================================================================

/// Arithmetic right shift with round-half-up toward +infinity.
/// Formula: `(x + (1 << (s - 1))) >> s` for `s > 0`; identity for `s == 0`.
#[inline]
pub fn rshift_rnd(x: i64, s: i32) -> i64 {
    assert!(s >= 0, "Shift amount s must be non-negative, got {s}");
    if s == 0 {
        return x;
    }
    if s >= 63 {
        return if x >= 0 { 0 } else { -1 };
    }
    let offset = 1i64 << (s - 1);
    (x.wrapping_add(offset)) >> s
}

/// Vectorized `rshift_rnd` across a slice.
pub fn rshift_rnd_vec(xs: &[i64], s: i32) -> Vec<i64> {
    xs.iter().map(|&x| rshift_rnd(x, s)).collect()
}

/// Saturate integer to `[min_val, max_val]`.
#[inline]
pub fn sat(x: i64, min_val: i64, max_val: i64) -> i64 {
    x.clamp(min_val, max_val)
}

/// Symmetric INT8 saturation `[-127, 127]` per Numeric Contract N2-2.
#[inline]
pub fn sat_int8(x: i64) -> i8 {
    x.clamp(-127, 127) as i8
}

/// Symmetric INT24 saturation `[-8388608, 8388607]` per Numeric Contract N2-2.
#[inline]
pub fn sat_int24(x: i64) -> i32 {
    x.clamp(-(1 << 23), (1 << 23) - 1) as i32
}

/// Scalar dyadic requantization: `y = sat(rshift_rnd(acc * M, s), out_min, out_max)`.
#[inline]
pub fn requant(acc: i64, m: i64, s: i32, out_min: i64, out_max: i64) -> i64 {
    let prod = (acc as i128) * (m as i128);
    let shifted = if s == 0 {
        prod
    } else {
        let offset = 1i128 << (s - 1);
        (prod + offset) >> s
    };
    let clamped = shifted.clamp(out_min as i128, out_max as i128);
    clamped as i64
}

/// Per-channel dyadic requantization over an accumulator slice with per-channel multipliers `m_vec` and shift `s`.
pub fn requant_per_channel(
    acc: &[i64],
    m_vec: &[i64],
    s: i32,
    out_min: i64,
    out_max: i64,
) -> Vec<i64> {
    assert_eq!(acc.len(), m_vec.len());
    acc.iter()
        .zip(m_vec.iter())
        .map(|(&a, &m)| requant(a, m, s, out_min, out_max))
        .collect()
}

/// Per-channel dyadic requantization with individual `(M_i, s_i)` per channel.
pub fn requant_per_channel_ms(
    acc: &[i64],
    m_vec: &[i64],
    s_vec: &[i32],
    out_min: i64,
    out_max: i64,
) -> Vec<i64> {
    assert_eq!(acc.len(), m_vec.len());
    assert_eq!(acc.len(), s_vec.len());
    acc.iter()
        .zip(m_vec.iter().zip(s_vec.iter()))
        .map(|(&a, (&m, &s))| requant(a, m, s, out_min, out_max))
        .collect()
}

/// Round-half-to-even (`np.round` / IEEE-754 banker's rounding) for `f64`.
#[inline]
pub fn round_half_to_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// Convert a positive real float scale into an exact dyadic multiplier `(M, s)` matching `numerics.ops.float_to_dyadic_scale` on Python `float64`.
pub fn float_to_dyadic_scale(scale: f64, max_m_bits: i32) -> (i64, i32) {
    if scale <= 0.0 {
        return (0, 0);
    }
    let mut s = if scale < 1.0 {
        ((1.0 / scale).log2().ceil() as i32) + (max_m_bits - 1)
    } else {
        max_m_bits - 1
    };
    s = s.clamp(0, 62);
    let mut m = round_half_to_even(scale * 2.0f64.powi(s)) as i64;
    while s > 0 && m % 2 == 0 {
        m /= 2;
        s -= 1;
    }
    (m, s)
}

/// Convert a NumPy `np.float32` scalar scale into `(M, s)` matching `np.ceil(np.log2(np.float32(1.0 / scale)))` and `np.round(np.float32(scale) * (2**s))`.
pub fn float_to_dyadic_scale_f32(scale_f32: f32, max_m_bits: i32) -> (i64, i32) {
    if scale_f32 <= 0.0 {
        return (0, 0);
    }
    let mut s = if scale_f32 < 1.0 {
        let inv_f32 = 1.0f32 / scale_f32;
        (inv_f32.log2().ceil() as i32) + (max_m_bits - 1)
    } else {
        max_m_bits - 1
    };
    s = s.clamp(0, 62);
    let prod = (scale_f32 as f64) * 2.0f64.powi(s);
    let mut m = round_half_to_even(prod) as i64;
    while s > 0 && m % 2 == 0 {
        m /= 2;
        s -= 1;
    }
    (m, s)
}

/// Convert a signed real float scale into a signed dyadic multiplier `(M, s)` matching `numerics.ops.float_to_signed_dyadic_scale`.
pub fn float_to_signed_dyadic_scale(scale: f64, max_m_bits: i32) -> (i64, i32) {
    if scale == 0.0 {
        return (0, 0);
    }
    let sign = if scale < 0.0 { -1i64 } else { 1i64 };
    let mag = scale.abs();
    let mut s = ((max_m_bits - 1) as f64 - mag.log2()).floor() as i32;
    s = s.clamp(0, 62);
    let mut m_mag = round_half_to_even(mag * 2.0f64.powi(s)) as i64;
    while m_mag >= (1i64 << max_m_bits) && s > 0 {
        s -= 1;
        m_mag = round_half_to_even(mag * 2.0f64.powi(s)) as i64;
    }
    while s > 0 && m_mag % 2 == 0 {
        m_mag /= 2;
        s -= 1;
    }
    (sign * m_mag, s)
}

// ==============================================================================
// 2. Offline LUT Builders (Float64 with Explicit Rounding, N2-14 & ADR-2-2)
// ==============================================================================

/// Build lookup table for negative fractional base-2 exponent: `2^(-k / table_size)` in `Q0.out_bits` (ADR-2-2).
pub fn build_exp2_fraction_lut(table_size: usize, out_bits: u32) -> Vec<i64> {
    let scale = (1u64 << out_bits) as f64;
    (0..table_size)
        .map(|k| {
            let frac = (k as f64) / (table_size as f64);
            let val = (-frac).exp2();
            round_half_to_even(val * scale) as i64
        })
        .collect()
}

/// Build reciprocal square root mantissa LUT for RMSNorm (ADR-2-2): `lut[0] = 2^out_bits - 1`, `lut[k] = min(max_val, round(2^19 / sqrt(k)))`.
pub fn build_rsqrt_mantissa_lut(table_size: usize, out_bits: u32) -> Vec<i64> {
    let max_val = (1i64 << out_bits) - 1;
    let mut lut = vec![0i64; table_size];
    if table_size > 0 {
        lut[0] = max_val;
    }
    for (k, item) in lut.iter_mut().enumerate().skip(1) {
        let val = round_half_to_even((2.0f64.powi(19)) / (k as f64).sqrt()) as i64;
        *item = val.min(max_val);
    }
    lut
}

/// Build reciprocal mantissa LUT for Softmax normalizer sum: `round((1.0 / (1.0 + k / table_size)) * 2^out_bits)`.
pub fn build_reciprocal_lut(table_size: usize, out_bits: u32) -> Vec<i64> {
    let scale = (1u64 << out_bits) as f64;
    (0..table_size)
        .map(|k| {
            let m = 1.0 + (k as f64) / (table_size as f64);
            let val = 1.0 / m;
            round_half_to_even(val * scale) as i64
        })
        .collect()
}

/// Build symmetric lookup table for `SiLU(x) = x * sigmoid(x)`.
pub fn build_silu_lut(
    in_min: i32,
    in_max: i32,
    in_scale: f64,
    out_scale: f64,
    out_min: i64,
    out_max: i64,
) -> Vec<i8> {
    (in_min..=in_max)
        .map(|idx| {
            let x_real = (idx as f64) * in_scale;
            let sigmoid = 1.0 / (1.0 + (-x_real).exp());
            let y_real = x_real * sigmoid;
            let y_int = round_half_to_even(y_real / out_scale) as i64;
            y_int.clamp(out_min, out_max) as i8
        })
        .collect()
}

/// Build fixed-point `cos` and `sin` tables for RoPE in `Q1.out_bits` format.
/// Returns `(cos_table, sin_table)` flattened or as `Vec<Vec<i16>>` of shape `[max_ctx][head_dim / 2]`.
pub fn build_rope_cos_sin_tables(
    max_ctx: usize,
    head_dim: usize,
    theta: f64,
    out_bits: u32,
) -> (Vec<Vec<i16>>, Vec<Vec<i16>>) {
    let half_dim = head_dim / 2;
    let scale = (1u64 << out_bits) as f64;
    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / theta.powf((2.0 * (i as f64)) / (head_dim as f64)))
        .collect();

    let mut cos_table = Vec::with_capacity(max_ctx);
    let mut sin_table = Vec::with_capacity(max_ctx);

    for pos in 0..max_ctx {
        let mut cos_row = Vec::with_capacity(half_dim);
        let mut sin_row = Vec::with_capacity(half_dim);
        for &freq in &inv_freq {
            let angle = (pos as f64) * freq;
            cos_row.push(round_half_to_even(angle.cos() * scale) as i16);
            sin_row.push(round_half_to_even(angle.sin() * scale) as i16);
        }
        cos_table.push(cos_row);
        sin_table.push(sin_row);
    }
    (cos_table, sin_table)
}

// ==============================================================================
// 3. Integer Autoregressive Decoder Oracle (Task UA.4 / numerics/decoder.py)
// ==============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegerModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_ctx: usize,
    pub rope_theta: f64,
    pub residual_bits: u32,
    pub res_min: i64,
    pub res_max: i64,
}

impl Default for IntegerModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            vocab_size: 256,
            max_ctx: 32,
            rope_theta: 10000.0,
            residual_bits: 24,
            res_min: -(1 << 23),
            res_max: (1 << 23) - 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IntegerLayerWeights {
    pub q_proj: Vec<i8>,    // [N_q * K] row-major
    pub k_proj: Vec<i8>,    // [N_kv * K]
    pub v_proj: Vec<i8>,    // [N_kv * K]
    pub o_proj: Vec<i8>,    // [K * N_q]
    pub gate_proj: Vec<i8>, // [FFN * K]
    pub up_proj: Vec<i8>,   // [FFN * K]
    pub down_proj: Vec<i8>, // [K * FFN]

    pub q_m: Vec<i64>,
    pub q_s: i32,
    pub k_m: Vec<i64>,
    pub k_s: i32,
    pub v_m: Vec<i64>,
    pub v_s: i32,
    pub o_m: Vec<i64>,
    pub o_s: i32,
    pub gate_m: Vec<i64>,
    pub gate_s: i32,
    pub up_m: Vec<i64>,
    pub up_s: i32,
    pub down_m: Vec<i64>,
    pub down_s: i32,

    pub in_norm_gamma: Vec<i64>,
    pub in_norm_m: i64,
    pub in_norm_s: i32,
    pub post_norm_gamma: Vec<i64>,
    pub post_norm_m: i64,
    pub post_norm_s: i32,
}

#[derive(Debug, Clone)]
pub struct IntegerModelWeights {
    pub embed_table: Vec<i8>, // [vocab_size * hidden_size]
    pub embed_m: i64,
    pub embed_s: i32,
    pub layers: Vec<IntegerLayerWeights>,
    pub final_norm_gamma: Vec<i64>,
    pub final_norm_m: i64,
    pub final_norm_s: i32,
    pub lm_head: Vec<i8>, // [vocab_size * hidden_size]
    pub lm_head_m: Vec<i64>,
    pub lm_head_s: i32,
}

#[derive(Debug, Clone)]
pub struct IntegerKVCache {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub max_ctx: usize,
    pub head_dim: usize,
    pub k: Vec<i8>, // [num_layers][num_kv_heads][max_ctx][head_dim]
    pub v: Vec<i8>,
}

impl IntegerKVCache {
    pub fn new(num_layers: usize, num_kv_heads: usize, max_ctx: usize, head_dim: usize) -> Self {
        let total = num_layers * num_kv_heads * max_ctx * head_dim;
        Self {
            num_layers,
            num_kv_heads,
            max_ctx,
            head_dim,
            k: vec![0i8; total],
            v: vec![0i8; total],
        }
    }

    #[inline]
    fn offset(&self, layer_idx: usize, kv_head: usize, pos: usize) -> usize {
        (((layer_idx * self.num_kv_heads + kv_head) * self.max_ctx) + pos) * self.head_dim
    }

    pub fn write(&mut self, layer_idx: usize, pos: usize, k_vec: &[i8], v_vec: &[i8]) {
        for h in 0..self.num_kv_heads {
            let base = self.offset(layer_idx, h, pos);
            let src_base = h * self.head_dim;
            for d in 0..self.head_dim {
                self.k[base + d] = k_vec[src_base + d].clamp(-127, 127);
                self.v[base + d] = v_vec[src_base + d].clamp(-127, 127);
            }
        }
    }

    pub fn get_head_at(&self, layer_idx: usize, kv_head: usize, pos: usize) -> (&[i8], &[i8]) {
        let base = self.offset(layer_idx, kv_head, pos);
        (
            &self.k[base..base + self.head_dim],
            &self.v[base..base + self.head_dim],
        )
    }
}

/// Matrix-vector multiply `y[n] = sum_k x[k] * w[n, k]` in `i64`.
pub fn matvec_i8_to_i64(x: &[i8], w_row_major: &[i8], out_features: usize, in_features: usize) -> Vec<i64> {
    assert_eq!(x.len(), in_features);
    assert_eq!(w_row_major.len(), out_features * in_features);
    let mut out = vec![0i64; out_features];
    for (n, out_val) in out.iter_mut().enumerate() {
        let row = &w_row_major[n * in_features..(n + 1) * in_features];
        let mut acc = 0i64;
        for (&xk, &wnk) in x.iter().zip(row.iter()) {
            acc += (xk as i64) * (wnk as i64);
        }
        *out_val = acc;
    }
    out
}

#[inline]
pub fn bit_length_u64(x: u64) -> u32 {
    if x == 0 {
        0
    } else {
        64 - x.leading_zeros()
    }
}

pub struct IntegerDecoder {
    pub cfg: IntegerModelConfig,
    pub weights: IntegerModelWeights,
    pub kv_cache: IntegerKVCache,
    pub exp2_lut: Vec<i64>,
    pub rsqrt_lut: Vec<i64>,
    pub recip_lut: Vec<i64>,
    pub silu_lut: Vec<i8>,
    pub cos_table: Vec<Vec<i16>>,
    pub sin_table: Vec<Vec<i16>>,
}

impl IntegerDecoder {
    pub fn new(config: IntegerModelConfig, weights: IntegerModelWeights) -> Self {
        let kv_cache = IntegerKVCache::new(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.max_ctx,
            config.head_dim,
        );
        let exp2_lut = build_exp2_fraction_lut(256, 16);
        let rsqrt_lut = build_rsqrt_mantissa_lut(256, 16);
        let recip_lut = build_reciprocal_lut(256, 16);
        let silu_lut = build_silu_lut(-128, 127, 0.05, 0.05, -127, 127);
        let (cos_table, sin_table) =
            build_rope_cos_sin_tables(config.max_ctx, config.head_dim, config.rope_theta, 14);

        Self {
            cfg: config,
            weights,
            kv_cache,
            exp2_lut,
            rsqrt_lut,
            recip_lut,
            silu_lut,
            cos_table,
            sin_table,
        }
    }

    pub fn reset(&mut self) {
        self.kv_cache = IntegerKVCache::new(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.max_ctx,
            self.cfg.head_dim,
        );
    }

    pub fn integer_rmsnorm(&self, x: &[i64], gamma: &[i64], m: i64, s: i32) -> Vec<i8> {
        let ss: i64 = x.iter().map(|&v| v * v).sum();
        let mean = ss / (self.cfg.hidden_size as i64);
        if mean <= 0 {
            return vec![0i8; self.cfg.hidden_size];
        }
        let b = bit_length_u64(mean as u64) as i32;
        let sh_amt = if b > 8 { (b - 7) & !1 } else { 0 };
        let mantissa = ((mean >> sh_amt) & 0xFF) as usize;
        let rsqrt_val = self.rsqrt_lut[mantissa];
        let exp_sh = sh_amt / 2;

        x.iter()
            .zip(gamma.iter())
            .map(|(&xi, &gi)| {
                let stg1 = if exp_sh > 0 {
                    rshift_rnd(xi * rsqrt_val, exp_sh)
                } else {
                    xi * rsqrt_val
                };
                sat_int8(requant(stg1 * gi, m, 19 + s, -127, 127))
            })
            .collect()
    }

    pub fn integer_rope(&self, x: &[i8], num_heads: usize, pos: usize) -> Vec<i8> {
        let hd = self.cfg.head_dim;
        let half = hd / 2;
        let cos_m = &self.cos_table[pos];
        let sin_m = &self.sin_table[pos];
        let mut out = vec![0i8; num_heads * hd];

        for h in 0..num_heads {
            let base = h * hd;
            for i in 0..half {
                let x1 = x[base + i] as i64;
                let x2 = x[base + half + i] as i64;
                let c = cos_m[i] as i64;
                let s = sin_m[i] as i64;
                out[base + i] = sat_int8(rshift_rnd(x1 * c - x2 * s, 14));
                out[base + half + i] = sat_int8(rshift_rnd(x2 * c + x1 * s, 14));
            }
        }
        out
    }

    pub fn decode_step(&mut self, token_id: usize, pos: usize) -> (Vec<i64>, usize) {
        let d = self.cfg.hidden_size;
        let ffn = self.cfg.intermediate_size;
        let n_q = self.cfg.num_attention_heads;
        let n_kv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let gqa_group = n_q / n_kv;

        // 1. Embedding Gather
        let emb_row = &self.weights.embed_table[token_id * d..(token_id + 1) * d];
        let mut h: Vec<i64> = emb_row
            .iter()
            .map(|&v| {
                let scaled = rshift_rnd((v as i64) * self.weights.embed_m, self.weights.embed_s);
                sat(scaled, self.cfg.res_min, self.cfg.res_max)
            })
            .collect();

        // 2. Transformer Layers
        for l_idx in 0..self.cfg.num_hidden_layers {
            let layer = &self.weights.layers[l_idx];
            let a = self.integer_rmsnorm(&h, &layer.in_norm_gamma, layer.in_norm_m, layer.in_norm_s);

            let q_acc = matvec_i8_to_i64(&a, &layer.q_proj, n_q * hd, d);
            let k_acc = matvec_i8_to_i64(&a, &layer.k_proj, n_kv * hd, d);
            let v_acc = matvec_i8_to_i64(&a, &layer.v_proj, n_kv * hd, d);

            let q_raw: Vec<i8> = requant_per_channel(&q_acc, &layer.q_m, layer.q_s, -127, 127)
                .into_iter()
                .map(|v| v as i8)
                .collect();
            let k_raw: Vec<i8> = requant_per_channel(&k_acc, &layer.k_m, layer.k_s, -127, 127)
                .into_iter()
                .map(|v| v as i8)
                .collect();
            let v_raw: Vec<i8> = requant_per_channel(&v_acc, &layer.v_m, layer.v_s, -127, 127)
                .into_iter()
                .map(|v| v as i8)
                .collect();

            let q = self.integer_rope(&q_raw, n_q, pos);
            let k = self.integer_rope(&k_raw, n_kv, pos);

            self.kv_cache.write(l_idx, pos, &k, &v_raw);

            let mut attn_flat = vec![0i8; n_q * hd];
            for q_idx in 0..n_q {
                let kv_idx = q_idx / gqa_group;
                let q_head = &q[q_idx * hd..(q_idx + 1) * hd];

                let mut scores = Vec::with_capacity(pos + 1);
                for t in 0..=pos {
                    let (k_head, _) = self.kv_cache.get_head_at(l_idx, kv_idx, t);
                    let dot: i64 = q_head
                        .iter()
                        .zip(k_head.iter())
                        .map(|(&qi, &ki)| (qi as i64) * (ki as i64))
                        .sum();
                    scores.push(dot >> 1);
                }

                let m = *scores.iter().max().unwrap_or(&0);
                let mut exp_vals = Vec::with_capacity(pos + 1);
                for &sc in &scores {
                    let diff = (m - sc).unsigned_abs();
                    let int_part = (diff >> 8) as usize;
                    let frac_idx = (diff & 0xFF) as usize;
                    let frac_val = self.exp2_lut[frac_idx];
                    let sh = int_part.min(16);
                    let exp_v = (frac_val >> sh).max(1);
                    exp_vals.push(exp_v);
                }
                let sigma = exp_vals.iter().sum::<i64>().max(1);
                let probs: Vec<i64> = exp_vals.iter().map(|&ev| (ev * 32767) / sigma).collect();

                for d_i in 0..hd {
                    let mut acc_v = 0i64;
                    for (t, &pt) in probs.iter().enumerate() {
                        let (_, v_head) = self.kv_cache.get_head_at(l_idx, kv_idx, t);
                        acc_v += pt * (v_head[d_i] as i64);
                    }
                    attn_flat[q_idx * hd + d_i] = sat_int8(acc_v >> 15);
                }
            }

            let o_acc = matvec_i8_to_i64(&attn_flat, &layer.o_proj, d, n_q * hd);
            for i in 0..d {
                let o_res = rshift_rnd(o_acc[i] * layer.o_m[i], layer.o_s);
                h[i] = sat(h[i] + o_res, self.cfg.res_min, self.cfg.res_max);
            }

            // SwiGLU MLP block
            let m_act =
                self.integer_rmsnorm(&h, &layer.post_norm_gamma, layer.post_norm_m, layer.post_norm_s);
            let gate_acc = matvec_i8_to_i64(&m_act, &layer.gate_proj, ffn, d);
            let up_acc = matvec_i8_to_i64(&m_act, &layer.up_proj, ffn, d);

            let gate = requant_per_channel(&gate_acc, &layer.gate_m, layer.gate_s, -127, 127);
            let up = requant_per_channel(&up_acc, &layer.up_m, layer.up_s, -127, 127);

            let mut swiglu_prod = vec![0i8; ffn];
            for i in 0..ffn {
                let idx = (gate[i] + 128).clamp(0, 255) as usize;
                let silu_val = self.silu_lut[idx] as i64;
                swiglu_prod[i] = sat_int8(requant(silu_val * up[i], 1, 6, -127, 127));
            }

            let down_acc = matvec_i8_to_i64(&swiglu_prod, &layer.down_proj, d, ffn);
            for i in 0..d {
                let down_res = rshift_rnd(down_acc[i] * layer.down_m[i], layer.down_s);
                h[i] = sat(h[i] + down_res, self.cfg.res_min, self.cfg.res_max);
            }
        }

        // 3. Final RMSNorm
        let f = self.integer_rmsnorm(
            &h,
            &self.weights.final_norm_gamma,
            self.weights.final_norm_m,
            self.weights.final_norm_s,
        );

        // 4. LM Head Projection with scale alignment
        let raw_logits = matvec_i8_to_i64(&f, &self.weights.lm_head, self.cfg.vocab_size, d);
        let aligned_logits: Vec<i64> = raw_logits
            .iter()
            .zip(self.weights.lm_head_m.iter())
            .map(|(&raw, &m)| rshift_rnd(raw * m, self.weights.lm_head_s))
            .collect();

        // 5. Greedy ArgMax (lowest token ID tie-break)
        let mut best_id = 0usize;
        let mut best_val = aligned_logits[0];
        for (idx, &val) in aligned_logits.iter().enumerate().skip(1) {
            if val > best_val {
                best_val = val;
                best_id = idx;
            }
        }

        (aligned_logits, best_id)
    }

    pub fn score_sequence(&mut self, sequence_tokens: &[usize]) -> Vec<usize> {
        self.reset();
        sequence_tokens
            .iter()
            .enumerate()
            .map(|(pos, &tok)| self.decode_step(tok, pos).1)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rshift_rnd_and_saturation() {
        assert_eq!(rshift_rnd(42, 0), 42);
        assert_eq!(rshift_rnd(-42, 0), -42);
        // Round-half-up toward +infinity
        assert_eq!(rshift_rnd(1, 1), 1); // 0.5 -> 1
        assert_eq!(rshift_rnd(-1, 1), 0); // -0.5 -> 0
        assert_eq!(rshift_rnd(3, 1), 2); // 1.5 -> 2
        assert_eq!(rshift_rnd(-3, 1), -1); // -1.5 -> -1
        assert_eq!(sat_int8(200), 127);
        assert_eq!(sat_int8(-128), -127);
        assert_eq!(sat_int24(1 << 25), (1 << 23) - 1);
        assert_eq!(sat_int24(-(1 << 25)), -(1 << 23));
    }

    #[test]
    fn test_luts_adr_2_2_properties() {
        let exp2 = build_exp2_fraction_lut(256, 16);
        assert_eq!(exp2.len(), 256);
        assert_eq!(exp2[0], 65536);
        assert_eq!(exp2[255], 32857);
        // Monotonically non-increasing
        for w in exp2.windows(2) {
            assert!(w[0] >= w[1]);
        }

        let rsqrt = build_rsqrt_mantissa_lut(256, 16);
        assert_eq!(rsqrt.len(), 256);
        assert_eq!(rsqrt[0], 65535);
        assert_eq!(rsqrt[64], 65535);
        assert_eq!(rsqrt[65], 65030);
        assert_eq!(rsqrt[255], 32832);

        let recip = build_reciprocal_lut(256, 16);
        assert_eq!(recip[0], 65536);
        assert!(recip[255] > 32768);

        let silu = build_silu_lut(-256, 255, 0.05, 0.05, -127, 127);
        assert_eq!(silu.len(), 512);
        assert_eq!(silu[256], 0); // x = 0 -> SiLU(0) = 0
        assert!(silu[511] > 0);
    }

    #[test]
    fn test_dyadic_scale_conversion() {
        let (m, s) = float_to_dyadic_scale(0.5, 31);
        assert_eq!(m, 1);
        assert_eq!(s, 1);

        let (ms, ss) = float_to_signed_dyadic_scale(-0.25, 15);
        assert_eq!(ms, -1);
        assert_eq!(ss, 2);
    }
}
