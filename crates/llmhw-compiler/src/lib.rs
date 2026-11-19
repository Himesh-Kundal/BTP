//! Compiler Passes (Part I Quantization & Part II Format Assignment, ROM Packing, Schedule, Memory Planner, and CompiledArtifactInterpreter) (Phase R3).
//!
//! Strict Layering (Golden Rule 6): No ML framework dependencies.

use llmhw_ir::{sha256_hex, EdgeFormat, GraphV2, NpyArray, OpKind, TensorKind};
use llmhw_numerics::{
    bit_length_u64, build_exp2_fraction_lut, build_reciprocal_lut, build_rope_cos_sin_tables,
    build_rsqrt_mantissa_lut, build_silu_lut, float_to_dyadic_scale, float_to_signed_dyadic_scale,
    requant, round_half_to_even, rshift_rnd, sat, sat_int8,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

// ==============================================================================
// 1. Part I Symmetric INT8 Quantization & Accumulator Bounds
// ==============================================================================

/// Rule N2 per-channel symmetric INT8 weight quantization (`[-127, 127]`, round-half-to-even).
/// `w` is `[out_features, in_features]` row-major. Returns `(w_q, scales)`.
pub fn quantize_weights(w: &[f64], out_features: usize, in_features: usize) -> (Vec<i8>, Vec<f64>) {
    assert_eq!(w.len(), out_features * in_features);
    let mut w_q = vec![0i8; w.len()];
    let mut scales = vec![0.0f64; out_features];

    for n in 0..out_features {
        let row = &w[n * in_features..(n + 1) * in_features];
        // Match NumPy float32 ch_max = np.max(np.abs(w_arr), axis=-1)
        let max_abs_f32 = row
            .iter()
            .map(|&v| (v as f32).abs())
            .fold(0.0f32, f32::max);
        let ch_max_f32 = max_abs_f32.max(1e-8f32);
        scales[n] = (ch_max_f32 as f64) / 127.0;
        let out_row = &mut w_q[n * in_features..(n + 1) * in_features];
        for (dst, &val) in out_row.iter_mut().zip(row.iter()) {
            // Match NumPy: (float32 / float32) * float32(127.0) -> round_ties_even
            let ratio_f32 = (val as f32) / ch_max_f32;
            let prod_f32 = ratio_f32 * 127.0f32;
            let q = prod_f32.round_ties_even() as i64;
            *dst = q.clamp(-127, 127) as i8;
        }
    }
    (w_q, scales)
}

/// Rule N3 static per-tensor activation quantization (`[-127, 127]`, round-half-to-even).
pub fn quantize_activations(x: &[f64], act_scale: Option<f64>) -> (Vec<i8>, f64) {
    let s_x = act_scale.unwrap_or_else(|| {
        let max_abs = x.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        if max_abs < 1e-8 {
            1e-8 / 127.0
        } else {
            max_abs / 127.0
        }
    });
    let x_q = x
        .iter()
        .map(|&v| (round_half_to_even(v / s_x) as i64).clamp(-127, 127) as i8)
        .collect();
    (x_q, s_x)
}

/// Rule N4 worst-case 32-bit signed accumulator bound check: `K * 127 * 127 + max_bias < 2^31`.
pub fn check_accumulator_bounds(in_features: usize, max_abs_bias: i64) -> anyhow::Result<i64> {
    let worst_case = (in_features as i64) * 127 * 127 + max_abs_bias.abs();
    if worst_case >= (1i64 << 31) {
        anyhow::bail!(
            "Accumulator overflow risk: worst_case={worst_case} >= 2^31 for K={in_features}"
        );
    }
    Ok(worst_case)
}

/// Pad 2D matrix `[n, k]` with zeros to multiples of `(tile_n, tile_k)`.
pub fn pad_matrix_i8(
    w: &[i8],
    n: usize,
    k: usize,
    tile_n: usize,
    tile_k: usize,
) -> (Vec<i8>, usize, usize) {
    let n_pad = n.div_ceil(tile_n) * tile_n;
    let k_pad = k.div_ceil(tile_k) * tile_k;
    let mut padded = vec![0i8; n_pad * k_pad];
    for r in 0..n {
        padded[r * k_pad..r * k_pad + k].copy_from_slice(&w[r * k..(r + 1) * k]);
    }
    (padded, n_pad, k_pad)
}

// ==============================================================================
// 2. Part II Format & Dyadic Scale Assignment Pass (compiler/format_assignment.py)
// ==============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequantParamSet {
    #[serde(rename = "M")]
    pub m: Vec<i64>,
    pub shift: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatAssignmentResult {
    #[serde(skip)]
    pub graph: Option<GraphV2>,
    pub format_table: BTreeMap<String, Value>,
    pub requant_params: BTreeMap<String, RequantParamSet>,
    #[serde(skip)]
    pub requant_params_order: Vec<String>,
    pub silu_scale: f64,
}

impl FormatAssignmentResult {
    pub fn save_json(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(self)?;
        fs::write(path, text)?;
        Ok(())
    }
}

pub fn assign_formats_and_scales(
    mut graph: GraphV2,
    weights_dict: &HashMap<String, NpyArray<f64>>,
    frozen_config: &Value,
    calib_scales: Option<&BTreeMap<String, Value>>,
) -> anyhow::Result<FormatAssignmentResult> {
    let empty_calib = BTreeMap::new();
    let calib = calib_scales.unwrap_or(&empty_calib);

    let res_bits = frozen_config
        .pointer("/formats/residual_stream/bits")
        .and_then(|v| v.as_u64())
        .unwrap_or(24) as u32;
    let w_bits = frozen_config
        .pointer("/formats/linear_weights/bits")
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as u32;
    let act_bits = frozen_config
        .pointer("/formats/linear_activations/bits")
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as u32;

    let s_act = 0.05f64;
    let s_res = 0.01f64;
    let silu_scale = calib
        .get("__s_gate__")
        .and_then(|v| v.as_f64())
        .unwrap_or(s_act);

    let fmt_token = EdgeFormat::new(16, false, 1.0, "UINT16")?;
    let fmt_res = EdgeFormat::new(res_bits, true, s_res, format!("INT{res_bits}"))?;
    let fmt_act = EdgeFormat::new(act_bits, true, s_act, format!("INT{act_bits}"))?;
    let fmt_weight = EdgeFormat::new(w_bits, true, 0.002, format!("INT{w_bits}"))?;
    let fmt_rope = EdgeFormat::new(16, true, 1.0 / 16384.0, "Q1.14")?;

    let mut format_table = BTreeMap::new();
    let mut requant_params = BTreeMap::new();
    let mut requant_params_order = Vec::new();

    for (name, tensor) in graph.tensors.iter_mut() {
        if tensor.kind == TensorKind::STATE {
            tensor.format = fmt_act.clone();
        } else if tensor.kind == TensorKind::CONSTANT {
            if name.contains("rope") {
                tensor.format = fmt_rope.clone();
            } else {
                tensor.format = fmt_weight.clone();
            }
        } else if name == "tok_in" || name == "tok_out" {
            tensor.format = fmt_token.clone();
        } else if !name.contains("norm")
            && (name.contains("logits")
                || name.contains("res")
                || name.contains("down_")
                || name.contains("o_proj_")
                || name == "h_embed")
        {
            tensor.format = fmt_res.clone();
        } else {
            let t_scale = calib
                .get(name)
                .and_then(|v| v.as_f64())
                .unwrap_or(s_act);
            tensor.format = EdgeFormat::new(act_bits, true, t_scale, format!("INT{act_bits}"))?;
        }

        format_table.insert(
            name.clone(),
            json!({
                "bits": tensor.format.bits,
                "signed": tensor.format.signed,
                "scale": tensor.format.scale,
                "fixed_point": tensor.format.fixed_point,
                "min_val": tensor.format.min_val() as i64,
                "max_val": tensor.format.max_val() as i64,
                "shape": tensor.shape,
            }),
        );
    }

    for node in &graph.nodes {
        match node.op {
            OpKind::Embed => {
                let w_name = &node.inputs[1];
                if let Some(w_arr) = weights_dict.get(w_name) {
                    let rows = w_arr.shape[0];
                    let cols = w_arr.shape[1];
                    let mut m_list = Vec::with_capacity(rows);
                    let mut s_list = Vec::with_capacity(rows);
                    for r in 0..rows {
                        let row = &w_arr.data[r * cols..(r + 1) * cols];
                        let row_max = row
                            .iter()
                            .map(|&v| (v as f32).abs())
                            .fold(0.0f32, f32::max)
                            .max(1e-8f32);
                        let s_w = row_max / 127.0f32;
                        let real_scale = s_w / (s_res as f32);
                        let (m, s) = llmhw_numerics::float_to_dyadic_scale_f32(real_scale, 31);
                        m_list.push(m);
                        s_list.push(s);
                    }
                    requant_params_order.push(node.id.clone());
                    requant_params.insert(node.id.clone(), RequantParamSet { m: m_list, shift: s_list });
                }
            }
            OpKind::RMSNorm => {
                let g_name = &node.inputs[1];
                if let Some(gamma) = weights_dict.get(g_name) {
                    let s_out = graph
                        .tensors
                        .get(&node.outputs[0])
                        .map(|t| t.format.scale)
                        .unwrap_or(s_act);
                    let mut m_list = Vec::with_capacity(gamma.data.len());
                    let mut s_list = Vec::with_capacity(gamma.data.len());
                    for &g_val in &gamma.data {
                        let real_scale = (g_val as f32 as f64) / s_out;
                        let (m, s) = float_to_signed_dyadic_scale(real_scale, 15);
                        m_list.push(m);
                        s_list.push(s);
                    }
                    requant_params_order.push(node.id.clone());
                    requant_params.insert(node.id.clone(), RequantParamSet { m: m_list, shift: s_list });
                }
            }
            OpKind::Linear => {
                let w_name = &node.inputs[1];
                if let Some(w_arr) = weights_dict.get(w_name) {
                    let rows = w_arr.shape[0];
                    let cols = w_arr.shape[1];
                    let s_out = graph.tensors[&node.outputs[0]].format.scale;
                    let s_in = graph.tensors[&node.inputs[0]].format.scale;
                    let ch_out_list = calib
                        .get(&format!("__ch_scale__{}", node.id))
                        .and_then(|v| v.as_array());

                    let mut m_list = Vec::with_capacity(rows);
                    let mut s_list = Vec::with_capacity(rows);

                    for r in 0..rows {
                        let row = &w_arr.data[r * cols..(r + 1) * cols];
                        let ch_max_f32 = row
                            .iter()
                            .map(|&v| (v as f32).abs())
                            .fold(0.0f32, f32::max)
                            .max(1e-8f32);
                        let s_w_f32 = ch_max_f32 / 127.0f32;
                        if let Some(s_out_c) = ch_out_list
                            .and_then(|arr| arr.get(r))
                            .and_then(|v| v.as_f64())
                        {
                            // Python: float((s_in * s_w_f32) / float(s_out_c)) where s_in * s_w_f32 is np.float32
                            let num_f32 = (s_in as f32) * s_w_f32;
                            let real_scale_f32 = num_f32 / (s_out_c as f32);
                            let (m, s) = float_to_dyadic_scale(real_scale_f32 as f64, 31);
                            m_list.push(m);
                            s_list.push(s);
                        } else {
                            let real_scale_f32 = ((s_in as f32) * s_w_f32) / (s_out as f32);
                            let (m, s) = llmhw_numerics::float_to_dyadic_scale_f32(real_scale_f32, 31);
                            m_list.push(m);
                            s_list.push(s);
                        }
                    }
                    requant_params_order.push(node.id.clone());
                    requant_params.insert(node.id.clone(), RequantParamSet { m: m_list, shift: s_list });
                }
            }
            OpKind::Attention => {
                let hd = node
                    .attributes
                    .get("head_dim")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(64) as f64;
                let (m, s) = float_to_dyadic_scale(1.0 / hd.sqrt(), 31);
                requant_params_order.push(node.id.clone());
                requant_params.insert(node.id.clone(), RequantParamSet { m: vec![m], shift: vec![s] });
            }
            _ => {}
        }
    }

    Ok(FormatAssignmentResult {
        graph: Some(graph),
        format_table,
        requant_params,
        requant_params_order,
        silu_scale,
    })
}

// ==============================================================================
// 3. Part II ROM Packing Pass (compiler/rom_packing.py)
// ==============================================================================

#[inline]
pub fn int_to_twos_comp_hex(val: i64, bits: u32) -> String {
    let mask: u64 = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let uval = (val as u64) & mask;
    let hex_len = (bits as usize).div_ceil(4);
    format!("{uval:0>width$x}", width = hex_len)
}

#[allow(clippy::too_many_arguments)]
pub fn pack_rom_artifacts(
    graph: &GraphV2,
    format_result: &FormatAssignmentResult,
    weights_dict: &HashMap<String, NpyArray<f64>>,
    out_dir: &Path,
    tile_r: usize,
    tile_c: usize,
    max_ctx: usize,
    rope_theta: f64,
    head_dim: usize,
) -> anyhow::Result<Value> {
    fs::create_dir_all(out_dir)?;

    let mut weights_meta = serde_json::Map::new();
    let mut requant_meta = serde_json::Map::new();
    let mut tables_meta = serde_json::Map::new();

    // 1. Pack Weight ROM (weights.hex)
    let weight_hex_path = out_dir.join("weights.hex");
    let mut f_w = BufWriter::with_capacity(4 * 1024 * 1024, File::create(&weight_hex_path)?);
    let mut w_hasher = Sha256::new();
    let mut current_weight_offset = 0usize;

    let mut hex_lut = [[0u8; 3]; 256];
    for (i, entry) in hex_lut.iter_mut().enumerate() {
        let s = format!("{i:02x}\n");
        entry.copy_from_slice(s.as_bytes());
    }

    for node in &graph.nodes {
        if !matches!(node.op, OpKind::Linear | OpKind::Embed) || node.inputs.len() < 2 {
            continue;
        }
        let w_name = &node.inputs[1];
        let Some(w_arr) = weights_dict.get(w_name) else {
            continue;
        };
        let orig_n = w_arr.shape[0];
        let orig_k = w_arr.shape[1];

        let (w_int8, _) = quantize_weights(&w_arr.data, orig_n, orig_k);
        let (w_padded, n_pad, k_pad) = pad_matrix_i8(&w_int8, orig_n, orig_k, tile_r, tile_c);

        let num_tiles_r = n_pad / tile_r;
        let num_tiles_c = k_pad / tile_c;
        let start_offset = current_weight_offset;
        let length = n_pad * k_pad;

        let mut chunk_buf = Vec::with_capacity(tile_r * tile_c * 3);
        for tr in 0..num_tiles_r {
            for tc in 0..num_tiles_c {
                chunk_buf.clear();
                for r in 0..tile_r {
                    let row_base = (tr * tile_r + r) * k_pad + tc * tile_c;
                    for c in 0..tile_c {
                        let b = w_padded[row_base + c] as u8;
                        chunk_buf.extend_from_slice(&hex_lut[b as usize]);
                    }
                }
                f_w.write_all(&chunk_buf)?;
                w_hasher.update(&chunk_buf);
            }
        }

        current_weight_offset += length;
        weights_meta.insert(
            node.id.clone(),
            json!({
                "tensor_name": w_name,
                "base_addr": start_offset,
                "length": length,
                "orig_shape": [orig_n, orig_k],
                "padded_shape": [n_pad, k_pad],
                "num_tiles": [num_tiles_r, num_tiles_c],
            }),
        );
    }
    f_w.flush()?;
    let w_sha = hex::encode(w_hasher.finalize());

    // 2. Pack Requant Parameters ROM (requant_params.hex)
    let mut param_hex_lines = Vec::new();
    let mut current_param_offset = 0usize;

    for node_id in &format_result.requant_params_order {
        let params = &format_result.requant_params[node_id];
        let start_offset = current_param_offset;
        for (&m_val, &s_val) in params.m.iter().zip(params.shift.iter()) {
            let m_hex = int_to_twos_comp_hex(m_val, 32);
            let s_hex = int_to_twos_comp_hex(s_val as i64, 32);
            param_hex_lines.push(format!("{m_hex}{s_hex}"));
            current_param_offset += 1;
        }
        let length = current_param_offset - start_offset;
        requant_meta.insert(
            node_id.clone(),
            json!({
                "base_addr": start_offset,
                "length": length,
                "channels": params.m.len(),
            }),
        );
    }

    let param_hex_path = out_dir.join("requant_params.hex");
    let param_content = format!("{}\n", param_hex_lines.join("\n"));
    fs::write(&param_hex_path, param_content.as_bytes())?;
    let param_sha = sha256_hex(param_content.as_bytes());

    // 3. Pack Mathematical Lookup Tables ROM (tables.hex)
    let mut table_hex_lines = Vec::new();
    let mut current_table_offset = 0usize;

    let mut add_table = |name: &str, values: &[i64], shape: Vec<usize>, bits: u32| {
        let start_offset = current_table_offset;
        for &v in values {
            table_hex_lines.push(int_to_twos_comp_hex(v, bits));
            current_table_offset += 1;
        }
        let length = current_table_offset - start_offset;
        tables_meta.insert(
            name.to_string(),
            json!({
                "base_addr": start_offset,
                "length": length,
                "shape": shape,
                "bits": bits,
            }),
        );
    };

    let (cos_tab, sin_tab) = build_rope_cos_sin_tables(max_ctx, head_dim, rope_theta, 14);
    let half_dim = head_dim / 2;
    let cos_flat: Vec<i64> = cos_tab.iter().flatten().map(|&v| v as i64).collect();
    let sin_flat: Vec<i64> = sin_tab.iter().flatten().map(|&v| v as i64).collect();
    add_table("rope_cos", &cos_flat, vec![max_ctx, half_dim], 16);
    add_table("rope_sin", &sin_flat, vec![max_ctx, half_dim], 16);

    let rsqrt_lut = build_rsqrt_mantissa_lut(256, 16);
    add_table("rsqrt_mantissa_lut", &rsqrt_lut, vec![256], 16);

    let exp2_lut = build_exp2_fraction_lut(256, 16);
    add_table("exp2_lut", &exp2_lut, vec![256], 16);

    let recip_lut = build_reciprocal_lut(256, 16);
    add_table("reciprocal_lut", &recip_lut, vec![256], 16);

    let s_gate = format_result.silu_scale;
    let silu_lut: Vec<i64> = build_silu_lut(-256, 255, s_gate, s_gate, -127, 127)
        .into_iter()
        .map(|v| v as i64)
        .collect();
    add_table("silu_lut", &silu_lut, vec![512], 8);

    let table_hex_path = out_dir.join("tables.hex");
    let table_content = format!("{}\n", table_hex_lines.join("\n"));
    fs::write(&table_hex_path, table_content.as_bytes())?;
    let table_sha = sha256_hex(table_content.as_bytes());

    let address_map = json!({
        "engine_config": {"tile_r": tile_r, "tile_c": tile_c},
        "weights": weights_meta,
        "weights_image": {
            "file": "weights.hex",
            "total_words": current_weight_offset,
            "word_bits": 8,
            "sha256": w_sha,
        },
        "requant_params": requant_meta,
        "requant_image": {
            "file": "requant_params.hex",
            "total_words": current_param_offset,
            "word_bits": 64,
            "sha256": param_sha,
        },
        "tables": tables_meta,
        "tables_image": {
            "file": "tables.hex",
            "total_words": current_table_offset,
            "word_bits": 16,
            "sha256": table_sha,
        },
    });

    fs::write(
        out_dir.join("address_map.json"),
        serde_json::to_string_pretty(&address_map)?,
    )?;

    Ok(address_map)
}

// ==============================================================================
// 4. Sequencer Microprogram Schedule (compiler/schedule.py)
// ==============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Opcode {
    HALT = 0,
    EMBED = 1,
    RMSNORM = 2,
    LINEAR = 3,
    ROPE = 4,
    ATTENTION = 5,
    ADD = 6,
    SWIGLU = 7,
    ARGMAX = 8,
}

impl Opcode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::EMBED,
            2 => Self::RMSNORM,
            3 => Self::LINEAR,
            4 => Self::ROPE,
            5 => Self::ATTENTION,
            6 => Self::ADD,
            7 => Self::SWIGLU,
            8 => Self::ARGMAX,
            _ => Self::HALT,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: String,
    pub opcode_val: u8,
    pub name: String,
    pub src_buf: usize,
    pub dst_buf: usize,
    pub dims: Vec<usize>,
    pub rom_weight_base: usize,
    pub rom_param_base: usize,
    pub flags: u8,
    pub cycle_estimate: usize,
}

impl Instruction {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        op: Opcode,
        name: impl Into<String>,
        src_buf: usize,
        dst_buf: usize,
        dims: Vec<usize>,
        rom_weight_base: usize,
        rom_param_base: usize,
        flags: u8,
        cycle_estimate: usize,
    ) -> Self {
        Self {
            opcode: format!("{op:?}"),
            opcode_val: op as u8,
            name: name.into(),
            src_buf,
            dst_buf,
            dims,
            rom_weight_base,
            rom_param_base,
            flags,
            cycle_estimate,
        }
    }

    pub fn to_hex_word(&self) -> String {
        let dim0 = self.dims.first().copied().unwrap_or(0) & 0xFFFF;
        let dim1 = self.dims.get(1).copied().unwrap_or(0) & 0xFFFF;
        let w_high: u32 = ((self.opcode_val as u32) << 24)
            | (((self.src_buf as u32) & 0xFF) << 16)
            | (((self.dst_buf as u32) & 0xFF) << 8)
            | ((self.flags as u32) & 0xFF);
        format!(
            "{w_high:08x}{:08x}{:08x}{dim0:04x}{dim1:04x}",
            (self.rom_weight_base as u32),
            (self.rom_param_base as u32),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Microprogram {
    pub instructions: Vec<Instruction>,
    pub layer_loop_start: usize,
    pub layer_loop_end: usize,
    pub num_layers: usize,
    pub layer_rom_bases: Vec<BTreeMap<String, usize>>,
    pub total_cycles_per_token: usize,
}

impl Microprogram {
    pub fn save(&self, out_dir: &Path) -> anyhow::Result<()> {
        fs::create_dir_all(out_dir)?;
        fs::write(
            out_dir.join("schedule.json"),
            serde_json::to_string_pretty(self)?,
        )?;
        let hex_lines: Vec<String> = self.instructions.iter().map(|i| i.to_hex_word()).collect();
        fs::write(out_dir.join("program.hex"), format!("{}\n", hex_lines.join("\n")))?;
        Ok(())
    }
}

pub fn estimate_op_cycles(
    opcode: Opcode,
    dims: &[usize],
    tile_r: usize,
    tile_c: usize,
    max_ctx: usize,
) -> usize {
    match opcode {
        Opcode::LINEAR => {
            let n = dims[0];
            let k = dims[1];
            let tiles_n = n.div_ceil(tile_r);
            let tiles_k = k.div_ceil(tile_c);
            (tiles_n * tiles_k * tile_r) + 8
        }
        Opcode::RMSNORM => (2 * dims[0]) + 12,
        Opcode::ROPE => (dims[0] / 2) + 4,
        Opcode::ATTENTION => (dims[0] * max_ctx) + 24,
        Opcode::ADD => dims[0] + 4,
        Opcode::SWIGLU => dims[0] + 8,
        Opcode::EMBED => dims[0] + 2,
        Opcode::ARGMAX => dims[0] + 4,
        Opcode::HALT => 1,
    }
}

pub fn generate_schedule(
    address_map: &Value,
    config: &Value,
    tile_r: usize,
    tile_c: usize,
) -> Microprogram {
    let hidden_size = config["hidden_size"].as_u64().unwrap() as usize;
    let intermediate_size = config["intermediate_size"].as_u64().unwrap() as usize;
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let num_layers = config["num_hidden_layers"].as_u64().unwrap() as usize;
    let num_q_heads = config["num_attention_heads"].as_u64().unwrap() as usize;
    let num_kv_heads = config["num_key_value_heads"].as_u64().unwrap() as usize;
    let head_dim = config["head_dim"].as_u64().unwrap() as usize;
    let max_ctx = config
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as usize;

    let get_w_base = |key: &str| -> usize {
        address_map
            .pointer(&format!("/weights/{key}/base_addr"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize
    };
    let get_p_base = |key: &str| -> usize {
        address_map
            .pointer(&format!("/requant_params/{key}/base_addr"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize
    };

    let mut instructions = Vec::new();
    let mut layer_rom_bases = Vec::with_capacity(num_layers);

    instructions.push(Instruction::new(
        Opcode::EMBED,
        "embed_gather",
        0,
        0,
        vec![hidden_size],
        get_w_base("embed_tokens"),
        get_p_base("embed_tokens"),
        0,
        estimate_op_cycles(Opcode::EMBED, &[hidden_size], tile_r, tile_c, max_ctx),
    ));

    let loop_start_pc = instructions.len();

    for l_idx in 0..num_layers {
        let mut l_bases = BTreeMap::new();
        for proj in [
            "attn_norm",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "mlp_norm",
            "gate_proj",
            "up_proj",
            "down_proj",
        ] {
            let node_id = format!("{proj}_{l_idx}");
            l_bases.insert(format!("{proj}_w"), get_w_base(&node_id));
            l_bases.insert(format!("{proj}_p"), get_p_base(&node_id));
        }
        layer_rom_bases.push(l_bases);
    }

    instructions.push(Instruction::new(
        Opcode::RMSNORM,
        "layer_attn_norm",
        0,
        1,
        vec![hidden_size],
        0,
        0,
        0,
        estimate_op_cycles(Opcode::RMSNORM, &[hidden_size], tile_r, tile_c, max_ctx),
    ));
    instructions.push(Instruction::new(
        Opcode::LINEAR,
        "layer_q_proj",
        1,
        2,
        vec![num_q_heads * head_dim, hidden_size],
        0,
        0,
        0,
        estimate_op_cycles(
            Opcode::LINEAR,
            &[num_q_heads * head_dim, hidden_size],
            tile_r,
            tile_c,
            max_ctx,
        ),
    ));
    instructions.push(Instruction::new(
        Opcode::LINEAR,
        "layer_k_proj",
        1,
        3,
        vec![num_kv_heads * head_dim, hidden_size],
        0,
        0,
        0,
        estimate_op_cycles(
            Opcode::LINEAR,
            &[num_kv_heads * head_dim, hidden_size],
            tile_r,
            tile_c,
            max_ctx,
        ),
    ));
    instructions.push(Instruction::new(
        Opcode::LINEAR,
        "layer_v_proj",
        1,
        4,
        vec![num_kv_heads * head_dim, hidden_size],
        0,
        0,
        0,
        estimate_op_cycles(
            Opcode::LINEAR,
            &[num_kv_heads * head_dim, hidden_size],
            tile_r,
            tile_c,
            max_ctx,
        ),
    ));
    instructions.push(Instruction::new(
        Opcode::ROPE,
        "layer_q_rope",
        2,
        2,
        vec![num_q_heads, head_dim],
        0,
        0,
        0,
        estimate_op_cycles(Opcode::ROPE, &[num_q_heads * head_dim], tile_r, tile_c, max_ctx),
    ));
    instructions.push(Instruction::new(
        Opcode::ROPE,
        "layer_k_rope",
        3,
        3,
        vec![num_kv_heads, head_dim],
        0,
        0,
        0,
        estimate_op_cycles(Opcode::ROPE, &[num_kv_heads * head_dim], tile_r, tile_c, max_ctx),
    ));
    instructions.push(Instruction::new(
        Opcode::ATTENTION,
        "layer_attention",
        2,
        5,
        vec![num_q_heads, num_kv_heads, head_dim],
        0,
        0,
        0,
        estimate_op_cycles(
            Opcode::ATTENTION,
            &[num_q_heads, num_kv_heads, head_dim],
            tile_r,
            tile_c,
            max_ctx,
        ),
    ));
    instructions.push(Instruction::new(
        Opcode::LINEAR,
        "layer_o_proj",
        5,
        1,
        vec![hidden_size, hidden_size],
        0,
        0,
        0,
        estimate_op_cycles(Opcode::LINEAR, &[hidden_size, hidden_size], tile_r, tile_c, max_ctx),
    ));
    instructions.push(Instruction::new(
        Opcode::ADD,
        "layer_res_add_attn",
        1,
        0,
        vec![hidden_size],
        0,
        0,
        0,
        estimate_op_cycles(Opcode::ADD, &[hidden_size], tile_r, tile_c, max_ctx),
    ));
    instructions.push(Instruction::new(
        Opcode::RMSNORM,
        "layer_mlp_norm",
        0,
        1,
        vec![hidden_size],
        0,
        0,
        0,
        estimate_op_cycles(Opcode::RMSNORM, &[hidden_size], tile_r, tile_c, max_ctx),
    ));
    instructions.push(Instruction::new(
        Opcode::LINEAR,
        "layer_gate_proj",
        1,
        6,
        vec![intermediate_size, hidden_size],
        0,
        0,
        0,
        estimate_op_cycles(
            Opcode::LINEAR,
            &[intermediate_size, hidden_size],
            tile_r,
            tile_c,
            max_ctx,
        ),
    ));
    instructions.push(Instruction::new(
        Opcode::LINEAR,
        "layer_up_proj",
        1,
        7,
        vec![intermediate_size, hidden_size],
        0,
        0,
        0,
        estimate_op_cycles(
            Opcode::LINEAR,
            &[intermediate_size, hidden_size],
            tile_r,
            tile_c,
            max_ctx,
        ),
    ));
    instructions.push(Instruction::new(
        Opcode::SWIGLU,
        "layer_swiglu",
        6,
        8,
        vec![intermediate_size],
        0,
        0,
        0,
        estimate_op_cycles(Opcode::SWIGLU, &[intermediate_size], tile_r, tile_c, max_ctx),
    ));
    instructions.push(Instruction::new(
        Opcode::LINEAR,
        "layer_down_proj",
        8,
        1,
        vec![hidden_size, intermediate_size],
        0,
        0,
        0,
        estimate_op_cycles(
            Opcode::LINEAR,
            &[hidden_size, intermediate_size],
            tile_r,
            tile_c,
            max_ctx,
        ),
    ));
    instructions.push(Instruction::new(
        Opcode::ADD,
        "layer_res_add_mlp",
        1,
        0,
        vec![hidden_size],
        0,
        0,
        0,
        estimate_op_cycles(Opcode::ADD, &[hidden_size], tile_r, tile_c, max_ctx),
    ));

    let loop_end_pc = instructions.len() - 1;

    instructions.push(Instruction::new(
        Opcode::RMSNORM,
        "final_norm",
        0,
        1,
        vec![hidden_size],
        0,
        get_p_base("final_norm"),
        0,
        estimate_op_cycles(Opcode::RMSNORM, &[hidden_size], tile_r, tile_c, max_ctx),
    ));
    instructions.push(Instruction::new(
        Opcode::LINEAR,
        "lm_head",
        1,
        9,
        vec![vocab_size, hidden_size],
        get_w_base("lm_head"),
        get_p_base("lm_head"),
        0,
        estimate_op_cycles(Opcode::LINEAR, &[vocab_size, hidden_size], tile_r, tile_c, max_ctx),
    ));
    instructions.push(Instruction::new(
        Opcode::ARGMAX,
        "argmax",
        9,
        0,
        vec![vocab_size],
        0,
        0,
        0,
        estimate_op_cycles(Opcode::ARGMAX, &[vocab_size], tile_r, tile_c, max_ctx),
    ));
    instructions.push(Instruction::new(
        Opcode::HALT,
        "halt",
        0,
        0,
        vec![0],
        0,
        0,
        0,
        1,
    ));

    let prologue_cycles: usize = instructions[..loop_start_pc]
        .iter()
        .map(|i| i.cycle_estimate)
        .sum();
    let layer_cycles: usize = instructions[loop_start_pc..=loop_end_pc]
        .iter()
        .map(|i| i.cycle_estimate)
        .sum();
    let epilogue_cycles: usize = instructions[loop_end_pc + 1..]
        .iter()
        .map(|i| i.cycle_estimate)
        .sum();
    let total_cycles_per_token = prologue_cycles + layer_cycles * num_layers + epilogue_cycles;

    Microprogram {
        instructions,
        layer_loop_start: loop_start_pc,
        layer_loop_end: loop_end_pc,
        num_layers,
        layer_rom_bases,
        total_cycles_per_token,
    }
}

// ==============================================================================
// 5. Memory Planner (compiler/memory_planner.py)
// ==============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferSpec {
    pub buffer_id: usize,
    pub name: String,
    pub num_elements: usize,
    pub element_bits: usize,
    pub total_bytes: usize,
    pub live_start: usize,
    pub live_end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub max_ctx: usize,
    pub activation_buffers: Vec<BufferSpec>,
    pub activation_bytes_total: usize,
    pub kv_cache_shape: Vec<usize>,
    pub kv_cache_bytes: usize,
    pub total_dynamic_sram_bytes: usize,
    pub total_dynamic_sram_mb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_config: Option<Value>,
}

pub fn plan_memory(config: &Value, max_ctx: usize, res_bits: usize) -> MemoryPlan {
    let hidden_size = config["hidden_size"].as_u64().unwrap() as usize;
    let intermediate_size = config["intermediate_size"].as_u64().unwrap() as usize;
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let num_layers = config["num_hidden_layers"].as_u64().unwrap() as usize;
    let num_q_heads = config["num_attention_heads"].as_u64().unwrap() as usize;
    let num_kv_heads = config["num_key_value_heads"].as_u64().unwrap() as usize;
    let head_dim = config["head_dim"].as_u64().unwrap() as usize;

    let bytes_for = |elems: usize, bits: usize| (elems * bits).div_ceil(8);

    let buffers = vec![
        BufferSpec {
            buffer_id: 0,
            name: "BUF_RES".into(),
            num_elements: hidden_size,
            element_bits: res_bits,
            total_bytes: bytes_for(hidden_size, res_bits),
            live_start: 0,
            live_end: 18,
        },
        BufferSpec {
            buffer_id: 1,
            name: "BUF_ACT".into(),
            num_elements: hidden_size,
            element_bits: 8,
            total_bytes: bytes_for(hidden_size, 8),
            live_start: 1,
            live_end: 16,
        },
        BufferSpec {
            buffer_id: 2,
            name: "BUF_Q".into(),
            num_elements: num_q_heads * head_dim,
            element_bits: 8,
            total_bytes: bytes_for(num_q_heads * head_dim, 8),
            live_start: 2,
            live_end: 7,
        },
        BufferSpec {
            buffer_id: 3,
            name: "BUF_K".into(),
            num_elements: num_kv_heads * head_dim,
            element_bits: 8,
            total_bytes: bytes_for(num_kv_heads * head_dim, 8),
            live_start: 3,
            live_end: 7,
        },
        BufferSpec {
            buffer_id: 4,
            name: "BUF_V".into(),
            num_elements: num_kv_heads * head_dim,
            element_bits: 8,
            total_bytes: bytes_for(num_kv_heads * head_dim, 8),
            live_start: 4,
            live_end: 7,
        },
        BufferSpec {
            buffer_id: 5,
            name: "BUF_ATTN_OUT".into(),
            num_elements: hidden_size,
            element_bits: 8,
            total_bytes: bytes_for(hidden_size, 8),
            live_start: 7,
            live_end: 8,
        },
        BufferSpec {
            buffer_id: 6,
            name: "BUF_MLP_GATE".into(),
            num_elements: intermediate_size,
            element_bits: 8,
            total_bytes: bytes_for(intermediate_size, 8),
            live_start: 11,
            live_end: 13,
        },
        BufferSpec {
            buffer_id: 7,
            name: "BUF_MLP_UP".into(),
            num_elements: intermediate_size,
            element_bits: 8,
            total_bytes: bytes_for(intermediate_size, 8),
            live_start: 12,
            live_end: 13,
        },
        BufferSpec {
            buffer_id: 8,
            name: "BUF_MLP_SWIGLU".into(),
            num_elements: intermediate_size,
            element_bits: 8,
            total_bytes: bytes_for(intermediate_size, 8),
            live_start: 13,
            live_end: 14,
        },
        BufferSpec {
            buffer_id: 9,
            name: "BUF_LOGITS".into(),
            num_elements: vocab_size,
            element_bits: res_bits,
            total_bytes: bytes_for(vocab_size, res_bits),
            live_start: 17,
            live_end: 18,
        },
    ];

    let activation_bytes_total: usize = buffers.iter().map(|b| b.total_bytes).sum();
    let kv_cache_shape = vec![num_layers, 2, num_kv_heads, max_ctx, head_dim];
    let kv_cache_bytes = num_layers * 2 * num_kv_heads * max_ctx * head_dim;
    let total_dynamic_sram_bytes = activation_bytes_total + kv_cache_bytes;
    let total_dynamic_sram_mb =
        ((total_dynamic_sram_bytes as f64) / (1024.0 * 1024.0) * 10000.0).round() / 10000.0;

    MemoryPlan {
        max_ctx,
        activation_buffers: buffers,
        activation_bytes_total,
        kv_cache_shape,
        kv_cache_bytes,
        total_dynamic_sram_bytes,
        total_dynamic_sram_mb,
        model_config: Some(config.clone()),
    }
}

// ==============================================================================
// 6. High-Performance Compiled-Artifact Software Interpreter (compiler/interpreter.py)
// ==============================================================================

#[derive(Debug, Clone)]
struct WeightMatrixCache {
    pub orig_n: usize,
    pub orig_k: usize,
    pub data: Vec<i8>, // Row-major [orig_n * orig_k]
}

pub struct CompiledArtifactInterpreter {
    pub rom_dir: PathBuf,
    pub addr_map: Value,
    pub schedule: Microprogram,
    pub weight_rom: Vec<i8>,
    pub requant_rom: Vec<(i64, i32)>,
    pub table_rom: Vec<i16>,
    pub tile_r: usize,
    pub tile_c: usize,
    pub rope_cos: Vec<i64>,
    pub rope_sin: Vec<i64>,
    pub rope_half_dim: usize,
    pub rsqrt_lut: Vec<i64>,
    pub exp2_lut: Vec<i64>,
    pub recip_lut: Vec<i64>,
    pub silu_lut: Vec<i64>,
    pub buffers: HashMap<usize, Vec<i64>>,
    pub kv_cache: HashMap<usize, (Vec<i8>, Vec<i8>)>, // layer_idx -> (K, V) [num_kv_heads * 256 * hd]
    pub traces: HashMap<String, Vec<i64>>,
    weight_cache: HashMap<String, WeightMatrixCache>,
}

impl CompiledArtifactInterpreter {
    pub fn new(rom_dir: &Path) -> anyhow::Result<Self> {
        let addr_map: Value =
            serde_json::from_str(&fs::read_to_string(rom_dir.join("address_map.json"))?)?;
        let schedule: Microprogram =
            serde_json::from_str(&fs::read_to_string(rom_dir.join("schedule.json"))?)?;

        let tile_r = addr_map["engine_config"]["tile_r"].as_u64().unwrap() as usize;
        let tile_c = addr_map["engine_config"]["tile_c"].as_u64().unwrap() as usize;

        let weight_rom = Self::load_hex_i8(&rom_dir.join("weights.hex"))?;
        let requant_rom = Self::load_param_hex(&rom_dir.join("requant_params.hex"))?;
        let table_rom = Self::load_hex_i16(&rom_dir.join("tables.hex"))?;

        let tbl_info = &addr_map["tables"];
        let cos_base = tbl_info["rope_cos"]["base_addr"].as_u64().unwrap() as usize;
        let cos_len = tbl_info["rope_cos"]["length"].as_u64().unwrap() as usize;
        let rope_half_dim = tbl_info["rope_cos"]["shape"][1].as_u64().unwrap() as usize;
        let rope_cos: Vec<i64> = table_rom[cos_base..cos_base + cos_len]
            .iter()
            .map(|&v| v as i64)
            .collect();

        let sin_base = tbl_info["rope_sin"]["base_addr"].as_u64().unwrap() as usize;
        let sin_len = tbl_info["rope_sin"]["length"].as_u64().unwrap() as usize;
        let rope_sin: Vec<i64> = table_rom[sin_base..sin_base + sin_len]
            .iter()
            .map(|&v| v as i64)
            .collect();

        let r_base = tbl_info["rsqrt_mantissa_lut"]["base_addr"].as_u64().unwrap() as usize;
        let r_len = tbl_info["rsqrt_mantissa_lut"]["length"].as_u64().unwrap() as usize;
        let rsqrt_lut: Vec<i64> = table_rom[r_base..r_base + r_len]
            .iter()
            .map(|&v| (v as u16) as i64)
            .collect();

        let exp2_lut = build_exp2_fraction_lut(256, 16);

        let rec_base = tbl_info["reciprocal_lut"]["base_addr"].as_u64().unwrap() as usize;
        let rec_len = tbl_info["reciprocal_lut"]["length"].as_u64().unwrap() as usize;
        let recip_lut: Vec<i64> = table_rom[rec_base..rec_base + rec_len]
            .iter()
            .map(|&v| (v as u16) as i64)
            .collect();

        let silu_base = tbl_info["silu_lut"]["base_addr"].as_u64().unwrap() as usize;
        let silu_len = tbl_info["silu_lut"]["length"].as_u64().unwrap() as usize;
        let silu_lut: Vec<i64> = table_rom[silu_base..silu_base + silu_len]
            .iter()
            .map(|&v| (v as i8) as i64)
            .collect();

        let mut interp = Self {
            rom_dir: rom_dir.to_path_buf(),
            addr_map,
            schedule,
            weight_rom,
            requant_rom,
            table_rom,
            tile_r,
            tile_c,
            rope_cos,
            rope_sin,
            rope_half_dim,
            rsqrt_lut,
            exp2_lut,
            recip_lut,
            silu_lut,
            buffers: HashMap::new(),
            kv_cache: HashMap::new(),
            traces: HashMap::new(),
            weight_cache: HashMap::new(),
        };
        interp.precache_all_weights();
        Ok(interp)
    }

    fn load_hex_i8(path: &Path) -> anyhow::Result<Vec<i8>> {
        let raw = fs::read(path)?;
        let mut lut = [0u8; 256];
        for i in 0..10 {
            lut[(b'0' + i) as usize] = i;
        }
        for i in 0..6 {
            lut[(b'a' + i) as usize] = 10 + i;
            lut[(b'A' + i) as usize] = 10 + i;
        }
        if !raw.is_empty() && raw.len() % 3 == 0 && raw[2] == b'\n' {
            let count = raw.len() / 3;
            let mut out = vec![0i8; count];
            out.par_iter_mut().enumerate().for_each(|(idx, dst)| {
                let b0 = raw[idx * 3] as usize;
                let b1 = raw[idx * 3 + 1] as usize;
                *dst = ((lut[b0] << 4) | lut[b1]) as i8;
            });
            return Ok(out);
        }
        let text = String::from_utf8(raw)?;
        let mut out = Vec::new();
        for line in text.lines() {
            let s = line.trim();
            if !s.is_empty() {
                let v = u8::from_str_radix(s, 16)?;
                out.push(v as i8);
            }
        }
        Ok(out)
    }

    fn load_hex_i16(path: &Path) -> anyhow::Result<Vec<i16>> {
        let text = fs::read_to_string(path)?;
        let mut out = Vec::new();
        for line in text.lines() {
            let s = line.trim();
            if !s.is_empty() {
                let v = u16::from_str_radix(s, 16)?;
                out.push(v as i16);
            }
        }
        Ok(out)
    }

    fn load_param_hex(path: &Path) -> anyhow::Result<Vec<(i64, i32)>> {
        let text = fs::read_to_string(path)?;
        let mut out = Vec::new();
        for line in text.lines() {
            let s = line.trim();
            if s.len() >= 16 {
                let m_u32 = u32::from_str_radix(&s[0..8], 16)?;
                let s_u32 = u32::from_str_radix(&s[8..16], 16)?;
                let m_i64 = (m_u32 as i32) as i64;
                let s_i32 = s_u32 as i32;
                out.push((m_i64, s_i32));
            }
        }
        Ok(out)
    }

    fn precache_all_weights(&mut self) {
        let Some(weights_obj) = self.addr_map.get("weights").and_then(|v| v.as_object()) else {
            return;
        };
        let keys: Vec<String> = weights_obj.keys().cloned().collect();
        for node_id in keys {
            let info = &self.addr_map["weights"][&node_id];
            let base = info["base_addr"].as_u64().unwrap() as usize;
            let len = info["length"].as_u64().unwrap() as usize;
            let n_pad = info["padded_shape"][0].as_u64().unwrap() as usize;
            let k_pad = info["padded_shape"][1].as_u64().unwrap() as usize;
            let num_tr = info["num_tiles"][0].as_u64().unwrap() as usize;
            let num_tc = info["num_tiles"][1].as_u64().unwrap() as usize;
            let orig_n = info["orig_shape"][0].as_u64().unwrap() as usize;
            let orig_k = info["orig_shape"][1].as_u64().unwrap() as usize;

            let raw = &self.weight_rom[base..base + len];
            let mut padded = vec![0i8; n_pad * k_pad];
            let tr_size = self.tile_r;
            let tc_size = self.tile_c;
            let tile_elems = tr_size * tc_size;

            for tr in 0..num_tr {
                for tc in 0..num_tc {
                    let tile_idx = tr * num_tc + tc;
                    let tile_slice = &raw[tile_idx * tile_elems..(tile_idx + 1) * tile_elems];
                    for r in 0..tr_size {
                        let dst_row = (tr * tr_size + r) * k_pad + tc * tc_size;
                        padded[dst_row..dst_row + tc_size]
                            .copy_from_slice(&tile_slice[r * tc_size..(r + 1) * tc_size]);
                    }
                }
            }

            let mut unpadded = vec![0i8; orig_n * orig_k];
            for r in 0..orig_n {
                unpadded[r * orig_k..(r + 1) * orig_k]
                    .copy_from_slice(&padded[r * k_pad..r * k_pad + orig_k]);
            }
            self.weight_cache.insert(
                node_id,
                WeightMatrixCache {
                    orig_n,
                    orig_k,
                    data: unpadded,
                },
            );
        }
    }

    pub fn reset(&mut self) {
        self.buffers.clear();
        self.kv_cache.clear();
        self.traces.clear();
    }

    pub fn decode_step(&mut self, token_id: usize, pos: usize) -> (Vec<i64>, usize) {
        let num_layers = self.schedule.num_layers;
        let loop_start = self.schedule.layer_loop_start;
        let loop_end = self.schedule.layer_loop_end;
        let instructions = self.schedule.instructions.clone();

        for ins in &instructions[..loop_start] {
            self.execute_instruction(ins, 0, pos, token_id);
        }
        for l_idx in 0..num_layers {
            for ins in &instructions[loop_start..=loop_end] {
                self.execute_instruction(ins, l_idx, pos, token_id);
            }
        }
        for ins in &instructions[loop_end + 1..] {
            if Opcode::from_u8(ins.opcode_val) == Opcode::HALT {
                break;
            }
            self.execute_instruction(ins, 0, pos, token_id);
        }

        let logits = self.buffers.get(&9).cloned().unwrap_or_else(|| vec![0]);
        let next_tok = self
            .buffers
            .get(&0)
            .and_then(|v| v.first().copied())
            .unwrap_or(0) as usize;
        (logits, next_tok)
    }

    fn execute_instruction(
        &mut self,
        ins: &Instruction,
        layer_idx: usize,
        pos: usize,
        token_id: usize,
    ) {
        let opcode = Opcode::from_u8(ins.opcode_val);
        let name = &ins.name;
        let src = ins.src_buf;
        let dst = ins.dst_buf;
        let dims = &ins.dims;

        match opcode {
            Opcode::EMBED => {
                let w = &self.weight_cache["embed_tokens"];
                let d = w.orig_k;
                let row = &w.data[token_id * d..(token_id + 1) * d];
                let p_base = self.addr_map["requant_params"]["embed_tokens"]["base_addr"]
                    .as_u64()
                    .unwrap() as usize
                    + token_id;
                let (m, s) = self.requant_rom[p_base];
                let res: Vec<i64> = row
                    .iter()
                    .map(|&v| sat(rshift_rnd((v as i64) * m, s), -8388608, 8388607))
                    .collect();
                self.traces
                    .insert(format!("embed_tokens_tok{token_id}"), res.clone());
                self.traces.insert("embed_tokens_current".into(), res.clone());
                self.buffers.insert(dst, res);
            }
            Opcode::RMSNORM => {
                let x = &self.buffers[&src];
                let d = dims[0];
                let ss: i64 = x.iter().map(|&v| v * v).sum();
                let mean = ss / (d as i64);
                if mean <= 0 {
                    let zeros = vec![0i64; d];
                    self.traces.insert(format!("{name}_L{layer_idx}"), zeros.clone());
                    self.buffers.insert(dst, zeros);
                    return;
                }
                let b = bit_length_u64(mean as u64) as i32;
                let sh_amt = if b > 8 { (b - 7) & !1 } else { 0 };
                let mantissa = ((mean >> sh_amt) & 0xFF) as usize;
                let rsqrt_val = self.rsqrt_lut[mantissa];
                let exp_sh = sh_amt / 2;

                let node_key = if name.contains("attn") {
                    format!("attn_norm_{layer_idx}")
                } else if name.contains("mlp") {
                    format!("mlp_norm_{layer_idx}")
                } else {
                    "final_norm".to_string()
                };

                let out: Vec<i64> = if let Some(p_info) =
                    self.addr_map["requant_params"].get(&node_key)
                {
                    let p_base = p_info["base_addr"].as_u64().unwrap() as usize;
                    let params = &self.requant_rom[p_base..p_base + d];
                    x.iter()
                        .zip(params.iter())
                        .map(|(&xi, &(m_i, s_i))| {
                            let stg1 = if exp_sh > 0 {
                                rshift_rnd(xi * rsqrt_val, exp_sh)
                            } else {
                                xi * rsqrt_val
                            };
                            requant(stg1 * m_i, 1, 19 + s_i, -127, 127)
                        })
                        .collect()
                } else {
                    x.iter()
                        .map(|&xi| {
                            let stg1 = if exp_sh > 0 {
                                rshift_rnd(xi * rsqrt_val, exp_sh)
                            } else {
                                xi * rsqrt_val
                            };
                            sat(rshift_rnd(stg1, 19), -127, 127)
                        })
                        .collect()
                };
                self.traces.insert(format!("{name}_L{layer_idx}"), out.clone());
                self.buffers.insert(dst, out);
            }
            Opcode::LINEAR => {
                let x = &self.buffers[&src];
                let node_key = if let Some(stripped) = name.strip_prefix("layer_") {
                    format!("{stripped}_{layer_idx}")
                } else {
                    name.clone()
                };
                let w = &self.weight_cache[&node_key];
                let out_n = w.orig_n;
                let in_k = w.orig_k;
                let is_res_out =
                    dst == 0 || dst == 9 || name.contains("o_proj") || name.contains("down_proj");
                let out_min = if is_res_out { -8388608i64 } else { -127i64 };
                let out_max = if is_res_out { 8388607i64 } else { 127i64 };

                let p_base_opt = self.addr_map["requant_params"]
                    .get(&node_key)
                    .and_then(|p| p["base_addr"].as_u64())
                    .map(|v| v as usize);

                let res: Vec<i64> = if out_n >= 256 {
                    // Multi-threaded parallel matrix-vector product (massive speedup on lm_head / MLP)
                    (0..out_n)
                        .into_par_iter()
                        .map(|r| {
                            let row = &w.data[r * in_k..(r + 1) * in_k];
                            let mut acc = 0i64;
                            for (&xk, &wr) in x.iter().zip(row.iter()) {
                                acc += xk * (wr as i64);
                            }
                            if let Some(p_base) = p_base_opt {
                                let (m, s) = self.requant_rom[p_base + r];
                                requant(acc, m, s, out_min, out_max)
                            } else {
                                acc
                            }
                        })
                        .collect()
                } else {
                    (0..out_n)
                        .map(|r| {
                            let row = &w.data[r * in_k..(r + 1) * in_k];
                            let mut acc = 0i64;
                            for (&xk, &wr) in x.iter().zip(row.iter()) {
                                acc += xk * (wr as i64);
                            }
                            if let Some(p_base) = p_base_opt {
                                let (m, s) = self.requant_rom[p_base + r];
                                requant(acc, m, s, out_min, out_max)
                            } else {
                                acc
                            }
                        })
                        .collect()
                };

                self.traces.insert(format!("{name}_L{layer_idx}"), res.clone());
                self.buffers.insert(dst, res);
            }
            Opcode::ROPE => {
                let x = &self.buffers[&src];
                let num_heads = dims[0];
                let hd = if dims.len() > 1 {
                    dims[1]
                } else {
                    x.len() / num_heads
                };
                let half = hd / 2;
                let cos_base = pos * self.rope_half_dim;
                let cos_m = &self.rope_cos[cos_base..cos_base + half];
                let sin_m = &self.rope_sin[cos_base..cos_base + half];

                let mut out = vec![0i64; num_heads * hd];
                for h in 0..num_heads {
                    let base = h * hd;
                    for i in 0..half {
                        let x1 = x[base + i];
                        let x2 = x[base + half + i];
                        let c = cos_m[i];
                        let s = sin_m[i];
                        out[base + i] = sat(rshift_rnd(x1 * c - x2 * s, 14), -127, 127);
                        out[base + half + i] = sat(rshift_rnd(x2 * c + x1 * s, 14), -127, 127);
                    }
                }
                self.traces.insert(format!("{name}_L{layer_idx}"), out.clone());
                self.buffers.insert(dst, out);
            }
            Opcode::ATTENTION => {
                let num_q_heads = dims[0];
                let num_kv_heads = dims[1];
                let hd = dims[2];
                let q = self.buffers[&2].clone();
                let k = self.buffers[&3].clone();
                let v = self.buffers[&4].clone();

                let entry = self.kv_cache.entry(layer_idx).or_insert_with(|| {
                    (
                        vec![0i8; num_kv_heads * 256 * hd],
                        vec![0i8; num_kv_heads * 256 * hd],
                    )
                });
                for h in 0..num_kv_heads {
                    let dst_base = (h * 256 + pos) * hd;
                    let src_base = h * hd;
                    for d_i in 0..hd {
                        entry.0[dst_base + d_i] = sat_int8(k[src_base + d_i]);
                        entry.1[dst_base + d_i] = sat_int8(v[src_base + d_i]);
                    }
                }

                let (k_cache, v_cache) = &self.kv_cache[&layer_idx];
                let gqa_group = num_q_heads / num_kv_heads;
                let mut attn_out = vec![0i64; num_q_heads * hd];

                for q_idx in 0..num_q_heads {
                    let kv_idx = q_idx / gqa_group;
                    let q_head = &q[q_idx * hd..(q_idx + 1) * hd];

                    let mut scores = Vec::with_capacity(pos + 1);
                    for t in 0..=pos {
                        let k_base = (kv_idx * 256 + t) * hd;
                        let k_head = &k_cache[k_base..k_base + hd];
                        let mut dot = 0i64;
                        for (&qi, &ki) in q_head.iter().zip(k_head.iter()) {
                            dot += qi * (ki as i64);
                        }
                        scores.push(dot >> 1);
                    }
                    let max_score = *scores.iter().max().unwrap_or(&0);
                    let mut exp_vals = Vec::with_capacity(pos + 1);
                    for &sc in &scores {
                        let diff = (max_score - sc).unsigned_abs();
                        let int_part = (diff >> 8) as usize;
                        let frac_idx = (diff & 0xFF) as usize;
                        let frac_val = self.exp2_lut[frac_idx];
                        let exp_v = (frac_val >> int_part.min(16)).max(1);
                        exp_vals.push(exp_v);
                    }
                    let sigma = exp_vals.iter().sum::<i64>().max(1);
                    let probs: Vec<i64> = exp_vals.iter().map(|&ev| (ev * 32767) / sigma).collect();

                    for d_i in 0..hd {
                        let mut acc_v = 0i64;
                        for (t, &pt) in probs.iter().enumerate() {
                            let v_base = (kv_idx * 256 + t) * hd;
                            acc_v += pt * (v_cache[v_base + d_i] as i64);
                        }
                        attn_out[q_idx * hd + d_i] = sat(rshift_rnd(acc_v, 15), -127, 127);
                    }
                }
                self.traces
                    .insert(format!("{name}_L{layer_idx}"), attn_out.clone());
                self.buffers.insert(dst, attn_out);
            }
            Opcode::ADD => {
                let x_src = &self.buffers[&src];
                let x_dst = &self.buffers[&dst];
                let res: Vec<i64> = x_src
                    .iter()
                    .zip(x_dst.iter())
                    .map(|(&a, &b)| sat(a + b, -8388608, 8388607))
                    .collect();
                self.traces.insert(format!("{name}_L{layer_idx}"), res.clone());
                self.buffers.insert(dst, res);
            }
            Opcode::SWIGLU => {
                let gate = &self.buffers[&src];
                let up = &self.buffers[&7];
                let res: Vec<i64> = gate
                    .iter()
                    .zip(up.iter())
                    .map(|(&g, &u)| {
                        let idx = (g + 256).clamp(0, 511) as usize;
                        let silu_val = self.silu_lut[idx];
                        sat(rshift_rnd(silu_val * u, 6), -127, 127)
                    })
                    .collect();
                self.traces.insert(format!("{name}_L{layer_idx}"), res.clone());
                self.buffers.insert(dst, res);
            }
            Opcode::ARGMAX => {
                let logits = &self.buffers[&src];
                let mut best_id = 0usize;
                let mut best_val = logits[0];
                for (idx, &val) in logits.iter().enumerate().skip(1) {
                    if val > best_val {
                        best_val = val;
                        best_id = idx;
                    }
                }
                let out = vec![best_id as i64];
                self.traces.insert(format!("{name}_tok"), out.clone());
                self.buffers.insert(dst, out);
            }
            Opcode::HALT => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_weights_and_bounds() {
        let w = vec![1.0, -0.5, 0.0, 0.25];
        let (wq, scales) = quantize_weights(&w, 2, 2);
        assert_eq!(wq[0], 127);
        assert_eq!(wq[1], -64);
        assert_eq!(scales.len(), 2);
        assert!(check_accumulator_bounds(576, 0).is_ok());
    }

    #[test]
    fn test_rust_interpreter_matches_python_on_smollm2_bundle() {
        let hw_dir = Path::new("../../build/smollm2_hw");
        let ue3_path = Path::new("../../results/part2/ue3_layer0_traces.json");
        if !hw_dir.join("top_accel.v").is_file() || !ue3_path.is_file() {
            return;
        }
        let ue3: Value = serde_json::from_str(&fs::read_to_string(ue3_path).unwrap()).unwrap();
        let prompt_tokens: Vec<usize> = ue3["prompt_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let golden_tokens: Vec<usize> = ue3["golden_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let mut interp = CompiledArtifactInterpreter::new(hw_dir).unwrap();
        let mut rust_tokens = Vec::with_capacity(golden_tokens.len());
        for pos in 0..golden_tokens.len() {
            let tok_in = if pos < prompt_tokens.len() {
                prompt_tokens[pos]
            } else {
                *rust_tokens.last().unwrap()
            };
            let (_, tok_out) = interp.decode_step(tok_in, pos);
            rust_tokens.push(tok_out);
        }
        assert_eq!(
            rust_tokens, golden_tokens,
            "Rust CompiledArtifactInterpreter must match Python UE.3 30-layer SmolLM2 tokens 100% bit-for-bit!"
        );
    }
}
