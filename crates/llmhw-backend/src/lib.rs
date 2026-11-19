//! Verilog-2001/2005 Hardware Codegen (Part I Design F/P/P_bus & Part II `top_generator`) (Phase R4).
//!
//! Strict Layering (Golden Rule 6): No ML framework dependencies.

use llmhw_compiler::{
    assign_formats_and_scales, generate_schedule, pack_rom_artifacts, pad_matrix_i8,
    FormatAssignmentResult, Microprogram,
};
use llmhw_ir::{sha256_hex, GraphV2, LinearOp, NpyArray};
use llmhw_numerics::build_exp2_fraction_lut;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

// ==============================================================================
// 1. Part I Identifier Sanitization, Tiling & Constant-Weight Encoding
// ==============================================================================

pub fn sanitize_identifier(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    if out.is_empty() || out.chars().next().unwrap().is_ascii_digit() {
        format!("m_{out}")
    } else {
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileInfo {
    pub tile_i: usize,
    pub tile_j: usize,
    pub row_start: usize,
    pub row_end: usize,
    pub col_start: usize,
    pub col_end: usize,
    pub weights_sha256: String,
    pub nonzero_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileManifest {
    pub layer_name: String,
    pub n_orig: usize,
    pub k_orig: usize,
    pub n_padded: usize,
    pub k_padded: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub num_tiles_n: usize,
    pub num_tiles_k: usize,
    pub tiles: Vec<TileInfo>,
    pub manifest_hash: String,
}

impl TileManifest {
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let s = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&s)?)
    }
}

pub fn tile_weight_matrix(
    w: &[i8],
    n: usize,
    k: usize,
    tile_n: usize,
    tile_k: usize,
    layer_name: &str,
) -> (TileManifest, Vec<Vec<i8>>) {
    let (w_pad, n_padded, k_padded) = pad_matrix_i8(w, n, k, tile_n, tile_k);
    let num_tiles_n = n_padded / tile_n;
    let num_tiles_k = k_padded / tile_k;

    let mut tiles = Vec::with_capacity(num_tiles_n * num_tiles_k);
    let mut tile_arrays = Vec::with_capacity(num_tiles_n * num_tiles_k);

    for ti in 0..num_tiles_n {
        for tj in 0..num_tiles_k {
            let mut sub = Vec::with_capacity(tile_n * tile_k);
            let mut nz = 0usize;
            for r in 0..tile_n {
                let row_idx = (ti * tile_n + r) * k_padded + tj * tile_k;
                for c in 0..tile_k {
                    let v = w_pad[row_idx + c];
                    if v != 0 {
                        nz += 1;
                    }
                    sub.push(v);
                }
            }
            let raw_bytes: Vec<u8> = sub.iter().map(|&x| x as u8).collect();
            let sha = sha256_hex(&raw_bytes);
            tiles.push(TileInfo {
                tile_i: ti,
                tile_j: tj,
                row_start: ti * tile_n,
                row_end: (ti + 1) * tile_n,
                col_start: tj * tile_k,
                col_end: (tj + 1) * tile_k,
                weights_sha256: sha,
                nonzero_count: nz,
            });
            tile_arrays.push(sub);
        }
    }

    let summary_json = serde_json::to_string(&tiles).unwrap_or_default();
    let manifest_hash = sha256_hex(summary_json.as_bytes());

    (
        TileManifest {
            layer_name: layer_name.to_string(),
            n_orig: n,
            k_orig: k,
            n_padded,
            k_padded,
            tile_n,
            tile_k,
            num_tiles_n,
            num_tiles_k,
            tiles,
            manifest_hash,
        },
        tile_arrays,
    )
}

/// Format an INT8 weight value `[-127, 127]` as a Verilog signed literal (`8'sd...` or `-8'sd...`).
pub fn format_weight_literal(w: i8) -> anyhow::Result<String> {
    if w == -128 {
        anyhow::bail!("Weight -128 is outside symmetric INT8 range [-127, 127]");
    }
    if w < 0 {
        Ok(format!("-8'sd{}", -(w as i32)))
    } else {
        Ok(format!("8'sd{w}"))
    }
}

/// Encode single product term `x_name * w` with zero-skipping.
pub fn encode_product_term(x_name: &str, w: i8) -> anyhow::Result<Option<String>> {
    if w == 0 {
        return Ok(None);
    }
    let lit = format_weight_literal(w)?;
    Ok(Some(format!("($signed({x_name}) * $signed({lit}))")))
}

/// Generate synthesizable fixed-weight Verilog-2001 module (Part I Design F).
pub fn generate_full_module(
    op: &LinearOp,
    manifest: &TileManifest,
    module_name: Option<&str>,
) -> anyhow::Result<String> {
    let raw_name = module_name
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("llmhw_{}", op.name));
    let top_name = sanitize_identifier(&raw_name);
    let n = op.out_features;
    let k = op.in_features;

    let mut lines = Vec::new();
    lines.push("// Generated by LLMHW (Rust Edition) — Design F (Fixed-Weight)".into());
    lines.push(format!("// Layer: {} (N={n}, K={k}), Manifest Hash: {}", op.name, manifest.manifest_hash));
    lines.push(format!("module {top_name} ("));
    lines.push("    input  wire                  clk,".into());
    lines.push("    input  wire                  rst_n,".into());
    lines.push("    input  wire                  valid_in,".into());
    lines.push(format!("    input  wire [{}:0] x_in,", k * 8 - 1));
    lines.push("    output reg                   valid_out,".into());
    lines.push(format!("    output reg  [{}:0] y_out", n * 32 - 1));
    lines.push(");".into());
    lines.push("".into());

    for col in 0..k {
        let hi = (col + 1) * 8 - 1;
        let lo = col * 8;
        lines.push(format!("    wire signed [7:0] x_{col} = x_in[{hi}:{lo}];"));
    }
    lines.push("".into());

    for row in 0..n {
        let mut terms = Vec::new();
        for col in 0..k {
            let w = op.weight[row * k + col];
            if let Some(term) = encode_product_term(&format!("x_{col}"), w)? {
                terms.push(term);
            }
        }
        let expr = if terms.is_empty() {
            "32'sd0".to_string()
        } else {
            terms.join(" + ")
        };
        lines.push(format!("    wire signed [31:0] acc_{row} = {expr};"));
    }
    lines.push("".into());
    lines.push("    always @(posedge clk or negedge rst_n) begin".into());
    lines.push("        if (!rst_n) begin".into());
    lines.push("            valid_out <= 1'b0;".into());
    lines.push(format!("            y_out     <= {}'d0;", n * 32));
    lines.push("        end else begin".into());
    lines.push("            valid_out <= valid_in;".into());
    lines.push("            if (valid_in) begin".into());
    for row in 0..n {
        let hi = (row + 1) * 32 - 1;
        let lo = row * 32;
        lines.push(format!("                y_out[{hi}:{lo}] <= acc_{row};"));
    }
    lines.push("            end".into());
    lines.push("        end".into());
    lines.push("    end".into());
    lines.push("endmodule".into());

    Ok(format!("{}\n", lines.join("\n")))
}

/// Generate synthesizable programmable baseline Verilog-2001 module (Part I Design P).
pub fn generate_full_module_p(op: &LinearOp, manifest: &TileManifest) -> String {
    let top_name = sanitize_identifier(&format!("llmhw_{}_p", op.name));
    let n = op.out_features;
    let k = op.in_features;
    let total_w = n * k;
    let addr_w = (total_w.next_power_of_two().trailing_zeros() as usize).max(1);

    let mut lines = Vec::new();
    lines.push("// Generated by LLMHW (Rust Edition) — Design P (Programmable Baseline)".into());
    lines.push(format!("// Layer: {} (N={n}, K={k}), Manifest Hash: {}", op.name, manifest.manifest_hash));
    lines.push(format!("module {top_name} ("));
    lines.push("    input  wire                  clk,".into());
    lines.push("    input  wire                  rst_n,".into());
    lines.push("    input  wire                  w_we,".into());
    lines.push(format!("    input  wire [{}:0] w_addr,", addr_w - 1));
    lines.push("    input  wire signed [7:0]     w_data,".into());
    lines.push("    input  wire                  valid_in,".into());
    lines.push(format!("    input  wire [{}:0] x_in,", k * 8 - 1));
    lines.push("    output reg                   valid_out,".into());
    lines.push(format!("    output reg  [{}:0] y_out", n * 32 - 1));
    lines.push(");".into());
    lines.push(format!("    reg signed [7:0] w_reg [0:{}];", total_w - 1));
    lines.push("    integer wi;".into());
    lines.push("    always @(posedge clk or negedge rst_n) begin".into());
    lines.push("        if (!rst_n) begin".into());
    lines.push(format!("            for (wi = 0; wi < {total_w}; wi = wi + 1) w_reg[wi] <= 8'sd0;"));
    lines.push("        end else if (w_we) begin".into());
    lines.push("            w_reg[w_addr] <= w_data;".into());
    lines.push("        end".into());
    lines.push("    end".into());

    for col in 0..k {
        let hi = (col + 1) * 8 - 1;
        let lo = col * 8;
        lines.push(format!("    wire signed [7:0] x_{col} = x_in[{hi}:{lo}];"));
    }
    for row in 0..n {
        let terms: Vec<String> = (0..k)
            .map(|col| format!("($signed(x_{col}) * $signed(w_reg[{}]))", row * k + col))
            .collect();
        lines.push(format!("    wire signed [31:0] acc_{row} = {};", terms.join(" + ")));
    }
    lines.push("    always @(posedge clk or negedge rst_n) begin".into());
    lines.push("        if (!rst_n) begin".into());
    lines.push("            valid_out <= 1'b0;".into());
    lines.push(format!("            y_out     <= {}'d0;", n * 32));
    lines.push("        end else begin".into());
    lines.push("            valid_out <= valid_in;".into());
    lines.push("            if (valid_in) begin".into());
    for row in 0..n {
        let hi = (row + 1) * 32 - 1;
        let lo = row * 32;
        lines.push(format!("                y_out[{hi}:{lo}] <= acc_{row};"));
    }
    lines.push("            end".into());
    lines.push("        end".into());
    lines.push("    end".into());
    lines.push("endmodule".into());

    format!("{}\n", lines.join("\n"))
}

// ==============================================================================
// 2. Part II Top-Level Accelerator Generator (`backend/top_generator.py`)
// ==============================================================================

pub fn generate_top_accel_verilog(
    config: &Value,
    rom_files: Option<&BTreeMap<String, String>>,
    module_name: &str,
    rom_depths: Option<&BTreeMap<String, usize>>,
    template_content: &str,
) -> String {
    let get_usize = |k: &str, def: usize| -> usize {
        config.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(def)
    };
    let vocab_size = get_usize("vocab_size", 256);
    let hidden_size = get_usize("hidden_size", 64);
    let intermediate_size = get_usize("intermediate_size", 128);
    let num_layers = get_usize("num_hidden_layers", 2);
    let num_q_heads = get_usize("num_attention_heads", 4);
    let num_kv_heads = get_usize("num_key_value_heads", 2);
    let head_dim = get_usize("head_dim", 16);
    let max_ctx = get_usize("max_position_embeddings", 32);
    let tile_r = get_usize("tile_r", 8);
    let tile_c = get_usize("tile_c", 16);

    let empty_depths = BTreeMap::new();
    let depths = rom_depths.unwrap_or(&empty_depths);
    let w_depth = (*depths.get("weights").unwrap_or(&32768)).max(32768);
    let p_depth = (*depths.get("params").unwrap_or(&2048)).max(2048);
    let prog_depth = (*depths.get("prog").unwrap_or(&256)).max(256);
    let bases_depth = (*depths.get("bases").unwrap_or(&512)).max(512);

    let bit_len = |x: usize| -> usize { (usize::BITS - x.leading_zeros()) as usize };
    let w_addr_w = bit_len(w_depth - 1).max(16);
    let p_addr_w = bit_len(p_depth - 1).max(16);
    let bases_addr_w = bit_len(bases_depth - 1).max(9);

    let empty_files = BTreeMap::new();
    let files = rom_files.unwrap_or(&empty_files);
    let f_get = |k: &str| files.get(k).map(|s| s.as_str()).unwrap_or("");

    let param_block = format!(
        "module {module_name} #(\n\
         \x20   parameter VOCAB_SIZE        = {vocab_size},\n\
         \x20   parameter HIDDEN_SIZE       = {hidden_size},\n\
         \x20   parameter INTERMEDIATE_SIZE = {intermediate_size},\n\
         \x20   parameter NUM_LAYERS        = {num_layers},\n\
         \x20   parameter NUM_Q_HEADS       = {num_q_heads},\n\
         \x20   parameter NUM_KV_HEADS      = {num_kv_heads},\n\
         \x20   parameter HEAD_DIM          = {head_dim},\n\
         \x20   parameter MAX_CTX           = {max_ctx},\n\
         \x20   parameter TILE_R            = {tile_r},\n\
         \x20   parameter TILE_C            = {tile_c},\n\
         \x20   parameter PROG_ADDR_W       = 8,\n\
         \x20   parameter LAYER_W           = 5,\n\
         \x20   parameter ROM_WEIGHTS_ADDR_W = {w_addr_w},\n\
         \x20   parameter ROM_PARAMS_ADDR_W  = {p_addr_w},\n\
         \x20   parameter ROM_BASES_ADDR_W   = {bases_addr_w},\n\
         \x20   parameter ROM_WEIGHTS_DEPTH = {w_depth},\n\
         \x20   parameter ROM_PARAMS_DEPTH  = {p_depth},\n\
         \x20   parameter ROM_PROG_DEPTH    = {prog_depth},\n\
         \x20   parameter ROM_BASES_DEPTH   = {bases_depth},\n\
         \x20   parameter ROM_WEIGHTS_FILE  = \"{}\",\n\
         \x20   parameter ROM_PARAMS_FILE   = \"{}\",\n\
         \x20   parameter ROM_PROG_FILE     = \"{}\",\n\
         \x20   parameter ROM_BASES_FILE    = \"{}\",\n\
         \x20   parameter ROM_EXP2_FILE     = \"{}\",\n\
         \x20   parameter ROM_RSQRT_FILE    = \"{}\",\n\
         \x20   parameter ROM_SILU_FILE     = \"{}\",\n\
         \x20   parameter ROM_COS_FILE      = \"{}\",\n\
         \x20   parameter ROM_SIN_FILE      = \"{}\"\n\
         )(",
        f_get("weights"),
        f_get("requant_params"),
        f_get("program"),
        f_get("layer_bases"),
        f_get("exp2"),
        f_get("rsqrt"),
        f_get("silu"),
        f_get("rope_cos"),
        f_get("rope_sin"),
    );

    let re = Regex::new(r"(?s)module\s+\w+\s*#\(.*?\)\s*\(").unwrap();
    re.replace(template_content, param_block.as_str()).into_owned()
}

pub struct HardwareBundleResult {
    pub out_dir: String,
    pub address_map: Value,
    pub microprogram: Microprogram,
    pub format_result: FormatAssignmentResult,
    pub top_file: String,
}

#[allow(clippy::too_many_arguments)]
pub fn emit_hardware_bundle(
    graph: GraphV2,
    weights_dict: &HashMap<String, NpyArray<f64>>,
    config: &Value,
    frozen_config: &Value,
    out_dir: &Path,
    tile_r: usize,
    tile_c: usize,
    max_ctx: usize,
    calib_scales: Option<&BTreeMap<String, Value>>,
    top_accel_template: &str,
) -> anyhow::Result<HardwareBundleResult> {
    fs::create_dir_all(out_dir)?;

    let fmt_res = assign_formats_and_scales(graph.clone(), weights_dict, frozen_config, calib_scales)?;
    fmt_res.save_json(&out_dir.join("format_table.json"))?;

    let head_dim = config.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let rope_theta = config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(100000.0);

    let addr_map = pack_rom_artifacts(
        fmt_res.graph.as_ref().unwrap(),
        &fmt_res,
        weights_dict,
        out_dir,
        tile_r,
        tile_c,
        max_ctx,
        rope_theta,
        head_dim,
    )?;

    let microprog = generate_schedule(&addr_map, config, tile_r, tile_c);
    microprog.save(out_dir)?;

    // 4. Emit 128-bit words for weights (weights_128b.hex)
    let w_hex_path = out_dir.join("weights.hex");
    let w128_path = out_dir.join("weights_128b.hex");
    let raw_w_bytes = fs::read(&w_hex_path)?;
    let chunk_size = tile_c;
    let total_bytes = raw_w_bytes.len() / 3;
    let w128_count = total_bytes.div_ceil(chunk_size);

    {
        let mut f128 = BufWriter::with_capacity(4 * 1024 * 1024, File::create(&w128_path)?);
        let mut line_buf = Vec::with_capacity(chunk_size * 2 + 1);
        for word_idx in 0..w128_count {
            line_buf.clear();
            let base = word_idx * chunk_size;
            for offset in (0..chunk_size).rev() {
                let idx = base + offset;
                if idx < total_bytes {
                    line_buf.push(raw_w_bytes[idx * 3]);
                    line_buf.push(raw_w_bytes[idx * 3 + 1]);
                } else {
                    line_buf.extend_from_slice(b"00");
                }
            }
            line_buf.push(b'\n');
            f128.write_all(&line_buf)?;
        }
        f128.flush()?;
    }

    // 5. Emit Layer Bases ROM hex (layer_bases.hex)
    let num_layers = config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;
    let mut layer_bases_lines = Vec::with_capacity(num_layers * 16);

    let proj_by_op = [
        (0usize, "attn_norm"),
        (1, "q_proj"),
        (2, "k_proj"),
        (3, "v_proj"),
        (7, "o_proj"),
        (9, "mlp_norm"),
        (10, "gate_proj"),
        (11, "up_proj"),
        (13, "down_proj"),
    ];
    let proj_map: HashMap<usize, &str> = proj_by_op.into_iter().collect();

    for l_idx in 0..num_layers {
        for op_idx in 0..16 {
            let (w_base, p_base) = if let Some(&proj) = proj_map.get(&op_idx) {
                let node_id = format!("{proj}_{l_idx}");
                let wb = addr_map
                    .pointer(&format!("/weights/{node_id}/base_addr"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                let pb = addr_map
                    .pointer(&format!("/requant_params/{node_id}/base_addr"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                (wb, pb)
            } else {
                (0u32, 0u32)
            };
            layer_bases_lines.push(format!("{w_base:08x}{p_base:08x}"));
        }
    }
    fs::write(
        out_dir.join("layer_bases.hex"),
        format!("{}\n", layer_bases_lines.join("\n")),
    )?;

    // 6. Emit individual table hex files
    let tables_text = fs::read_to_string(out_dir.join("tables.hex"))?;
    let tables_lines: Vec<&str> = tables_text.lines().collect();

    for (tbl_name, short_name) in [
        ("rope_cos", "rope_cos"),
        ("rope_sin", "rope_sin"),
        ("rsqrt_mantissa_lut", "rsqrt"),
        ("exp2_lut", "exp2"),
        ("silu_lut", "silu"),
    ] {
        if tbl_name == "exp2_lut" {
            let exp2_arr = build_exp2_fraction_lut(256, 16);
            let sub_lines: Vec<String> = exp2_arr
                .iter()
                .map(|&v| format!("{:05x}", (v as u32) & 0x3FFFF))
                .collect();
            fs::write(
                out_dir.join(format!("tables_{short_name}.hex")),
                format!("{}\n", sub_lines.join("\n")),
            )?;
        } else if let Some(info) = addr_map.pointer(&format!("/tables/{tbl_name}")) {
            let b = info["base_addr"].as_u64().unwrap() as usize;
            let len = info["length"].as_u64().unwrap() as usize;
            let sub = &tables_lines[b..b + len];
            fs::write(
                out_dir.join(format!("tables_{short_name}.hex")),
                format!("{}\n", sub.join("\n")),
            )?;
        }
    }

    // 7. Emit top_accel.v
    let mut rom_files = BTreeMap::new();
    let posix = |name: &str| out_dir.join(name).to_string_lossy().replace('\\', "/");
    rom_files.insert("weights".into(), posix("weights_128b.hex"));
    rom_files.insert("requant_params".into(), posix("requant_params.hex"));
    rom_files.insert("program".into(), posix("program.hex"));
    rom_files.insert("layer_bases".into(), posix("layer_bases.hex"));
    rom_files.insert("exp2".into(), posix("tables_exp2.hex"));
    rom_files.insert("rsqrt".into(), posix("tables_rsqrt.hex"));
    rom_files.insert("silu".into(), posix("tables_silu.hex"));
    rom_files.insert("rope_cos".into(), posix("tables_rope_cos.hex"));
    rom_files.insert("rope_sin".into(), posix("tables_rope_sin.hex"));

    let mut rom_depths = BTreeMap::new();
    rom_depths.insert("weights".into(), w128_count);
    rom_depths.insert(
        "params".into(),
        addr_map["requant_image"]["total_words"].as_u64().unwrap_or(2048) as usize,
    );
    rom_depths.insert("prog".into(), 256);
    rom_depths.insert("bases".into(), layer_bases_lines.len().max(512));

    let top_verilog = generate_top_accel_verilog(
        config,
        Some(&rom_files),
        "top_accel",
        Some(&rom_depths),
        top_accel_template,
    );
    let top_file = out_dir.join("top_accel.v");
    fs::write(&top_file, top_verilog)?;

    Ok(HardwareBundleResult {
        out_dir: out_dir.to_string_lossy().into_owned(),
        address_map: addr_map,
        microprogram: microprog,
        format_result: fmt_res,
        top_file: top_file.to_string_lossy().into_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_design_f_and_p_generation() {
        let w = vec![12i8, -7, 0, 15, -3, 8, 22, -1];
        let op = LinearOp::new("toy", 4, 2, w.clone(), None, true).unwrap();
        let (manifest, _) = tile_weight_matrix(&w, 2, 4, 2, 4, "toy");
        let rtl_f = generate_full_module(&op, &manifest, None).unwrap();
        assert!(rtl_f.contains("module llmhw_toy"));
        assert!(!rtl_f.contains("w_we"));

        let rtl_p = generate_full_module_p(&op, &manifest);
        assert!(rtl_p.contains("module llmhw_toy_p"));
        assert!(rtl_p.contains("w_we"));
    }
}
