//! Verification Suite: Independent INT64 Oracle, Trace Diff, Weight Audit v1/v2, Structural Audit, and Multi-Tool Linter (Phase R5).
//!
//! Strict Layering (Golden Rule 6): No ML framework dependencies.

use llmhw_backend::generate_top_accel_verilog;
use llmhw_ir::sha256_hex;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ==============================================================================
// 1. Part I Independent NumPy INT64 Reference Oracle (verification/reference.py)
// ==============================================================================

/// Compute `Y = X @ W.T + b` in exact 64-bit signed integer arithmetic.
/// `x`: `[batch, k]`, `w`: `[n, k]`, `bias`: optional `[n]`. Returns `[batch * n]`.
pub fn linear_reference_int64(
    x: &[i8],
    batch: usize,
    n: usize,
    k: usize,
    w: &[i8],
    bias: Option<&[i32]>,
) -> Vec<i64> {
    assert_eq!(x.len(), batch * k);
    assert_eq!(w.len(), n * k);
    let mut out = vec![0i64; batch * n];
    for b in 0..batch {
        let x_row = &x[b * k..(b + 1) * k];
        for r in 0..n {
            let w_row = &w[r * k..(r + 1) * k];
            let mut acc = bias.map(|bs| bs[r] as i64).unwrap_or(0i64);
            for (&xk, &wrk) in x_row.iter().zip(w_row.iter()) {
                acc += (xk as i64) * (wrk as i64);
            }
            out[b * n + r] = acc;
        }
    }
    out
}

// ==============================================================================
// 2. Verilog Comment Stripping & Weight Audit v2 (verification/weight_audit_v2.py)
// ==============================================================================

pub fn strip_verilog_comments(code: &str) -> String {
    let re_block = Regex::new(r"(?s)/\*.*?\*/").unwrap();
    let no_block = re_block.replace_all(code, "");
    let re_line = Regex::new(r"(?m)//.*$").unwrap();
    re_line.replace_all(&no_block, "").into_owned()
}

pub fn audit_rom_module_source(rtl_content: &str, module_name: Option<&str>) -> anyhow::Result<Value> {
    let clean = strip_verilog_comments(rtl_content);
    let mut violations = Vec::new();

    let re_in = Regex::new(r"input\s+(?:wire\s+)?(?:\[[^\]]+\]\s+)?([a-zA-Z0-9_]+)").unwrap();
    let mut input_ports = Vec::new();
    for cap in re_in.captures_iter(&clean) {
        let port = cap[1].to_string();
        let lower = port.to_ascii_lowercase();
        if ["we", "wr", "write", "din", "wdata"]
            .iter()
            .any(|term| lower.contains(term))
        {
            violations.push(format!("Forbidden write-related input port found: '{port}'"));
        } else if !["clk", "addr", "rst_n", "rst", "enable", "en", "ren"].contains(&lower.as_str()) {
            violations.push(format!("Unexpected input port in ROM module: '{port}'"));
        }
        input_ports.push(port);
    }

    let re_write = Regex::new(r"([a-zA-Z0-9_]+)\s*\[[^\]]+\]\s*(?:<=|=)\s*[^;]+;").unwrap();
    let re_always = Regex::new(r"(?s)always\s*@[^b]+begin(.*?)end").unwrap();
    for cap in re_write.captures_iter(&clean) {
        let target = &cap[1];
        for blk in re_always.captures_iter(&clean) {
            let body = &blk[1];
            let pat = Regex::new(&format!(r"\b{}\s*\[", regex::escape(target))).unwrap();
            if pat.is_match(body) {
                violations.push(format!(
                    "Runtime write assignment to memory array '{target}' in always block"
                ));
            }
        }
    }

    let re_out =
        Regex::new(r"output\s+(?:(?:reg|wire)\s+)?(?:signed\s+)?(?:\[[^\]]+\]\s+)?([a-zA-Z0-9_]+)")
            .unwrap();
    let mut output_ports = Vec::new();
    for cap in re_out.captures_iter(&clean) {
        let out_port = cap[1].to_string();
        let lower = out_port.to_ascii_lowercase();
        if !["data", "dout", "val"].iter().any(|t| lower.contains(t)) {
            violations.push(format!("Unexpected output port in ROM module: '{out_port}'"));
        }
        output_ports.push(out_port);
    }

    if !violations.is_empty() {
        anyhow::bail!(
            "ROM Audit failed for {}:\n  - {}",
            module_name.unwrap_or("module"),
            violations.join("\n  - ")
        );
    }

    Ok(json!({
        "status": "PASS",
        "input_ports": input_ports,
        "output_ports": output_ports,
        "violations": violations,
    }))
}

pub fn audit_rom_hex_images(address_map_path: &Path) -> anyhow::Result<Value> {
    if !address_map_path.is_file() {
        anyhow::bail!("Address map not found: {}", address_map_path.display());
    }
    let meta: Value = serde_json::from_str(&fs::read_to_string(address_map_path)?)?;
    let base_dir = address_map_path.parent().unwrap_or_else(|| Path::new("."));
    let mut checked = Vec::new();

    for (key_a, key_b) in [
        ("weights_image", "weights"),
        ("requant_image", "requant_params"),
        ("tables_image", "tables"),
    ] {
        let Some(rom_info) = meta.get(key_a).or_else(|| meta.get(key_b)) else {
            continue;
        };
        let Some(file_name) = rom_info.get("file").and_then(|v| v.as_str()) else {
            continue;
        };
        let hex_path = base_dir.join(file_name);
        if !hex_path.is_file() {
            anyhow::bail!("Required ROM hex file not found: {}", hex_path.display());
        }

        let hex_bytes = fs::read(&hex_path)?;
        let actual_sha256 = sha256_hex(&hex_bytes);
        if let Some(expected_sha) = rom_info.get("sha256").and_then(|v| v.as_str()) {
            if actual_sha256 != expected_sha {
                anyhow::bail!(
                    "ROM integrity violation for '{key_a}' ({file_name}): expected sha256={expected_sha}, actual sha256={actual_sha256}"
                );
            }
        }

        let line_count = String::from_utf8_lossy(&hex_bytes)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count();
        if let Some(exp_words) = rom_info.get("total_words").and_then(|v| v.as_u64()) {
            if line_count != (exp_words as usize) {
                anyhow::bail!(
                    "ROM size mismatch for '{key_a}' ({file_name}): expected {exp_words} words, found {line_count} lines"
                );
            }
        }

        checked.push(json!({
            "rom": key_a,
            "file": file_name,
            "words": line_count,
            "sha256": actual_sha256,
        }));
    }

    Ok(json!({
        "status": "PASS",
        "checked_roms": checked,
    }))
}

// ==============================================================================
// 3. Structural, Determinism & Lint Audits (verification/structural_audit.py)
// ==============================================================================

const AUTHORIZED_DYNAMIC_BUFFERS: &[&str] = &[
    "buf_res",
    "buf_act",
    "buf_q",
    "buf_k",
    "buf_v",
    "buf_attn_out",
    "buf_gate",
    "buf_up",
    "buf_swiglu",
    "buf_logits",
];

const FORBIDDEN_ORACLE_PATTERNS: &[&str] = &[
    r"(?i)\bexp(?:ected)?_tokens?\b",
    r"(?i)\bgolden_traces?\b",
    r"(?i)\boracle_tokens?\b",
    r"(?i)\bexp_mem\b",
    r"(?i)\bgolden_mem\b",
    r"(?i)\btb_exp\b",
];

pub fn audit_no_expected_values_in_rtl(rtl_content: &str, module_name: &str) -> anyhow::Result<Value> {
    let clean = strip_verilog_comments(rtl_content);
    let mut violations = Vec::new();

    for pat in FORBIDDEN_ORACLE_PATTERNS {
        let re = Regex::new(pat).unwrap();
        if let Some(m) = re.find(&clean) {
            violations.push(format!(
                "Forbidden oracle signal or memory pattern detected: '{}'",
                m.as_str()
            ));
        }
    }

    let re_tok_arr =
        Regex::new(r"(?i)reg\s+(?:\[[^\]]+\]\s+)?(tokens?|golden|expected)\s*\[\s*\d+\s*:\s*\d+\s*\]")
            .unwrap();
    for cap in re_tok_arr.captures_iter(&clean) {
        violations.push(format!(
            "Forbidden hardcoded token array detected in RTL: '{}'",
            &cap[1]
        ));
    }

    let re_readmem = Regex::new(r#"\$readmem[hb]\s*\(\s*"([^"]+)""#).unwrap();
    for cap in re_readmem.captures_iter(&clean) {
        let f = &cap[1];
        let fl = f.to_ascii_lowercase();
        if ["exp", "golden", "oracle", "expected", "trace"]
            .iter()
            .any(|t| fl.contains(t))
        {
            violations.push(format!("Forbidden readmem of oracle file in RTL: '{f}'"));
        }
    }

    if !violations.is_empty() {
        anyhow::bail!(
            "No-Expected-Values Audit failed for {module_name}:\n  - {}",
            violations.join("\n  - ")
        );
    }

    Ok(json!({
        "status": "PASS",
        "module": module_name,
        "violations": [],
    }))
}

pub fn audit_top_accel_memory_structure(rtl_content: &str) -> anyhow::Result<Value> {
    let clean = strip_verilog_comments(rtl_content);
    let mut violations = Vec::new();
    let authorized: BTreeSet<&str> = AUTHORIZED_DYNAMIC_BUFFERS.iter().copied().collect();

    let re_mem =
        Regex::new(r"reg\s+(?:signed\s+)?(?:\[[^\]]+\]\s+)?([a-zA-Z0-9_]+)\s*\[\s*0\s*:\s*([^\]]+)\s*\]\s*;")
            .unwrap();
    let mut found_buffers = BTreeSet::new();
    for cap in re_mem.captures_iter(&clean) {
        let buf_name = cap[1].to_string();
        if !authorized.contains(buf_name.as_str()) {
            violations.push(format!("Unauthorized writable memory array declared: '{buf_name}'"));
        }
        found_buffers.insert(buf_name);
    }

    let re_rom = Regex::new(r"(?s)rom_sync\s*#\s*\((.*?)\)\s*([a-zA-Z0-9_]+)\s*\((.*?)\);").unwrap();
    let re_port = Regex::new(r"\.([a-zA-Z0-9_]+)\s*\(").unwrap();
    let mut instantiated_roms = Vec::new();

    for cap in re_rom.captures_iter(&clean) {
        let inst_name = cap[2].to_string();
        let ports = &cap[3];
        for pcap in re_port.captures_iter(ports) {
            let p = &pcap[1];
            let pl = p.to_ascii_lowercase();
            if ["we", "wr", "write", "din", "wdata"]
                .iter()
                .any(|t| pl.contains(t))
            {
                violations.push(format!(
                    "Forbidden write port '{p}' connected to ROM instance '{inst_name}'"
                ));
            }
        }
        instantiated_roms.push(inst_name);
    }

    let required_roms: BTreeSet<&str> = [
        "prog_rom_inst",
        "layer_bases_rom_inst",
        "rom_weights_inst",
        "rom_params_inst",
        "rom_rope_cos_inst",
        "rom_rope_sin_inst",
        "rom_rsqrt_inst",
        "rom_exp2_inst",
        "rom_silu_inst",
    ]
    .into_iter()
    .collect();

    let inst_set: BTreeSet<&str> = instantiated_roms.iter().map(|s| s.as_str()).collect();
    let missing: Vec<&str> = required_roms.difference(&inst_set).copied().collect();
    if !missing.is_empty() {
        violations.push(format!(
            "Missing required ROM instances in top_accel: {missing:?}"
        ));
    }

    if !violations.is_empty() {
        anyhow::bail!(
            "Top Accelerator Memory Audit failed:\n  - {}",
            violations.join("\n  - ")
        );
    }

    Ok(json!({
        "status": "PASS",
        "found_buffers": found_buffers,
        "instantiated_roms": instantiated_roms,
        "violations": [],
    }))
}

pub fn audit_generator_determinism(
    config: &Value,
    rom_files: &BTreeMap<String, String>,
    template_content: &str,
) -> anyhow::Result<Value> {
    let code1 = generate_top_accel_verilog(config, Some(rom_files), "top_accel", None, template_content);
    let code2 = generate_top_accel_verilog(config, Some(rom_files), "top_accel", None, template_content);
    let sha1 = sha256_hex(code1.as_bytes());
    let sha2 = sha256_hex(code2.as_bytes());
    if sha1 != sha2 || code1 != code2 {
        anyhow::bail!("Determinism violation: sha1={sha1}, sha2={sha2}");
    }
    Ok(json!({
        "status": "PASS",
        "sha256": sha1,
        "byte_length": code1.len(),
    }))
}

pub fn run_multi_tool_lint(rtl_lib_dir: &Path, top_module: &str) -> anyhow::Result<Value> {
    let file_names = [
        "top_accel.v",
        "rom_sync.v",
        "sequencer.v",
        "embed_gather.v",
        "rmsnorm.v",
        "rope.v",
        "attention.v",
        "residual_add.v",
        "swiglu.v",
        "argmax.v",
        "linear_engine.v",
        "kv_cache.v",
        "softmax.v",
    ];
    let files: Vec<String> = file_names
        .iter()
        .map(|f| rtl_lib_dir.join(f).to_string_lossy().into_owned())
        .collect();
    let inc = rtl_lib_dir.to_string_lossy().into_owned();

    // 1. Icarus Verilog Lint
    let mut cmd_iv = Command::new("iverilog");
    cmd_iv.args(["-Wall", "-g2005", &format!("-I{inc}"), &format!("-s{top_module}")]);
    for f in &files {
        cmd_iv.arg(f);
    }
    let res_iv = cmd_iv.output()?;
    if !res_iv.status.success() {
        anyhow::bail!(
            "Icarus Verilog lint failed:\n{}",
            String::from_utf8_lossy(&res_iv.stderr)
        );
    }

    // 2. Verilator Lint
    let mut cmd_ver = Command::new("verilator");
    cmd_ver.args([
        "--lint-only",
        "-Wall",
        "-Wno-DECLFILENAME",
        "-Wno-UNUSEDSIGNAL",
        "-Wno-UNDRIVEN",
        "-Wno-SYNCASYNCNET",
        "-Wno-WIDTHEXPAND",
        "-Wno-WIDTHTRUNC",
        "-Wno-BLKSEQ",
        &format!("-I{inc}"),
        "--top-module",
        top_module,
    ]);
    for f in &files {
        cmd_ver.arg(f);
    }
    let res_ver = cmd_ver.output()?;
    if !res_ver.status.success() {
        anyhow::bail!(
            "Verilator lint failed:\n{}",
            String::from_utf8_lossy(&res_ver.stderr)
        );
    }

    // 3. Yosys Hierarchy & Proc Check
    let yosys_script = format!(
        "read_verilog -I{inc} {}; hierarchy -check -top {top_module}; proc; check -assert",
        files.join(" ")
    );
    let res_yo = Command::new("yosys")
        .args(["-q", "-p", &yosys_script])
        .output()?;
    if !res_yo.status.success() {
        anyhow::bail!(
            "Yosys check failed:\n{}",
            String::from_utf8_lossy(&res_yo.stderr)
        );
    }

    Ok(json!({
        "status": "PASS",
        "iverilog": {"returncode": 0},
        "verilator": {"returncode": 0},
        "yosys": {"returncode": 0},
    }))
}

// ==============================================================================
// 4. First-Divergence Hierarchical Trace Diff Engine (verification/trace_diff.py)
// ==============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DivergencePoint {
    pub seq_id: String,
    pub token_idx: String,
    pub layer: String,
    pub op_name: String,
    pub tensor_name: String,
    pub element_idx: usize,
    pub expected_hex: String,
    pub actual_hex: String,
}

impl fmt::Display for DivergencePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DIVERGENCE at seq={}, tok={}, layer={}, op={}, tensor={}, elem[{}]: expected={}, actual={}",
            self.seq_id,
            self.token_idx,
            self.layer,
            self.op_name,
            self.tensor_name,
            self.element_idx,
            self.expected_hex,
            self.actual_hex
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffResult {
    pub is_identical: bool,
    pub total_files_checked: usize,
    pub total_elements_checked: usize,
    pub first_divergence: Option<DivergencePoint>,
    pub missing_files: Vec<String>,
}

fn collect_hex_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let p = entry?.path();
        if p.is_dir() {
            collect_hex_files_recursive(&p, out)?;
        } else if p.extension().and_then(|e| e.to_str()) == Some("hex") {
            out.push(p);
        }
    }
    Ok(())
}

pub fn compare_trace_directories(golden_dir: &Path, actual_dir: &Path) -> anyhow::Result<DiffResult> {
    let mut golden_files = Vec::new();
    collect_hex_files_recursive(golden_dir, &mut golden_files)?;
    golden_files.sort();

    let mut total_files = 0usize;
    let mut total_elements = 0usize;
    let mut missing = Vec::new();

    for g_path in &golden_files {
        let rel = g_path.strip_prefix(golden_dir)?;
        let a_path = actual_dir.join(rel);
        if !a_path.is_file() {
            missing.push(rel.to_string_lossy().into_owned());
            continue;
        }
        total_files += 1;
        let g_text = fs::read_to_string(g_path)?;
        let a_text = fs::read_to_string(&a_path)?;
        let g_lines: Vec<&str> = g_text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
        let a_lines: Vec<&str> = a_text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();

        let min_len = g_lines.len().min(a_lines.len());
        let mut mismatch_found: Option<(usize, String, String)> = None;

        for idx in 0..min_len {
            if !g_lines[idx].eq_ignore_ascii_case(a_lines[idx]) {
                mismatch_found = Some((idx, g_lines[idx].to_string(), a_lines[idx].to_string()));
                break;
            }
        }
        if mismatch_found.is_none() && g_lines.len() != a_lines.len() {
            mismatch_found = Some((
                min_len,
                format!("len={}", g_lines.len()),
                format!("len={}", a_lines.len()),
            ));
        }

        total_elements += g_lines.len();

        if let Some((elem_idx, exp_val, act_val)) = mismatch_found {
            let comps: Vec<String> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            let seq_id = comps.first().cloned().unwrap_or_else(|| "?".into());
            let token_idx = comps.get(1).cloned().unwrap_or_else(|| "?".into());
            let layer = comps.get(2).cloned().unwrap_or_else(|| "?".into());
            let stem = g_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("?");
            let (op_name, tensor_name) = if let Some((op, ten)) = stem.split_once('_') {
                (op.to_string(), ten.to_string())
            } else {
                (stem.to_string(), stem.to_string())
            };

            return Ok(DiffResult {
                is_identical: false,
                total_files_checked: total_files,
                total_elements_checked: total_elements,
                first_divergence: Some(DivergencePoint {
                    seq_id,
                    token_idx,
                    layer,
                    op_name,
                    tensor_name,
                    element_idx: elem_idx,
                    expected_hex: exp_val,
                    actual_hex: act_val,
                }),
                missing_files: missing,
            });
        }
    }

    Ok(DiffResult {
        is_identical: missing.is_empty(),
        total_files_checked: total_files,
        total_elements_checked: total_elements,
        first_divergence: None,
        missing_files: missing,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rom_and_structural_audits_and_negative_controls() {
        let rom_path = Path::new("../../rtl/lib/rom_sync.v");
        let top_path = Path::new("../../rtl/lib/top_accel.v");
        if !rom_path.is_file() || !top_path.is_file() {
            return;
        }
        let rom_src = fs::read_to_string(rom_path).unwrap();
        let top_src = fs::read_to_string(top_path).unwrap();

        assert!(audit_rom_module_source(&rom_src, Some("rom_sync")).is_ok());
        assert!(audit_no_expected_values_in_rtl(&top_src, "top_accel").is_ok());
        assert!(audit_top_accel_memory_structure(&top_src).is_ok());

        // Negative control 1: injected write port in ROM
        let bad_rom = rom_src.replace("input  wire                  clk,", "input wire clk, input wire we,");
        assert!(audit_rom_module_source(&bad_rom, Some("rom_sync")).is_err());

        // Negative control 2: injected oracle token array in top_accel
        let bad_top = format!("{top_src}\nreg [15:0] expected_tokens [0:15];\n");
        assert!(audit_no_expected_values_in_rtl(&bad_top, "top_accel").is_err());

        // Negative control 3: unauthorized writable memory array in top_accel
        let bad_mem = format!("{top_src}\nreg [7:0] secret_weight_ram [0:1023];\n");
        assert!(audit_top_accel_memory_structure(&bad_mem).is_err());
    }
}
