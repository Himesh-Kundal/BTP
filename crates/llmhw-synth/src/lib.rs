//! Yosys Logic Synthesis Runner, Metric Parser, and Synthesis Evaluation Drivers (Phase R6).
//!
//! Strict Layering (Golden Rule 6): No ML framework dependencies.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResult {
    pub top_module: String,
    pub target: String,
    pub flatten: bool,
    pub returncode: i32,
    pub wall_time_seconds: f64,
    pub num_wires: usize,
    pub num_wire_bits: usize,
    pub num_ports: usize,
    pub num_port_bits: usize,
    pub num_cells_total: usize,
    pub num_cells_comb: usize,
    pub num_cells_seq: usize,
    pub num_lut4: usize,
    pub ltp_depth: usize,
    pub cell_counts: BTreeMap<String, usize>,
    pub stdout: String,
    pub stderr: String,
}

pub fn parse_yosys_stat_json(stdout: &str) -> anyhow::Result<Value> {
    let idx = stdout
        .find("\"invocation\": \"stat -json")
        .or_else(|| stdout.find("\"modules\":"))
        .ok_or_else(|| anyhow::anyhow!("Could not find Yosys 'stat -json' output block"))?;
    let start = stdout[..idx]
        .rfind('{')
        .ok_or_else(|| anyhow::anyhow!("Could not find opening brace for 'stat -json'"))?;

    let bytes = stdout.as_bytes();
    let mut brace_depth = 0i32;
    let mut end = None;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if b == b'{' {
            brace_depth += 1;
        } else if b == b'}' {
            brace_depth -= 1;
            if brace_depth == 0 {
                end = Some(i + 1);
                break;
            }
        }
    }
    let end_idx = end.ok_or_else(|| anyhow::anyhow!("Unterminated JSON block in Yosys stdout"))?;
    let val: Value = serde_json::from_str(&stdout[start..end_idx])?;
    Ok(val)
}

pub fn parse_yosys_ltp(stdout: &str) -> usize {
    let re = Regex::new(r"Longest topological path in \S+ \(length=(\d+)\):").unwrap();
    re.captures(stdout)
        .and_then(|c| c[1].parse::<usize>().ok())
        .unwrap_or(0)
}

pub fn classify_cells(cell_counts: &BTreeMap<String, usize>) -> (usize, usize, usize) {
    let mut seq = 0usize;
    let mut lut4 = 0usize;
    let mut comb = 0usize;
    for (ct, &cnt) in cell_counts {
        if ct == "$scopeinfo" {
            continue;
        }
        let up = ct.to_ascii_uppercase();
        if up.contains("DFF") {
            seq += cnt;
        } else if up.contains("LUT4") {
            lut4 += cnt;
            comb += cnt;
        } else {
            comb += cnt;
        }
    }
    (seq, comb, lut4)
}

pub fn run_yosys_synthesis(
    verilog_paths: &[&Path],
    top_module: &str,
    target: &str,
    flatten: bool,
    include_dir: Option<&Path>,
) -> anyhow::Result<SynthesisResult> {
    if target != "generic" && target != "ice40" {
        anyhow::bail!("Unsupported synthesis target: '{target}'. Must be 'generic' or 'ice40'.");
    }

    let mut script_lines = Vec::new();
    let inc_flag = include_dir
        .and_then(|d| fs::canonicalize(d).ok())
        .map(|d| format!("-I{} ", d.display()))
        .unwrap_or_default();
    for p in verilog_paths {
        let abs_p = fs::canonicalize(p)?;
        script_lines.push(format!("read_verilog {inc_flag}{}", abs_p.display()));
    }
    script_lines.push(format!("hierarchy -check -top {top_module}"));
    let flatten_flag = if flatten { " -flatten" } else { "" };
    if target == "generic" {
        script_lines.push(format!("synth -top {top_module}{flatten_flag}"));
    } else {
        script_lines.push(format!("synth_ice40 -top {top_module}{flatten_flag}"));
    }
    script_lines.push("stat -json".into());
    script_lines.push("ltp".into());

    let temp_dir = tempfile::tempdir()?;
    let script_file = temp_dir.path().join("synth.ys");
    fs::write(&script_file, format!("{}\n", script_lines.join("\n")))?;

    let t0 = Instant::now();
    let out = Command::new("yosys")
        .arg("-s")
        .arg(&script_file)
        .current_dir(temp_dir.path())
        .output()?;
    let elapsed = t0.elapsed().as_secs_f64();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let rc = out.status.code().unwrap_or(1);

    if rc != 0 {
        return Ok(SynthesisResult {
            top_module: top_module.to_string(),
            target: target.to_string(),
            flatten,
            returncode: rc,
            wall_time_seconds: elapsed,
            num_wires: 0,
            num_wire_bits: 0,
            num_ports: 0,
            num_port_bits: 0,
            num_cells_total: 0,
            num_cells_comb: 0,
            num_cells_seq: 0,
            num_lut4: 0,
            ltp_depth: 0,
            cell_counts: BTreeMap::new(),
            stdout,
            stderr,
        });
    }

    let stat_data = parse_yosys_stat_json(&stdout)?;
    let design = &stat_data["design"];
    let mut cell_counts = BTreeMap::new();
    if let Some(obj) = design.get("num_cells_by_type").and_then(|v| v.as_object()) {
        for (k, v) in obj {
            cell_counts.insert(k.clone(), v.as_u64().unwrap_or(0) as usize);
        }
    }
    let ltp_depth = parse_yosys_ltp(&stdout);
    let (seq, comb, lut4) = classify_cells(&cell_counts);
    let u = |k: &str| design.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;

    Ok(SynthesisResult {
        top_module: top_module.to_string(),
        target: target.to_string(),
        flatten,
        returncode: 0,
        wall_time_seconds: elapsed,
        num_wires: u("num_wires"),
        num_wire_bits: u("num_wire_bits"),
        num_ports: u("num_ports"),
        num_port_bits: u("num_port_bits"),
        num_cells_total: u("num_cells"),
        num_cells_comb: comb,
        num_cells_seq: seq,
        num_lut4: lut4,
        ltp_depth,
        cell_counts,
        stdout,
        stderr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yosys_synth_residual_add() {
        let rtl_lib = Path::new("../../rtl/lib");
        let unit_file = rtl_lib.join("residual_add.v");
        if !unit_file.is_file() {
            return;
        }
        let res =
            run_yosys_synthesis(&[&unit_file], "residual_add", "generic", true, Some(rtl_lib))
                .unwrap();
        assert_eq!(res.returncode, 0);
        assert!(res.num_cells_total > 0);
        assert!(res.ltp_depth > 0);
    }
}
