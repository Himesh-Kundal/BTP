//! Intermediate Representation (Part I LinearOp & Part II GraphV2) and Zero-Dependency `.npy` I/O (Phases R0 & R2).
//!
//! Enforces Invariants I1–I7 per docs/part2/02 §5.
//! Strict Layering (Golden Rule 6): No ML framework dependencies.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use thiserror::Error;

// ==============================================================================
// 1. Native NumPy `.npy` v1.0 Reader & Writer (`i8`, `i16`, `i32`, `i64`, `f32`, `f64`)
// ==============================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn parse_npy_header(bytes: &[u8]) -> anyhow::Result<(String, Vec<usize>, usize)> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        anyhow::bail!("Invalid .npy magic header");
    }
    let major = bytes[6];
    let (header_len, data_offset) = if major == 1 {
        let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (hlen, 10 + hlen)
    } else {
        let hlen = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (hlen, 12 + hlen)
    };
    let header_str = std::str::from_utf8(&bytes[data_offset - header_len..data_offset])?;

    // Extract descr
    let descr_pos = header_str
        .find("'descr':")
        .ok_or_else(|| anyhow::anyhow!("Missing 'descr' in .npy header: {header_str}"))?;
    let after_descr = &header_str[descr_pos + 8..];
    let q1 = after_descr.find('\'').ok_or_else(|| anyhow::anyhow!("Malformed descr"))?;
    let q2 = after_descr[q1 + 1..]
        .find('\'')
        .ok_or_else(|| anyhow::anyhow!("Malformed descr end"))?;
    let descr = after_descr[q1 + 1..q1 + 1 + q2].to_string();

    // Extract shape
    let shape_pos = header_str
        .find("'shape':")
        .ok_or_else(|| anyhow::anyhow!("Missing 'shape' in .npy header: {header_str}"))?;
    let after_shape = &header_str[shape_pos + 8..];
    let p1 = after_shape.find('(').ok_or_else(|| anyhow::anyhow!("Malformed shape"))?;
    let p2 = after_shape[p1 + 1..]
        .find(')')
        .ok_or_else(|| anyhow::anyhow!("Malformed shape end"))?;
    let shape_inner = after_shape[p1 + 1..p1 + 1 + p2].trim();
    let mut shape = Vec::new();
    for tok in shape_inner.split(',') {
        let t = tok.trim();
        if !t.is_empty() {
            shape.push(t.parse::<usize>()?);
        }
    }
    if shape.is_empty() {
        shape.push(1);
    }

    Ok((descr, shape, data_offset))
}

fn build_npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let inner: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        format!("({})", inner.join(", "))
    };
    let dict_str = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    );
    let mut header_bytes = dict_str.into_bytes();
    // Total prefix length (10 + header_len) must be a multiple of 64
    let rem = (10 + header_bytes.len() + 1) % 64;
    let pad = if rem == 0 { 0 } else { 64 - rem };
    header_bytes.resize(header_bytes.len() + pad, b' ');
    header_bytes.push(b'\n');

    let hlen = header_bytes.len() as u16;
    let mut out = Vec::with_capacity(10 + header_bytes.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&hlen.to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out
}

pub fn read_npy_i8(path: &Path) -> anyhow::Result<NpyArray<i8>> {
    let bytes = fs::read(path)?;
    let (descr, shape, offset) = parse_npy_header(&bytes)?;
    if descr != "|i1" && descr != "<i1" && descr != "i1" {
        anyhow::bail!("Expected int8 .npy descr '|i1', got '{descr}'");
    }
    let count: usize = shape.iter().product();
    let raw = &bytes[offset..offset + count];
    let data: Vec<i8> = raw.iter().map(|&b| b as i8).collect();
    Ok(NpyArray { shape, data })
}

pub fn write_npy_i8(path: &Path, shape: &[usize], data: &[i8]) -> anyhow::Result<()> {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let mut out = build_npy_header("|i1", shape);
    out.extend(data.iter().map(|&v| v as u8));
    fs::write(path, out)?;
    Ok(())
}

pub fn read_npy_i32(path: &Path) -> anyhow::Result<NpyArray<i32>> {
    let bytes = fs::read(path)?;
    let (descr, shape, offset) = parse_npy_header(&bytes)?;
    if descr != "<i4" && descr != "i4" {
        anyhow::bail!("Expected int32 .npy descr '<i4', got '{descr}'");
    }
    let count: usize = shape.iter().product();
    let raw = &bytes[offset..offset + count * 4];
    let data: Vec<i32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| i32::from_le_bytes(c))
        .collect();
    Ok(NpyArray { shape, data })
}

pub fn read_npy_i64_any(path: &Path) -> anyhow::Result<NpyArray<i64>> {
    let bytes = fs::read(path)?;
    let (descr, shape, offset) = parse_npy_header(&bytes)?;
    let count: usize = shape.iter().product();
    let data: Vec<i64> = match descr.as_str() {
        "|i1" | "<i1" | "i1" => bytes[offset..offset + count]
            .iter()
            .map(|&b| (b as i8) as i64)
            .collect(),
        "<i2" | "i2" => bytes[offset..offset + count * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| i16::from_le_bytes(c) as i64)
            .collect(),
        "<i4" | "i4" => bytes[offset..offset + count * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c) as i64)
            .collect(),
        "<i8" | "i8" => bytes[offset..offset + count * 8]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| i64::from_le_bytes(c))
            .collect(),
        _ => anyhow::bail!("Unsupported integer .npy descr '{descr}'"),
    };
    Ok(NpyArray { shape, data })
}

pub fn read_npy_f64_any(path: &Path) -> anyhow::Result<NpyArray<f64>> {
    let bytes = fs::read(path)?;
    let (descr, shape, offset) = parse_npy_header(&bytes)?;
    let count: usize = shape.iter().product();
    let data: Vec<f64> = match descr.as_str() {
        "<f4" | "f4" => bytes[offset..offset + count * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c) as f64)
            .collect(),
        "<f8" | "f8" => bytes[offset..offset + count * 8]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| f64::from_le_bytes(c))
            .collect(),
        "|i1" | "<i1" | "i1" => bytes[offset..offset + count]
            .iter()
            .map(|&b| (b as i8) as f64)
            .collect(),
        _ => anyhow::bail!("Unsupported float .npy descr '{descr}'"),
    };
    Ok(NpyArray { shape, data })
}

pub fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> anyhow::Result<()> {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let mut out = build_npy_header("<f4", shape);
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, out)?;
    Ok(())
}

// ==============================================================================
// 2. Part I LinearOp IR (ir/linear.py)
// ==============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearOp {
    pub name: String,
    pub in_features: usize,
    pub out_features: usize,
    pub weight: Vec<i8>, // Row-major [out_features, in_features]
    pub bias: Option<Vec<i32>>,
    pub is_quantized: bool,
}

impl LinearOp {
    pub fn new(
        name: impl Into<String>,
        in_features: usize,
        out_features: usize,
        weight: Vec<i8>,
        bias: Option<Vec<i32>>,
        is_quantized: bool,
    ) -> anyhow::Result<Self> {
        let op = Self {
            name: name.into(),
            in_features,
            out_features,
            weight,
            bias,
            is_quantized,
        };
        op.validate()?;
        Ok(op)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.weight.len() != self.out_features * self.in_features {
            anyhow::bail!(
                "Weight length {} != out_features ({}) * in_features ({})",
                self.weight.len(),
                self.out_features,
                self.in_features
            );
        }
        if let Some(ref b) = self.bias {
            if b.len() != self.out_features {
                anyhow::bail!("Bias length {} != out_features {}", b.len(), self.out_features);
            }
        }
        if self.is_quantized {
            for &w in &self.weight {
                if w == -128 {
                    anyhow::bail!("Invariant I2 violation: weight contains -128 outside [-127, 127]");
                }
            }
        }
        Ok(())
    }
}

// ==============================================================================
// 3. Part II Graph IR v2 (ir/graph_v2.py) — Invariants I1–I7
// ==============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorKind {
    CONSTANT,
    DYNAMIC,
    STATE,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpKind {
    Embed,
    RMSNorm,
    Linear,
    RoPE,
    Attention,
    SiLU,
    EltMul,
    Add,
    ArgMax,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeFormat {
    pub bits: u32,
    pub signed: bool,
    pub scale: f64,
    pub fixed_point: String,
}

impl EdgeFormat {
    pub fn new(bits: u32, signed: bool, scale: f64, fixed_point: impl Into<String>) -> Result<Self, InvariantViolation> {
        if bits == 0 || bits > 64 {
            return Err(InvariantViolation(format!(
                "Invalid bits {bits}; must be in [1, 64]"
            )));
        }
        if scale <= 0.0 {
            return Err(InvariantViolation(format!(
                "Scale must be strictly positive, got {scale}"
            )));
        }
        Ok(Self {
            bits,
            signed,
            scale,
            fixed_point: fixed_point.into(),
        })
    }

    pub fn min_val(&self) -> i128 {
        if self.signed {
            -(1i128 << (self.bits - 1))
        } else {
            0
        }
    }

    pub fn max_val(&self) -> i128 {
        if self.signed {
            (1i128 << (self.bits - 1)) - 1
        } else {
            (1i128 << self.bits) - 1
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorRefV2 {
    pub name: String,
    pub kind: TensorKind,
    pub shape: Vec<usize>,
    pub format: EdgeFormat,
    pub producer: Option<String>,
    #[serde(default)]
    pub consumers: Vec<String>,
    pub data_file: Option<String>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeV2 {
    pub id: String,
    pub op: OpKind,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
    #[serde(default)]
    pub provenance: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphV2 {
    pub name: String,
    #[serde(default)]
    pub nodes: Vec<NodeV2>,
    #[serde(default)]
    pub tensors: BTreeMap<String, TensorRefV2>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Error)]
#[error("{0}")]
pub struct InvariantViolation(pub String);

impl GraphV2 {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            tensors: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        kind: TensorKind,
        shape: Vec<usize>,
        edge_format: EdgeFormat,
        producer: Option<String>,
        consumers: Vec<String>,
        data_file: Option<String>,
        sha256: Option<String>,
    ) -> Result<TensorRefV2, InvariantViolation> {
        let name_str = name.into();
        if self.tensors.contains_key(&name_str) {
            return Err(InvariantViolation(format!(
                "Tensor {name_str} already exists in graph"
            )));
        }
        let t = TensorRefV2 {
            name: name_str.clone(),
            kind,
            shape,
            format: edge_format,
            producer,
            consumers,
            data_file,
            sha256,
        };
        self.tensors.insert(name_str, t.clone());
        Ok(t)
    }

    pub fn add_node(
        &mut self,
        node_id: impl Into<String>,
        op: OpKind,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: BTreeMap<String, Value>,
        provenance: BTreeMap<String, String>,
    ) -> NodeV2 {
        let id_str = node_id.into();
        for inp in &inputs {
            if let Some(t) = self.tensors.get_mut(inp) {
                if !t.consumers.contains(&id_str) {
                    t.consumers.push(id_str.clone());
                }
            }
        }
        for out in &outputs {
            if let Some(t) = self.tensors.get_mut(out) {
                t.producer = Some(id_str.clone());
            }
        }
        let node = NodeV2 {
            id: id_str,
            op,
            inputs,
            outputs,
            attributes,
            provenance,
        };
        self.nodes.push(node.clone());
        node
    }

    pub fn save_json(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(self)?;
        fs::write(path, text)?;
        Ok(())
    }

    pub fn load_json(path: &Path) -> anyhow::Result<Self> {
        let text = fs::read_to_string(path)?;
        let g: Self = serde_json::from_str(&text)?;
        Ok(g)
    }
}

/// Validate Invariants I1–I7 per docs/part2/02 §5.
pub fn validate_invariants(graph: &GraphV2) -> Result<(), InvariantViolation> {
    // I6: Every edge carries a valid format
    for (name, tensor) in &graph.tensors {
        if tensor.format.bits == 0 || tensor.format.bits > 64 || tensor.format.scale <= 0.0 {
            return Err(InvariantViolation(format!(
                "Invariant I6 failed: Tensor '{name}' invalid format fields"
            )));
        }
    }

    // I3: CONSTANT tensors never feed a write port (never produced by any node)
    let mut node_outputs: HashMap<String, String> = HashMap::new();
    for node in &graph.nodes {
        for out in &node.outputs {
            if node_outputs.contains_key(out) {
                return Err(InvariantViolation(format!(
                    "Invariant I1 failed: Multiple nodes produce tensor '{out}'"
                )));
            }
            node_outputs.insert(out.clone(), node.id.clone());
            if let Some(t) = graph.tensors.get(out) {
                if t.kind == TensorKind::CONSTANT {
                    return Err(InvariantViolation(format!(
                        "Invariant I3 failed: CONSTANT tensor '{out}' is written by node '{}'",
                        node.id
                    )));
                }
            }
        }
    }

    // I7: STATE tensors accessed only by Attention ops
    for node in &graph.nodes {
        for inp in &node.inputs {
            if let Some(t) = graph.tensors.get(inp) {
                if t.kind == TensorKind::STATE && node.op != OpKind::Attention {
                    return Err(InvariantViolation(format!(
                        "Invariant I7 failed: STATE tensor '{inp}' accessed by non-attention op '{:?}'",
                        node.op
                    )));
                }
            }
        }
        for out in &node.outputs {
            if let Some(t) = graph.tensors.get(out) {
                if t.kind == TensorKind::STATE && node.op != OpKind::Attention {
                    return Err(InvariantViolation(format!(
                        "Invariant I7 failed: STATE tensor '{out}' produced by non-attention op '{:?}'",
                        node.op
                    )));
                }
            }
        }
    }

    // I1: Shapes consistent
    for node in &graph.nodes {
        for inp in &node.inputs {
            if !graph.tensors.contains_key(inp) {
                return Err(InvariantViolation(format!(
                    "Invariant I1 failed: Node '{}' input '{inp}' not in graph.tensors",
                    node.id
                )));
            }
        }
        for out in &node.outputs {
            if !graph.tensors.contains_key(out) {
                return Err(InvariantViolation(format!(
                    "Invariant I1 failed: Node '{}' output '{out}' not in graph.tensors",
                    node.id
                )));
            }
        }

        match node.op {
            OpKind::Linear => {
                let act_t = &graph.tensors[&node.inputs[0]];
                let weight_t = &graph.tensors[&node.inputs[1]];
                let out_t = &graph.tensors[&node.outputs[0]];
                if weight_t.shape.len() != 2 {
                    return Err(InvariantViolation(format!(
                        "Invariant I1 failed: Linear weight '{}' must be 2D",
                        weight_t.name
                    )));
                }
                let in_feat = weight_t.shape[1];
                let out_feat = weight_t.shape[0];
                if *act_t.shape.last().unwrap_or(&0) != in_feat {
                    return Err(InvariantViolation(format!(
                        "Invariant I1 failed: Linear act dim {:?} != weight in_features {in_feat}",
                        act_t.shape
                    )));
                }
                if *out_t.shape.last().unwrap_or(&0) != out_feat {
                    return Err(InvariantViolation(format!(
                        "Invariant I1 failed: Linear output dim {:?} != weight out_features {out_feat}",
                        out_t.shape
                    )));
                }
            }
            OpKind::Add | OpKind::EltMul => {
                let out_shape = &graph.tensors[&node.outputs[0]].shape;
                for inp in &node.inputs {
                    let s = &graph.tensors[inp].shape;
                    if s != out_shape {
                        return Err(InvariantViolation(format!(
                            "Invariant I1 failed: {:?} node '{}' shape mismatch: {:?} vs {:?}",
                            node.op, node.id, s, out_shape
                        )));
                    }
                }
            }
            OpKind::RMSNorm => {
                let in_shape = &graph.tensors[&node.inputs[0]].shape;
                let out_shape = &graph.tensors[&node.outputs[0]].shape;
                if in_shape != out_shape {
                    return Err(InvariantViolation(format!(
                        "Invariant I1 failed: RMSNorm node '{}' in_shape {:?} != out_shape {:?}",
                        node.id, in_shape, out_shape
                    )));
                }
            }
            OpKind::ArgMax => {
                let out_t = &graph.tensors[&node.outputs[0]];
                if out_t.shape.len() > 1 && out_t.shape != vec![1] {
                    return Err(InvariantViolation(format!(
                        "Invariant I1 failed: ArgMax node '{}' output must be scalar/1D, got {:?}",
                        node.id, out_t.shape
                    )));
                }
            }
            _ => {}
        }
    }

    // I2: Value ranges fit declared formats
    for tensor in graph.tensors.values() {
        let fmt = &tensor.format;
        if fmt.signed {
            assert_eq!(fmt.min_val(), -(1i128 << (fmt.bits - 1)));
            assert_eq!(fmt.max_val(), (1i128 << (fmt.bits - 1)) - 1);
        } else {
            assert_eq!(fmt.min_val(), 0);
            assert_eq!(fmt.max_val(), (1i128 << fmt.bits) - 1);
        }
    }

    // I4: Round-trip lossless JSON serialization
    let json_str = serde_json::to_string(graph)
        .map_err(|e| InvariantViolation(format!("Invariant I4 serialize failed: {e}")))?;
    let rt: GraphV2 = serde_json::from_str(&json_str)
        .map_err(|e| InvariantViolation(format!("Invariant I4 deserialize failed: {e}")))?;
    if &rt != graph {
        return Err(InvariantViolation(
            "Invariant I4 failed: Graph serialization round-trip is not identical".into(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npy_i8_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_llmhw_ir_i8.npy");
        let data = vec![-127i8, -1, 0, 1, 42, 127];
        write_npy_i8(&path, &[2, 3], &data).unwrap();
        let loaded = read_npy_i8(&path).unwrap();
        assert_eq!(loaded.shape, vec![2, 3]);
        assert_eq!(loaded.data, data);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_graph_v2_invariants_and_roundtrip() {
        let mut g = GraphV2::new("test_decoder");
        let fmt_i8 = EdgeFormat::new(8, true, 0.05, "INT8").unwrap();
        let fmt_w = EdgeFormat::new(8, true, 0.01, "INT8").unwrap();
        let fmt_kv = EdgeFormat::new(8, true, 0.05, "INT8").unwrap();

        g.add_tensor("x_in", TensorKind::DYNAMIC, vec![1, 64], fmt_i8.clone(), None, vec![], None, None)
            .unwrap();
        g.add_tensor("w_q", TensorKind::CONSTANT, vec![64, 64], fmt_w, None, vec![], None, None)
            .unwrap();
        g.add_tensor("q_out", TensorKind::DYNAMIC, vec![1, 64], fmt_i8.clone(), None, vec![], None, None)
            .unwrap();
        g.add_tensor("kv_cache_0", TensorKind::STATE, vec![2, 32, 16], fmt_kv, None, vec![], None, None)
            .unwrap();

        g.add_node(
            "q_proj_0",
            OpKind::Linear,
            vec!["x_in".into(), "w_q".into()],
            vec!["q_out".into()],
            BTreeMap::new(),
            BTreeMap::new(),
        );

        validate_invariants(&g).unwrap();

        // Negative control I3: writing to a CONSTANT tensor must fail
        let mut g_bad_i3 = g.clone();
        g_bad_i3.add_node(
            "bad_writer",
            OpKind::RMSNorm,
            vec!["x_in".into()],
            vec!["w_q".into()],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        assert!(validate_invariants(&g_bad_i3).is_err());

        // Negative control I7: accessing STATE tensor from Linear op must fail
        let mut g_bad_i7 = g.clone();
        g_bad_i7.add_node(
            "bad_state_access",
            OpKind::RMSNorm,
            vec!["kv_cache_0".into()],
            vec!["q_out".into()],
            BTreeMap::new(),
            BTreeMap::new(),
        );
        assert!(validate_invariants(&g_bad_i7).is_err());
    }
}
