//! Frontend Model Artifacts Loader, Tiny Model Generator, SafeTensors Parser, and `GraphV2` Builder (Phase R7).

use llmhw_ir::{
    read_npy_f64_any, sha256_hex, write_npy_f32, EdgeFormat, GraphV2, NpyArray, OpKind, TensorKind,
};
use memmap2::Mmap;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::path::Path;

pub const DEFAULT_MODEL_ID: &str = "HuggingFaceTB/SmolLM2-135M";
pub const PINNED_REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";

// ==============================================================================
// 1. Deterministic Tiny Model Generator (frontend/tiny_model.py)
// ==============================================================================

pub fn get_tiny_model_config_json() -> Value {
    json!({
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
    })
}

/// Deterministically generate and dump v2 tiny model artifacts to `target_dir`.
pub fn generate_and_dump_tiny_model_v2(target_dir: &Path, seed: u64) -> anyhow::Result<Value> {
    fs::create_dir_all(target_dir)?;
    let cfg = get_tiny_model_config_json();
    let vocab_size = 256usize;
    let hidden_size = 64usize;
    let intermediate_size = 128usize;
    let num_layers = 2usize;
    let num_q_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 16usize;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 0.04f32).unwrap();

    let mut tensors_meta = serde_json::Map::new();

    let mut save_tensor = |name: &str,
                           rel_path: &str,
                           shape: &[usize],
                           data: &[f32],
                           tied_to: Option<&str>|
     -> anyhow::Result<()> {
        let fpath = target_dir.join(rel_path);
        if let Some(parent) = fpath.parent() {
            fs::create_dir_all(parent)?;
        }
        write_npy_f32(&fpath, shape, data)?;
        let sha = sha256_hex(&fs::read(&fpath)?);
        tensors_meta.insert(
            name.to_string(),
            json!({
                "path": rel_path,
                "shape": shape,
                "dtype": "float32",
                "sha256": sha,
                "tied_to": tied_to,
            }),
        );
        Ok(())
    };

    let mut rand_vec = |len: usize| -> Vec<f32> { (0..len).map(|_| normal.sample(&mut rng)).collect() };

    let embed_w = rand_vec(vocab_size * hidden_size);
    save_tensor(
        "model.embed_tokens.weight",
        "embed_tokens.npy",
        &[vocab_size, hidden_size],
        &embed_w,
        None,
    )?;
    save_tensor(
        "lm_head.weight",
        "lm_head.npy",
        &[vocab_size, hidden_size],
        &embed_w,
        Some("model.embed_tokens.weight"),
    )?;

    let ones_h = vec![1.0f32; hidden_size];
    save_tensor(
        "model.norm.weight",
        "final_norm.npy",
        &[hidden_size],
        &ones_h,
        None,
    )?;

    for l_idx in 0..num_layers {
        let l_prefix = format!("model.layers.{l_idx}");
        let l_dir = format!("layer_{l_idx}");
        save_tensor(
            &format!("{l_prefix}.input_layernorm.weight"),
            &format!("{l_dir}/input_layernorm.npy"),
            &[hidden_size],
            &ones_h,
            None,
        )?;
        save_tensor(
            &format!("{l_prefix}.post_attention_layernorm.weight"),
            &format!("{l_dir}/post_attention_layernorm.npy"),
            &[hidden_size],
            &ones_h,
            None,
        )?;

        for (p_name, slug, rows, cols) in [
            ("self_attn.q_proj.weight", "q_proj", num_q_heads * head_dim, hidden_size),
            ("self_attn.k_proj.weight", "k_proj", num_kv_heads * head_dim, hidden_size),
            ("self_attn.v_proj.weight", "v_proj", num_kv_heads * head_dim, hidden_size),
            ("self_attn.o_proj.weight", "o_proj", hidden_size, hidden_size),
            ("mlp.gate_proj.weight", "gate_proj", intermediate_size, hidden_size),
            ("mlp.up_proj.weight", "up_proj", intermediate_size, hidden_size),
            ("mlp.down_proj.weight", "down_proj", hidden_size, intermediate_size),
        ] {
            let w = rand_vec(rows * cols);
            save_tensor(
                &format!("{l_prefix}.{p_name}"),
                &format!("{l_dir}/{slug}.npy"),
                &[rows, cols],
                &w,
                None,
            )?;
        }
    }

    let manifest = json!({
        "model_id": "tiny-decoder-v2",
        "revision": "tiny-seed-42",
        "config": cfg,
        "tensors": tensors_meta,
    });
    fs::write(
        target_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

// ==============================================================================
// 2. Native SafeTensors Extractor (`model.safetensors` -> v2 Artifacts)
// ==============================================================================

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn extract_safetensors_to_v2_artifacts(
    safetensors_path: &Path,
    config_json_path: &Path,
    target_dir: &Path,
    model_id: &str,
    revision: &str,
    max_ctx_override: Option<usize>,
) -> anyhow::Result<Value> {
    fs::create_dir_all(target_dir)?;
    let raw_cfg: Value = serde_json::from_str(&fs::read_to_string(config_json_path)?)?;
    let hidden_size = raw_cfg["hidden_size"].as_u64().unwrap() as usize;
    let num_q_heads = raw_cfg["num_attention_heads"].as_u64().unwrap() as usize;
    let head_dim = raw_cfg
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_q_heads);
    let max_ctx = max_ctx_override.unwrap_or_else(|| {
        raw_cfg
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize
    });
    let num_layers = raw_cfg["num_hidden_layers"].as_u64().unwrap() as usize;
    let tie_word_embeddings = raw_cfg
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let cfg = json!({
        "vocab_size": raw_cfg["vocab_size"].as_u64().unwrap(),
        "hidden_size": hidden_size,
        "intermediate_size": raw_cfg["intermediate_size"].as_u64().unwrap(),
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": raw_cfg["num_key_value_heads"].as_u64().unwrap(),
        "head_dim": head_dim,
        "max_position_embeddings": max_ctx,
        "rms_norm_eps": raw_cfg["rms_norm_eps"].as_f64().unwrap_or(1e-5),
        "rope_theta": raw_cfg.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(100000.0),
        "tie_word_embeddings": tie_word_embeddings,
    });

    let file = File::open(safetensors_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mut tensors_meta = serde_json::Map::new();

    let mut dump_key =
        |st_key: &str, manifest_key: &str, rel_path: &str, tied_to: Option<&str>| -> anyhow::Result<()> {
            let view = st.tensor(st_key)?;
            let shape: Vec<usize> = view.shape().to_vec();
            let raw = view.data();
            let f32_data: Vec<f32> = match view.dtype() {
                Dtype::F32 => raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&c| f32::from_le_bytes(c))
                    .collect(),
                Dtype::BF16 => raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&c| bf16_to_f32(u16::from_le_bytes(c)))
                    .collect(),
                other => anyhow::bail!("Unsupported SafeTensors dtype: {other:?}"),
            };
            let fpath = target_dir.join(rel_path);
            if let Some(p) = fpath.parent() {
                fs::create_dir_all(p)?;
            }
            write_npy_f32(&fpath, &shape, &f32_data)?;
            let sha = sha256_hex(&fs::read(&fpath)?);
            tensors_meta.insert(
                manifest_key.to_string(),
                json!({
                    "path": rel_path,
                    "shape": shape,
                    "dtype": "float32",
                    "sha256": sha,
                    "tied_to": tied_to,
                }),
            );
            Ok(())
        };

    dump_key(
        "model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.npy",
        None,
    )?;
    if tie_word_embeddings {
        dump_key(
            "model.embed_tokens.weight",
            "lm_head.weight",
            "lm_head.npy",
            Some("model.embed_tokens.weight"),
        )?;
    } else {
        dump_key("lm_head.weight", "lm_head.weight", "lm_head.npy", None)?;
    }
    dump_key("model.norm.weight", "model.norm.weight", "final_norm.npy", None)?;

    for l_idx in 0..num_layers {
        let l_prefix = format!("model.layers.{l_idx}");
        let l_dir = format!("layer_{l_idx}");
        let k_in = format!("{l_prefix}.input_layernorm.weight");
        dump_key(&k_in, &k_in, &format!("{l_dir}/input_layernorm.npy"), None)?;
        let k_post = format!("{l_prefix}.post_attention_layernorm.weight");
        dump_key(
            &k_post,
            &k_post,
            &format!("{l_dir}/post_attention_layernorm.npy"),
            None,
        )?;
        for (p_name, slug) in [
            ("self_attn.q_proj.weight", "q_proj"),
            ("self_attn.k_proj.weight", "k_proj"),
            ("self_attn.v_proj.weight", "v_proj"),
            ("self_attn.o_proj.weight", "o_proj"),
            ("mlp.gate_proj.weight", "gate_proj"),
            ("mlp.up_proj.weight", "up_proj"),
            ("mlp.down_proj.weight", "down_proj"),
        ] {
            let k_p = format!("{l_prefix}.{p_name}");
            dump_key(&k_p, &k_p, &format!("{l_dir}/{slug}.npy"), None)?;
        }
    }

    let manifest = json!({
        "model_id": model_id,
        "revision": revision,
        "config": cfg,
        "tensors": tensors_meta,
    });
    fs::write(
        target_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

// ==============================================================================
// 3. Reloading Verified v2 Artifacts & Building `GraphV2` (`importer_v2.py`)
// ==============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedArtifactsV2 {
    pub manifest: Value,
    pub config: Value,
    #[serde(skip)]
    pub tensors: HashMap<String, NpyArray<f64>>,
    pub calib_scales: BTreeMap<String, Value>,
}

pub fn load_model_artifacts_v2(target_dir: &Path) -> anyhow::Result<LoadedArtifactsV2> {
    let manifest_file = target_dir.join("manifest.json");
    if !manifest_file.is_file() {
        anyhow::bail!("Manifest not found at {}", manifest_file.display());
    }
    let manifest: Value = serde_json::from_str(&fs::read_to_string(&manifest_file)?)?;
    let config = manifest["config"].clone();

    let mut tensors = HashMap::new();
    if let Some(t_obj) = manifest.get("tensors").and_then(|v| v.as_object()) {
        for (name, info) in t_obj {
            let rel_path = info["path"].as_str().unwrap();
            let file_path = target_dir.join(rel_path);
            if !file_path.is_file() {
                anyhow::bail!("Artifact file missing: {}", file_path.display());
            }
            let raw_bytes = fs::read(&file_path)?;
            let computed_sha = sha256_hex(&raw_bytes);
            let exp_sha = info["sha256"].as_str().unwrap();
            if computed_sha != exp_sha {
                anyhow::bail!("SHA256 mismatch for {name}: expected {exp_sha}, got {computed_sha}");
            }
            let arr = read_npy_f64_any(&file_path)?;
            tensors.insert(name.clone(), arr);
        }
    }

    let calib_file = target_dir.join("calib_scales.json");
    let calib_scales: BTreeMap<String, Value> = if calib_file.is_file() {
        serde_json::from_str(&fs::read_to_string(&calib_file)?)?
    } else {
        BTreeMap::new()
    };

    Ok(LoadedArtifactsV2 {
        manifest,
        config,
        tensors,
        calib_scales,
    })
}

pub fn build_graph_v2_from_artifacts(target_dir: &Path) -> anyhow::Result<GraphV2> {
    let manifest_file = target_dir.join("manifest.json");
    let manifest: Value = serde_json::from_str(&fs::read_to_string(&manifest_file)?)?;
    let config = &manifest["config"];

    let hidden_size = config["hidden_size"].as_u64().unwrap() as usize;
    let intermediate_size = config["intermediate_size"].as_u64().unwrap() as usize;
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let num_layers = config["num_hidden_layers"].as_u64().unwrap() as usize;
    let num_q_heads = config["num_attention_heads"].as_u64().unwrap() as usize;
    let num_kv_heads = config["num_key_value_heads"].as_u64().unwrap() as usize;
    let head_dim = config["head_dim"].as_u64().unwrap() as usize;
    let max_ctx = config["max_position_embeddings"].as_u64().unwrap() as usize;
    let eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-5);
    let rope_theta = config["rope_theta"].as_f64().unwrap_or(100000.0);

    let model_id = manifest
        .get("model_id")
        .and_then(|v| v.as_str())
        .unwrap_or("transformer_decoder");
    let mut g = GraphV2::new(model_id);
    g.metadata.insert("config".into(), config.clone());

    let fmt_tok = EdgeFormat::new(16, false, 1.0, "UINT16")?;
    let fmt_res = EdgeFormat::new(16, true, 0.01, "INT16")?;
    let fmt_act = EdgeFormat::new(8, true, 0.05, "INT8")?;
    let fmt_w = EdgeFormat::new(8, true, 0.002, "INT8")?;
    let fmt_kv = EdgeFormat::new(8, true, 0.05, "INT8")?;

    let get_path_sha = |k: &str| -> (Option<String>, Option<String>) {
        let info = &manifest["tensors"][k];
        (
            info.get("path").and_then(|v| v.as_str()).map(|s| s.to_string()),
            info.get("sha256").and_then(|v| v.as_str()).map(|s| s.to_string()),
        )
    };
    let prov = |m: &str| -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        map.insert("module".into(), m.to_string());
        map
    };

    g.add_tensor("tok_in", TensorKind::DYNAMIC, vec![1], fmt_tok.clone(), None, vec![], None, None)?;

    let (emb_p, emb_s) = get_path_sha("model.embed_tokens.weight");
    g.add_tensor(
        "model.embed_tokens.weight",
        TensorKind::CONSTANT,
        vec![vocab_size, hidden_size],
        fmt_w.clone(),
        None,
        vec![],
        emb_p,
        emb_s,
    )?;
    g.add_tensor("h_0", TensorKind::DYNAMIC, vec![1, hidden_size], fmt_res.clone(), None, vec![], None, None)?;
    g.add_node(
        "embed_tokens",
        OpKind::Embed,
        vec!["tok_in".into(), "model.embed_tokens.weight".into()],
        vec!["h_0".into()],
        BTreeMap::new(),
        prov("model.embed_tokens"),
    );

    let mut h_curr = "h_0".to_string();
    for l_idx in 0..num_layers {
        let l_prefix = format!("model.layers.{l_idx}");

        let in_norm_key = format!("{l_prefix}.input_layernorm.weight");
        let (in_p, in_s) = get_path_sha(&in_norm_key);
        g.add_tensor(&in_norm_key, TensorKind::CONSTANT, vec![hidden_size], fmt_w.clone(), None, vec![], in_p, in_s)?;
        let h_attn_norm = format!("h_attn_norm_{l_idx}");
        g.add_tensor(&h_attn_norm, TensorKind::DYNAMIC, vec![1, hidden_size], fmt_act.clone(), None, vec![], None, None)?;
        let mut norm_attrs = BTreeMap::new();
        norm_attrs.insert("eps".into(), json!(eps));
        g.add_node(
            format!("attn_norm_{l_idx}"),
            OpKind::RMSNorm,
            vec![h_curr.clone(), in_norm_key],
            vec![h_attn_norm.clone()],
            norm_attrs.clone(),
            prov(&format!("{l_prefix}.input_layernorm")),
        );

        for (proj, out_dim) in [
            ("q_proj", num_q_heads * head_dim),
            ("k_proj", num_kv_heads * head_dim),
            ("v_proj", num_kv_heads * head_dim),
        ] {
            let w_key = format!("{l_prefix}.self_attn.{proj}.weight");
            let (wp, ws) = get_path_sha(&w_key);
            g.add_tensor(&w_key, TensorKind::CONSTANT, vec![out_dim, hidden_size], fmt_w.clone(), None, vec![], wp, ws)?;
            let out_name = format!("{proj}_{l_idx}");
            g.add_tensor(&out_name, TensorKind::DYNAMIC, vec![1, out_dim], fmt_act.clone(), None, vec![], None, None)?;
            g.add_node(
                &out_name,
                OpKind::Linear,
                vec![h_attn_norm.clone(), w_key],
                vec![out_name.clone()],
                BTreeMap::new(),
                prov(&format!("{l_prefix}.self_attn.{proj}")),
            );
        }

        let q_rope = format!("q_rope_{l_idx}");
        let k_rope = format!("k_rope_{l_idx}");
        g.add_tensor(&q_rope, TensorKind::DYNAMIC, vec![1, num_q_heads * head_dim], fmt_act.clone(), None, vec![], None, None)?;
        g.add_tensor(&k_rope, TensorKind::DYNAMIC, vec![1, num_kv_heads * head_dim], fmt_act.clone(), None, vec![], None, None)?;
        let mut rope_attrs = BTreeMap::new();
        rope_attrs.insert("head_dim".into(), json!(head_dim));
        rope_attrs.insert("theta".into(), json!(rope_theta));
        g.add_node(
            &q_rope,
            OpKind::RoPE,
            vec![format!("q_proj_{l_idx}")],
            vec![q_rope.clone()],
            rope_attrs.clone(),
            prov(&format!("{l_prefix}.self_attn.rotary_emb")),
        );
        g.add_node(
            &k_rope,
            OpKind::RoPE,
            vec![format!("k_proj_{l_idx}")],
            vec![k_rope.clone()],
            rope_attrs,
            prov(&format!("{l_prefix}.self_attn.rotary_emb")),
        );

        let kv_cache = format!("kv_cache_{l_idx}");
        g.add_tensor(&kv_cache, TensorKind::STATE, vec![2, num_kv_heads, max_ctx, head_dim], fmt_kv.clone(), None, vec![], None, None)?;
        let attn_ctx = format!("attn_context_{l_idx}");
        g.add_tensor(&attn_ctx, TensorKind::DYNAMIC, vec![1, hidden_size], fmt_act.clone(), None, vec![], None, None)?;
        let mut attn_attrs = BTreeMap::new();
        attn_attrs.insert("num_q_heads".into(), json!(num_q_heads));
        attn_attrs.insert("num_kv_heads".into(), json!(num_kv_heads));
        attn_attrs.insert("head_dim".into(), json!(head_dim));
        attn_attrs.insert("max_ctx".into(), json!(max_ctx));
        g.add_node(
            format!("attention_{l_idx}"),
            OpKind::Attention,
            vec![q_rope, k_rope, format!("v_proj_{l_idx}"), kv_cache],
            vec![attn_ctx.clone()],
            attn_attrs,
            prov(&format!("{l_prefix}.self_attn")),
        );

        let o_key = format!("{l_prefix}.self_attn.o_proj.weight");
        let (op_p, op_s) = get_path_sha(&o_key);
        g.add_tensor(&o_key, TensorKind::CONSTANT, vec![hidden_size, hidden_size], fmt_w.clone(), None, vec![], op_p, op_s)?;
        let o_proj = format!("o_proj_{l_idx}");
        g.add_tensor(&o_proj, TensorKind::DYNAMIC, vec![1, hidden_size], fmt_res.clone(), None, vec![], None, None)?;
        g.add_node(
            &o_proj,
            OpKind::Linear,
            vec![attn_ctx, o_key],
            vec![o_proj.clone()],
            BTreeMap::new(),
            prov(&format!("{l_prefix}.self_attn.o_proj")),
        );

        let h_mid = format!("h_mid_{l_idx}");
        g.add_tensor(&h_mid, TensorKind::DYNAMIC, vec![1, hidden_size], fmt_res.clone(), None, vec![], None, None)?;
        g.add_node(
            format!("res_add_attn_{l_idx}"),
            OpKind::Add,
            vec![h_curr, o_proj],
            vec![h_mid.clone()],
            BTreeMap::new(),
            prov(&format!("{l_prefix}.residual_1")),
        );

        let post_norm_key = format!("{l_prefix}.post_attention_layernorm.weight");
        let (pn_p, pn_s) = get_path_sha(&post_norm_key);
        g.add_tensor(&post_norm_key, TensorKind::CONSTANT, vec![hidden_size], fmt_w.clone(), None, vec![], pn_p, pn_s)?;
        let h_mlp_norm = format!("h_mlp_norm_{l_idx}");
        g.add_tensor(&h_mlp_norm, TensorKind::DYNAMIC, vec![1, hidden_size], fmt_act.clone(), None, vec![], None, None)?;
        g.add_node(
            format!("mlp_norm_{l_idx}"),
            OpKind::RMSNorm,
            vec![h_mid.clone(), post_norm_key],
            vec![h_mlp_norm.clone()],
            norm_attrs,
            prov(&format!("{l_prefix}.post_attention_layernorm")),
        );

        for (proj, t_out) in [("gate_proj", format!("gate_{l_idx}")), ("up_proj", format!("up_{l_idx}"))] {
            let w_key = format!("{l_prefix}.mlp.{proj}.weight");
            let (wp, ws) = get_path_sha(&w_key);
            g.add_tensor(&w_key, TensorKind::CONSTANT, vec![intermediate_size, hidden_size], fmt_w.clone(), None, vec![], wp, ws)?;
            g.add_tensor(&t_out, TensorKind::DYNAMIC, vec![1, intermediate_size], fmt_act.clone(), None, vec![], None, None)?;
            g.add_node(
                format!("{proj}_{l_idx}"),
                OpKind::Linear,
                vec![h_mlp_norm.clone(), w_key],
                vec![t_out],
                BTreeMap::new(),
                prov(&format!("{l_prefix}.mlp.{proj}")),
            );
        }

        let silu_t = format!("silu_{l_idx}");
        g.add_tensor(&silu_t, TensorKind::DYNAMIC, vec![1, intermediate_size], fmt_act.clone(), None, vec![], None, None)?;
        g.add_node(
            &silu_t,
            OpKind::SiLU,
            vec![format!("gate_{l_idx}")],
            vec![silu_t.clone()],
            BTreeMap::new(),
            prov(&format!("{l_prefix}.mlp.act_fn")),
        );

        let swiglu_t = format!("swiglu_{l_idx}");
        g.add_tensor(&swiglu_t, TensorKind::DYNAMIC, vec![1, intermediate_size], fmt_act.clone(), None, vec![], None, None)?;
        g.add_node(
            &swiglu_t,
            OpKind::EltMul,
            vec![silu_t, format!("up_{l_idx}")],
            vec![swiglu_t.clone()],
            BTreeMap::new(),
            prov(&format!("{l_prefix}.mlp.mul")),
        );

        let down_key = format!("{l_prefix}.mlp.down_proj.weight");
        let (dp, ds) = get_path_sha(&down_key);
        g.add_tensor(&down_key, TensorKind::CONSTANT, vec![hidden_size, intermediate_size], fmt_w.clone(), None, vec![], dp, ds)?;
        let down_t = format!("down_{l_idx}");
        g.add_tensor(&down_t, TensorKind::DYNAMIC, vec![1, hidden_size], fmt_res.clone(), None, vec![], None, None)?;
        g.add_node(
            format!("down_proj_{l_idx}"),
            OpKind::Linear,
            vec![swiglu_t, down_key],
            vec![down_t.clone()],
            BTreeMap::new(),
            prov(&format!("{l_prefix}.mlp.down_proj")),
        );

        let h_next = format!("h_{}", l_idx + 1);
        g.add_tensor(&h_next, TensorKind::DYNAMIC, vec![1, hidden_size], fmt_res.clone(), None, vec![], None, None)?;
        g.add_node(
            format!("res_add_mlp_{l_idx}"),
            OpKind::Add,
            vec![h_mid, down_t],
            vec![h_next.clone()],
            BTreeMap::new(),
            prov(&format!("{l_prefix}.residual_2")),
        );

        h_curr = h_next;
    }

    let (fn_p, fn_s) = get_path_sha("model.norm.weight");
    g.add_tensor("model.norm.weight", TensorKind::CONSTANT, vec![hidden_size], fmt_w.clone(), None, vec![], fn_p, fn_s)?;
    g.add_tensor("h_final_norm", TensorKind::DYNAMIC, vec![1, hidden_size], fmt_act, None, vec![], None, None)?;
    let mut fn_attrs = BTreeMap::new();
    fn_attrs.insert("eps".into(), json!(eps));
    g.add_node(
        "final_norm",
        OpKind::RMSNorm,
        vec![h_curr, "model.norm.weight".into()],
        vec!["h_final_norm".into()],
        fn_attrs,
        prov("model.norm"),
    );

    let (lm_p, lm_s) = get_path_sha("lm_head.weight");
    g.add_tensor("lm_head.weight", TensorKind::CONSTANT, vec![vocab_size, hidden_size], fmt_w, None, vec![], lm_p, lm_s)?;
    g.add_tensor("logits", TensorKind::DYNAMIC, vec![1, vocab_size], fmt_res, None, vec![], None, None)?;
    g.add_node(
        "lm_head",
        OpKind::Linear,
        vec!["h_final_norm".into(), "lm_head.weight".into()],
        vec!["logits".into()],
        BTreeMap::new(),
        prov("lm_head"),
    );

    g.add_tensor("tok_out", TensorKind::DYNAMIC, vec![1], fmt_tok, None, vec![], None, None)?;
    g.add_node(
        "argmax",
        OpKind::ArgMax,
        vec!["logits".into()],
        vec!["tok_out".into()],
        BTreeMap::new(),
        prov("argmax"),
    );

    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmhw_ir::validate_invariants;

    #[test]
    fn test_tiny_model_v2_dump_reload_and_graph_build() {
        let dir = std::env::temp_dir().join("llmhw_rust_tiny_v2_test");
        let _ = fs::remove_dir_all(&dir);
        generate_and_dump_tiny_model_v2(&dir, 42).unwrap();
        let loaded = load_model_artifacts_v2(&dir).unwrap();
        assert_eq!(loaded.tensors.len(), 21);
        let graph = build_graph_v2_from_artifacts(&dir).unwrap();
        validate_invariants(&graph).unwrap();
        let _ = fs::remove_dir_all(&dir);
    }
}
