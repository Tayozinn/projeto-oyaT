//! Agente 2 — Engenharia do Modelo Neural Próprio (Transformer Decoder-Only em Rust Puro).
//!
//! Implementa do zero:
//! - Embeddings de token e posição
//! - Blocos Transformer Decoder-only (RMSNorm, Multi-Head Causal Self-Attention, MLP GELU, residuais)
//! - Função de perda Cross-Entropy + Backpropagation analítica completa
//! - Otimizador AdamW com clipping de norma de gradiente
//! - Inferência autoregressiva (greedy e top-k/temperatura)
//! - Salvamento e carregamento de checkpoints binários completos (pesos + estado do otimizador).

#![allow(
    clippy::approx_constant,
    clippy::chunks_exact_to_as_chunks,
    clippy::derivable_impls,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::manual_c_str_literals,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::missing_transmute_annotations,
    clippy::needless_lifetimes,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::unnecessary_map_or
)]

pub mod gpu;
pub mod ops;
pub mod quant;
pub mod spirv;

pub use self::gpu::{ComputeDevice, VulkanDevice, VulkanTrainingSession};
use self::ops::{axpy_f32, dot_f32, matmul_nn, matmul_nt, matmul_tn};
pub use self::quant::{
    load_quantized_checkpoint, quantize_model_in_memory, save_quantized_checkpoint,
    KvCachePrecision, WeightPrecision,
};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const CHECKPOINT_MAGIC: &[u8; 4] = b"SMNN";
const CHECKPOINT_VERSION_V1: u32 = 1;
const CHECKPOINT_VERSION: u32 = 2;
const MAX_CHECKPOINT_META_BYTES: usize = 1_048_576; // 1 MiB
const CHECKPOINT_IO_BUF_CAP: usize = 256 * 1024; // 256 KiB

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
}

impl TransformerConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0
            || self.max_seq_len == 0
            || self.d_model == 0
            || self.n_layers == 0
            || self.d_ff == 0
        {
            return Err("Dimensões do Transformer devem ser maiores que zero.".to_string());
        }
        if self.n_heads == 0 || self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model ({}) deve ser divisível por n_heads ({}).",
                self.d_model, self.n_heads
            ));
        }
        Ok(())
    }

    pub fn checked_total_params(&self) -> Option<usize> {
        let tok_e = self.vocab_size.checked_mul(self.d_model)?;
        let pos_e = self.max_seq_len.checked_mul(self.d_model)?;
        let embed = tok_e.checked_add(pos_e)?;

        let qkvo = 4usize
            .checked_mul(self.d_model)?
            .checked_mul(self.d_model)?;
        let ffn = 2usize.checked_mul(self.d_model)?.checked_mul(self.d_ff)?;
        let norms_biases = self.d_ff.checked_add(3usize.checked_mul(self.d_model)?)?;
        let per_layer = qkvo.checked_add(ffn)?.checked_add(norms_biases)?;
        let all_layers = self.n_layers.checked_mul(per_layer)?;

        let head = self
            .d_model
            .checked_add(self.d_model.checked_mul(self.vocab_size)?)?;
        embed.checked_add(all_layers)?.checked_add(head)
    }

    pub fn total_params(&self) -> usize {
        self.checked_total_params().unwrap_or(usize::MAX)
    }
}

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub ln1_gamma: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ln2_gamma: Vec<f32>,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ServerMindTransformer {
    pub config: TransformerConfig,
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f_gamma: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub quantized: Option<Box<quant::QuantizedModelData>>,
}

#[derive(Debug, Clone)]
pub struct BlockGrads {
    pub ln1_gamma: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ln2_gamma: Vec<f32>,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TransformerGrads {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub blocks: Vec<BlockGrads>,
    pub ln_f_gamma: Vec<f32>,
    pub lm_head: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct AdamWState {
    pub step: usize,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

/// Cache de chaves e valores (KV Cache) para inferência autoregressiva incremental,
/// evitando recomputar projeções e atenção de posições anteriores (`O(T^2)` total em vez de `O(T^3)`).
/// Suporta armazenamento em `Fp32` (padrão), `Fp16` (50% de redução de RAM) ou `Int8` (~74% de redução de RAM).
#[derive(Debug, Clone)]
pub struct KvCache {
    pub precision: KvCachePrecision,
    pub k_layers: Vec<Vec<f32>>,
    pub v_layers: Vec<Vec<f32>>,
    pub k_fp16: Vec<Vec<u16>>,
    pub v_fp16: Vec<Vec<u16>>,
    pub k_int8: Vec<Vec<i8>>,
    pub v_int8: Vec<Vec<i8>>,
    pub k_scales: Vec<Vec<f32>>,
    pub v_scales: Vec<Vec<f32>>,
    pub cached_len: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_layers: usize,
}

impl KvCache {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(config: &TransformerConfig) -> Self {
        Self::with_precision(config, KvCachePrecision::Fp32)
    }

    pub fn with_precision(config: &TransformerConfig, precision: KvCachePrecision) -> Self {
        let cap = config.max_seq_len * config.d_model;
        let l = config.n_layers;
        match precision {
            KvCachePrecision::Fp32 => Self {
                precision,
                k_layers: (0..l).map(|_| vec![0.0; cap]).collect(),
                v_layers: (0..l).map(|_| vec![0.0; cap]).collect(),
                k_fp16: Vec::new(),
                v_fp16: Vec::new(),
                k_int8: Vec::new(),
                v_int8: Vec::new(),
                k_scales: Vec::new(),
                v_scales: Vec::new(),
                cached_len: 0,
                max_seq_len: config.max_seq_len,
                d_model: config.d_model,
                n_layers: l,
            },
            KvCachePrecision::Fp16 => Self {
                precision,
                k_layers: Vec::new(),
                v_layers: Vec::new(),
                k_fp16: (0..l).map(|_| vec![0u16; cap]).collect(),
                v_fp16: (0..l).map(|_| vec![0u16; cap]).collect(),
                k_int8: Vec::new(),
                v_int8: Vec::new(),
                k_scales: Vec::new(),
                v_scales: Vec::new(),
                cached_len: 0,
                max_seq_len: config.max_seq_len,
                d_model: config.d_model,
                n_layers: l,
            },
            KvCachePrecision::Int8 => Self {
                precision,
                k_layers: Vec::new(),
                v_layers: Vec::new(),
                k_fp16: Vec::new(),
                v_fp16: Vec::new(),
                k_int8: (0..l).map(|_| vec![0i8; cap]).collect(),
                v_int8: (0..l).map(|_| vec![0i8; cap]).collect(),
                k_scales: (0..l).map(|_| vec![0.0f32; config.max_seq_len]).collect(),
                v_scales: (0..l).map(|_| vec![0.0f32; config.max_seq_len]).collect(),
                cached_len: 0,
                max_seq_len: config.max_seq_len,
                d_model: config.d_model,
                n_layers: l,
            },
        }
    }

    pub fn clear(&mut self) {
        self.cached_len = 0;
    }

    pub fn is_compatible(&self, config: &TransformerConfig) -> bool {
        self.max_seq_len == config.max_seq_len
            && self.d_model == config.d_model
            && self.n_layers == config.n_layers
            && match self.precision {
                KvCachePrecision::Fp32 => {
                    self.k_layers.len() == config.n_layers && self.v_layers.len() == config.n_layers
                }
                KvCachePrecision::Fp16 => {
                    self.k_fp16.len() == config.n_layers && self.v_fp16.len() == config.n_layers
                }
                KvCachePrecision::Int8 => {
                    self.k_int8.len() == config.n_layers
                        && self.v_int8.len() == config.n_layers
                        && self.k_scales.len() == config.n_layers
                        && self.v_scales.len() == config.n_layers
                }
            }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn memory_bytes(&self) -> usize {
        let f32_bytes: usize = self
            .k_layers
            .iter()
            .chain(self.v_layers.iter())
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum();
        let fp16_bytes: usize = self
            .k_fp16
            .iter()
            .chain(self.v_fp16.iter())
            .map(|v| v.len() * std::mem::size_of::<u16>())
            .sum();
        let int8_bytes: usize = self
            .k_int8
            .iter()
            .chain(self.v_int8.iter())
            .map(|v| v.len())
            .sum::<usize>()
            + self
                .k_scales
                .iter()
                .chain(self.v_scales.iter())
                .map(|v| v.len() * std::mem::size_of::<f32>())
                .sum::<usize>();
        f32_bytes + fp16_bytes + int8_bytes
    }

    fn store_kv_range(&mut self, layer: usize, pos_start: usize, k_src: &[f32], v_src: &[f32]) {
        let d = self.d_model;
        let n_new = k_src.len() / d;
        match self.precision {
            KvCachePrecision::Fp32 => {
                let start = pos_start * d;
                let end = (pos_start + n_new) * d;
                self.k_layers[layer][start..end].copy_from_slice(k_src);
                self.v_layers[layer][start..end].copy_from_slice(v_src);
            }
            KvCachePrecision::Fp16 => {
                let start = pos_start * d;
                let k_dst = &mut self.k_fp16[layer][start..start + n_new * d];
                let v_dst = &mut self.v_fp16[layer][start..start + n_new * d];
                for i in 0..(n_new * d) {
                    k_dst[i] = quant::f32_to_f16(k_src[i]);
                    v_dst[i] = quant::f32_to_f16(v_src[i]);
                }
            }
            KvCachePrecision::Int8 => {
                for step in 0..n_new {
                    let abs_pos = pos_start + step;
                    let kr = &k_src[step * d..(step + 1) * d];
                    let vr = &v_src[step * d..(step + 1) * d];
                    let k_max = kr
                        .iter()
                        .copied()
                        .filter(|x| x.is_finite())
                        .fold(0.0f32, |m, x| m.max(x.abs()));
                    let v_max = vr
                        .iter()
                        .copied()
                        .filter(|x| x.is_finite())
                        .fold(0.0f32, |m, x| m.max(x.abs()));
                    let ks = if k_max > 0.0 { k_max / 127.0 } else { 0.0 };
                    let vs = if v_max > 0.0 { v_max / 127.0 } else { 0.0 };
                    self.k_scales[layer][abs_pos] = ks;
                    self.v_scales[layer][abs_pos] = vs;

                    let k_dst = &mut self.k_int8[layer][abs_pos * d..(abs_pos + 1) * d];
                    let v_dst = &mut self.v_int8[layer][abs_pos * d..(abs_pos + 1) * d];
                    if ks > 0.0 {
                        let inv = 1.0 / ks;
                        for c in 0..d {
                            k_dst[c] = (kr[c] * inv).round().clamp(-127.0, 127.0) as i8;
                        }
                    } else {
                        k_dst.fill(0);
                    }
                    if vs > 0.0 {
                        let inv = 1.0 / vs;
                        for c in 0..d {
                            v_dst[c] = (vr[c] * inv).round().clamp(-127.0, 127.0) as i8;
                        }
                    } else {
                        v_dst.fill(0);
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct WorkspaceLayerCache {
    x_in: Vec<f32>,
    ln1: Vec<f32>,
    inv_rms1: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_probs: Vec<f32>,
    attn_ctx: Vec<f32>,
    x_mid: Vec<f32>,
    ln2: Vec<f32>,
    inv_rms2: Vec<f32>,
    h_pre: Vec<f32>,
    h_act: Vec<f32>,
}

/// Área de trabalho pré-alocada para treinamento zero-alloc (`SM-005`),
/// reutilizando todas as ativações intermediárias, gradientes temporários e `d_probs`
/// ao longo das amostras do batch e dos passos de otimização.
#[derive(Debug, Clone)]
pub struct TrainingWorkspace {
    config: TransformerConfig,
    layers: Vec<WorkspaceLayerCache>,
    x: Vec<f32>,
    attn_out: Vec<f32>,
    mlp_out: Vec<f32>,
    ln_f: Vec<f32>,
    inv_rms_f: Vec<f32>,
    logits: Vec<f32>,
    d_logits: Vec<f32>,
    d_ln_f: Vec<f32>,
    d_x: Vec<f32>,
    d_x_mid: Vec<f32>,
    d_x_in: Vec<f32>,
    d_rms_tmp: Vec<f32>,
    d_h_act: Vec<f32>,
    d_h_pre: Vec<f32>,
    d_ln2: Vec<f32>,
    d_attn_ctx: Vec<f32>,
    d_q: Vec<f32>,
    d_k: Vec<f32>,
    d_v: Vec<f32>,
    d_ln1: Vec<f32>,
    d_probs: Vec<f32>,
}

impl TrainingWorkspace {
    pub fn new(config: &TransformerConfig) -> Self {
        let max_t = config.max_seq_len;
        let d = config.d_model;
        let ff = config.d_ff;
        let v = config.vocab_size;
        let h = config.n_heads;
        let td = max_t * d;
        let tff = max_t * ff;
        let tv = max_t * v;
        let htt = h * max_t * max_t;

        let layers = (0..config.n_layers)
            .map(|_| WorkspaceLayerCache {
                x_in: vec![0.0; td],
                ln1: vec![0.0; td],
                inv_rms1: vec![0.0; max_t],
                q: vec![0.0; td],
                k: vec![0.0; td],
                v: vec![0.0; td],
                attn_probs: vec![0.0; htt],
                attn_ctx: vec![0.0; td],
                x_mid: vec![0.0; td],
                ln2: vec![0.0; td],
                inv_rms2: vec![0.0; max_t],
                h_pre: vec![0.0; tff],
                h_act: vec![0.0; tff],
            })
            .collect();

        Self {
            config: config.clone(),
            layers,
            x: vec![0.0; td],
            attn_out: vec![0.0; td],
            mlp_out: vec![0.0; td],
            ln_f: vec![0.0; td],
            inv_rms_f: vec![0.0; max_t],
            logits: vec![0.0; tv],
            d_logits: vec![0.0; tv],
            d_ln_f: vec![0.0; td],
            d_x: vec![0.0; td],
            d_x_mid: vec![0.0; td],
            d_x_in: vec![0.0; td],
            d_rms_tmp: vec![0.0; td],
            d_h_act: vec![0.0; tff],
            d_h_pre: vec![0.0; tff],
            d_ln2: vec![0.0; td],
            d_attn_ctx: vec![0.0; td],
            d_q: vec![0.0; td],
            d_k: vec![0.0; td],
            d_v: vec![0.0; td],
            d_ln1: vec![0.0; td],
            d_probs: vec![0.0; max_t],
        }
    }

    #[inline]
    pub fn ensure_compatible(&mut self, config: &TransformerConfig) {
        if &self.config != config {
            *self = Self::new(config);
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn memory_bytes(&self) -> usize {
        let per_layer: usize = self
            .layers
            .iter()
            .map(|l| {
                l.x_in.len()
                    + l.ln1.len()
                    + l.inv_rms1.len()
                    + l.q.len()
                    + l.k.len()
                    + l.v.len()
                    + l.attn_probs.len()
                    + l.attn_ctx.len()
                    + l.x_mid.len()
                    + l.ln2.len()
                    + l.inv_rms2.len()
                    + l.h_pre.len()
                    + l.h_act.len()
            })
            .sum();
        let scratch = self.x.len()
            + self.attn_out.len()
            + self.mlp_out.len()
            + self.ln_f.len()
            + self.inv_rms_f.len()
            + self.logits.len()
            + self.d_logits.len()
            + self.d_ln_f.len()
            + self.d_x.len()
            + self.d_x_mid.len()
            + self.d_x_in.len()
            + self.d_rms_tmp.len()
            + self.d_h_act.len()
            + self.d_h_pre.len()
            + self.d_ln2.len()
            + self.d_attn_ctx.len()
            + self.d_q.len()
            + self.d_k.len()
            + self.d_v.len()
            + self.d_ln1.len()
            + self.d_probs.len();
        (per_layer + scratch) * std::mem::size_of::<f32>()
    }
}

pub(super) mod serde_loss_f32 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(val: &f32, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if val.is_finite() {
            serializer.serialize_some(val)
        } else {
            serializer.serialize_none()
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<f32, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt = Option::<f32>::deserialize(deserializer)?;
        Ok(match opt {
            Some(v) if v.is_finite() => v,
            _ => f32::INFINITY,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub config: TransformerConfig,
    pub step: usize,
    #[serde(with = "serde_loss_f32")]
    pub best_val_loss: f32,
    pub has_optimizer: bool,
    #[serde(default)]
    pub optimizer_step: Option<usize>,
}

impl TransformerGrads {
    pub fn zeros_like(config: &TransformerConfig) -> Self {
        let d = config.d_model;
        let ff = config.d_ff;
        Self {
            token_emb: vec![0.0; config.vocab_size * d],
            pos_emb: vec![0.0; config.max_seq_len * d],
            blocks: (0..config.n_layers)
                .map(|_| BlockGrads {
                    ln1_gamma: vec![0.0; d],
                    wq: vec![0.0; d * d],
                    wk: vec![0.0; d * d],
                    wv: vec![0.0; d * d],
                    wo: vec![0.0; d * d],
                    ln2_gamma: vec![0.0; d],
                    w1: vec![0.0; d * ff],
                    b1: vec![0.0; ff],
                    w2: vec![0.0; ff * d],
                    b2: vec![0.0; d],
                })
                .collect(),
            ln_f_gamma: vec![0.0; d],
            lm_head: vec![0.0; d * config.vocab_size],
        }
    }

    pub fn zero_out(&mut self) {
        self.token_emb.fill(0.0);
        self.pos_emb.fill(0.0);
        for b in &mut self.blocks {
            b.ln1_gamma.fill(0.0);
            b.wq.fill(0.0);
            b.wk.fill(0.0);
            b.wv.fill(0.0);
            b.wo.fill(0.0);
            b.ln2_gamma.fill(0.0);
            b.w1.fill(0.0);
            b.b1.fill(0.0);
            b.w2.fill(0.0);
            b.b2.fill(0.0);
        }
        self.ln_f_gamma.fill(0.0);
        self.lm_head.fill(0.0);
    }

    pub fn scale(&mut self, factor: f32) {
        for x in &mut self.token_emb {
            *x *= factor;
        }
        for x in &mut self.pos_emb {
            *x *= factor;
        }
        for b in &mut self.blocks {
            for v in [
                &mut b.ln1_gamma,
                &mut b.wq,
                &mut b.wk,
                &mut b.wv,
                &mut b.wo,
                &mut b.ln2_gamma,
                &mut b.w1,
                &mut b.b1,
                &mut b.w2,
                &mut b.b2,
            ] {
                for x in v.iter_mut() {
                    *x *= factor;
                }
            }
        }
        for x in &mut self.ln_f_gamma {
            *x *= factor;
        }
        for x in &mut self.lm_head {
            *x *= factor;
        }
    }

    pub fn l2_norm(&self) -> f32 {
        let mut sum = 0.0f64;
        for x in &self.token_emb {
            sum += (*x as f64) * (*x as f64);
        }
        for x in &self.pos_emb {
            sum += (*x as f64) * (*x as f64);
        }
        for b in &self.blocks {
            for v in [
                &b.ln1_gamma,
                &b.wq,
                &b.wk,
                &b.wv,
                &b.wo,
                &b.ln2_gamma,
                &b.w1,
                &b.b1,
                &b.w2,
                &b.b2,
            ] {
                for x in v {
                    sum += (*x as f64) * (*x as f64);
                }
            }
        }
        for x in &self.ln_f_gamma {
            sum += (*x as f64) * (*x as f64);
        }
        for x in &self.lm_head {
            sum += (*x as f64) * (*x as f64);
        }
        sum.sqrt() as f32
    }
}

impl ServerMindTransformer {
    /// Inicializa um modelo Transformer Decoder-only com pesos aleatórios determinísticos (Xavier).
    pub fn new_random(config: TransformerConfig, seed: u64) -> Result<Self, String> {
        config.validate()?;
        let mut rng = SplitMix64::new(seed);
        let d = config.d_model;
        let ff = config.d_ff;
        let std_emb = 0.02;
        let std_attn = (1.0 / d as f32).sqrt();
        let std_ff1 = (1.0 / d as f32).sqrt();
        let std_ff2 = (1.0 / (ff as f32 * 2.0 * config.n_layers as f32).max(1.0)).sqrt();
        let std_proj = (1.0 / (d as f32 * 2.0 * config.n_layers as f32).max(1.0)).sqrt();

        let token_emb = rng.normal_vec(config.vocab_size * d, std_emb);
        let pos_emb = rng.normal_vec(config.max_seq_len * d, std_emb);

        let blocks = (0..config.n_layers)
            .map(|_| TransformerBlock {
                ln1_gamma: vec![1.0; d],
                wq: rng.normal_vec(d * d, std_attn),
                wk: rng.normal_vec(d * d, std_attn),
                wv: rng.normal_vec(d * d, std_attn),
                wo: rng.normal_vec(d * d, std_proj),
                ln2_gamma: vec![1.0; d],
                w1: rng.normal_vec(d * ff, std_ff1),
                b1: vec![0.0; ff],
                w2: rng.normal_vec(ff * d, std_ff2),
                b2: vec![0.0; d],
            })
            .collect();

        Ok(Self {
            token_emb,
            pos_emb,
            blocks,
            ln_f_gamma: vec![1.0; d],
            lm_head: rng.normal_vec(d * config.vocab_size, std_attn),
            config,
            quantized: None,
        })
    }

    /// Aloca um modelo com tensores zerados (sem custo de PRNG/Box-Muller) para preenchimento
    /// direto durante `load_checkpoint`, validando dimensões e overflow aritmético.
    pub fn new_zeros(config: TransformerConfig) -> Result<Self, String> {
        config.validate()?;
        if config.checked_total_params().is_none() {
            return Err("Overflow aritmético nas dimensões do Transformer.".to_string());
        }
        let d = config.d_model;
        let ff = config.d_ff;
        let blocks = (0..config.n_layers)
            .map(|_| TransformerBlock {
                ln1_gamma: vec![0.0; d],
                wq: vec![0.0; d * d],
                wk: vec![0.0; d * d],
                wv: vec![0.0; d * d],
                wo: vec![0.0; d * d],
                ln2_gamma: vec![0.0; d],
                w1: vec![0.0; d * ff],
                b1: vec![0.0; ff],
                w2: vec![0.0; ff * d],
                b2: vec![0.0; d],
            })
            .collect();

        Ok(Self {
            token_emb: vec![0.0; config.vocab_size * d],
            pos_emb: vec![0.0; config.max_seq_len * d],
            blocks,
            ln_f_gamma: vec![0.0; d],
            lm_head: vec![0.0; d * config.vocab_size],
            config,
            quantized: None,
        })
    }

    pub fn num_float_params(&self) -> usize {
        self.config.total_params()
    }

    #[inline]
    pub fn is_quantized(&self) -> bool {
        self.quantized.is_some()
    }

    pub fn weight_precision_name(&self) -> &'static str {
        match self.quantized {
            Some(ref q) => q.precision.as_str(),
            None => "fp32",
        }
    }

    /// Retorna o consumo real em bytes de todos os pesos residentes na RAM
    /// (incluindo os tensores `FP32` remanescentes e, quando quantizado, as matrizes comprimidas).
    pub fn actual_weight_bytes(&self) -> usize {
        let mut f32_elems =
            self.token_emb.len() + self.pos_emb.len() + self.ln_f_gamma.len() + self.lm_head.len();
        for b in &self.blocks {
            f32_elems += b.ln1_gamma.len()
                + b.wq.len()
                + b.wk.len()
                + b.wv.len()
                + b.wo.len()
                + b.ln2_gamma.len()
                + b.w1.len()
                + b.b1.len()
                + b.w2.len()
                + b.b2.len();
        }
        let q_bytes = self
            .quantized
            .as_ref()
            .map(|q| q.memory_bytes())
            .unwrap_or(0);
        f32_elems * std::mem::size_of::<f32>() + q_bytes
    }

    /// Executa apenas o forward pass e retorna os logits `[seq_len, vocab_size]`.
    /// Opera nativamente tanto em `FP32` quanto sobre modelos quantizados (`FP16`, `BF16`, `INT8`, `INT4`)
    /// sem expandir as matrizes de pesos na RAM.
    pub fn forward_logits(&self, input_ids: &[usize]) -> Vec<f32> {
        let t = input_ids.len().min(self.config.max_seq_len);
        let d = self.config.d_model;
        let v = self.config.vocab_size;
        if t == 0 || d == 0 || v == 0 {
            return Vec::new();
        }

        let mut x = vec![0.0f32; t * d];
        if let Some(ref qdata) = self.quantized {
            let mut te_row = vec![0.0f32; d];
            for (pos, &tok_id) in input_ids[..t].iter().enumerate() {
                let tid = tok_id.min(v.saturating_sub(1));
                qdata.token_emb.dequantize_row_into(tid, &mut te_row);
                let pe = &self.pos_emb[pos * d..(pos + 1) * d];
                let out = &mut x[pos * d..(pos + 1) * d];
                for i in 0..d {
                    out[i] = te_row[i] + pe[i];
                }
            }

            for (l, block) in self.blocks.iter().enumerate() {
                let qb = &qdata.blocks[l];
                let (ln1, _) = rmsnorm_forward(&x, &block.ln1_gamma, t, d);
                let attn = causal_mha_forward_quant(&ln1, qb, t, d, self.config.n_heads);
                for i in 0..x.len() {
                    x[i] += attn[i];
                }

                let (ln2, _) = rmsnorm_forward(&x, &block.ln2_gamma, t, d);
                let mlp = mlp_forward_quant(&ln2, block, qb, t, d, self.config.d_ff);
                for i in 0..x.len() {
                    x[i] += mlp[i];
                }
            }

            let (ln_f, _) = rmsnorm_forward(&x, &self.ln_f_gamma, t, d);
            let mut logits = vec![0.0f32; t * v];
            qdata.lm_head.matmul_nn(&ln_f, &mut logits, t);
            return logits;
        }

        for (pos, &tok_id) in input_ids[..t].iter().enumerate() {
            let tid = tok_id.min(v.saturating_sub(1));
            let te = &self.token_emb[tid * d..(tid + 1) * d];
            let pe = &self.pos_emb[pos * d..(pos + 1) * d];
            let out = &mut x[pos * d..(pos + 1) * d];
            for i in 0..d {
                out[i] = te[i] + pe[i];
            }
        }

        for block in &self.blocks {
            let (ln1, _) = rmsnorm_forward(&x, &block.ln1_gamma, t, d);
            let attn = causal_mha_forward(&ln1, block, t, d, self.config.n_heads);
            for i in 0..x.len() {
                x[i] += attn[i];
            }

            let (ln2, _) = rmsnorm_forward(&x, &block.ln2_gamma, t, d);
            let mlp = mlp_forward(&ln2, block, t, d, self.config.d_ff);
            for i in 0..x.len() {
                x[i] += mlp[i];
            }
        }

        let (ln_f, _) = rmsnorm_forward(&x, &self.ln_f_gamma, t, d);
        let mut logits = vec![0.0f32; t * v];
        matmul_nn(&ln_f, &self.lm_head, &mut logits, t, d, v);
        logits
    }

    /// Executa forward incremental usando `KvCache` (`Fp32`, `Fp16` ou `Int8`) para `new_tokens`
    /// e projeta `lm_head` **apenas para a última posição** (`[1, vocab_size]`), suportando tanto
    /// pesos `FP32` quanto pesos quantizados (`FP16`, `BF16`, `INT8`, `INT4`).
    pub fn forward_last_logits_cached(
        &self,
        new_tokens: &[usize],
        cache: &mut KvCache,
    ) -> Vec<f32> {
        let d = self.config.d_model;
        let v = self.config.vocab_size;
        let max_t = self.config.max_seq_len;
        if new_tokens.is_empty() || v == 0 || d == 0 || max_t == 0 {
            return Vec::new();
        }
        if !cache.is_compatible(&self.config) {
            let prec = cache.precision;
            *cache = KvCache::with_precision(&self.config, prec);
        }

        // Se os novos tokens ultrapassarem o espaço livre do cache, reinicia com os últimos `max_t` tokens.
        let tokens = if new_tokens.len() >= max_t {
            cache.clear();
            &new_tokens[new_tokens.len() - max_t..]
        } else if cache.cached_len + new_tokens.len() > max_t {
            cache.clear();
            new_tokens
        } else {
            new_tokens
        };

        let pos_start = cache.cached_len;
        let n_new = tokens.len();

        let mut x = vec![0.0f32; n_new * d];
        if let Some(ref qdata) = self.quantized {
            let mut te_row = vec![0.0f32; d];
            for (step_idx, &tok_id) in tokens.iter().enumerate() {
                let abs_pos = pos_start + step_idx;
                let tid = tok_id.min(v - 1);
                qdata.token_emb.dequantize_row_into(tid, &mut te_row);
                let pe = &self.pos_emb[abs_pos * d..(abs_pos + 1) * d];
                let out = &mut x[step_idx * d..(step_idx + 1) * d];
                for i in 0..d {
                    out[i] = te_row[i] + pe[i];
                }
            }
        } else {
            for (step_idx, &tok_id) in tokens.iter().enumerate() {
                let abs_pos = pos_start + step_idx;
                let tid = tok_id.min(v - 1);
                let te = &self.token_emb[tid * d..(tid + 1) * d];
                let pe = &self.pos_emb[abs_pos * d..(abs_pos + 1) * d];
                let out = &mut x[step_idx * d..(step_idx + 1) * d];
                for i in 0..d {
                    out[i] = te[i] + pe[i];
                }
            }
        }

        let n_heads = self.config.n_heads;
        let head_dim = d / n_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_len = pos_start + n_new;
        let mut q_new = vec![0.0f32; n_new * d];
        let mut k_new = vec![0.0f32; n_new * d];
        let mut v_new = vec![0.0f32; n_new * d];
        let mut ctx = vec![0.0f32; n_new * d];
        let mut attn_out = vec![0.0f32; n_new * d];
        let mut probs = vec![0.0f32; total_len];

        for (l, block) in self.blocks.iter().enumerate() {
            let (ln1, _) = rmsnorm_forward(&x, &block.ln1_gamma, n_new, d);

            q_new.fill(0.0);
            if let Some(ref qdata) = self.quantized {
                let qb = &qdata.blocks[l];
                qb.wq.matmul_nn(&ln1, &mut q_new, n_new);
                if cache.precision == KvCachePrecision::Fp32 {
                    let k_slice = &mut cache.k_layers[l][pos_start * d..total_len * d];
                    let v_slice = &mut cache.v_layers[l][pos_start * d..total_len * d];
                    k_slice.fill(0.0);
                    v_slice.fill(0.0);
                    qb.wk.matmul_nn(&ln1, k_slice, n_new);
                    qb.wv.matmul_nn(&ln1, v_slice, n_new);
                } else {
                    k_new.fill(0.0);
                    v_new.fill(0.0);
                    qb.wk.matmul_nn(&ln1, &mut k_new, n_new);
                    qb.wv.matmul_nn(&ln1, &mut v_new, n_new);
                    cache.store_kv_range(l, pos_start, &k_new, &v_new);
                }
            } else if cache.precision == KvCachePrecision::Fp32 {
                matmul_nn(&ln1, &block.wq, &mut q_new, n_new, d, d);
                let k_slice = &mut cache.k_layers[l][pos_start * d..total_len * d];
                let v_slice = &mut cache.v_layers[l][pos_start * d..total_len * d];
                k_slice.fill(0.0);
                v_slice.fill(0.0);
                matmul_nn(&ln1, &block.wk, k_slice, n_new, d, d);
                matmul_nn(&ln1, &block.wv, v_slice, n_new, d, d);
            } else {
                matmul_nn(&ln1, &block.wq, &mut q_new, n_new, d, d);
                k_new.fill(0.0);
                v_new.fill(0.0);
                matmul_nn(&ln1, &block.wk, &mut k_new, n_new, d, d);
                matmul_nn(&ln1, &block.wv, &mut v_new, n_new, d, d);
                cache.store_kv_range(l, pos_start, &k_new, &v_new);
            }

            ctx.fill(0.0);
            match cache.precision {
                KvCachePrecision::Fp32 => {
                    let k_all = &cache.k_layers[l][..total_len * d];
                    let v_all = &cache.v_layers[l][..total_len * d];
                    for h in 0..n_heads {
                        let h_off = h * head_dim;
                        for step_idx in 0..n_new {
                            let abs_i = pos_start + step_idx;
                            let q_row =
                                &q_new[step_idx * d + h_off..step_idx * d + h_off + head_dim];
                            let mut max_s = f32::NEG_INFINITY;
                            for j in 0..=abs_i {
                                let k_row = &k_all[j * d + h_off..j * d + h_off + head_dim];
                                let s = dot_f32(q_row, k_row) * scale;
                                probs[j] = s;
                                if s > max_s {
                                    max_s = s;
                                }
                            }
                            let mut sum_e = 0.0f32;
                            for j in 0..=abs_i {
                                let e = (probs[j] - max_s).exp();
                                probs[j] = e;
                                sum_e += e;
                            }
                            let inv = 1.0 / sum_e.max(1e-12);
                            let ctx_row =
                                &mut ctx[step_idx * d + h_off..step_idx * d + h_off + head_dim];
                            for j in 0..=abs_i {
                                let p = probs[j] * inv;
                                let v_row = &v_all[j * d + h_off..j * d + h_off + head_dim];
                                axpy_f32(p, v_row, ctx_row);
                            }
                        }
                    }
                }
                KvCachePrecision::Fp16 => {
                    let k_all = &cache.k_fp16[l][..total_len * d];
                    let v_all = &cache.v_fp16[l][..total_len * d];
                    for h in 0..n_heads {
                        let h_off = h * head_dim;
                        for step_idx in 0..n_new {
                            let abs_i = pos_start + step_idx;
                            let q_row =
                                &q_new[step_idx * d + h_off..step_idx * d + h_off + head_dim];
                            let mut max_s = f32::NEG_INFINITY;
                            for j in 0..=abs_i {
                                let k_row = &k_all[j * d + h_off..j * d + h_off + head_dim];
                                let mut dot = 0.0f32;
                                for hd in 0..head_dim {
                                    dot += q_row[hd] * quant::f16_to_f32(k_row[hd]);
                                }
                                let s = dot * scale;
                                probs[j] = s;
                                if s > max_s {
                                    max_s = s;
                                }
                            }
                            let mut sum_e = 0.0f32;
                            for j in 0..=abs_i {
                                let e = (probs[j] - max_s).exp();
                                probs[j] = e;
                                sum_e += e;
                            }
                            let inv = 1.0 / sum_e.max(1e-12);
                            let ctx_row =
                                &mut ctx[step_idx * d + h_off..step_idx * d + h_off + head_dim];
                            for j in 0..=abs_i {
                                let p = probs[j] * inv;
                                let v_row = &v_all[j * d + h_off..j * d + h_off + head_dim];
                                for hd in 0..head_dim {
                                    ctx_row[hd] += p * quant::f16_to_f32(v_row[hd]);
                                }
                            }
                        }
                    }
                }
                KvCachePrecision::Int8 => {
                    let k_all = &cache.k_int8[l][..total_len * d];
                    let v_all = &cache.v_int8[l][..total_len * d];
                    let ks = &cache.k_scales[l][..total_len];
                    let vs = &cache.v_scales[l][..total_len];
                    for h in 0..n_heads {
                        let h_off = h * head_dim;
                        for step_idx in 0..n_new {
                            let abs_i = pos_start + step_idx;
                            let q_row =
                                &q_new[step_idx * d + h_off..step_idx * d + h_off + head_dim];
                            let mut max_s = f32::NEG_INFINITY;
                            for j in 0..=abs_i {
                                let k_row = &k_all[j * d + h_off..j * d + h_off + head_dim];
                                let mut dot = 0.0f32;
                                for hd in 0..head_dim {
                                    dot += q_row[hd] * (k_row[hd] as f32);
                                }
                                let s = dot * ks[j] * scale;
                                probs[j] = s;
                                if s > max_s {
                                    max_s = s;
                                }
                            }
                            let mut sum_e = 0.0f32;
                            for j in 0..=abs_i {
                                let e = (probs[j] - max_s).exp();
                                probs[j] = e;
                                sum_e += e;
                            }
                            let inv = 1.0 / sum_e.max(1e-12);
                            let ctx_row =
                                &mut ctx[step_idx * d + h_off..step_idx * d + h_off + head_dim];
                            for j in 0..=abs_i {
                                let alpha = probs[j] * inv * vs[j];
                                let v_row = &v_all[j * d + h_off..j * d + h_off + head_dim];
                                for hd in 0..head_dim {
                                    ctx_row[hd] += alpha * (v_row[hd] as f32);
                                }
                            }
                        }
                    }
                }
            }

            attn_out.fill(0.0);
            if let Some(ref qdata) = self.quantized {
                qdata.blocks[l].wo.matmul_nn(&ctx, &mut attn_out, n_new);
            } else {
                matmul_nn(&ctx, &block.wo, &mut attn_out, n_new, d, d);
            }
            for i in 0..x.len() {
                x[i] += attn_out[i];
            }

            let (ln2, _) = rmsnorm_forward(&x, &block.ln2_gamma, n_new, d);
            let mlp = if let Some(ref qdata) = self.quantized {
                mlp_forward_quant(&ln2, block, &qdata.blocks[l], n_new, d, self.config.d_ff)
            } else {
                mlp_forward(&ln2, block, n_new, d, self.config.d_ff)
            };
            for i in 0..x.len() {
                x[i] += mlp[i];
            }
        }

        cache.cached_len = total_len;

        let last_x = &x[(n_new - 1) * d..n_new * d];
        let (ln_f_last, _) = rmsnorm_forward(last_x, &self.ln_f_gamma, 1, d);
        let mut last_logits = vec![0.0f32; v];
        if let Some(ref qdata) = self.quantized {
            qdata.lm_head.matmul_nn(&ln_f_last, &mut last_logits, 1);
        } else {
            matmul_nn(&ln_f_last, &self.lm_head, &mut last_logits, 1, d, v);
        }
        last_logits
    }

    /// Executa forward, perda Cross-Entropy (ignorando `pad_id`) e backpropagation analítica,
    /// acumulando gradientes em `grads`. Retorna `(soma_da_perda, tokens_validos)`.
    /// Para treinos em lote sem realocação por amostra, prefira [`Self::forward_and_backward_with_workspace`].
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn forward_and_backward(
        &self,
        input_ids: &[usize],
        target_ids: &[usize],
        pad_id: usize,
        grads: &mut TransformerGrads,
    ) -> (f32, usize) {
        let mut ws = TrainingWorkspace::new(&self.config);
        self.forward_and_backward_with_workspace(input_ids, target_ids, pad_id, grads, &mut ws)
    }

    /// Variante zero-alloc de [`Self::forward_and_backward`] que reutiliza integralmente os
    /// buffers de [`TrainingWorkspace`] (ativações de camada, gradientes intermediários e `d_probs`).
    pub fn forward_and_backward_with_workspace(
        &self,
        input_ids: &[usize],
        target_ids: &[usize],
        pad_id: usize,
        grads: &mut TransformerGrads,
        ws: &mut TrainingWorkspace,
    ) -> (f32, usize) {
        if self.is_quantized() {
            return (f32::NAN, 0);
        }
        let t = input_ids
            .len()
            .min(target_ids.len())
            .min(self.config.max_seq_len);
        if t == 0 {
            return (0.0, 0);
        }
        ws.ensure_compatible(&self.config);

        let d = self.config.d_model;
        let ff = self.config.d_ff;
        let v = self.config.vocab_size;
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let td = t * d;
        let tff = t * ff;
        let tv = t * v;

        for (pos, &tok_id) in input_ids[..t].iter().enumerate() {
            let tid = tok_id.min(v.saturating_sub(1));
            let te = &self.token_emb[tid * d..(tid + 1) * d];
            let pe = &self.pos_emb[pos * d..(pos + 1) * d];
            let out = &mut ws.x[pos * d..(pos + 1) * d];
            for i in 0..d {
                out[i] = te[i] + pe[i];
            }
        }

        for (l, block) in self.blocks.iter().enumerate() {
            let lc = &mut ws.layers[l];
            lc.x_in[..td].copy_from_slice(&ws.x[..td]);
            rmsnorm_forward_into(
                &lc.x_in[..td],
                &block.ln1_gamma,
                &mut lc.ln1[..td],
                &mut lc.inv_rms1[..t],
                t,
                d,
            );

            lc.q[..td].fill(0.0);
            lc.k[..td].fill(0.0);
            lc.v[..td].fill(0.0);
            matmul_nn(&lc.ln1[..td], &block.wq, &mut lc.q[..td], t, d, d);
            matmul_nn(&lc.ln1[..td], &block.wk, &mut lc.k[..td], t, d, d);
            matmul_nn(&lc.ln1[..td], &block.wv, &mut lc.v[..td], t, d, d);

            lc.attn_ctx[..td].fill(0.0);
            for h in 0..n_heads {
                let h_off = h * head_dim;
                for i in 0..t {
                    let mut max_score = f32::NEG_INFINITY;
                    let prob_row = &mut lc.attn_probs[(h * t + i) * t..(h * t + i + 1) * t];
                    let q_slice = &lc.q[i * d + h_off..i * d + h_off + head_dim];
                    for j in 0..=i {
                        let k_slice = &lc.k[j * d + h_off..j * d + h_off + head_dim];
                        let s = dot_f32(q_slice, k_slice) * scale;
                        prob_row[j] = s;
                        if s > max_score {
                            max_score = s;
                        }
                    }
                    let mut sum_exp = 0.0f32;
                    for j in 0..=i {
                        let e = (prob_row[j] - max_score).exp();
                        prob_row[j] = e;
                        sum_exp += e;
                    }
                    let inv = 1.0 / sum_exp.max(1e-12);
                    let ctx_slice = &mut lc.attn_ctx[i * d + h_off..i * d + h_off + head_dim];
                    for j in 0..=i {
                        let p = prob_row[j] * inv;
                        prob_row[j] = p;
                        let v_slice = &lc.v[j * d + h_off..j * d + h_off + head_dim];
                        axpy_f32(p, v_slice, ctx_slice);
                    }
                }
            }

            ws.attn_out[..td].fill(0.0);
            matmul_nn(
                &lc.attn_ctx[..td],
                &block.wo,
                &mut ws.attn_out[..td],
                t,
                d,
                d,
            );
            for i in 0..td {
                lc.x_mid[i] = lc.x_in[i] + ws.attn_out[i];
            }

            rmsnorm_forward_into(
                &lc.x_mid[..td],
                &block.ln2_gamma,
                &mut lc.ln2[..td],
                &mut lc.inv_rms2[..t],
                t,
                d,
            );
            lc.h_pre[..tff].fill(0.0);
            matmul_nn(&lc.ln2[..td], &block.w1, &mut lc.h_pre[..tff], t, d, ff);
            for pos in 0..t {
                let hp = &mut lc.h_pre[pos * ff..(pos + 1) * ff];
                let ha = &mut lc.h_act[pos * ff..(pos + 1) * ff];
                for j in 0..ff {
                    let z = hp[j] + block.b1[j];
                    hp[j] = z;
                    ha[j] = gelu(z);
                }
            }

            ws.mlp_out[..td].fill(0.0);
            matmul_nn(&lc.h_act[..tff], &block.w2, &mut ws.mlp_out[..td], t, ff, d);
            for pos in 0..t {
                let mo = &ws.mlp_out[pos * d..(pos + 1) * d];
                let xm = &lc.x_mid[pos * d..(pos + 1) * d];
                let xr = &mut ws.x[pos * d..(pos + 1) * d];
                for j in 0..d {
                    xr[j] = xm[j] + mo[j] + block.b2[j];
                }
            }
        }

        rmsnorm_forward_into(
            &ws.x[..td],
            &self.ln_f_gamma,
            &mut ws.ln_f[..td],
            &mut ws.inv_rms_f[..t],
            t,
            d,
        );
        ws.logits[..tv].fill(0.0);
        matmul_nn(&ws.ln_f[..td], &self.lm_head, &mut ws.logits[..tv], t, d, v);

        // Cross-entropy exata via log-sum-exp em f64 + d_logits analítico
        let mut total_loss = 0.0f64;
        let mut valid_tokens = 0usize;
        ws.d_logits[..tv].fill(0.0);

        for pos in 0..t {
            let tgt = target_ids[pos];
            if tgt == pad_id || tgt >= v {
                continue;
            }
            valid_tokens += 1;
            let row = &ws.logits[pos * v..(pos + 1) * v];
            let d_row = &mut ws.d_logits[pos * v..(pos + 1) * v];
            if row.iter().any(|x| !x.is_finite()) {
                total_loss = f64::NAN;
                d_row.fill(f32::NAN);
                continue;
            }
            let max_l = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_e = 0.0f64;
            for j in 0..v {
                let e = ((row[j] - max_l) as f64).exp();
                d_row[j] = e as f32;
                sum_e += e;
            }
            let safe_sum = sum_e.max(1e-30);
            let inv = (1.0 / safe_sum) as f32;
            for j in 0..v {
                d_row[j] *= inv;
            }
            let log_prob = (row[tgt] - max_l) as f64 - safe_sum.ln();
            total_loss += -log_prob;
            d_row[tgt] -= 1.0;
        }

        if valid_tokens == 0 {
            return (0.0, 0);
        }

        // Backprop LM Head: d_lm_head += ln_f^T @ d_logits, d_ln_f = d_logits @ lm_head^T
        matmul_tn(
            &ws.ln_f[..td],
            &ws.d_logits[..tv],
            &mut grads.lm_head,
            d,
            t,
            v,
        );
        ws.d_ln_f[..td].fill(0.0);
        matmul_nt(
            &ws.d_logits[..tv],
            &self.lm_head,
            &mut ws.d_ln_f[..td],
            t,
            v,
            d,
        );

        rmsnorm_backward_into(
            &ws.x[..td],
            &self.ln_f_gamma,
            &ws.inv_rms_f[..t],
            &ws.d_ln_f[..td],
            &mut grads.ln_f_gamma,
            &mut ws.d_x[..td],
            t,
            d,
        );

        // Backprop através dos blocos em ordem reversa sem alocações no heap.
        for l in (0..self.blocks.len()).rev() {
            let block = &self.blocks[l];
            let bg = &mut grads.blocks[l];
            let c = &ws.layers[l];

            ws.d_x_mid[..td].copy_from_slice(&ws.d_x[..td]);

            for pos in 0..t {
                let dx_row = &ws.d_x[pos * d..(pos + 1) * d];
                for j in 0..d {
                    bg.b2[j] += dx_row[j];
                }
            }
            matmul_tn(&c.h_act[..tff], &ws.d_x[..td], &mut bg.w2, ff, t, d);
            ws.d_h_act[..tff].fill(0.0);
            matmul_nt(&ws.d_x[..td], &block.w2, &mut ws.d_h_act[..tff], t, d, ff);

            for i in 0..tff {
                ws.d_h_pre[i] = ws.d_h_act[i] * gelu_grad(c.h_pre[i]);
            }

            for pos in 0..t {
                let dhp_row = &ws.d_h_pre[pos * ff..(pos + 1) * ff];
                for j in 0..ff {
                    bg.b1[j] += dhp_row[j];
                }
            }
            matmul_tn(&c.ln2[..td], &ws.d_h_pre[..tff], &mut bg.w1, d, t, ff);
            ws.d_ln2[..td].fill(0.0);
            matmul_nt(&ws.d_h_pre[..tff], &block.w1, &mut ws.d_ln2[..td], t, ff, d);

            rmsnorm_backward_into(
                &c.x_mid[..td],
                &block.ln2_gamma,
                &c.inv_rms2[..t],
                &ws.d_ln2[..td],
                &mut bg.ln2_gamma,
                &mut ws.d_rms_tmp[..td],
                t,
                d,
            );
            for i in 0..td {
                ws.d_x_mid[i] += ws.d_rms_tmp[i];
            }

            ws.d_x_in[..td].copy_from_slice(&ws.d_x_mid[..td]);
            matmul_tn(&c.attn_ctx[..td], &ws.d_x_mid[..td], &mut bg.wo, d, t, d);
            ws.d_attn_ctx[..td].fill(0.0);
            matmul_nt(
                &ws.d_x_mid[..td],
                &block.wo,
                &mut ws.d_attn_ctx[..td],
                t,
                d,
                d,
            );

            ws.d_q[..td].fill(0.0);
            ws.d_k[..td].fill(0.0);
            ws.d_v[..td].fill(0.0);

            for h in 0..n_heads {
                let h_off = h * head_dim;
                for i in 0..t {
                    let prob_row = &c.attn_probs[(h * t + i) * t..(h * t + i) * t + i + 1];
                    let out_base = i * d + h_off;
                    let d_out_slice = &ws.d_attn_ctx[out_base..out_base + head_dim];
                    let d_probs = &mut ws.d_probs[..=i];

                    for j in 0..=i {
                        let p = prob_row[j];
                        let v_base = j * d + h_off;
                        let v_slice = &c.v[v_base..v_base + head_dim];
                        d_probs[j] = dot_f32(d_out_slice, v_slice);
                        axpy_f32(p, d_out_slice, &mut ws.d_v[v_base..v_base + head_dim]);
                    }

                    let dot_pd = dot_f32(prob_row, d_probs);
                    let q_base = i * d + h_off;
                    let q_slice = &c.q[q_base..q_base + head_dim];
                    for j in 0..=i {
                        let d_score = prob_row[j] * (d_probs[j] - dot_pd) * scale;
                        let k_base = j * d + h_off;
                        let k_slice = &c.k[k_base..k_base + head_dim];
                        axpy_f32(d_score, k_slice, &mut ws.d_q[q_base..q_base + head_dim]);
                        axpy_f32(d_score, q_slice, &mut ws.d_k[k_base..k_base + head_dim]);
                    }
                }
            }

            matmul_tn(&c.ln1[..td], &ws.d_q[..td], &mut bg.wq, d, t, d);
            matmul_tn(&c.ln1[..td], &ws.d_k[..td], &mut bg.wk, d, t, d);
            matmul_tn(&c.ln1[..td], &ws.d_v[..td], &mut bg.wv, d, t, d);

            ws.d_ln1[..td].fill(0.0);
            matmul_nt(&ws.d_q[..td], &block.wq, &mut ws.d_ln1[..td], t, d, d);
            matmul_nt(&ws.d_k[..td], &block.wk, &mut ws.d_ln1[..td], t, d, d);
            matmul_nt(&ws.d_v[..td], &block.wv, &mut ws.d_ln1[..td], t, d, d);

            rmsnorm_backward_into(
                &c.x_in[..td],
                &block.ln1_gamma,
                &c.inv_rms1[..t],
                &ws.d_ln1[..td],
                &mut bg.ln1_gamma,
                &mut ws.d_rms_tmp[..td],
                t,
                d,
            );
            for i in 0..td {
                ws.d_x[i] = ws.d_x_in[i] + ws.d_rms_tmp[i];
            }
        }

        // Embeddings backward
        for (pos, &tok_id) in input_ids[..t].iter().enumerate() {
            let tid = tok_id.min(v.saturating_sub(1));
            let dx_row = &ws.d_x[pos * d..(pos + 1) * d];
            let te_g = &mut grads.token_emb[tid * d..(tid + 1) * d];
            let pe_g = &mut grads.pos_emb[pos * d..(pos + 1) * d];
            for i in 0..d {
                te_g[i] += dx_row[i];
                pe_g[i] += dx_row[i];
            }
        }

        (total_loss as f32, valid_tokens)
    }

    /// Implementação de referência escalar pura do `forward_and_backward` (Regra 3),
    /// preservada permanentemente para validação numérica contra `TrainingWorkspace` + AVX2/FMA.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn forward_and_backward_reference(
        &self,
        input_ids: &[usize],
        target_ids: &[usize],
        pad_id: usize,
        grads: &mut TransformerGrads,
    ) -> (f32, usize) {
        use self::ops::{matmul_nn_reference, matmul_nt_reference, matmul_tn_reference};
        if self.is_quantized() {
            return (f32::NAN, 0);
        }
        let t = input_ids
            .len()
            .min(target_ids.len())
            .min(self.config.max_seq_len);
        if t == 0 {
            return (0.0, 0);
        }
        let d = self.config.d_model;
        let ff = self.config.d_ff;
        let v = self.config.vocab_size;
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        struct RefLayerCache {
            x_in: Vec<f32>,
            ln1: Vec<f32>,
            inv_rms1: Vec<f32>,
            q: Vec<f32>,
            k: Vec<f32>,
            v: Vec<f32>,
            attn_probs: Vec<f32>,
            attn_ctx: Vec<f32>,
            x_mid: Vec<f32>,
            ln2: Vec<f32>,
            inv_rms2: Vec<f32>,
            h_pre: Vec<f32>,
            h_act: Vec<f32>,
        }

        let mut x = vec![0.0f32; t * d];
        for (pos, &tok_id) in input_ids[..t].iter().enumerate() {
            let tid = tok_id.min(v.saturating_sub(1));
            let te = &self.token_emb[tid * d..(tid + 1) * d];
            let pe = &self.pos_emb[pos * d..(pos + 1) * d];
            let out = &mut x[pos * d..(pos + 1) * d];
            for i in 0..d {
                out[i] = te[i] + pe[i];
            }
        }

        let mut caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let x_in = x.clone();
            let (ln1, inv_rms1) = rmsnorm_forward(&x_in, &block.ln1_gamma, t, d);

            let mut q = vec![0.0f32; t * d];
            let mut k = vec![0.0f32; t * d];
            let mut val = vec![0.0f32; t * d];
            matmul_nn_reference(&ln1, &block.wq, &mut q, t, d, d);
            matmul_nn_reference(&ln1, &block.wk, &mut k, t, d, d);
            matmul_nn_reference(&ln1, &block.wv, &mut val, t, d, d);

            let mut attn_probs = vec![0.0f32; n_heads * t * t];
            let mut attn_ctx = vec![0.0f32; t * d];

            for h in 0..n_heads {
                let h_off = h * head_dim;
                for i in 0..t {
                    let mut max_score = f32::NEG_INFINITY;
                    let prob_row = &mut attn_probs[(h * t + i) * t..(h * t + i + 1) * t];
                    for j in 0..=i {
                        let mut dot = 0.0f32;
                        let q_base = i * d + h_off;
                        let k_base = j * d + h_off;
                        for hd in 0..head_dim {
                            dot += q[q_base + hd] * k[k_base + hd];
                        }
                        let s = dot * scale;
                        prob_row[j] = s;
                        if s > max_score {
                            max_score = s;
                        }
                    }
                    let mut sum_exp = 0.0f32;
                    for j in 0..=i {
                        let e = (prob_row[j] - max_score).exp();
                        prob_row[j] = e;
                        sum_exp += e;
                    }
                    let inv = 1.0 / sum_exp.max(1e-12);
                    for j in 0..=i {
                        prob_row[j] *= inv;
                    }
                    let out_base = i * d + h_off;
                    for j in 0..=i {
                        let p = prob_row[j];
                        let v_base = j * d + h_off;
                        for hd in 0..head_dim {
                            attn_ctx[out_base + hd] += p * val[v_base + hd];
                        }
                    }
                }
            }

            let mut attn_out = vec![0.0f32; t * d];
            matmul_nn_reference(&attn_ctx, &block.wo, &mut attn_out, t, d, d);

            let mut x_mid = vec![0.0f32; t * d];
            for i in 0..x_mid.len() {
                x_mid[i] = x_in[i] + attn_out[i];
            }

            let (ln2, inv_rms2) = rmsnorm_forward(&x_mid, &block.ln2_gamma, t, d);
            let mut h_pre = vec![0.0f32; t * ff];
            matmul_nn_reference(&ln2, &block.w1, &mut h_pre, t, d, ff);
            for pos in 0..t {
                for j in 0..ff {
                    h_pre[pos * ff + j] += block.b1[j];
                }
            }
            let h_act: Vec<f32> = h_pre.iter().map(|&z| gelu(z)).collect();
            let mut mlp_out = vec![0.0f32; t * d];
            matmul_nn_reference(&h_act, &block.w2, &mut mlp_out, t, ff, d);
            for pos in 0..t {
                for j in 0..d {
                    mlp_out[pos * d + j] += block.b2[j];
                }
            }

            for i in 0..x.len() {
                x[i] = x_mid[i] + mlp_out[i];
            }

            caches.push(RefLayerCache {
                x_in,
                ln1,
                inv_rms1,
                q,
                k,
                v: val,
                attn_probs,
                attn_ctx,
                x_mid,
                ln2,
                inv_rms2,
                h_pre,
                h_act,
            });
        }

        let (ln_f, inv_rms_f) = rmsnorm_forward(&x, &self.ln_f_gamma, t, d);
        let mut logits = vec![0.0f32; t * v];
        matmul_nn_reference(&ln_f, &self.lm_head, &mut logits, t, d, v);

        let mut total_loss = 0.0f64;
        let mut valid_tokens = 0usize;
        let mut d_logits = vec![0.0f32; t * v];

        for pos in 0..t {
            let tgt = target_ids[pos];
            if tgt == pad_id || tgt >= v {
                continue;
            }
            valid_tokens += 1;
            let row = &logits[pos * v..(pos + 1) * v];
            let d_row = &mut d_logits[pos * v..(pos + 1) * v];
            if row.iter().any(|x| !x.is_finite()) {
                total_loss = f64::NAN;
                d_row.fill(f32::NAN);
                continue;
            }
            let max_l = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_e = 0.0f64;
            for j in 0..v {
                let e = ((row[j] - max_l) as f64).exp();
                d_row[j] = e as f32;
                sum_e += e;
            }
            let safe_sum = sum_e.max(1e-30);
            let inv = (1.0 / safe_sum) as f32;
            for j in 0..v {
                d_row[j] *= inv;
            }
            let log_prob = (row[tgt] - max_l) as f64 - safe_sum.ln();
            total_loss += -log_prob;
            d_row[tgt] -= 1.0;
        }

        if valid_tokens == 0 {
            return (0.0, 0);
        }

        matmul_tn_reference(&ln_f, &d_logits, &mut grads.lm_head, d, t, v);
        let mut d_ln_f = vec![0.0f32; t * d];
        matmul_nt_reference(&d_logits, &self.lm_head, &mut d_ln_f, t, v, d);

        let mut d_x = rmsnorm_backward(
            &x,
            &self.ln_f_gamma,
            &inv_rms_f,
            &d_ln_f,
            &mut grads.ln_f_gamma,
            t,
            d,
        );

        for l in (0..self.blocks.len()).rev() {
            let block = &self.blocks[l];
            let bg = &mut grads.blocks[l];
            let c = &caches[l];

            let mut d_x_mid = d_x.clone();

            for pos in 0..t {
                for j in 0..d {
                    bg.b2[j] += d_x[pos * d + j];
                }
            }
            matmul_tn_reference(&c.h_act, &d_x, &mut bg.w2, ff, t, d);
            let mut d_h_act = vec![0.0f32; t * ff];
            matmul_nt_reference(&d_x, &block.w2, &mut d_h_act, t, d, ff);

            let mut d_h_pre = vec![0.0f32; t * ff];
            for i in 0..d_h_act.len() {
                d_h_pre[i] = d_h_act[i] * gelu_grad(c.h_pre[i]);
            }

            for pos in 0..t {
                for j in 0..ff {
                    bg.b1[j] += d_h_pre[pos * ff + j];
                }
            }
            matmul_tn_reference(&c.ln2, &d_h_pre, &mut bg.w1, d, t, ff);
            let mut d_ln2 = vec![0.0f32; t * d];
            matmul_nt_reference(&d_h_pre, &block.w1, &mut d_ln2, t, ff, d);

            let d_x_mid_from_ln2 = rmsnorm_backward(
                &c.x_mid,
                &block.ln2_gamma,
                &c.inv_rms2,
                &d_ln2,
                &mut bg.ln2_gamma,
                t,
                d,
            );
            for i in 0..d_x_mid.len() {
                d_x_mid[i] += d_x_mid_from_ln2[i];
            }

            let mut d_x_in = d_x_mid.clone();

            matmul_tn_reference(&c.attn_ctx, &d_x_mid, &mut bg.wo, d, t, d);
            let mut d_attn_ctx = vec![0.0f32; t * d];
            matmul_nt_reference(&d_x_mid, &block.wo, &mut d_attn_ctx, t, d, d);

            let mut d_q = vec![0.0f32; t * d];
            let mut d_k = vec![0.0f32; t * d];
            let mut d_v = vec![0.0f32; t * d];
            let mut d_probs = vec![0.0f32; t];

            for h in 0..n_heads {
                let h_off = h * head_dim;
                for i in 0..t {
                    let prob_row = &c.attn_probs[(h * t + i) * t..(h * t + i + 1) * t];
                    let out_base = i * d + h_off;

                    for j in 0..=i {
                        let p = prob_row[j];
                        let v_base = j * d + h_off;
                        let mut dp = 0.0f32;
                        for hd in 0..head_dim {
                            let d_out = d_attn_ctx[out_base + hd];
                            dp += d_out * c.v[v_base + hd];
                            d_v[v_base + hd] += p * d_out;
                        }
                        d_probs[j] = dp;
                    }

                    let mut dot_pd = 0.0f32;
                    for j in 0..=i {
                        dot_pd += prob_row[j] * d_probs[j];
                    }
                    let q_base = i * d + h_off;
                    for j in 0..=i {
                        let d_score = prob_row[j] * (d_probs[j] - dot_pd) * scale;
                        let k_base = j * d + h_off;
                        for hd in 0..head_dim {
                            d_q[q_base + hd] += d_score * c.k[k_base + hd];
                            d_k[k_base + hd] += d_score * c.q[q_base + hd];
                        }
                    }
                }
            }

            matmul_tn_reference(&c.ln1, &d_q, &mut bg.wq, d, t, d);
            matmul_tn_reference(&c.ln1, &d_k, &mut bg.wk, d, t, d);
            matmul_tn_reference(&c.ln1, &d_v, &mut bg.wv, d, t, d);

            let mut d_ln1 = vec![0.0f32; t * d];
            matmul_nt_reference(&d_q, &block.wq, &mut d_ln1, t, d, d);
            matmul_nt_reference(&d_k, &block.wk, &mut d_ln1, t, d, d);
            matmul_nt_reference(&d_v, &block.wv, &mut d_ln1, t, d, d);

            let d_x_in_from_ln1 = rmsnorm_backward(
                &c.x_in,
                &block.ln1_gamma,
                &c.inv_rms1,
                &d_ln1,
                &mut bg.ln1_gamma,
                t,
                d,
            );
            for i in 0..d_x_in.len() {
                d_x_in[i] += d_x_in_from_ln1[i];
            }

            d_x = d_x_in;
        }

        for (pos, &tok_id) in input_ids[..t].iter().enumerate() {
            let tid = tok_id.min(v.saturating_sub(1));
            let dx_row = &d_x[pos * d..(pos + 1) * d];
            let te_g = &mut grads.token_emb[tid * d..(tid + 1) * d];
            let pe_g = &mut grads.pos_emb[pos * d..(pos + 1) * d];
            for i in 0..d {
                te_g[i] += dx_row[i];
                pe_g[i] += dx_row[i];
            }
        }

        (total_loss as f32, valid_tokens)
    }

    /// Implementação de referência da geração autoregressiva sem KV Cache (Regra 3),
    /// preservada para verificação de equivalência numérica e benchmarks comparativos.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn generate_reference(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        eos_id: usize,
        seed: u64,
    ) -> Vec<usize> {
        let v = self.config.vocab_size;
        if prompt_ids.is_empty() || max_new_tokens == 0 || v == 0 {
            return prompt_ids.to_vec();
        }

        let mut seq = prompt_ids.to_vec();
        let mut rng = SplitMix64::new(seed);
        let fallback_id = eos_id.min(v - 1);

        for _ in 0..max_new_tokens {
            let start = seq.len().saturating_sub(self.config.max_seq_len);
            let ctx = &seq[start..];
            let logits = self.forward_logits(ctx);
            let t_eff = logits.len() / v;
            if t_eff == 0 {
                break;
            }
            let last_off = (t_eff - 1) * v;
            let last_logits = &logits[last_off..last_off + v];

            let next_id = if !temperature.is_finite() || temperature <= 1e-6 {
                argmax_finite(last_logits).unwrap_or(fallback_id)
            } else {
                sample_top_k(last_logits, temperature, top_k.max(1), &mut rng)
                    .unwrap_or(fallback_id)
            };

            seq.push(next_id);
            if next_id == eos_id {
                break;
            }
        }
        seq
    }

    /// Gera novos tokens autoregressivamente acelerado por `KvCache`, com suporte a greedy
    /// (`temperature <= 0.0`) ou top-k, preservando equivalência com [`Self::generate_reference`].
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn generate(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        eos_id: usize,
        seed: u64,
    ) -> Vec<usize> {
        self.generate_with_options(
            prompt_ids,
            max_new_tokens,
            temperature,
            top_k,
            &[eos_id],
            seed,
            None,
        )
    }

    /// Geração autoregressiva com `KvCache` (`Fp32`), múltiplos `stop_tokens` e callback incremental por token.
    pub fn generate_with_options(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        stop_tokens: &[usize],
        seed: u64,
        on_token: Option<&mut dyn FnMut(usize)>,
    ) -> Vec<usize> {
        self.generate_with_kv_precision(
            prompt_ids,
            max_new_tokens,
            temperature,
            top_k,
            stop_tokens,
            seed,
            KvCachePrecision::Fp32,
            on_token,
        )
    }

    /// Geração autoregressiva parametrizada pela precisão do `KvCache` (`Fp32`, `Fp16` ou `Int8`).
    pub fn generate_with_kv_precision(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        stop_tokens: &[usize],
        seed: u64,
        kv_precision: KvCachePrecision,
        mut on_token: Option<&mut dyn FnMut(usize)>,
    ) -> Vec<usize> {
        let v = self.config.vocab_size;
        let max_t = self.config.max_seq_len;
        if prompt_ids.is_empty() || max_new_tokens == 0 || v == 0 || max_t == 0 {
            return prompt_ids.to_vec();
        }

        let mut seq = prompt_ids.to_vec();
        let mut rng = SplitMix64::new(seed);
        let fallback_id = stop_tokens.first().copied().unwrap_or(0).min(v - 1);
        let mut cache = KvCache::with_precision(&self.config, kv_precision);

        for step_idx in 0..max_new_tokens {
            let last_logits = if step_idx == 0 {
                let start = seq.len().saturating_sub(max_t);
                self.forward_last_logits_cached(&seq[start..], &mut cache)
            } else if cache.cached_len < max_t {
                let last_tok = *seq.last().unwrap();
                self.forward_last_logits_cached(&[last_tok], &mut cache)
            } else {
                // Janela deslizante atingiu `max_seq_len`: reconstrói contexto deslocado
                cache.clear();
                let start = seq.len().saturating_sub(max_t);
                self.forward_last_logits_cached(&seq[start..], &mut cache)
            };

            if last_logits.len() < v {
                break;
            }

            let next_id = if !temperature.is_finite() || temperature <= 1e-6 {
                argmax_finite(&last_logits).unwrap_or(fallback_id)
            } else {
                sample_top_k(&last_logits, temperature, top_k.max(1), &mut rng)
                    .unwrap_or(fallback_id)
            };

            seq.push(next_id);
            if stop_tokens.contains(&next_id) {
                break;
            }
            if let Some(cb) = on_token.as_deref_mut() {
                cb(next_id);
            }
        }
        seq
    }

    fn visit_param_slices_mut<'a>(&'a mut self, mut f: impl FnMut(&mut [f32])) {
        f(&mut self.token_emb);
        f(&mut self.pos_emb);
        for b in &mut self.blocks {
            f(&mut b.ln1_gamma);
            f(&mut b.wq);
            f(&mut b.wk);
            f(&mut b.wv);
            f(&mut b.wo);
            f(&mut b.ln2_gamma);
            f(&mut b.w1);
            f(&mut b.b1);
            f(&mut b.w2);
            f(&mut b.b2);
        }
        f(&mut self.ln_f_gamma);
        f(&mut self.lm_head);
    }

    fn visit_param_slices<'a>(&'a self, mut f: impl FnMut(&[f32])) {
        f(&self.token_emb);
        f(&self.pos_emb);
        for b in &self.blocks {
            f(&b.ln1_gamma);
            f(&b.wq);
            f(&b.wk);
            f(&b.wv);
            f(&b.wo);
            f(&b.ln2_gamma);
            f(&b.w1);
            f(&b.b1);
            f(&b.w2);
            f(&b.b2);
        }
        f(&self.ln_f_gamma);
        f(&self.lm_head);
    }
}

impl AdamWState {
    pub fn new(model: &ServerMindTransformer) -> Self {
        let total = model.num_float_params();
        Self {
            step: 0,
            m: vec![0.0; total],
            v: vec![0.0; total],
        }
    }

    /// Executa um passo AdamW com clipping de norma de gradiente.
    /// Retorna `false` sem alterar `step`, `m`, `v` ou os pesos do modelo caso os gradientes
    /// ou hiperparâmetros contenham `NaN` ou `Inf`.
    pub fn step(
        &mut self,
        model: &mut ServerMindTransformer,
        grads: &mut TransformerGrads,
        lr: f32,
        weight_decay: f32,
        max_grad_norm: f32,
    ) -> bool {
        if model.is_quantized() {
            return false;
        }
        let norm = grads.l2_norm();
        if !norm.is_finite() || !lr.is_finite() || !weight_decay.is_finite() {
            return false;
        }
        if max_grad_norm.is_finite() && max_grad_norm > 0.0 && norm > max_grad_norm {
            grads.scale(max_grad_norm / norm.max(1e-8));
        }

        self.step += 1;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let step_i32 = self.step.min(i32::MAX as usize) as i32;
        let bc1 = (1.0 - beta1.powi(step_i32)).max(1e-8);
        let bc2 = (1.0 - beta2.powi(step_i32)).max(1e-12);

        let mut offset = 0usize;
        let m_all = &mut self.m;
        let v_all = &mut self.v;

        let mut update_slice = |params: &mut [f32], g_slice: &[f32], apply_decay: bool| {
            for (i, (p, &g)) in params.iter_mut().zip(g_slice.iter()).enumerate() {
                let idx = offset + i;
                let m = beta1 * m_all[idx] + (1.0 - beta1) * g;
                let v = beta2 * v_all[idx] + (1.0 - beta2) * g * g;
                m_all[idx] = m;
                v_all[idx] = v;
                let m_hat = m / bc1;
                let v_hat = v / bc2;
                if apply_decay && weight_decay > 0.0 {
                    *p -= lr * weight_decay * *p;
                }
                *p -= lr * m_hat / (v_hat.sqrt() + eps);
            }
            offset += params.len();
        };

        update_slice(&mut model.token_emb, &grads.token_emb, false);
        update_slice(&mut model.pos_emb, &grads.pos_emb, false);
        for (b, bg) in model.blocks.iter_mut().zip(grads.blocks.iter()) {
            update_slice(&mut b.ln1_gamma, &bg.ln1_gamma, false);
            update_slice(&mut b.wq, &bg.wq, true);
            update_slice(&mut b.wk, &bg.wk, true);
            update_slice(&mut b.wv, &bg.wv, true);
            update_slice(&mut b.wo, &bg.wo, true);
            update_slice(&mut b.ln2_gamma, &bg.ln2_gamma, false);
            update_slice(&mut b.w1, &bg.w1, true);
            update_slice(&mut b.b1, &bg.b1, false);
            update_slice(&mut b.w2, &bg.w2, true);
            update_slice(&mut b.b2, &bg.b2, false);
        }
        update_slice(&mut model.ln_f_gamma, &grads.ln_f_gamma, false);
        update_slice(&mut model.lm_head, &grads.lm_head, true);
        true
    }
}

/// Salva pesos e estado opcional do otimizador AdamW no formato binário `SMNN` v2:
/// - Valida dimensões, contagens de parâmetros e finitude (`!NaN && !Inf`) antes de gravar;
/// - Calcula checksum CRC32 (IEEE 802.3) incremental sobre todo o conteúdo e grava no trailer;
/// - Utiliza escrita atômica (`temp file + flush + sync_all + rename` com retry curto no Windows).
pub fn save_checkpoint(
    path: &Path,
    model: &ServerMindTransformer,
    optimizer: Option<&AdamWState>,
    step: usize,
    best_val_loss: f32,
) -> Result<(), String> {
    if model.is_quantized() {
        return Err(
            "save_checkpoint requer pesos FP32 completos; para modelos quantizados use save_quantized_checkpoint."
                .to_string(),
        );
    }
    model.config.validate()?;
    let expected_params = model
        .config
        .checked_total_params()
        .ok_or_else(|| "Overflow no total de parâmetros ao salvar checkpoint.".to_string())?;

    if let Some(opt) = optimizer {
        if opt.m.len() != expected_params || opt.v.len() != expected_params {
            return Err(format!(
                "Estado AdamW inconsistente ao salvar checkpoint: esperado {} floats, obtido m={} v={}.",
                expected_params,
                opt.m.len(),
                opt.v.len()
            ));
        }
    }

    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => {
            fs::create_dir_all(p)
                .map_err(|e| format!("Falha ao criar pasta de checkpoint {:?}: {}", p, e))?;
            p
        }
        _ => Path::new("."),
    };

    let meta = CheckpointMetadata {
        config: model.config.clone(),
        step,
        best_val_loss,
        has_optimizer: optimizer.is_some(),
        optimizer_step: optimizer.map(|o| o.step),
    };
    let meta_bytes = serde_json::to_vec(&meta)
        .map_err(|e| format!("Falha ao serializar metadados do checkpoint: {}", e))?;
    if meta_bytes.len() > MAX_CHECKPOINT_META_BYTES {
        return Err(format!(
            "Metadados do checkpoint excedem o limite máximo permitido ({} > {} bytes).",
            meta_bytes.len(),
            MAX_CHECKPOINT_META_BYTES
        ));
    }
    let meta_len_u32 = u32::try_from(meta_bytes.len())
        .map_err(|_| "Tamanho de metadados excede u32::MAX.".to_string())?;

    let tmp_path = make_atomic_temp_path(parent, path);
    let write_result = (|| -> Result<(), String> {
        let file = File::create(&tmp_path).map_err(|e| {
            format!(
                "Falha ao criar arquivo temporário de checkpoint {:?}: {}",
                tmp_path, e
            )
        })?;
        let mut w = BufWriter::with_capacity(CHECKPOINT_IO_BUF_CAP, file);
        let mut crc = Crc32Hasher::new();

        write_bytes_hashed(&mut w, &mut crc, CHECKPOINT_MAGIC)?;
        write_bytes_hashed(&mut w, &mut crc, &CHECKPOINT_VERSION.to_le_bytes())?;
        write_bytes_hashed(&mut w, &mut crc, &meta_len_u32.to_le_bytes())?;
        write_bytes_hashed(&mut w, &mut crc, &meta_bytes)?;

        let mut written_params = 0usize;
        let mut slice_err: Option<String> = None;
        model.visit_param_slices(|slice| {
            if slice_err.is_some() {
                return;
            }
            if let Err(e) = write_f32_slice_hashed(&mut w, &mut crc, slice, "peso do modelo") {
                slice_err = Some(e);
                return;
            }
            written_params += slice.len();
        });
        if let Some(err) = slice_err {
            return Err(err);
        }
        if written_params != expected_params {
            return Err(format!(
                "Contagem de parâmetros divergente ao gravar checkpoint ({} != {}).",
                written_params, expected_params
            ));
        }

        if let Some(opt) = optimizer {
            write_f32_slice_hashed(&mut w, &mut crc, &opt.m, "AdamW m")?;
            write_f32_slice_hashed(&mut w, &mut crc, &opt.v, "AdamW v")?;
        }

        let checksum = crc.finalize();
        w.write_all(&checksum.to_le_bytes())
            .map_err(|e| format!("Falha ao gravar CRC32 do checkpoint: {}", e))?;
        w.flush()
            .map_err(|e| format!("Falha no flush do checkpoint: {}", e))?;
        w.get_ref()
            .sync_all()
            .map_err(|e| format!("Falha no sync_all do checkpoint: {}", e))?;
        Ok(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }

    atomic_rename_with_retry(&tmp_path, path)
}

/// Carrega um checkpoint binário completo (`SMNN` v1 ou v2), validando limites de memória,
/// tamanho exato do arquivo (incluindo EOF), finitude dos tensores e checksum CRC32 (em v2).
pub fn load_checkpoint(
    path: &Path,
) -> Result<
    (
        ServerMindTransformer,
        Option<AdamWState>,
        CheckpointMetadata,
    ),
    String,
> {
    let file =
        File::open(path).map_err(|e| format!("Falha ao abrir checkpoint {:?}: {}", path, e))?;
    let file_len = file
        .metadata()
        .map_err(|e| format!("Falha ao ler metadados de arquivo {:?}: {}", path, e))?
        .len();
    if file_len < 12 {
        return Err("Arquivo de checkpoint inválido ou truncado (< 12 bytes).".to_string());
    }

    let mut r = BufReader::with_capacity(CHECKPOINT_IO_BUF_CAP, file);
    let mut crc = Crc32Hasher::new();

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .map_err(|e| format!("Falha ao ler cabeçalho do checkpoint: {}", e))?;
    if &magic != CHECKPOINT_MAGIC {
        return Err("Arquivo de checkpoint inválido (magic header incorreto).".to_string());
    }
    crc.update(&magic);

    let mut u32_buf = [0u8; 4];
    r.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let ver = u32::from_le_bytes(u32_buf);
    if ver != CHECKPOINT_VERSION_V1 && ver != CHECKPOINT_VERSION {
        return Err(format!("Versão de checkpoint não suportada: {}", ver));
    }
    crc.update(&u32_buf);

    r.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let meta_len = u32::from_le_bytes(u32_buf) as usize;
    if meta_len == 0 || meta_len > MAX_CHECKPOINT_META_BYTES {
        return Err(format!(
            "Tamanho de metadados inválido no checkpoint: {} bytes (limite {} bytes).",
            meta_len, MAX_CHECKPOINT_META_BYTES
        ));
    }
    if (12 + meta_len as u64) > file_len {
        return Err("Checkpoint truncado na seção de metadados.".to_string());
    }
    crc.update(&u32_buf);

    let mut meta_bytes = vec![0u8; meta_len];
    r.read_exact(&mut meta_bytes).map_err(|e| e.to_string())?;
    crc.update(&meta_bytes);

    let meta: CheckpointMetadata = serde_json::from_slice(&meta_bytes)
        .map_err(|e| format!("Metadados de checkpoint inválidos: {}", e))?;
    meta.config.validate()?;

    let total_params = meta
        .config
        .checked_total_params()
        .ok_or_else(|| "Overflow nas dimensões do modelo no checkpoint.".to_string())?;
    let float_factor = if meta.has_optimizer { 3usize } else { 1usize };
    let total_floats = total_params
        .checked_mul(float_factor)
        .ok_or_else(|| "Overflow na contagem de floats do checkpoint.".to_string())?;
    let payload_bytes = (total_floats as u64)
        .checked_mul(4)
        .ok_or_else(|| "Overflow no tamanho em bytes do checkpoint.".to_string())?;
    let trailer_bytes = if ver == CHECKPOINT_VERSION {
        4u64
    } else {
        0u64
    };
    let expected_file_len = 12u64
        .checked_add(meta_len as u64)
        .and_then(|x| x.checked_add(payload_bytes))
        .and_then(|x| x.checked_add(trailer_bytes))
        .ok_or_else(|| "Overflow no tamanho total esperado do checkpoint.".to_string())?;

    if file_len != expected_file_len {
        return Err(format!(
            "Tamanho do arquivo de checkpoint inconsistente (esperado {} bytes, encontrado {} bytes): arquivo truncado ou corrompido.",
            expected_file_len, file_len
        ));
    }

    let mut model = ServerMindTransformer::new_zeros(meta.config.clone())?;
    let use_crc = ver == CHECKPOINT_VERSION;
    let mut io_err: Option<String> = None;
    model.visit_param_slices_mut(|slice| {
        if io_err.is_some() {
            return;
        }
        let crc_opt = if use_crc { Some(&mut crc) } else { None };
        if let Err(e) = read_f32_slice_checked(&mut r, crc_opt, slice, "peso do modelo") {
            io_err = Some(e);
        }
    });
    if let Some(err) = io_err {
        return Err(err);
    }

    let optimizer = if meta.has_optimizer {
        let mut opt = AdamWState::new(&model);
        opt.step = meta.optimizer_step.unwrap_or(meta.step);
        read_f32_slice_checked(
            &mut r,
            if use_crc { Some(&mut crc) } else { None },
            &mut opt.m,
            "AdamW m",
        )?;
        read_f32_slice_checked(
            &mut r,
            if use_crc { Some(&mut crc) } else { None },
            &mut opt.v,
            "AdamW v",
        )?;
        Some(opt)
    } else {
        None
    };

    if ver == CHECKPOINT_VERSION {
        let mut crc_buf = [0u8; 4];
        r.read_exact(&mut crc_buf)
            .map_err(|e| format!("Falha ao ler CRC32 do checkpoint: {}", e))?;
        let stored_crc = u32::from_le_bytes(crc_buf);
        let computed_crc = crc.finalize();
        if stored_crc != computed_crc {
            return Err(format!(
                "Falha de integridade no checkpoint {:?}: CRC32 inválido (esperado {:08x}, calculado {:08x}).",
                path, stored_crc, computed_crc
            ));
        }
    }

    Ok((model, optimizer, meta))
}

pub(super) struct Crc32Hasher {
    state: u32,
}

impl Crc32Hasher {
    pub(super) fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }

    pub(super) fn update(&mut self, bytes: &[u8]) {
        static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
        let table = TABLE.get_or_init(|| {
            let mut t = [0u32; 256];
            for i in 0..256u32 {
                let mut c = i;
                for _ in 0..8 {
                    c = if c & 1 != 0 {
                        0xEDB8_8320 ^ (c >> 1)
                    } else {
                        c >> 1
                    };
                }
                t[i as usize] = c;
            }
            t
        });
        let mut s = self.state;
        for &b in bytes {
            let idx = ((s ^ (b as u32)) & 0xFF) as usize;
            s = table[idx] ^ (s >> 8);
        }
        self.state = s;
    }

    pub(super) fn finalize(&self) -> u32 {
        !self.state
    }
}

fn write_bytes_hashed(
    w: &mut BufWriter<File>,
    crc: &mut Crc32Hasher,
    bytes: &[u8],
) -> Result<(), String> {
    crc.update(bytes);
    w.write_all(bytes).map_err(|e| e.to_string())
}

fn write_f32_slice_hashed(
    w: &mut BufWriter<File>,
    crc: &mut Crc32Hasher,
    slice: &[f32],
    label: &str,
) -> Result<(), String> {
    const CHUNK_FLOATS: usize = 4096;
    let mut byte_buf = [0u8; CHUNK_FLOATS * 4];
    for chunk in slice.chunks(CHUNK_FLOATS) {
        let byte_len = chunk.len() * 4;
        for (i, &val) in chunk.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!(
                    "Tentativa bloqueada de gravar valor não-finito ({}) em {}.",
                    val, label
                ));
            }
            byte_buf[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
        }
        crc.update(&byte_buf[..byte_len]);
        w.write_all(&byte_buf[..byte_len])
            .map_err(|e| format!("Erro de I/O ao gravar {}: {}", label, e))?;
    }
    Ok(())
}

fn read_f32_slice_checked(
    r: &mut BufReader<File>,
    mut crc: Option<&mut Crc32Hasher>,
    slice: &mut [f32],
    label: &str,
) -> Result<(), String> {
    const CHUNK_FLOATS: usize = 4096;
    let mut byte_buf = [0u8; CHUNK_FLOATS * 4];
    for chunk in slice.chunks_mut(CHUNK_FLOATS) {
        let byte_len = chunk.len() * 4;
        r.read_exact(&mut byte_buf[..byte_len])
            .map_err(|e| format!("Falha ao ler {} do checkpoint: {}", label, e))?;
        if let Some(c) = crc.as_deref_mut() {
            c.update(&byte_buf[..byte_len]);
        }
        for (i, slot) in chunk.iter_mut().enumerate() {
            let val = f32::from_le_bytes([
                byte_buf[i * 4],
                byte_buf[i * 4 + 1],
                byte_buf[i * 4 + 2],
                byte_buf[i * 4 + 3],
            ]);
            if !val.is_finite() {
                return Err(format!(
                    "Checkpoint corrompido: valor não-finito ({}) detectado em {}.",
                    val, label
                ));
            }
            *slot = val;
        }
    }
    Ok(())
}

fn make_atomic_temp_path(parent: &Path, target: &Path) -> std::path::PathBuf {
    static ATOMIC_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let seq = ATOMIC_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let fname = target
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("checkpoint.bin");
    parent.join(format!(".{}.tmp.{}.{}", fname, std::process::id(), seq))
}

fn atomic_rename_with_retry(tmp_path: &Path, final_path: &Path) -> Result<(), String> {
    let mut last_err = None;
    for attempt in 0..3 {
        match fs::rename(tmp_path, final_path) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last_err = Some(e);
                if attempt < 2 {
                    std::thread::sleep(std::time::Duration::from_millis(15));
                }
            }
        }
    }
    let _ = fs::remove_file(tmp_path);
    Err(format!(
        "Falha ao substituir atomicamente {:?}: {}",
        final_path,
        last_err.map(|e| e.to_string()).unwrap_or_default()
    ))
}

// ---------------------------------------------------------------------------
// Primitivas matemáticas puras + MatMul multi-thread
// ---------------------------------------------------------------------------

fn rmsnorm_forward_into(
    x: &[f32],
    gamma: &[f32],
    out: &mut [f32],
    inv_rms: &mut [f32],
    t: usize,
    d: usize,
) {
    let eps = 1e-5f32;
    for pos in 0..t {
        let row = &x[pos * d..(pos + 1) * d];
        let ss = dot_f32(row, row);
        let ir = 1.0 / (ss / d as f32 + eps).sqrt();
        inv_rms[pos] = ir;
        let out_row = &mut out[pos * d..(pos + 1) * d];
        for i in 0..d {
            out_row[i] = row[i] * ir * gamma[i];
        }
    }
}

fn rmsnorm_forward(x: &[f32], gamma: &[f32], t: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; t * d];
    let mut inv_rms = vec![0.0f32; t];
    rmsnorm_forward_into(x, gamma, &mut out, &mut inv_rms, t, d);
    (out, inv_rms)
}

fn rmsnorm_backward_into(
    x: &[f32],
    gamma: &[f32],
    inv_rms: &[f32],
    d_out: &[f32],
    d_gamma: &mut [f32],
    d_x: &mut [f32],
    t: usize,
    d: usize,
) {
    let inv_d = 1.0 / d as f32;
    for pos in 0..t {
        let ir = inv_rms[pos];
        let x_row = &x[pos * d..(pos + 1) * d];
        let do_row = &d_out[pos * d..(pos + 1) * d];
        let dx_row = &mut d_x[pos * d..(pos + 1) * d];

        let mut dot = 0.0f32;
        for i in 0..d {
            d_gamma[i] += do_row[i] * x_row[i] * ir;
            dot += do_row[i] * gamma[i] * x_row[i];
        }
        let coeff = dot * ir * ir * ir * inv_d;
        for i in 0..d {
            dx_row[i] = do_row[i] * gamma[i] * ir - x_row[i] * coeff;
        }
    }
}

fn rmsnorm_backward(
    x: &[f32],
    gamma: &[f32],
    inv_rms: &[f32],
    d_out: &[f32],
    d_gamma: &mut [f32],
    t: usize,
    d: usize,
) -> Vec<f32> {
    let mut d_x = vec![0.0f32; t * d];
    rmsnorm_backward_into(x, gamma, inv_rms, d_out, d_gamma, &mut d_x, t, d);
    d_x
}

fn causal_mha_forward(
    ln1: &[f32],
    block: &TransformerBlock,
    t: usize,
    d: usize,
    n_heads: usize,
) -> Vec<f32> {
    let head_dim = d / n_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; t * d];
    let mut k = vec![0.0f32; t * d];
    let mut v = vec![0.0f32; t * d];
    matmul_nn(ln1, &block.wq, &mut q, t, d, d);
    matmul_nn(ln1, &block.wk, &mut k, t, d, d);
    matmul_nn(ln1, &block.wv, &mut v, t, d, d);

    let mut ctx = vec![0.0f32; t * d];
    let mut probs = vec![0.0f32; t];
    for h in 0..n_heads {
        let h_off = h * head_dim;
        for i in 0..t {
            let mut max_s = f32::NEG_INFINITY;
            let q_slice = &q[i * d + h_off..i * d + h_off + head_dim];
            for j in 0..=i {
                let k_slice = &k[j * d + h_off..j * d + h_off + head_dim];
                let s = dot_f32(q_slice, k_slice) * scale;
                probs[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut sum_e = 0.0f32;
            for j in 0..=i {
                let e = (probs[j] - max_s).exp();
                probs[j] = e;
                sum_e += e;
            }
            let inv = 1.0 / sum_e.max(1e-12);
            let ctx_slice = &mut ctx[i * d + h_off..i * d + h_off + head_dim];
            for j in 0..=i {
                let p = probs[j] * inv;
                let v_slice = &v[j * d + h_off..j * d + h_off + head_dim];
                axpy_f32(p, v_slice, ctx_slice);
            }
        }
    }

    let mut out = vec![0.0f32; t * d];
    matmul_nn(&ctx, &block.wo, &mut out, t, d, d);
    out
}

fn mlp_forward(ln2: &[f32], block: &TransformerBlock, t: usize, d: usize, ff: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; t * ff];
    matmul_nn(ln2, &block.w1, &mut h, t, d, ff);
    for pos in 0..t {
        for j in 0..ff {
            h[pos * ff + j] = gelu(h[pos * ff + j] + block.b1[j]);
        }
    }
    let mut out = vec![0.0f32; t * d];
    matmul_nn(&h, &block.w2, &mut out, t, ff, d);
    for pos in 0..t {
        for j in 0..d {
            out[pos * d + j] += block.b2[j];
        }
    }
    out
}

fn causal_mha_forward_quant(
    ln1: &[f32],
    qb: &quant::QuantizedBlock,
    t: usize,
    d: usize,
    n_heads: usize,
) -> Vec<f32> {
    let head_dim = d / n_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; t * d];
    let mut k = vec![0.0f32; t * d];
    let mut v = vec![0.0f32; t * d];
    qb.wq.matmul_nn(ln1, &mut q, t);
    qb.wk.matmul_nn(ln1, &mut k, t);
    qb.wv.matmul_nn(ln1, &mut v, t);

    let mut ctx = vec![0.0f32; t * d];
    let mut probs = vec![0.0f32; t];
    for h in 0..n_heads {
        let h_off = h * head_dim;
        for i in 0..t {
            let mut max_s = f32::NEG_INFINITY;
            let q_slice = &q[i * d + h_off..i * d + h_off + head_dim];
            for j in 0..=i {
                let k_slice = &k[j * d + h_off..j * d + h_off + head_dim];
                let s = dot_f32(q_slice, k_slice) * scale;
                probs[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut sum_e = 0.0f32;
            for j in 0..=i {
                let e = (probs[j] - max_s).exp();
                probs[j] = e;
                sum_e += e;
            }
            let inv = 1.0 / sum_e.max(1e-12);
            let ctx_slice = &mut ctx[i * d + h_off..i * d + h_off + head_dim];
            for j in 0..=i {
                let p = probs[j] * inv;
                let v_slice = &v[j * d + h_off..j * d + h_off + head_dim];
                axpy_f32(p, v_slice, ctx_slice);
            }
        }
    }

    let mut out = vec![0.0f32; t * d];
    qb.wo.matmul_nn(&ctx, &mut out, t);
    out
}

fn mlp_forward_quant(
    ln2: &[f32],
    block: &TransformerBlock,
    qb: &quant::QuantizedBlock,
    t: usize,
    d: usize,
    ff: usize,
) -> Vec<f32> {
    let mut h = vec![0.0f32; t * ff];
    qb.w1.matmul_nn(ln2, &mut h, t);
    for pos in 0..t {
        for j in 0..ff {
            h[pos * ff + j] = gelu(h[pos * ff + j] + block.b1[j]);
        }
    }
    let mut out = vec![0.0f32; t * d];
    qb.w2.matmul_nn(&h, &mut out, t);
    for pos in 0..t {
        for j in 0..d {
            out[pos * d + j] += block.b2[j];
        }
    }
    out
}

#[inline]
fn gelu(x: f32) -> f32 {
    if x >= 10.0 {
        return x;
    }
    if x <= -10.0 {
        return 0.0;
    }
    let k = 0.7978845608f32; // sqrt(2 / pi)
    let c = 0.044715f32;
    let inner = k * (x + c * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

#[inline]
fn gelu_grad(x: f32) -> f32 {
    if x >= 10.0 {
        return 1.0;
    }
    if x <= -10.0 {
        return 0.0;
    }
    let k = 0.7978845608f32;
    let c = 0.044715f32;
    let inner = k * (x + c * x * x * x);
    let t = inner.tanh();
    let sech2 = 1.0 - t * t;
    let d_inner = k * (1.0 + 3.0 * c * x * x);
    0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
}

fn argmax_finite(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, val)| val.is_finite())
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(idx, _)| idx)
}

fn sample_top_k(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    rng: &mut SplitMix64,
) -> Option<usize> {
    let mut pairs: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, l)| l.is_finite())
        .collect();
    if pairs.is_empty() {
        return None;
    }
    let k = top_k.max(1).min(pairs.len());
    if k < pairs.len() {
        pairs.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        pairs.truncate(k);
    }
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Subtrai o máximo antes de dividir por `temperature` para evitar overflow (+Inf -> NaN)
    // para qualquer `temperature > 0`, garantindo `probs[0] == 1.0` e `sum >= 1.0`.
    let max_raw = pairs[0].1;
    let probs: Vec<f32> = pairs
        .iter()
        .map(|(_, l)| ((*l - max_raw) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Some(pairs[0].0);
    }
    let r = rng.uniform_f32() * sum;
    let mut acc = 0.0f32;
    for ((idx, _), &p) in pairs.iter().zip(probs.iter()) {
        acc += p;
        if acc >= r {
            return Some(*idx);
        }
    }
    Some(pairs[0].0)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E3779B97F4A7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn uniform_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        u.clamp(1e-7, 1.0 - 1e-7)
    }

    fn normal_vec(&mut self, len: usize, std: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            let u1 = self.uniform_f32();
            let u2 = self.uniform_f32();
            let r = (-2.0 * u1.ln()).sqrt() * std;
            let theta = 2.0 * std::f32::consts::PI * u2;
            out.push(r * theta.cos());
            if out.len() < len {
                out.push(r * theta.sin());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{
        gelu, gelu_grad, load_checkpoint, matmul_nn, matmul_nt, matmul_tn, sample_top_k,
        save_checkpoint, AdamWState, ServerMindTransformer, SplitMix64, TransformerConfig,
        TransformerGrads,
    };
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_SEQ: AtomicU64 = AtomicU64::new(1);

    fn unique_temp_path(label: &str) -> std::path::PathBuf {
        let seq = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "sm_{}_{}_{}_{}.bin",
            label,
            std::process::id(),
            seq,
            nanos
        ))
    }

    #[test]
    fn analytical_backprop_matches_finite_differences() {
        // Modelo de 2 camadas para verificar propagação entre blocos, tokens repetidos,
        // mascaramento de `pad_id` e todos os 14 grupos de parâmetros.
        let cfg = TransformerConfig {
            vocab_size: 16,
            max_seq_len: 8,
            d_model: 8,
            n_heads: 2,
            n_layers: 2,
            d_ff: 16,
        };
        let mut model = ServerMindTransformer::new_random(cfg.clone(), 42).unwrap();
        // Token `1` repetido (testa `+=` em `token_emb`); `0` = pad_id e `99` = OOB em target.
        let input = [1usize, 5, 1, 7, 3];
        let target = [5usize, 0, 7, 99, 2];

        let mut grads = TransformerGrads::zeros_like(&cfg);
        let (loss, valid) = model.forward_and_backward(&input, &target, 0, &mut grads);
        assert_eq!(valid, 3);
        assert!(loss.is_finite() && loss > 0.0);

        // Consistência exata entre `forward_and_backward` e `training::evaluate_loss`
        let win = vec![1usize, 5, 1, 7, 3, 2];
        let mut g_win = TransformerGrads::zeros_like(&cfg);
        let (l_sum, n_val) = model.forward_and_backward(&win[..5], &win[1..], 0, &mut g_win);
        let eval_l = crate::training::evaluate_loss(&model, &[win], 0).unwrap();
        assert!(
            ((l_sum / n_val as f32) - eval_l).abs() < 1e-5,
            "train_loss={} vs eval_loss={}",
            l_sum / n_val as f32,
            eval_l
        );

        let eps = 5e-4f32;
        let atol = 1e-3f32;
        let rtol = 2e-2f32;
        let check_grad =
            |model: &mut ServerMindTransformer,
             name: &str,
             analytic: f32,
             mut getter: Box<dyn FnMut(&mut ServerMindTransformer) -> &mut f32>| {
                let orig = *getter(model);
                *getter(model) = orig + eps;
                let mut g_dummy = TransformerGrads::zeros_like(&model.config);
                let (l_plus, _) = model.forward_and_backward(&input, &target, 0, &mut g_dummy);
                *getter(model) = orig - eps;
                let (l_minus, _) = model.forward_and_backward(&input, &target, 0, &mut g_dummy);
                *getter(model) = orig;
                let numeric = (l_plus - l_minus) / (2.0 * eps);
                let diff = (analytic - numeric).abs();
                let bound = atol + rtol * numeric.abs().max(analytic.abs());
                assert!(
                    diff <= bound,
                    "grad mismatch em {}: analytic={} numeric={} diff={} bound={}",
                    name,
                    analytic,
                    numeric,
                    diff,
                    bound
                );
            };

        // Top-level: token_emb (repetido, único e não-usado), pos_emb, ln_f_gamma, lm_head
        check_grad(
            &mut model,
            "token_emb[tok=1]",
            grads.token_emb[1 * 8 + 3],
            Box::new(|m| &mut m.token_emb[1 * 8 + 3]),
        );
        check_grad(
            &mut model,
            "token_emb[tok=5]",
            grads.token_emb[5 * 8 + 1],
            Box::new(|m| &mut m.token_emb[5 * 8 + 1]),
        );
        assert_eq!(grads.token_emb[9 * 8 + 2], 0.0);
        check_grad(
            &mut model,
            "pos_emb[pos=2]",
            grads.pos_emb[2 * 8 + 4],
            Box::new(|m| &mut m.pos_emb[2 * 8 + 4]),
        );
        check_grad(
            &mut model,
            "ln_f_gamma[2]",
            grads.ln_f_gamma[2],
            Box::new(|m| &mut m.ln_f_gamma[2]),
        );
        check_grad(
            &mut model,
            "lm_head[5]",
            grads.lm_head[5],
            Box::new(|m| &mut m.lm_head[5]),
        );

        // Todos os 10 tensores internos em cada uma das 2 camadas
        for layer in 0..2 {
            let b_grad = grads.blocks[layer].clone();
            check_grad(
                &mut model,
                &format!("L{}.ln1_gamma", layer),
                b_grad.ln1_gamma[1],
                Box::new(move |m| &mut m.blocks[layer].ln1_gamma[1]),
            );
            check_grad(
                &mut model,
                &format!("L{}.wq", layer),
                b_grad.wq[4],
                Box::new(move |m| &mut m.blocks[layer].wq[4]),
            );
            check_grad(
                &mut model,
                &format!("L{}.wk", layer),
                b_grad.wk[7],
                Box::new(move |m| &mut m.blocks[layer].wk[7]),
            );
            check_grad(
                &mut model,
                &format!("L{}.wv", layer),
                b_grad.wv[2],
                Box::new(move |m| &mut m.blocks[layer].wv[2]),
            );
            check_grad(
                &mut model,
                &format!("L{}.wo", layer),
                b_grad.wo[9],
                Box::new(move |m| &mut m.blocks[layer].wo[9]),
            );
            check_grad(
                &mut model,
                &format!("L{}.ln2_gamma", layer),
                b_grad.ln2_gamma[3],
                Box::new(move |m| &mut m.blocks[layer].ln2_gamma[3]),
            );
            check_grad(
                &mut model,
                &format!("L{}.w1", layer),
                b_grad.w1[5],
                Box::new(move |m| &mut m.blocks[layer].w1[5]),
            );
            check_grad(
                &mut model,
                &format!("L{}.b1", layer),
                b_grad.b1[6],
                Box::new(move |m| &mut m.blocks[layer].b1[6]),
            );
            check_grad(
                &mut model,
                &format!("L{}.w2", layer),
                b_grad.w2[11],
                Box::new(move |m| &mut m.blocks[layer].w2[11]),
            );
            check_grad(
                &mut model,
                &format!("L{}.b2", layer),
                b_grad.b2[4],
                Box::new(move |m| &mut m.blocks[layer].b2[4]),
            );
        }
    }

    #[test]
    fn matmul_multithread_matches_singlethread_reference() {
        // m * k * n = 45 * 64 * 50 = 144_000 (>= 65_536), com dimensões não divisíveis por threads.
        let (m, k, n) = (45usize, 64usize, 50usize);
        let mut rng = SplitMix64::new(999);
        let a_nn = rng.normal_vec(m * k, 0.5);
        let b_nn = rng.normal_vec(k * n, 0.5);
        let mut c_mt = vec![0.25f32; m * n];
        let mut c_ref = c_mt.clone();
        matmul_nn(&a_nn, &b_nn, &mut c_mt, m, k, n);
        for i in 0..m {
            for kk in 0..k {
                let av = a_nn[i * k + kk];
                for j in 0..n {
                    c_ref[i * n + j] += av * b_nn[kk * n + j];
                }
            }
        }
        for (x, y) in c_mt.iter().zip(c_ref.iter()) {
            assert!((x - y).abs() < 1e-4, "matmul_nn diff: {} vs {}", x, y);
        }

        // matmul_tn: A[m, k]^T @ B[m, n] => [k, n]
        let mut c_tn_mt = vec![-0.1f32; k * n];
        let mut c_tn_ref = c_tn_mt.clone();
        matmul_tn(&a_nn, &c_ref, &mut c_tn_mt, k, m, n);
        for row_m in 0..m {
            for i in 0..k {
                let av = a_nn[row_m * k + i];
                for j in 0..n {
                    c_tn_ref[i * n + j] += av * c_ref[row_m * n + j];
                }
            }
        }
        for (x, y) in c_tn_mt.iter().zip(c_tn_ref.iter()) {
            assert!((x - y).abs() < 1e-3, "matmul_tn diff: {} vs {}", x, y);
        }

        // matmul_nt: A[m, n] @ B[k, n]^T => [m, k]
        let mut c_nt_mt = vec![0.05f32; m * k];
        let mut c_nt_ref = c_nt_mt.clone();
        matmul_nt(&c_ref, &b_nn, &mut c_nt_mt, m, n, k);
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0f32;
                for col in 0..n {
                    sum += c_ref[i * n + col] * b_nn[j * n + col];
                }
                c_nt_ref[i * k + j] += sum;
            }
        }
        for (x, y) in c_nt_mt.iter().zip(c_nt_ref.iter()) {
            assert!((x - y).abs() < 1e-3, "matmul_nt diff: {} vs {}", x, y);
        }
    }

    #[test]
    fn adamw_rejects_nan_and_inf_without_corrupting_weights() {
        let cfg = TransformerConfig {
            vocab_size: 8,
            max_seq_len: 4,
            d_model: 4,
            n_heads: 2,
            n_layers: 1,
            d_ff: 8,
        };
        let mut model = ServerMindTransformer::new_random(cfg.clone(), 7).unwrap();
        let orig_tok = model.token_emb.clone();
        let orig_head = model.lm_head.clone();
        let mut opt = AdamWState::new(&model);

        let mut nan_grads = TransformerGrads::zeros_like(&cfg);
        nan_grads.lm_head[0] = f32::NAN;
        assert!(!opt.step(&mut model, &mut nan_grads, 1e-3, 0.01, 1.0));
        assert_eq!(opt.step, 0);
        assert_eq!(model.token_emb, orig_tok);
        assert_eq!(model.lm_head, orig_head);

        let mut inf_grads = TransformerGrads::zeros_like(&cfg);
        inf_grads.blocks[0].wq[1] = f32::INFINITY;
        assert!(!opt.step(&mut model, &mut inf_grads, 1e-3, 0.01, 1.0));
        assert_eq!(opt.step, 0);
        assert_eq!(model.lm_head, orig_head);
    }

    #[test]
    fn generate_sample_top_k_and_gelu_handle_edge_cases_safely() {
        let mut bad_cfg = TransformerConfig {
            vocab_size: 8,
            max_seq_len: 4,
            d_model: 4,
            n_heads: 2,
            n_layers: 1,
            d_ff: 0,
        };
        assert!(bad_cfg.validate().is_err());
        bad_cfg.d_ff = 8;
        let model = ServerMindTransformer::new_random(bad_cfg, 11).unwrap();

        // Prompt vazio não pode entrar em panic
        assert!(model.generate(&[], 5, 0.8, 4, 2, 42).is_empty());
        // Greedy e temperatura extrema produzem IDs válidos
        let greedy = model.generate(&[1, 3], 4, 0.0, 4, 99, 42);
        let tiny_temp = model.generate(&[1, 3], 4, 1e-30, 4, 99, 42);
        assert_eq!(greedy, tiny_temp);

        // `sample_top_k` ignora NaN/+Inf e nunca entra em overflow com temperatura minúscula
        let mut rng = SplitMix64::new(55);
        let logits_with_nan = [f32::NAN, -1.0, 2.5, f32::INFINITY, 0.5];
        for _ in 0..20 {
            let idx = sample_top_k(&logits_with_nan, 1e-20, 3, &mut rng).unwrap();
            assert_eq!(idx, 2);
        }
        assert!(sample_top_k(&[f32::NAN, f32::INFINITY], 1.0, 2, &mut rng).is_none());

        // `gelu` e `gelu_grad` não geram NaN em magnitudes extremas
        assert!(gelu(1e20).is_finite());
        assert_eq!(gelu(-1e20), 0.0);
        assert_eq!(gelu_grad(1e20), 1.0);
        assert_eq!(gelu_grad(-1e20), 0.0);
    }

    #[test]
    fn checkpoint_saves_and_restores_weights_and_optimizer_including_infinite_loss() {
        let cfg = TransformerConfig {
            vocab_size: 32,
            max_seq_len: 16,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
        };
        let model = ServerMindTransformer::new_random(cfg, 123).unwrap();
        let opt = AdamWState::new(&model);
        let tmp = unique_temp_path("ckpt_finite");
        save_checkpoint(&tmp, &model, Some(&opt), 17, 1.42).unwrap();

        let (loaded, loaded_opt, meta) = load_checkpoint(&tmp).unwrap();
        assert_eq!(meta.step, 17);
        assert!((meta.best_val_loss - 1.42).abs() < 1e-6);
        assert_eq!(model.token_emb, loaded.token_emb);
        assert_eq!(model.pos_emb, loaded.pos_emb);
        assert_eq!(model.ln_f_gamma, loaded.ln_f_gamma);
        assert_eq!(model.lm_head, loaded.lm_head);
        assert!(loaded_opt.is_some());
        let _ = std::fs::remove_file(&tmp);

        // Testa `best_val_loss = f32::INFINITY` (serializado como `null` em JSON)
        let tmp_inf = unique_temp_path("ckpt_inf");
        save_checkpoint(&tmp_inf, &model, None, 0, f32::INFINITY).unwrap();
        let (_, opt_none, meta_inf) = load_checkpoint(&tmp_inf).unwrap();
        assert!(opt_none.is_none());
        assert!(meta_inf.best_val_loss.is_infinite());
        let _ = std::fs::remove_file(&tmp_inf);
    }

    #[test]
    fn kv_cache_matches_reference_forward_and_generation() {
        let cfg = TransformerConfig {
            vocab_size: 48,
            max_seq_len: 16,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            d_ff: 64,
        };
        let model = ServerMindTransformer::new_random(cfg.clone(), 2026).unwrap();
        let mut cache = super::KvCache::new(&cfg);
        assert_eq!(cache.memory_bytes(), 2 * 2 * 16 * 32 * 4);

        // 1. Equivalência numérica exata entre `forward_last_logits_cached` (prefill + passo a passo)
        // e a última linha de `forward_logits` de referência.
        let mut seq = vec![1usize, 5, 9, 14];
        let mut cached_logits = model.forward_last_logits_cached(&seq, &mut cache);
        for &next_tok in &[3usize, 18, 7, 22, 11] {
            let ref_all = model.forward_logits(&seq);
            let ref_last = &ref_all[(seq.len() - 1) * cfg.vocab_size..];
            for (a, b) in cached_logits.iter().zip(ref_last.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "divergência KV Cache vs referência: {} vs {}",
                    a,
                    b
                );
            }
            seq.push(next_tok);
            cached_logits = model.forward_last_logits_cached(&[next_tok], &mut cache);
        }

        // 2. Equivalência de geração greedy e top-k (incluindo janela deslizante além de `max_seq_len = 16`)
        let prompt = [2usize, 6, 10, 15];
        let greedy_kv = model.generate(&prompt, 18, 0.0, 4, 999, 77);
        let greedy_ref = model.generate_reference(&prompt, 18, 0.0, 4, 999, 77);
        assert_eq!(greedy_kv, greedy_ref);

        let topk_kv = model.generate(&prompt, 18, 0.8, 6, 999, 77);
        let topk_ref = model.generate_reference(&prompt, 18, 0.8, 6, 999, 77);
        assert_eq!(topk_kv, topk_ref);

        // 3. Parada imediata em `stop_tokens` secundário
        let target_stop = greedy_kv[prompt.len() + 2];
        let stopped =
            model.generate_with_options(&prompt, 18, 0.0, 4, &[999, target_stop], 77, None);
        assert!(stopped.len() <= prompt.len() + 3);
        assert_eq!(*stopped.last().unwrap(), target_stop);
    }

    #[test]
    #[ignore]
    fn benchmark_kv_cache_vs_reference_14m_model() {
        let cfg = TransformerConfig {
            vocab_size: 1024,
            max_seq_len: 128,
            d_model: 192,
            n_heads: 6,
            n_layers: 3,
            d_ff: 512,
        };
        let model = ServerMindTransformer::new_random(cfg, 2026).unwrap();
        let prompt: Vec<usize> = (1..=32).collect();

        let t0 = std::time::Instant::now();
        let out_ref = model.generate_reference(&prompt, 80, 0.0, 8, 9999, 1);
        let ms_ref = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let out_kv = model.generate(&prompt, 80, 0.0, 8, 9999, 1);
        let ms_kv = t1.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(out_ref, out_kv);
        println!(
            "BENCH_KV_80_TOK: ref={:.2} ms ({:.1} tok/s) | kv={:.2} ms ({:.1} tok/s) | speedup={:.2}x",
            ms_ref,
            80_000.0 / ms_ref,
            ms_kv,
            80_000.0 / ms_kv,
            ms_ref / ms_kv.max(1e-6)
        );
    }

    fn write_legacy_v1_checkpoint(
        path: &std::path::Path,
        model: &ServerMindTransformer,
        opt: Option<&AdamWState>,
        step: usize,
        best_val_loss: f32,
    ) {
        use std::io::Write;
        let meta = serde_json::json!({
            "config": model.config,
            "step": step,
            "best_val_loss": best_val_loss,
            "has_optimizer": opt.is_some(),
        });
        let meta_bytes = serde_json::to_vec(&meta).unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut w = std::io::BufWriter::new(file);
        w.write_all(b"SMNN").unwrap();
        w.write_all(&1u32.to_le_bytes()).unwrap();
        w.write_all(&(meta_bytes.len() as u32).to_le_bytes())
            .unwrap();
        w.write_all(&meta_bytes).unwrap();
        model.visit_param_slices(|slice| {
            for &val in slice {
                w.write_all(&val.to_le_bytes()).unwrap();
            }
        });
        if let Some(o) = opt {
            for &val in &o.m {
                w.write_all(&val.to_le_bytes()).unwrap();
            }
            for &val in &o.v {
                w.write_all(&val.to_le_bytes()).unwrap();
            }
        }
        w.flush().unwrap();
    }

    #[test]
    fn checkpoint_v2_detects_all_corruption_classes_and_reads_legacy_v1() {
        let cfg = TransformerConfig {
            vocab_size: 24,
            max_seq_len: 12,
            d_model: 16,
            n_heads: 2,
            n_layers: 2,
            d_ff: 32,
        };
        let model = ServerMindTransformer::new_random(cfg.clone(), 321).unwrap();
        let mut opt = AdamWState::new(&model);
        opt.step = 9;
        opt.m[0] = 0.125;
        opt.v[0] = 0.0625;

        // Round-trip v2 preservando `opt.step != step` e substituição atômica segura
        let p_valid = unique_temp_path("v2_valid");
        save_checkpoint(&p_valid, &model, Some(&opt), 15, 2.5).unwrap();
        let (loaded, loaded_opt, meta) = load_checkpoint(&p_valid).unwrap();
        assert_eq!(meta.step, 15);
        assert_eq!(meta.optimizer_step, Some(9));
        let lo = loaded_opt.unwrap();
        assert_eq!(lo.step, 9);
        assert_eq!(lo.m, opt.m);
        assert_eq!(lo.v, opt.v);
        assert_eq!(loaded.blocks[1].w2, model.blocks[1].w2);

        // Se uma gravação subsequente falhar (ex.: NaN em peso), o checkpoint válido anterior permanece intacto!
        let mut bad_nan_model = model.clone();
        bad_nan_model.lm_head[0] = f32::NAN;
        assert!(save_checkpoint(&p_valid, &bad_nan_model, None, 16, 2.0).is_err());
        let (_, _, preserved_meta) = load_checkpoint(&p_valid).unwrap();
        assert_eq!(preserved_meta.step, 15);

        let valid_bytes = std::fs::read(&p_valid).unwrap();
        let _ = std::fs::remove_file(&p_valid);

        // 1. Magic inválido
        let p1 = unique_temp_path("bad_magic");
        let mut b1 = valid_bytes.clone();
        b1[0..4].copy_from_slice(b"XXXX");
        std::fs::write(&p1, &b1).unwrap();
        assert!(load_checkpoint(&p1).is_err());
        let _ = std::fs::remove_file(&p1);

        // 2. Versão desconhecida (99)
        let p2 = unique_temp_path("bad_ver");
        let mut b2 = valid_bytes.clone();
        b2[4..8].copy_from_slice(&99u32.to_le_bytes());
        std::fs::write(&p2, &b2).unwrap();
        assert!(load_checkpoint(&p2).is_err());
        let _ = std::fs::remove_file(&p2);

        // 3. `meta_len = u32::MAX` (rejeitado imediatamente sem OOM)
        let p3 = unique_temp_path("huge_meta");
        let mut b3 = valid_bytes[..32].to_vec();
        b3[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&p3, &b3).unwrap();
        assert!(load_checkpoint(&p3).is_err());
        let _ = std::fs::remove_file(&p3);

        // 4. Truncado no meio dos pesos
        let p4 = unique_temp_path("truncated");
        std::fs::write(&p4, &valid_bytes[..valid_bytes.len() - 20]).unwrap();
        assert!(load_checkpoint(&p4).is_err());
        let _ = std::fs::remove_file(&p4);

        // 5. Bytes excedentes no fim (EOF sujo)
        let p5 = unique_temp_path("trailing");
        let mut b5 = valid_bytes.clone();
        b5.extend_from_slice(&[0u8; 8]);
        std::fs::write(&p5, &b5).unwrap();
        assert!(load_checkpoint(&p5).is_err());
        let _ = std::fs::remove_file(&p5);

        // 6. Bit adulterado em um peso (mesmo tamanho, detectado pelo CRC32)
        let p6 = unique_temp_path("bitflip");
        let mut b6 = valid_bytes.clone();
        let mid = b6.len() / 2;
        b6[mid] ^= 0x04;
        std::fs::write(&p6, &b6).unwrap();
        let err_crc = load_checkpoint(&p6).unwrap_err();
        assert!(
            err_crc.contains("CRC32"),
            "esperado erro CRC32, obtido: {}",
            err_crc
        );
        let _ = std::fs::remove_file(&p6);

        // 7. Retrocompatibilidade SMNN v1 + rejeição de `NaN` em v1
        let pv1 = unique_temp_path("legacy_v1");
        write_legacy_v1_checkpoint(&pv1, &model, Some(&opt), 11, 3.14);
        let (m_v1, opt_v1, meta_v1) = load_checkpoint(&pv1).unwrap();
        assert_eq!(meta_v1.step, 11);
        assert_eq!(opt_v1.unwrap().step, 11);
        assert_eq!(m_v1.token_emb, model.token_emb);

        let mut nan_v1_model = model.clone();
        nan_v1_model.token_emb[2] = f32::NAN;
        write_legacy_v1_checkpoint(&pv1, &nan_v1_model, None, 11, 3.14);
        assert!(load_checkpoint(&pv1).is_err());
        let _ = std::fs::remove_file(&pv1);
    }

    #[test]
    #[ignore]
    fn benchmark_checkpoint_io_v1_vs_v2() {
        let cfg = TransformerConfig {
            vocab_size: 1024,
            max_seq_len: 96,
            d_model: 192,
            n_heads: 6,
            n_layers: 3,
            d_ff: 512,
        };
        let model = ServerMindTransformer::new_random(cfg, 2026).unwrap();
        let opt = AdamWState::new(&model);
        let p_v1 = unique_temp_path("bench_v1");
        let p_v2 = unique_temp_path("bench_v2");

        let t0 = std::time::Instant::now();
        write_legacy_v1_checkpoint(&p_v1, &model, Some(&opt), 100, 4.0);
        let save_v1_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        // Simula load antigo (`new_random` + leitura v1)
        let _ = ServerMindTransformer::new_random(model.config.clone(), 0).unwrap();
        let _ = load_checkpoint(&p_v1).unwrap();
        let load_v1_sim_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = std::time::Instant::now();
        save_checkpoint(&p_v2, &model, Some(&opt), 100, 4.0).unwrap();
        let save_v2_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let t3 = std::time::Instant::now();
        let _ = load_checkpoint(&p_v2).unwrap();
        let load_v2_ms = t3.elapsed().as_secs_f64() * 1000.0;

        let size_mb = std::fs::metadata(&p_v2).unwrap().len() as f64 / (1024.0 * 1024.0);
        let _ = std::fs::remove_file(&p_v1);
        let _ = std::fs::remove_file(&p_v2);
        println!(
            "BENCH_CKPT ({:.2} MB): v1_save={:.2} ms | v2_save_atomic_crc={:.2} ms | v1_load_with_rng={:.2} ms | v2_load_new_zeros_crc={:.2} ms",
            size_mb, save_v1_ms, save_v2_ms, load_v1_sim_ms, load_v2_ms
        );
    }

    #[test]
    fn ops_simd_and_pool_match_reference_on_prime_and_gemv_shapes() {
        use super::ops::{
            axpy_f32, dot_f32, matmul_nn, matmul_nn_reference, matmul_nt, matmul_nt_reference,
            matmul_tn, matmul_tn_reference,
        };

        let mut rng = SplitMix64::new(202605);

        // Testa primitivas vetoriais em comprimentos primos/não-alinhados
        for len in [0usize, 1, 7, 8, 15, 16, 31, 32, 33, 97] {
            let x = rng.normal_vec(len, 0.7);
            let y = rng.normal_vec(len, 0.7);
            let d_opt = dot_f32(&x, &y);
            let d_ref: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            assert!(
                (d_opt - d_ref).abs() <= 1e-4 * (1.0 + d_ref.abs()),
                "dot_f32 len={} diff: {} vs {}",
                len,
                d_opt,
                d_ref
            );

            let mut y_opt = y.clone();
            let mut y_ref = y.clone();
            axpy_f32(-0.37, &x, &mut y_opt);
            for i in 0..len {
                y_ref[i] += -0.37 * x[i];
            }
            for i in 0..len {
                assert!((y_opt[i] - y_ref[i]).abs() <= 1e-5);
            }
        }

        // Formas cobrindo: escalar, GEMV single-thread, GEMV multi-thread (1x263x1031),
        // bordas de tile (6x128x256) e GEMM multi-thread primo (53x137x139).
        let shapes = [
            (1usize, 1usize, 1usize),
            (1, 37, 53),
            (1, 263, 1031),
            (3, 15, 31),
            (6, 128, 256),
            (7, 65, 129),
            (13, 131, 263),
            (53, 137, 139),
        ];

        for (m, k, n) in shapes {
            let a = rng.normal_vec(m * k, 0.5);
            let b = rng.normal_vec(k * n, 0.5);
            let c_init = rng.normal_vec(m * n, 0.2);

            // 1. matmul_nn
            let mut c_opt = c_init.clone();
            let mut c_ref = c_init.clone();
            matmul_nn(&a, &b, &mut c_opt, m, k, n);
            matmul_nn_reference(&a, &b, &mut c_ref, m, k, n);
            for idx in 0..(m * n) {
                let diff = (c_opt[idx] - c_ref[idx]).abs();
                let tol = 1e-4 + 1e-4 * c_ref[idx].abs();
                assert!(
                    diff <= tol,
                    "matmul_nn ({},{},{}) idx={} diff={}: opt={} ref={}",
                    m,
                    k,
                    n,
                    idx,
                    diff,
                    c_opt[idx],
                    c_ref[idx]
                );
            }

            // 2. matmul_tn: A[m, k]^T @ B_tn[m, n] => C_tn[k, n]
            let b_tn = rng.normal_vec(m * n, 0.5);
            let c_tn_init = rng.normal_vec(k * n, 0.2);
            let mut c_tn_opt = c_tn_init.clone();
            let mut c_tn_ref = c_tn_init;
            matmul_tn(&a, &b_tn, &mut c_tn_opt, k, m, n);
            matmul_tn_reference(&a, &b_tn, &mut c_tn_ref, k, m, n);
            for idx in 0..(k * n) {
                let diff = (c_tn_opt[idx] - c_tn_ref[idx]).abs();
                let tol = 1e-4 + 1e-4 * c_tn_ref[idx].abs();
                assert!(
                    diff <= tol,
                    "matmul_tn ({},{},{}) idx={} diff={}: opt={} ref={}",
                    k,
                    m,
                    n,
                    idx,
                    diff,
                    c_tn_opt[idx],
                    c_tn_ref[idx]
                );
            }

            // 3. matmul_nt: A_nt[m, n] @ B_nt[k, n]^T => C_nt[m, k]
            let a_nt = rng.normal_vec(m * n, 0.5);
            let b_nt = rng.normal_vec(k * n, 0.5);
            let c_nt_init = rng.normal_vec(m * k, 0.2);
            let mut c_nt_opt = c_nt_init.clone();
            let mut c_nt_ref = c_nt_init;
            matmul_nt(&a_nt, &b_nt, &mut c_nt_opt, m, n, k);
            matmul_nt_reference(&a_nt, &b_nt, &mut c_nt_ref, m, n, k);
            for idx in 0..(m * k) {
                let diff = (c_nt_opt[idx] - c_nt_ref[idx]).abs();
                let tol = 1e-4 + 1e-4 * c_nt_ref[idx].abs();
                assert!(
                    diff <= tol,
                    "matmul_nt ({},{},{}) idx={} diff={}: opt={} ref={}",
                    m,
                    n,
                    k,
                    idx,
                    diff,
                    c_nt_opt[idx],
                    c_nt_ref[idx]
                );
            }
        }
    }

    #[test]
    fn training_workspace_matches_reference_across_variable_length_reuses() {
        let cfg = TransformerConfig {
            vocab_size: 67,
            max_seq_len: 33,
            d_model: 48,
            n_heads: 6,
            n_layers: 2,
            d_ff: 131,
        };
        let model = ServerMindTransformer::new_random(cfg.clone(), 202606).unwrap();
        let mut ws = super::TrainingWorkspace::new(&cfg);
        let mem_before = ws.memory_bytes();
        assert!(mem_before > 0);

        let mut grads_ws = TransformerGrads::zeros_like(&cfg);
        let mut grads_ref = TransformerGrads::zeros_like(&cfg);

        // Amostra 1: janela cheia (`t = 33`), seguida de Amostra 2 menor (`t = 19` com `pad_id = 0`)
        // reutilizando o MESMO `TrainingWorkspace` para detectar qualquer vazamento de estado residual.
        let in1: Vec<usize> = (0..33).map(|i| (i * 5 + 1) % 67).collect();
        let tg1: Vec<usize> = (0..33).map(|i| (i * 7 + 3) % 67).collect();
        let in2: Vec<usize> = (0..19).map(|i| (i * 11 + 2) % 67).collect();
        let mut tg2: Vec<usize> = (0..19).map(|i| (i * 3 + 1) % 67).collect();
        tg2[5] = 0; // token de padding ignorado

        let (loss1_ws, tok1_ws) =
            model.forward_and_backward_with_workspace(&in1, &tg1, 0, &mut grads_ws, &mut ws);
        let (loss1_ref, tok1_ref) =
            model.forward_and_backward_reference(&in1, &tg1, 0, &mut grads_ref);
        assert_eq!(tok1_ws, tok1_ref);
        assert!((loss1_ws - loss1_ref).abs() <= 1e-3);

        let (loss2_ws, tok2_ws) =
            model.forward_and_backward_with_workspace(&in2, &tg2, 0, &mut grads_ws, &mut ws);
        let (loss2_ref, tok2_ref) =
            model.forward_and_backward_reference(&in2, &tg2, 0, &mut grads_ref);
        assert_eq!(tok2_ws, tok2_ref);
        assert!((loss2_ws - loss2_ref).abs() <= 1e-3);
        assert_eq!(ws.memory_bytes(), mem_before);

        let flatten_grads = |g: &TransformerGrads| -> Vec<f32> {
            let mut out = Vec::new();
            out.extend_from_slice(&g.token_emb);
            out.extend_from_slice(&g.pos_emb);
            for b in &g.blocks {
                out.extend_from_slice(&b.ln1_gamma);
                out.extend_from_slice(&b.wq);
                out.extend_from_slice(&b.wk);
                out.extend_from_slice(&b.wv);
                out.extend_from_slice(&b.wo);
                out.extend_from_slice(&b.ln2_gamma);
                out.extend_from_slice(&b.w1);
                out.extend_from_slice(&b.b1);
                out.extend_from_slice(&b.w2);
                out.extend_from_slice(&b.b2);
            }
            out.extend_from_slice(&g.ln_f_gamma);
            out.extend_from_slice(&g.lm_head);
            out
        };
        let flat_ws = flatten_grads(&grads_ws);
        let flat_ref = flatten_grads(&grads_ref);
        assert_eq!(flat_ws.len(), flat_ref.len());
        for (idx, (&gw, &gr)) in flat_ws.iter().zip(flat_ref.iter()).enumerate() {
            let diff = (gw - gr).abs();
            let tol = 1e-4 + 1e-3 * gr.abs();
            assert!(
                diff <= tol,
                "grad mismatch at idx {}: ws={} vs ref={} (diff={})",
                idx,
                gw,
                gr,
                diff
            );
        }
    }

    #[test]
    #[ignore]
    fn benchmark_sm005_gemm_and_workspace_vs_reference() {
        use super::ops::{
            matmul_nn, matmul_nn_reference, matmul_nt, matmul_nt_reference, matmul_tn,
            matmul_tn_reference, simd_backend_name, worker_threads,
        };

        let (m, k, n) = (256usize, 384usize, 1536usize);
        let flops_per_call = 2.0 * (m as f64) * (k as f64) * (n as f64);
        let mut rng = SplitMix64::new(77);
        let a = rng.normal_vec(m * k, 0.5);
        let b = rng.normal_vec(k * n, 0.5);
        let mut c = vec![0.0f32; m * n];

        // Warmup pool
        matmul_nn(&a, &b, &mut c, m, k, n);

        let iters = 10usize;

        let t0 = std::time::Instant::now();
        for _ in 0..2 {
            matmul_nn_reference(&a, &b, &mut c, m, k, n);
        }
        let nn_ref_gf = (2.0 * flops_per_call / t0.elapsed().as_secs_f64()) / 1e9;

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            matmul_nn(&a, &b, &mut c, m, k, n);
        }
        let nn_opt_gf = ((iters as f64) * flops_per_call / t1.elapsed().as_secs_f64()) / 1e9;

        let mut c_tn = vec![0.0f32; k * n];
        let t2 = std::time::Instant::now();
        for _ in 0..2 {
            matmul_tn_reference(&a, &c, &mut c_tn, k, m, n);
        }
        let tn_ref_gf = (2.0 * flops_per_call / t2.elapsed().as_secs_f64()) / 1e9;

        let t3 = std::time::Instant::now();
        for _ in 0..iters {
            matmul_tn(&a, &c, &mut c_tn, k, m, n);
        }
        let tn_opt_gf = ((iters as f64) * flops_per_call / t3.elapsed().as_secs_f64()) / 1e9;

        let mut c_nt = vec![0.0f32; m * k];
        let t4 = std::time::Instant::now();
        for _ in 0..2 {
            matmul_nt_reference(&c, &b, &mut c_nt, m, n, k);
        }
        let nt_ref_gf = (2.0 * flops_per_call / t4.elapsed().as_secs_f64()) / 1e9;

        let t5 = std::time::Instant::now();
        for _ in 0..iters {
            matmul_nt(&c, &b, &mut c_nt, m, n, k);
        }
        let nt_opt_gf = ((iters as f64) * flops_per_call / t5.elapsed().as_secs_f64()) / 1e9;

        println!(
            "BENCH_SM005_GEMM (256x384x1536, threads={}, simd={}):\n  matmul_nn: ref={:.2} GFLOP/s -> opt={:.2} GFLOP/s ({:.2}x)\n  matmul_tn: ref={:.2} GFLOP/s -> opt={:.2} GFLOP/s ({:.2}x)\n  matmul_nt: ref={:.2} GFLOP/s -> opt={:.2} GFLOP/s ({:.2}x)",
            worker_threads(),
            simd_backend_name(),
            nn_ref_gf,
            nn_opt_gf,
            nn_opt_gf / nn_ref_gf.max(1e-6),
            tn_ref_gf,
            tn_opt_gf,
            tn_opt_gf / tn_ref_gf.max(1e-6),
            nt_ref_gf,
            nt_opt_gf,
            nt_opt_gf / nt_ref_gf.max(1e-6),
        );

        let cfg = TransformerConfig {
            vocab_size: 1024,
            max_seq_len: 96,
            d_model: 192,
            n_heads: 6,
            n_layers: 3,
            d_ff: 512,
        };
        let model = ServerMindTransformer::new_random(cfg.clone(), 2026).unwrap();
        let mut ws = super::TrainingWorkspace::new(&cfg);
        let mut grads = TransformerGrads::zeros_like(&cfg);
        let inp: Vec<usize> = (1..=96).collect();
        let tgt: Vec<usize> = (2..=97).collect();

        let tr0 = std::time::Instant::now();
        for _ in 0..4 {
            let _ = model.forward_and_backward_reference(&inp, &tgt, 0, &mut grads);
        }
        let ref_ms = tr0.elapsed().as_secs_f64() * 1000.0 / 4.0;

        let tr1 = std::time::Instant::now();
        for _ in 0..16 {
            let _ = model.forward_and_backward_with_workspace(&inp, &tgt, 0, &mut grads, &mut ws);
        }
        let opt_ms = tr1.elapsed().as_secs_f64() * 1000.0 / 16.0;

        println!(
            "BENCH_SM005_STEP (1.45M, T=96, ws={:.2} MB): ref_scalar={:.2} ms/sample ({:.1} tok/s) -> opt_ws={:.2} ms/sample ({:.1} tok/s) | speedup={:.2}x",
            ws.memory_bytes() as f64 / (1024.0 * 1024.0),
            ref_ms,
            96_000.0 / ref_ms,
            opt_ms,
            96_000.0 / opt_ms,
            ref_ms / opt_ms.max(1e-6),
        );
        std::hint::black_box(&grads);
    }

    #[test]
    fn spirv_modules_are_valid_and_compute_device_parses() {
        use super::spirv::{validate_spirv_header, NeuralSpirvKernels};
        let k = NeuralSpirvKernels::build_all();
        for (name, words) in [
            ("fill_f32", &k.fill_f32),
            ("vec_add_inplace", &k.vec_add_inplace),
            ("embed_fwd", &k.embed_fwd),
            ("embed_bwd", &k.embed_bwd),
            ("rmsnorm_fwd", &k.rmsnorm_fwd),
            ("rmsnorm_bwd_dx", &k.rmsnorm_bwd_dx),
            ("rmsnorm_bwd_dgamma", &k.rmsnorm_bwd_dgamma),
            ("matmul_nn_tiled", &k.matmul_nn_tiled),
            ("matmul_tn_tiled", &k.matmul_tn_tiled),
            ("matmul_nt_tiled", &k.matmul_nt_tiled),
            ("causal_mha_fwd", &k.causal_mha_fwd),
            ("causal_mha_bwd_q_ds", &k.causal_mha_bwd_q_ds),
            ("causal_mha_bwd_kv", &k.causal_mha_bwd_kv),
            ("bias_gelu_fwd", &k.bias_gelu_fwd),
            ("gelu_bwd", &k.gelu_bwd),
            ("bias_res_add_fwd", &k.bias_res_add_fwd),
            ("bias_grad_sum", &k.bias_grad_sum),
            ("cross_entropy_fwd_bwd", &k.cross_entropy_fwd_bwd),
            ("reduce_sum_f32_to_io", &k.reduce_sum_f32_to_io),
            ("scale_and_grad_sq", &k.scale_and_grad_sq),
            ("adamw_step", &k.adamw_step),
        ] {
            validate_spirv_header(words)
                .unwrap_or_else(|e| panic!("SPIR-V '{}' inválido: {}", name, e));
        }

        assert_eq!(
            super::ComputeDevice::parse("auto").unwrap(),
            super::ComputeDevice::Auto
        );
        assert_eq!(
            super::ComputeDevice::parse("CPU").unwrap(),
            super::ComputeDevice::Cpu
        );
        assert_eq!(
            super::ComputeDevice::parse("gpu").unwrap(),
            super::ComputeDevice::Gpu
        );
        assert_eq!(
            super::ComputeDevice::parse("vulkan").unwrap(),
            super::ComputeDevice::Gpu
        );
        assert!(super::ComputeDevice::parse("cuda").is_err());
    }

    #[test]
    fn vulkan_gpu_session_matches_cpu_reference_forward_backward_and_adamw() {
        let Ok(dev) = super::VulkanDevice::global() else {
            // Permite execução limpa em máquinas/CI sem runtime Vulkan instalado.
            return;
        };
        assert!(!dev.info.device_name.is_empty());

        // Dimensões propositalmente não múltiplas de 16 ou 64 (`V=53, T=21, d=36, H=6, L=2, ff=75`)
        // para estressar todas as máscaras de borda nos tiles 16x16 e reduções de 64/256 lanes.
        let cfg = TransformerConfig {
            vocab_size: 53,
            max_seq_len: 21,
            d_model: 36,
            n_heads: 6,
            n_layers: 2,
            d_ff: 75,
        };
        let mut model_cpu = ServerMindTransformer::new_random(cfg.clone(), 202607).unwrap();
        let mut model_gpu = model_cpu.clone();
        let mut opt_cpu = AdamWState::new(&model_cpu);
        let mut opt_gpu = opt_cpu.clone();

        let mut sess = super::VulkanTrainingSession::new(dev, &model_gpu, &opt_gpu).unwrap();
        assert!(sess.vram_allocated_mb() > 0.0);

        // Duas amostras com comprimentos distintos (`21` e `13`), tokens repetidos e `pad_id = 0`
        let in1: Vec<usize> = (0..21).map(|i| (i * 3 + 2) % 11 + 1).collect();
        let mut tg1: Vec<usize> = (0..21).map(|i| (i * 5 + 1) % 53).collect();
        tg1[4] = 0;
        let in2: Vec<usize> = (0..13).map(|i| (i * 7 + 1) % 9 + 1).collect();
        let tg2: Vec<usize> = (0..13).map(|i| (i * 2 + 3) % 53).collect();

        sess.zero_grads().unwrap();
        let mut grads_cpu = TransformerGrads::zeros_like(&cfg);

        let (l1_gpu, c1_gpu) = sess.forward_and_backward_sample(&in1, &tg1, 0).unwrap();
        let (l1_cpu, c1_cpu) =
            model_cpu.forward_and_backward_reference(&in1, &tg1, 0, &mut grads_cpu);
        assert_eq!(c1_gpu, c1_cpu);
        assert!(
            (l1_gpu - l1_cpu).abs() <= 2e-3 * (1.0 + l1_cpu.abs()),
            "loss1 gpu={} vs cpu={}",
            l1_gpu,
            l1_cpu
        );

        let (l2_gpu, c2_gpu) = sess.forward_and_backward_sample(&in2, &tg2, 0).unwrap();
        let (l2_cpu, c2_cpu) =
            model_cpu.forward_and_backward_reference(&in2, &tg2, 0, &mut grads_cpu);
        assert_eq!(c2_gpu, c2_cpu);
        assert!(
            (l2_gpu - l2_cpu).abs() <= 2e-3 * (1.0 + l2_cpu.abs()),
            "loss2 gpu={} vs cpu={}",
            l2_gpu,
            l2_cpu
        );

        // Verifica equivalência numérica de TODOS os tensores de gradiente na VRAM antes do AdamW
        let mut grads_gpu = TransformerGrads::zeros_like(&cfg);
        sess.download_grads(&mut grads_gpu).unwrap();

        let flatten_grads = |g: &TransformerGrads| -> Vec<f32> {
            let mut out = Vec::new();
            out.extend_from_slice(&g.token_emb);
            out.extend_from_slice(&g.pos_emb);
            for b in &g.blocks {
                out.extend_from_slice(&b.ln1_gamma);
                out.extend_from_slice(&b.wq);
                out.extend_from_slice(&b.wk);
                out.extend_from_slice(&b.wv);
                out.extend_from_slice(&b.wo);
                out.extend_from_slice(&b.ln2_gamma);
                out.extend_from_slice(&b.w1);
                out.extend_from_slice(&b.b1);
                out.extend_from_slice(&b.w2);
                out.extend_from_slice(&b.b2);
            }
            out.extend_from_slice(&g.ln_f_gamma);
            out.extend_from_slice(&g.lm_head);
            out
        };
        let fg = flatten_grads(&grads_gpu);
        let fc = flatten_grads(&grads_cpu);
        assert_eq!(fg.len(), fc.len());
        for (idx, (&gv, &cv)) in fg.iter().zip(fc.iter()).enumerate() {
            let diff = (gv - cv).abs();
            let tol = 1e-3 + 5e-3 * cv.abs();
            assert!(
                diff <= tol,
                "GPU vs CPU grad mismatch at idx {}: gpu={} vs cpu={} (diff={})",
                idx,
                gv,
                cv,
                diff
            );
        }

        // Verifica `evaluate_loss` residente em VRAM vs CPU
        let eval_win = vec![vec![1usize, 4, 7, 2, 9, 12, 0, 5]];
        let val_gpu = sess.evaluate_loss(&eval_win, 0).unwrap().unwrap();
        let val_cpu = crate::training::evaluate_loss(&model_cpu, &eval_win, 0).unwrap();
        assert!((val_gpu - val_cpu).abs() <= 1e-3);

        // Executa `step_adamw` na GPU (com weight_decay e clipping ativos) e compara pesos + m + v
        let total_toks = c1_cpu + c2_cpu;
        grads_cpu.scale(1.0 / total_toks as f32);
        let gn_cpu = grads_cpu.l2_norm();
        assert!(opt_cpu.step(&mut model_cpu, &mut grads_cpu, 3e-3, 0.05, 0.5));

        let gn_gpu = sess
            .step_adamw(&mut opt_gpu.step, total_toks, 3e-3, 0.05, 0.5)
            .unwrap()
            .unwrap();
        assert!((gn_gpu - gn_cpu).abs() <= 2e-3 * (1.0 + gn_cpu));
        assert_eq!(opt_gpu.step, opt_cpu.step);

        sess.sync_to_host(&mut model_gpu, &mut opt_gpu).unwrap();

        let mut wp_gpu = Vec::new();
        let mut wp_cpu = Vec::new();
        model_gpu.visit_param_slices(|s| wp_gpu.extend_from_slice(s));
        model_cpu.visit_param_slices(|s| wp_cpu.extend_from_slice(s));
        for (idx, (&wg, &wc)) in wp_gpu.iter().zip(wp_cpu.iter()).enumerate() {
            let diff = (wg - wc).abs();
            let tol = 2e-4 + 2e-3 * wc.abs();
            assert!(
                diff <= tol,
                "GPU vs CPU weight after AdamW mismatch at idx {}: gpu={} vs cpu={}",
                idx,
                wg,
                wc
            );
        }
        for idx in 0..opt_cpu.m.len() {
            assert!((opt_gpu.m[idx] - opt_cpu.m[idx]).abs() <= 2e-4 + 5e-3 * opt_cpu.m[idx].abs());
            assert!((opt_gpu.v[idx] - opt_cpu.v[idx]).abs() <= 2e-4 + 5e-3 * opt_cpu.v[idx].abs());
        }

        // Garante que checkpoint `SMNN` v2 salvo após treino GPU recarrega e gera normalmente em CPU
        let ckpt_tmp = unique_temp_path("gpu_to_cpu_ckpt");
        save_checkpoint(&ckpt_tmp, &model_gpu, Some(&opt_gpu), 1, val_gpu).unwrap();
        let (reloaded, _, _) = load_checkpoint(&ckpt_tmp).unwrap();
        let _ = std::fs::remove_file(&ckpt_tmp);
        let gen = reloaded.generate(&[1, 3, 5], 6, 0.0, 4, 999, 42);
        assert_eq!(gen.len(), 9);
    }

    #[test]
    #[ignore]
    fn benchmark_sm006_rx7600_gpu_vs_cpu_crossover() {
        let dev =
            super::VulkanDevice::global().expect("Vulkan GPU requerida para benchmark SM-006");
        println!(
            "=== GPU DETECTADA: {} | VRAM={} MB | API={} ===",
            dev.info.device_name, dev.info.device_local_vram_mb, dev.info.api_version_str
        );

        for sz in [512usize, 1024, 2048] {
            let (gpu_ms, gpu_gf) = dev.benchmark_vram_matmul(sz).unwrap();
            let (_cpu_ms, cpu_gf) = crate::infrastructure::benchmark_cpu_matmul(sz);
            println!(
                "BENCH_MATMUL_{}x{}: CPU={:.2} GFLOP/s | GPU_VRAM={:.2} GFLOP/s ({:.2} ms) | speedup={:.2}x",
                sz,
                sz,
                cpu_gf,
                gpu_gf,
                gpu_ms,
                gpu_gf / cpu_gf.max(1e-6)
            );
        }

        let configs = [
            (
                "Small_1.45M",
                TransformerConfig {
                    vocab_size: 1024,
                    max_seq_len: 96,
                    d_model: 192,
                    n_heads: 6,
                    n_layers: 3,
                    d_ff: 512,
                },
                8usize,
            ),
            (
                "Medium_13.9M",
                TransformerConfig {
                    vocab_size: 4096,
                    max_seq_len: 256,
                    d_model: 384,
                    n_heads: 6,
                    n_layers: 6,
                    d_ff: 1536,
                },
                4usize,
            ),
        ];

        for (label, cfg, iters) in configs {
            let t = cfg.max_seq_len;
            let mut m_cpu = ServerMindTransformer::new_random(cfg.clone(), 2026).unwrap();
            let mut opt_cpu = AdamWState::new(&m_cpu);
            let mut grads_cpu = TransformerGrads::zeros_like(&cfg);
            let mut ws_cpu = super::TrainingWorkspace::new(&cfg);

            let m_gpu = m_cpu.clone();
            let mut opt_gpu = opt_cpu.clone();
            let mut sess =
                super::VulkanTrainingSession::new(dev.clone(), &m_gpu, &opt_gpu).unwrap();

            let inp: Vec<usize> = (1..=t).map(|i| i % cfg.vocab_size).collect();
            let tgt: Vec<usize> = (2..=t + 1).map(|i| i % cfg.vocab_size).collect();

            // Warmup GPU
            sess.zero_grads().unwrap();
            let _ = sess.forward_and_backward_sample(&inp, &tgt, 0).unwrap();
            let _ = sess
                .step_adamw(&mut opt_gpu.step, t, 1e-3, 0.01, 1.0)
                .unwrap();

            // Mede CPU (forward + backward + AdamW)
            let tc0 = std::time::Instant::now();
            for _ in 0..iters {
                grads_cpu.zero_out();
                let (_, cnt) = m_cpu.forward_and_backward_with_workspace(
                    &inp,
                    &tgt,
                    0,
                    &mut grads_cpu,
                    &mut ws_cpu,
                );
                grads_cpu.scale(1.0 / cnt.max(1) as f32);
                let _ = opt_cpu.step(&mut m_cpu, &mut grads_cpu, 1e-3, 0.01, 1.0);
            }
            let cpu_step_ms = tc0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // Mede GPU Residente em VRAM (zero_grads + forward + backward + step_adamw)
            let tg0 = std::time::Instant::now();
            for _ in 0..iters {
                sess.zero_grads().unwrap();
                let (_, cnt) = sess.forward_and_backward_sample(&inp, &tgt, 0).unwrap();
                let _ = sess
                    .step_adamw(&mut opt_gpu.step, cnt.max(1), 1e-3, 0.01, 1.0)
                    .unwrap();
            }
            let gpu_step_ms = tg0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            println!(
                "BENCH_TRAIN_{} (params={}, T={}, vram={:.1} MB): CPU={:.2} ms/step ({:.1} tok/s) | GPU_RX7600={:.2} ms/step ({:.1} tok/s) | speedup={:.2}x",
                label,
                cfg.total_params(),
                t,
                sess.vram_allocated_mb(),
                cpu_step_ms,
                (t as f64 * 1000.0) / cpu_step_ms,
                gpu_step_ms,
                (t as f64 * 1000.0) / gpu_step_ms,
                cpu_step_ms / gpu_step_ms.max(1e-6)
            );
        }
    }

    #[test]
    fn fp16_and_bf16_ieee754_converters_handle_edges_subnormals_and_ties() {
        use super::quant::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};

        // Zeros positivos e negativos
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());

        // Saturação finita segura (nunca emite Inf/NaN em FP16 ou BF16)
        assert_eq!(f16_to_f32(f32_to_f16(100_000.0)), 65504.0);
        assert_eq!(f16_to_f32(f32_to_f16(-100_000.0)), -65504.0);
        assert_eq!(f16_to_f32(f32_to_f16(f32::INFINITY)), 65504.0);
        assert_eq!(f16_to_f32(f32_to_f16(f32::NAN)), 0.0);
        assert!(bf16_to_f32(f32_to_bf16(f32::INFINITY)).is_finite());
        assert_eq!(bf16_to_f32(f32_to_bf16(f32::NAN)), 0.0);

        // Subnormais FP16 (`2^-24` = menor subnormal positivo = `0x0001`)
        let min_sub = 2.0f32.powi(-24);
        assert_eq!(f32_to_f16(min_sub), 0x0001);
        assert!((f16_to_f32(0x0001) - min_sub).abs() < 1e-12);

        // Round-to-nearest-even em FP16: `1.0 + 2^-11` empata exatamente entre `0x3C00` (par) e `0x3C01`
        let tie_even = 1.0f32 + 2.0f32.powi(-11);
        assert_eq!(f32_to_f16(tie_even), 0x3C00);
        // `1.0 + 3 * 2^-11` empata entre `0x3C01` (ímpar) e `0x3C02` (par) -> arredonda para `0x3C02`
        let tie_odd = 1.0f32 + 3.0 * 2.0f32.powi(-11);
        assert_eq!(f32_to_f16(tie_odd), 0x3C02);

        // Precisão relativa em faixa típica de pesos neurais
        for &v in &[-3.1415926f32, -0.123456, -1e-4, 1e-4, 0.042, 1.75, 12.5] {
            let rt_f16 = f16_to_f32(f32_to_f16(v));
            let rt_bf16 = bf16_to_f32(f32_to_bf16(v));
            assert!((rt_f16 - v).abs() <= 1e-3 * v.abs().max(1e-2));
            assert!((rt_bf16 - v).abs() <= 1e-2 * v.abs().max(1e-2));
        }
    }

    #[test]
    fn quant_matrix_avx2_matches_scalar_reference_across_all_precisions_and_prime_shapes() {
        use super::quant::{QuantMatrix, WeightPrecision};

        let mut rng = SplitMix64::new(202608);
        // Dimensões propositalmente primas/ímpares (<32, =32, >32) para estressar todos os tails
        // de 8, 16 e 32 colunas (incluindo nibbles ímpares no INT4).
        let shapes = [
            (1usize, 19usize, 29usize),
            (3, 33, 47),
            (5, 64, 97),
            (2, 71, 131),
        ];
        let precisions = [
            WeightPrecision::Fp16,
            WeightPrecision::Bf16,
            WeightPrecision::Int8,
            WeightPrecision::Int4,
        ];

        for (m, k, n) in shapes {
            let a = rng.normal_vec(m * k, 0.6);
            let w = rng.normal_vec(k * n, 0.4);
            let c_init = rng.normal_vec(m * n, 0.15);

            for prec in precisions {
                let qm = QuantMatrix::from_f32(&w, k, n, prec);
                let mut c_avx = c_init.clone();
                let mut c_ref = c_init.clone();
                qm.matmul_nn(&a, &mut c_avx, m);
                qm.matmul_nn_reference(&a, &mut c_ref, m);

                for idx in 0..(m * n) {
                    let diff = (c_avx[idx] - c_ref[idx]).abs();
                    let tol = 1e-4 + 1e-4 * c_ref[idx].abs();
                    assert!(
                        diff <= tol,
                        "QuantMatrix {:?} ({},{},{}) idx={} mismatch: opt={} vs ref={} (diff={})",
                        prec,
                        m,
                        k,
                        n,
                        idx,
                        c_avx[idx],
                        c_ref[idx],
                        diff
                    );
                }
            }
        }
    }

    #[test]
    fn smq1_quantized_packages_reduce_ram_verify_crc32_and_preserve_kv_cache_accuracy() {
        use super::{
            load_quantized_checkpoint, quantize_model_in_memory, save_quantized_checkpoint,
            KvCache, KvCachePrecision, WeightPrecision,
        };

        let cfg = TransformerConfig {
            vocab_size: 73,
            max_seq_len: 24,
            d_model: 48,
            n_heads: 6,
            n_layers: 2,
            d_ff: 103,
        };
        let fp32_model = ServerMindTransformer::new_random(cfg.clone(), 202609).unwrap();
        let fp32_bytes = fp32_model.actual_weight_bytes();
        assert_eq!(fp32_bytes, cfg.total_params() * 4);

        // 1. `KvCache` em `Fp16` (50% RAM) e `Int8` (~28% RAM) vs `Fp32`
        let kv_f32 = KvCache::with_precision(&cfg, KvCachePrecision::Fp32);
        let kv_f16 = KvCache::with_precision(&cfg, KvCachePrecision::Fp16);
        let kv_i8 = KvCache::with_precision(&cfg, KvCachePrecision::Int8);
        assert_eq!(kv_f16.memory_bytes() * 2, kv_f32.memory_bytes());
        assert!(kv_i8.memory_bytes() < kv_f16.memory_bytes());

        let prompt = [1usize, 7, 15, 23, 31, 42];
        let mut c32 = kv_f32;
        let mut c16 = kv_f16;
        let mut ci8 = kv_i8;
        let _ = fp32_model.forward_last_logits_cached(&prompt[..4], &mut c32);
        let _ = fp32_model.forward_last_logits_cached(&prompt[..4], &mut c16);
        let _ = fp32_model.forward_last_logits_cached(&prompt[..4], &mut ci8);
        let l32 = fp32_model.forward_last_logits_cached(&prompt[4..], &mut c32);
        let l16 = fp32_model.forward_last_logits_cached(&prompt[4..], &mut c16);
        let li8 = fp32_model.forward_last_logits_cached(&prompt[4..], &mut ci8);
        for j in 0..cfg.vocab_size {
            assert!(
                (l32[j] - l16[j]).abs() < 5e-3,
                "kv fp16 diff: {} vs {}",
                l32[j],
                l16[j]
            );
            assert!(
                (l32[j] - li8[j]).abs() < 4e-2,
                "kv int8 diff: {} vs {}",
                l32[j],
                li8[j]
            );
        }

        // 2. Round-trip completo `SMQ1` em todas as 4 precisões + garantia de zero pesos 2D FP32 na RAM
        let eval_win = vec![vec![2usize, 9, 14, 27, 35, 41, 12, 6]];
        let loss_fp32 = crate::training::evaluate_loss(&fp32_model, &eval_win, 0).unwrap();

        let mut prev_bytes = fp32_bytes;
        for (prec, max_loss_delta) in [
            (WeightPrecision::Fp16, 0.02f32),
            (WeightPrecision::Bf16, 0.05f32),
            (WeightPrecision::Int8, 0.08f32),
            (WeightPrecision::Int4, 0.35f32),
        ] {
            let qm = quantize_model_in_memory(&fp32_model, prec).unwrap();
            assert!(qm.is_quantized());
            assert!(qm.token_emb.is_empty() && qm.lm_head.is_empty());
            assert!(qm.blocks.iter().all(|b| b.wq.is_empty() && b.w1.is_empty()));

            // Bloqueia salvamento FP32 acidental ou treino sobre modelo quantizado
            let dummy_p = unique_temp_path("forbid_fp32_save");
            assert!(save_checkpoint(&dummy_p, &qm, None, 1, 1.0).is_err());

            let smq_path = unique_temp_path(&format!("smq_{}", prec.as_str()));
            save_quantized_checkpoint(&smq_path, &qm, 42, loss_fp32).unwrap();
            let (reloaded, meta) = load_quantized_checkpoint(&smq_path).unwrap();
            assert_eq!(meta.step, 42);
            assert_eq!(meta.precision, prec);
            assert_eq!(reloaded.actual_weight_bytes(), qm.actual_weight_bytes());

            // Verifica redução estrita de memória: FP32 > FP16 == BF16 > INT8 > INT4
            if prec != WeightPrecision::Bf16 {
                assert!(reloaded.actual_weight_bytes() < prev_bytes);
                prev_bytes = reloaded.actual_weight_bytes();
            }

            // Consistência exata entre `forward_logits` e `forward_last_logits_cached` no modelo quantizado
            let full_q = reloaded.forward_logits(&prompt);
            let last_full = &full_q[(prompt.len() - 1) * cfg.vocab_size..];
            let mut qcache = KvCache::new(&cfg);
            let _ = reloaded.forward_last_logits_cached(&prompt[..3], &mut qcache);
            let last_cached = reloaded.forward_last_logits_cached(&prompt[3..], &mut qcache);
            for j in 0..cfg.vocab_size {
                assert!(
                    (last_full[j] - last_cached[j]).abs() <= 1e-4,
                    "quant {:?} cached vs full mismatch at {}: {} vs {}",
                    prec,
                    j,
                    last_cached[j],
                    last_full[j]
                );
            }

            let loss_q = crate::training::evaluate_loss(&reloaded, &eval_win, 0).unwrap();
            assert!(
                (loss_q - loss_fp32).abs() <= max_loss_delta,
                "degradação excessiva de loss em {:?}: fp32={} vs quant={}",
                prec,
                loss_fp32,
                loss_q
            );

            // Adulteração de 1 bit no payload deve ser bloqueada pelo CRC32 do `SMQ1`
            let mut raw = std::fs::read(&smq_path).unwrap();
            let mid = raw.len() / 2;
            raw[mid] ^= 0x01;
            std::fs::write(&smq_path, &raw).unwrap();
            let err_msg = load_quantized_checkpoint(&smq_path).unwrap_err();
            assert!(
                err_msg.contains("CRC32"),
                "esperado erro CRC32, obtido: {}",
                err_msg
            );
            let _ = std::fs::remove_file(&smq_path);
        }
    }

    #[test]
    #[ignore]
    fn benchmark_sm007_quantization_memory_perplexity_and_speed() {
        use super::{
            load_quantized_checkpoint, quantize_model_in_memory, save_quantized_checkpoint,
            KvCache, KvCachePrecision, WeightPrecision,
        };

        // 1. Se `checkpoints/servermind-neural-v2` existir, mede no modelo real treinado do projeto;
        //    além disso, mede na arquitetura de referência `Medium_13.9M` para projetar escala.
        let real_dir = std::path::Path::new("checkpoints/servermind-neural-v2");
        if real_dir.join("best.bin").is_file() && real_dir.join("tokenizer.json").is_file() {
            let engine_fp32 = crate::inference::NeuralEngine::load_from_dir(real_dir).unwrap();
            let raw_ds =
                std::fs::read_to_string("datasets/servermind_corpus_v2.jsonl").unwrap_or_default();
            let mut val_texts = Vec::new();
            for line in raw_ds.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    let p = v.get("prompt").and_then(|x| x.as_str()).unwrap_or("");
                    let r = v.get("response").and_then(|x| x.as_str()).unwrap_or("");
                    val_texts.push(format!("<user> {}\n<assistant> {}", p, r));
                }
            }
            let val_wins = crate::training::pack_eval_sequences(
                &engine_fp32.tokenizer,
                &val_texts,
                engine_fp32.model.config.max_seq_len,
            );
            let pad = engine_fp32.tokenizer.pad_id();
            let best_disk_mb = std::fs::metadata(real_dir.join("best.bin")).unwrap().len() as f64
                / (1024.0 * 1024.0);
            let fp32_ram_mb = engine_fp32.weight_bytes() as f64 / (1024.0 * 1024.0);
            let fp32_loss =
                crate::training::evaluate_loss(&engine_fp32.model, &val_wins, pad).unwrap_or(0.0);
            let prompt_ids: Vec<usize> = (1..=24)
                .map(|i| i % engine_fp32.model.config.vocab_size)
                .collect();

            let t0 = std::time::Instant::now();
            for _ in 0..4 {
                let _ = engine_fp32.model.generate_with_kv_precision(
                    &prompt_ids,
                    64,
                    0.0,
                    8,
                    &[999999],
                    42,
                    KvCachePrecision::Fp32,
                    None,
                );
            }
            let fp32_tok_s = (4.0 * 64.0) / t0.elapsed().as_secs_f64();
            println!(
                "REAL_CKPT_V2 [FP32]       : disk={:.3} MB (w/opt) | ram_weights={:.3} MB | val_loss={:.4} | ppl={:.2} | decode={:.1} tok/s",
                best_disk_mb,
                fp32_ram_mb,
                fp32_loss,
                (fp32_loss as f64).exp(),
                fp32_tok_s
            );

            for prec in [
                WeightPrecision::Fp16,
                WeightPrecision::Bf16,
                WeightPrecision::Int8,
                WeightPrecision::Int4,
            ] {
                let qm = quantize_model_in_memory(&engine_fp32.model, prec).unwrap();
                let p_smq = unique_temp_path(&format!("bench_v2_{}", prec.as_str()));
                save_quantized_checkpoint(&p_smq, &qm, engine_fp32.step, fp32_loss).unwrap();
                let disk_mb = std::fs::metadata(&p_smq).unwrap().len() as f64 / (1024.0 * 1024.0);
                let (reloaded, _) = load_quantized_checkpoint(&p_smq).unwrap();
                let _ = std::fs::remove_file(&p_smq);

                let ram_mb = reloaded.actual_weight_bytes() as f64 / (1024.0 * 1024.0);
                let qloss =
                    crate::training::evaluate_loss(&reloaded, &val_wins, pad).unwrap_or(0.0);
                let tq = std::time::Instant::now();
                for _ in 0..4 {
                    let _ = reloaded.generate_with_kv_precision(
                        &prompt_ids,
                        64,
                        0.0,
                        8,
                        &[999999],
                        42,
                        KvCachePrecision::Fp32,
                        None,
                    );
                }
                let q_tok_s = (4.0 * 64.0) / tq.elapsed().as_secs_f64();
                println!(
                    "REAL_CKPT_V2 [{:<4}]       : disk={:.3} MB ({:.2}x vs best.bin) | ram_weights={:.3} MB ({:.2}x vs fp32) | val_loss={:.4} (Δ={:+.4}) | ppl={:.2} | decode={:.1} tok/s",
                    prec.as_str().to_uppercase(),
                    disk_mb,
                    best_disk_mb / disk_mb,
                    ram_mb,
                    fp32_ram_mb / ram_mb,
                    qloss,
                    qloss - fp32_loss,
                    (qloss as f64).exp(),
                    q_tok_s
                );
            }
        }

        // 2. Benchmark de escala em `Medium_13.9M` (`V=4096, T=256, d=384, H=6, L=6, ff=1536`)
        let cfg_14m = TransformerConfig {
            vocab_size: 4096,
            max_seq_len: 256,
            d_model: 384,
            n_heads: 6,
            n_layers: 6,
            d_ff: 1536,
        };
        let m14 = ServerMindTransformer::new_random(cfg_14m.clone(), 2026).unwrap();
        let fp32_mb = m14.actual_weight_bytes() as f64 / (1024.0 * 1024.0);
        let prompt14: Vec<usize> = (1..=32).collect();

        for kv_p in [
            KvCachePrecision::Fp32,
            KvCachePrecision::Fp16,
            KvCachePrecision::Int8,
        ] {
            let kv = KvCache::with_precision(&cfg_14m, kv_p);
            let t0 = std::time::Instant::now();
            let _ = m14.generate_with_kv_precision(&prompt14, 48, 0.0, 8, &[99999], 7, kv_p, None);
            let tok_s = 48.0 / t0.elapsed().as_secs_f64();
            println!(
                "KV_CACHE_13.9M [{:<4}]     : kv_ram={:.3} MB | decode={:.1} tok/s",
                kv_p.as_str().to_uppercase(),
                kv.memory_bytes() as f64 / (1024.0 * 1024.0),
                tok_s
            );
        }

        for prec in [
            WeightPrecision::Fp16,
            WeightPrecision::Bf16,
            WeightPrecision::Int8,
            WeightPrecision::Int4,
        ] {
            let q14 = quantize_model_in_memory(&m14, prec).unwrap();
            let q_mb = q14.actual_weight_bytes() as f64 / (1024.0 * 1024.0);
            let t0 = std::time::Instant::now();
            let _ = q14.generate_with_kv_precision(
                &prompt14,
                48,
                0.0,
                8,
                &[99999],
                7,
                KvCachePrecision::Fp16,
                None,
            );
            let tok_s = 48.0 / t0.elapsed().as_secs_f64();
            println!(
                "MODEL_13.9M [{:<4}+KV_FP16]: ram_weights={:.2} MB (fp32={:.2} MB -> {:.2}x menor) | decode={:.1} tok/s",
                prec.as_str().to_uppercase(),
                q_mb,
                fp32_mb,
                fp32_mb / q_mb,
                tok_s
            );
        }
    }
}
