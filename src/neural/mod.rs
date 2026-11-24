//! Agente 2 — Engenharia do Modelo Neural Próprio (Transformer Decoder-Only em Rust Puro).
//!
//! Implementa do zero:
//! - Embeddings de token e posição
//! - Blocos Transformer Decoder-only (RMSNorm, Multi-Head Causal Self-Attention, MLP GELU, residuais)
//! - Função de perda Cross-Entropy + Backpropagation analítica completa
//! - Otimizador AdamW com clipping de norma de gradiente
//! - Inferência autoregressiva (greedy e top-k/temperatura)
//! - Salvamento e carregamento de checkpoints binários completos (pesos + estado do otimizador).

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const CHECKPOINT_MAGIC: &[u8; 4] = b"SMNN";
const CHECKPOINT_VERSION: u32 = 1;

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
        if self.vocab_size == 0 || self.max_seq_len == 0 || self.d_model == 0 || self.n_layers == 0
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

    pub fn total_params(&self) -> usize {
        let embed = self.vocab_size * self.d_model + self.max_seq_len * self.d_model;
        let per_layer = 4 * self.d_model * self.d_model
            + 2 * self.d_model * self.d_ff
            + self.d_ff
            + 3 * self.d_model;
        let head = self.d_model + self.d_model * self.vocab_size;
        embed + self.n_layers * per_layer + head
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub config: TransformerConfig,
    pub step: usize,
    pub best_val_loss: f32,
    pub has_optimizer: bool,
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
        })
    }

    pub fn num_float_params(&self) -> usize {
        self.config.total_params()
    }

    /// Executa apenas o forward pass e retorna os logits `[seq_len, vocab_size]`.
    pub fn forward_logits(&self, input_ids: &[usize]) -> Vec<f32> {
        let t = input_ids.len().min(self.config.max_seq_len);
        let d = self.config.d_model;
        let v = self.config.vocab_size;

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

    /// Executa forward, perda Cross-Entropy (ignorando `pad_id`) e backpropagation analítica,
    /// acumulando gradientes em `grads`. Retorna `(soma_da_perda, tokens_validos)`.
    pub fn forward_and_backward(
        &self,
        input_ids: &[usize],
        target_ids: &[usize],
        pad_id: usize,
        grads: &mut TransformerGrads,
    ) -> (f32, usize) {
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

        // Cache de ativações por camada para backprop exata.
        struct LayerCache {
            x_in: Vec<f32>,
            ln1: Vec<f32>,
            inv_rms1: Vec<f32>,
            q: Vec<f32>,
            k: Vec<f32>,
            v: Vec<f32>,
            attn_probs: Vec<f32>, // [n_heads, t, t]
            attn_ctx: Vec<f32>,   // [t, d]
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
            matmul_nn(&ln1, &block.wq, &mut q, t, d, d);
            matmul_nn(&ln1, &block.wk, &mut k, t, d, d);
            matmul_nn(&ln1, &block.wv, &mut val, t, d, d);

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
            matmul_nn(&attn_ctx, &block.wo, &mut attn_out, t, d, d);

            let mut x_mid = vec![0.0f32; t * d];
            for i in 0..x_mid.len() {
                x_mid[i] = x_in[i] + attn_out[i];
            }

            let (ln2, inv_rms2) = rmsnorm_forward(&x_mid, &block.ln2_gamma, t, d);
            let mut h_pre = vec![0.0f32; t * ff];
            matmul_nn(&ln2, &block.w1, &mut h_pre, t, d, ff);
            for pos in 0..t {
                for j in 0..ff {
                    h_pre[pos * ff + j] += block.b1[j];
                }
            }
            let h_act: Vec<f32> = h_pre.iter().map(|&z| gelu(z)).collect();
            let mut mlp_out = vec![0.0f32; t * d];
            matmul_nn(&h_act, &block.w2, &mut mlp_out, t, ff, d);
            for pos in 0..t {
                for j in 0..d {
                    mlp_out[pos * d + j] += block.b2[j];
                }
            }

            for i in 0..x.len() {
                x[i] = x_mid[i] + mlp_out[i];
            }

            caches.push(LayerCache {
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
        matmul_nn(&ln_f, &self.lm_head, &mut logits, t, d, v);

        // Cross-entropy + d_logits
        let mut total_loss = 0.0f32;
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
            let max_l = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_e = 0.0f32;
            for j in 0..v {
                let e = (row[j] - max_l).exp();
                d_row[j] = e;
                sum_e += e;
            }
            let inv = 1.0 / sum_e.max(1e-12);
            for j in 0..v {
                d_row[j] *= inv;
            }
            let p_tgt = d_row[tgt].max(1e-12);
            total_loss += -p_tgt.ln();
            d_row[tgt] -= 1.0;
        }

        if valid_tokens == 0 {
            return (0.0, 0);
        }

        // Backprop LM Head: d_lm_head += ln_f^T @ d_logits, d_ln_f = d_logits @ lm_head^T
        matmul_tn(&ln_f, &d_logits, &mut grads.lm_head, d, t, v);
        let mut d_ln_f = vec![0.0f32; t * d];
        matmul_nt(&d_logits, &self.lm_head, &mut d_ln_f, t, v, d);

        let mut d_x = rmsnorm_backward(
            &x,
            &self.ln_f_gamma,
            &inv_rms_f,
            &d_ln_f,
            &mut grads.ln_f_gamma,
            t,
            d,
        );

        // Backprop através dos blocos em ordem reversa.
        for l in (0..self.blocks.len()).rev() {
            let block = &self.blocks[l];
            let bg = &mut grads.blocks[l];
            let c = &caches[l];

            // Residual pós-MLP: d_mlp_out = d_x, d_x_mid = d_x
            let mut d_x_mid = d_x.clone();

            // b2 grad
            for pos in 0..t {
                for j in 0..d {
                    bg.b2[j] += d_x[pos * d + j];
                }
            }
            // w2 grad: h_act^T @ d_x
            matmul_tn(&c.h_act, &d_x, &mut bg.w2, ff, t, d);
            // d_h_act = d_x @ w2^T
            let mut d_h_act = vec![0.0f32; t * ff];
            matmul_nt(&d_x, &block.w2, &mut d_h_act, t, d, ff);

            // GELU backward
            let mut d_h_pre = vec![0.0f32; t * ff];
            for i in 0..d_h_act.len() {
                d_h_pre[i] = d_h_act[i] * gelu_grad(c.h_pre[i]);
            }

            // b1 grad & w1 grad
            for pos in 0..t {
                for j in 0..ff {
                    bg.b1[j] += d_h_pre[pos * ff + j];
                }
            }
            matmul_tn(&c.ln2, &d_h_pre, &mut bg.w1, d, t, ff);
            let mut d_ln2 = vec![0.0f32; t * d];
            matmul_nt(&d_h_pre, &block.w1, &mut d_ln2, t, ff, d);

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

            // Residual pós-Attention: d_attn_out = d_x_mid, d_x_in = d_x_mid
            let mut d_x_in = d_x_mid.clone();

            matmul_tn(&c.attn_ctx, &d_x_mid, &mut bg.wo, d, t, d);
            let mut d_attn_ctx = vec![0.0f32; t * d];
            matmul_nt(&d_x_mid, &block.wo, &mut d_attn_ctx, t, d, d);

            // Multi-Head Causal Self-Attention backward
            let mut d_q = vec![0.0f32; t * d];
            let mut d_k = vec![0.0f32; t * d];
            let mut d_v = vec![0.0f32; t * d];

            for h in 0..n_heads {
                let h_off = h * head_dim;
                for i in 0..t {
                    let prob_row = &c.attn_probs[(h * t + i) * t..(h * t + i + 1) * t];
                    let out_base = i * d + h_off;
                    let mut d_probs = vec![0.0f32; i + 1];

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

                    // Softmax backward: d_score_j = p_j * (d_prob_j - sum_k(p_k * d_prob_k))
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

            matmul_tn(&c.ln1, &d_q, &mut bg.wq, d, t, d);
            matmul_tn(&c.ln1, &d_k, &mut bg.wk, d, t, d);
            matmul_tn(&c.ln1, &d_v, &mut bg.wv, d, t, d);

            let mut d_ln1 = vec![0.0f32; t * d];
            matmul_nt(&d_q, &block.wq, &mut d_ln1, t, d, d);
            matmul_nt(&d_k, &block.wk, &mut d_ln1, t, d, d);
            matmul_nt(&d_v, &block.wv, &mut d_ln1, t, d, d);

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

        // Embeddings backward
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

        (total_loss, valid_tokens)
    }

    /// Gera novos tokens autoregressivamente com suporte a greedy (`temperature <= 0.0`) ou top-k.
    pub fn generate(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        eos_id: usize,
        seed: u64,
    ) -> Vec<usize> {
        let mut seq = prompt_ids.to_vec();
        let mut rng = SplitMix64::new(seed);
        let v = self.config.vocab_size;

        for _ in 0..max_new_tokens {
            let start = seq.len().saturating_sub(self.config.max_seq_len);
            let ctx = &seq[start..];
            let logits = self.forward_logits(ctx);
            let last_off = (ctx.len() - 1) * v;
            let last_logits = &logits[last_off..last_off + v];

            let next_id = if temperature <= 0.0 {
                last_logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(idx, _)| idx)
                    .unwrap_or(eos_id)
            } else {
                sample_top_k(last_logits, temperature, top_k.max(1), &mut rng)
            };

            seq.push(next_id);
            if next_id == eos_id {
                break;
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
    pub fn step(
        &mut self,
        model: &mut ServerMindTransformer,
        grads: &mut TransformerGrads,
        lr: f32,
        weight_decay: f32,
        max_grad_norm: f32,
    ) {
        let norm = grads.l2_norm();
        if max_grad_norm > 0.0 && norm > max_grad_norm {
            grads.scale(max_grad_norm / norm.max(1e-8));
        }

        self.step += 1;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let bc1 = 1.0 - beta1.powi(self.step as i32);
        let bc2 = 1.0 - beta2.powi(self.step as i32);

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
    }
}

/// Salva pesos e estado opcional do otimizador AdamW em arquivo binário versionado.
pub fn save_checkpoint(
    path: &Path,
    model: &ServerMindTransformer,
    optimizer: Option<&AdamWState>,
    step: usize,
    best_val_loss: f32,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Falha ao criar pasta de checkpoint {:?}: {}", parent, e))?;
    }
    let meta = CheckpointMetadata {
        config: model.config.clone(),
        step,
        best_val_loss,
        has_optimizer: optimizer.is_some(),
    };
    let meta_bytes = serde_json::to_vec(&meta)
        .map_err(|e| format!("Falha ao serializar metadados do checkpoint: {}", e))?;

    let file = File::create(path)
        .map_err(|e| format!("Falha ao criar arquivo de checkpoint {:?}: {}", path, e))?;
    let mut w = BufWriter::new(file);
    w.write_all(CHECKPOINT_MAGIC).map_err(|e| e.to_string())?;
    w.write_all(&CHECKPOINT_VERSION.to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(&(meta_bytes.len() as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(&meta_bytes).map_err(|e| e.to_string())?;

    let mut io_err: Option<String> = None;
    model.visit_param_slices(|slice| {
        if io_err.is_some() {
            return;
        }
        for &val in slice {
            if let Err(e) = w.write_all(&val.to_le_bytes()) {
                io_err = Some(e.to_string());
                return;
            }
        }
    });
    if let Some(err) = io_err {
        return Err(format!("Erro ao gravar pesos do checkpoint: {}", err));
    }

    if let Some(opt) = optimizer {
        for &val in &opt.m {
            w.write_all(&val.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        for &val in &opt.v {
            w.write_all(&val.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }

    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Carrega um checkpoint binário completo (modelo + metadados + estado opcional do AdamW).
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
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .map_err(|e| format!("Falha ao ler cabeçalho do checkpoint: {}", e))?;
    if &magic != CHECKPOINT_MAGIC {
        return Err("Arquivo de checkpoint inválido (magic header incorreto).".to_string());
    }

    let mut u32_buf = [0u8; 4];
    r.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let ver = u32::from_le_bytes(u32_buf);
    if ver != CHECKPOINT_VERSION {
        return Err(format!("Versão de checkpoint não suportada: {}", ver));
    }

    r.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let meta_len = u32::from_le_bytes(u32_buf) as usize;
    let mut meta_bytes = vec![0u8; meta_len];
    r.read_exact(&mut meta_bytes).map_err(|e| e.to_string())?;
    let meta: CheckpointMetadata = serde_json::from_slice(&meta_bytes)
        .map_err(|e| format!("Metadados de checkpoint inválidos: {}", e))?;

    let mut model = ServerMindTransformer::new_random(meta.config.clone(), 0)?;
    let mut io_err: Option<String> = None;
    model.visit_param_slices_mut(|slice| {
        if io_err.is_some() {
            return;
        }
        let mut f_buf = [0u8; 4];
        for slot in slice.iter_mut() {
            if let Err(e) = r.read_exact(&mut f_buf) {
                io_err = Some(e.to_string());
                return;
            }
            *slot = f32::from_le_bytes(f_buf);
        }
    });
    if let Some(err) = io_err {
        return Err(format!("Falha ao ler pesos do checkpoint: {}", err));
    }

    let optimizer = if meta.has_optimizer {
        let mut opt = AdamWState::new(&model);
        opt.step = meta.step;
        let mut f_buf = [0u8; 4];
        for slot in &mut opt.m {
            r.read_exact(&mut f_buf).map_err(|e| e.to_string())?;
            *slot = f32::from_le_bytes(f_buf);
        }
        for slot in &mut opt.v {
            r.read_exact(&mut f_buf).map_err(|e| e.to_string())?;
            *slot = f32::from_le_bytes(f_buf);
        }
        Some(opt)
    } else {
        None
    };

    Ok((model, optimizer, meta))
}

// ---------------------------------------------------------------------------
// Primitivas matemáticas puras + MatMul multi-thread
// ---------------------------------------------------------------------------

fn rmsnorm_forward(x: &[f32], gamma: &[f32], t: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let eps = 1e-5f32;
    let mut out = vec![0.0f32; t * d];
    let mut inv_rms = vec![0.0f32; t];
    for pos in 0..t {
        let row = &x[pos * d..(pos + 1) * d];
        let mut ss = 0.0f32;
        for &val in row {
            ss += val * val;
        }
        let ir = 1.0 / (ss / d as f32 + eps).sqrt();
        inv_rms[pos] = ir;
        let out_row = &mut out[pos * d..(pos + 1) * d];
        for i in 0..d {
            out_row[i] = row[i] * ir * gamma[i];
        }
    }
    (out, inv_rms)
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
            for j in 0..=i {
                let mut dot = 0.0f32;
                for hd in 0..head_dim {
                    dot += q[i * d + h_off + hd] * k[j * d + h_off + hd];
                }
                let s = dot * scale;
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
            for j in 0..=i {
                let p = probs[j] * inv;
                for hd in 0..head_dim {
                    ctx[i * d + h_off + hd] += p * v[j * d + h_off + hd];
                }
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

#[inline]
fn gelu(x: f32) -> f32 {
    let k = 0.7978845608f32; // sqrt(2 / pi)
    let c = 0.044715f32;
    let inner = k * (x + c * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

#[inline]
fn gelu_grad(x: f32) -> f32 {
    let k = 0.7978845608f32;
    let c = 0.044715f32;
    let inner = k * (x + c * x * x * x);
    let t = inner.tanh();
    let sech2 = 1.0 - t * t;
    let d_inner = k * (1.0 + 3.0 * c * x * x);
    0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
}

#[inline]
fn worker_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|p| p.get().clamp(1, 16))
            .unwrap_or(2)
    })
}

/// C += A[m, k] @ B[k, n]
fn matmul_nn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let work = m * k * n;
    let max_threads = worker_threads();
    if work >= 65_536 && m >= 4 && max_threads > 1 {
        let threads = max_threads.min(m);
        let chunk = (m + threads - 1) / threads;
        std::thread::scope(|s| {
            for (t_idx, c_chunk) in c.chunks_mut(chunk * n).enumerate() {
                let row_start = t_idx * chunk;
                let rows = c_chunk.len() / n;
                let a_chunk = &a[row_start * k..(row_start + rows) * k];
                s.spawn(move || {
                    for i in 0..rows {
                        let a_row = &a_chunk[i * k..(i + 1) * k];
                        let c_row = &mut c_chunk[i * n..(i + 1) * n];
                        for (kk, &av) in a_row.iter().enumerate() {
                            let b_row = &b[kk * n..(kk + 1) * n];
                            for (cv, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                                *cv += av * bv;
                            }
                        }
                    }
                });
            }
        });
        return;
    }

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        for (kk, &av) in a_row.iter().enumerate() {
            let b_row = &b[kk * n..(kk + 1) * n];
            for (cv, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                *cv += av * bv;
            }
        }
    }
}

/// C += A[m, k]^T @ B[m, n]  => [k, n]
fn matmul_tn(a: &[f32], b: &[f32], c: &mut [f32], k: usize, m: usize, n: usize) {
    let work = k * m * n;
    let max_threads = worker_threads();
    if work >= 65_536 && k >= 8 && max_threads > 1 {
        let threads = max_threads.min(k);
        let chunk = (k + threads - 1) / threads;
        std::thread::scope(|s| {
            for (t_idx, c_chunk) in c.chunks_mut(chunk * n).enumerate() {
                let row_start = t_idx * chunk;
                let rows = c_chunk.len() / n;
                s.spawn(move || {
                    for row_m in 0..m {
                        let a_row = &a[row_m * k + row_start..row_m * k + row_start + rows];
                        let b_row = &b[row_m * n..(row_m + 1) * n];
                        for (i, &av) in a_row.iter().enumerate() {
                            let c_row = &mut c_chunk[i * n..(i + 1) * n];
                            for (cv, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                                *cv += av * bv;
                            }
                        }
                    }
                });
            }
        });
        return;
    }

    for row_m in 0..m {
        let a_row = &a[row_m * k..(row_m + 1) * k];
        let b_row = &b[row_m * n..(row_m + 1) * n];
        for (i, &av) in a_row.iter().enumerate() {
            let c_row = &mut c[i * n..(i + 1) * n];
            for (cv, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                *cv += av * bv;
            }
        }
    }
}

/// C += A[m, n] @ B[k, n]^T => [m, k]
fn matmul_nt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let work = m * n * k;
    let max_threads = worker_threads();
    if work >= 65_536 && m >= 4 && max_threads > 1 {
        let threads = max_threads.min(m);
        let chunk = (m + threads - 1) / threads;
        std::thread::scope(|s| {
            for (t_idx, c_chunk) in c.chunks_mut(chunk * k).enumerate() {
                let row_start = t_idx * chunk;
                let rows = c_chunk.len() / k;
                let a_chunk = &a[row_start * n..(row_start + rows) * n];
                s.spawn(move || {
                    for i in 0..rows {
                        let a_row = &a_chunk[i * n..(i + 1) * n];
                        let c_row = &mut c_chunk[i * k..(i + 1) * k];
                        for j in 0..k {
                            let b_row = &b[j * n..(j + 1) * n];
                            let mut sum = 0.0f32;
                            for (&av, &bv) in a_row.iter().zip(b_row.iter()) {
                                sum += av * bv;
                            }
                            c_row[j] += sum;
                        }
                    }
                });
            }
        });
        return;
    }

    for i in 0..m {
        let a_row = &a[i * n..(i + 1) * n];
        let c_row = &mut c[i * k..(i + 1) * k];
        for j in 0..k {
            let b_row = &b[j * n..(j + 1) * n];
            let mut sum = 0.0f32;
            for (&av, &bv) in a_row.iter().zip(b_row.iter()) {
                sum += av * bv;
            }
            c_row[j] += sum;
        }
    }
}

fn sample_top_k(logits: &[f32], temperature: f32, top_k: usize, rng: &mut SplitMix64) -> usize {
    let mut pairs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l / temperature))
        .collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    pairs.truncate(top_k.min(pairs.len()));

    let max_l = pairs[0].1;
    let mut probs: Vec<f32> = pairs.iter().map(|(_, l)| (*l - max_l).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let r = rng.uniform_f32() * sum;
    let mut acc = 0.0f32;
    for ((idx, _), p) in pairs.iter().zip(probs.iter_mut()) {
        acc += *p;
        if acc >= r {
            return *idx;
        }
    }
    pairs[0].0
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
        load_checkpoint, save_checkpoint, AdamWState, ServerMindTransformer, TransformerConfig,
        TransformerGrads,
    };

    #[test]
    fn analytical_backprop_matches_finite_differences() {
        let cfg = TransformerConfig {
            vocab_size: 16,
            max_seq_len: 8,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 16,
        };
        let mut model = ServerMindTransformer::new_random(cfg.clone(), 42).unwrap();
        let input = [1usize, 5, 3, 7];
        let target = [5usize, 3, 7, 2];

        let mut grads = TransformerGrads::zeros_like(&cfg);
        let (loss, valid) = model.forward_and_backward(&input, &target, 0, &mut grads);
        assert_eq!(valid, 4);
        assert!(loss > 0.0);

        let eps = 1e-3f32;
        let check_grad =
            |model: &mut ServerMindTransformer,
             analytic: f32,
             getter: fn(&mut ServerMindTransformer) -> &mut f32| {
                let orig = *getter(model);
                *getter(model) = orig + eps;
                let mut g_dummy = TransformerGrads::zeros_like(&model.config);
                let (l_plus, _) = model.forward_and_backward(&input, &target, 0, &mut g_dummy);
                *getter(model) = orig - eps;
                let (l_minus, _) = model.forward_and_backward(&input, &target, 0, &mut g_dummy);
                *getter(model) = orig;
                let numeric = (l_plus - l_minus) / (2.0 * eps);
                assert!(
                    (analytic - numeric).abs() < 0.05,
                    "grad mismatch: analytic={} numeric={}",
                    analytic,
                    numeric
                );
            };

        check_grad(&mut model, grads.lm_head[3], |m| &mut m.lm_head[3]);
        check_grad(&mut model, grads.ln_f_gamma[2], |m| &mut m.ln_f_gamma[2]);
        check_grad(&mut model, grads.blocks[0].w1[5], |m| {
            &mut m.blocks[0].w1[5]
        });
        check_grad(&mut model, grads.blocks[0].wq[4], |m| {
            &mut m.blocks[0].wq[4]
        });
        check_grad(&mut model, grads.blocks[0].wk[7], |m| {
            &mut m.blocks[0].wk[7]
        });
        check_grad(&mut model, grads.blocks[0].wv[2], |m| {
            &mut m.blocks[0].wv[2]
        });
        check_grad(&mut model, grads.token_emb[1 * 8 + 3], |m| {
            &mut m.token_emb[8 + 3]
        });
    }

    #[test]
    fn checkpoint_saves_and_restores_weights_and_optimizer() {
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
        let tmp = std::env::temp_dir().join(format!("sm_ckpt_{}.bin", std::process::id()));
        save_checkpoint(&tmp, &model, Some(&opt), 17, 1.42).unwrap();

        let (loaded, loaded_opt, meta) = load_checkpoint(&tmp).unwrap();
        assert_eq!(meta.step, 17);
        assert!((meta.best_val_loss - 1.42).abs() < 1e-6);
        assert_eq!(model.token_emb, loaded.token_emb);
        assert_eq!(model.lm_head, loaded.lm_head);
        assert!(loaded_opt.is_some());
        let _ = std::fs::remove_file(&tmp);
    }
}
