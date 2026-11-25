//! Agente 6 — Abstração Unificada de Inferência (`Ollama` fallback vs `ServerMindNeural`).
//!
//! Permite alternar dinamicamente entre o backend externo temporário (Ollama)
//! e o motor neural próprio do ServerMind (`BpeTokenizer` + `ServerMindTransformer`),
//! mantendo a CLI, os guardrails de segurança e a suíte de avaliação funcionando.

use crate::infrastructure::RuntimeBudget;
use crate::models::{Message, Role};
use crate::neural::{
    load_checkpoint, load_quantized_checkpoint, KvCachePrecision, ServerMindTransformer,
    TransformerConfig,
};
use crate::ollama::OllamaClient;
use crate::tokenizer::{
    BpeTokenizer, ASSISTANT_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SYSTEM_TOKEN, USER_TOKEN,
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            temperature: 0.7,
            top_k: 16,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceBackendKind {
    Ollama,
    ServerMindNeural,
}

impl InferenceBackendKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::ServerMindNeural => "neural",
        }
    }

    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "ollama" => Ok(Self::Ollama),
            "neural" | "servermind" | "servermind-neural" => Ok(Self::ServerMindNeural),
            other => Err(format!(
                "Backend de inferência desconhecido: '{}'. Use 'ollama' ou 'neural'.",
                other
            )),
        }
    }
}

pub struct NeuralEngine {
    pub model: ServerMindTransformer,
    pub tokenizer: BpeTokenizer,
    pub checkpoint_dir: PathBuf,
    pub step: usize,
    pub best_val_loss: f32,
    pub kv_precision: KvCachePrecision,
}

impl NeuralEngine {
    pub fn load_from_dir(dir: &Path) -> Result<Self, String> {
        Self::load_from_dir_with_budget(dir, RuntimeBudget::auto_detect())
    }

    /// Carrega um modelo neural respeitando preventivamente o orçamento de RAM (`RuntimeBudget`):
    /// inspeciona apenas o cabeçalho JSON do arquivo (`.bin` ou `.smq`) **antes** de alocar
    /// os tensores de pesos na RAM, bloqueando cargas que excedam `budget.max_inference_ram_bytes`.
    pub fn load_from_dir_with_budget(dir: &Path, budget: RuntimeBudget) -> Result<Self, String> {
        if let Some(t) = budget.max_threads {
            crate::neural::ops::set_max_active_threads(Some(t));
        }

        let tok_path = dir.join("tokenizer.json");
        if !tok_path.is_file() {
            return Err(format!(
                "Tokenizer próprio não encontrado em '{}'. Execute `:train` para treinar o ServerMind Neural.",
                tok_path.display()
            ));
        }
        let tokenizer = BpeTokenizer::load(&tok_path)?;

        let best_path = dir.join("best.bin");
        let latest_path = dir.join("latest.bin");
        let smq_path = dir.join("quantized.smq");

        let (selected_path, is_smq) = if best_path.is_file() {
            (best_path, false)
        } else if smq_path.is_file() {
            (smq_path, true)
        } else if latest_path.is_file() {
            (latest_path, false)
        } else {
            return Err(format!(
                "Nenhum checkpoint ('best.bin', 'quantized.smq' ou 'latest.bin') encontrado em '{}'.",
                dir.display()
            ));
        };

        let kv_precision = budget.default_kv_precision;
        if let Some(max_bytes) = budget.max_inference_ram_bytes {
            let est_bytes = peek_checkpoint_runtime_bytes(&selected_path, is_smq, kv_precision)?;
            if est_bytes > max_bytes {
                let mb = 1024.0 * 1024.0;
                return Err(format!(
                    "Orçamento de memória excedido ao carregar '{}': estimado {:.2} MB (pesos + KV {} + prefill) > limite {:.2} MB (perfil {}).\nPróxima ação: exporte um pacote quantizado compacto com `:quantize {} <pasta_saida> int8` (ou `int4`).",
                    selected_path.display(),
                    est_bytes as f64 / mb,
                    kv_precision.as_str(),
                    max_bytes as f64 / mb,
                    budget.profile.as_str(),
                    dir.display()
                ));
            }
        }

        let (model, step, best_val_loss) = if is_smq {
            let (m, meta) = load_quantized_checkpoint(&selected_path)?;
            (m, meta.step, meta.best_val_loss)
        } else {
            let (m, _opt, meta) = load_checkpoint(&selected_path)?;
            (m, meta.step, meta.best_val_loss)
        };

        if tokenizer.vocab_size() != model.config.vocab_size {
            return Err(format!(
                "Incompatibilidade entre tokenizer (vocab={}) e checkpoint '{}' (vocab={}).",
                tokenizer.vocab_size(),
                selected_path.display(),
                model.config.vocab_size
            ));
        }

        let engine = Self {
            model,
            tokenizer,
            checkpoint_dir: dir.to_path_buf(),
            step,
            best_val_loss,
            kv_precision,
        };

        if let Some(max_bytes) = budget.max_inference_ram_bytes {
            let actual_rt = engine.estimated_runtime_bytes();
            if actual_rt > max_bytes {
                let mb = 1024.0 * 1024.0;
                return Err(format!(
                    "Orçamento de memória excedido após inspeção de '{}': {:.2} MB > limite {:.2} MB.",
                    selected_path.display(),
                    actual_rt as f64 / mb,
                    max_bytes as f64 / mb
                ));
            }
        }

        Ok(engine)
    }

    pub fn with_kv_precision(mut self, kv_precision: KvCachePrecision) -> Self {
        self.kv_precision = kv_precision;
        self
    }

    pub fn precision_label(&self) -> &'static str {
        self.model.weight_precision_name()
    }

    pub fn weight_bytes(&self) -> usize {
        self.model.actual_weight_bytes()
    }

    /// Retorna o footprint total estimado de inferência em bytes (`pesos + KvCache ativo + scratch de prefill`).
    pub fn estimated_runtime_bytes(&self) -> usize {
        self.weight_bytes()
            .saturating_add(kv_cache_bytes_for(&self.model.config, self.kv_precision))
            .saturating_add(prefill_scratch_bytes_for(&self.model.config))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn generate_reply(
        &self,
        user_prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> String {
        self.generate_reply_with_context(&[], user_prompt, max_new_tokens, temperature, top_k)
    }

    pub fn generate_reply_with_context(
        &self,
        request_messages: &[Message],
        user_prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> String {
        let cfg = GenerationConfig {
            max_new_tokens,
            temperature,
            top_k,
            seed: None,
        };
        let (reply, _) = self.generate_with_config(request_messages, user_prompt, cfg, None);
        reply
    }

    pub fn generate_with_config(
        &self,
        request_messages: &[Message],
        user_prompt: &str,
        cfg: GenerationConfig,
        mut on_chunk: Option<&mut dyn FnMut(&str)>,
    ) -> (String, usize) {
        let prompt_ids =
            self.build_prompt_ids(request_messages, user_prompt, cfg.max_new_tokens);
        let stop_tokens = self.stop_token_ids();

        let seed = cfg.seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1337)
                ^ (self.step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        });

        let mut streamed_tokens: Vec<usize> = Vec::with_capacity(cfg.max_new_tokens);
        let mut emitted_bytes = 0usize;

        let generated = {
            let tok_ref = &self.tokenizer;
            let mut token_cb = |tok_id: usize| {
                streamed_tokens.push(tok_id);
                if let Some(cb) = on_chunk.as_deref_mut() {
                    let partial = sanitize_decoded_tokens(&tok_ref.decode(&streamed_tokens));
                    if !partial.ends_with('\u{FFFD}') && partial.len() > emitted_bytes {
                        cb(&partial[emitted_bytes..]);
                        emitted_bytes = partial.len();
                    }
                }
            };
            self.model.generate_with_kv_precision(
                &prompt_ids,
                cfg.max_new_tokens,
                cfg.temperature,
                cfg.top_k,
                &stop_tokens,
                seed,
                self.kv_precision,
                Some(&mut token_cb),
            )
        };

        let raw_new = if generated.len() > prompt_ids.len() {
            &generated[prompt_ids.len()..]
        } else {
            &[]
        };
        let token_count = raw_new.len();
        // Exclui eventual stop token final da decodificação textual
        let content_tokens = match raw_new.last() {
            Some(last) if stop_tokens.contains(last) => &raw_new[..raw_new.len() - 1],
            _ => raw_new,
        };

        let decoded = sanitize_decoded_tokens(&self.tokenizer.decode(content_tokens));
        if let Some(cb) = on_chunk.as_deref_mut() {
            if decoded.len() > emitted_bytes {
                cb(&decoded[emitted_bytes..]);
            }
        }

        let cleaned = decoded.trim().to_string();
        let final_text = if cleaned.is_empty() {
            let msg = format!(
                "[ServerMind Neural | step={} | val_loss={:.4}] (sem tokens gerados após prompt)",
                self.step, self.best_val_loss
            );
            if let Some(cb) = on_chunk {
                if emitted_bytes == 0 {
                    cb(&msg);
                }
            }
            msg
        } else {
            cleaned
        };
        (final_text, token_count)
    }

    fn stop_token_ids(&self) -> Vec<usize> {
        let mut stops = vec![self.tokenizer.eos_id()];
        for sp in [USER_TOKEN, SYSTEM_TOKEN, ASSISTANT_TOKEN] {
            if let Some(id) = self.tokenizer.token_id(sp) {
                if !stops.contains(&id) {
                    stops.push(id);
                }
            }
        }
        stops
    }

    fn build_prompt_ids(
        &self,
        request_messages: &[Message],
        user_prompt: &str,
        max_new_tokens: usize,
    ) -> Vec<usize> {
        let mut formatted = String::new();

        // Em modelos neurais próprios compactos (`max_seq_len <= 256`), omitimos blocos longos
        // `<system>` para preservar 100% da janela útil no padrão `<user> ...\n<assistant> `
        // aprendido no fine-tuning supervisionado (`SM-009`).
        if self.model.config.max_seq_len > 256 {
            for sys_note in request_messages
                .iter()
                .skip(1)
                .filter(|m| m.role == Role::System)
            {
                let content = sys_note.content.trim();
                if !content.is_empty() {
                    formatted.push_str("<system> ");
                    formatted.push_str(content);
                    formatted.push('\n');
                }
            }
        }

        let recent_non_sys: Vec<&Message> = request_messages
            .iter()
            .filter(|m| m.role != Role::System)
            .collect();
        let max_hist_turns = if self.model.config.max_seq_len <= 256 {
            2
        } else {
            4
        };
        let skip_turns = recent_non_sys.len().saturating_sub(max_hist_turns);
        for msg in recent_non_sys.into_iter().skip(skip_turns) {
            match msg.role {
                Role::User => {
                    formatted.push_str("<user> ");
                    formatted.push_str(msg.content.trim());
                    formatted.push('\n');
                }
                Role::Assistant => {
                    formatted.push_str("<assistant> ");
                    formatted.push_str(msg.content.trim());
                    formatted.push('\n');
                }
                Role::System => {}
            }
        }
        if !formatted.ends_with(&format!("<user> {}\n", user_prompt.trim())) {
            formatted.push_str("<user> ");
            formatted.push_str(user_prompt.trim());
            formatted.push('\n');
        }
        formatted.push_str("<assistant> ");

        let prompt_ids = self.tokenizer.encode(&formatted, true, false);
        let max_prompt = self
            .model
            .config
            .max_seq_len
            .saturating_sub(max_new_tokens.min(self.model.config.max_seq_len / 2))
            .max(8);
        if prompt_ids.len() > max_prompt {
            // Preserva `<bos>` na posição 0 e mantém os últimos `max_prompt - 1` tokens
            let tail_len = max_prompt - 1;
            let start = prompt_ids.len() - tail_len;
            let mut truncated = Vec::with_capacity(max_prompt);
            truncated.push(self.tokenizer.bos_id());
            truncated.extend_from_slice(&prompt_ids[start..]);
            truncated
        } else {
            prompt_ids
        }
    }
}

fn kv_cache_bytes_for(cfg: &TransformerConfig, kv_precision: KvCachePrecision) -> usize {
    let elems = 2usize
        .saturating_mul(cfg.n_layers)
        .saturating_mul(cfg.max_seq_len)
        .saturating_mul(cfg.d_model);
    match kv_precision {
        KvCachePrecision::Fp32 => elems.saturating_mul(4),
        KvCachePrecision::Fp16 => elems.saturating_mul(2),
        KvCachePrecision::Int8 => elems.saturating_add(
            2usize
                .saturating_mul(cfg.n_layers)
                .saturating_mul(cfg.max_seq_len)
                .saturating_mul(4),
        ),
    }
}

fn prefill_scratch_bytes_for(cfg: &TransformerConfig) -> usize {
    // Buffers temporários de prefill em `forward_last_logits_cached`: `x`, `q_new`, `k_new`, `v_new`,
    // `ctx`, `attn_out`, `ln1`/`ln2`, `mlp` (`t * d_ff`) e `last_logits` (`v`).
    let t = cfg.max_seq_len;
    let d = cfg.d_model;
    let ff = cfg.d_ff;
    let v = cfg.vocab_size;
    let elems = t
        .saturating_mul(8 * d + ff + 1)
        .saturating_add(v);
    elems.saturating_mul(4)
}

fn quantized_weight_bytes_for(cfg: &TransformerConfig, prec: crate::neural::WeightPrecision) -> usize {
    let d = cfg.d_model;
    let ff = cfg.d_ff;
    let v = cfg.vocab_size;
    let t = cfg.max_seq_len;
    let l = cfg.n_layers;
    let kept_f32 = t * d + l * (3 * d + ff) + d;
    let kept_bytes = kept_f32.saturating_mul(4);

    let matrix_bytes = |rows: usize, cols: usize| -> usize {
        match prec {
            crate::neural::WeightPrecision::Fp16 | crate::neural::WeightPrecision::Bf16 => {
                rows.saturating_mul(cols).saturating_mul(2)
            }
            crate::neural::WeightPrecision::Int8 => {
                rows.saturating_mul(4).saturating_add(rows.saturating_mul(cols))
            }
            crate::neural::WeightPrecision::Int4 => {
                let g = crate::neural::quant::INT4_GROUP_SIZE;
                let blocks = (cols + g - 1) / g;
                let packed = (cols + 1) / 2;
                rows.saturating_mul(blocks * 2 + packed)
            }
        }
    };

    let per_block = 4usize
        .saturating_mul(matrix_bytes(d, d))
        .saturating_add(matrix_bytes(d, ff))
        .saturating_add(matrix_bytes(ff, d));
    kept_bytes
        .saturating_add(matrix_bytes(v, d))
        .saturating_add(l.saturating_mul(per_block))
        .saturating_add(matrix_bytes(d, v))
}

fn peek_checkpoint_runtime_bytes(
    path: &Path,
    is_smq: bool,
    kv_precision: KvCachePrecision,
) -> Result<usize, String> {
    let mut f = File::open(path)
        .map_err(|e| format!("Falha ao abrir '{}' para inspeção de memória: {}", path.display(), e))?;
    let mut hdr = [0u8; 12];
    f.read_exact(&mut hdr)
        .map_err(|_| format!("Cabeçalho truncado em '{}'.", path.display()))?;

    let expected_magic: &[u8; 4] = if is_smq { b"SMQ1" } else { b"SMNN" };
    if &hdr[0..4] != expected_magic {
        return Err(format!("Assinatura inválida em '{}'.", path.display()));
    }
    let meta_len = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]) as usize;
    if meta_len == 0 || meta_len > 1_048_576 {
        return Err(format!("Tamanho de metadados inválido em '{}'.", path.display()));
    }
    let mut meta_bytes = vec![0u8; meta_len];
    f.read_exact(&mut meta_bytes)
        .map_err(|_| format!("Metadados truncados em '{}'.", path.display()))?;

    let weights_and_cfg = if is_smq {
        let meta: crate::neural::quant::QuantizedCheckpointMetadata =
            serde_json::from_slice(&meta_bytes)
                .map_err(|e| format!("Metadados SMQ1 inválidos: {}", e))?;
        meta.config.validate()?;
        if meta.config.checked_total_params().is_none() {
            return Err("Overflow aritmético nas dimensões do modelo SMQ1.".to_string());
        }
        let wb = quantized_weight_bytes_for(&meta.config, meta.precision);
        (wb, meta.config)
    } else {
        let meta: crate::neural::CheckpointMetadata = serde_json::from_slice(&meta_bytes)
            .map_err(|e| format!("Metadados SMNN inválidos: {}", e))?;
        meta.config.validate()?;
        let total = meta
            .config
            .checked_total_params()
            .ok_or_else(|| "Overflow aritmético nas dimensões do checkpoint.".to_string())?;
        let wb = total
            .checked_mul(4)
            .ok_or_else(|| "Overflow em bytes de pesos FP32.".to_string())?;
        (wb, meta.config)
    };

    Ok(weights_and_cfg
        .0
        .saturating_add(kv_cache_bytes_for(&weights_and_cfg.1, kv_precision))
        .saturating_add(prefill_scratch_bytes_for(&weights_and_cfg.1)))
}

fn sanitize_decoded_tokens(raw: &str) -> String {
    raw.replace(EOS_TOKEN, "")
        .replace(PAD_TOKEN, "")
        .replace(BOS_TOKEN, "")
        .replace(USER_TOKEN, "")
        .replace(ASSISTANT_TOKEN, "")
        .replace(SYSTEM_TOKEN, "")
}

pub enum InferenceRouter {
    Ollama,
    ServerMindNeural(NeuralEngine),
}

impl InferenceRouter {
    pub fn kind(&self) -> InferenceBackendKind {
        match self {
            Self::Ollama => InferenceBackendKind::Ollama,
            Self::ServerMindNeural(_) => InferenceBackendKind::ServerMindNeural,
        }
    }

    pub fn generate(
        &self,
        ollama: &OllamaClient,
        model_name: &str,
        request_messages: &[Message],
        user_prompt: &str,
        on_chunk: Option<&mut dyn FnMut(&str)>,
    ) -> Result<String, String> {
        match self {
            Self::Ollama => match on_chunk {
                Some(cb) => ollama.send_chat_streaming(model_name, request_messages, cb),
                None => ollama.send_chat(model_name, request_messages),
            },
            Self::ServerMindNeural(engine) => {
                let (reply, _) = engine.generate_with_config(
                    request_messages,
                    user_prompt,
                    GenerationConfig::default(),
                    on_chunk,
                );
                Ok(reply)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::{save_checkpoint, TransformerConfig};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn neural_engine_loads_and_generates_without_ollama() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("servermind_infer_test_{}", nanos));
        fs::create_dir_all(&dir).unwrap();

        let corpus = [
            "<user> Como checar portas no Linux? <assistant> Use ss -tulnp para listar portas abertas.",
            "<user> Como ver logs do nginx? <assistant> Use journalctl -u nginx --no-pager.",
        ];
        let tok = BpeTokenizer::train_from_corpus(&corpus, 320);
        tok.save(&dir.join("tokenizer.json")).unwrap();

        let cfg = TransformerConfig {
            vocab_size: tok.vocab.len(),
            max_seq_len: 48,
            d_model: 32,
            n_heads: 4,
            n_layers: 1,
            d_ff: 64,
        };
        let model = ServerMindTransformer::new_random(cfg, 99).unwrap();
        save_checkpoint(&dir.join("best.bin"), &model, None, 5, 1.25).unwrap();

        let engine = NeuralEngine::load_from_dir(&dir).unwrap();
        assert_eq!(engine.step, 5);
        let reply = engine.generate_reply("Como checar portas no Linux?", 16, 0.8, 8);
        assert!(!reply.trim().is_empty());

        let router = InferenceRouter::ServerMindNeural(engine);
        assert_eq!(router.kind(), InferenceBackendKind::ServerMindNeural);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn neural_engine_validates_vocab_mismatch_and_streams_deterministically() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("servermind_infer_sm003_{}", nanos));
        fs::create_dir_all(&dir).unwrap();

        let corpus = [
            "<user> Diagnosticar nginx <assistant> Use systemctl status nginx.",
            "<user> Ver portas <assistant> Use ss -tulnp.",
        ];
        let tok = BpeTokenizer::train_from_corpus(&corpus, 300);
        tok.save(&dir.join("tokenizer.json")).unwrap();

        // Checkpoint com `vocab_size` incompatível deve falhar explicitamente no load
        let bad_cfg = TransformerConfig {
            vocab_size: tok.vocab_size() + 10,
            max_seq_len: 32,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
        };
        let bad_model = ServerMindTransformer::new_random(bad_cfg, 1).unwrap();
        save_checkpoint(&dir.join("best.bin"), &bad_model, None, 1, 1.0).unwrap();
        assert!(NeuralEngine::load_from_dir(&dir).is_err());

        // Checkpoint compatível: seed fixa reproduz exatamente a mesma saída e streaming concatena igual
        let ok_cfg = TransformerConfig {
            vocab_size: tok.vocab_size(),
            max_seq_len: 32,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
        };
        let ok_model = ServerMindTransformer::new_random(ok_cfg, 2).unwrap();
        save_checkpoint(&dir.join("best.bin"), &ok_model, None, 2, 1.0).unwrap();
        let engine = NeuralEngine::load_from_dir(&dir).unwrap();

        let gcfg = GenerationConfig {
            max_new_tokens: 12,
            temperature: 0.7,
            top_k: 8,
            seed: Some(4242),
        };
        let mut streamed = String::new();
        let mut cb = |chunk: &str| streamed.push_str(chunk);
        let (r1, n1) = engine.generate_with_config(&[], "Diagnosticar nginx", gcfg, Some(&mut cb));
        let (r2, n2) = engine.generate_with_config(&[], "Diagnosticar nginx", gcfg, None);
        assert_eq!(r1, r2);
        assert_eq!(n1, n2);
        assert_eq!(streamed.trim(), r1);

        // Exporta pacote compacto `quantized.smq` (`INT8`) sem `best.bin` e valida carregamento automático
        let q_dir = dir.join("smq_only");
        fs::create_dir_all(&q_dir).unwrap();
        tok.save(&q_dir.join("tokenizer.json")).unwrap();
        let q_model = crate::neural::quantize_model_in_memory(
            &ok_model,
            crate::neural::WeightPrecision::Int8,
        )
        .unwrap();
        crate::neural::save_quantized_checkpoint(&q_dir.join("quantized.smq"), &q_model, 7, 0.95)
            .unwrap();

        let q_engine = NeuralEngine::load_from_dir(&q_dir)
            .unwrap()
            .with_kv_precision(KvCachePrecision::Fp16);
        assert_eq!(q_engine.step, 7);
        assert_eq!(q_engine.precision_label(), "int8");
        assert!(q_engine.weight_bytes() < engine.weight_bytes());
        let (rq, nq) = q_engine.generate_with_config(&[], "Diagnosticar nginx", gcfg, None);
        assert!(!rq.trim().is_empty());
        assert!(nq > 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sm008_low_end_8gb_budget_blocks_oversized_and_certifies_cpu_only_e2e_pipeline() {
        use crate::infrastructure::{RuntimeBudget, RuntimeProfileKind};
        use crate::neural::ops::{effective_worker_threads, set_max_active_threads, worker_threads};
        use crate::neural::quant::{QuantMatrix, WeightPrecision};
        use crate::neural::{quantize_model_in_memory, save_quantized_checkpoint, ComputeDevice};
        use crate::training::{pack_eval_sequences, pack_sequences, Trainer, TrainingConfig};

        // 1. Equivalência bit-a-bit entre `QuantMatrix::matmul_nn` multi-thread (`m=14`) e single-thread (`threads=1`)
        let (m, k, n) = (14usize, 131usize, 137usize); // m*k*n = 251_258 >= 131_072
        let a: Vec<f32> = (0..(m * k)).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
        let w: Vec<f32> = (0..(k * n)).map(|i| ((i % 31) as f32 - 15.0) * 0.03).collect();
        for prec in [
            WeightPrecision::Fp16,
            WeightPrecision::Bf16,
            WeightPrecision::Int8,
            WeightPrecision::Int4,
        ] {
            let qm = QuantMatrix::from_f32(&w, k, n, prec);
            set_max_active_threads(Some(1));
            assert_eq!(effective_worker_threads(), 1);
            let mut c_st = vec![0.125f32; m * n];
            qm.matmul_nn(&a, &mut c_st, m);

            set_max_active_threads(None);
            assert_eq!(effective_worker_threads(), worker_threads());
            let mut c_mt = vec![0.125f32; m * n];
            qm.matmul_nn(&a, &mut c_mt, m);
            assert_eq!(c_st, c_mt, "divergência bit-a-bit em prefill MT {:?}", prec);
        }

        // 2. Certificação E2E CPU-Only (zero Vulkan) -> Exportação SMQ1 INT8/INT4 -> Validação de Orçamento RAM
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sm008_low_end_cert_{}", nanos));
        let fp32_dir = root.join("ckpt_fp32");
        let int8_dir = root.join("pkg_int8");
        let int4_dir = root.join("pkg_int4");
        fs::create_dir_all(&fp32_dir).unwrap();
        fs::create_dir_all(&int8_dir).unwrap();
        fs::create_dir_all(&int4_dir).unwrap();

        let corpus = [
            "<user> Como checar portas abertas no Linux? <assistant> Execute ss -tulnp para inspecionar sockets TCP e UDP.",
            "<user> Container reiniciando em loop no Docker <assistant> Verifique docker logs e docker inspect para diagnosticar o exit code.",
            "<user> Erro 502 Bad Gateway no Nginx <assistant> Confira systemctl status nginx e se o upstream responde na porta local.",
            "<user> Como auditar logins SSH suspeitos? <assistant> Consulte journalctl -u ssh e revise regras do ufw.",
        ];
        let tok = BpeTokenizer::train_from_corpus(&corpus, 340);
        tok.save(&fp32_dir.join("tokenizer.json")).unwrap();
        tok.save(&int8_dir.join("tokenizer.json")).unwrap();
        tok.save(&int4_dir.join("tokenizer.json")).unwrap();

        let cfg = TransformerConfig {
            vocab_size: tok.vocab_size(),
            max_seq_len: 48,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 192,
        };
        let model = ServerMindTransformer::new_random(cfg, 202608).unwrap();
        let tcfg = TrainingConfig {
            run_name: "sm008_cpu_cert".to_string(),
            checkpoint_dir: fp32_dir.clone(),
            max_steps: 6,
            batch_size: 2,
            seq_len: 48,
            learning_rate: 3e-3,
            min_learning_rate: 3e-4,
            warmup_steps: 1,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            eval_interval: 3,
            checkpoint_interval: 6,
            max_val_windows: 16,
            // Força estritamente CPU-only (simulando máquina de 8 GB RAM sem GPU dedicada)
            device: ComputeDevice::Cpu,
        };
        let train_str: Vec<String> = corpus.iter().map(|s| s.to_string()).collect();
        let train_wins = pack_sequences(&tok, &train_str, 48);
        let val_wins = pack_eval_sequences(&tok, &train_str, 48);
        let mut trainer = Trainer::new(model, tcfg);

        // Limita a 2 threads para simular dual-core modesto durante o treino e inferência
        set_max_active_threads(Some(2));
        let report = trainer
            .train_with_progress(&train_wins, &val_wins, tok.pad_id(), None)
            .unwrap();
        assert!(
            report.active_device.to_ascii_lowercase().starts_with("cpu"),
            "active_device={}",
            report.active_device
        );
        assert!(report.final_train_loss < report.initial_train_loss);

        // Carrega FP32 e exporta pacotes `SMQ1` em `INT8` e `INT4`
        let fp32_engine = NeuralEngine::load_from_dir(&fp32_dir).unwrap();
        let fp32_rt_bytes = fp32_engine.estimated_runtime_bytes();

        let q_int8 = quantize_model_in_memory(&fp32_engine.model, WeightPrecision::Int8).unwrap();
        save_quantized_checkpoint(
            &int8_dir.join("quantized.smq"),
            &q_int8,
            fp32_engine.step,
            fp32_engine.best_val_loss,
        )
        .unwrap();

        let q_int4 = quantize_model_in_memory(&fp32_engine.model, WeightPrecision::Int4).unwrap();
        save_quantized_checkpoint(
            &int4_dir.join("quantized.smq"),
            &q_int4,
            fp32_engine.step,
            fp32_engine.best_val_loss,
        )
        .unwrap();

        // 3. Testa bloqueio preventivo `peek_checkpoint_runtime_bytes`:
        // Define orçamento entre o tamanho do pacote `INT4` + KV FP16 e o tamanho do `FP32`!
        let mut tight_budget = RuntimeBudget::low_end_8gb();
        assert_eq!(tight_budget.profile, RuntimeProfileKind::LowEnd8Gb);
        let int4_probe =
            NeuralEngine::load_from_dir_with_budget(&int4_dir, tight_budget).unwrap();
        let int4_rt_bytes = int4_probe.estimated_runtime_bytes();
        assert!(int4_rt_bytes < fp32_rt_bytes);

        // Orçamento intermediário: bloqueia o checkpoint `FP32` ANTES de alocar pesos, mas aprova `INT4`!
        tight_budget.max_inference_ram_bytes = Some((int4_rt_bytes + fp32_rt_bytes) / 2);
        let err_fp32 = match NeuralEngine::load_from_dir_with_budget(&fp32_dir, tight_budget) {
            Ok(_) => panic!("esperado bloqueio de orçamento de RAM para FP32"),
            Err(e) => e,
        };
        assert!(
            err_fp32.contains("Orçamento de memória excedido"),
            "mensagem inesperada: {}",
            err_fp32
        );

        let int4_ok = NeuralEngine::load_from_dir_with_budget(&int4_dir, tight_budget).unwrap();
        assert_eq!(int4_ok.kv_precision, KvCachePrecision::Fp16);
        assert_eq!(int4_ok.precision_label(), "int4");

        // Carrega `INT8` sob perfil padrão `LowEnd8Gb` (`2048 MB`) e valida determinismo de streaming
        let int8_low =
            NeuralEngine::load_from_dir_with_budget(&int8_dir, RuntimeBudget::low_end_8gb())
                .unwrap();
        assert_eq!(int8_low.kv_precision, KvCachePrecision::Fp16);
        let gcfg = GenerationConfig {
            max_new_tokens: 16,
            temperature: 0.6,
            top_k: 8,
            seed: Some(202609),
        };
        let mut streamed = String::new();
        let mut cb = |s: &str| streamed.push_str(s);
        let (out1, n1) =
            int8_low.generate_with_config(&[], "Como checar portas abertas no Linux?", gcfg, Some(&mut cb));
        let (out2, n2) =
            int8_low.generate_with_config(&[], "Como checar portas abertas no Linux?", gcfg, None);
        assert_eq!(out1, out2);
        assert_eq!(n1, n2);
        assert_eq!(streamed.trim(), out1);

        // Restaura todas as threads do pool ao final do teste
        set_max_active_threads(None);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    #[ignore]
    fn benchmark_sm008_prefill_mt_and_low_end_8gb_footprint() {
        use crate::infrastructure::RuntimeBudget;
        use crate::neural::ops::{set_max_active_threads, worker_threads};
        use crate::neural::quant::WeightPrecision;
        use crate::neural::{quantize_model_in_memory, TransformerConfig};

        let cfg = TransformerConfig {
            vocab_size: 4096,
            max_seq_len: 256,
            d_model: 384,
            n_heads: 6,
            n_layers: 6,
            d_ff: 1536,
        };
        let fp32 = ServerMindTransformer::new_random(cfg.clone(), 2026).unwrap();
        let prompt_128: Vec<usize> = (1..=128).collect();

        for prec in [
            WeightPrecision::Fp16,
            WeightPrecision::Bf16,
            WeightPrecision::Int8,
            WeightPrecision::Int4,
        ] {
            let qm = quantize_model_in_memory(&fp32, prec).unwrap();

            set_max_active_threads(Some(1));
            let t_st = std::time::Instant::now();
            for _ in 0..4 {
                let _ = qm.forward_logits(&prompt_128);
            }
            let st_ms = t_st.elapsed().as_secs_f64() * 1000.0 / 4.0;

            set_max_active_threads(None);
            let t_mt = std::time::Instant::now();
            for _ in 0..8 {
                let _ = qm.forward_logits(&prompt_128);
            }
            let mt_ms = t_mt.elapsed().as_secs_f64() * 1000.0 / 8.0;

            let rt_mb = (qm.actual_weight_bytes()
                + super::kv_cache_bytes_for(&cfg, KvCachePrecision::Fp16)
                + super::prefill_scratch_bytes_for(&cfg)) as f64
                / (1024.0 * 1024.0);
            let budget_mb =
                RuntimeBudget::low_end_8gb().max_inference_ram_bytes.unwrap() as f64 / (1024.0 * 1024.0);

            println!(
                "SM008_PREFILL_13.9M [{:<4}, T=128, threads=1->{}]: 1T={:.2} ms -> MT={:.2} ms (speedup={:.2}x) | total_runtime_ram={:.2} MB / {:.0} MB budget ({:.2}%)",
                prec.as_str().to_uppercase(),
                worker_threads(),
                st_ms,
                mt_ms,
                st_ms / mt_ms.max(1e-6),
                rt_mb,
                budget_mb,
                (rt_mb / budget_mb) * 100.0
            );
        }
        set_max_active_threads(None);
    }
}
