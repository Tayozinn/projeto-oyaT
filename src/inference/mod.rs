//! Agente 6 — Abstração Unificada de Inferência (`Ollama` fallback vs `ServerMindNeural`).
//!
//! Permite alternar dinamicamente entre o backend externo temporário (Ollama)
//! e o motor neural próprio do ServerMind (`BpeTokenizer` + `ServerMindTransformer`),
//! mantendo a CLI, os guardrails de segurança e a suíte de avaliação funcionando.

use crate::models::{Message, Role};
use crate::neural::{load_checkpoint, ServerMindTransformer};
use crate::ollama::OllamaClient;
use crate::tokenizer::BpeTokenizer;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

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
}

impl NeuralEngine {
    pub fn load_from_dir(dir: &Path) -> Result<Self, String> {
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
        let ckpt_path = if best_path.is_file() {
            best_path
        } else if latest_path.is_file() {
            latest_path
        } else {
            return Err(format!(
                "Nenhum checkpoint ('best.bin' ou 'latest.bin') encontrado em '{}'.",
                dir.display()
            ));
        };

        let (model, _opt, meta) = load_checkpoint(&ckpt_path)?;
        Ok(Self {
            model,
            tokenizer,
            checkpoint_dir: dir.to_path_buf(),
            step: meta.step,
            best_val_loss: meta.best_val_loss,
        })
    }

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
        let mut formatted = String::new();
        // Inclui notas de grounding (mensagens de sistema além da instrução principal) e turnos recentes
        let recent_non_sys: Vec<&Message> = request_messages
            .iter()
            .filter(|m| m.role != Role::System)
            .collect();
        let skip_turns = recent_non_sys.len().saturating_sub(4);
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
        if let Some(kb_note) = request_messages
            .iter()
            .skip(1)
            .find(|m| m.role == Role::System)
        {
            formatted.push_str("<system> ");
            formatted.push_str(kb_note.content.trim());
            formatted.push('\n');
        }
        formatted.push_str("<assistant> ");

        let mut prompt_ids = self.tokenizer.encode(&formatted, true, false);
        let max_prompt = self
            .model
            .config
            .max_seq_len
            .saturating_sub(max_new_tokens.min(self.model.config.max_seq_len / 2))
            .max(8);
        if prompt_ids.len() > max_prompt {
            let start = prompt_ids.len() - max_prompt;
            prompt_ids = prompt_ids[start..].to_vec();
        }

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1337)
            ^ (self.step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);

        let generated = self.model.generate(
            &prompt_ids,
            max_new_tokens,
            temperature,
            top_k,
            self.tokenizer.eos_id(),
            seed,
        );
        let new_tokens = if generated.len() > prompt_ids.len() {
            &generated[prompt_ids.len()..]
        } else {
            &[]
        };

        let decoded = self.tokenizer.decode(new_tokens);
        let cleaned = decoded
            .replace("<eos>", "")
            .replace("<pad>", "")
            .replace("<bos>", "")
            .trim()
            .to_string();

        if cleaned.is_empty() {
            format!(
                "[ServerMind Neural | step={} | val_loss={:.4}] (sem tokens gerados após prompt)",
                self.step, self.best_val_loss
            )
        } else {
            cleaned
        }
    }
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
                let reply =
                    engine.generate_reply_with_context(request_messages, user_prompt, 64, 0.7, 16);
                if let Some(cb) = on_chunk {
                    cb(&reply);
                }
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
}
