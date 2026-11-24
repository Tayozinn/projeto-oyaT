//! Agente 7 — Avaliação e Qualidade do ServerMind Neural.
//!
//! Avalia versões próprias do modelo neural em tarefas técnicas (Linux, Docker, Redes,
//! Segurança, Depuração e Programação), mede perplexity/val_loss, latência de inferência
//! e compara objetivamente versões entre si e com o baseline Ollama.

use crate::inference::NeuralEngine;
use crate::training::{evaluate_loss, pack_sequences};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralBenchmarkCase {
    pub id: String,
    pub domain: String,
    pub prompt: String,
    pub expected_keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCaseResult {
    pub id: String,
    pub domain: String,
    pub latency_ms: u128,
    pub keyword_hits: usize,
    pub expected_keywords: usize,
    pub output_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralEvalSummary {
    pub checkpoint_dir: String,
    pub step: usize,
    pub total_params: usize,
    pub val_loss: f32,
    pub perplexity: f32,
    pub avg_latency_ms: f64,
    pub keyword_recall: f64,
    pub cases: Vec<NeuralCaseResult>,
}

pub fn default_technical_suite() -> Vec<NeuralBenchmarkCase> {
    vec![
        NeuralBenchmarkCase {
            id: "linux_cpu".into(),
            domain: "linux".into(),
            prompt: "Como diagnosticar uso alto de CPU e memória em um servidor Linux?".into(),
            expected_keywords: vec!["uptime".into(), "top".into(), "free".into(), "ps".into()],
        },
        NeuralBenchmarkCase {
            id: "docker_crash".into(),
            domain: "docker".into(),
            prompt: "Um container Docker reinicia em loop. Como descobrir a causa raiz?".into(),
            expected_keywords: vec!["docker".into(), "logs".into(), "inspect".into()],
        },
        NeuralBenchmarkCase {
            id: "net_ports".into(),
            domain: "network".into(),
            prompt: "Como verificar quais portas TCP/UDP estão abertas no Linux?".into(),
            expected_keywords: vec!["ss".into(), "tulnp".into(), "nc".into()],
        },
        NeuralBenchmarkCase {
            id: "sec_ssh".into(),
            domain: "security".into(),
            prompt: "Quais passos seguir após tentativa de força bruta SSH?".into(),
            expected_keywords: vec!["journalctl".into(), "ssh".into(), "ufw".into()],
        },
        NeuralBenchmarkCase {
            id: "dbg_502".into(),
            domain: "debugging".into(),
            prompt: "Uma API retorna 502 Bad Gateway atrás do Nginx. O que checar?".into(),
            expected_keywords: vec!["systemctl".into(), "nginx".into(), "ss".into()],
        },
        NeuralBenchmarkCase {
            id: "prog_sqlite".into(),
            domain: "programming".into(),
            prompt: "Como garantir atomicidade ao gravar múltiplos registros no SQLite em Rust?"
                .into(),
            expected_keywords: vec!["transaction".into(), "commit".into(), "rollback".into()],
        },
    ]
}

pub fn evaluate_neural_engine(
    engine: &NeuralEngine,
    validation_texts: &[String],
) -> NeuralEvalSummary {
    let seq_len = engine.model.config.max_seq_len.min(96);
    let val_windows = pack_sequences(&engine.tokenizer, validation_texts, seq_len);
    let val_loss = if val_windows.is_empty() {
        engine.best_val_loss
    } else {
        evaluate_loss(&engine.model, &val_windows, engine.tokenizer.pad_id())
    };
    let perplexity = val_loss.min(20.0).exp();

    let suite = default_technical_suite();
    let mut results = Vec::with_capacity(suite.len());
    let mut total_hits = 0usize;
    let mut total_expected = 0usize;
    let mut total_latency = 0u128;

    for case in suite {
        let t0 = Instant::now();
        let reply = engine.generate_reply(&case.prompt, 40, 0.7, 16);
        let latency_ms = t0.elapsed().as_millis();
        let lower = reply.to_ascii_lowercase();
        let hits = case
            .expected_keywords
            .iter()
            .filter(|kw| lower.contains(&kw.to_ascii_lowercase()))
            .count();
        total_hits += hits;
        total_expected += case.expected_keywords.len();
        total_latency += latency_ms;

        let preview: String = reply.chars().take(160).collect();
        results.push(NeuralCaseResult {
            id: case.id,
            domain: case.domain,
            latency_ms,
            keyword_hits: hits,
            expected_keywords: case.expected_keywords.len(),
            output_preview: preview,
        });
    }

    let n = results.len().max(1) as f64;
    NeuralEvalSummary {
        checkpoint_dir: engine.checkpoint_dir.to_string_lossy().to_string(),
        step: engine.step,
        total_params: engine.model.config.total_params(),
        val_loss,
        perplexity,
        avg_latency_ms: total_latency as f64 / n,
        keyword_recall: if total_expected == 0 {
            0.0
        } else {
            total_hits as f64 / total_expected as f64
        },
        cases: results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::{save_checkpoint, ServerMindTransformer, TransformerConfig};
    use crate::tokenizer::BpeTokenizer;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn evaluates_neural_engine_metrics_and_cases() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("servermind_eval_neural_{}", nanos));
        fs::create_dir_all(&dir).unwrap();

        let corpus = vec![
            "<user> Como verificar portas? <assistant> Use ss -tulnp.".to_string(),
            "<user> Logs nginx? <assistant> Use journalctl -u nginx.".to_string(),
        ];
        let refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        let tok = BpeTokenizer::train_from_corpus(&refs, 320);
        tok.save(&dir.join("tokenizer.json")).unwrap();

        let cfg = TransformerConfig {
            vocab_size: tok.vocab.len(),
            max_seq_len: 48,
            d_model: 32,
            n_heads: 4,
            n_layers: 1,
            d_ff: 64,
        };
        let model = ServerMindTransformer::new_random(cfg, 7).unwrap();
        save_checkpoint(&dir.join("best.bin"), &model, None, 10, 2.0).unwrap();

        let engine = NeuralEngine::load_from_dir(&dir).unwrap();
        let summary = evaluate_neural_engine(&engine, &corpus);
        assert_eq!(summary.cases.len(), 6);
        assert!(summary.val_loss >= 0.0);
        assert!(summary.perplexity >= 1.0);

        let _ = fs::remove_dir_all(&dir);
    }
}
