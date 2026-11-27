//! Agente 7 — Avaliação e Qualidade do ServerMind Neural.
//!
//! Avalia versões próprias do modelo neural em tarefas técnicas (Linux, Docker, Redes,
//! Segurança, Depuração e Programação), mede perplexity/val_loss, latência de inferência
//! e compara objetivamente versões entre si e com o baseline Ollama.

use crate::neural::inference::{GenerationConfig, NeuralEngine};
use crate::neural::training::{evaluate_loss, pack_eval_sequences};
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
    pub generated_tokens: usize,
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
    pub tokens_per_sec: f64,
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
    let seq_len = engine.model.config.max_seq_len.max(8);
    let val_windows = pack_eval_sequences(&engine.tokenizer, validation_texts, seq_len);
    let val_loss = evaluate_loss(&engine.model, &val_windows, engine.tokenizer.pad_id())
        .filter(|v| v.is_finite())
        .unwrap_or(engine.best_val_loss);
    let perplexity = if val_loss.is_finite() {
        (val_loss as f64).exp().min(f32::MAX as f64) as f32
    } else {
        f32::INFINITY
    };

    let suite = default_technical_suite();
    let mut results = Vec::with_capacity(suite.len());
    let mut total_hits = 0usize;
    let mut total_expected = 0usize;
    let mut total_latency = 0u128;
    let mut total_gen_tokens = 0usize;

    for (idx, case) in suite.into_iter().enumerate() {
        let gen_cfg = GenerationConfig {
            max_new_tokens: 40,
            temperature: 0.7,
            top_k: 16,
            // Seed determinística por caso para reprodutibilidade exata entre execuções
            seed: Some(0x5EED_2026_u64.wrapping_add(idx as u64)),
        };
        let t0 = Instant::now();
        let (reply, generated_tokens) =
            engine.generate_with_config(&[], &case.prompt, gen_cfg, None);
        let latency_ms = t0.elapsed().as_millis();
        let lower = reply.to_ascii_lowercase();
        let hits = case
            .expected_keywords
            .iter()
            .filter(|kw| contains_keyword_boundary(&lower, &kw.to_ascii_lowercase()))
            .count();
        total_hits += hits;
        total_expected += case.expected_keywords.len();
        total_latency += latency_ms;
        total_gen_tokens += generated_tokens;

        let preview: String = reply.chars().take(160).collect();
        results.push(NeuralCaseResult {
            id: case.id,
            domain: case.domain,
            latency_ms,
            generated_tokens,
            keyword_hits: hits,
            expected_keywords: case.expected_keywords.len(),
            output_preview: preview,
        });
    }

    let n = results.len().max(1) as f64;
    let total_secs = (total_latency as f64 / 1000.0).max(1e-6);
    NeuralEvalSummary {
        checkpoint_dir: engine.checkpoint_dir.to_string_lossy().to_string(),
        step: engine.step,
        total_params: engine.model.config.total_params(),
        val_loss,
        perplexity,
        avg_latency_ms: total_latency as f64 / n,
        tokens_per_sec: total_gen_tokens as f64 / total_secs,
        keyword_recall: if total_expected == 0 {
            0.0
        } else {
            total_hits as f64 / total_expected as f64
        },
        cases: results,
    }
}

/// Verifica presença de `keyword` respeitando fronteira alfanumérica (evita falsos positivos
/// como `"ss"` dentro de `"processo"`/`"passos"`, `"ps"` em `"https"` ou `"nc"` em `"banco"`).
pub fn contains_keyword_boundary(haystack_lower: &str, kw_lower: &str) -> bool {
    if kw_lower.is_empty() {
        return false;
    }
    let mut start = 0usize;
    while let Some(rel) = haystack_lower[start..].find(kw_lower) {
        let abs_start = start + rel;
        let abs_end = abs_start + kw_lower.len();
        let prev_ok = haystack_lower[..abs_start]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric() && c != '_');
        let next_ok = haystack_lower[abs_end..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric() && c != '_');
        if prev_ok && next_ok {
            return true;
        }
        start = abs_start + kw_lower.chars().next().map_or(1, |c| c.len_utf8());
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::tokenizer::BpeTokenizer;
    use crate::neural::{save_checkpoint, ServerMindTransformer, TransformerConfig};
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
        assert!(summary.tokens_per_sec >= 0.0);

        // Determinismo: duas avaliações seguidas sobre o mesmo checkpoint produzem os mesmos previews e recall
        let summary2 = evaluate_neural_engine(&engine, &corpus);
        assert_eq!(summary.keyword_recall, summary2.keyword_recall);
        for (c1, c2) in summary.cases.iter().zip(summary2.cases.iter()) {
            assert_eq!(c1.output_preview, c2.output_preview);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn keyword_boundary_rejects_substring_false_positives() {
        assert!(!contains_keyword_boundary(
            "verifique o processo e siga os passos de acesso",
            "ss"
        ));
        assert!(!contains_keyword_boundary(
            "acesse https://exemplo.local",
            "ps"
        ));
        assert!(!contains_keyword_boundary("o banco iniciou", "nc"));
        assert!(!contains_keyword_boundary(
            "use systemctl stop nginx",
            "top"
        ));

        assert!(contains_keyword_boundary("execute `ss -tulnp` agora", "ss"));
        assert!(contains_keyword_boundary("rode ps aux e top -b", "ps"));
        assert!(contains_keyword_boundary("rode ps aux e top -b", "top"));
        assert!(contains_keyword_boundary("teste porta com nc -zv", "nc"));
    }
}
