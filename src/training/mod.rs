//! Agente 4 — Sistema de Treinamento Progressivo do ServerMind Neural.
//!
//! Implementa:
//! - Empacotamento de sequências (`pack_sequences`) para Language Modeling causal
//! - Mini-batches com acumulação de gradientes e clipping de norma
//! - Agendamento de learning rate (linear warmup + cosine decay)
//! - Avaliação periódica no conjunto de validação
//! - Checkpoints `latest.bin` e `best.bin` + log `metrics.jsonl`
//! - Recuperação de treinamentos interrompidos (`resume_from_checkpoint`)
//! - Comparação entre execuções (`compare_training_runs`).

use crate::neural::{
    load_checkpoint, save_checkpoint, AdamWState, ServerMindTransformer, TransformerGrads,
};
use crate::tokenizer::BpeTokenizer;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn default_max_val_windows() -> usize {
    128
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub run_name: String,
    pub checkpoint_dir: PathBuf,
    pub max_steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub learning_rate: f32,
    pub min_learning_rate: f32,
    pub warmup_steps: usize,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub eval_interval: usize,
    pub checkpoint_interval: usize,
    #[serde(default = "default_max_val_windows")]
    pub max_val_windows: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            run_name: "servermind-neural-v1".to_string(),
            checkpoint_dir: PathBuf::from("checkpoints/servermind-neural-v1"),
            max_steps: 200,
            batch_size: 4,
            seq_len: 96,
            learning_rate: 3e-3,
            min_learning_rate: 3e-4,
            warmup_steps: 20,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            eval_interval: 25,
            checkpoint_interval: 50,
            max_val_windows: default_max_val_windows(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainStepMetrics {
    pub step: usize,
    pub train_loss: f32,
    pub val_loss: Option<f32>,
    pub learning_rate: f32,
    pub tokens_processed: usize,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    pub run_name: String,
    pub steps_completed: usize,
    pub initial_train_loss: f32,
    pub final_train_loss: f32,
    pub best_val_loss: f32,
    pub total_tokens: usize,
    pub total_elapsed_ms: u128,
    pub tokens_per_sec: f64,
    pub history: Vec<TrainStepMetrics>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunComparison {
    pub run_a_name: String,
    pub run_b_name: String,
    pub run_a_best_val: f32,
    pub run_b_best_val: f32,
    pub delta_val_loss: f32,
    pub summary: String,
}

/// Empacota textos tokenizados em janelas contíguas de tamanho `seq_len + 1`
/// (entrada `0..seq_len`, alvo `1..seq_len+1`).
pub fn pack_sequences(
    tokenizer: &BpeTokenizer,
    texts: &[String],
    seq_len: usize,
) -> Vec<Vec<usize>> {
    let mut stream = Vec::new();
    for text in texts {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        let ids = tokenizer.encode(trimmed, true, true);
        if ids.len() >= 2 {
            stream.extend_from_slice(&ids);
        }
    }

    let window_len = seq_len.max(8) + 1;
    let mut windows = Vec::new();
    if stream.len() < 2 {
        return windows;
    }

    if stream.len() < window_len {
        let mut padded = stream;
        while padded.len() < window_len {
            padded.push(tokenizer.pad_id());
        }
        windows.push(padded);
        return windows;
    }

    let stride = (window_len / 2).max(1);
    let mut start = 0;
    while start + window_len <= stream.len() {
        windows.push(stream[start..start + window_len].to_vec());
        start += stride;
    }
    if start < stream.len() && stream.len() >= window_len {
        windows.push(stream[stream.len() - window_len..].to_vec());
    }
    windows
}

pub fn cosine_lr(
    step: usize,
    warmup_steps: usize,
    max_steps: usize,
    max_lr: f32,
    min_lr: f32,
) -> f32 {
    if warmup_steps > 0 && step < warmup_steps {
        return max_lr * ((step + 1) as f32 / warmup_steps as f32);
    }
    let decay_steps = max_steps.saturating_sub(warmup_steps).max(1);
    let progress =
        ((step.saturating_sub(warmup_steps)) as f32 / decay_steps as f32).clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    min_lr + (max_lr - min_lr) * cosine
}

/// Avalia perda média de cross-entropy em janelas de validação sem alterar pesos.
pub fn evaluate_loss(model: &ServerMindTransformer, windows: &[Vec<usize>], pad_id: usize) -> f32 {
    if windows.is_empty() {
        return 0.0;
    }
    let vocab = model.config.vocab_size;
    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;

    for win in windows {
        if win.len() < 2 {
            continue;
        }
        let inputs = &win[..win.len() - 1];
        let targets = &win[1..];
        let logits = model.forward_logits(inputs);
        for (t, &target) in targets.iter().enumerate() {
            if target == pad_id || target >= vocab {
                continue;
            }
            let row = &logits[t * vocab..(t + 1) * vocab];
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f64;
            for &v in row {
                sum_exp += ((v - max_v) as f64).exp();
            }
            let log_prob = (row[target] - max_v) as f64 - sum_exp.ln();
            total_loss += -log_prob;
            total_tokens += 1;
        }
    }

    if total_tokens == 0 {
        0.0
    } else {
        (total_loss / total_tokens as f64) as f32
    }
}

pub struct Trainer {
    pub model: ServerMindTransformer,
    pub optimizer: AdamWState,
    pub config: TrainingConfig,
    pub current_step: usize,
    pub best_val_loss: f32,
    pub total_tokens: usize,
}

impl Trainer {
    pub fn new(model: ServerMindTransformer, config: TrainingConfig) -> Self {
        let optimizer = AdamWState::new(&model);
        Self {
            model,
            optimizer,
            config,
            current_step: 0,
            best_val_loss: f32::INFINITY,
            total_tokens: 0,
        }
    }

    #[allow(dead_code)]
    pub fn resume_from_checkpoint(
        checkpoint_path: &Path,
        config: TrainingConfig,
    ) -> Result<Self, String> {
        let (model, opt_opt, meta) = load_checkpoint(checkpoint_path)?;
        let optimizer = opt_opt.unwrap_or_else(|| AdamWState::new(&model));
        Ok(Self {
            model,
            optimizer,
            config,
            current_step: meta.step,
            best_val_loss: meta.best_val_loss,
            total_tokens: 0,
        })
    }

    pub fn train(
        &mut self,
        train_windows: &[Vec<usize>],
        val_windows: &[Vec<usize>],
        pad_id: usize,
    ) -> Result<TrainingReport, String> {
        if train_windows.is_empty() {
            return Err("Conjunto de treinamento vazio após empacotamento.".to_string());
        }
        fs::create_dir_all(&self.config.checkpoint_dir).map_err(|e| {
            format!(
                "Falha ao criar diretório de checkpoints '{}': {}",
                self.config.checkpoint_dir.display(),
                e
            )
        })?;

        let metrics_path = self.config.checkpoint_dir.join("metrics.jsonl");
        if self.current_step == 0 && metrics_path.exists() {
            let _ = fs::remove_file(&metrics_path);
        }
        let latest_path = self.config.checkpoint_dir.join("latest.bin");
        let best_path = self.config.checkpoint_dir.join("best.bin");

        let start_all = Instant::now();
        let mut grads = TransformerGrads::zeros_like(&self.model.config);
        let mut history = Vec::new();
        let mut initial_train_loss = None;
        let mut final_train_loss = 0.0f32;
        let batch_size = self.config.batch_size.max(1);
        let total_seen_windows = self.current_step * batch_size;
        let mut epoch = total_seen_windows / train_windows.len();
        let mut epoch_pos = total_seen_windows % train_windows.len();
        let mut order = shuffled_indices(train_windows.len(), 0x5EED_2026_u64 ^ (epoch as u64));

        // Subamostragem uniforme determinística das janelas de validação para evitar gargalo em corpora grandes.
        let eval_subset: Vec<Vec<usize>> = select_eval_windows(val_windows, self.config.max_val_windows);

        while self.current_step < self.config.max_steps {
            let step_start = Instant::now();
            grads.zero_out();

            let mut step_loss_sum = 0.0f32;
            let mut step_tokens = 0usize;

            for _ in 0..batch_size {
                if epoch_pos >= order.len() {
                    epoch += 1;
                    epoch_pos = 0;
                    order = shuffled_indices(train_windows.len(), 0x5EED_2026_u64 ^ (epoch as u64));
                }
                let win = &train_windows[order[epoch_pos]];
                epoch_pos += 1;
                let inputs = &win[..win.len() - 1];
                let targets = &win[1..];
                let (loss_sum, count) = self
                    .model
                    .forward_and_backward(inputs, targets, pad_id, &mut grads);
                step_loss_sum += loss_sum;
                step_tokens += count;
            }

            if step_tokens > 0 {
                grads.scale(1.0 / step_tokens as f32);
            }
            let train_loss = if step_tokens > 0 {
                step_loss_sum / step_tokens as f32
            } else {
                0.0
            };
            if initial_train_loss.is_none() {
                initial_train_loss = Some(train_loss);
            }
            final_train_loss = train_loss;

            let lr = cosine_lr(
                self.current_step,
                self.config.warmup_steps,
                self.config.max_steps,
                self.config.learning_rate,
                self.config.min_learning_rate,
            );
            self.optimizer.step(
                &mut self.model,
                &mut grads,
                lr,
                self.config.weight_decay,
                self.config.max_grad_norm,
            );

            self.current_step += 1;
            self.total_tokens += step_tokens;

            let should_eval = self.config.eval_interval > 0
                && (self.current_step % self.config.eval_interval == 0
                    || self.current_step == self.config.max_steps);
            let val_loss = if should_eval && !eval_subset.is_empty() {
                let v = evaluate_loss(&self.model, &eval_subset, pad_id);
                if v < self.best_val_loss {
                    self.best_val_loss = v;
                    save_checkpoint(
                        &best_path,
                        &self.model,
                        Some(&self.optimizer),
                        self.current_step,
                        self.best_val_loss,
                    )?;
                }
                Some(v)
            } else {
                None
            };

            if self.config.checkpoint_interval > 0
                && (self.current_step % self.config.checkpoint_interval == 0
                    || self.current_step == self.config.max_steps)
            {
                save_checkpoint(
                    &latest_path,
                    &self.model,
                    Some(&self.optimizer),
                    self.current_step,
                    self.best_val_loss,
                )?;
            }

            let step_metric = TrainStepMetrics {
                step: self.current_step,
                train_loss,
                val_loss,
                learning_rate: lr,
                tokens_processed: step_tokens,
                elapsed_ms: step_start.elapsed().as_millis(),
            };
            append_metric_jsonl(&metrics_path, &step_metric)?;
            history.push(step_metric);
        }

        if !best_path.exists() {
            save_checkpoint(
                &best_path,
                &self.model,
                Some(&self.optimizer),
                self.current_step,
                self.best_val_loss,
            )?;
        }

        let total_elapsed_ms = start_all.elapsed().as_millis();
        let secs = (total_elapsed_ms as f64 / 1000.0).max(1e-6);
        Ok(TrainingReport {
            run_name: self.config.run_name.clone(),
            steps_completed: self.current_step,
            initial_train_loss: initial_train_loss.unwrap_or(final_train_loss),
            final_train_loss,
            best_val_loss: self.best_val_loss,
            total_tokens: self.total_tokens,
            total_elapsed_ms,
            tokens_per_sec: self.total_tokens as f64 / secs,
            history,
        })
    }
}

fn shuffled_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..len).collect();
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for i in (1..len).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let j = (z as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx
}

pub fn select_eval_windows(val_windows: &[Vec<usize>], max_val_windows: usize) -> Vec<Vec<usize>> {
    if val_windows.is_empty() || max_val_windows == 0 || val_windows.len() <= max_val_windows {
        return val_windows.to_vec();
    }
    let mut out = Vec::with_capacity(max_val_windows);
    let n = val_windows.len();
    for i in 0..max_val_windows {
        let pos = (i * n) / max_val_windows;
        out.push(val_windows[pos].clone());
    }
    out
}

fn append_metric_jsonl(path: &Path, metric: &TrainStepMetrics) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("Falha ao gravar métrica em '{}': {}", path.display(), e))?;
    let line = serde_json::to_string(metric)
        .map_err(|e| format!("Falha ao serializar métrica de treino: {}", e))?;
    writeln!(file, "{}", line).map_err(|e| format!("Falha ao escrever linha de métrica: {}", e))
}

#[allow(dead_code)]
pub fn compare_training_runs(run_a: &TrainingReport, run_b: &TrainingReport) -> RunComparison {
    let delta = run_b.best_val_loss - run_a.best_val_loss;
    let verdict = if delta < -1e-4 {
        format!(
            "'{}' melhorou val_loss em {:.4} comparado a '{}'",
            run_b.run_name, -delta, run_a.run_name
        )
    } else if delta > 1e-4 {
        format!(
            "'{}' piorou val_loss em {:.4} comparado a '{}'",
            run_b.run_name, delta, run_a.run_name
        )
    } else {
        format!(
            "'{}' e '{}' empataram em val_loss ({:.4})",
            run_a.run_name, run_b.run_name, run_b.best_val_loss
        )
    };
    RunComparison {
        run_a_name: run_a.run_name.clone(),
        run_b_name: run_b.run_name.clone(),
        run_a_best_val: run_a.best_val_loss,
        run_b_best_val: run_b.best_val_loss,
        delta_val_loss: delta,
        summary: verdict,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::TransformerConfig;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("servermind_train_{}_{}", label, nanos))
    }

    #[test]
    fn training_reduces_loss_and_resumes_from_checkpoint() {
        let corpus = [
            "systemctl status nginx journalctl -u nginx --no-pager".to_string(),
            "docker ps -a docker logs api_server --tail 100".to_string(),
            "ss -tulnp df -h free -m uptime".to_string(),
        ];
        let refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        let tok = BpeTokenizer::train_from_corpus(&refs, 340);
        let windows = pack_sequences(&tok, &corpus, 20);
        assert!(!windows.is_empty());

        let cfg = TransformerConfig {
            vocab_size: tok.vocab.len(),
            max_seq_len: 20,
            d_model: 32,
            n_heads: 4,
            n_layers: 1,
            d_ff: 64,
        };
        let model = ServerMindTransformer::new_random(cfg, 42).unwrap();
        let dir = temp_dir("resume");
        let train_cfg = TrainingConfig {
            run_name: "unit-test-v1".to_string(),
            checkpoint_dir: dir.clone(),
            max_steps: 12,
            batch_size: 2,
            seq_len: 20,
            learning_rate: 1e-2,
            min_learning_rate: 1e-3,
            warmup_steps: 2,
            weight_decay: 0.0,
            max_grad_norm: 1.0,
            eval_interval: 6,
            checkpoint_interval: 6,
            max_val_windows: 64,
        };

        let mut trainer = Trainer::new(model, train_cfg.clone());
        let report1 = trainer.train(&windows, &windows, tok.pad_id()).unwrap();
        assert_eq!(report1.steps_completed, 12);
        assert!(
            report1.final_train_loss < report1.initial_train_loss,
            "Esperado queda de loss: inicial {} -> final {}",
            report1.initial_train_loss,
            report1.final_train_loss
        );
        assert!(dir.join("latest.bin").is_file());
        assert!(dir.join("best.bin").is_file());
        assert!(dir.join("metrics.jsonl").is_file());

        let mut resumed_cfg = train_cfg;
        resumed_cfg.run_name = "unit-test-v2".to_string();
        resumed_cfg.max_steps = 16;
        let mut resumed =
            Trainer::resume_from_checkpoint(&dir.join("latest.bin"), resumed_cfg).unwrap();
        assert_eq!(resumed.current_step, 12);
        let report2 = resumed.train(&windows, &windows, tok.pad_id()).unwrap();
        assert_eq!(report2.steps_completed, 16);

        let cmp = compare_training_runs(&report1, &report2);
        assert!(!cmp.summary.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn shuffle_and_eval_window_cap_are_deterministic() {
        let a = shuffled_indices(50, 42);
        let b = shuffled_indices(50, 42);
        assert_eq!(a, b);
        assert_ne!(a, (0..50).collect::<Vec<_>>());
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());

        let windows: Vec<Vec<usize>> = (0..1000).map(|i| vec![i, i + 1]).collect();
        let sampled = select_eval_windows(&windows, 64);
        assert_eq!(sampled.len(), 64);
        assert_eq!(sampled[0][0], 0);
        assert!(sampled[63][0] > 950);
    }
}
