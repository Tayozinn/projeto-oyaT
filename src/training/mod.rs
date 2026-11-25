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

use crate::neural::gpu::{GPU_AUTO_MIN_PARAMS};
use crate::neural::{
    load_checkpoint, save_checkpoint, AdamWState, ComputeDevice, ServerMindTransformer,
    TrainingWorkspace, TransformerGrads, VulkanDevice, VulkanTrainingSession,
};
use crate::tokenizer::BpeTokenizer;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_IN_MEMORY_HISTORY: usize = 4_096;
const MAX_CHECKPOINT_WALL_SECS: u64 = 300; // Salva no máximo a cada 5 minutos em treinos longos

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
    #[serde(default)]
    pub device: ComputeDevice,
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
            device: ComputeDevice::Auto,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainStepMetrics {
    pub step: usize,
    pub train_loss: f32,
    pub val_loss: Option<f32>,
    #[serde(default)]
    pub grad_norm: f32,
    pub learning_rate: f32,
    pub tokens_processed: usize,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    pub run_name: String,
    #[serde(default)]
    pub active_device: String,
    pub steps_completed: usize,
    pub initial_train_loss: f32,
    pub final_train_loss: f32,
    pub best_val_loss: f32,
    pub total_tokens: usize,
    pub total_elapsed_ms: u128,
    pub tokens_per_sec: f64,
    #[serde(default)]
    pub stopped_by_duration: bool,
    #[serde(default)]
    pub cancelled: bool,
    pub history: Vec<TrainStepMetrics>,
}

/// Opções de controle por tempo de relógio (`--max-duration`, ex.: `6h`), cancelamento limpo (`Ctrl+C`)
/// e callbacks periódicos de marco com os pesos já sincronizados da GPU para geração de amostras ao vivo.
pub struct TimedSessionOptions<'a> {
    pub max_duration: Option<Duration>,
    pub eval_every_secs: Option<u64>,
    pub cancel_flag: Option<Arc<AtomicBool>>,
    pub install_ctrlc_handler: bool,
    pub on_milestone: Option<&'a mut dyn FnMut(&ServerMindTransformer, &TrainStepMetrics, f64, Duration)>,
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
/// (entrada `0..seq_len`, alvo `1..seq_len+1`). Em corpora pequenos usa stride de 50%;
/// em corpora grandes (`>= 64` janelas) usa janelas disjuntas (`stride = seq_len`).
pub fn pack_sequences(
    tokenizer: &BpeTokenizer,
    texts: &[String],
    seq_len: usize,
) -> Vec<Vec<usize>> {
    let eff_seq = seq_len.max(8);
    pack_sequences_internal(tokenizer, texts, eff_seq, None)
}

/// Empacota janelas de validação sem sobreposição (`stride = seq_len`) para avaliação imparcial
/// (evitando dupla contagem de tokens centrais e duplicação de cauda).
pub fn pack_eval_sequences(
    tokenizer: &BpeTokenizer,
    texts: &[String],
    seq_len: usize,
) -> Vec<Vec<usize>> {
    let eff_seq = seq_len.max(8);
    pack_sequences_with_stride(tokenizer, texts, eff_seq, eff_seq)
}

/// Empacota textos tokenizados em janelas de `seq_len.max(8) + 1` usando passo explícito `stride`,
/// sem jamais duplicar a última janela quando divisível pelo stride.
pub fn pack_sequences_with_stride(
    tokenizer: &BpeTokenizer,
    texts: &[String],
    seq_len: usize,
    stride: usize,
) -> Vec<Vec<usize>> {
    let eff_seq = seq_len.max(8);
    pack_sequences_internal(tokenizer, texts, eff_seq, Some(stride.max(1)))
}

fn pack_sequences_internal(
    tokenizer: &BpeTokenizer,
    texts: &[String],
    eff_seq: usize,
    explicit_stride: Option<usize>,
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

    let window_len = eff_seq + 1;
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

    let stride = explicit_stride.unwrap_or_else(|| {
        if stream.len() >= window_len * 64 {
            eff_seq
        } else {
            (window_len / 2).max(1)
        }
    });

    let mut start = 0usize;
    let mut last_start = 0usize;
    while start + window_len <= stream.len() {
        windows.push(stream[start..start + window_len].to_vec());
        last_start = start;
        start += stride;
    }
    let tail_start = stream.len() - window_len;
    if tail_start > last_start {
        windows.push(stream[tail_start..].to_vec());
    }
    windows
}

pub fn cosine_lr_temporal(
    elapsed: Duration,
    target: Duration,
    max_lr: f32,
    min_lr: f32,
) -> f32 {
    let total = target.as_secs_f64().max(1e-3);
    let cur = elapsed.as_secs_f64().clamp(0.0, total);
    let warmup = (total * 0.03).clamp(0.05, 600.0).min(total * 0.25);
    if cur < warmup {
        return max_lr * ((cur / warmup).clamp(0.05, 1.0) as f32);
    }
    let progress = ((cur - warmup) / (total - warmup).max(1e-6)).clamp(0.0, 1.0) as f32;
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    min_lr + (max_lr - min_lr) * cosine
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
/// Trunca cada janela ao limite `model.config.max_seq_len` (consistente com `forward_logits`).
pub fn evaluate_loss(model: &ServerMindTransformer, windows: &[Vec<usize>], pad_id: usize) -> f32 {
    if windows.is_empty() {
        return 0.0;
    }
    let vocab = model.config.vocab_size;
    if vocab == 0 {
        return 0.0;
    }
    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;

    for win in windows {
        if win.len() < 2 {
            continue;
        }
        let inputs = &win[..win.len() - 1];
        let targets = &win[1..];
        let logits = model.forward_logits(inputs);
        let t_eff = (logits.len() / vocab).min(targets.len());
        for (t, &target) in targets[..t_eff].iter().enumerate() {
            if target == pad_id || target >= vocab {
                continue;
            }
            let row = &logits[t * vocab..(t + 1) * vocab];
            if row.iter().any(|x| !x.is_finite()) {
                return f32::NAN;
            }
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f64;
            for &v in row {
                sum_exp += ((v - max_v) as f64).exp();
            }
            let safe_sum = sum_exp.max(1e-30);
            let log_prob = (row[target] - max_v) as f64 - safe_sum.ln();
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

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn train(
        &mut self,
        train_windows: &[Vec<usize>],
        val_windows: &[Vec<usize>],
        pad_id: usize,
    ) -> Result<TrainingReport, String> {
        self.train_with_progress(train_windows, val_windows, pad_id, None)
    }

    pub fn train_with_progress(
        &mut self,
        train_windows: &[Vec<usize>],
        val_windows: &[Vec<usize>],
        pad_id: usize,
        on_progress: Option<&mut dyn FnMut(&TrainStepMetrics, f64)>,
    ) -> Result<TrainingReport, String> {
        let opts = TimedSessionOptions {
            max_duration: None,
            eval_every_secs: None,
            cancel_flag: None,
            install_ctrlc_handler: false,
            on_milestone: None,
        };
        self.train_with_session_options(train_windows, val_windows, pad_id, on_progress, opts)
    }

    pub fn train_with_session_options<'a>(
        &mut self,
        train_windows: &[Vec<usize>],
        val_windows: &[Vec<usize>],
        pad_id: usize,
        mut on_progress: Option<&mut dyn FnMut(&TrainStepMetrics, f64)>,
        mut session_opts: TimedSessionOptions<'a>,
    ) -> Result<TrainingReport, String> {
        if train_windows.is_empty() {
            return Err("Conjunto de treinamento vazio após empacotamento.".to_string());
        }
        if session_opts.max_duration.is_none() && self.current_step >= self.config.max_steps {
            return Err(format!(
                "Passo atual do checkpoint ({}) já atingiu ou ultrapassou max_steps ({}). Informe um --steps maior.",
                self.current_step, self.config.max_steps
            ));
        }
        fs::create_dir_all(&self.config.checkpoint_dir).map_err(|e| {
            format!(
                "Falha ao criar diretório de checkpoints '{}': {}",
                self.config.checkpoint_dir.display(),
                e
            )
        })?;

        let ctrlc_guard = if session_opts.install_ctrlc_handler {
            Some(crate::eval_runner::CtrlCGuard::install())
        } else {
            None
        };
        let is_cancelled = |ext_flag: &Option<Arc<AtomicBool>>,
                            guard: &Option<crate::eval_runner::CtrlCGuard>|
         -> bool {
            ext_flag
                .as_ref()
                .map(|f| f.load(Ordering::SeqCst))
                .unwrap_or(false)
                || guard.as_ref().map(|g| g.is_triggered()).unwrap_or(false)
        };

        let metrics_path = self.config.checkpoint_dir.join("metrics.jsonl");
        let latest_path = self.config.checkpoint_dir.join("latest.bin");
        let best_path = self.config.checkpoint_dir.join("best.bin");
        if self.current_step == 0 {
            for stale in [&metrics_path, &latest_path, &best_path] {
                if stale.exists() {
                    let _ = fs::remove_file(stale);
                }
            }
        }

        let start_all = Instant::now();
        let mut gpu_session: Option<VulkanTrainingSession> = match self.config.device {
            ComputeDevice::Cpu => None,
            ComputeDevice::Gpu => {
                let dev = VulkanDevice::global()?;
                Some(VulkanTrainingSession::new(dev, &self.model, &self.optimizer)?)
            }
            ComputeDevice::Auto => {
                if self.model.config.total_params() >= GPU_AUTO_MIN_PARAMS {
                    if let Ok(dev) = VulkanDevice::global() {
                        if dev.info.is_discrete_gpu {
                            VulkanTrainingSession::new(dev, &self.model, &self.optimizer).ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        };

        let active_device = if let Some(ref gpu) = gpu_session {
            format!(
                "Vulkan GPU: {} ({:.1} MB VRAM arena)",
                gpu.device_name(),
                gpu.vram_allocated_mb()
            )
        } else {
            format!(
                "CPU ({} threads, {})",
                crate::neural::ops::worker_threads(),
                crate::neural::ops::simd_backend_name()
            )
        };

        let mut grads = if gpu_session.is_none() {
            Some(TransformerGrads::zeros_like(&self.model.config))
        } else {
            None
        };
        let mut workspace = if gpu_session.is_none() {
            Some(TrainingWorkspace::new(&self.model.config))
        } else {
            None
        };
        let mut host_synced = true;
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

        let mut stopped_by_duration = false;
        let mut cancelled = false;
        let mut last_eval_wall = Instant::now();
        let mut last_ckpt_wall = Instant::now();

        loop {
            if session_opts.max_duration.is_none() && self.current_step >= self.config.max_steps {
                break;
            }
            if let Some(max_dur) = session_opts.max_duration {
                if start_all.elapsed() >= max_dur && self.current_step > 0 {
                    stopped_by_duration = true;
                    break;
                }
                if self.current_step >= self.config.max_steps {
                    break;
                }
            }
            if is_cancelled(&session_opts.cancel_flag, &ctrlc_guard) {
                cancelled = true;
                break;
            }

            let step_start = Instant::now();
            if let Some(ref mut gpu) = gpu_session {
                gpu.zero_grads()?;
            } else if let Some(ref mut g) = grads {
                g.zero_out();
            }

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
                if win.len() < 2 {
                    continue;
                }
                let inputs = &win[..win.len() - 1];
                let targets = &win[1..];
                let (loss_sum, count) = if let Some(ref mut gpu) = gpu_session {
                    gpu.forward_and_backward_sample(inputs, targets, pad_id)?
                } else {
                    self.model.forward_and_backward_with_workspace(
                        inputs,
                        targets,
                        pad_id,
                        grads.as_mut().expect("CPU grads"),
                        workspace.as_mut().expect("CPU workspace"),
                    )
                };
                step_loss_sum += loss_sum;
                step_tokens += count;
            }

            if !step_loss_sum.is_finite() {
                return Err(format!(
                    "Divergência numérica detectada no passo {}: loss não-finita ({}). Pesos preservados.",
                    self.current_step + 1,
                    step_loss_sum
                ));
            }

            let lr = if let Some(max_dur) = session_opts.max_duration {
                cosine_lr_temporal(
                    start_all.elapsed(),
                    max_dur,
                    self.config.learning_rate,
                    self.config.min_learning_rate,
                )
            } else {
                cosine_lr(
                    self.current_step,
                    self.config.warmup_steps,
                    self.config.max_steps,
                    self.config.learning_rate,
                    self.config.min_learning_rate,
                )
            };

            let (train_loss, grad_norm) = if step_tokens > 0 {
                if let Some(ref mut gpu) = gpu_session {
                    match gpu.step_adamw(
                        &mut self.optimizer.step,
                        step_tokens,
                        lr,
                        self.config.weight_decay,
                        self.config.max_grad_norm,
                    )? {
                        Some(gn) => {
                            host_synced = false;
                            (step_loss_sum / step_tokens as f32, gn)
                        }
                        None => {
                            return Err(format!(
                                "Divergência numérica detectada no passo {}: gradiente não-finito (NaN/Inf). Pesos preservados.",
                                self.current_step + 1
                            ));
                        }
                    }
                } else {
                    let g = grads.as_mut().expect("CPU grads");
                    g.scale(1.0 / step_tokens as f32);
                    let gn = g.l2_norm();
                    if !self.optimizer.step(
                        &mut self.model,
                        g,
                        lr,
                        self.config.weight_decay,
                        self.config.max_grad_norm,
                    ) {
                        return Err(format!(
                            "Divergência numérica detectada no passo {}: gradiente não-finito (NaN/Inf). Pesos preservados.",
                            self.current_step + 1
                        ));
                    }
                    (step_loss_sum / step_tokens as f32, gn)
                }
            } else {
                (0.0, 0.0)
            };

            if initial_train_loss.is_none() {
                initial_train_loss = Some(train_loss);
            }
            final_train_loss = train_loss;

            self.current_step += 1;
            self.total_tokens += step_tokens;

            let time_eval_due = session_opts
                .eval_every_secs
                .map(|s| s > 0 && last_eval_wall.elapsed().as_secs() >= s)
                .unwrap_or(false);
            let step_eval_due = self.config.eval_interval > 0
                && (self.current_step % self.config.eval_interval == 0
                    || (session_opts.max_duration.is_none()
                        && self.current_step == self.config.max_steps));
            let dur_just_reached = session_opts
                .max_duration
                .map(|d| start_all.elapsed() >= d)
                .unwrap_or(false);
            let ctrlc_now = is_cancelled(&session_opts.cancel_flag, &ctrlc_guard);

            let should_eval = time_eval_due
                || step_eval_due
                || ((dur_just_reached || ctrlc_now) && self.config.eval_interval > 0);

            let val_loss = if should_eval && !eval_subset.is_empty() {
                last_eval_wall = Instant::now();
                let v = if let Some(ref mut gpu) = gpu_session {
                    gpu.evaluate_loss(&eval_subset, pad_id)?
                } else {
                    evaluate_loss(&self.model, &eval_subset, pad_id)
                };
                if v < self.best_val_loss {
                    self.best_val_loss = v;
                    if let Some(ref mut gpu) = gpu_session {
                        if !host_synced {
                            gpu.sync_to_host(&mut self.model, &mut self.optimizer)?;
                            host_synced = true;
                        }
                    }
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

            let wall_ckpt_due = session_opts.max_duration.is_some()
                && last_ckpt_wall.elapsed().as_secs() >= MAX_CHECKPOINT_WALL_SECS;
            let step_ckpt_due = self.config.checkpoint_interval > 0
                && (self.current_step % self.config.checkpoint_interval == 0
                    || (session_opts.max_duration.is_none()
                        && self.current_step == self.config.max_steps));

            if step_ckpt_due || wall_ckpt_due || dur_just_reached || ctrlc_now {
                last_ckpt_wall = Instant::now();
                if let Some(ref mut gpu) = gpu_session {
                    if !host_synced {
                        gpu.sync_to_host(&mut self.model, &mut self.optimizer)?;
                        host_synced = true;
                    }
                }
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
                grad_norm,
                learning_rate: lr,
                tokens_processed: step_tokens,
                elapsed_ms: step_start.elapsed().as_millis(),
            };
            append_metric_jsonl(&metrics_path, &step_metric)?;
            if should_eval {
                let secs_so_far = start_all.elapsed().as_secs_f64().max(1e-6);
                let tok_per_sec = self.total_tokens as f64 / secs_so_far;
                if let Some(cb) = on_progress.as_deref_mut() {
                    cb(&step_metric, tok_per_sec);
                }
                if let Some(m_cb) = session_opts.on_milestone.as_deref_mut() {
                    if let Some(ref mut gpu) = gpu_session {
                        if !host_synced {
                            gpu.sync_to_host(&mut self.model, &mut self.optimizer)?;
                            host_synced = true;
                        }
                    }
                    m_cb(&self.model, &step_metric, tok_per_sec, start_all.elapsed());
                }
            }
            if history.len() < MAX_IN_MEMORY_HISTORY || should_eval {
                history.push(step_metric);
            }

            if dur_just_reached {
                stopped_by_duration = true;
                break;
            }
            if ctrlc_now {
                cancelled = true;
                break;
            }
        }

        if let Some(ref mut gpu) = gpu_session {
            if !host_synced {
                gpu.sync_to_host(&mut self.model, &mut self.optimizer)?;
            }
        }

        if self.current_step > 0 {
            save_checkpoint(
                &latest_path,
                &self.model,
                Some(&self.optimizer),
                self.current_step,
                self.best_val_loss,
            )?;
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
            active_device,
            steps_completed: self.current_step,
            initial_train_loss: initial_train_loss.unwrap_or(final_train_loss),
            final_train_loss,
            best_val_loss: self.best_val_loss,
            total_tokens: self.total_tokens,
            total_elapsed_ms,
            tokens_per_sec: self.total_tokens as f64 / secs,
            stopped_by_duration,
            cancelled,
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
            device: ComputeDevice::Cpu,
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

    #[test]
    fn evaluate_loss_truncates_windows_longer_than_max_seq_len_and_skips_zero_token_step() {
        let cfg = TransformerConfig {
            vocab_size: 16,
            max_seq_len: 4,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 16,
        };
        let model = ServerMindTransformer::new_random(cfg, 21).unwrap();
        // Janela com 9 tokens (> max_seq_len = 4), como gerada por `pack_sequences(..., 4)`
        let long_win = vec![1usize, 2, 3, 4, 5, 6, 7, 8, 9];
        let loss = evaluate_loss(&model, &[long_win], 0);
        assert!(loss.is_finite() && loss > 0.0);

        // Lote composto apenas por `pad_id` (step_tokens == 0) não deve avançar `optimizer.step`
        // nem aplicar weight decay espúrio nos pesos.
        let orig_head = model.lm_head.clone();
        let dir = temp_dir("zero_tok");
        let train_cfg = TrainingConfig {
            run_name: "zero-tok-test".to_string(),
            checkpoint_dir: dir.clone(),
            max_steps: 2,
            batch_size: 2,
            seq_len: 4,
            learning_rate: 1e-2,
            min_learning_rate: 1e-3,
            warmup_steps: 1,
            weight_decay: 0.1,
            max_grad_norm: 1.0,
            eval_interval: 0,
            checkpoint_interval: 2,
            max_val_windows: 8,
            device: ComputeDevice::Cpu,
        };
        let mut trainer = Trainer::new(model, train_cfg);
        let pad_windows = vec![vec![0usize, 0, 0, 0, 0]];
        let report = trainer.train(&pad_windows, &[], 0).unwrap();
        assert_eq!(report.total_tokens, 0);
        assert_eq!(trainer.optimizer.step, 0);
        assert_eq!(trainer.model.lm_head, orig_head);

        // `best.bin` salvo sem validação (`best_val_loss = INFINITY`) deve recarregar sem erro
        let (_, _, meta) = crate::neural::load_checkpoint(&dir.join("best.bin")).unwrap();
        assert!(meta.best_val_loss.is_infinite());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pack_sequences_never_duplicates_tail_window_and_records_grad_norm() {
        let corpus = ["a b c d e f g h i j k l m n o p q r s t u v w x y z".to_string()];
        let refs = [corpus[0].as_str()];
        let tok = BpeTokenizer::train_from_corpus(&refs, 280);

        for seq_len in [8usize, 9, 10, 12, 16] {
            let wins = pack_sequences_with_stride(&tok, &corpus, seq_len, 4);
            if wins.len() >= 2 {
                assert_ne!(
                    wins[wins.len() - 1],
                    wins[wins.len() - 2],
                    "janela final duplicada para seq_len={}",
                    seq_len
                );
            }
            let eval_wins = pack_eval_sequences(&tok, &corpus, seq_len);
            if eval_wins.len() >= 2 {
                assert_ne!(eval_wins[eval_wins.len() - 1], eval_wins[eval_wins.len() - 2]);
            }
        }

        // Verifica registro de `grad_norm > 0` e limpeza de checkpoint obsoleto em novo treino
        let cfg = TransformerConfig {
            vocab_size: tok.vocab_size(),
            max_seq_len: 12,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
        };
        let model = ServerMindTransformer::new_random(cfg, 77).unwrap();
        let dir = temp_dir("grad_norm_and_clean");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("best.bin"), b"stale_corrupt_bytes").unwrap();

        let train_cfg = TrainingConfig {
            run_name: "sm002-gn".to_string(),
            checkpoint_dir: dir.clone(),
            max_steps: 2,
            batch_size: 1,
            seq_len: 12,
            learning_rate: 1e-3,
            min_learning_rate: 1e-4,
            warmup_steps: 1,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            eval_interval: 1,
            checkpoint_interval: 2,
            max_val_windows: 8,
            device: ComputeDevice::Cpu,
        };
        let wins = pack_sequences(&tok, &corpus, 12);
        let mut progress_hits = 0usize;
        let mut trainer = Trainer::new(model, train_cfg);
        let mut cb = |_m: &TrainStepMetrics, _tps: f64| {
            progress_hits += 1;
        };
        let report = trainer
            .train_with_progress(&wins, &wins, tok.pad_id(), Some(&mut cb))
            .unwrap();
        assert_eq!(progress_hits, 2);
        assert!(report.history.iter().all(|h| h.grad_norm > 0.0 && h.grad_norm.is_finite()));

        // O `best.bin` corrompido antigo foi substituído por um checkpoint válido
        assert!(crate::neural::load_checkpoint(&dir.join("best.bin")).is_ok());

        // `current_step >= max_steps` deve retornar erro explícito
        assert!(trainer.train(&wins, &wins, tok.pad_id()).is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sm009_timed_marathon_and_cancel_flag_generate_evolution_sequence_and_clean_checkpoint() {
        let dir = temp_dir("sm009_marathon");
        let q_dir = temp_dir("sm009_marathon_int8");
        let ds_file = dir.join("mini_qa.jsonl");
        let books_dir = dir.join("mini_books");
        fs::create_dir_all(&books_dir).unwrap();
        fs::write(
            &ds_file,
            "{\"domain\":\"linux\",\"prompt\":\"Como checar portas?\",\"response\":\"Use ss -tulnp.\"}\n\
             {\"domain\":\"docker\",\"prompt\":\"Container em loop?\",\"response\":\"Use docker logs.\"}\n\
             {\"domain\":\"network\",\"prompt\":\"Erro 502 no Nginx?\",\"response\":\"Cheque systemctl status nginx.\"}\n",
        )
        .unwrap();
        fs::write(
            books_dir.join("cap1.txt"),
            "Servidores Linux utilizam systemd, journalctl, ss e nginx para operação confiável em produção.",
        )
        .unwrap();

        let out_ckpt = dir.join("ckpt");
        let args = vec![
            "--marathon".to_string(),
            "--max-duration".to_string(),
            "2s".to_string(),
            "--eval-every-secs".to_string(),
            "1".to_string(),
            "--dataset".to_string(),
            ds_file.to_string_lossy().to_string(),
            "--corpus-dir".to_string(),
            books_dir.to_string_lossy().to_string(),
            "--out".to_string(),
            out_ckpt.to_string_lossy().to_string(),
            "--export-quant".to_string(),
            q_dir.to_string_lossy().to_string(),
            "--device".to_string(),
            "cpu".to_string(),
            "--d-model".to_string(),
            "32".to_string(),
            "--layers".to_string(),
            "1".to_string(),
            "--heads".to_string(),
            "4".to_string(),
            "--ff".to_string(),
            "64".to_string(),
            "--seq-len".to_string(),
            "32".to_string(),
            "--vocab".to_string(),
            "300".to_string(),
            "--batch-size".to_string(),
            "2".to_string(),
        ];
        let mut out_buf = Vec::new();
        let mut err_buf = Vec::new();
        let code = crate::cli::run_train_neural_cli_with_cancel(
            &args,
            false,
            None,
            &mut out_buf,
            &mut err_buf,
        );
        assert_eq!(
            code,
            0,
            "stderr={}",
            String::from_utf8_lossy(&err_buf)
        );
        assert!(out_ckpt.join("best.bin").is_file());
        assert!(out_ckpt.join("latest.bin").is_file());
        assert!(out_ckpt.join("training_evolution.md").is_file());
        assert!(out_ckpt.join("evolution_log.jsonl").is_file());
        assert!(q_dir.join("quantized.smq").is_file());

        let md = fs::read_to_string(out_ckpt.join("training_evolution.md")).unwrap();
        assert!(md.contains("Marco #00"));
        assert!(md.contains("Marco Final Consolidado"));

        // Testa parada limpa por `cancel_flag` salvando checkpoint íntegro com código 130
        let cancel_flag = Arc::new(AtomicBool::new(true));
        let cancel_dir = dir.join("ckpt_cancel");
        let cancel_args = vec![
            "--dataset".to_string(),
            ds_file.to_string_lossy().to_string(),
            "--out".to_string(),
            cancel_dir.to_string_lossy().to_string(),
            "--max-duration".to_string(),
            "10s".to_string(),
            "--device".to_string(),
            "cpu".to_string(),
            "--d-model".to_string(),
            "16".to_string(),
            "--layers".to_string(),
            "1".to_string(),
            "--heads".to_string(),
            "2".to_string(),
            "--ff".to_string(),
            "32".to_string(),
            "--seq-len".to_string(),
            "16".to_string(),
        ];
        let code_cancel = crate::cli::run_train_neural_cli_with_cancel(
            &cancel_args,
            false,
            Some(cancel_flag),
            &mut out_buf,
            &mut err_buf,
        );
        assert_eq!(code_cancel, 130);
        assert!(cancel_dir.join("best.bin").is_file());

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&q_dir);
    }
}
