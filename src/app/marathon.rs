use super::corpus::{check_no_existing_training_artifacts, load_seed_corpus};
use super::quantize::export_quantized_package;
use crate::cli::error::{CliError, CliResult};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

pub(crate) fn format_hms(d: Duration) -> String {
    let total = d.as_secs();
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    format!("{:02}h:{:02}m:{:02}s", h, m, s)
}

#[derive(Debug, Clone, serde::Serialize)]
struct ProbeOutputSample {
    domain: String,
    prompt: String,
    response: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct EvolutionMilestoneEntry {
    index: usize,
    phase: String,
    elapsed_hms: String,
    elapsed_secs: f64,
    step: usize,
    train_loss: Option<f32>,
    val_loss: Option<f32>,
    perplexity: Option<f32>,
    grad_norm: f32,
    learning_rate: f32,
    tokens_per_sec: f64,
    samples: Vec<ProbeOutputSample>,
}

pub(crate) struct EvolutionSequenceRecorder {
    run_name: String,
    device_hint: String,
    total_params: usize,
    planned_duration: Option<Duration>,
    pub md_path: PathBuf,
    pub jsonl_path: PathBuf,
    entries: Vec<EvolutionMilestoneEntry>,
}

impl EvolutionSequenceRecorder {
    pub(crate) fn new(
        out_dir: &std::path::Path,
        run_name: &str,
        device_hint: &str,
        total_params: usize,
        planned_duration: Option<Duration>,
    ) -> Self {
        let _ = std::fs::create_dir_all(out_dir);
        let md_path = out_dir.join("training_evolution.md");
        let jsonl_path = out_dir.join("evolution_log.jsonl");
        Self {
            run_name: run_name.to_string(),
            device_hint: device_hint.to_string(),
            total_params,
            planned_duration,
            md_path,
            jsonl_path,
            entries: Vec::new(),
        }
    }

    fn probe_prompts() -> [(&'static str, &'static str); 4] {
        [
            (
                "identity",
                "Olá! Quem criou você e no que você pode me ajudar?",
            ),
            (
                "linux",
                "Como diagnosticar uso alto de CPU e memória em um servidor Linux sem interromper processos?",
            ),
            (
                "docker",
                "Um container Docker reinicia em loop (CrashLoop). Como descobrir a causa raiz?",
            ),
            (
                "network",
                "Uma API retorna 502 Bad Gateway atrás do Nginx. Quais verificações fazer em ordem?",
            ),
        ]
    }

    fn sample_model_reply(
        model: &crate::neural::ServerMindTransformer,
        tok: &crate::neural::tokenizer::BpeTokenizer,
        prompt: &str,
        seed: u64,
    ) -> String {
        let formatted = format!("<user> {}\n<assistant> ", prompt.trim());
        let mut p_ids = tok.encode(&formatted, true, false);
        let max_p = model.config.max_seq_len.saturating_sub(40).max(8);
        if p_ids.len() > max_p {
            let start = p_ids.len() - (max_p - 1);
            let mut trunc = vec![tok.bos_id()];
            trunc.extend_from_slice(&p_ids[start..]);
            p_ids = trunc;
        }
        let mut stops = vec![tok.eos_id()];
        for sp in [
            crate::neural::tokenizer::USER_TOKEN,
            crate::neural::tokenizer::SYSTEM_TOKEN,
            crate::neural::tokenizer::ASSISTANT_TOKEN,
        ] {
            if let Some(id) = tok.token_id(sp) {
                if !stops.contains(&id) {
                    stops.push(id);
                }
            }
        }
        let gen = model.generate_with_kv_precision(
            &p_ids,
            40,
            0.2,
            5,
            &stops,
            seed,
            crate::neural::KvCachePrecision::Fp16,
            None,
        );
        let new_slice = if gen.len() > p_ids.len() {
            &gen[p_ids.len()..]
        } else {
            &[]
        };
        let clean_ids = match new_slice.last() {
            Some(last) if stops.contains(last) => &new_slice[..new_slice.len() - 1],
            _ => new_slice,
        };
        let decoded = tok
            .decode(clean_ids)
            .replace(crate::neural::tokenizer::EOS_TOKEN, "")
            .replace(crate::neural::tokenizer::PAD_TOKEN, "")
            .replace(crate::neural::tokenizer::BOS_TOKEN, "")
            .replace('\n', " ");
        let trimmed = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
        if trimmed.is_empty() {
            "(vazio)".to_string()
        } else {
            trimmed.chars().take(180).collect()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_milestone(
        &mut self,
        phase: &str,
        model: &crate::neural::ServerMindTransformer,
        tok: &crate::neural::tokenizer::BpeTokenizer,
        step: usize,
        train_loss: Option<f32>,
        val_loss: Option<f32>,
        grad_norm: f32,
        learning_rate: f32,
        tokens_per_sec: f64,
        elapsed: Duration,
        out: &mut dyn Write,
    ) {
        let idx = self.entries.len();
        let mut samples = Vec::with_capacity(4);
        for (k, (dom, q)) in Self::probe_prompts().iter().enumerate() {
            let reply = Self::sample_model_reply(model, tok, q, 0x2026_5EED_u64 + k as u64);
            samples.push(ProbeOutputSample {
                domain: (*dom).to_string(),
                prompt: (*q).to_string(),
                response: reply,
            });
        }
        let ppl = val_loss
            .or(train_loss)
            .filter(|l| l.is_finite())
            .map(|l| (l as f64).exp().min(999_999.0) as f32);

        let entry = EvolutionMilestoneEntry {
            index: idx,
            phase: phase.to_string(),
            elapsed_hms: format_hms(elapsed),
            elapsed_secs: elapsed.as_secs_f64(),
            step,
            train_loss,
            val_loss,
            perplexity: ppl,
            grad_norm,
            learning_rate,
            tokens_per_sec,
            samples,
        };

        if let Ok(mut jf) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.jsonl_path)
        {
            if let Ok(line) = serde_json::to_string(&entry) {
                let _ = writeln!(jf, "{}", line);
            }
        }

        let eta_str = match self.planned_duration {
            Some(tot) if elapsed < tot => format!(" | ETA={}", format_hms(tot - elapsed)),
            Some(_) => " | ETA=00h:00m:00s".to_string(),
            None => String::new(),
        };
        let tl_s = train_loss
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "-".to_string());
        let vl_s = val_loss
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "-".to_string());
        let ppl_s = ppl
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "-".to_string());

        let _ = writeln!(
            out,
            "\n┌─ [MARCO #{:02} | {} | t={}{}] step={} | train_loss={} | val_loss={} | ppl={} | lr={:.5} | {:.1} tok/s",
            entry.index,
            entry.phase,
            entry.elapsed_hms,
            eta_str,
            entry.step,
            tl_s,
            vl_s,
            ppl_s,
            entry.learning_rate,
            entry.tokens_per_sec
        );
        for s in &entry.samples {
            let _ = writeln!(out, "│  [{}] Q: {}", s.domain, s.prompt);
            let _ = writeln!(out, "│         A: {}", s.response);
        }
        let _ = writeln!(
            out,
            "└──────────────────────────────────────────────────────────────────────────"
        );
        let _ = out.flush();

        self.entries.push(entry);
        let _ = self.write_markdown_summary(&self.device_hint, "Em andamento");
    }

    pub(crate) fn write_markdown_summary(
        &self,
        active_device: &str,
        status_label: &str,
    ) -> Result<(), String> {
        let mut md = String::new();
        md.push_str(&format!(
            "# Sequência Evolutiva de Treinamento — `{}`\n\n",
            self.run_name
        ));
        md.push_str(&format!(
            "- **Parâmetros:** `{}` (`{:.2}M`)\n- **Dispositivo:** `{}`\n- **Status:** `{}`\n",
            self.total_params,
            self.total_params as f64 / 1e6,
            active_device,
            status_label
        ));
        if let Some(d) = self.planned_duration {
            md.push_str(&format!("- **Duração Alvo:** `{}`\n", format_hms(d)));
        }
        md.push_str("\n## 1. Curva Cronológica de Métricas\n\n");
        md.push_str("| Marco | Fase | Tempo | Step | `train_loss` | `val_loss` | Perplexidade | `lr` | `tok/s` |\n");
        md.push_str("| ---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for e in &self.entries {
            let tl = e
                .train_loss
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".into());
            let vl = e
                .val_loss
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".into());
            let pp = e
                .perplexity
                .map(|v| format!("{:.2}", v))
                .unwrap_or_else(|| "-".into());
            md.push_str(&format!(
                "| **#{:02}** | {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{:.5}` | `{:.1}` |\n",
                e.index,
                e.phase,
                e.elapsed_hms,
                e.step,
                tl,
                vl,
                pp,
                e.learning_rate,
                e.tokens_per_sec
            ));
        }

        md.push_str("\n## 2. Evolução Real das Respostas Marco a Marco\n\n");
        for e in &self.entries {
            let tl = e
                .train_loss
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".into());
            let vl = e
                .val_loss
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".into());
            md.push_str(&format!(
                "### Marco #{:02} — `{}` (`{}` | `step={}` | `train_loss={}` | `val_loss={}`)\n\n",
                e.index, e.phase, e.elapsed_hms, e.step, tl, vl
            ));
            for s in &e.samples {
                md.push_str(&format!(
                    "- **`[{}]` Pergunta:** {}\n  - **Resposta gerada:** `{}`\n",
                    s.domain, s.prompt, s.response
                ));
            }
            md.push('\n');
        }

        std::fs::write(&self.md_path, md)
            .map_err(|e| format!("Falha ao gravar '{}': {}", self.md_path.display(), e))
    }
}

pub(crate) struct MarathonParams {
    pub(crate) out_dir: PathBuf,
    pub(crate) quant_out_dir: PathBuf,
    pub(crate) export_bits: crate::neural::WeightPrecision,
    pub(crate) total_duration: Duration,
    pub(crate) eval_every_secs: Option<u64>,
    pub(crate) dataset_override: Option<PathBuf>,
    pub(crate) corpus_dirs_override: Vec<PathBuf>,
    pub(crate) chunk_chars: usize,
    pub(crate) resume: bool,
    pub(crate) d_model: usize,
    pub(crate) n_heads: usize,
    pub(crate) n_layers: usize,
    pub(crate) d_ff: usize,
    pub(crate) seq_len: usize,
    pub(crate) batch_size: usize,
    pub(crate) vocab_size: usize,
    pub(crate) lr_override: Option<f32>,
    pub(crate) max_val_windows: usize,
    pub(crate) device: crate::neural::ComputeDevice,
    pub(crate) install_ctrlc_handler: bool,
    pub(crate) cancel_flag: Option<Arc<AtomicBool>>,
}

fn load_qa_datasets_combined(explicit_ds: Option<&PathBuf>) -> Result<Vec<String>, String> {
    let mut qa = Vec::new();
    if let Some(ds) = explicit_ds {
        qa.extend(load_seed_corpus(ds)?);
    } else {
        let v3 = crate::chat::config::AppConfig::resolve_from_config_dir(
            "datasets/servermind_corpus_v3_balanced.jsonl",
        );
        if v3.is_file() {
            qa.extend(load_seed_corpus(&v3)?);
        } else {
            for cand in [
                "datasets/servermind_corpus_v2.jsonl",
                "datasets/servermind_seed_v1.jsonl",
            ] {
                let p = crate::chat::config::AppConfig::resolve_from_config_dir(cand);
                if p.is_file() {
                    qa.extend(load_seed_corpus(&p)?);
                }
            }
        }
    }
    if qa.is_empty() {
        return Err("Nenhum exemplo Q&A encontrado para o treinamento maratona.".to_string());
    }
    Ok(qa)
}

pub(crate) fn run_autonomous_marathon(
    params: MarathonParams,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> CliResult<()> {
    let qa_examples =
        load_qa_datasets_combined(params.dataset_override.as_ref()).map_err(CliError::runtime)?;

    let mut book_chunks = Vec::new();
    let book_dirs = if !params.corpus_dirs_override.is_empty() {
        params.corpus_dirs_override.clone()
    } else {
        let default_books =
            crate::chat::config::AppConfig::resolve_from_config_dir("datasets/pretrain_books");
        if default_books.is_dir() {
            vec![default_books]
        } else {
            Vec::new()
        }
    };
    for bdir in &book_dirs {
        let mut c = crate::neural::tokenizer::load_corpus_directory(bdir, params.chunk_chars)
            .map_err(CliError::runtime)?;
        book_chunks.append(&mut c);
    }

    let has_pretrain_books = !book_chunks.is_empty();
    let mut combined_for_vocab = Vec::new();
    if book_chunks.len() > 400 {
        let step = book_chunks.len() / 400;
        for idx in (0..book_chunks.len()).step_by(step.max(1)).take(400) {
            combined_for_vocab.push(book_chunks[idx].clone());
        }
    } else {
        combined_for_vocab.extend(book_chunks.iter().cloned());
    }
    for _ in 0..6 {
        combined_for_vocab.extend(qa_examples.iter().cloned());
    }

    let run_name = params
        .out_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("servermind-neural-marathon")
        .to_string();
    let tok_path = params.out_dir.join("tokenizer.json");
    let latest_ckpt = params.out_dir.join("latest.bin");

    if !params.resume {
        check_no_existing_training_artifacts(&params.out_dir).map_err(CliError::runtime)?;
    }

    let (tokenizer, mut trainer) = if params.resume {
        if !tok_path.is_file() || !latest_ckpt.is_file() {
            return Err(CliError::runtime(format!(
                "Flag --resume exige '{}' e '{}' existentes.",
                tok_path.display(),
                latest_ckpt.display()
            )));
        }
        let tok =
            crate::neural::tokenizer::BpeTokenizer::load(&tok_path).map_err(CliError::runtime)?;
        let cfg = crate::neural::training::TrainingConfig {
            run_name: run_name.clone(),
            checkpoint_dir: params.out_dir.clone(),
            max_steps: usize::MAX / 2,
            batch_size: params.batch_size,
            seq_len: params.seq_len,
            learning_rate: params.lr_override.unwrap_or(1.2e-3),
            min_learning_rate: 5e-5,
            warmup_steps: 10,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            eval_interval: 1_000,
            checkpoint_interval: 2_000,
            max_val_windows: params.max_val_windows,
            device: params.device,
        };
        let tr = crate::neural::training::Trainer::resume_from_checkpoint(&latest_ckpt, cfg)
            .map_err(CliError::runtime)?;
        (tok, tr)
    } else {
        std::fs::create_dir_all(&params.out_dir).map_err(|e| {
            CliError::runtime(format!(
                "Falha ao criar '{}': {}",
                params.out_dir.display(),
                e
            ))
        })?;
        let _ = writeln!(
            out,
            "[1/3] Treinando tokenizador BPE próprio (vocab_alvo={}) sobre {} blocos de amostra...",
            params.vocab_size,
            combined_for_vocab.len()
        );
        let _ = out.flush();
        let refs: Vec<&str> = combined_for_vocab.iter().map(|s| s.as_str()).collect();
        let tok =
            crate::neural::tokenizer::BpeTokenizer::train_from_corpus(&refs, params.vocab_size);
        tok.save(&tok_path).map_err(CliError::runtime)?;
        let mcfg = crate::neural::TransformerConfig {
            vocab_size: tok.vocab_size(),
            max_seq_len: params.seq_len,
            d_model: params.d_model,
            n_heads: params.n_heads,
            n_layers: params.n_layers,
            d_ff: params.d_ff,
        };
        let model = crate::neural::ServerMindTransformer::new_random(mcfg, 2026)
            .map_err(CliError::runtime)?;
        let tcfg = crate::neural::training::TrainingConfig {
            run_name: run_name.clone(),
            checkpoint_dir: params.out_dir.clone(),
            max_steps: usize::MAX / 2,
            batch_size: params.batch_size,
            seq_len: params.seq_len,
            learning_rate: params.lr_override.unwrap_or(1.5e-3),
            min_learning_rate: 2.5e-4,
            warmup_steps: 20,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            eval_interval: 1_000,
            checkpoint_interval: 2_000,
            max_val_windows: params.max_val_windows,
            device: params.device,
        };
        (tok, crate::neural::training::Trainer::new(model, tcfg))
    };

    let active_seq_len = trainer.model.config.max_seq_len;
    let total_params = trainer.model.config.total_params();

    let total_secs = params.total_duration.as_secs_f64().max(1.0);
    let (stage1_dur, stage2_dur) = if has_pretrain_books {
        let s1 = Duration::from_secs_f64((total_secs * 0.35).max(0.5));
        let s2 = params
            .total_duration
            .saturating_sub(s1)
            .max(Duration::from_millis(500));
        (Some(s1), s2)
    } else {
        (None, params.total_duration)
    };

    let eval_every = params
        .eval_every_secs
        .unwrap_or_else(|| ((total_secs / 24.0).round() as u64).clamp(1, 900));

    let qa_split = crate::neural::tokenizer::split_dataset(&qa_examples, &run_name, 0.12, 0.05);
    let qa_train_masked = crate::neural::training::pack_chat_masked_sequences(
        &tokenizer,
        &qa_split.train,
        active_seq_len,
    );
    let qa_val_src = if qa_split.validation.is_empty() {
        &qa_split.train
    } else {
        &qa_split.validation
    };
    let qa_val_masked =
        crate::neural::training::pack_chat_masked_sequences(&tokenizer, qa_val_src, active_seq_len);

    let _ = writeln!(
        out,
        "=========================================================================="
    );
    let _ = writeln!(
        out,
        "  MARATONA AUTÔNOMA SERVERMIND NEURAL — DURAÇÃO ALVO: {}",
        format_hms(params.total_duration)
    );
    let _ = writeln!(
        out,
        "  Modelo: {:.2}M params (d={}, L={}, H={}, seq={}, vocab={}) | device={}",
        total_params as f64 / 1e6,
        trainer.model.config.d_model,
        trainer.model.config.n_layers,
        trainer.model.config.n_heads,
        active_seq_len,
        tokenizer.vocab_size(),
        params.device.as_str()
    );
    if let Some(s1) = stage1_dur {
        let _ = writeln!(
            out,
            "  Currículo: Estágio 1 Pré-Treino ({}) -> Estágio 2 Chat Fine-Tuning ({})",
            format_hms(s1),
            format_hms(stage2_dur)
        );
    } else {
        let _ = writeln!(
            out,
            "  Currículo: Fine-Tuning Direto em Q&A ({})",
            format_hms(stage2_dur)
        );
    }
    let _ = writeln!(
        out,
        "  Marcos evolutivos a cada ~{}s -> '{}/training_evolution.md' (Ctrl+C salva limpo)",
        eval_every,
        params.out_dir.display()
    );
    let _ = writeln!(
        out,
        "=========================================================================="
    );
    let _ = out.flush();

    let mut recorder = EvolutionSequenceRecorder::new(
        &params.out_dir,
        &run_name,
        params.device.as_str(),
        total_params,
        Some(params.total_duration),
    );
    let global_start = std::time::Instant::now();

    recorder.record_milestone(
        "Marco Zero (Antes do Treino)",
        &trainer.model,
        &tokenizer,
        trainer.current_step,
        None,
        None,
        0.0,
        trainer.config.learning_rate,
        0.0,
        Duration::ZERO,
        out,
    );

    let mut active_device_str = params.device.as_str().to_string();
    let mut was_cancelled = false;

    if let Some(s1_dur) = stage1_dur {
        let mut stage1_corpus = book_chunks;
        for _ in 0..4 {
            stage1_corpus.extend(qa_split.train.iter().cloned());
        }
        let s1_train_wins =
            crate::neural::training::pack_sequences(&tokenizer, &stage1_corpus, active_seq_len);
        trainer.config.learning_rate = params.lr_override.unwrap_or(1.5e-3);
        trainer.config.min_learning_rate = 3.0e-4;
        trainer.config.eval_interval = usize::MAX / 4;

        let s1_res = {
            let tok_ref = &tokenizer;
            let mut m_cb = |model: &crate::neural::ServerMindTransformer,
                            m: &crate::neural::training::TrainStepMetrics,
                            tok_s: f64,
                            _st_elapsed: Duration| {
                recorder.record_milestone(
                    "Fase 1/2 — Pré-Treino Base",
                    model,
                    tok_ref,
                    m.step,
                    Some(m.train_loss),
                    m.val_loss,
                    m.grad_norm,
                    m.learning_rate,
                    tok_s,
                    global_start.elapsed(),
                    out,
                );
            };
            let opts = crate::neural::training::TimedSessionOptions {
                max_duration: Some(s1_dur),
                eval_every_secs: Some(eval_every),
                cancel_flag: params.cancel_flag.clone(),
                install_ctrlc_handler: params.install_ctrlc_handler,
                on_milestone: Some(&mut m_cb),
                masked_train: None,
                masked_val: Some(&qa_val_masked),
            };
            trainer.train_with_session_options(&s1_train_wins, &[], tokenizer.pad_id(), None, opts)
        };

        let rep = s1_res
            .map_err(|e| CliError::runtime(format!("Erro no Estágio 1 da maratona: {}", e)))?;
        active_device_str = rep.active_device;
        was_cancelled = rep.cancelled;
        let _ = writeln!(
            out,
            "\n>> [Transição de Fase] Estágio 1 finalizado em step={} (loss {:.4} -> {:.4}). Iniciando Estágio 2 (Chat Fine-Tuning)...",
            rep.steps_completed, rep.initial_train_loss, rep.final_train_loss
        );
        let _ = out.flush();
    }

    if !was_cancelled {
        trainer.best_val_loss = match crate::neural::training::evaluate_masked_loss(
            &trainer.model,
            &qa_val_masked,
            tokenizer.pad_id(),
        ) {
            Some(v) if v.is_finite() => v,
            Some(non_finite) => {
                let _ = writeln!(
                    err,
                    "[AVISO DE VALIDAÇÃO] Início da Fase 2 com val_loss não finita ({}); best_val_loss mantido em INFINITY.",
                    non_finite
                );
                f32::INFINITY
            }
            None => f32::INFINITY,
        };
        let _ = crate::neural::save_checkpoint(
            &params.out_dir.join("best.bin"),
            &trainer.model,
            Some(&trainer.optimizer),
            trainer.current_step,
            trainer.best_val_loss,
        );

        trainer.config.learning_rate = params.lr_override.map(|v| v * 0.4).unwrap_or(5.0e-4);
        trainer.config.min_learning_rate = 3.0e-5;
        trainer.config.eval_interval = usize::MAX / 4;

        let s2_res = {
            let tok_ref = &tokenizer;
            let mut m_cb = |model: &crate::neural::ServerMindTransformer,
                            m: &crate::neural::training::TrainStepMetrics,
                            tok_s: f64,
                            _st_elapsed: Duration| {
                recorder.record_milestone(
                    "Fase 2/2 — Chat Fine-Tuning",
                    model,
                    tok_ref,
                    m.step,
                    Some(m.train_loss),
                    m.val_loss,
                    m.grad_norm,
                    m.learning_rate,
                    tok_s,
                    global_start.elapsed(),
                    out,
                );
            };
            let opts = crate::neural::training::TimedSessionOptions {
                max_duration: Some(stage2_dur),
                eval_every_secs: Some(eval_every),
                cancel_flag: params.cancel_flag.clone(),
                install_ctrlc_handler: params.install_ctrlc_handler,
                on_milestone: Some(&mut m_cb),
                masked_train: Some(&qa_train_masked),
                masked_val: Some(&qa_val_masked),
            };
            trainer.train_with_session_options(&[], &[], tokenizer.pad_id(), None, opts)
        };

        let rep = s2_res
            .map_err(|e| CliError::runtime(format!("Erro no Estágio 2 da maratona: {}", e)))?;
        active_device_str = rep.active_device;
        was_cancelled = rep.cancelled;
        let final_val = crate::neural::training::evaluate_masked_loss(
            &trainer.model,
            &qa_val_masked,
            tokenizer.pad_id(),
        )
        .filter(|v| v.is_finite());
        recorder.record_milestone(
            if was_cancelled {
                "Marco Final (Pausado em Ctrl+C)"
            } else {
                "Marco Final Consolidado"
            },
            &trainer.model,
            &tokenizer,
            rep.steps_completed,
            Some(rep.final_train_loss),
            final_val,
            0.0,
            trainer.config.min_learning_rate,
            rep.tokens_per_sec,
            global_start.elapsed(),
            out,
        );
    }

    let final_status = if was_cancelled {
        "Pausado pelo usuário (Ctrl+C, salvo)"
    } else {
        "Concluído"
    };
    let _ = recorder.write_markdown_summary(&active_device_str, final_status);

    match export_quantized_package(&params.out_dir, &params.quant_out_dir, params.export_bits) {
        Ok(st) => {
            let _ = writeln!(
                out,
                "\n[Pacote SMQ1 ({}) atualizado em '{}': disco {:.2} MB -> {:.2} MB ({:.2}x menor) | RAM pesos {:.2} MB -> {:.2} MB]",
                st.precision.as_str(),
                params.quant_out_dir.display(),
                st.src_disk_mb,
                st.dst_disk_mb,
                st.compression_ratio,
                st.src_ram_mb,
                st.dst_ram_mb
            );
        }
        Err(e) => {
            let _ = writeln!(err, "Aviso ao exportar pacote quantizado: {}", e);
        }
    }

    let _ = writeln!(
        out,
        "=========================================================================="
    );
    let _ = writeln!(
        out,
        "  SEQUÊNCIA EVOLUTIVA GRAVADA EM: {}",
        recorder.md_path.display()
    );
    let _ = writeln!(
        out,
        "  LOG JSONL CRONOLÓGICO         : {}",
        recorder.jsonl_path.display()
    );
    let _ = writeln!(
        out,
        "  PARA CONVERSAR NO CHAT        : :backend neural {}",
        params.out_dir.display()
    );
    let _ = writeln!(
        out,
        "=========================================================================="
    );

    if was_cancelled {
        Err(CliError::cancelled())
    } else {
        Ok(())
    }
}
