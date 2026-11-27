use super::corpus::{check_no_existing_training_artifacts, load_seed_corpus};
use super::marathon::{
    format_hms, run_autonomous_marathon, EvolutionSequenceRecorder, MarathonParams,
};
use super::quantize::export_quantized_package;
use crate::cli::error::{CliError, CliResult};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

pub(crate) const DEFAULT_TRAIN_STEPS: usize = 80;

#[derive(Debug)]
pub(crate) struct TrainNeuralRequest {
    pub(crate) dataset_path: Option<PathBuf>,
    pub(crate) corpus_dirs: Vec<PathBuf>,
    pub(crate) chunk_chars: usize,
    pub(crate) out_dir_opt: Option<PathBuf>,
    pub(crate) run_name: Option<String>,
    pub(crate) resume: bool,
    pub(crate) max_steps_opt: Option<usize>,
    pub(crate) max_duration_opt: Option<Duration>,
    pub(crate) eval_every_secs_opt: Option<u64>,
    pub(crate) marathon: bool,
    pub(crate) d_model_opt: Option<usize>,
    pub(crate) n_heads_opt: Option<usize>,
    pub(crate) n_layers_opt: Option<usize>,
    pub(crate) d_ff_opt: Option<usize>,
    pub(crate) seq_len_opt: Option<usize>,
    pub(crate) batch_size_opt: Option<usize>,
    pub(crate) vocab_size_opt: Option<usize>,
    pub(crate) lr_opt: Option<f32>,
    pub(crate) min_learning_rate: Option<f32>,
    pub(crate) max_val_windows: usize,
    pub(crate) device: crate::neural::ComputeDevice,
    pub(crate) export_quant_dir: Option<PathBuf>,
    pub(crate) export_bits: crate::neural::WeightPrecision,
    pub(crate) threads_opt: Option<usize>,
}

#[derive(Debug)]
pub(crate) struct EvalNeuralRequest {
    pub(crate) ckpt_dir_opt: Option<PathBuf>,
    pub(crate) dataset_path_opt: Option<PathBuf>,
    pub(crate) quantize_out: Option<PathBuf>,
    pub(crate) bits_opt: Option<crate::neural::WeightPrecision>,
    pub(crate) kv_cache_opt: Option<crate::neural::KvCachePrecision>,
    pub(crate) low_ram: bool,
    pub(crate) max_ram_mb_opt: Option<usize>,
    pub(crate) threads_opt: Option<usize>,
}

pub(crate) fn execute_train_neural(
    req: TrainNeuralRequest,
    install_ctrlc_handler: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> CliResult<()> {
    let TrainNeuralRequest {
        dataset_path,
        corpus_dirs,
        chunk_chars,
        out_dir_opt,
        run_name,
        resume,
        max_steps_opt,
        max_duration_opt,
        eval_every_secs_opt,
        marathon,
        d_model_opt,
        n_heads_opt,
        n_layers_opt,
        d_ff_opt,
        seq_len_opt,
        batch_size_opt,
        vocab_size_opt,
        lr_opt,
        min_learning_rate,
        max_val_windows,
        device,
        export_quant_dir,
        export_bits,
        threads_opt,
    } = req;

    if let Some(t) = threads_opt {
        crate::neural::ops::set_max_active_threads(Some(t));
    }

    if marathon {
        let total_dur = max_duration_opt.unwrap_or_else(|| Duration::from_secs(6 * 3600));
        let out_dir = out_dir_opt.unwrap_or_else(|| {
            crate::chat::config::AppConfig::resolve_from_config_dir(
                "checkpoints/servermind-neural-marathon",
            )
        });
        let quant_out = export_quant_dir
            .unwrap_or_else(|| PathBuf::from(format!("{}-int8", out_dir.display())));
        return run_autonomous_marathon(
            MarathonParams {
                out_dir,
                quant_out_dir: quant_out,
                export_bits,
                total_duration: total_dur,
                eval_every_secs: eval_every_secs_opt,
                dataset_override: dataset_path,
                corpus_dirs_override: corpus_dirs,
                chunk_chars,
                resume,
                d_model: d_model_opt.unwrap_or(256),
                n_heads: n_heads_opt.unwrap_or(8),
                n_layers: n_layers_opt.unwrap_or(4),
                d_ff: d_ff_opt.unwrap_or(768),
                seq_len: seq_len_opt.unwrap_or(192),
                batch_size: batch_size_opt.unwrap_or(4),
                vocab_size: vocab_size_opt.unwrap_or(1536),
                lr_override: lr_opt,
                max_val_windows,
                device,
                install_ctrlc_handler,
                cancel_flag,
            },
            out,
            err,
        );
    }

    let out_dir = out_dir_opt.unwrap_or_else(|| {
        crate::chat::config::AppConfig::resolve_from_config_dir("checkpoints/servermind-neural-v1")
    });
    let max_steps = max_steps_opt.unwrap_or_else(|| {
        if max_duration_opt.is_some() {
            usize::MAX / 2
        } else {
            DEFAULT_TRAIN_STEPS
        }
    });
    let d_model = d_model_opt.unwrap_or(128);
    let n_heads = n_heads_opt.unwrap_or(4);
    let n_layers = n_layers_opt.unwrap_or(2);
    let d_ff = d_ff_opt.unwrap_or(256);
    let seq_len = seq_len_opt.unwrap_or(64);
    let batch_size = batch_size_opt.unwrap_or(4);
    let vocab_size = vocab_size_opt.unwrap_or(768);
    let learning_rate = lr_opt.unwrap_or(3e-3);

    let resolved_min_lr = min_learning_rate.unwrap_or(learning_rate * 0.1);
    if resolved_min_lr > learning_rate {
        return Err(CliError::usage(format!(
            "'--min-lr' ({}) não pode ser maior que '--lr' ({}).",
            resolved_min_lr, learning_rate
        )));
    }

    let mut examples = Vec::new();
    for dir in &corpus_dirs {
        let mut chunks = crate::neural::tokenizer::load_corpus_directory(dir, chunk_chars)
            .map_err(CliError::runtime)?;
        examples.append(&mut chunks);
    }

    let effective_dataset = if let Some(ref ds) = dataset_path {
        Some(ds.clone())
    } else if corpus_dirs.is_empty() {
        Some(crate::chat::config::AppConfig::resolve_from_config_dir(
            "datasets/servermind_seed_v1.jsonl",
        ))
    } else {
        None
    };

    if let Some(ds_path) = effective_dataset {
        let mut ds_examples = load_seed_corpus(&ds_path).map_err(CliError::runtime)?;
        examples.append(&mut ds_examples);
    }

    if examples.is_empty() {
        return Err(CliError::runtime(
            "Nenhum exemplo válido carregado para treinamento.",
        ));
    }

    let resolved_run_name = run_name.unwrap_or_else(|| {
        out_dir
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("servermind-neural")
            .to_string()
    });

    let split = crate::neural::tokenizer::split_dataset(&examples, &resolved_run_name, 0.15, 0.10);
    let latest_ckpt = out_dir.join("latest.bin");
    let tok_path = out_dir.join("tokenizer.json");

    let (tokenizer, mut trainer) = if resume {
        if !tok_path.is_file() || !latest_ckpt.is_file() {
            return Err(CliError::runtime(format!(
                "Flag --resume exige '{}' e '{}' existentes.",
                tok_path.display(),
                latest_ckpt.display()
            )));
        }
        let tokenizer =
            crate::neural::tokenizer::BpeTokenizer::load(&tok_path).map_err(CliError::runtime)?;
        let train_cfg = crate::neural::training::TrainingConfig {
            run_name: resolved_run_name.clone(),
            checkpoint_dir: out_dir.clone(),
            max_steps,
            batch_size,
            seq_len,
            learning_rate,
            min_learning_rate: resolved_min_lr,
            warmup_steps: 0,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            eval_interval: (max_steps / 5).max(5),
            checkpoint_interval: (max_steps / 2).max(10),
            max_val_windows,
            device,
        };
        let mut resumed =
            crate::neural::training::Trainer::resume_from_checkpoint(&latest_ckpt, train_cfg)
                .map_err(CliError::runtime)?;
        if tokenizer.vocab_size() != resumed.model.config.vocab_size {
            return Err(CliError::runtime(format!(
                "Incompatibilidade em --resume: tokenizer tem vocab={} mas checkpoint tem vocab={}.",
                tokenizer.vocab_size(),
                resumed.model.config.vocab_size
            )));
        }
        if max_duration_opt.is_none() {
            if let Some(explicit_steps) = max_steps_opt {
                if explicit_steps <= resumed.current_step {
                    return Err(CliError::usage(format!(
                        "'--steps' ({}) deve ser maior que o passo atual do checkpoint ({}).",
                        explicit_steps, resumed.current_step
                    )));
                }
                resumed.config.max_steps = explicit_steps;
            } else {
                resumed.config.max_steps = resumed.current_step.saturating_add(DEFAULT_TRAIN_STEPS);
            }
        }
        resumed.config.seq_len = resumed.model.config.max_seq_len;
        if max_duration_opt.is_some() {
            resumed.config.eval_interval = 500;
            resumed.config.checkpoint_interval = 1_000;
        } else {
            let remaining = resumed
                .config
                .max_steps
                .saturating_sub(resumed.current_step);
            resumed.config.eval_interval = (remaining / 5).max(5);
            resumed.config.checkpoint_interval = (remaining / 2).max(10);
        }
        (tokenizer, resumed)
    } else {
        check_no_existing_training_artifacts(&out_dir).map_err(CliError::runtime)?;
        let corpus_refs: Vec<&str> = split.train.iter().map(|s| s.as_str()).collect();
        let tokenizer =
            crate::neural::tokenizer::BpeTokenizer::train_from_corpus(&corpus_refs, vocab_size);

        std::fs::create_dir_all(&out_dir).map_err(|e| {
            CliError::runtime(format!("Falha ao criar '{}': {}", out_dir.display(), e))
        })?;
        tokenizer.save(&tok_path).map_err(CliError::runtime)?;

        let cfg = crate::neural::TransformerConfig {
            vocab_size: tokenizer.vocab_size(),
            max_seq_len: seq_len,
            d_model,
            n_heads,
            n_layers,
            d_ff,
        };
        let model = crate::neural::ServerMindTransformer::new_random(cfg, 2026)
            .map_err(CliError::runtime)?;
        let (warmup_s, eval_int, ckpt_int) = if max_duration_opt.is_some() {
            (10, 500, 1_000)
        } else {
            (
                (max_steps / 10).max(2),
                (max_steps / 5).max(5),
                (max_steps / 2).max(10),
            )
        };
        let train_cfg = crate::neural::training::TrainingConfig {
            run_name: resolved_run_name.clone(),
            checkpoint_dir: out_dir.clone(),
            max_steps,
            batch_size,
            seq_len,
            learning_rate,
            min_learning_rate: resolved_min_lr,
            warmup_steps: warmup_s,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            eval_interval: eval_int,
            checkpoint_interval: ckpt_int,
            max_val_windows,
            device,
        };
        (
            tokenizer,
            crate::neural::training::Trainer::new(model, train_cfg),
        )
    };

    let active_seq_len = trainer.model.config.max_seq_len;
    let use_chat_mask = corpus_dirs.is_empty();
    let val_src = if split.validation.is_empty() {
        &split.train
    } else {
        &split.validation
    };
    let (train_windows, val_windows, masked_train_seqs, masked_val_seqs) = if use_chat_mask {
        (
            Vec::new(),
            Vec::new(),
            Some(crate::neural::training::pack_chat_masked_sequences(
                &tokenizer,
                &split.train,
                active_seq_len,
            )),
            Some(crate::neural::training::pack_chat_masked_sequences(
                &tokenizer,
                val_src,
                active_seq_len,
            )),
        )
    } else {
        (
            crate::neural::training::pack_sequences(&tokenizer, &split.train, active_seq_len),
            crate::neural::training::pack_eval_sequences(&tokenizer, val_src, active_seq_len),
            None,
            None,
        )
    };
    let effective_train_count = masked_train_seqs
        .as_ref()
        .map(|m| m.len())
        .unwrap_or(train_windows.len());
    let effective_val_count = masked_val_seqs
        .as_ref()
        .map(|m| m.len())
        .unwrap_or(val_windows.len());
    let total_params = trainer.model.config.total_params();

    let target_steps = trainer.config.max_steps;
    let target_desc = if let Some(dur) = max_duration_opt {
        format!("duração={}", format_hms(dur))
    } else {
        format!("step={} -> {}", trainer.current_step, target_steps)
    };
    let _ = writeln!(
        out,
        "Iniciando treinamento ServerMind Neural ({}): params={} | vocab={} | seq_len={} | lr={:.5} | device={} | train_windows={} | val_windows={} | {}",
        resolved_run_name,
        total_params,
        tokenizer.vocab.len(),
        active_seq_len,
        learning_rate,
        device.as_str(),
        effective_train_count,
        effective_val_count,
        target_desc
    );
    let _ = out.flush();

    let mut recorder = if max_duration_opt.is_some() || eval_every_secs_opt.is_some() {
        let mut rec = EvolutionSequenceRecorder::new(
            &out_dir,
            &resolved_run_name,
            device.as_str(),
            total_params,
            max_duration_opt,
        );
        rec.record_milestone(
            "Baseline Inicial (Step 0 / Pré-Sessão)",
            &trainer.model,
            &tokenizer,
            trainer.current_step,
            None,
            None,
            0.0,
            learning_rate,
            0.0,
            Duration::ZERO,
            out,
        );
        Some(rec)
    } else {
        None
    };

    let auto_eval_secs =
        eval_every_secs_opt.or_else(|| max_duration_opt.map(|d| (d.as_secs() / 24).clamp(2, 900)));

    let run_res = if let Some(ref mut rec) = recorder {
        let tok_ref = &tokenizer;
        let mut milestone_cb = |model: &crate::neural::ServerMindTransformer,
                                m: &crate::neural::training::TrainStepMetrics,
                                tok_s: f64,
                                elapsed: Duration| {
            rec.record_milestone(
                "Sessão Contínua",
                model,
                tok_ref,
                m.step,
                Some(m.train_loss),
                m.val_loss,
                m.grad_norm,
                m.learning_rate,
                tok_s,
                elapsed,
                out,
            );
        };
        let opts = crate::neural::training::TimedSessionOptions {
            max_duration: max_duration_opt,
            eval_every_secs: auto_eval_secs,
            cancel_flag,
            install_ctrlc_handler,
            on_milestone: Some(&mut milestone_cb),
            masked_train: masked_train_seqs.as_deref(),
            masked_val: masked_val_seqs.as_deref(),
        };
        trainer.train_with_session_options(
            &train_windows,
            &val_windows,
            tokenizer.pad_id(),
            None,
            opts,
        )
    } else {
        let mut progress_cb = |m: &crate::neural::training::TrainStepMetrics, tok_s: f64| {
            let val_str = m
                .val_loss
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".to_string());
            let _ = writeln!(
                out,
                "  [step {}/{}] train_loss={:.4} | val_loss={} | grad_norm={:.3} | lr={:.5} | {:.1} tok/s",
                m.step, target_steps, m.train_loss, val_str, m.grad_norm, m.learning_rate, tok_s
            );
            let _ = out.flush();
        };
        let opts = crate::neural::training::TimedSessionOptions {
            max_duration: None,
            eval_every_secs: None,
            cancel_flag,
            install_ctrlc_handler,
            on_milestone: None,
            masked_train: masked_train_seqs.as_deref(),
            masked_val: masked_val_seqs.as_deref(),
        };
        trainer.train_with_session_options(
            &train_windows,
            &val_windows,
            tokenizer.pad_id(),
            Some(&mut progress_cb),
            opts,
        )
    };

    let report = run_res.map_err(|e| CliError::runtime(format!("Erro no treinamento: {}", e)))?;
    if let Some(ref rec) = recorder {
        let final_status = if report.cancelled {
            "Pausado pelo usuário (Ctrl+C, salvo)"
        } else {
            "Concluído"
        };
        let _ = rec.write_markdown_summary(&report.active_device, final_status);
        let _ = writeln!(
            out,
            "Sequência evolutiva completa salva em '{}' e '{}'",
            rec.md_path.display(),
            rec.jsonl_path.display()
        );
    }
    let status_lbl = if report.cancelled {
        "pausado com segurança (Ctrl+C)"
    } else if report.stopped_by_duration {
        "concluído no tempo programado"
    } else {
        "concluído"
    };
    let _ = writeln!(
        out,
        "Treinamento {} [{}]: steps={} | initial_loss={:.4} -> final_loss={:.4} | best_val_loss={:.4} | {:.1} tok/s | elapsed={}",
        status_lbl,
        report.active_device,
        report.steps_completed,
        report.initial_train_loss,
        report.final_train_loss,
        report.best_val_loss,
        report.tokens_per_sec,
        format_hms(Duration::from_millis(report.total_elapsed_ms as u64))
    );
    let _ = writeln!(out, "Artefatos salvos em '{}'", out_dir.display());
    if let Some(ref qdir) = export_quant_dir {
        let info = export_quantized_package(&out_dir, qdir, export_bits)
            .map_err(|e| CliError::runtime(format!("Erro ao exportar pacote quantizado: {}", e)))?;
        let _ = writeln!(
            out,
            "Pacote quantizado SMQ1 ({}) exportado para '{}': disco {:.2} MB -> {:.2} MB ({:.2}x menor) | pesos RAM {:.2} MB -> {:.2} MB",
            info.precision.as_str(),
            qdir.display(),
            info.src_disk_mb,
            info.dst_disk_mb,
            info.compression_ratio,
            info.src_ram_mb,
            info.dst_ram_mb
        );
    }
    if report.cancelled {
        Err(CliError::cancelled())
    } else {
        Ok(())
    }
}

pub(crate) fn execute_eval_neural(
    req: EvalNeuralRequest,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> CliResult<()> {
    let ckpt_dir = req.ckpt_dir_opt.unwrap_or_else(|| {
        crate::chat::config::AppConfig::resolve_from_config_dir("checkpoints/servermind-neural-v1")
    });
    let dataset_path = req.dataset_path_opt.unwrap_or_else(|| {
        crate::chat::config::AppConfig::resolve_from_config_dir("datasets/servermind_seed_v1.jsonl")
    });
    let quantize_out = req.quantize_out;
    let bits_opt = req.bits_opt;
    let kv_cache_opt = req.kv_cache_opt;
    let low_ram = req.low_ram;
    let max_ram_mb_opt = req.max_ram_mb_opt;
    let threads_opt = req.threads_opt;

    let mut budget = if low_ram {
        crate::platform::hardware::RuntimeBudget::low_end_8gb()
    } else {
        crate::platform::hardware::RuntimeBudget::auto_detect()
    };
    if let Some(mb) = max_ram_mb_opt {
        budget = budget.with_max_ram_mb(mb);
    }
    if let Some(t) = threads_opt {
        budget = budget.with_max_threads(t);
    }
    if let Some(kv) = kv_cache_opt {
        budget.default_kv_precision = kv;
    }

    let eval_target_dir = if let Some(ref qdir) = quantize_out {
        let prec = bits_opt.unwrap_or(crate::neural::WeightPrecision::Int8);
        let st = export_quantized_package(&ckpt_dir, qdir, prec).map_err(CliError::runtime)?;
        let _ = writeln!(
            err,
            "[SMQ1 exportado | bits={} | disco {:.2} MB -> {:.2} MB ({:.2}x) | RAM pesos {:.2} MB -> {:.2} MB]",
            st.precision.as_str(),
            st.src_disk_mb,
            st.dst_disk_mb,
            st.compression_ratio,
            st.src_ram_mb,
            st.dst_ram_mb
        );
        qdir.clone()
    } else {
        ckpt_dir
    };

    let mut engine =
        crate::neural::inference::NeuralEngine::load_from_dir_with_budget(&eval_target_dir, budget)
            .map_err(CliError::runtime)?;
    if let Some(kv) = kv_cache_opt {
        engine = engine.with_kv_precision(kv);
    }

    if quantize_out.is_none() {
        if let Some(prec) = bits_opt {
            if !engine.model.is_quantized() {
                engine.model = crate::neural::quantize_model_in_memory(&engine.model, prec)
                    .map_err(CliError::runtime)?;
            }
        }
    }

    let val_texts = load_seed_corpus(&dataset_path).map_err(CliError::runtime)?;
    let summary = crate::eval::neural_eval::evaluate_neural_engine(&engine, &val_texts);
    let mut json_val = serde_json::to_value(&summary).map_err(|e| {
        CliError::runtime(format!("Falha ao serializar resumo de avaliação: {}", e))
    })?;
    if let Some(obj) = json_val.as_object_mut() {
        obj.insert(
            "runtime_profile".to_string(),
            serde_json::Value::String(budget.profile.as_str().to_string()),
        );
        obj.insert(
            "active_cpu_threads".to_string(),
            serde_json::json!(crate::neural::ops::effective_worker_threads()),
        );
        obj.insert(
            "weight_precision".to_string(),
            serde_json::Value::String(engine.precision_label().to_string()),
        );
        obj.insert(
            "kv_cache_precision".to_string(),
            serde_json::Value::String(engine.kv_precision.as_str().to_string()),
        );
        obj.insert(
            "weight_ram_mb".to_string(),
            serde_json::json!(
                (engine.weight_bytes() as f64 / (1024.0 * 1024.0) * 1000.0).round() / 1000.0
            ),
        );
        obj.insert(
            "estimated_runtime_ram_mb".to_string(),
            serde_json::json!(
                (engine.estimated_runtime_bytes() as f64 / (1024.0 * 1024.0) * 1000.0).round()
                    / 1000.0
            ),
        );
    }
    let json = serde_json::to_string_pretty(&json_val)
        .map_err(|e| CliError::runtime(format!("Falha ao formatar JSON de avaliação: {}", e)))?;
    let _ = writeln!(out, "{}", json);
    Ok(())
}
