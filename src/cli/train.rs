use super::marathon::{
    format_hms, run_autonomous_marathon, EvolutionSequenceRecorder, MarathonParams,
};
use super::quantize::export_quantized_package;
use crate::eval::runner::parse_duration_str;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

pub(super) fn load_seed_corpus(dataset_path: &std::path::Path) -> Result<Vec<String>, String> {
    let raw = std::fs::read_to_string(dataset_path)
        .map_err(|e| format!("Falha ao ler dataset '{}': {}", dataset_path.display(), e))?;
    let mut examples = Vec::new();
    for (idx, line) in raw.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
            format!(
                "Linha JSON inválida em '{}' (linha {}): {}",
                dataset_path.display(),
                line_no,
                e
            )
        })?;
        let dialogue = if let Some(turns_val) = val.get("turns") {
            let turns = turns_val.as_array().ok_or_else(|| {
                format!(
                    "Campo 'turns' inválido em '{}' (linha {}): esperado array não vazio.",
                    dataset_path.display(),
                    line_no
                )
            })?;
            if turns.is_empty() {
                return Err(format!(
                    "Campo 'turns' vazio em '{}' (linha {}).",
                    dataset_path.display(),
                    line_no
                ));
            }
            let mut buf = String::new();
            for (t_idx, turn) in turns.iter().enumerate() {
                let u = turn
                    .get("user")
                    .or_else(|| turn.get("prompt"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim())
                    .unwrap_or("");
                let a = turn
                    .get("assistant")
                    .or_else(|| turn.get("response"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim())
                    .unwrap_or("");
                if u.is_empty() || a.is_empty() {
                    return Err(format!(
                        "Turno #{} incompleto em '{}' (linha {}): 'user' e 'assistant' são obrigatórios e não podem ser vazios.",
                        t_idx + 1,
                        dataset_path.display(),
                        line_no
                    ));
                }
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(&format!("<user> {}\n<assistant> {}", u, a));
            }
            buf
        } else {
            let prompt = val
                .get("prompt")
                .and_then(|v| v.as_str())
                .map(|s| s.trim())
                .unwrap_or("");
            let response = val
                .get("response")
                .and_then(|v| v.as_str())
                .map(|s| s.trim())
                .unwrap_or("");
            if prompt.is_empty() || response.is_empty() {
                return Err(format!(
                    "Registro inválido em '{}' (linha {}): campos 'prompt' e 'response' são obrigatórios e não podem ser vazios.",
                    dataset_path.display(),
                    line_no
                ));
            }
            format!("<user> {}\n<assistant> {}", prompt, response)
        };
        let norm = crate::neural::tokenizer::normalize_corpus_text(&dialogue);
        if !norm.is_empty() {
            examples.push(norm);
        }
    }
    if examples.is_empty() {
        return Err("Nenhum exemplo válido encontrado no dataset.".to_string());
    }
    Ok(examples)
}

pub(super) fn check_no_existing_training_artifacts(
    out_dir: &std::path::Path,
) -> Result<(), String> {
    const PROTECTED_FILES: &[&str] = &[
        "best.bin",
        "latest.bin",
        "metrics.jsonl",
        "training_evolution.md",
        "evolution_log.jsonl",
    ];
    let existing: Vec<&str> = PROTECTED_FILES
        .iter()
        .copied()
        .filter(|name| out_dir.join(name).exists())
        .collect();
    if !existing.is_empty() {
        return Err(format!(
            "O diretório '{}' já contém artefatos de treinamento ({}). Para não perder progresso, use `--resume` para continuar ou informe um novo diretório em `--out`.",
            out_dir.display(),
            existing.join(", ")
        ));
    }
    Ok(())
}

pub(super) const DEFAULT_TRAIN_STEPS: usize = 80;

#[derive(Debug)]
pub(crate) enum TrainCliParsed {
    Help,
    Run(Box<ParsedTrainNeuralArgs>),
}

#[derive(Debug)]
pub(crate) struct ParsedTrainNeuralArgs {
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

fn print_train_neural_help(out: &mut dyn Write) {
    let _ = writeln!(out, "Uso: servermind-ai train-neural [opções]");
    let _ = writeln!(out, "  --marathon                  Ativa Maratona Autônoma de 6h em 2 fases (Pré-treino + Fine-Tuning Chat + Sequência Evolutiva + SMQ1 INT8)");
    let _ = writeln!(out, "  --max-duration <DUR>        Controla o tempo real de treino (ex.: 30s, 20m, 6h, 1h30m; salva limpo ao atingir ou em Ctrl+C)");
    let _ = writeln!(out, "  --eval-every-secs <N>       Intervalo em segundos entre marcos da sequência evolutiva (automático se omitido)");
    let _ = writeln!(out, "  --dataset <arquivo.jsonl>   Corpus Q&A em JSONL (padrão se --corpus-dir não for informado)");
    let _ = writeln!(out, "  --corpus-dir <pasta>        Diretório recursivo de texto/código para pré-treino (repetível)");
    let _ = writeln!(out, "  --chunk-chars <N>           Tamanho máximo de bloco por arquivo de corpus (padrão: 1200)");
    let _ = writeln!(
        out,
        "  --out <pasta>               Diretório de saída dos checkpoints"
    );
    let _ = writeln!(
        out,
        "  --run-name <nome>           Nome da execução nos relatórios"
    );
    let _ = writeln!(
        out,
        "  --resume                    Retoma latest.bin e tokenizer.json existentes em --out"
    );
    let _ = writeln!(out, "  --steps <N>                 Número alvo de passos (padrão: 80 se --max-duration não for usado)");
    let _ = writeln!(out, "  --d-model <N> --layers <N> --heads <N> --ff <N> --seq-len <N> --vocab <N> --batch-size <N>");
    let _ = writeln!(
        out,
        "  --lr <f32> --min-lr <f32>   Taxa de aprendizado inicial e mínima (cosine decay)"
    );
    let _ = writeln!(out, "  --max-val-windows <N>       Limite uniforme de janelas na validação periódica (padrão: 128)");
    let _ = writeln!(out, "  --device <auto|cpu|gpu>     Backend computacional (padrão: auto; gpu usa Vulkan RX 7600)");
    let _ = writeln!(out, "  --export-quant <pasta>      Exporta pacote quantizado SMQ1 ('quantized.smq' + 'tokenizer.json') ao concluir");
    let _ = writeln!(
        out,
        "  --bits <fp16|bf16|int8|int4> Precisão para --export-quant (padrão: int8)"
    );
    let _ = writeln!(
        out,
        "  --threads <N>               Limita o número máximo de threads CPU ativas (1..=64)"
    );
}

pub(crate) fn parse_train_neural_args(args: &[String]) -> Result<TrainCliParsed, String> {
    let mut p = ParsedTrainNeuralArgs {
        dataset_path: None,
        corpus_dirs: Vec::new(),
        chunk_chars: 1200,
        out_dir_opt: None,
        run_name: None,
        resume: false,
        max_steps_opt: None,
        max_duration_opt: None,
        eval_every_secs_opt: None,
        marathon: false,
        d_model_opt: None,
        n_heads_opt: None,
        n_layers_opt: None,
        d_ff_opt: None,
        seq_len_opt: None,
        batch_size_opt: None,
        vocab_size_opt: None,
        lr_opt: None,
        min_learning_rate: None,
        max_val_windows: 128,
        device: crate::neural::ComputeDevice::Auto,
        export_quant_dir: None,
        export_bits: crate::neural::WeightPrecision::Int8,
        threads_opt: None,
    };

    let parse_pos_usize = |flag: &str, raw: &str| -> Result<usize, String> {
        let v: usize = raw.parse().map_err(|_| {
            format!(
                "Valor inválido '{}' para '{}': esperado inteiro positivo.",
                raw, flag
            )
        })?;
        if v == 0 {
            return Err(format!("'{}' deve ser maior que zero.", flag));
        }
        Ok(v)
    };

    let parse_pos_f32 = |flag: &str, raw: &str| -> Result<f32, String> {
        let v: f32 = raw.parse().map_err(|_| {
            format!(
                "Valor inválido '{}' para '{}': esperado número decimal positivo.",
                raw, flag
            )
        })?;
        if !v.is_finite() || v <= 0.0 {
            return Err(format!("'{}' deve ser maior que zero.", flag));
        }
        Ok(v)
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => return Ok(TrainCliParsed::Help),
            "--marathon" => {
                p.marathon = true;
                i += 1;
            }
            "--max-duration" if i + 1 < args.len() => {
                p.max_duration_opt = Some(parse_duration_str(&args[i + 1])?);
                i += 2;
            }
            "--eval-every-secs" if i + 1 < args.len() => {
                p.eval_every_secs_opt =
                    Some(parse_pos_usize("--eval-every-secs", &args[i + 1])? as u64);
                i += 2;
            }
            "--threads" if i + 1 < args.len() => {
                p.threads_opt = Some(parse_pos_usize("--threads", &args[i + 1])?.min(64));
                i += 2;
            }
            "--export-quant" if i + 1 < args.len() => {
                p.export_quant_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--bits" if i + 1 < args.len() => {
                p.export_bits = crate::neural::WeightPrecision::parse(&args[i + 1])?;
                i += 2;
            }
            "--device" if i + 1 < args.len() => {
                p.device = crate::neural::ComputeDevice::parse(&args[i + 1])?;
                i += 2;
            }
            "--dataset" if i + 1 < args.len() => {
                p.dataset_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--corpus-dir" if i + 1 < args.len() => {
                p.corpus_dirs.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--chunk-chars" if i + 1 < args.len() => {
                p.chunk_chars = parse_pos_usize("--chunk-chars", &args[i + 1])?;
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                p.out_dir_opt = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--run-name" if i + 1 < args.len() => {
                p.run_name = Some(args[i + 1].clone());
                i += 2;
            }
            "--resume" => {
                p.resume = true;
                i += 1;
            }
            "--lr" if i + 1 < args.len() => {
                p.lr_opt = Some(parse_pos_f32("--lr", &args[i + 1])?);
                i += 2;
            }
            "--min-lr" if i + 1 < args.len() => {
                p.min_learning_rate = Some(parse_pos_f32("--min-lr", &args[i + 1])?);
                i += 2;
            }
            "--max-val-windows" if i + 1 < args.len() => {
                p.max_val_windows = parse_pos_usize("--max-val-windows", &args[i + 1])?;
                i += 2;
            }
            "--steps" if i + 1 < args.len() => {
                p.max_steps_opt = Some(parse_pos_usize("--steps", &args[i + 1])?);
                i += 2;
            }
            "--d-model" if i + 1 < args.len() => {
                p.d_model_opt = Some(parse_pos_usize("--d-model", &args[i + 1])?);
                i += 2;
            }
            "--layers" if i + 1 < args.len() => {
                p.n_layers_opt = Some(parse_pos_usize("--layers", &args[i + 1])?);
                i += 2;
            }
            "--heads" if i + 1 < args.len() => {
                p.n_heads_opt = Some(parse_pos_usize("--heads", &args[i + 1])?);
                i += 2;
            }
            "--ff" if i + 1 < args.len() => {
                p.d_ff_opt = Some(parse_pos_usize("--ff", &args[i + 1])?);
                i += 2;
            }
            "--seq-len" if i + 1 < args.len() => {
                p.seq_len_opt = Some(parse_pos_usize("--seq-len", &args[i + 1])?);
                i += 2;
            }
            "--batch-size" if i + 1 < args.len() => {
                p.batch_size_opt = Some(parse_pos_usize("--batch-size", &args[i + 1])?);
                i += 2;
            }
            "--vocab" if i + 1 < args.len() => {
                p.vocab_size_opt = Some(parse_pos_usize("--vocab", &args[i + 1])?);
                i += 2;
            }
            other => {
                return Err(format!(
                    "Argumento desconhecido em train-neural: '{}'",
                    other
                ))
            }
        }
    }
    Ok(TrainCliParsed::Run(Box::new(p)))
}

pub fn run_train_neural_cli(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    run_train_neural_cli_with_cancel(args, true, None, out, err)
}

pub fn run_train_neural_cli_with_cancel(
    args: &[String],
    install_ctrlc_handler: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    let parsed = match parse_train_neural_args(args) {
        Ok(TrainCliParsed::Help) => {
            print_train_neural_help(out);
            return 0;
        }
        Ok(TrainCliParsed::Run(p)) => *p,
        Err(msg) => {
            let _ = writeln!(err, "{}", msg);
            return 2;
        }
    };

    let ParsedTrainNeuralArgs {
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
    } = parsed;

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

    let mut examples = Vec::new();
    for dir in &corpus_dirs {
        match crate::neural::tokenizer::load_corpus_directory(dir, chunk_chars) {
            Ok(mut chunks) => examples.append(&mut chunks),
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        }
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
        match load_seed_corpus(&ds_path) {
            Ok(mut ds_examples) => examples.append(&mut ds_examples),
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        }
    }

    if examples.is_empty() {
        let _ = writeln!(err, "Nenhum exemplo válido carregado para treinamento.");
        return 1;
    }

    let resolved_run_name = run_name.unwrap_or_else(|| {
        out_dir
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("servermind-neural")
            .to_string()
    });
    let resolved_min_lr = min_learning_rate.unwrap_or(learning_rate * 0.1);
    if resolved_min_lr > learning_rate {
        let _ = writeln!(
            err,
            "'--min-lr' ({}) não pode ser maior que '--lr' ({}).",
            resolved_min_lr, learning_rate
        );
        return 2;
    }

    let split = crate::neural::tokenizer::split_dataset(&examples, &resolved_run_name, 0.15, 0.10);
    let latest_ckpt = out_dir.join("latest.bin");
    let tok_path = out_dir.join("tokenizer.json");

    let (tokenizer, mut trainer) = if resume {
        if !tok_path.is_file() || !latest_ckpt.is_file() {
            let _ = writeln!(
                err,
                "Flag --resume exige '{}' e '{}' existentes.",
                tok_path.display(),
                latest_ckpt.display()
            );
            return 1;
        }
        let tokenizer = match crate::neural::tokenizer::BpeTokenizer::load(&tok_path) {
            Ok(t) => t,
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        };
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
            match crate::neural::training::Trainer::resume_from_checkpoint(&latest_ckpt, train_cfg)
            {
                Ok(t) => t,
                Err(e) => {
                    let _ = writeln!(err, "{}", e);
                    return 1;
                }
            };
        if tokenizer.vocab_size() != resumed.model.config.vocab_size {
            let _ = writeln!(
                err,
                "Incompatibilidade em --resume: tokenizer tem vocab={} mas checkpoint tem vocab={}.",
                tokenizer.vocab_size(),
                resumed.model.config.vocab_size
            );
            return 1;
        }
        if max_duration_opt.is_none() {
            if let Some(explicit_steps) = max_steps_opt {
                if explicit_steps <= resumed.current_step {
                    let _ = writeln!(
                        err,
                        "'--steps' ({}) deve ser maior que o passo atual do checkpoint ({}).",
                        explicit_steps, resumed.current_step
                    );
                    return 2;
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
        if let Err(msg) = check_no_existing_training_artifacts(&out_dir) {
            let _ = writeln!(err, "{}", msg);
            return 1;
        }
        let corpus_refs: Vec<&str> = split.train.iter().map(|s| s.as_str()).collect();
        let tokenizer =
            crate::neural::tokenizer::BpeTokenizer::train_from_corpus(&corpus_refs, vocab_size);

        if let Err(e) = std::fs::create_dir_all(&out_dir) {
            let _ = writeln!(err, "Falha ao criar '{}': {}", out_dir.display(), e);
            return 1;
        }
        if let Err(e) = tokenizer.save(&tok_path) {
            let _ = writeln!(err, "{}", e);
            return 1;
        }

        let cfg = crate::neural::TransformerConfig {
            vocab_size: tokenizer.vocab_size(),
            max_seq_len: seq_len,
            d_model,
            n_heads,
            n_layers,
            d_ff,
        };
        let model = match crate::neural::ServerMindTransformer::new_random(cfg, 2026) {
            Ok(m) => m,
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        };
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

    match run_res {
        Ok(report) => {
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
                match export_quantized_package(&out_dir, qdir, export_bits) {
                    Ok(info) => {
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
                    Err(e) => {
                        let _ = writeln!(err, "Erro ao exportar pacote quantizado: {}", e);
                        return 1;
                    }
                }
            }
            if report.cancelled {
                130
            } else {
                0
            }
        }
        Err(e) => {
            let _ = writeln!(err, "Erro no treinamento: {}", e);
            1
        }
    }
}

#[derive(Debug)]
pub(crate) enum EvalNeuralCliParsed {
    Help,
    Run(ParsedEvalNeuralArgs),
}

#[derive(Debug)]
pub(crate) struct ParsedEvalNeuralArgs {
    pub(crate) ckpt_dir_opt: Option<PathBuf>,
    pub(crate) dataset_path_opt: Option<PathBuf>,
    pub(crate) quantize_out: Option<PathBuf>,
    pub(crate) bits_opt: Option<crate::neural::WeightPrecision>,
    pub(crate) kv_cache_opt: Option<crate::neural::KvCachePrecision>,
    pub(crate) low_ram: bool,
    pub(crate) max_ram_mb_opt: Option<usize>,
    pub(crate) threads_opt: Option<usize>,
}

fn print_eval_neural_help(out: &mut dyn Write) {
    let _ = writeln!(out, "Uso: servermind-ai eval-neural [opções]");
    let _ = writeln!(
        out,
        "  --dir <pasta>                 Diretório do checkpoint ('best.bin', 'quantized.smq' ou 'latest.bin')"
    );
    let _ = writeln!(
        out,
        "  --dataset <arquivo.jsonl>     Dataset JSONL de validação (padrão: datasets/servermind_seed_v1.jsonl)"
    );
    let _ = writeln!(
        out,
        "  --quantize-out <pasta>        Exporta pacote quantizado SMQ1 para <pasta> e avalia o pacote exportado"
    );
    let _ = writeln!(
        out,
        "  --bits <fp16|bf16|int8|int4>  Precisão de pesos em memória ou para --quantize-out (padrão com --quantize-out: int8)"
    );
    let _ = writeln!(
        out,
        "  --kv-cache <fp32|fp16|int8>   Precisão do KV Cache durante a geração autoregressiva"
    );
    let _ = writeln!(
        out,
        "  --low-ram                     Ativa perfil Low-End 8 GB RAM (teto 2048 MB + KV Cache FP16 padrão)"
    );
    let _ = writeln!(
        out,
        "  --max-ram-mb <N>              Impõe teto máximo estrito de RAM (pesos + KV Cache + prefill) em MiB"
    );
    let _ = writeln!(
        out,
        "  --threads <N>                 Limita o número de threads CPU ativas (1..=64)"
    );
}

pub(crate) fn parse_eval_neural_args(args: &[String]) -> Result<EvalNeuralCliParsed, String> {
    let mut p = ParsedEvalNeuralArgs {
        ckpt_dir_opt: None,
        dataset_path_opt: None,
        quantize_out: None,
        bits_opt: None,
        kv_cache_opt: None,
        low_ram: false,
        max_ram_mb_opt: None,
        threads_opt: None,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => return Ok(EvalNeuralCliParsed::Help),
            "--dir" if i + 1 < args.len() => {
                p.ckpt_dir_opt = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--dataset" if i + 1 < args.len() => {
                p.dataset_path_opt = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--quantize-out" if i + 1 < args.len() => {
                p.quantize_out = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--bits" if i + 1 < args.len() => {
                p.bits_opt = Some(crate::neural::WeightPrecision::parse(&args[i + 1])?);
                i += 2;
            }
            "--kv-cache" if i + 1 < args.len() => {
                p.kv_cache_opt = Some(crate::neural::KvCachePrecision::parse(&args[i + 1])?);
                i += 2;
            }
            "--low-ram" => {
                p.low_ram = true;
                i += 1;
            }
            "--max-ram-mb" if i + 1 < args.len() => {
                match args[i + 1].parse::<usize>() {
                    Ok(v) if v > 0 => p.max_ram_mb_opt = Some(v),
                    _ => {
                        return Err(
                            "Valor inválido para '--max-ram-mb': esperado inteiro > 0.".to_string()
                        );
                    }
                }
                i += 2;
            }
            "--threads" if i + 1 < args.len() => {
                match args[i + 1].parse::<usize>() {
                    Ok(v) if v > 0 => p.threads_opt = Some(v.min(64)),
                    _ => {
                        return Err(
                            "Valor inválido para '--threads': esperado inteiro > 0.".to_string()
                        );
                    }
                }
                i += 2;
            }
            other => {
                return Err(format!(
                    "Argumento desconhecido em eval-neural: '{}'",
                    other
                ))
            }
        }
    }
    Ok(EvalNeuralCliParsed::Run(p))
}

pub fn run_eval_neural_cli(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let parsed = match parse_eval_neural_args(args) {
        Ok(EvalNeuralCliParsed::Help) => {
            print_eval_neural_help(out);
            return 0;
        }
        Ok(EvalNeuralCliParsed::Run(p)) => p,
        Err(msg) => {
            let _ = writeln!(err, "{}", msg);
            return 2;
        }
    };

    let ckpt_dir = parsed.ckpt_dir_opt.unwrap_or_else(|| {
        crate::chat::config::AppConfig::resolve_from_config_dir("checkpoints/servermind-neural-v1")
    });
    let dataset_path = parsed.dataset_path_opt.unwrap_or_else(|| {
        crate::chat::config::AppConfig::resolve_from_config_dir("datasets/servermind_seed_v1.jsonl")
    });
    let quantize_out = parsed.quantize_out;
    let bits_opt = parsed.bits_opt;
    let kv_cache_opt = parsed.kv_cache_opt;
    let low_ram = parsed.low_ram;
    let max_ram_mb_opt = parsed.max_ram_mb_opt;
    let threads_opt = parsed.threads_opt;

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
        match export_quantized_package(&ckpt_dir, qdir, prec) {
            Ok(st) => {
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
            }
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        }
        qdir.clone()
    } else {
        ckpt_dir
    };

    let mut engine = match crate::neural::inference::NeuralEngine::load_from_dir_with_budget(
        &eval_target_dir,
        budget,
    ) {
        Ok(e) => e,
        Err(e) => {
            let _ = writeln!(err, "{}", e);
            return 1;
        }
    };
    if let Some(kv) = kv_cache_opt {
        engine = engine.with_kv_precision(kv);
    }

    if quantize_out.is_none() {
        if let Some(prec) = bits_opt {
            if !engine.model.is_quantized() {
                match crate::neural::quantize_model_in_memory(&engine.model, prec) {
                    Ok(qm) => engine.model = qm,
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 1;
                    }
                }
            }
        }
    }

    let val_texts = match load_seed_corpus(&dataset_path) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(err, "{}", e);
            return 1;
        }
    };
    let summary = crate::eval::neural_eval::evaluate_neural_engine(&engine, &val_texts);
    let mut json_val = match serde_json::to_value(&summary) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(err, "Falha ao serializar resumo de avaliação: {}", e);
            return 1;
        }
    };
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
    let json = match serde_json::to_string_pretty(&json_val) {
        Ok(s) => s,
        Err(e) => {
            let _ = writeln!(err, "Falha ao formatar JSON de avaliação: {}", e);
            return 1;
        }
    };
    let _ = writeln!(out, "{}", json);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_vec(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn test_phase2_t4_train_neural_parser_table() {
        assert!(matches!(
            parse_train_neural_args(&to_vec(&["--help"])).unwrap(),
            TrainCliParsed::Help
        ));
        assert!(matches!(
            parse_train_neural_args(&to_vec(&["-h"])).unwrap(),
            TrainCliParsed::Help
        ));

        let TrainCliParsed::Run(p_def) = parse_train_neural_args(&to_vec(&[])).unwrap() else {
            panic!("esperava Run");
        };
        assert!(!p_def.marathon);
        assert!(!p_def.resume);
        assert_eq!(p_def.chunk_chars, 1200);
        assert_eq!(p_def.max_val_windows, 128);
        assert_eq!(p_def.export_bits, crate::neural::WeightPrecision::Int8);

        let TrainCliParsed::Run(p_full) = parse_train_neural_args(&to_vec(&[
            "--marathon",
            "--resume",
            "--max-duration",
            "2h30m",
            "--eval-every-secs",
            "120",
            "--dataset",
            "datasets/qa.jsonl",
            "--corpus-dir",
            "docs_a",
            "--corpus-dir",
            "docs_b",
            "--chunk-chars",
            "2048",
            "--out",
            "checkpoints/custom",
            "--run-name",
            "exp-alpha",
            "--steps",
            "500",
            "--d-model",
            "256",
            "--layers",
            "4",
            "--heads",
            "8",
            "--ff",
            "768",
            "--seq-len",
            "128",
            "--batch-size",
            "8",
            "--vocab",
            "2048",
            "--lr",
            "0.001",
            "--min-lr",
            "0.0001",
            "--max-val-windows",
            "64",
            "--device",
            "cpu",
            "--export-quant",
            "checkpoints/custom-q",
            "--bits",
            "int4",
            "--threads",
            "128",
        ]))
        .expect("train-neural completo válido") else {
            panic!("esperava Run");
        };
        assert!(p_full.marathon);
        assert!(p_full.resume);
        assert_eq!(p_full.max_duration_opt, Some(Duration::from_secs(9000)));
        assert_eq!(p_full.eval_every_secs_opt, Some(120));
        assert_eq!(p_full.corpus_dirs.len(), 2);
        assert_eq!(p_full.chunk_chars, 2048);
        assert_eq!(p_full.max_steps_opt, Some(500));
        assert_eq!(p_full.d_model_opt, Some(256));
        assert_eq!(p_full.n_layers_opt, Some(4));
        assert_eq!(p_full.n_heads_opt, Some(8));
        assert_eq!(p_full.d_ff_opt, Some(768));
        assert_eq!(p_full.seq_len_opt, Some(128));
        assert_eq!(p_full.batch_size_opt, Some(8));
        assert_eq!(p_full.vocab_size_opt, Some(2048));
        assert_eq!(p_full.threads_opt, Some(64), "threads deve limitar em 64");
        assert_eq!(p_full.export_bits, crate::neural::WeightPrecision::Int4);

        let invalid_table: &[(&[&str], &str)] = &[
            (&["--steps", "0"], "--steps"),
            (&["--steps", "abc"], "--steps"),
            (&["--steps"], "--steps"),
            (&["--d-model", "0"], "--d-model"),
            (&["--layers", "0"], "--layers"),
            (&["--heads", "0"], "--heads"),
            (&["--ff", "0"], "--ff"),
            (&["--seq-len", "0"], "--seq-len"),
            (&["--batch-size", "0"], "--batch-size"),
            (&["--vocab", "0"], "--vocab"),
            (&["--chunk-chars", "0"], "--chunk-chars"),
            (&["--eval-every-secs", "0"], "--eval-every-secs"),
            (&["--max-val-windows", "0"], "--max-val-windows"),
            (&["--threads", "0"], "--threads"),
            (&["--lr", "0"], "--lr"),
            (&["--lr", "-0.1"], "--lr"),
            (&["--lr", "NaN"], "--lr"),
            (&["--min-lr", "0"], "--min-lr"),
            (&["--max-duration", "0s"], "maior que zero"),
            (&["--device", "tpu"], "--device"),
            (&["--bits", "int2"], "quantização"),
            (&["--flag-invalida"], "desconhecido"),
        ];
        for (raw, expected_substr) in invalid_table {
            let err = parse_train_neural_args(&to_vec(raw))
                .err()
                .unwrap_or_else(|| panic!("deveria rejeitar {:?}", raw));
            assert!(
                err.to_lowercase().contains(&expected_substr.to_lowercase()),
                "para {:?} esperava '{}' em '{}'",
                raw,
                expected_substr,
                err
            );
        }
    }

    #[test]
    fn test_phase2_t4_eval_neural_parser_table() {
        assert!(matches!(
            parse_eval_neural_args(&to_vec(&["--help"])).unwrap(),
            EvalNeuralCliParsed::Help
        ));

        let EvalNeuralCliParsed::Run(p_ok) = parse_eval_neural_args(&to_vec(&[
            "--dir",
            "checkpoints/m1",
            "--dataset",
            "datasets/val.jsonl",
            "--quantize-out",
            "checkpoints/m1-int8",
            "--bits",
            "fp16",
            "--kv-cache",
            "int8",
            "--low-ram",
            "--max-ram-mb",
            "1536",
            "--threads",
            "99",
        ]))
        .expect("eval-neural válido") else {
            panic!("esperava Run");
        };
        assert_eq!(p_ok.ckpt_dir_opt, Some(PathBuf::from("checkpoints/m1")));
        assert_eq!(p_ok.bits_opt, Some(crate::neural::WeightPrecision::Fp16));
        assert_eq!(
            p_ok.kv_cache_opt,
            Some(crate::neural::KvCachePrecision::Int8)
        );
        assert!(p_ok.low_ram);
        assert_eq!(p_ok.max_ram_mb_opt, Some(1536));
        assert_eq!(p_ok.threads_opt, Some(64));

        let invalid_table: &[(&[&str], &str)] = &[
            (&["--dir"], "--dir"),
            (&["--dataset"], "--dataset"),
            (&["--quantize-out"], "--quantize-out"),
            (&["--bits", "fp8"], "quantização"),
            (&["--kv-cache", "int4"], "KV"),
            (&["--max-ram-mb", "0"], "--max-ram-mb"),
            (&["--max-ram-mb", "abc"], "--max-ram-mb"),
            (&["--threads", "0"], "--threads"),
            (&["--desconhecido"], "desconhecido"),
        ];
        for (raw, expected_substr) in invalid_table {
            let err = parse_eval_neural_args(&to_vec(raw))
                .err()
                .unwrap_or_else(|| panic!("deveria rejeitar {:?}", raw));
            assert!(
                err.to_lowercase().contains(&expected_substr.to_lowercase()),
                "para {:?} esperava '{}' em '{}'",
                raw,
                expected_substr,
                err
            );
        }
    }
}
