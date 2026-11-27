pub(super) use crate::app::neural_service::DEFAULT_TRAIN_STEPS;
use crate::app::neural_service::{
    execute_eval_neural, execute_train_neural, EvalNeuralRequest, TrainNeuralRequest,
};
use crate::cli::error::{CliError, CliResult};
use crate::eval::runner::parse_duration_str;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) enum TrainCliParsed {
    Help,
    Run(Box<TrainNeuralRequest>),
}

#[derive(Debug)]
pub(crate) enum EvalNeuralCliParsed {
    Help,
    Run(EvalNeuralRequest),
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
    let mut p = TrainNeuralRequest {
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

pub fn run_train_neural_cli_result(
    args: &[String],
    install_ctrlc_handler: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> CliResult<()> {
    let parsed = match parse_train_neural_args(args) {
        Ok(TrainCliParsed::Help) => {
            print_train_neural_help(out);
            return Ok(());
        }
        Ok(TrainCliParsed::Run(p)) => *p,
        Err(msg) => return Err(CliError::usage(msg)),
    };

    execute_train_neural(parsed, install_ctrlc_handler, cancel_flag, out, err)
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
    match run_train_neural_cli_result(args, install_ctrlc_handler, cancel_flag, out, err) {
        Ok(()) => 0,
        Err(e) => e.emit_and_code(err),
    }
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
    let mut p = EvalNeuralRequest {
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

pub fn run_eval_neural_cli_result(
    args: &[String],
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> CliResult<()> {
    let parsed = match parse_eval_neural_args(args) {
        Ok(EvalNeuralCliParsed::Help) => {
            print_eval_neural_help(out);
            return Ok(());
        }
        Ok(EvalNeuralCliParsed::Run(p)) => p,
        Err(msg) => return Err(CliError::usage(msg)),
    };

    execute_eval_neural(parsed, out, err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDirGuard;
    use std::fs;
    use std::time::Duration;

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

    /// `T10` (Fase 4): teste ponta a ponta da maratona e cancelamento movido de `neural::training`
    /// para os testes de `cli::train`, respeitando a hierarquia `cli -> app -> neural`.
    #[test]
    fn sm009_timed_marathon_and_cancel_flag_generate_evolution_sequence_and_clean_checkpoint() {
        let dir = TempDirGuard::new("sm009_marathon");
        let q_dir = TempDirGuard::new("sm009_marathon_int8");
        let ds_file = dir.path.join("mini_qa.jsonl");
        let books_dir = dir.path.join("mini_books");
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

        let out_ckpt = dir.path.join("ckpt");
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
            q_dir.path.to_string_lossy().to_string(),
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
        let code = run_train_neural_cli_with_cancel(&args, false, None, &mut out_buf, &mut err_buf);
        assert_eq!(code, 0, "stderr={}", String::from_utf8_lossy(&err_buf));
        assert!(out_ckpt.join("best.bin").is_file());
        assert!(out_ckpt.join("latest.bin").is_file());
        assert!(out_ckpt.join("training_evolution.md").is_file());
        assert!(out_ckpt.join("evolution_log.jsonl").is_file());
        assert!(q_dir.path.join("quantized.smq").is_file());

        let md = fs::read_to_string(out_ckpt.join("training_evolution.md")).unwrap();
        assert!(md.contains("Marco #00"));
        assert!(md.contains("Marco Final Consolidado"));

        let cancel_flag = Arc::new(AtomicBool::new(true));
        let cancel_dir = dir.path.join("ckpt_cancel");
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
        let code_cancel = run_train_neural_cli_with_cancel(
            &cancel_args,
            false,
            Some(cancel_flag),
            &mut out_buf,
            &mut err_buf,
        );
        assert_eq!(code_cancel, 130);
        assert!(cancel_dir.join("best.bin").is_file());
    }
}
