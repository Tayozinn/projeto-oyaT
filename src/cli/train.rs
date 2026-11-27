pub(super) use crate::app::neural_service::DEFAULT_TRAIN_STEPS;
use crate::app::neural_service::{
    execute_eval_neural, execute_train_neural, EvalNeuralRequest, TrainNeuralRequest,
};
use crate::cli::error::{CliError, CliResult};
use crate::eval::runner::parse_duration_str;
use clap::error::ErrorKind as ClapErrorKind;
use clap::{ArgAction, Error as ClapError, Parser};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

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

fn parse_positive_usize_clap(raw: &str) -> Result<usize, String> {
    let v: usize = raw
        .parse()
        .map_err(|_| format!("valor inválido '{}': esperado inteiro maior que zero.", raw))?;
    if v == 0 {
        return Err("o valor deve ser maior que zero.".to_string());
    }
    Ok(v)
}

fn parse_positive_u64_clap(raw: &str) -> Result<u64, String> {
    let v: u64 = raw
        .parse()
        .map_err(|_| format!("valor inválido '{}': esperado inteiro maior que zero.", raw))?;
    if v == 0 {
        return Err("o valor deve ser maior que zero.".to_string());
    }
    Ok(v)
}

fn parse_threads_clap(raw: &str) -> Result<usize, String> {
    Ok(parse_positive_usize_clap(raw)?.min(64))
}

fn parse_positive_f32_clap(raw: &str) -> Result<f32, String> {
    let v: f32 = raw.parse().map_err(|_| {
        format!(
            "valor inválido '{}': esperado número decimal maior que zero.",
            raw
        )
    })?;
    if !v.is_finite() || v <= 0.0 {
        return Err("o valor deve ser maior que zero.".to_string());
    }
    Ok(v)
}

pub(crate) fn format_clap_error_pt(err: &ClapError, subcommand_label: &str) -> String {
    let raw = err.to_string();
    match err.kind() {
        ClapErrorKind::UnknownArgument => {
            format!(
                "Argumento ou opção desconhecida em `{}`.\nDetalhe: {}",
                subcommand_label,
                raw.trim()
            )
        }
        ClapErrorKind::ArgumentConflict => {
            format!(
                "Opção informada mais de uma vez ou em conflito em `{}`.\nDetalhe: {}",
                subcommand_label,
                raw.trim()
            )
        }
        _ => format!(
            "Argumentos inválidos em `{}`.\nDetalhe: {}",
            subcommand_label,
            raw.trim()
        ),
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "train-neural",
    disable_help_flag = true,
    disable_version_flag = true,
    allow_negative_numbers = true
)]
struct TrainNeuralClapArgs {
    #[arg(long = "help", short = 'h', action = ArgAction::SetTrue)]
    help: bool,

    #[arg(long = "marathon", action = ArgAction::SetTrue)]
    marathon: bool,

    #[arg(long = "max-duration", value_parser = parse_duration_str)]
    max_duration: Option<Duration>,

    #[arg(long = "eval-every-secs", value_parser = parse_positive_u64_clap)]
    eval_every_secs: Option<u64>,

    #[arg(long = "dataset")]
    dataset: Option<PathBuf>,

    #[arg(long = "corpus-dir", action = ArgAction::Append)]
    corpus_dirs: Vec<PathBuf>,

    #[arg(long = "chunk-chars", default_value_t = 1200, value_parser = parse_positive_usize_clap)]
    chunk_chars: usize,

    #[arg(long = "out")]
    out_dir: Option<PathBuf>,

    #[arg(long = "run-name")]
    run_name: Option<String>,

    #[arg(long = "resume", action = ArgAction::SetTrue)]
    resume: bool,

    #[arg(long = "steps", value_parser = parse_positive_usize_clap)]
    steps: Option<usize>,

    #[arg(long = "d-model", value_parser = parse_positive_usize_clap)]
    d_model: Option<usize>,

    #[arg(long = "layers", value_parser = parse_positive_usize_clap)]
    layers: Option<usize>,

    #[arg(long = "heads", value_parser = parse_positive_usize_clap)]
    heads: Option<usize>,

    #[arg(long = "ff", value_parser = parse_positive_usize_clap)]
    ff: Option<usize>,

    #[arg(long = "seq-len", value_parser = parse_positive_usize_clap)]
    seq_len: Option<usize>,

    #[arg(long = "batch-size", value_parser = parse_positive_usize_clap)]
    batch_size: Option<usize>,

    #[arg(long = "vocab", value_parser = parse_positive_usize_clap)]
    vocab: Option<usize>,

    #[arg(long = "lr", value_parser = parse_positive_f32_clap)]
    lr: Option<f32>,

    #[arg(long = "min-lr", value_parser = parse_positive_f32_clap)]
    min_lr: Option<f32>,

    #[arg(long = "max-val-windows", default_value_t = 128, value_parser = parse_positive_usize_clap)]
    max_val_windows: usize,

    #[arg(
        long = "device",
        default_value = "auto",
        value_parser = crate::neural::ComputeDevice::parse
    )]
    device: crate::neural::ComputeDevice,

    #[arg(long = "export-quant")]
    export_quant: Option<PathBuf>,

    #[arg(
        long = "bits",
        default_value = "int8",
        value_parser = crate::neural::WeightPrecision::parse
    )]
    bits: crate::neural::WeightPrecision,

    #[arg(long = "threads", value_parser = parse_threads_clap)]
    threads: Option<usize>,
}

#[derive(Debug, Parser)]
#[command(
    name = "eval-neural",
    disable_help_flag = true,
    disable_version_flag = true,
    allow_negative_numbers = true
)]
struct EvalNeuralClapArgs {
    #[arg(long = "help", short = 'h', action = ArgAction::SetTrue)]
    help: bool,

    #[arg(long = "dir")]
    dir: Option<PathBuf>,

    #[arg(long = "dataset")]
    dataset: Option<PathBuf>,

    #[arg(long = "quantize-out")]
    quantize_out: Option<PathBuf>,

    #[arg(long = "bits", value_parser = crate::neural::WeightPrecision::parse)]
    bits: Option<crate::neural::WeightPrecision>,

    #[arg(long = "kv-cache", value_parser = crate::neural::KvCachePrecision::parse)]
    kv_cache: Option<crate::neural::KvCachePrecision>,

    #[arg(long = "low-ram", action = ArgAction::SetTrue)]
    low_ram: bool,

    #[arg(long = "max-ram-mb", value_parser = parse_positive_usize_clap)]
    max_ram_mb: Option<usize>,

    #[arg(long = "threads", value_parser = parse_threads_clap)]
    threads: Option<usize>,
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
    if args.iter().any(|a| a == "--help" || a == "-h")
        && !args.iter().any(|a| a.starts_with("--help="))
    {
        // Valida que não há flags desconhecidas/malformadas junto com --help se for apenas --help
        if args.len() == 1 {
            return Ok(TrainCliParsed::Help);
        }
    }
    let parsed = TrainNeuralClapArgs::try_parse_from(
        std::iter::once("train-neural").chain(args.iter().map(|s| s.as_str())),
    )
    .map_err(|e| format_clap_error_pt(&e, "train-neural"))?;

    if parsed.help {
        return Ok(TrainCliParsed::Help);
    }

    Ok(TrainCliParsed::Run(Box::new(TrainNeuralRequest {
        dataset_path: parsed.dataset,
        corpus_dirs: parsed.corpus_dirs,
        chunk_chars: parsed.chunk_chars,
        out_dir_opt: parsed.out_dir,
        run_name: parsed.run_name,
        resume: parsed.resume,
        max_steps_opt: parsed.steps,
        max_duration_opt: parsed.max_duration,
        eval_every_secs_opt: parsed.eval_every_secs,
        marathon: parsed.marathon,
        d_model_opt: parsed.d_model,
        n_heads_opt: parsed.heads,
        n_layers_opt: parsed.layers,
        d_ff_opt: parsed.ff,
        seq_len_opt: parsed.seq_len,
        batch_size_opt: parsed.batch_size,
        vocab_size_opt: parsed.vocab,
        lr_opt: parsed.lr,
        min_learning_rate: parsed.min_lr,
        max_val_windows: parsed.max_val_windows,
        device: parsed.device,
        export_quant_dir: parsed.export_quant,
        export_bits: parsed.bits,
        threads_opt: parsed.threads,
    })))
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
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        return Ok(EvalNeuralCliParsed::Help);
    }
    let parsed = EvalNeuralClapArgs::try_parse_from(
        std::iter::once("eval-neural").chain(args.iter().map(|s| s.as_str())),
    )
    .map_err(|e| format_clap_error_pt(&e, "eval-neural"))?;

    if parsed.help {
        return Ok(EvalNeuralCliParsed::Help);
    }

    Ok(EvalNeuralCliParsed::Run(EvalNeuralRequest {
        ckpt_dir_opt: parsed.dir,
        dataset_path_opt: parsed.dataset,
        quantize_out: parsed.quantize_out,
        bits_opt: parsed.bits,
        kv_cache_opt: parsed.kv_cache,
        low_ram: parsed.low_ram,
        max_ram_mb_opt: parsed.max_ram_mb,
        threads_opt: parsed.threads,
    }))
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
            (&["--flag-invalida"], "desconhecid"),
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
            (&["--desconhecido"], "desconhecid"),
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
