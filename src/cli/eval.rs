use crate::chat::config::AppConfig;
use crate::chat::ollama::{InstalledModelInfo, OllamaClient};
use crate::cli::error::{CliError, CliResult};
use crate::eval::dataset::{filter_cases, load_and_validate_dataset, DatasetFilter, EvalCategory};
use crate::eval::report::CaseExecutionStatus;
use crate::eval::runner::{
    default_dataset_path, parse_duration_str, run_evaluation_with_progress, validate_repeat,
    EvalProgressEvent, EvalRunConfig, EvalRunOutcome, DEFAULT_CONSECUTIVE_GLOBAL_FAILURES_LIMIT,
    DEFAULT_EVAL_REPEAT,
};
use clap::error::ErrorKind as ClapErrorKind;
use clap::{ArgAction, Error as ClapError, Parser};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

pub fn run_eval_cli_result<W: Write>(
    args: &[String],
    install_ctrlc_handler: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
    out: &mut W,
) -> CliResult<()> {
    if args.is_empty() {
        return Err(CliError::usage(
            "[ERRO DE USO]\nSubcomando de avaliação ausente.\nPróxima ação: execute `servermind-ai eval --help` para ver os subcomandos disponíveis (list-cases, list-models, run).",
        ));
    }

    match args[0].as_str() {
        "--help" | "-h" | "help" => {
            if args.len() > 1 {
                return Err(CliError::usage(format!(
                    "[ERRO DE USO]\nArgumento inesperado '{}' após '{}'.\nPróxima ação: execute `servermind-ai eval --help`.",
                    args[1], args[0]
                )));
            }
            print_eval_help(out);
            Ok(())
        }
        "list-cases" => run_eval_list_cases(&args[1..], out),
        "list-models" => run_eval_list_models(&args[1..], out),
        "run" => run_eval_run(&args[1..], install_ctrlc_handler, cancel_flag, out),
        other => {
            if other.starts_with('-') {
                Err(CliError::usage(format!(
                    "[OPÇÃO DESCONHECIDA]\nA opção '{}' não é reconhecida antes de um subcomando de avaliação.\nPróxima ação: execute `servermind-ai eval --help`.",
                    other
                )))
            } else {
                Err(CliError::usage(format!(
                    "[SUBCOMANDO DESCONHECIDO]\nO subcomando 'eval {}' não existe.\nPróxima ação: use `list-cases`, `list-models` ou `run` (veja `servermind-ai eval --help`).",
                    other
                )))
            }
        }
    }
}

/// Ponto de entrada testável para subcomandos `eval`, permitindo capturar `stdout`/`stderr`,
/// desabilitar handler global de `Ctrl+C` em testes paralelos e injetar `cancel_flag`.
#[cfg(test)]
pub fn run_eval_cli_with_options<W1: Write, W2: Write>(
    args: &[String],
    install_ctrlc_handler: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
    out: &mut W1,
    err: &mut W2,
) -> i32 {
    match run_eval_cli_result(args, install_ctrlc_handler, cancel_flag, out) {
        Ok(()) => 0,
        Err(e) => e.emit_and_code(err),
    }
}

fn print_eval_help<W: Write>(out: &mut W) {
    let _ = writeln!(
        out,
        "SERVERMIND AI — Avaliação Automatizada Offline (Estritamente Consultiva)\n"
    );
    let _ = writeln!(out, "Subcomandos:");
    let _ = writeln!(out, "  eval --help\n      Exibe esta ajuda.");
    let _ = writeln!(
        out,
        "  eval list-cases [--dataset <PATH>] [--category <CAT>] [--case <ID>]\n      Lista os casos do conjunto versionado (100% offline, sem Ollama nem SQLite)."
    );
    let _ = writeln!(
        out,
        "  eval list-models [--ollama-url <URL>] [--timeout-secs <N>]\n      Lista os modelos já instalados no Ollama local (sem baixar modelos)."
    );
    let _ = writeln!(
        out,
        "  eval run [opções]\n      Executa a suíte com concorrência fixa = 1 e gera relatórios .json e .md."
    );
    let _ = writeln!(out, "\nOpções de `eval run`:");
    let _ = writeln!(
        out,
        "  --backend <BACKEND>      Backend alvo: ollama (padrão local), neural, claude, gemini, groq, mistral, openrouter, openai ou free-*"
    );
    let _ = writeln!(
        out,
        "  --model <MODEL>          Modelo instalado ou remoto a avaliar (padrão: config.toml/env ou default do backend)"
    );
    let _ = writeln!(
        out,
        "  --dataset <PATH>         Caminho do dataset JSON (padrão: eval/cases_v1.json)"
    );
    let _ = writeln!(
        out,
        "  --category <CAT>         Filtra por categoria (repetível ou separada por vírgula)"
    );
    let _ = writeln!(
        out,
        "  --case <ID>              Filtra por ID de caso (repetível ou separado por vírgula)"
    );
    let _ = writeln!(
        out,
        "  --limit <N>              Limita a quantidade máxima de casos após o filtro (> 0)"
    );
    let _ = writeln!(
        out,
        "  --out-dir <DIR>          Pasta base para os relatórios (padrão: eval_reports)"
    );
    let _ = writeln!(
        out,
        "  --timeout-secs <N>       Timeout por requisição ao Ollama em segundos (> 0)"
    );
    let _ = writeln!(
        out,
        "  --max-duration <DUR>     Duração máxima total (ex.: 30s, 15m, 7h, 1h30m; até 7d)"
    );
    let _ = writeln!(
        out,
        "  --repeat <N>             Repetições por caso (1..={}; padrão: 1)",
        crate::eval::runner::MAX_EVAL_REPEAT
    );
    let _ = writeln!(
        out,
        "  --judge-model <MODEL>    Juiz LLM local opcional (sempre marcado como estimativa)"
    );
    let _ = writeln!(
        out,
        "  --ollama-url <URL>       URL local de loopback do Ollama (127.0.0.1/localhost/[::1])"
    );
    let _ = writeln!(
        out,
        "  --with-kb                Enriquece cada caso com notas relevantes da base de conhecimento local"
    );
    let _ = writeln!(out, "\nGarantias:");
    let _ = writeln!(
        out,
        "  - Nunca lê nem escreve `servermind.db` e nunca inclui `system_prompt` privado nos relatórios."
    );
    let _ = writeln!(
        out,
        "  - Não treina modelos, não altera pesos/Modelfiles e não executa comandos sugeridos."
    );
    let _ = writeln!(
        out,
        "  - Códigos de saída: 0 = sucesso/parada no teto `--max-duration`; 1 = erro ou falha global; 2 = uso inválido; 130 = cancelado (Ctrl+C)."
    );
}

fn run_eval_list_cases<W: Write>(args: &[String], out: &mut W) -> CliResult<()> {
    let parsed = parse_list_cases_args(args)
        .map_err(|msg| CliError::usage(format!("[ERRO DE ARGUMENTOS]\n{}", msg)))?;

    if parsed.help_requested {
        print_eval_help(out);
        return Ok(());
    }

    let dataset_path = parsed.dataset.unwrap_or_else(default_dataset_path);
    let loaded = load_and_validate_dataset(&dataset_path)
        .map_err(|msg| CliError::runtime(format!("[ERRO AO CARREGAR DATASET]\n{}", msg)))?;

    let filter = DatasetFilter {
        categories: parsed.categories,
        case_ids: parsed.case_ids,
        limit: None,
    };

    let selected = filter_cases(&loaded.dataset, &filter)
        .map_err(|msg| CliError::runtime(format!("[ERRO DE FILTRO]\n{}", msg)))?;

    let _ = writeln!(
        out,
        "Dataset v{} (schema {}) | SHA-256: {}",
        loaded.dataset.dataset_version, loaded.dataset.schema_version, loaded.sha256_hex
    );
    let _ = writeln!(
        out,
        "Casos selecionados: {} de {}\n",
        selected.len(),
        loaded.dataset.cases.len()
    );

    for case in selected {
        let review_tag = if case.requires_human_review_by_default {
            "obrigatória"
        } else {
            "condicional"
        };
        let _ = writeln!(
            out,
            "  {} | {} ({}) | comportamento={} | revisão_humana={} | {}",
            case.id,
            case.category.as_str(),
            case.category.label_pt(),
            case.expected_behavior.as_str(),
            review_tag,
            case.title
        );
    }

    Ok(())
}

fn run_eval_list_models<W: Write>(args: &[String], out: &mut W) -> CliResult<()> {
    let parsed = parse_list_models_args(args)
        .map_err(|msg| CliError::usage(format!("[ERRO DE ARGUMENTOS]\n{}", msg)))?;

    if parsed.help_requested {
        print_eval_help(out);
        return Ok(());
    }

    let (ollama_url, timeout_secs, _) =
        resolve_eval_connection(parsed.ollama_url, parsed.timeout_secs, None, false)
            .map_err(|msg| CliError::runtime(format!("[ERRO DE CONFIGURAÇÃO]\n{}", msg)))?;

    let client = build_eval_client(&ollama_url, timeout_secs)
        .map_err(|msg| CliError::runtime(format!("[ERRO DE CLIENTE OLLAMA]\n{}", msg)))?;

    let models = client
        .list_installed_model_details()
        .map_err(|msg| CliError::runtime(format!("[OLLAMA INDISPONÍVEL]\n{}", msg)))?;

    let _ = writeln!(out, "Ollama local: {}", client.base_url());
    if models.is_empty() {
        let _ = writeln!(
            out,
            "Nenhum modelo instalado localmente. O ServerMind não baixa modelos automaticamente; execute `ollama pull llama3.2` em outro terminal."
        );
        return Ok(());
    }

    let _ = writeln!(out, "Modelos instalados ({}):", models.len());
    for info in &models {
        let meta = format_model_info_line(info);
        if meta.is_empty() {
            let _ = writeln!(out, "  - {}", info.name);
        } else {
            let _ = writeln!(out, "  - {} ({})", info.name, meta);
        }
    }

    Ok(())
}

fn run_eval_run<W: Write>(
    args: &[String],
    install_ctrlc_handler: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
    out: &mut W,
) -> CliResult<()> {
    let parsed = parse_eval_run_args(args)
        .map_err(|msg| CliError::usage(format!("[ERRO DE ARGUMENTOS]\n{}", msg)))?;

    if parsed.help_requested {
        print_eval_help(out);
        return Ok(());
    }

    let backend_kind = parsed
        .backend
        .unwrap_or(crate::chat::backend::BackendKind::Ollama);

    let build_common_run_config = |model_name: String, timeout_secs: u64| -> EvalRunConfig {
        let mut run_config = EvalRunConfig::new(model_name);
        if let Some(ref ds_path) = parsed.dataset {
            run_config.dataset_path = ds_path.clone();
        }
        if let Some(ref out_dir) = parsed.out_dir {
            run_config.out_dir = out_dir.clone();
        }
        run_config.filter = DatasetFilter {
            categories: parsed.categories.clone(),
            case_ids: parsed.case_ids.clone(),
            limit: parsed.limit,
        };
        run_config.judge_model = parsed.judge_model.clone();
        run_config.timeout_secs = timeout_secs;
        run_config.max_duration = parsed.max_duration;
        run_config.repeat = parsed.repeat.unwrap_or(DEFAULT_EVAL_REPEAT);
        run_config.max_consecutive_global_failures = DEFAULT_CONSECUTIVE_GLOBAL_FAILURES_LIMIT;
        run_config.cancel_flag = cancel_flag.clone();
        run_config.install_ctrlc_handler = install_ctrlc_handler;
        run_config.use_kb = parsed.with_kb;
        run_config
    };

    let outcome = match backend_kind {
        crate::chat::backend::BackendKind::Ollama => {
            let (ollama_url, timeout_secs, model) =
                resolve_eval_connection(parsed.ollama_url, parsed.timeout_secs, parsed.model, true)
                    .map_err(|msg| CliError::runtime(format!("[ERRO DE CONFIGURAÇÃO]\n{}", msg)))?;
            let client = build_eval_client(&ollama_url, timeout_secs)
                .map_err(|msg| CliError::runtime(format!("[ERRO DE CLIENTE OLLAMA]\n{}", msg)))?;
            let run_config = build_common_run_config(model, timeout_secs);
            run_evaluation_with_progress(&client, &run_config, |event| {
                print_progress_event(out, event);
            })
            .map_err(|msg| CliError::runtime(format!("[ERRO NA AVALIAÇÃO]\n{}", msg)))?
        }
        crate::chat::backend::BackendKind::Neural => {
            let base_dir = AppConfig::resolve_active_config_dir();
            let ckpt_dir = parsed
                .model
                .as_deref()
                .map(|m| {
                    let p = std::path::Path::new(m);
                    if p.is_absolute() {
                        p.to_path_buf()
                    } else {
                        base_dir.join(p)
                    }
                })
                .unwrap_or_else(|| super::repl::resolve_default_neural_checkpoint_dir(&base_dir));
            let nb =
                crate::chat::backend::NeuralBackend::load_from_dir(&ckpt_dir).map_err(|msg| {
                    CliError::runtime(format!("[ERRO AO CARREGAR BACKEND NEURAL]\n{}", msg))
                })?;
            let timeout_secs = parsed.timeout_secs.unwrap_or(120);
            let run_config = build_common_run_config(nb.model_label(), timeout_secs);
            run_evaluation_with_progress(&nb, &run_config, |event| {
                print_progress_event(out, event);
            })
            .map_err(|msg| CliError::runtime(format!("[ERRO NA AVALIAÇÃO]\n{}", msg)))?
        }
        crate::chat::backend::BackendKind::Claude => {
            let timeout_secs = parsed
                .timeout_secs
                .unwrap_or(crate::chat::backend::DEFAULT_REMOTE_TIMEOUT_SECS);
            let limits = crate::chat::backend::RemoteLimits::from_env_with_timeout(timeout_secs)
                .map_err(|msg| {
                    CliError::runtime(format!("[ERRO DE CONFIGURAÇÃO REMOTA]\n{}", msg))
                })?;
            let cb = crate::chat::backend::ClaudeBackend::from_env(parsed.model.as_deref(), limits)
                .map_err(|msg| CliError::runtime(format!("[ERRO DE BACKEND CLAUDE]\n{}", msg)))?;
            let _ = writeln!(
                out,
                "{}",
                crate::chat::backend::format_remote_backend_banner(
                    "Anthropic Claude API",
                    crate::chat::backend::ModelBackend::endpoint_url(&cb),
                    cb.default_model(),
                    cb.limits(),
                )
            );
            let run_config = build_common_run_config(cb.default_model().to_string(), timeout_secs);
            run_evaluation_with_progress(&cb, &run_config, |event| {
                print_progress_event(out, event);
            })
            .map_err(|msg| CliError::runtime(format!("[ERRO NA AVALIAÇÃO]\n{}", msg)))?
        }
        remote_kind => {
            let (preset, preset_model) = remote_kind
                .resolve_openai_preset_and_default_model()
                .expect("OpenAI-compat preset kind");
            let timeout_secs = parsed
                .timeout_secs
                .unwrap_or(crate::chat::backend::DEFAULT_REMOTE_TIMEOUT_SECS);
            let limits = crate::chat::backend::RemoteLimits::from_env_with_timeout(timeout_secs)
                .map_err(|msg| {
                    CliError::runtime(format!("[ERRO DE CONFIGURAÇÃO REMOTA]\n{}", msg))
                })?;
            let effective_model = parsed.model.as_deref().or(preset_model);
            let ob = crate::chat::backend::OpenAiCompatBackend::from_env(
                preset,
                effective_model,
                limits,
            )
            .map_err(|msg| CliError::runtime(format!("[ERRO DE BACKEND REMOTO]\n{}", msg)))?;
            let _ = writeln!(
                out,
                "{}",
                crate::chat::backend::format_remote_backend_banner(
                    preset.display_name(),
                    crate::chat::backend::ModelBackend::endpoint_url(&ob),
                    ob.default_model(),
                    ob.limits(),
                )
            );
            let run_config = build_common_run_config(ob.default_model().to_string(), timeout_secs);
            run_evaluation_with_progress(&ob, &run_config, |event| {
                print_progress_event(out, event);
            })
            .map_err(|msg| CliError::runtime(format!("[ERRO NA AVALIAÇÃO]\n{}", msg)))?
        }
    };

    print_eval_run_summary(out, &outcome);

    let stats = &outcome.report.execution_stats;
    if stats.cancelled {
        return Err(CliError::cancelled_with_message(
            "[AVALIAÇÃO CANCELADA]\nExecução interrompida por solicitação do usuário (Ctrl+C). Relatório parcial consistente salvo.",
        ));
    }

    if stats.global_failure_detected {
        let reason = stats.early_abort_reason.as_deref().unwrap_or(
            "todas as execuções planejadas falharam, expiraram ou retornaram respostas vazias",
        );
        return Err(CliError::runtime(format!(
            "[FALHA GLOBAL DETECTADA]\nCausa: {}\nPróxima ação: verifique o serviço Ollama local, os logs do modelo e os relatórios parciais gerados acima.",
            reason
        )));
    }

    if stats.max_duration_reached {
        let _ = writeln!(
            out,
            "[AVISO DE DURAÇÃO MÁXIMA]\nLimite `--max-duration` atingido antes da conclusão completa ({} de {} execuções realizadas); relatório parcial consistente gravado.",
            stats.executed_count, stats.planned_executions
        );
    }

    Ok(())
}

struct ParsedListCasesArgs {
    dataset: Option<PathBuf>,
    categories: Vec<EvalCategory>,
    case_ids: Vec<String>,
    help_requested: bool,
}

struct ParsedListModelsArgs {
    ollama_url: Option<String>,
    timeout_secs: Option<u64>,
    help_requested: bool,
}

struct ParsedEvalRunArgs {
    backend: Option<crate::chat::backend::BackendKind>,
    model: Option<String>,
    dataset: Option<PathBuf>,
    categories: Vec<EvalCategory>,
    case_ids: Vec<String>,
    limit: Option<usize>,
    out_dir: Option<PathBuf>,
    timeout_secs: Option<u64>,
    max_duration: Option<Duration>,
    repeat: Option<usize>,
    judge_model: Option<String>,
    ollama_url: Option<String>,
    with_kb: bool,
    help_requested: bool,
}

fn parse_non_empty_trimmed(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("o valor informado não pode ser vazio.".to_string());
    }
    Ok(trimmed.to_string())
}

fn parse_non_empty_path(raw: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(parse_non_empty_trimmed(raw)?))
}

fn parse_local_ollama_url_clap(raw: &str) -> Result<String, String> {
    let val = parse_non_empty_trimmed(raw)?;
    AppConfig::validate_local_url(&val)?;
    Ok(val)
}

fn parse_backend_clap(raw: &str) -> Result<crate::chat::backend::BackendKind, String> {
    let val = parse_non_empty_trimmed(raw)?;
    crate::chat::backend::BackendKind::parse(&val)
}

fn parse_positive_usize_eval_clap(raw: &str) -> Result<usize, String> {
    let val = raw.trim().parse::<usize>().map_err(|_| {
        format!(
            "valor inválido '{}': informe um número inteiro positivo.",
            raw
        )
    })?;
    if val == 0 {
        return Err("o valor deve ser um número inteiro maior que zero.".to_string());
    }
    Ok(val)
}

fn parse_positive_u64_eval_clap(raw: &str) -> Result<u64, String> {
    let val = raw.trim().parse::<u64>().map_err(|_| {
        format!(
            "valor inválido '{}': informe um número inteiro positivo em segundos.",
            raw
        )
    })?;
    if val == 0 {
        return Err("o valor deve ser um número inteiro maior que zero.".to_string());
    }
    Ok(val)
}

fn parse_repeat_clap(raw: &str) -> Result<usize, String> {
    let n = parse_positive_usize_eval_clap(raw)?;
    validate_repeat(n)
}

fn parse_duration_clap(raw: &str) -> Result<Duration, String> {
    let val = parse_non_empty_trimmed(raw)?;
    parse_duration_str(&val)
}

fn expand_csv_items(raw_list: &[String], flag: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for raw in raw_list {
        for part in raw.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                return Err(format!(
                    "Lista inválida para '{}': há um item vazio em '{}'.",
                    flag, raw
                ));
            }
            out.push(trimmed.to_string());
        }
    }
    Ok(out)
}

fn expand_categories(raw_list: &[String]) -> Result<Vec<EvalCategory>, String> {
    let names = expand_csv_items(raw_list, "--category")?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        out.push(EvalCategory::parse(&name)?);
    }
    Ok(out)
}

fn format_eval_clap_error(err: &ClapError, subcommand: &str) -> String {
    let raw = err.to_string();
    match err.kind() {
        ClapErrorKind::ArgumentConflict => format!(
            "A opção foi informada mais de uma vez em `eval {}`.\nDetalhe: {}",
            subcommand,
            raw.trim()
        ),
        ClapErrorKind::UnknownArgument => {
            if raw.contains("unexpected argument") && !raw.contains("'--") && !raw.contains("'-") {
                format!(
                    "Argumento posicional inesperado em `eval {}`.\nPróxima ação: use as opções nomeadas conforme `servermind-ai eval --help`.\nDetalhe: {}",
                    subcommand,
                    raw.trim()
                )
            } else {
                format!(
                    "Opção desconhecida para `eval {}`.\nPróxima ação: execute `servermind-ai eval --help`.\nDetalhe: {}",
                    subcommand,
                    raw.trim()
                )
            }
        }
        ClapErrorKind::TooManyValues => format!(
            "A opção não aceita valor adicional ou argumento posicional inesperado em `eval {}`.\nDetalhe: {}",
            subcommand,
            raw.trim()
        ),
        _ => format!(
            "Argumentos inválidos em `eval {}`.\nDetalhe: {}",
            subcommand,
            raw.trim()
        ),
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "list-cases",
    disable_help_flag = true,
    disable_version_flag = true,
    allow_negative_numbers = true
)]
struct ListCasesClapArgs {
    #[arg(long = "help", short = 'h', action = ArgAction::SetTrue)]
    help: bool,

    #[arg(long = "dataset", value_parser = parse_non_empty_path)]
    dataset: Option<PathBuf>,

    #[arg(long = "category", action = ArgAction::Append)]
    categories: Vec<String>,

    #[arg(long = "case", action = ArgAction::Append)]
    cases: Vec<String>,
}

#[derive(Debug, Parser)]
#[command(
    name = "list-models",
    disable_help_flag = true,
    disable_version_flag = true,
    allow_negative_numbers = true
)]
struct ListModelsClapArgs {
    #[arg(long = "help", short = 'h', action = ArgAction::SetTrue)]
    help: bool,

    #[arg(long = "ollama-url", value_parser = parse_local_ollama_url_clap)]
    ollama_url: Option<String>,

    #[arg(long = "timeout-secs", value_parser = parse_positive_u64_eval_clap)]
    timeout_secs: Option<u64>,
}

#[derive(Debug, Parser)]
#[command(
    name = "run",
    disable_help_flag = true,
    disable_version_flag = true,
    allow_negative_numbers = true
)]
struct EvalRunClapArgs {
    #[arg(long = "help", short = 'h', action = ArgAction::SetTrue)]
    help: bool,

    #[arg(long = "backend", value_parser = parse_backend_clap)]
    backend: Option<crate::chat::backend::BackendKind>,

    #[arg(long = "model", value_parser = parse_non_empty_trimmed)]
    model: Option<String>,

    #[arg(long = "dataset", value_parser = parse_non_empty_path)]
    dataset: Option<PathBuf>,

    #[arg(long = "category", action = ArgAction::Append)]
    categories: Vec<String>,

    #[arg(long = "case", action = ArgAction::Append)]
    cases: Vec<String>,

    #[arg(long = "limit", value_parser = parse_positive_usize_eval_clap)]
    limit: Option<usize>,

    #[arg(long = "out-dir", value_parser = parse_non_empty_path)]
    out_dir: Option<PathBuf>,

    #[arg(long = "timeout-secs", value_parser = parse_positive_u64_eval_clap)]
    timeout_secs: Option<u64>,

    #[arg(long = "max-duration", value_parser = parse_duration_clap)]
    max_duration: Option<Duration>,

    #[arg(long = "repeat", value_parser = parse_repeat_clap)]
    repeat: Option<usize>,

    #[arg(long = "judge-model", value_parser = parse_non_empty_trimmed)]
    judge_model: Option<String>,

    #[arg(long = "ollama-url", value_parser = parse_local_ollama_url_clap)]
    ollama_url: Option<String>,

    #[arg(long = "with-kb", action = ArgAction::SetTrue)]
    with_kb: bool,
}

fn parse_list_cases_args(args: &[String]) -> Result<ParsedListCasesArgs, String> {
    let raw = ListCasesClapArgs::try_parse_from(
        std::iter::once("list-cases").chain(args.iter().map(|s| s.as_str())),
    )
    .map_err(|e| format_eval_clap_error(&e, "list-cases"))?;

    let categories = expand_categories(&raw.categories)?;
    let case_ids = expand_csv_items(&raw.cases, "--case")?;
    Ok(ParsedListCasesArgs {
        dataset: raw.dataset,
        categories,
        case_ids,
        help_requested: raw.help,
    })
}

fn parse_list_models_args(args: &[String]) -> Result<ParsedListModelsArgs, String> {
    let raw = ListModelsClapArgs::try_parse_from(
        std::iter::once("list-models").chain(args.iter().map(|s| s.as_str())),
    )
    .map_err(|e| format_eval_clap_error(&e, "list-models"))?;

    Ok(ParsedListModelsArgs {
        ollama_url: raw.ollama_url,
        timeout_secs: raw.timeout_secs,
        help_requested: raw.help,
    })
}

fn parse_eval_run_args(args: &[String]) -> Result<ParsedEvalRunArgs, String> {
    let raw = EvalRunClapArgs::try_parse_from(
        std::iter::once("run").chain(args.iter().map(|s| s.as_str())),
    )
    .map_err(|e| format_eval_clap_error(&e, "run"))?;

    let categories = expand_categories(&raw.categories)?;
    let case_ids = expand_csv_items(&raw.cases, "--case")?;

    Ok(ParsedEvalRunArgs {
        backend: raw.backend,
        model: raw.model,
        dataset: raw.dataset,
        categories,
        case_ids,
        limit: raw.limit,
        out_dir: raw.out_dir,
        timeout_secs: raw.timeout_secs,
        max_duration: raw.max_duration,
        repeat: raw.repeat,
        judge_model: raw.judge_model,
        ollama_url: raw.ollama_url,
        with_kb: raw.with_kb,
        help_requested: raw.help,
    })
}

fn resolve_eval_connection(
    ollama_url_opt: Option<String>,
    timeout_opt: Option<u64>,
    model_opt: Option<String>,
    need_model: bool,
) -> Result<(String, u64, String), String> {
    let need_config =
        ollama_url_opt.is_none() || timeout_opt.is_none() || (need_model && model_opt.is_none());

    let loaded_cfg = if need_config {
        Some(AppConfig::load()?)
    } else {
        None
    };

    let ollama_url = match ollama_url_opt {
        Some(url) => url,
        None => loaded_cfg
            .as_ref()
            .map(|c| c.ollama_url.clone())
            .unwrap_or_else(|| crate::chat::config::DEFAULT_OLLAMA_URL.to_string()),
    };
    AppConfig::validate_local_url(&ollama_url)?;

    let timeout_secs = match timeout_opt {
        Some(secs) => secs,
        None => loaded_cfg
            .as_ref()
            .map(|c| c.request_timeout_secs)
            .unwrap_or(crate::chat::config::DEFAULT_TIMEOUT_SECS),
    };
    if timeout_secs == 0 {
        return Err("O timeout de requisição do Ollama deve ser maior que zero.".to_string());
    }

    let model = if need_model {
        match model_opt {
            Some(m) => m,
            None => loaded_cfg
                .as_ref()
                .map(|c| c.model.clone())
                .unwrap_or_else(|| crate::chat::config::DEFAULT_MODEL.to_string()),
        }
    } else {
        String::new()
    };

    Ok((ollama_url, timeout_secs, model))
}

fn build_eval_client(ollama_url: &str, timeout_secs: u64) -> Result<OllamaClient, String> {
    AppConfig::validate_local_url(ollama_url)?;
    OllamaClient::new(ollama_url.to_string(), timeout_secs)
}

fn format_model_info_line(info: &InstalledModelInfo) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(details) = &info.details {
        if let Some(family) = details.family.as_deref().filter(|s| !s.trim().is_empty()) {
            parts.push(format!("família={}", family));
        }
        if let Some(param) = details
            .parameter_size
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            parts.push(param.to_string());
        }
        if let Some(quant) = details
            .quantization_level
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            parts.push(quant.to_string());
        }
    }
    if let Some(bytes) = info.size {
        parts.push(format_bytes_human(bytes));
    }
    parts.join(", ")
}

fn format_bytes_human(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else {
        format!("{} B", bytes)
    }
}

fn print_progress_event<W: Write>(out: &mut W, event: &EvalProgressEvent) {
    match event {
        EvalProgressEvent::Started {
            model_resolved,
            judge_resolved,
            selected_cases,
            repeat,
            planned_executions,
        } => {
            let judge_info = match judge_resolved {
                Some(j) if j.eq_ignore_ascii_case(model_resolved) => {
                    format!(" | juiz='{}' (estimativa/autoavaliação)", j)
                }
                Some(j) => format!(" | juiz='{}' (estimativa)", j),
                None => String::new(),
            };
            let _ = writeln!(
                out,
                "[eval] Iniciando avaliação: modelo='{}' | casos={} | repeat={} | execuções={}{} | concorrência=1",
                model_resolved, selected_cases, repeat, planned_executions, judge_info
            );
            let _ = out.flush();
        }
        EvalProgressEvent::CaseStarted {
            execution_index,
            planned_executions,
            case_id,
            repeat_index,
            category,
            title,
        } => {
            let _ = writeln!(
                out,
                "[eval] [{}/{}] Executando {} (rep {}, {}) — {}",
                execution_index,
                planned_executions,
                case_id,
                repeat_index,
                category.as_str(),
                title
            );
            let _ = out.flush();
        }
        EvalProgressEvent::CaseFinished {
            execution_index,
            planned_executions,
            case_id,
            repeat_index,
            status,
            duration_ms,
            deterministic_total,
            deterministic_max,
            needs_human_review,
        } => {
            let score_part = match deterministic_total {
                Some(s) => format!(" | det={}/{}", s, deterministic_max),
                None => String::new(),
            };
            let review_part = if *needs_human_review {
                " | revisão_humana=sim"
            } else {
                ""
            };
            let status_tag = match status {
                CaseExecutionStatus::Succeeded => "OK",
                CaseExecutionStatus::EmptyResponse => "VAZIA",
                CaseExecutionStatus::Timeout => "TIMEOUT",
                CaseExecutionStatus::Error => "ERRO",
            };
            let _ = writeln!(
                out,
                "[eval] [{}/{}] {} (rep {}): {} | {}ms{}{}",
                execution_index,
                planned_executions,
                case_id,
                repeat_index,
                status_tag,
                duration_ms,
                score_part,
                review_part
            );
            let _ = out.flush();
        }
        EvalProgressEvent::Interrupted {
            cancelled,
            max_duration_reached,
            early_abort_reason,
        } => {
            let cause = if *cancelled {
                "Ctrl+C / cancelamento"
            } else if *max_duration_reached {
                "duração máxima atingida"
            } else if let Some(reason) = early_abort_reason {
                reason.as_str()
            } else {
                "interrupção antecipada"
            };
            let _ = writeln!(out, "[eval] Execução interrompida: {}", cause);
            let _ = out.flush();
        }
        EvalProgressEvent::ReportWritten { paths } => {
            let _ = writeln!(
                out,
                "[eval] Relatórios gravados em '{}'",
                paths.directory.display()
            );
            let _ = out.flush();
        }
    }
}

fn print_eval_run_summary<W: Write>(out: &mut W, outcome: &EvalRunOutcome) {
    let report = &outcome.report;
    let stats = &report.execution_stats;
    let scores = &report.overall_scores;

    let _ = writeln!(out, "\n=== Resumo da Avaliação ===");
    let _ = writeln!(out, "Modelo avaliado   : {}", report.model.resolved_name);
    let _ = writeln!(
        out,
        "Dataset           : v{} (SHA-256: {})",
        report.dataset.dataset_version, report.dataset.sha256_hex
    );
    let _ = writeln!(
        out,
        "Execuções         : {}/{} (OK: {}, vazias: {}, timeout: {}, erro: {}, não executadas: {})",
        stats.executed_count,
        stats.planned_executions,
        stats.succeeded_count,
        stats.empty_response_count,
        stats.timeout_count,
        stats.error_count,
        stats.skipped_count
    );
    if let Some(mean) = scores.deterministic_mean_total {
        let _ = writeln!(
            out,
            "Média determiníst.: {:.2}/{} ({} casos avaliados)",
            mean, scores.max_total_score, scores.deterministic_evaluated_count
        );
    }
    if scores.judge_enabled {
        let self_tag = if scores.judge_self_judging {
            " | AVISO: autoavaliação"
        } else {
            ""
        };
        let judge_name = report
            .judge_model
            .as_ref()
            .map(|j| j.resolved_name.as_str())
            .unwrap_or("-");
        match scores.judge_estimate_mean_total {
            Some(mean) => {
                let _ = writeln!(
                    out,
                    "Média juiz (estim.): {:.2}/{} (modelo: {}, válidos: {}{})",
                    mean, scores.max_total_score, judge_name, scores.judge_estimate_count, self_tag
                );
            }
            None => {
                let _ = writeln!(
                    out,
                    "Juiz local (estim.): sem pontuações válidas (modelo: {}{})",
                    judge_name, self_tag
                );
            }
        }
    }
    let _ = writeln!(
        out,
        "Revisão humana    : {} caso(s) sinalizado(s)",
        stats.human_review_count
    );
    let _ = writeln!(
        out,
        "Relatório JSON    : {}",
        outcome.paths.json_path.display()
    );
    let _ = writeln!(
        out,
        "Relatório MD      : {}",
        outcome.paths.markdown_path.display()
    );
    let _ = writeln!(
        out,
        "Aviso             : Avaliação estritamente consultiva; notas não garantem correção nem alteram modelos/pesos."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_vec(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn test_phase2_t4_eval_list_cases_parser_table() {
        // Casos válidos (espaço, inline `=`, CSV e repetição)
        let p_empty = parse_list_cases_args(&to_vec(&[])).expect("sem args é válido");
        assert!(p_empty.dataset.is_none());
        assert!(p_empty.categories.is_empty());
        assert!(p_empty.case_ids.is_empty());
        assert!(!p_empty.help_requested);

        let p_full = parse_list_cases_args(&to_vec(&[
            "--dataset=eval/cases_v1.json",
            "--category",
            "explicacao_conceitos,fora_de_escopo",
            "--case",
            "exp-01,inc-01",
            "--case=dst-01",
        ]))
        .expect("combinação válida em list-cases");
        assert_eq!(p_full.dataset, Some(PathBuf::from("eval/cases_v1.json")));
        assert_eq!(p_full.categories.len(), 2);
        assert_eq!(p_full.case_ids, vec!["exp-01", "inc-01", "dst-01"]);

        let p_help = parse_list_cases_args(&to_vec(&["-h"])).expect("-h válido");
        assert!(p_help.help_requested);

        // Tabela de entradas inválidas
        let invalid_table: &[(&[&str], &str)] = &[
            (&["--help=1"], "--help"),
            (&["--dataset"], "--dataset"),
            (
                &["--dataset", "a.json", "--dataset", "b.json"],
                "mais de uma vez",
            ),
            (&["--category", ""], "--category"),
            (&["--category", "explicacao_conceitos,"], "--category"),
            (&["--category", "invalida"], "Categoria"),
            (&["--case", ""], "--case"),
            (&["--case", "exp-01, ,inc-01"], "--case"),
            (&["posicional_solto"], "posicional inesperado"),
            (&["--flag-desconhecida"], "desconhecida"),
        ];
        for (raw, expected_substr) in invalid_table {
            let err = parse_list_cases_args(&to_vec(raw))
                .err()
                .unwrap_or_else(|| panic!("deveria rejeitar {:?}", raw));
            assert!(
                err.contains(expected_substr),
                "para {:?} esperava '{}' em '{}'",
                raw,
                expected_substr,
                err
            );
        }
    }

    #[test]
    fn test_phase2_t4_eval_list_models_parser_table() {
        let p_ok = parse_list_models_args(&to_vec(&[
            "--ollama-url=http://localhost:11434",
            "--timeout-secs",
            "15",
        ]))
        .expect("list-models válido");
        assert_eq!(p_ok.ollama_url.as_deref(), Some("http://localhost:11434"));
        assert_eq!(p_ok.timeout_secs, Some(15));

        let invalid_table: &[(&[&str], &str)] = &[
            (&["--help=sim"], "--help"),
            (&["--ollama-url"], "--ollama-url"),
            (&["--ollama-url", "http://10.0.0.8:11434"], "loopback"),
            (
                &[
                    "--ollama-url",
                    "http://127.0.0.1:11434",
                    "--ollama-url",
                    "http://localhost:11434",
                ],
                "mais de uma vez",
            ),
            (&["--timeout-secs", "0"], "--timeout-secs"),
            (&["--timeout-secs", "abc"], "--timeout-secs"),
            (
                &["--timeout-secs", "5", "--timeout-secs", "10"],
                "mais de uma vez",
            ),
            (&["extra"], "posicional inesperado"),
        ];
        for (raw, expected_substr) in invalid_table {
            let err = parse_list_models_args(&to_vec(raw))
                .err()
                .unwrap_or_else(|| panic!("deveria rejeitar {:?}", raw));
            assert!(
                err.contains(expected_substr),
                "para {:?} esperava '{}' em '{}'",
                raw,
                expected_substr,
                err
            );
        }
    }

    #[test]
    fn test_phase2_t4_eval_run_parser_table() {
        let p = parse_eval_run_args(&to_vec(&[
            "--backend=claude",
            "--model",
            "qwen2.5:7b",
            "--dataset=eval/cases_v1.json",
            "--category=explicacao_conceitos",
            "--case=exp-01",
            "--limit=5",
            "--out-dir",
            "out_custom",
            "--timeout-secs=45",
            "--max-duration=1h30m",
            "--repeat=3",
            "--judge-model=llama3.2",
            "--ollama-url=http://[::1]:11434",
            "--with-kb",
        ]))
        .expect("eval run completo válido");
        assert_eq!(p.backend, Some(crate::chat::backend::BackendKind::Claude));
        assert_eq!(p.model.as_deref(), Some("qwen2.5:7b"));
        assert_eq!(p.limit, Some(5));
        assert_eq!(p.timeout_secs, Some(45));
        assert_eq!(p.max_duration, Some(Duration::from_secs(5400)));
        assert_eq!(p.repeat, Some(3));
        assert_eq!(p.judge_model.as_deref(), Some("llama3.2"));
        assert!(p.with_kb);

        let invalid_table: &[(&[&str], &str)] = &[
            (&["--backend", "invalido"], "Backend desconhecido"),
            (
                &["--backend", "ollama", "--backend", "claude"],
                "mais de uma vez",
            ),
            (&["--with-kb=true"], "--with-kb"),
            (&["--help=true"], "--help"),
            (&["--limit", "0"], "--limit"),
            (&["--limit", "-2"], "--limit"),
            (&["--limit", "2", "--limit", "3"], "mais de uma vez"),
            (&["--repeat", "0"], "--repeat"),
            (&["--repeat", "21"], "20"),
            (&["--max-duration", "0s"], "maior que zero"),
            (&["--max-duration", "999h"], "limite"),
            (&["--ollama-url", "https://api.openai.com"], "loopback"),
            (&["--model"], "--model"),
            (&["--judge-model", "   "], "vazio"),
            (&["arg_solto"], "posicional inesperado"),
        ];
        for (raw, expected_substr) in invalid_table {
            let err = parse_eval_run_args(&to_vec(raw))
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
    fn test_phase3_c5_eval_run_supports_remote_claude_backend() {
        use crate::test_support::{lock_env_cwd, EnvVarGuard, TempDirGuard};
        use std::io::{Read, Write as IoWrite};
        use std::net::{Shutdown, TcpListener};

        let _lock = lock_env_cwd();
        let temp = TempDirGuard::new("c5_eval_claude");
        let _env = EnvVarGuard::capture(&["ANTHROPIC_API_KEY", "SERVERMIND_ANTHROPIC_BASE_URL"]);

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mock_url = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let body = serde_json::json!({
                "id": "msg_eval_1",
                "model": "claude-sonnet-4-0",
                "content": [{
                    "type": "text",
                    "text": "No systemd, .service define o processo gerenciado e .timer agenda execuções periódicas via OnCalendar com logs no journalctl."
                }]
            })
            .to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
            let _ = stream.shutdown(Shutdown::Write);
        });

        std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-eval-test-key");
        std::env::set_var("SERVERMIND_ANTHROPIC_BASE_URL", &mock_url);

        let ds_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("eval/cases_v1.json")
            .to_string_lossy()
            .into_owned();
        let out_dir = temp.path.join("eval_out");
        let out_str = out_dir.to_string_lossy().into_owned();

        let mut out = Vec::new();
        let mut err = Vec::new();
        let code = run_eval_cli_with_options(
            &to_vec(&[
                "run",
                "--backend",
                "claude",
                "--dataset",
                &ds_path,
                "--case",
                "exp-01",
                "--out-dir",
                &out_str,
            ]),
            false,
            None,
            &mut out,
            &mut err,
        );
        handle.join().unwrap();

        let stdout_str = String::from_utf8_lossy(&out);
        let stderr_str = String::from_utf8_lossy(&err);
        assert_eq!(code, 0, "stderr: {}", stderr_str);
        assert!(stdout_str.contains("[AVISO DE BACKEND REMOTO ATIVO"));
        assert!(stdout_str.contains("claude-sonnet-4-0"));
        assert!(stdout_str.contains("=== Resumo da Avaliação ==="));
    }
}
