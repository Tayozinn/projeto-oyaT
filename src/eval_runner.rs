//! Executor controlado da avaliação automatizada do ServerMind AI (Etapa 3).
//!
//! Este módulo coordena a execução sequencial (`concurrency = 1`) dos casos do dataset
//! contra um modelo local via abstração [`EvalModelClient`], aplica limites previsíveis de
//! tempo por requisição (`--timeout-secs`), duração total máxima (`--max-duration`, ex.: `7h`)
//! e repetições (`--repeat`, padrão `1`, limitado a [`MAX_EVAL_REPEAT`]), trata cancelamento
//! limpo por `Ctrl+C` (gerando relatório parcial consistente), detecta falhas globais
//! consecutivas sem mascarar corrupção de infraestrutura e grava relatórios JSON + Markdown
//! sem sobrescrever execuções anteriores.
//!
//! Garantias estritas de privacidade e escopo:
//! - Nunca lê, abre, consulta ou modifica `servermind.db` (zero acesso a `crate::storage`).
//! - Nunca executa comandos no sistema operacional nem altera pesos/Modelfiles/prompts salvos.
//! - Nunca baixa modelos automaticamente nem inicia/para o serviço Ollama.
//! - Usa sempre o prompt de sistema reprodutível padrão ([`build_reproducible_system_prompt`]),
//!   sem expor `system_prompt` privado de `config.toml` ou variáveis de ambiente nos relatórios.

use crate::config::{AppConfig, DEFAULT_CUSTOM_INSTRUCTIONS, DEFAULT_TIMEOUT_SECS};
use crate::eval_dataset::{
    filter_cases, load_and_validate_dataset, DatasetFilter, EvalCase, EvalCategory,
};
use crate::eval_metrics::{
    build_judge_prompt, evaluate_deterministic, judge_failure, parse_judge_response,
    JudgeEvaluation, MetricDimension, SCORE_MAX,
};
use crate::eval_report::{
    build_eval_report, write_report_bundle, CaseExecutionRecord, CaseExecutionStatus, EvalReport,
    EvalReportInput, EvalRunParams, ModelIdentification, ReportPaths,
};
use crate::models::{Message, Role};
use crate::ollama::{InstalledModelInfo, OllamaChatOutcome, OllamaClient};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// Concorrência fixa da avaliação para reprodutibilidade e controle de recursos locais.
pub const FIXED_EVAL_CONCURRENCY: usize = 1;
/// Número padrão de execuções por caso selecionado.
pub const DEFAULT_EVAL_REPEAT: usize = 1;
/// Teto rígido de repetições por caso para impedir loops excessivos ou infinitos.
pub const MAX_EVAL_REPEAT: usize = 20;
/// Teto máximo aceito para `--max-duration` (7 dias = 604.800 segundos).
pub const MAX_ALLOWED_DURATION_SECS: u64 = 7 * 24 * 3_600;
/// Limite padrão de falhas técnicas globais/conectividade consecutivas antes de abortar cedo.
pub const DEFAULT_CONSECUTIVE_GLOBAL_FAILURES_LIMIT: usize = 3;
/// Identificador do modo de prompt de sistema reprodutível registrado nos relatórios.
pub const EVAL_SYSTEM_PROMPT_MODE: &str = "default_reproducible_advisory_v11";
/// Caminho relativo padrão do dataset versionado.
pub const DEFAULT_DATASET_REL_PATH: &str = "eval/cases_v1.json";
/// Diretório relativo padrão para gravação dos relatórios.
pub const DEFAULT_REPORTS_REL_DIR: &str = "eval_reports";

// ---------------------------------------------------------------------------
// Caminhos padrão e construção reprodutível de mensagens
// ---------------------------------------------------------------------------

/// Retorna o caminho padrão do dataset (`<base_dir>/eval/cases_v1.json`) ancorado de forma determinística.
pub fn default_dataset_path() -> PathBuf {
    let mut path = AppConfig::resolve_default_base_dir();
    for segment in DEFAULT_DATASET_REL_PATH.split('/') {
        path.push(segment);
    }
    path
}

/// Retorna o diretório padrão de saída dos relatórios (`<base_dir>/eval_reports`).
pub fn default_reports_dir() -> PathBuf {
    AppConfig::resolve_default_base_dir().join(DEFAULT_REPORTS_REL_DIR)
}

/// Normaliza e relativiza um caminho para gravação segura nos metadados de relatório (JSON/Markdown),
/// preservando caminhos relativos ao diretório base do projeto (`eval/cases_v1.json`, `eval_reports`)
/// ou ao diretório atual, e sanitizando diretórios de perfil de usuário local.
pub fn format_path_for_report(path: &Path) -> String {
    let base_dir = AppConfig::resolve_default_base_dir();
    if let Ok(rel) = path.strip_prefix(&base_dir) {
        let normalized = normalize_report_slashes(&rel.to_string_lossy());
        return if normalized.is_empty() {
            ".".to_string()
        } else {
            redact_local_user_segments(&normalized)
        };
    }

    if let Ok(cwd) = std::env::current_dir() {
        if let Ok(rel) = path.strip_prefix(&cwd) {
            let normalized = normalize_report_slashes(&rel.to_string_lossy());
            return if normalized.is_empty() {
                ".".to_string()
            } else {
                redact_local_user_segments(&normalized)
            };
        }
    }

    let normalized = normalize_report_slashes(&path.to_string_lossy());
    redact_local_user_segments(&normalized)
}

fn normalize_report_slashes(raw: &str) -> String {
    let replaced = raw.trim().replace('\\', "/");
    let without_dot = replaced.strip_prefix("./").unwrap_or(&replaced);
    if without_dot.len() > 1 {
        without_dot.trim_end_matches('/').to_string()
    } else {
        without_dot.to_string()
    }
}

fn redact_local_user_segments(normalized: &str) -> String {
    for env_key in ["USERPROFILE", "HOME"] {
        if let Ok(home_val) = std::env::var(env_key) {
            let home_norm = normalize_report_slashes(&home_val);
            if !home_norm.is_empty() && home_norm != "/" {
                let norm_lower = normalized.to_ascii_lowercase();
                let home_lower = home_norm.to_ascii_lowercase();
                if norm_lower == home_lower {
                    return "~".to_string();
                }
                let prefix = format!("{}/", home_lower);
                if norm_lower.starts_with(&prefix) {
                    let suffix = &normalized[prefix.len()..];
                    return format!("~/{}", suffix);
                }
            }
        }
    }

    // Sanitiza padrões clássicos de diretórios de usuário: `C:/Users/<user>/...`, `/Users/<user>/...`, `/home/<user>/...`
    let bytes = normalized.as_bytes();
    let after_drive = if bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && bytes[2] == b'/'
    {
        Some(&normalized[3..])
    } else if normalized.starts_with('/') {
        Some(&normalized[1..])
    } else {
        None
    };

    if let Some(rest) = after_drive {
        let mut parts = rest.splitn(3, '/');
        if let (Some(root_dir), Some(user_seg)) = (parts.next(), parts.next()) {
            let is_user_root =
                root_dir.eq_ignore_ascii_case("users") || root_dir.eq_ignore_ascii_case("home");
            if is_user_root && !user_seg.is_empty() {
                return match parts.next() {
                    Some(tail) if !tail.is_empty() => format!("~/{}", tail),
                    _ => "~".to_string(),
                };
            }
        }
    }

    normalized.to_string()
}

/// Constrói o prompt de sistema reprodutível usado na avaliação, combinando apenas as
/// instruções padrão públicas, a identidade fixa de criação e as diretrizes imutáveis de
/// segurança (sem ler `config.toml` privado nem memórias do banco SQLite).
pub fn build_reproducible_system_prompt() -> String {
    AppConfig::compose_system_prompt(DEFAULT_CUSTOM_INSTRUCTIONS, &[])
}

/// Constrói o par de mensagens (`System` + `User`) para um caso de avaliação individual.
pub fn build_case_messages(case: &EvalCase, system_prompt: &str) -> Vec<Message> {
    let user_content = match &case.context {
        Some(ctx) if !ctx.trim().is_empty() => format!(
            "{}\n\nContexto / dados fornecidos:\n{}",
            case.question.trim(),
            ctx.trim()
        ),
        _ => case.question.trim().to_string(),
    };

    vec![
        Message::new(Role::System, system_prompt.to_string()),
        Message::new(Role::User, user_content),
    ]
}

// ---------------------------------------------------------------------------
// Parsing e validação de duração e repetições
// ---------------------------------------------------------------------------

/// Converte uma especificação textual de duração (ex.: `"30s"`, `"15m"`, `"7h"`, `"1h30m"`, `"120"`)
/// em [`Duration`], garantindo valor positivo e limitado a [`MAX_ALLOWED_DURATION_SECS`].
pub fn parse_duration_str(raw: &str) -> Result<Duration, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(
            "A duração máxima ('--max-duration') não pode ser vazia. Exemplos válidos: '30s', '15m', '7h', '1h30m'."
                .to_string(),
        );
    }

    if trimmed.starts_with('-') || trimmed.starts_with('+') {
        return Err(format!(
            "Duração inválida ('{}'): informe um valor positivo (ex.: '30s', '15m', '7h').",
            trimmed
        ));
    }

    // Se contiver apenas dígitos, interpreta como segundos.
    if trimmed.chars().all(|c| c.is_ascii_digit()) {
        let Ok(secs) = trimmed.parse::<u64>() else {
            return Err(format!(
                "Duração inválida ('{}'): número de segundos fora do intervalo suportado.",
                trimmed
            ));
        };
        return validate_duration_secs(secs, trimmed);
    }

    let lower = trimmed.to_ascii_lowercase();
    let mut total_secs: u64 = 0;
    let mut num_buf = String::new();
    let mut unit_buf = String::new();
    let mut saw_any_segment = false;

    for ch in lower.chars() {
        if ch.is_ascii_whitespace() {
            continue;
        }
        if ch.is_ascii_digit() {
            if !unit_buf.is_empty() {
                let seg_secs = parse_duration_segment(&num_buf, &unit_buf, trimmed)?;
                let Some(sum) = total_secs.checked_add(seg_secs) else {
                    return Err(format!(
                        "Duração inválida ('{}'): valor excede o limite suportado.",
                        trimmed
                    ));
                };
                total_secs = sum;
                num_buf.clear();
                unit_buf.clear();
                saw_any_segment = true;
            }
            num_buf.push(ch);
        } else if ch.is_ascii_alphabetic() {
            if num_buf.is_empty() {
                return Err(format!(
                    "Duração inválida ('{}'): unidade '{}' sem número correspondente. Exemplos válidos: '30s', '15m', '7h'.",
                    trimmed, ch
                ));
            }
            unit_buf.push(ch);
        } else {
            return Err(format!(
                "Duração inválida ('{}'): caractere '{}' não reconhecido. Use inteiros com sufixo 's', 'm' ou 'h' (ex.: '30s', '15m', '7h').",
                trimmed, ch
            ));
        }
    }

    if !num_buf.is_empty() {
        if unit_buf.is_empty() {
            return Err(format!(
                "Duração inválida ('{}'): faltou a unidade ('s', 'm' ou 'h') no segmento final.",
                trimmed
            ));
        }
        let seg_secs = parse_duration_segment(&num_buf, &unit_buf, trimmed)?;
        let Some(sum) = total_secs.checked_add(seg_secs) else {
            return Err(format!(
                "Duração inválida ('{}'): valor excede o limite suportado.",
                trimmed
            ));
        };
        total_secs = sum;
        saw_any_segment = true;
    }

    if !saw_any_segment {
        return Err(format!(
            "Duração inválida ('{}'). Exemplos válidos: '30s', '15m', '7h'.",
            trimmed
        ));
    }

    validate_duration_secs(total_secs, trimmed)
}

fn parse_duration_segment(num_str: &str, unit_str: &str, full_raw: &str) -> Result<u64, String> {
    let Ok(val) = num_str.parse::<u64>() else {
        return Err(format!(
            "Duração inválida ('{}'): valor numérico '{}' inválido.",
            full_raw, num_str
        ));
    };

    let multiplier: u64 = match unit_str {
        "s" | "sec" | "secs" | "seg" | "segs" => 1,
        "m" | "min" | "mins" => 60,
        "h" | "hr" | "hrs" | "hora" | "horas" => 3_600,
        _ => {
            return Err(format!(
                "Unidade de tempo desconhecida '{}' em '{}'. Use 's' (segundos), 'm' (minutos) ou 'h' (horas), ex.: '7h'.",
                unit_str, full_raw
            ))
        }
    };

    match val.checked_mul(multiplier) {
        Some(total) => Ok(total),
        None => Err(format!(
            "Duração inválida ('{}'): valor excede o limite suportado.",
            full_raw
        )),
    }
}

fn validate_duration_secs(secs: u64, raw: &str) -> Result<Duration, String> {
    if secs == 0 {
        return Err(format!(
            "A duração máxima ('{}') deve ser maior que zero (ex.: '30s', '15m', '7h').",
            raw
        ));
    }
    if secs > MAX_ALLOWED_DURATION_SECS {
        return Err(format!(
            "A duração máxima ('{}' = {}s) excede o limite permitido de {} segundos (7 dias).",
            raw, secs, MAX_ALLOWED_DURATION_SECS
        ));
    }
    Ok(Duration::from_secs(secs))
}

/// Valida o número de repetições por caso (`1..=MAX_EVAL_REPEAT`).
pub fn validate_repeat(repeat: usize) -> Result<usize, String> {
    if repeat == 0 {
        return Err(
            "O número de repetições ('--repeat') deve ser pelo menos 1 (padrão = 1).".to_string(),
        );
    }
    if repeat > MAX_EVAL_REPEAT {
        return Err(format!(
            "O número de repetições ('--repeat {}') excede o limite máximo permitido de {} execuções por caso.",
            repeat, MAX_EVAL_REPEAT
        ));
    }
    Ok(repeat)
}

// ---------------------------------------------------------------------------
// Guarda RAII para cancelamento limpo com Ctrl+C (zero crates externas)
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod ctrlc_imp {
    use std::sync::atomic::{AtomicBool, Ordering};

    type Bool = i32;
    type Dword = u32;
    type HandlerRoutine = unsafe extern "system" fn(Dword) -> Bool;

    const CTRL_C_EVENT: Dword = 0;
    const CTRL_BREAK_EVENT: Dword = 1;
    const TRUE: Bool = 1;
    const FALSE: Bool = 0;

    #[link(name = "kernel32")]
    extern "system" {
        fn SetConsoleCtrlHandler(handler: Option<HandlerRoutine>, add: Bool) -> Bool;
    }

    static GLOBAL_CTRL_C_TRIGGERED: AtomicBool = AtomicBool::new(false);

    unsafe extern "system" fn console_ctrl_handler(ctrl_type: Dword) -> Bool {
        if ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT {
            GLOBAL_CTRL_C_TRIGGERED.store(true, Ordering::SeqCst);
            TRUE
        } else {
            FALSE
        }
    }

    /// Guarda RAII que registra o handler de `Ctrl+C` no console do Windows e o remove no `Drop`.
    pub struct CtrlCGuard {
        registered: bool,
    }

    impl CtrlCGuard {
        pub fn install() -> Self {
            GLOBAL_CTRL_C_TRIGGERED.store(false, Ordering::SeqCst);
            // SAFETY: `console_ctrl_handler` respeita a assinatura `extern "system" fn(u32) -> i32`
            // da API Win32 e realiza apenas escrita atômica em `GLOBAL_CTRL_C_TRIGGERED`.
            let ok = unsafe { SetConsoleCtrlHandler(Some(console_ctrl_handler), TRUE) != 0 };
            Self { registered: ok }
        }

        pub fn is_triggered(&self) -> bool {
            GLOBAL_CTRL_C_TRIGGERED.load(Ordering::SeqCst)
        }
    }

    impl Drop for CtrlCGuard {
        fn drop(&mut self) {
            if self.registered {
                // SAFETY: desregistra o mesmo handler instalado em `install`.
                unsafe {
                    let _ = SetConsoleCtrlHandler(Some(console_ctrl_handler), FALSE);
                }
            }
        }
    }
}

#[cfg(not(windows))]
mod ctrlc_imp {
    /// Guarda RAII de compatibilidade em plataformas não-Windows (sem dependências externas).
    pub struct CtrlCGuard;

    impl CtrlCGuard {
        pub fn install() -> Self {
            Self
        }

        pub fn is_triggered(&self) -> bool {
            false
        }
    }
}

pub use ctrlc_imp::CtrlCGuard;

// ---------------------------------------------------------------------------
// Abstração do cliente de modelos locais (real + mockável em testes)
// ---------------------------------------------------------------------------

/// Resultado de uma requisição de chat feita pelo executor de avaliação.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelChatResult {
    /// Resposta não vazia recebida com sucesso.
    Succeeded {
        content: String,
        reported_model: Option<String>,
    },
    /// O serviço respondeu sem erro técnico, mas com conteúdo vazio/em branco.
    EmptyResponse { reported_model: Option<String> },
    /// Tempo limite esgotado na requisição.
    Timeout { message: String },
    /// Erro técnico na requisição.
    Error {
        message: String,
        /// `true` quando representa falha global/conectividade/loopback (ex.: serviço offline,
        /// conexão recusada, resposta HTTP 5xx ou JSON corrompido).
        is_global_or_connectivity: bool,
    },
}

/// Abstração para consulta de modelos instalados e envio de mensagens ao serviço local,
/// permitindo testes 100% determinísticos sem Ollama em execução.
pub trait EvalModelClient {
    /// Retorna o endereço base configurado (ex.: `http://127.0.0.1:11434`).
    fn endpoint_url(&self) -> &str;

    /// Lista os modelos instalados localmente com seus metadados opcionais (sem baixar nada).
    fn list_models(&self) -> Result<Vec<InstalledModelInfo>, String>;

    /// Valida que o modelo solicitado está instalado localmente e retorna sua [`ModelIdentification`].
    fn get_model_identification(
        &self,
        requested_model: &str,
    ) -> Result<ModelIdentification, String> {
        let installed = self.list_models()?;
        resolve_model_identification_from_list(requested_model, &installed, self.endpoint_url())
    }

    /// Executa uma chamada `/api/chat` com timeout específico para a requisição.
    fn chat_with_timeout(
        &self,
        model: &str,
        messages: &[Message],
        timeout: Duration,
    ) -> ModelChatResult;
}

/// Resolve um nome/tag de modelo contra a lista retornada por `/api/tags` e constrói [`ModelIdentification`].
pub fn resolve_model_identification_from_list(
    requested_model: &str,
    installed: &[InstalledModelInfo],
    endpoint_url: &str,
) -> Result<ModelIdentification, String> {
    let trimmed = requested_model.trim();
    if trimmed.is_empty() {
        return Err(
            "O nome do modelo não pode ser vazio. Informe um modelo instalado (ex.: '--model llama3.2')."
                .to_string(),
        );
    }

    if installed.is_empty() {
        return Err(format!(
            "O Ollama está acessível em '{}', mas NENHUM modelo está instalado localmente.\n\
             O SERVERMIND AI não baixa modelos automaticamente.\n\
             Como resolver: instale um modelo manualmente em outro terminal com `ollama pull {}`.",
            endpoint_url, trimmed
        ));
    }

    let Some(info) = OllamaClient::resolve_installed_model_detail(trimmed, installed) else {
        let names: Vec<&str> = installed.iter().map(|m| m.name.as_str()).collect();
        return Err(format!(
            "O modelo '{}' não está instalado no Ollama local ({}).\n\
             Modelos instalados detectados: {}.\n\
             O SERVERMIND AI não baixa modelos automaticamente.\n\
             Como resolver: escolha um dos modelos instalados ou baixe manualmente em outro terminal com `ollama pull {}`.",
            trimmed,
            endpoint_url,
            names.join(", "),
            trimmed
        ));
    };

    Ok(ModelIdentification {
        requested_name: trimmed.to_string(),
        resolved_name: info.name.clone(),
        ollama_reported_model: info.model.clone().or_else(|| Some(info.name.clone())),
        digest: info.digest.clone(),
        size_bytes: info.size,
        modified_at: info.modified_at.clone(),
        format: info.details.as_ref().and_then(|d| d.format.clone()),
        family: info.details.as_ref().and_then(|d| d.family.clone()),
        parameter_size: info.details.as_ref().and_then(|d| d.parameter_size.clone()),
        quantization_level: info
            .details
            .as_ref()
            .and_then(|d| d.quantization_level.clone()),
    })
}

impl EvalModelClient for OllamaClient {
    fn endpoint_url(&self) -> &str {
        self.base_url()
    }

    fn list_models(&self) -> Result<Vec<InstalledModelInfo>, String> {
        self.list_installed_model_details()
    }

    fn chat_with_timeout(
        &self,
        model: &str,
        messages: &[Message],
        timeout: Duration,
    ) -> ModelChatResult {
        match self.send_chat_with_timeout(model, messages, Some(timeout)) {
            OllamaChatOutcome::Succeeded {
                content,
                reported_model,
            } => ModelChatResult::Succeeded {
                content,
                reported_model,
            },
            OllamaChatOutcome::EmptyResponse { reported_model } => {
                ModelChatResult::EmptyResponse { reported_model }
            }
            OllamaChatOutcome::Timeout { message } => ModelChatResult::Timeout { message },
            OllamaChatOutcome::Error {
                message,
                is_connectivity_error,
            } => ModelChatResult::Error {
                message,
                is_global_or_connectivity: is_connectivity_error,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Configuração, eventos de progresso e execução controlada
// ---------------------------------------------------------------------------

/// Parâmetros de configuração do executor de avaliação (`eval run`).
#[derive(Debug, Clone)]
pub struct EvalRunConfig {
    /// Caminho do arquivo de dataset (ex.: `eval/cases_v1.json`).
    pub dataset_path: PathBuf,
    /// Filtro de seleção de casos (categorias, IDs, limite).
    pub filter: DatasetFilter,
    /// Modelo alvo a ser avaliado.
    pub model: String,
    /// Modelo juiz local opcional (`--judge-model`), sempre marcado como estimativa.
    pub judge_model: Option<String>,
    /// Diretório base onde será criado o subdiretório único do relatório.
    pub out_dir: PathBuf,
    /// Timeout máximo por requisição ao Ollama (em segundos, `> 0`).
    pub timeout_secs: u64,
    /// Duração máxima total da execução (ex.: `Some(Duration::from_secs(7 * 3600))`).
    pub max_duration: Option<Duration>,
    /// Número de repetições por caso (`1..=MAX_EVAL_REPEAT`, padrão `1`).
    pub repeat: usize,
    /// Limite de falhas técnicas globais/conectividade consecutivas antes de abortar cedo.
    pub max_consecutive_global_failures: usize,
    /// Flag atômica opcional para cancelamento externo/testes.
    pub cancel_flag: Option<Arc<AtomicBool>>,
    /// Quando `true`, instala o handler de console do sistema para `Ctrl+C`.
    pub install_ctrlc_handler: bool,
    /// Quando `true` (`--with-kb`), injeta notas técnicas da base local `knowledge/sysadmin_kb.json`.
    pub use_kb: bool,
}

impl EvalRunConfig {
    /// Cria uma configuração padrão para o modelo informado.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            dataset_path: default_dataset_path(),
            filter: DatasetFilter::default(),
            model: model.into(),
            judge_model: None,
            out_dir: default_reports_dir(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            max_duration: None,
            repeat: DEFAULT_EVAL_REPEAT,
            max_consecutive_global_failures: DEFAULT_CONSECUTIVE_GLOBAL_FAILURES_LIMIT,
            cancel_flag: None,
            install_ctrlc_handler: false,
            use_kb: false,
        }
    }
}

/// Eventos de progresso emitidos sequencialmente durante [`run_evaluation_with_progress`].
#[derive(Debug, Clone)]
pub enum EvalProgressEvent {
    Started {
        model_resolved: String,
        judge_resolved: Option<String>,
        selected_cases: usize,
        repeat: usize,
        planned_executions: usize,
    },
    CaseStarted {
        execution_index: usize,
        planned_executions: usize,
        case_id: String,
        repeat_index: usize,
        category: EvalCategory,
        title: String,
    },
    CaseFinished {
        execution_index: usize,
        planned_executions: usize,
        case_id: String,
        repeat_index: usize,
        status: CaseExecutionStatus,
        duration_ms: u64,
        deterministic_total: Option<u32>,
        deterministic_max: u32,
        needs_human_review: bool,
    },
    Interrupted {
        cancelled: bool,
        max_duration_reached: bool,
        early_abort_reason: Option<String>,
    },
    ReportWritten {
        paths: ReportPaths,
    },
}

/// Resultado consolidado de [`run_evaluation`], contendo o relatório em memória e os
/// caminhos gravados em disco.
#[derive(Debug, Clone)]
pub struct EvalRunOutcome {
    pub report: EvalReport,
    pub paths: ReportPaths,
}

/// Executa a avaliação automatizada sem callback de progresso.
#[cfg(test)]
pub fn run_evaluation<C: EvalModelClient + ?Sized>(
    client: &C,
    config: &EvalRunConfig,
) -> Result<EvalRunOutcome, String> {
    run_evaluation_with_progress(client, config, |_| {})
}

fn evaluate_optional_judge<C: EvalModelClient + ?Sized>(
    client: &C,
    case: &EvalCase,
    response_content: &str,
    evaluated_model_name: &str,
    judge_ident: &mut ModelIdentification,
    base_req_timeout: Duration,
    max_duration: Option<Duration>,
    run_timer: &Instant,
    is_cancelled: bool,
    cancelled_out: &mut bool,
    max_duration_reached_out: &mut bool,
) -> JudgeEvaluation {
    if is_cancelled {
        *cancelled_out = true;
        return judge_failure(
            &judge_ident.resolved_name,
            evaluated_model_name,
            "execução cancelada (Ctrl+C) antes da chamada ao juiz local",
        );
    }

    let j_timeout = match max_duration {
        Some(max_dur) => {
            let elapsed = run_timer.elapsed();
            if elapsed >= max_dur {
                *max_duration_reached_out = true;
                return judge_failure(
                    &judge_ident.resolved_name,
                    evaluated_model_name,
                    "tempo máximo da avaliação esgotado antes da chamada ao juiz local",
                );
            }
            let rem = max_dur.saturating_sub(elapsed);
            base_req_timeout.min(rem).max(Duration::from_millis(1))
        }
        None => base_req_timeout,
    };

    let j_prompt = build_judge_prompt(case, response_content);
    match client.chat_with_timeout(
        &judge_ident.resolved_name,
        &j_prompt.to_messages(),
        j_timeout,
    ) {
        ModelChatResult::Succeeded {
            content: j_raw,
            reported_model: j_rep,
        } => {
            if let Some(rep) = j_rep.filter(|s| !s.trim().is_empty()) {
                judge_ident.ollama_reported_model = Some(rep);
            }
            parse_judge_response(&j_raw, &judge_ident.resolved_name, evaluated_model_name)
        }
        ModelChatResult::EmptyResponse { .. } => judge_failure(
            &judge_ident.resolved_name,
            evaluated_model_name,
            "o modelo juiz retornou resposta vazia",
        ),
        ModelChatResult::Timeout { message } | ModelChatResult::Error { message, .. } => {
            judge_failure(&judge_ident.resolved_name, evaluated_model_name, &message)
        }
    }
}

/// Executa a avaliação automatizada sequencialmente (`concurrency = 1`), validando
/// previamente o dataset e os modelos instalados, respeitando timeouts, `--max-duration`,
/// `--repeat` e `Ctrl+C`, e gravando o pacote de relatórios (JSON + Markdown) ao final.
pub fn run_evaluation_with_progress<C, F>(
    client: &C,
    config: &EvalRunConfig,
    mut on_progress: F,
) -> Result<EvalRunOutcome, String>
where
    C: EvalModelClient + ?Sized,
    F: FnMut(&EvalProgressEvent),
{
    // 1. Validação dos parâmetros de execução.
    if config.timeout_secs == 0 {
        return Err(
            "O timeout por requisição ('--timeout-secs') deve ser maior que zero.".to_string(),
        );
    }
    validate_repeat(config.repeat)?;

    if let Some(max_dur) = config.max_duration {
        if max_dur.is_zero() {
            return Err("A duração máxima ('--max-duration') deve ser maior que zero.".to_string());
        }
        if max_dur.as_secs() > MAX_ALLOWED_DURATION_SECS {
            return Err(format!(
                "A duração máxima ({}s) excede o limite permitido de {} segundos (7 dias).",
                max_dur.as_secs(),
                MAX_ALLOWED_DURATION_SECS
            ));
        }
    }

    let model_trimmed = config.model.trim();
    if model_trimmed.is_empty() {
        return Err(
            "O nome do modelo a avaliar não pode ser vazio. Informe '--model <nome>'.".to_string(),
        );
    }

    if config.out_dir.as_os_str().is_empty() {
        return Err("O diretório de saída ('--out-dir') não pode ser vazio.".to_string());
    }
    if config.out_dir.exists() && !config.out_dir.is_dir() {
        return Err(format!(
            "O caminho de saída '{}' já existe e não é um diretório.",
            config.out_dir.display()
        ));
    }

    // 2. Carregamento, validação e filtragem do dataset (100% offline).
    let loaded_dataset = load_and_validate_dataset(&config.dataset_path)?;
    let selected_refs = filter_cases(&loaded_dataset.dataset, &config.filter)?;
    let selected_cases: Vec<EvalCase> = selected_refs.into_iter().cloned().collect();

    // 3. Pré-checagem de conectividade com o Ollama e presença local do(s) modelo(s) sem baixar.
    let mut model_ident = client.get_model_identification(model_trimmed)?;
    let mut judge_ident: Option<ModelIdentification> = match &config.judge_model {
        Some(raw_jm) => {
            let trimmed_jm = raw_jm.trim();
            if trimmed_jm.is_empty() {
                return Err("O nome em '--judge-model' não pode ser vazio.".to_string());
            }
            Some(client.get_model_identification(trimmed_jm)?)
        }
        None => None,
    };

    // 4. Instalação opcional do handler de Ctrl+C e preparação do loop sequencial.
    let ctrlc_guard = if config.install_ctrlc_handler {
        Some(CtrlCGuard::install())
    } else {
        None
    };

    let check_cancelled = || -> bool {
        let ext = config
            .cancel_flag
            .as_ref()
            .map_or(false, |f| f.load(Ordering::SeqCst));
        let sys = ctrlc_guard.as_ref().map_or(false, |g| g.is_triggered());
        if sys {
            if let Some(flag) = &config.cancel_flag {
                flag.store(true, Ordering::SeqCst);
            }
        }
        ext || sys
    };

    let planned_executions = selected_cases.len().saturating_mul(config.repeat);
    let deterministic_max = (MetricDimension::ALL.len() as u32) * (SCORE_MAX as u32);
    let system_prompt = build_reproducible_system_prompt();
    let base_req_timeout = Duration::from_secs(config.timeout_secs);
    let consecutive_limit = config.max_consecutive_global_failures.max(1);

    on_progress(&EvalProgressEvent::Started {
        model_resolved: model_ident.resolved_name.clone(),
        judge_resolved: judge_ident.as_ref().map(|j| j.resolved_name.clone()),
        selected_cases: selected_cases.len(),
        repeat: config.repeat,
        planned_executions,
    });

    let started_at = SystemTime::now();
    let run_timer = Instant::now();

    let mut records: Vec<CaseExecutionRecord> = Vec::with_capacity(planned_executions);
    let mut cancelled = false;
    let mut max_duration_reached = false;
    let mut early_abort_reason: Option<String> = None;
    let mut consecutive_global_failures: usize = 0;
    let mut execution_index: usize = 0;

    'outer: for repeat_index in 1..=config.repeat {
        for case in &selected_cases {
            if check_cancelled() {
                cancelled = true;
                break 'outer;
            }

            let req_timeout = match config.max_duration {
                Some(max_dur) => {
                    let elapsed = run_timer.elapsed();
                    if elapsed >= max_dur {
                        max_duration_reached = true;
                        break 'outer;
                    }
                    let remaining = max_dur.saturating_sub(elapsed);
                    base_req_timeout
                        .min(remaining)
                        .max(Duration::from_millis(1))
                }
                None => base_req_timeout,
            };

            execution_index += 1;
            on_progress(&EvalProgressEvent::CaseStarted {
                execution_index,
                planned_executions,
                case_id: case.id.clone(),
                repeat_index,
                category: case.category,
                title: case.title.clone(),
            });

            let mut messages = build_case_messages(case, &system_prompt);
            if config.use_kb {
                let kb_hits = crate::knowledge::retrieve_relevant_notes(&case.question, 2);
                if let Some(grounding) = crate::knowledge::format_grounding_note(&kb_hits) {
                    messages.insert(1, Message::new(Role::System, grounding));
                }
            }
            let case_timer = Instant::now();
            let chat_result =
                client.chat_with_timeout(&model_ident.resolved_name, &messages, req_timeout);
            let duration_ms = case_timer.elapsed().as_millis().min(u64::MAX as u128) as u64;

            let record = match chat_result {
                ModelChatResult::Succeeded {
                    content,
                    reported_model,
                } => {
                    consecutive_global_failures = 0;
                    if let Some(rep) = reported_model.filter(|s| !s.trim().is_empty()) {
                        model_ident.ollama_reported_model = Some(rep);
                    }

                    if content.trim().is_empty() {
                        let det = evaluate_deterministic(case, "");
                        CaseExecutionRecord {
                            case: case.clone(),
                            repeat_index,
                            status: CaseExecutionStatus::EmptyResponse,
                            raw_response: Some(String::new()),
                            error_message: Some(
                                "O modelo avaliado retornou resposta vazia.".to_string(),
                            ),
                            duration_ms,
                            deterministic: Some(det),
                            judge_estimate: None,
                        }
                    } else {
                        let user_prompt_text = messages
                            .last()
                            .map(|m| m.content.as_str())
                            .unwrap_or(&case.question);
                        let guarded_content = crate::guardrails::apply_advisory_guardrails(
                            user_prompt_text,
                            &content,
                        );
                        let det = evaluate_deterministic(case, &guarded_content);
                        let is_canc_before_judge = check_cancelled();
                        let judge_estimate = match judge_ident.as_mut() {
                            Some(j_ident) => Some(evaluate_optional_judge(
                                client,
                                case,
                                &guarded_content,
                                &model_ident.resolved_name,
                                j_ident,
                                base_req_timeout,
                                config.max_duration,
                                &run_timer,
                                is_canc_before_judge,
                                &mut cancelled,
                                &mut max_duration_reached,
                            )),
                            None => None,
                        };

                        CaseExecutionRecord {
                            case: case.clone(),
                            repeat_index,
                            status: CaseExecutionStatus::Succeeded,
                            raw_response: Some(guarded_content),
                            error_message: None,
                            duration_ms,
                            deterministic: Some(det),
                            judge_estimate,
                        }
                    }
                }
                ModelChatResult::EmptyResponse { reported_model } => {
                    consecutive_global_failures = 0;
                    if let Some(rep) = reported_model.filter(|s| !s.trim().is_empty()) {
                        model_ident.ollama_reported_model = Some(rep);
                    }
                    let det = evaluate_deterministic(case, "");
                    CaseExecutionRecord {
                        case: case.clone(),
                        repeat_index,
                        status: CaseExecutionStatus::EmptyResponse,
                        raw_response: Some(String::new()),
                        error_message: Some(
                            "O modelo avaliado retornou resposta vazia.".to_string(),
                        ),
                        duration_ms,
                        deterministic: Some(det),
                        judge_estimate: None,
                    }
                }
                ModelChatResult::Timeout { message } => {
                    if let Some(max_dur) = config.max_duration {
                        if run_timer.elapsed() >= max_dur {
                            max_duration_reached = true;
                        }
                    }
                    CaseExecutionRecord {
                        case: case.clone(),
                        repeat_index,
                        status: CaseExecutionStatus::Timeout,
                        raw_response: None,
                        error_message: Some(message),
                        duration_ms,
                        deterministic: None,
                        judge_estimate: None,
                    }
                }
                ModelChatResult::Error {
                    message,
                    is_global_or_connectivity,
                } => {
                    if is_global_or_connectivity {
                        consecutive_global_failures += 1;
                    } else {
                        consecutive_global_failures = 0;
                    }

                    if consecutive_global_failures >= consecutive_limit {
                        early_abort_reason = Some(format!(
                            "Execução interrompida após {} falhas técnicas globais/conectividade consecutivas. Último erro: {}",
                            consecutive_global_failures, message
                        ));
                    }

                    CaseExecutionRecord {
                        case: case.clone(),
                        repeat_index,
                        status: CaseExecutionStatus::Error,
                        raw_response: None,
                        error_message: Some(message),
                        duration_ms,
                        deterministic: None,
                        judge_estimate: None,
                    }
                }
            };

            let det_total = record.deterministic.as_ref().map(|d| d.total_score);
            let needs_hr = record.case.requires_human_review_by_default
                || record.status != CaseExecutionStatus::Succeeded
                || record
                    .deterministic
                    .as_ref()
                    .map_or(false, |d| d.needs_human_review)
                || record
                    .judge_estimate
                    .as_ref()
                    .map_or(false, |j| j.needs_human_review);

            let finished_status = record.status;
            records.push(record);

            on_progress(&EvalProgressEvent::CaseFinished {
                execution_index,
                planned_executions,
                case_id: case.id.clone(),
                repeat_index,
                status: finished_status,
                duration_ms,
                deterministic_total: det_total,
                deterministic_max,
                needs_human_review: needs_hr,
            });

            if check_cancelled() {
                cancelled = true;
            }
            if let Some(max_dur) = config.max_duration {
                if run_timer.elapsed() >= max_dur {
                    max_duration_reached = true;
                }
            }
            if cancelled || max_duration_reached || early_abort_reason.is_some() {
                break 'outer;
            }
        }
    }

    let finished_at = SystemTime::now();

    if cancelled || max_duration_reached || early_abort_reason.is_some() {
        on_progress(&EvalProgressEvent::Interrupted {
            cancelled,
            max_duration_reached,
            early_abort_reason: early_abort_reason.clone(),
        });
    }

    let run_params = EvalRunParams {
        ollama_url: client.endpoint_url().to_string(),
        concurrency: FIXED_EVAL_CONCURRENCY,
        timeout_secs: config.timeout_secs,
        max_duration_secs: config.max_duration.map(|d| d.as_secs()),
        repeat: config.repeat,
        judge_model: judge_ident.as_ref().map(|j| j.resolved_name.clone()),
        categories_filter: config
            .filter
            .categories
            .iter()
            .map(|c| c.as_str().to_string())
            .collect(),
        case_ids_filter: config.filter.case_ids.clone(),
        limit: config.filter.limit,
        out_dir: format_path_for_report(&config.out_dir),
        system_prompt_mode: EVAL_SYSTEM_PROMPT_MODE.to_string(),
    };

    let report = build_eval_report(EvalReportInput {
        started_at,
        finished_at,
        run_params,
        model: model_ident,
        judge_model: judge_ident,
        dataset_path: format_path_for_report(&config.dataset_path),
        loaded_dataset,
        selected_cases,
        cancelled,
        max_duration_reached,
        early_abort_reason,
        records,
    });

    let paths = write_report_bundle(&config.out_dir, &report)?;
    on_progress(&EvalProgressEvent::ReportWritten {
        paths: paths.clone(),
    });

    Ok(EvalRunOutcome { report, paths })
}

// ---------------------------------------------------------------------------
// Testes unitários (100% offline, sem Ollama ativo, sem tocar em servermind.db)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CREATOR_IDENTITY_INSTRUCTION, IMMUTABLE_SAFETY_INSTRUCTIONS};
    use crate::ollama::OllamaModelDetails;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::AtomicU64;
    use std::thread;
    use std::time::UNIX_EPOCH;

    static TEST_DIR_SEQ: AtomicU64 = AtomicU64::new(0);

    struct TempDirGuard {
        path: PathBuf,
    }

    impl TempDirGuard {
        fn new(prefix: &str) -> Self {
            let seq = TEST_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "servermind_runner_{}_{}_{}_{}",
                prefix,
                std::process::id(),
                nanos,
                seq
            ));
            fs::create_dir_all(&path).expect("deve criar diretório temporário de teste");
            Self { path }
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn real_dataset_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("eval")
            .join("cases_v1.json")
    }

    struct MockEvalClient {
        endpoint: String,
        installed: Result<Vec<InstalledModelInfo>, String>,
        responses: RefCell<VecDeque<ModelChatResult>>,
        recorded_timeouts: RefCell<Vec<Duration>>,
        per_call_sleep: Duration,
        cancel_after_calls: Option<(usize, Arc<AtomicBool>)>,
        calls_made: RefCell<usize>,
    }

    impl MockEvalClient {
        fn with_responses(responses: Vec<ModelChatResult>) -> Self {
            let installed = Ok(vec![
                InstalledModelInfo {
                    name: "llama3.2:latest".to_string(),
                    model: Some("llama3.2:latest".to_string()),
                    modified_at: Some("2026-03-10T00:00:00Z".to_string()),
                    size: Some(2_000_000_000),
                    digest: Some("sha256:11223344556677889900aabbccddeeff".to_string()),
                    details: Some(OllamaModelDetails {
                        format: Some("gguf".to_string()),
                        family: Some("llama".to_string()),
                        parameter_size: Some("3.2B".to_string()),
                        quantization_level: Some("Q4_K_M".to_string()),
                    }),
                },
                InstalledModelInfo {
                    name: "qwen2.5:7b".to_string(),
                    model: Some("qwen2.5:7b".to_string()),
                    modified_at: None,
                    size: None,
                    digest: None,
                    details: None,
                },
            ]);
            Self {
                endpoint: "http://127.0.0.1:11434".to_string(),
                installed,
                responses: RefCell::new(VecDeque::from(responses)),
                recorded_timeouts: RefCell::new(Vec::new()),
                per_call_sleep: Duration::ZERO,
                cancel_after_calls: None,
                calls_made: RefCell::new(0),
            }
        }
    }

    impl EvalModelClient for MockEvalClient {
        fn endpoint_url(&self) -> &str {
            &self.endpoint
        }

        fn list_models(&self) -> Result<Vec<InstalledModelInfo>, String> {
            self.installed.clone()
        }

        fn chat_with_timeout(
            &self,
            _model: &str,
            _messages: &[Message],
            timeout: Duration,
        ) -> ModelChatResult {
            self.recorded_timeouts.borrow_mut().push(timeout);
            if !self.per_call_sleep.is_zero() {
                thread::sleep(self.per_call_sleep);
            }
            let mut count = self.calls_made.borrow_mut();
            *count += 1;
            let current = *count;
            drop(count);

            if let Some((limit, ref flag)) = self.cancel_after_calls {
                if current >= limit {
                    flag.store(true, Ordering::SeqCst);
                }
            }

            self.responses
                .borrow_mut()
                .pop_front()
                .unwrap_or_else(|| ModelChatResult::Succeeded {
                    content: "No systemd, uma unidade .service define o serviço/processo gerenciado e uma unidade .timer agenda a execução periódica por calendário (OnCalendar) como alternativa ao cron, registrando logs no journalctl. Revise manualmente antes de aplicar.".to_string(),
                    reported_model: Some("llama3.2:latest".to_string()),
                })
        }
    }

    #[test]
    fn test_parse_duration_str_accepts_valid_and_rejects_invalid() {
        assert_eq!(
            parse_duration_str("30s").expect("30s válido"),
            Duration::from_secs(30)
        );
        assert_eq!(
            parse_duration_str("15m").expect("15m válido"),
            Duration::from_secs(900)
        );
        assert_eq!(
            parse_duration_str("7h").expect("7h válido"),
            Duration::from_secs(7 * 3_600)
        );
        assert_eq!(
            parse_duration_str("120").expect("inteiro puro em segundos"),
            Duration::from_secs(120)
        );
        assert_eq!(
            parse_duration_str("1h30m10s").expect("composto válido"),
            Duration::from_secs(5_410)
        );

        assert!(parse_duration_str("").is_err());
        assert!(parse_duration_str("   ").is_err());
        assert!(parse_duration_str("0").is_err());
        assert!(parse_duration_str("0s").is_err());
        assert!(parse_duration_str("0h").is_err());
        assert!(parse_duration_str("-10s").is_err());
        assert!(parse_duration_str("abc").is_err());
        assert!(parse_duration_str("10d").is_err());
        assert!(parse_duration_str("200h").is_err());
    }

    #[test]
    fn test_validate_repeat_enforces_bounds() {
        assert_eq!(validate_repeat(1).expect("1 é válido"), 1);
        assert_eq!(
            validate_repeat(MAX_EVAL_REPEAT).expect("MAX é válido"),
            MAX_EVAL_REPEAT
        );
        assert!(validate_repeat(0).is_err());
        assert!(validate_repeat(MAX_EVAL_REPEAT + 1).is_err());
    }

    #[test]
    fn test_reproducible_system_prompt_does_not_leak_custom_or_env() {
        let prompt = build_reproducible_system_prompt();
        assert!(prompt.contains(DEFAULT_CUSTOM_INSTRUCTIONS.trim()));
        assert_eq!(prompt.matches(CREATOR_IDENTITY_INSTRUCTION).count(), 1);
        assert!(prompt.contains(IMMUTABLE_SAFETY_INSTRUCTIONS.trim()));
        assert!(!prompt.contains("Memórias explícitas"));
    }

    #[test]
    fn test_run_evaluation_success_and_judge_estimate_and_anti_overwrite() {
        let tmp = TempDirGuard::new("success_judge");
        let judge_json = r#"{
            "scores": {
                "technical_correctness": {"score": 2, "justification": "Explicou .service e .timer com precisão."},
                "relevance": {"score": 2, "justification": "Direto ao ponto."},
                "clarity": {"score": 2, "justification": "Texto claro em PT-BR."},
                "uncertainty_recognition": {"score": 2, "justification": "Sem excesso de confiança."},
                "safety": {"score": 2, "justification": "Seguro e consultivo."},
                "expected_behavior_fit": {"score": 2, "justification": "Respondeu adequadamente."}
            },
            "overall_comment": "Boa resposta técnica."
        }"#;

        let client = MockEvalClient::with_responses(vec![
            ModelChatResult::Succeeded {
                content: "No systemd, uma unidade .service define como iniciar e gerenciar um processo ou serviço, enquanto uma unidade .timer agenda a ativação por calendário (OnCalendar) ou intervalo, substituindo tarefas do cron com integração ao journalctl.".to_string(),
                reported_model: Some("llama3.2:latest".to_string()),
            },
            ModelChatResult::Succeeded {
                content: judge_json.to_string(),
                reported_model: Some("llama3.2:latest".to_string()),
            },
        ]);

        let mut cfg = EvalRunConfig::new("llama3.2");
        cfg.dataset_path = real_dataset_path();
        cfg.out_dir = tmp.path.clone();
        cfg.filter.case_ids = vec!["exp-01".to_string()];
        cfg.judge_model = Some("llama3.2".to_string());

        let outcome1 = run_evaluation(&client, &cfg).expect("avaliação 1 deve ter sucesso");
        assert_eq!(outcome1.report.execution_stats.executed_count, 1);
        assert_eq!(outcome1.report.execution_stats.succeeded_count, 1);
        assert!(!outcome1.report.execution_stats.global_failure_detected);
        assert!(outcome1.report.overall_scores.judge_enabled);
        assert!(outcome1.report.overall_scores.judge_is_estimate);
        assert!(outcome1.report.overall_scores.judge_self_judging);
        assert!(outcome1
            .report
            .overall_scores
            .judge_self_judging_warning
            .is_some());
        assert_eq!(
            outcome1.report.model.digest.as_deref(),
            Some("sha256:11223344556677889900aabbccddeeff")
        );
        assert_eq!(outcome1.report.dataset.path, "eval/cases_v1.json");
        assert_eq!(
            outcome1.report.run_params.system_prompt_mode,
            "default_reproducible_advisory_v11"
        );
        assert!(!outcome1
            .report
            .run_params
            .out_dir
            .to_ascii_lowercase()
            .contains("users/"));
        assert!(outcome1.paths.json_path.is_file());
        assert!(outcome1.paths.markdown_path.is_file());

        // Segunda execução no mesmo out_dir jamais deve sobrescrever a primeira.
        let outcome2 = run_evaluation(&client, &cfg).expect("avaliação 2 deve ter sucesso");
        assert_ne!(outcome1.paths.directory, outcome2.paths.directory);
        assert!(outcome1.paths.json_path.is_file());
        assert!(outcome2.paths.json_path.is_file());
    }

    #[test]
    fn test_format_path_for_report_relativizes_base_dir_and_redacts_local_user() {
        assert_eq!(
            format_path_for_report(&default_dataset_path()),
            "eval/cases_v1.json"
        );
        assert_eq!(
            format_path_for_report(&default_reports_dir()),
            "eval_reports"
        );
        assert_eq!(
            format_path_for_report(Path::new(r"C:\Users\usuario_secreto\Desktop\relatorios")),
            "~/Desktop/relatorios"
        );
        assert_eq!(
            format_path_for_report(Path::new("/home/usuario_secreto/eval_out")),
            "~/eval_out"
        );
    }

    #[test]
    fn test_run_evaluation_handles_empty_timeout_and_error_statuses() {
        let tmp = TempDirGuard::new("statuses");
        let client = MockEvalClient::with_responses(vec![
            ModelChatResult::Succeeded {
                content: "No systemd, .service gerencia processos e .timer agenda ativações periódicas como no cron com logs no journalctl.".to_string(),
                reported_model: Some("llama3.2:latest".to_string()),
            },
            ModelChatResult::EmptyResponse {
                reported_model: Some("llama3.2:latest".to_string()),
            },
            ModelChatResult::Timeout {
                message: "Tempo limite de 30 segundo(s) esgotado.".to_string(),
            },
            ModelChatResult::Error {
                message: "Erro específico de processamento do modelo (HTTP 400).".to_string(),
                is_global_or_connectivity: false,
            },
        ]);

        let mut cfg = EvalRunConfig::new("llama3.2");
        cfg.dataset_path = real_dataset_path();
        cfg.out_dir = tmp.path.clone();
        cfg.filter.limit = Some(4);

        let outcome = run_evaluation(&client, &cfg).expect("deve concluir registrando falhas");
        let st = &outcome.report.execution_stats;
        assert_eq!(st.planned_executions, 4);
        assert_eq!(st.executed_count, 4);
        assert_eq!(st.succeeded_count, 1);
        assert_eq!(st.empty_response_count, 1);
        assert_eq!(st.timeout_count, 1);
        assert_eq!(st.error_count, 1);
        assert_eq!(st.failed_count, 2);
        assert!(!st.global_failure_detected);
        assert!(st.early_abort_reason.is_none());
    }

    #[test]
    fn test_run_evaluation_aborts_early_on_consecutive_global_failures() {
        let tmp = TempDirGuard::new("global_abort");
        let client = MockEvalClient::with_responses(vec![
            ModelChatResult::Error {
                message: "Não foi possível conectar ao serviço local do Ollama.".to_string(),
                is_global_or_connectivity: true,
            },
            ModelChatResult::Error {
                message: "Não foi possível conectar ao serviço local do Ollama.".to_string(),
                is_global_or_connectivity: true,
            },
            ModelChatResult::Error {
                message: "Não foi possível conectar ao serviço local do Ollama.".to_string(),
                is_global_or_connectivity: true,
            },
        ]);

        let mut cfg = EvalRunConfig::new("llama3.2");
        cfg.dataset_path = real_dataset_path();
        cfg.out_dir = tmp.path.clone();
        cfg.filter.limit = Some(5);
        cfg.max_consecutive_global_failures = 3;

        let outcome = run_evaluation(&client, &cfg).expect("deve gravar relatório mesmo abortando");
        let st = &outcome.report.execution_stats;
        assert_eq!(st.planned_executions, 5);
        assert_eq!(st.executed_count, 3);
        assert_eq!(st.skipped_count, 2);
        assert_eq!(st.error_count, 3);
        assert!(st.global_failure_detected);
        assert!(st.early_abort_reason.is_some());
        assert!(outcome.paths.json_path.is_file());
        assert!(outcome.paths.markdown_path.is_file());
    }

    #[test]
    fn test_run_evaluation_respects_repeat_and_cancellation_flag() {
        let tmp = TempDirGuard::new("cancel_repeat");
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let mut client = MockEvalClient::with_responses(Vec::new());
        client.cancel_after_calls = Some((2, Arc::clone(&cancel_flag)));

        let mut cfg = EvalRunConfig::new("llama3.2");
        cfg.dataset_path = real_dataset_path();
        cfg.out_dir = tmp.path.clone();
        cfg.filter.limit = Some(2);
        cfg.repeat = 2;
        cfg.cancel_flag = Some(cancel_flag);

        let outcome = run_evaluation(&client, &cfg).expect("cancelamento deve gerar relatório");
        let st = &outcome.report.execution_stats;
        assert_eq!(st.planned_executions, 4);
        assert_eq!(st.executed_count, 2);
        assert_eq!(st.skipped_count, 2);
        assert!(st.cancelled);
        assert!(outcome.paths.json_path.is_file());
        assert!(outcome.paths.markdown_path.is_file());
    }

    #[test]
    fn test_run_evaluation_respects_max_duration_and_caps_request_timeout() {
        let tmp = TempDirGuard::new("max_duration");
        let mut client = MockEvalClient::with_responses(Vec::new());
        client.per_call_sleep = Duration::from_millis(25);

        let mut cfg = EvalRunConfig::new("llama3.2");
        cfg.dataset_path = real_dataset_path();
        cfg.out_dir = tmp.path.clone();
        cfg.filter.limit = Some(4);
        cfg.timeout_secs = 120;
        cfg.max_duration = Some(Duration::from_millis(35));

        let outcome = run_evaluation(&client, &cfg).expect("max_duration deve gerar relatório");
        let st = &outcome.report.execution_stats;
        assert!(st.max_duration_reached);
        assert!(st.executed_count >= 1 && st.executed_count < 4);

        let timeouts = client.recorded_timeouts.borrow();
        assert!(!timeouts.is_empty());
        for t in timeouts.iter() {
            assert!(*t <= Duration::from_millis(35));
        }
    }

    #[test]
    fn test_run_evaluation_preflight_rejects_missing_model_or_judge() {
        let tmp = TempDirGuard::new("preflight");
        let client = MockEvalClient::with_responses(Vec::new());

        let mut cfg = EvalRunConfig::new("modelo-inexistente:99b");
        cfg.dataset_path = real_dataset_path();
        cfg.out_dir = tmp.path.clone();
        cfg.filter.limit = Some(1);

        let err = run_evaluation(&client, &cfg).expect_err("deve rejeitar modelo não instalado");
        assert!(err.contains("não está instalado"));

        let mut cfg_judge = EvalRunConfig::new("llama3.2");
        cfg_judge.dataset_path = real_dataset_path();
        cfg_judge.out_dir = tmp.path.clone();
        cfg_judge.filter.limit = Some(1);
        cfg_judge.judge_model = Some("juiz-inexistente".to_string());

        let err_judge =
            run_evaluation(&client, &cfg_judge).expect_err("deve rejeitar juiz não instalado");
        assert!(err_judge.contains("não está instalado"));
    }

    #[test]
    fn test_ctrlc_guard_install_and_drop() {
        let guard = CtrlCGuard::install();
        assert!(!guard.is_triggered());
        drop(guard);
    }
}
