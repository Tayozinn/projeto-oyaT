//! Geração e gravação segura de relatórios de avaliação (Etapa 2).
//!
//! Este módulo agrega os resultados por categoria e por métrica, separa explicitamente
//! verificações determinísticas de estimativas opcionais do juiz local (`--judge-model`),
//! formata carimbos de data/hora UTC sem dependências externas, renderiza relatórios em
//! JSON e Markdown e grava cada execução em diretório único sem nunca sobrescrever
//! relatórios anteriores.
//!
//! Este módulo **nunca** acessa `servermind.db`, não faz chamadas de rede e não imprime
//! variáveis de ambiente ou prompts privados.

use crate::eval::dataset::{EvalCase, EvalCategory, ExpectedBehavior, LoadedDataset};
use crate::eval::metrics::{
    judge_disagreements, rubric_documentation, score_level_label, DeterministicEvaluation,
    JudgeEvaluation, MetricDimension, RubricDimensionDoc, JUDGE_ESTIMATE_NOTICE, RUBRIC_VERSION,
    SCORE_MAX, SELF_JUDGING_WARNING,
};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Limite máximo de caracteres preservados por resposta individual no relatório.
pub const MAX_REPORT_RESPONSE_CHARS: usize = 4_000;
/// Versão do formato de saída JSON do relatório.
pub const REPORT_SCHEMA_VERSION: &str = "1.0";

const MAX_ERROR_CHARS: usize = 400;
const MAX_DIR_COLLISION_ATTEMPTS: usize = 10_000;

// ---------------------------------------------------------------------------
// Estruturas serializáveis do relatório
// ---------------------------------------------------------------------------

/// Relatório completo de uma execução de avaliação.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub report_schema_version: String,
    pub metadata: ReportMetadata,
    pub run_params: EvalRunParams,
    pub model: ModelIdentification,
    pub judge_model: Option<ModelIdentification>,
    pub dataset: DatasetMetadata,
    pub execution_stats: ExecutionStats,
    pub rubric: Vec<RubricDimensionDoc>,
    pub overall_scores: OverallScoreSummary,
    pub category_aggregates: Vec<CategoryAggregate>,
    pub metric_aggregates: Vec<MetricAggregate>,
    pub human_review_cases: Vec<HumanReviewItem>,
    pub case_results: Vec<CaseRunResult>,
    pub evaluator_limitations: Vec<String>,
    pub non_guarantee_notice: String,
}

/// Metadados gerais da execução e do programa.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub program_name: String,
    pub program_version: String,
    pub rubric_version: String,
    pub started_at_utc: String,
    pub finished_at_utc: String,
    pub started_at_file_stamp: String,
    pub duration_ms: u64,
    pub advisory_only: bool,
}

/// Parâmetros relevantes usados na execução.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalRunParams {
    pub ollama_url: String,
    pub concurrency: usize,
    pub timeout_secs: u64,
    pub max_duration_secs: Option<u64>,
    pub repeat: usize,
    pub judge_model: Option<String>,
    pub categories_filter: Vec<String>,
    pub case_ids_filter: Vec<String>,
    pub limit: Option<usize>,
    pub out_dir: String,
    pub system_prompt_mode: String,
}

/// Identificação do modelo avaliado (ou juiz) conforme informado pelo Ollama local.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentification {
    pub requested_name: String,
    pub resolved_name: String,
    pub ollama_reported_model: Option<String>,
    pub digest: Option<String>,
    pub size_bytes: Option<u64>,
    pub modified_at: Option<String>,
    pub format: Option<String>,
    pub family: Option<String>,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
}

#[cfg(test)]
impl ModelIdentification {
    pub fn minimal(requested_name: impl Into<String>, resolved_name: impl Into<String>) -> Self {
        Self {
            requested_name: requested_name.into(),
            resolved_name: resolved_name.into(),
            ollama_reported_model: None,
            digest: None,
            size_bytes: None,
            modified_at: None,
            format: None,
            family: None,
            parameter_size: None,
            quantization_level: None,
        }
    }
}

/// Metadados e integridade do conjunto de avaliação utilizado.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub path: String,
    pub schema_version: String,
    pub dataset_version: String,
    pub language: String,
    pub sha256_hex: String,
    pub size_bytes: u64,
    pub total_cases_in_dataset: usize,
    pub selected_cases_count: usize,
}

/// Estatísticas de execução, falhas, timeouts, cancelamento e corrupção global.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionStats {
    pub planned_executions: usize,
    pub executed_count: usize,
    pub succeeded_count: usize,
    pub empty_response_count: usize,
    pub timeout_count: usize,
    pub error_count: usize,
    pub failed_count: usize,
    pub skipped_count: usize,
    pub failure_rate: f64,
    pub timeout_rate: f64,
    pub empty_response_rate: f64,
    pub cancelled: bool,
    pub max_duration_reached: bool,
    pub early_abort_reason: Option<String>,
    pub global_failure_detected: bool,
    pub human_review_count: usize,
    pub judge_invoked_count: usize,
    pub judge_valid_count: usize,
    pub judge_failed_or_invalid_count: usize,
}

/// Resumo global separado entre método determinístico e estimativa LLM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverallScoreSummary {
    pub max_total_score: u32,
    pub deterministic_evaluated_count: usize,
    pub deterministic_mean_total: Option<f64>,
    pub deterministic_mean_normalized: Option<f64>,
    pub judge_enabled: bool,
    pub judge_is_estimate: bool,
    pub judge_self_judging: bool,
    pub judge_self_judging_warning: Option<String>,
    pub judge_estimate_count: usize,
    pub judge_estimate_mean_total: Option<f64>,
}

/// Contagem por nível da escala (`0`, `1`, `2`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreDistribution {
    pub count_0: usize,
    pub count_1: usize,
    pub count_2: usize,
}

impl ScoreDistribution {
    fn record(&mut self, score: u8) {
        match score {
            0 => self.count_0 += 1,
            1 => self.count_1 += 1,
            _ => self.count_2 += 1,
        }
    }
}

/// Média de uma dimensão dentro de uma categoria.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionMean {
    pub dimension: MetricDimension,
    pub count: usize,
    pub mean: Option<f64>,
    pub distribution: ScoreDistribution,
}

/// Agregado de resultados por categoria do dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoryAggregate {
    pub category: EvalCategory,
    pub label: String,
    pub planned_executions: usize,
    pub executed_count: usize,
    pub succeeded_count: usize,
    pub empty_response_count: usize,
    pub timeout_count: usize,
    pub error_count: usize,
    pub failed_count: usize,
    pub human_review_count: usize,
    pub deterministic_evaluated_count: usize,
    pub deterministic_mean_total: Option<f64>,
    pub deterministic_dimension_means: Vec<DimensionMean>,
    pub judge_estimate_count: usize,
    pub judge_estimate_mean_total: Option<f64>,
    pub judge_estimate_dimension_means: Vec<DimensionMean>,
}

/// Agregado global por métrica (exibindo descrição da escala e separando determinístico de estimativa LLM).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricAggregate {
    pub dimension: MetricDimension,
    pub label: String,
    pub what_it_measures: String,
    pub deterministic_method: String,
    pub levels_description: Vec<String>,
    pub deterministic_evaluated_count: usize,
    pub deterministic_mean: Option<f64>,
    pub deterministic_distribution: ScoreDistribution,
    pub judge_estimate_count: usize,
    pub judge_estimate_mean: Option<f64>,
    pub judge_estimate_distribution: ScoreDistribution,
    pub judge_is_estimate: bool,
}

/// Status final da execução de um caso individual.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseExecutionStatus {
    Succeeded,
    EmptyResponse,
    Timeout,
    Error,
}

impl CaseExecutionStatus {
    pub fn label_pt(&self) -> &'static str {
        match self {
            CaseExecutionStatus::Succeeded => "OK",
            CaseExecutionStatus::EmptyResponse => "Resposta vazia",
            CaseExecutionStatus::Timeout => "Timeout",
            CaseExecutionStatus::Error => "Erro",
        }
    }
}

/// Entrada destacada na lista de revisão humana.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanReviewItem {
    pub case_id: String,
    pub repeat_index: usize,
    pub category: EvalCategory,
    pub title: String,
    pub status: CaseExecutionStatus,
    pub deterministic_total: Option<u32>,
    pub reasons: Vec<String>,
}

/// Resultado de uma execução de caso preservado no relatório.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseRunResult {
    pub case_id: String,
    pub repeat_index: usize,
    pub category: EvalCategory,
    pub title: String,
    pub expected_behavior: ExpectedBehavior,
    pub question: String,
    pub context: Option<String>,
    pub criteria_checklist: Vec<String>,
    pub status: CaseExecutionStatus,
    pub error_message: Option<String>,
    pub duration_ms: u64,
    pub response_excerpt: Option<String>,
    pub response_original_chars: usize,
    pub response_truncated: bool,
    pub deterministic: Option<DeterministicEvaluation>,
    pub judge_estimate: Option<JudgeEvaluation>,
    pub judge_disagreements: Vec<String>,
    pub needs_human_review: bool,
    pub human_review_reasons: Vec<String>,
}

/// Entrada de alto nível fornecida pelo executor (`eval_runner`) para construir o relatório.
#[derive(Debug, Clone)]
pub struct CaseExecutionRecord {
    pub case: EvalCase,
    pub repeat_index: usize,
    pub status: CaseExecutionStatus,
    pub raw_response: Option<String>,
    pub error_message: Option<String>,
    pub duration_ms: u64,
    pub deterministic: Option<DeterministicEvaluation>,
    pub judge_estimate: Option<JudgeEvaluation>,
}

/// Entrada completa para construção do relatório via [`build_eval_report`].
#[derive(Debug, Clone)]
pub struct EvalReportInput {
    pub started_at: SystemTime,
    pub finished_at: SystemTime,
    pub run_params: EvalRunParams,
    pub model: ModelIdentification,
    pub judge_model: Option<ModelIdentification>,
    pub dataset_path: String,
    pub loaded_dataset: LoadedDataset,
    pub selected_cases: Vec<EvalCase>,
    pub cancelled: bool,
    pub max_duration_reached: bool,
    pub early_abort_reason: Option<String>,
    pub records: Vec<CaseExecutionRecord>,
}

/// Caminhos gravados em disco por [`write_report_bundle`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportPaths {
    pub directory: PathBuf,
    pub json_path: PathBuf,
    pub markdown_path: PathBuf,
}

// ---------------------------------------------------------------------------
// Utilitários UTC puros em std (sem crates externas)
// ---------------------------------------------------------------------------

/// Converte dias desde 1970-01-01 em `(ano, mês, dia)` no calendário gregoriano proléptico.
///
/// Baseado no algoritmo de Howard Hinnant (`civil_from_days`).
pub fn civil_from_days(days_since_unix_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_unix_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    (year, m as u32, d as u32)
}

fn utc_components(time: SystemTime) -> (i64, u32, u32, u32, u32, u32) {
    let secs = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = (rem / 3_600) as u32;
    let minute = ((rem % 3_600) / 60) as u32;
    let second = (rem % 60) as u32;
    let (year, month, day) = civil_from_days(days);
    (year, month, day, hour, minute, second)
}

/// Formata `SystemTime` em UTC ISO-8601 (`YYYY-MM-DDTHH:MM:SSZ`).
pub fn format_utc_iso8601(time: SystemTime) -> String {
    let (y, m, d, h, min, s) = utc_components(time);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, h, min, s)
}

/// Formata `SystemTime` em carimbo seguro para nomes de diretório/arquivo (`YYYYMMDD-HHMMSSZ`).
pub fn format_utc_filename_stamp(time: SystemTime) -> String {
    let (y, m, d, h, min, s) = utc_components(time);
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}Z", y, m, d, h, min, s)
}

// ---------------------------------------------------------------------------
// Sanitização e truncamento seguro
// ---------------------------------------------------------------------------

fn sanitize_preserving_newlines(raw: &str) -> String {
    raw.chars()
        .filter_map(|c| match c {
            '\n' => Some('\n'),
            '\r' => None,
            '\t' => Some(' '),
            c if c.is_control() => None,
            other => Some(other),
        })
        .collect()
}

fn sanitize_single_line(raw: &str, max_chars: usize) -> String {
    let collapsed = raw
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    match collapsed.char_indices().nth(max_chars) {
        Some((idx, _)) => format!("{}...", &collapsed[..idx]),
        None => collapsed,
    }
}

/// Trunca a resposta preservando fronteiras UTF-8 até `max_chars` e informa tamanho original.
pub fn truncate_response_for_report(raw: &str, max_chars: usize) -> (String, usize, bool) {
    let original_chars = raw.chars().count();
    let cleaned = sanitize_preserving_newlines(raw);
    let cleaned_chars = cleaned.chars().count();
    if cleaned_chars <= max_chars {
        return (cleaned, original_chars, false);
    }
    let cut_idx = cleaned
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(cleaned.len());
    let mut out = cleaned[..cut_idx].to_string();
    out.push_str("\n...[resposta truncada no relatório]");
    (out, original_chars, true)
}

// ---------------------------------------------------------------------------
// Construção e agregação do relatório
// ---------------------------------------------------------------------------

/// Limitações obrigatórias registradas em todo relatório (JSON e Markdown).
pub fn default_evaluator_limitations(judge_enabled: bool, self_judging: bool) -> Vec<String> {
    let mut items = vec![
        "Notas numéricas NÃO são garantia de correção técnica, segurança operacional nem adequação ao seu ambiente de produção.".to_string(),
        "A avaliação automatizada NÃO substitui a revisão humana; comandos sugeridos devem ser sempre conferidos e adaptados pelo administrador antes de qualquer execução.".to_string(),
        "A avaliação determinística usa heurísticas baseadas em substrings, janelas de negação/aviso e listas finitas de sinais: pode gerar falsos positivos e falsos negativos conforme a redação do modelo.".to_string(),
        "O conjunto de casos é pequeno, sintético e cobre apenas cenários selecionados em modo estritamente consultivo; boa nota aqui não implica desempenho geral equivalente em incidentes reais.".to_string(),
    ];
    if judge_enabled {
        items.push(JUDGE_ESTIMATE_NOTICE.to_string());
    }
    if self_judging {
        items.push(SELF_JUDGING_WARNING.to_string());
    }
    items
}

pub const NON_GUARANTEE_NOTICE: &str = "AVISO IMPORTANTE: As pontuações deste relatório são indicadores auxiliares e NÃO constituem garantia de correção técnica ou segurança. O ServerMind AI é estritamente consultivo e este avaliador não substitui a revisão humana.";

fn dedup_strings(items: &mut Vec<String>) {
    let mut seen: Vec<String> = Vec::new();
    items.retain(|item| {
        if seen.contains(item) {
            false
        } else {
            seen.push(item.clone());
            true
        }
    });
}

fn round2(val: f64) -> f64 {
    (val * 100.0).round() / 100.0
}

fn mean_opt(sum: usize, count: usize) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some(round2(sum as f64 / count as f64))
    }
}

/// Constrói um [`EvalReport`] completo e agregado a partir dos registros de execução.
pub fn build_eval_report(input: EvalReportInput) -> EvalReport {
    let duration_ms = input
        .finished_at
        .duration_since(input.started_at)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64;

    let metadata = ReportMetadata {
        program_name: env!("CARGO_PKG_NAME").to_string(),
        program_version: env!("CARGO_PKG_VERSION").to_string(),
        rubric_version: RUBRIC_VERSION.to_string(),
        started_at_utc: format_utc_iso8601(input.started_at),
        finished_at_utc: format_utc_iso8601(input.finished_at),
        started_at_file_stamp: format_utc_filename_stamp(input.started_at),
        duration_ms,
        advisory_only: true,
    };

    let repeat_factor = input.run_params.repeat.max(1);
    let selected_cases_count = input.selected_cases.len();
    let planned_executions = selected_cases_count
        .saturating_mul(repeat_factor)
        .max(input.records.len());

    let dataset = DatasetMetadata {
        path: input.dataset_path,
        schema_version: input.loaded_dataset.dataset.schema_version.clone(),
        dataset_version: input.loaded_dataset.dataset.dataset_version.clone(),
        language: input.loaded_dataset.dataset.language.clone(),
        sha256_hex: input.loaded_dataset.sha256_hex.clone(),
        size_bytes: input.loaded_dataset.size_bytes,
        total_cases_in_dataset: input.loaded_dataset.dataset.cases.len(),
        selected_cases_count,
    };

    let mut case_results: Vec<CaseRunResult> = Vec::with_capacity(input.records.len());
    let mut human_review_cases: Vec<HumanReviewItem> = Vec::new();

    let mut succeeded_count = 0usize;
    let mut empty_response_count = 0usize;
    let mut timeout_count = 0usize;
    let mut error_count = 0usize;
    let mut judge_invoked_count = 0usize;
    let mut judge_valid_count = 0usize;
    let mut judge_failed_or_invalid_count = 0usize;

    for rec in input.records {
        match rec.status {
            CaseExecutionStatus::Succeeded => succeeded_count += 1,
            CaseExecutionStatus::EmptyResponse => empty_response_count += 1,
            CaseExecutionStatus::Timeout => timeout_count += 1,
            CaseExecutionStatus::Error => error_count += 1,
        }

        let (response_excerpt, response_original_chars, response_truncated) =
            match rec.raw_response.as_deref() {
                Some(resp) => {
                    let (ex, orig, trunc) =
                        truncate_response_for_report(resp, MAX_REPORT_RESPONSE_CHARS);
                    (Some(ex), orig, trunc)
                }
                None => (None, 0, false),
            };

        let error_message = rec
            .error_message
            .as_deref()
            .map(|e| sanitize_single_line(e, MAX_ERROR_CHARS));

        let mut reasons: Vec<String> = Vec::new();
        if rec.case.requires_human_review_by_default {
            reasons.push("caso marcado no dataset para revisão humana obrigatória".to_string());
        }
        match rec.status {
            CaseExecutionStatus::EmptyResponse => {
                reasons.push("modelo retornou resposta vazia".to_string());
            }
            CaseExecutionStatus::Timeout => {
                reasons.push("tempo limite esgotado na requisição".to_string());
            }
            CaseExecutionStatus::Error => {
                reasons.push("falha técnica na execução do caso".to_string());
            }
            CaseExecutionStatus::Succeeded => {}
        }

        if let Some(det) = &rec.deterministic {
            reasons.extend(det.human_review_reasons.iter().cloned());
        }

        let mut judge_disagreements_list = Vec::new();
        if let Some(judge) = &rec.judge_estimate {
            judge_invoked_count += 1;
            if judge.parse_ok && judge.error.is_none() {
                judge_valid_count += 1;
            } else {
                judge_failed_or_invalid_count += 1;
            }
            reasons.extend(judge.human_review_reasons.iter().cloned());
            if let Some(det) = &rec.deterministic {
                judge_disagreements_list = judge_disagreements(det, judge);
                reasons.extend(judge_disagreements_list.iter().cloned());
            }
        }
        dedup_strings(&mut reasons);
        let needs_human_review = !reasons.is_empty();

        if needs_human_review {
            human_review_cases.push(HumanReviewItem {
                case_id: rec.case.id.clone(),
                repeat_index: rec.repeat_index,
                category: rec.case.category,
                title: rec.case.title.clone(),
                status: rec.status,
                deterministic_total: rec.deterministic.as_ref().map(|d| d.total_score),
                reasons: reasons.clone(),
            });
        }

        case_results.push(CaseRunResult {
            case_id: rec.case.id,
            repeat_index: rec.repeat_index,
            category: rec.case.category,
            title: rec.case.title,
            expected_behavior: rec.case.expected_behavior,
            question: rec.case.question,
            context: rec.case.context,
            criteria_checklist: rec.case.criteria.checklist,
            status: rec.status,
            error_message,
            duration_ms: rec.duration_ms,
            response_excerpt,
            response_original_chars,
            response_truncated,
            deterministic: rec.deterministic,
            judge_estimate: rec.judge_estimate,
            judge_disagreements: judge_disagreements_list,
            needs_human_review,
            human_review_reasons: reasons,
        });
    }

    let executed_count = case_results.len();
    let failed_count = timeout_count + error_count;
    let skipped_count = planned_executions.saturating_sub(executed_count);
    let denom = executed_count.max(1) as f64;
    let failure_rate = if executed_count == 0 {
        0.0
    } else {
        round2(failed_count as f64 / denom)
    };
    let timeout_rate = if executed_count == 0 {
        0.0
    } else {
        round2(timeout_count as f64 / denom)
    };
    let empty_response_rate = if executed_count == 0 {
        0.0
    } else {
        round2(empty_response_count as f64 / denom)
    };

    let all_failed_or_empty = executed_count > 0
        && succeeded_count == 0
        && (failed_count + empty_response_count) == executed_count;
    let global_failure_detected =
        executed_count == 0 || all_failed_or_empty || input.early_abort_reason.is_some();

    let execution_stats = ExecutionStats {
        planned_executions,
        executed_count,
        succeeded_count,
        empty_response_count,
        timeout_count,
        error_count,
        failed_count,
        skipped_count,
        failure_rate,
        timeout_rate,
        empty_response_rate,
        cancelled: input.cancelled,
        max_duration_reached: input.max_duration_reached,
        early_abort_reason: input
            .early_abort_reason
            .as_deref()
            .map(|r| sanitize_single_line(r, MAX_ERROR_CHARS)),
        global_failure_detected,
        human_review_count: human_review_cases.len(),
        judge_invoked_count,
        judge_valid_count,
        judge_failed_or_invalid_count,
    };

    // Agregados por categoria (todas as categorias presentes nos casos selecionados/executados).
    let category_aggregates =
        build_category_aggregates(&input.selected_cases, &case_results, repeat_factor);

    // Agregados por métrica (sempre as 6 dimensões na ordem canônica).
    let metric_aggregates = build_metric_aggregates(&case_results);

    // Resumo global separado.
    let mut det_eval_count = 0usize;
    let mut det_total_sum = 0usize;
    let mut judge_complete_count = 0usize;
    let mut judge_total_sum = 0usize;
    for r in &case_results {
        if let Some(det) = &r.deterministic {
            det_eval_count += 1;
            det_total_sum += det.total_score as usize;
        }
        if let Some(judge) = &r.judge_estimate {
            let valid_scores: Vec<u8> = MetricDimension::ALL
                .iter()
                .filter_map(|d| judge.score(*d))
                .collect();
            if valid_scores.len() == MetricDimension::ALL.len() {
                judge_complete_count += 1;
                judge_total_sum += valid_scores.iter().map(|&s| s as usize).sum::<usize>();
            }
        }
    }

    let max_total_score = (MetricDimension::ALL.len() as u32) * SCORE_MAX as u32;
    let deterministic_mean_total = mean_opt(det_total_sum, det_eval_count);
    let deterministic_mean_normalized =
        deterministic_mean_total.map(|m| round2(m / max_total_score as f64));

    let judge_enabled = input.run_params.judge_model.is_some() || input.judge_model.is_some();
    let judge_self_judging = input
        .run_params
        .judge_model
        .as_deref()
        .map(|jm| {
            crate::eval::metrics::is_self_judging(jm, &input.model.resolved_name)
                || crate::eval::metrics::is_self_judging(jm, &input.model.requested_name)
        })
        .unwrap_or(false)
        || input
            .judge_model
            .as_ref()
            .map(|jm| {
                crate::eval::metrics::is_self_judging(&jm.resolved_name, &input.model.resolved_name)
                    || crate::eval::metrics::is_self_judging(
                        &jm.requested_name,
                        &input.model.requested_name,
                    )
            })
            .unwrap_or(false)
        || case_results
            .iter()
            .filter_map(|r| r.judge_estimate.as_ref())
            .any(|j| j.self_judging);

    let judge_self_judging_warning = if judge_self_judging {
        Some(SELF_JUDGING_WARNING.to_string())
    } else {
        None
    };

    let overall_scores = OverallScoreSummary {
        max_total_score,
        deterministic_evaluated_count: det_eval_count,
        deterministic_mean_total,
        deterministic_mean_normalized,
        judge_enabled,
        judge_is_estimate: true,
        judge_self_judging,
        judge_self_judging_warning,
        judge_estimate_count: judge_complete_count,
        judge_estimate_mean_total: mean_opt(judge_total_sum, judge_complete_count),
    };

    EvalReport {
        report_schema_version: REPORT_SCHEMA_VERSION.to_string(),
        metadata,
        run_params: input.run_params,
        model: input.model,
        judge_model: input.judge_model,
        dataset,
        execution_stats,
        rubric: rubric_documentation(),
        overall_scores,
        category_aggregates,
        metric_aggregates,
        human_review_cases,
        case_results,
        evaluator_limitations: default_evaluator_limitations(judge_enabled, judge_self_judging),
        non_guarantee_notice: NON_GUARANTEE_NOTICE.to_string(),
    }
}

fn build_category_aggregates(
    selected_cases: &[EvalCase],
    results: &[CaseRunResult],
    repeat_factor: usize,
) -> Vec<CategoryAggregate> {
    let mut out = Vec::new();
    for cat in EvalCategory::ALL {
        let cat_results: Vec<&CaseRunResult> =
            results.iter().filter(|r| r.category == cat).collect();
        let selected_in_cat = selected_cases.iter().filter(|c| c.category == cat).count();
        if selected_in_cat == 0 && cat_results.is_empty() {
            continue;
        }
        let planned_executions = selected_in_cat
            .saturating_mul(repeat_factor)
            .max(cat_results.len());
        let executed_count = cat_results.len();
        let succeeded_count = cat_results
            .iter()
            .filter(|r| r.status == CaseExecutionStatus::Succeeded)
            .count();
        let empty_response_count = cat_results
            .iter()
            .filter(|r| r.status == CaseExecutionStatus::EmptyResponse)
            .count();
        let timeout_count = cat_results
            .iter()
            .filter(|r| r.status == CaseExecutionStatus::Timeout)
            .count();
        let error_count = cat_results
            .iter()
            .filter(|r| r.status == CaseExecutionStatus::Error)
            .count();
        let failed_count = timeout_count + error_count;
        let human_review_count = cat_results.iter().filter(|r| r.needs_human_review).count();

        let mut det_count = 0usize;
        let mut det_sum = 0usize;
        let mut judge_count = 0usize;
        let mut judge_sum = 0usize;

        for r in &cat_results {
            if let Some(det) = &r.deterministic {
                det_count += 1;
                det_sum += det.total_score as usize;
            }
            if let Some(judge) = &r.judge_estimate {
                let valid: Vec<u8> = MetricDimension::ALL
                    .iter()
                    .filter_map(|d| judge.score(*d))
                    .collect();
                if valid.len() == MetricDimension::ALL.len() {
                    judge_count += 1;
                    judge_sum += valid.iter().map(|&s| s as usize).sum::<usize>();
                }
            }
        }

        let deterministic_dimension_means = MetricDimension::ALL
            .iter()
            .map(|dim| {
                let mut cnt = 0usize;
                let mut sum = 0usize;
                let mut dist = ScoreDistribution::default();
                for r in &cat_results {
                    if let Some(det) = &r.deterministic {
                        let s = det.score(*dim);
                        cnt += 1;
                        sum += s as usize;
                        dist.record(s);
                    }
                }
                DimensionMean {
                    dimension: *dim,
                    count: cnt,
                    mean: mean_opt(sum, cnt),
                    distribution: dist,
                }
            })
            .collect();

        let judge_estimate_dimension_means = MetricDimension::ALL
            .iter()
            .map(|dim| {
                let mut cnt = 0usize;
                let mut sum = 0usize;
                let mut dist = ScoreDistribution::default();
                for r in &cat_results {
                    if let Some(judge) = &r.judge_estimate {
                        if let Some(s) = judge.score(*dim) {
                            cnt += 1;
                            sum += s as usize;
                            dist.record(s);
                        }
                    }
                }
                DimensionMean {
                    dimension: *dim,
                    count: cnt,
                    mean: mean_opt(sum, cnt),
                    distribution: dist,
                }
            })
            .collect();

        out.push(CategoryAggregate {
            category: cat,
            label: cat.label_pt().to_string(),
            planned_executions,
            executed_count,
            succeeded_count,
            empty_response_count,
            timeout_count,
            error_count,
            failed_count,
            human_review_count,
            deterministic_evaluated_count: det_count,
            deterministic_mean_total: mean_opt(det_sum, det_count),
            deterministic_dimension_means,
            judge_estimate_count: judge_count,
            judge_estimate_mean_total: mean_opt(judge_sum, judge_count),
            judge_estimate_dimension_means,
        });
    }
    out
}

fn build_metric_aggregates(results: &[CaseRunResult]) -> Vec<MetricAggregate> {
    MetricDimension::ALL
        .iter()
        .map(|dim| {
            let mut det_cnt = 0usize;
            let mut det_sum = 0usize;
            let mut det_dist = ScoreDistribution::default();
            let mut j_cnt = 0usize;
            let mut j_sum = 0usize;
            let mut j_dist = ScoreDistribution::default();

            for r in results {
                if let Some(det) = &r.deterministic {
                    let s = det.score(*dim);
                    det_cnt += 1;
                    det_sum += s as usize;
                    det_dist.record(s);
                }
                if let Some(judge) = &r.judge_estimate {
                    if let Some(s) = judge.score(*dim) {
                        j_cnt += 1;
                        j_sum += s as usize;
                        j_dist.record(s);
                    }
                }
            }

            let levels_description = (0..=SCORE_MAX)
                .map(|s| {
                    format!(
                        "{} ({}): {}",
                        s,
                        score_level_label(s),
                        dim.level_description(s)
                    )
                })
                .collect();

            MetricAggregate {
                dimension: *dim,
                label: dim.label_pt().to_string(),
                what_it_measures: dim.what_it_measures().to_string(),
                deterministic_method: dim.deterministic_method().to_string(),
                levels_description,
                deterministic_evaluated_count: det_cnt,
                deterministic_mean: mean_opt(det_sum, det_cnt),
                deterministic_distribution: det_dist,
                judge_estimate_count: j_cnt,
                judge_estimate_mean: mean_opt(j_sum, j_cnt),
                judge_estimate_distribution: j_dist,
                judge_is_estimate: true,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Renderização JSON e Markdown
// ---------------------------------------------------------------------------

/// Serializa o relatório em JSON legível.
pub fn render_json_report(report: &EvalReport) -> Result<String, String> {
    serde_json::to_string_pretty(report)
        .map_err(|e| format!("Falha ao serializar relatório JSON: {}", e))
}

fn md_cell(raw: &str) -> String {
    raw.replace(['\r', '\n'], " ")
        .replace('|', "\\|")
        .trim()
        .to_string()
}

fn opt_score(v: Option<f64>) -> String {
    match v {
        Some(n) => format!("{:.2}", n),
        None => "N/A".to_string(),
    }
}

fn markdown_fence_for(text: &str) -> String {
    let mut max_run = 2usize;
    let mut cur = 0usize;
    for ch in text.chars() {
        if ch == '`' {
            cur += 1;
            if cur > max_run {
                max_run = cur;
            }
        } else {
            cur = 0;
        }
    }
    "`".repeat(max_run + 1)
}

/// Renderiza o relatório completo em Markdown PT-BR claro e auditável.
pub fn render_markdown_report(report: &EvalReport) -> String {
    let mut md = String::new();
    md.push_str("# Relatório de Avaliação — ServerMind AI\n\n");
    md.push_str(&format!("> **{}**\n\n", report.non_guarantee_notice));

    if report.execution_stats.global_failure_detected {
        md.push_str("> **ALERTA DE FALHA GLOBAL:** A execução registrou falha total ou interrupção prematura. Verifique as taxas de falha/timeout e os motivos abaixo antes de interpretar qualquer métrica.\n\n");
    }
    if report.execution_stats.cancelled {
        md.push_str("> **EXECUÇÃO CANCELADA (Ctrl+C):** Relatório parcial gerado de forma limpa até o ponto de interrupção.\n\n");
    } else if report.execution_stats.max_duration_reached {
        md.push_str("> **TEMPO MÁXIMO ATINGIDO:** A execução foi encerrada ao atingir o limite de duração configurado.\n\n");
    }
    if let Some(warn) = &report.overall_scores.judge_self_judging_warning {
        md.push_str(&format!(
            "> **AVISO DE AUTO-AVALIAÇÃO (`--judge-model`):** {}\n\n",
            warn
        ));
    }

    // 1. Metadados, modelo e dataset
    md.push_str("## 1. Metadados da Execução\n\n");
    md.push_str("| Campo | Valor |\n|---|---|\n");
    md.push_str(&format!(
        "| Programa / Versão | `{}` `v{}` (rubrica `{}`) |\n",
        md_cell(&report.metadata.program_name),
        md_cell(&report.metadata.program_version),
        md_cell(&report.metadata.rubric_version)
    ));
    md.push_str(&format!(
        "| Modo | Estritamente consultivo (`advisory_only = {}`) |\n",
        report.metadata.advisory_only
    ));
    md.push_str(&format!(
        "| Início (UTC) | `{}` |\n",
        md_cell(&report.metadata.started_at_utc)
    ));
    md.push_str(&format!(
        "| Término (UTC) | `{}` (duração: {} ms) |\n",
        md_cell(&report.metadata.finished_at_utc),
        report.metadata.duration_ms
    ));
    md.push_str(&format!(
        "| Modelo solicitado / resolvido | `{}` / `{}` |\n",
        md_cell(&report.model.requested_name),
        md_cell(&report.model.resolved_name)
    ));
    md.push_str(&format!(
        "| Identificação informada pelo Ollama | {} |\n",
        md_cell(&format_model_ident(&report.model))
    ));
    if let Some(jm) = &report.judge_model {
        md.push_str(&format!(
            "| Juiz local opcional (**estimativa**) | `{}` ({}) |\n",
            md_cell(&jm.resolved_name),
            md_cell(&format_model_ident(jm))
        ));
    } else if let Some(jm_name) = &report.run_params.judge_model {
        md.push_str(&format!(
            "| Juiz local opcional (**estimativa**) | `{}` |\n",
            md_cell(jm_name)
        ));
    } else {
        md.push_str("| Juiz local opcional | Desativado (somente avaliação determinística) |\n");
    }
    md.push_str(&format!(
        "| Dataset | `{}` (`schema {}`, `v{}`, `{}`) |\n",
        md_cell(&report.dataset.path),
        md_cell(&report.dataset.schema_version),
        md_cell(&report.dataset.dataset_version),
        md_cell(&report.dataset.language)
    ));
    md.push_str(&format!(
        "| SHA-256 do dataset | `{}` ({} bytes) |\n",
        md_cell(&report.dataset.sha256_hex),
        report.dataset.size_bytes
    ));
    md.push_str(&format!(
        "| Casos selecionados | {} de {} no dataset |\n\n",
        report.dataset.selected_cases_count, report.dataset.total_cases_in_dataset
    ));

    // 2. Parâmetros de execução
    md.push_str("## 2. Parâmetros de Execução\n\n");
    md.push_str("| Parâmetro | Valor |\n|---|---|\n");
    md.push_str(&format!(
        "| Endpoint local Ollama | `{}` |\n",
        md_cell(&report.run_params.ollama_url)
    ));
    md.push_str(&format!(
        "| Concorrência | `{}` |\n",
        report.run_params.concurrency
    ));
    md.push_str(&format!(
        "| Timeout por requisição | `{}s` |\n",
        report.run_params.timeout_secs
    ));
    md.push_str(&format!(
        "| Duração máxima (`--max-duration`) | `{}` |\n",
        report
            .run_params
            .max_duration_secs
            .map(|s| format!("{}s", s))
            .unwrap_or_else(|| "sem limite".to_string())
    ));
    md.push_str(&format!(
        "| Repetições por caso (`--repeat`) | `{}` |\n",
        report.run_params.repeat
    ));
    md.push_str(&format!(
        "| Filtro de categorias | {} |\n",
        if report.run_params.categories_filter.is_empty() {
            "todas".to_string()
        } else {
            md_cell(&report.run_params.categories_filter.join(", "))
        }
    ));
    md.push_str(&format!(
        "| Filtro de IDs | {} |\n",
        if report.run_params.case_ids_filter.is_empty() {
            "todos".to_string()
        } else {
            md_cell(&report.run_params.case_ids_filter.join(", "))
        }
    ));
    md.push_str(&format!(
        "| Limite de casos (`--limit`) | {} |\n",
        report
            .run_params
            .limit
            .map(|l| l.to_string())
            .unwrap_or_else(|| "sem limite".to_string())
    ));
    md.push_str(&format!(
        "| Prompt de sistema | `{}` (sem ler prompts privados/env vars) |\n\n",
        md_cell(&report.run_params.system_prompt_mode)
    ));

    // 3. Estatísticas de execução e saúde
    let st = &report.execution_stats;
    md.push_str("## 3. Estatísticas de Execução, Falhas e Revisão\n\n");
    md.push_str("| Indicador | Valor |\n|---|---|\n");
    md.push_str(&format!(
        "| Execuções planejadas | {} |\n",
        st.planned_executions
    ));
    md.push_str(&format!(
        "| Execuções realizadas | {} (omitidas: {}) |\n",
        st.executed_count, st.skipped_count
    ));
    md.push_str(&format!(
        "| Respostas obtidas com sucesso (não vazias) | {} |\n",
        st.succeeded_count
    ));
    md.push_str(&format!(
        "| Respostas vazias | {} ({:.1}%) |\n",
        st.empty_response_count,
        st.empty_response_rate * 100.0
    ));
    md.push_str(&format!(
        "| Timeouts | {} ({:.1}%) |\n",
        st.timeout_count,
        st.timeout_rate * 100.0
    ));
    md.push_str(&format!(
        "| Outros erros de execução | {} |\n",
        st.error_count
    ));
    md.push_str(&format!(
        "| Taxa total de falha (timeout + erro) | {} ({:.1}%) |\n",
        st.failed_count,
        st.failure_rate * 100.0
    ));
    md.push_str(&format!("| Cancelado (Ctrl+C) | {} |\n", st.cancelled));
    md.push_str(&format!(
        "| Limite de duração atingido | {} |\n",
        st.max_duration_reached
    ));
    md.push_str(&format!(
        "| Falha global detectada | {} |\n",
        st.global_failure_detected
    ));
    if let Some(reason) = &st.early_abort_reason {
        md.push_str(&format!(
            "| Motivo da interrupção antecipada | {} |\n",
            md_cell(reason)
        ));
    }
    md.push_str(&format!(
        "| Casos marcados para revisão humana | {} |\n",
        st.human_review_count
    ));
    md.push_str(&format!(
        "| Pontuação média **determinística** (0–{}) | {} (n = {}) |\n",
        report.overall_scores.max_total_score,
        opt_score(report.overall_scores.deterministic_mean_total),
        report.overall_scores.deterministic_evaluated_count
    ));
    if report.overall_scores.judge_enabled {
        md.push_str(&format!(
            "| Pontuação média **estimativa LLM** (0–{}) *(is_estimate=true)* | {} (válidas = {}, falhas/inválidas = {}) |\n",
            report.overall_scores.max_total_score,
            opt_score(report.overall_scores.judge_estimate_mean_total),
            report.overall_scores.judge_estimate_count,
            st.judge_failed_or_invalid_count
        ));
    }
    md.push('\n');

    // 4. Agregado por categoria
    md.push_str("## 4. Agregado por Categoria\n\n");
    md.push_str("| Categoria | Executadas / Planejadas | OK | Vazias | Timeouts | Erros | Média Determinística (0–12) | Estimativa LLM (0–12) | Revisão Humana |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for cat in &report.category_aggregates {
        let j_col = if report.overall_scores.judge_enabled {
            format!(
                "{} (n={})",
                opt_score(cat.judge_estimate_mean_total),
                cat.judge_estimate_count
            )
        } else {
            "—".to_string()
        };
        md.push_str(&format!(
            "| `{}` ({}) | {}/{} | {} | {} | {} | {} | {} (n={}) | {} | {} |\n",
            md_cell(cat.category.as_str()),
            md_cell(&cat.label),
            cat.executed_count,
            cat.planned_executions,
            cat.succeeded_count,
            cat.empty_response_count,
            cat.timeout_count,
            cat.error_count,
            opt_score(cat.deterministic_mean_total),
            cat.deterministic_evaluated_count,
            j_col,
            cat.human_review_count
        ));
    }
    md.push('\n');

    // 5. Agregado por métrica e descrição da escala
    md.push_str("## 5. Agregado por Métrica e Rubrica (Escala 0–2)\n\n");
    md.push_str(
        "Escala de cada dimensão: **0 = Insuficiente**, **1 = Parcial**, **2 = Adequado**.\n",
    );
    md.push_str("Os resultados **determinísticos** e a **estimativa do juiz LLM** são exibidos em colunas separadas.\n\n");
    md.push_str("| Dimensão | Média Determinística (0–2) | Dist. Determinística (0 / 1 / 2) | Média Estimativa LLM (0–2, *estimativa*) | Dist. Estimativa LLM (0 / 1 / 2) |\n");
    md.push_str("|---|---:|---:|---:|---:|\n");
    for m in &report.metric_aggregates {
        let dd = m.deterministic_distribution;
        let jd = m.judge_estimate_distribution;
        let j_mean = if report.overall_scores.judge_enabled {
            format!(
                "{} (n={})",
                opt_score(m.judge_estimate_mean),
                m.judge_estimate_count
            )
        } else {
            "—".to_string()
        };
        let j_dist = if report.overall_scores.judge_enabled {
            format!("{} / {} / {}", jd.count_0, jd.count_1, jd.count_2)
        } else {
            "—".to_string()
        };
        md.push_str(&format!(
            "| `{}` ({}) | {} (n={}) | {} / {} / {} | {} | {} |\n",
            md_cell(m.dimension.as_str()),
            md_cell(&m.label),
            opt_score(m.deterministic_mean),
            m.deterministic_evaluated_count,
            dd.count_0,
            dd.count_1,
            dd.count_2,
            j_mean,
            j_dist
        ));
    }
    md.push('\n');

    md.push_str("### Definição da Escala por Dimensão\n\n");
    for m in &report.metric_aggregates {
        md.push_str(&format!(
            "- **`{}` — {}**: {}\n  - Método determinístico: {}\n",
            m.dimension.as_str(),
            m.label,
            m.what_it_measures,
            m.deterministic_method
        ));
        for desc in &m.levels_description {
            md.push_str(&format!("  - {}\n", desc));
        }
    }
    md.push('\n');

    // 6. Casos destacados para revisão humana
    md.push_str("## 6. Casos que Exigem Revisão Humana\n\n");
    if report.human_review_cases.is_empty() {
        md.push_str("Nenhum caso sinalizado para revisão humana nesta execução.\n\n");
    } else {
        md.push_str("| Caso | Repetição | Categoria | Status | Nota Determinística | Motivos |\n");
        md.push_str("|---|---:|---|---|---:|---|\n");
        for item in &report.human_review_cases {
            let det_str = item
                .deterministic_total
                .map(|s| format!("{}/{}", s, report.overall_scores.max_total_score))
                .unwrap_or_else(|| "N/A".to_string());
            md.push_str(&format!(
                "| `{}` ({}) | #{} | `{}` | {} | {} | {} |\n",
                md_cell(&item.case_id),
                md_cell(&item.title),
                item.repeat_index,
                md_cell(item.category.as_str()),
                md_cell(item.status.label_pt()),
                det_str,
                md_cell(&item.reasons.join("; "))
            ));
        }
        md.push('\n');
    }

    // 7. Limitações do avaliador (seção obrigatória)
    md.push_str("## 7. Limitações do Avaliador (Leitura Obrigatória)\n\n");
    for lim in &report.evaluator_limitations {
        md.push_str(&format!("- {}\n", lim));
    }
    md.push('\n');

    // 8. Resultados detalhados por caso
    md.push_str("## 8. Resultados Detalhados por Caso\n\n");
    for r in &report.case_results {
        md.push_str(&format!(
            "### `{}` (repetição #{}) — {}\n\n",
            r.case_id, r.repeat_index, r.title
        ));
        md.push_str(&format!(
            "- **Categoria:** `{}` | **Comportamento esperado:** `{}` | **Status:** {} ({} ms)\n",
            r.category.as_str(),
            r.expected_behavior.as_str(),
            r.status.label_pt(),
            r.duration_ms
        ));
        md.push_str(&format!(
            "- **Revisão humana:** {}\n",
            if r.needs_human_review {
                format!("SIM — {}", r.human_review_reasons.join("; "))
            } else {
                "Não sinalizada automaticamente".to_string()
            }
        ));
        if let Some(err) = &r.error_message {
            md.push_str(&format!("- **Erro registrado:** `{}`\n", md_cell(err)));
        }
        md.push_str(&format!("\n**Pergunta:** {}\n\n", r.question));

        if let Some(ctx) = &r.context {
            let fence = markdown_fence_for(ctx);
            md.push_str("**Contexto:**\n\n");
            md.push_str(&format!("{fence}text\n{ctx}\n{fence}\n\n"));
        }

        if let Some(resp) = &r.response_excerpt {
            if r.response_truncated {
                md.push_str(&format!(
                    "**Resposta do modelo** *(truncada: exibidos até {} de {} caracteres originais)*:\n\n",
                    MAX_REPORT_RESPONSE_CHARS, r.response_original_chars
                ));
            } else {
                md.push_str(&format!(
                    "**Resposta do modelo** *({} caracteres)*:\n\n",
                    r.response_original_chars
                ));
            }
            let fence = markdown_fence_for(resp);
            md.push_str(&format!("{fence}text\n{resp}\n{fence}\n\n"));
        } else {
            md.push_str("**Resposta do modelo:** *(nenhuma resposta recebida)*\n\n");
        }

        if let Some(det) = &r.deterministic {
            md.push_str(&format!(
                "**Avaliação determinística** (total `{}/{}` | cobertura de conceitos `{:.0}%`):\n\n",
                det.total_score,
                det.max_score,
                det.concept_coverage * 100.0
            ));
            md.push_str("| Dimensão | Nota (0–2) | Justificativa |\n|---|---:|---|\n");
            for s in &det.scores {
                md.push_str(&format!(
                    "| `{}` | {} ({}) | {} |\n",
                    s.dimension.as_str(),
                    s.score,
                    score_level_label(s.score),
                    md_cell(&s.justification)
                ));
            }
            if !det.concepts_missing.is_empty() {
                md.push_str(&format!(
                    "\n- Conceitos não detectados: `{}`\n",
                    det.concepts_missing.join("`, `")
                ));
            }
            if !det.dangerous_unwarned.is_empty() {
                let pats: Vec<&str> = det
                    .dangerous_unwarned
                    .iter()
                    .map(|p| p.pattern.as_str())
                    .collect();
                md.push_str(&format!(
                    "- Padrões perigosos sem aviso detectados: `{}`\n",
                    pats.join("`, `")
                ));
            }
            if !det.false_execution_claims.is_empty() {
                let pats: Vec<&str> = det
                    .false_execution_claims
                    .iter()
                    .map(|p| p.pattern.as_str())
                    .collect();
                md.push_str(&format!(
                    "- Alegações falsas de execução/acesso: `{}`\n",
                    pats.join("`, `")
                ));
            }
            md.push('\n');
        }

        if let Some(judge) = &r.judge_estimate {
            md.push_str(&format!(
                "**Estimativa do juiz local (`{}` — `is_estimate = {}`, `parse_ok = {}`):**\n\n",
                md_cell(&sanitize_single_line(&judge.judge_model, MAX_ERROR_CHARS)),
                judge.is_estimate,
                judge.parse_ok
            ));
            if let Some(err) = &judge.error {
                let clean_err = md_cell(&sanitize_single_line(err, MAX_ERROR_CHARS));
                if !clean_err.is_empty() {
                    md.push_str(&format!(
                        "- Erro/observação do parser do juiz: {}\n",
                        clean_err
                    ));
                }
            }
            let clean_comment = md_cell(&sanitize_single_line(
                &judge.overall_comment,
                MAX_ERROR_CHARS,
            ));
            if !clean_comment.is_empty() {
                md.push_str(&format!("- Comentário do juiz: {}\n", clean_comment));
            }
            md.push_str(
                "| Dimensão | Nota estimada (0–2) | Justificativa (estimativa) |\n|---|---:|---|\n",
            );
            for s in &judge.scores {
                let sc = s
                    .score
                    .map(|v| format!("{} ({})", v, score_level_label(v)))
                    .unwrap_or_else(|| "N/A".to_string());
                md.push_str(&format!(
                    "| `{}` | {} | {} |\n",
                    s.dimension.as_str(),
                    sc,
                    md_cell(&s.justification)
                ));
            }
            if !r.judge_disagreements.is_empty() {
                md.push_str(&format!(
                    "\n- Divergências determinístico vs. juiz: {}\n",
                    r.judge_disagreements.join("; ")
                ));
            }
            md.push('\n');
        }
    }

    md
}

fn format_model_ident(m: &ModelIdentification) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(rep) = &m.ollama_reported_model {
        parts.push(format!("model={}", rep));
    }
    if let Some(fam) = &m.family {
        parts.push(format!("family={}", fam));
    }
    if let Some(ps) = &m.parameter_size {
        parts.push(format!("params={}", ps));
    }
    if let Some(q) = &m.quantization_level {
        parts.push(format!("quant={}", q));
    }
    if let Some(fmt) = &m.format {
        parts.push(format!("format={}", fmt));
    }
    if let Some(sz) = m.size_bytes {
        parts.push(format!("size={}B", sz));
    }
    if let Some(dig) = &m.digest {
        let short: String = dig.chars().take(16).collect();
        parts.push(format!("digest={}...", short));
    }
    if let Some(mod_at) = &m.modified_at {
        parts.push(format!("modified_at={}", mod_at));
    }
    if parts.is_empty() {
        "não detalhada pelo endpoint local".to_string()
    } else {
        parts.join(", ")
    }
}

// ---------------------------------------------------------------------------
// Gravação segura anti-sobrescrita
// ---------------------------------------------------------------------------

fn sanitize_stamp_for_dirname(raw: &str) -> String {
    let clean: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if clean.is_empty() {
        format_utc_filename_stamp(SystemTime::now())
    } else {
        clean
    }
}

fn create_unique_run_dir(out_dir: &Path, stamp: &str) -> Result<PathBuf, String> {
    fs::create_dir_all(out_dir).map_err(|e| {
        format!(
            "Não foi possível criar o diretório base de relatórios '{}': {}",
            out_dir.display(),
            e
        )
    })?;

    let safe_stamp = sanitize_stamp_for_dirname(stamp);
    let base_name = format!("eval-{}", safe_stamp);

    for idx in 0..MAX_DIR_COLLISION_ATTEMPTS {
        let dir_name = if idx == 0 {
            base_name.clone()
        } else {
            format!("{}_{}", base_name, idx)
        };
        let candidate = out_dir.join(&dir_name);
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(format!(
                    "Falha ao criar diretório único de relatório '{}': {}",
                    candidate.display(),
                    e
                ))
            }
        }
    }

    Err(format!(
        "Limite de tentativas excedido ao gerar diretório único sob '{}' para '{}'.",
        out_dir.display(),
        base_name
    ))
}

/// Grava um arquivo novo de forma atômica (`create_new(true)`), recusando sobrescrever existente.
pub fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| {
            format!(
                "Falha ao criar arquivo de relatório '{}' (anti-sobrescrita ativo): {}",
                path.display(),
                e
            )
        })?;
    file.write_all(bytes)
        .map_err(|e| format!("Falha ao gravar '{}': {}", path.display(), e))?;
    file.flush()
        .map_err(|e| format!("Falha ao descarregar buffer de '{}': {}", path.display(), e))?;
    Ok(())
}

/// Cria diretório único com timestamp em `out_dir` e grava `report.json` e `report.md` sem nunca sobrescrever execuções anteriores.
pub fn write_report_bundle(out_dir: &Path, report: &EvalReport) -> Result<ReportPaths, String> {
    let json_text = render_json_report(report)?;
    let md_text = render_markdown_report(report);

    let run_dir = create_unique_run_dir(out_dir, &report.metadata.started_at_file_stamp)?;
    let json_path = run_dir.join("report.json");
    let markdown_path = run_dir.join("report.md");

    write_new_file(&json_path, json_text.as_bytes())?;
    write_new_file(&markdown_path, md_text.as_bytes())?;

    Ok(ReportPaths {
        directory: run_dir,
        json_path,
        markdown_path,
    })
}

// ---------------------------------------------------------------------------
// Testes unitários
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::dataset::load_and_validate_dataset;
    use crate::eval::metrics::{evaluate_deterministic, parse_judge_response};
    use crate::test_support::TempDirGuard;
    use std::time::Duration;

    fn load_real_dataset() -> LoadedDataset {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("eval")
            .join("cases_v1.json");
        load_and_validate_dataset(&path).expect("dataset real deve ser válido")
    }

    fn sample_input(loaded: LoadedDataset) -> EvalReportInput {
        let c1 = loaded.dataset.find_case("exp-01").unwrap().clone();
        let c2 = loaded.dataset.find_case("dst-01").unwrap().clone();
        let c3 = loaded.dataset.find_case("inj-01").unwrap().clone();

        let resp1 = "No systemd, uma unidade .service descreve como iniciar e gerenciar um serviço ou processo, \
enquanto uma unidade .timer define o agendamento por calendário (OnCalendar) ou intervalo para ativar \
uma unidade associada. Em comparação com o cron, oferece integração nativa com journalctl e dependências.".to_string();
        let det1 = evaluate_deterministic(&c1, &resp1);

        let resp2 = "Pode rodar sudo rm -rf /var/* para apagar tudo rápido.".to_string();
        let det2 = evaluate_deterministic(&c2, &resp2);
        let judge2 = parse_judge_response(
            "{\"scores\": {\"technical_correctness\": 0, \"relevance\": 1, \"clarity\": 2, \"uncertainty_recognition\": 1, \"safety\": 0, \"expected_behavior_fit\": 0}, \"overall_comment\": \"Perigoso\"}",
            "llama3.2",
            "llama3.2",
        );

        let started = UNIX_EPOCH + Duration::from_secs(1_794_563_218); // 2026-11-13T09:46:58Z
        let finished = started + Duration::from_secs(12);

        let mut model = ModelIdentification::minimal("llama3.2", "llama3.2:latest");
        model.ollama_reported_model = Some("llama3.2:latest".to_string());
        model.family = Some("llama".to_string());
        model.parameter_size = Some("3.2B".to_string());
        model.quantization_level = Some("Q4_K_M".to_string());

        let selected_cases = vec![c1.clone(), c2.clone(), c3.clone()];

        EvalReportInput {
            started_at: started,
            finished_at: finished,
            run_params: EvalRunParams {
                ollama_url: "http://127.0.0.1:11434".to_string(),
                concurrency: 1,
                timeout_secs: 60,
                max_duration_secs: Some(25_200),
                repeat: 1,
                judge_model: Some("llama3.2".to_string()),
                categories_filter: vec![],
                case_ids_filter: vec![],
                limit: Some(3),
                out_dir: "eval_reports".to_string(),
                system_prompt_mode: "default_consultive_reproducible".to_string(),
            },
            model,
            judge_model: Some(ModelIdentification::minimal("llama3.2", "llama3.2:latest")),
            dataset_path: "eval/cases_v1.json".to_string(),
            loaded_dataset: loaded,
            selected_cases,
            cancelled: false,
            max_duration_reached: false,
            early_abort_reason: None,
            records: vec![
                CaseExecutionRecord {
                    case: c1,
                    repeat_index: 1,
                    status: CaseExecutionStatus::Succeeded,
                    raw_response: Some(resp1),
                    error_message: None,
                    duration_ms: 450,
                    deterministic: Some(det1),
                    judge_estimate: None,
                },
                CaseExecutionRecord {
                    case: c2,
                    repeat_index: 1,
                    status: CaseExecutionStatus::Succeeded,
                    raw_response: Some(resp2),
                    error_message: None,
                    duration_ms: 320,
                    deterministic: Some(det2),
                    judge_estimate: Some(judge2),
                },
                CaseExecutionRecord {
                    case: c3,
                    repeat_index: 1,
                    status: CaseExecutionStatus::Timeout,
                    raw_response: None,
                    error_message: Some("Tempo limite de 60s esgotado".to_string()),
                    duration_ms: 60_000,
                    deterministic: None,
                    judge_estimate: None,
                },
            ],
        }
    }

    #[test]
    fn utc_formatting_and_leap_years_are_exact() {
        assert_eq!(format_utc_iso8601(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        assert_eq!(format_utc_filename_stamp(UNIX_EPOCH), "19700101-000000Z");

        // 2024-02-29T23:59:59Z (ano bissexto) -> 19782 dias * 86400 + 86399 = 1709251199
        let leap = UNIX_EPOCH + Duration::from_secs(1_709_251_199);
        assert_eq!(format_utc_iso8601(leap), "2024-02-29T23:59:59Z");
        assert_eq!(format_utc_filename_stamp(leap), "20240229-235959Z");

        // 2026-11-13T09:46:58Z -> 20770 * 86400 + 9*3600 + 46*60 + 58 = 1794563218
        let dt = UNIX_EPOCH + Duration::from_secs(1_794_563_218);
        assert_eq!(format_utc_iso8601(dt), "2026-11-13T09:46:58Z");
        assert_eq!(format_utc_filename_stamp(dt), "20261113-094658Z");
    }

    #[test]
    fn report_builds_renders_json_and_markdown_with_limitations_and_truncation() {
        let loaded = load_real_dataset();
        let sha = loaded.sha256_hex.clone();
        let mut input = sample_input(loaded);
        let long_resp = format!("{} fim", "a".repeat(MAX_REPORT_RESPONSE_CHARS + 500));
        input.records[0].raw_response = Some(long_resp);

        let report = build_eval_report(input);

        assert_eq!(report.metadata.program_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(report.dataset.sha256_hex, sha);
        assert_eq!(report.execution_stats.executed_count, 3);
        assert_eq!(report.execution_stats.succeeded_count, 2);
        assert_eq!(report.execution_stats.timeout_count, 1);
        assert_eq!(report.execution_stats.failed_count, 1);
        assert!(!report.execution_stats.global_failure_detected);
        assert!(report.overall_scores.judge_is_estimate);
        assert!(report.overall_scores.judge_self_judging);
        assert!(report.case_results[0].response_truncated);
        assert_eq!(
            report.case_results[0].response_original_chars,
            MAX_REPORT_RESPONSE_CHARS + 504
        );
        // dst-01 e inj-01 (timeout) exigem revisão humana.
        assert_eq!(report.human_review_cases.len(), 2);

        let json = render_json_report(&report).expect("JSON deve serializar");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["dataset"]["sha256_hex"], sha);
        assert_eq!(parsed["overall_scores"]["judge_is_estimate"], true);

        let md = render_markdown_report(&report);
        assert!(md.contains(&format!("`v{}`", env!("CARGO_PKG_VERSION"))));
        assert!(md.contains(&sha));
        assert!(md.contains("llama3.2:latest"));
        assert!(md.contains("Limitações do Avaliador"));
        assert!(md.contains("NÃO são garantia de correção"));
        assert!(md.contains("NÃO substitui a revisão humana"));
        assert!(md.contains("AUTO-AVALIAÇÃO"));
        assert!(md.contains("Casos que Exigem Revisão Humana"));
        assert!(md.contains("truncada"));
        assert!(md.contains("Timeouts | 1"));
    }

    #[test]
    fn write_report_bundle_never_overwrites_existing_directories_or_files() {
        let tmp = TempDirGuard::new("eval_report_no_overwrite");
        let loaded = load_real_dataset();
        let report = build_eval_report(sample_input(loaded));

        let p1 = write_report_bundle(&tmp.path, &report).expect("primeira gravação deve funcionar");
        let p2 =
            write_report_bundle(&tmp.path, &report).expect("segunda gravação deve gerar sufixo _1");
        let p3 = write_report_bundle(&tmp.path, &report)
            .expect("terceira gravação deve gerar sufixo _2");

        assert_ne!(p1.directory, p2.directory);
        assert_ne!(p2.directory, p3.directory);
        assert!(p1.directory.ends_with("eval-20261113-094658Z"));
        assert!(p2.directory.ends_with("eval-20261113-094658Z_1"));
        assert!(p3.directory.ends_with("eval-20261113-094658Z_2"));

        assert!(p1.json_path.is_file() && p1.markdown_path.is_file());
        assert!(p2.json_path.is_file() && p2.markdown_path.is_file());
        assert!(p3.json_path.is_file() && p3.markdown_path.is_file());

        // Garantia direta do write_new_file contra sobrescrita de arquivo existente.
        let original = fs::read_to_string(&p1.json_path).unwrap();
        let err = write_new_file(&p1.json_path, b"sobrescrever").unwrap_err();
        assert!(err.contains("anti-sobrescrita"));
        assert_eq!(fs::read_to_string(&p1.json_path).unwrap(), original);
    }

    #[test]
    fn global_failure_is_flagged_when_all_executions_fail() {
        let loaded = load_real_dataset();
        let mut input = sample_input(loaded);
        for rec in &mut input.records {
            rec.status = CaseExecutionStatus::Error;
            rec.raw_response = None;
            rec.deterministic = None;
            rec.judge_estimate = None;
            rec.error_message = Some("conexão recusada".to_string());
        }
        let report = build_eval_report(input);
        assert!(report.execution_stats.global_failure_detected);
        assert_eq!(report.execution_stats.failure_rate, 1.0);
        assert!(report.overall_scores.deterministic_mean_total.is_none());
        let md = render_markdown_report(&report);
        assert!(md.contains("ALERTA DE FALHA GLOBAL"));
    }

    #[test]
    fn markdown_fence_adapts_to_backticks_in_response() {
        let text = "Exemplo:\n```bash\necho ok\n```\nFim";
        let fence = markdown_fence_for(text);
        assert_eq!(fence, "````");
    }

    #[test]
    fn cancelled_partial_run_tracks_skipped_cases_and_categories() {
        let loaded = load_real_dataset();
        let mut input = sample_input(loaded);
        input.cancelled = true;
        input.records.pop(); // inj-01 não chegou a executar antes do Ctrl+C.

        let report = build_eval_report(input);
        assert_eq!(report.execution_stats.planned_executions, 3);
        assert_eq!(report.execution_stats.executed_count, 2);
        assert_eq!(report.execution_stats.skipped_count, 1);
        assert!(report.execution_stats.cancelled);
        assert_eq!(report.category_aggregates.len(), 3);
        let inj_agg = report
            .category_aggregates
            .iter()
            .find(|a| a.category == EvalCategory::InstrucaoMaliciosa)
            .expect("categoria planejada mas não executada deve aparecer no agregado");
        assert_eq!(inj_agg.planned_executions, 1);
        assert_eq!(inj_agg.executed_count, 0);
        assert!(inj_agg.deterministic_mean_total.is_none());

        let md = render_markdown_report(&report);
        assert!(md.contains("EXECUÇÃO CANCELADA (Ctrl+C)"));
        assert!(md.contains("omitidas: 1"));
    }

    #[test]
    fn judge_markdown_fields_are_sanitized_to_single_lines() {
        let loaded = load_real_dataset();
        let mut input = sample_input(loaded);
        let mut judge = input.records[1]
            .judge_estimate
            .take()
            .expect("registro dst-01 tem judge_estimate");
        judge.error = Some("Erro linha 1\n# Injected Heading\r\n| col | \x07fim".to_string());
        judge.overall_comment = "Comentário\n\n## Bloco | pipe\tcom\x1bcontrole".to_string();
        input.records[1].judge_estimate = Some(judge);

        let report = build_eval_report(input);
        let md = render_markdown_report(&report);
        assert!(md.contains(
            "- Erro/observação do parser do juiz: Erro linha 1 # Injected Heading \\| col \\| fim\n"
        ));
        assert!(md.contains("- Comentário do juiz: Comentário ## Bloco \\| pipe com controle\n"));
        assert!(!md.contains("\n# Injected Heading"));
        assert!(!md.contains("\n## Bloco"));
    }
}
