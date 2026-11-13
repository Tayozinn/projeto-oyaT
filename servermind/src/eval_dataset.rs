//! Conjunto de avaliação versionado do ServerMind AI (Etapa 1 da avaliação automatizada).
//!
//! Este módulo apenas **carrega e valida** o dataset (`eval/cases_v1.json`), calcula o
//! SHA-256 dos bytes brutos do arquivo e oferece filtros explícitos de seleção de casos.
//! Ele não faz chamadas de rede, não executa comandos, não lê nem grava em `servermind.db`
//! e usa somente `std` + `serde`/`serde_json` (já presentes no projeto).
//!
//! Convenções de casamento de texto para quem consome os critérios (ex.: `eval_metrics`):
//! `expected_concepts[].any_of`, `dangerous_patterns` e `incorrect_claims` são **substrings**
//! comparadas após [`normalize_text`] (minúsculas e sem acentos) tanto no padrão quanto na
//! resposta avaliada. A detecção de negação/aviso (ex.: "NÃO rode `rm -rf /`") é
//! responsabilidade do avaliador.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Única versão de schema aceita por este módulo.
pub const SUPPORTED_SCHEMA_VERSION: &str = "1.0";

// ---------------------------------------------------------------------------
// Categorias e comportamento esperado
// ---------------------------------------------------------------------------

/// Categorias obrigatórias do dataset (strings estáveis em `snake_case`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvalCategory {
    #[serde(rename = "explicacao_conceitos")]
    ExplicacaoConceitos,
    #[serde(rename = "diagnostico_evidencia_suficiente")]
    DiagnosticoEvidenciaSuficiente,
    #[serde(rename = "diagnostico_info_insuficiente")]
    DiagnosticoInfoInsuficiente,
    #[serde(rename = "interpretacao_logs")]
    InterpretacaoLogs,
    #[serde(rename = "sugestao_comandos_revisao")]
    SugestaoComandosRevisao,
    #[serde(rename = "prevencao_comandos_destrutivos")]
    PrevencaoComandosDestrutivos,
    #[serde(rename = "fora_de_escopo")]
    ForaDeEscopo,
    #[serde(rename = "instrucao_maliciosa")]
    InstrucaoMaliciosa,
    #[serde(rename = "admissao_incerteza")]
    AdmissaoIncerteza,
}

impl EvalCategory {
    /// Todas as categorias, na ordem canônica usada em relatórios.
    pub const ALL: [EvalCategory; 9] = [
        EvalCategory::ExplicacaoConceitos,
        EvalCategory::DiagnosticoEvidenciaSuficiente,
        EvalCategory::DiagnosticoInfoInsuficiente,
        EvalCategory::InterpretacaoLogs,
        EvalCategory::SugestaoComandosRevisao,
        EvalCategory::PrevencaoComandosDestrutivos,
        EvalCategory::ForaDeEscopo,
        EvalCategory::InstrucaoMaliciosa,
        EvalCategory::AdmissaoIncerteza,
    ];

    /// Identificador estável (o mesmo valor usado no JSON e na CLI).
    pub fn as_str(&self) -> &'static str {
        match self {
            EvalCategory::ExplicacaoConceitos => "explicacao_conceitos",
            EvalCategory::DiagnosticoEvidenciaSuficiente => "diagnostico_evidencia_suficiente",
            EvalCategory::DiagnosticoInfoInsuficiente => "diagnostico_info_insuficiente",
            EvalCategory::InterpretacaoLogs => "interpretacao_logs",
            EvalCategory::SugestaoComandosRevisao => "sugestao_comandos_revisao",
            EvalCategory::PrevencaoComandosDestrutivos => "prevencao_comandos_destrutivos",
            EvalCategory::ForaDeEscopo => "fora_de_escopo",
            EvalCategory::InstrucaoMaliciosa => "instrucao_maliciosa",
            EvalCategory::AdmissaoIncerteza => "admissao_incerteza",
        }
    }

    /// Rótulo legível em PT-BR para relatórios Markdown.
    pub fn label_pt(&self) -> &'static str {
        match self {
            EvalCategory::ExplicacaoConceitos => "Explicação de conceitos",
            EvalCategory::DiagnosticoEvidenciaSuficiente => "Diagnóstico com evidência suficiente",
            EvalCategory::DiagnosticoInfoInsuficiente => "Diagnóstico com informação insuficiente",
            EvalCategory::InterpretacaoLogs => "Interpretação de logs",
            EvalCategory::SugestaoComandosRevisao => "Sugestão de comandos para revisão",
            EvalCategory::PrevencaoComandosDestrutivos => "Prevenção de comandos destrutivos",
            EvalCategory::ForaDeEscopo => "Fora de escopo",
            EvalCategory::InstrucaoMaliciosa => "Instrução maliciosa embutida",
            EvalCategory::AdmissaoIncerteza => "Admissão de incerteza",
        }
    }

    /// Converte o identificador textual (ex.: valor de `--category`) em categoria.
    pub fn parse(value: &str) -> Result<Self, String> {
        let wanted = value.trim();
        EvalCategory::ALL
            .iter()
            .copied()
            .find(|c| c.as_str() == wanted)
            .ok_or_else(|| {
                format!(
                    "Categoria desconhecida: '{}'. Categorias válidas: {}.",
                    wanted,
                    EvalCategory::ALL
                        .iter()
                        .map(|c| c.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }
}

/// Comportamento esperado do assistente para o caso.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpectedBehavior {
    /// Responder normalmente (inclui responder ignorando instruções maliciosas embutidas).
    #[serde(rename = "responder")]
    Responder,
    /// Pedir esclarecimentos antes de concluir.
    #[serde(rename = "esclarecer")]
    Esclarecer,
    /// Declarar explicitamente incerteza/falta de dados em vez de inventar fatos.
    #[serde(rename = "admitir_incerteza")]
    AdmitirIncerteza,
    /// Recusar a ação pedida (fora do escopo consultivo ou insegura), oferecendo alternativa segura.
    #[serde(rename = "recusar")]
    Recusar,
}

impl ExpectedBehavior {
    #[cfg(test)]
    pub const ALL: [ExpectedBehavior; 4] = [
        ExpectedBehavior::Responder,
        ExpectedBehavior::Esclarecer,
        ExpectedBehavior::AdmitirIncerteza,
        ExpectedBehavior::Recusar,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            ExpectedBehavior::Responder => "responder",
            ExpectedBehavior::Esclarecer => "esclarecer",
            ExpectedBehavior::AdmitirIncerteza => "admitir_incerteza",
            ExpectedBehavior::Recusar => "recusar",
        }
    }

    #[cfg(test)]
    pub fn parse(value: &str) -> Result<Self, String> {
        let wanted = value.trim();
        ExpectedBehavior::ALL
            .iter()
            .copied()
            .find(|b| b.as_str() == wanted)
            .ok_or_else(|| format!("Comportamento esperado desconhecido: '{}'.", wanted))
    }
}

// ---------------------------------------------------------------------------
// Estruturas do dataset
// ---------------------------------------------------------------------------

/// Conceito esperado: basta que **qualquer** variante de `any_of` apareça na resposta.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedConcept {
    pub name: String,
    pub any_of: Vec<String>,
}

/// Critérios de avaliação do caso.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalCriteria {
    /// Lista legível de critérios (para revisão humana e para o juiz opcional).
    pub checklist: Vec<String>,
    /// Sinais/conceitos esperados verificáveis de forma determinística.
    pub expected_concepts: Vec<ExpectedConcept>,
    /// Tamanho mínimo (em caracteres) para considerar a resposta substantiva.
    pub min_chars: usize,
}

/// Padrões perigosos ou afirmações incorretas que devem ser penalizados.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DangerSignals {
    /// Comandos/ações perigosas que não devem ser recomendados sem aviso.
    #[serde(default)]
    pub dangerous_patterns: Vec<String>,
    /// Afirmações incorretas ou inventadas (ex.: alegar ter executado algo).
    #[serde(default)]
    pub incorrect_claims: Vec<String>,
}

/// Um caso de avaliação.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalCase {
    pub id: String,
    pub category: EvalCategory,
    pub title: String,
    pub question: String,
    #[serde(default)]
    pub context: Option<String>,
    pub criteria: EvalCriteria,
    pub dangerous_or_incorrect_signals: DangerSignals,
    pub expected_behavior: ExpectedBehavior,
    pub requires_human_review_by_default: bool,
}

/// Dataset completo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalDataset {
    pub schema_version: String,
    pub dataset_version: String,
    pub language: String,
    pub description: String,
    pub cases: Vec<EvalCase>,
}

#[cfg(test)]
impl EvalDataset {
    /// Quantidade de casos por categoria, na ordem canônica (inclui categorias com zero casos).
    pub fn case_counts_by_category(&self) -> Vec<(EvalCategory, usize)> {
        EvalCategory::ALL
            .iter()
            .map(|cat| {
                (
                    *cat,
                    self.cases.iter().filter(|c| c.category == *cat).count(),
                )
            })
            .collect()
    }

    /// Categorias obrigatórias sem nenhum caso.
    pub fn missing_categories(&self) -> Vec<EvalCategory> {
        self.case_counts_by_category()
            .into_iter()
            .filter(|(_, n)| *n == 0)
            .map(|(c, _)| c)
            .collect()
    }

    pub fn find_case(&self, id: &str) -> Option<&EvalCase> {
        self.cases.iter().find(|c| c.id == id)
    }
}

/// Dataset validado + metadados do arquivo de origem.
#[derive(Debug, Clone)]
pub struct LoadedDataset {
    pub dataset: EvalDataset,
    /// SHA-256 (hex minúsculo, 64 caracteres) dos bytes brutos do arquivo.
    pub sha256_hex: String,
    /// Tamanho do arquivo em bytes.
    pub size_bytes: u64,
}

/// Filtro explícito de seleção de casos. Vetores vazios significam "sem restrição".
#[derive(Debug, Clone, Default)]
pub struct DatasetFilter {
    pub categories: Vec<EvalCategory>,
    pub case_ids: Vec<String>,
    pub limit: Option<usize>,
}

// ---------------------------------------------------------------------------
// Carregamento e validação
// ---------------------------------------------------------------------------

/// Lê o arquivo, valida o dataset e calcula o SHA-256 dos bytes brutos.
pub fn load_and_validate_dataset(path: &Path) -> Result<LoadedDataset, String> {
    let bytes = fs::read(path).map_err(|e| {
        format!(
            "Não foi possível ler o dataset de avaliação '{}': {}",
            path.display(),
            e
        )
    })?;
    let dataset = parse_dataset(&bytes)
        .map_err(|e| format!("Dataset inválido em '{}': {}", path.display(), e))?;
    Ok(LoadedDataset {
        dataset,
        sha256_hex: compute_sha256_hex(&bytes),
        size_bytes: bytes.len() as u64,
    })
}

/// Faz o parse (UTF-8 + JSON) e valida o dataset a partir de bytes.
pub fn parse_dataset(bytes: &[u8]) -> Result<EvalDataset, String> {
    let text =
        std::str::from_utf8(bytes).map_err(|e| format!("arquivo não é UTF-8 válido: {}", e))?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("JSON inválido: {}", e))?;

    // Verifica a versão do schema antes do parse estrutural, para dar uma mensagem clara.
    match value.get("schema_version").and_then(|v| v.as_str()) {
        Some(SUPPORTED_SCHEMA_VERSION) => {}
        Some(other) => {
            return Err(format!(
                "schema_version '{}' não suportada (esperado '{}')",
                other, SUPPORTED_SCHEMA_VERSION
            ))
        }
        None => {
            return Err(format!(
                "campo 'schema_version' ausente ou não é texto (esperado '{}')",
                SUPPORTED_SCHEMA_VERSION
            ))
        }
    }

    let dataset: EvalDataset =
        serde_json::from_value(value).map_err(|e| format!("estrutura inválida: {}", e))?;
    validate_dataset(&dataset)?;
    Ok(dataset)
}

fn is_blank(s: &str) -> bool {
    s.trim().is_empty()
}

fn is_valid_case_id(id: &str) -> bool {
    let len_ok = (3..=40).contains(&id.len());
    let starts_ok = id.chars().next().map_or(false, |c| c.is_ascii_lowercase());
    let chars_ok = id
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    len_ok && starts_ok && chars_ok && !id.ends_with('-')
}

fn is_valid_semver(v: &str) -> bool {
    let parts: Vec<&str> = v.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
}

/// Valida regras de schema 1.0 (campos não vazios, IDs únicos, coerência de comportamento e
/// dados estritamente fictícios).
pub fn validate_dataset(dataset: &EvalDataset) -> Result<(), String> {
    if dataset.schema_version != SUPPORTED_SCHEMA_VERSION {
        return Err(format!(
            "schema_version '{}' não suportada (esperado '{}')",
            dataset.schema_version, SUPPORTED_SCHEMA_VERSION
        ));
    }
    if !is_valid_semver(&dataset.dataset_version) {
        return Err(format!(
            "dataset_version '{}' inválida (use MAJOR.MINOR.PATCH, ex.: 1.0.0)",
            dataset.dataset_version
        ));
    }
    if is_blank(&dataset.language) {
        return Err("campo 'language' vazio".to_string());
    }
    if is_blank(&dataset.description) {
        return Err("campo 'description' vazio".to_string());
    }
    if dataset.cases.is_empty() {
        return Err("o dataset não contém nenhum caso".to_string());
    }

    let mut seen_ids: HashSet<&str> = HashSet::new();
    for case in &dataset.cases {
        let label = if case.id.is_empty() {
            "<sem id>".to_string()
        } else {
            case.id.clone()
        };
        validate_case(case).map_err(|e| format!("caso '{}': {}", label, e))?;
        if !seen_ids.insert(case.id.as_str()) {
            return Err(format!("ID de caso duplicado: '{}'", case.id));
        }
    }
    Ok(())
}

fn validate_case(case: &EvalCase) -> Result<(), String> {
    if !is_valid_case_id(&case.id) {
        return Err(
            "id inválido (use 3-40 caracteres: minúsculas, dígitos e '-', começando por letra; ex.: 'adm-01')"
                .to_string(),
        );
    }
    if is_blank(&case.title) {
        return Err("campo 'title' vazio".to_string());
    }
    if case.title.chars().count() > 100 {
        return Err("campo 'title' excede 100 caracteres".to_string());
    }
    if is_blank(&case.question) {
        return Err("campo 'question' vazio".to_string());
    }
    match &case.context {
        Some(ctx) if is_blank(ctx) => {
            return Err(
                "campo 'context' presente mas vazio (use null quando não houver)".to_string(),
            )
        }
        None => {
            if matches!(
                case.category,
                EvalCategory::InterpretacaoLogs | EvalCategory::InstrucaoMaliciosa
            ) {
                return Err(format!(
                    "a categoria '{}' exige 'context' (log ou configuração fictícia)",
                    case.category.as_str()
                ));
            }
        }
        _ => {}
    }

    // Critérios.
    let criteria = &case.criteria;
    if criteria.checklist.is_empty() || criteria.checklist.iter().any(|c| is_blank(c)) {
        return Err(
            "'criteria.checklist' deve ter ao menos um item e nenhum item vazio".to_string(),
        );
    }
    if criteria.expected_concepts.is_empty() {
        return Err("'criteria.expected_concepts' deve ter ao menos um conceito".to_string());
    }
    let mut concept_names: HashSet<&str> = HashSet::new();
    for concept in &criteria.expected_concepts {
        if is_blank(&concept.name) {
            return Err("conceito esperado com 'name' vazio".to_string());
        }
        if !concept_names.insert(concept.name.as_str()) {
            return Err(format!("conceito esperado duplicado: '{}'", concept.name));
        }
        if concept.any_of.is_empty() || concept.any_of.iter().any(|s| is_blank(s)) {
            return Err(format!(
                "conceito '{}' deve ter 'any_of' não vazio e sem itens vazios",
                concept.name
            ));
        }
    }
    if criteria.min_chars == 0 {
        return Err("'criteria.min_chars' deve ser maior que zero".to_string());
    }

    // Sinais perigosos/incorretos.
    let signals = &case.dangerous_or_incorrect_signals;
    if signals.dangerous_patterns.iter().any(|s| is_blank(s))
        || signals.incorrect_claims.iter().any(|s| is_blank(s))
    {
        return Err("'dangerous_or_incorrect_signals' contém item vazio".to_string());
    }
    if signals.dangerous_patterns.is_empty() && signals.incorrect_claims.is_empty() {
        return Err(
            "'dangerous_or_incorrect_signals' deve ter ao menos um padrão ou afirmação incorreta"
                .to_string(),
        );
    }

    // Coerência categoria x comportamento esperado.
    let behavior_ok = match case.category {
        EvalCategory::DiagnosticoInfoInsuficiente => {
            case.expected_behavior == ExpectedBehavior::Esclarecer
        }
        EvalCategory::AdmissaoIncerteza => {
            case.expected_behavior == ExpectedBehavior::AdmitirIncerteza
        }
        EvalCategory::ForaDeEscopo => case.expected_behavior == ExpectedBehavior::Recusar,
        _ => true,
    };
    if !behavior_ok {
        return Err(format!(
            "expected_behavior '{}' incompatível com a categoria '{}'",
            case.expected_behavior.as_str(),
            case.category.as_str()
        ));
    }

    // Dados estritamente fictícios.
    let mut texts: Vec<&str> = vec![&case.title, &case.question];
    if let Some(ctx) = &case.context {
        texts.push(ctx);
    }
    texts.extend(criteria.checklist.iter().map(|s| s.as_str()));
    for text in texts {
        let markers = find_non_fictional_markers(text);
        if let Some(first) = markers.first() {
            return Err(format!(
                "possível dado não fictício ({}). Use IPs RFC 5737 (192.0.2.x, 198.51.100.x, 203.0.113.x) e domínios .example/.local",
                first
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Verificação de dados fictícios
// ---------------------------------------------------------------------------

fn ipv4_is_allowed(octets: &[u32; 4]) -> bool {
    let [a, b, c, _] = *octets;
    (a == 192 && b == 0 && c == 2)
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a == 127
        || octets == &[0, 0, 0, 0]
}

fn host_is_allowed(host: &str) -> bool {
    let host = host.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.' && c != '-');
    let host = host.to_ascii_lowercase();
    if host.is_empty() {
        return true;
    }
    const ALLOWED_SUFFIXES: [&str; 5] = [".example", ".local", ".test", ".invalid", ".localhost"];
    const ALLOWED_EXACT: [&str; 7] = [
        "localhost",
        "example",
        "example.com",
        "example.org",
        "example.net",
        "corp.local",
        "local",
    ];
    ALLOWED_EXACT.contains(&host.as_str())
        || ALLOWED_SUFFIXES.iter().any(|s| host.ends_with(s))
        || host.ends_with(".example.com")
        || host.ends_with(".example.org")
        || host.ends_with(".example.net")
}

/// Procura marcadores de dados possivelmente reais em um texto: IPv4 fora dos blocos de
/// documentação RFC 5737 (ou loopback), e-mails/URLs com domínios fora de `.example`,
/// `.local`, `.test`, `.invalid` ou `example.{com,org,net}`.
///
/// É uma heurística conservadora de higiene, não uma garantia de anonimização: a revisão
/// humana do dataset continua obrigatória (ver `eval/README.md`).
pub fn find_non_fictional_markers(text: &str) -> Vec<String> {
    let mut found = Vec::new();

    // IPv4: sequências de dígitos e pontos com exatamente quatro octetos.
    let mut run = String::new();
    let mut runs: Vec<String> = Vec::new();
    for ch in text.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_digit() || ch == '.' {
            run.push(ch);
        } else {
            if !run.is_empty() {
                runs.push(std::mem::take(&mut run));
            }
        }
    }
    for r in runs {
        let trimmed = r.trim_matches('.');
        let parts: Vec<&str> = trimmed.split('.').collect();
        if parts.len() != 4 {
            continue;
        }
        let nums: Vec<Option<u32>> = parts
            .iter()
            .map(|p| {
                if !p.is_empty() && p.len() <= 3 {
                    p.parse::<u32>().ok().filter(|n| *n <= 255)
                } else {
                    None
                }
            })
            .collect();
        if nums.iter().all(|n| n.is_some()) {
            let octets = [
                nums[0].unwrap_or(0),
                nums[1].unwrap_or(0),
                nums[2].unwrap_or(0),
                nums[3].unwrap_or(0),
            ];
            if !ipv4_is_allowed(&octets) {
                found.push(format!("IPv4 fora de RFC 5737: {}", trimmed));
            }
        }
    }

    // E-mails e URLs.
    for token in
        text.split(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == '<' || c == '>')
    {
        if let Some(pos) = token.find('@') {
            let domain = &token[pos + 1..];
            if !host_is_allowed(domain) {
                found.push(format!("e-mail com domínio não fictício: {}", token));
            }
        }
        for scheme in ["http://", "https://"] {
            if let Some(pos) = token.find(scheme) {
                let rest = &token[pos + scheme.len()..];
                let host: String = rest
                    .chars()
                    .take_while(|c| *c != '/' && *c != ':' && *c != '?' && *c != '#')
                    .collect();
                let is_ip_like = host.chars().all(|c| c.is_ascii_digit() || c == '.');
                // Hosts numéricos já foram verificados pela regra de IPv4.
                if !is_ip_like && !host_is_allowed(&host) {
                    found.push(format!("URL com domínio não fictício: {}", token));
                }
            }
        }
    }
    found
}

// ---------------------------------------------------------------------------
// Filtros
// ---------------------------------------------------------------------------

/// Seleciona casos preservando a ordem do dataset. Falha se qualquer categoria ou ID pedido
/// não existir, se `limit == 0` ou se a combinação de filtros não selecionar nenhum caso.
pub fn filter_cases<'a>(
    dataset: &'a EvalDataset,
    filter: &DatasetFilter,
) -> Result<Vec<&'a EvalCase>, String> {
    if filter.limit == Some(0) {
        return Err("O limite de casos deve ser maior que zero.".to_string());
    }
    for category in &filter.categories {
        if !dataset.cases.iter().any(|c| c.category == *category) {
            return Err(format!(
                "A categoria '{}' não possui casos neste dataset.",
                category.as_str()
            ));
        }
    }
    for id in &filter.case_ids {
        if !dataset.cases.iter().any(|c| &c.id == id) {
            return Err(format!("ID de caso inexistente no dataset: '{}'.", id));
        }
    }

    let mut selected: Vec<&'a EvalCase> = dataset
        .cases
        .iter()
        .filter(|c| filter.categories.is_empty() || filter.categories.contains(&c.category))
        .filter(|c| filter.case_ids.is_empty() || filter.case_ids.iter().any(|id| id == &c.id))
        .collect();

    if selected.is_empty() {
        return Err(
            "A combinação de filtros (categoria/ID) não selecionou nenhum caso.".to_string(),
        );
    }
    if let Some(limit) = filter.limit {
        selected.truncate(limit);
    }
    Ok(selected)
}

// ---------------------------------------------------------------------------
// Normalização de texto para casamento
// ---------------------------------------------------------------------------

/// Minúsculas e sem acentos do português (á→a, ç→c, ...), para casamento tolerante.
pub fn normalize_text(input: &str) -> String {
    input
        .chars()
        .flat_map(|c| c.to_lowercase())
        .map(|c| match c {
            'á' | 'à' | 'â' | 'ã' | 'ä' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'õ' | 'ö' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ç' => 'c',
            other => other,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), implementação pura em std
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Calcula o SHA-256 de `data` e devolve 64 caracteres hexadecimais minúsculos.
pub fn compute_sha256_hex(data: &[u8]) -> String {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Preenchimento: 0x80, zeros até 56 mod 64 e o comprimento em bits (64 bits big-endian).
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut message = Vec::with_capacity(data.len() + 72);
    message.extend_from_slice(data);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut hex = String::with_capacity(64);
    for word in state {
        for byte in word.to_be_bytes() {
            hex.push_str(&format!("{:02x}", byte));
        }
    }
    hex
}

// ---------------------------------------------------------------------------
// Testes
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::path::PathBuf;

    fn real_dataset_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("eval")
            .join("cases_v1.json")
    }

    fn case_json(id: &str, category: &str, behavior: &str) -> Value {
        json!({
            "id": id,
            "category": category,
            "title": "Título curto",
            "question": "Pergunta fictícia sobre o host srv01.corp.local?",
            "context": null,
            "criteria": {
                "checklist": ["Responde de forma clara."],
                "expected_concepts": [{ "name": "conceito", "any_of": ["alfa", "beta"] }],
                "min_chars": 50
            },
            "dangerous_or_incorrect_signals": {
                "dangerous_patterns": ["rm -rf /"],
                "incorrect_claims": []
            },
            "expected_behavior": behavior,
            "requires_human_review_by_default": false
        })
    }

    fn dataset_json(cases: Vec<Value>) -> Value {
        json!({
            "schema_version": "1.0",
            "dataset_version": "1.0.0",
            "language": "pt-BR",
            "description": "Dataset de teste fictício.",
            "cases": cases
        })
    }

    fn parse_value(value: &Value) -> Result<EvalDataset, String> {
        parse_dataset(value.to_string().as_bytes())
    }

    fn small_dataset() -> EvalDataset {
        parse_value(&dataset_json(vec![
            case_json("exp-01", "explicacao_conceitos", "responder"),
            case_json("exp-02", "explicacao_conceitos", "responder"),
            case_json("dst-01", "prevencao_comandos_destrutivos", "responder"),
            case_json("fds-01", "fora_de_escopo", "recusar"),
        ]))
        .expect("dataset pequeno deve ser válido")
    }

    // --- SHA-256 ---

    #[test]
    fn sha256_nist_empty() {
        assert_eq!(
            compute_sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_nist_abc() {
        assert_eq!(
            compute_sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_nist_two_block_message() {
        assert_eq!(
            compute_sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn sha256_padding_boundaries_are_stable() {
        // 55, 56 e 64 bytes exercitam os limites de preenchimento; só confere formato/determinismo.
        for len in [55usize, 56, 63, 64, 65, 119, 120] {
            let data = vec![b'a'; len];
            let h1 = compute_sha256_hex(&data);
            let h2 = compute_sha256_hex(&data);
            assert_eq!(h1, h2);
            assert_eq!(h1.len(), 64);
            assert!(h1
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        }
        assert_ne!(
            compute_sha256_hex(&vec![b'a'; 55]),
            compute_sha256_hex(&vec![b'a'; 56])
        );
    }

    // --- Dataset real ---

    #[test]
    fn real_dataset_loads_and_covers_all_categories() {
        let loaded = load_and_validate_dataset(&real_dataset_path())
            .expect("eval/cases_v1.json deve ser válido");
        let ds = &loaded.dataset;
        assert_eq!(ds.schema_version, "1.0");
        assert_eq!(ds.dataset_version, "1.0.0");
        assert!(
            ds.cases.len() >= 18,
            "esperado >= 18 casos, obtido {}",
            ds.cases.len()
        );
        assert!(ds.missing_categories().is_empty());
        for (cat, n) in ds.case_counts_by_category() {
            assert!(
                n >= 2,
                "categoria {} tem apenas {} caso(s)",
                cat.as_str(),
                n
            );
        }
        let ids: HashSet<&str> = ds.cases.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids.len(), ds.cases.len(), "IDs devem ser únicos");
        assert_eq!(loaded.sha256_hex.len(), 64);
        let raw = fs::read(real_dataset_path()).unwrap();
        assert_eq!(loaded.sha256_hex, compute_sha256_hex(&raw));
        assert_eq!(loaded.size_bytes, raw.len() as u64);
    }

    #[test]
    fn real_dataset_is_fictional_and_consistent() {
        let loaded = load_and_validate_dataset(&real_dataset_path()).unwrap();
        for case in &loaded.dataset.cases {
            let mut texts = vec![case.title.as_str(), case.question.as_str()];
            if let Some(ctx) = &case.context {
                texts.push(ctx.as_str());
            }
            for t in texts {
                assert!(
                    find_non_fictional_markers(t).is_empty(),
                    "caso {} contém possível dado real",
                    case.id
                );
            }
            assert!(case.criteria.min_chars > 0);
            assert!(!case.criteria.checklist.is_empty());
        }
        // Casos de log e injeção sempre trazem contexto.
        for case in &loaded.dataset.cases {
            if matches!(
                case.category,
                EvalCategory::InterpretacaoLogs | EvalCategory::InstrucaoMaliciosa
            ) {
                assert!(case.context.is_some(), "caso {} sem contexto", case.id);
            }
        }
    }

    // --- Rejeições ---

    #[test]
    fn rejects_invalid_schema_version() {
        let mut v = dataset_json(vec![case_json(
            "exp-01",
            "explicacao_conceitos",
            "responder",
        )]);
        v["schema_version"] = json!("2.0");
        let err = parse_value(&v).unwrap_err();
        assert!(err.contains("schema_version"), "{}", err);

        let mut v = dataset_json(vec![case_json(
            "exp-01",
            "explicacao_conceitos",
            "responder",
        )]);
        v.as_object_mut().unwrap().remove("schema_version");
        assert!(parse_value(&v).unwrap_err().contains("schema_version"));
    }

    #[test]
    fn rejects_duplicate_ids() {
        let v = dataset_json(vec![
            case_json("exp-01", "explicacao_conceitos", "responder"),
            case_json("exp-01", "fora_de_escopo", "recusar"),
        ]);
        let err = parse_value(&v).unwrap_err();
        assert!(err.contains("duplicado"), "{}", err);
    }

    #[test]
    fn rejects_unstable_or_empty_ids() {
        for bad in ["", "Exp-01", "exp 01", "1-exp", "ex", "exp-01-"] {
            let v = dataset_json(vec![case_json(bad, "explicacao_conceitos", "responder")]);
            assert!(
                parse_value(&v).is_err(),
                "id '{}' deveria ser rejeitado",
                bad
            );
        }
    }

    #[test]
    fn rejects_empty_fields() {
        let mutations: Vec<Box<dyn Fn(&mut Value)>> = vec![
            Box::new(|c: &mut Value| c["title"] = json!("  ")),
            Box::new(|c: &mut Value| c["question"] = json!("")),
            Box::new(|c: &mut Value| c["context"] = json!("   ")),
            Box::new(|c: &mut Value| c["criteria"]["checklist"] = json!([])),
            Box::new(|c: &mut Value| c["criteria"]["checklist"] = json!([""])),
            Box::new(|c: &mut Value| c["criteria"]["expected_concepts"] = json!([])),
            Box::new(|c: &mut Value| c["criteria"]["expected_concepts"][0]["any_of"] = json!([])),
            Box::new(|c: &mut Value| c["criteria"]["expected_concepts"][0]["name"] = json!("")),
            Box::new(|c: &mut Value| c["criteria"]["min_chars"] = json!(0)),
            Box::new(|c: &mut Value| {
                c["dangerous_or_incorrect_signals"] =
                    json!({ "dangerous_patterns": [], "incorrect_claims": [] })
            }),
            Box::new(|c: &mut Value| {
                c["dangerous_or_incorrect_signals"]["dangerous_patterns"] = json!([" "])
            }),
        ];
        for (i, mutate) in mutations.iter().enumerate() {
            let mut case = case_json("exp-01", "explicacao_conceitos", "responder");
            mutate(&mut case);
            assert!(
                parse_value(&dataset_json(vec![case])).is_err(),
                "mutação #{} deveria ser rejeitada",
                i
            );
        }
    }

    #[test]
    fn rejects_invalid_category_and_behavior() {
        let v = dataset_json(vec![case_json(
            "exp-01",
            "categoria_inexistente",
            "responder",
        )]);
        assert!(parse_value(&v).is_err());
        let v = dataset_json(vec![case_json(
            "exp-01",
            "explicacao_conceitos",
            "executar",
        )]);
        assert!(parse_value(&v).is_err());
    }

    #[test]
    fn rejects_inconsistent_behavior_for_category() {
        let v = dataset_json(vec![case_json(
            "dgi-01",
            "diagnostico_info_insuficiente",
            "responder",
        )]);
        assert!(parse_value(&v).unwrap_err().contains("incompatível"));
        let v = dataset_json(vec![case_json("inc-01", "admissao_incerteza", "responder")]);
        assert!(parse_value(&v).is_err());
        let v = dataset_json(vec![case_json("fds-01", "fora_de_escopo", "responder")]);
        assert!(parse_value(&v).is_err());
    }

    #[test]
    fn requires_context_for_logs_and_injection() {
        let v = dataset_json(vec![case_json("log-01", "interpretacao_logs", "responder")]);
        assert!(parse_value(&v).unwrap_err().contains("context"));
        let v = dataset_json(vec![case_json(
            "inj-01",
            "instrucao_maliciosa",
            "responder",
        )]);
        assert!(parse_value(&v).unwrap_err().contains("context"));
    }

    #[test]
    fn rejects_unknown_fields_and_missing_file() {
        let mut case = case_json("exp-01", "explicacao_conceitos", "responder");
        case["campo_extra"] = json!(1);
        assert!(parse_value(&dataset_json(vec![case])).is_err());

        let err = load_and_validate_dataset(Path::new("eval/nao_existe_xyz.json")).unwrap_err();
        assert!(err.contains("Não foi possível ler"));
    }

    #[test]
    fn rejects_non_fictional_data() {
        for bad in [
            "Servidor em 8.8.8.8 está lento",
            "Fale com joao@gmail.com",
            "Acesse https://www.minhaempresa.com.br/admin",
        ] {
            let mut case = case_json("exp-01", "explicacao_conceitos", "responder");
            case["question"] = json!(bad);
            let err = parse_value(&dataset_json(vec![case])).unwrap_err();
            assert!(err.contains("não fictício"), "{} -> {}", bad, err);
        }
        for good in [
            "Servidor em 192.0.2.10 e 198.51.100.7 e 203.0.113.250/24",
            "Fale com ops@corp.local ou ops@empresa.example",
            "Acesse http://app.corp.local:8080/x e http://203.0.113.50/install.sh",
            "Versão 1.18.0 do nginx, timestamp 2026-03-10 08:14:02 e [91233.412]",
        ] {
            assert!(find_non_fictional_markers(good).is_empty(), "{}", good);
        }
    }

    // --- Filtros ---

    #[test]
    fn filter_by_category_and_id_and_limit() {
        let ds = small_dataset();

        let all = filter_cases(&ds, &DatasetFilter::default()).unwrap();
        assert_eq!(all.len(), 4);

        let by_cat = filter_cases(
            &ds,
            &DatasetFilter {
                categories: vec![EvalCategory::ExplicacaoConceitos],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            by_cat.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
            ["exp-01", "exp-02"]
        );

        let by_id = filter_cases(
            &ds,
            &DatasetFilter {
                case_ids: vec!["fds-01".to_string(), "exp-02".to_string()],
                ..Default::default()
            },
        )
        .unwrap();
        // Preserva a ordem do dataset, não a ordem do filtro.
        assert_eq!(
            by_id.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
            ["exp-02", "fds-01"]
        );

        let limited = filter_cases(
            &ds,
            &DatasetFilter {
                categories: vec![EvalCategory::ExplicacaoConceitos],
                limit: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(limited.len(), 1);
        assert_eq!(limited[0].id, "exp-01");

        let big_limit = filter_cases(
            &ds,
            &DatasetFilter {
                limit: Some(100),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(big_limit.len(), 4);
    }

    #[test]
    fn filter_fails_explicitly_for_unknown_or_empty_selection() {
        let ds = small_dataset();

        // Categoria válida no schema, mas sem casos neste dataset.
        let err = filter_cases(
            &ds,
            &DatasetFilter {
                categories: vec![EvalCategory::InstrucaoMaliciosa],
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("instrucao_maliciosa"));

        let err = filter_cases(
            &ds,
            &DatasetFilter {
                case_ids: vec!["zzz-99".to_string()],
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("zzz-99"));

        // Um ID válido e um inexistente: falha, não ignora silenciosamente.
        assert!(filter_cases(
            &ds,
            &DatasetFilter {
                case_ids: vec!["exp-01".to_string(), "zzz-99".to_string()],
                ..Default::default()
            },
        )
        .is_err());

        // Interseção vazia.
        let err = filter_cases(
            &ds,
            &DatasetFilter {
                categories: vec![EvalCategory::ForaDeEscopo],
                case_ids: vec!["exp-01".to_string()],
                limit: None,
            },
        )
        .unwrap_err();
        assert!(err.contains("nenhum caso"));

        assert!(filter_cases(
            &ds,
            &DatasetFilter {
                limit: Some(0),
                ..Default::default()
            }
        )
        .is_err());
    }

    // --- Utilitários ---

    #[test]
    fn category_and_behavior_parse_roundtrip() {
        for cat in EvalCategory::ALL {
            assert_eq!(EvalCategory::parse(cat.as_str()).unwrap(), cat);
            let json = serde_json::to_string(&cat).unwrap();
            assert_eq!(json, format!("\"{}\"", cat.as_str()));
        }
        for b in ExpectedBehavior::ALL {
            assert_eq!(ExpectedBehavior::parse(b.as_str()).unwrap(), b);
            assert_eq!(
                serde_json::to_string(&b).unwrap(),
                format!("\"{}\"", b.as_str())
            );
        }
        let err = EvalCategory::parse("inventada").unwrap_err();
        assert!(err.contains("explicacao_conceitos"));
        assert!(ExpectedBehavior::parse("x").is_err());
    }

    #[test]
    fn normalize_text_strips_accents_and_case() {
        assert_eq!(
            normalize_text("Força BRUTA: Não é Possível"),
            "forca bruta: nao e possivel"
        );
        assert_eq!(normalize_text("Rotação ÇÃO"), "rotacao cao");
    }
}
