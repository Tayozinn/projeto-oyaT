//! Fase 3 (`C1`–`C4`) — Interface `ModelBackend` trocável e adaptadores locais/remotos:
//! - Locais (offline, zero vazamento externo): [`OllamaBackend`] (só loopback) e [`NeuralBackend`]
//! - Remotos (ativação explícita, chave apenas em variável de ambiente, `max_tokens`, `timeout`,
//!   limite de histórico e redação prévia de segredos `C3`): [`ClaudeBackend`] e [`OpenAiCompatBackend`]
//!   (com presets prontos para **Google Gemini AI Studio**, **Groq**, **OpenRouter** e **OpenAI**).

#[cfg(test)]
use crate::chat::config::AppConfig;
use crate::chat::models::{Message, Role};
use crate::chat::ollama::{InstalledModelInfo, OllamaChatOutcome, OllamaClient};
use crate::chat::redaction::redact_messages_for_remote;
use crate::eval::report::ModelIdentification;
use crate::eval::runner::{EvalModelClient, ModelChatResult};
use crate::neural::inference::{GenerationConfig, NeuralEngine};
use reqwest::blocking::Client;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

pub const DEFAULT_CLAUDE_BASE_URL: &str = "https://api.anthropic.com";
pub const DEFAULT_CLAUDE_MODEL: &str = "claude-sonnet-4-0";
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";

pub const DEFAULT_REMOTE_MAX_TOKENS: u32 = 1024;
pub const MAX_REMOTE_MAX_TOKENS_CAP: u32 = 8192;
pub const DEFAULT_REMOTE_TIMEOUT_SECS: u64 = 60;
pub const DEFAULT_REMOTE_MAX_HISTORY_MESSAGES: usize = 10;
pub const MAX_REMOTE_HISTORY_CAP: usize = 40;

/// Especificação declarativa de ferramenta pronta para a Fase 5 (`Tool` + loop do agente).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Chamada estruturada de ferramenta retornada pelo modelo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Saída padronizada de [`ModelBackend::generate`] (`C1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendResponse {
    Text(String),
    ToolCalls {
        assistant_text: Option<String>,
        calls: Vec<ToolCall>,
    },
}

impl BackendResponse {
    pub fn into_display_text(self) -> String {
        match self {
            Self::Text(t) => t,
            Self::ToolCalls {
                assistant_text,
                calls,
            } => {
                let mut out = assistant_text.unwrap_or_default();
                for c in calls {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&format!("[tool_call: {}({})]", c.name, c.arguments));
                }
                out
            }
        }
    }
}

/// Requisição unificada enviada a qualquer [`ModelBackend`].
pub struct BackendRequest<'a> {
    pub model: &'a str,
    pub messages: &'a [Message],
    pub tools: &'a [ToolSpec],
    pub user_prompt: &'a str,
    pub timeout_override: Option<Duration>,
}

/// Contrato único do cérebro trocável do ServerMind AI (`C1`).
pub trait ModelBackend {
    fn backend_id(&self) -> &'static str;
    fn display_name(&self) -> &str;
    fn endpoint_url(&self) -> &str;
    fn is_remote(&self) -> bool;

    fn generate(
        &self,
        req: BackendRequest<'_>,
        on_chunk: Option<&mut dyn FnMut(&str)>,
    ) -> Result<BackendResponse, String>;
}

/// Limites de custo, latência e contexto para backends remotos (`C2`).
#[derive(Debug, Clone, Copy)]
pub struct RemoteLimits {
    pub max_tokens: u32,
    pub timeout: Duration,
    pub max_history_messages: usize,
}

impl Default for RemoteLimits {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_REMOTE_MAX_TOKENS,
            timeout: Duration::from_secs(DEFAULT_REMOTE_TIMEOUT_SECS),
            max_history_messages: DEFAULT_REMOTE_MAX_HISTORY_MESSAGES,
        }
    }
}

impl RemoteLimits {
    /// Carrega limites remotos com suporte opcional a `SERVERMIND_REMOTE_MAX_TOKENS` e
    /// `SERVERMIND_REMOTE_MAX_HISTORY`, aplicando tetos rígidos contra explosão de custo (`C2`).
    pub fn from_env_with_timeout(default_timeout_secs: u64) -> Result<Self, String> {
        let max_tokens = match env::var("SERVERMIND_REMOTE_MAX_TOKENS") {
            Ok(raw) if !raw.trim().is_empty() => {
                let parsed = raw.trim().parse::<u32>().map_err(|_| {
                    format!(
                        "Valor inválido em SERVERMIND_REMOTE_MAX_TOKENS ('{}'): informe inteiro entre 1 e {}.",
                        raw.trim(),
                        MAX_REMOTE_MAX_TOKENS_CAP
                    )
                })?;
                if parsed == 0 || parsed > MAX_REMOTE_MAX_TOKENS_CAP {
                    return Err(format!(
                        "SERVERMIND_REMOTE_MAX_TOKENS ({}) fora do intervalo permitido (1..={}).",
                        parsed, MAX_REMOTE_MAX_TOKENS_CAP
                    ));
                }
                parsed
            }
            _ => DEFAULT_REMOTE_MAX_TOKENS,
        };

        let max_history_messages = match env::var("SERVERMIND_REMOTE_MAX_HISTORY") {
            Ok(raw) if !raw.trim().is_empty() => {
                let parsed = raw.trim().parse::<usize>().map_err(|_| {
                    format!(
                        "Valor inválido em SERVERMIND_REMOTE_MAX_HISTORY ('{}'): informe inteiro entre 2 e {}.",
                        raw.trim(),
                        MAX_REMOTE_HISTORY_CAP
                    )
                })?;
                if !(2..=MAX_REMOTE_HISTORY_CAP).contains(&parsed) {
                    return Err(format!(
                        "SERVERMIND_REMOTE_MAX_HISTORY ({}) fora do intervalo permitido (2..={}).",
                        parsed, MAX_REMOTE_HISTORY_CAP
                    ));
                }
                parsed
            }
            _ => DEFAULT_REMOTE_MAX_HISTORY_MESSAGES,
        };

        Ok(Self {
            max_tokens,
            timeout: Duration::from_secs(default_timeout_secs.max(1)),
            max_history_messages,
        })
    }
}

/// Valida endpoint de backend remoto (`C4`):
/// - Para hosts externos, exige obrigatoriamente `https://` sem credenciais na URL (`user:pass@`);
/// - Permite `http://` exclusivamente em loopback (`127.0.0.1`, `localhost`, `[::1]`) para mocks de teste!
pub fn validate_remote_endpoint_url(raw: &str, provider_label: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!(
            "A URL base do backend '{}' não pode ser vazia.",
            provider_label
        ));
    }
    let parsed = Url::parse(trimmed).map_err(|_| {
        format!(
            "URL inválida para o backend '{}': '{}'. Use um endpoint HTTPS válido.",
            provider_label, trimmed
        )
    })?;
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(format!(
            "A URL do backend '{}' não pode conter credenciais embutidas ('user:pass@'). Use apenas variável de ambiente para a chave.",
            provider_label
        ));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(format!(
            "A URL base do backend '{}' não deve conter query string ('?') nem fragmento ('#').",
            provider_label
        ));
    }

    match parsed.scheme() {
        "https" => Ok(trimmed.trim_end_matches('/').to_string()),
        "http" => {
            // Em HTTP sem TLS, aceita apenas loopback local (ex.: testes determinísticos com MockServer)
            let host = parsed.host_str().unwrap_or("");
            let is_loopback = host == "127.0.0.1"
                || host.eq_ignore_ascii_case("localhost")
                || host == "[::1]"
                || host == "::1";
            if is_loopback {
                Ok(trimmed.trim_end_matches('/').to_string())
            } else {
                Err(format!(
                    "Endpoint inseguro para '{}': conexões remotas exigem obrigatoriamente 'https://' (recebido '{}').",
                    provider_label, trimmed
                ))
            }
        }
        other => Err(format!(
            "Esquema '{}' não suportado para o backend '{}': use 'https://'.",
            other, provider_label
        )),
    }
}

/// Formata o banner explícito exibido sempre que um backend remoto está ativo (`C3` + `C4`).
pub fn format_remote_backend_banner(
    provider_name: &str,
    endpoint_url: &str,
    active_model: &str,
    limits: RemoteLimits,
) -> String {
    format!(
        "======================================================================\n\
         [AVISO DE BACKEND REMOTO ATIVO — DADOS ENVIADOS PELA REDE]\n\
         Provedor / Endpoint : {} ({})\n\
         Modelo remoto       : {}\n\
         Limites de custo    : max_tokens={} | histórico={} msgs | timeout={}s\n\
         Proteção ativa (C3) : Redação automática de senhas, tokens, URIs, JWTs e chaves PEM\n\
         Atenção             : Nunca envie chaves reais ou dados sigilosos sem necessidade.\n\
         Para voltar ao modo 100% local/offline, use `:backend ollama`.\n\
         ======================================================================",
        provider_name,
        endpoint_url,
        active_model,
        limits.max_tokens,
        limits.max_history_messages,
        limits.timeout.as_secs()
    )
}

/// Aplica limite de janela de histórico (`max_history_messages` mensagens não-System mais recentes)
/// preservando mensagens `Role::System` (prompt base + grounding da KB) e redige segredos (`C3`).
pub fn prepare_remote_messages(
    messages: &[Message],
    max_history_messages: usize,
) -> (Vec<Message>, usize) {
    let mut system_msgs = Vec::new();
    let mut conv_msgs = Vec::new();
    for m in messages {
        if m.role == Role::System {
            system_msgs.push(m.clone());
        } else {
            conv_msgs.push(m.clone());
        }
    }
    let cap = max_history_messages.max(1);
    let skip = conv_msgs.len().saturating_sub(cap);
    let mut trimmed = Vec::with_capacity(system_msgs.len() + conv_msgs.len().saturating_sub(skip));
    trimmed.extend(system_msgs);
    trimmed.extend(conv_msgs.into_iter().skip(skip));

    redact_messages_for_remote(&trimmed)
}

fn sanitize_remote_error(raw: &str, secret_key: &str) -> String {
    let without_key = if !secret_key.is_empty() {
        raw.replace(secret_key, "[CHAVE_OCULTADA]")
    } else {
        raw.to_string()
    };
    let collapsed: String = without_key
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let trimmed = collapsed.split_whitespace().collect::<Vec<_>>().join(" ");
    if trimmed.chars().count() <= 260 {
        trimmed
    } else {
        let short: String = trimmed.chars().take(260).collect();
        format!("{}...", short)
    }
}

// ===========================================================================
// 1. OllamaBackend (100% local / só loopback — padrão `C4`)
// ===========================================================================

pub struct OllamaBackend {
    client: OllamaClient,
}

impl OllamaBackend {
    pub fn new(client: OllamaClient) -> Self {
        Self { client }
    }
}

impl ModelBackend for OllamaBackend {
    fn backend_id(&self) -> &'static str {
        "ollama"
    }

    fn display_name(&self) -> &str {
        "Ollama Local (loopback)"
    }

    fn endpoint_url(&self) -> &str {
        self.client.base_url()
    }

    fn is_remote(&self) -> bool {
        false
    }

    fn generate(
        &self,
        req: BackendRequest<'_>,
        on_chunk: Option<&mut dyn FnMut(&str)>,
    ) -> Result<BackendResponse, String> {
        if let Some(cb) = on_chunk {
            let text = self
                .client
                .send_chat_streaming(req.model, req.messages, cb)?;
            return Ok(BackendResponse::Text(text));
        }

        match self
            .client
            .send_chat_with_timeout(req.model, req.messages, req.timeout_override)
        {
            OllamaChatOutcome::Succeeded { content, .. } => Ok(BackendResponse::Text(content)),
            OllamaChatOutcome::EmptyResponse { .. } => Err(format!(
                "O modelo '{}' no Ollama respondeu sem conteúdo de mensagem (resposta vazia).",
                req.model
            )),
            OllamaChatOutcome::Timeout { message } | OllamaChatOutcome::Error { message, .. } => {
                Err(message)
            }
        }
    }
}

// ===========================================================================
// 2. NeuralBackend (Motor próprio ServerMind Transformer em Rust — 100% local)
// ===========================================================================

pub struct NeuralBackend {
    pub engine: NeuralEngine,
}

impl NeuralBackend {
    pub fn load_from_dir(dir: &std::path::Path) -> Result<Self, String> {
        let engine = NeuralEngine::load_from_dir(dir)?;
        Ok(Self { engine })
    }
}

impl ModelBackend for NeuralBackend {
    fn backend_id(&self) -> &'static str {
        "neural"
    }

    fn display_name(&self) -> &str {
        "ServerMind Neural (local)"
    }

    fn endpoint_url(&self) -> &str {
        self.engine
            .checkpoint_dir
            .to_str()
            .unwrap_or("checkpoints/servermind-neural")
    }

    fn is_remote(&self) -> bool {
        false
    }

    fn generate(
        &self,
        req: BackendRequest<'_>,
        on_chunk: Option<&mut dyn FnMut(&str)>,
    ) -> Result<BackendResponse, String> {
        let fallback_prompt = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .unwrap_or(req.user_prompt);
        let (reply, _) = self.engine.generate_with_config(
            req.messages,
            fallback_prompt,
            GenerationConfig::default(),
            on_chunk,
        );
        Ok(BackendResponse::Text(reply))
    }
}

impl EvalModelClient for NeuralBackend {
    fn endpoint_url(&self) -> &str {
        ModelBackend::endpoint_url(self)
    }

    fn list_models(&self) -> Result<Vec<InstalledModelInfo>, String> {
        Ok(vec![InstalledModelInfo {
            name: self.model_label(),
            model: Some(self.model_label()),
            modified_at: None,
            size: Some(self.engine.weight_bytes() as u64),
            digest: None,
            details: None,
        }])
    }

    fn get_model_identification(
        &self,
        requested_model: &str,
    ) -> Result<ModelIdentification, String> {
        let resolved = self.model_label();
        Ok(ModelIdentification {
            requested_name: requested_model.to_string(),
            resolved_name: resolved.clone(),
            ollama_reported_model: Some(resolved),
            digest: None,
            size_bytes: Some(self.engine.weight_bytes() as u64),
            modified_at: None,
            format: Some(format!("smq/{}", self.engine.precision_label())),
            family: Some("servermind-transformer".to_string()),
            parameter_size: Some(format!(
                "{:.1}M",
                self.engine.model.config.total_params() as f64 / 1e6
            )),
            quantization_level: Some(self.engine.precision_label().to_string()),
        })
    }

    fn chat_with_timeout(
        &self,
        model: &str,
        messages: &[Message],
        timeout: Duration,
    ) -> ModelChatResult {
        match self.generate(
            BackendRequest {
                model,
                messages,
                tools: &[],
                user_prompt: "",
                timeout_override: Some(timeout),
            },
            None,
        ) {
            Ok(resp) => {
                let text = resp.into_display_text();
                if text.trim().is_empty() {
                    ModelChatResult::EmptyResponse {
                        reported_model: Some(self.model_label()),
                    }
                } else {
                    ModelChatResult::Succeeded {
                        content: text,
                        reported_model: Some(self.model_label()),
                    }
                }
            }
            Err(message) => ModelChatResult::Error {
                message,
                is_global_or_connectivity: false,
            },
        }
    }
}

impl NeuralBackend {
    pub fn model_label(&self) -> String {
        let dir_name = self
            .engine
            .checkpoint_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("servermind-neural");
        format!("neural:{}", dir_name)
    }
}

// ===========================================================================
// 3. ClaudeBackend (API oficial Anthropic Messages `/v1/messages` — `C2`–`C4`)
// ===========================================================================

pub struct ClaudeBackend {
    base_url: String,
    api_key: String,
    default_model: String,
    limits: RemoteLimits,
    http: Client,
}

#[derive(Serialize)]
struct AnthropicToolPayload<'a> {
    name: &'a str,
    description: &'a str,
    input_schema: &'a serde_json::Value,
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicToolPayload<'a>>,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    content: Vec<AnthropicContentBlock>,
    #[serde(default)]
    error: Option<AnthropicErrorDetail>,
}

#[derive(Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct AnthropicErrorEnvelope {
    #[serde(default)]
    error: Option<AnthropicErrorDetail>,
}

#[derive(Deserialize)]
struct AnthropicErrorDetail {
    #[serde(default)]
    message: Option<String>,
}

impl ClaudeBackend {
    pub fn from_env(model_override: Option<&str>, limits: RemoteLimits) -> Result<Self, String> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                "Variável de ambiente 'ANTHROPIC_API_KEY' não definida ou vazia.\n\
                 Por segurança (C2/C4), chaves de API remotas nunca ficam no config.toml nem no SQLite.\n\
                 Como resolver: defina `$env:ANTHROPIC_API_KEY = \"sk-ant-...\"` na sessão atual do terminal."
                    .to_string()
            })?;

        let raw_base = env::var("SERVERMIND_ANTHROPIC_BASE_URL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_CLAUDE_BASE_URL.to_string());

        let default_model = model_override
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .unwrap_or(DEFAULT_CLAUDE_MODEL)
            .to_string();

        Self::new(raw_base, api_key, default_model, limits)
    }

    pub fn new(
        raw_base_url: String,
        api_key: String,
        default_model: String,
        limits: RemoteLimits,
    ) -> Result<Self, String> {
        if api_key.trim().is_empty() {
            return Err("A chave de API do Claude não pode ser vazia.".to_string());
        }
        let base_url = validate_remote_endpoint_url(&raw_base_url, "claude")?;
        let http = Client::builder()
            .timeout(limits.timeout)
            .connect_timeout(Duration::from_secs(10).min(limits.timeout))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| {
                format!(
                    "Falha ao inicializar cliente HTTPS para Claude: {}",
                    sanitize_remote_error(&e.to_string(), &api_key)
                )
            })?;

        Ok(Self {
            base_url,
            api_key,
            default_model,
            limits,
            http,
        })
    }

    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    pub fn limits(&self) -> RemoteLimits {
        self.limits
    }

    fn execute_messages_call(
        &self,
        req: BackendRequest<'_>,
    ) -> Result<(BackendResponse, Option<String>), (String, bool, bool)> {
        let effective_model = if req.model.trim().is_empty() {
            self.default_model.as_str()
        } else {
            req.model.trim()
        };

        let (redacted_msgs, _) =
            prepare_remote_messages(req.messages, self.limits.max_history_messages);

        let mut system_parts = Vec::new();
        let mut chat_msgs = Vec::new();
        for m in &redacted_msgs {
            match m.role {
                Role::System => {
                    if !m.content.trim().is_empty() {
                        system_parts.push(m.content.trim().to_string());
                    }
                }
                Role::User => chat_msgs.push(AnthropicMessage {
                    role: "user",
                    content: &m.content,
                }),
                Role::Assistant => chat_msgs.push(AnthropicMessage {
                    role: "assistant",
                    content: &m.content,
                }),
            }
        }
        if chat_msgs.is_empty() {
            return Err((
                "Nenhuma mensagem de usuário disponível para enviar ao Claude.".to_string(),
                false,
                false,
            ));
        }

        let tools_payload: Vec<AnthropicToolPayload<'_>> = req
            .tools
            .iter()
            .map(|t| AnthropicToolPayload {
                name: &t.name,
                description: &t.description,
                input_schema: &t.input_schema,
            })
            .collect();

        let payload = AnthropicRequest {
            model: effective_model,
            max_tokens: self.limits.max_tokens,
            system: if system_parts.is_empty() {
                None
            } else {
                Some(system_parts.join("\n\n"))
            },
            messages: chat_msgs,
            tools: tools_payload,
        };

        let url = format!("{}/v1/messages", self.base_url);
        let timeout = req.timeout_override.unwrap_or(self.limits.timeout);
        let resp = self
            .http
            .post(&url)
            .timeout(timeout)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("content-type", "application/json")
            .json(&payload)
            .send()
            .map_err(|e| {
                let is_timeout = e.is_timeout();
                let safe = sanitize_remote_error(&e.to_string(), &self.api_key);
                (
                    format!(
                        "Falha ao contatar API Anthropic ('{}'): {}",
                        self.base_url, safe
                    ),
                    is_timeout,
                    true,
                )
            })?;

        let status = resp.status();
        if !status.is_success() {
            let detail = resp
                .json::<AnthropicErrorEnvelope>()
                .ok()
                .and_then(|env| env.error)
                .and_then(|d| d.message)
                .map(|m| sanitize_remote_error(&m, &self.api_key))
                .unwrap_or_else(|| "sem detalhe adicional da API".to_string());
            let is_global =
                status.is_server_error() || status.as_u16() == 401 || status.as_u16() == 403;
            return Err((
                format!(
                    "A API Anthropic retornou status HTTP {} para o modelo '{}': {}",
                    status, effective_model, detail
                ),
                false,
                is_global,
            ));
        }

        let parsed: AnthropicResponse = resp.json().map_err(|e| {
            (
                format!(
                    "Resposta JSON inválida da API Anthropic: {}",
                    sanitize_remote_error(&e.to_string(), &self.api_key)
                ),
                false,
                true,
            )
        })?;

        if let Some(err) = parsed.error.and_then(|d| d.message) {
            return Err((
                format!(
                    "Erro reportado pela API Anthropic: {}",
                    sanitize_remote_error(&err, &self.api_key)
                ),
                false,
                false,
            ));
        }

        let mut texts = Vec::new();
        let mut calls = Vec::new();
        for block in parsed.content {
            match block.kind.as_str() {
                "text" => {
                    if let Some(t) = block.text.filter(|s| !s.is_empty()) {
                        texts.push(t);
                    }
                }
                "tool_use" => {
                    if let Some(name) = block.name.filter(|s| !s.trim().is_empty()) {
                        calls.push(ToolCall {
                            id: block.id.unwrap_or_else(|| "tool_call_1".to_string()),
                            name,
                            arguments: block.input.unwrap_or_else(|| serde_json::json!({})),
                        });
                    }
                }
                _ => {}
            }
        }

        let joined_text = texts.join("\n").trim().to_string();
        let reported_model = parsed.model.filter(|m| !m.trim().is_empty());

        if !calls.is_empty() {
            Ok((
                BackendResponse::ToolCalls {
                    assistant_text: if joined_text.is_empty() {
                        None
                    } else {
                        Some(joined_text)
                    },
                    calls,
                },
                reported_model,
            ))
        } else {
            Ok((BackendResponse::Text(joined_text), reported_model))
        }
    }
}

impl ModelBackend for ClaudeBackend {
    fn backend_id(&self) -> &'static str {
        "claude"
    }

    fn display_name(&self) -> &str {
        "Anthropic Claude API (remoto)"
    }

    fn endpoint_url(&self) -> &str {
        &self.base_url
    }

    fn is_remote(&self) -> bool {
        true
    }

    fn generate(
        &self,
        req: BackendRequest<'_>,
        on_chunk: Option<&mut dyn FnMut(&str)>,
    ) -> Result<BackendResponse, String> {
        let (resp, _) = self.execute_messages_call(req).map_err(|(m, _, _)| m)?;
        if let BackendResponse::Text(ref t) = resp {
            if t.trim().is_empty() {
                return Err("A API Anthropic retornou resposta vazia.".to_string());
            }
            if let Some(cb) = on_chunk {
                cb(t);
            }
        }
        Ok(resp)
    }
}

impl EvalModelClient for ClaudeBackend {
    fn endpoint_url(&self) -> &str {
        &self.base_url
    }

    fn list_models(&self) -> Result<Vec<InstalledModelInfo>, String> {
        Ok(vec![InstalledModelInfo {
            name: self.default_model.clone(),
            model: Some(self.default_model.clone()),
            modified_at: None,
            size: None,
            digest: None,
            details: None,
        }])
    }

    fn get_model_identification(
        &self,
        requested_model: &str,
    ) -> Result<ModelIdentification, String> {
        let resolved = if requested_model.trim().is_empty() {
            self.default_model.clone()
        } else {
            requested_model.trim().to_string()
        };
        Ok(ModelIdentification {
            requested_name: resolved.clone(),
            resolved_name: resolved.clone(),
            ollama_reported_model: Some(resolved),
            digest: None,
            size_bytes: None,
            modified_at: None,
            format: Some("remote-api:anthropic".to_string()),
            family: Some("claude".to_string()),
            parameter_size: None,
            quantization_level: None,
        })
    }

    fn chat_with_timeout(
        &self,
        model: &str,
        messages: &[Message],
        timeout: Duration,
    ) -> ModelChatResult {
        match self.execute_messages_call(BackendRequest {
            model,
            messages,
            tools: &[],
            user_prompt: "",
            timeout_override: Some(timeout),
        }) {
            Ok((resp, reported_model)) => {
                let content = resp.into_display_text();
                if content.trim().is_empty() {
                    ModelChatResult::EmptyResponse { reported_model }
                } else {
                    ModelChatResult::Succeeded {
                        content,
                        reported_model,
                    }
                }
            }
            Err((message, true, _)) => ModelChatResult::Timeout { message },
            Err((message, false, is_global)) => ModelChatResult::Error {
                message,
                is_global_or_connectivity: is_global,
            },
        }
    }
}

// ===========================================================================
// 4. OpenAiCompatBackend (Gemini AI Studio / Groq / Mistral / OpenRouter / OpenAI)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiProviderPreset {
    Gemini,
    Groq,
    Mistral,
    OpenRouter,
    OpenAi,
}

impl OpenAiProviderPreset {
    pub fn backend_id(self) -> &'static str {
        match self {
            Self::Gemini => "gemini",
            Self::Groq => "groq",
            Self::Mistral => "mistral",
            Self::OpenRouter => "openrouter",
            Self::OpenAi => "openai",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Gemini => "Google Gemini AI Studio (OpenAI-compat remoto)",
            Self::Groq => "Groq Cloud API (OpenAI-compat remoto)",
            Self::Mistral => "Mistral La Plateforme API (OpenAI-compat remoto)",
            Self::OpenRouter => "OpenRouter API (OpenAI-compat remoto)",
            Self::OpenAi => "OpenAI-Compatible API (remoto)",
        }
    }

    pub fn default_base_url(self) -> &'static str {
        match self {
            Self::Gemini => "https://generativelanguage.googleapis.com/v1beta/openai",
            Self::Groq => "https://api.groq.com/openai/v1",
            Self::Mistral => "https://api.mistral.ai/v1",
            Self::OpenRouter => "https://openrouter.ai/api/v1",
            Self::OpenAi => "https://api.openai.com/v1",
        }
    }

    pub fn default_model(self) -> &'static str {
        match self {
            Self::Gemini => "gemini-flash-latest",
            Self::Groq => "openai/gpt-oss-120b",
            Self::Mistral => "mistral-small-latest",
            Self::OpenRouter => "openrouter/free",
            Self::OpenAi => "gpt-4o-mini",
        }
    }

    pub fn env_key_candidates(self) -> &'static [&'static str] {
        match self {
            Self::Gemini => &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            Self::Groq => &["GROQ_API_KEY"],
            Self::Mistral => &["MISTRAL_API_KEY"],
            Self::OpenRouter => &["OPENROUTER_API_KEY"],
            Self::OpenAi => &["SERVERMIND_OPENAI_API_KEY", "OPENAI_API_KEY"],
        }
    }

    pub fn base_url_env_var(self) -> &'static str {
        match self {
            Self::Gemini => "SERVERMIND_GEMINI_BASE_URL",
            Self::Groq => "SERVERMIND_GROQ_BASE_URL",
            Self::Mistral => "SERVERMIND_MISTRAL_BASE_URL",
            Self::OpenRouter => "SERVERMIND_OPENROUTER_BASE_URL",
            Self::OpenAi => "SERVERMIND_OPENAI_BASE_URL",
        }
    }
}

/// Catálogo curado de presets gratuitos oficiais prontos para uso sem cartão de crédito
/// via chave de API em variável de ambiente (`:backend presets` / `--backend <alias>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeBackendPresetInfo {
    pub alias: &'static str,
    pub provider: OpenAiProviderPreset,
    pub model: &'static str,
    pub env_var: &'static str,
    pub signup_hint: &'static str,
    pub summary: &'static str,
}

pub const FREE_BACKEND_PRESETS: &[FreeBackendPresetInfo] = &[
    FreeBackendPresetInfo {
        alias: "free-gemini",
        provider: OpenAiProviderPreset::Gemini,
        model: "gemini-flash-latest",
        env_var: "GEMINI_API_KEY",
        signup_hint: "aistudio.google.com/apikey",
        summary: "Google AI Studio Flash mais recente (ótimo equilíbrio e janela ampla)",
    },
    FreeBackendPresetInfo {
        alias: "free-groq",
        provider: OpenAiProviderPreset::Groq,
        model: "openai/gpt-oss-120b",
        env_var: "GROQ_API_KEY",
        signup_hint: "console.groq.com/keys",
        summary: "Groq GPT-OSS 120B em baixa latência (raciocínio forte para infra e código)",
    },
    FreeBackendPresetInfo {
        alias: "free-groq-qwen",
        provider: OpenAiProviderPreset::Groq,
        model: "qwen/qwen3.8-27b",
        env_var: "GROQ_API_KEY",
        signup_hint: "console.groq.com/keys",
        summary: "Groq Qwen 3.8 27B (excelente aderência a instruções técnicas e scripts)",
    },
    FreeBackendPresetInfo {
        alias: "free-mistral",
        provider: OpenAiProviderPreset::Mistral,
        model: "mistral-small-latest",
        env_var: "MISTRAL_API_KEY",
        signup_hint: "console.mistral.ai/api-keys",
        summary: "Mistral Free Mode (sem cartão; rápido para diagnóstico Linux e YAML/Bash)",
    },
    FreeBackendPresetInfo {
        alias: "free-openrouter",
        provider: OpenAiProviderPreset::OpenRouter,
        model: "openrouter/free",
        env_var: "OPENROUTER_API_KEY",
        signup_hint: "openrouter.ai/keys",
        summary:
            "Auto-router gratuito do OpenRouter (escolhe automaticamente um modelo :free ativo)",
    },
    FreeBackendPresetInfo {
        alias: "free-nemotron",
        provider: OpenAiProviderPreset::OpenRouter,
        model: "nvidia/nemotron-3-ultra:free",
        env_var: "OPENROUTER_API_KEY",
        signup_hint: "openrouter.ai/keys",
        summary: "NVIDIA Nemotron 3 Ultra :free via OpenRouter (contexto longo para logs extensos)",
    },
];

pub fn resolve_free_preset(alias: &str) -> Option<&'static FreeBackendPresetInfo> {
    let normalized = alias.trim().to_ascii_lowercase();
    FREE_BACKEND_PRESETS.iter().find(|p| p.alias == normalized)
}

pub fn format_free_presets_catalog() -> String {
    let mut out = String::from(
        "Presets gratuitos prontos (sem salvar chave em disco; exigem apenas variável de ambiente na sessão):\n",
    );
    for p in FREE_BACKEND_PRESETS {
        let key_ready = env::var(p.env_var)
            .ok()
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false);
        let status = if key_ready {
            "chave detectada"
        } else {
            "definir env"
        };
        out.push_str(&format!(
            "  - {:<16} -> {:<10} | modelo='{:<30}' | ${:<18} ({}) | {}\n      Obter chave gratuita: https://{}\n",
            p.alias,
            p.provider.backend_id(),
            p.model,
            p.env_var,
            status,
            p.summary,
            p.signup_hint
        ));
    }
    out.push_str(
        "Ativação rápida no REPL : `:backend free-gemini` (ou qualquer alias acima)\n\
         Ativação rápida no eval : `servermind-ai eval run --backend free-groq --limit 6`",
    );
    out
}

pub struct OpenAiCompatBackend {
    preset: OpenAiProviderPreset,
    base_url: String,
    api_key: String,
    default_model: String,
    limits: RemoteLimits,
    http: Client,
}

#[derive(Serialize)]
struct OpenAiChatMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Serialize)]
struct OpenAiFunctionDef<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a serde_json::Value,
}

#[derive(Serialize)]
struct OpenAiToolDef<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunctionDef<'a>,
}

#[derive(Serialize)]
struct OpenAiChatRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<OpenAiChatMessage<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAiToolDef<'a>>,
}

#[derive(Deserialize)]
struct OpenAiChatResponse {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    error: Option<OpenAiErrorDetail>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiRespMessage,
}

#[derive(Deserialize)]
struct OpenAiRespMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAiRespToolCall>,
}

#[derive(Deserialize)]
struct OpenAiRespToolCall {
    #[serde(default)]
    id: Option<String>,
    function: OpenAiRespFunction,
}

#[derive(Deserialize)]
struct OpenAiRespFunction {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize)]
struct OpenAiErrorEnvelope {
    #[serde(default)]
    error: Option<OpenAiErrorDetail>,
}

#[derive(Deserialize)]
struct OpenAiErrorDetail {
    #[serde(default)]
    message: Option<String>,
}

impl OpenAiCompatBackend {
    pub fn from_env(
        preset: OpenAiProviderPreset,
        model_override: Option<&str>,
        limits: RemoteLimits,
    ) -> Result<Self, String> {
        let env_keys = preset.env_key_candidates();
        let api_key = env_keys
            .iter()
            .find_map(|&k| {
                env::var(k)
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
            })
            .ok_or_else(|| {
                let keys_joined = env_keys.join(" ou ");
                format!(
                    "Variável de ambiente '{}' não definida ou vazia para o backend '{}'.\n\
                     Por segurança (C2/C4), chaves de API remotas nunca ficam no config.toml nem no SQLite.\n\
                     Como resolver: defina `$env:{} = \"sua_chave\"` na sessão atual do terminal.",
                    keys_joined,
                    preset.backend_id(),
                    env_keys[0]
                )
            })?;

        let raw_base = env::var(preset.base_url_env_var())
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| preset.default_base_url().to_string());

        let default_model = model_override
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| preset.default_model())
            .to_string();

        Self::new(preset, raw_base, api_key, default_model, limits)
    }

    pub fn new(
        preset: OpenAiProviderPreset,
        raw_base_url: String,
        api_key: String,
        default_model: String,
        limits: RemoteLimits,
    ) -> Result<Self, String> {
        if api_key.trim().is_empty() {
            return Err("A chave de API não pode ser vazia.".to_string());
        }
        let base_url = validate_remote_endpoint_url(&raw_base_url, preset.backend_id())?;
        let http = Client::builder()
            .timeout(limits.timeout)
            .connect_timeout(Duration::from_secs(10).min(limits.timeout))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| {
                format!(
                    "Falha ao inicializar cliente HTTPS para '{}': {}",
                    preset.backend_id(),
                    sanitize_remote_error(&e.to_string(), &api_key)
                )
            })?;

        Ok(Self {
            preset,
            base_url,
            api_key,
            default_model,
            limits,
            http,
        })
    }

    pub fn preset(&self) -> OpenAiProviderPreset {
        self.preset
    }

    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    pub fn limits(&self) -> RemoteLimits {
        self.limits
    }

    fn execute_chat_completions(
        &self,
        req: BackendRequest<'_>,
    ) -> Result<(BackendResponse, Option<String>), (String, bool, bool)> {
        let effective_model = if req.model.trim().is_empty() {
            self.default_model.as_str()
        } else {
            req.model.trim()
        };

        let (redacted_msgs, _) =
            prepare_remote_messages(req.messages, self.limits.max_history_messages);

        let wire_messages: Vec<OpenAiChatMessage<'_>> = redacted_msgs
            .iter()
            .map(|m| OpenAiChatMessage {
                role: match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        if !redacted_msgs.iter().any(|m| m.role == Role::User) {
            return Err((
                "Nenhuma mensagem de usuário disponível para enviar ao backend remoto.".to_string(),
                false,
                false,
            ));
        }

        let tools_payload: Vec<OpenAiToolDef<'_>> = req
            .tools
            .iter()
            .map(|t| OpenAiToolDef {
                kind: "function",
                function: OpenAiFunctionDef {
                    name: &t.name,
                    description: &t.description,
                    parameters: &t.input_schema,
                },
            })
            .collect();

        let payload = OpenAiChatRequest {
            model: effective_model,
            max_tokens: self.limits.max_tokens,
            messages: wire_messages,
            tools: tools_payload,
        };

        let url = format!("{}/chat/completions", self.base_url);
        let timeout = req.timeout_override.unwrap_or(self.limits.timeout);
        let resp = self
            .http
            .post(&url)
            .timeout(timeout)
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&payload)
            .send()
            .map_err(|e| {
                let is_timeout = e.is_timeout();
                let safe = sanitize_remote_error(&e.to_string(), &self.api_key);
                (
                    format!(
                        "Falha ao contatar '{}' ('{}'): {}",
                        self.preset.display_name(),
                        self.base_url,
                        safe
                    ),
                    is_timeout,
                    true,
                )
            })?;

        let status = resp.status();
        if !status.is_success() {
            let detail = resp
                .json::<OpenAiErrorEnvelope>()
                .ok()
                .and_then(|env| env.error)
                .and_then(|d| d.message)
                .map(|m| sanitize_remote_error(&m, &self.api_key))
                .unwrap_or_else(|| "sem detalhe adicional da API".to_string());
            let is_global =
                status.is_server_error() || status.as_u16() == 401 || status.as_u16() == 403;
            return Err((
                format!(
                    "O provedor '{}' retornou status HTTP {} para o modelo '{}': {}",
                    self.preset.backend_id(),
                    status,
                    effective_model,
                    detail
                ),
                false,
                is_global,
            ));
        }

        let parsed: OpenAiChatResponse = resp.json().map_err(|e| {
            (
                format!(
                    "Resposta JSON inválida de '{}': {}",
                    self.preset.backend_id(),
                    sanitize_remote_error(&e.to_string(), &self.api_key)
                ),
                false,
                true,
            )
        })?;

        if let Some(err) = parsed.error.and_then(|d| d.message) {
            return Err((
                format!(
                    "Erro reportado por '{}': {}",
                    self.preset.backend_id(),
                    sanitize_remote_error(&err, &self.api_key)
                ),
                false,
                false,
            ));
        }

        let Some(choice) = parsed.choices.into_iter().next() else {
            return Ok((
                BackendResponse::Text(String::new()),
                parsed.model.filter(|m| !m.trim().is_empty()),
            ));
        };

        let reported_model = parsed.model.filter(|m| !m.trim().is_empty());
        let content_text = choice
            .message
            .content
            .unwrap_or_default()
            .trim()
            .to_string();

        if !choice.message.tool_calls.is_empty() {
            let calls = choice
                .message
                .tool_calls
                .into_iter()
                .map(|tc| {
                    let args_val =
                        serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                            .unwrap_or_else(
                                |_| serde_json::json!({ "raw": tc.function.arguments }),
                            );
                    ToolCall {
                        id: tc.id.unwrap_or_else(|| "call_1".to_string()),
                        name: tc.function.name,
                        arguments: args_val,
                    }
                })
                .collect();
            Ok((
                BackendResponse::ToolCalls {
                    assistant_text: if content_text.is_empty() {
                        None
                    } else {
                        Some(content_text)
                    },
                    calls,
                },
                reported_model,
            ))
        } else {
            Ok((BackendResponse::Text(content_text), reported_model))
        }
    }
}

impl ModelBackend for OpenAiCompatBackend {
    fn backend_id(&self) -> &'static str {
        self.preset.backend_id()
    }

    fn display_name(&self) -> &str {
        self.preset.display_name()
    }

    fn endpoint_url(&self) -> &str {
        &self.base_url
    }

    fn is_remote(&self) -> bool {
        true
    }

    fn generate(
        &self,
        req: BackendRequest<'_>,
        on_chunk: Option<&mut dyn FnMut(&str)>,
    ) -> Result<BackendResponse, String> {
        let (resp, _) = self.execute_chat_completions(req).map_err(|(m, _, _)| m)?;
        if let BackendResponse::Text(ref t) = resp {
            if t.trim().is_empty() {
                return Err(format!(
                    "O provedor '{}' retornou resposta vazia.",
                    self.preset.backend_id()
                ));
            }
            if let Some(cb) = on_chunk {
                cb(t);
            }
        }
        Ok(resp)
    }
}

impl EvalModelClient for OpenAiCompatBackend {
    fn endpoint_url(&self) -> &str {
        &self.base_url
    }

    fn list_models(&self) -> Result<Vec<InstalledModelInfo>, String> {
        Ok(vec![InstalledModelInfo {
            name: self.default_model.clone(),
            model: Some(self.default_model.clone()),
            modified_at: None,
            size: None,
            digest: None,
            details: None,
        }])
    }

    fn get_model_identification(
        &self,
        requested_model: &str,
    ) -> Result<ModelIdentification, String> {
        let resolved = if requested_model.trim().is_empty() {
            self.default_model.clone()
        } else {
            requested_model.trim().to_string()
        };
        Ok(ModelIdentification {
            requested_name: resolved.clone(),
            resolved_name: resolved.clone(),
            ollama_reported_model: Some(resolved),
            digest: None,
            size_bytes: None,
            modified_at: None,
            format: Some(format!("remote-api:{}", self.preset.backend_id())),
            family: Some(self.preset.backend_id().to_string()),
            parameter_size: None,
            quantization_level: None,
        })
    }

    fn chat_with_timeout(
        &self,
        model: &str,
        messages: &[Message],
        timeout: Duration,
    ) -> ModelChatResult {
        match self.execute_chat_completions(BackendRequest {
            model,
            messages,
            tools: &[],
            user_prompt: "",
            timeout_override: Some(timeout),
        }) {
            Ok((resp, reported_model)) => {
                let content = resp.into_display_text();
                if content.trim().is_empty() {
                    ModelChatResult::EmptyResponse { reported_model }
                } else {
                    ModelChatResult::Succeeded {
                        content,
                        reported_model,
                    }
                }
            }
            Err((message, true, _)) => ModelChatResult::Timeout { message },
            Err((message, false, is_global)) => ModelChatResult::Error {
                message,
                is_global_or_connectivity: is_global,
            },
        }
    }
}

// ===========================================================================
// Seleção unificada de Backend (`BackendSelection` + `ActiveBackend`)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Ollama,
    Neural,
    Claude,
    Gemini,
    Groq,
    Mistral,
    OpenRouter,
    OpenAi,
    FreePreset(&'static FreeBackendPresetInfo),
}

impl BackendKind {
    pub fn parse(raw: &str) -> Result<Self, String> {
        let lower = raw.trim().to_ascii_lowercase();
        if let Some(preset_info) = resolve_free_preset(&lower) {
            return Ok(Self::FreePreset(preset_info));
        }
        match lower.as_str() {
            "ollama" => Ok(Self::Ollama),
            "neural" | "servermind" | "servermind-neural" => Ok(Self::Neural),
            "claude" | "anthropic" => Ok(Self::Claude),
            "gemini" | "google" => Ok(Self::Gemini),
            "groq" => Ok(Self::Groq),
            "mistral" => Ok(Self::Mistral),
            "openrouter" => Ok(Self::OpenRouter),
            "openai" | "openai-compat" => Ok(Self::OpenAi),
            other => Err(format!(
                "Backend desconhecido: '{}'. Opções suportadas: 'ollama' (padrão local), 'neural' (local), 'claude', 'gemini', 'groq', 'mistral', 'openrouter', 'openai' ou presets gratuitos ('free-gemini', 'free-groq', 'free-groq-qwen', 'free-mistral', 'free-openrouter', 'free-nemotron').",
                other
            )),
        }
    }

    pub fn resolve_openai_preset_and_default_model(
        self,
    ) -> Option<(OpenAiProviderPreset, Option<&'static str>)> {
        match self {
            Self::Gemini => Some((OpenAiProviderPreset::Gemini, None)),
            Self::Groq => Some((OpenAiProviderPreset::Groq, None)),
            Self::Mistral => Some((OpenAiProviderPreset::Mistral, None)),
            Self::OpenRouter => Some((OpenAiProviderPreset::OpenRouter, None)),
            Self::OpenAi => Some((OpenAiProviderPreset::OpenAi, None)),
            Self::FreePreset(info) => Some((info.provider, Some(info.model))),
            Self::Ollama | Self::Neural | Self::Claude => None,
        }
    }
}

/// Instância ativa polimórfica de [`ModelBackend`] usada por [`crate::chat::agent::AgentSession`].
pub enum ActiveBackend {
    Ollama(OllamaBackend),
    Neural(Box<NeuralBackend>),
    Claude(ClaudeBackend),
    OpenAiCompat(OpenAiCompatBackend),
}

impl ActiveBackend {
    pub fn as_model_backend(&self) -> &dyn ModelBackend {
        match self {
            Self::Ollama(b) => b,
            Self::Neural(b) => b.as_ref(),
            Self::Claude(b) => b,
            Self::OpenAiCompat(b) => b,
        }
    }
}

/// Garante explicitamente que a regra "só loopback" permanece exclusiva do Ollama (`C4`),
/// enquanto endpoints remotos exigem HTTPS (exceto loopback em testes).
#[cfg(test)]
fn _assert_loopback_vs_remote_separation() {
    assert!(AppConfig::validate_local_url("https://api.anthropic.com").is_err());
    assert!(validate_remote_endpoint_url("https://api.anthropic.com", "claude").is_ok());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{lock_env_cwd, EnvVarGuard};
    use std::io::{Read, Write};
    use std::net::{Shutdown, TcpListener};
    use std::thread;

    fn spawn_single_shot_json_server(
        expected_status: &str,
        response_body: String,
    ) -> (String, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock remote server");
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{}", port);
        let status_owned = expected_status.to_string();

        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept request");
            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
            let mut buf = Vec::new();
            let mut tmp = [0u8; 1024];
            let header_end = loop {
                let n = stream.read(&mut tmp).unwrap();
                buf.extend_from_slice(&tmp[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let headers = String::from_utf8_lossy(&buf[..header_end]);
            let content_len = headers
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    if k.trim().eq_ignore_ascii_case("content-length") {
                        v.trim().parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            while buf.len() < header_end + content_len {
                let n = stream.read(&mut tmp).unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let raw_req = String::from_utf8_lossy(&buf).into_owned();
            let http_resp = format!(
                "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status_owned,
                response_body.len(),
                response_body
            );
            let _ = stream.write_all(http_resp.as_bytes());
            let _ = stream.flush();
            let _ = stream.shutdown(Shutdown::Write);
            raw_req
        });

        (url, handle)
    }

    #[test]
    fn test_phase3_c4_loopback_rule_applies_only_to_ollama_and_remote_requires_https() {
        // Ollama continua estritamente restrito a loopback
        assert!(AppConfig::validate_local_url("https://api.anthropic.com").is_err());
        assert!(AppConfig::validate_local_url("https://openrouter.ai/api/v1").is_err());
        assert!(AppConfig::validate_local_url("http://127.0.0.1:11434").is_ok());

        // Backends remotos aceitam HTTPS externo, mas rejeitam HTTP sem TLS fora de loopback e credenciais na URL
        assert!(validate_remote_endpoint_url("https://api.anthropic.com", "claude").is_ok());
        assert!(
            validate_remote_endpoint_url("https://openrouter.ai/api/v1/", "openrouter").is_ok()
        );
        assert!(validate_remote_endpoint_url("http://api.anthropic.com", "claude").is_err());
        assert!(
            validate_remote_endpoint_url("https://user:pass@api.anthropic.com", "claude").is_err()
        );
    }

    #[test]
    fn test_phase3_c2_c3_claude_backend_redacts_secrets_limits_history_and_parses_tool_calls() {
        let resp_json = serde_json::json!({
            "id": "msg_01",
            "model": "claude-sonnet-4-0",
            "content": [
                { "type": "text", "text": "Vou verificar o uso de disco primeiro." },
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "df_check",
                    "input": { "filesystem": "/var" }
                }
            ]
        })
        .to_string();
        let (mock_url, handle) = spawn_single_shot_json_server("200 OK", resp_json);

        let limits = RemoteLimits {
            max_tokens: 512,
            timeout: Duration::from_secs(5),
            max_history_messages: 2,
        };
        let backend = ClaudeBackend::new(
            mock_url,
            "sk-ant-chave-super-secreta-teste".to_string(),
            "claude-sonnet-4-0".to_string(),
            limits,
        )
        .unwrap();

        let messages = vec![
            Message::new(Role::System, "Sistema seguro TayoX."),
            Message::new(Role::User, "Mensagem antiga 1 que deve sair do histórico"),
            Message::new(Role::Assistant, "Resposta antiga 1"),
            Message::new(Role::User, "Turno recente 2"),
            Message::new(
                Role::User,
                "Log: postgres://app:SenhaSuperSecretaProd@10.0.0.5:5432/db falhou",
            ),
        ];
        let tools = vec![ToolSpec {
            name: "df_check".to_string(),
            description: "Lista uso de disco".to_string(),
            input_schema: serde_json::json!({ "type": "object" }),
        }];

        let out = backend
            .generate(
                BackendRequest {
                    model: "",
                    messages: &messages,
                    tools: &tools,
                    user_prompt: "",
                    timeout_override: None,
                },
                None,
            )
            .expect("chamada Claude mockada deve ter sucesso");

        let raw_http = handle.join().unwrap();
        // 1. Limite de histórico (`max_history_messages = 2`) removeu "Mensagem antiga 1"
        assert!(!raw_http.contains("Mensagem antiga 1"));
        assert!(raw_http.contains("Turno recente 2"));
        // 2. Redação prévia (`C3`) impediu o segredo do banco de sair da máquina!
        assert!(
            !raw_http.contains("SenhaSuperSecretaProd"),
            "vazou senha no payload HTTP remoto: {}",
            raw_http
        );
        assert!(raw_http.contains("[SENHA_REDIGIDA]"));
        assert!(raw_http.contains("\"max_tokens\":512"));

        // 3. `C1`: suporta retorno estruturado de `ToolCalls`
        match out {
            BackendResponse::ToolCalls {
                assistant_text,
                calls,
            } => {
                assert_eq!(
                    assistant_text.as_deref(),
                    Some("Vou verificar o uso de disco primeiro.")
                );
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "df_check");
                assert_eq!(calls[0].arguments["filesystem"], "/var");
            }
            other => panic!("esperava ToolCalls, obtido {:?}", other),
        }
    }

    #[test]
    fn test_phase3_openai_compat_presets_redaction_and_missing_env_error() {
        let _lock = lock_env_cwd();
        let _env = EnvVarGuard::capture(&[
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
        ]);
        env::remove_var("ANTHROPIC_API_KEY");
        env::remove_var("GEMINI_API_KEY");
        env::remove_var("GOOGLE_API_KEY");

        // Sem variável de ambiente definida, falha com instrução clara sem tocar em arquivos
        let err_claude = ClaudeBackend::from_env(None, RemoteLimits::default())
            .err()
            .expect("deve exigir ANTHROPIC_API_KEY");
        assert!(err_claude.contains("ANTHROPIC_API_KEY"));

        let err_gemini = OpenAiCompatBackend::from_env(
            OpenAiProviderPreset::Gemini,
            None,
            RemoteLimits::default(),
        )
        .err()
        .expect("deve exigir GEMINI_API_KEY");
        assert!(err_gemini.contains("GEMINI_API_KEY"));

        // Simula OpenRouter / Gemini respondendo em `/chat/completions` e garante redação de token
        let resp_json = serde_json::json!({
            "model": "qwen/qwen-2.5-coder-32b-instruct:free",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Analisei o erro com segurança e sem expor o token."
                }
            }]
        })
        .to_string();
        let (mock_url, handle) = spawn_single_shot_json_server("200 OK", resp_json);
        let backend = OpenAiCompatBackend::new(
            OpenAiProviderPreset::OpenRouter,
            mock_url,
            "or-key-secreta-999".to_string(),
            "qwen/qwen-2.5-coder-32b-instruct:free".to_string(),
            RemoteLimits::default(),
        )
        .unwrap();

        let msgs = vec![
            Message::new(Role::System, "Consultor Linux"),
            Message::new(
                Role::User,
                "Erro com Authorization: Bearer SegredoJwtOuBearerReal12345 no Nginx",
            ),
        ];
        let mut streamed = String::new();
        let mut cb = |chunk: &str| streamed.push_str(chunk);
        let res = backend
            .generate(
                BackendRequest {
                    model: "",
                    messages: &msgs,
                    tools: &[],
                    user_prompt: "",
                    timeout_override: None,
                },
                Some(&mut cb),
            )
            .unwrap();

        let raw_http = handle.join().unwrap();
        assert!(!raw_http.contains("SegredoJwtOuBearerReal12345"));
        assert!(raw_http.contains("[TOKEN_REDIGIDO]"));
        assert_eq!(
            res,
            BackendResponse::Text("Analisei o erro com segurança e sem expor o token.".to_string())
        );
        assert_eq!(
            streamed,
            "Analisei o erro com segurança e sem expor o token."
        );

        // Catálogo de presets gratuitos prontos deve resolver aliases e exibir instruções
        assert_eq!(FREE_BACKEND_PRESETS.len(), 6);
        for info in FREE_BACKEND_PRESETS {
            let parsed = BackendKind::parse(info.alias).expect("alias free-* deve ser reconhecido");
            let (prov, mdl) = parsed
                .resolve_openai_preset_and_default_model()
                .expect("deve mapear para OpenAiProviderPreset");
            assert_eq!(prov, info.provider);
            assert_eq!(mdl, Some(info.model));
        }
        let catalog = format_free_presets_catalog();
        assert!(catalog.contains("free-gemini"));
        assert!(catalog.contains("free-groq"));
        assert!(catalog.contains("free-mistral"));
        assert!(catalog.contains("free-openrouter"));
    }
}
