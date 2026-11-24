use crate::config::AppConfig;
use crate::models::{Message, OllamaHealth};
use reqwest::blocking::Client;
use reqwest::redirect::Policy;
use reqwest::{StatusCode, Url};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::net::ToSocketAddrs;
use std::time::Duration;

const MAX_DIAGNOSTIC_CHARS: usize = 160;

/// Cliente HTTP síncrono para comunicação exclusiva com a API local do Ollama.
pub struct OllamaClient {
    base_url: String,
    timeout_secs: u64,
    client: Client,
}

#[derive(Debug, Serialize)]
struct ChatOptions {
    num_ctx: usize,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ChatOptions>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    model: Option<String>,
    message: Option<ChatMessagePayload>,
    error: Option<String>,
    #[serde(default)]
    done: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatMessagePayload {
    content: String,
}

/// Detalhes técnicos opcionais reportados pelo endpoint `/api/tags` do Ollama.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OllamaModelDetails {
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub parameter_size: Option<String>,
    #[serde(default)]
    pub quantization_level: Option<String>,
}

/// Metadados de um modelo instalado conforme retornado por `/api/tags` no Ollama local.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledModelInfo {
    pub name: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub modified_at: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub details: Option<OllamaModelDetails>,
}

/// Resultado estruturado de uma chamada a `/api/chat`, distinguindo sucesso,
/// resposta vazia, timeout e erro técnico/conectividade.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OllamaChatOutcome {
    Succeeded {
        content: String,
        reported_model: Option<String>,
    },
    EmptyResponse {
        reported_model: Option<String>,
    },
    Timeout {
        message: String,
    },
    Error {
        message: String,
        is_connectivity_error: bool,
    },
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Option<Vec<InstalledModelInfo>>,
}

#[derive(Debug, Deserialize)]
struct ErrorPayload {
    error: Option<String>,
}

impl OllamaClient {
    pub fn new(base_url: String, timeout_secs: u64) -> Result<Self, String> {
        if timeout_secs == 0 {
            return Err(
                "Tempo limite inválido (0 segundos) ao inicializar o cliente do Ollama.\n\
                 Como resolver: configure 'request_timeout_secs' em 'config.toml' ou 'SERVERMIND_TIMEOUT_SECS' com um valor inteiro positivo (ex: 120)."
                    .to_string(),
            );
        }

        AppConfig::validate_local_url(&base_url)?;
        let normalized_url = base_url.trim().trim_end_matches('/').to_string();
        Self::verify_loopback_resolution(&normalized_url)?;

        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .no_proxy()
            .redirect(Policy::none())
            .build()
            .map_err(|e| {
                format!(
                    "Falha ao inicializar o cliente HTTP local para o Ollama.\n\
                     Detalhe técnico: {}\n\
                     Como resolver: verifique as configurações locais de rede do sistema operacional.",
                    Self::sanitize_diagnostic(&e.to_string())
                )
            })?;

        Ok(Self {
            base_url: normalized_url,
            timeout_secs,
            client,
        })
    }

    /// Retorna o endereço base normalizado do endpoint local do Ollama.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Garante que o host configurado (especialmente `localhost`) resolva exclusivamente
    /// para endereços IP de loopback (`127.0.0.0/8` ou `::1`) no resolvedor do sistema operacional.
    pub fn verify_loopback_resolution(base_url: &str) -> Result<(), String> {
        let parsed = Url::parse(base_url).map_err(|e| {
            format!(
                "URL do Ollama inválida ('{}'): {}",
                base_url,
                Self::sanitize_diagnostic(&e.to_string())
            )
        })?;

        let host = parsed.host_str().unwrap_or("");
        let port = parsed.port_or_known_default().unwrap_or(11434);

        if host.eq_ignore_ascii_case("localhost") {
            let addrs: Vec<_> = ("localhost", port)
                .to_socket_addrs()
                .map_err(|e| {
                    format!(
                        "Não foi possível resolver o endereço 'localhost:{}' no sistema operacional.\n\
                         Detalhe técnico: {}\n\
                         Como resolver: use diretamente o IP de loopback 'http://127.0.0.1:{}' em 'config.toml' ou SERVERMIND_OLLAMA_URL.",
                        port,
                        Self::sanitize_diagnostic(&e.to_string()),
                        port
                    )
                })?
                .collect();

            if addrs.is_empty() {
                return Err(format!(
                    "O nome 'localhost' não retornou nenhum endereço IP local ao resolver a porta {}.\n\
                     Como resolver: configure 'ollama_url = \"http://127.0.0.1:{}\"' em 'config.toml'.",
                    port, port
                ));
            }

            if !addrs.iter().all(|addr| addr.ip().is_loopback()) {
                return Err(
                    "Política de privacidade local violada: o nome 'localhost' resolveu para um endereço IP que não é de loopback no sistema.\n\
                     Como resolver: verifique o arquivo 'hosts' do sistema ou use diretamente 'http://127.0.0.1:11434' em 'config.toml'."
                        .to_string(),
                );
            }
        }

        Ok(())
    }

    /// Consulta o endpoint `/api/tags` do Ollama local e retorna os metadados completos dos modelos instalados.
    pub fn list_installed_model_details(&self) -> Result<Vec<InstalledModelInfo>, String> {
        Self::verify_loopback_resolution(&self.base_url)?;
        let url = format!("{}/api/tags", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|err| self.format_connection_error(&err))?;

        let status = response.status();
        if status.is_redirection() {
            return Err(format!(
                "O endpoint '{}' retornou um redirecionamento HTTP ({}), que foi bloqueado por segurança.\n\
                 O SERVERMIND AI não segue redirecionamentos para proteger seus dados locais.\n\
                 Como resolver: confirme se 'ollama_url' aponta diretamente para a porta local do Ollama (ex: http://127.0.0.1:11434).",
                url, status
            ));
        }

        if !status.is_success() {
            return Err(format!(
                "O endpoint '{}' respondeu com status HTTP inesperado ({}).\n\
                 Causa provável: outro serviço pode estar ocupando esta porta ou o serviço local do Ollama não está pronto.\n\
                 Como resolver: verifique se o Ollama está em execução (`ollama serve`) e se 'ollama_url' está correta.",
                url, status
            ));
        }

        let tags: TagsResponse = response.json().map_err(|err| {
            format!(
                "O serviço em '{}' respondeu, mas não retornou um JSON válido do Ollama.\n\
                 Detalhe técnico: {}\n\
                 Como resolver: confirme se a porta configurada pertence realmente ao serviço Ollama.",
                url,
                Self::sanitize_diagnostic(&err.to_string())
            )
        })?;

        Ok(tags.models.unwrap_or_default())
    }

    /// Consulta o endpoint `/api/tags` do Ollama local e retorna a lista de nomes dos modelos instalados.
    pub fn list_installed_models(&self) -> Result<Vec<String>, String> {
        let details = self.list_installed_model_details()?;
        Ok(details.into_iter().map(|m| m.name).collect())
    }

    /// Verifica se o serviço Ollama está respondendo em `/api/tags` e se o modelo configurado está instalado.
    pub fn check_health(&self, target_model: &str) -> OllamaHealth {
        let available_models = match self.list_installed_models() {
            Ok(models) => models,
            Err(err_msg) => {
                return OllamaHealth {
                    server_reachable: false,
                    model_installed: false,
                    available_models: Vec::new(),
                    error_message: Some(err_msg),
                };
            }
        };

        if available_models.is_empty() {
            return OllamaHealth {
                server_reachable: true,
                model_installed: false,
                available_models,
                error_message: Some(format!(
                    "O Ollama está ativo em '{}', mas NENHUM modelo está instalado localmente.\n\
                     O SERVERMIND AI não baixa modelos automaticamente.\n\
                     Como resolver:\n\
                     1. Instale um modelo manualmente em outro terminal: ollama pull {}\n\
                     2. Depois, execute `:models` (ou `/modelos`) para listar e selecionar o modelo instalado.",
                    self.base_url, target_model
                )),
            };
        }

        let model_installed =
            Self::resolve_installed_model(target_model, &available_models).is_some();

        let error_message = if !model_installed {
            let installed_list = available_models.join(", ");
            Some(format!(
                "O modelo ativo '{}' não foi encontrado no Ollama local.\n\
                 Modelos instalados detectados: {}.\n\
                 Como resolver:\n\
                 1. Use `:models` ou `:model <nome>` para alternar para um dos modelos já instalados.\n\
                 2. Ou baixe manualmente este modelo em outro terminal executando: ollama pull {}",
                target_model, installed_list, target_model
            ))
        } else {
            None
        };

        OllamaHealth {
            server_reachable: true,
            model_installed,
            available_models,
            error_message,
        }
    }

    fn post_chat_request(
        &self,
        model: &str,
        messages: &[Message],
        stream: bool,
        timeout_override: Option<Duration>,
    ) -> Result<reqwest::blocking::Response, OllamaChatOutcome> {
        if let Err(err_msg) = Self::verify_loopback_resolution(&self.base_url) {
            return Err(OllamaChatOutcome::Error {
                message: err_msg,
                is_connectivity_error: true,
            });
        }

        let url = format!("{}/api/chat", self.base_url);
        let payload = ChatRequest {
            model,
            messages,
            stream,
            keep_alive: if stream { Some("15m") } else { None },
            options: if stream {
                Some(ChatOptions { num_ctx: 4096 })
            } else {
                None
            },
        };

        let effective_timeout =
            timeout_override.unwrap_or_else(|| Duration::from_secs(self.timeout_secs));
        let effective_secs = effective_timeout.as_secs().max(1);

        let response = match self
            .client
            .post(&url)
            .timeout(effective_timeout)
            .json(&payload)
            .send()
        {
            Ok(resp) => resp,
            Err(err) => {
                if err.is_timeout() {
                    return Err(OllamaChatOutcome::Timeout {
                        message: self.format_timeout_error(effective_secs),
                    });
                }
                return Err(OllamaChatOutcome::Error {
                    message: self.format_connection_error(&err),
                    is_connectivity_error: true,
                });
            }
        };

        let status = response.status();
        if status.is_redirection() {
            return Err(OllamaChatOutcome::Error {
                message: format!(
                    "O endpoint '{}' retornou um redirecionamento HTTP ({}), que foi bloqueado por segurança.\n\
                     Como resolver: confirme se 'ollama_url' aponta diretamente para o serviço local do Ollama.",
                    url, status
                ),
                is_connectivity_error: true,
            });
        }

        if status == StatusCode::NOT_FOUND {
            let body_err = response
                .json::<ErrorPayload>()
                .ok()
                .and_then(|p| p.error)
                .map(|e| Self::sanitize_diagnostic(&e))
                .unwrap_or_default();

            return Err(OllamaChatOutcome::Error {
                message: format!(
                    "Modelo '{}' não está instalado ou o endpoint não foi encontrado no Ollama (HTTP 404{}).\n\
                     O SERVERMIND AI não baixa modelos automaticamente.\n\
                     Como resolver:\n\
                     1. Digite `:models` para ver os modelos instalados e escolher um disponível.\n\
                     2. Ou instale este modelo manualmente em outro terminal com: ollama pull {}",
                    model,
                    if body_err.is_empty() {
                        String::new()
                    } else {
                        format!(": {}", body_err)
                    },
                    model
                ),
                is_connectivity_error: false,
            });
        }

        if !status.is_success() {
            let safe_detail = response
                .json::<ErrorPayload>()
                .ok()
                .and_then(|p| p.error)
                .map(|e| Self::sanitize_diagnostic(&e))
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "sem detalhes JSON adicionais do serviço".to_string());

            return Err(OllamaChatOutcome::Error {
                message: format!(
                    "O Ollama retornou status HTTP inesperado ({}) ao processar a solicitação.\n\
                     Detalhe técnico: {}\n\
                     Como resolver: verifique com `:status` ou `:models` se o modelo '{}' está disponível e se há memória suficiente.",
                    status, safe_detail, model
                ),
                is_connectivity_error: status.is_server_error(),
            });
        }

        Ok(response)
    }

    /// Envia mensagens a `/api/chat` com timeout específico opcional e retorna classificação estruturada.
    pub fn send_chat_with_timeout(
        &self,
        model: &str,
        messages: &[Message],
        timeout_override: Option<Duration>,
    ) -> OllamaChatOutcome {
        let response = match self.post_chat_request(model, messages, false, timeout_override) {
            Ok(resp) => resp,
            Err(outcome) => return outcome,
        };

        let url = format!("{}/api/chat", self.base_url);
        let chat_resp: ChatResponse = match response.json() {
            Ok(parsed) => parsed,
            Err(e) => {
                return OllamaChatOutcome::Error {
                    message: format!(
                        "O serviço em '{}' respondeu com formato JSON inválido ou inesperado.\n\
                         Detalhe técnico: {}\n\
                         Como resolver: verifique se o serviço Ollama está atualizado e operando normalmente (`:status`).",
                        url,
                        Self::sanitize_diagnostic(&e.to_string())
                    ),
                    is_connectivity_error: true,
                };
            }
        };

        if let Some(err_msg) = chat_resp.error {
            let clean_err = Self::sanitize_diagnostic(&err_msg);
            return OllamaChatOutcome::Error {
                message: format!(
                    "O Ollama reportou um erro ao executar o modelo '{}': {}\n\
                     Como resolver: execute `:status` ou `:models` para verificar os modelos instalados.",
                    model, clean_err
                ),
                is_connectivity_error: false,
            };
        }

        let reported_model = chat_resp
            .model
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty());

        match chat_resp
            .message
            .map(|m| m.content.trim().to_string())
            .filter(|c| !c.is_empty())
        {
            Some(content) => OllamaChatOutcome::Succeeded {
                content,
                reported_model,
            },
            None => OllamaChatOutcome::EmptyResponse { reported_model },
        }
    }

    /// Envia o histórico da conversa atual ao endpoint `/api/chat` do Ollama e retorna a resposta do modelo.
    pub fn send_chat(&self, model: &str, messages: &[Message]) -> Result<String, String> {
        match self.send_chat_with_timeout(model, messages, None) {
            OllamaChatOutcome::Succeeded { content, .. } => Ok(content),
            OllamaChatOutcome::EmptyResponse { .. } => Err(format!(
                "O modelo '{}' no Ollama respondeu sem conteúdo de mensagem (resposta vazia).\n\
                 Como resolver: tente reenviar sua mensagem, consulte `:status` ou selecione outro modelo com `:models`.",
                model
            )),
            OllamaChatOutcome::Timeout { message } | OllamaChatOutcome::Error { message, .. } => {
                Err(message)
            }
        }
    }

    /// Envia o histórico da conversa ao endpoint `/api/chat` com `stream: true`, entregando
    /// cada fragmento de texto incrementalmente via `on_chunk` e retornando a resposta completa.
    pub fn send_chat_streaming<F>(
        &self,
        model: &str,
        messages: &[Message],
        mut on_chunk: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        let response = match self.post_chat_request(model, messages, true, None) {
            Ok(resp) => resp,
            Err(OllamaChatOutcome::Timeout { message })
            | Err(OllamaChatOutcome::Error { message, .. }) => return Err(message),
            Err(OllamaChatOutcome::EmptyResponse { .. })
            | Err(OllamaChatOutcome::Succeeded { .. }) => {
                return Err("Estado inesperado ao iniciar streaming com o Ollama.".to_string());
            }
        };

        let mut reader = BufReader::new(response);
        let mut line = String::new();
        let mut full_content = String::new();

        loop {
            line.clear();
            let bytes = reader.read_line(&mut line).map_err(|err| {
                if err.kind() == std::io::ErrorKind::TimedOut {
                    self.format_timeout_error(self.timeout_secs.max(1))
                } else {
                    format!(
                        "Falha ao ler fluxo de resposta (streaming) do Ollama: {}\n\
                         Como resolver: verifique `:status` e tente novamente.",
                        Self::sanitize_diagnostic(&err.to_string())
                    )
                }
            })?;
            if bytes == 0 {
                break;
            }
            let trimmed_line = line.trim();
            if trimmed_line.is_empty() {
                continue;
            }

            let chunk: ChatResponse = serde_json::from_str(trimmed_line).map_err(|e| {
                format!(
                    "O serviço em '{}' enviou um fragmento JSON inválido no streaming.\n\
                     Detalhe técnico: {}\n\
                     Como resolver: verifique se o serviço Ollama está atualizado (`:status`).",
                    self.base_url,
                    Self::sanitize_diagnostic(&e.to_string())
                )
            })?;

            if let Some(err_msg) = chunk.error {
                let clean_err = Self::sanitize_diagnostic(&err_msg);
                return Err(format!(
                    "O Ollama reportou um erro ao executar o modelo '{}': {}\n\
                     Como resolver: execute `:status` ou `:models` para verificar os modelos instalados.",
                    model, clean_err
                ));
            }

            if let Some(msg) = chunk.message {
                if !msg.content.is_empty() {
                    on_chunk(&msg.content);
                    full_content.push_str(&msg.content);
                }
            }

            if chunk.done == Some(true) {
                break;
            }
        }

        let final_trimmed = full_content.trim().to_string();
        if final_trimmed.is_empty() {
            return Err(format!(
                "O modelo '{}' no Ollama respondeu sem conteúdo de mensagem (resposta vazia).\n\
                 Como resolver: tente reenviar sua mensagem, consulte `:status` ou selecione outro modelo com `:models`.",
                model
            ));
        }

        Ok(final_trimmed)
    }

    /// Resolve a seleção do usuário (por índice numérico `1..=N` ou por nome/tag) contra a lista
    /// de modelos efetivamente instalados no Ollama local.
    pub fn resolve_installed_model(selection: &str, available: &[String]) -> Option<String> {
        let trimmed = selection.trim();
        if trimmed.is_empty() {
            return None;
        }

        // 1. Permite selecionar pelo índice numérico exibido em `:models` (1..=N)
        if let Ok(index) = trimmed.parse::<usize>() {
            if index >= 1 && index <= available.len() {
                return Some(available[index - 1].clone());
            }
        }

        let target_norm = trimmed.to_lowercase();
        let target_with_latest = if target_norm.contains(':') {
            target_norm.clone()
        } else {
            format!("{}:latest", target_norm)
        };

        // 2. Busca correspondência exata ou tag `:latest`
        for item in available {
            let cand = item.trim().to_lowercase();
            if cand == target_norm || cand == target_with_latest {
                return Some(item.clone());
            }
        }

        // 3. Se o usuário digitou apenas o prefixo antes de `:` (ex: "qwen2.5") e há um modelo correspondente
        if !target_norm.contains(':') {
            for item in available {
                let cand = item.trim().to_lowercase();
                if let Some((base, _tag)) = cand.split_once(':') {
                    if base == target_norm {
                        return Some(item.clone());
                    }
                }
            }
        }

        None
    }

    /// Resolve a seleção do usuário contra a lista de metadados de modelos instalados.
    pub fn resolve_installed_model_detail<'a>(
        selection: &str,
        available: &'a [InstalledModelInfo],
    ) -> Option<&'a InstalledModelInfo> {
        let names: Vec<String> = available.iter().map(|m| m.name.clone()).collect();
        let resolved_name = Self::resolve_installed_model(selection, &names)?;
        available.iter().find(|m| m.name == resolved_name)
    }

    fn format_timeout_error(&self, timeout_secs: u64) -> String {
        format!(
            "Tempo limite de {} segundo(s) esgotado ao aguardar resposta do Ollama em '{}'.\n\
             Causa provável: o modelo pode estar sendo carregado na memória (RAM/VRAM) pela primeira vez ou o hardware está sobrecarregado.\n\
             Como resolver: aguarde alguns segundos e tente novamente, ou aumente 'request_timeout_secs' em 'config.toml' (ou via SERVERMIND_TIMEOUT_SECS).",
            timeout_secs, self.base_url
        )
    }

    fn format_connection_error(&self, err: &reqwest::Error) -> String {
        if err.is_timeout() {
            self.format_timeout_error(self.timeout_secs)
        } else if err.is_redirect() {
            format!(
                "Redirecionamento HTTP bloqueado por segurança ao acessar '{}'.\n\
                 Como resolver: verifique se 'ollama_url' aponta diretamente para o serviço local do Ollama.",
                self.base_url
            )
        } else {
            format!(
                "Não foi possível conectar ao serviço local do Ollama em '{}'.\n\
                 Detalhe técnico: {}\n\
                 Como resolver:\n\
                 1. Verifique se o Ollama está instalado (https://ollama.com).\n\
                 2. Inicie o serviço local em outro terminal executando: ollama serve\n\
                 3. Confirme se o endereço/porta em 'config.toml' ou SERVERMIND_OLLAMA_URL está correto.",
                self.base_url,
                Self::sanitize_diagnostic(&err.to_string())
            )
        }
    }

    fn sanitize_diagnostic(raw: &str) -> String {
        let single_line: String = raw
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        let collapsed = single_line.split_whitespace().collect::<Vec<_>>().join(" ");
        if collapsed.chars().count() <= MAX_DIAGNOSTIC_CHARS {
            collapsed
        } else {
            let truncated: String = collapsed.chars().take(MAX_DIAGNOSTIC_CHARS).collect();
            format!("{}...", truncated)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_client_new_validates_and_normalizes_url() {
        let client = OllamaClient::new("http://127.0.0.1:11434/".to_string(), 60)
            .expect("deve criar cliente com URL local válida");
        assert_eq!(client.base_url, "http://127.0.0.1:11434");
        assert_eq!(client.timeout_secs, 60);

        let client_v6 = OllamaClient::new("http://[::1]:11434".to_string(), 30)
            .expect("deve aceitar IPv6 loopback");
        assert_eq!(client_v6.base_url, "http://[::1]:11434");

        let client_localhost = OllamaClient::new("http://localhost:11434/".to_string(), 30)
            .expect("deve aceitar localhost com resolução de loopback");
        assert_eq!(client_localhost.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_ollama_client_new_rejects_invalid_urls_and_zero_timeout() {
        assert!(OllamaClient::new("http://127.0.0.1:11434".to_string(), 0).is_err());
        assert!(OllamaClient::new("http://0.0.0.0:11434".to_string(), 30).is_err());
        assert!(OllamaClient::new("http://127.0.0.1@evil.com:11434".to_string(), 30).is_err());
        assert!(OllamaClient::new("https://api.openai.com/v1".to_string(), 30).is_err());
        assert!(OllamaClient::new("http://192.168.1.20:11434".to_string(), 30).is_err());
    }

    #[test]
    fn test_sanitize_diagnostic_collapses_control_chars_and_truncates() {
        let raw = format!("Erro\ncom\tquebra   de linha e {}", "x".repeat(250));
        let clean = OllamaClient::sanitize_diagnostic(&raw);
        assert!(!clean.contains('\n'));
        assert!(!clean.contains('\t'));
        assert!(clean.ends_with("..."));
        assert!(clean.chars().count() <= MAX_DIAGNOSTIC_CHARS + 3);
    }

    #[test]
    fn test_tags_response_deserializes_minimal_and_enriched_metadata() {
        let minimal_json = r#"{"models":[{"name":"llama3.2:latest"}]}"#;
        let minimal: TagsResponse =
            serde_json::from_str(minimal_json).expect("deve aceitar JSON mínimo de /api/tags");
        let list = minimal.models.expect("deve conter lista de modelos");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "llama3.2:latest");
        assert!(list[0].size.is_none());
        assert!(list[0].digest.is_none());
        assert!(list[0].details.is_none());

        let full_json = r#"{
            "models": [
                {
                    "name": "qwen2.5:7b",
                    "model": "qwen2.5:7b",
                    "modified_at": "2026-03-10T08:00:00Z",
                    "size": 4683087332,
                    "digest": "sha256:abcdef0123456789",
                    "details": {
                        "format": "gguf",
                        "family": "qwen2",
                        "parameter_size": "7.6B",
                        "quantization_level": "Q4_K_M"
                    }
                }
            ]
        }"#;
        let full: TagsResponse =
            serde_json::from_str(full_json).expect("deve aceitar JSON completo de /api/tags");
        let full_list = full.models.expect("deve conter lista de modelos");
        assert_eq!(full_list.len(), 1);
        let item = &full_list[0];
        assert_eq!(item.name, "qwen2.5:7b");
        assert_eq!(item.size, Some(4683087332));
        assert_eq!(item.digest.as_deref(), Some("sha256:abcdef0123456789"));
        let det = item.details.as_ref().expect("deve conter details");
        assert_eq!(det.family.as_deref(), Some("qwen2"));
        assert_eq!(det.parameter_size.as_deref(), Some("7.6B"));
        assert_eq!(det.quantization_level.as_deref(), Some("Q4_K_M"));

        let resolved = OllamaClient::resolve_installed_model_detail("qwen2.5", &full_list)
            .expect("deve resolver por prefixo");
        assert_eq!(resolved.name, "qwen2.5:7b");
    }
}
