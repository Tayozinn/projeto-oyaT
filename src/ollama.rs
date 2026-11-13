use crate::models::{Message, OllamaHealth};
use reqwest::blocking::Client;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Cliente HTTP síncrono para comunicação exclusiva com a API local do Ollama.
pub struct OllamaClient {
    base_url: String,
    client: Client,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: Option<ChatMessagePayload>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessagePayload {
    content: String,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Option<Vec<ModelTag>>,
}

#[derive(Debug, Deserialize)]
struct ModelTag {
    name: String,
}

#[derive(Debug, Deserialize)]
struct ErrorPayload {
    error: Option<String>,
}

impl OllamaClient {
    pub fn new(base_url: String, timeout_secs: u64) -> Result<Self, String> {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| format!("Falha ao inicializar cliente HTTP: {}", e))?;

        Ok(Self { base_url, client })
    }

    /// Consulta o endpoint `/api/tags` do Ollama local e retorna a lista de modelos instalados.
    pub fn list_installed_models(&self) -> Result<Vec<String>, String> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|err| self.format_connection_error(&err))?;

        if !response.status().is_success() {
            return Err(format!(
                "O endpoint '{}' respondeu com status HTTP inesperado: {}. \
                 Verifique se o serviço local do Ollama está rodando corretamente.",
                url,
                response.status()
            ));
        }

        let tags: TagsResponse = response.json().map_err(|err| {
            format!(
                "O serviço respondeu em '{}', mas não retornou um JSON válido do Ollama: {}",
                url, err
            )
        })?;

        let available_models: Vec<String> = tags
            .models
            .unwrap_or_default()
            .into_iter()
            .map(|m| m.name)
            .collect();

        Ok(available_models)
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

        let model_installed = Self::resolve_installed_model(target_model, &available_models).is_some();

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

    /// Envia o histórico da conversa atual ao endpoint `/api/chat` do Ollama e retorna a resposta do modelo.
    pub fn send_chat(&self, model: &str, messages: &[Message]) -> Result<String, String> {
        let url = format!("{}/api/chat", self.base_url);
        let payload = ChatRequest {
            model,
            messages,
            stream: false,
        };

        let response = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .map_err(|e| self.format_connection_error(&e))?;

        let status = response.status();
        if status == StatusCode::NOT_FOUND {
            let body_err = response
                .json::<ErrorPayload>()
                .ok()
                .and_then(|p| p.error)
                .unwrap_or_default();

            return Err(format!(
                "Modelo '{}' não está instalado no Ollama (HTTP 404{}).\n\
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
            ));
        }

        if !status.is_success() {
            let raw_text = response.text().unwrap_or_default();
            return Err(format!(
                "O Ollama retornou erro HTTP {} ao processar a mensagem.\nDetalhes: {}",
                status,
                if raw_text.trim().is_empty() {
                    "sem detalhes adicionais no corpo da resposta"
                } else {
                    raw_text.trim()
                }
            ));
        }

        let chat_resp: ChatResponse = response.json().map_err(|e| {
            format!(
                "Erro ao decodificar resposta JSON do Ollama em '{}': {}",
                url, e
            )
        })?;

        if let Some(err_msg) = chat_resp.error {
            return Err(format!("Erro reportado pelo Ollama: {}", err_msg));
        }

        chat_resp
            .message
            .map(|m| m.content.trim().to_string())
            .ok_or_else(|| "O Ollama respondeu sem conteúdo de mensagem.".to_string())
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

    fn format_connection_error(&self, err: &reqwest::Error) -> String {
        if err.is_timeout() {
            format!(
                "Tempo limite esgotado ao aguardar resposta do Ollama em '{}'.\n\
                 O modelo pode estar carregando na memória pela primeira vez ou o hardware está sobrecarregado.\n\
                 Dica: tente novamente em alguns segundos ou aumente 'request_timeout_secs' em config.toml.",
                self.base_url
            )
        } else {
            format!(
                "Não foi possível conectar ao serviço local do Ollama em '{}'.\n\
                 Causa técnica: {}\n\
                 Como resolver:\n\
                 1. Verifique se o Ollama está instalado (https://ollama.com).\n\
                 2. Inicie o serviço local executando em outro terminal: ollama serve\n\
                 3. Confirme se a porta/URL em 'config.toml' ou SERVERMIND_OLLAMA_URL está correta.",
                self.base_url, err
            )
        }
    }
}
