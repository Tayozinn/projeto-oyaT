use serde::{Deserialize, Serialize};

/// Papel de cada mensagem no histórico e na API do Ollama.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    pub fn from_db_str(value: &str) -> Self {
        match value.trim().to_lowercase().as_str() {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            _ => Role::User,
        }
    }
}

/// Representa uma mensagem trocada ou instrução de sistema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<i64>,
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            id: None,
            conversation_id: None,
            role,
            content: content.into(),
            created_at: None,
        }
    }
}

/// Metadados de uma conversa armazenada no SQLite.
#[derive(Debug, Clone)]
pub struct Conversation {
    pub id: i64,
    pub title: String,
    pub model: String,
    pub created_at: String,
    pub message_count: i64,
}

/// Registro de memória explícita salva pelo administrador.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub id: i64,
    pub content: String,
    pub created_at: String,
}

/// Resultado da verificação de disponibilidade do Ollama e do modelo configurado.
#[derive(Debug, Clone)]
pub struct OllamaHealth {
    pub server_reachable: bool,
    pub model_installed: bool,
    pub available_models: Vec<String>,
    pub error_message: Option<String>,
}
