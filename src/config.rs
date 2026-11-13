use crate::models::MemoryEntry;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;

pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";
pub const DEFAULT_MODEL: &str = "llama3.2";
pub const DEFAULT_DB_PATH: &str = "servermind.db";
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;
pub const DEFAULT_CONFIG_FILE: &str = "config.toml";

/// Limite máximo de caracteres para instruções personalizadas do assistente.
pub const MAX_INSTRUCTIONS_CHARS: usize = 2000;

/// Instruções padrão personalizáveis do assistente.
pub const DEFAULT_CUSTOM_INSTRUCTIONS: &str = "\
Você é o SERVERMIND AI, um assistente local para administradores de sistemas e servidores. \
Responda sempre em português do Brasil, de forma clara, técnica, segura e objetiva.";

/// Instruções imutáveis de segurança que são sempre preservadas independentemente da personalização.
pub const IMMUTABLE_SAFETY_INSTRUCTIONS: &str = "\
Diretrizes imutáveis de segurança e escopo (obrigatórias em todas as interações):\n\
- Você é estritamente conversacional e consultivo.\n\
- Você NÃO executa comandos no sistema operacional, NÃO altera arquivos ou configurações, NÃO instala pacotes e NÃO acessa servidores ou redes diretamente.\n\
- Você NÃO salva memórias automaticamente a partir de mensagens comuns; memórias só são registradas quando o administrador usa o comando explícito de memória.\n\
- Quando o administrador pedir ajuda com tarefas administrativas, forneça orientações passo a passo, explicações técnicas e exemplos de comandos para que o próprio administrador revise e execute manualmente.";

#[derive(Debug, Serialize, Deserialize, Default)]
struct FileConfig {
    ollama_url: Option<String>,
    model: Option<String>,
    database_path: Option<String>,
    request_timeout_secs: Option<u64>,
    system_prompt: Option<String>,
}

/// Configuração consolidada da aplicação.
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub config_file_path: String,
    pub ollama_url: String,
    pub model: String,
    pub database_path: String,
    pub request_timeout_secs: u64,
    pub custom_instructions: String,
    pub config_source: String,
    pub model_from_env: bool,
}

impl AppConfig {
    /// Carrega a configuração combinando valores padrão, arquivo `config.toml` (se existir)
    /// e variáveis de ambiente (`SERVERMIND_OLLAMA_URL`, `SERVERMIND_MODEL`, `SERVERMIND_DB_PATH`,
    /// `SERVERMIND_TIMEOUT_SECS`, `SERVERMIND_CONFIG`).
    pub fn load() -> Result<Self, String> {
        let config_path = env::var("SERVERMIND_CONFIG")
            .unwrap_or_else(|_| DEFAULT_CONFIG_FILE.to_string());

        let (file_cfg, source_desc) = if Path::new(&config_path).exists() {
            let raw = fs::read_to_string(&config_path).map_err(|e| {
                format!(
                    "Erro ao ler o arquivo de configuração '{}': {}",
                    config_path, e
                )
            })?;
            let parsed: FileConfig = toml::from_str(&raw).map_err(|e| {
                format!(
                    "Arquivo '{}' contém sintaxe TOML inválida: {}",
                    config_path, e
                )
            })?;
            (parsed, format!("arquivo '{}'", config_path))
        } else {
            (FileConfig::default(), "valores padrão internos".to_string())
        };

        let mut ollama_url = env::var("SERVERMIND_OLLAMA_URL")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .or(file_cfg.ollama_url)
            .unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_string());

        while ollama_url.ends_with('/') {
            ollama_url.pop();
        }

        Self::validate_local_url(&ollama_url)?;

        let env_model = env::var("SERVERMIND_MODEL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let model_from_env = env_model.is_some();

        let model = env_model
            .or(file_cfg.model)
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());

        let database_path = env::var("SERVERMIND_DB_PATH")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .or(file_cfg.database_path)
            .unwrap_or_else(|| DEFAULT_DB_PATH.to_string());

        let request_timeout_secs = env::var("SERVERMIND_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .or(file_cfg.request_timeout_secs)
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        let custom_instructions = match file_cfg.system_prompt {
            Some(raw_prompt) => {
                let cleaned = Self::strip_legacy_safety_suffix(&raw_prompt);
                if cleaned.is_empty() {
                    DEFAULT_CUSTOM_INSTRUCTIONS.to_string()
                } else {
                    Self::validate_custom_instructions(&cleaned)?
                }
            }
            None => DEFAULT_CUSTOM_INSTRUCTIONS.to_string(),
        };

        Ok(Self {
            config_file_path: config_path,
            ollama_url,
            model,
            database_path,
            request_timeout_secs,
            custom_instructions,
            config_source: source_desc,
            model_from_env,
        })
    }

    /// Remove o sufixo legado da Fase 1 caso esteja presente no `config.toml`, evitando duplicação
    /// com as diretrizes imutáveis de segurança da Fase 2.
    fn strip_legacy_safety_suffix(raw: &str) -> String {
        let trimmed = raw.trim();
        if let Some(idx) = trimmed.find("Limites importantes desta fase:") {
            trimmed[..idx].trim().to_string()
        } else {
            trimmed.to_string()
        }
    }

    /// Valida as instruções personalizadas do assistente contra entradas vazias ou excessivamente longas.
    pub fn validate_custom_instructions(input: &str) -> Result<String, String> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(
                "As instruções personalizadas não podem ser vazias.\n\
                 Dica: informe um texto descritivo (ex: `:instructions Responda de forma concisa focada em Linux`) \
                 ou restaure o comportamento padrão com `:instructions reset` (ou `/instrucoes padrao`)."
                    .to_string(),
            );
        }

        let char_count = trimmed.chars().count();
        if char_count > MAX_INSTRUCTIONS_CHARS {
            return Err(format!(
                "As instruções informadas são excessivamente longas ({} caracteres). \
                 O limite máximo permitido é de {} caracteres.\n\
                 Dica: reduza o texto das instruções ou volte ao padrão executando `:instructions reset` (ou `/instrucoes padrao`).",
                char_count, MAX_INSTRUCTIONS_CHARS
            ));
        }

        Ok(trimmed.to_string())
    }

    /// Compõe o prompt de sistema efetivo combinando:
    /// 1. As instruções personalizadas do assistente;
    /// 2. As instruções imutáveis de segurança (sempre ativas);
    /// 3. As memórias salvas explicitamente pelo administrador (se houver).
    pub fn effective_system_prompt(&self, memories: &[MemoryEntry]) -> String {
        let mut prompt = format!(
            "{}\n\n{}",
            self.custom_instructions.trim(),
            IMMUTABLE_SAFETY_INSTRUCTIONS
        );

        if !memories.is_empty() {
            prompt.push_str("\n\nMemórias explícitas salvas pelo administrador (utilize como contexto quando relevante):\n");
            for mem in memories {
                prompt.push_str(&format!("- [Memória #{}] {}\n", mem.id, mem.content));
            }
        }

        prompt
    }

    /// Salva a configuração atual no arquivo TOML local (`config.toml`), garantindo persistência
    /// entre reinicializações.
    pub fn save_to_file(&self) -> Result<(), String> {
        let file_cfg = FileConfig {
            ollama_url: Some(self.ollama_url.clone()),
            model: Some(self.model.clone()),
            database_path: Some(self.database_path.clone()),
            request_timeout_secs: Some(self.request_timeout_secs),
            system_prompt: Some(self.custom_instructions.clone()),
        };

        let body = toml::to_string_pretty(&file_cfg)
            .map_err(|e| format!("Erro ao serializar configuração TOML: {}", e))?;

        let content = format!(
            "# Configuração local do SERVERMIND AI (atualizada automaticamente)\n\
             # Para voltar as instruções ao padrão pelo terminal, use: :instructions reset\n\n{}",
            body
        );

        fs::write(&self.config_file_path, content).map_err(|e| {
            format!(
                "Não foi possível salvar a configuração em '{}': {}",
                self.config_file_path, e
            )
        })
    }

    /// Garante que a URL do Ollama aponte exclusivamente para um endereço de loopback local,
    /// impedindo o envio acidental de mensagens ou histórico para servidores externos.
    pub fn validate_local_url(url: &str) -> Result<(), String> {
        let lower = url.trim().to_lowercase();
        let without_scheme = lower
            .strip_prefix("http://")
            .or_else(|| lower.strip_prefix("https://"))
            .ok_or_else(|| {
                format!(
                    "URL inválida '{}': deve iniciar com http:// ou https://",
                    url
                )
            })?;

        let host_port = without_scheme.split('/').next().unwrap_or("");
        let host = if host_port.starts_with('[') {
            host_port
                .split(']')
                .next()
                .map(|s| format!("{}]", s))
                .unwrap_or_default()
        } else {
            host_port.split(':').next().unwrap_or("").to_string()
        };

        let is_local = matches!(
            host.as_str(),
            "localhost" | "127.0.0.1" | "[::1]" | "0.0.0.0"
        );

        if !is_local {
            return Err(format!(
                "Política de privacidade local violada: a URL '{}' aponta para o host externo '{}'. \
                 O SERVERMIND AI permite apenas endpoints locais (127.0.0.1, localhost ou [::1]).",
                url, host
            ));
        }

        Ok(())
    }
}
