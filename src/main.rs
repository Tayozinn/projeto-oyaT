mod agent;
mod cli;
mod config;
mod models;
mod ollama;
mod storage;

use agent::AgentSession;
use config::AppConfig;
use ollama::OllamaClient;
use std::process;
use storage::Storage;

fn main() {
    let config = match AppConfig::load() {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("[ERRO DE CONFIGURAÇÃO]\n{}", err);
            process::exit(1);
        }
    };

    let storage = match Storage::new(&config.database_path) {
        Ok(db) => db,
        Err(err) => {
            eprintln!("[ERRO DE BANCO DE DADOS SQLITE]\n{}", err);
            process::exit(1);
        }
    };

    let ollama = match OllamaClient::new(config.ollama_url.clone(), config.request_timeout_secs) {
        Ok(client) => client,
        Err(err) => {
            eprintln!("[ERRO DE CLIENTE OLLAMA]\n{}", err);
            process::exit(1);
        }
    };

    let mut session = match AgentSession::new(config, storage, ollama) {
        Ok(sess) => sess,
        Err(err) => {
            eprintln!("[ERRO AO INICIAR SESSÃO]\n{}", err);
            process::exit(1);
        }
    };

    cli::run_terminal_loop(&mut session);
}

#[cfg(test)]
mod tests {
    use super::config::{
        AppConfig, DEFAULT_CUSTOM_INSTRUCTIONS, IMMUTABLE_SAFETY_INSTRUCTIONS,
        MAX_INSTRUCTIONS_CHARS,
    };
    use super::models::Role;
    use super::ollama::OllamaClient;
    use super::storage::Storage;

    #[test]
    fn test_local_url_validation() {
        assert!(AppConfig::validate_local_url("http://127.0.0.1:11434").is_ok());
        assert!(AppConfig::validate_local_url("http://localhost:11434").is_ok());
        assert!(AppConfig::validate_local_url("http://[::1]:11434").is_ok());
        assert!(AppConfig::validate_local_url("https://api.openai.com/v1").is_err());
        assert!(AppConfig::validate_local_url("http://192.168.1.50:11434").is_err());
    }

    #[test]
    fn test_model_resolution() {
        let installed = vec![
            "llama3.2:latest".to_string(),
            "mistral:7b".to_string(),
            "qwen2.5-coder:latest".to_string(),
        ];

        // Seleção por índice numérico (1..=N)
        assert_eq!(
            OllamaClient::resolve_installed_model("1", &installed),
            Some("llama3.2:latest".to_string())
        );
        assert_eq!(
            OllamaClient::resolve_installed_model("2", &installed),
            Some("mistral:7b".to_string())
        );

        // Seleção por nome sem tag ou com tag completa
        assert_eq!(
            OllamaClient::resolve_installed_model("llama3.2", &installed),
            Some("llama3.2:latest".to_string())
        );
        assert_eq!(
            OllamaClient::resolve_installed_model("mistral:7b", &installed),
            Some("mistral:7b".to_string())
        );
        assert_eq!(
            OllamaClient::resolve_installed_model("mistral", &installed),
            Some("mistral:7b".to_string())
        );

        // Modelo não instalado deve retornar None
        assert_eq!(
            OllamaClient::resolve_installed_model("deepseek-r1", &installed),
            None
        );
        assert_eq!(OllamaClient::resolve_installed_model("99", &installed), None);
    }

    #[test]
    fn test_conversation_history_and_switching_non_destructive() {
        let storage = Storage::new(":memory:").expect("deve criar banco em memória");

        let conv1 = storage
            .create_conversation("Diagnóstico Nginx", "llama3.2")
            .expect("deve criar conversa 1");
        storage
            .save_message(conv1.id, Role::User, "Como testar configuração do Nginx?")
            .expect("deve salvar pergunta 1");
        storage
            .save_message(conv1.id, Role::Assistant, "Execute nginx -t para validar a sintaxe.")
            .expect("deve salvar resposta 1");

        let conv2 = storage
            .create_conversation("Análise de Firewall", "mistral:7b")
            .expect("deve criar conversa 2");
        storage
            .save_message(conv2.id, Role::User, "Como listar regras no ufw?")
            .expect("deve salvar pergunta 2");

        // Verifica que ambas as conversas coexistem intactas no histórico
        let list = storage.list_conversations(10).expect("deve listar conversas");
        assert_eq!(list.len(), 2);

        let loaded_conv1 = storage
            .get_conversation(conv1.id)
            .expect("consulta ok")
            .expect("conversa 1 existe");
        assert_eq!(loaded_conv1.message_count, 2);

        let msgs_conv1 = storage
            .get_messages(conv1.id)
            .expect("deve carregar mensagens da conversa 1");
        assert_eq!(msgs_conv1.len(), 2);
        assert_eq!(msgs_conv1[0].content, "Como testar configuração do Nginx?");

        let msgs_conv2 = storage
            .get_messages(conv2.id)
            .expect("deve carregar mensagens da conversa 2");
        assert_eq!(msgs_conv2.len(), 1);
    }

    #[test]
    fn test_custom_instructions_validation_and_safety() {
        // Rejeita entradas vazias ou apenas espaços
        assert!(AppConfig::validate_custom_instructions("").is_err());
        assert!(AppConfig::validate_custom_instructions("    \n\t  ").is_err());

        // Rejeita instruções excessivamente longas (> MAX_INSTRUCTIONS_CHARS)
        let too_long = "a".repeat(MAX_INSTRUCTIONS_CHARS + 1);
        let err_msg = AppConfig::validate_custom_instructions(&too_long)
            .expect_err("deve rejeitar instruções longas");
        assert!(err_msg.contains(":instructions reset"));

        // Aceita instruções válidas e preserva sempre as diretrizes de segurança
        let valid = AppConfig::validate_custom_instructions(
            "Priorize exemplos para servidores Rocky Linux e Debian.",
        )
        .expect("deve aceitar instruções válidas");

        let cfg = AppConfig {
            config_file_path: "config.toml".to_string(),
            ollama_url: "http://127.0.0.1:11434".to_string(),
            model: "llama3.2".to_string(),
            database_path: ":memory:".to_string(),
            request_timeout_secs: 120,
            custom_instructions: valid.clone(),
            config_source: "teste".to_string(),
            model_from_env: false,
        };

        let effective = cfg.effective_system_prompt(&[]);
        assert!(effective.contains(&valid));
        assert!(effective.contains(IMMUTABLE_SAFETY_INSTRUCTIONS));
        assert!(!DEFAULT_CUSTOM_INSTRUCTIONS.is_empty());
    }

    #[test]
    fn test_explicit_memories_and_settings_persistence() {
        let storage = Storage::new(":memory:").expect("deve criar banco em memória");

        storage
            .set_setting("active_model", "qwen2.5:7b")
            .expect("deve salvar modelo ativo");
        assert_eq!(
            storage.get_setting("active_model").expect("leitura ok"),
            Some("qwen2.5:7b".to_string())
        );

        let mem = storage
            .add_memory("O servidor principal usa Debian 12 na porta SSH 2222.")
            .expect("deve salvar memória explícita");
        assert_eq!(mem.id, 1);

        let memories = storage.list_memories().expect("deve listar memórias");
        assert_eq!(memories.len(), 1);
        assert!(memories[0].content.contains("Debian 12"));

        assert!(storage.delete_memory(mem.id).expect("deve remover memória"));
        assert!(storage.list_memories().expect("lista vazia").is_empty());
    }
}
