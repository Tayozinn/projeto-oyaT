use crate::config::{AppConfig, DEFAULT_CUSTOM_INSTRUCTIONS};
use crate::models::{Conversation, MemoryEntry, Message, OllamaHealth, Role};
use crate::ollama::OllamaClient;
use crate::storage::Storage;

pub const SETTING_ACTIVE_MODEL: &str = "active_model";
pub const SETTING_CUSTOM_INSTRUCTIONS: &str = "custom_instructions";

/// Núcleo conversacional do SERVERMIND AI (Fase 2).
/// Desacoplado da interface de terminal e responsável por coordenar modelo ativo,
/// histórico de conversas, instruções personalizadas com travas de segurança e memórias explícitas.
pub struct AgentSession {
    config: AppConfig,
    storage: Storage,
    ollama: OllamaClient,
    current_conversation: Conversation,
    context_messages: Vec<Message>,
}

impl AgentSession {
    pub fn new(
        mut config: AppConfig,
        storage: Storage,
        ollama: OllamaClient,
    ) -> Result<Self, String> {
        // Restaura o modelo persistido localmente (a menos que SERVERMIND_MODEL tenha sido forçado no ambiente)
        if !config.model_from_env {
            if let Ok(Some(saved_model)) = storage.get_setting(SETTING_ACTIVE_MODEL) {
                let trimmed = saved_model.trim();
                if !trimmed.is_empty() {
                    config.model = trimmed.to_string();
                }
            }
        }

        // Restaura instruções personalizadas persistidas no SQLite, se existirem e forem válidas
        if let Ok(Some(saved_instructions)) = storage.get_setting(SETTING_CUSTOM_INSTRUCTIONS) {
            if let Ok(valid) = AppConfig::validate_custom_instructions(&saved_instructions) {
                config.custom_instructions = valid;
            }
        }

        let memories = storage.list_memories().unwrap_or_default();
        let system_prompt = config.effective_system_prompt(&memories);

        let current_conversation =
            storage.create_conversation("Nova conversa", &config.model)?;

        let system_msg = Message::new(Role::System, system_prompt);

        Ok(Self {
            config,
            storage,
            ollama,
            current_conversation,
            context_messages: vec![system_msg],
        })
    }

    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    pub fn current_conversation(&self) -> &Conversation {
        &self.current_conversation
    }

    /// Verifica a saúde do Ollama local e a disponibilidade do modelo ativo.
    pub fn check_health(&self) -> OllamaHealth {
        self.ollama.check_health(&self.config.model)
    }

    /// Consulta o Ollama local e retorna a lista de modelos instalados.
    pub fn list_installed_models(&self) -> Result<Vec<String>, String> {
        self.ollama.list_installed_models()
    }

    /// Seleciona um modelo instalado no Ollama local (por nome ou índice `1..=N`),
    /// valida sua existência sem baixar modelos automaticamente e persiste a escolha localmente.
    pub fn select_installed_model(&mut self, selection: &str) -> Result<String, String> {
        let trimmed = selection.trim();
        if trimmed.is_empty() {
            return Err("Informe o nome ou o número do modelo instalado que deseja selecionar.".to_string());
        }

        let installed = self.ollama.list_installed_models()?;
        if installed.is_empty() {
            return Err(format!(
                "Nenhum modelo está instalado no Ollama local ({}).\n\
                 O SERVERMIND AI não baixa modelos automaticamente.\n\
                 Como resolver:\n\
                 1. Em outro terminal, instale um modelo manualmente: ollama pull llama3.2\n\
                 2. Em seguida, execute `:models` novamente para selecioná-lo.",
                self.config.ollama_url
            ));
        }

        let resolved = OllamaClient::resolve_installed_model(trimmed, &installed).ok_or_else(|| {
            let numbered: Vec<String> = installed
                .iter()
                .enumerate()
                .map(|(i, m)| format!("  [{}] {}", i + 1, m))
                .collect();
            format!(
                "O modelo '{}' não está instalado no Ollama local.\n\
                 O SERVERMIND AI não realiza download automático de modelos.\n\
                 Modelos instalados disponíveis para seleção imediata:\n{}\n\
                 Use `:model <número ou nome>` para escolher um deles, ou execute `ollama pull {}` manualmente no terminal.",
                trimmed,
                numbered.join("\n"),
                trimmed
            )
        })?;

        self.config.model = resolved.clone();
        self.current_conversation.model = resolved.clone();

        // Persiste no SQLite e no arquivo config.toml para manter ativo ao reiniciar
        self.storage.set_setting(SETTING_ACTIVE_MODEL, &resolved)?;
        let _ = self
            .storage
            .update_conversation_model(self.current_conversation.id, &resolved);
        self.config.save_to_file()?;

        Ok(resolved)
    }

    /// Inicia uma nova conversa separada no SQLite, preservando integralmente todas as conversas anteriores.
    pub fn start_new_conversation(&mut self) -> Result<&Conversation, String> {
        let conv = self
            .storage
            .create_conversation("Nova conversa", &self.config.model)?;
        self.current_conversation = conv;
        let system_prompt = self.build_effective_system_prompt()?;
        self.context_messages = vec![Message::new(Role::System, system_prompt)];
        Ok(&self.current_conversation)
    }

    /// Carrega e reabre uma conversa existente do SQLite pelo ID, restaurando todo o seu contexto
    /// sem apagar nem sobrescrever outras conversas.
    pub fn load_conversation(&mut self, conversation_id: i64) -> Result<usize, String> {
        let conv = self
            .storage
            .get_conversation(conversation_id)?
            .ok_or_else(|| {
                format!(
                    "Conversa #{} não encontrada no banco SQLite. Use `:history` para ver os IDs disponíveis.",
                    conversation_id
                )
            })?;

        let saved_messages = self.storage.get_messages(conversation_id)?;
        let system_prompt = self.build_effective_system_prompt()?;
        let mut context = vec![Message::new(Role::System, system_prompt)];
        let count = saved_messages.len();
        context.extend(saved_messages);

        self.current_conversation = conv;
        self.context_messages = context;
        Ok(count)
    }

    /// Lista as conversas salvas no banco local sem modificá-las.
    pub fn list_conversations(&self, limit: usize) -> Result<Vec<Conversation>, String> {
        self.storage.list_conversations(limit)
    }

    /// Retorna as mensagens da conversa atual (excluindo o prompt de sistema interno).
    pub fn visible_messages(&self) -> Vec<&Message> {
        self.context_messages
            .iter()
            .filter(|m| m.role != Role::System)
            .collect()
    }

    /// Retorna as instruções personalizadas atualmente ativas.
    pub fn custom_instructions(&self) -> &str {
        &self.config.custom_instructions
    }

    /// Indica se as instruções atuais são exatamente as instruções padrão.
    pub fn is_default_instructions(&self) -> bool {
        self.config.custom_instructions.trim() == DEFAULT_CUSTOM_INSTRUCTIONS.trim()
    }

    /// Valida, aplica e persiste localmente novas instruções para o assistente,
    /// preservando sempre as diretrizes imutáveis de segurança.
    pub fn set_custom_instructions(&mut self, raw_input: &str) -> Result<(), String> {
        let validated = AppConfig::validate_custom_instructions(raw_input)?;
        self.config.custom_instructions = validated.clone();

        self.storage
            .set_setting(SETTING_CUSTOM_INSTRUCTIONS, &validated)?;
        self.config.save_to_file()?;
        self.refresh_system_message_in_context()
    }

    /// Restaura as instruções personalizadas para o valor padrão e persiste a alteração localmente.
    pub fn reset_custom_instructions(&mut self) -> Result<(), String> {
        let default_text = DEFAULT_CUSTOM_INSTRUCTIONS.to_string();
        self.config.custom_instructions = default_text.clone();

        self.storage
            .set_setting(SETTING_CUSTOM_INSTRUCTIONS, &default_text)?;
        self.config.save_to_file()?;
        self.refresh_system_message_in_context()
    }

    /// Salva uma memória explicitamente solicitada pelo administrador por comando dedicado.
    pub fn add_explicit_memory(&mut self, content: &str) -> Result<MemoryEntry, String> {
        let entry = self.storage.add_memory(content)?;
        self.refresh_system_message_in_context()?;
        Ok(entry)
    }

    /// Lista todas as memórias explícitas salvas no SQLite.
    pub fn list_memories(&self) -> Result<Vec<MemoryEntry>, String> {
        self.storage.list_memories()
    }

    /// Remove uma memória explícita pelo ID e atualiza o contexto de sistema.
    pub fn delete_memory(&mut self, memory_id: i64) -> Result<bool, String> {
        let removed = self.storage.delete_memory(memory_id)?;
        if removed {
            self.refresh_system_message_in_context()?;
        }
        Ok(removed)
    }

    /// Envia uma mensagem comum do administrador ao modelo ativo no Ollama, mantendo o contexto
    /// da conversa atual e salvando apenas as mensagens da conversa no SQLite (nunca salva memórias implicitamente).
    pub fn send_user_message(&mut self, user_input: &str) -> Result<String, String> {
        let trimmed = user_input.trim();
        if trimmed.is_empty() {
            return Err("A mensagem não pode estar vazia.".to_string());
        }

        // Atualiza o título da conversa automaticamente na primeira mensagem do usuário
        let is_first_user_msg = !self
            .context_messages
            .iter()
            .any(|m| m.role == Role::User);

        let previous_title = self.current_conversation.title.clone();
        if is_first_user_msg && previous_title == "Nova conversa" {
            let title = Self::summarize_title(trimmed);
            let _ = self
                .storage
                .update_conversation_title(self.current_conversation.id, &title);
            self.current_conversation.title = title;
        }

        let user_msg = Message::new(Role::User, trimmed);
        self.context_messages.push(user_msg);

        match self
            .ollama
            .send_chat(&self.config.model, &self.context_messages)
        {
            Ok(reply) => {
                self.storage.save_message(
                    self.current_conversation.id,
                    Role::User,
                    trimmed,
                )?;
                self.storage.save_message(
                    self.current_conversation.id,
                    Role::Assistant,
                    &reply,
                )?;

                self.current_conversation.message_count += 2;
                self.context_messages
                    .push(Message::new(Role::Assistant, reply.clone()));

                Ok(reply)
            }
            Err(err) => {
                self.context_messages.pop();
                if is_first_user_msg && previous_title == "Nova conversa" {
                    let _ = self
                        .storage
                        .update_conversation_title(self.current_conversation.id, &previous_title);
                    self.current_conversation.title = previous_title;
                }
                Err(err)
            }
        }
    }

    fn build_effective_system_prompt(&self) -> Result<String, String> {
        let memories = self.storage.list_memories()?;
        Ok(self.config.effective_system_prompt(&memories))
    }

    fn refresh_system_message_in_context(&mut self) -> Result<(), String> {
        let new_prompt = self.build_effective_system_prompt()?;
        if let Some(first) = self.context_messages.first_mut() {
            if first.role == Role::System {
                first.content = new_prompt;
                return Ok(());
            }
        }
        self.context_messages
            .insert(0, Message::new(Role::System, new_prompt));
        Ok(())
    }

    fn summarize_title(first_message: &str) -> String {
        let single_line = first_message.lines().next().unwrap_or("Nova conversa").trim();
        let max_chars = 50;
        let char_count = single_line.chars().count();
        if char_count <= max_chars {
            single_line.to_string()
        } else {
            let truncated: String = single_line.chars().take(max_chars).collect();
            format!("{}...", truncated)
        }
    }
}
