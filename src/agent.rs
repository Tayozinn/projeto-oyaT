use crate::config::{AppConfig, DEFAULT_CUSTOM_INSTRUCTIONS};
use crate::inference::{InferenceBackendKind, InferenceRouter, NeuralEngine};
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
    inference: InferenceRouter,
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
            if let Some(saved_model) = storage.get_setting(SETTING_ACTIVE_MODEL)? {
                let trimmed = saved_model.trim();
                if !trimmed.is_empty() {
                    config.model = trimmed.to_string();
                }
            }
        }

        // Restaura instruções personalizadas persistidas no SQLite, propagando erros reais do banco
        if let Some(saved_instructions) = storage.get_setting(SETTING_CUSTOM_INSTRUCTIONS)? {
            config.custom_instructions =
                AppConfig::validate_custom_instructions(&saved_instructions).map_err(|_| {
                    "As instruções personalizadas salvas no banco SQLite são inválidas (vazias ou acima do limite permitido).\n\
                     Causa: o valor armazenado em 'app_settings' não atende às regras atuais de validação.\n\
                     Próxima ação: corrija ou remova a chave 'custom_instructions' em 'app_settings' no banco SQLite, ou defina outro banco via SERVERMIND_DB_PATH."
                        .to_string()
                })?;
        }

        let memories = storage.list_memories()?;
        let system_prompt = config.effective_system_prompt(&memories);

        let current_conversation = storage.create_conversation("Nova conversa", &config.model)?;

        let system_msg = Message::new(Role::System, system_prompt);

        Ok(Self {
            config,
            storage,
            ollama,
            inference: InferenceRouter::Ollama,
            current_conversation,
            context_messages: vec![system_msg],
        })
    }

    pub fn backend_kind(&self) -> InferenceBackendKind {
        self.inference.kind()
    }

    pub fn use_ollama_backend(&mut self) {
        self.inference = InferenceRouter::Ollama;
    }

    pub fn use_neural_backend(
        &mut self,
        checkpoint_dir: &std::path::Path,
    ) -> Result<(usize, f32), String> {
        let engine = NeuralEngine::load_from_dir(checkpoint_dir)?;
        let info = (engine.step, engine.best_val_loss);
        self.inference = InferenceRouter::ServerMindNeural(engine);
        Ok(info)
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
            return Err(
                "Informe o nome ou o número do modelo instalado que deseja selecionar.".to_string(),
            );
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

        let mut updated_cfg = self.config.clone();
        updated_cfg.model = resolved.clone();

        // Persiste no SQLite e no arquivo config.toml antes de mutar o estado em memória
        self.storage.set_setting(SETTING_ACTIVE_MODEL, &resolved)?;
        if let Err(err) = self
            .storage
            .update_conversation_model(self.current_conversation.id, &resolved)
        {
            let _ = self
                .storage
                .set_setting(SETTING_ACTIVE_MODEL, &self.config.model);
            return Err(err);
        }
        if let Err(err) = updated_cfg.save_to_file() {
            let _ = self
                .storage
                .set_setting(SETTING_ACTIVE_MODEL, &self.config.model);
            let _ = self.storage.update_conversation_model(
                self.current_conversation.id,
                &self.current_conversation.model,
            );
            return Err(err);
        }

        self.config.model = resolved.clone();
        self.current_conversation.model = resolved.clone();

        Ok(resolved)
    }

    /// Inicia uma nova conversa separada no SQLite, preservando integralmente todas as conversas anteriores.
    pub fn start_new_conversation(&mut self) -> Result<&Conversation, String> {
        let system_prompt = self.build_effective_system_prompt()?;
        let conv = self
            .storage
            .create_conversation("Nova conversa", &self.config.model)?;
        self.current_conversation = conv;
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
        self.apply_custom_instructions(validated)
    }

    /// Restaura as instruções personalizadas para o valor padrão e persiste a alteração localmente.
    pub fn reset_custom_instructions(&mut self) -> Result<(), String> {
        self.apply_custom_instructions(DEFAULT_CUSTOM_INSTRUCTIONS.to_string())
    }

    fn apply_custom_instructions(&mut self, new_instructions: String) -> Result<(), String> {
        let mut updated_cfg = self.config.clone();
        updated_cfg.custom_instructions = new_instructions.clone();

        let memories = self.storage.list_memories()?;
        let new_prompt = updated_cfg.effective_system_prompt(&memories);

        self.storage
            .set_setting(SETTING_CUSTOM_INSTRUCTIONS, &new_instructions)?;
        if let Err(err) = updated_cfg.save_to_file() {
            let _ = self.storage.set_setting(
                SETTING_CUSTOM_INSTRUCTIONS,
                &self.config.custom_instructions,
            );
            return Err(err);
        }

        self.config.custom_instructions = new_instructions;
        self.replace_system_message_in_context(new_prompt);
        Ok(())
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
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn send_user_message(&mut self, user_input: &str) -> Result<String, String> {
        self.send_user_message_inner(user_input, None)
    }

    /// Versão em streaming de [`Self::send_user_message`]: transmite cada fragmento recebido
    /// do Ollama via `on_chunk` e aplica os guardrails determinísticos antes de persistir.
    pub fn send_user_message_streaming<F>(
        &mut self,
        user_input: &str,
        mut on_chunk: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        self.send_user_message_inner(user_input, Some(&mut on_chunk))
    }

    fn send_user_message_inner(
        &mut self,
        user_input: &str,
        on_chunk: Option<&mut dyn FnMut(&str)>,
    ) -> Result<String, String> {
        let trimmed = user_input.trim();
        if trimmed.is_empty() {
            return Err("A mensagem não pode estar vazia.".to_string());
        }

        let is_first_user_msg = !self.context_messages.iter().any(|m| m.role == Role::User);
        let is_default_title = self.current_conversation.title == "Nova conversa";
        let new_title = if is_first_user_msg && is_default_title {
            Some(Self::summarize_title(trimmed))
        } else {
            None
        };

        let user_msg = Message::new(Role::User, trimmed);
        const MAX_CONTEXT_HISTORY_MSGS: usize = 12;
        let mut request_messages = Vec::with_capacity(MAX_CONTEXT_HISTORY_MSGS + 3);
        if let Some(sys) = self
            .context_messages
            .first()
            .filter(|m| m.role == Role::System)
        {
            request_messages.push(sys.clone());
        }
        let non_system: Vec<&Message> = self
            .context_messages
            .iter()
            .filter(|m| m.role != Role::System)
            .collect();
        let skip = non_system.len().saturating_sub(MAX_CONTEXT_HISTORY_MSGS);
        for msg in non_system.into_iter().skip(skip) {
            request_messages.push(msg.clone());
        }
        let kb_hits = crate::knowledge::retrieve_relevant_notes(trimmed, 2);
        if let Some(grounding) = crate::knowledge::format_grounding_note(&kb_hits) {
            request_messages.push(Message::new(Role::System, grounding));
        }
        request_messages.push(user_msg.clone());

        let raw_reply = self.inference.generate(
            &self.ollama,
            &self.config.model,
            &request_messages,
            trimmed,
            on_chunk,
        )?;
        let reply = crate::guardrails::apply_advisory_guardrails(trimmed, &raw_reply);

        let _ = self.storage.save_conversation_turn(
            self.current_conversation.id,
            trimmed,
            &reply,
            new_title.as_deref(),
        )?;

        if let Some(title) = new_title {
            self.current_conversation.title = title;
        }
        self.current_conversation.message_count += 2;
        self.context_messages.push(user_msg);
        self.context_messages
            .push(Message::new(Role::Assistant, reply.clone()));

        Ok(reply)
    }

    fn build_effective_system_prompt(&self) -> Result<String, String> {
        let memories = self.storage.list_memories()?;
        Ok(self.config.effective_system_prompt(&memories))
    }

    fn refresh_system_message_in_context(&mut self) -> Result<(), String> {
        let new_prompt = self.build_effective_system_prompt()?;
        self.replace_system_message_in_context(new_prompt);
        Ok(())
    }

    fn replace_system_message_in_context(&mut self, new_prompt: String) {
        if let Some(first) = self.context_messages.first_mut() {
            if first.role == Role::System {
                first.content = new_prompt;
                return;
            }
        }
        self.context_messages
            .insert(0, Message::new(Role::System, new_prompt));
    }

    fn summarize_title(first_message: &str) -> String {
        let single_line = first_message
            .lines()
            .next()
            .unwrap_or("Nova conversa")
            .trim();
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
