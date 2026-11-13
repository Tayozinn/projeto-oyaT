use crate::models::{Conversation, MemoryEntry, Message, Role};
use rusqlite::{params, Connection, Result as SqlResult};

/// Camada de persistência local em SQLite para conversas, mensagens, configurações e memórias explícitas.
pub struct Storage {
    conn: Connection,
}

impl Storage {
    /// Abre (ou cria) o arquivo SQLite local e inicializa o esquema de tabelas de forma não destrutiva.
    pub fn new(db_path: &str) -> Result<Self, String> {
        let conn = Connection::open(db_path).map_err(|e| {
            format!(
                "Falha ao abrir ou criar banco SQLite em '{}': {}",
                db_path, e
            )
        })?;

        let storage = Self { conn };
        storage.init_schema().map_err(|e| {
            format!(
                "Falha ao inicializar tabelas no banco SQLite '{}': {}",
                db_path, e
            )
        })?;

        Ok(storage)
    }

    fn init_schema(&self) -> SqlResult<()> {
        self.conn.execute_batch(
            "PRAGMA foreign_keys = ON;

             CREATE TABLE IF NOT EXISTS conversations (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 title TEXT NOT NULL,
                 model TEXT NOT NULL,
                 created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
             );

             CREATE TABLE IF NOT EXISTS messages (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 conversation_id INTEGER NOT NULL,
                 role TEXT NOT NULL,
                 content TEXT NOT NULL,
                 created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                 FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
             );

             CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                 ON messages(conversation_id);

             CREATE TABLE IF NOT EXISTS app_settings (
                 key TEXT PRIMARY KEY,
                 value TEXT NOT NULL,
                 updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
             );

             CREATE TABLE IF NOT EXISTS memories (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 content TEXT NOT NULL,
                 created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
             );",
        )
    }

    /// Obtém um valor de configuração persistido localmente no SQLite.
    pub fn get_setting(&self, key: &str) -> Result<Option<String>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT value FROM app_settings WHERE key = ?1")
            .map_err(|e| format!("Erro ao consultar configuração '{}': {}", key, e))?;

        let mut rows = stmt
            .query(params![key])
            .map_err(|e| format!("Erro ao ler configuração '{}': {}", key, e))?;

        if let Some(row) = rows
            .next()
            .map_err(|e| format!("Erro ao extrair linha de configuração: {}", e))?
        {
            let val: String = row.get(0).unwrap_or_default();
            Ok(Some(val))
        } else {
            Ok(None)
        }
    }

    /// Salva ou atualiza uma configuração persistida localmente no SQLite.
    pub fn set_setting(&self, key: &str, value: &str) -> Result<(), String> {
        self.conn
            .execute(
                "INSERT INTO app_settings (key, value, updated_at)
                 VALUES (?1, ?2, datetime('now', 'localtime'))
                 ON CONFLICT(key) DO UPDATE SET
                     value = excluded.value,
                     updated_at = excluded.updated_at",
                params![key, value],
            )
            .map_err(|e| format!("Erro ao salvar configuração '{}' no SQLite: {}", key, e))?;
        Ok(())
    }

    /// Cria uma nova conversa no banco de dados e retorna o registro criado.
    pub fn create_conversation(&self, title: &str, model: &str) -> Result<Conversation, String> {
        self.conn
            .execute(
                "INSERT INTO conversations (title, model) VALUES (?1, ?2)",
                params![title, model],
            )
            .map_err(|e| format!("Erro ao criar nova conversa no SQLite: {}", e))?;

        let id = self.conn.last_insert_rowid();
        self.get_conversation(id)?.ok_or_else(|| {
            "Erro inesperado: conversa recém-criada não encontrada no SQLite.".to_string()
        })
    }

    /// Atualiza o título de uma conversa (por exemplo, a partir da primeira pergunta do usuário).
    pub fn update_conversation_title(&self, conversation_id: i64, title: &str) -> Result<(), String> {
        self.conn
            .execute(
                "UPDATE conversations SET title = ?1 WHERE id = ?2",
                params![title, conversation_id],
            )
            .map_err(|e| format!("Erro ao atualizar título da conversa #{}: {}", conversation_id, e))?;
        Ok(())
    }

    /// Atualiza o modelo associado à conversa atual.
    pub fn update_conversation_model(&self, conversation_id: i64, model: &str) -> Result<(), String> {
        self.conn
            .execute(
                "UPDATE conversations SET model = ?1 WHERE id = ?2",
                params![model, conversation_id],
            )
            .map_err(|e| format!("Erro ao atualizar modelo da conversa #{}: {}", conversation_id, e))?;
        Ok(())
    }

    /// Busca os metadados de uma conversa pelo seu ID.
    pub fn get_conversation(&self, conversation_id: i64) -> Result<Option<Conversation>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT c.id, c.title, c.model, c.created_at,
                        (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count
                 FROM conversations c
                 WHERE c.id = ?1",
            )
            .map_err(|e| format!("Erro ao preparar consulta de conversa: {}", e))?;

        let mut rows = stmt
            .query(params![conversation_id])
            .map_err(|e| format!("Erro ao consultar conversa #{}: {}", conversation_id, e))?;

        if let Some(row) = rows
            .next()
            .map_err(|e| format!("Erro ao ler linha da conversa: {}", e))?
        {
            Ok(Some(Conversation {
                id: row.get(0).unwrap_or_default(),
                title: row.get(1).unwrap_or_default(),
                model: row.get(2).unwrap_or_default(),
                created_at: row.get(3).unwrap_or_default(),
                message_count: row.get(4).unwrap_or(0),
            }))
        } else {
            Ok(None)
        }
    }

    /// Lista as conversas mais recentes armazenadas no SQLite.
    pub fn list_conversations(&self, limit: usize) -> Result<Vec<Conversation>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT c.id, c.title, c.model, c.created_at,
                        (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count
                 FROM conversations c
                 ORDER BY c.id DESC
                 LIMIT ?1",
            )
            .map_err(|e| format!("Erro ao listar conversas: {}", e))?;

        let rows = stmt
            .query_map(params![limit as i64], |row| {
                Ok(Conversation {
                    id: row.get(0)?,
                    title: row.get(1)?,
                    model: row.get(2)?,
                    created_at: row.get(3)?,
                    message_count: row.get(4)?,
                })
            })
            .map_err(|e| format!("Erro ao mapear conversas do SQLite: {}", e))?;

        let mut list = Vec::new();
        for item in rows {
            list.push(item.map_err(|e| format!("Erro ao ler conversa: {}", e))?);
        }
        Ok(list)
    }

    /// Salva uma mensagem vinculada a uma conversa no SQLite.
    pub fn save_message(
        &self,
        conversation_id: i64,
        role: Role,
        content: &str,
    ) -> Result<Message, String> {
        self.conn
            .execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?1, ?2, ?3)",
                params![conversation_id, role.as_str(), content],
            )
            .map_err(|e| format!("Erro ao salvar mensagem no SQLite: {}", e))?;

        let id = self.conn.last_insert_rowid();
        Ok(Message {
            id: Some(id),
            conversation_id: Some(conversation_id),
            role,
            content: content.to_string(),
            created_at: None,
        })
    }

    /// Carrega todas as mensagens de uma conversa em ordem cronológica.
    pub fn get_messages(&self, conversation_id: i64) -> Result<Vec<Message>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, conversation_id, role, content, created_at
                 FROM messages
                 WHERE conversation_id = ?1
                 ORDER BY id ASC",
            )
            .map_err(|e| format!("Erro ao preparar leitura de mensagens: {}", e))?;

        let rows = stmt
            .query_map(params![conversation_id], |row| {
                let role_str: String = row.get(2)?;
                Ok(Message {
                    id: Some(row.get(0)?),
                    conversation_id: Some(row.get(1)?),
                    role: Role::from_db_str(&role_str),
                    content: row.get(3)?,
                    created_at: Some(row.get(4)?),
                })
            })
            .map_err(|e| {
                format!(
                    "Erro ao carregar mensagens da conversa #{}: {}",
                    conversation_id, e
                )
            })?;

        let mut messages = Vec::new();
        for msg in rows {
            messages.push(msg.map_err(|e| format!("Erro ao ler mensagem: {}", e))?);
        }
        Ok(messages)
    }

    /// Salva uma memória explicitamente solicitada pelo administrador.
    pub fn add_memory(&self, content: &str) -> Result<MemoryEntry, String> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Err("O conteúdo da memória não pode ser vazio.".to_string());
        }

        self.conn
            .execute(
                "INSERT INTO memories (content) VALUES (?1)",
                params![trimmed],
            )
            .map_err(|e| format!("Erro ao salvar memória no SQLite: {}", e))?;

        let id = self.conn.last_insert_rowid();
        let mut stmt = self
            .conn
            .prepare("SELECT id, content, created_at FROM memories WHERE id = ?1")
            .map_err(|e| format!("Erro ao consultar memória recém-criada: {}", e))?;

        stmt.query_row(params![id], |row| {
            Ok(MemoryEntry {
                id: row.get(0)?,
                content: row.get(1)?,
                created_at: row.get(2)?,
            })
        })
        .map_err(|e| format!("Erro ao ler memória recém-salva: {}", e))
    }

    /// Lista todas as memórias explícitas salvas no SQLite.
    pub fn list_memories(&self) -> Result<Vec<MemoryEntry>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, content, created_at FROM memories ORDER BY id ASC")
            .map_err(|e| format!("Erro ao listar memórias: {}", e))?;

        let rows = stmt
            .query_map([], |row| {
                Ok(MemoryEntry {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    created_at: row.get(2)?,
                })
            })
            .map_err(|e| format!("Erro ao mapear memórias do SQLite: {}", e))?;

        let mut list = Vec::new();
        for item in rows {
            list.push(item.map_err(|e| format!("Erro ao ler memória: {}", e))?);
        }
        Ok(list)
    }

    /// Remove uma memória explícita pelo seu ID. Retorna `true` se um registro foi removido.
    pub fn delete_memory(&self, memory_id: i64) -> Result<bool, String> {
        let affected = self
            .conn
            .execute("DELETE FROM memories WHERE id = ?1", params![memory_id])
            .map_err(|e| format!("Erro ao remover memória #{}: {}", memory_id, e))?;
        Ok(affected > 0)
    }
}
