use crate::models::{Conversation, MemoryEntry, Message, Role};
use rusqlite::{params, Connection, ErrorCode, Result as SqlResult};
use std::fs;
use std::io;
use std::path::Path;

/// Camada de persistência local em SQLite para conversas, mensagens, configurações e memórias explícitas.
pub struct Storage {
    conn: Connection,
    db_path: String,
}

impl Storage {
    /// Abre (ou cria) o arquivo SQLite local e inicializa o esquema de tabelas de forma não destrutiva.
    pub fn new(db_path: &str) -> Result<Self, String> {
        Self::validate_db_path(db_path)?;

        let conn = Connection::open(db_path)
            .map_err(|e| Self::format_sqlite_error(db_path, "abrir ou criar o banco SQLite", &e))?;

        let storage = Self {
            conn,
            db_path: db_path.to_string(),
        };
        storage.init_schema().map_err(|e| {
            Self::format_sqlite_error(&storage.db_path, "inicializar tabelas no banco SQLite", &e)
        })?;

        Ok(storage)
    }

    /// Valida caminho, diretório pai e permissões antes de abrir o SQLite (mantendo `:memory:`).
    fn validate_db_path(db_path: &str) -> Result<(), String> {
        if db_path.trim().is_empty() {
            return Err(
                "Caminho do banco SQLite inválido: o caminho não pode estar vazio.\n\
                 Causa: nenhum arquivo foi informado para persistência local.\n\
                 Próxima ação: defina um arquivo .db válido em SERVERMIND_DB_PATH ou em database_path no config.toml."
                    .to_string(),
            );
        }

        if db_path == ":memory:" {
            return Ok(());
        }

        if db_path.contains('\0') {
            return Err(
                "Caminho do banco SQLite inválido: contém caractere nulo.\n\
                 Próxima ação: informe um caminho de arquivo válido em SERVERMIND_DB_PATH ou database_path."
                    .to_string(),
            );
        }

        if db_path.ends_with('/') || db_path.ends_with('\\') {
            return Err(format!(
                "Caminho do banco SQLite inválido: '{}' termina com separador de diretório.\n\
                 Causa: o caminho aponta para uma pasta, e não para um arquivo .db.\n\
                 Próxima ação: informe o caminho completo de um arquivo (ex.: servermind.db) em SERVERMIND_DB_PATH ou database_path.",
                db_path
            ));
        }

        let path = Path::new(db_path);
        if path.file_name().is_none() {
            return Err(format!(
                "Caminho do banco SQLite inválido: '{}'.\n\
                 Causa: o caminho não especifica um nome de arquivo válido.\n\
                 Próxima ação: informe um arquivo .db válido (ex.: servermind.db) em SERVERMIND_DB_PATH ou database_path.",
                db_path
            ));
        }

        match fs::metadata(path) {
            Ok(meta) => {
                if meta.is_dir() {
                    return Err(format!(
                        "Caminho do banco SQLite inválido: '{}' é um diretório.\n\
                         Causa: o caminho configurado aponta para uma pasta existente, não para um arquivo.\n\
                         Próxima ação: informe um arquivo .db dentro da pasta desejada em SERVERMIND_DB_PATH ou database_path.",
                        db_path
                    ));
                }
                if meta.permissions().readonly() {
                    return Err(format!(
                        "O arquivo de banco SQLite '{}' está marcado como somente leitura.\n\
                         Causa: o arquivo não permite gravações necessárias para conversas e configurações.\n\
                         Próxima ação: remova o atributo somente leitura, ajuste as permissões do arquivo ou aponte SERVERMIND_DB_PATH / database_path para outro arquivo gravável.",
                        db_path
                    ));
                }
                fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path)
                    .map_err(|e| Self::format_path_io_error(db_path, &e))?;
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                let parent = match path.parent() {
                    Some(p) if !p.as_os_str().is_empty() => p,
                    _ => Path::new("."),
                };
                match fs::metadata(parent) {
                    Ok(parent_meta) => {
                        if !parent_meta.is_dir() {
                            return Err(format!(
                                "Caminho pai inválido para o banco SQLite '{}': '{}' não é um diretório.\n\
                                 Próxima ação: ajuste SERVERMIND_DB_PATH ou database_path para um diretório existente.",
                                db_path,
                                parent.display()
                            ));
                        }
                    }
                    Err(parent_err) => {
                        return Err(Self::format_parent_io_error(db_path, parent, &parent_err));
                    }
                }
            }
            Err(err) => {
                return Err(Self::format_path_io_error(db_path, &err));
            }
        }

        Ok(())
    }

    fn format_parent_io_error(db_path: &str, parent: &Path, err: &io::Error) -> String {
        let (cause, action) = match err.kind() {
            io::ErrorKind::NotFound => (
                format!("o diretório pai '{}' não existe.", parent.display()),
                "Crie o diretório manualmente ou aponte SERVERMIND_DB_PATH / database_path para uma pasta existente.",
            ),
            io::ErrorKind::PermissionDenied => (
                format!("sem permissão para acessar o diretório pai '{}'.", parent.display()),
                "Ajuste as permissões da pasta ou escolha outro diretório gravável em SERVERMIND_DB_PATH / database_path.",
            ),
            _ => (
                format!("falha ao verificar o diretório pai '{}'.", parent.display()),
                "Verifique o caminho configurado em SERVERMIND_DB_PATH ou database_path.",
            ),
        };
        format!(
            "Não foi possível preparar o caminho do banco SQLite '{}'.\nCausa: {}\nPróxima ação: {}\nDetalhe técnico: {:?} ({})",
            db_path,
            cause,
            action,
            err.kind(),
            err
        )
    }

    fn format_path_io_error(db_path: &str, err: &io::Error) -> String {
        let (cause, action) = match err.kind() {
            io::ErrorKind::PermissionDenied => (
                "sem permissão para acessar o arquivo do banco SQLite.",
                "Verifique as permissões de leitura e gravação do arquivo/pasta ou escolha outro arquivo em SERVERMIND_DB_PATH.",
            ),
            _ => (
                "falha de entrada/saída ao verificar o caminho do banco SQLite.",
                "Confirme se o caminho em SERVERMIND_DB_PATH ou database_path é válido e acessível.",
            ),
        };
        format!(
            "Não foi possível acessar o banco SQLite '{}'.\nCausa: {}\nPróxima ação: {}\nDetalhe técnico: {:?} ({})",
            db_path,
            cause,
            action,
            err.kind(),
            err
        )
    }

    fn format_sqlite_error(db_path: &str, operation: &str, err: &rusqlite::Error) -> String {
        let (cause, action) = match err {
            rusqlite::Error::SqliteFailure(ffi_err, _) => match ffi_err.code {
                ErrorCode::ReadOnly => (
                    "o banco SQLite está em modo somente leitura ou sem permissão de escrita.",
                    "Verifique as permissões do arquivo e do diretório ou defina outro arquivo gravável em SERVERMIND_DB_PATH / database_path.",
                ),
                ErrorCode::PermissionDenied => (
                    "permissão negada ao acessar o banco SQLite.",
                    "Verifique as permissões de leitura/gravação do arquivo e da pasta ou escolha outro caminho em SERVERMIND_DB_PATH.",
                ),
                ErrorCode::CannotOpen => (
                    "não foi possível abrir ou criar o arquivo de banco SQLite.",
                    "Confirme se o diretório pai existe, se há permissão de escrita e se o caminho em SERVERMIND_DB_PATH / database_path está correto.",
                ),
                ErrorCode::NotADatabase => (
                    "o arquivo selecionado não é um banco SQLite válido ou está corrompido.",
                    "Verifique o caminho em SERVERMIND_DB_PATH / database_path ou aponte para um novo arquivo .db para iniciar um banco limpo.",
                ),
                ErrorCode::DatabaseCorrupt => (
                    "o arquivo de banco SQLite está corrompido.",
                    "Preserve uma cópia de segurança do arquivo atual e configure um banco íntegro em SERVERMIND_DB_PATH / database_path.",
                ),
                ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked => (
                    "o banco SQLite está bloqueado por outro processo ou conexão ativa.",
                    "Feche outras instâncias do ServerMind AI ou sessões abertas (como abrir-sqlite.py) e tente novamente.",
                ),
                ErrorCode::DiskFull => (
                    "espaço em disco insuficiente para gravar no banco SQLite.",
                    "Libere espaço em disco na unidade de destino e tente novamente.",
                ),
                _ => (
                    "falha retornada pelo motor SQLite.",
                    "Verifique a integridade e permissões do banco ou teste outro caminho via SERVERMIND_DB_PATH.",
                ),
            },
            _ => (
                "erro ao ler ou converter dados do banco SQLite.",
                "Verifique se o arquivo pertence a uma versão compatível do ServerMind AI ou aponte SERVERMIND_DB_PATH para outro arquivo.",
            ),
        };

        format!(
            "Erro ao {} (banco: '{}').\nCausa: {}\nPróxima ação: {}\nDetalhe técnico: {}",
            operation, db_path, cause, action, err
        )
    }

    fn err(&self, operation: &str, err: &rusqlite::Error) -> String {
        Self::format_sqlite_error(&self.db_path, operation, err)
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
            .map_err(|e| self.err(&format!("preparar leitura da configuração '{}'", key), &e))?;

        let mut rows = stmt
            .query(params![key])
            .map_err(|e| self.err(&format!("consultar configuração '{}'", key), &e))?;

        if let Some(row) = rows
            .next()
            .map_err(|e| self.err(&format!("ler linha da configuração '{}'", key), &e))?
        {
            let val: String = row
                .get(0)
                .map_err(|e| self.err(&format!("decodificar configuração '{}'", key), &e))?;
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
            .map_err(|e| self.err(&format!("salvar configuração '{}'", key), &e))?;
        Ok(())
    }

    /// Cria uma nova conversa no banco de dados e retorna o registro criado.
    pub fn create_conversation(&self, title: &str, model: &str) -> Result<Conversation, String> {
        self.conn
            .execute(
                "INSERT INTO conversations (title, model) VALUES (?1, ?2)",
                params![title, model],
            )
            .map_err(|e| self.err("criar nova conversa", &e))?;

        let id = self.conn.last_insert_rowid();
        self.get_conversation(id)?.ok_or_else(|| {
            format!(
                "Erro inesperado ao criar conversa #{} (banco: '{}').\n\
                 Causa: a conversa foi inserida, mas não foi localizada em seguida.\n\
                 Próxima ação: verifique a integridade do banco SQLite ou tente reiniciar a sessão.",
                id, self.db_path
            )
        })
    }

    /// Atualiza o título de uma conversa (por exemplo, a partir da primeira pergunta do usuário).
    #[allow(dead_code)]
    pub fn update_conversation_title(
        &self,
        conversation_id: i64,
        title: &str,
    ) -> Result<(), String> {
        self.conn
            .execute(
                "UPDATE conversations SET title = ?1 WHERE id = ?2",
                params![title, conversation_id],
            )
            .map_err(|e| {
                self.err(
                    &format!("atualizar título da conversa #{}", conversation_id),
                    &e,
                )
            })?;
        Ok(())
    }

    /// Atualiza o modelo associado à conversa atual.
    pub fn update_conversation_model(
        &self,
        conversation_id: i64,
        model: &str,
    ) -> Result<(), String> {
        self.conn
            .execute(
                "UPDATE conversations SET model = ?1 WHERE id = ?2",
                params![model, conversation_id],
            )
            .map_err(|e| {
                self.err(
                    &format!("atualizar modelo da conversa #{}", conversation_id),
                    &e,
                )
            })?;
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
            .map_err(|e| self.err("preparar consulta de conversa", &e))?;

        let mut rows = stmt
            .query(params![conversation_id])
            .map_err(|e| self.err(&format!("consultar conversa #{}", conversation_id), &e))?;

        if let Some(row) = rows
            .next()
            .map_err(|e| self.err(&format!("ler conversa #{}", conversation_id), &e))?
        {
            let conv = Conversation {
                id: row
                    .get(0)
                    .map_err(|e| self.err("decodificar ID da conversa", &e))?,
                title: row
                    .get(1)
                    .map_err(|e| self.err("decodificar título da conversa", &e))?,
                model: row
                    .get(2)
                    .map_err(|e| self.err("decodificar modelo da conversa", &e))?,
                created_at: row
                    .get(3)
                    .map_err(|e| self.err("decodificar data da conversa", &e))?,
                message_count: row
                    .get(4)
                    .map_err(|e| self.err("decodificar contagem de mensagens", &e))?,
            };
            Ok(Some(conv))
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
            .map_err(|e| self.err("listar conversas", &e))?;

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
            .map_err(|e| self.err("mapear conversas", &e))?;

        let mut list = Vec::new();
        for item in rows {
            list.push(item.map_err(|e| self.err("ler item de conversa", &e))?);
        }
        Ok(list)
    }

    /// Salva uma mensagem vinculada a uma conversa no SQLite.
    #[allow(dead_code)]
    pub fn save_message(
        &self,
        conversation_id: i64,
        role: Role,
        content: &str,
    ) -> Result<Message, String> {
        Self::insert_message(
            &self.conn,
            &self.db_path,
            conversation_id,
            role,
            content,
            &format!("salvar mensagem na conversa #{}", conversation_id),
        )
    }

    /// Grava atomicamente (em uma única transação SQLite) a mensagem do usuário,
    /// a resposta do assistente e, opcionalmente, a atualização de título da conversa.
    /// Se qualquer etapa falhar, executa `tx.rollback()` explicitamente e preserva
    /// o erro original (incluindo informação adicional segura caso o rollback também falhe).
    pub fn save_conversation_turn(
        &mut self,
        conversation_id: i64,
        user_content: &str,
        assistant_content: &str,
        new_title: Option<&str>,
    ) -> Result<(Message, Message), String> {
        let db_path = self.db_path.clone();
        let tx = self.conn.transaction().map_err(|e| {
            Self::format_sqlite_error(
                &db_path,
                &format!("iniciar transação na conversa #{}", conversation_id),
                &e,
            )
        })?;

        let step_result: Result<(Message, Message), String> = (|| {
            if let Some(title) = new_title {
                let affected = tx
                    .execute(
                        "UPDATE conversations SET title = ?1 WHERE id = ?2",
                        params![title, conversation_id],
                    )
                    .map_err(|e| {
                        Self::format_sqlite_error(
                            &db_path,
                            &format!("atualizar título da conversa #{}", conversation_id),
                            &e,
                        )
                    })?;

                if affected == 0 {
                    return Err(format!(
                        "Erro ao atualizar título da conversa #{} (banco: '{}').\n\
                         Causa: a conversa #{} não foi encontrada no banco SQLite.\n\
                         Próxima ação: inicie uma nova conversa com `:new` ou selecione uma existente em `:history`.",
                        conversation_id, db_path, conversation_id
                    ));
                }
            }

            let user_msg = Self::insert_message(
                &tx,
                &db_path,
                conversation_id,
                Role::User,
                user_content,
                &format!(
                    "salvar mensagem do administrador na conversa #{}",
                    conversation_id
                ),
            )?;

            let assistant_msg = Self::insert_message(
                &tx,
                &db_path,
                conversation_id,
                Role::Assistant,
                assistant_content,
                &format!(
                    "salvar resposta do assistente na conversa #{}",
                    conversation_id
                ),
            )?;

            Ok((user_msg, assistant_msg))
        })();

        match step_result {
            Ok(pair) => {
                tx.commit().map_err(|e| {
                    Self::format_sqlite_error(
                        &db_path,
                        &format!(
                            "confirmar transação (commit) da conversa #{}",
                            conversation_id
                        ),
                        &e,
                    )
                })?;
                Ok(pair)
            }
            Err(op_err) => match tx.rollback() {
                Ok(()) => Err(op_err),
                Err(rb_err) => {
                    let rb_msg = Self::format_sqlite_error(
                        &db_path,
                        &format!(
                            "reverter transação (rollback) da conversa #{}",
                            conversation_id
                        ),
                        &rb_err,
                    );
                    Err(format!(
                        "{}\n[ERRO ADICIONAL NO ROLLBACK]\n{}",
                        op_err, rb_msg
                    ))
                }
            },
        }
    }

    fn insert_message(
        conn: &Connection,
        db_path: &str,
        conversation_id: i64,
        role: Role,
        content: &str,
        operation_desc: &str,
    ) -> Result<Message, String> {
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?1, ?2, ?3)",
            params![conversation_id, role.as_str(), content],
        )
        .map_err(|e| Self::format_sqlite_error(db_path, operation_desc, &e))?;

        let id = conn.last_insert_rowid();
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
            .map_err(|e| self.err("preparar leitura de mensagens", &e))?;

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
                self.err(
                    &format!("carregar mensagens da conversa #{}", conversation_id),
                    &e,
                )
            })?;

        let mut messages = Vec::new();
        for msg in rows {
            messages.push(msg.map_err(|e| self.err("ler mensagem da conversa", &e))?);
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
            .map_err(|e| self.err("salvar memória explícita", &e))?;

        let id = self.conn.last_insert_rowid();
        let mut stmt = self
            .conn
            .prepare("SELECT id, content, created_at FROM memories WHERE id = ?1")
            .map_err(|e| self.err("consultar memória recém-criada", &e))?;

        stmt.query_row(params![id], |row| {
            Ok(MemoryEntry {
                id: row.get(0)?,
                content: row.get(1)?,
                created_at: row.get(2)?,
            })
        })
        .map_err(|e| self.err("ler memória recém-salva", &e))
    }

    /// Lista todas as memórias explícitas salvas no SQLite.
    pub fn list_memories(&self) -> Result<Vec<MemoryEntry>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, content, created_at FROM memories ORDER BY id ASC")
            .map_err(|e| self.err("listar memórias", &e))?;

        let rows = stmt
            .query_map([], |row| {
                Ok(MemoryEntry {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    created_at: row.get(2)?,
                })
            })
            .map_err(|e| self.err("mapear memórias", &e))?;

        let mut list = Vec::new();
        for item in rows {
            list.push(item.map_err(|e| self.err("ler memória salva", &e))?);
        }
        Ok(list)
    }

    /// Remove uma memória explícita pelo seu ID. Retorna `true` se um registro foi removido.
    pub fn delete_memory(&self, memory_id: i64) -> Result<bool, String> {
        let affected = self
            .conn
            .execute("DELETE FROM memories WHERE id = ?1", params![memory_id])
            .map_err(|e| self.err(&format!("remover memória #{}", memory_id), &e))?;
        Ok(affected > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(1);

    struct TempDirGuard {
        path: PathBuf,
    }

    impl TempDirGuard {
        fn new(prefix: &str) -> Self {
            let seq = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let dir = env::temp_dir().join(format!(
                "servermind_storage_tx_test_{}_{}_{}_{}",
                prefix,
                std::process::id(),
                seq,
                nanos
            ));
            fs::create_dir_all(&dir).expect("deve criar diretório temporário de teste");
            Self { path: dir }
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn test_save_conversation_turn_commits_messages_and_title_atomically() {
        let temp = TempDirGuard::new("commit_ok");
        let db_path = temp.path.join("turn_ok.db");
        let mut storage =
            Storage::new(db_path.to_str().unwrap()).expect("deve criar banco temporário");

        let conv = storage
            .create_conversation("Nova conversa", "llama3.2")
            .expect("deve criar conversa inicial");

        let (user_msg, assistant_msg) = storage
            .save_conversation_turn(
                conv.id,
                "Como verificar uso de memória no Linux?",
                "Use `free -h` ou `vmstat` para inspecionar a RAM e swap.",
                Some("Como verificar uso de memória no Linux?"),
            )
            .expect("deve confirmar transação completa");

        assert!(user_msg.id.is_some());
        assert!(assistant_msg.id.is_some());
        assert_eq!(user_msg.role, Role::User);
        assert_eq!(assistant_msg.role, Role::Assistant);

        let updated_conv = storage
            .get_conversation(conv.id)
            .expect("consulta ok")
            .expect("conversa existe");
        assert_eq!(
            updated_conv.title,
            "Como verificar uso de memória no Linux?"
        );
        assert_eq!(updated_conv.message_count, 2);

        let msgs = storage
            .get_messages(conv.id)
            .expect("deve listar mensagens da conversa");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[0].content, "Como verificar uso de memória no Linux?");
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(
            msgs[1].content,
            "Use `free -h` ou `vmstat` para inspecionar a RAM e swap."
        );
    }

    #[test]
    fn test_save_conversation_turn_rolls_back_when_second_insert_fails() {
        let temp = TempDirGuard::new("rollback_abort");
        let db_path = temp.path.join("turn_rollback.db");
        let mut storage =
            Storage::new(db_path.to_str().unwrap()).expect("deve criar banco temporário");

        let conv = storage
            .create_conversation("Nova conversa", "llama3.2")
            .expect("deve criar conversa inicial");

        // Trigger que permite o UPDATE do título e o INSERT do usuário, mas aborta o INSERT do assistente
        storage
            .conn
            .execute_batch(
                "CREATE TRIGGER abort_on_assistant_insert
                 BEFORE INSERT ON messages
                 WHEN NEW.role = 'assistant'
                 BEGIN
                     SELECT RAISE(ABORT, 'falha deterministica na gravacao do assistente');
                 END;",
            )
            .expect("deve criar trigger de falha na 2ª mensagem");

        let err = storage
            .save_conversation_turn(
                conv.id,
                "PERGUNTA_CONFIDENCIAL_USUARIO",
                "RESPOSTA_CONFIDENCIAL_ASSISTENTE",
                Some("TITULO_CONFIDENCIAL_NAO_SALVO"),
            )
            .expect_err("deve falhar quando a gravação da 2ª mensagem é abortada");

        assert!(err.contains("salvar resposta do assistente"));
        assert!(!err.contains("PERGUNTA_CONFIDENCIAL_USUARIO"));
        assert!(!err.contains("RESPOSTA_CONFIDENCIAL_ASSISTENTE"));
        assert!(!err.contains("TITULO_CONFIDENCIAL_NAO_SALVO"));

        // Verifica que o rollback reverteu tanto a 1ª mensagem (User) quanto a alteração de título
        let msgs = storage
            .get_messages(conv.id)
            .expect("consulta de mensagens deve funcionar após rollback");
        assert!(
            msgs.is_empty(),
            "nenhuma mensagem pode permanecer gravada após falha na 2ª mensagem"
        );

        let loaded_conv = storage
            .get_conversation(conv.id)
            .expect("consulta de conversa ok")
            .expect("conversa deve continuar existindo");
        assert_eq!(loaded_conv.title, "Nova conversa");
        assert_eq!(loaded_conv.message_count, 0);
    }

    #[test]
    fn test_save_conversation_turn_preserves_original_error_and_reports_rollback_failure() {
        let temp = TempDirGuard::new("rollback_err");
        let db_path = temp.path.join("turn_rollback_err.db");
        let mut storage =
            Storage::new(db_path.to_str().unwrap()).expect("deve criar banco temporário");

        let conv = storage
            .create_conversation("Nova conversa", "llama3.2")
            .expect("deve criar conversa inicial");

        // RAISE(ROLLBACK, ...) encerra a transação no motor SQLite durante o INSERT,
        // fazendo com que a chamada subsequente a `tx.rollback()` também retorne erro.
        storage
            .conn
            .execute_batch(
                "CREATE TRIGGER force_engine_rollback_on_assistant
                 BEFORE INSERT ON messages
                 WHEN NEW.role = 'assistant'
                 BEGIN
                     SELECT RAISE(ROLLBACK, 'erro que encerra transacao nativamente');
                 END;",
            )
            .expect("deve criar trigger RAISE(ROLLBACK)");

        let err = storage
            .save_conversation_turn(
                conv.id,
                "PERGUNTA_PRIVADA_ROLLBACK",
                "RESPOSTA_PRIVADA_ROLLBACK",
                Some("TITULO_PRIVADO_ROLLBACK"),
            )
            .expect_err("deve falhar reportando erro original e erro de rollback");

        assert!(err.contains("salvar resposta do assistente"));
        assert!(err.contains("[ERRO ADICIONAL NO ROLLBACK]"));
        assert!(err.contains("reverter transação (rollback)"));
        assert!(!err.contains("PERGUNTA_PRIVADA_ROLLBACK"));
        assert!(!err.contains("RESPOSTA_PRIVADA_ROLLBACK"));
        assert!(!err.contains("TITULO_PRIVADO_ROLLBACK"));

        let msgs = storage
            .get_messages(conv.id)
            .expect("consulta ok após falha");
        assert!(msgs.is_empty());
        let loaded_conv = storage
            .get_conversation(conv.id)
            .expect("consulta ok")
            .expect("conversa existe");
        assert_eq!(loaded_conv.title, "Nova conversa");
        assert_eq!(loaded_conv.message_count, 0);
    }
}
