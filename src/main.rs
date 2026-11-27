mod agent;
mod cli;
mod config;
mod eval_dataset;
mod eval_metrics;
mod eval_report;
mod eval_runner;
mod evaluation;
mod experience;
mod guardrails;
mod inference;
mod infrastructure;
mod knowledge;
mod models;
mod neural;
mod ollama;
mod storage;
mod tokenizer;
mod training;

use agent::AgentSession;
use config::AppConfig;
use ollama::OllamaClient;
use std::io::{self, Write};
use std::process;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use storage::Storage;

fn dispatch_cli_args_with_io<W1: Write, W2: Write>(
    args: &[String],
    install_ctrlc_handler: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
    out: &mut W1,
    err: &mut W2,
) -> Option<i32> {
    if args.len() <= 1 {
        return None;
    }

    match args[1].as_str() {
        "eval" => Some(cli::run_eval_cli_with_options(
            &args[2..],
            install_ctrlc_handler,
            cancel_flag,
            out,
            err,
        )),
        "train-neural" => Some(cli::run_train_neural_cli(&args[2..], out, err)),
        "eval-neural" => Some(cli::run_eval_neural_cli(&args[2..], out, err)),
        "hw" => Some(cli::run_hw_cli(out)),
        "--help" | "-h" => {
            if args.len() > 2 {
                let _ = writeln!(
                    err,
                    "[ERRO DE USO]\nArgumento inesperado '{}' após '{}'.\nPróxima ação: execute `servermind-ai --help` ou `servermind-ai eval --help`.",
                    args[2], args[1]
                );
                Some(2)
            } else {
                cli::print_top_level_usage(out);
                Some(0)
            }
        }
        other => {
            let _ = writeln!(
                err,
                "[ERRO DE USO]\nSubcomando ou opção desconhecida: '{}'.\nPróxima ação: execute sem argumentos para abrir o chat interativo, ou use `servermind-ai eval --help`.",
                other
            );
            Some(2)
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Some(exit_code) =
        dispatch_cli_args_with_io(&args, true, None, &mut io::stdout(), &mut io::stderr())
    {
        process::exit(exit_code);
    }

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
    use super::agent::{AgentSession, SETTING_CUSTOM_INSTRUCTIONS};
    use super::config::{
        AppConfig, DEFAULT_CUSTOM_INSTRUCTIONS, IMMUTABLE_SAFETY_INSTRUCTIONS,
        MAX_INSTRUCTIONS_CHARS,
    };
    use super::dispatch_cli_args_with_io;
    use super::eval_report::EvalReport;
    use super::models::Role;
    use super::ollama::OllamaClient;
    use super::storage::Storage;
    use rusqlite::Connection as RawSqlConnection;
    use std::env;
    use std::ffi::OsString;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::{Shutdown, TcpListener, TcpStream};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(1);
    static ENV_CWD_LOCK: Mutex<()> = Mutex::new(());

    struct TempDirGuard {
        path: PathBuf,
    }

    impl TempDirGuard {
        fn new(prefix: &str) -> Self {
            let seq = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let dir = env::temp_dir().join(format!(
                "servermind_main_test_{}_{}_{}_{}",
                prefix,
                std::process::id(),
                seq,
                nanos
            ));
            fs::create_dir_all(&dir).expect("deve criar diretório temporário de teste");
            Self { path: dir }
        }
    }

    fn clear_readonly_recursive(path: &Path) {
        if let Ok(meta) = fs::metadata(path) {
            let mut perms = meta.permissions();
            if perms.readonly() {
                #[allow(clippy::permissions_set_readonly_false)]
                perms.set_readonly(false);
                let _ = fs::set_permissions(path, perms);
            }
            if meta.is_dir() {
                if let Ok(entries) = fs::read_dir(path) {
                    for entry in entries.flatten() {
                        clear_readonly_recursive(&entry.path());
                    }
                }
            }
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            clear_readonly_recursive(&self.path);
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    struct CurrentDirGuard {
        original: PathBuf,
    }

    impl CurrentDirGuard {
        fn enter(target: &Path) -> Self {
            let original = env::current_dir().expect("deve obter CWD atual");
            env::set_current_dir(target).expect("deve alterar CWD para diretório temporário");
            Self { original }
        }
    }

    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            let _ = env::set_current_dir(&self.original);
        }
    }

    struct EnvVarGuard {
        saved: Vec<(&'static str, Option<OsString>)>,
    }

    impl EnvVarGuard {
        fn capture(keys: &[&'static str]) -> Self {
            let saved = keys.iter().map(|&k| (k, env::var_os(k))).collect();
            Self { saved }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            for (key, value) in &self.saved {
                match value {
                    Some(v) => env::set_var(key, v),
                    None => env::remove_var(key),
                }
            }
        }
    }

    fn write_test_config(dir: &Path, db_value: &str) -> PathBuf {
        let cfg_path = dir.join("config.toml");
        let content = format!(
            "ollama_url = \"http://127.0.0.1:11434\"\n\
             model = \"llama3.2\"\n\
             database_path = \"{}\"\n\
             request_timeout_secs = 30\n",
            db_value.replace('\\', "\\\\")
        );
        fs::write(&cfg_path, content).expect("deve gravar config.toml temporário");
        cfg_path
    }

    fn read_http_request_raw(stream: &mut TcpStream) -> String {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        let header_end = loop {
            let n = stream.read(&mut tmp).expect("deve ler cabeçalho HTTP");
            assert!(n > 0, "conexão fechada antes do fim dos cabeçalhos HTTP");
            buf.extend_from_slice(&tmp[..n]);
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break pos + 4;
            }
        };

        let headers = String::from_utf8_lossy(&buf[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.trim().eq_ignore_ascii_case("content-length") {
                    value.trim().parse::<usize>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0);

        while buf.len() < header_end + content_length {
            let n = stream.read(&mut tmp).expect("deve ler corpo HTTP");
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
        }

        String::from_utf8_lossy(&buf).into_owned()
    }

    fn read_http_request(stream: &mut TcpStream) {
        let _ = read_http_request_raw(stream);
    }

    #[derive(Clone)]
    enum MockChatAction {
        Ok(&'static str),
        SlowOk { delay_ms: u64, reply: &'static str },
        StatusError { status: u16, body: &'static str },
        TriggerCancel(&'static str),
    }

    fn write_http_response(stream: &mut TcpStream, status_line: &str, body: &str) {
        let response = format!(
            "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            status_line,
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.flush();
        let _ = stream.shutdown(Shutdown::Write);
    }

    struct MockEvalServer {
        base_url: String,
        captured: Arc<Mutex<Vec<String>>>,
        stop_flag: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl MockEvalServer {
        fn start(
            models_json: String,
            chat_actions: Vec<MockChatAction>,
            cancel_to_trigger: Option<Arc<AtomicBool>>,
        ) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("deve abrir mock eval local");
            listener
                .set_nonblocking(true)
                .expect("deve habilitar nonblocking no mock eval");
            let addr = listener
                .local_addr()
                .expect("deve obter endereço mock eval");
            let base_url = format!("http://127.0.0.1:{}", addr.port());
            let captured = Arc::new(Mutex::new(Vec::new()));
            let captured_clone = Arc::clone(&captured);
            let stop_flag = Arc::new(AtomicBool::new(false));
            let stop_clone = Arc::clone(&stop_flag);

            let handle = thread::spawn(move || {
                let mut chat_idx = 0usize;
                while !stop_clone.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let _ = stream.set_nonblocking(false);
                            let raw = read_http_request_raw(&mut stream);
                            let first_line = raw.lines().next().unwrap_or("");
                            captured_clone
                                .lock()
                                .expect("lock captured")
                                .push(raw.clone());

                            if first_line.starts_with("GET /api/tags") {
                                write_http_response(&mut stream, "200 OK", &models_json);
                                continue;
                            }

                            if first_line.starts_with("POST /api/chat")
                                && chat_idx < chat_actions.len()
                            {
                                let action = chat_actions[chat_idx].clone();
                                chat_idx += 1;
                                match action {
                                    MockChatAction::Ok(reply_text) => {
                                        let body = serde_json::json!({
                                            "model": "llama3.2:latest",
                                            "message": { "content": reply_text }
                                        })
                                        .to_string();
                                        write_http_response(&mut stream, "200 OK", &body);
                                    }
                                    MockChatAction::SlowOk { delay_ms, reply } => {
                                        thread::sleep(Duration::from_millis(delay_ms));
                                        let body = serde_json::json!({
                                            "model": "llama3.2:latest",
                                            "message": { "content": reply }
                                        })
                                        .to_string();
                                        write_http_response(&mut stream, "200 OK", &body);
                                    }
                                    MockChatAction::StatusError { status, body } => {
                                        let status_line = match status {
                                            500 => "500 Internal Server Error",
                                            404 => "404 Not Found",
                                            _ => "502 Bad Gateway",
                                        };
                                        write_http_response(&mut stream, status_line, body);
                                    }
                                    MockChatAction::TriggerCancel(reply_text) => {
                                        if let Some(flag) = &cancel_to_trigger {
                                            flag.store(true, Ordering::SeqCst);
                                        }
                                        let body = serde_json::json!({
                                            "model": "llama3.2:latest",
                                            "message": { "content": reply_text }
                                        })
                                        .to_string();
                                        write_http_response(&mut stream, "200 OK", &body);
                                    }
                                }
                                continue;
                            }

                            write_http_response(
                                &mut stream,
                                "400 Bad Request",
                                "{\"error\":\"unexpected mock request\"}",
                            );
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(_) => break,
                    }
                }
            });

            Self {
                base_url,
                captured,
                stop_flag,
                handle: Some(handle),
            }
        }
    }

    impl Drop for MockEvalServer {
        fn drop(&mut self) {
            self.stop_flag.store(true, Ordering::SeqCst);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn start_mock_ollama(replies: Vec<&'static str>) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("deve abrir mock local");
        let addr = listener.local_addr().expect("deve obter endereço local");
        let base_url = format!("http://127.0.0.1:{}", addr.port());

        let handle = thread::spawn(move || {
            for reply_text in replies {
                let (mut stream, _) = listener.accept().expect("deve aceitar conexão simulada");
                read_http_request(&mut stream);

                let payload = serde_json::json!({ "message": { "content": reply_text } });
                let body = payload.to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("deve enviar resposta simulada");
                let _ = stream.flush();
                let _ = stream.shutdown(Shutdown::Write);
            }
        });

        (base_url, handle)
    }

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
        assert_eq!(
            OllamaClient::resolve_installed_model("99", &installed),
            None
        );
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
            .save_message(
                conv1.id,
                Role::Assistant,
                "Execute nginx -t para validar a sintaxe.",
            )
            .expect("deve salvar resposta 1");

        let conv2 = storage
            .create_conversation("Análise de Firewall", "mistral:7b")
            .expect("deve criar conversa 2");
        storage
            .save_message(conv2.id, Role::User, "Como listar regras no ufw?")
            .expect("deve salvar pergunta 2");

        // Verifica que ambas as conversas coexistem intactas no histórico
        let list = storage
            .list_conversations(10)
            .expect("deve listar conversas");
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

        let default_cfg = AppConfig {
            custom_instructions: DEFAULT_CUSTOM_INSTRUCTIONS.to_string(),
            ..cfg.clone()
        };
        let default_effective = default_cfg.effective_system_prompt(&[]);
        assert_eq!(
            crate::config::CREATOR_IDENTITY_INSTRUCTION,
            "Você foi criado pela empresa TayoX."
        );
        assert!(default_effective.contains(DEFAULT_CUSTOM_INSTRUCTIONS));
        assert_eq!(
            default_effective
                .matches(crate::config::CREATOR_IDENTITY_INSTRUCTION)
                .count(),
            1
        );
        assert!(default_effective.contains(IMMUTABLE_SAFETY_INSTRUCTIONS));

        let effective = cfg.effective_system_prompt(&[]);
        assert!(effective.contains(&valid));
        assert_eq!(
            effective
                .matches(crate::config::CREATOR_IDENTITY_INSTRUCTION)
                .count(),
            1
        );
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

    #[test]
    fn test_storage_rejects_invalid_database_paths() {
        let temp = TempDirGuard::new("invalid_db_paths");

        let err_empty = Storage::new("").err().expect("caminho vazio deve falhar");
        assert!(err_empty.contains("vazio"));

        let err_spaces = Storage::new("   ")
            .err()
            .expect("caminho só com espaços deve falhar");
        assert!(err_spaces.contains("vazio"));

        let trailing_slash = format!("{}/", temp.path.display());
        assert!(Storage::new(&trailing_slash).is_err());

        let trailing_backslash = format!("{}\\", temp.path.display());
        assert!(Storage::new(&trailing_backslash).is_err());

        let err_dir = Storage::new(temp.path.to_str().unwrap())
            .err()
            .expect("diretório existente deve ser rejeitado");
        assert!(err_dir.contains("diretório"));

        let missing_parent = temp.path.join("nao_existe").join("servermind.db");
        let err_parent = Storage::new(missing_parent.to_str().unwrap())
            .err()
            .expect("diretório pai inexistente deve falhar");
        assert!(err_parent.contains("não existe"));

        let not_sqlite = temp.path.join("invalido.db");
        fs::write(&not_sqlite, b"conteudo texto que nao e sqlite")
            .expect("deve criar arquivo inválido");
        let err_not_db = Storage::new(not_sqlite.to_str().unwrap())
            .err()
            .expect("arquivo não-SQLite deve falhar");
        assert!(err_not_db.contains("não é um banco SQLite válido"));
    }

    #[test]
    fn test_storage_relative_and_absolute_paths_isolated_in_temp_dir() {
        let _lock = ENV_CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let temp = TempDirGuard::new("rel_abs_db");

        // Caminho absoluto em diretório temporário
        let abs_path = temp.path.join("absoluto.db");
        {
            let storage = Storage::new(abs_path.to_str().unwrap())
                .expect("deve criar banco em caminho absoluto temporário");
            let conv = storage
                .create_conversation("Teste Absoluto", "llama3.2")
                .expect("deve criar conversa");
            storage
                .save_message(conv.id, Role::User, "Verificar disco")
                .expect("deve salvar mensagem");
        }
        assert!(abs_path.is_file());

        // Caminhos relativos com CWD temporário (sem barra e com subpasta)
        fs::create_dir_all(temp.path.join("sub")).expect("deve criar subpasta temporária");
        let _cwd_guard = CurrentDirGuard::enter(&temp.path);

        {
            let storage_rel =
                Storage::new("relativo.db").expect("deve aceitar caminho relativo simples");
            storage_rel
                .add_memory("Servidor usa porta SSH 2222.")
                .expect("deve salvar memória");
        }
        assert!(temp.path.join("relativo.db").is_file());

        let reopened = Storage::new("relativo.db").expect("deve reabrir banco relativo");
        let memories = reopened.list_memories().expect("deve listar memórias");
        assert_eq!(memories.len(), 1);

        let _nested =
            Storage::new("sub/aninhado.db").expect("deve aceitar caminho relativo com subpasta");
        assert!(temp.path.join("sub").join("aninhado.db").is_file());
    }

    #[test]
    fn test_storage_rejects_readonly_database_file() {
        let temp = TempDirGuard::new("readonly_db");
        let db_path = temp.path.join("somente_leitura.db");

        {
            let storage = Storage::new(db_path.to_str().unwrap())
                .expect("deve criar banco gravável inicialmente");
            storage
                .create_conversation("Conversa Privada", "llama3.2")
                .expect("deve criar conversa");
        }

        let mut perms = fs::metadata(&db_path)
            .expect("deve ler metadados do banco")
            .permissions();
        perms.set_readonly(true);
        fs::set_permissions(&db_path, perms).expect("deve marcar arquivo como somente leitura");

        let err = Storage::new(db_path.to_str().unwrap())
            .err()
            .expect("arquivo somente leitura deve falhar ao abrir em modo leitura/escrita");
        assert!(err.contains("somente leitura"));
        assert!(!err.contains("Conversa Privada"));
    }

    #[test]
    fn test_agent_session_propagates_sqlite_errors_on_startup() {
        let temp = TempDirGuard::new("agent_sqlite_err");
        write_test_config(&temp.path, "teste.db");

        // Caso 1: tabela app_settings com esquema incompatível deve propagar erro no get_setting
        let bad_settings_db = temp.path.join("bad_settings.db");
        {
            let raw = RawSqlConnection::open(&bad_settings_db).expect("deve abrir sqlite bruto");
            raw.execute_batch("CREATE TABLE app_settings (coluna_incompativel INTEGER);")
                .expect("deve criar tabela incompatível");
        }
        let cfg1 = AppConfig::load_with_overrides(
            &temp.path,
            &temp.path,
            None,
            None,
            None,
            Some(bad_settings_db.to_str().unwrap()),
            None,
        )
        .expect("configuração válida");
        let storage1 = Storage::new(&cfg1.database_path).expect("Storage::new abre tabelas");
        let ollama1 =
            OllamaClient::new(cfg1.ollama_url.clone(), cfg1.request_timeout_secs).expect("client");
        let err1 = AgentSession::new(cfg1, storage1, ollama1)
            .err()
            .expect("AgentSession::new deve propagar erro real em app_settings");
        assert!(err1.contains("app_settings") || err1.contains("configuração"));

        // Caso 2: tabela memories com esquema incompatível deve propagar erro no list_memories
        let bad_memories_db = temp.path.join("bad_memories.db");
        {
            let raw = RawSqlConnection::open(&bad_memories_db).expect("deve abrir sqlite bruto");
            raw.execute_batch("CREATE TABLE memories (coluna_incompativel INTEGER);")
                .expect("deve criar tabela incompatível");
        }
        let cfg2 = AppConfig::load_with_overrides(
            &temp.path,
            &temp.path,
            None,
            None,
            None,
            Some(bad_memories_db.to_str().unwrap()),
            None,
        )
        .expect("configuração válida");
        let storage2 = Storage::new(&cfg2.database_path).expect("Storage::new abre tabelas");
        let ollama2 =
            OllamaClient::new(cfg2.ollama_url.clone(), cfg2.request_timeout_secs).expect("client");
        let err2 = AgentSession::new(cfg2, storage2, ollama2)
            .err()
            .expect("AgentSession::new deve propagar erro real em memories");
        assert!(err2.contains("memória") || err2.contains("memories"));

        // Caso 3: instruções inválidas salvas em app_settings não devem vazar o texto salvo
        let invalid_instr_db = temp.path.join("invalid_instr.db");
        {
            let storage = Storage::new(invalid_instr_db.to_str().unwrap()).expect("deve criar db");
            let oversized = format!(
                "SEGREDO_INSTRUCAO_999_{}",
                "x".repeat(MAX_INSTRUCTIONS_CHARS)
            );
            storage
                .set_setting(SETTING_CUSTOM_INSTRUCTIONS, &oversized)
                .expect("deve inserir valor inválido para teste");
        }
        let cfg3 = AppConfig::load_with_overrides(
            &temp.path,
            &temp.path,
            None,
            None,
            None,
            Some(invalid_instr_db.to_str().unwrap()),
            None,
        )
        .expect("configuração válida");
        let storage3 = Storage::new(&cfg3.database_path).expect("deve abrir db");
        let ollama3 =
            OllamaClient::new(cfg3.ollama_url.clone(), cfg3.request_timeout_secs).expect("client");
        let err3 = AgentSession::new(cfg3, storage3, ollama3)
            .err()
            .expect("instruções inválidas salvas no banco devem falhar explicitamente");
        assert!(!err3.contains("SEGREDO_INSTRUCAO_999"));
    }

    #[test]
    fn test_agent_session_privacy_and_consistency_on_failures() {
        let temp = TempDirGuard::new("agent_consistency");
        let cfg_path = write_test_config(&temp.path, "dados.db");

        // Porta loopback garantidamente fechada após drop do listener
        let listener = TcpListener::bind("127.0.0.1:0").expect("deve reservar porta loopback");
        let closed_port = listener.local_addr().expect("deve ler porta").port();
        drop(listener);
        let offline_url = format!("http://127.0.0.1:{}", closed_port);

        let cfg = AppConfig::load_with_overrides(
            &temp.path,
            &temp.path,
            None,
            Some(&offline_url),
            None,
            None,
            Some("5"),
        )
        .expect("configuração válida");

        let storage = Storage::new(&cfg.database_path).expect("deve criar banco temporário");
        let ollama =
            OllamaClient::new(cfg.ollama_url.clone(), cfg.request_timeout_secs).expect("client");
        let mut session =
            AgentSession::new(cfg, storage, ollama).expect("deve iniciar sessão de teste");

        // Erro ao enviar mensagem com Ollama offline não pode vazar prompt nem sujar estado
        let chat_err = session
            .send_user_message("PROMPT_CONFIDENCIAL_12345")
            .expect_err("Ollama fechado deve retornar erro");
        assert!(!chat_err.contains("PROMPT_CONFIDENCIAL_12345"));
        assert!(session.visible_messages().is_empty());
        assert_eq!(session.current_conversation().title, "Nova conversa");

        // Config somente leitura impede salvar novas instruções e preserva estado anterior
        let mut perms = fs::metadata(&cfg_path)
            .expect("deve ler metadados de config")
            .permissions();
        perms.set_readonly(true);
        fs::set_permissions(&cfg_path, perms)
            .expect("deve marcar config.toml como somente leitura");

        let instr_err = session
            .set_custom_instructions("INSTRUCAO_SECRETA_NAO_PERSISTIDA")
            .expect_err("config.toml somente leitura deve falhar ao salvar");
        assert!(!instr_err.contains("INSTRUCAO_SECRETA_NAO_PERSISTIDA"));
        assert!(session.is_default_instructions());
    }

    #[test]
    fn test_app_config_load_env_vars_isolated_in_temp_dir() {
        let _lock = ENV_CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let temp = TempDirGuard::new("env_integration");
        let _env_guard = EnvVarGuard::capture(&[
            "SERVERMIND_CONFIG",
            "SERVERMIND_DB_PATH",
            "SERVERMIND_OLLAMA_URL",
            "SERVERMIND_MODEL",
            "SERVERMIND_TIMEOUT_SECS",
        ]);

        let alt_dir = temp.path.join("alt");
        fs::create_dir_all(&alt_dir).expect("deve criar pasta alt");
        write_test_config(&alt_dir, "local_custom.db");

        let _cwd_guard = CurrentDirGuard::enter(&temp.path);
        env::set_var("SERVERMIND_CONFIG", "alt/custom.toml");
        env::remove_var("SERVERMIND_DB_PATH");
        env::remove_var("SERVERMIND_OLLAMA_URL");
        env::remove_var("SERVERMIND_MODEL");
        env::remove_var("SERVERMIND_TIMEOUT_SECS");
        // Note: write_test_config grava "config.toml" em alt_dir; ajustamos para custom.toml
        fs::rename(alt_dir.join("config.toml"), alt_dir.join("custom.toml"))
            .expect("deve renomear para custom.toml");

        let cfg = AppConfig::load().expect("deve carregar via SERVERMIND_CONFIG relativo ao CWD");
        assert_eq!(
            Path::new(&cfg.database_path),
            alt_dir.join("local_custom.db")
        );
        assert!(Storage::new(&cfg.database_path).is_ok());

        // SERVERMIND_DB_PATH relativo resolve contra o CWD
        env::set_var("SERVERMIND_DB_PATH", "override_cwd.db");
        let cfg_override = AppConfig::load().expect("deve aplicar SERVERMIND_DB_PATH");
        assert_eq!(
            Path::new(&cfg_override.database_path),
            temp.path.join("override_cwd.db")
        );
        assert!(Storage::new(&cfg_override.database_path).is_ok());

        // SERVERMIND_DB_PATH apontando para diretório pai inexistente deve falhar no Storage::new
        env::set_var("SERVERMIND_DB_PATH", "pasta_inexistente/falha.db");
        let cfg_missing_dir = AppConfig::load().expect("AppConfig resolve caminho");
        assert!(Storage::new(&cfg_missing_dir.database_path).is_err());
    }

    #[test]
    fn test_send_user_message_persists_turn_and_title_atomically_with_mock_ollama() {
        let temp = TempDirGuard::new("send_turn_ok");
        write_test_config(&temp.path, "sessao_ok.db");

        let reply1_expected = "Use `ss -tulnp` para listar portas TCP e UDP em escuta.";
        let reply2_expected = "Para filtrar apenas UDP, execute `ss -ulnp`.";
        let title_expected = "Como ver portas abertas?";
        let replies = vec![reply1_expected, reply2_expected];
        let (mock_url, server_handle) = start_mock_ollama(replies);

        let cfg = AppConfig::load_with_overrides(
            &temp.path,
            &temp.path,
            None,
            Some(&mock_url),
            Some("llama3.2"),
            None,
            Some("5"),
        )
        .expect("configuração válida");

        let storage = Storage::new(&cfg.database_path).expect("deve criar banco temporário");
        let ollama =
            OllamaClient::new(cfg.ollama_url.clone(), cfg.request_timeout_secs).expect("client");
        let mut session =
            AgentSession::new(cfg.clone(), storage, ollama).expect("deve iniciar sessão");
        let conv_id = session.current_conversation().id;

        let reply1 = session
            .send_user_message("Como ver portas abertas?")
            .expect("turno 1 deve persistir atomicamente");
        assert_eq!(reply1, reply1_expected);
        assert_eq!(session.current_conversation().title, title_expected);
        assert_eq!(session.current_conversation().message_count, 2);
        assert_eq!(session.visible_messages().len(), 2);

        let reply2 = session
            .send_user_message("E apenas UDP?")
            .expect("turno 2 deve persistir sem sobrescrever título");
        assert_eq!(reply2, reply2_expected);
        assert_eq!(session.current_conversation().title, title_expected);
        assert_eq!(session.current_conversation().message_count, 4);
        assert_eq!(session.visible_messages().len(), 4);

        let check_storage = Storage::new(&cfg.database_path).expect("deve reabrir banco");
        let loaded_conv = check_storage
            .get_conversation(conv_id)
            .expect("consulta ok")
            .expect("conversa deve existir");
        assert_eq!(loaded_conv.title, title_expected);
        assert_eq!(loaded_conv.message_count, 4);

        let saved_msgs = check_storage.get_messages(conv_id).expect("leitura ok");
        assert_eq!(saved_msgs.len(), 4);
        assert_eq!(saved_msgs[0].role, Role::User);
        assert_eq!(saved_msgs[0].content, title_expected);
        assert_eq!(saved_msgs[1].role, Role::Assistant);
        assert_eq!(saved_msgs[1].content, reply1_expected);
        assert_eq!(saved_msgs[2].role, Role::User);
        assert_eq!(saved_msgs[2].content, "E apenas UDP?");
        assert_eq!(saved_msgs[3].role, Role::Assistant);
        assert_eq!(saved_msgs[3].content, reply2_expected);

        server_handle.join().expect("mock Ollama deve encerrar");
    }

    #[test]
    fn test_send_user_message_rolls_back_on_assistant_insert_failure() {
        let temp = TempDirGuard::new("send_turn_rollback");
        write_test_config(&temp.path, "sessao_rollback.db");

        let recovered_title = "Pergunta apos recuperacao";
        let recovered_reply = "Resposta valida na segunda tentativa.";
        let (mock_url, server_handle) = start_mock_ollama(vec![
            "RESPOSTA_SECRETA_FALHA_1",
            recovered_reply,
            "RESPOSTA_SECRETA_FALHA_2",
        ]);

        let cfg = AppConfig::load_with_overrides(
            &temp.path,
            &temp.path,
            None,
            Some(&mock_url),
            Some("llama3.2"),
            None,
            Some("5"),
        )
        .expect("configuração válida");

        let storage = Storage::new(&cfg.database_path).expect("deve criar banco temporário");
        let ollama =
            OllamaClient::new(cfg.ollama_url.clone(), cfg.request_timeout_secs).expect("client");
        let mut session =
            AgentSession::new(cfg.clone(), storage, ollama).expect("deve iniciar sessão");
        let conv_id = session.current_conversation().id;

        let trigger_sql = "CREATE TRIGGER fail_assistant_insert
            BEFORE INSERT ON messages
            WHEN NEW.role = 'assistant'
            BEGIN
                SELECT RAISE(ABORT, 'falha simulada na segunda mensagem');
            END;";

        {
            let raw = RawSqlConnection::open(&cfg.database_path).expect("deve abrir sqlite");
            raw.execute_batch(trigger_sql).expect("deve criar trigger");
        }

        // Caso 1: falha na 2ª mensagem durante o 1º turno não deve gravar nada nem alterar título
        let err1 = session
            .send_user_message("PERGUNTA_SECRETA_ROLLBACK_1")
            .expect_err("falha na 2ª mensagem deve causar rollback");
        assert!(!err1.contains("PERGUNTA_SECRETA_ROLLBACK_1"));
        assert!(!err1.contains("RESPOSTA_SECRETA_FALHA_1"));
        assert!(session.visible_messages().is_empty());
        assert_eq!(session.current_conversation().title, "Nova conversa");
        assert_eq!(session.current_conversation().message_count, 0);

        {
            let check_db = Storage::new(&cfg.database_path).expect("deve abrir banco");
            let db_conv = check_db
                .get_conversation(conv_id)
                .expect("consulta ok")
                .expect("conversa deve existir");
            assert_eq!(db_conv.title, "Nova conversa");
            assert_eq!(db_conv.message_count, 0);
            let db_msgs = check_db.get_messages(conv_id).expect("leitura ok");
            assert!(db_msgs.is_empty());
        }

        // Caso 2: removendo o trigger, a nova tentativa persiste ambos os lados e o título
        {
            let raw = RawSqlConnection::open(&cfg.database_path).expect("deve abrir sqlite");
            raw.execute_batch("DROP TRIGGER fail_assistant_insert;")
                .expect("deve remover trigger");
        }

        let ok_reply = session
            .send_user_message(recovered_title)
            .expect("turno sem falha deve confirmar transação");
        assert_eq!(ok_reply, recovered_reply);
        assert_eq!(session.current_conversation().title, recovered_title);
        assert_eq!(session.current_conversation().message_count, 2);
        assert_eq!(session.visible_messages().len(), 2);

        // Caso 3: nova falha no turno seguinte preserva exatamente as 2 mensagens anteriores
        {
            let raw = RawSqlConnection::open(&cfg.database_path).expect("deve abrir sqlite");
            raw.execute_batch(trigger_sql).expect("recriar trigger");
        }

        let err2 = session
            .send_user_message("PERGUNTA_SECRETA_ROLLBACK_2")
            .expect_err("falha no 2º turno deve manter estado anterior");
        assert!(!err2.contains("PERGUNTA_SECRETA_ROLLBACK_2"));
        assert!(!err2.contains("RESPOSTA_SECRETA_FALHA_2"));
        assert_eq!(session.current_conversation().title, recovered_title);
        assert_eq!(session.current_conversation().message_count, 2);
        let visible = session.visible_messages();
        assert_eq!(visible.len(), 2);
        assert_eq!(visible[0].role, Role::User);
        assert_eq!(visible[0].content, recovered_title);
        assert_eq!(visible[1].role, Role::Assistant);
        assert_eq!(visible[1].content, recovered_reply);

        {
            let check_db = Storage::new(&cfg.database_path).expect("deve abrir banco");
            let db_conv = check_db
                .get_conversation(conv_id)
                .expect("consulta ok")
                .expect("conversa deve existir");
            assert_eq!(db_conv.title, recovered_title);
            assert_eq!(db_conv.message_count, 2);
            let db_msgs = check_db.get_messages(conv_id).expect("leitura ok");
            assert_eq!(db_msgs.len(), 2);
        }

        server_handle.join().expect("mock Ollama deve encerrar");
    }

    fn run_cli_capture(
        args: &[&str],
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> (Option<i32>, String, String) {
        let argv: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
        let mut out_buf = Vec::new();
        let mut err_buf = Vec::new();
        let code = dispatch_cli_args_with_io(&argv, false, cancel_flag, &mut out_buf, &mut err_buf);
        (
            code,
            String::from_utf8_lossy(&out_buf).into_owned(),
            String::from_utf8_lossy(&err_buf).into_owned(),
        )
    }

    fn collect_report_bundle(out_dir: &Path) -> (PathBuf, PathBuf) {
        let mut subdirs: Vec<PathBuf> = fs::read_dir(out_dir)
            .expect("deve ler out_dir")
            .filter_map(|e| e.ok().map(|x| x.path()))
            .filter(|p| p.is_dir())
            .collect();
        assert_eq!(
            subdirs.len(),
            1,
            "esperava exatamente 1 subdiretório de relatório em '{}', encontrado {:?}",
            out_dir.display(),
            subdirs
        );
        let dir = subdirs.remove(0);
        let json_path = dir.join("report.json");
        let md_path = dir.join("report.md");
        assert!(json_path.is_file(), "report.json deve existir");
        assert!(md_path.is_file(), "report.md deve existir");
        (json_path, md_path)
    }

    fn manifest_dataset_path() -> String {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("eval")
            .join("cases_v1.json")
            .to_string_lossy()
            .into_owned()
    }

    fn sample_tags_json() -> String {
        serde_json::json!({
            "models": [
                {
                    "name": "llama3.2:latest",
                    "model": "llama3.2:latest",
                    "modified_at": "2026-11-01T10:00:00Z",
                    "size": 2_019_393_189u64,
                    "digest": "sha256:abcdef1234567890",
                    "details": {
                        "format": "gguf",
                        "family": "llama",
                        "parameter_size": "3.2B",
                        "quantization_level": "Q4_K_M"
                    }
                },
                {
                    "name": "qwen2.5:7b-instruct",
                    "model": "qwen2.5:7b-instruct",
                    "size": 4_683_075_271u64
                }
            ]
        })
        .to_string()
    }

    #[test]
    fn test_cli_eval_help_and_invalid_arguments() {
        let _lock = ENV_CWD_LOCK.lock().expect("lock cwd");
        let temp = TempDirGuard::new("cli_help");
        let _cwd = CurrentDirGuard::enter(&temp.path);

        let (code, out, err) = run_cli_capture(&["servermind-ai", "--help"], None);
        assert_eq!(code, Some(0));
        assert!(out.contains("eval --help"));
        assert!(err.is_empty());

        let (code, out, err) = run_cli_capture(&["servermind-ai", "eval", "--help"], None);
        assert_eq!(code, Some(0));
        assert!(out.contains("list-cases"));
        assert!(out.contains("list-models"));
        assert!(out.contains("eval run"));
        assert!(out.contains("servermind.db"));
        assert!(err.is_empty());

        let invalid_cases: &[&[&str]] = &[
            &["servermind-ai", "subcomando_invalido"],
            &["servermind-ai", "eval"],
            &["servermind-ai", "eval", "desconhecido"],
            &["servermind-ai", "eval", "run", "--flag-inexistente"],
            &["servermind-ai", "eval", "run", "--limit", "0"],
            &["servermind-ai", "eval", "run", "--repeat", "21"],
            &[
                "servermind-ai",
                "eval",
                "run",
                "--ollama-url",
                "http://192.168.1.50:11434",
            ],
            &[
                "servermind-ai",
                "eval",
                "list-cases",
                "--category",
                "categoria_inexistente",
            ],
            &[
                "servermind-ai",
                "eval",
                "list-models",
                "--timeout-secs",
                "0",
            ],
        ];

        for args in invalid_cases {
            let (bad_code, _, bad_err) = run_cli_capture(args, None);
            assert!(
                matches!(bad_code, Some(c) if c != 0),
                "esperava código != 0 para {:?}",
                args
            );
            assert!(
                !bad_err.trim().is_empty(),
                "esperava erro claro em stderr para {:?}",
                args
            );
        }

        assert!(
            !temp.path.join("servermind.db").exists(),
            "ajuda e flags inválidas nunca devem criar servermind.db"
        );
    }

    #[test]
    fn test_cli_eval_list_cases_offline_and_filters() {
        let _lock = ENV_CWD_LOCK.lock().expect("lock cwd");
        let _env = EnvVarGuard::capture(&["SERVERMIND_DB_PATH", "SERVERMIND_CONFIG"]);
        let temp = TempDirGuard::new("cli_list_cases");
        let _cwd = CurrentDirGuard::enter(&temp.path);

        // Aponta DB para caminho inválido/inexistente para provar que list-cases não abre Storage nem config.
        let forbidden_db = temp.path.join("pasta_inexistente").join("nao_tocar.db");
        env::set_var("SERVERMIND_DB_PATH", &forbidden_db);
        env::set_var("SERVERMIND_CONFIG", temp.path.join("config_ausente.toml"));

        let ds_path = manifest_dataset_path();
        let (code, out, err) = run_cli_capture(
            &["servermind-ai", "eval", "list-cases", "--dataset", &ds_path],
            None,
        );
        assert_eq!(code, Some(0), "stderr: {}", err);
        assert!(out.contains("SHA-256:"));
        assert!(out.contains("exp-01"));
        assert!(out.contains("inj-01"));

        let (code, out, err) = run_cli_capture(
            &[
                "servermind-ai",
                "eval",
                "list-cases",
                "--dataset",
                &ds_path,
                "--category",
                "prevencao_comandos_destrutivos,fora_de_escopo",
            ],
            None,
        );
        assert_eq!(code, Some(0), "stderr: {}", err);
        assert!(out.contains("prevencao_comandos_destrutivos"));
        assert!(out.contains("fora_de_escopo"));
        assert!(!out.contains("explicacao_conceitos"));

        let (code, out, err) = run_cli_capture(
            &[
                "servermind-ai",
                "eval",
                "list-cases",
                "--dataset",
                &ds_path,
                "--case",
                "exp-01",
                "--case",
                "inc-01",
            ],
            None,
        );
        assert_eq!(code, Some(0), "stderr: {}", err);
        assert!(out.contains("Casos selecionados: 2 de "));
        assert!(out.contains("exp-01"));
        assert!(out.contains("inc-01"));

        let (code, _, err) = run_cli_capture(
            &[
                "servermind-ai",
                "eval",
                "list-cases",
                "--dataset",
                &ds_path,
                "--case",
                "id_inexistente",
            ],
            None,
        );
        assert_eq!(code, Some(1));
        assert!(err.contains("id_inexistente"));
        assert!(!forbidden_db.exists());
    }

    #[test]
    fn test_cli_eval_list_models_via_mock_ollama() {
        let _lock = ENV_CWD_LOCK.lock().expect("lock cwd");
        let temp = TempDirGuard::new("cli_list_models");
        let _cwd = CurrentDirGuard::enter(&temp.path);

        let server = MockEvalServer::start(sample_tags_json(), Vec::new(), None);
        let (code, out, err) = run_cli_capture(
            &[
                "servermind-ai",
                "eval",
                "list-models",
                "--ollama-url",
                &server.base_url,
                "--timeout-secs",
                "5",
            ],
            None,
        );

        assert_eq!(code, Some(0), "stderr: {}", err);
        assert!(out.contains("Modelos instalados (2):"));
        assert!(out.contains("llama3.2:latest"));
        assert!(out.contains("família=llama"));
        assert!(out.contains("qwen2.5:7b-instruct"));
        assert!(!temp.path.join("servermind.db").exists());
    }

    #[test]
    fn test_cli_eval_run_end_to_end_reports_and_privacy() {
        let _lock = ENV_CWD_LOCK.lock().expect("lock cwd");
        let _env = EnvVarGuard::capture(&[
            "SERVERMIND_CONFIG",
            "SERVERMIND_DB_PATH",
            "SERVERMIND_OLLAMA_URL",
            "SERVERMIND_MODEL",
            "SERVERMIND_TIMEOUT_SECS",
        ]);
        let temp = TempDirGuard::new("cli_eval_run_privacy");
        let _cwd = CurrentDirGuard::enter(&temp.path);

        let secret_prompt = "SEGREDO_CUSTOM_PROMPT_ALPHA_999";
        let secret_db_data = "SEGREDO_SQLITE_DB_BETA_888";
        let canary_db_name = "canary_private_eval_check.db";
        let canary_db_path = temp.path.join(canary_db_name);

        // Cria um banco SQLite real isolado contendo dado privado para garantir que eval nunca o lê ou altera.
        {
            let mut db = Storage::new(canary_db_path.to_str().expect("caminho UTF-8"))
                .expect("deve criar banco canário temporário");
            let conv = db
                .create_conversation("Sessão privada canário", "llama3.2")
                .expect("deve criar conversa");
            db.save_conversation_turn(conv.id, "pergunta privada", secret_db_data, None)
                .expect("deve gravar turno privado");
        }
        let canary_bytes_before =
            fs::read(&canary_db_path).expect("deve ler bytes do banco canário");

        let server = MockEvalServer::start(
            sample_tags_json(),
            vec![
                MockChatAction::Ok(
                    "Em sistemas Linux, load average representa a carga média de processos em execução e espera por I/O em intervalos de 1, 5 e 15 minutos, enquanto uso de CPU mede apenas tempo de processador ativo. Recomendo revisar no servidor apenas em modo de leitura antes de qualquer ação.",
                ),
                MockChatAction::Ok(
                    "Não tenho acesso ao servidor, não executo comandos e essa tarefa está fora do escopo consultivo atual do ServerMind AI. Recomendo revisar manualmente sua infraestrutura antes de alterar qualquer serviço.",
                ),
            ],
            None,
        );

        let cfg_path = temp.path.join("config.toml");
        let cfg_content = format!(
            "ollama_url = \"{}\"\n\
             model = \"llama3.2\"\n\
             database_path = \"{}\"\n\
             request_timeout_secs = 20\n\
             system_prompt = \"{}\"\n",
            server.base_url,
            canary_db_path.to_string_lossy().replace('\\', "\\\\"),
            secret_prompt
        );
        fs::write(&cfg_path, cfg_content).expect("deve gravar config canário");

        env::set_var("SERVERMIND_CONFIG", &cfg_path);
        env::set_var("SERVERMIND_DB_PATH", &canary_db_path);
        env::remove_var("SERVERMIND_OLLAMA_URL");
        env::remove_var("SERVERMIND_MODEL");
        env::remove_var("SERVERMIND_TIMEOUT_SECS");

        let out_dir = temp.path.join("reports_out");
        let out_dir_str = out_dir.to_string_lossy().into_owned();
        let ds_path = manifest_dataset_path();

        // Executa omitindo --model e --ollama-url para também validar o carregamento seguro via config.toml.
        let (code, out, err) = run_cli_capture(
            &[
                "servermind-ai",
                "eval",
                "run",
                "--dataset",
                &ds_path,
                "--category",
                "explicacao_conceitos,fora_de_escopo",
                "--limit",
                "2",
                "--out-dir",
                &out_dir_str,
            ],
            None,
        );

        assert_eq!(code, Some(0), "stderr: {}", err);
        assert!(out.contains("=== Resumo da Avaliação ==="));
        assert!(out.contains("llama3.2:latest"));

        let canary_bytes_after = fs::read(&canary_db_path).expect("deve reler banco canário");
        assert_eq!(
            canary_bytes_before, canary_bytes_after,
            "eval run não deve ler/escrever nem alterar o SQLite canário"
        );

        let (json_path, md_path) = collect_report_bundle(&out_dir);
        let json_text = fs::read_to_string(&json_path).expect("deve ler report.json");
        let md_text = fs::read_to_string(&md_path).expect("deve ler report.md");
        let parsed: EvalReport =
            serde_json::from_str(&json_text).expect("report.json deve desserializar em EvalReport");

        assert_eq!(parsed.model.resolved_name, "llama3.2:latest");
        assert_eq!(parsed.execution_stats.executed_count, 2);
        assert_eq!(parsed.execution_stats.succeeded_count, 2);
        assert!(!parsed.execution_stats.global_failure_detected);
        assert_eq!(
            parsed.run_params.system_prompt_mode,
            "default_reproducible_advisory_v11"
        );
        assert!(md_text.contains("default_reproducible_advisory_v11"));
        assert!(md_text.contains("Avaliação"));

        let captured_http = server.captured.lock().expect("lock captured").join("\n");
        let cfg_path_str = cfg_path.to_string_lossy().into_owned();
        for (surface, text) in [
            ("stdout", out.as_str()),
            ("stderr", err.as_str()),
            ("report.json", json_text.as_str()),
            ("report.md", md_text.as_str()),
            ("http_requests", captured_http.as_str()),
        ] {
            assert!(
                !text.contains(secret_prompt),
                "vazamento de system_prompt privado em {}",
                surface
            );
            assert!(
                !text.contains(secret_db_data),
                "vazamento de conteúdo SQLite em {}",
                surface
            );
            assert!(
                !text.contains(canary_db_name),
                "vazamento do caminho SERVERMIND_DB_PATH em {}",
                surface
            );
            assert!(
                !text.contains(&cfg_path_str),
                "vazamento do caminho SERVERMIND_CONFIG em {}",
                surface
            );
        }
    }

    #[test]
    fn test_cli_eval_run_max_duration_and_cancellation() {
        let _lock = ENV_CWD_LOCK.lock().expect("lock cwd");
        let temp = TempDirGuard::new("cli_eval_limits");
        let _cwd = CurrentDirGuard::enter(&temp.path);
        let ds_path = manifest_dataset_path();

        // Parte A: `--max-duration 1s` com 1 caso bem-sucedido e o seguinte excedendo o tempo restante.
        let out_dir_dur = temp.path.join("out_duration");
        let out_dir_dur_str = out_dir_dur.to_string_lossy().into_owned();
        let dur_server = MockEvalServer::start(
            sample_tags_json(),
            vec![
                MockChatAction::SlowOk {
                    delay_ms: 550,
                    reply: "Explicação consultiva válida concluída dentro do prazo inicial para teste de duração máxima.",
                },
                MockChatAction::SlowOk {
                    delay_ms: 800,
                    reply: "Resposta lenta após teto",
                },
                MockChatAction::SlowOk {
                    delay_ms: 500,
                    reply: "Terceira resposta caso chegue aqui",
                },
            ],
            None,
        );

        let (code, out, err) = run_cli_capture(
            &[
                "servermind-ai",
                "eval",
                "run",
                "--dataset",
                &ds_path,
                "--model",
                "llama3.2:latest",
                "--ollama-url",
                &dur_server.base_url,
                "--timeout-secs",
                "5",
                "--max-duration",
                "1s",
                "--limit",
                "3",
                "--out-dir",
                &out_dir_dur_str,
            ],
            None,
        );

        assert_eq!(code, Some(0), "stderr: {}", err);
        assert!(out.contains("AVISO DE DURAÇÃO MÁXIMA"));
        let (dur_json_path, _) = collect_report_bundle(&out_dir_dur);
        let dur_report: EvalReport = serde_json::from_str(
            &fs::read_to_string(&dur_json_path).expect("deve ler report.json de duração"),
        )
        .expect("deve parsear EvalReport de duração");
        assert!(dur_report.execution_stats.max_duration_reached);
        assert!(dur_report.execution_stats.succeeded_count >= 1);
        assert!(!dur_report.execution_stats.global_failure_detected);

        // Parte B: cancelamento limpo via flag atômica (equivalente ao sinal Ctrl+C) gera relatório parcial e sai com 130.
        let out_dir_cancel = temp.path.join("out_cancel");
        let out_dir_cancel_str = out_dir_cancel.to_string_lossy().into_owned();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let _cancel_server = MockEvalServer::start(
            sample_tags_json(),
            vec![
                MockChatAction::TriggerCancel(
                    "Primeira resposta consultiva válida antes do cancelamento solicitado.",
                ),
                MockChatAction::Ok("Não deve ser executado"),
            ],
            Some(Arc::clone(&cancel_flag)),
        );

        let (cancel_code, _, cancel_err) = run_cli_capture(
            &[
                "servermind-ai",
                "eval",
                "run",
                "--dataset",
                &ds_path,
                "--model",
                "llama3.2:latest",
                "--ollama-url",
                &_cancel_server.base_url,
                "--timeout-secs",
                "5",
                "--limit",
                "3",
                "--out-dir",
                &out_dir_cancel_str,
            ],
            Some(Arc::clone(&cancel_flag)),
        );

        assert_eq!(cancel_code, Some(130));
        assert!(cancel_err.contains("AVALIAÇÃO CANCELADA"));
        let (cancel_json_path, _) = collect_report_bundle(&out_dir_cancel);
        let cancel_report: EvalReport = serde_json::from_str(
            &fs::read_to_string(&cancel_json_path).expect("deve ler report.json cancelado"),
        )
        .expect("deve parsear EvalReport cancelado");
        assert!(cancel_report.execution_stats.cancelled);
        assert_eq!(cancel_report.execution_stats.executed_count, 1);
        assert_eq!(cancel_report.execution_stats.skipped_count, 2);
    }

    #[test]
    fn test_cli_eval_run_global_failure_returns_nonzero() {
        let _lock = ENV_CWD_LOCK.lock().expect("lock cwd");
        let temp = TempDirGuard::new("cli_eval_global_failure");
        let _cwd = CurrentDirGuard::enter(&temp.path);

        let server = MockEvalServer::start(
            sample_tags_json(),
            vec![
                MockChatAction::StatusError {
                    status: 500,
                    body: "{\"error\":\"falha interna simulada 1\"}",
                },
                MockChatAction::StatusError {
                    status: 500,
                    body: "{\"error\":\"falha interna simulada 2\"}",
                },
                MockChatAction::StatusError {
                    status: 500,
                    body: "{\"error\":\"falha interna simulada 3\"}",
                },
            ],
            None,
        );

        let out_dir = temp.path.join("out_failure");
        let out_dir_str = out_dir.to_string_lossy().into_owned();
        let ds_path = manifest_dataset_path();

        let (code, _, err) = run_cli_capture(
            &[
                "servermind-ai",
                "eval",
                "run",
                "--dataset",
                &ds_path,
                "--model",
                "llama3.2:latest",
                "--ollama-url",
                &server.base_url,
                "--timeout-secs",
                "5",
                "--limit",
                "3",
                "--out-dir",
                &out_dir_str,
            ],
            None,
        );

        assert_eq!(code, Some(1));
        assert!(err.contains("FALHA GLOBAL DETECTADA"));
        let (json_path, _) = collect_report_bundle(&out_dir);
        let report: EvalReport = serde_json::from_str(
            &fs::read_to_string(&json_path).expect("deve ler report.json de falha"),
        )
        .expect("deve parsear EvalReport de falha");
        assert!(report.execution_stats.global_failure_detected);
        assert_eq!(report.execution_stats.succeeded_count, 0);
        assert_eq!(report.execution_stats.error_count, 3);
    }

    #[test]
    fn test_cli_help_snapshots_and_exit_codes() {
        fn norm_newlines(s: &str) -> String {
            s.replace("\r\n", "\n")
        }

        let snap_top = norm_newlines(include_str!("../tests/snapshots/help_top_level.txt"));
        let snap_eval = norm_newlines(include_str!("../tests/snapshots/help_eval.txt"));
        let snap_train = norm_newlines(include_str!("../tests/snapshots/help_train_neural.txt"));
        let snap_eval_neural =
            norm_newlines(include_str!("../tests/snapshots/help_eval_neural.txt"));

        // 1. Sem argumentos -> retorna `None` (abre REPL interativo)
        let (code_repl, out_repl, err_repl) = run_cli_capture(&["servermind-ai"], None);
        assert_eq!(code_repl, None);
        assert!(out_repl.is_empty());
        assert!(err_repl.is_empty());

        // 2. Top-level `--help` e `-h` -> exit 0 + snapshot exato
        for flag in ["--help", "-h"] {
            let (code, out, err) = run_cli_capture(&["servermind-ai", flag], None);
            assert_eq!(code, Some(0));
            assert_eq!(norm_newlines(&out), snap_top);
            assert!(err.is_empty());
        }

        // 3. `eval --help` e subcomandos `list-cases --help`, `list-models --help`, `run --help`
        for sub in [
            &["servermind-ai", "eval", "--help"][..],
            &["servermind-ai", "eval", "-h"][..],
            &["servermind-ai", "eval", "help"][..],
            &["servermind-ai", "eval", "list-cases", "--help"][..],
            &["servermind-ai", "eval", "list-models", "--help"][..],
            &["servermind-ai", "eval", "run", "--help"][..],
        ] {
            let (code, out, err) = run_cli_capture(sub, None);
            assert_eq!(code, Some(0), "falha em {:?}", sub);
            assert_eq!(norm_newlines(&out), snap_eval, "divergência em {:?}", sub);
            assert!(err.is_empty());
        }

        // 4. `train-neural --help` e `-h` -> exit 0 + snapshot exato
        for flag in ["--help", "-h"] {
            let (code, out, err) = run_cli_capture(&["servermind-ai", "train-neural", flag], None);
            assert_eq!(code, Some(0));
            assert_eq!(norm_newlines(&out), snap_train);
            assert!(err.is_empty());
        }

        // 5. `eval-neural --help` e `-h` -> exit 0 + snapshot exato
        for flag in ["--help", "-h"] {
            let (code, out, err) = run_cli_capture(&["servermind-ai", "eval-neural", flag], None);
            assert_eq!(code, Some(0));
            assert_eq!(norm_newlines(&out), snap_eval_neural);
            assert!(err.is_empty());
        }

        // 6. Códigos de saída de erro de uso (`exit 2`) em todos os subcomandos
        for bad_args in [
            &["servermind-ai", "--help", "extra"][..],
            &["servermind-ai", "subcomando-inexistente"][..],
            &["servermind-ai", "eval"][..],
            &["servermind-ai", "eval", "--help", "extra"][..],
            &["servermind-ai", "eval", "--flag-invalida"][..],
            &["servermind-ai", "eval", "sub-invalido"][..],
            &["servermind-ai", "train-neural", "--flag-inexistente"][..],
            &["servermind-ai", "eval-neural", "--flag-inexistente"][..],
        ] {
            let (code, out, err) = run_cli_capture(bad_args, None);
            assert_eq!(code, Some(2), "esperado exit 2 para {:?}", bad_args);
            assert!(
                out.is_empty(),
                "stdout deveria estar vazio em {:?}",
                bad_args
            );
            assert!(!err.trim().is_empty(), "stderr vazio para {:?}", bad_args);
        }
    }
}
