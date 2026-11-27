//! Utilitários compartilhados de isolamento de testes (diretórios temporários, CWD, variáveis
//! de ambiente, verificação de permissões e servidor HTTP local simulado do Ollama).

use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(1);
static ENV_CWD_LOCK: Mutex<()> = Mutex::new(());

/// Adquire o lock global de CWD/variáveis de ambiente sem propagar pânico (`PoisonError`)
/// caso algum teste anterior tenha falhado com o lock retido (`T6`).
pub(crate) fn lock_env_cwd() -> MutexGuard<'static, ()> {
    ENV_CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) struct TempDirGuard {
    pub(crate) path: PathBuf,
}

impl TempDirGuard {
    pub(crate) fn new(prefix: &str) -> Self {
        let seq = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!(
            "servermind_test_{}_{}_{}_{}",
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

pub(crate) struct CurrentDirGuard {
    original: PathBuf,
}

impl CurrentDirGuard {
    pub(crate) fn enter(target: &Path) -> Self {
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

pub(crate) struct EnvVarGuard {
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvVarGuard {
    pub(crate) fn capture(keys: &[&'static str]) -> Self {
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

/// Verifica empiricamente se o sistema operacional e o usuário atual aplicam bloqueio de escrita
/// em arquivos marcados como somente leitura (`T7`: em containers Docker rodando como `root`,
/// `CAP_DAC_OVERRIDE` ignora permissões `0444`).
pub(crate) fn readonly_permissions_enforced_in_dir(dir: &Path) -> bool {
    let seq = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let probe = dir.join(format!(".readonly_probe_{}", seq));
    if fs::write(&probe, b"probe").is_err() {
        return true;
    }
    let enforced = match fs::metadata(&probe) {
        Ok(meta) => {
            let mut perms = meta.permissions();
            perms.set_readonly(true);
            if fs::set_permissions(&probe, perms).is_err() {
                true
            } else {
                fs::OpenOptions::new().write(true).open(&probe).is_err()
            }
        }
        Err(_) => true,
    };
    clear_readonly_recursive(&probe);
    let _ = fs::remove_file(&probe);
    enforced
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

#[derive(Clone)]
pub(crate) enum MockChatAction {
    Ok(&'static str),
    SlowOk { delay_ms: u64, reply: &'static str },
    StatusError { status: u16, body: &'static str },
    TriggerCancel(&'static str),
}

/// Servidor HTTP local único para simular `/api/tags` e `/api/chat` do Ollama em testes (`T9`),
/// encerrando a thread limpa e determinística no `Drop`.
pub(crate) struct MockOllamaServer {
    pub(crate) base_url: String,
    pub(crate) captured: Arc<Mutex<Vec<String>>>,
    stop_flag: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl MockOllamaServer {
    pub(crate) fn from_replies(replies: Vec<&'static str>) -> Self {
        let actions = replies.into_iter().map(MockChatAction::Ok).collect();
        Self::start("{\"models\":[]}".to_string(), actions, None)
    }

    pub(crate) fn start(
        models_json: String,
        chat_actions: Vec<MockChatAction>,
        cancel_to_trigger: Option<Arc<AtomicBool>>,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("deve abrir mock Ollama local");
        listener
            .set_nonblocking(true)
            .expect("deve habilitar nonblocking no mock Ollama");
        let addr = listener
            .local_addr()
            .expect("deve obter endereço mock Ollama");
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
                            .unwrap_or_else(|e| e.into_inner())
                            .push(raw.clone());

                        if first_line.starts_with("GET /api/tags") {
                            write_http_response(&mut stream, "200 OK", &models_json);
                            continue;
                        }

                        if first_line.starts_with("POST /api/chat") && chat_idx < chat_actions.len()
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

impl Drop for MockOllamaServer {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
