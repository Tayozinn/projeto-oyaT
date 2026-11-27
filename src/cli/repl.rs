use super::hw::handle_hw_command;
use super::quantize::handle_quantize_command;
use super::train::{run_train_neural_cli, DEFAULT_TRAIN_STEPS};
use crate::chat::agent::AgentSession;
use crate::chat::config::{
    CREATOR_IDENTITY_INSTRUCTION, IMMUTABLE_SAFETY_INSTRUCTIONS, MAX_INSTRUCTIONS_CHARS,
};
use crate::chat::models::{OllamaHealth, Role};
use crate::chat::ollama::OllamaClient;
use std::io::{self, BufRead, IsTerminal, Write};

/// Interface interativa de terminal (CLI) para o administrador.
pub fn run_terminal_loop(session: &mut AgentSession) {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut out = io::stdout();
    let mut err = io::stderr();
    run_terminal_loop_with_io(session, &mut reader, &mut out, &mut err, true);
}

/// Versão com `BufRead` e `Write` injetados do loop interativo do REPL (`T5`).
pub(crate) fn run_terminal_loop_with_io(
    session: &mut AgentSession,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    err: &mut dyn Write,
    enable_spinner: bool,
) {
    print_banner(session, out);
    print_health_report(session, out, err);

    let _ = writeln!(
        out,
        "\nDigite sua mensagem e pressione Enter, ou use um comando (ex: :help, :models, :history, :new, :instructions, :quit)."
    );

    loop {
        let model_lbl = session.active_model_label();
        let _ = write!(
            out,
            "\n[Conversa #{} | {}] Administrador > ",
            session.current_conversation().id,
            model_lbl
        );
        let _ = out.flush();

        let mut line = String::new();
        match input.read_line(&mut line) {
            Ok(0) => {
                let _ = writeln!(out, "\nEncerrando SERVERMIND AI. Até logo!");
                break;
            }
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if is_repl_command_input(trimmed) {
                    if handle_command_with_io(trimmed, session, input, out, err) {
                        break;
                    }
                    continue;
                }

                let mut spinner = SpinnerHandle::spawn(enable_spinner);
                let mut para_buf = ParagraphStreamBuffer::new(trimmed);
                let mut started_output = false;
                let stream_res = session.send_user_message_streaming(trimmed, |chunk| {
                    if let Some(ready_para) = para_buf.push_chunk(chunk) {
                        if !started_output {
                            spinner.stop_and_join();
                            let _ = writeln!(out, "\nSERVERMIND AI >");
                            started_output = true;
                        }
                        let _ = write!(out, "{}", ready_para);
                        let _ = out.flush();
                    }
                });
                spinner.stop_and_join();
                match stream_res {
                    Ok(guarded_reply) => {
                        let tail = para_buf.finish(&guarded_reply);
                        if !started_output {
                            let _ = writeln!(out, "\nSERVERMIND AI >\n{}", tail);
                        } else if !tail.is_empty() {
                            let _ = write!(out, "{}", tail);
                            if !tail.ends_with('\n') {
                                let _ = writeln!(out);
                            }
                        }
                    }
                    Err(e) => {
                        if started_output {
                            let _ = writeln!(out);
                        }
                        let _ = writeln!(err, "\n[ERRO AO PROCESSAR MENSAGEM]\n{}", e);
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(err, "\nErro ao ler entrada do terminal: {}", e);
                break;
            }
        }
    }
}

fn read_prompt_line(input: &mut dyn BufRead) -> Result<Option<String>, String> {
    let mut buf = String::new();
    match input.read_line(&mut buf) {
        Ok(0) => Ok(None),
        Ok(_) => Ok(Some(buf)),
        Err(err) => Err(format!(
            "Erro ao ler entrada do terminal: {}\nPróxima ação: verifique o terminal ou tente executar o comando novamente.",
            err
        )),
    }
}

fn print_banner(session: &AgentSession, out: &mut dyn Write) {
    let instr_status = if session.is_default_instructions() {
        "Padrão"
    } else {
        "Personalizadas"
    };
    let _ = writeln!(
        out,
        "======================================================================"
    );
    let _ = writeln!(
        out,
        "       SERVERMIND AI - Assistente Consultivo Local                    "
    );
    let _ = writeln!(
        out,
        "       Assistente Conversacional Local para Administradores           "
    );
    let _ = writeln!(
        out,
        "======================================================================"
    );
    let _ = writeln!(
        out,
        "  Origem da config       : {} ({})",
        session.config().config_source,
        session.config().config_file_path
    );
    let _ = writeln!(
        out,
        "  URL do Ollama (local)  : {}",
        session.config().ollama_url
    );
    let _ = writeln!(out, "  Modelo ativo           : {}", session.config().model);
    let _ = writeln!(
        out,
        "  Instruções do agente   : {} (+ travas de segurança ativas)",
        instr_status
    );
    let _ = writeln!(
        out,
        "  Banco SQLite local     : {}",
        session.config().database_path
    );
    let _ = writeln!(
        out,
        "  Modo de operação       : Estritamente conversacional (sem execução)"
    );
    let _ = writeln!(
        out,
        "======================================================================"
    );
}

fn print_health_report(session: &AgentSession, out: &mut dyn Write, err: &mut dyn Write) {
    let health: OllamaHealth = session.check_health();
    if health.server_reachable && health.model_installed {
        let _ = writeln!(
            out,
            "[STATUS OK] Conectado ao Ollama local ({} modelo(s) instalado(s)). Modelo ativo: '{}'.",
            health.available_models.len(),
            session.config().model
        );
    } else {
        let _ = writeln!(err, "\n[ATENÇÃO - VERIFICAÇÃO DE AMBIENTE]");
        if let Some(msg) = health.error_message {
            let _ = writeln!(err, "{}", msg);
        }
    }
}

/// Handle de spinner assíncrono controlado por canal (`A8`):
/// - Só desenha em terminais interativos (`stdout().is_terminal()`);
/// - Aguarda `200ms` antes do 1º frame (evitando *flicker* em respostas instantâneas);
/// - Usa `recv_timeout` para acordar em `< 1ms` quando `stop_and_join` é chamado antes do 1º print!
pub(crate) struct SpinnerHandle {
    stop_tx: Option<std::sync::mpsc::Sender<()>>,
    join_handle: Option<std::thread::JoinHandle<bool>>,
}

impl SpinnerHandle {
    pub(crate) fn spawn(enabled: bool) -> Self {
        if !enabled {
            return Self {
                stop_tx: None,
                join_handle: None,
            };
        }
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let draw_to_tty = io::stdout().is_terminal();
        let handle = std::thread::spawn(move || {
            let start = std::time::Instant::now();
            // Espera inicial curta: se o 1º parágrafo chegar rápido (< 200ms), nem desenha spinner
            if rx
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_ok()
                || !draw_to_tty
            {
                if !draw_to_tty {
                    let _ = rx.recv();
                }
                return false;
            }
            loop {
                let secs = start.elapsed().as_secs();
                print!("\r[SERVERMIND AI pensando... {}s]   ", secs);
                let _ = io::stdout().flush();
                match rx.recv_timeout(std::time::Duration::from_millis(250)) {
                    Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
            true
        });
        Self {
            stop_tx: Some(tx),
            join_handle: Some(handle),
        }
    }

    /// Sinaliza parada imediata pelo canal, faz `join` da thread antes de qualquer `print!` do chat
    /// e limpa a linha do spinner caso algum frame tenha sido desenhado no TTY.
    pub(crate) fn stop_and_join(&mut self) -> bool {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.join_handle.take() {
            if let Ok(true) = h.join() {
                print!("\r                                    \r");
                let _ = io::stdout().flush();
            }
            true
        } else {
            false
        }
    }
}

impl Drop for SpinnerHandle {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

fn find_safe_paragraph_cut(buf: &str) -> Option<usize> {
    let mut in_fence = false;
    let mut offset = 0usize;
    for line in buf.split_inclusive('\n') {
        let next_offset = offset + line.len();
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        }
        // Quando estamos fora de qualquer cerca ``` e encontramos uma linha em branco (fechamento de parágrafo)
        if !in_fence && line.trim().is_empty() && line.ends_with('\n') && offset > 0 {
            return Some(next_offset);
        }
        offset = next_offset;
    }
    None
}

/// Acumulador de streaming por parágrafo (`\n\n`) respeitando blocos ```` ``` ```` abertos (`A11`).
pub(crate) struct ParagraphStreamBuffer {
    user_prompt: String,
    pending: String,
    printed_so_far: String,
}

impl ParagraphStreamBuffer {
    pub(crate) fn new(user_prompt: &str) -> Self {
        Self {
            user_prompt: user_prompt.to_string(),
            pending: String::new(),
            printed_so_far: String::new(),
        }
    }

    pub(crate) fn push_chunk(&mut self, chunk: &str) -> Option<String> {
        self.pending.push_str(chunk);
        let mut emitted = String::new();
        while let Some(cut) = find_safe_paragraph_cut(&self.pending) {
            let raw_para: String = self.pending.drain(..cut).collect();
            let mut sanitized =
                crate::chat::guardrails::sanitize_streaming_paragraph(&self.user_prompt, &raw_para);
            if !sanitized.ends_with("\n\n") {
                if sanitized.ends_with('\n') {
                    sanitized.push('\n');
                } else {
                    sanitized.push_str("\n\n");
                }
            }
            self.printed_so_far.push_str(&sanitized);
            emitted.push_str(&sanitized);
        }
        if emitted.is_empty() {
            None
        } else {
            Some(emitted)
        }
    }

    pub(crate) fn finish(&mut self, final_guarded: &str) -> String {
        if self.printed_so_far.is_empty() {
            self.pending.clear();
            return final_guarded.to_string();
        }
        let mut tail = String::new();
        let rem = std::mem::take(&mut self.pending);
        if !rem.trim().is_empty() {
            let sanitized_rem =
                crate::chat::guardrails::sanitize_streaming_paragraph(&self.user_prompt, &rem);
            self.printed_so_far.push_str(&sanitized_rem);
            tail.push_str(&sanitized_rem);
        }
        let notices =
            crate::chat::guardrails::trailing_advisory_notices(final_guarded, &self.printed_so_far);
        if !notices.is_empty() {
            if !tail.is_empty() && !tail.ends_with('\n') {
                tail.push_str("\n\n");
            } else if (tail.ends_with('\n') && !tail.ends_with("\n\n"))
                || (tail.is_empty() && !self.printed_so_far.ends_with("\n\n"))
            {
                tail.push('\n');
            }
            tail.push_str(&notices);
        }
        tail
    }
}

fn is_known_repl_command_name(cmd: &str) -> bool {
    matches!(
        cmd,
        "quit"
            | "exit"
            | "sair"
            | "q"
            | "help"
            | "ajuda"
            | "h"
            | "?"
            | "models"
            | "model"
            | "modelos"
            | "modelo"
            | "history"
            | "historico"
            | "hist"
            | "load"
            | "open"
            | "carregar"
            | "abrir"
            | "new"
            | "nova"
            | "instructions"
            | "instruction"
            | "prompt"
            | "instrucoes"
            | "instrucao"
            | "remember"
            | "lembrar"
            | "memorizar"
            | "memories"
            | "memorias"
            | "memory"
            | "forget"
            | "esquecer"
            | "context"
            | "contexto"
            | "status"
            | "kb"
            | "estudo"
            | "study"
            | "backend"
            | "hw"
            | "hardware"
            | "train"
            | "treinar"
            | "exp"
            | "experiencias"
            | "quantize"
            | "quantizar"
    )
}

/// Verifica se a linha digitada no REPL deve ser tratada como comando (e não mensagem de chat).
/// Comandos com `:` sempre entram no dispatcher (para sinalizar erro caso desconheça `:foo`),
/// enquanto entradas com `/` só são comandos se a primeira palavra for um comando conhecido
/// (permitindo enviar caminhos Unix como `/var/log/nginx/error.log` normalmente ao chat).
pub(crate) fn is_repl_command_input(trimmed: &str) -> bool {
    if trimmed.starts_with(':') {
        return true;
    }
    if let Some(without_slash) = trimmed.strip_prefix('/') {
        let raw_cmd = without_slash
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_lowercase();
        return is_known_repl_command_name(&raw_cmd);
    }
    false
}

/// Processa comandos iniciados por `:` ou `/` usando `BufRead` e `Write` injetados (`T5`).
/// Retorna `true` se o programa deve encerrar.
pub(crate) fn handle_command_with_io(
    cmd_line: &str,
    session: &mut AgentSession,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> bool {
    let prefix = &cmd_line[..1];
    let without_prefix = &cmd_line[1..];
    let mut parts = without_prefix.splitn(2, char::is_whitespace);
    let raw_cmd = parts.next().unwrap_or("").to_lowercase();
    let arg = parts.next().map(|s| s.trim()).unwrap_or("");

    match raw_cmd.as_str() {
        "quit" | "exit" | "sair" | "q" => {
            let _ = writeln!(
                out,
                "Encerrando SERVERMIND AI. Histórico e configurações salvos localmente em '{}' e '{}'.",
                session.config().database_path,
                session.config().config_file_path
            );
            return true;
        }
        "help" | "ajuda" | "h" | "?" => {
            print_help(out);
        }
        "models" | "model" | "modelos" | "modelo" => {
            handle_models_command(arg, session, input, out, err);
        }
        "history" | "historico" | "hist" => {
            handle_history_command(arg, session, input, out, err);
        }
        "load" | "open" | "carregar" | "abrir" => {
            if arg.is_empty() {
                let _ = writeln!(
                    err,
                    "Uso correto: :history (para selecionar da lista) ou :load <id> (ex: :load 1)"
                );
            } else {
                reopen_conversation_by_str(arg, session, out, err);
            }
        }
        "new" | "nova" => match session.start_new_conversation() {
            Ok(conv) => {
                let _ = writeln!(
                    out,
                    "Nova conversa separada iniciada (ID #{}, modelo '{}'). As conversas anteriores permanecem intactas em :history.",
                    conv.id, conv.model
                );
            }
            Err(e) => {
                let _ = writeln!(err, "Erro ao iniciar nova conversa: {}", e);
            }
        },
        "instructions" | "instruction" | "prompt" | "instrucoes" | "instrucao" => {
            handle_instructions_command(arg, session, input, out, err);
        }
        "remember" | "lembrar" | "memorizar" => {
            if arg.is_empty() {
                let _ = writeln!(
                    err,
                    "Uso correto: :remember <informação que deseja salvar na memória explícita>"
                );
            } else {
                match session.add_explicit_memory(arg) {
                    Ok(mem) => {
                        let _ = writeln!(
                            out,
                            "Memória explícita #{} salva localmente no SQLite: \"{}\"",
                            mem.id, mem.content
                        );
                    }
                    Err(e) => {
                        let _ = writeln!(err, "Erro ao salvar memória: {}", e);
                    }
                }
            }
        }
        "memories" | "memorias" | "memory" => {
            handle_memories_command(arg, session, out, err);
        }
        "forget" | "esquecer" => {
            if arg.is_empty() {
                let _ = writeln!(
                    err,
                    "Uso correto: :forget <id_da_memoria> (consulte os IDs com :memories)"
                );
            } else {
                match arg.parse::<i64>() {
                    Ok(id) if id > 0 => match session.delete_memory(id) {
                        Ok(true) => {
                            let _ =
                                writeln!(out, "Memória explícita #{} removida com sucesso.", id);
                        }
                        Ok(false) => {
                            let _ = writeln!(
                                err,
                                "Memória #{} não encontrada. Use :memories para listar.",
                                id
                            );
                        }
                        Err(e) => {
                            let _ = writeln!(err, "Erro ao remover memória: {}", e);
                        }
                    },
                    _ => {
                        let _ = writeln!(
                            err,
                            "ID inválido '{}'. Informe um número inteiro positivo da memória.",
                            arg
                        );
                    }
                }
            }
        }
        "context" | "contexto" => {
            print_current_conversation_messages(session, out);
        }
        "status" => {
            print_health_report(session, out, err);
        }
        "kb" | "estudo" | "study" => {
            handle_kb_command(arg, session, out);
        }
        "backend" => {
            handle_backend_command(arg, session, out, err);
        }
        "hw" | "hardware" => {
            handle_hw_command(out);
        }
        "train" | "treinar" => {
            handle_train_command(arg, out, err);
        }
        "exp" | "experiencias" => {
            handle_exp_command(arg, session, out, err);
        }
        "quantize" | "quantizar" => {
            let base_dir = session.config().config_dir();
            handle_quantize_command(arg, &base_dir, out, err);
        }
        _ => {
            let _ = writeln!(
                err,
                "Comando desconhecido: '{}{}'. Digite :help (ou /ajuda) para ver os comandos disponíveis.",
                prefix, raw_cmd
            );
        }
    }

    false
}

fn print_help(out: &mut dyn Write) {
    let _ = writeln!(out, "\nComandos disponíveis (aceitam prefixo ':' ou '/'):");
    let _ = writeln!(out, "  --- Seleção de Modelos (Ollama Local) ---");
    let _ = writeln!(
        out,
        "  :models                  - Lista os modelos instalados no Ollama local e permite trocar interativamente"
    );
    let _ = writeln!(
        out,
        "  :model <nome_ou_numero>  - Troca para outro modelo instalado e persiste a escolha localmente"
    );
    let _ = writeln!(
        out,
        "  :status                  - Verifica a disponibilidade do Ollama e do modelo ativo"
    );
    let _ = writeln!(out, "\n  --- Histórico e Conversas ---");
    let _ = writeln!(
        out,
        "  :history                 - Lista conversas salvas e permite escolher uma para reabrir com seu contexto"
    );
    let _ = writeln!(
        out,
        "  :history <id>            - Reabre diretamente a conversa pelo ID informado"
    );
    let _ = writeln!(
        out,
        "  :new                     - Inicia uma nova conversa separada sem apagar as anteriores"
    );
    let _ = writeln!(
        out,
        "  :context                 - Exibe todas as mensagens da conversa ativa"
    );
    let _ = writeln!(out, "\n  --- Instruções do Assistente ---");
    let _ = writeln!(
        out,
        "  :instructions            - Exibe as instruções atuais, regras de segurança e como personalizar"
    );
    let _ = writeln!(
        out,
        "  :instructions <texto>    - Define e persiste novas instruções personalizadas (máx. {} caracteres)",
        MAX_INSTRUCTIONS_CHARS
    );
    let _ = writeln!(
        out,
        "  :instructions edit       - Abre prompt interativo para digitar novas instruções"
    );
    let _ = writeln!(
        out,
        "  :instructions reset      - Restaura as instruções padrão do assistente"
    );
    let _ = writeln!(out, "\n  --- Memórias Explícitas ---");
    let _ = writeln!(
        out,
        "  :remember <texto>        - Salva uma memória explicitamente no SQLite"
    );
    let _ = writeln!(
        out,
        "  :memories                - Lista todas as memórias salvas explicitamente"
    );
    let _ = writeln!(
        out,
        "  :forget <id>             - Remove uma memória explícita pelo ID"
    );
    let _ = writeln!(out, "\n  --- Geral ---");
    let _ = writeln!(
        out,
        "  :kb [consulta]           - Lista ou busca runbooks na base técnica local"
    );
    let _ = writeln!(
        out,
        "  :backend [nome|presets] [arg] - Alterna backend (ollama, neural, claude ou presets gratuitos como free-gemini/free-groq)"
    );
    let _ = writeln!(
        out,
        "  :hw                      - Inspeciona hardware, VRAM/Vulkan e benchmark de matmul CPU"
    );
    let _ = writeln!(
        out,
        "  :train [steps | marathon [6h]] - Treina o ServerMind Neural (padrão: {} passos; ex.: `:train marathon 6h`)",
        DEFAULT_TRAIN_STEPS
    );
    let _ = writeln!(
        out,
        "  :exp                     - Exibe estatísticas da memória em 3 camadas (experiências)"
    );
    let _ = writeln!(
        out,
        "  :quantize <src> <dst> [fp16|bf16|int8|int4] - Exporta pacote quantizado compacto SMQ1"
    );
    let _ = writeln!(
        out,
        "  :help                    - Exibe esta mensagem de ajuda"
    );
    let _ = writeln!(
        out,
        "  :quit                    - Encerra o programa com segurança"
    );
    let _ = writeln!(out, "\n  --- Avaliação Offline (CLI) ---");
    let _ = writeln!(
        out,
        "  Para executar avaliações reprodutíveis sem acessar `servermind.db`,"
    );
    let _ = writeln!(
        out,
        "  encerre o chat e execute no terminal: servermind-ai eval --help"
    );
}

fn handle_models_command(
    arg: &str,
    session: &mut AgentSession,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    err: &mut dyn Write,
) {
    if !arg.is_empty() {
        match session.select_installed_model(arg) {
            Ok(selected) => {
                let _ = writeln!(
                    out,
                    "Modelo ativo alterado para '{}' e persistido localmente (em '{}' e no SQLite).",
                    selected,
                    session.config().config_file_path
                );
            }
            Err(e) => {
                let _ = writeln!(err, "\n[NÃO FOI POSSÍVEL TROCAR DE MODELO]\n{}", e);
            }
        }
        return;
    }

    let _ = writeln!(
        out,
        "\nModelo configurado atualmente: '{}'",
        session.config().model
    );
    let _ = writeln!(
        out,
        "Consultando modelos instalados no Ollama local ({})...",
        session.config().ollama_url
    );

    let installed = match session.list_installed_models() {
        Ok(list) => list,
        Err(e) => {
            let _ = writeln!(err, "\n[OLLAMA INDISPONÍVEL]\n{}", e);
            return;
        }
    };

    if installed.is_empty() {
        let _ = writeln!(
            err,
            "\n[NENHUM MODELO INSTALADO]\n\
             O Ollama está respondendo em '{}', mas não há nenhum modelo instalado localmente.\n\
             O SERVERMIND AI não baixa modelos automaticamente.\n\
             Como resolver:\n\
             1. Abra outro terminal e execute: ollama pull llama3.2\n\
             2. Depois, execute `:models` novamente para selecioná-lo.",
            session.config().ollama_url
        );
        return;
    }

    let active_resolved =
        OllamaClient::resolve_installed_model(&session.config().model, &installed);

    let _ = writeln!(out, "\nModelos instalados localmente no Ollama:");
    for (idx, model_name) in installed.iter().enumerate() {
        let is_active = active_resolved
            .as_deref()
            .map(|act| act.eq_ignore_ascii_case(model_name))
            .unwrap_or_else(|| session.config().model.eq_ignore_ascii_case(model_name));

        let badge = if is_active { "* [ATIVO]" } else { "         " };
        let _ = writeln!(out, "  [{}] {} {}", idx + 1, badge, model_name);
    }

    if active_resolved.is_none() {
        let _ = writeln!(
            err,
            "\nAviso: O modelo configurado '{}' não consta entre os instalados acima.",
            session.config().model
        );
    }

    let _ = write!(
        out,
        "\nDigite o número (1-{}) ou nome do modelo para ativá-lo (ou pressione Enter para manter '{}'): ",
        installed.len(),
        session.config().model
    );
    let _ = out.flush();

    match read_prompt_line(input) {
        Ok(Some(choice)) => {
            let trimmed = choice.trim();
            if trimmed.is_empty() {
                let _ = writeln!(out, "Mantendo o modelo '{}'.", session.config().model);
            } else {
                match session.select_installed_model(trimmed) {
                    Ok(selected) => {
                        let _ = writeln!(
                            out,
                            "Modelo ativo alterado para '{}' e persistido localmente (em '{}' e no SQLite).",
                            selected,
                            session.config().config_file_path
                        );
                    }
                    Err(e) => {
                        let _ = writeln!(err, "\n[NÃO FOI POSSÍVEL TROCAR DE MODELO]\n{}", e);
                    }
                }
            }
        }
        Ok(None) => {
            let _ = writeln!(
                out,
                "\nEntrada encerrada. Mantendo o modelo '{}'.",
                session.config().model
            );
        }
        Err(e) => {
            let _ = writeln!(err, "\n{}", e);
        }
    }
}

fn handle_history_command(
    arg: &str,
    session: &mut AgentSession,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    err: &mut dyn Write,
) {
    if !arg.is_empty() {
        reopen_conversation_by_str(arg, session, out, err);
        return;
    }

    let list = match session.list_conversations(30) {
        Ok(convs) => convs,
        Err(e) => {
            let _ = writeln!(err, "Erro ao consultar histórico de conversas: {}", e);
            return;
        }
    };

    if list.is_empty() {
        let _ = writeln!(out, "Nenhuma conversa encontrada no banco de dados SQLite.");
        return;
    }

    let _ = writeln!(out, "\nConversas salvas localmente no SQLite:");
    for c in &list {
        let marker = if c.id == session.current_conversation().id {
            "* [ATUAL]"
        } else {
            "         "
        };
        let _ = writeln!(
            out,
            "  #{} {} | {} | modelo: {} | {} msg(s) | {}",
            c.id, marker, c.created_at, c.model, c.message_count, c.title
        );
    }

    let _ = write!(
        out,
        "\nDigite o ID da conversa (ex: {}) para reabri-la (ou pressione Enter para permanecer na #{}): ",
        list.first().map(|c| c.id).unwrap_or(1),
        session.current_conversation().id
    );
    let _ = out.flush();

    match read_prompt_line(input) {
        Ok(Some(selection)) => {
            let trimmed = selection.trim().trim_start_matches('#');
            if trimmed.is_empty() {
                let _ = writeln!(
                    out,
                    "Permanecendo na conversa atual (#{} - {}).",
                    session.current_conversation().id,
                    session.current_conversation().title
                );
            } else {
                reopen_conversation_by_str(trimmed, session, out, err);
            }
        }
        Ok(None) => {
            let _ = writeln!(
                out,
                "\nEntrada encerrada. Permanecendo na conversa atual (#{}).",
                session.current_conversation().id
            );
        }
        Err(e) => {
            let _ = writeln!(err, "\n{}", e);
        }
    }
}

fn reopen_conversation_by_str(
    raw_id: &str,
    session: &mut AgentSession,
    out: &mut dyn Write,
    err: &mut dyn Write,
) {
    let cleaned = raw_id.trim().trim_start_matches('#');
    match cleaned.parse::<i64>() {
        Ok(id) if id > 0 => match session.load_conversation(id) {
            Ok(count) => {
                let _ = writeln!(
                    out,
                    "\nConversa #{} ('{}') reaberta com sucesso! {} mensagem(ns) restaurada(s) no contexto.",
                    id,
                    session.current_conversation().title,
                    count
                );
                print_current_conversation_messages(session, out);
            }
            Err(e) => {
                let _ = writeln!(err, "{}", e);
            }
        },
        _ => {
            let _ = writeln!(
                err,
                "ID de conversa inválido: '{}'. Informe um número inteiro positivo (ex: `:history 1`).",
                raw_id
            );
        }
    }
}

fn print_current_conversation_messages(session: &AgentSession, out: &mut dyn Write) {
    let msgs = session.visible_messages();
    if msgs.is_empty() {
        let _ = writeln!(
            out,
            "A conversa atual (#{} - '{}') ainda não possui mensagens trocadas.",
            session.current_conversation().id,
            session.current_conversation().title
        );
    } else {
        let _ = writeln!(
            out,
            "\n--- Histórico da Conversa Ativa (#{} - {}) ---",
            session.current_conversation().id,
            session.current_conversation().title
        );
        for m in msgs {
            let speaker = match m.role {
                Role::User => "Administrador",
                Role::Assistant => "SERVERMIND AI",
                Role::System => "Sistema",
            };
            let _ = writeln!(out, "\n[{}]\n{}", speaker, m.content);
        }
        let _ = writeln!(out, "----------------------------------------------");
    }
}

fn handle_instructions_command(
    arg: &str,
    session: &mut AgentSession,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    err: &mut dyn Write,
) {
    let trimmed = arg.trim();

    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("show")
        || trimmed.eq_ignore_ascii_case("mostrar")
    {
        let status_label = if session.is_default_instructions() {
            "PADRÃO"
        } else {
            "PERSONALIZADAS"
        };
        let char_len = session.custom_instructions().chars().count();

        let _ = writeln!(out, "\n=== CONFIGURAÇÃO DE INSTRUÇÕES DO ASSISTENTE ===");
        let _ = writeln!(
            out,
            "Estado atual : {} ({} / {} caracteres)",
            status_label, char_len, MAX_INSTRUCTIONS_CHARS
        );
        let _ = writeln!(out, "\n[1] Instruções personalizáveis ativas:");
        let _ = writeln!(out, "  {}", session.custom_instructions());
        let _ = writeln!(out, "\n[2] Identidade fixa de criação (sempre aplicada):");
        let _ = writeln!(out, "  {}", CREATOR_IDENTITY_INSTRUCTION);
        let _ = writeln!(
            out,
            "\n[3] Diretrizes de segurança imutáveis (sempre aplicadas):"
        );
        for line in IMMUTABLE_SAFETY_INSTRUCTIONS.lines() {
            let _ = writeln!(out, "  {}", line);
        }
        let _ = writeln!(out, "\nComo alterar ou restaurar:");
        let _ = writeln!(
            out,
            "  - Definir novas instruções : :instructions <suas instruções em português>"
        );
        let _ = writeln!(out, "  - Editar interativamente   : :instructions edit");
        let _ = writeln!(
            out,
            "  - Voltar ao padrão original: :instructions reset  (ou /instrucoes padrao)"
        );
        return;
    }

    if matches!(
        trimmed.to_lowercase().as_str(),
        "reset" | "default" | "padrao" | "padrão"
    ) {
        match session.reset_custom_instructions() {
            Ok(()) => {
                let _ = writeln!(
                    out,
                    "Instruções do assistente restauradas para o padrão original e salvas em '{}' e no SQLite.",
                    session.config().config_file_path
                );
                let _ = writeln!(
                    out,
                    "As diretrizes de segurança (modo apenas conversacional) permanecem ativas."
                );
            }
            Err(e) => {
                let _ = writeln!(err, "Erro ao restaurar instruções padrão: {}", e);
            }
        }
        return;
    }

    if matches!(trimmed.to_lowercase().as_str(), "edit" | "editar" | "set") {
        let _ = writeln!(
            out,
            "\nDigite as novas instruções para o assistente (máximo de {} caracteres).",
            MAX_INSTRUCTIONS_CHARS
        );
        let _ = writeln!(
            out,
            "Dica: Para cancelar, deixe em branco e pressione Enter. Para voltar ao padrão, use `:instructions reset`."
        );
        let _ = write!(out, "Novas instruções > ");
        let _ = out.flush();

        match read_prompt_line(input) {
            Ok(Some(new_text)) => {
                let input_clean = new_text.trim();
                if input_clean.is_empty() {
                    let _ = writeln!(
                        out,
                        "Edição cancelada. As instruções atuais foram mantidas. (Use `:instructions reset` se desejar voltar ao padrão)."
                    );
                    return;
                }
                apply_new_instructions(input_clean, session, out, err);
            }
            Ok(None) => {
                let _ = writeln!(
                    out,
                    "\nEdição cancelada (entrada encerrada). As instruções atuais foram mantidas."
                );
            }
            Err(e) => {
                let _ = writeln!(err, "\n{}", e);
            }
        }
        return;
    }

    // Caso o usuário tenha escrito `:instructions set <texto>` ou `:instructions definir <texto>`
    let text_to_set = match trimmed.split_once(char::is_whitespace) {
        Some((head, rest))
            if head.eq_ignore_ascii_case("set") || head.eq_ignore_ascii_case("definir") =>
        {
            rest.trim()
        }
        _ => trimmed,
    };

    apply_new_instructions(text_to_set, session, out, err);
}

fn apply_new_instructions(
    text: &str,
    session: &mut AgentSession,
    out: &mut dyn Write,
    err: &mut dyn Write,
) {
    match session.set_custom_instructions(text) {
        Ok(()) => {
            let _ = writeln!(
                out,
                "\nInstruções personalizadas atualizadas e salvas localmente em '{}' e no SQLite!",
                session.config().config_file_path
            );
            let _ = writeln!(
                out,
                "  Novas instruções: \"{}\"",
                session.custom_instructions()
            );
            let _ = writeln!(
                out,
                "  Nota de segurança: O agente permanece estritamente conversacional e sem executar comandos."
            );
            let _ = writeln!(
                out,
                "  Para voltar às instruções padrão a qualquer momento, execute: :instructions reset"
            );
        }
        Err(e) => {
            let _ = writeln!(err, "\n[INSTRUÇÕES INVÁLIDAS]\n{}", e);
        }
    }
}

fn handle_kb_command(arg: &str, session: &AgentSession, out: &mut dyn Write) {
    let base_dir = session.config().config_dir();
    let query = arg.trim();
    if query.is_empty() {
        let entries = crate::chat::knowledge::load_kb_entries_from_dir(&base_dir);
        let _ = writeln!(
            out,
            "\nBase técnica local ({} runbooks em '{}'):",
            entries.len(),
            base_dir
                .join(crate::chat::knowledge::DEFAULT_KB_RELATIVE_PATH)
                .display()
        );
        for entry in entries {
            let _ = writeln!(
                out,
                "  - [{}] {} — {}",
                entry.id, entry.title, entry.summary
            );
        }
        let _ = writeln!(
            out,
            "Dica: use `:kb <termo>` para testar a recuperação ou edite o JSON para expandir o estudo."
        );
        return;
    }

    let hits = crate::chat::knowledge::retrieve_relevant_notes_from_dir(&base_dir, query, 3);
    if hits.is_empty() {
        let _ = writeln!(
            out,
            "\nNenhum runbook atingiu relevância mínima para '{}'. Use `:kb` para listar todos.",
            query
        );
        return;
    }
    let _ = writeln!(out, "\nRunbooks recuperados para '{}':", query);
    for hit in hits {
        let _ = writeln!(
            out,
            "  [score={}] {} ({})",
            hit.score, hit.entry.title, hit.entry.id
        );
        for line in &hit.entry.guidance {
            let _ = writeln!(out, "    {}", line);
        }
    }
}

fn handle_memories_command(
    arg: &str,
    session: &mut AgentSession,
    out: &mut dyn Write,
    err: &mut dyn Write,
) {
    let trimmed = arg.trim();
    if let Some(rest) = trimmed.strip_prefix("add ") {
        match session.add_explicit_memory(rest) {
            Ok(mem) => {
                let _ = writeln!(
                    out,
                    "Memória explícita #{} salva: \"{}\"",
                    mem.id, mem.content
                );
            }
            Err(e) => {
                let _ = writeln!(err, "Erro ao salvar memória: {}", e);
            }
        }
        return;
    }

    match session.list_memories() {
        Ok(list) => {
            if list.is_empty() {
                let _ = writeln!(
                    out,
                    "Nenhuma memória explícita salva. Use `:remember <texto>` para salvar uma informação importante."
                );
            } else {
                let _ = writeln!(out, "\nMemórias explícitas salvas localmente no SQLite:");
                for m in list {
                    let _ = writeln!(out, "  #{} [{}] {}", m.id, m.created_at, m.content);
                }
                let _ = writeln!(
                    out,
                    "Dica: Use `:remember <texto>` para adicionar ou `:forget <id>` para remover."
                );
            }
        }
        Err(e) => {
            let _ = writeln!(err, "Erro ao listar memórias: {}", e);
        }
    }
}

pub(crate) fn resolve_default_neural_checkpoint_dir(
    base_dir: &std::path::Path,
) -> std::path::PathBuf {
    const CANDIDATES: &[&str] = &[
        "checkpoints/servermind-neural-110m-int8",
        "checkpoints/servermind-neural-110m",
        "checkpoints/servermind-neural-marathon-int8",
        "checkpoints/servermind-neural-marathon",
        "checkpoints/servermind-neural-v4-int8",
        "checkpoints/servermind-neural-v4",
        "checkpoints/servermind-neural-v3",
        "checkpoints/servermind-neural-v2",
        "checkpoints/servermind-neural-v1",
    ];
    for dir in CANDIDATES {
        let p = base_dir.join(dir);
        if p.join("tokenizer.json").is_file()
            && (p.join("best.bin").is_file()
                || p.join("quantized.smq").is_file()
                || p.join("latest.bin").is_file())
        {
            return p;
        }
    }
    base_dir.join("checkpoints/servermind-neural-v1")
}

fn handle_backend_command(
    arg: &str,
    session: &mut AgentSession,
    out: &mut dyn Write,
    err: &mut dyn Write,
) {
    let base_dir = session.config().config_dir();
    let default_dir = resolve_default_neural_checkpoint_dir(&base_dir);
    let trimmed = arg.trim();
    if trimmed.is_empty() {
        let mb = session.active_backend().as_model_backend();
        let _ = writeln!(
            out,
            "Backend ativo : {} ({}) | modelo='{}' | endpoint='{}' | remoto={}",
            mb.backend_id(),
            mb.display_name(),
            session.active_model_label(),
            mb.endpoint_url(),
            mb.is_remote()
        );
        let _ = writeln!(
            out,
            "Uso local     : :backend ollama  |  :backend neural [diretorio_checkpoint] (padrão: '{}')",
            default_dir.display()
        );
        let _ = writeln!(
            out,
            "Uso remoto    : :backend claude [modelo]  |  :backend gemini [modelo]  |  :backend groq [modelo]  |  :backend mistral [modelo]  |  :backend openrouter [modelo]  |  :backend openai [modelo]"
        );
        let _ = writeln!(
            out,
            "Presets grátis: :backend presets  (lista atalhos prontos: free-gemini, free-groq, free-groq-qwen, free-mistral, free-openrouter, free-nemotron)"
        );
        let _ = writeln!(
            out,
            "Variáveis API : ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY ou SERVERMIND_OPENAI_API_KEY"
        );
        return;
    }

    if matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "presets" | "free" | "gratis" | "gratuitos" | "list"
    ) {
        let _ = writeln!(
            out,
            "{}",
            crate::chat::backend::format_free_presets_catalog()
        );
        return;
    }

    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let kind_str = parts.next().unwrap_or("");
    let opt_arg = parts.next().map(|s| s.trim()).filter(|s| !s.is_empty());

    let parsed_kind = match crate::chat::backend::BackendKind::parse(kind_str) {
        Ok(k) => k,
        Err(e) => {
            let _ = writeln!(err, "{}", e);
            return;
        }
    };

    match parsed_kind {
        crate::chat::backend::BackendKind::Ollama => {
            session.use_ollama_backend();
            let _ = writeln!(
                out,
                "Backend alterado para 'ollama' (100% local via loopback '{}', modelo '{}').",
                session.config().ollama_url,
                session.config().model
            );
        }
        crate::chat::backend::BackendKind::Neural => {
            let resolved_dir = opt_arg
                .map(|s| session.config().resolve_path(s))
                .unwrap_or(default_dir);
            match session.use_neural_backend(&resolved_dir) {
                Ok((step, best_val)) => {
                    let _ = writeln!(
                        out,
                        "Backend alterado para 'neural' (ServerMind Neural próprio | dir='{}' | step={} | best_val_loss={:.4}).",
                        resolved_dir.display(),
                        step,
                        best_val
                    );
                }
                Err(e) => {
                    let _ = writeln!(err, "[ERRO AO CARREGAR BACKEND NEURAL]\n{}", e);
                }
            }
        }
        crate::chat::backend::BackendKind::Claude => match session.use_claude_backend(opt_arg) {
            Ok((endpoint, model_name, limits)) => {
                let _ = writeln!(
                    out,
                    "{}",
                    crate::chat::backend::format_remote_backend_banner(
                        "Anthropic Claude API",
                        &endpoint,
                        &model_name,
                        limits,
                    )
                );
            }
            Err(e) => {
                let _ = writeln!(err, "[ERRO AO ATIVAR BACKEND CLAUDE]\n{}", e);
            }
        },
        remote_preset_kind => {
            let (preset, preset_model) = remote_preset_kind
                .resolve_openai_preset_and_default_model()
                .expect("OpenAI-compat preset kind");
            let effective_model_override = opt_arg.or(preset_model);
            match session.use_openai_compat_backend(preset, effective_model_override) {
                Ok((endpoint, model_name, limits)) => {
                    let _ = writeln!(
                        out,
                        "{}",
                        crate::chat::backend::format_remote_backend_banner(
                            preset.display_name(),
                            &endpoint,
                            &model_name,
                            limits,
                        )
                    );
                }
                Err(e) => {
                    let _ = writeln!(
                        err,
                        "[ERRO AO ATIVAR BACKEND REMOTO ({})]\n{}",
                        preset.backend_id(),
                        e
                    );
                }
            }
        }
    }
}

fn handle_train_command(arg: &str, out: &mut dyn Write, err: &mut dyn Write) {
    let trimmed = arg.trim();
    let args: Vec<String> = if trimmed.is_empty() {
        vec!["--steps".to_string(), DEFAULT_TRAIN_STEPS.to_string()]
    } else {
        let toks: Vec<&str> = trimmed.split_whitespace().collect();
        if toks[0].eq_ignore_ascii_case("marathon") || toks[0].eq_ignore_ascii_case("maratona") {
            let mut v = vec!["--marathon".to_string()];
            if let Some(&dur) = toks.get(1) {
                v.push("--max-duration".to_string());
                v.push(dur.to_string());
            }
            v
        } else if toks.len() == 1 && toks[0].chars().all(|c| c.is_ascii_digit()) {
            vec!["--steps".to_string(), toks[0].to_string()]
        } else {
            toks.into_iter().map(|s| s.to_string()).collect()
        }
    };
    let _ = run_train_neural_cli(&args, out, err);
}

fn handle_exp_command(arg: &str, session: &AgentSession, out: &mut dyn Write, err: &mut dyn Write) {
    let db_path = session.config().resolve_path("servermind_experiences.db");
    handle_exp_command_with_io(arg, &db_path, out, err);
}

pub(crate) fn handle_exp_command_with_io(
    arg: &str,
    db_path: &std::path::Path,
    out: &mut dyn Write,
    err: &mut dyn Write,
) {
    let store = match crate::neural::experience_store::ExperienceStore::open(db_path) {
        Ok(s) => s,
        Err(e) => {
            let _ = writeln!(err, "Erro ao abrir ExperienceStore: {}", e);
            return;
        }
    };

    let trimmed = arg.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("stats") {
        match store.stats() {
            Ok(st) => {
                let _ = writeln!(
                    out,
                    "Memória 3 Camadas ('{}'): operacional={}, experiências={} (verificadas={}), treino_aprovado={}, treino_rejeitado={}",
                    db_path.display(),
                    st.operational_count,
                    st.experience_count,
                    st.verified_experience_count,
                    st.approved_training_count,
                    st.rejected_training_count
                );
                let _ = writeln!(out, "Uso: :exp op <scope>|<key>|<value>  |  :exp add <dominio>|<problema>|<diagnostico>|<solucao>|<outcome>  |  :exp approve <id>  |  :exp export [arquivo.jsonl]");
            }
            Err(e) => {
                let _ = writeln!(err, "Erro ao consultar estatísticas: {}", e);
            }
        }
        return;
    }

    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("").to_ascii_lowercase();
    let rest = parts.next().map(|s| s.trim()).unwrap_or("");

    match sub.as_str() {
        "op" => {
            let cols: Vec<&str> = rest.split('|').map(|s| s.trim()).collect();
            if cols.len() != 3 {
                let _ = writeln!(err, "Uso: :exp op <scope> | <key> | <value>");
                return;
            }
            match store.upsert_operational(cols[0], cols[1], cols[2]) {
                Ok(()) => {
                    let _ = writeln!(
                        out,
                        "Memória operacional atualizada ({}:{}).",
                        cols[0], cols[1]
                    );
                }
                Err(e) => {
                    let _ = writeln!(err, "Erro ao gravar memória operacional: {}", e);
                }
            }
        }
        "add" => {
            let cols: Vec<&str> = rest.split('|').map(|s| s.trim()).collect();
            if cols.len() < 5 {
                let _ = writeln!(
                    err,
                    "Uso: :exp add <dominio> | <problema> | <diagnostico> | <solucao> | <outcome>"
                );
                return;
            }
            // A17: `:exp add` registra sempre como pendente de revisão (`verified = false`),
            // exigindo `:exp approve <id>` em uma etapa separada antes de entrar no dataset de treino!
            let exp_id =
                match store.record_experience(&crate::neural::experience_store::NewExperience {
                    domain: cols[0].to_string(),
                    problem: cols[1].to_string(),
                    diagnosis: cols[2].to_string(),
                    solution: cols[3].to_string(),
                    outcome: cols[4].to_string(),
                    error_notes: String::new(),
                    verified: false,
                }) {
                    Ok(id) => id,
                    Err(e) => {
                        let _ = writeln!(err, "Erro ao registrar experiência: {}", e);
                        return;
                    }
                };
            let _ = writeln!(
                out,
                "Experiência #{} registrada na Camada 2 (pendente de revisão). Revise com `:exp list` e aprove para treinamento com `:exp approve {}`.",
                exp_id, exp_id
            );
        }
        "approve" | "aprovar" => {
            let cleaned = rest.trim_start_matches('#').trim();
            let Ok(exp_id) = cleaned.parse::<i64>() else {
                let _ = writeln!(err, "Uso: :exp approve <id_da_experiencia>");
                return;
            };
            match store.approve_experience_for_training(exp_id, "exp_v1") {
                Ok(rec) if rec.approved => {
                    let _ = writeln!(
                        out,
                        "Experiência #{} revisada, verificada e APROVADA para treinamento (score={:.2}).",
                        exp_id, rec.quality_score
                    );
                }
                Ok(rec) => {
                    let _ = writeln!(
                        out,
                        "Experiência #{} revisada, mas REPROVADA no gate de qualidade: {}",
                        exp_id,
                        rec.rejection_reason.unwrap_or_default()
                    );
                }
                Err(e) => {
                    let _ = writeln!(err, "Erro ao aprovar experiência #{}: {}", exp_id, e);
                }
            }
        }
        "export" => {
            let anchor = db_path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .map(|p| p.to_path_buf())
                .unwrap_or_else(crate::chat::config::AppConfig::resolve_active_config_dir);
            let out_path = if rest.is_empty() {
                anchor.join("datasets/experiences_approved.jsonl")
            } else {
                let p = std::path::Path::new(rest);
                if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    anchor.join(p)
                }
            };
            if let Some(parent) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
                let _ = std::fs::create_dir_all(parent);
            }
            match store.export_approved_jsonl(&out_path, None) {
                Ok(n) => {
                    let _ = writeln!(
                        out,
                        "Exportados {} exemplo(s) aprovado(s) para '{}'.",
                        n,
                        out_path.display()
                    );
                }
                Err(e) => {
                    let _ = writeln!(err, "Erro ao exportar experiências: {}", e);
                }
            }
        }
        "list" => {
            if let Ok(ops) = store.list_operational(None) {
                let _ = writeln!(out, "Memória Operacional ({}):", ops.len());
                for op in ops.iter().take(10) {
                    let _ = writeln!(out, "  [{}] {} = {}", op.scope, op.key, op.value);
                }
            }
            if let Ok(exps) = store.list_experiences(10) {
                let _ = writeln!(out, "Últimas Experiências ({}):", exps.len());
                for e in &exps {
                    let _ = writeln!(
                        out,
                        "  #{} [{} | verified={} | {}] {}",
                        e.id, e.domain, e.verified, e.outcome, e.problem
                    );
                }
            }
        }
        _ => {
            let _ = writeln!(
                err,
                "Subcomando :exp desconhecido '{}'. Use :exp stats | list | op | add | approve | export.",
                sub
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::config::AppConfig;
    use crate::chat::storage::Storage;
    use crate::test_support::{
        lock_env_cwd, CurrentDirGuard, EnvVarGuard, MockChatAction, MockOllamaServer, TempDirGuard,
    };
    use std::fs;
    use std::io::Cursor;

    fn build_session_in_dir(dir: &std::path::Path, ollama_url: &str) -> AgentSession {
        let cfg_path = dir.join("config.toml");
        let content = format!(
            "ollama_url = \"{}\"\n\
             model = \"llama3.2\"\n\
             database_path = \"sessao_repl.db\"\n\
             request_timeout_secs = 10\n",
            ollama_url
        );
        fs::write(&cfg_path, content).expect("deve gravar config.toml");
        let cfg = AppConfig::load_with_overrides(dir, dir, None, None, None, None, None)
            .expect("deve carregar config");
        let storage = Storage::new(&cfg.database_path).expect("deve abrir sqlite");
        let ollama =
            OllamaClient::new(cfg.ollama_url.clone(), cfg.request_timeout_secs).expect("client");
        AgentSession::new(cfg, storage, ollama).expect("deve criar AgentSession")
    }

    #[test]
    fn test_phase2_t5_repl_writer_injected_commands_unknown_and_unix_paths() {
        let temp = TempDirGuard::new("t5_repl_io");
        let tags_json = serde_json::json!({
            "models": [{ "name": "llama3.2:latest", "model": "llama3.2:latest" }]
        })
        .to_string();
        let mock = MockOllamaServer::start(
            tags_json,
            vec![MockChatAction::Ok(
                "Inspecione `/var/log/nginx/error.log` com `tail -n 50` em modo somente leitura.",
            )],
            None,
        );
        let mut session = build_session_in_dir(&temp.path, &mock.base_url);

        let script = "\
:help\n\
/ajuda\n\
:comando_inexistente_xyz\n\
/var/log/nginx/error.log está enchendo o disco, como inspecionar?\n\
:remember Servidor usa Debian 12\n\
:memories\n\
:context\n\
:new\n\
:quit\n";
        let mut input = Cursor::new(script.as_bytes());
        let mut out = Vec::new();
        let mut err = Vec::new();

        run_terminal_loop_with_io(&mut session, &mut input, &mut out, &mut err, false);

        let stdout_str = String::from_utf8_lossy(&out);
        let stderr_str = String::from_utf8_lossy(&err);

        // B9: banner sem "Fase 0" e help exibindo o default unificado de 80 passos
        assert!(
            !stdout_str.contains("Fase 0"),
            "banner não deve mais exibir 'Fase 0': {}",
            stdout_str
        );
        assert!(
            stdout_str.contains("padrão: 80 passos"),
            "help deve documentar o default unificado de 80 passos: {}",
            stdout_str
        );
        assert_eq!(
            stdout_str
                .matches("Comandos disponíveis (aceitam prefixo ':' ou '/'):")
                .count(),
            2,
            ":help e /ajuda devem imprimir a ajuda duas vezes"
        );

        // Comando desconhecido com `:` emite aviso claro em stderr sem derrubar o loop
        assert!(
            stderr_str.contains("Comando desconhecido: ':comando_inexistente_xyz'"),
            "stderr deveria acusar comando desconhecido, obtido: {}",
            stderr_str
        );

        // A4 + T5: `/var/log/nginx/error.log ...` foi encaminhado ao chat (1 POST /api/chat), não tratado como comando!
        assert!(
            stdout_str.contains("tail -n 50"),
            "mensagem iniciada por /var/log deveria ter sido respondida pelo chat: {}",
            stdout_str
        );
        let chat_posts = mock
            .captured
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|req| req.starts_with("POST /api/chat"))
            .count();
        assert_eq!(
            chat_posts, 1,
            "exatamente 1 chamada de chat esperada (para /var/log), obtido {}",
            chat_posts
        );

        // `:remember`, `:memories`, `:context`, `:new`, `:quit` funcionaram nos writers injetados
        assert!(stdout_str.contains("Memória explícita #1 salva"));
        assert!(stdout_str.contains("Debian 12"));
        assert!(stdout_str.contains("Histórico da Conversa Ativa"));
        assert!(stdout_str.contains("Nova conversa separada iniciada"));
        assert!(stdout_str.contains("Encerrando SERVERMIND AI. Histórico e configurações salvos"));
    }

    #[test]
    fn test_phase2_b8_paths_resolve_from_config_dir_regardless_of_cwd() {
        let _lock = lock_env_cwd();
        let cfg_root = TempDirGuard::new("b8_cfg_root");
        let empty_cwd = TempDirGuard::new("b8_empty_cwd");
        let _env = EnvVarGuard::capture(&["SERVERMIND_CONFIG"]);

        // 1. Prepara `knowledge/sysadmin_kb.json` customizado e checkpoint neural dentro de `cfg_root`
        let kb_dir = cfg_root.path.join("knowledge");
        fs::create_dir_all(&kb_dir).unwrap();
        fs::write(
            kb_dir.join("sysadmin_kb.json"),
            r#"[{
                "id": "kb-b8-canary",
                "title": "Runbook Canário B8",
                "tags": ["canaryb8", "nginx"],
                "summary": "Valida resolução pelo diretório da config",
                "guidance": ["Sempre resolver pelo diretório da config."]
            }]"#,
        )
        .unwrap();

        let ckpt_dir = cfg_root.path.join("checkpoints/servermind-neural-v1");
        fs::create_dir_all(&ckpt_dir).unwrap();
        let corpus = ["<user> Olá <assistant> Olá! Posso ajudar no servidor."];
        let tok = crate::neural::tokenizer::BpeTokenizer::train_from_corpus(&corpus, 280);
        tok.save(&ckpt_dir.join("tokenizer.json")).unwrap();
        let tcfg = crate::neural::TransformerConfig {
            vocab_size: tok.vocab_size(),
            max_seq_len: 16,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
        };
        let model = crate::neural::ServerMindTransformer::new_random(tcfg, 11).unwrap();
        crate::neural::save_checkpoint(&ckpt_dir.join("best.bin"), &model, None, 3, 1.8).unwrap();

        let mock = MockOllamaServer::from_replies(vec![]);
        let mut session = build_session_in_dir(&cfg_root.path, &mock.base_url);
        std::env::set_var("SERVERMIND_CONFIG", cfg_root.path.join("config.toml"));

        // 2. Entra em um CWD completamente vazio e diferente de `cfg_root`!
        let _cwd = CurrentDirGuard::enter(&empty_cwd.path);

        let mut input = Cursor::new(b"" as &[u8]);
        let mut out = Vec::new();
        let mut err = Vec::new();

        assert!(!handle_command_with_io(
            ":kb",
            &mut session,
            &mut input,
            &mut out,
            &mut err
        ));
        assert!(!handle_command_with_io(
            ":backend neural",
            &mut session,
            &mut input,
            &mut out,
            &mut err
        ));
        assert!(!handle_command_with_io(
            ":exp add linux | Uso alto de CPU no host | Worker travado em loop | Reiniciar unidade systemd com cautela | resolved",
            &mut session,
            &mut input,
            &mut out,
            &mut err
        ));
        assert!(!handle_command_with_io(
            ":exp approve 1",
            &mut session,
            &mut input,
            &mut out,
            &mut err
        ));
        assert!(!handle_command_with_io(
            ":exp export",
            &mut session,
            &mut input,
            &mut out,
            &mut err
        ));

        let stdout_str = String::from_utf8_lossy(&out);
        let stderr_str = String::from_utf8_lossy(&err);
        assert!(stderr_str.is_empty(), "stderr inesperado: {}", stderr_str);
        assert!(
            stdout_str.contains("kb-b8-canary"),
            ":kb deveria carregar sysadmin_kb.json de cfg_root: {}",
            stdout_str
        );
        assert!(
            stdout_str.contains("Backend alterado para 'neural'"),
            ":backend neural deveria encontrar checkpoints em cfg_root: {}",
            stdout_str
        );
        assert!(
            cfg_root.path.join("servermind_experiences.db").is_file(),
            "servermind_experiences.db deveria ser gravado em cfg_root!"
        );
        assert!(
            cfg_root
                .path
                .join("datasets/experiences_approved.jsonl")
                .is_file(),
            "experiences_approved.jsonl deveria ser exportado em cfg_root!"
        );

        let cwd_entries: Vec<_> = fs::read_dir(&empty_cwd.path).unwrap().collect();
        assert!(
            cwd_entries.is_empty(),
            "nenhum arquivo deveria ser criado no CWD externo, encontrado: {:?}",
            cwd_entries
        );
    }

    #[test]
    fn test_phase3_c3_c4_repl_remote_backend_banner_and_secret_redaction() {
        use std::io::{Read, Write as IoWrite};
        use std::net::{Shutdown, TcpListener};

        let _lock = lock_env_cwd();
        let temp = TempDirGuard::new("phase3_repl_remote");
        let _env = EnvVarGuard::capture(&["ANTHROPIC_API_KEY", "SERVERMIND_ANTHROPIC_BASE_URL"]);

        // Servidor HTTP local simulando `/v1/messages` da Anthropic
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mock_claude_url = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let claude_thread = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
            let mut buf = Vec::new();
            let mut tmp = [0u8; 1024];
            let header_end = loop {
                let n = stream.read(&mut tmp).unwrap();
                buf.extend_from_slice(&tmp[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let headers = String::from_utf8_lossy(&buf[..header_end]);
            let content_len = headers
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    if k.trim().eq_ignore_ascii_case("content-length") {
                        v.trim().parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            while buf.len() < header_end + content_len {
                let n = stream.read(&mut tmp).unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let raw_req = String::from_utf8_lossy(&buf).into_owned();
            let body = serde_json::json!({
                "id": "msg_repl_1",
                "model": "claude-sonnet-4-0",
                "content": [{ "type": "text", "text": "Use `df -h` em modo somente leitura para inspecionar o disco." }]
            })
            .to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
            let _ = stream.shutdown(Shutdown::Write);
            raw_req
        });

        std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-test-key-nao-persistida");
        std::env::set_var("SERVERMIND_ANTHROPIC_BASE_URL", &mock_claude_url);

        let mock_ollama = MockOllamaServer::from_replies(vec![]);
        let mut session = build_session_in_dir(&temp.path, &mock_ollama.base_url);

        // Padrão inicial deve continuar sendo `ollama` (`C4`)
        assert_eq!(
            session.active_backend().as_model_backend().backend_id(),
            "ollama"
        );

        let script = "\
:backend claude\n\
Erro em postgres://admin:SenhaSuperSecreta123@10.0.0.9:5432/prod\n\
:backend ollama\n\
:quit\n";
        let mut input = Cursor::new(script.as_bytes());
        let mut out = Vec::new();
        let mut err = Vec::new();

        run_terminal_loop_with_io(&mut session, &mut input, &mut out, &mut err, false);

        let stdout_str = String::from_utf8_lossy(&out);
        let raw_http = claude_thread.join().unwrap();

        // Banner explícito de backend remoto (`C3` + `C4`) + prompt marcado com `[REMOTO]`
        assert!(stdout_str.contains("[AVISO DE BACKEND REMOTO ATIVO"));
        assert!(stdout_str.contains("[REMOTO]"));
        assert!(stdout_str.contains("df -h"));
        assert!(stdout_str.contains("Backend alterado para 'ollama'"));

        // Nenhum segredo vazou no payload HTTP enviado ao backend remoto nem no config.toml!
        assert!(!raw_http.contains("SenhaSuperSecreta123"));
        assert!(raw_http.contains("[SENHA_REDIGIDA]"));
        let saved_cfg = fs::read_to_string(temp.path.join("config.toml")).unwrap();
        assert!(!saved_cfg.contains("sk-ant-test-key-nao-persistida"));
    }
}
