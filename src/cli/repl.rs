use super::hw::handle_hw_command;
use super::quantize::handle_quantize_command;
use super::train::run_train_neural_cli;
use crate::chat::agent::AgentSession;
use crate::chat::config::{
    CREATOR_IDENTITY_INSTRUCTION, IMMUTABLE_SAFETY_INSTRUCTIONS, MAX_INSTRUCTIONS_CHARS,
};
use crate::chat::models::{OllamaHealth, Role};
use crate::chat::ollama::OllamaClient;
use std::io::{self, IsTerminal, Write};

/// Interface interativa de terminal (CLI) para o administrador.
pub fn run_terminal_loop(session: &mut AgentSession) {
    print_banner(session);
    print_health_report(session);

    println!(
        "\nDigite sua mensagem e pressione Enter, ou use um comando (ex: :help, :models, :history, :new, :instructions, :quit)."
    );

    let stdin = io::stdin();
    loop {
        print!(
            "\n[Conversa #{} | {}] Administrador > ",
            session.current_conversation().id,
            session.config().model
        );
        let _ = io::stdout().flush();

        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(0) => {
                println!("\nEncerrando SERVERMIND AI. Até logo!");
                break;
            }
            Ok(_) => {
                let trimmed = input.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if is_repl_command_input(trimmed) {
                    if handle_command(trimmed, session) {
                        break;
                    }
                    continue;
                }

                let mut spinner = SpinnerHandle::spawn(true);
                let mut para_buf = ParagraphStreamBuffer::new(trimmed);
                let mut started_output = false;
                let stream_res = session.send_user_message_streaming(trimmed, |chunk| {
                    if let Some(ready_para) = para_buf.push_chunk(chunk) {
                        if !started_output {
                            spinner.stop_and_join();
                            println!("\nSERVERMIND AI >");
                            started_output = true;
                        }
                        print!("{}", ready_para);
                        let _ = io::stdout().flush();
                    }
                });
                spinner.stop_and_join();
                match stream_res {
                    Ok(guarded_reply) => {
                        let tail = para_buf.finish(&guarded_reply);
                        if !started_output {
                            println!("\nSERVERMIND AI >\n{}", tail);
                        } else if !tail.is_empty() {
                            print!("{}", tail);
                            if !tail.ends_with('\n') {
                                println!();
                            }
                        }
                    }
                    Err(err) => {
                        if started_output {
                            println!();
                        }
                        eprintln!("\n[ERRO AO PROCESSAR MENSAGEM]\n{}", err);
                    }
                }
            }
            Err(err) => {
                eprintln!("\nErro ao ler entrada do terminal: {}", err);
                break;
            }
        }
    }
}

fn read_prompt_line() -> Result<Option<String>, String> {
    let mut buf = String::new();
    match io::stdin().read_line(&mut buf) {
        Ok(0) => Ok(None),
        Ok(_) => Ok(Some(buf)),
        Err(err) => Err(format!(
            "Erro ao ler entrada do terminal: {}\nPróxima ação: verifique o terminal ou tente executar o comando novamente.",
            err
        )),
    }
}

fn print_banner(session: &AgentSession) {
    let instr_status = if session.is_default_instructions() {
        "Padrão"
    } else {
        "Personalizadas"
    };
    println!("======================================================================");
    println!("       SERVERMIND AI - Assistente Conversacional Local (Fase 0)       ");
    println!("       Assistente Conversacional Local para Administradores           ");
    println!("======================================================================");
    println!(
        "  Origem da config       : {} ({})",
        session.config().config_source,
        session.config().config_file_path
    );
    println!("  URL do Ollama (local)  : {}", session.config().ollama_url);
    println!("  Modelo ativo           : {}", session.config().model);
    println!(
        "  Instruções do agente   : {} (+ travas de segurança ativas)",
        instr_status
    );
    println!(
        "  Banco SQLite local     : {}",
        session.config().database_path
    );
    println!("  Modo de operação       : Estritamente conversacional (sem execução)");
    println!("======================================================================");
}

fn print_health_report(session: &AgentSession) {
    let health: OllamaHealth = session.check_health();
    if health.server_reachable && health.model_installed {
        println!(
            "[STATUS OK] Conectado ao Ollama local ({} modelo(s) instalado(s)). Modelo ativo: '{}'.",
            health.available_models.len(),
            session.config().model
        );
    } else {
        eprintln!("\n[ATENÇÃO - VERIFICAÇÃO DE AMBIENTE]");
        if let Some(msg) = health.error_message {
            eprintln!("{}", msg);
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

/// Processa comandos iniciados por `:` ou `/`. Retorna `true` se o programa deve encerrar.
fn handle_command(cmd_line: &str, session: &mut AgentSession) -> bool {
    let prefix = &cmd_line[..1];
    let without_prefix = &cmd_line[1..];
    let mut parts = without_prefix.splitn(2, char::is_whitespace);
    let raw_cmd = parts.next().unwrap_or("").to_lowercase();
    let arg = parts.next().map(|s| s.trim()).unwrap_or("");

    match raw_cmd.as_str() {
        "quit" | "exit" | "sair" | "q" => {
            println!(
                "Encerrando SERVERMIND AI. Histórico e configurações salvos localmente em '{}' e '{}'.",
                session.config().database_path,
                session.config().config_file_path
            );
            return true;
        }
        "help" | "ajuda" | "h" | "?" => {
            print_help();
        }
        "models" | "model" | "modelos" | "modelo" => {
            handle_models_command(arg, session);
        }
        "history" | "historico" | "hist" => {
            handle_history_command(arg, session);
        }
        "load" | "open" | "carregar" | "abrir" => {
            if arg.is_empty() {
                eprintln!(
                    "Uso correto: :history (para selecionar da lista) ou :load <id> (ex: :load 1)"
                );
            } else {
                reopen_conversation_by_str(arg, session);
            }
        }
        "new" | "nova" => match session.start_new_conversation() {
            Ok(conv) => {
                println!(
                    "Nova conversa separada iniciada (ID #{}, modelo '{}'). As conversas anteriores permanecem intactas em :history.",
                    conv.id, conv.model
                );
            }
            Err(err) => eprintln!("Erro ao iniciar nova conversa: {}", err),
        },
        "instructions" | "instruction" | "prompt" | "instrucoes" | "instrucao" => {
            handle_instructions_command(arg, session);
        }
        "remember" | "lembrar" | "memorizar" => {
            if arg.is_empty() {
                eprintln!(
                    "Uso correto: :remember <informação que deseja salvar na memória explícita>"
                );
            } else {
                match session.add_explicit_memory(arg) {
                    Ok(mem) => {
                        println!(
                            "Memória explícita #{} salva localmente no SQLite: \"{}\"",
                            mem.id, mem.content
                        );
                    }
                    Err(err) => eprintln!("Erro ao salvar memória: {}", err),
                }
            }
        }
        "memories" | "memorias" | "memory" => {
            handle_memories_command(arg, session);
        }
        "forget" | "esquecer" => {
            if arg.is_empty() {
                eprintln!("Uso correto: :forget <id_da_memoria> (consulte os IDs com :memories)");
            } else {
                match arg.parse::<i64>() {
                    Ok(id) if id > 0 => match session.delete_memory(id) {
                        Ok(true) => println!("Memória explícita #{} removida com sucesso.", id),
                        Ok(false) => {
                            eprintln!("Memória #{} não encontrada. Use :memories para listar.", id)
                        }
                        Err(err) => eprintln!("Erro ao remover memória: {}", err),
                    },
                    _ => eprintln!(
                        "ID inválido '{}'. Informe um número inteiro positivo da memória.",
                        arg
                    ),
                }
            }
        }
        "context" | "contexto" => {
            print_current_conversation_messages(session);
        }
        "status" => {
            print_health_report(session);
        }
        "kb" | "estudo" | "study" => {
            handle_kb_command(arg);
        }
        "backend" => {
            handle_backend_command(arg, session);
        }
        "hw" | "hardware" => {
            handle_hw_command();
        }
        "train" | "treinar" => {
            handle_train_command(arg);
        }
        "exp" | "experiencias" => {
            handle_exp_command(arg);
        }
        "quantize" | "quantizar" => {
            handle_quantize_command(arg);
        }
        _ => {
            eprintln!(
                "Comando desconhecido: '{}{}'. Digite :help (ou /ajuda) para ver os comandos disponíveis.",
                prefix, raw_cmd
            );
        }
    }

    false
}

fn print_help() {
    println!("\nComandos disponíveis (aceitam prefixo ':' ou '/'):");
    println!("  --- Seleção de Modelos (Ollama Local) ---");
    println!("  :models                  - Lista os modelos instalados no Ollama local e permite trocar interativamente");
    println!("  :model <nome_ou_numero>  - Troca para outro modelo instalado e persiste a escolha localmente");
    println!("  :status                  - Verifica a disponibilidade do Ollama e do modelo ativo");
    println!("\n  --- Histórico e Conversas ---");
    println!("  :history                 - Lista conversas salvas e permite escolher uma para reabrir com seu contexto");
    println!("  :history <id>            - Reabre diretamente a conversa pelo ID informado");
    println!(
        "  :new                     - Inicia uma nova conversa separada sem apagar as anteriores"
    );
    println!("  :context                 - Exibe todas as mensagens da conversa ativa");
    println!("\n  --- Instruções do Assistente ---");
    println!("  :instructions            - Exibe as instruções atuais, regras de segurança e como personalizar");
    println!("  :instructions <texto>    - Define e persiste novas instruções personalizadas (máx. {} caracteres)", MAX_INSTRUCTIONS_CHARS);
    println!("  :instructions edit       - Abre prompt interativo para digitar novas instruções");
    println!("  :instructions reset      - Restaura as instruções padrão do assistente");
    println!("\n  --- Memórias Explícitas ---");
    println!("  :remember <texto>        - Salva uma memória explicitamente no SQLite");
    println!("  :memories                - Lista todas as memórias salvas explicitamente");
    println!("  :forget <id>             - Remove uma memória explícita pelo ID");
    println!("\n  --- Geral ---");
    println!("  :kb [consulta]           - Lista ou busca runbooks na base técnica local");
    println!("  :backend [ollama|neural [dir]] - Alterna entre Ollama (fallback) e ServerMind Neural próprio");
    println!(
        "  :hw                      - Inspeciona hardware, VRAM/Vulkan e benchmark de matmul CPU"
    );
    println!(
        "  :train [steps | marathon [6h]] - Treina o ServerMind Neural (ex.: `:train marathon 6h`)"
    );
    println!(
        "  :exp                     - Exibe estatísticas da memória em 3 camadas (experiências)"
    );
    println!(
        "  :quantize <src> <dst> [fp16|bf16|int8|int4] - Exporta pacote quantizado compacto SMQ1"
    );
    println!("  :help                    - Exibe esta mensagem de ajuda");
    println!("  :quit                    - Encerra o programa com segurança");
    println!("\n  --- Avaliação Offline (CLI) ---");
    println!("  Para executar avaliações reprodutíveis sem acessar `servermind.db`,");
    println!("  encerre o chat e execute no terminal: servermind-ai eval --help");
}

fn handle_models_command(arg: &str, session: &mut AgentSession) {
    if !arg.is_empty() {
        match session.select_installed_model(arg) {
            Ok(selected) => {
                println!(
                    "Modelo ativo alterado para '{}' e persistido localmente (em '{}' e no SQLite).",
                    selected,
                    session.config().config_file_path
                );
            }
            Err(err) => {
                eprintln!("\n[NÃO FOI POSSÍVEL TROCAR DE MODELO]\n{}", err);
            }
        }
        return;
    }

    println!(
        "\nModelo configurado atualmente: '{}'",
        session.config().model
    );
    println!(
        "Consultando modelos instalados no Ollama local ({})...",
        session.config().ollama_url
    );

    let installed = match session.list_installed_models() {
        Ok(list) => list,
        Err(err) => {
            eprintln!("\n[OLLAMA INDISPONÍVEL]\n{}", err);
            return;
        }
    };

    if installed.is_empty() {
        eprintln!(
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

    println!("\nModelos instalados localmente no Ollama:");
    for (idx, model_name) in installed.iter().enumerate() {
        let is_active = active_resolved
            .as_deref()
            .map(|act| act.eq_ignore_ascii_case(model_name))
            .unwrap_or_else(|| session.config().model.eq_ignore_ascii_case(model_name));

        let badge = if is_active { "* [ATIVO]" } else { "         " };
        println!("  [{}] {} {}", idx + 1, badge, model_name);
    }

    if active_resolved.is_none() {
        eprintln!(
            "\nAviso: O modelo configurado '{}' não consta entre os instalados acima.",
            session.config().model
        );
    }

    print!(
        "\nDigite o número (1-{}) ou nome do modelo para ativá-lo (ou pressione Enter para manter '{}'): ",
        installed.len(),
        session.config().model
    );
    let _ = io::stdout().flush();

    match read_prompt_line() {
        Ok(Some(choice)) => {
            let trimmed = choice.trim();
            if trimmed.is_empty() {
                println!("Mantendo o modelo '{}'.", session.config().model);
            } else {
                match session.select_installed_model(trimmed) {
                    Ok(selected) => {
                        println!(
                            "Modelo ativo alterado para '{}' e persistido localmente (em '{}' e no SQLite).",
                            selected,
                            session.config().config_file_path
                        );
                    }
                    Err(err) => {
                        eprintln!("\n[NÃO FOI POSSÍVEL TROCAR DE MODELO]\n{}", err);
                    }
                }
            }
        }
        Ok(None) => println!(
            "\nEntrada encerrada. Mantendo o modelo '{}'.",
            session.config().model
        ),
        Err(err) => eprintln!("\n{}", err),
    }
}

fn handle_history_command(arg: &str, session: &mut AgentSession) {
    if !arg.is_empty() {
        reopen_conversation_by_str(arg, session);
        return;
    }

    let list = match session.list_conversations(30) {
        Ok(convs) => convs,
        Err(err) => {
            eprintln!("Erro ao consultar histórico de conversas: {}", err);
            return;
        }
    };

    if list.is_empty() {
        println!("Nenhuma conversa encontrada no banco de dados SQLite.");
        return;
    }

    println!("\nConversas salvas localmente no SQLite:");
    for c in &list {
        let marker = if c.id == session.current_conversation().id {
            "* [ATUAL]"
        } else {
            "         "
        };
        println!(
            "  #{} {} | {} | modelo: {} | {} msg(s) | {}",
            c.id, marker, c.created_at, c.model, c.message_count, c.title
        );
    }

    print!(
        "\nDigite o ID da conversa (ex: {}) para reabri-la (ou pressione Enter para permanecer na #{}): ",
        list.first().map(|c| c.id).unwrap_or(1),
        session.current_conversation().id
    );
    let _ = io::stdout().flush();

    match read_prompt_line() {
        Ok(Some(selection)) => {
            let trimmed = selection.trim().trim_start_matches('#');
            if trimmed.is_empty() {
                println!(
                    "Permanecendo na conversa atual (#{} - {}).",
                    session.current_conversation().id,
                    session.current_conversation().title
                );
            } else {
                reopen_conversation_by_str(trimmed, session);
            }
        }
        Ok(None) => println!(
            "\nEntrada encerrada. Permanecendo na conversa atual (#{}).",
            session.current_conversation().id
        ),
        Err(err) => eprintln!("\n{}", err),
    }
}

fn reopen_conversation_by_str(raw_id: &str, session: &mut AgentSession) {
    let cleaned = raw_id.trim().trim_start_matches('#');
    match cleaned.parse::<i64>() {
        Ok(id) if id > 0 => match session.load_conversation(id) {
            Ok(count) => {
                println!(
                    "\nConversa #{} ('{}') reaberta com sucesso! {} mensagem(ns) restaurada(s) no contexto.",
                    id,
                    session.current_conversation().title,
                    count
                );
                print_current_conversation_messages(session);
            }
            Err(err) => eprintln!("{}", err),
        },
        _ => {
            eprintln!(
                "ID de conversa inválido: '{}'. Informe um número inteiro positivo (ex: `:history 1`).",
                raw_id
            );
        }
    }
}

fn print_current_conversation_messages(session: &AgentSession) {
    let msgs = session.visible_messages();
    if msgs.is_empty() {
        println!(
            "A conversa atual (#{} - '{}') ainda não possui mensagens trocadas.",
            session.current_conversation().id,
            session.current_conversation().title
        );
    } else {
        println!(
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
            println!("\n[{}]\n{}", speaker, m.content);
        }
        println!("----------------------------------------------");
    }
}

fn handle_instructions_command(arg: &str, session: &mut AgentSession) {
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

        println!("\n=== CONFIGURAÇÃO DE INSTRUÇÕES DO ASSISTENTE ===");
        println!(
            "Estado atual : {} ({} / {} caracteres)",
            status_label, char_len, MAX_INSTRUCTIONS_CHARS
        );
        println!("\n[1] Instruções personalizáveis ativas:");
        println!("  {}", session.custom_instructions());
        println!("\n[2] Identidade fixa de criação (sempre aplicada):");
        println!("  {}", CREATOR_IDENTITY_INSTRUCTION);
        println!("\n[3] Diretrizes de segurança imutáveis (sempre aplicadas):");
        for line in IMMUTABLE_SAFETY_INSTRUCTIONS.lines() {
            println!("  {}", line);
        }
        println!("\nComo alterar ou restaurar:");
        println!("  - Definir novas instruções : :instructions <suas instruções em português>");
        println!("  - Editar interativamente   : :instructions edit");
        println!("  - Voltar ao padrão original: :instructions reset  (ou /instrucoes padrao)");
        return;
    }

    if matches!(
        trimmed.to_lowercase().as_str(),
        "reset" | "default" | "padrao" | "padrão"
    ) {
        match session.reset_custom_instructions() {
            Ok(()) => {
                println!(
                    "Instruções do assistente restauradas para o padrão original e salvas em '{}' e no SQLite.",
                    session.config().config_file_path
                );
                println!(
                    "As diretrizes de segurança (modo apenas conversacional) permanecem ativas."
                );
            }
            Err(err) => eprintln!("Erro ao restaurar instruções padrão: {}", err),
        }
        return;
    }

    if matches!(trimmed.to_lowercase().as_str(), "edit" | "editar" | "set") {
        println!(
            "\nDigite as novas instruções para o assistente (máximo de {} caracteres).",
            MAX_INSTRUCTIONS_CHARS
        );
        println!("Dica: Para cancelar, deixe em branco e pressione Enter. Para voltar ao padrão, use `:instructions reset`.");
        print!("Novas instruções > ");
        let _ = io::stdout().flush();

        match read_prompt_line() {
            Ok(Some(new_text)) => {
                let input_clean = new_text.trim();
                if input_clean.is_empty() {
                    println!(
                        "Edição cancelada. As instruções atuais foram mantidas. (Use `:instructions reset` se desejar voltar ao padrão)."
                    );
                    return;
                }
                apply_new_instructions(input_clean, session);
            }
            Ok(None) => println!(
                "\nEdição cancelada (entrada encerrada). As instruções atuais foram mantidas."
            ),
            Err(err) => eprintln!("\n{}", err),
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

    apply_new_instructions(text_to_set, session);
}

fn apply_new_instructions(text: &str, session: &mut AgentSession) {
    match session.set_custom_instructions(text) {
        Ok(()) => {
            println!(
                "\nInstruções personalizadas atualizadas e salvas localmente em '{}' e no SQLite!",
                session.config().config_file_path
            );
            println!("  Novas instruções: \"{}\"", session.custom_instructions());
            println!("  Nota de segurança: O agente permanece estritamente conversacional e sem executar comandos.");
            println!("  Para voltar às instruções padrão a qualquer momento, execute: :instructions reset");
        }
        Err(err) => {
            eprintln!("\n[INSTRUÇÕES INVÁLIDAS]\n{}", err);
        }
    }
}

fn handle_kb_command(arg: &str) {
    let query = arg.trim();
    if query.is_empty() {
        let entries = crate::chat::knowledge::load_kb_entries();
        println!(
            "\nBase técnica local ({} runbooks em '{}'):",
            entries.len(),
            crate::chat::knowledge::DEFAULT_KB_RELATIVE_PATH
        );
        for entry in entries {
            println!("  - [{}] {} — {}", entry.id, entry.title, entry.summary);
        }
        println!("Dica: use `:kb <termo>` para testar a recuperação ou edite o JSON para expandir o estudo.");
        return;
    }

    let hits = crate::chat::knowledge::retrieve_relevant_notes(query, 3);
    if hits.is_empty() {
        println!(
            "\nNenhum runbook atingiu relevância mínima para '{}'. Use `:kb` para listar todos.",
            query
        );
        return;
    }
    println!("\nRunbooks recuperados para '{}':", query);
    for hit in hits {
        println!(
            "  [score={}] {} ({})",
            hit.score, hit.entry.title, hit.entry.id
        );
        for line in &hit.entry.guidance {
            println!("    {}", line);
        }
    }
}

fn handle_memories_command(arg: &str, session: &mut AgentSession) {
    let trimmed = arg.trim();
    if let Some(rest) = trimmed.strip_prefix("add ") {
        match session.add_explicit_memory(rest) {
            Ok(mem) => println!("Memória explícita #{} salva: \"{}\"", mem.id, mem.content),
            Err(err) => eprintln!("Erro ao salvar memória: {}", err),
        }
        return;
    }

    match session.list_memories() {
        Ok(list) => {
            if list.is_empty() {
                println!(
                    "Nenhuma memória explícita salva. Use `:remember <texto>` para salvar uma informação importante."
                );
            } else {
                println!("\nMemórias explícitas salvas localmente no SQLite:");
                for m in list {
                    println!("  #{} [{}] {}", m.id, m.created_at, m.content);
                }
                println!(
                    "Dica: Use `:remember <texto>` para adicionar ou `:forget <id>` para remover."
                );
            }
        }
        Err(err) => eprintln!("Erro ao listar memórias: {}", err),
    }
}

fn resolve_default_neural_checkpoint_dir() -> &'static str {
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
        let p = std::path::Path::new(dir);
        if p.join("tokenizer.json").is_file()
            && (p.join("best.bin").is_file()
                || p.join("quantized.smq").is_file()
                || p.join("latest.bin").is_file())
        {
            return dir;
        }
    }
    "checkpoints/servermind-neural-v1"
}

fn handle_backend_command(arg: &str, session: &mut AgentSession) {
    let trimmed = arg.trim();
    if trimmed.is_empty() {
        println!("Backend ativo: {}", session.backend_kind().as_str());
        println!(
            "Uso: :backend ollama  |  :backend neural [diretorio_checkpoint] (padrão detectado: '{}')",
            resolve_default_neural_checkpoint_dir()
        );
        return;
    }
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let kind_str = parts.next().unwrap_or("");
    let default_dir = resolve_default_neural_checkpoint_dir();
    let dir_str = parts
        .next()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or(default_dir);

    match crate::neural::inference::InferenceBackendKind::parse(kind_str) {
        Ok(crate::neural::inference::InferenceBackendKind::Ollama) => {
            session.use_ollama_backend();
            println!("Backend alterado para 'ollama' (fallback local preservado).");
        }
        Ok(crate::neural::inference::InferenceBackendKind::ServerMindNeural) => {
            match session.use_neural_backend(std::path::Path::new(dir_str)) {
                Ok((step, best_val)) => {
                    println!(
                        "Backend alterado para 'neural' (ServerMind Neural próprio | dir='{}' | step={} | best_val_loss={:.4}).",
                        dir_str, step, best_val
                    );
                }
                Err(err) => {
                    eprintln!("[ERRO AO CARREGAR BACKEND NEURAL]\n{}", err);
                }
            }
        }
        Err(err) => eprintln!("{}", err),
    }
}

fn handle_train_command(arg: &str) {
    let trimmed = arg.trim();
    let args: Vec<String> = if trimmed.is_empty() {
        vec!["--steps".to_string(), "60".to_string()]
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
    let mut out = io::stdout();
    let mut err = io::stderr();
    let _ = run_train_neural_cli(&args, &mut out, &mut err);
}

fn handle_exp_command(arg: &str) {
    let db_path = std::path::Path::new("servermind_experiences.db");
    let mut out = io::stdout();
    let mut err = io::stderr();
    handle_exp_command_with_io(arg, db_path, &mut out, &mut err);
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
            let out_path = if rest.is_empty() {
                std::path::Path::new("datasets/experiences_approved.jsonl")
            } else {
                std::path::Path::new(rest)
            };
            if let Some(parent) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
                let _ = std::fs::create_dir_all(parent);
            }
            match store.export_approved_jsonl(out_path, None) {
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
