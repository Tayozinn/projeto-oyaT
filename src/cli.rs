use crate::agent::AgentSession;
use crate::config::{
    AppConfig, CREATOR_IDENTITY_INSTRUCTION, IMMUTABLE_SAFETY_INSTRUCTIONS, MAX_INSTRUCTIONS_CHARS,
};
use crate::eval_dataset::{filter_cases, load_and_validate_dataset, DatasetFilter, EvalCategory};
use crate::eval_report::CaseExecutionStatus;
use crate::eval_runner::{
    default_dataset_path, parse_duration_str, run_evaluation_with_progress, validate_repeat,
    EvalProgressEvent, EvalRunConfig, EvalRunOutcome, DEFAULT_CONSECUTIVE_GLOBAL_FAILURES_LIMIT,
    DEFAULT_EVAL_REPEAT,
};
use crate::models::{OllamaHealth, Role};
use crate::ollama::{InstalledModelInfo, OllamaClient};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

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

                let waiting_first_token =
                    std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
                let waiting_flag = std::sync::Arc::clone(&waiting_first_token);
                let ticker = std::thread::spawn(move || {
                    let start = std::time::Instant::now();
                    while waiting_flag.load(std::sync::atomic::Ordering::Relaxed) {
                        let secs = start.elapsed().as_secs();
                        print!("\r[SERVERMIND AI pensando... {}s]   ", secs);
                        let _ = io::stdout().flush();
                        std::thread::sleep(std::time::Duration::from_millis(500));
                    }
                });
                let mut streamed_raw = String::new();
                let mut started_output = false;
                let stream_res = session.send_user_message_streaming(trimmed, |chunk| {
                    if !started_output {
                        waiting_first_token.store(false, std::sync::atomic::Ordering::Relaxed);
                        print!("\r                                    \r\nSERVERMIND AI >\n");
                        started_output = true;
                    }
                    print!("{}", chunk);
                    let _ = io::stdout().flush();
                    streamed_raw.push_str(chunk);
                });
                waiting_first_token.store(false, std::sync::atomic::Ordering::Relaxed);
                let _ = ticker.join();
                match stream_res {
                    Ok(guarded_reply) => {
                        let raw_trimmed = streamed_raw.trim();
                        if !started_output {
                            println!(
                                "\r                                    \r\nSERVERMIND AI >\n{}",
                                guarded_reply
                            );
                        } else if let Some(suffix) = guarded_reply.strip_prefix(raw_trimmed) {
                            if !suffix.is_empty() {
                                print!("{}", suffix);
                            }
                            println!();
                        } else if guarded_reply != raw_trimmed {
                            println!(
                                "\n\n[Resposta ajustada pelos guardrails de segurança]\n{}",
                                guarded_reply
                            );
                        } else {
                            println!();
                        }
                    }
                    Err(err) => {
                        if started_output {
                            println!();
                        } else {
                            print!("\r                                    \r");
                            let _ = io::stdout().flush();
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
        let entries = crate::knowledge::load_kb_entries();
        println!(
            "\nBase técnica local ({} runbooks em '{}'):",
            entries.len(),
            crate::knowledge::DEFAULT_KB_RELATIVE_PATH
        );
        for entry in entries {
            println!("  - [{}] {} — {}", entry.id, entry.title, entry.summary);
        }
        println!("Dica: use `:kb <termo>` para testar a recuperação ou edite o JSON para expandir o estudo.");
        return;
    }

    let hits = crate::knowledge::retrieve_relevant_notes(query, 3);
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

/// Imprime a ajuda de linha de comando de alto nível do binário.
pub fn print_top_level_usage<W: Write>(out: &mut W) {
    let _ = writeln!(out, "SERVERMIND AI - Assistente Consultivo Local\n");
    let _ = writeln!(out, "Uso:");
    let _ = writeln!(
        out,
        "  servermind-ai                    Inicia o chat interativo local (usa config.toml e servermind.db)"
    );
    let _ = writeln!(
        out,
        "  servermind-ai eval --help        Executa e gerencia avaliações offline reproduzíveis (sem tocar em servermind.db)"
    );
    let _ = writeln!(
        out,
        "  servermind-ai train-neural       Treina o modelo neural próprio (--marathon [6h], --max-duration, --resume)"
    );
    let _ = writeln!(
        out,
        "  servermind-ai eval-neural        Avalia ou quantiza (--quantize-out, --bits, --kv-cache) um modelo neural próprio"
    );
    let _ = writeln!(
        out,
        "  servermind-ai hw                 Inspeciona hardware local e benchmark de CPU"
    );
    let _ = writeln!(out, "  servermind-ai --help             Exibe esta ajuda");
}

/// Ponto de entrada testável para subcomandos `eval`, permitindo capturar `stdout`/`stderr`,
/// desabilitar handler global de `Ctrl+C` em testes paralelos e injetar `cancel_flag`.
pub fn run_eval_cli_with_options<W1: Write, W2: Write>(
    args: &[String],
    install_ctrlc_handler: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
    out: &mut W1,
    err: &mut W2,
) -> i32 {
    if args.is_empty() {
        let _ = writeln!(
            err,
            "[ERRO DE USO]\nSubcomando de avaliação ausente.\nPróxima ação: execute `servermind-ai eval --help` para ver os subcomandos disponíveis (list-cases, list-models, run)."
        );
        return 2;
    }

    match args[0].as_str() {
        "--help" | "-h" | "help" => {
            if args.len() > 1 {
                let _ = writeln!(
                    err,
                    "[ERRO DE USO]\nArgumento inesperado '{}' após '{}'.\nPróxima ação: execute `servermind-ai eval --help`.",
                    args[1], args[0]
                );
                return 2;
            }
            print_eval_help(out);
            0
        }
        "list-cases" => run_eval_list_cases(&args[1..], out, err),
        "list-models" => run_eval_list_models(&args[1..], out, err),
        "run" => run_eval_run(&args[1..], install_ctrlc_handler, cancel_flag, out, err),
        other => {
            if other.starts_with('-') {
                let _ = writeln!(
                    err,
                    "[OPÇÃO DESCONHECIDA]\nA opção '{}' não é reconhecida antes de um subcomando de avaliação.\nPróxima ação: execute `servermind-ai eval --help`.",
                    other
                );
            } else {
                let _ = writeln!(
                    err,
                    "[SUBCOMANDO DESCONHECIDO]\nO subcomando 'eval {}' não existe.\nPróxima ação: use `list-cases`, `list-models` ou `run` (veja `servermind-ai eval --help`).",
                    other
                );
            }
            2
        }
    }
}

fn print_eval_help<W: Write>(out: &mut W) {
    let _ = writeln!(
        out,
        "SERVERMIND AI — Avaliação Automatizada Offline (Estritamente Consultiva)\n"
    );
    let _ = writeln!(out, "Subcomandos:");
    let _ = writeln!(out, "  eval --help\n      Exibe esta ajuda.");
    let _ = writeln!(
        out,
        "  eval list-cases [--dataset <PATH>] [--category <CAT>] [--case <ID>]\n      Lista os casos do conjunto versionado (100% offline, sem Ollama nem SQLite)."
    );
    let _ = writeln!(
        out,
        "  eval list-models [--ollama-url <URL>] [--timeout-secs <N>]\n      Lista os modelos já instalados no Ollama local (sem baixar modelos)."
    );
    let _ = writeln!(
        out,
        "  eval run [opções]\n      Executa a suíte com concorrência fixa = 1 e gera relatórios .json e .md."
    );
    let _ = writeln!(out, "\nOpções de `eval run`:");
    let _ = writeln!(out, "  --model <MODEL>          Modelo instalado a avaliar (padrão: config.toml/env ou llama3.2)");
    let _ = writeln!(
        out,
        "  --dataset <PATH>         Caminho do dataset JSON (padrão: eval/cases_v1.json)"
    );
    let _ = writeln!(
        out,
        "  --category <CAT>         Filtra por categoria (repetível ou separada por vírgula)"
    );
    let _ = writeln!(
        out,
        "  --case <ID>              Filtra por ID de caso (repetível ou separado por vírgula)"
    );
    let _ = writeln!(
        out,
        "  --limit <N>              Limita a quantidade máxima de casos após o filtro (> 0)"
    );
    let _ = writeln!(
        out,
        "  --out-dir <DIR>          Pasta base para os relatórios (padrão: eval_reports)"
    );
    let _ = writeln!(
        out,
        "  --timeout-secs <N>       Timeout por requisição ao Ollama em segundos (> 0)"
    );
    let _ = writeln!(
        out,
        "  --max-duration <DUR>     Duração máxima total (ex.: 30s, 15m, 7h, 1h30m; até 7d)"
    );
    let _ = writeln!(
        out,
        "  --repeat <N>             Repetições por caso (1..={}; padrão: 1)",
        crate::eval_runner::MAX_EVAL_REPEAT
    );
    let _ = writeln!(
        out,
        "  --judge-model <MODEL>    Juiz LLM local opcional (sempre marcado como estimativa)"
    );
    let _ = writeln!(
        out,
        "  --ollama-url <URL>       URL local de loopback do Ollama (127.0.0.1/localhost/[::1])"
    );
    let _ = writeln!(out, "\nGarantias:");
    let _ = writeln!(
        out,
        "  - Nunca lê nem escreve `servermind.db` e nunca inclui `system_prompt` privado nos relatórios."
    );
    let _ = writeln!(
        out,
        "  - Não treina modelos, não altera pesos/Modelfiles e não executa comandos sugeridos."
    );
    let _ = writeln!(
        out,
        "  - Códigos de saída: 0 = sucesso/parada no teto `--max-duration`; 1 = erro ou falha global; 2 = uso inválido; 130 = cancelado (Ctrl+C)."
    );
}

fn run_eval_list_cases<W1: Write, W2: Write>(args: &[String], out: &mut W1, err: &mut W2) -> i32 {
    let parsed = match parse_list_cases_args(args) {
        Ok(p) => p,
        Err(msg) => {
            let _ = writeln!(err, "[ERRO DE ARGUMENTOS]\n{}", msg);
            return 2;
        }
    };

    if parsed.help_requested {
        print_eval_help(out);
        return 0;
    }

    let dataset_path = parsed.dataset.unwrap_or_else(default_dataset_path);
    let loaded = match load_and_validate_dataset(&dataset_path) {
        Ok(ds) => ds,
        Err(msg) => {
            let _ = writeln!(err, "[ERRO AO CARREGAR DATASET]\n{}", msg);
            return 1;
        }
    };

    let filter = DatasetFilter {
        categories: parsed.categories,
        case_ids: parsed.case_ids,
        limit: None,
    };

    let selected = match filter_cases(&loaded.dataset, &filter) {
        Ok(list) => list,
        Err(msg) => {
            let _ = writeln!(err, "[ERRO DE FILTRO]\n{}", msg);
            return 1;
        }
    };

    let _ = writeln!(
        out,
        "Dataset v{} (schema {}) | SHA-256: {}",
        loaded.dataset.dataset_version, loaded.dataset.schema_version, loaded.sha256_hex
    );
    let _ = writeln!(
        out,
        "Casos selecionados: {} de {}\n",
        selected.len(),
        loaded.dataset.cases.len()
    );

    for case in selected {
        let review_tag = if case.requires_human_review_by_default {
            "obrigatória"
        } else {
            "condicional"
        };
        let _ = writeln!(
            out,
            "  {} | {} ({}) | comportamento={} | revisão_humana={} | {}",
            case.id,
            case.category.as_str(),
            case.category.label_pt(),
            case.expected_behavior.as_str(),
            review_tag,
            case.title
        );
    }

    0
}

fn run_eval_list_models<W1: Write, W2: Write>(args: &[String], out: &mut W1, err: &mut W2) -> i32 {
    let parsed = match parse_list_models_args(args) {
        Ok(p) => p,
        Err(msg) => {
            let _ = writeln!(err, "[ERRO DE ARGUMENTOS]\n{}", msg);
            return 2;
        }
    };

    if parsed.help_requested {
        print_eval_help(out);
        return 0;
    }

    let (ollama_url, timeout_secs, _) =
        match resolve_eval_connection(parsed.ollama_url, parsed.timeout_secs, None, false) {
            Ok(cfg) => cfg,
            Err(msg) => {
                let _ = writeln!(err, "[ERRO DE CONFIGURAÇÃO]\n{}", msg);
                return 1;
            }
        };

    let client = match build_eval_client(&ollama_url, timeout_secs) {
        Ok(c) => c,
        Err(msg) => {
            let _ = writeln!(err, "[ERRO DE CLIENTE OLLAMA]\n{}", msg);
            return 1;
        }
    };

    let models = match client.list_installed_model_details() {
        Ok(list) => list,
        Err(msg) => {
            let _ = writeln!(err, "[OLLAMA INDISPONÍVEL]\n{}", msg);
            return 1;
        }
    };

    let _ = writeln!(out, "Ollama local: {}", client.base_url());
    if models.is_empty() {
        let _ = writeln!(
            out,
            "Nenhum modelo instalado localmente. O ServerMind não baixa modelos automaticamente; execute `ollama pull llama3.2` em outro terminal."
        );
        return 0;
    }

    let _ = writeln!(out, "Modelos instalados ({}):", models.len());
    for info in &models {
        let meta = format_model_info_line(info);
        if meta.is_empty() {
            let _ = writeln!(out, "  - {}", info.name);
        } else {
            let _ = writeln!(out, "  - {} ({})", info.name, meta);
        }
    }

    0
}

fn run_eval_run<W1: Write, W2: Write>(
    args: &[String],
    install_ctrlc_handler: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
    out: &mut W1,
    err: &mut W2,
) -> i32 {
    let parsed = match parse_eval_run_args(args) {
        Ok(p) => p,
        Err(msg) => {
            let _ = writeln!(err, "[ERRO DE ARGUMENTOS]\n{}", msg);
            return 2;
        }
    };

    if parsed.help_requested {
        print_eval_help(out);
        return 0;
    }

    let (ollama_url, timeout_secs, model) =
        match resolve_eval_connection(parsed.ollama_url, parsed.timeout_secs, parsed.model, true) {
            Ok(cfg) => cfg,
            Err(msg) => {
                let _ = writeln!(err, "[ERRO DE CONFIGURAÇÃO]\n{}", msg);
                return 1;
            }
        };

    let client = match build_eval_client(&ollama_url, timeout_secs) {
        Ok(c) => c,
        Err(msg) => {
            let _ = writeln!(err, "[ERRO DE CLIENTE OLLAMA]\n{}", msg);
            return 1;
        }
    };

    let mut run_config = EvalRunConfig::new(model);
    if let Some(ds_path) = parsed.dataset {
        run_config.dataset_path = ds_path;
    }
    if let Some(out_dir) = parsed.out_dir {
        run_config.out_dir = out_dir;
    }
    run_config.filter = DatasetFilter {
        categories: parsed.categories,
        case_ids: parsed.case_ids,
        limit: parsed.limit,
    };
    run_config.judge_model = parsed.judge_model;
    run_config.timeout_secs = timeout_secs;
    run_config.max_duration = parsed.max_duration;
    run_config.repeat = parsed.repeat.unwrap_or(DEFAULT_EVAL_REPEAT);
    run_config.max_consecutive_global_failures = DEFAULT_CONSECUTIVE_GLOBAL_FAILURES_LIMIT;
    run_config.cancel_flag = cancel_flag;
    run_config.install_ctrlc_handler = install_ctrlc_handler;
    run_config.use_kb = parsed.with_kb;

    let outcome = match run_evaluation_with_progress(&client, &run_config, |event| {
        print_progress_event(out, event);
    }) {
        Ok(res) => res,
        Err(msg) => {
            let _ = writeln!(err, "[ERRO NA AVALIAÇÃO]\n{}", msg);
            return 1;
        }
    };

    print_eval_run_summary(out, &outcome);

    let stats = &outcome.report.execution_stats;
    if stats.cancelled {
        let _ = writeln!(
            err,
            "[AVALIAÇÃO CANCELADA]\nExecução interrompida por solicitação do usuário (Ctrl+C). Relatório parcial consistente salvo."
        );
        return 130;
    }

    if stats.global_failure_detected {
        let reason = stats.early_abort_reason.as_deref().unwrap_or(
            "todas as execuções planejadas falharam, expiraram ou retornaram respostas vazias",
        );
        let _ = writeln!(
            err,
            "[FALHA GLOBAL DETECTADA]\nCausa: {}\nPróxima ação: verifique o serviço Ollama local, os logs do modelo e os relatórios parciais gerados acima.",
            reason
        );
        return 1;
    }

    if stats.max_duration_reached {
        let _ = writeln!(
            out,
            "[AVISO DE DURAÇÃO MÁXIMA]\nLimite `--max-duration` atingido antes da conclusão completa ({} de {} execuções realizadas); relatório parcial consistente gravado.",
            stats.executed_count, stats.planned_executions
        );
    }

    0
}

struct ParsedListCasesArgs {
    dataset: Option<PathBuf>,
    categories: Vec<EvalCategory>,
    case_ids: Vec<String>,
    help_requested: bool,
}

struct ParsedListModelsArgs {
    ollama_url: Option<String>,
    timeout_secs: Option<u64>,
    help_requested: bool,
}

struct ParsedEvalRunArgs {
    model: Option<String>,
    dataset: Option<PathBuf>,
    categories: Vec<EvalCategory>,
    case_ids: Vec<String>,
    limit: Option<usize>,
    out_dir: Option<PathBuf>,
    timeout_secs: Option<u64>,
    max_duration: Option<Duration>,
    repeat: Option<usize>,
    judge_model: Option<String>,
    ollama_url: Option<String>,
    with_kb: bool,
    help_requested: bool,
}

fn split_flag(arg: &str) -> (&str, Option<&str>) {
    match arg.split_once('=') {
        Some((name, val)) => (name, Some(val)),
        None => (arg, None),
    }
}

fn take_flag_value(
    args: &[String],
    idx: &mut usize,
    flag: &str,
    inline: Option<&str>,
) -> Result<String, String> {
    let raw = match inline {
        Some(v) => v,
        None => {
            let next_idx = *idx + 1;
            let Some(next) = args.get(next_idx) else {
                return Err(format!(
                    "A opção '{}' requer um valor.\nPróxima ação: consulte `servermind-ai eval --help`.",
                    flag
                ));
            };
            if next.starts_with("--") {
                return Err(format!(
                    "A opção '{}' requer um valor antes de '{}'.",
                    flag, next
                ));
            }
            *idx = next_idx;
            next.as_str()
        }
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!(
            "O valor informado para '{}' não pode ser vazio.",
            flag
        ));
    }
    Ok(trimmed.to_string())
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!(
            "A opção '{}' foi informada mais de uma vez. Informe-a apenas uma vez.",
            flag
        ));
    }
    *slot = Some(value);
    Ok(())
}

fn push_csv_items(raw: &str, flag: &str, out_items: &mut Vec<String>) -> Result<(), String> {
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            return Err(format!(
                "Lista inválida para '{}': há um item vazio em '{}'.",
                flag, raw
            ));
        }
        out_items.push(trimmed.to_string());
    }
    Ok(())
}

fn parse_categories_arg(raw: &str, categories: &mut Vec<EvalCategory>) -> Result<(), String> {
    let mut names = Vec::new();
    push_csv_items(raw, "--category", &mut names)?;
    for name in names {
        let parsed = EvalCategory::parse(&name)?;
        categories.push(parsed);
    }
    Ok(())
}

fn parse_positive_usize(raw: &str, flag: &str) -> Result<usize, String> {
    let val = raw.parse::<usize>().map_err(|_| {
        format!(
            "Valor inválido para '{}' ('{}'): informe um número inteiro positivo.",
            flag, raw
        )
    })?;
    if val == 0 {
        return Err(format!(
            "O valor de '{}' deve ser um número inteiro maior que zero.",
            flag
        ));
    }
    Ok(val)
}

fn parse_positive_u64(raw: &str, flag: &str) -> Result<u64, String> {
    let val = raw.parse::<u64>().map_err(|_| {
        format!(
            "Valor inválido para '{}' ('{}'): informe um número inteiro positivo em segundos.",
            flag, raw
        )
    })?;
    if val == 0 {
        return Err(format!(
            "O valor de '{}' deve ser um número inteiro maior que zero.",
            flag
        ));
    }
    Ok(val)
}

fn reject_unexpected_arg(arg: &str, subcommand: &str) -> String {
    if arg.starts_with('-') {
        format!(
            "Opção desconhecida '{}' para `eval {}`.\nPróxima ação: execute `servermind-ai eval --help`.",
            arg, subcommand
        )
    } else {
        format!(
            "Argumento posicional inesperado '{}' em `eval {}`.\nPróxima ação: use as opções nomeadas conforme `servermind-ai eval --help`.",
            arg, subcommand
        )
    }
}

fn parse_list_cases_args(args: &[String]) -> Result<ParsedListCasesArgs, String> {
    let mut dataset = None;
    let mut categories = Vec::new();
    let mut case_ids = Vec::new();
    let mut help_requested = false;

    let mut idx = 0;
    while idx < args.len() {
        let (flag, inline) = split_flag(&args[idx]);
        match flag {
            "--help" | "-h" => {
                if inline.is_some() {
                    return Err("A opção '--help' não aceita valor.".to_string());
                }
                help_requested = true;
            }
            "--dataset" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                set_once(&mut dataset, PathBuf::from(val), flag)?;
            }
            "--category" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                parse_categories_arg(&val, &mut categories)?;
            }
            "--case" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                push_csv_items(&val, flag, &mut case_ids)?;
            }
            _ => return Err(reject_unexpected_arg(&args[idx], "list-cases")),
        }
        idx += 1;
    }

    Ok(ParsedListCasesArgs {
        dataset,
        categories,
        case_ids,
        help_requested,
    })
}

fn parse_list_models_args(args: &[String]) -> Result<ParsedListModelsArgs, String> {
    let mut ollama_url = None;
    let mut timeout_secs = None;
    let mut help_requested = false;

    let mut idx = 0;
    while idx < args.len() {
        let (flag, inline) = split_flag(&args[idx]);
        match flag {
            "--help" | "-h" => {
                if inline.is_some() {
                    return Err("A opção '--help' não aceita valor.".to_string());
                }
                help_requested = true;
            }
            "--ollama-url" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                AppConfig::validate_local_url(&val)?;
                set_once(&mut ollama_url, val, flag)?;
            }
            "--timeout-secs" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                let parsed = parse_positive_u64(&val, flag)?;
                set_once(&mut timeout_secs, parsed, flag)?;
            }
            _ => return Err(reject_unexpected_arg(&args[idx], "list-models")),
        }
        idx += 1;
    }

    Ok(ParsedListModelsArgs {
        ollama_url,
        timeout_secs,
        help_requested,
    })
}

fn parse_eval_run_args(args: &[String]) -> Result<ParsedEvalRunArgs, String> {
    let mut parsed = ParsedEvalRunArgs {
        model: None,
        dataset: None,
        categories: Vec::new(),
        case_ids: Vec::new(),
        limit: None,
        out_dir: None,
        timeout_secs: None,
        max_duration: None,
        repeat: None,
        judge_model: None,
        ollama_url: None,
        with_kb: false,
        help_requested: false,
    };

    let mut idx = 0;
    while idx < args.len() {
        let (flag, inline) = split_flag(&args[idx]);
        match flag {
            "--help" | "-h" => {
                if inline.is_some() {
                    return Err("A opção '--help' não aceita valor.".to_string());
                }
                parsed.help_requested = true;
            }
            "--model" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                set_once(&mut parsed.model, val, flag)?;
            }
            "--dataset" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                set_once(&mut parsed.dataset, PathBuf::from(val), flag)?;
            }
            "--category" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                parse_categories_arg(&val, &mut parsed.categories)?;
            }
            "--case" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                push_csv_items(&val, flag, &mut parsed.case_ids)?;
            }
            "--limit" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                let num = parse_positive_usize(&val, flag)?;
                set_once(&mut parsed.limit, num, flag)?;
            }
            "--out-dir" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                set_once(&mut parsed.out_dir, PathBuf::from(val), flag)?;
            }
            "--timeout-secs" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                let secs = parse_positive_u64(&val, flag)?;
                set_once(&mut parsed.timeout_secs, secs, flag)?;
            }
            "--max-duration" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                let dur = parse_duration_str(&val)?;
                set_once(&mut parsed.max_duration, dur, flag)?;
            }
            "--repeat" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                let raw_rep = parse_positive_usize(&val, flag)?;
                let rep = validate_repeat(raw_rep)?;
                set_once(&mut parsed.repeat, rep, flag)?;
            }
            "--judge-model" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                set_once(&mut parsed.judge_model, val, flag)?;
            }
            "--ollama-url" => {
                let val = take_flag_value(args, &mut idx, flag, inline)?;
                AppConfig::validate_local_url(&val)?;
                set_once(&mut parsed.ollama_url, val, flag)?;
            }
            "--with-kb" => {
                if inline.is_some() {
                    return Err("A opção '--with-kb' não aceita valor.".to_string());
                }
                parsed.with_kb = true;
            }
            _ => return Err(reject_unexpected_arg(&args[idx], "run")),
        }
        idx += 1;
    }

    Ok(parsed)
}

fn resolve_eval_connection(
    ollama_url_opt: Option<String>,
    timeout_opt: Option<u64>,
    model_opt: Option<String>,
    need_model: bool,
) -> Result<(String, u64, String), String> {
    let need_config =
        ollama_url_opt.is_none() || timeout_opt.is_none() || (need_model && model_opt.is_none());

    let loaded_cfg = if need_config {
        Some(AppConfig::load()?)
    } else {
        None
    };

    let ollama_url = match ollama_url_opt {
        Some(url) => url,
        None => loaded_cfg
            .as_ref()
            .map(|c| c.ollama_url.clone())
            .unwrap_or_else(|| crate::config::DEFAULT_OLLAMA_URL.to_string()),
    };
    AppConfig::validate_local_url(&ollama_url)?;

    let timeout_secs = match timeout_opt {
        Some(secs) => secs,
        None => loaded_cfg
            .as_ref()
            .map(|c| c.request_timeout_secs)
            .unwrap_or(crate::config::DEFAULT_TIMEOUT_SECS),
    };
    if timeout_secs == 0 {
        return Err("O timeout de requisição do Ollama deve ser maior que zero.".to_string());
    }

    let model = if need_model {
        match model_opt {
            Some(m) => m,
            None => loaded_cfg
                .as_ref()
                .map(|c| c.model.clone())
                .unwrap_or_else(|| crate::config::DEFAULT_MODEL.to_string()),
        }
    } else {
        String::new()
    };

    Ok((ollama_url, timeout_secs, model))
}

fn build_eval_client(ollama_url: &str, timeout_secs: u64) -> Result<OllamaClient, String> {
    AppConfig::validate_local_url(ollama_url)?;
    OllamaClient::new(ollama_url.to_string(), timeout_secs)
}

fn format_model_info_line(info: &InstalledModelInfo) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(details) = &info.details {
        if let Some(family) = details.family.as_deref().filter(|s| !s.trim().is_empty()) {
            parts.push(format!("família={}", family));
        }
        if let Some(param) = details
            .parameter_size
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            parts.push(param.to_string());
        }
        if let Some(quant) = details
            .quantization_level
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            parts.push(quant.to_string());
        }
    }
    if let Some(bytes) = info.size {
        parts.push(format_bytes_human(bytes));
    }
    parts.join(", ")
}

fn format_bytes_human(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else {
        format!("{} B", bytes)
    }
}

fn print_progress_event<W: Write>(out: &mut W, event: &EvalProgressEvent) {
    match event {
        EvalProgressEvent::Started {
            model_resolved,
            judge_resolved,
            selected_cases,
            repeat,
            planned_executions,
        } => {
            let judge_info = match judge_resolved {
                Some(j) if j.eq_ignore_ascii_case(model_resolved) => {
                    format!(" | juiz='{}' (estimativa/autoavaliação)", j)
                }
                Some(j) => format!(" | juiz='{}' (estimativa)", j),
                None => String::new(),
            };
            let _ = writeln!(
                out,
                "[eval] Iniciando avaliação: modelo='{}' | casos={} | repeat={} | execuções={}{} | concorrência=1",
                model_resolved, selected_cases, repeat, planned_executions, judge_info
            );
            let _ = out.flush();
        }
        EvalProgressEvent::CaseStarted {
            execution_index,
            planned_executions,
            case_id,
            repeat_index,
            category,
            title,
        } => {
            let _ = writeln!(
                out,
                "[eval] [{}/{}] Executando {} (rep {}, {}) — {}",
                execution_index,
                planned_executions,
                case_id,
                repeat_index,
                category.as_str(),
                title
            );
            let _ = out.flush();
        }
        EvalProgressEvent::CaseFinished {
            execution_index,
            planned_executions,
            case_id,
            repeat_index,
            status,
            duration_ms,
            deterministic_total,
            deterministic_max,
            needs_human_review,
        } => {
            let score_part = match deterministic_total {
                Some(s) => format!(" | det={}/{}", s, deterministic_max),
                None => String::new(),
            };
            let review_part = if *needs_human_review {
                " | revisão_humana=sim"
            } else {
                ""
            };
            let status_tag = match status {
                CaseExecutionStatus::Succeeded => "OK",
                CaseExecutionStatus::EmptyResponse => "VAZIA",
                CaseExecutionStatus::Timeout => "TIMEOUT",
                CaseExecutionStatus::Error => "ERRO",
            };
            let _ = writeln!(
                out,
                "[eval] [{}/{}] {} (rep {}): {} | {}ms{}{}",
                execution_index,
                planned_executions,
                case_id,
                repeat_index,
                status_tag,
                duration_ms,
                score_part,
                review_part
            );
            let _ = out.flush();
        }
        EvalProgressEvent::Interrupted {
            cancelled,
            max_duration_reached,
            early_abort_reason,
        } => {
            let cause = if *cancelled {
                "Ctrl+C / cancelamento"
            } else if *max_duration_reached {
                "duração máxima atingida"
            } else if let Some(reason) = early_abort_reason {
                reason.as_str()
            } else {
                "interrupção antecipada"
            };
            let _ = writeln!(out, "[eval] Execução interrompida: {}", cause);
            let _ = out.flush();
        }
        EvalProgressEvent::ReportWritten { paths } => {
            let _ = writeln!(
                out,
                "[eval] Relatórios gravados em '{}'",
                paths.directory.display()
            );
            let _ = out.flush();
        }
    }
}

fn print_eval_run_summary<W: Write>(out: &mut W, outcome: &EvalRunOutcome) {
    let report = &outcome.report;
    let stats = &report.execution_stats;
    let scores = &report.overall_scores;

    let _ = writeln!(out, "\n=== Resumo da Avaliação ===");
    let _ = writeln!(out, "Modelo avaliado   : {}", report.model.resolved_name);
    let _ = writeln!(
        out,
        "Dataset           : v{} (SHA-256: {})",
        report.dataset.dataset_version, report.dataset.sha256_hex
    );
    let _ = writeln!(
        out,
        "Execuções         : {}/{} (OK: {}, vazias: {}, timeout: {}, erro: {}, não executadas: {})",
        stats.executed_count,
        stats.planned_executions,
        stats.succeeded_count,
        stats.empty_response_count,
        stats.timeout_count,
        stats.error_count,
        stats.skipped_count
    );
    if let Some(mean) = scores.deterministic_mean_total {
        let _ = writeln!(
            out,
            "Média determiníst.: {:.2}/{} ({} casos avaliados)",
            mean, scores.max_total_score, scores.deterministic_evaluated_count
        );
    }
    if scores.judge_enabled {
        let self_tag = if scores.judge_self_judging {
            " | AVISO: autoavaliação"
        } else {
            ""
        };
        let judge_name = report
            .judge_model
            .as_ref()
            .map(|j| j.resolved_name.as_str())
            .unwrap_or("-");
        match scores.judge_estimate_mean_total {
            Some(mean) => {
                let _ = writeln!(
                    out,
                    "Média juiz (estim.): {:.2}/{} (modelo: {}, válidos: {}{})",
                    mean, scores.max_total_score, judge_name, scores.judge_estimate_count, self_tag
                );
            }
            None => {
                let _ = writeln!(
                    out,
                    "Juiz local (estim.): sem pontuações válidas (modelo: {}{})",
                    judge_name, self_tag
                );
            }
        }
    }
    let _ = writeln!(
        out,
        "Revisão humana    : {} caso(s) sinalizado(s)",
        stats.human_review_count
    );
    let _ = writeln!(
        out,
        "Relatório JSON    : {}",
        outcome.paths.json_path.display()
    );
    let _ = writeln!(
        out,
        "Relatório MD      : {}",
        outcome.paths.markdown_path.display()
    );
    let _ = writeln!(
        out,
        "Aviso             : Avaliação estritamente consultiva; notas não garantem correção nem alteram modelos/pesos."
    );
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

    match crate::inference::InferenceBackendKind::parse(kind_str) {
        Ok(crate::inference::InferenceBackendKind::Ollama) => {
            session.use_ollama_backend();
            println!("Backend alterado para 'ollama' (fallback local preservado).");
        }
        Ok(crate::inference::InferenceBackendKind::ServerMindNeural) => {
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

fn handle_hw_command() {
    let mut out = io::stdout();
    let _ = run_hw_cli(&mut out);
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
    let store = match crate::experience::ExperienceStore::open(db_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Erro ao abrir ExperienceStore: {}", e);
            return;
        }
    };

    let trimmed = arg.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("stats") {
        match store.stats() {
            Ok(st) => {
                println!(
                    "Memória 3 Camadas ('{}'): operacional={}, experiências={} (verificadas={}), treino_aprovado={}, treino_rejeitado={}",
                    db_path.display(),
                    st.operational_count,
                    st.experience_count,
                    st.verified_experience_count,
                    st.approved_training_count,
                    st.rejected_training_count
                );
                println!("Uso: :exp op <scope>|<key>|<value>  |  :exp add <dominio>|<problema>|<diagnostico>|<solucao>|<outcome>|<verified:true|false>  |  :exp export [arquivo.jsonl]");
            }
            Err(e) => eprintln!("Erro ao consultar estatísticas: {}", e),
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
                eprintln!("Uso: :exp op <scope> | <key> | <value>");
                return;
            }
            match store.upsert_operational(cols[0], cols[1], cols[2]) {
                Ok(()) => println!("Memória operacional atualizada ({}:{}).", cols[0], cols[1]),
                Err(e) => eprintln!("Erro ao gravar memória operacional: {}", e),
            }
        }
        "add" => {
            let cols: Vec<&str> = rest.split('|').map(|s| s.trim()).collect();
            if cols.len() < 6 {
                eprintln!("Uso: :exp add <dominio> | <problema> | <diagnostico> | <solucao> | <outcome> | <true|false>");
                return;
            }
            let verified = matches!(
                cols[5].to_ascii_lowercase().as_str(),
                "true" | "1" | "sim" | "yes"
            );
            let exp_id = match store.record_experience(&crate::experience::NewExperience {
                domain: cols[0].to_string(),
                problem: cols[1].to_string(),
                diagnosis: cols[2].to_string(),
                solution: cols[3].to_string(),
                outcome: cols[4].to_string(),
                error_notes: String::new(),
                verified,
            }) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("Erro ao registrar experiência: {}", e);
                    return;
                }
            };
            let ideal = format!("Diagnóstico: {}. Solução: {}", cols[2], cols[3]);
            match store.submit_for_training(
                Some(exp_id),
                cols[1],
                &ideal,
                cols[0],
                cols[4],
                verified,
                "exp_v1",
            ) {
                Ok(rec) if rec.approved => {
                    println!(
                        "Experiência #{} registrada e APROVADA para treinamento (score={:.2}).",
                        exp_id, rec.quality_score
                    );
                }
                Ok(rec) => {
                    println!(
                        "Experiência #{} registrada, mas REPROVADA no gate de treinamento: {}",
                        exp_id,
                        rec.rejection_reason.unwrap_or_default()
                    );
                }
                Err(e) => eprintln!("Erro ao validar exemplo de treinamento: {}", e),
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
                Ok(n) => println!(
                    "Exportados {} exemplo(s) aprovado(s) para '{}'.",
                    n,
                    out_path.display()
                ),
                Err(e) => eprintln!("Erro ao exportar experiências: {}", e),
            }
        }
        "list" => {
            if let Ok(ops) = store.list_operational(None) {
                println!("Memória Operacional ({}):", ops.len());
                for op in ops.iter().take(10) {
                    println!("  [{}] {} = {}", op.scope, op.key, op.value);
                }
            }
            if let Ok(exps) = store.list_experiences(10) {
                println!("Últimas Experiências ({}):", exps.len());
                for e in &exps {
                    println!(
                        "  #{} [{} | verified={} | {}] {}",
                        e.id, e.domain, e.verified, e.outcome, e.problem
                    );
                }
            }
        }
        _ => {
            eprintln!(
                "Subcomando :exp desconhecido '{}'. Use :exp stats | list | op | add | export.",
                sub
            );
        }
    }
}

pub fn run_hw_cli(out: &mut dyn Write) -> i32 {
    let hw = crate::infrastructure::probe_hardware();
    let (ms, gflops) = crate::infrastructure::benchmark_cpu_matmul(192);
    let mem_10m = crate::infrastructure::estimate_transformer_memory_with_heads(
        4096, 256, 384, 6, 6, 1536, 1,
    );
    let _ = writeln!(out, "=== Hardware & Infraestrutura ServerMind ===");
    let _ = writeln!(out, "OS / Arch        : {} / {}", hw.os, hw.arch);
    let _ = writeln!(out, "CPUs lógicas     : {}", hw.logical_cpus);
    if let (Some(tot_mb), Some(avail_mb)) = (hw.total_system_ram_mb, hw.available_system_ram_mb) {
        let prof = hw
            .recommended_runtime_profile
            .as_deref()
            .unwrap_or("standard");
        let _ = writeln!(
            out,
            "RAM Física Host  : {} MB total ({} MB livres) | perfil={}",
            tot_mb, avail_mb, prof
        );
    }
    let _ = writeln!(out, "GPU hint         : {}", hw.detected_gpu_hint);
    let _ = writeln!(
        out,
        "Vulkan / ROCm    : {} / {}",
        hw.vulkan_available, hw.rocm_available
    );
    let _ = writeln!(out, "Dispositivo ativo: {}", hw.recommended_training_device);
    let _ = writeln!(
        out,
        "Benchmark CPU    : 192x192 em {} ms ({:.2} GFLOPS)",
        ms, gflops
    );
    if let Some((gpu_ms, gpu_gflops)) = crate::infrastructure::benchmark_gpu_matmul(512) {
        let _ = writeln!(
            out,
            "Benchmark GPU    : 512x512 VRAM em {:.2} ms ({:.2} GFLOPS)",
            gpu_ms, gpu_gflops
        );
    }
    let _ = writeln!(
        out,
        "Estimativa ~14M  : {:.2} M params | pesos_f32={:.1} MB (fp16={:.1} | int8={:.1} | int4={:.1} MB) | treino_seq={:.1} MB | infer_kv(f32={:.1} | fp16={:.1} | int8={:.1} MB) | cabe_8gb_ram={} | cabe_rx7600_8gb={}",
        mem_10m.total_params as f64 / 1e6,
        mem_10m.weights_f32_mb,
        mem_10m.weights_fp16_mb,
        mem_10m.weights_int8_mb,
        mem_10m.weights_int4_mb,
        mem_10m.total_training_est_mb,
        mem_10m.inference_kv_fp32_mb,
        mem_10m.inference_kv_fp16_mb,
        mem_10m.inference_kv_int8_mb,
        mem_10m.fits_8gb_ram_inference,
        mem_10m.fits_8gb_vram_rx7600
    );
    0
}

fn load_seed_corpus(dataset_path: &std::path::Path) -> Result<Vec<String>, String> {
    let raw = std::fs::read_to_string(dataset_path)
        .map_err(|e| format!("Falha ao ler dataset '{}': {}", dataset_path.display(), e))?;
    let mut examples = Vec::new();
    for (idx, line) in raw.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
            format!(
                "Linha JSON inválida em '{}' (linha {}): {}",
                dataset_path.display(),
                line_no,
                e
            )
        })?;
        let dialogue = if let Some(turns_val) = val.get("turns") {
            let turns = turns_val.as_array().ok_or_else(|| {
                format!(
                    "Campo 'turns' inválido em '{}' (linha {}): esperado array não vazio.",
                    dataset_path.display(),
                    line_no
                )
            })?;
            if turns.is_empty() {
                return Err(format!(
                    "Campo 'turns' vazio em '{}' (linha {}).",
                    dataset_path.display(),
                    line_no
                ));
            }
            let mut buf = String::new();
            for (t_idx, turn) in turns.iter().enumerate() {
                let u = turn
                    .get("user")
                    .or_else(|| turn.get("prompt"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim())
                    .unwrap_or("");
                let a = turn
                    .get("assistant")
                    .or_else(|| turn.get("response"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim())
                    .unwrap_or("");
                if u.is_empty() || a.is_empty() {
                    return Err(format!(
                        "Turno #{} incompleto em '{}' (linha {}): 'user' e 'assistant' são obrigatórios e não podem ser vazios.",
                        t_idx + 1,
                        dataset_path.display(),
                        line_no
                    ));
                }
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(&format!("<user> {}\n<assistant> {}", u, a));
            }
            buf
        } else {
            let prompt = val
                .get("prompt")
                .and_then(|v| v.as_str())
                .map(|s| s.trim())
                .unwrap_or("");
            let response = val
                .get("response")
                .and_then(|v| v.as_str())
                .map(|s| s.trim())
                .unwrap_or("");
            if prompt.is_empty() || response.is_empty() {
                return Err(format!(
                    "Registro inválido em '{}' (linha {}): campos 'prompt' e 'response' são obrigatórios e não podem ser vazios.",
                    dataset_path.display(),
                    line_no
                ));
            }
            format!("<user> {}\n<assistant> {}", prompt, response)
        };
        let norm = crate::tokenizer::normalize_corpus_text(&dialogue);
        if !norm.is_empty() {
            examples.push(norm);
        }
    }
    if examples.is_empty() {
        return Err("Nenhum exemplo válido encontrado no dataset.".to_string());
    }
    Ok(examples)
}

fn check_no_existing_training_artifacts(out_dir: &std::path::Path) -> Result<(), String> {
    const PROTECTED_FILES: &[&str] = &[
        "best.bin",
        "latest.bin",
        "metrics.jsonl",
        "training_evolution.md",
        "evolution_log.jsonl",
    ];
    let existing: Vec<&str> = PROTECTED_FILES
        .iter()
        .copied()
        .filter(|name| out_dir.join(name).exists())
        .collect();
    if !existing.is_empty() {
        return Err(format!(
            "O diretório '{}' já contém artefatos de treinamento ({}). Para não perder progresso, use `--resume` para continuar ou informe um novo diretório em `--out`.",
            out_dir.display(),
            existing.join(", ")
        ));
    }
    Ok(())
}

pub fn run_train_neural_cli(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    run_train_neural_cli_with_cancel(args, true, None, out, err)
}

pub fn run_train_neural_cli_with_cancel(
    args: &[String],
    install_ctrlc_handler: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    let mut dataset_path: Option<PathBuf> = None;
    let mut corpus_dirs: Vec<PathBuf> = Vec::new();
    let mut chunk_chars: usize = 1200;
    let mut out_dir_opt: Option<PathBuf> = None;
    let mut run_name: Option<String> = None;
    let mut resume = false;
    let mut max_steps_opt: Option<usize> = None;
    let mut max_duration_opt: Option<Duration> = None;
    let mut eval_every_secs_opt: Option<u64> = None;
    let mut marathon = false;
    let mut d_model_opt: Option<usize> = None;
    let mut n_heads_opt: Option<usize> = None;
    let mut n_layers_opt: Option<usize> = None;
    let mut d_ff_opt: Option<usize> = None;
    let mut seq_len_opt: Option<usize> = None;
    let mut batch_size_opt: Option<usize> = None;
    let mut vocab_size_opt: Option<usize> = None;
    let mut lr_opt: Option<f32> = None;
    let mut min_learning_rate: Option<f32> = None;
    let mut max_val_windows: usize = 128;
    let mut device = crate::neural::ComputeDevice::Auto;
    let mut export_quant_dir: Option<PathBuf> = None;
    let mut export_bits = crate::neural::WeightPrecision::Int8;
    let mut threads_opt: Option<usize> = None;

    let parse_pos_usize = |flag: &str, raw: &str| -> Result<usize, String> {
        let v: usize = raw.parse().map_err(|_| {
            format!(
                "Valor inválido '{}' para '{}': esperado inteiro positivo.",
                raw, flag
            )
        })?;
        if v == 0 {
            return Err(format!("'{}' deve ser maior que zero.", flag));
        }
        Ok(v)
    };

    let parse_pos_f32 = |flag: &str, raw: &str| -> Result<f32, String> {
        let v: f32 = raw.parse().map_err(|_| {
            format!(
                "Valor inválido '{}' para '{}': esperado número decimal positivo.",
                raw, flag
            )
        })?;
        if !v.is_finite() || v <= 0.0 {
            return Err(format!("'{}' deve ser maior que zero.", flag));
        }
        Ok(v)
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                let _ = writeln!(out, "Uso: servermind-ai train-neural [opções]");
                let _ = writeln!(out, "  --marathon                  Ativa Maratona Autônoma de 6h em 2 fases (Pré-treino + Fine-Tuning Chat + Sequência Evolutiva + SMQ1 INT8)");
                let _ = writeln!(out, "  --max-duration <DUR>        Controla o tempo real de treino (ex.: 30s, 20m, 6h, 1h30m; salva limpo ao atingir ou em Ctrl+C)");
                let _ = writeln!(out, "  --eval-every-secs <N>       Intervalo em segundos entre marcos da sequência evolutiva (automático se omitido)");
                let _ = writeln!(out, "  --dataset <arquivo.jsonl>   Corpus Q&A em JSONL (padrão se --corpus-dir não for informado)");
                let _ = writeln!(out, "  --corpus-dir <pasta>        Diretório recursivo de texto/código para pré-treino (repetível)");
                let _ = writeln!(out, "  --chunk-chars <N>           Tamanho máximo de bloco por arquivo de corpus (padrão: 1200)");
                let _ = writeln!(
                    out,
                    "  --out <pasta>               Diretório de saída dos checkpoints"
                );
                let _ = writeln!(
                    out,
                    "  --run-name <nome>           Nome da execução nos relatórios"
                );
                let _ = writeln!(out, "  --resume                    Retoma latest.bin e tokenizer.json existentes em --out");
                let _ = writeln!(out, "  --steps <N>                 Número alvo de passos (padrão: 80 se --max-duration não for usado)");
                let _ = writeln!(out, "  --d-model <N> --layers <N> --heads <N> --ff <N> --seq-len <N> --vocab <N> --batch-size <N>");
                let _ = writeln!(out, "  --lr <f32> --min-lr <f32>   Taxa de aprendizado inicial e mínima (cosine decay)");
                let _ = writeln!(out, "  --max-val-windows <N>       Limite uniforme de janelas na validação periódica (padrão: 128)");
                let _ = writeln!(out, "  --device <auto|cpu|gpu>     Backend computacional (padrão: auto; gpu usa Vulkan RX 7600)");
                let _ = writeln!(out, "  --export-quant <pasta>      Exporta pacote quantizado SMQ1 ('quantized.smq' + 'tokenizer.json') ao concluir");
                let _ = writeln!(
                    out,
                    "  --bits <fp16|bf16|int8|int4> Precisão para --export-quant (padrão: int8)"
                );
                let _ = writeln!(out, "  --threads <N>               Limita o número máximo de threads CPU ativas (1..=64)");
                return 0;
            }
            "--marathon" => {
                marathon = true;
                i += 1;
            }
            "--max-duration" if i + 1 < args.len() => {
                match parse_duration_str(&args[i + 1]) {
                    Ok(d) => max_duration_opt = Some(d),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--eval-every-secs" if i + 1 < args.len() => {
                match parse_pos_usize("--eval-every-secs", &args[i + 1]) {
                    Ok(s) => eval_every_secs_opt = Some(s as u64),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--threads" if i + 1 < args.len() => {
                match parse_pos_usize("--threads", &args[i + 1]) {
                    Ok(t) => threads_opt = Some(t.min(64)),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--export-quant" if i + 1 < args.len() => {
                export_quant_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--bits" if i + 1 < args.len() => {
                match crate::neural::WeightPrecision::parse(&args[i + 1]) {
                    Ok(p) => export_bits = p,
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--device" if i + 1 < args.len() => {
                match crate::neural::ComputeDevice::parse(&args[i + 1]) {
                    Ok(d) => device = d,
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--dataset" if i + 1 < args.len() => {
                dataset_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--corpus-dir" if i + 1 < args.len() => {
                corpus_dirs.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--chunk-chars" if i + 1 < args.len() => {
                match parse_pos_usize("--chunk-chars", &args[i + 1]) {
                    Ok(v) => chunk_chars = v,
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_dir_opt = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--run-name" if i + 1 < args.len() => {
                run_name = Some(args[i + 1].clone());
                i += 2;
            }
            "--resume" => {
                resume = true;
                i += 1;
            }
            "--lr" if i + 1 < args.len() => {
                match parse_pos_f32("--lr", &args[i + 1]) {
                    Ok(v) => lr_opt = Some(v),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--min-lr" if i + 1 < args.len() => {
                match parse_pos_f32("--min-lr", &args[i + 1]) {
                    Ok(v) => min_learning_rate = Some(v),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--max-val-windows" if i + 1 < args.len() => {
                match parse_pos_usize("--max-val-windows", &args[i + 1]) {
                    Ok(v) => max_val_windows = v,
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--steps" if i + 1 < args.len() => {
                match parse_pos_usize("--steps", &args[i + 1]) {
                    Ok(v) => max_steps_opt = Some(v),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--d-model" if i + 1 < args.len() => {
                match parse_pos_usize("--d-model", &args[i + 1]) {
                    Ok(v) => d_model_opt = Some(v),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--layers" if i + 1 < args.len() => {
                match parse_pos_usize("--layers", &args[i + 1]) {
                    Ok(v) => n_layers_opt = Some(v),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--heads" if i + 1 < args.len() => {
                match parse_pos_usize("--heads", &args[i + 1]) {
                    Ok(v) => n_heads_opt = Some(v),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--ff" if i + 1 < args.len() => {
                match parse_pos_usize("--ff", &args[i + 1]) {
                    Ok(v) => d_ff_opt = Some(v),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--seq-len" if i + 1 < args.len() => {
                match parse_pos_usize("--seq-len", &args[i + 1]) {
                    Ok(v) => seq_len_opt = Some(v),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--batch-size" if i + 1 < args.len() => {
                match parse_pos_usize("--batch-size", &args[i + 1]) {
                    Ok(v) => batch_size_opt = Some(v),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--vocab" if i + 1 < args.len() => {
                match parse_pos_usize("--vocab", &args[i + 1]) {
                    Ok(v) => vocab_size_opt = Some(v),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            other => {
                let _ = writeln!(err, "Argumento desconhecido em train-neural: '{}'", other);
                return 2;
            }
        }
    }

    if let Some(t) = threads_opt {
        crate::neural::ops::set_max_active_threads(Some(t));
    }

    if marathon {
        let total_dur = max_duration_opt.unwrap_or_else(|| Duration::from_secs(6 * 3600));
        let out_dir =
            out_dir_opt.unwrap_or_else(|| PathBuf::from("checkpoints/servermind-neural-marathon"));
        let quant_out = export_quant_dir
            .unwrap_or_else(|| PathBuf::from(format!("{}-int8", out_dir.display())));
        return run_autonomous_marathon(
            MarathonParams {
                out_dir,
                quant_out_dir: quant_out,
                export_bits,
                total_duration: total_dur,
                eval_every_secs: eval_every_secs_opt,
                dataset_override: dataset_path,
                corpus_dirs_override: corpus_dirs,
                chunk_chars,
                resume,
                d_model: d_model_opt.unwrap_or(256),
                n_heads: n_heads_opt.unwrap_or(8),
                n_layers: n_layers_opt.unwrap_or(4),
                d_ff: d_ff_opt.unwrap_or(768),
                seq_len: seq_len_opt.unwrap_or(192),
                batch_size: batch_size_opt.unwrap_or(4),
                vocab_size: vocab_size_opt.unwrap_or(1536),
                lr_override: lr_opt,
                max_val_windows,
                device,
                install_ctrlc_handler,
                cancel_flag,
            },
            out,
            err,
        );
    }

    let out_dir = out_dir_opt.unwrap_or_else(|| PathBuf::from("checkpoints/servermind-neural-v1"));
    let max_steps = max_steps_opt.unwrap_or_else(|| {
        if max_duration_opt.is_some() {
            usize::MAX / 2
        } else {
            80
        }
    });
    let d_model = d_model_opt.unwrap_or(128);
    let n_heads = n_heads_opt.unwrap_or(4);
    let n_layers = n_layers_opt.unwrap_or(2);
    let d_ff = d_ff_opt.unwrap_or(256);
    let seq_len = seq_len_opt.unwrap_or(64);
    let batch_size = batch_size_opt.unwrap_or(4);
    let vocab_size = vocab_size_opt.unwrap_or(768);
    let learning_rate = lr_opt.unwrap_or(3e-3);

    let mut examples = Vec::new();
    for dir in &corpus_dirs {
        match crate::tokenizer::load_corpus_directory(dir, chunk_chars) {
            Ok(mut chunks) => examples.append(&mut chunks),
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        }
    }

    let effective_dataset = if let Some(ref ds) = dataset_path {
        Some(ds.clone())
    } else if corpus_dirs.is_empty() {
        Some(PathBuf::from("datasets/servermind_seed_v1.jsonl"))
    } else {
        None
    };

    if let Some(ds_path) = effective_dataset {
        match load_seed_corpus(&ds_path) {
            Ok(mut ds_examples) => examples.append(&mut ds_examples),
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        }
    }

    if examples.is_empty() {
        let _ = writeln!(err, "Nenhum exemplo válido carregado para treinamento.");
        return 1;
    }

    let resolved_run_name = run_name.unwrap_or_else(|| {
        out_dir
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("servermind-neural")
            .to_string()
    });
    let resolved_min_lr = min_learning_rate.unwrap_or(learning_rate * 0.1);
    if resolved_min_lr > learning_rate {
        let _ = writeln!(
            err,
            "'--min-lr' ({}) não pode ser maior que '--lr' ({}).",
            resolved_min_lr, learning_rate
        );
        return 2;
    }

    let split = crate::tokenizer::split_dataset(&examples, &resolved_run_name, 0.15, 0.10);
    let latest_ckpt = out_dir.join("latest.bin");
    let tok_path = out_dir.join("tokenizer.json");

    let (tokenizer, mut trainer) = if resume {
        if !tok_path.is_file() || !latest_ckpt.is_file() {
            let _ = writeln!(
                err,
                "Flag --resume exige '{}' e '{}' existentes.",
                tok_path.display(),
                latest_ckpt.display()
            );
            return 1;
        }
        let tokenizer = match crate::tokenizer::BpeTokenizer::load(&tok_path) {
            Ok(t) => t,
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        };
        let train_cfg = crate::training::TrainingConfig {
            run_name: resolved_run_name.clone(),
            checkpoint_dir: out_dir.clone(),
            max_steps,
            batch_size,
            seq_len,
            learning_rate,
            min_learning_rate: resolved_min_lr,
            warmup_steps: 0,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            eval_interval: (max_steps / 5).max(5),
            checkpoint_interval: (max_steps / 2).max(10),
            max_val_windows,
            device,
        };
        let mut resumed =
            match crate::training::Trainer::resume_from_checkpoint(&latest_ckpt, train_cfg) {
                Ok(t) => t,
                Err(e) => {
                    let _ = writeln!(err, "{}", e);
                    return 1;
                }
            };
        if tokenizer.vocab_size() != resumed.model.config.vocab_size {
            let _ = writeln!(
                err,
                "Incompatibilidade em --resume: tokenizer tem vocab={} mas checkpoint tem vocab={}.",
                tokenizer.vocab_size(),
                resumed.model.config.vocab_size
            );
            return 1;
        }
        if max_duration_opt.is_none() && max_steps <= resumed.current_step {
            let _ = writeln!(
                err,
                "'--steps' ({}) deve ser maior que o passo atual do checkpoint ({}).",
                max_steps, resumed.current_step
            );
            return 2;
        }
        resumed.config.seq_len = resumed.model.config.max_seq_len;
        if max_duration_opt.is_some() {
            resumed.config.eval_interval = 500;
            resumed.config.checkpoint_interval = 1_000;
        } else {
            let remaining = max_steps.saturating_sub(resumed.current_step);
            resumed.config.eval_interval = (remaining / 5).max(5);
            resumed.config.checkpoint_interval = (remaining / 2).max(10);
        }
        (tokenizer, resumed)
    } else {
        if let Err(msg) = check_no_existing_training_artifacts(&out_dir) {
            let _ = writeln!(err, "{}", msg);
            return 1;
        }
        let corpus_refs: Vec<&str> = split.train.iter().map(|s| s.as_str()).collect();
        let tokenizer = crate::tokenizer::BpeTokenizer::train_from_corpus(&corpus_refs, vocab_size);

        if let Err(e) = std::fs::create_dir_all(&out_dir) {
            let _ = writeln!(err, "Falha ao criar '{}': {}", out_dir.display(), e);
            return 1;
        }
        if let Err(e) = tokenizer.save(&tok_path) {
            let _ = writeln!(err, "{}", e);
            return 1;
        }

        let cfg = crate::neural::TransformerConfig {
            vocab_size: tokenizer.vocab_size(),
            max_seq_len: seq_len,
            d_model,
            n_heads,
            n_layers,
            d_ff,
        };
        let model = match crate::neural::ServerMindTransformer::new_random(cfg, 2026) {
            Ok(m) => m,
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        };
        let (warmup_s, eval_int, ckpt_int) = if max_duration_opt.is_some() {
            (10, 500, 1_000)
        } else {
            (
                (max_steps / 10).max(2),
                (max_steps / 5).max(5),
                (max_steps / 2).max(10),
            )
        };
        let train_cfg = crate::training::TrainingConfig {
            run_name: resolved_run_name.clone(),
            checkpoint_dir: out_dir.clone(),
            max_steps,
            batch_size,
            seq_len,
            learning_rate,
            min_learning_rate: resolved_min_lr,
            warmup_steps: warmup_s,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            eval_interval: eval_int,
            checkpoint_interval: ckpt_int,
            max_val_windows,
            device,
        };
        (tokenizer, crate::training::Trainer::new(model, train_cfg))
    };

    let active_seq_len = trainer.model.config.max_seq_len;
    let use_chat_mask = corpus_dirs.is_empty();
    let val_src = if split.validation.is_empty() {
        &split.train
    } else {
        &split.validation
    };
    let (train_windows, val_windows, masked_train_seqs, masked_val_seqs) = if use_chat_mask {
        (
            Vec::new(),
            Vec::new(),
            Some(crate::training::pack_chat_masked_sequences(
                &tokenizer,
                &split.train,
                active_seq_len,
            )),
            Some(crate::training::pack_chat_masked_sequences(
                &tokenizer,
                val_src,
                active_seq_len,
            )),
        )
    } else {
        (
            crate::training::pack_sequences(&tokenizer, &split.train, active_seq_len),
            crate::training::pack_eval_sequences(&tokenizer, val_src, active_seq_len),
            None,
            None,
        )
    };
    let effective_train_count = masked_train_seqs
        .as_ref()
        .map(|m| m.len())
        .unwrap_or(train_windows.len());
    let effective_val_count = masked_val_seqs
        .as_ref()
        .map(|m| m.len())
        .unwrap_or(val_windows.len());
    let total_params = trainer.model.config.total_params();

    let target_desc = if let Some(dur) = max_duration_opt {
        format!("duração={}", format_hms(dur))
    } else {
        format!("step={} -> {}", trainer.current_step, max_steps)
    };
    let _ = writeln!(
        out,
        "Iniciando treinamento ServerMind Neural ({}): params={} | vocab={} | seq_len={} | lr={:.5} | device={} | train_windows={} | val_windows={} | {}",
        resolved_run_name,
        total_params,
        tokenizer.vocab.len(),
        active_seq_len,
        learning_rate,
        device.as_str(),
        effective_train_count,
        effective_val_count,
        target_desc
    );
    let _ = out.flush();

    let mut recorder = if max_duration_opt.is_some() || eval_every_secs_opt.is_some() {
        let mut rec = EvolutionSequenceRecorder::new(
            &out_dir,
            &resolved_run_name,
            total_params,
            max_duration_opt,
        );
        rec.record_milestone(
            "Baseline Inicial (Step 0 / Pré-Sessão)",
            &trainer.model,
            &tokenizer,
            trainer.current_step,
            None,
            None,
            0.0,
            learning_rate,
            0.0,
            Duration::ZERO,
            out,
        );
        Some(rec)
    } else {
        None
    };

    let auto_eval_secs =
        eval_every_secs_opt.or_else(|| max_duration_opt.map(|d| (d.as_secs() / 24).clamp(2, 900)));

    let target_steps = max_steps;
    let run_res = if let Some(ref mut rec) = recorder {
        let tok_ref = &tokenizer;
        let mut milestone_cb = |model: &crate::neural::ServerMindTransformer,
                                m: &crate::training::TrainStepMetrics,
                                tok_s: f64,
                                elapsed: Duration| {
            rec.record_milestone(
                "Sessão Contínua",
                model,
                tok_ref,
                m.step,
                Some(m.train_loss),
                m.val_loss,
                m.grad_norm,
                m.learning_rate,
                tok_s,
                elapsed,
                out,
            );
        };
        let opts = crate::training::TimedSessionOptions {
            max_duration: max_duration_opt,
            eval_every_secs: auto_eval_secs,
            cancel_flag,
            install_ctrlc_handler,
            on_milestone: Some(&mut milestone_cb),
            masked_train: masked_train_seqs.as_deref(),
            masked_val: masked_val_seqs.as_deref(),
        };
        trainer.train_with_session_options(
            &train_windows,
            &val_windows,
            tokenizer.pad_id(),
            None,
            opts,
        )
    } else {
        let mut progress_cb = |m: &crate::training::TrainStepMetrics, tok_s: f64| {
            let val_str = m
                .val_loss
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".to_string());
            let _ = writeln!(
                out,
                "  [step {}/{}] train_loss={:.4} | val_loss={} | grad_norm={:.3} | lr={:.5} | {:.1} tok/s",
                m.step, target_steps, m.train_loss, val_str, m.grad_norm, m.learning_rate, tok_s
            );
            let _ = out.flush();
        };
        let opts = crate::training::TimedSessionOptions {
            max_duration: None,
            eval_every_secs: None,
            cancel_flag,
            install_ctrlc_handler,
            on_milestone: None,
            masked_train: masked_train_seqs.as_deref(),
            masked_val: masked_val_seqs.as_deref(),
        };
        trainer.train_with_session_options(
            &train_windows,
            &val_windows,
            tokenizer.pad_id(),
            Some(&mut progress_cb),
            opts,
        )
    };

    match run_res {
        Ok(report) => {
            if let Some(ref rec) = recorder {
                let _ = rec.write_markdown_summary(&report.active_device, report.cancelled);
                let _ = writeln!(
                    out,
                    "Sequência evolutiva completa salva em '{}' e '{}'",
                    rec.md_path.display(),
                    rec.jsonl_path.display()
                );
            }
            let status_lbl = if report.cancelled {
                "pausado com segurança (Ctrl+C)"
            } else if report.stopped_by_duration {
                "concluído no tempo programado"
            } else {
                "concluído"
            };
            let _ = writeln!(
                out,
                "Treinamento {} [{}]: steps={} | initial_loss={:.4} -> final_loss={:.4} | best_val_loss={:.4} | {:.1} tok/s | elapsed={}",
                status_lbl,
                report.active_device,
                report.steps_completed,
                report.initial_train_loss,
                report.final_train_loss,
                report.best_val_loss,
                report.tokens_per_sec,
                format_hms(Duration::from_millis(report.total_elapsed_ms as u64))
            );
            let _ = writeln!(out, "Artefatos salvos em '{}'", out_dir.display());
            if let Some(ref qdir) = export_quant_dir {
                match export_quantized_package(&out_dir, qdir, export_bits) {
                    Ok(info) => {
                        let _ = writeln!(
                            out,
                            "Pacote quantizado SMQ1 ({}) exportado para '{}': disco {:.2} MB -> {:.2} MB ({:.2}x menor) | pesos RAM {:.2} MB -> {:.2} MB",
                            info.precision.as_str(),
                            qdir.display(),
                            info.src_disk_mb,
                            info.dst_disk_mb,
                            info.compression_ratio,
                            info.src_ram_mb,
                            info.dst_ram_mb
                        );
                    }
                    Err(e) => {
                        let _ = writeln!(err, "Erro ao exportar pacote quantizado: {}", e);
                        return 1;
                    }
                }
            }
            if report.cancelled {
                130
            } else {
                0
            }
        }
        Err(e) => {
            let _ = writeln!(err, "Erro no treinamento: {}", e);
            1
        }
    }
}

fn format_hms(d: Duration) -> String {
    let total = d.as_secs();
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    format!("{:02}h:{:02}m:{:02}s", h, m, s)
}

#[derive(Debug, Clone, serde::Serialize)]
struct ProbeOutputSample {
    domain: String,
    prompt: String,
    response: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct EvolutionMilestoneEntry {
    index: usize,
    phase: String,
    elapsed_hms: String,
    elapsed_secs: f64,
    step: usize,
    train_loss: Option<f32>,
    val_loss: Option<f32>,
    perplexity: Option<f32>,
    grad_norm: f32,
    learning_rate: f32,
    tokens_per_sec: f64,
    samples: Vec<ProbeOutputSample>,
}

struct EvolutionSequenceRecorder {
    run_name: String,
    total_params: usize,
    planned_duration: Option<Duration>,
    pub md_path: PathBuf,
    pub jsonl_path: PathBuf,
    entries: Vec<EvolutionMilestoneEntry>,
}

impl EvolutionSequenceRecorder {
    fn new(
        out_dir: &std::path::Path,
        run_name: &str,
        total_params: usize,
        planned_duration: Option<Duration>,
    ) -> Self {
        let _ = std::fs::create_dir_all(out_dir);
        let md_path = out_dir.join("training_evolution.md");
        let jsonl_path = out_dir.join("evolution_log.jsonl");
        Self {
            run_name: run_name.to_string(),
            total_params,
            planned_duration,
            md_path,
            jsonl_path,
            entries: Vec::new(),
        }
    }

    fn probe_prompts() -> [(&'static str, &'static str); 4] {
        [
            (
                "identity",
                "Olá! Quem criou você e no que você pode me ajudar?",
            ),
            (
                "linux",
                "Como diagnosticar uso alto de CPU e memória em um servidor Linux sem interromper processos?",
            ),
            (
                "docker",
                "Um container Docker reinicia em loop (CrashLoop). Como descobrir a causa raiz?",
            ),
            (
                "network",
                "Uma API retorna 502 Bad Gateway atrás do Nginx. Quais verificações fazer em ordem?",
            ),
        ]
    }

    fn sample_model_reply(
        model: &crate::neural::ServerMindTransformer,
        tok: &crate::tokenizer::BpeTokenizer,
        prompt: &str,
        seed: u64,
    ) -> String {
        let formatted = format!("<user> {}\n<assistant> ", prompt.trim());
        let mut p_ids = tok.encode(&formatted, true, false);
        let max_p = model.config.max_seq_len.saturating_sub(40).max(8);
        if p_ids.len() > max_p {
            let start = p_ids.len() - (max_p - 1);
            let mut trunc = vec![tok.bos_id()];
            trunc.extend_from_slice(&p_ids[start..]);
            p_ids = trunc;
        }
        let mut stops = vec![tok.eos_id()];
        for sp in [
            crate::tokenizer::USER_TOKEN,
            crate::tokenizer::SYSTEM_TOKEN,
            crate::tokenizer::ASSISTANT_TOKEN,
        ] {
            if let Some(id) = tok.token_id(sp) {
                if !stops.contains(&id) {
                    stops.push(id);
                }
            }
        }
        let gen = model.generate_with_kv_precision(
            &p_ids,
            40,
            0.2,
            5,
            &stops,
            seed,
            crate::neural::KvCachePrecision::Fp16,
            None,
        );
        let new_slice = if gen.len() > p_ids.len() {
            &gen[p_ids.len()..]
        } else {
            &[]
        };
        let clean_ids = match new_slice.last() {
            Some(last) if stops.contains(last) => &new_slice[..new_slice.len() - 1],
            _ => new_slice,
        };
        let decoded = tok
            .decode(clean_ids)
            .replace(crate::tokenizer::EOS_TOKEN, "")
            .replace(crate::tokenizer::PAD_TOKEN, "")
            .replace(crate::tokenizer::BOS_TOKEN, "")
            .replace('\n', " ");
        let trimmed = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
        if trimmed.is_empty() {
            "(vazio)".to_string()
        } else {
            trimmed.chars().take(180).collect()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_milestone(
        &mut self,
        phase: &str,
        model: &crate::neural::ServerMindTransformer,
        tok: &crate::tokenizer::BpeTokenizer,
        step: usize,
        train_loss: Option<f32>,
        val_loss: Option<f32>,
        grad_norm: f32,
        learning_rate: f32,
        tokens_per_sec: f64,
        elapsed: Duration,
        out: &mut dyn Write,
    ) {
        let idx = self.entries.len();
        let mut samples = Vec::with_capacity(4);
        for (k, (dom, q)) in Self::probe_prompts().iter().enumerate() {
            let reply = Self::sample_model_reply(model, tok, q, 0x2026_5EED_u64 + k as u64);
            samples.push(ProbeOutputSample {
                domain: (*dom).to_string(),
                prompt: (*q).to_string(),
                response: reply,
            });
        }
        let ppl = val_loss
            .or(train_loss)
            .filter(|l| l.is_finite())
            .map(|l| (l as f64).exp().min(999_999.0) as f32);

        let entry = EvolutionMilestoneEntry {
            index: idx,
            phase: phase.to_string(),
            elapsed_hms: format_hms(elapsed),
            elapsed_secs: elapsed.as_secs_f64(),
            step,
            train_loss,
            val_loss,
            perplexity: ppl,
            grad_norm,
            learning_rate,
            tokens_per_sec,
            samples,
        };

        if let Ok(mut jf) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.jsonl_path)
        {
            if let Ok(line) = serde_json::to_string(&entry) {
                let _ = writeln!(jf, "{}", line);
            }
        }

        let eta_str = match self.planned_duration {
            Some(tot) if elapsed < tot => format!(" | ETA={}", format_hms(tot - elapsed)),
            Some(_) => " | ETA=00h:00m:00s".to_string(),
            None => String::new(),
        };
        let tl_s = train_loss
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "-".to_string());
        let vl_s = val_loss
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "-".to_string());
        let ppl_s = ppl
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "-".to_string());

        let _ = writeln!(
            out,
            "\n┌─ [MARCO #{:02} | {} | t={}{}] step={} | train_loss={} | val_loss={} | ppl={} | lr={:.5} | {:.1} tok/s",
            entry.index,
            entry.phase,
            entry.elapsed_hms,
            eta_str,
            entry.step,
            tl_s,
            vl_s,
            ppl_s,
            entry.learning_rate,
            entry.tokens_per_sec
        );
        for s in &entry.samples {
            let _ = writeln!(out, "│  [{}] Q: {}", s.domain, s.prompt);
            let _ = writeln!(out, "│         A: {}", s.response);
        }
        let _ = writeln!(
            out,
            "└──────────────────────────────────────────────────────────────────────────"
        );
        let _ = out.flush();

        self.entries.push(entry);
        let _ = self.write_markdown_summary("em andamento", false);
    }

    fn write_markdown_summary(&self, active_device: &str, cancelled: bool) -> Result<(), String> {
        let mut md = String::new();
        md.push_str(&format!(
            "# Sequência Evolutiva de Treinamento — `{}`\n\n",
            self.run_name
        ));
        md.push_str(&format!(
            "- **Parâmetros:** `{}` (`{:.2}M`)\n- **Dispositivo:** `{}`\n- **Status:** `{}`\n",
            self.total_params,
            self.total_params as f64 / 1e6,
            active_device,
            if cancelled {
                "Pausado pelo usuário (Ctrl+C, salvo)"
            } else {
                "Ativo / Concluído"
            }
        ));
        if let Some(d) = self.planned_duration {
            md.push_str(&format!("- **Duração Alvo:** `{}`\n", format_hms(d)));
        }
        md.push_str("\n## 1. Curva Cronológica de Métricas\n\n");
        md.push_str("| Marco | Fase | Tempo | Step | `train_loss` | `val_loss` | Perplexidade | `lr` | `tok/s` |\n");
        md.push_str("| ---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for e in &self.entries {
            let tl = e
                .train_loss
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".into());
            let vl = e
                .val_loss
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".into());
            let pp = e
                .perplexity
                .map(|v| format!("{:.2}", v))
                .unwrap_or_else(|| "-".into());
            md.push_str(&format!(
                "| **#{:02}** | {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{:.5}` | `{:.1}` |\n",
                e.index,
                e.phase,
                e.elapsed_hms,
                e.step,
                tl,
                vl,
                pp,
                e.learning_rate,
                e.tokens_per_sec
            ));
        }

        md.push_str("\n## 2. Evolução Real das Respostas Marco a Marco\n\n");
        for e in &self.entries {
            let tl = e
                .train_loss
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".into());
            let vl = e
                .val_loss
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".into());
            md.push_str(&format!(
                "### Marco #{:02} — `{}` (`{}` | `step={}` | `train_loss={}` | `val_loss={}`)\n\n",
                e.index, e.phase, e.elapsed_hms, e.step, tl, vl
            ));
            for s in &e.samples {
                md.push_str(&format!(
                    "- **`[{}]` Pergunta:** {}\n  - **Resposta gerada:** `{}`\n",
                    s.domain, s.prompt, s.response
                ));
            }
            md.push('\n');
        }

        std::fs::write(&self.md_path, md)
            .map_err(|e| format!("Falha ao gravar '{}': {}", self.md_path.display(), e))
    }
}

struct MarathonParams {
    out_dir: PathBuf,
    quant_out_dir: PathBuf,
    export_bits: crate::neural::WeightPrecision,
    total_duration: Duration,
    eval_every_secs: Option<u64>,
    dataset_override: Option<PathBuf>,
    corpus_dirs_override: Vec<PathBuf>,
    chunk_chars: usize,
    resume: bool,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    d_ff: usize,
    seq_len: usize,
    batch_size: usize,
    vocab_size: usize,
    lr_override: Option<f32>,
    max_val_windows: usize,
    device: crate::neural::ComputeDevice,
    install_ctrlc_handler: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
}

fn load_qa_datasets_combined(explicit_ds: Option<&PathBuf>) -> Result<Vec<String>, String> {
    let mut qa = Vec::new();
    if let Some(ds) = explicit_ds {
        qa.extend(load_seed_corpus(ds)?);
    } else {
        let v3 = std::path::Path::new("datasets/servermind_corpus_v3_balanced.jsonl");
        if v3.is_file() {
            qa.extend(load_seed_corpus(v3)?);
        } else {
            for cand in [
                "datasets/servermind_corpus_v2.jsonl",
                "datasets/servermind_seed_v1.jsonl",
            ] {
                let p = std::path::Path::new(cand);
                if p.is_file() {
                    qa.extend(load_seed_corpus(p)?);
                }
            }
        }
    }
    if qa.is_empty() {
        return Err("Nenhum exemplo Q&A encontrado para o treinamento maratona.".to_string());
    }
    Ok(qa)
}

fn run_autonomous_marathon(
    params: MarathonParams,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    let qa_examples = match load_qa_datasets_combined(params.dataset_override.as_ref()) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(err, "{}", e);
            return 1;
        }
    };

    let mut book_chunks = Vec::new();
    let book_dirs = if !params.corpus_dirs_override.is_empty() {
        params.corpus_dirs_override.clone()
    } else {
        let default_books = PathBuf::from("datasets/pretrain_books");
        if default_books.is_dir() {
            vec![default_books]
        } else {
            Vec::new()
        }
    };
    for bdir in &book_dirs {
        match crate::tokenizer::load_corpus_directory(bdir, params.chunk_chars) {
            Ok(mut c) => book_chunks.append(&mut c),
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        }
    }

    let has_pretrain_books = !book_chunks.is_empty();
    // Subamostra até 400 blocos representativos dos livros para treinar o BPE em < 3s,
    // mantendo todos os `book_chunks` completos para o treinamento neural da Fase 1!
    let mut combined_for_vocab = Vec::new();
    if book_chunks.len() > 400 {
        let step = book_chunks.len() / 400;
        for idx in (0..book_chunks.len()).step_by(step.max(1)).take(400) {
            combined_for_vocab.push(book_chunks[idx].clone());
        }
    } else {
        combined_for_vocab.extend(book_chunks.iter().cloned());
    }
    // Repete Q&A 6x no vocabulário para que comandos técnicos (`systemctl`, `docker`, `ss -tulnp`, `nginx`)
    // recebam tokens BPE dedicados de alta prioridade!
    for _ in 0..6 {
        combined_for_vocab.extend(qa_examples.iter().cloned());
    }

    let run_name = params
        .out_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("servermind-neural-marathon")
        .to_string();
    let tok_path = params.out_dir.join("tokenizer.json");
    let latest_ckpt = params.out_dir.join("latest.bin");

    if !params.resume {
        if let Err(msg) = check_no_existing_training_artifacts(&params.out_dir) {
            let _ = writeln!(err, "{}", msg);
            return 1;
        }
    }

    let (tokenizer, mut trainer) = if params.resume {
        if !tok_path.is_file() || !latest_ckpt.is_file() {
            let _ = writeln!(
                err,
                "Flag --resume exige '{}' e '{}' existentes.",
                tok_path.display(),
                latest_ckpt.display()
            );
            return 1;
        }
        let tok = match crate::tokenizer::BpeTokenizer::load(&tok_path) {
            Ok(t) => t,
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        };
        let cfg = crate::training::TrainingConfig {
            run_name: run_name.clone(),
            checkpoint_dir: params.out_dir.clone(),
            max_steps: usize::MAX / 2,
            batch_size: params.batch_size,
            seq_len: params.seq_len,
            learning_rate: params.lr_override.unwrap_or(1.2e-3),
            min_learning_rate: 5e-5,
            warmup_steps: 10,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            eval_interval: 1_000,
            checkpoint_interval: 2_000,
            max_val_windows: params.max_val_windows,
            device: params.device,
        };
        let tr = match crate::training::Trainer::resume_from_checkpoint(&latest_ckpt, cfg) {
            Ok(t) => t,
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        };
        (tok, tr)
    } else {
        if let Err(e) = std::fs::create_dir_all(&params.out_dir) {
            let _ = writeln!(err, "Falha ao criar '{}': {}", params.out_dir.display(), e);
            return 1;
        }
        let _ = writeln!(
            out,
            "[1/3] Treinando tokenizador BPE próprio (vocab_alvo={}) sobre {} blocos de amostra...",
            params.vocab_size,
            combined_for_vocab.len()
        );
        let _ = out.flush();
        let refs: Vec<&str> = combined_for_vocab.iter().map(|s| s.as_str()).collect();
        let tok = crate::tokenizer::BpeTokenizer::train_from_corpus(&refs, params.vocab_size);
        if let Err(e) = tok.save(&tok_path) {
            let _ = writeln!(err, "{}", e);
            return 1;
        }
        let mcfg = crate::neural::TransformerConfig {
            vocab_size: tok.vocab_size(),
            max_seq_len: params.seq_len,
            d_model: params.d_model,
            n_heads: params.n_heads,
            n_layers: params.n_layers,
            d_ff: params.d_ff,
        };
        let model = match crate::neural::ServerMindTransformer::new_random(mcfg, 2026) {
            Ok(m) => m,
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        };
        let tcfg = crate::training::TrainingConfig {
            run_name: run_name.clone(),
            checkpoint_dir: params.out_dir.clone(),
            max_steps: usize::MAX / 2,
            batch_size: params.batch_size,
            seq_len: params.seq_len,
            learning_rate: params.lr_override.unwrap_or(1.5e-3),
            min_learning_rate: 2.5e-4,
            warmup_steps: 20,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            eval_interval: 1_000,
            checkpoint_interval: 2_000,
            max_val_windows: params.max_val_windows,
            device: params.device,
        };
        (tok, crate::training::Trainer::new(model, tcfg))
    };

    let active_seq_len = trainer.model.config.max_seq_len;
    let total_params = trainer.model.config.total_params();

    // Divide a sessão total em 2 estágios quando há livros de pré-treino:
    // Estágio 1 (35% do tempo): Pré-Treino Geral (Livros + Q&A)
    // Estágio 2 (65% do tempo): Fine-Tuning Supervisionado Focado em Q&A (`<user> ... <assistant>`)
    let total_secs = params.total_duration.as_secs_f64().max(1.0);
    let (stage1_dur, stage2_dur) = if has_pretrain_books {
        let s1 = Duration::from_secs_f64((total_secs * 0.35).max(0.5));
        let s2 = params
            .total_duration
            .saturating_sub(s1)
            .max(Duration::from_millis(500));
        (Some(s1), s2)
    } else {
        (None, params.total_duration)
    };

    // Cerca de ~24 marcos distribuídos ao longo da sessão completa (ex.: a cada 15 min numa sessão de 6h)
    let eval_every = params
        .eval_every_secs
        .unwrap_or_else(|| ((total_secs / 24.0).round() as u64).clamp(1, 900));

    let qa_split = crate::tokenizer::split_dataset(&qa_examples, &run_name, 0.12, 0.05);
    let qa_train_masked =
        crate::training::pack_chat_masked_sequences(&tokenizer, &qa_split.train, active_seq_len);
    let qa_val_src = if qa_split.validation.is_empty() {
        &qa_split.train
    } else {
        &qa_split.validation
    };
    let qa_val_masked =
        crate::training::pack_chat_masked_sequences(&tokenizer, qa_val_src, active_seq_len);

    let _ = writeln!(
        out,
        "=========================================================================="
    );
    let _ = writeln!(
        out,
        "  MARATONA AUTÔNOMA SERVERMIND NEURAL — DURAÇÃO ALVO: {}",
        format_hms(params.total_duration)
    );
    let _ = writeln!(
        out,
        "  Modelo: {:.2}M params (d={}, L={}, H={}, seq={}, vocab={}) | device={}",
        total_params as f64 / 1e6,
        trainer.model.config.d_model,
        trainer.model.config.n_layers,
        trainer.model.config.n_heads,
        active_seq_len,
        tokenizer.vocab_size(),
        params.device.as_str()
    );
    if let Some(s1) = stage1_dur {
        let _ = writeln!(
            out,
            "  Currículo: Estágio 1 Pré-Treino ({}) -> Estágio 2 Chat Fine-Tuning ({})",
            format_hms(s1),
            format_hms(stage2_dur)
        );
    } else {
        let _ = writeln!(
            out,
            "  Currículo: Fine-Tuning Direto em Q&A ({})",
            format_hms(stage2_dur)
        );
    }
    let _ = writeln!(
        out,
        "  Marcos evolutivos a cada ~{}s -> '{}/training_evolution.md' (Ctrl+C salva limpo)",
        eval_every,
        params.out_dir.display()
    );
    let _ = writeln!(
        out,
        "=========================================================================="
    );
    let _ = out.flush();

    let mut recorder = EvolutionSequenceRecorder::new(
        &params.out_dir,
        &run_name,
        total_params,
        Some(params.total_duration),
    );
    let global_start = std::time::Instant::now();

    recorder.record_milestone(
        "Marco Zero (Antes do Treino)",
        &trainer.model,
        &tokenizer,
        trainer.current_step,
        None,
        None,
        0.0,
        trainer.config.learning_rate,
        0.0,
        Duration::ZERO,
        out,
    );

    let mut active_device_str = params.device.as_str().to_string();
    let mut was_cancelled = false;

    // --- ESTÁGIO 1: PRÉ-TREINO EM LIVROS + Q&A ---
    if let Some(s1_dur) = stage1_dur {
        let mut stage1_corpus = book_chunks;
        for _ in 0..4 {
            stage1_corpus.extend(qa_split.train.iter().cloned());
        }
        let s1_train_wins =
            crate::training::pack_sequences(&tokenizer, &stage1_corpus, active_seq_len);
        trainer.config.learning_rate = params.lr_override.unwrap_or(1.5e-3);
        trainer.config.min_learning_rate = 3.0e-4;
        trainer.config.eval_interval = usize::MAX / 4; // Guiado por relógio (`eval_every`)

        let s1_res = {
            let tok_ref = &tokenizer;
            let mut m_cb = |model: &crate::neural::ServerMindTransformer,
                            m: &crate::training::TrainStepMetrics,
                            tok_s: f64,
                            _st_elapsed: Duration| {
                recorder.record_milestone(
                    "Fase 1/2 — Pré-Treino Base",
                    model,
                    tok_ref,
                    m.step,
                    Some(m.train_loss),
                    m.val_loss,
                    m.grad_norm,
                    m.learning_rate,
                    tok_s,
                    global_start.elapsed(),
                    out,
                );
            };
            let opts = crate::training::TimedSessionOptions {
                max_duration: Some(s1_dur),
                eval_every_secs: Some(eval_every),
                cancel_flag: params.cancel_flag.clone(),
                install_ctrlc_handler: params.install_ctrlc_handler,
                on_milestone: Some(&mut m_cb),
                masked_train: None,
                masked_val: Some(&qa_val_masked),
            };
            trainer.train_with_session_options(&s1_train_wins, &[], tokenizer.pad_id(), None, opts)
        };

        match s1_res {
            Ok(rep) => {
                active_device_str = rep.active_device;
                was_cancelled = rep.cancelled;
                let _ = writeln!(
                    out,
                    "\n>> [Transição de Fase] Estágio 1 finalizado em step={} (loss {:.4} -> {:.4}). Iniciando Estágio 2 (Chat Fine-Tuning)...",
                    rep.steps_completed, rep.initial_train_loss, rep.final_train_loss
                );
                let _ = out.flush();
            }
            Err(e) => {
                let _ = writeln!(err, "Erro no Estágio 1 da maratona: {}", e);
                return 1;
            }
        }
    }

    // --- ESTÁGIO 2: FINE-TUNING SUPERVISIONADO EM CHAT TÉCNICO (`<user> ... <assistant>`) ---
    if !was_cancelled {
        // Reinicia `best_val_loss` especificamente para selecionar o melhor modelo de diálogo na Fase 2
        trainer.best_val_loss = match crate::training::evaluate_masked_loss(
            &trainer.model,
            &qa_val_masked,
            tokenizer.pad_id(),
        ) {
            Some(v) if v.is_finite() => v,
            Some(non_finite) => {
                let _ = writeln!(
                    err,
                    "[AVISO DE VALIDAÇÃO] Início da Fase 2 com val_loss não finita ({}); best_val_loss mantido em INFINITY.",
                    non_finite
                );
                f32::INFINITY
            }
            None => f32::INFINITY,
        };
        let _ = crate::neural::save_checkpoint(
            &params.out_dir.join("best.bin"),
            &trainer.model,
            Some(&trainer.optimizer),
            trainer.current_step,
            trainer.best_val_loss,
        );

        trainer.config.learning_rate = params.lr_override.map(|v| v * 0.4).unwrap_or(5.0e-4);
        trainer.config.min_learning_rate = 3.0e-5;
        trainer.config.eval_interval = usize::MAX / 4;

        let s2_res = {
            let tok_ref = &tokenizer;
            let mut m_cb = |model: &crate::neural::ServerMindTransformer,
                            m: &crate::training::TrainStepMetrics,
                            tok_s: f64,
                            _st_elapsed: Duration| {
                recorder.record_milestone(
                    "Fase 2/2 — Chat Fine-Tuning",
                    model,
                    tok_ref,
                    m.step,
                    Some(m.train_loss),
                    m.val_loss,
                    m.grad_norm,
                    m.learning_rate,
                    tok_s,
                    global_start.elapsed(),
                    out,
                );
            };
            let opts = crate::training::TimedSessionOptions {
                max_duration: Some(stage2_dur),
                eval_every_secs: Some(eval_every),
                cancel_flag: params.cancel_flag.clone(),
                install_ctrlc_handler: params.install_ctrlc_handler,
                on_milestone: Some(&mut m_cb),
                masked_train: Some(&qa_train_masked),
                masked_val: Some(&qa_val_masked),
            };
            trainer.train_with_session_options(&[], &[], tokenizer.pad_id(), None, opts)
        };

        match s2_res {
            Ok(rep) => {
                active_device_str = rep.active_device;
                was_cancelled = rep.cancelled;
                // Garante um marco final consolidado ao encerrar a Fase 2
                let final_val = crate::training::evaluate_masked_loss(
                    &trainer.model,
                    &qa_val_masked,
                    tokenizer.pad_id(),
                )
                .filter(|v| v.is_finite());
                recorder.record_milestone(
                    if was_cancelled {
                        "Marco Final (Pausado em Ctrl+C)"
                    } else {
                        "Marco Final Consolidado"
                    },
                    &trainer.model,
                    &tokenizer,
                    rep.steps_completed,
                    Some(rep.final_train_loss),
                    final_val,
                    0.0,
                    trainer.config.min_learning_rate,
                    rep.tokens_per_sec,
                    global_start.elapsed(),
                    out,
                );
            }
            Err(e) => {
                let _ = writeln!(err, "Erro no Estágio 2 da maratona: {}", e);
                return 1;
            }
        }
    }

    let _ = recorder.write_markdown_summary(&active_device_str, was_cancelled);

    match export_quantized_package(&params.out_dir, &params.quant_out_dir, params.export_bits) {
        Ok(st) => {
            let _ = writeln!(
                out,
                "\n[Pacote SMQ1 ({}) atualizado em '{}': disco {:.2} MB -> {:.2} MB ({:.2}x menor) | RAM pesos {:.2} MB -> {:.2} MB]",
                st.precision.as_str(),
                params.quant_out_dir.display(),
                st.src_disk_mb,
                st.dst_disk_mb,
                st.compression_ratio,
                st.src_ram_mb,
                st.dst_ram_mb
            );
        }
        Err(e) => {
            let _ = writeln!(err, "Aviso ao exportar pacote quantizado: {}", e);
        }
    }

    let _ = writeln!(
        out,
        "=========================================================================="
    );
    let _ = writeln!(
        out,
        "  SEQUÊNCIA EVOLUTIVA GRAVADA EM: {}",
        recorder.md_path.display()
    );
    let _ = writeln!(
        out,
        "  LOG JSONL CRONOLÓGICO         : {}",
        recorder.jsonl_path.display()
    );
    let _ = writeln!(
        out,
        "  PARA CONVERSAR NO CHAT        : :backend neural {}",
        params.out_dir.display()
    );
    let _ = writeln!(
        out,
        "=========================================================================="
    );

    if was_cancelled {
        130
    } else {
        0
    }
}

struct QuantExportStats {
    precision: crate::neural::WeightPrecision,
    src_disk_mb: f64,
    dst_disk_mb: f64,
    src_ram_mb: f64,
    dst_ram_mb: f64,
    compression_ratio: f64,
}

fn export_quantized_package(
    src_dir: &std::path::Path,
    dst_dir: &std::path::Path,
    precision: crate::neural::WeightPrecision,
) -> Result<QuantExportStats, String> {
    let tok_src = src_dir.join("tokenizer.json");
    if !tok_src.is_file() {
        return Err(format!(
            "Tokenizer '{}' não encontrado no diretório de origem.",
            tok_src.display()
        ));
    }
    let best_path = src_dir.join("best.bin");
    let latest_path = src_dir.join("latest.bin");
    let src_ckpt = if best_path.is_file() {
        best_path
    } else if latest_path.is_file() {
        latest_path
    } else {
        return Err(format!(
            "Nenhum checkpoint FP32 ('best.bin' ou 'latest.bin') encontrado em '{}'.",
            src_dir.display()
        ));
    };

    let src_disk_bytes = std::fs::metadata(&src_ckpt)
        .map_err(|e| e.to_string())?
        .len();
    let (fp32_model, _opt, meta) = crate::neural::load_checkpoint(&src_ckpt)?;
    let src_ram_bytes = fp32_model.actual_weight_bytes();

    let qmodel = crate::neural::quantize_model_in_memory(&fp32_model, precision)?;
    let dst_ram_bytes = qmodel.actual_weight_bytes();

    std::fs::create_dir_all(dst_dir).map_err(|e| {
        format!(
            "Falha ao criar pasta destino '{}': {}",
            dst_dir.display(),
            e
        )
    })?;
    let tok_dst = dst_dir.join("tokenizer.json");
    if src_dir != dst_dir {
        std::fs::copy(&tok_src, &tok_dst).map_err(|e| {
            format!(
                "Falha ao copiar tokenizer para '{}': {}",
                tok_dst.display(),
                e
            )
        })?;
    }
    let smq_path = dst_dir.join("quantized.smq");
    crate::neural::save_quantized_checkpoint(&smq_path, &qmodel, meta.step, meta.best_val_loss)?;
    let dst_disk_bytes = std::fs::metadata(&smq_path)
        .map_err(|e| e.to_string())?
        .len();

    let mb = 1024.0 * 1024.0;
    let src_disk_mb = src_disk_bytes as f64 / mb;
    let dst_disk_mb = dst_disk_bytes as f64 / mb;
    Ok(QuantExportStats {
        precision,
        src_disk_mb,
        dst_disk_mb,
        src_ram_mb: src_ram_bytes as f64 / mb,
        dst_ram_mb: dst_ram_bytes as f64 / mb,
        compression_ratio: src_disk_mb / dst_disk_mb.max(1e-9),
    })
}

fn handle_quantize_command(arg: &str) {
    let parts: Vec<&str> = arg.split_whitespace().collect();
    if parts.len() < 2 {
        println!("Uso: :quantize <pasta_ckpt_fp32> <pasta_saida_smq> [fp16|bf16|int8|int4]");
        return;
    }
    let bits_raw = parts.get(2).copied().unwrap_or("int8");
    let prec = match crate::neural::WeightPrecision::parse(bits_raw) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };
    match export_quantized_package(
        std::path::Path::new(parts[0]),
        std::path::Path::new(parts[1]),
        prec,
    ) {
        Ok(st) => {
            println!(
                "Pacote SMQ1 ({}) gravado em '{}': disco {:.2} MB -> {:.2} MB ({:.2}x menor) | RAM pesos {:.2} MB -> {:.2} MB.",
                st.precision.as_str(),
                parts[1],
                st.src_disk_mb,
                st.dst_disk_mb,
                st.compression_ratio,
                st.src_ram_mb,
                st.dst_ram_mb
            );
        }
        Err(e) => eprintln!("Erro ao quantizar checkpoint: {}", e),
    }
}

pub fn run_eval_neural_cli(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let mut ckpt_dir = PathBuf::from("checkpoints/servermind-neural-v1");
    let mut dataset_path = PathBuf::from("datasets/servermind_seed_v1.jsonl");
    let mut quantize_out: Option<PathBuf> = None;
    let mut bits_opt: Option<crate::neural::WeightPrecision> = None;
    let mut kv_cache_opt: Option<crate::neural::KvCachePrecision> = None;
    let mut low_ram = false;
    let mut max_ram_mb_opt: Option<usize> = None;
    let mut threads_opt: Option<usize> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                let _ = writeln!(out, "Uso: servermind-ai eval-neural [opções]");
                let _ = writeln!(out, "  --dir <pasta>                 Diretório do checkpoint ('best.bin', 'quantized.smq' ou 'latest.bin')");
                let _ = writeln!(out, "  --dataset <arquivo.jsonl>     Dataset JSONL de validação (padrão: datasets/servermind_seed_v1.jsonl)");
                let _ = writeln!(out, "  --quantize-out <pasta>        Exporta pacote quantizado SMQ1 para <pasta> e avalia o pacote exportado");
                let _ = writeln!(out, "  --bits <fp16|bf16|int8|int4>  Precisão de pesos em memória ou para --quantize-out (padrão com --quantize-out: int8)");
                let _ = writeln!(out, "  --kv-cache <fp32|fp16|int8>   Precisão do KV Cache durante a geração autoregressiva");
                let _ = writeln!(out, "  --low-ram                     Ativa perfil Low-End 8 GB RAM (teto 2048 MB + KV Cache FP16 padrão)");
                let _ = writeln!(out, "  --max-ram-mb <N>              Impõe teto máximo estrito de RAM (pesos + KV Cache + prefill) em MiB");
                let _ = writeln!(out, "  --threads <N>                 Limita o número de threads CPU ativas (1..=64)");
                return 0;
            }
            "--dir" if i + 1 < args.len() => {
                ckpt_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--dataset" if i + 1 < args.len() => {
                dataset_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--quantize-out" if i + 1 < args.len() => {
                quantize_out = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--bits" if i + 1 < args.len() => {
                match crate::neural::WeightPrecision::parse(&args[i + 1]) {
                    Ok(p) => bits_opt = Some(p),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--kv-cache" if i + 1 < args.len() => {
                match crate::neural::KvCachePrecision::parse(&args[i + 1]) {
                    Ok(k) => kv_cache_opt = Some(k),
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 2;
                    }
                }
                i += 2;
            }
            "--low-ram" => {
                low_ram = true;
                i += 1;
            }
            "--max-ram-mb" if i + 1 < args.len() => {
                match args[i + 1].parse::<usize>() {
                    Ok(v) if v > 0 => max_ram_mb_opt = Some(v),
                    _ => {
                        let _ = writeln!(
                            err,
                            "Valor inválido para '--max-ram-mb': esperado inteiro > 0."
                        );
                        return 2;
                    }
                }
                i += 2;
            }
            "--threads" if i + 1 < args.len() => {
                match args[i + 1].parse::<usize>() {
                    Ok(v) if v > 0 => threads_opt = Some(v.min(64)),
                    _ => {
                        let _ = writeln!(
                            err,
                            "Valor inválido para '--threads': esperado inteiro > 0."
                        );
                        return 2;
                    }
                }
                i += 2;
            }
            other => {
                let _ = writeln!(err, "Argumento desconhecido em eval-neural: '{}'", other);
                return 2;
            }
        }
    }

    let mut budget = if low_ram {
        crate::infrastructure::RuntimeBudget::low_end_8gb()
    } else {
        crate::infrastructure::RuntimeBudget::auto_detect()
    };
    if let Some(mb) = max_ram_mb_opt {
        budget = budget.with_max_ram_mb(mb);
    }
    if let Some(t) = threads_opt {
        budget = budget.with_max_threads(t);
    }
    if let Some(kv) = kv_cache_opt {
        budget.default_kv_precision = kv;
    }

    let eval_target_dir = if let Some(ref qdir) = quantize_out {
        let prec = bits_opt.unwrap_or(crate::neural::WeightPrecision::Int8);
        match export_quantized_package(&ckpt_dir, qdir, prec) {
            Ok(st) => {
                let _ = writeln!(
                    err,
                    "[SMQ1 exportado | bits={} | disco {:.2} MB -> {:.2} MB ({:.2}x) | RAM pesos {:.2} MB -> {:.2} MB]",
                    st.precision.as_str(),
                    st.src_disk_mb,
                    st.dst_disk_mb,
                    st.compression_ratio,
                    st.src_ram_mb,
                    st.dst_ram_mb
                );
            }
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        }
        qdir.clone()
    } else {
        ckpt_dir
    };

    let mut engine =
        match crate::inference::NeuralEngine::load_from_dir_with_budget(&eval_target_dir, budget) {
            Ok(e) => e,
            Err(e) => {
                let _ = writeln!(err, "{}", e);
                return 1;
            }
        };
    if let Some(kv) = kv_cache_opt {
        engine = engine.with_kv_precision(kv);
    }

    if quantize_out.is_none() {
        if let Some(prec) = bits_opt {
            if !engine.model.is_quantized() {
                match crate::neural::quantize_model_in_memory(&engine.model, prec) {
                    Ok(qm) => engine.model = qm,
                    Err(e) => {
                        let _ = writeln!(err, "{}", e);
                        return 1;
                    }
                }
            }
        }
    }

    let val_texts = match load_seed_corpus(&dataset_path) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(err, "{}", e);
            return 1;
        }
    };
    let summary = crate::evaluation::evaluate_neural_engine(&engine, &val_texts);
    let mut json_val = match serde_json::to_value(&summary) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(err, "Falha ao serializar resumo de avaliação: {}", e);
            return 1;
        }
    };
    if let Some(obj) = json_val.as_object_mut() {
        obj.insert(
            "runtime_profile".to_string(),
            serde_json::Value::String(budget.profile.as_str().to_string()),
        );
        obj.insert(
            "active_cpu_threads".to_string(),
            serde_json::json!(crate::neural::ops::effective_worker_threads()),
        );
        obj.insert(
            "weight_precision".to_string(),
            serde_json::Value::String(engine.precision_label().to_string()),
        );
        obj.insert(
            "kv_cache_precision".to_string(),
            serde_json::Value::String(engine.kv_precision.as_str().to_string()),
        );
        obj.insert(
            "weight_ram_mb".to_string(),
            serde_json::json!(
                (engine.weight_bytes() as f64 / (1024.0 * 1024.0) * 1000.0).round() / 1000.0
            ),
        );
        obj.insert(
            "estimated_runtime_ram_mb".to_string(),
            serde_json::json!(
                (engine.estimated_runtime_bytes() as f64 / (1024.0 * 1024.0) * 1000.0).round()
                    / 1000.0
            ),
        );
    }
    let json = match serde_json::to_string_pretty(&json_val) {
        Ok(s) => s,
        Err(e) => {
            let _ = writeln!(err, "Falha ao formatar JSON de avaliação: {}", e);
            return 1;
        }
    };
    let _ = writeln!(out, "{}", json);
    0
}
