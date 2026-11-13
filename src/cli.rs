use crate::agent::AgentSession;
use crate::config::{IMMUTABLE_SAFETY_INSTRUCTIONS, MAX_INSTRUCTIONS_CHARS};
use crate::models::{OllamaHealth, Role};
use crate::ollama::OllamaClient;
use std::io::{self, Write};

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

                if trimmed.starts_with(':') || trimmed.starts_with('/') {
                    if handle_command(trimmed, session) {
                        break;
                    }
                    continue;
                }

                println!("\n[SERVERMIND AI pensando...]");
                match session.send_user_message(trimmed) {
                    Ok(reply) => {
                        println!("\nSERVERMIND AI >\n{}", reply);
                    }
                    Err(err) => {
                        eprintln!("\n[ERRO DE COMUNICAÇÃO / INFERÊNCIA]\n{}", err);
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

fn print_banner(session: &AgentSession) {
    let instr_status = if session.is_default_instructions() {
        "Padrão"
    } else {
        "Personalizadas"
    };
    println!("======================================================================");
    println!("                   SERVERMIND AI - Fase 2 (Local)                     ");
    println!("       Assistente Conversacional Local para Administradores           ");
    println!("======================================================================");
    println!("  Origem da config       : {} ({})", session.config().config_source, session.config().config_file_path);
    println!("  URL do Ollama (local)  : {}", session.config().ollama_url);
    println!("  Modelo ativo           : {}", session.config().model);
    println!("  Instruções do agente   : {} (+ travas de segurança ativas)", instr_status);
    println!("  Banco SQLite local     : {}", session.config().database_path);
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

/// Processa comandos iniciados por `:` ou `/`. Retorna `true` se o programa deve encerrar.
fn handle_command(cmd_line: &str, session: &mut AgentSession) -> bool {
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
                eprintln!("Uso correto: :history (para selecionar da lista) ou :load <id> (ex: :load 1)");
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
                eprintln!("Uso correto: :remember <informação que deseja salvar na memória explícita>");
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
                    Ok(id) => match session.delete_memory(id) {
                        Ok(true) => println!("Memória explícita #{} removida com sucesso.", id),
                        Ok(false) => eprintln!("Memória #{} não encontrada. Use :memories para listar.", id),
                        Err(err) => eprintln!("Erro ao remover memória: {}", err),
                    },
                    Err(_) => eprintln!("ID inválido '{}'. Informe o número inteiro da memória.", arg),
                }
            }
        }
        "context" | "contexto" => {
            print_current_conversation_messages(session);
        }
        "status" => {
            print_health_report(session);
        }
        _ => {
            eprintln!(
                "Comando desconhecido: '{}'. Digite :help (ou /ajuda) para ver os comandos disponíveis.",
                cmd_line
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
    println!("  :new                     - Inicia uma nova conversa separada sem apagar as anteriores");
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
    println!("  :help                    - Exibe esta mensagem de ajuda");
    println!("  :quit                    - Encerra o programa com segurança");
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

    println!("\nModelo configurado atualmente: '{}'", session.config().model);
    println!("Consultando modelos instalados no Ollama local ({})...", session.config().ollama_url);

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

    let active_resolved = OllamaClient::resolve_installed_model(&session.config().model, &installed);

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

    let mut choice = String::new();
    if io::stdin().read_line(&mut choice).is_ok() {
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

    let mut selection = String::new();
    if io::stdin().read_line(&mut selection).is_ok() {
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
}

fn reopen_conversation_by_str(raw_id: &str, session: &mut AgentSession) {
    let cleaned = raw_id.trim().trim_start_matches('#');
    match cleaned.parse::<i64>() {
        Ok(id) => match session.load_conversation(id) {
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
        Err(_) => {
            eprintln!(
                "ID de conversa inválido: '{}'. Informe um número inteiro (ex: `:history 1`).",
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
        println!("\n[2] Diretrizes de segurança imutáveis (sempre aplicadas):");
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
                println!("As diretrizes de segurança (modo apenas conversacional) permanecem ativas.");
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

        let mut new_text = String::new();
        if io::stdin().read_line(&mut new_text).is_ok() {
            let input_clean = new_text.trim();
            if input_clean.is_empty() {
                println!(
                    "Edição cancelada. As instruções atuais foram mantidas. (Use `:instructions reset` se desejar voltar ao padrão)."
                );
                return;
            }
            apply_new_instructions(input_clean, session);
        }
        return;
    }

    // Caso o usuário tenha escrito `:instructions set <texto>`
    let text_to_set = if let Some(rest) = trimmed.strip_prefix("set ") {
        rest.trim()
    } else if let Some(rest) = trimmed.strip_prefix("definir ") {
        rest.trim()
    } else {
        trimmed
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
                println!("Dica: Use `:remember <texto>` para adicionar ou `:forget <id>` para remover.");
            }
        }
        Err(err) => eprintln!("Erro ao listar memórias: {}", err),
    }
}
