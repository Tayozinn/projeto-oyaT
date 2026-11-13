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
