pub(crate) mod eval;
pub(crate) mod hw;
pub(crate) mod quantize;
pub(crate) mod repl;
pub(crate) mod train;

pub(crate) use eval::run_eval_cli_with_options;
pub(crate) use hw::run_hw_cli;
pub(crate) use repl::run_terminal_loop;
#[cfg(test)]
pub(crate) use repl::{
    handle_exp_command_with_io, is_repl_command_input, ParagraphStreamBuffer, SpinnerHandle,
};
pub(crate) use train::{run_eval_neural_cli, run_train_neural_cli_with_cancel};

use std::io::Write;

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
