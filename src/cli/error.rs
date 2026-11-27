use std::fmt;
use std::io::Write;
use std::process::ExitCode;

/// Erro tipado de linha de comando com código de saída associado (`B11` — Fase 4).
///
/// Convenção de códigos de saída preservada em todo o binário:
/// - `0`: sucesso (`Ok(())`);
/// - `1`: erro operacional, de configuração, I/O ou falha global (`CliError::Runtime`);
/// - `2`: erro de sintaxe, flag inválida ou argumento fora de contrato (`CliError::Usage`);
/// - `130`: interrupção solicitada pelo usuário via `Ctrl+C` (`CliError::Cancelled`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    /// Uso inválido de argumentos/flags (código de saída `2`).
    Usage(String),
    /// Falha de execução, configuração ou conectividade (código de saída `1`).
    Runtime(String),
    /// Execução cancelada pelo operador via `Ctrl+C` / `cancel_flag` (código de saída `130`).
    Cancelled(Option<String>),
}

impl CliError {
    pub fn usage(msg: impl Into<String>) -> Self {
        Self::Usage(msg.into())
    }

    pub fn runtime(msg: impl Into<String>) -> Self {
        Self::Runtime(msg.into())
    }

    pub fn cancelled() -> Self {
        Self::Cancelled(None)
    }

    pub fn cancelled_with_message(msg: impl Into<String>) -> Self {
        Self::Cancelled(Some(msg.into()))
    }

    pub const fn exit_code_u8(&self) -> u8 {
        match self {
            Self::Usage(_) => 2,
            Self::Runtime(_) => 1,
            Self::Cancelled(_) => 130,
        }
    }

    pub const fn exit_code_i32(&self) -> i32 {
        self.exit_code_u8() as i32
    }

    pub fn to_exit_code(&self) -> ExitCode {
        ExitCode::from(self.exit_code_u8())
    }

    /// Emite a mensagem associada (se houver) em `err` e retorna o código de saída numérico.
    pub fn emit_and_code<W: Write + ?Sized>(&self, err: &mut W) -> i32 {
        match self {
            Self::Usage(msg) | Self::Runtime(msg) => {
                if !msg.is_empty() {
                    let _ = writeln!(err, "{}", msg);
                }
            }
            Self::Cancelled(Some(msg)) => {
                if !msg.is_empty() {
                    let _ = writeln!(err, "{}", msg);
                }
            }
            Self::Cancelled(None) => {}
        }
        self.exit_code_i32()
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(msg) | Self::Runtime(msg) => write!(f, "{}", msg),
            Self::Cancelled(Some(msg)) => write!(f, "{}", msg),
            Self::Cancelled(None) => write!(f, "Execução cancelada pelo usuário (Ctrl+C)."),
        }
    }
}

impl std::error::Error for CliError {}

pub type CliResult<T = ()> = Result<T, CliError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase4_b11_cli_error_exit_codes_and_emission() {
        let u = CliError::usage("flag inválida");
        assert_eq!(u.exit_code_u8(), 2);
        assert_eq!(u.to_exit_code(), ExitCode::from(2));

        let r = CliError::runtime("falha de banco");
        assert_eq!(r.exit_code_u8(), 1);
        assert_eq!(r.to_exit_code(), ExitCode::from(1));

        let c_silent = CliError::cancelled();
        assert_eq!(c_silent.exit_code_u8(), 130);
        assert_eq!(c_silent.to_exit_code(), ExitCode::from(130));
        let mut buf = Vec::new();
        assert_eq!(c_silent.emit_and_code(&mut buf), 130);
        assert!(buf.is_empty());

        let c_msg = CliError::cancelled_with_message("[CANCELADO]");
        assert_eq!(c_msg.emit_and_code(&mut buf), 130);
        assert_eq!(String::from_utf8_lossy(&buf).trim(), "[CANCELADO]");
    }
}
