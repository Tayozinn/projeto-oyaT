use servermind_ai::CliError;
use std::process::ExitCode;

#[test]
fn integration_cli_help_and_exit_code_mapping() {
    let mut out = Vec::new();
    let mut err = Vec::new();
    let args = vec!["servermind-ai".to_string(), "--help".to_string()];

    let handled =
        servermind_ai::dispatch_cli_result_with_io(&args, false, None, &mut out, &mut err)
            .expect("help deve suceder");
    assert!(handled);
    assert!(err.is_empty());
    let stdout_text = String::from_utf8_lossy(&out);
    assert!(stdout_text.contains("SERVERMIND AI"));

    let bad_args = vec!["servermind-ai".to_string(), "--invalida".to_string()];
    let cli_err =
        servermind_ai::dispatch_cli_result_with_io(&bad_args, false, None, &mut out, &mut err)
            .expect_err("opção inválida deve retornar CliError::Usage");
    assert!(matches!(cli_err, CliError::Usage(_)));
    assert_eq!(cli_err.to_exit_code(), ExitCode::from(2));
    assert_eq!(servermind_ai::exit_code_from_i32(0), ExitCode::SUCCESS);
    assert_eq!(CliError::cancelled().to_exit_code(), ExitCode::from(130));
}
