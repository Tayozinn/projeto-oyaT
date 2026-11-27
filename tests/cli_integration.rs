use std::process::ExitCode;

#[test]
fn integration_cli_help_and_exit_code_mapping() {
    let mut out = Vec::new();
    let mut err = Vec::new();
    let args = vec!["servermind-ai".to_string(), "--help".to_string()];

    let code = servermind_ai::dispatch_cli_args_with_io(&args, false, None, &mut out, &mut err);
    assert_eq!(code, Some(0));
    assert!(err.is_empty());
    let stdout_text = String::from_utf8_lossy(&out);
    assert!(stdout_text.contains("SERVERMIND AI"));
    assert_eq!(servermind_ai::exit_code_from_i32(0), ExitCode::SUCCESS);
    assert_eq!(servermind_ai::exit_code_from_i32(130), ExitCode::from(130));
}
