use std::path::Path;

pub(crate) fn load_seed_corpus(dataset_path: &Path) -> Result<Vec<String>, String> {
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
        let norm = crate::neural::tokenizer::normalize_corpus_text(&dialogue);
        if !norm.is_empty() {
            examples.push(norm);
        }
    }
    if examples.is_empty() {
        return Err("Nenhum exemplo válido encontrado no dataset.".to_string());
    }
    Ok(examples)
}

pub(crate) fn check_no_existing_training_artifacts(out_dir: &Path) -> Result<(), String> {
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
