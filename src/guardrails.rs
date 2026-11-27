//! Guardrails determinísticos pós-geração compartilhados entre o chat interativo (`agent.rs`)
//! e o avaliador reproduzível (`eval_runner.rs`).
//!
//! Opera exclusivamente em texto local, sem rede, sem execução de comandos e sem
//! dependência de LLM. Nunca mascara falhas de segurança: quando um padrão arriscado
//! escapa do modelo, anexa avisos explícitos ou oculta credenciais literais antes de
//! entregar a resposta ao administrador.

use std::collections::BTreeSet;

const MUTATING_TOKENS: &[&str] = &[
    "systemctl restart",
    "systemctl start",
    "systemctl stop",
    "systemctl reload",
    "systemctl enable",
    "systemctl disable",
    "systemctl reboot",
    "apachectl restart",
    "apachectl -k restart",
    "a2enmod",
    "a2dismod",
    "usermod",
    "useradd",
    "userdel",
    "groupmod",
    "kill ",
    "pkill",
    "killall",
    "rm ",
    "rmdir ",
    "mv ",
    "chmod ",
    "chown ",
    "apt install",
    "apt-get install",
    "apt remove",
    "apt-get remove",
    "apt purge",
    "yum install",
    "yum remove",
    "dnf install",
    "dnf remove",
    "sed -i",
    "truncate ",
    "iptables -a",
    "iptables -i",
    "iptables -d",
    "ufw allow",
    "ufw deny",
    "ufw disable",
    "shutdown -r",
    "shutdown -h",
    "sudo reboot",
    "docker rm",
    "docker system prune",
    "git reset --hard",
];

const CAUTION_KEYWORDS: &[&str] = &[
    "revis",
    "backup",
    "copia de seguranca",
    "antes de executar",
    "valide antes",
    "confirme antes",
    "teste em homolog",
];

const LINE_NEGATION_KEYWORDS: &[&str] = &[
    "nao execute",
    "nao rode",
    "nao use",
    "nao utilize",
    "nao faca",
    "nao recomendo",
    "nao posso recomendar",
    "nunca ",
    "jamais ",
    "evite ",
    "proibid",
    "perigos",
    "suspeit",
    "malicios",
    "insegur",
    "sem executar",
];

const SECRET_PREFIXES: &[&str] = &[
    "--password=",
    "password=",
    "passwd=",
    "db_pass=",
    "db_password=",
    "secret=",
    "api_key=",
    "apikey=",
    "token=",
    "pass=",
];

/// Sanitiza um parágrafo ou bloco fechado durante o streaming antes de exibi-lo na tela (`A11`):
/// remove caracteres CJK/Cirílicos vazados, oculta credenciais literais do prompt e insere
/// o lembrete consultivo logo após blocos/linhas de comando mutável sem quebrar cercas ```` ``` ````.
pub fn sanitize_streaming_paragraph(user_prompt: &str, paragraph: &str) -> String {
    if paragraph.trim().is_empty() {
        return paragraph.to_string();
    }
    let cleaned = strip_non_latin_script_leaks(paragraph);
    let (redacted, _) = redact_prompt_secrets(user_prompt, &cleaned);
    insert_mutating_caution_reminders(&redacted)
}

/// Calcula avisos de rodapé globais que ainda não tenham sido exibidos nos parágrafos transmitidos.
pub fn trailing_advisory_notices(final_guarded: &str, already_printed: &str) -> String {
    let mut notices = Vec::new();
    for marker in [
        "> **Aviso de segurança:** Credenciais expostas foram ocultadas.",
        "> **Aviso de segurança:** Nunca utilize `debug=True`",
        "> **Aviso de segurança:** Não execute remoção recursiva",
    ] {
        if let Some(pos) = final_guarded.find(marker) {
            let line = final_guarded[pos..].lines().next().unwrap_or("").trim_end();
            if !line.is_empty() && !already_printed.contains(line) {
                notices.push(line.to_string());
            }
        }
    }
    notices.join("\n\n")
}

/// Aplica guardrails determinísticos sobre a resposta gerada pelo modelo assistente.
pub fn apply_advisory_guardrails(user_prompt: &str, response: &str) -> String {
    if response.trim().is_empty() {
        return response.to_string();
    }

    let mut text = strip_non_latin_script_leaks(response);
    if text.trim().is_empty() {
        return "Não foi possível gerar uma resposta válida em português. Envie novamente os detalhes técnicos do ambiente para análise consultiva."
            .to_string();
    }

    let (redacted_text, redacted_any) = redact_prompt_secrets(user_prompt, &text);
    text = redacted_text;

    text = insert_mutating_caution_reminders(&text);
    text = append_danger_warnings_if_needed(&text);

    if redacted_any
        && !contains_normalized(&text, "rotacion")
        && !contains_normalized(&text, "trocar a senha")
    {
        text.push_str(
            "\n\n> **Aviso de segurança:** Credenciais expostas foram ocultadas. Rotacione a senha/token comprometido e utilize arquivo de configuração restrito (ex.: permissões `600`) ou variáveis protegidas.",
        );
    }

    text
}

fn is_disallowed_script_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x4E00..=0x9FFF // CJK Unified Ideographs
            | 0x3400..=0x4DBF // CJK Extension A
            | 0x3040..=0x30FF // Hiragana + Katakana
            | 0xAC00..=0xD7AF // Hangul
            | 0x3000..=0x303F // CJK Symbols and Punctuation
            | 0xFF00..=0xFFEF // Halfwidth and Fullwidth Forms
            | 0x0400..=0x04FF // Cyrillic
    )
}

fn strip_non_latin_script_leaks(input: &str) -> String {
    if !input.chars().any(is_disallowed_script_char) {
        return input.to_string();
    }

    let mut out_lines: Vec<String> = Vec::new();
    for line in input.lines() {
        if !line.chars().any(is_disallowed_script_char) {
            out_lines.push(line.to_string());
            continue;
        }
        let filtered: String = line
            .chars()
            .filter(|c| !is_disallowed_script_char(*c))
            .collect();
        let cleaned = filtered
            .replace("()", "")
            .replace("( )", "")
            .replace("  ", " ");
        out_lines.push(cleaned.trim_end().to_string());
    }
    let mut joined = out_lines.join("\n");
    if input.ends_with('\n') {
        joined.push('\n');
    }
    joined
}

fn extract_secrets_from_prompt(user_prompt: &str) -> Vec<String> {
    let mut secrets = BTreeSet::new();
    let lower = user_prompt.to_ascii_lowercase();

    for prefix in SECRET_PREFIXES {
        let mut search_from = 0;
        while let Some(rel) = lower[search_from..].find(prefix) {
            let start = search_from + rel + prefix.len();
            if let Some(val) = take_secret_token(&user_prompt[start..]) {
                secrets.insert(val);
            }
            search_from = start;
        }
    }

    // Detecta `-p<senha>` colado (ex.: `mysql -u app -pMinhaSenha123`), ignorando `-p 8080` ou `-p8080`.
    for raw_tok in user_prompt.split_whitespace() {
        let tok = raw_tok.trim_matches(|c: char| ",;\"'`()[]{}".contains(c));
        if let Some(rest) = tok.strip_prefix("-p") {
            if rest.len() >= 4
                && !rest.chars().all(|c| c.is_ascii_digit())
                && !is_placeholder_secret(rest)
            {
                secrets.insert(rest.to_string());
            }
        }
    }

    // Detecta `Bearer <token>`
    let mut search_from = 0;
    while let Some(rel) = lower[search_from..].find("bearer ") {
        let start = search_from + rel + "bearer ".len();
        if let Some(val) = take_secret_token(&user_prompt[start..]) {
            secrets.insert(val);
        }
        search_from = start;
    }

    secrets.into_iter().collect()
}

fn take_secret_token(slice: &str) -> Option<String> {
    let trimmed = slice.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    let quote = trimmed.chars().next().filter(|c| *c == '\'' || *c == '"');
    let raw = if let Some(q) = quote {
        let rest = &trimmed[1..];
        rest.split(q).next().unwrap_or("")
    } else {
        trimmed
            .split(|c: char| c.is_whitespace() || ",;)`]}>".contains(c))
            .next()
            .unwrap_or("")
    };
    let clean = raw.trim();
    if clean.len() >= 4 && !is_placeholder_secret(clean) {
        Some(clean.to_string())
    } else {
        None
    }
}

fn is_placeholder_secret(val: &str) -> bool {
    let low = val.to_ascii_lowercase();
    low.starts_with('<')
        || low.starts_with('$')
        || low.contains("***")
        || matches!(
            low.as_str(),
            "senha" | "password" | "token" | "secret" | "true" | "false" | "none" | "null"
        )
}

fn redact_prompt_secrets(user_prompt: &str, response: &str) -> (String, bool) {
    let secrets = extract_secrets_from_prompt(user_prompt);
    if secrets.is_empty() {
        return (response.to_string(), false);
    }
    let mut out = response.to_string();
    let mut redacted = false;
    for secret in secrets {
        if out.contains(&secret) {
            out = out.replace(&secret, "<SENHA_OCULTADA>");
            redacted = true;
        }
    }
    (out, redacted)
}

fn insert_mutating_caution_reminders(response: &str) -> String {
    let lines: Vec<&str> = response.lines().collect();
    if lines.is_empty() {
        return response.to_string();
    }

    // Mapeia cada linha para (start_byte, end_byte, in_code_block, code_block_end_line)
    let mut in_block = false;
    let mut line_in_block = vec![false; lines.len()];
    let mut block_end_line = vec![None; lines.len()];
    let mut block_lines: Vec<usize> = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            if !in_block {
                in_block = true;
                block_lines.clear();
            } else {
                for &b_idx in &block_lines {
                    block_end_line[b_idx] = Some(idx);
                }
                block_lines.clear();
                in_block = false;
            }
            continue;
        }
        if in_block {
            line_in_block[idx] = true;
            block_lines.push(idx);
        }
    }

    let norm_lines: Vec<String> = lines.iter().map(|l| normalize_ascii_lower(l)).collect();
    let norm_full = norm_lines.join("\n");
    let mut line_offsets: Vec<(usize, usize)> = Vec::with_capacity(norm_lines.len());
    let mut cursor = 0;
    for norm_line in &norm_lines {
        let start = cursor;
        let end = start + norm_line.len();
        line_offsets.push((start, end));
        cursor = end + 1;
    }

    let mut insert_after_line: BTreeSet<usize> = BTreeSet::new();

    for (idx, norm_line) in norm_lines.iter().enumerate() {
        if !line_in_block[idx] && LINE_NEGATION_KEYWORDS.iter().any(|k| norm_line.contains(k)) {
            continue;
        }
        let has_mutating = MUTATING_TOKENS
            .iter()
            .any(|tok| contains_command_token(norm_line, tok));
        if !has_mutating {
            continue;
        }

        // Verifica janela aproximada de +-160 caracteres ao redor da linha no texto normalizado.
        let (l_start, l_end) = line_offsets[idx];
        let win_start = floor_char_boundary(&norm_full, l_start.saturating_sub(160));
        let win_end = ceil_char_boundary(&norm_full, (l_end + 160).min(norm_full.len()));
        let window = &norm_full[win_start..win_end];
        if CAUTION_KEYWORDS.iter().any(|kw| window.contains(kw)) {
            continue;
        }

        let target_line = block_end_line[idx].unwrap_or(idx);
        insert_after_line.insert(target_line);
    }

    if insert_after_line.is_empty() {
        return response.to_string();
    }

    let mut rebuilt = Vec::with_capacity(lines.len() + insert_after_line.len());
    for (idx, line) in lines.iter().enumerate() {
        rebuilt.push((*line).to_string());
        if insert_after_line.contains(&idx) {
            rebuilt.push(
                "> *Revisão manual e backup recomendados antes de executar comandos que alteram o estado do sistema.*"
                    .to_string(),
            );
        }
    }

    let mut out = rebuilt.join("\n");
    if response.ends_with('\n') {
        out.push('\n');
    }
    out
}

fn append_danger_warnings_if_needed(response: &str) -> String {
    let mut in_block = false;
    let mut unnegated_destructive = false;
    let mut unnegated_debug_true = false;

    for raw_line in response.lines() {
        let trimmed = raw_line.trim_start();
        if trimmed.starts_with("```") {
            in_block = !in_block;
            continue;
        }

        let norm = normalize_ascii_lower(raw_line);
        let is_negated = !in_block && LINE_NEGATION_KEYWORDS.iter().any(|k| norm.contains(k));

        let is_destructive = norm.contains("rm -rf /")
            || norm.contains("rm -fr /")
            || norm.contains("rm -rf *")
            || norm.contains("rm -fr *")
            || (norm.contains("find ")
                && (norm.contains(" -delete") || norm.contains(" -exec rm")))
            || norm.contains("chmod -r 777")
            || norm.contains("chmod 777 /")
            || ((norm.contains("curl ") || norm.contains("wget "))
                && (norm.contains("| sh")
                    || norm.contains("| bash")
                    || norm.contains("| sudo sh")
                    || norm.contains("| sudo bash")));

        if is_destructive && !is_negated {
            unnegated_destructive = true;
        }
        if norm.contains("debug=true") && !is_negated {
            unnegated_debug_true = true;
        }
    }

    let mut out = response.to_string();
    if unnegated_debug_true && !contains_normalized(&out, "nunca utilize `debug=true`") {
        out.push_str(
            "\n\n> **Aviso de segurança:** Nunca utilize `debug=True` em produção (permite execução remota pelo debugger); mantenha `debug=False`.",
        );
    }
    if unnegated_destructive && !contains_normalized(&out, "nao execute remocao recursiva") {
        out.push_str(
            "\n\n> **Aviso de segurança:** Não execute remoção recursiva (`rm -rf`, `find ... -delete/-exec rm`), permissões amplas (`777`) ou scripts remotos via pipe sem inspeção prévia. Priorize comandos somente leitura e rotação controlada (`logrotate`) com revisão manual e backup.",
        );
    }
    out
}

fn contains_command_token(line: &str, token: &str) -> bool {
    let mut start = 0;
    while let Some(rel) = line[start..].find(token) {
        let abs = start + rel;
        let boundary_ok = abs == 0
            || line[..abs]
                .chars()
                .next_back()
                .map(|c| !c.is_alphanumeric() && c != '_' && c != '-')
                .unwrap_or(true);
        if boundary_ok {
            return true;
        }
        start = abs + token.len().max(1);
    }
    false
}

fn contains_normalized(haystack: &str, needle_ascii_lower: &str) -> bool {
    normalize_ascii_lower(haystack).contains(needle_ascii_lower)
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

fn normalize_ascii_lower(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        let mapped = match ch {
            'á' | 'à' | 'â' | 'ã' | 'ä' | 'Á' | 'À' | 'Â' | 'Ã' | 'Ä' => 'a',
            'é' | 'è' | 'ê' | 'ë' | 'É' | 'È' | 'Ê' | 'Ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' | 'Í' | 'Ì' | 'Î' | 'Ï' => 'i',
            'ó' | 'ò' | 'ô' | 'õ' | 'ö' | 'Ó' | 'Ò' | 'Ô' | 'Õ' | 'Ö' => 'o',
            'ú' | 'ù' | 'û' | 'ü' | 'Ú' | 'Ù' | 'Û' | 'Ü' => 'u',
            'ç' | 'Ç' => 'c',
            other => other.to_ascii_lowercase(),
        };
        out.push(mapped);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::apply_advisory_guardrails;

    #[test]
    fn strips_cjk_leaks_cleanly() {
        let raw = "Verifique os logs do serviço (检查日志) antes de reiniciar.";
        let out = apply_advisory_guardrails("Erro no nginx", raw);
        assert!(!out.contains('检'));
        assert!(out.contains("Verifique os logs do serviço"));
    }

    #[test]
    fn redacts_inline_password_from_prompt() {
        let prompt = "Rode mysql -u app -pSuperSecret99! -e 'SHOW PROCESSLIST;'";
        let raw =
            "Você pode usar `mysql -u app -pSuperSecret99! -e 'SHOW STATUS;'` para inspecionar.";
        let out = apply_advisory_guardrails(prompt, raw);
        assert!(!out.contains("SuperSecret99!"));
        assert!(out.contains("<SENHA_OCULTADA>"));
        assert!(out.contains("Rotacione"));
    }

    #[test]
    fn inserts_caution_near_mutating_command_when_missing() {
        let raw = "Para reiniciar o serviço:\n```bash\nsudo systemctl restart nginx\n```\n";
        let out = apply_advisory_guardrails("nginx caiu", raw);
        assert!(out.contains("Revisão manual e backup recomendados"));
    }

    #[test]
    fn does_not_duplicate_caution_when_already_present() {
        let raw =
            "Antes de executar, faça backup e revise:\n```bash\nsudo systemctl restart nginx\n```";
        let out = apply_advisory_guardrails("nginx caiu", raw);
        assert!(!out.contains("Revisão manual e backup recomendados"));
    }

    #[test]
    fn warns_on_unnegated_destructive_pattern_but_not_on_prohibition() {
        let bad = "Execute:\n```bash\nfind /var/log -mtime +7 -delete\n```";
        let out_bad = apply_advisory_guardrails("disco cheio", bad);
        assert!(out_bad.contains("Não execute remoção recursiva"));

        let good = "Nunca execute `find /var/log -mtime +7 -delete` diretamente sem inspeção.";
        let out_good = apply_advisory_guardrails("disco cheio", good);
        assert!(!out_good.contains("Não execute remoção recursiva"));
    }

    #[test]
    fn does_not_match_rm_inside_long_term_support() {
        let raw = "Recomendo o uso de uma versão LTS (Long Term Support) para maior estabilidade.";
        let out = apply_advisory_guardrails("montar servidor", raw);
        assert!(!out.contains("Revisão manual e backup recomendados"));
    }
}
