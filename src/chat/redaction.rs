//! Redação determinística de segredos de saída antes do envio para backends remotos (`C3`).
//!
//! Logs de servidores, comandos históricos e arquivos de configuração frequentemente contêm
//! senhas em URLs de banco, tokens `Bearer`, chaves de API (`sk-ant-`, `ghp_`, `AKIA...`,
//! `AIza...`, `gsk_...`), JWTs e chaves privadas PEM. Este módulo remove esses segredos
//! **antes** de serializar qualquer payload HTTP destinado a provedores externos.

use crate::chat::models::Message;

const PEM_BEGIN_MARKERS: &[&str] = &[
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "-----BEGIN EC PRIVATE KEY-----",
    "-----BEGIN DSA PRIVATE KEY-----",
    "-----BEGIN PGP PRIVATE KEY BLOCK-----",
];

const KNOWN_TOKEN_PREFIXES: &[(&str, usize)] = &[
    ("sk-ant-", 14),
    ("sk-proj-", 16),
    ("sk_live_", 16),
    ("rk_live_", 16),
    ("github_pat_", 20),
    ("ghp_", 16),
    ("gho_", 16),
    ("ghs_", 16),
    ("ghr_", 16),
    ("glpat-", 14),
    ("xoxb-", 14),
    ("xoxp-", 14),
    ("xoxa-", 14),
    ("gsk_", 16),
    ("AIza", 24),
    ("AKIA", 16),
];

const KV_SECRET_KEYS: &[&str] = &[
    "--password",
    "--token",
    "--api-key",
    "--apikey",
    "--secret",
    "db_password",
    "db_pass",
    "database_password",
    "client_secret",
    "access_token",
    "refresh_token",
    "auth_token",
    "anthropic_api_key",
    "openai_api_key",
    "gemini_api_key",
    "groq_api_key",
    "openrouter_api_key",
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "token",
    "pwd",
];

const URI_SCHEMES_WITH_AUTH: &[&str] = &[
    "postgres://",
    "postgresql://",
    "mysql://",
    "mariadb://",
    "mongodb://",
    "mongodb+srv://",
    "redis://",
    "rediss://",
    "amqp://",
    "amqps://",
];

/// Redige segredos de uma lista de mensagens antes do envio a um backend remoto,
/// retornando as mensagens sanitizadas e o total de substituições efetuadas.
pub fn redact_messages_for_remote(messages: &[Message]) -> (Vec<Message>, usize) {
    let mut total = 0usize;
    let sanitized = messages
        .iter()
        .map(|m| {
            let (clean, count) = redact_text_for_remote(&m.content);
            total += count;
            Message::new(m.role, clean)
        })
        .collect();
    (sanitized, total)
}

/// Redige chaves privadas PEM, tokens de API, cabeçalhos `Bearer`, JWTs, senhas em URIs
/// e pares `chave=segredo` em texto livre ou logs de servidor (`C3`).
pub fn redact_text_for_remote(input: &str) -> (String, usize) {
    if input.is_empty() {
        return (String::new(), 0);
    }

    let mut count = 0usize;
    let mut text = redact_pem_blocks(input, &mut count);
    text = redact_uri_credentials(&text, &mut count);
    text = redact_bearer_tokens(&text, &mut count);
    text = redact_kv_assignments(&text, &mut count);
    text = redact_inline_cli_p_flags(&text, &mut count);
    text = redact_known_api_tokens_and_jwts(&text, &mut count);

    (text, count)
}

fn redact_pem_blocks(input: &str, count: &mut usize) -> String {
    let mut current = input.to_string();
    for begin in PEM_BEGIN_MARKERS {
        while let Some(start) = current.find(begin) {
            let rest = &current[start..];
            let end_offset = if let Some(end_tag_pos) = rest.find("-----END ") {
                let after_end = &rest[end_tag_pos + "-----END ".len()..];
                if let Some(close_dashes) = after_end.find("-----") {
                    start + end_tag_pos + "-----END ".len() + close_dashes + "-----".len()
                } else {
                    current.len()
                }
            } else {
                current.len()
            };
            current.replace_range(start..end_offset, "[CHAVE_PRIVADA_REDIGIDA]");
            *count += 1;
        }
    }
    current
}

fn redact_uri_credentials(input: &str, count: &mut usize) -> String {
    let mut out = String::with_capacity(input.len());
    let mut idx = 0usize;

    while idx < input.len() {
        let lower_rem = input[idx..].to_ascii_lowercase();
        let mut next_match: Option<(usize, usize)> = None;
        for scheme in URI_SCHEMES_WITH_AUTH {
            if let Some(rel) = lower_rem.find(scheme) {
                let abs = idx + rel;
                if next_match.is_none_or(|(best_abs, _)| abs < best_abs) {
                    next_match = Some((abs, scheme.len()));
                }
            }
        }

        let Some((scheme_pos, scheme_len)) = next_match else {
            out.push_str(&input[idx..]);
            break;
        };

        out.push_str(&input[idx..scheme_pos + scheme_len]);
        let after_scheme = scheme_pos + scheme_len;
        let uri_end_rel = input[after_scheme..]
            .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '`' | ')' | ']' | '>'))
            .unwrap_or(input.len() - after_scheme);
        let uri_body = &input[after_scheme..after_scheme + uri_end_rel];
        let authority_end = uri_body.find('/').unwrap_or(uri_body.len());
        let authority = &uri_body[..authority_end];

        if let Some(at_pos) = authority.rfind('@') {
            let userinfo = &authority[..at_pos];
            if let Some(colon_pos) = userinfo.find(':') {
                let user = &userinfo[..colon_pos];
                let pass = &userinfo[colon_pos + 1..];
                if !pass.is_empty() && !is_placeholder_or_redacted(pass) {
                    out.push_str(user);
                    out.push_str(":[SENHA_REDIGIDA]@");
                    out.push_str(&uri_body[at_pos + 1..]);
                    *count += 1;
                    idx = after_scheme + uri_end_rel;
                    continue;
                }
            }
        }

        out.push_str(uri_body);
        idx = after_scheme + uri_end_rel;
    }

    out
}

fn redact_bearer_tokens(input: &str, count: &mut usize) -> String {
    let mut out = String::with_capacity(input.len());
    let mut cursor = 0usize;
    let lower = input.to_ascii_lowercase();

    while let Some(rel) = lower[cursor..].find("bearer ") {
        let match_start = cursor + rel;
        let prev_ok = match_start == 0
            || !input[..match_start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric());
        let after_kw = match_start + "bearer ".len();
        out.push_str(&input[cursor..after_kw]);

        if !prev_ok {
            cursor = after_kw;
            continue;
        }

        let ws_len: usize = input[after_kw..]
            .chars()
            .take_while(|c| *c == ' ' || *c == '\t')
            .map(|c| c.len_utf8())
            .sum();
        out.push_str(&input[after_kw..after_kw + ws_len]);
        let tok_start = after_kw + ws_len;
        let tok_len = secret_value_span(&input[tok_start..]);
        let candidate = &input[tok_start..tok_start + tok_len];

        if candidate.len() >= 6 && !is_placeholder_or_redacted(candidate) {
            out.push_str("[TOKEN_REDIGIDO]");
            *count += 1;
            cursor = tok_start + tok_len;
        } else {
            cursor = tok_start;
        }
    }

    out.push_str(&input[cursor..]);
    out
}

fn redact_kv_assignments(input: &str, count: &mut usize) -> String {
    let mut current = input.to_string();

    for key in KV_SECRET_KEYS {
        let mut out = String::with_capacity(current.len());
        let lower = current.to_ascii_lowercase();
        let mut cursor = 0usize;

        while let Some(rel) = lower[cursor..].find(key) {
            let key_start = cursor + rel;
            let key_end = key_start + key.len();

            // Garante fronteira de palavra antes da chave (evita casar dentro de identificadores maiores)
            let left_boundary_ok = key_start == 0
                || !current[..key_start]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');

            if !left_boundary_ok {
                out.push_str(&current[cursor..key_end]);
                cursor = key_end;
                continue;
            }

            // Aceita separador `=` ou `:` com espaços opcionais (ex.: `password = "abc"`, `api_key: xyz`)
            let after_key = &current[key_end..];
            let ws1: usize = after_key
                .chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .map(|c| c.len_utf8())
                .sum();
            let sep_slice = &after_key[ws1..];
            let Some(sep_ch) = sep_slice.chars().next().filter(|c| *c == '=' || *c == ':') else {
                out.push_str(&current[cursor..key_end]);
                cursor = key_end;
                continue;
            };
            // Evita `::` de Rust (ex.: `token::...`)
            if sep_ch == ':' && sep_slice.starts_with("::") {
                out.push_str(&current[cursor..key_end]);
                cursor = key_end;
                continue;
            }

            let after_sep_idx = key_end + ws1 + sep_ch.len_utf8();
            let ws2: usize = current[after_sep_idx..]
                .chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .map(|c| c.len_utf8())
                .sum();
            let val_start = after_sep_idx + ws2;
            out.push_str(&current[cursor..val_start]);

            let val_slice = &current[val_start..];
            if let Some(quote) = val_slice.chars().next().filter(|c| *c == '"' || *c == '\'') {
                let inner = &val_slice[quote.len_utf8()..];
                if let Some(end_q) = inner.find(quote) {
                    let secret_val = &inner[..end_q];
                    if secret_val.len() >= 4 && !is_placeholder_or_redacted(secret_val) {
                        out.push(quote);
                        out.push_str("[SEGREDO_REDIGIDO]");
                        out.push(quote);
                        *count += 1;
                    } else {
                        out.push(quote);
                        out.push_str(secret_val);
                        out.push(quote);
                    }
                    cursor = val_start + quote.len_utf8() + end_q + quote.len_utf8();
                    continue;
                }
            }

            let val_len = secret_value_span(val_slice);
            let secret_val = &val_slice[..val_len];
            if secret_val.len() >= 4 && !is_placeholder_or_redacted(secret_val) {
                out.push_str("[SEGREDO_REDIGIDO]");
                *count += 1;
                cursor = val_start + val_len;
            } else {
                cursor = val_start;
            }
        }

        out.push_str(&current[cursor..]);
        current = out;
    }

    current
}

fn redact_inline_cli_p_flags(input: &str, count: &mut usize) -> String {
    let mut out = String::with_capacity(input.len());
    let mut cursor = 0usize;

    while let Some(rel) = input[cursor..].find("-p") {
        let pos = cursor + rel;
        let preceded_by_space = pos == 0
            || input[..pos]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_whitespace() || c == '`');
        let after_p = pos + 2;
        out.push_str(&input[cursor..after_p]);

        if !preceded_by_space || after_p >= input.len() {
            cursor = after_p;
            continue;
        }

        let span = secret_value_span(&input[after_p..]);
        let rest = &input[after_p..after_p + span];
        // Ignora flags comuns como `-p 2222`, `-p8080` (só dígitos) ou nomes curtos
        if rest.len() >= 6
            && !rest.chars().all(|c| c.is_ascii_digit())
            && !rest.starts_with('-')
            && !is_placeholder_or_redacted(rest)
        {
            out.push_str("[SENHA_REDIGIDA]");
            *count += 1;
            cursor = after_p + span;
        } else {
            cursor = after_p;
        }
    }

    out.push_str(&input[cursor..]);
    out
}

fn redact_known_api_tokens_and_jwts(input: &str, count: &mut usize) -> String {
    let mut current = input.to_string();

    for &(prefix, min_len) in KNOWN_TOKEN_PREFIXES {
        let mut out = String::with_capacity(current.len());
        let mut cursor = 0usize;
        while let Some(rel) = current[cursor..].find(prefix) {
            let start = cursor + rel;
            let left_ok = start == 0
                || !current[..start]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
            if !left_ok {
                out.push_str(&current[cursor..start + prefix.len()]);
                cursor = start + prefix.len();
                continue;
            }
            let tok_len: usize = current[start..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                .map(|c| c.len_utf8())
                .sum();
            out.push_str(&current[cursor..start]);
            if tok_len >= min_len {
                out.push_str("[TOKEN_REDIGIDO]");
                *count += 1;
                cursor = start + tok_len;
            } else {
                out.push_str(&current[start..start + prefix.len()]);
                cursor = start + prefix.len();
            }
        }
        out.push_str(&current[cursor..]);
        current = out;
    }

    // OpenAI `sk-<20+ alnum>` genérico e JWTs `eyJ...`
    let mut out = String::with_capacity(current.len());
    let mut cursor = 0usize;
    while cursor < current.len() {
        let rem = &current[cursor..];
        let next_sk = rem.find("sk-");
        let next_jwt = rem.find("eyJ");
        let next = match (next_sk, next_jwt) {
            (Some(a), Some(b)) => Some((a.min(b), a <= b)),
            (Some(a), None) => Some((a, true)),
            (None, Some(b)) => Some((b, false)),
            (None, None) => None,
        };
        let Some((rel, is_sk)) = next else {
            out.push_str(rem);
            break;
        };
        let start = cursor + rel;
        let left_ok = start == 0
            || !current[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
        out.push_str(&current[cursor..start]);
        if !left_ok {
            out.push_str(&current[start..start + 3]);
            cursor = start + 3;
            continue;
        }

        if is_sk {
            let tok_len: usize = current[start..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                .map(|c| c.len_utf8())
                .sum();
            if tok_len >= 24 {
                out.push_str("[TOKEN_REDIGIDO]");
                *count += 1;
                cursor = start + tok_len;
            } else {
                out.push_str("sk-");
                cursor = start + 3;
            }
        } else {
            let jwt_len: usize = current[start..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
                .map(|c| c.len_utf8())
                .sum();
            let candidate = current[start..start + jwt_len].trim_end_matches('.');
            let parts: Vec<&str> = candidate.split('.').collect();
            if parts.len() == 3
                && parts[0].starts_with("eyJ")
                && parts[1].starts_with("eyJ")
                && parts.iter().all(|p| p.len() >= 10)
            {
                out.push_str("[JWT_REDIGIDO]");
                *count += 1;
                cursor = start + candidate.len();
            } else {
                out.push_str("eyJ");
                cursor = start + 3;
            }
        }
    }

    out
}

fn secret_value_span(slice: &str) -> usize {
    slice
        .find(|c: char| {
            c.is_whitespace() || matches!(c, ',' | ';' | '"' | '\'' | '`' | ')' | ']' | '}' | '>')
        })
        .unwrap_or(slice.len())
}

fn is_placeholder_or_redacted(val: &str) -> bool {
    let trimmed = val.trim();
    let low = trimmed.to_ascii_lowercase();
    low.is_empty()
        || low.starts_with('<')
        || low.starts_with('[')
        || low.starts_with('$')
        || low.starts_with('{')
        || low.contains("***")
        || low.contains("redigid")
        || low.contains("ocultad")
        || matches!(
            low.as_str(),
            "senha"
                | "password"
                | "sua_senha"
                | "your_password"
                | "token"
                | "secret"
                | "true"
                | "false"
                | "none"
                | "null"
                | "changeme"
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::models::Role;

    #[test]
    fn test_phase3_c3_redacts_tokens_passwords_uris_jwts_and_pem_keys() {
        let log_sample = "\
Erro em produção: postgres://admin:SuperSegredoPg99@10.0.0.12:5432/appdb\n\
Header capturado: Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlRlc3RlIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c\n\
ANTHROPIC_API_KEY=sk-ant-api03-SegredoRealDaAnthropic1234567890\n\
Token GitHub: ghp_1234567890abcdefGHIJKLMNOPQRSTUV\n\
Comando usado: mysql -u root -pSenhaBancoProd2026 -h 127.0.0.1\n\
-----BEGIN OPENSSH PRIVATE KEY-----\n\
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ==\n\
-----END OPENSSH PRIVATE KEY-----\n\
Placeholders seguros: password=<senha> e token=$API_TOKEN e -p 2222.";

        let msgs = vec![Message::new(Role::User, log_sample)];
        let (redacted_msgs, count) = redact_messages_for_remote(&msgs);
        let out = &redacted_msgs[0].content;

        assert!(
            count >= 6,
            "esperava >= 6 redações, obtido {}: {}",
            count,
            out
        );
        assert!(
            !out.contains("SuperSegredoPg99"),
            "vazou senha URI: {}",
            out
        );
        assert!(out.contains("postgres://admin:[SENHA_REDIGIDA]@10.0.0.12:5432/appdb"));
        assert!(
            !out.contains("SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"),
            "vazou JWT: {}",
            out
        );
        assert!(
            !out.contains("SegredoRealDaAnthropic"),
            "vazou chave Anthropic: {}",
            out
        );
        assert!(
            !out.contains("ghp_1234567890abcdef"),
            "vazou PAT GitHub: {}",
            out
        );
        assert!(
            !out.contains("SenhaBancoProd2026"),
            "vazou senha CLI -p: {}",
            out
        );
        assert!(
            !out.contains("b3BlbnNzaC1rZXktdjE"),
            "vazou bloco PEM: {}",
            out
        );
        assert!(out.contains("[CHAVE_PRIVADA_REDIGIDA]"));

        // Placeholders didáticos e portas numéricas (`-p 2222`) continuam intactos!
        assert!(out.contains("password=<senha>"));
        assert!(out.contains("token=$API_TOKEN"));
        assert!(out.contains("-p 2222"));
    }
}
