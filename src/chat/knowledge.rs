//! Base de conhecimento técnica local (RAG consultivo determinístico).
//!
//! Carrega runbooks técnicos estruturados de `knowledge/sysadmin_kb.json` (com fallback
//! embutido via `include_str!`) sem dependências externas, sem rede e sem alterar `servermind.db`.
//! Fornece busca por similaridade léxica para fundamentar respostas do chat interativo
//! (`agent.rs`) e inspeção direta via comando `:kb`.

use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const EMBEDDED_KB_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/knowledge/sysadmin_kb.json"
));
pub const DEFAULT_KB_RELATIVE_PATH: &str = "knowledge/sysadmin_kb.json";

#[derive(Debug, Clone, Deserialize)]
pub struct KbEntry {
    pub id: String,
    pub title: String,
    pub tags: Vec<String>,
    pub summary: String,
    pub guidance: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct KbHit {
    pub entry: KbEntry,
    pub score: usize,
}

/// Carrega a base técnica local de `<base_dir>/knowledge/sysadmin_kb.json` quando presente,
/// ou usa a cópia embutida no binário para funcionar de qualquer pasta (`B8`).
pub fn load_kb_entries_from_dir(base_dir: &Path) -> Vec<KbEntry> {
    let path = base_dir.join(DEFAULT_KB_RELATIVE_PATH);
    if let Ok(raw) = fs::read_to_string(&path) {
        if let Ok(parsed) = serde_json::from_str::<Vec<KbEntry>>(&raw) {
            if !parsed.is_empty() {
                return parsed;
            }
        }
    }
    serde_json::from_str::<Vec<KbEntry>>(EMBEDDED_KB_JSON).unwrap_or_default()
}

/// Carrega a base técnica local resolvendo `knowledge/sysadmin_kb.json` pelo diretório ativo da config.
#[cfg(test)]
pub fn load_kb_entries() -> Vec<KbEntry> {
    let base_dir = crate::chat::config::AppConfig::resolve_active_config_dir();
    load_kb_entries_from_dir(&base_dir)
}

/// Busca as `limit` notas técnicas mais relevantes resolvendo a KB pelo diretório ativo da config.
pub fn retrieve_relevant_notes(query: &str, limit: usize) -> Vec<KbHit> {
    let base_dir = crate::chat::config::AppConfig::resolve_active_config_dir();
    retrieve_relevant_notes_from_dir(&base_dir, query, limit)
}

/// Busca as `limit` notas técnicas mais relevantes a partir de `base_dir` (`B8`).
pub fn retrieve_relevant_notes_from_dir(base_dir: &Path, query: &str, limit: usize) -> Vec<KbHit> {
    if limit == 0 || query.trim().is_empty() {
        return Vec::new();
    }
    let entries = load_kb_entries_from_dir(base_dir);
    let norm_query = normalize_ascii_lower(query);
    let query_tokens = tokenize(&norm_query);
    if query_tokens.is_empty() {
        return Vec::new();
    }

    let mut scored: Vec<KbHit> = entries
        .into_iter()
        .filter_map(|entry| {
            let mut score = 0usize;
            for tag in &entry.tags {
                let norm_tag = normalize_ascii_lower(tag);
                if norm_query.contains(&norm_tag) {
                    score += if norm_tag.contains(' ') { 6 } else { 4 };
                } else {
                    for t_tok in tokenize(&norm_tag) {
                        if query_tokens.contains(&t_tok) {
                            score += 2;
                        }
                    }
                }
            }
            let title_tokens = tokenize(&normalize_ascii_lower(&entry.title));
            for q_tok in &query_tokens {
                if title_tokens.contains(q_tok) {
                    score += 2;
                }
            }
            if score >= 4 {
                Some(KbHit { entry, score })
            } else {
                None
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.entry.id.cmp(&b.entry.id))
    });
    scored.truncate(limit);
    scored
}

/// Formata as notas recuperadas como bloco de referência consultiva para o contexto do modelo.
pub fn format_grounding_note(hits: &[KbHit]) -> Option<String> {
    if hits.is_empty() {
        return None;
    }
    let mut out = String::from(
        "[REFERÊNCIA TÉCNICA LOCAL VERIFICADA — use como base factual sem inventar URLs, flags ou caminhos fora dela]\n",
    );
    for hit in hits {
        out.push_str(&format!("\n### {} ({})\n", hit.entry.title, hit.entry.id));
        for line in &hit.entry.guidance {
            out.push_str(line);
            out.push('\n');
        }
    }
    Some(out)
}

fn tokenize(norm: &str) -> BTreeSet<String> {
    norm.split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '/')
        .filter(|w| w.len() >= 3)
        .filter(|w| {
            !matches!(
                *w,
                "como"
                    | "para"
                    | "com"
                    | "sem"
                    | "que"
                    | "uma"
                    | "nao"
                    | "não"
                    | "sobre"
                    | "qual"
                    | "meu"
                    | "minha"
                    | "isso"
                    | "esta"
                    | "esse"
                    | "voce"
                    | "quero"
            )
        })
        .map(|w| w.to_string())
        .collect()
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
    use super::{format_grounding_note, load_kb_entries, retrieve_relevant_notes};

    #[test]
    fn loads_embedded_or_disk_kb_entries() {
        let entries = load_kb_entries();
        assert!(entries.len() >= 8);
    }

    #[test]
    fn retrieves_dynamic_ip_cgnat_note_for_home_server_question() {
        let hits =
            retrieve_relevant_notes("quero montar um servidor na minha maquina sem ip fixo", 2);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].entry.id, "self-hosting-dynamic-ip-cgnat");
        let formatted = format_grounding_note(&hits).unwrap();
        assert!(formatted.contains("CGNAT"));
    }

    #[test]
    fn retrieves_scope_note_for_realtime_or_non_infra_question() {
        let hits =
            retrieve_relevant_notes("sabe o que aconteceu com a nvidia nos ultimos dias?", 2);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].entry.id, "scope-realtime-and-non-infra");
    }
}
