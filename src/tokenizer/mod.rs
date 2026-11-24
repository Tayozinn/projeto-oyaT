//! Agente 3 — Tokenizer BPE Próprio e Processamento de Dados.
//!
//! Implementa um tokenizer Byte-Pair Encoding (BPE) em nível de byte/UTF-8 treinado do zero,
//! com tokens especiais, persistência em JSON, limpeza de corpus e divisão determinística
//! entre conjuntos de treino, validação e teste.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

pub const PAD_TOKEN: &str = "<pad>";
pub const BOS_TOKEN: &str = "<bos>";
pub const EOS_TOKEN: &str = "<eos>";
pub const UNK_TOKEN: &str = "<unk>";
pub const SYSTEM_TOKEN: &str = "<system>";
pub const USER_TOKEN: &str = "<user>";
pub const ASSISTANT_TOKEN: &str = "<assistant>";

pub const SPECIAL_TOKENS: &[&str] = &[
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeTokenizer {
    pub vocab: Vec<String>,
    pub merges: Vec<(String, String)>,
    #[serde(skip)]
    token_to_id: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSplit {
    pub version: String,
    pub train: Vec<String>,
    pub validation: Vec<String>,
    pub test: Vec<String>,
}

impl BpeTokenizer {
    /// Treina um tokenizer BPE próprio a partir de um corpus de textos até `target_vocab_size`.
    pub fn train_from_corpus(texts: &[&str], target_vocab_size: usize) -> Self {
        let mut vocab: Vec<String> = SPECIAL_TOKENS.iter().map(|s| (*s).to_string()).collect();
        let mut token_to_id: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();

        // Vocabulário base de todos os 256 bytes possíveis codificados como tokens `<0xNN>` ou ASCII imprimível,
        // usando representação UTF-8 por caractere + fallback de byte para cobertura total sem `<unk>`.
        for b in 0u8..=255 {
            let tok = byte_token(b);
            if !token_to_id.contains_key(&tok) {
                let id = vocab.len();
                token_to_id.insert(tok.clone(), id);
                vocab.push(tok);
            }
        }

        // Pré-tokenização em palavras/espaços para aprender merges frequentes dentro de palavras.
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
        for text in texts {
            let cleaned = normalize_corpus_text(text);
            for piece in pretokenize(&cleaned) {
                let seq: Vec<String> = piece.as_bytes().iter().map(|&b| byte_token(b)).collect();
                if !seq.is_empty() {
                    *word_freqs.entry(seq).or_insert(0) += 1;
                }
            }
        }

        let mut merges: Vec<(String, String)> = Vec::new();
        while vocab.len() < target_vocab_size {
            let mut pair_counts: BTreeMap<(String, String), usize> = BTreeMap::new();
            for (seq, freq) in &word_freqs {
                if seq.len() < 2 {
                    continue;
                }
                for pair in seq.windows(2) {
                    *pair_counts
                        .entry((pair[0].clone(), pair[1].clone()))
                        .or_insert(0) += freq;
                }
            }

            let Some(((a, b), count)) = pair_counts.into_iter().max_by_key(|(_, c)| *c) else {
                break;
            };
            if count < 2 {
                break;
            }

            let merged = format!("{}{}", a, b);
            if !token_to_id.contains_key(&merged) {
                let id = vocab.len();
                token_to_id.insert(merged.clone(), id);
                vocab.push(merged.clone());
            }
            merges.push((a.clone(), b.clone()));

            // Aplica o merge em todas as palavras do corpus de treino.
            let mut next_freqs: HashMap<Vec<String>, usize> =
                HashMap::with_capacity(word_freqs.len());
            for (seq, freq) in word_freqs {
                let new_seq = merge_pair_in_seq(&seq, &a, &b, &merged);
                *next_freqs.entry(new_seq).or_insert(0) += freq;
            }
            word_freqs = next_freqs;
        }

        Self {
            vocab,
            merges,
            token_to_id,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn token_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    pub fn bos_id(&self) -> usize {
        self.token_id(BOS_TOKEN).unwrap_or(1)
    }

    pub fn eos_id(&self) -> usize {
        self.token_id(EOS_TOKEN).unwrap_or(2)
    }

    pub fn pad_id(&self) -> usize {
        self.token_id(PAD_TOKEN).unwrap_or(0)
    }

    /// Codifica um texto em IDs de tokens usando os merges aprendidos e preservando tokens especiais.
    pub fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<usize> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.bos_id());
        }

        let unk_id = self.token_id(UNK_TOKEN).unwrap_or(3);
        let mut cursor = 0usize;
        while cursor < text.len() {
            if let Some((special, sp_id)) = self.match_special_prefix(&text[cursor..]) {
                ids.push(sp_id);
                cursor += special.len();
                continue;
            }
            let next_special = find_next_special(&text[cursor..]);
            let segment = &text[cursor..cursor + next_special];
            for piece in pretokenize(segment) {
                let mut seq: Vec<String> =
                    piece.as_bytes().iter().map(|&b| byte_token(b)).collect();
                for (a, b) in &self.merges {
                    if seq.len() < 2 {
                        break;
                    }
                    let merged = format!("{}{}", a, b);
                    seq = merge_pair_in_seq(&seq, a, b, &merged);
                }
                for tok in seq {
                    ids.push(self.token_to_id.get(&tok).copied().unwrap_or(unk_id));
                }
            }
            cursor += next_special;
        }

        if add_eos {
            ids.push(self.eos_id());
        }
        ids
    }

    /// Decodifica uma sequência de IDs de volta para texto UTF-8 legível.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        let mut out = String::new();

        for &id in ids {
            let Some(tok) = self.vocab.get(id) else {
                continue;
            };
            if SPECIAL_TOKENS.contains(&tok.as_str()) {
                flush_bytes(&mut bytes, &mut out);
                if tok == USER_TOKEN || tok == ASSISTANT_TOKEN || tok == SYSTEM_TOKEN {
                    out.push_str(tok);
                }
                continue;
            }
            decode_token_bytes(tok, &mut bytes);
        }
        flush_bytes(&mut bytes, &mut out);
        out
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Falha ao criar diretório do tokenizer: {}", e))?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Falha ao serializar tokenizer: {}", e))?;
        fs::write(path, json).map_err(|e| format!("Falha ao gravar tokenizer em {:?}: {}", path, e))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = fs::read_to_string(path)
            .map_err(|e| format!("Falha ao ler tokenizer em {:?}: {}", path, e))?;
        let mut tok: Self = serde_json::from_str(&raw)
            .map_err(|e| format!("Falha ao desserializar tokenizer: {}", e))?;
        tok.token_to_id = tok
            .vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();
        Ok(tok)
    }

    fn match_special_prefix<'a>(&self, slice: &'a str) -> Option<(&'static str, usize)> {
        for &sp in SPECIAL_TOKENS {
            if slice.starts_with(sp) {
                if let Some(id) = self.token_to_id.get(sp).copied() {
                    return Some((sp, id));
                }
            }
        }
        None
    }
}

/// Normaliza espaços e quebras de linha preservando identação e comandos técnicos.
pub fn normalize_corpus_text(text: &str) -> String {
    text.replace("\r\n", "\n").trim().to_string()
}

/// Extensões de texto/código aceitas no carregamento recursivo de diretórios de pré-treino.
pub const CORPUS_ALLOWED_EXTENSIONS: &[&str] = &[
    "txt", "md", "rs", "py", "sh", "toml", "json", "jsonl", "yaml", "yml", "sql", "conf",
];

/// Diretórios ignorados automaticamente ao varrer um corpus local.
pub const CORPUS_IGNORED_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    ".venv",
    "__pycache__",
    "checkpoints",
    "eval_reports",
];

const MAX_CORPUS_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// Carrega recursivamente arquivos de texto/código de um diretório de pré-treino em ordem
/// determinística, dividindo documentos longos em blocos de até `max_chunk_chars` caracteres
/// e filtrando diretórios de build/controle de versão e chaves privadas literais.
pub fn load_corpus_directory(dir: &Path, max_chunk_chars: usize) -> Result<Vec<String>, String> {
    if !dir.exists() {
        return Err(format!(
            "Diretório de corpus não encontrado: '{}'.",
            dir.display()
        ));
    }
    if !dir.is_dir() {
        return Err(format!(
            "O caminho informado em --corpus-dir não é um diretório: '{}'.",
            dir.display()
        ));
    }

    let mut files = Vec::new();
    collect_corpus_files(dir, &mut files)?;
    files.sort();

    let chunk_limit = max_chunk_chars.max(256);
    let mut chunks = Vec::new();

    for path in files {
        let meta = match fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_CORPUS_FILE_BYTES {
            continue;
        }
        let Ok(raw) = fs::read_to_string(&path) else {
            continue;
        };
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        if ext == "jsonl" {
            for line in raw.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || contains_private_key_marker(trimmed) {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    if let (Some(p), Some(r)) = (
                        val.get("prompt").and_then(|v| v.as_str()),
                        val.get("response").and_then(|v| v.as_str()),
                    ) {
                        let norm =
                            normalize_corpus_text(&format!("<user> {}\n<assistant> {}", p, r));
                        if !norm.is_empty() && !contains_private_key_marker(&norm) {
                            chunks.push(norm);
                        }
                        continue;
                    }
                }
                let norm = normalize_corpus_text(trimmed);
                if !norm.is_empty() {
                    chunks.push(norm);
                }
            }
        } else {
            let norm = normalize_corpus_text(&raw);
            if norm.is_empty() || contains_private_key_marker(&norm) {
                continue;
            }
            split_text_into_chunks(&norm, chunk_limit, &mut chunks);
        }
    }

    if chunks.is_empty() {
        return Err(format!(
            "Nenhum texto elegível encontrado em '{}'. Extensões aceitas: {}.",
            dir.display(),
            CORPUS_ALLOWED_EXTENSIONS.join(", ")
        ));
    }

    Ok(chunks)
}

fn collect_corpus_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir)
        .map_err(|e| format!("Falha ao ler diretório '{}': {}", dir.display(), e))?;
    let mut paths: Vec<std::path::PathBuf> = entries
        .filter_map(|e| e.ok().map(|entry| entry.path()))
        .collect();
    paths.sort();

    for path in paths {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let lower_name = name.to_ascii_lowercase();
        if path.is_dir() {
            if CORPUS_IGNORED_DIRS.contains(&lower_name.as_str()) {
                continue;
            }
            collect_corpus_files(&path, out)?;
        } else if path.is_file() {
            if lower_name.starts_with(".env")
                || lower_name.ends_with(".db")
                || lower_name.ends_with(".sqlite")
                || lower_name.ends_with(".pem")
                || lower_name.ends_with(".key")
            {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            if CORPUS_ALLOWED_EXTENSIONS.contains(&ext.as_str()) {
                out.push(path);
            }
        }
    }
    Ok(())
}

fn contains_private_key_marker(text: &str) -> bool {
    let upper = text.to_ascii_uppercase();
    upper.contains("-----BEGIN PRIVATE KEY")
        || upper.contains("-----BEGIN RSA PRIVATE KEY")
        || upper.contains("-----BEGIN OPENSSH PRIVATE KEY")
        || upper.contains("-----BEGIN EC PRIVATE KEY")
}

fn split_text_into_chunks(text: &str, max_chars: usize, out: &mut Vec<String>) {
    let mut current = String::new();
    let mut current_chars = 0usize;

    for line in text.lines() {
        let line_chars = line.chars().count();
        if current_chars > 0 && current_chars + line_chars + 1 > max_chars {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            current.clear();
            current_chars = 0;
        }
        if line_chars > max_chars {
            let mut slice = String::new();
            let mut count = 0usize;
            for ch in line.chars() {
                slice.push(ch);
                count += 1;
                if count >= max_chars {
                    let trimmed = slice.trim();
                    if !trimmed.is_empty() {
                        out.push(trimmed.to_string());
                    }
                    slice.clear();
                    count = 0;
                }
            }
            if !slice.trim().is_empty() {
                current.push_str(slice.trim());
                current_chars = current.chars().count();
            }
            continue;
        }
        if !current.is_empty() {
            current.push('\n');
            current_chars += 1;
        }
        current.push_str(line);
        current_chars += line_chars;
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
}

/// Divide exemplos limpos de forma determinística em treino, validação e teste.
pub fn split_dataset(
    raw_examples: &[String],
    version: &str,
    val_ratio: f64,
    test_ratio: f64,
) -> DatasetSplit {
    let mut clean: Vec<String> = raw_examples
        .iter()
        .map(|s| normalize_corpus_text(s))
        .filter(|s| !s.is_empty())
        .collect();
    // Deduplicação preservando ordem determinística
    let mut seen = std::collections::HashSet::new();
    clean.retain(|item| seen.insert(item.clone()));

    let n = clean.len();
    let test_count = ((n as f64) * test_ratio).round() as usize;
    let val_count = ((n as f64) * val_ratio).round() as usize;
    let train_end = n.saturating_sub(val_count + test_count);
    let val_end = n.saturating_sub(test_count);

    DatasetSplit {
        version: version.to_string(),
        train: clean[..train_end].to_vec(),
        validation: clean[train_end..val_end].to_vec(),
        test: clean[val_end..].to_vec(),
    }
}

fn byte_token(b: u8) -> String {
    if b.is_ascii_graphic() && b != b'<' && b != b'>' && b != b'\\' {
        (b as char).to_string()
    } else {
        format!("<0x{:02X}>", b)
    }
}

fn decode_token_bytes(tok: &str, out_bytes: &mut Vec<u8>) {
    let mut rest = tok;
    while !rest.is_empty() {
        if rest.starts_with("<0x") && rest.len() >= 6 && rest.as_bytes()[5] == b'>' {
            if let Ok(byte) = u8::from_str_radix(&rest[3..5], 16) {
                out_bytes.push(byte);
                rest = &rest[6..];
                continue;
            }
        }
        let ch = rest.chars().next().unwrap();
        let mut buf = [0u8; 4];
        out_bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        rest = &rest[ch.len_utf8()..];
    }
}

fn flush_bytes(bytes: &mut Vec<u8>, out: &mut String) {
    if !bytes.is_empty() {
        out.push_str(&String::from_utf8_lossy(bytes));
        bytes.clear();
    }
}

fn pretokenize(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut prev_ws = None;
    for (idx, ch) in text.char_indices() {
        let is_ws = ch.is_whitespace();
        if let Some(pw) = prev_ws {
            if pw != is_ws {
                out.push(&text[start..idx]);
                start = idx;
            }
        }
        prev_ws = Some(is_ws);
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

fn find_next_special(slice: &str) -> usize {
    let mut best = slice.len();
    for &sp in SPECIAL_TOKENS {
        if let Some(pos) = slice.find(sp) {
            if pos > 0 && pos < best {
                best = pos;
            }
        }
    }
    best
}

fn merge_pair_in_seq(seq: &[String], a: &str, b: &str, merged: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(seq.len());
    let mut i = 0;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == a && seq[i + 1] == b {
            out.push(merged.to_string());
            i += 2;
        } else {
            out.push(seq[i].clone());
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{split_dataset, BpeTokenizer, ASSISTANT_TOKEN, USER_TOKEN};
    use std::path::PathBuf;

    #[test]
    fn bpe_trains_encodes_decodes_portuguese_and_commands() {
        let corpus = [
            "<user> verificar disco cheio <assistant> use df -hT e du -sh /var/log",
            "<user> verificar portas abertas <assistant> use ss -tulpn para listar portas",
            "<user> verificar disco cheio no servidor <assistant> execute df -i para inodes",
        ];
        let tok = BpeTokenizer::train_from_corpus(&corpus, 320);
        assert!(tok.vocab_size() > 263);

        let sample = format!(
            "{} verificar disco cheio {} use df -hT",
            USER_TOKEN, ASSISTANT_TOKEN
        );
        let ids = tok.encode(&sample, true, true);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, sample);
    }

    #[test]
    fn bpe_saves_and_loads_roundtrip() {
        let corpus = ["systemctl status nginx && journalctl -u nginx -n 50"];
        let tok = BpeTokenizer::train_from_corpus(&corpus, 290);
        let tmp_dir = std::env::temp_dir().join(format!("sm_tok_test_{}", std::process::id()));
        let path: PathBuf = tmp_dir.join("tokenizer.json");
        tok.save(&path).unwrap();
        let loaded = BpeTokenizer::load(&path).unwrap();
        assert_eq!(tok.vocab_size(), loaded.vocab_size());
        assert_eq!(
            tok.encode("systemctl status", false, false),
            loaded.encode("systemctl status", false, false)
        );
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn splits_dataset_deterministically_with_dedup() {
        let raw: Vec<String> = (0..10)
            .map(|i| format!("exemplo {}", i))
            .chain(std::iter::once("exemplo 0".to_string()))
            .collect();
        let split = split_dataset(&raw, "1.0.0", 0.2, 0.1);
        assert_eq!(
            split.train.len() + split.validation.len() + split.test.len(),
            10
        );
        assert_eq!(split.validation.len(), 2);
        assert_eq!(split.test.len(), 1);
    }

    #[test]
    fn loads_corpus_directory_recursively_and_skips_ignored_or_secrets() {
        let tmp = std::env::temp_dir().join(format!(
            "sm_corpus_dir_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let sub = tmp.join("guides");
        let ignored = tmp.join("target");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::create_dir_all(&ignored).unwrap();

        std::fs::write(
            sub.join("linux.md"),
            "Use `systemctl status nginx` para verificar servicos.\nUse `df -hT` para disco.",
        )
        .unwrap();
        std::fs::write(
            tmp.join("qa.jsonl"),
            r#"{"prompt":"Como ver portas?","response":"Use ss -tulnp."}"#,
        )
        .unwrap();
        std::fs::write(ignored.join("skip.rs"), "fn ignored() {}").unwrap();
        std::fs::write(tmp.join(".env"), "SECRET=abc").unwrap();
        std::fs::write(
            tmp.join("secret.txt"),
            "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0\n-----END PRIVATE KEY-----",
        )
        .unwrap();

        let chunks = super::load_corpus_directory(&tmp, 300).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(chunks.iter().any(|c| c.contains("systemctl status nginx")));
        assert!(chunks
            .iter()
            .any(|c| c.contains("<user> Como ver portas?") && c.contains("<assistant> Use ss -tulnp.")));
        assert!(!chunks.iter().any(|c| c.contains("ignored")));
        assert!(!chunks.iter().any(|c| c.contains("BEGIN PRIVATE KEY")));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
