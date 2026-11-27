//! Agente 3 — Tokenizer BPE Próprio e Processamento de Dados.
//!
//! Implementa um tokenizer Byte-Pair Encoding (BPE) em nível de byte/UTF-8 treinado do zero,
//! com tokens especiais, persistência em JSON, limpeza de corpus e divisão determinística
//! entre conjuntos de treino, validação e teste.

#![allow(
    clippy::manual_find,
    clippy::needless_lifetimes,
    clippy::unnecessary_map_or
)]

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
    #[serde(skip)]
    merge_ranks: HashMap<(String, String), (usize, String)>,
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

        // Pré-tokenização em palavras/espaços isolando SPECIAL_TOKENS para não desperdiçar merges
        // fundindo `<user>`, `<assistant>`, `<system>`, etc.
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
        for text in texts {
            let cleaned = normalize_corpus_text(text);
            for segment in split_non_special_segments(&cleaned) {
                for piece in pretokenize(segment) {
                    let seq: Vec<String> =
                        piece.as_bytes().iter().map(|&b| byte_token(b)).collect();
                    if !seq.is_empty() {
                        *word_freqs.entry(seq).or_insert(0) += 1;
                    }
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

            // Aplica o merge apenas nas sequências que contêm o par adjacente `(a, b)`.
            let mut next_freqs: HashMap<Vec<String>, usize> =
                HashMap::with_capacity(word_freqs.len());
            for (mut seq, freq) in word_freqs {
                merge_pair_in_place(&mut seq, &a, &b, Some(&merged));
                *next_freqs.entry(seq).or_insert(0) += freq;
            }
            word_freqs = next_freqs;
        }

        let merge_ranks = build_merge_ranks(&merges);
        Self {
            vocab,
            merges,
            token_to_id,
            merge_ranks,
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
    /// Normaliza `\r\n` para `\n` e evita alocações em merges que não ocorrem na palavra.
    pub fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<usize> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.bos_id());
        }

        let normalized: std::borrow::Cow<'_, str> = if text.contains("\r\n") {
            std::borrow::Cow::Owned(text.replace("\r\n", "\n"))
        } else {
            std::borrow::Cow::Borrowed(text)
        };
        let input = normalized.as_ref();

        let unk_id = self.token_id(UNK_TOKEN).unwrap_or(3);
        let mut cursor = 0usize;
        while cursor < input.len() {
            if let Some((special, sp_id)) = self.match_special_prefix(&input[cursor..]) {
                ids.push(sp_id);
                cursor += special.len();
                continue;
            }
            let next_special = find_next_special(&input[cursor..]);
            let segment = &input[cursor..cursor + next_special];
            for piece in pretokenize(segment) {
                let mut seq: Vec<String> =
                    piece.as_bytes().iter().map(|&b| byte_token(b)).collect();
                if !self.merge_ranks.is_empty() {
                    while seq.len() >= 2 {
                        let mut best: Option<(usize, String, String, String)> = None;
                        for pair in seq.windows(2) {
                            if let Some((rank, merged)) =
                                self.merge_ranks.get(&(pair[0].clone(), pair[1].clone()))
                            {
                                if best
                                    .as_ref()
                                    .map_or(true, |(b_rank, _, _, _)| *rank < *b_rank)
                                {
                                    best = Some((
                                        *rank,
                                        pair[0].clone(),
                                        pair[1].clone(),
                                        merged.clone(),
                                    ));
                                }
                            }
                        }
                        let Some((_, a, b, merged)) = best else {
                            break;
                        };
                        merge_pair_in_place(&mut seq, &a, &b, Some(&merged));
                    }
                } else {
                    for (a, b) in &self.merges {
                        if seq.len() < 2 {
                            break;
                        }
                        merge_pair_in_place(&mut seq, a, b, None);
                    }
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
        use std::io::Write;
        let parent = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => {
                fs::create_dir_all(p)
                    .map_err(|e| format!("Falha ao criar diretório do tokenizer: {}", e))?;
                p
            }
            _ => Path::new("."),
        };
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Falha ao serializar tokenizer: {}", e))?;

        static TOK_ATOMIC_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let seq = TOK_ATOMIC_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let fname = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("tokenizer.json");
        let tmp_path = parent.join(format!(".{}.tmp.{}.{}", fname, std::process::id(), seq));

        let write_res = (|| -> Result<(), String> {
            let mut file = fs::File::create(&tmp_path).map_err(|e| {
                format!(
                    "Falha ao criar arquivo temporário do tokenizer {:?}: {}",
                    tmp_path, e
                )
            })?;
            file.write_all(json.as_bytes())
                .map_err(|e| format!("Falha ao escrever tokenizer: {}", e))?;
            file.flush()
                .map_err(|e| format!("Falha no flush do tokenizer: {}", e))?;
            file.sync_all()
                .map_err(|e| format!("Falha no sync_all do tokenizer: {}", e))?;
            Ok(())
        })();

        if let Err(e) = write_res {
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }

        let mut last_err = None;
        for attempt in 0..3 {
            match fs::rename(&tmp_path, path) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    last_err = Some(e);
                    if attempt < 2 {
                        std::thread::sleep(std::time::Duration::from_millis(15));
                    }
                }
            }
        }
        let _ = fs::remove_file(&tmp_path);
        Err(format!(
            "Falha ao gravar atomicamente tokenizer em {:?}: {}",
            path,
            last_err.map(|e| e.to_string()).unwrap_or_default()
        ))
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
        tok.merge_ranks = build_merge_ranks(&tok.merges);
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

fn build_merge_ranks(merges: &[(String, String)]) -> HashMap<(String, String), (usize, String)> {
    let mut map = HashMap::with_capacity(merges.len());
    for (rank, (a, b)) in merges.iter().enumerate() {
        map.entry((a.clone(), b.clone()))
            .or_insert_with(|| (rank, format!("{}{}", a, b)));
    }
    map
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

/// Divide exemplos limpos de forma determinística (com deduplicação e embaralhamento reproduzível)
/// em treino, validação e teste. Valida que `val_ratio` e `test_ratio` deixem fração positiva de treino.
pub fn try_split_dataset(
    raw_examples: &[String],
    version: &str,
    val_ratio: f64,
    test_ratio: f64,
) -> Result<DatasetSplit, String> {
    if !val_ratio.is_finite()
        || !test_ratio.is_finite()
        || val_ratio < 0.0
        || test_ratio < 0.0
        || val_ratio + test_ratio >= 1.0
    {
        return Err(format!(
            "Proporções de divisão inválidas (val_ratio={}, test_ratio={}): exigido >= 0 e soma < 1.0.",
            val_ratio, test_ratio
        ));
    }

    let mut clean: Vec<String> = raw_examples
        .iter()
        .map(|s| normalize_corpus_text(s))
        .filter(|s| !s.is_empty())
        .collect();
    // Deduplicação preservando ordem canônica inicial
    let mut seen = std::collections::HashSet::new();
    clean.retain(|item| seen.insert(item.clone()));

    // Embaralhamento Fisher-Yates determinístico (SplitMix64 ancorado em `version`)
    // para evitar que exemplos no fim da lista (ex.: `--dataset` após `--corpus-dir`)
    // caiam 100% em validação/teste.
    deterministic_shuffle(&mut clean, hash_version_seed(version));

    let n = clean.len();
    let mut test_count = ((n as f64) * test_ratio).round() as usize;
    let mut val_count = ((n as f64) * val_ratio).round() as usize;
    if n > 0 && val_count + test_count >= n {
        // Garante ao menos 1 exemplo em `train` quando `n >= 1`
        let max_holdout = n - 1;
        test_count = test_count.min(max_holdout);
        val_count = val_count.min(max_holdout - test_count);
    }
    let train_end = n.saturating_sub(val_count + test_count);
    let val_end = n.saturating_sub(test_count);

    Ok(DatasetSplit {
        version: version.to_string(),
        train: clean[..train_end].to_vec(),
        validation: clean[train_end..val_end].to_vec(),
        test: clean[val_end..].to_vec(),
    })
}

/// Versão infalível de [`try_split_dataset`], limitando proporções fora de faixa de forma segura.
pub fn split_dataset(
    raw_examples: &[String],
    version: &str,
    val_ratio: f64,
    test_ratio: f64,
) -> DatasetSplit {
    let v = if val_ratio.is_finite() {
        val_ratio.clamp(0.0, 0.45)
    } else {
        0.15
    };
    let t = if test_ratio.is_finite() {
        test_ratio.clamp(0.0, 0.45)
    } else {
        0.10
    };
    try_split_dataset(raw_examples, version, v, t).unwrap_or_else(|_| DatasetSplit {
        version: version.to_string(),
        train: Vec::new(),
        validation: Vec::new(),
        test: Vec::new(),
    })
}

fn hash_version_seed(version: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in version.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h ^ 0x534D_3030_325F_5350 // "SM002_SP"
}

fn deterministic_shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for i in (1..items.len()).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let j = (z as usize) % (i + 1);
        items.swap(i, j);
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

fn match_special_token_prefix(slice: &str) -> Option<&'static str> {
    for &sp in SPECIAL_TOKENS {
        if slice.starts_with(sp) {
            return Some(sp);
        }
    }
    None
}

fn split_non_special_segments(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < text.len() {
        if let Some(sp) = match_special_token_prefix(&text[cursor..]) {
            cursor += sp.len();
            continue;
        }
        let next_special = find_next_special(&text[cursor..]);
        let segment = &text[cursor..cursor + next_special];
        if !segment.is_empty() {
            out.push(segment);
        }
        cursor += next_special;
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

#[inline]
fn has_adjacent_pair(seq: &[String], a: &str, b: &str) -> bool {
    seq.windows(2).any(|w| w[0] == a && w[1] == b)
}

fn merge_pair_in_place(seq: &mut Vec<String>, a: &str, b: &str, precomputed_merged: Option<&str>) {
    if !has_adjacent_pair(seq, a, b) {
        return;
    }
    let merged_owned;
    let merged = match precomputed_merged {
        Some(m) => m,
        None => {
            merged_owned = format!("{}{}", a, b);
            merged_owned.as_str()
        }
    };
    let mut write = 0usize;
    let mut read = 0usize;
    while read < seq.len() {
        if read + 1 < seq.len() && seq[read] == a && seq[read + 1] == b {
            seq[write] = merged.to_string();
            write += 1;
            read += 2;
        } else {
            if write != read {
                seq.swap(write, read);
            }
            write += 1;
            read += 1;
        }
    }
    seq.truncate(write);
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
            .any(|c| c.contains("<user> Como ver portas?")
                && c.contains("<assistant> Use ss -tulnp.")));
        assert!(!chunks.iter().any(|c| c.contains("ignored")));
        assert!(!chunks.iter().any(|c| c.contains("BEGIN PRIVATE KEY")));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn bpe_isolates_special_tokens_and_merge_ranks_match_sequential_merges() {
        let corpus = [
            "<user> diagnosticar servidor linux\r\n<assistant> execute systemctl status nginx",
            "<user> diagnosticar servidor web\n<assistant> execute journalctl -u nginx",
            "<user> diagnosticar portas abertas\n<assistant> execute ss -tulnp",
        ];
        let tok = BpeTokenizer::train_from_corpus(&corpus, 310);

        // Nenhum merge deve desperdiçar vocabulário recriando `<user>` ou `<assistant>`
        for (a, b) in &tok.merges {
            let m = format!("{}{}", a, b);
            assert!(
                !m.contains("user") && !m.contains("assistant"),
                "merge indevido de token especial encontrado: {}",
                m
            );
        }

        // `encode` com CRLF e LF deve ser idêntico
        let a = tok.encode("<user> diagnosticar\r\n<assistant> execute", true, true);
        let b = tok.encode("<user> diagnosticar\n<assistant> execute", true, true);
        assert_eq!(a, b);

        // Equivalência bit-a-bit entre `merge_ranks` O(L^2) e fallback sequencial O(|M|*L)
        let mut tok_no_ranks = tok.clone();
        tok_no_ranks.merge_ranks.clear();
        let probe = "<user> diagnosticar servidor linux e portas <assistant> execute systemctl status nginx";
        assert_eq!(
            tok.encode(probe, true, true),
            tok_no_ranks.encode(probe, true, true)
        );
    }

    #[test]
    fn split_dataset_mixes_tail_qa_into_train_and_validates_ratios() {
        let mut combined: Vec<String> = (0..300).map(|i| format!("livro capitulo {}", i)).collect();
        for q in 0..100 {
            combined.push(format!("qa_item_{}", q));
        }
        let split = split_dataset(&combined, "v3-test", 0.15, 0.10);
        let qa_in_train = split
            .train
            .iter()
            .filter(|s| s.starts_with("qa_item_"))
            .count();
        // Com 75% de treino, espera-se ~75 exemplos Q&A (antes de SM-002 era 0!).
        assert!(
            (60..=90).contains(&qa_in_train),
            "Q&A mal distribuído em train: {}",
            qa_in_train
        );

        // Determinismo exato entre chamadas repetidas
        let split2 = split_dataset(&combined, "v3-test", 0.15, 0.10);
        assert_eq!(split.train, split2.train);
        assert_eq!(split.validation, split2.validation);

        // Validação explícita de proporções inválidas
        assert!(super::try_split_dataset(&combined, "v3", 0.6, 0.5).is_err());
        assert!(super::try_split_dataset(&combined, "v3", -0.1, 0.1).is_err());
    }
}
