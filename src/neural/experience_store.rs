//! Agente 5 — Memória em 3 Camadas e Aprendizado por Experiências Verificadas.
//!
//! Camada 1: Memória Operacional (`operational_memory`) — servidores, ambientes, projetos e configs.
//! Camada 2: Memória de Experiências (`experience_log`) — problemas, diagnósticos, soluções, resultados e erros.
//! Camada 3: Memória de Treinamento (`training_examples`) — exemplos aprovados por gate rigoroso
//!           para ciclos controlados de treinamento (nunca atualizando pesos a cada chat).

#![allow(clippy::too_many_arguments)]

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalRecord {
    pub id: i64,
    pub scope: String,
    pub key: String,
    pub value: String,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewExperience {
    pub domain: String,
    pub problem: String,
    pub diagnosis: String,
    pub solution: String,
    pub outcome: String,
    pub error_notes: String,
    pub verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceRecord {
    pub id: i64,
    pub domain: String,
    pub problem: String,
    pub diagnosis: String,
    pub solution: String,
    pub outcome: String,
    pub error_notes: String,
    pub verified: bool,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationVerdict {
    pub approved: bool,
    pub quality_score: f64,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExampleRecord {
    pub id: i64,
    pub experience_id: Option<i64>,
    pub prompt: String,
    pub ideal_response: String,
    pub domain: String,
    pub outcome: String,
    pub quality_score: f64,
    pub approved: bool,
    pub rejection_reason: Option<String>,
    pub dataset_version: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceStats {
    pub operational_count: usize,
    pub experience_count: usize,
    pub verified_experience_count: usize,
    pub approved_training_count: usize,
    pub rejected_training_count: usize,
}

pub struct ExperienceStore {
    conn: Connection,
}

impl ExperienceStore {
    pub fn open(path: &Path) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| {
            format!(
                "Falha ao abrir banco de experiências '{}': {}",
                path.display(),
                e
            )
        })?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    #[cfg(test)]
    pub fn in_memory() -> Result<Self, String> {
        let conn = Connection::open_in_memory()
            .map_err(|e| format!("Falha ao abrir ExperienceStore em memória: {}", e))?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<(), String> {
        self.conn
            .execute_batch(
                "
                CREATE TABLE IF NOT EXISTS operational_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(scope, key)
                );
                CREATE TABLE IF NOT EXISTS experience_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    problem TEXT NOT NULL,
                    diagnosis TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    error_notes TEXT NOT NULL,
                    verified INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experience_id INTEGER,
                    prompt TEXT NOT NULL,
                    ideal_response TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    approved INTEGER NOT NULL,
                    rejection_reason TEXT,
                    dataset_version TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
                ",
            )
            .map_err(|e| format!("Falha ao inicializar schema de experiências: {}", e))
    }

    pub fn upsert_operational(&self, scope: &str, key: &str, value: &str) -> Result<(), String> {
        let scope = scope.trim();
        let key = key.trim();
        let value = value.trim();
        if scope.is_empty() || key.is_empty() || value.is_empty() {
            return Err("Scope, key e value são obrigatórios na memória operacional.".to_string());
        }
        let now = unix_now();
        self.conn
            .execute(
                "INSERT INTO operational_memory (scope, key, value, updated_at)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(scope, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                params![scope, key, value, now],
            )
            .map_err(|e| format!("Falha ao gravar memória operacional: {}", e))?;
        Ok(())
    }

    pub fn list_operational(&self, scope: Option<&str>) -> Result<Vec<OperationalRecord>, String> {
        let mut out = Vec::new();
        if let Some(sc) = scope {
            let mut stmt = self
                .conn
                .prepare("SELECT id, scope, key, value, updated_at FROM operational_memory WHERE scope = ?1 ORDER BY key ASC")
                .map_err(|e| e.to_string())?;
            let rows = stmt
                .query_map(params![sc], |r| {
                    Ok(OperationalRecord {
                        id: r.get(0)?,
                        scope: r.get(1)?,
                        key: r.get(2)?,
                        value: r.get(3)?,
                        updated_at: r.get(4)?,
                    })
                })
                .map_err(|e| e.to_string())?;
            for row in rows {
                out.push(row.map_err(|e| e.to_string())?);
            }
        } else {
            let mut stmt = self
                .conn
                .prepare("SELECT id, scope, key, value, updated_at FROM operational_memory ORDER BY scope ASC, key ASC")
                .map_err(|e| e.to_string())?;
            let rows = stmt
                .query_map([], |r| {
                    Ok(OperationalRecord {
                        id: r.get(0)?,
                        scope: r.get(1)?,
                        key: r.get(2)?,
                        value: r.get(3)?,
                        updated_at: r.get(4)?,
                    })
                })
                .map_err(|e| e.to_string())?;
            for row in rows {
                out.push(row.map_err(|e| e.to_string())?);
            }
        }
        Ok(out)
    }

    pub fn record_experience(&self, entry: &NewExperience) -> Result<i64, String> {
        if entry.problem.trim().is_empty() || entry.solution.trim().is_empty() {
            return Err(
                "Problema e solução são obrigatórios para registrar experiência.".to_string(),
            );
        }
        let now = unix_now();
        self.conn
            .execute(
                "INSERT INTO experience_log (domain, problem, diagnosis, solution, outcome, error_notes, verified, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    entry.domain.trim(),
                    entry.problem.trim(),
                    entry.diagnosis.trim(),
                    entry.solution.trim(),
                    entry.outcome.trim(),
                    entry.error_notes.trim(),
                    if entry.verified { 1 } else { 0 },
                    now
                ],
            )
            .map_err(|e| format!("Falha ao registrar experiência: {}", e))?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Revisa e aprova explicitamente uma experiência existente na Camada 2 (`experience_log`),
    /// marcando `verified = 1` e submetendo-a ao gate da Camada 3 (`training_examples`) (`A17`).
    pub fn approve_experience_for_training(
        &self,
        exp_id: i64,
        dataset_version: &str,
    ) -> Result<TrainingExampleRecord, String> {
        let exp: ExperienceRecord = self
            .conn
            .query_row(
                "SELECT id, domain, problem, diagnosis, solution, outcome, error_notes, verified, created_at
                 FROM experience_log WHERE id = ?1",
                params![exp_id],
                |r| {
                    Ok(ExperienceRecord {
                        id: r.get(0)?,
                        domain: r.get(1)?,
                        problem: r.get(2)?,
                        diagnosis: r.get(3)?,
                        solution: r.get(4)?,
                        outcome: r.get(5)?,
                        error_notes: r.get(6)?,
                        verified: r.get::<_, i64>(7)? != 0,
                        created_at: r.get(8)?,
                    })
                },
            )
            .map_err(|_| format!("Experiência #{} não encontrada em `experience_log`.", exp_id))?;

        let existing_approved: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM training_examples WHERE experience_id = ?1 AND approved = 1",
                params![exp_id],
                |r| r.get(0),
            )
            .map_err(|e| e.to_string())?;
        if existing_approved > 0 {
            return Err(format!(
                "A experiência #{} já foi revisada e aprovada anteriormente para treinamento.",
                exp_id
            ));
        }

        self.conn
            .execute(
                "UPDATE experience_log SET verified = 1 WHERE id = ?1",
                params![exp_id],
            )
            .map_err(|e| {
                format!(
                    "Falha ao marcar experiência #{} como verificada: {}",
                    exp_id, e
                )
            })?;

        let ideal = if exp.diagnosis.trim().is_empty() {
            exp.solution.clone()
        } else {
            format!("Diagnóstico: {}. Solução: {}", exp.diagnosis, exp.solution)
        };
        self.submit_for_training(
            Some(exp.id),
            &exp.problem,
            &ideal,
            &exp.domain,
            &exp.outcome,
            true,
            dataset_version,
        )
    }

    pub fn list_experiences(&self, limit: usize) -> Result<Vec<ExperienceRecord>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, domain, problem, diagnosis, solution, outcome, error_notes, verified, created_at
                 FROM experience_log ORDER BY id DESC LIMIT ?1",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(params![limit as i64], |r| {
                Ok(ExperienceRecord {
                    id: r.get(0)?,
                    domain: r.get(1)?,
                    problem: r.get(2)?,
                    diagnosis: r.get(3)?,
                    solution: r.get(4)?,
                    outcome: r.get(5)?,
                    error_notes: r.get(6)?,
                    verified: r.get::<_, i64>(7)? != 0,
                    created_at: r.get(8)?,
                })
            })
            .map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(|e| e.to_string())?);
        }
        Ok(out)
    }

    /// Submete um exemplo à Camada 3 passando obrigatoriamente pelo gate de validação.
    /// Exemplos reprovados ficam auditáveis (`approved = 0`) com `rejection_reason` e nunca entram no export.
    pub fn submit_for_training(
        &self,
        experience_id: Option<i64>,
        prompt: &str,
        ideal_response: &str,
        domain: &str,
        outcome: &str,
        verified: bool,
        dataset_version: &str,
    ) -> Result<TrainingExampleRecord, String> {
        let verdict = validate_experience_for_training(prompt, ideal_response, outcome, verified);
        let rejection_reason = if verdict.approved {
            None
        } else {
            Some(verdict.reasons.join("; "))
        };
        let now = unix_now();
        self.conn
            .execute(
                "INSERT INTO training_examples (experience_id, prompt, ideal_response, domain, outcome, quality_score, approved, rejection_reason, dataset_version, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    experience_id,
                    prompt.trim(),
                    ideal_response.trim(),
                    domain.trim(),
                    outcome.trim(),
                    verdict.quality_score,
                    if verdict.approved { 1 } else { 0 },
                    rejection_reason,
                    dataset_version.trim(),
                    now
                ],
            )
            .map_err(|e| format!("Falha ao registrar exemplo de treinamento: {}", e))?;
        let id = self.conn.last_insert_rowid();
        Ok(TrainingExampleRecord {
            id,
            experience_id,
            prompt: prompt.trim().to_string(),
            ideal_response: ideal_response.trim().to_string(),
            domain: domain.trim().to_string(),
            outcome: outcome.trim().to_string(),
            quality_score: verdict.quality_score,
            approved: verdict.approved,
            rejection_reason,
            dataset_version: dataset_version.trim().to_string(),
            created_at: now,
        })
    }

    pub fn export_approved_jsonl(
        &self,
        output_path: &Path,
        dataset_version: Option<&str>,
    ) -> Result<usize, String> {
        let file = File::create(output_path).map_err(|e| {
            format!(
                "Falha ao criar arquivo de exportação '{}': {}",
                output_path.display(),
                e
            )
        })?;
        let mut writer = BufWriter::new(file);
        let mut count = 0usize;

        let mut query = String::from(
            "SELECT prompt, ideal_response, domain, outcome, dataset_version
             FROM training_examples WHERE approved = 1",
        );
        if dataset_version.is_some() {
            query.push_str(" AND dataset_version = ?1");
        }
        query.push_str(" ORDER BY id ASC");

        let mut stmt = self.conn.prepare(&query).map_err(|e| e.to_string())?;
        let mapper = |r: &rusqlite::Row<'_>| {
            Ok(serde_json::json!({
                "prompt": r.get::<_, String>(0)?,
                "response": r.get::<_, String>(1)?,
                "domain": r.get::<_, String>(2)?,
                "outcome": r.get::<_, String>(3)?,
                "dataset_version": r.get::<_, String>(4)?,
            }))
        };

        if let Some(ver) = dataset_version {
            let rows = stmt
                .query_map(params![ver], mapper)
                .map_err(|e| e.to_string())?;
            for row in rows {
                let val = row.map_err(|e| e.to_string())?;
                writeln!(writer, "{}", val).map_err(|e| e.to_string())?;
                count += 1;
            }
        } else {
            let rows = stmt.query_map([], mapper).map_err(|e| e.to_string())?;
            for row in rows {
                let val = row.map_err(|e| e.to_string())?;
                writeln!(writer, "{}", val).map_err(|e| e.to_string())?;
                count += 1;
            }
        }
        writer.flush().map_err(|e| e.to_string())?;
        Ok(count)
    }

    pub fn stats(&self) -> Result<ExperienceStats, String> {
        let operational_count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM operational_memory", [], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let experience_count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM experience_log", [], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let verified_experience_count: usize = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM experience_log WHERE verified = 1",
                [],
                |r| r.get(0),
            )
            .map_err(|e| e.to_string())?;
        let approved_training_count: usize = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM training_examples WHERE approved = 1",
                [],
                |r| r.get(0),
            )
            .map_err(|e| e.to_string())?;
        let rejected_training_count: usize = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM training_examples WHERE approved = 0",
                [],
                |r| r.get(0),
            )
            .map_err(|e| e.to_string())?;
        Ok(ExperienceStats {
            operational_count,
            experience_count,
            verified_experience_count,
            approved_training_count,
            rejected_training_count,
        })
    }
}

/// Gate de validação que impede que experiências não verificadas, inseguras ou triviais
/// contaminem o dataset de treinamento do modelo próprio.
pub fn validate_experience_for_training(
    prompt: &str,
    ideal_response: &str,
    outcome: &str,
    verified: bool,
) -> ValidationVerdict {
    let mut reasons = Vec::new();
    let mut score = 1.0f64;

    if !verified {
        reasons.push("Experiência não verificada".to_string());
        score -= 0.5;
    }

    let outcome_norm = outcome.trim().to_ascii_lowercase();
    if outcome_norm != "success" && outcome_norm != "resolved" && outcome_norm != "verified_failure"
    {
        reasons.push(format!("Outcome inválido ou não conclusivo: '{}'", outcome));
        score -= 0.4;
    }

    if prompt.trim().len() < 12 {
        reasons.push("Prompt muito curto para aprendizado útil".to_string());
        score -= 0.3;
    }
    if ideal_response.trim().len() < 20 {
        reasons.push("Resposta ideal muito curta".to_string());
        score -= 0.3;
    }

    let lower = format!("{}\n{}", prompt, ideal_response).to_ascii_lowercase();
    let forbidden_patterns = [
        ("rm -rf /", "Comando destrutivo na raiz"),
        ("chmod -r 777 /", "Permissão insegura recursiva na raiz"),
        ("mkfs.", "Formatação destrutiva de sistema de arquivos"),
        (":(){", "Fork bomb detectado"),
        ("| sh", "Execução remota não verificada via pipe para shell"),
        (
            "| bash",
            "Execução remota não verificada via pipe para bash",
        ),
        (
            "ignore previous instructions",
            "Tentativa de prompt injection",
        ),
        ("reveal system prompt", "Tentativa de vazamento de prompt"),
        ("password=", "Credencial literal detectada"),
        ("passwd=", "Credencial literal detectada"),
        ("secret=", "Segredo literal detectado"),
        ("token=", "Token literal detectado"),
        ("api_key=", "Chave literal detectada"),
        ("-----begin", "Chave privada PEM detectada"),
    ];
    for (pat, msg) in forbidden_patterns {
        if lower.contains(pat) {
            reasons.push(msg.to_string());
            score = 0.0;
        }
    }

    let approved = reasons.is_empty() && score >= 0.7;
    ValidationVerdict {
        approved,
        quality_score: score.clamp(0.0, 1.0),
        reasons,
    }
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_layer_memory_and_validation_gate_work() {
        let store = ExperienceStore::in_memory().unwrap();
        store
            .upsert_operational(
                "server",
                "prod-web-01",
                "Ubuntu 24.04 | Nginx 1.24 | 8GB RAM",
            )
            .unwrap();
        let ops = store.list_operational(Some("server")).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(ops[0].value.contains("Nginx"));

        let exp_id = store
            .record_experience(&NewExperience {
                domain: "linux".to_string(),
                problem: "Nginx retornando 502 Bad Gateway após restart".to_string(),
                diagnosis: "Socket unix do php-fpm ausente ou serviço parado".to_string(),
                solution: "Verificar `systemctl status php8.2-fpm` e `ss -xl | grep php`"
                    .to_string(),
                outcome: "success".to_string(),
                error_notes: "Permissão do socket estava em root".to_string(),
                verified: true,
            })
            .unwrap();

        let approved = store
            .submit_for_training(
                Some(exp_id),
                "Como diagnosticar 502 Bad Gateway no Nginx?",
                "Inspecione `journalctl -u nginx -n 50 --no-pager`, confirme o upstream com `ss -tulnp` e valide o status do serviço backend.",
                "linux",
                "success",
                true,
                "exp_v1",
            )
            .unwrap();
        assert!(approved.approved);

        let rejected_unverified = store
            .submit_for_training(
                None,
                "Como limpar disco cheio?",
                "Execute rm -rf /tmp/* ou rm -rf / imediatamente.",
                "linux",
                "success",
                false,
                "exp_v1",
            )
            .unwrap();
        assert!(!rejected_unverified.approved);
        assert!(rejected_unverified.rejection_reason.is_some());

        let stats = store.stats().unwrap();
        assert_eq!(stats.operational_count, 1);
        assert_eq!(stats.experience_count, 1);
        assert_eq!(stats.approved_training_count, 1);
        assert_eq!(stats.rejected_training_count, 1);

        let out_file = std::env::temp_dir().join("servermind_exp_export_test.jsonl");
        let exported = store
            .export_approved_jsonl(&out_file, Some("exp_v1"))
            .unwrap();
        assert_eq!(exported, 1);
        let _ = std::fs::remove_file(out_file);
    }
}
