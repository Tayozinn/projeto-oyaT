//! Camada de aplicação (`app/`) — Fase 4 (`B10`).
//!
//! Concentra a lógica de orquestração de alto nível (carregamento de corpora JSONL,
//! exportação de pacotes quantizados `SMQ1`, currículo e marcos evolutivos da maratona,
//! e casos de uso de treinamento/avaliação neural) fora de `src/cli/`, permitindo que
//! a camada CLI apenas traduza argumentos e códigos de saída.

pub(crate) mod corpus;
pub(crate) mod marathon;
pub(crate) mod neural_service;
pub(crate) mod quantize;
