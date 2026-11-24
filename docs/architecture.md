# Arquitetura do ServerMind AI (Motor Neural Próprio em Rust)

O **ServerMind AI** evoluiu para uma plataforma híbrida em Rust puro com motor neural próprio treinado do zero (sem pesos pré-treinados de terceiros), mantendo o cliente **Ollama** como fallback configurável.

## Organização de Módulos (`src/`)

- `src/main.rs`: Ponto de entrada e despacho de subcomandos CLI (`eval`, `train-neural`, `eval-neural`, `hw`) e loop interativo.
- `src/infrastructure/mod.rs` *(Agente 8 — Infraestrutura e Hardware)*:
  - Sondagem de hardware (`probe_hardware()`), detecção de CPUs lógicas, disponibilidade de `vulkaninfo` / ROCm, estimativa de memória de modelos Transformer (`estimate_transformer_memory()`) e benchmark prático de multiplicação matricial em CPU (`benchmark_cpu_matmul()`).
- `src/tokenizer/mod.rs` *(Agente 3 — Tokenizer e Dados)*:
  - Tokenizer Byte-Pair Encoding (`BpeTokenizer`) treinado do zero sobre o corpus técnico (`train_from_corpus`), com cobertura de 256 bytes base + tokens especiais (`<pad>`, `<bos>`, `<eos>`, `<unk>`, `<system>`, `<user>`, `<assistant>`), normalização (`normalize_corpus_text`) e divisão determinística com deduplicação (`split_dataset`).
- `src/neural/mod.rs` *(Agente 2 — Engenharia do Modelo Neural)*:
  - Transformer Decoder-only em Rust puro (`ServerMindTransformer`), com embeddings de token/posição, RMSNorm, Multi-Head Causal Self-Attention, MLP GELU, conexões residuais, Cross-Entropy Loss, backpropagation analítica completa (`forward_and_backward`), otimizador `AdamWState`, inferência autoregressiva (`generate`) e checkpoints binários completos (`SMNN`).
- `src/training/mod.rs` *(Agente 4 — Sistema de Treinamento)*:
  - Empacotamento de sequências (`pack_sequences`), mini-batches com acumulação de gradientes, warmup linear + cosine decay (`cosine_lr`), avaliação periódica (`evaluate_loss`), salvamento de `latest.bin`, `best.bin` e `metrics.jsonl`, retomada via `Trainer::resume_from_checkpoint` e comparação entre versões (`compare_training_runs`).
- `src/experience/mod.rs` *(Agente 5 — Memória em 3 Camadas)*:
  - Banco SQLite dedicado (`ExperienceStore`) com:
    1. **Memória Operacional (`operational_memory`)**: configurações, servidores, ambientes e projetos.
    2. **Memória de Experiências (`experience_log`)**: problemas, diagnósticos, soluções, resultados e erros.
    3. **Memória de Treinamento (`training_examples`)**: protegida por gate rigoroso (`validate_experience_for_training`), bloqueando exemplos não verificados, comandos destrutivos, credenciais literais ou tentativas de prompt injection.
- `src/inference/mod.rs` *(Agente 6 — Integração e Inferência)*:
  - `InferenceRouter` e `NeuralEngine` permitindo alternar dinamicamente entre `Ollama` (fallback) e `ServerMindNeural` via `:backend ollama` ou `:backend neural [dir]`.
- `src/evaluation/mod.rs` *(Agente 7 — Avaliação e Qualidade)*:
  - Suite técnica (`default_technical_suite`, `evaluate_neural_engine`) medindo `val_loss`, perplexidade, latência de inferência e cobertura técnica em Linux, Docker, Redes, Segurança, Depuração e Programação.
- Módulos originais preservados: `agent.rs`, `cli.rs`, `config.rs`, `storage.rs`, `ollama.rs`, `models.rs`, `guardrails.rs`, `knowledge.rs`, `eval_*.rs`.

## Hardware-Alvo e Decisão de Execução

- Máquina avaliada: **Intel Core i5-9600K (6 cores / 6 threads)**, **16 GB RAM**, **AMD Radeon RX 7600 (8 GB VRAM)**, **Windows 11**.
- Como o ecossistema `Candle` no Windows não oferece aceleração AMD/ROCm nativa e `Burn-wgpu` adicionaria grande árvore de dependências para modelos experimentais, o motor neural atual opera em **CPU multi-thread em Rust puro** (`std::thread::scope`), mantendo compilação rápida, zero dependências externas pesadas e total domínio matemático.
