# HANDOFF COMPLETO — ServerMind AI (Estado, Decisões, Problemas e Próximos Passos)

> **Para o agente que assumir no Ubuntu:** leia este documento antes de qualquer alteração. Ele concentra todo o histórico técnico, decisões arquiteturais, problemas já resolvidos, métricas reais medidas e o próximo passo exato da evolução do **ServerMind AI**.

---

## 1. Visão, Identidade e Regras Inegociáveis do Projeto

- **Projeto:** **ServerMind AI** — assistente e plataforma de IA autônoma em **Rust puro**, especializada em programação, engenharia de software, administração de servidores Linux/Windows, redes, segurança, Docker, virtualização e diagnóstico de falhas.
- **Identidade fixa do criador:** `Você foi criado por TayoX.` (`CREATOR_IDENTITY_INSTRUCTION` em `src/config.rs`).
- **Objetivo neural:** desenvolver um modelo Transformer próprio, **treinado do zero** (tokenizer BPE próprio + arquitetura + pesos próprios, **sem usar pesos pré-treinados** de Qwen, Llama, DeepSeek ou terceiros).
- **Regras operacionais obrigatórias:**
  1. Preservar **Rust** como linguagem principal (não migrar para Python).
  2. **Não remover** o cliente **Ollama** como fallback configurável até que o modelo próprio atinja qualidade satisfatória em avaliação objetiva.
  3. **Nunca** atualizar pesos do modelo a cada conversa; o aprendizado ocorre apenas em ciclos controlados de treinamento com gate de validação e comparação de métricas.
  4. Manter papel estritamente consultivo no chat, com guardrails determinísticos (`src/guardrails.rs`) e diretrizes imutáveis de segurança (`src/config.rs`).
  5. **Nunca** corromper ou sobrescrever `servermind.db` nem `eval/cases_v1.json` durante testes (testes devem usar `:memory:` ou `std::env::temp_dir()`).
  6. **Não** inventar resultados de testes, benchmarks ou métricas.

---

## 2. Hardware do Usuário, Discos e Plano de Migração para Ubuntu

### Hardware verificado na máquina
- **CPU:** Intel Core i5-9600K @ 3.70GHz (6 cores / 6 threads lógicas)
- **RAM:** 16 GB
- **GPU:** AMD Radeon RX 7600 (8 GB VRAM, arquitetura RDNA 3)
- **Sondagem do próprio binário (`servermind-ai hw` no Windows):**
  - `OS / Arch: windows / x86_64`, `CPUs lógicas: 6`, `Vulkan: true`, `ROCm: false`
  - `Benchmark MatMul: 192x192 em 1 ms (12.32 GFLOPS)`

### Mapeamento real dos SSDs (verificado via `Get-Disk` / `Get-Partition` / `Get-Volume`)
- **Disco 0 (`SSD`, 931,51 GB, GPT):**
  - Partição 1: Reserved (`0,02 GB`)
  - Partição 2 (`C:`): `930,58 GB` NTFS (`631,13 GB` livres) — **Windows 11 ativo**.
  - Partição 3: Recovery (`0,91 GB`).
  - *Atenção:* o Disco 0 **não possui partição EFI (`System`) própria**.
- **Disco 1 (`KINGSTON SA400S37480G`, 447,13 GB, GPT):**
  - Partição 1: **`System` (EFI, `0,10 GB` / 100 MB)** — **contém o boot (BCD) do Windows atual!**
  - Partição 2: Reserved (`0,02 GB`).
  - Partição 3 (`D: memória reserva`): `446,26 GB` NTFS (`446,15 GB` livres — **vazio**, apenas `System Volume Information`).
  - Partição 4: Recovery (`0,76 GB`) — sobra de formatação anterior.

### Decisões de migração para Linux (Dual-Boot)
- **Distro recomendada:** **Ubuntu 24.04 LTS** (ou **Xubuntu 24.04 LTS** para economizar mais RAM) pelo suporte oficial prioritário de **ROCm** para GPU AMD, economia de ~2–3,5 GB de RAM em relação ao Windows 11, e toolchain Rust/Docker/Vulkan nativa.
- **Motivo adicional:** no Windows, o projeto estava em `C:\Users\lealt\OneDrive\Área de Trabalho\servermind-ai` (sincronização do OneDrive gera overhead de I/O). No Ubuntu, manter em diretório nativo (ex.: `~/projetos/servermind-ai`).
- **ALERTA CRÍTICO PARA O DUAL-BOOT:** ao instalar o Ubuntu no **Disco 1**, **NÃO apagar o disco inteiro** e **NÃO formatar a Partição 1 (`EFI/System` de 100 MB)**, pois o Windows do `C:` depende dela para dar boot. Instalar o Ubuntu apenas na **Partição 3 (`D:`, 446 GB)** mantendo a partição EFI intacta.
- **Motivo das "3 opções de Windows" no boot:** a formatação reaproveitou a partição EFI do Disco 1 mantendo entradas antigas no BCD (removíveis opcionalmente via `msconfig` -> *Inicialização do Sistema*).

---

## 3. Problemas Enfrentados no Ciclo Atual e Soluções Aplicadas

1. **Timeout de 120s e respostas travadas no Ollama (ex.: `qwen2.5:14b`):**
   - Resolvido anteriormente com streaming NDJSON (`src/ollama.rs`, `src/agent.rs`, `src/cli.rs`), `num_ctx: 4096`, `keep_alive: "15m"`, janela deslizante de 12 mensagens, ticker de espera ao vivo e base de conhecimento local (`src/knowledge.rs` + `knowledge/sysadmin_kb.json`).
2. **Incompatibilidade de frameworks Rust com GPU AMD no Windows:**
   - `Candle` não oferece suporte estável para AMD/ROCm no Windows; `Burn-wgpu` adicionaria ~250+ crates pesadas.
   - **Decisão que funcionou melhor:** motor neural em **Rust puro multi-thread** (`std::thread::scope` em `src/neural/mod.rs`) com zero dependências novas no `Cargo.toml` (`reqwest`, `rusqlite`, `serde`, `serde_json`, `toml`), mantendo compilação rápida e domínio matemático completo. Aceleração GPU (ROCm/Vulkan) fica como evolução opcional no Ubuntu.
3. **Lock de binário no Windows (`Acesso negado (os error 5)`):**
   - Quando `target\release\servermind-ai.exe` está aberto em outro terminal, `cargo build --release` falha ao tentar sobrescrevê-lo. Sempre feche sessões antigas do executável antes de recompilar (ou use `--target-dir` temporário se necessário).
4. **Gaps identificados na revisão independente e já corrigidos no Passo 1:**
   - `InferenceRouter::generate` agora repassa contexto recente + notas de KB (`generate_reply_with_context`) e usa seed dinâmica por timestamp em vez de seed fixa `1337`.
   - `ExperienceStore` ganhou interface interativa completa via `:exp stats | op | add | export`, e o gate `validate_experience_for_training` passou a bloquear também `passwd=`, `secret=`, `token=` e `-----begin`.
   - `Trainer::train` limpa `metrics.jsonl` em treinos novos (`current_step == 0`) em vez de concatenar execuções antigas, e não sobrescreve `best_val_loss` com `train_loss` quando `val_windows` está vazio.
   - `run_train_neural_cli` retorna erro explícito (código `2`) para flags numéricas inválidas ou zero em vez de cair silenciosamente no padrão.

---

## 4. Arquitetura Atual do Repositório

### Módulos do Motor Próprio (Implementados e Testados)
- `src/infrastructure/mod.rs` — Sondagem de hardware (`probe_hardware`), estimativa de RAM/VRAM (`estimate_transformer_memory`) e benchmark CPU (`benchmark_cpu_matmul`).
- `src/tokenizer/mod.rs` — `BpeTokenizer` próprio em nível de byte/UTF-8, tokens especiais (`<pad>`..`<assistant>`), normalização e `split_dataset` determinístico com deduplicação.
- `src/neural/mod.rs` — `ServerMindTransformer` (Decoder-only: embeddings, RMSNorm, Causal MHA, MLP GELU, Cross-Entropy, backpropagation analítica verificada por diferenças finitas, `AdamWState`, geração autoregressiva e checkpoints binários `SMNN`).
- `src/training/mod.rs` — `Trainer`, `pack_sequences`, warmup + cosine decay (`cosine_lr`), avaliação periódica (`evaluate_loss`), checkpoints `latest.bin`/`best.bin`, `metrics.jsonl`, `resume_from_checkpoint` e `compare_training_runs`.
- `src/experience/mod.rs` — `ExperienceStore` SQLite em 3 camadas (`operational_memory`, `experience_log`, `training_examples`) com gate de validação rigoroso e exportação JSONL.
- `src/inference/mod.rs` — `InferenceRouter` e `NeuralEngine` unificando `Ollama` (fallback) e `ServerMindNeural`.
- `src/evaluation/mod.rs` — `evaluate_neural_engine` com suíte técnica em 6 domínios (`linux`, `docker`, `network`, `security`, `debugging`, `programming`).

### Módulos Originais Preservados
- `src/main.rs`, `src/agent.rs`, `src/cli.rs`, `src/config.rs`, `src/models.rs`, `src/ollama.rs`, `src/storage.rs`, `src/guardrails.rs`, `src/knowledge.rs`, `src/eval_dataset.rs`, `src/eval_metrics.rs`, `src/eval_report.rs`, `src/eval_runner.rs`.

### Comandos Disponíveis
- **Subcomandos CLI:** `hw`, `train-neural`, `eval-neural`, `eval`, `--help`.
- **Comandos no chat interativo:** `:backend [ollama | neural [dir]]`, `:hw`, `:train [steps]`, `:exp [stats | op | add | export]`, além de todos os comandos originais (`:models`, `:history`, `:load`, `:remember`, `:kb`, `:help`, `:quit`).

### Datasets, Checkpoints e Documentação
- `datasets/servermind_seed_v1.jsonl` — 20 exemplos iniciais (usado na `v1`).
- `datasets/servermind_corpus_v2.jsonl` — **60 exemplos técnicos curados** (pronto para treinar a `v2`).
- `checkpoints/servermind-neural-v1/` — contém `tokenizer.json`, `best.bin`, `latest.bin` e `metrics.jsonl` da `v1`.
- `docs/architecture.md`, `docs/training.md`, `docs/HANDOFF.md`.

---

## 5. Estado Verificado e Métricas Reais (Baseline v1)

- **Qualidade de código:** `cargo fmt --check`, `cargo test --locked` (**119 passed; 0 failed**) e `git diff --check` passaram com código `0`.
- **Treinamento real `v1` (sobre `servermind_seed_v1.jsonl`):**
  - Parâmetros: `455.040` (`d_model=128`, `n_layers=2`, `n_heads=4`, `d_ff=256`, `seq_len=64`, `vocab=716`)
  - `120` passos: `initial_loss=6.8204 -> final_loss=2.1373` | `best_val_loss=4.7808` | **1572.6 tok/s** (`19.535 ms`)
- **Avaliação real `eval-neural` (`v1`, checkpoint `step=30`):**
  - `val_loss=4.1209` | `perplexity=61.62` | `avg_latency_ms=243.67 ms` | `keyword_recall=15.79%` (3/19)
- **Teste ao vivo (`:backend neural`):** carrega e responde sem Ollama, mas por ter apenas 455K parâmetros e 20 exemplos ainda gera texto fragmentado (`de \`docker ...`).

---

## 6. Próximos Passos Exatos ao Retomar no Ubuntu

1. **Validar ambiente Linux:**
   - Instalar `rustup` (`cargo`), `git`, `build-essential`, `sqlite3`, `vulkan-tools` e drivers ROCm/Ollama no Ubuntu 24.04 LTS.
   - Rodar `cargo test --locked` e `cargo run --release --locked -- hw`.
2. **Passo 2 — Treinar e avaliar o `ServerMind Neural v2`:**
   - Usar `datasets/servermind_corpus_v2.jsonl` (e expandir ainda mais combinando conhecimento de `knowledge/sysadmin_kb.json` e experiências reais registradas via `:exp add` / `:exp export`).
   - Escalar o modelo experimental (ex.: `d_model=256`, `n_layers=4`, `n_heads=4`, `d_ff=512`, `seq_len=96`, `vocab=1536`, ~2M–4M parâmetros) salvando em `checkpoints/servermind-neural-v2/`.
   - Rodar `eval-neural --dir checkpoints/servermind-neural-v2 --dataset datasets/servermind_corpus_v2.jsonl` e comparar objetivamente com a `v1` (`val_loss`, `perplexity`, `keyword_recall` e qualidade real das respostas).
3. **Passo 3 — Aceleração em GPU AMD RX 7600 (ROCm / Vulkan):**
   - Avaliar backend opcional de multiplicação matricial em GPU (via Vulkan/`wgpu` ou ROCm/HIP) mantendo o motor em CPU puro como fallback garantido, para viabilizar modelos de **10M–30M parâmetros**.
