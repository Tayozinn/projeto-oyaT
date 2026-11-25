# Pipeline de Treinamento e Dados — Implementação `SM-002` (Etapa 3)

## 1. Objetivos e Problemas Resolvidos

1. **Divisão determinística sem viés de ordenação (`split_dataset` / `try_split_dataset`):**
   - Antes de `SM-002`, `split_dataset` fatiava o vetor na ordem de entrada (`..train_end`, `train_end..val_end`). Ao combinar `--corpus-dir` (ordenado alfabeticamente) com `--dataset` (anexado ao final), 100% dos exemplos Q&A caíam em validação/teste.
   - Agora, após deduplicação canônica, aplica-se um embaralhamento Fisher-Yates determinístico (`SplitMix64` ancorado em hash FNV-1a de `version`), garantindo mistura uniforme entre livros e Q&A (~75% treino / 15% validação / 10% teste) com reprodutibilidade exata.

2. **Isolamento de `SPECIAL_TOKENS` e Índice `merge_ranks` no BPE (`src/tokenizer/mod.rs`):**
   - `BpeTokenizer::train_from_corpus` agora isola `SPECIAL_TOKENS` antes de `pretokenize`, impedindo que slots de merge sejam desperdiçados fundindo `<user>`, `<assistant>` ou `<system>`.
   - `BpeTokenizer::encode` normaliza `\r\n` -> `\n` e substitui a varredura linear $O(|M| \cdot L)$ com realocações em massa por busca de par de menor rank em `merge_ranks` ($O(m \cdot L)$, matematicamente idêntica por indução sobre a ordem de criação dos tokens no BPE) com compactação in-place (`merge_pair_in_place`).

3. **Empacotamento sem duplicação de cauda (`pack_sequences`, `pack_eval_sequences`):**
   - Corrigida a condição `tail_start > last_start` para impedir duplicação exata da última janela quando `(stream.len() - window_len) % stride == 0`.
   - `pack_eval_sequences` usa `stride = seq_len` (janelas disjuntas), evitando dupla contagem de tokens centrais na validação.

4. **Continuidade de `--resume`, Limpeza de Checkpoints e Telemetria (`src/training/mod.rs`, `src/cli.rs`):**
   - Em `--resume`: `warmup_steps = 0` (evita salto de reentrada em warmup), validação explícita `tokenizer.vocab_size() == model.config.vocab_size`, validação `max_steps > current_step` e `min_lr <= lr`.
   - Em treino novo (`current_step == 0`): remove `metrics.jsonl`, `latest.bin` e `best.bin` antigos do diretório alvo.
   - `TrainStepMetrics` agora registra `grad_norm` (norma L2 pré-clipping) em `metrics.jsonl` e emite progresso ao vivo por `train_with_progress`.

5. **Estimativa Exata de Memória (`src/infrastructure/mod.rs`):**
   - `estimate_transformer_memory_with_heads` incorpora vieses `b1`/`b2` (100% igual a `TransformerConfig::total_params()`), `LayerCache` completo das $L$ camadas incluindo `attn_probs` ($LHT^2$), temporários do backward ($2TV + 14Td + 2Td_{\text{ff}} + T$), estimativa de inferência FP32 com KV Cache e flags `fits_8gb_ram_inference` e `fits_8gb_vram_rx7600`.
