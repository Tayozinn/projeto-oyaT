# SM-005 — Fundação de Desempenho em CPU (Thread Pool Persistente, GEMM/GEMV Tiled AVX2+FMA e TrainingWorkspace)

## 1. Visão Geral Arquitetural
A proposta **`SM-005` (`ETAPA 6 — PERFORMANCE FOUNDATION`)** reestrutura a base computacional de CPU do ServerMind Neural Engine em Rust puro (`100% std`, zero dependências externas), separando claramente a camada de operações numéricas de baixo nível em `src/neural/ops.rs` e eliminando realocações no loop de treinamento em `src/neural/mod.rs` e `src/training/mod.rs`.

## 2. Componentes Implementados

### 2.1. `CpuThreadPool` Persistente (`src/neural/ops.rs`)
- Inicializado sob demanda via `OnceLock<CpuThreadPool>` com `N = worker_threads()` (respeitando opcionalmente `SERVERMIND_THREADS` em `1..=64`).
- Mantém `N - 1` threads estacionadas em `Condvar` (consumo `0%` de CPU quando ociosas), enquanto a thread chamadora atua diretamente como worker `0`, eliminando as centenas/milhares de chamadas `CreateThread` por passo que ocorriam com `std::thread::scope`.
- **Segurança de memória e concorrência:**
  - `dispatch_lock` serializa despachos concorrentes (ex.: múltiplos testes paralelos em `cargo test`).
  - `parallel_disjoint_rows_mut` valida limites com aritmética verificada (`checked_mul` + `assert!`) e entrega fatias mutáveis estritamente disjuntas a cada worker.
  - `std::panic::catch_unwind` + `BgCompletionGuard` (RAII) garantem que a thread chamadora sempre aguarda `remaining == 0` antes de desempilhar o contexto, mesmo sob `panic`.

### 2.2. Kernels Matriciais e Vetoriais com Cache Blocking + AVX2/FMA (`src/neural/ops.rs`)
- Despacho dinâmico único em runtime via `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`, com fallback escalar blocado 100% seguro para qualquer CPU sem AVX2 ou arquiteturas não-`x86_64`.
- **`matmul_nn` (`C[m, n] += A[m, k] @ B[k, n]`):**
  - Tiling L1/L2 (`KC = 128`, `NC = 256`, mantendo o painel `B[KC, NC]` = `128 KiB` dentro do cache L2 privado de `256 KiB` do Core i5-9600K).
  - Micro-kernel registrador-residente **6×16** (`12` acumuladores `__m256` + `2` cargas vetoriais de `B` + `2` broadcasts de `A` = `16` registradores YMM sem spill), seguido de caudas `6×8`, `1×32`, `1×8` e escalar.
  - Caminho dedicado **GEMV (`m == 1`)** para decodificação autoregressiva com `KvCache`, acumulando faixas `1×32` em registradores YMM e paralelizando por blocos de 32 colunas quando `k * n >= 262_144`.
- **`matmul_tn` (`C[k, n] += A[m, k]^T @ B[m, n]`):**
  - Tiling sobre `(jc, mc)` (`MC = 128`, `NC = 256`) combinado com micro-kernel **6×16** registrador-residente em `C`, eliminando o *cache thrashing* em matrizes de gradiente grandes (ex.: `grads.lm_head`).
- **`matmul_nt` (`C[m, k] += A[m, n] @ B[k, n]^T`):**
  - Tiling sobre `(nc, jc)` combinado com micro-kernel **2×4** (`8` acumuladores YMM independentes + redução horizontal `hsum256_ps`), rompendo a cadeia de latência de adição escalar FP32.
- **`dot_f32` e `axpy_f32`:** primitivas vetoriais desdobradas em 4 acumuladores YMM (`32` floats/iteração) aplicadas também aos scores e valores da atenção multi-cabeça.
- **Preservação de Referência (Regra 3):** `matmul_nn_reference`, `matmul_tn_reference`, `matmul_nt_reference` e `ServerMindTransformer::forward_and_backward_reference` permanecem intactas e testadas continuamente contra os caminhos otimizados.

### 2.3. `TrainingWorkspace` Zero-Alloc (`src/neural/mod.rs` & `src/training/mod.rs`)
- Pré-aloca uma única vez por execução de treino todas as ativações de camada (`WorkspaceLayerCache`), logits, gradientes intermediários e o vetor `d_probs` da atenção reversa (que antes gerava `n_layers * n_heads * seq_len` alocações heap por amostra).
- `Trainer::train_with_progress` reutiliza o mesmo `TrainingWorkspace` em todas as amostras de todos os passos, reduzindo a zero as alocações dinâmicas por amostra nos tensores do Transformer.

## 3. Nota sobre Precisão Numérica IEEE-754 (Regra 2)
As intrínsecas `_mm256_fmadd_ps` realizam multiplicação-acumulação fundida $\text{rn}(a \cdot b + c)$ com **um único arredondamento** (contra dois arredondamentos na versão escalar) e reduzem somas parciais em árvore (`4` ou `8` faixas). Por não-associatividade de ponto flutuante, desvios de última unidade na ordem de $10^{-6}$ a $10^{-5}$ em relação à soma escalar sequencial são matematicamente esperados e validados dentro das tolerâncias estritas da suíte (`atol = 1e-4, rtol = 1e-4` para GEMM; `atol = 1e-4, rtol = 1e-3` para backprop completo e diferenças finitas).
