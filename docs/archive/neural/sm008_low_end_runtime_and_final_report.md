# SM-008 — Runtime Low-End (8 GB RAM CPU-Only), Orçamentos de Recursos e Relatório Consolidado Final (`ETAPAS 0–9`)

## 1. Objetivo da Etapa 9 (`SM-008`)
Certificar e blindar o **ServerMind Neural Engine** para operar com primeira classe em **computadores modestos (`>= 8 GB RAM`, dual/quad-core, sem GPU dedicada)**, concluindo o ciclo completo de engenharia (`ETAPA 0` a `ETAPA 9`) sobre a arquitetura **Transformer Decoder-only tradicional (`MODELO A`)** em Rust puro (`0` novas crates externas).

## 2. Componentes Implementados em `SM-008`
1. **Sondagem Nativa de RAM Física (`probe_system_ram_mb` em `src/infrastructure/mod.rs`):**
   - Windows: FFI direta para `GlobalMemoryStatusEx` (`kernel32.dll`, já vinculada pelo runtime padrão do Rust);
   - Linux: parser seguro de `MemTotal:` e `MemAvailable:` em `/proc/meminfo`;
   - Classifica automaticamente sistemas com `<= 8.704 MiB` como `RuntimeProfileKind::LowEnd8Gb`.
2. **Orçamento Preventivo Pré-Alocação (`RuntimeBudget` + `peek_checkpoint_runtime_bytes` em `src/inference/mod.rs`):**
   - Antes de alocar qualquer `Vec<f32>` ou matriz quantizada na RAM, lê apenas os metadados JSON do cabeçalho (`SMNN` ou `SMQ1`), calcula exatamente `pesos + KvCache + buffers de prefill` e bloqueia preventivamente cargas que ultrapassem `max_inference_ram_bytes` (padrão Low-End: **`2.048 MiB`**, preservando `~6 GB` livres para o SO em PCs de `8 GB RAM`);
   - Seleciona automaticamente `KvCachePrecision::Fp16` (`-50%` RAM no cache) no perfil `LowEnd8Gb`.
3. **Controle Dinâmico de Threads Ativas e Prefill Quantizado Multi-Thread (`src/neural/ops.rs` e `src/neural/quant.rs`):**
   - `set_max_active_threads` / `effective_worker_threads` permitem limitar o paralelismo do `CpuThreadPool` por execução (`--threads <N>`) sem destruir nem recriar threads;
   - `QuantMatrix::matmul_nn` agora particiona as linhas de entrada (`m >= 2`, `m * k * n >= 131_072`) entre os workers do `CpuThreadPool` com **equivalência bit-a-bit exata** em relação à execução single-thread.

## 3. Resultados Reais da Etapa 9 (`SM-008`)

### 3.1 Aceleração Multi-Thread de Prefill Quantizado (`Medium_13.9M`, `T=128`, i5-9600K `1 -> 6` threads)
| Formato `SMQ1` | Prefill `1 Thread` | Prefill `6 Threads` | Speedup Prefill | Footprint Total (`Pesos + KV FP16 + Prefill`) | Uso do Orçamento `LowEnd8Gb` (`2.048 MB`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **`FP16`** | `478,27 ms` | `163,60 ms` | **`2,92x`** | `33,45 MB` | `1,63%` |
| **`BF16`** | `249,24 ms` | `98,18 ms` | **`2,54x`** | `33,45 MB` | `1,63%` |
| **`INT8`** | `160,79 ms` | **`69,45 ms`** | **`2,32x`** | **`20,43 MB`** | **`1,00%`** |
| **`INT4` (G=32)** | `281,87 ms` | `129,99 ms` | **`2,17x`** | **`14,59 MB`** | **`0,71%`** |

### 3.2 Capacidade Máxima Certificada dentro do Teto Estrito de `2.048 MB` (PCs de `8 GB RAM` sem GPU)
| Escala do Modelo (`T=512`, `KV FP16`) | `FP32` | `SMQ1 FP16/BF16` | `SMQ1 INT8` | `SMQ1 INT4` (G=32) | Cabe no Orçamento `2.048 MB` (`8 GB RAM`)? |
| --- | ---: | ---: | ---: | ---: | --- |
| **`13,9M` params** (`d=384, L=6`) | `66 MB` | `38 MB` | **`25 MB`** | **`19 MB`** | **Sim (todos os formatos, `< 2%` do orçamento)** |
| **`125M` params** (`d=768, L=12`) | `518 MB` | `268 MB` | **`145 MB`** | **`88 MB`** | **Sim (todos os formatos, `INT8` usa apenas `7%`)** |
| **`350M` params** (`d=1024, L=24`) | `1.412 MB` | `732 MB` | **`395 MB`** | **`236 MB`** | **Sim (`INT8`/`INT4` com ampla margem)** |
| **`1,3B` params** (`d=2048, L=24`) | `5.150 MB` *(Bloqueado)* | `2.650 MB` *(Bloqueado)* | **`1.410 MB`** | **`835 MB`** | **Sim em `SMQ1 INT8` (`69%`) e `SMQ1 INT4` (`41%`)** |

---

## 4. Panorama Consolidado do Projeto (`ETAPA 0` -> `ETAPA 9`)

| Etapa | Proposta | Marco Principal Entregue e Medido |
| :--- | :--- | :--- |
| **Etapa 0** | Auditoria | Auditoria completa pelos 7 agentes especialistas sobre matemática, BPE, memória, CPU e RX 7600 |
| **Etapas 1–2** | `SM-001` | Cross-Entropy exata em `f64` (`log-sum-exp`), GELU seguro, guarda `NaN`/`Inf` no AdamW e teste de diferenças finitas nos 14 grupos de tensores |
| **Etapa 3** | `SM-002` | BPE `encode` $O(m \cdot L)$ (**`17,58x` mais rápido**: `94,17 s -> 5,36 s`), isolamento de `SPECIAL_TOKENS` e validação determinística |
| **Etapa 4** | `SM-003` | `KvCache` incremental + projeção `lm_head` apenas no último token (**`30,03x` speedup** em geração) + streaming UTF-8 seguro |
| **Etapa 5** | `SM-004` | Formato `SMNN` v2 com escrita atômica (`temp + sync_all + rename`), trailer `CRC32` IEEE 802.3, retrocompatibilidade v1 e `new_zeros` |
| **Etapa 6** | `SM-005` | `CpuThreadPool` persistente + micro-kernels `AVX2+FMA` (`matmul_nt` **`141,11x`**, passo `1,45M` **`11,63x`**) + `TrainingWorkspace` zero-alloc |
| **Etapa 7** | `SM-006` | Compilador SPIR-V 1.3 em Rust puro + backend Vulkan Compute dinâmico na **AMD RX 7600** (**`2.069 GFLOP/s`**, **`7,19x`** sobre CPU AVX2 em `13,9M`) |
| **Etapa 8** | `SM-007` | Quantização `FP16`/`BF16`/`INT8`/`INT4` + formato `SMQ1` (**`11,31x`** menor em `INT8`, **`19,52x`** em `INT4`, **`4.557 tok/s`** decode) + `KvCachePrecision` |
| **Etapa 9** | `SM-008` | Sondagem nativa de RAM do host, orçamento preventivo `RuntimeBudget` (`LowEnd8Gb`), `--threads`, prefill `SMQ1` multi-thread (**`2,92x`**) e certificação final |
