# SM-007 — Otimização de Memória, Quantização Pós-Treino (`FP16` / `BF16` / `INT8` / `INT4`) e Formato `SMQ1`

## 1. Objetivo Arquitetural
Reduzir drasticamente o consumo de RAM e armazenamento de modelos treinados pelo **ServerMind Neural Engine** para distribuição e inferência em computadores modestos (`>= 8 GB RAM`, sem GPU dedicada), sem alterar a arquitetura **Transformer Decoder-only tradicional (`MODELO A`)** e sem introduzir nenhuma crate externa.

## 2. Estratégia de Precisão Mista de Inferência
Para evitar instabilidades numéricas em normalização e deslocamentos residuais, `QuantizedModelData` aplica quantização seletiva:
- **Mantidos em `FP32` exato (`< 0,5%` dos parâmetros em escala 14M+):**
  - `pos_emb` (`[max_seq_len, d_model]`)
  - `ln1_gamma`, `ln2_gamma` e `ln_f_gamma` (`RMSNorm`)
  - `b1` e `b2` (biases da MLP)
- **Comprimidos em `QuantMatrix` (com `Vec<f32>` originais esvaziados para `0` bytes na RAM):**
  - `token_emb` (`[vocab_size, d_model]`, desquantizado sob demanda em $O(d_{\text{model}})$ apenas para os tokens consultados)
  - Projeções de atenção `wq`, `wk`, `wv`, `wo` (`[d_model, d_model]`)
  - Projeções MLP `w1` (`[d_model, d_ff]`) e `w2` (`[d_ff, d_model]`)
  - Cabeça de linguagem `lm_head` (`[d_model, vocab_size]`)

### Formatos Suportados (`WeightPrecision`)
1. **`Fp16` (IEEE 754 `binary16`, `2` bytes/peso):** conversor escalar bit-exato *round-to-nearest-even* com tratamento de subnormais e saturação finita segura em `±65504.0` (jamais introduz `Inf`/`NaN`), decodificado em registradores `AVX2` (`8` floats por instrução).
2. **`Bf16` (`Brain Float 16`, `2` bytes/peso):** *round-to-nearest-even* no bit 16 com saturação finita, decodificado via shift vetorial de 1 ciclo (`_mm256_cvtepu16_epi32` + `_mm256_slli_epi32::<16>`) + `_mm256_fmadd_ps`.
3. **`Int8` (Simétrico por linha/canal, `~1,01` byte/peso):**
   $$s_r = \frac{\max_c |W_{r,c}|}{127}, \qquad q_{r,c} = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{W_{r,c}}{s_r}\right), -127, 127\right)$$
   Kernel `gemv_int8_avx2` expande `32` pesos `i8` por iteração (`_mm256_cvtepi8_epi32` + `_mm256_cvtepi32_ps` + FMA) pré-multiplicando `alpha = a[k] * scale[k]`.
4. **`Int4` (Simétrico por grupo de `G = 32` colunas, `~0,56` byte/peso):**
   - Escala por bloco armazenada em `FP16` (`2` bytes a cada `32` pesos = `+0,0625` byte/peso);
   - Dois pesos empacotados por byte (`q in [-7, +7]`, codificados com offset `+8` em nibbles low/high);
   - Kernel `gemv_int4_avx2` desempacota `16` bytes (`32` nibbles) em registradores `XMM`/`YMM` via `_mm_and_si128`, `_mm_srli_epi16::<4>` e `_mm_unpacklo_epi8`/`_mm_unpackhi_epi8`.

### KV Cache Configurável (`KvCachePrecision`)
- **`Fp32` (padrão):** $8 \cdot L \cdot T \cdot d$ bytes;
- **`Fp16`:** $4 \cdot L \cdot T \cdot d$ bytes (**redução exata de `50,0%`**);
- **`Int8`:** $2 \cdot L \cdot T \cdot d + 8 \cdot L \cdot T$ bytes (escala simétrica por posição/camada, **redução de `~74,7%`** em `d=384`).

## 3. Formato Binário Compacto de Distribuição `SMQ1`
- Cabeçalho: `magic = b"SMQ1"` (4B) + `version = 1u32` (4B) + `meta_len: u32` (4B) + JSON `QuantizedCheckpointMetadata`.
- Payload: tensores `FP32` pequenos preservados seguidos das matrizes quantizadas compactas (sem estados AdamW `m`/`v`).
- Integridade e Atomicidade: trailer `CRC32` IEEE 802.3 sobre todo o arquivo + verificação de EOF exato + gravação atômica `temp + flush + sync_all + rename`.
- Carregamento automático em `NeuralEngine::load_from_dir`: detecta `best.bin`, `quantized.smq` ou `latest.bin`.

## 4. Resultados Experimentais Reais (i5-9600K `AVX2+FMA`)

### 4.1 Checkpoint Real Treinado (`checkpoints/servermind-neural-v2`, `1,45M` params, `step=180`)
| Formato | Disco (`best.bin` / `.smq`) | Redução Disco | RAM Pesos | Redução RAM | `val_loss` | `Δ loss` | Perplexidade | Decode (`tok/s`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`FP32` (`SMNN` v2 c/ AdamW)** | `16,563 MB` | `1,00x` | `5,521 MB` | `1,00x` | `3,9379` | `0,0000` | `51,31` | `1.689,3 tok/s` |
| **`SMQ1 FP16`** | `2,802 MB` | **`5,91x`** | `2,802 MB` | **`1,97x`** | `3,9379` | `-0,0000` | `51,31` | `1.138,4 tok/s` |
| **`SMQ1 BF16`** | `2,802 MB` | **`5,91x`** | `2,802 MB` | **`1,97x`** | `3,9379` | `+0,0000` | `51,31` | **`2.849,7 tok/s`** |
| **`SMQ1 INT8`** | `1,465 MB` | **`11,31x`** | `1,464 MB` | **`3,77x`** | `3,9380` | `+0,0001` | `51,31` | **`4.557,0 tok/s`** |
| **`SMQ1 INT4` (G=32)** | `0,848 MB` | **`19,52x`** | `0,848 MB` | **`6,51x`** | `3,9414` | `+0,0035` | `51,49` | **`3.047,8 tok/s`** |

### 4.2 Escala `Medium_13.9M` (`13,88M` params, `V=4096, T=256, d=384, H=6, L=6, ff=1536`)
| Configuração | RAM Pesos | RAM `KvCache` | Redução RAM Pesos | Decode (`tok/s`) |
| --- | ---: | ---: | ---: | ---: |
| `FP32` + `KV FP32` | `52,94 MB` | `4,500 MB` | `1,00x` | `310,0 tok/s` |
| `FP32` + `KV FP16` | `52,94 MB` | `2,250 MB` (`-50,0%`) | `1,00x` | `292,5 tok/s` |
| `FP32` + `KV INT8` | `52,94 MB` | `1,137 MB` (`-74,7%`) | `1,00x` | `304,0 tok/s` |
| `SMQ1 FP16` + `KV FP16` | `26,69 MB` | `2,250 MB` | **`1,98x`** | `155,8 tok/s` |
| `SMQ1 BF16` + `KV FP16` | `26,69 MB` | `2,250 MB` | **`1,98x`** | `277,1 tok/s` |
| **`SMQ1 INT8` + `KV FP16`** | **`13,66 MB`** | **`2,250 MB`** | **`3,88x`** | **`380,9 tok/s` (`+22,9%` vs FP32)** |
| **`SMQ1 INT4` + `KV FP16`** | **`7,82 MB`** | **`2,250 MB`** | **`6,77x`** | `253,8 tok/s` |

## 5. Tradeoffs Documentados (Regra 2)
- **`INT8` + `KV FP16` (Recomendação Padrão de Distribuição):** entrega simultaneamente **`3,88x` menos RAM de pesos**, **`11,31x` menor arquivo em disco**, perda numérica desprezível (`Δval_loss = +0,0001`) e **maior velocidade de decodificação em CPU** (`+22,9%` em `13,9M` e `2,70x` em `1,45M`) graças à menor pressão sobre banda de memória L2/L3/DRAM e conversão direta `_mm256_cvtepi8_epi32`.
- **`BF16` vs `FP16` sem instrução `F16C` dedicada:** como a decodificação `FP16` IEEE-754 completa (incluindo subnormais) é feita via emulação inteira `AVX2`, `BF16` decodifica `~1,8x` mais rápido em CPU com o mesmo tamanho (`2` bytes/peso), enquanto `FP16` preserva `10` bits de mantissa contra `7` bits do `BF16`.
- **`INT4` (Grupo 32):** atinge a máxima compressão (**`6,77x` menos RAM** e **`19,52x` menor em disco**), ideal para modelos maiores (`100M–500M` params) em máquinas de `8 GB RAM`, com leve aumento esperado de perplexidade (`51,31 -> 51,49`, `Δval_loss = +0,0035`).
- **Escopo estritamente de inferência:** modelos carregados de `.smq` bloqueiam explicitamente `save_checkpoint` FP32, `forward_and_backward` e `AdamWState::step`. O treinamento permanece em `FP32` (CPU/GPU).
