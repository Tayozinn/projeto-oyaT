# Checkpoint Engine (`SMNN` v2) — Implementação `SM-004` (Etapa 5)

## 1. Formato Binário `SMNN` v2 e Retrocompatibilidade v1

| Offset / Seção | Tamanho | Descrição |
| --- | --- | --- |
| `magic` | `4 bytes` | `"SMNN"` |
| `version` | `u32 LE` | `2` na escrita nova (`1` aceito na leitura retrocompatível) |
| `meta_len` | `u32 LE` | Tamanho do JSON de metadados (`1 ..= 1_048_576` bytes) |
| `meta_json` | `meta_len` | `CheckpointMetadata` (`config`, `step`, `best_val_loss`, `has_optimizer`, `optimizer_step`) |
| `weights` | `4 * P` | Parâmetros `f32 LE` em fatias canônicas (`visit_param_slices`) |
| `adamw_state` | `8 * P` *(opc.)* | Vetores `m` (`4 * P`) e `v` (`4 * P`) quando `has_optimizer == true` |
| `crc32` *(v2)* | `u32 LE` | Checksum CRC32 (IEEE 802.3, polinômio `0xEDB88320`) cobrindo desde `magic` até o fim do payload |

## 2. Garantias de Integridade, Segurança e Atomicidade

1. **Escrita Atômica (`save_checkpoint` e `BpeTokenizer::save`):**
   - Gravação completa em arquivo temporário oculto no mesmo diretório (`.<nome>.tmp.<pid>.<seq>`), seguida de `flush()`, `sync_all()`, fechamento do descritor e `fs::rename` com até 3 tentativas (`15 ms` de espera) para tolerar antivírus/indexadores no Windows sem jamais truncar um checkpoint válido anterior em caso de `Ctrl+C` ou falha de I/O.
2. **Defesa contra OOM, Truncamento e Corrupção Silenciosa (`load_checkpoint`):**
   - Limite estrito `MAX_CHECKPOINT_META_BYTES = 1 MiB` antes de alocar `meta_bytes`.
   - `TransformerConfig::checked_total_params()` previne overflow aritmético em metadados malformados.
   - Validação exata de `file_len == expected_file_len` **antes** de alocar tensores do modelo/otimizador (detecta truncamento e bytes extras imediatamente).
   - Verificação `val.is_finite()` na leitura e na escrita de todos os `f32` (pesos + `m` + `v`).
   - Verificação de CRC32 ponta a ponta em arquivos `SMNN` v2.
3. **Desempenho de I/O e Alocação:**
   - `ServerMindTransformer::new_zeros` elimina a geração descartada de normais Box-Muller durante o load.
   - Leitura/escrita em blocos de `16 KiB` (`4096` floats) sobre `BufReader`/`BufWriter` de `256 KiB`.
