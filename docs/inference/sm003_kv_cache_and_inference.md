# Motor de Inferência, KV Cache e Avaliação Determinística — `SM-003` (Etapa 4)

## 1. Fundamento Matemático do `KvCache`

Em um Transformer Decoder-only com máscara causal estrita ($j \le i$), as projeções $K_t^{(l)}$ e $V_t^{(l)}$ na camada $l$ e posição $t$ dependem **exclusivamente** de $x_{0..=t}$. Logo, ao gerar um novo token na posição $t_{\text{novo}}$, todos os vetores $K_{0..t_{\text{novo}}-1}^{(l)}$ e $V_{0..t_{\text{novo}}-1}^{(l)}$ permanecem invariantes.

- **Sem `KvCache` (`forward_logits` / `generate_reference`):**
  - Complexidade para gerar $N$ tokens até $T$: $O(T^3 d + T^2 d^2 + T^2 d V)$.
  - Alocação temporária de logits a cada passo: $T \cdot V \cdot 4$ bytes (descartando $T - 1$ linhas).
- **Com `KvCache` (`forward_last_logits_cached` / `generate_with_options`):**
  - Prefill inicial `[prompt_len, d]`, seguido de passos unitários `[1, d]`.
  - Projeção `lm_head` calculada **apenas na última posição** (`[1, d] @ [d, V]`).
  - Complexidade acumulada: $O(T^2 d + T d^2 + T d V)$.
  - Memória estática do `KvCache`: $2 \cdot L \cdot T_{\max} \cdot d \cdot 4$ bytes (`0,42 MB` para `1,45M` params; `4,50 MB` para `13,88M` params).

Conforme a **Regra 3**, `forward_logits` e `generate_reference` são preservados intactos para auditoria contínua de equivalência numérica (`max |diff| < 1e-4` nos logits e igualdade exata nas sequências geradas).

## 2. Melhorias de Amostragem, Prompt e Streaming

1. **`sample_top_k` em $O(V)$ médio:** usa `select_nth_unstable_by` para particionar os $k$ maiores logits finitos antes de ordenar apenas os $k$ finalistas ($O(V + k \log k)$), preservando determinismo por seed.
2. **Alinhamento de prompt e `stop_tokens` (`src/inference/mod.rs`):**
   - Notas `<system>` posicionadas no topo (antes dos turnos `<user>`/`<assistant>`).
   - Truncamento de contexto preserva `<bos>` no índice `0`.
   - Parada imediata em `<eos>`, `<user>`, `<system>` ou `<assistant>` + streaming incremental seguro em fronteiras UTF-8 via `on_chunk`.
   - Validação obrigatória `tokenizer.vocab_size() == model.config.vocab_size` em `NeuralEngine::load_from_dir`.
3. **Avaliação `eval-neural` determinística (`src/evaluation/mod.rs`):**
   - Seed fixa por caso (`GenerationConfig`), eliminando variação aleatória entre execuções do mesmo checkpoint.
   - `contains_keyword_boundary` substitui `contains` bruto, eliminando falsos positivos como `"ss"` em `"processo"`/`"passos"`.
   - Métrica explícita `tokens_per_sec` e `generated_tokens` adicionadas ao relatório.
