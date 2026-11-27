# Auditoria Matemática (Etapa 0) e Implementação `SM-001`

## 1. Escopo e Arquitetura Auditada

Arquitetura oficial preservada: **MODELO A — Transformer Decoder-only Tradicional** em Rust puro (`FP32`).

Fluxo analítico verificado linha a linha em `src/neural/mod.rs` e `src/training/mod.rs`:

1. **Embeddings (`[T, d_model]`):**
   $$x_{p,i} = E_{\text{tok}}[\text{id}_p, i] + E_{\text{pos}}[p, i]$$
   No backward, $\frac{\partial \mathcal{L}}{\partial E_{\text{tok}}[k]} = \sum_{p:\,\text{id}_p = k} \frac{\partial \mathcal{L}}{\partial x_p}$ e $\frac{\partial \mathcal{L}}{\partial E_{\text{pos}}[p]} = \frac{\partial \mathcal{L}}{\partial x_p}$.

2. **RMSNorm (`eps = 1e-5`):**
   $$r_p = \left(\frac{1}{d}\sum_{j=0}^{d-1} x_{p,j}^2 + \epsilon\right)^{-1/2}, \qquad y_{p,i} = \gamma_i \, x_{p,i} \, r_p$$
   Gradientes analíticos:
   $$\frac{\partial \mathcal{L}}{\partial \gamma_i} = \sum_p g_{p,i} \, x_{p,i} \, r_p, \qquad \frac{\partial \mathcal{L}}{\partial x_{p,k}} = g_{p,k}\gamma_k r_p - x_{p,k} \frac{r_p^3}{d}\sum_{i=0}^{d-1} g_{p,i}\gamma_i x_{p,i}$$

3. **Multi-Head Causal Self-Attention:**
   - Projeções $Q = \text{LN}_1 W_Q$, $K = \text{LN}_1 W_K$, $V = \text{LN}_1 W_V$, escala $s = d_h^{-1/2}$.
   - Máscara causal $j \in [0, i]$ e Softmax estável $P_{i,j} = \frac{\exp(S_{i,j} - \max_{k \le i} S_{i,k})}{\sum_{k \le i} \exp(S_{i,k} - \max_{m \le i} S_{i,m})}$.
   - Backward do Softmax: $dS_{i,j} = P_{i,j}\left(dP_{i,j} - \sum_{k \le i} P_{i,k} dP_{i,k}\right) \cdot s$.

4. **MLP com GELU aproximada (`tanh`):**
   $$u(x) = \sqrt{\frac{2}{\pi}}(x + 0{,}044715 x^3), \qquad \text{GELU}(x) = \frac{1}{2}x(1 + \tanh u(x))$$
   $$\text{GELU}'(x) = \frac{1}{2}(1 + \tanh u) + \frac{1}{2}x(1 - \tanh^2 u)\sqrt{\frac{2}{\pi}}(1 + 3 \cdot 0{,}044715 x^2)$$
   Com saturação exata em $|x| \ge 10{,}0$ ($\tanh(\pm 43{,}656) = \pm 1{,}0$ em `f32`) para impedir indeterminação `0.0 * inf = NaN`.

5. **Cross-Entropy via Log-Sum-Exp em `f64`:**
   $$\mathcal{L}_p = -\left(z_{p, y_p} - m_p - \ln \sum_{j=0}^{V-1} \exp(z_{p,j} - m_p)\right), \qquad \frac{\partial \mathcal{L}_p}{\partial z_{p,j}} = P_{p,j} - \mathbf{1}_{[j = y_p]}$$
   Unificada entre `forward_and_backward` e `evaluate_loss`, eliminando o clamp artificial `1e-12` (`27.63`) anterior.

---

## 2. Correções Implementadas em `SM-001`

| Item | Problema na Etapa 0 | Correção em `SM-001` |
| --- | --- | --- |
| `generate` | Underflow `(ctx.len() - 1) * v` com `prompt_ids` vazio | Retorno imediato seguro quando `prompt_ids.is_empty()`, `max_new_tokens == 0` ou `vocab_size == 0`. |
| `evaluate_loss` | Panic OOB quando `win.len() - 1 > max_seq_len` | Truncamento explícito `t_eff = (logits.len() / vocab).min(targets.len())`. |
| `sample_top_k` / Greedy | `f32::total_cmp` selecionava `+NaN` e `l / temperature` causava overflow para `T -> 0+` | Filtro `is_finite()`, greedy para `temperature <= 1e-6` e cálculo `((l - max_raw) / temperature).exp()` sem risco de `+Inf`. |
| `AdamWState::step` | `NaN` ignorava `norm > max_grad_norm` e corrompia pesos/`m`/`v` | Checagem `norm.is_finite()`; retorna `false` preservando pesos e otimizador intactos. |
| `Trainer::train` | Executava `optimizer.step()` mesmo com `step_tokens == 0` | Pula atualização do otimizador quando `step_tokens == 0` e aborta com erro explícito em perda/gradiente não-finito. |
| `CheckpointMetadata` | `best_val_loss = f32::INFINITY` virava `null` no JSON e quebrava `load_checkpoint` | Módulo `serde_loss_f32` compatível com checkpoints `v1`–`v3` existentes (`null <-> f32::INFINITY`). |
| `TransformerConfig` | Aceitava `d_ff == 0` | Validação `d_ff > 0` em `TransformerConfig::validate()`. |

---

## 3. Critérios de Verificação Automatizada (`cargo test --locked`)

- **Diferenças finitas multi-camada (`n_layers = 2`):** verifica os 14 grupos de parâmetros (`token_emb` com repetição, `pos_emb`, `ln1_gamma`, `wq`, `wk`, `wv`, `wo`, `ln2_gamma`, `w1`, `b1`, `w2`, `b2` nas camadas 0 e 1, `ln_f_gamma` e `lm_head`) com `pad_id` e alvo OOB sob tolerância combinada `|a - n| <= 1e-3 + 2e-2 * max(|a|, |n|)`.
- **Equivalência Multi-Thread vs. Single-Thread:** valida `matmul_nn`, `matmul_tn` e `matmul_nt` com `work = 144.000 >= 65.536` e dimensões não divisíveis pelo número de threads.
- **Imunidade a `NaN`/`Inf` e Round-trip de `INFINITY`:** testes dedicados em `neural::tests` e `training::tests`.
