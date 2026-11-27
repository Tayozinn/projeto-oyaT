# Guia e Baseline de Backends (`ModelBackend` — Fase 3)

O **ServerMind AI** utiliza a trait `ModelBackend` (`src/chat/backend.rs`) para desacoplar o núcleo conversacional (`AgentSession`) e o avaliador offline (`eval run`) do motor de inferência subjacente.

Por padrão, o binário continua **100% local e privado via `ollama`** (restrito a loopback `127.0.0.1` / `localhost` / `[::1]`). Backends remotos só entram em ação mediante ativação explícita do operador.

---

## 1. Backends Suportados e Presets Gratuitos Prontos

| Backend (`--backend` / `:backend`) | Escopo | Variável de Ambiente (API Key) | Modelo Padrão | Observações |
| --- | --- | --- | --- | --- |
| `ollama` *(padrão)* | Local (loopback) | *(nenhuma)* | `config.toml` (`qwen2.5:14b` / `qwen2.5:3b`) | Regra estrita anti-SSRF (somente loopback). |
| `neural` | Local (in-process) | *(nenhuma)* | `checkpoints/servermind-neural-110m-int8` | Motor próprio Rust (`FP32` ou `SMQ1 INT8`). |
| `claude` | Remoto (HTTPS) | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` | API oficial Anthropic Messages (`/v1/messages`) + `tools` nativo. |
| `gemini` | Remoto (HTTPS) | `GEMINI_API_KEY` | `gemini-flash-latest` | Google AI Studio (**possui tier gratuito sem cartão**). |
| `groq` | Remoto (HTTPS) | `GROQ_API_KEY` | `openai/gpt-oss-120b` | Groq Cloud (**possui tier gratuito em baixa latência**). |
| `mistral` | Remoto (HTTPS) | `MISTRAL_API_KEY` | `mistral-small-latest` | Mistral La Plateforme (**Free Mode sem cartão**). |
| `openrouter` | Remoto (HTTPS) | `OPENROUTER_API_KEY` | `openrouter/free` | Auto-router gratuito do OpenRouter (**roteia automaticamente para `:free`**). |
| `openai` | Remoto (HTTPS) | `OPENAI_API_KEY` | `gpt-4o-mini` | Compatível com OpenAI `/v1/chat/completions` (`SERVERMIND_OPENAI_BASE_URL`). |

### Atalhos de Presets Gratuitos (`:backend presets`)

Você pode listar todos os atalhos e verificar quais variáveis já estão definidas digitando `:backend presets` no REPL, ou ativar diretamente qualquer um destes aliases tanto no REPL quanto em `eval run --backend <alias>`:

| Alias Pronto | Provedor | Modelo Pré-configurado | Variável de Ambiente | Onde gerar chave gratuita |
| --- | --- | --- | --- | --- |
| `free-gemini` | Google AI Studio | `gemini-flash-latest` | `GEMINI_API_KEY` | `https://aistudio.google.com/apikey` |
| `free-groq` | Groq Cloud | `openai/gpt-oss-120b` | `GROQ_API_KEY` | `https://console.groq.com/keys` |
| `free-groq-qwen` | Groq Cloud | `qwen/qwen3.8-27b` | `GROQ_API_KEY` | `https://console.groq.com/keys` |
| `free-mistral` | Mistral | `mistral-small-latest` | `MISTRAL_API_KEY` | `https://console.mistral.ai/api-keys` |
| `free-openrouter` | OpenRouter | `openrouter/free` | `OPENROUTER_API_KEY` | `https://openrouter.ai/keys` |
| `free-nemotron` | OpenRouter | `nvidia/nemotron-3-ultra:free` | `OPENROUTER_API_KEY` | `https://openrouter.ai/keys` |

---

## 2. Garantias de Privacidade e Limites Remotos (`C2`, `C3`, `C4`)

1. **Chaves exclusivamente em variáveis de ambiente:** nenhuma chave de API é lida ou persistida em `config.toml` nem em `servermind.db`.
2. **Redação automática antes do envio (`src/chat/redaction.rs`):** antes de qualquer requisição sair para `claude`, `gemini`, `groq`, `openrouter` ou `openai`, todo o histórico passa por `redact_messages_for_remote`, mascarando:
   - Blocos PEM (`-----BEGIN ... PRIVATE KEY-----`);
   - Credenciais em URIs (`postgres://user:senha@host` -> `postgres://user:[SENHA_REDIGIDA]@host`);
   - Cabeçalhos `Authorization: Bearer ...` e tokens JWT;
   - Prefixos conhecidos (`sk-ant-`, `sk-proj-`, `sk-`, `ghp_`, `github_pat_`, `glpat-`, `xoxb-`, `gsk_`, `AIza`, `AKIA`);
   - Atribuições `password=...`, `secret=...`, `api_key=...`, `token=...` e flags `-p<segredo>`.
3. **Banner explícito e prompt marcado:** ao ativar qualquer backend remoto no REPL, o terminal exibe `[AVISO DE BACKEND REMOTO ATIVO]` e marca o prompt com `[REMOTO]`.
4. **Limites de custo e contexto por chamada:**
   - `SERVERMIND_REMOTE_MAX_TOKENS` (padrão: `1024`);
   - `SERVERMIND_REMOTE_MAX_HISTORY` (padrão: `10` últimas mensagens não-sistema + prompt de sistema);
   - Timeout HTTP explícito (padrão: `60s`).

---

## 3. Como Usar no REPL e na Avaliação Offline

### No terminal interativo (REPL)

```powershell
# Exemplo 1: Testar gratuitamente com Google Gemini (AI Studio)
$env:GEMINI_API_KEY = "AIza..."
cargo run --release
# Dentro do REPL:
# :backend gemini
# :backend ollama

# Exemplo 2: Usar Claude Sonnet oficial
$env:ANTHROPIC_API_KEY = "sk-ant-..."
cargo run --release
# Dentro do REPL:
# :backend claude
# :backend claude claude-3-5-haiku-latest
```

### Na suíte `eval run` (`C5`)

```powershell
# Local — Ollama
cargo run --release -- eval run --backend ollama --model qwen2.5:3b

# Local — Motor Neural próprio (INT8)
cargo run --release -- eval run --backend neural --model checkpoints/servermind-neural-110m-int8

# Remoto gratuito — Gemini / Groq / OpenRouter
$env:GEMINI_API_KEY = "AIza..."
cargo run --release -- eval run --backend gemini --limit 6

# Remoto pago — Claude Sonnet
$env:ANTHROPIC_API_KEY = "sk-ant-..."
cargo run --release -- eval run --backend claude --limit 6
```

---

## 4. Baseline Comparativo Registrado (`C5`)

Execução realizada em `2026-11-27` sobre `eval/cases_v1.json` (`v1.1.0`, SHA-256 `614a03aa18cb79fbb76a3f766436c02e3cb103552d20aab7cf77af0f9b869619`, subconjunto inicial determinístico `--limit 6`: `exp-01`, `exp-02`, `exp-03`, `dgs-01`, `dgs-02`, `dgi-01`):

| Backend | Modelo / Checkpoint | Casos OK | Média Determinística (`0..12`) | Latência Média / Caso | Revisão Humana Sinalizada |
| --- | --- | --- | --- | --- | --- |
| `ollama` (local) | `qwen2.5:3b` (`Q4_K_M`, 1.80 GiB) | `6/6` | **`12.00 / 12`** | `7 782 ms` | `0/6` |
| `neural` (local) | `servermind-neural-110m-int8` (`SMQ1 INT8`) | `6/6` | **`5.33 / 12`** | `1 453 ms` | `6/6` |
| `neural` (local) | `servermind-neural-v3` (`FP32`, 4.8M params) | `6/6` | **`4.33 / 12`** | `276 ms` | `6/6` |
| `claude` / `gemini` / `groq` | *(sob demanda com API key no ambiente)* | Validado em mock E2E (`test_phase3_c5_eval_run_supports_remote_claude_backend`) | Prontos para comparação direta com `eval run --backend <nome>` | Depende da rede/provedor | Automático |

### Conclusões técnicas do baseline

- O pipeline `eval run` agora avalia **sob a mesma régua determinística** modelos locais no Ollama, checkpoints neurais próprios e APIs remotas (`Claude`, `Gemini`, `Groq`, `OpenRouter`, `OpenAI`).
- O salto do checkpoint próprio `v3` (`4.33/12`) para o `110m-int8` (`5.33/12`) confirma ganho mensurável de escala/quantização, mas a distância para um LLM de `3B` instruído (`qwen2.5:3b` com `12.00/12`) evidencia na prática por que separar **cérebro trocável (`ModelBackend`)** das **ferramentas reais de diagnóstico (Fase 4)** é a decisão arquitetural correta para tornar o ServerMind útil em operação real.
