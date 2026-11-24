# Guia de Treinamento e Avaliação — ServerMind Neural

## 1. Preparação de Dados e Experiência

- O corpus inicial de domínio técnico fica em `datasets/servermind_seed_v1.jsonl` (Linux, Docker, Redes, Segurança, Depuração e Programação).
- Novas experiências reais podem ser registradas na **Memória em 3 Camadas** (`src/experience/mod.rs`), passando obrigatoriamente por `validate_experience_for_training()` antes de serem exportadas com `export_approved_jsonl()`.
- Pesos **nunca** são atualizados durante conversas normais: o aprendizado ocorre apenas em ciclos controlados de treinamento e validação.

## 2. Treinando do Zero via CLI

Compilar em modo release para máximo desempenho na CPU:

```powershell
cargo build --release --locked
.\target\release\servermind-ai.exe hw
.\target\release\servermind-ai.exe train-neural --steps 80 --d-model 128 --layers 2 --heads 4 --ff 256 --seq-len 64 --vocab 768
```

Artefatos gerados em `checkpoints/servermind-neural-v1/`:
- `tokenizer.json` — vocabulário e merges BPE próprios
- `best.bin` — melhor checkpoint em `val_loss`
- `latest.bin` — último checkpoint (pesos + estado AdamW para retomada)
- `metrics.jsonl` — histórico passo a passo (`train_loss`, `val_loss`, `lr`, `tokens_processed`, `elapsed_ms`)

## 3. Avaliando e Usando o Modelo Próprio

```powershell
.\target\release\servermind-ai.exe eval-neural --dir checkpoints/servermind-neural-v1
```

Dentro do chat interativo (`servermind-ai.exe`):
- `:hw` — inspeciona hardware e benchmark de matmul
- `:train 80` — executa ciclo de treinamento
- `:backend neural` — ativa o motor próprio **ServerMind Neural** (sem Ollama)
- `:backend ollama` — retorna ao fallback Ollama local
- `:exp` — exibe estatísticas da memória em 3 camadas
