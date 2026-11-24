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
.\target\release\servermind-ai.exe train-neural --dataset datasets/servermind_corpus_v2.jsonl --out checkpoints/servermind-neural-v2 --steps 180 --d-model 192 --layers 3 --heads 6 --ff 512 --seq-len 96 --vocab 1024 --lr 0.002 --min-lr 0.0002
```

### Pré-treino em diretórios de texto/código (`--corpus-dir`) e retomada (`--resume`)

Você pode combinar pré-treino em arquivos `.txt`, `.md`, `.rs`, `.py`, `.sh`, `.toml`, `.json`, `.jsonl`, `.yaml`, `.sql` e `.conf` com ajuste em Q&A:

```powershell
# Pré-treino puro em diretórios de documentação/código
.\target\release\servermind-ai.exe train-neural --corpus-dir docs --corpus-dir knowledge --out checkpoints/servermind-pretrain --steps 300 --lr 0.002

# Pré-treino + Q&A combinados
.\target\release\servermind-ai.exe train-neural --corpus-dir docs --dataset datasets/servermind_corpus_v2.jsonl --out checkpoints/servermind-neural-v3 --steps 300 --lr 0.002

# Retomar treino existente (preserva tokenizer.json + estado AdamW de latest.bin)
.\target\release\servermind-ai.exe train-neural --resume --dataset datasets/servermind_corpus_v2.jsonl --out checkpoints/servermind-neural-v2 --steps 360 --lr 0.001
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
