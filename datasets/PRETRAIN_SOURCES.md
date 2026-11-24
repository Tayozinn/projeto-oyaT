# Fontes de Livros Gratuitos de Pré-Treino (`datasets/pretrain_books/`)

Este diretório armazena textos técnicos limpos (UTF-8 `.md` e `.txt`) extraídos de **livros gratuitos e de licença aberta em português do Brasil**, preparados para pré-treino do **ServerMind Neural** via `--corpus-dir datasets/pretrain_books`.

> **Nota sobre Git:** a pasta `datasets/pretrain_books/` está listada no `.gitignore` para manter o repositório leve; este manifesto registra a procedência e as licenças de cada obra.

## Livros Preparados

| Pasta | Obra | Repositório Oficial | Licença | Arquivos | Caracteres | Palavras |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `guiafoca-linux/` | Guia Foca GNU/Linux (Iniciante, Intermediário e Avançado) | `https://github.com/guiafoca/guiafoca` | GNU GPL / GFDL | 94 | 1.904.418 | 291.340 |
| `progit2-pt-br/` | Pro Git — 2ª Edição (PT-BR) | `https://github.com/progit/progit2-pt-br` | CC BY-NC-SA 3.0 | 89 | 898.311 | 138.955 |
| `rust-book-pt-br/` | The Rust Programming Language (PT-BR) | `https://github.com/rust-br/rust-book-pt-br` | MIT / Apache-2.0 | 61 | 667.552 | 104.601 |
| `descomplicando-docker/` | Descomplicando o Docker (LINUXtips) | `https://github.com/badtuxx/DescomplicandoDocker` | Livro aberto no GitHub | 16 | 182.329 | 24.215 |
| **Total** | **4 livros técnicos em PT-BR** | — | — | **260** | **3.652.610** | **559.111** |

## Como usar no treinamento do ServerMind Neural

```powershell
# 1. Pré-treino puro nos 4 livros (gera vocabulário BPE + pesos base)
.\target\release\servermind-ai.exe train-neural `
  --corpus-dir datasets/pretrain_books `
  --out checkpoints/servermind-neural-v3-pretrain `
  --run-name servermind-neural-v3-pretrain `
  --d-model 192 --layers 3 --heads 6 --ff 512 --seq-len 96 --vocab 2048 `
  --batch-size 4 --steps 400 --lr 0.002 --min-lr 0.0002

# 2. Pré-treino + Q&A técnico combinados
.\target\release\servermind-ai.exe train-neural `
  --corpus-dir datasets/pretrain_books `
  --dataset datasets/servermind_corpus_v2.jsonl `
  --out checkpoints/servermind-neural-v3 `
  --run-name servermind-neural-v3 `
  --d-model 192 --layers 3 --heads 6 --ff 512 --seq-len 96 --vocab 2048 `
  --batch-size 4 --steps 500 --lr 0.002 --min-lr 0.0002

# 3. Retomar treinamento a partir do latest.bin salvo
.\target\release\servermind-ai.exe train-neural `
  --resume `
  --corpus-dir datasets/pretrain_books `
  --dataset datasets/servermind_corpus_v2.jsonl `
  --out checkpoints/servermind-neural-v3 `
  --steps 1000 --lr 0.001 --min-lr 0.0001
```
