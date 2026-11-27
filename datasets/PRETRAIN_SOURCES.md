# Acervo Balanceado de Pré-Treino e Fine-Tuning (`SM-010` — 50% Servidores / 50% Programação)

Este documento registra a procedência, as licenças abertas, o balanceamento matemático **50% / 50%** e o comando de treinamento da **Opção B (`112,9M` parâmetros — classe GPT-2 Base)** na **AMD Radeon RX 7600 (8 GB VRAM) + 16 GB RAM**.

> **Nota sobre Git:** `datasets/pretrain_books/` e `datasets/pretrain_corpus_v2/` estão no `.gitignore` para manter o repositório Git enxuto, enquanto o dataset supervisionado `datasets/servermind_corpus_v3_balanced.jsonl` e o script `scripts/run_marathon_110m.ps1` permanecem versionáveis no projeto.

---

## 1. Biblioteca de Pré-Treino `v2` (`datasets/pretrain_corpus_v2/` — `10,27 MB` UTF-8 Limpo)

| Eixo | Diretório | Obras e Guias Técnicos em PT-BR Incluídos | Capítulos Limpos | Caracteres | Proporção Exata |
| :--- | :--- | :--- | ---: | ---: | ---: |
| **Servidores & Infraestrutura** | `datasets/pretrain_corpus_v2/servidores/` | **15 obras:** *Guia Foca GNU/Linux* (Iniciante/Intermediário/Avançado), *Descomplicando Kubernetes*, *OWASP Web Security Testing Guide (WSTG PT-BR)*, *Descomplicando Prometheus*, *Descomplicando Docker*, *Linux4Noobs*, *Segurança de Redes*, *Descomplicando ArgoCD*, *OWASP Secure Coding Practices PT-BR*, *Guia DevOps*, *Descomplicando GitLab*, *Descomplicando Ansible*, *LinuxTips Terraform*, *Descomplicando Helm*, *Terraform 101* | **369** | **`5.153.094`** | **`50,19%`** |
| **Programação & Engenharia** | `datasets/pretrain_corpus_v2/programacao/` | **11 obras:** *Pense em Python (2ª Ed. PT-BR)*, *Pro Git (2ª Ed. PT-BR)*, *Eloquente JavaScript (PT-BR)*, *The Rust Programming Language (PT-BR)*, *Aprenda Go com Testes (PT-BR)*, *Rust com Exemplos (PT-BR)*, *Rust4Noobs*, *Python4Noobs*, *SQL4Noobs*, *TypeScript para Brasileiros*, *TypeScript Livro de Bolso* | **588** | **`5.114.844`** | **`49,81%`** |
| **TOTAL CONSOLIDADO** | `datasets/pretrain_corpus_v2/` | **26 livros e referenciais abertos em PT-BR** | **957** | **`10.267.938`** | **`100,00%`** |

---

## 2. Dataset Supervisionado `v3` (`datasets/servermind_corpus_v3_balanced.jsonl`)

- **Total de pares `<user> ... <assistant> ...`:** **`1.544` diálogos únicos** (`0` prompts duplicados, `100%` JSON válido — **`12,8x` maior** que o `v2` anterior):
  - **`772` pares (`50,0%`) — Servidores, Linux, Containers, Kubernetes, Redes, Prometheus & Segurança Defensiva**
  - **`772` pares (`50,0%`) — Programação (Rust, Python, TypeScript/JS, Go, SQL), Algoritmos, Estruturas de Dados, Git & Arquitetura**
- **Intercalação 1:1:** As linhas alternam estritamente `servidores -> programacao -> servidores -> programacao`, garantindo equilíbrio exato em qualquer split ou mini-batch.

---

## 3. Arquitetura Opção B — `ServerMind-110M` na AMD Radeon RX 7600

| Parâmetro Arquitetural | Valor | Detalhe Técnico |
| :--- | ---: | :--- |
| **`total_params`** | **`112.934.400` (`112,93M`)** | **32,6x maior** que o modelo experimental `3,46M` |
| **`d_model` / `n_layers` / `n_heads` / `d_ff`** | `768 / 15 / 12 / 3072` | Porte classe GPT-2 Base expandido (`head_dim = 64`) |
| **`max_seq_len` / `vocab_size`** | `512` / `4096` | Comporta perguntas completas + explicações + blocos multi-linha de código |
| **Arena VRAM Alocada (`DEVICE_LOCAL`)** | **`2.323,0 MB` (`2,27 GB`)** | Usa apenas **29%** dos `7.920 MB` da RX 7600 (bem abaixo do limite de `4.096 MB` por SSBO) |
| **Pacote Exportado `SMQ1 INT8`** | **`~108 MB`** | Inferência instantânea (`< 0,2s`) |

---

## 4. Como Disparar (Quando o Treino Atual Terminar ou Após `Ctrl+C`)

```powershell
# Noite 1 — Cria o modelo de 113M do zero, treina por 8h na RX 7600 e gera sequência evolutiva + INT8:
.\scripts\run_marathon_110m.ps1 -Duration "8h"

# Noites seguintes — Continua evoluindo exatamente de onde parou:
.\scripts\run_marathon_110m.ps1 -Resume -Duration "8h"

# Para conversar com o modelo de 113M no chat:
.\target\release\servermind-ai.exe
# e digite dentro do chat:
# :backend neural checkpoints/servermind-neural-110m
```
