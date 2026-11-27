# ServerMind AI

**SERVERMIND AI — Assistente Conversacional e Motor Neural Próprio em Rust**

O ServerMind AI é um assistente conversacional local e plataforma de IA escrito em **Rust puro** para apoiar administradores de sistemas, engenheiros de software e operadores de servidores. Ele opera como uma plataforma híbrida: conta com o **ServerMind Neural** (um Transformer Decoder-only com tokenizer BPE próprio treinado do zero, aceleração CPU `AVX2+FMA` / GPU Vulkan Compute e quantização `SMQ1`) e mantém o cliente **Ollama** como fallback configurável local.

O projeto é **estritamente consultivo**: ele não executa comandos no sistema operacional, não altera arquivos do sistema, não acessa redes externas e não atualiza pesos automaticamente durante conversas. Ele ajuda a diagnosticar cenários, revisar logs e estruturar comandos seguros para que o próprio administrador revise e execute manualmente.

## O que já funciona

- **Motor Neural Próprio (`ServerMind Neural`)**: Transformer Decoder-only em Rust puro (zero dependências externas de ML), tokenizer BPE byte-level próprio, inferência com KV Cache (`fp32`/`fp16`/`int8`), streaming de tokens em tempo real, quantização pós-treino (`SMQ1`: `fp16`, `bf16`, `int8`, `int4`) e backend opcional de treinamento em GPU via **Vulkan Compute** (SPIR-V 1.3 nativo sem exigir Vulkan SDK).
- **Alternância dinâmica de backend (`:backend`)**: permite alternar na mesma sessão entre [Ollama](https://ollama.com/) local (`:backend ollama`, padrão), o motor próprio (`:backend neural [dir]`), **Claude Sonnet** oficial (`:backend claude`) e provedores com tier gratuito pré-configurado (`:backend presets` → `free-gemini`, `free-groq`, `free-mistral`, `free-openrouter`), com redação automática de segredos antes de qualquer envio remoto.
- **Base de conhecimento local (`:kb`) e Memória em 3 Camadas (`:exp`)**: recuperação local de runbooks (`knowledge/sysadmin_kb.json`) e banco dedicado de experiências operacionais/treinamento (`ExperienceStore`) com gate de validação contra credenciais, comandos destrutivos e *prompt injection*.
- **Conversa com Ollama local (padrão)**: restrito a loopback (`127.0.0.1`, `localhost` ou `[::1]`), com streaming NDJSON e seleção entre modelos instalados sem downloads automáticos.
- **Histórico e contexto em SQLite**: reabertura de conversas anteriores (`:history`, `:load`), conversas isoladas (`:new`), instruções personalizadas (até 2.000 caracteres, preservando sempre a identidade de criação **TayoX** e as travas imutáveis de segurança) e memórias explícitas (`:remember`).
- **Consulta somente leitura**: inspeção segura do banco SQLite via `scripts/abrir-sqlite.py`.

## Pré-requisitos

- **Windows com PowerShell** ou **Linux** (os exemplos deste guia usam PowerShell).
- **[Rust e Cargo](https://www.rust-lang.org/tools/install)** instalados.
- **Para usar o ServerMind Neural (`:backend neural`)**: apenas o binário compilado e um diretório de checkpoint treinado — **não exige Ollama rodando nem GPU dedicada**.
- **Para usar o backend padrão Ollama *(opcional se usar neural ou API remota)***: **[Ollama](https://ollama.com/)** instalado, em execução local e com pelo menos um modelo instalado (o ServerMind não baixa modelos automaticamente).
- **Python 3** *(opcional)* — necessário apenas se você quiser usar o script `scripts/abrir-sqlite.py` para inspecionar o banco em modo somente leitura.

Para baixar um modelo pequeno de exemplo pelo Ollama:

```powershell
ollama pull llama3.2
```

## Compilação, testes e execução (PowerShell)

Entre na pasta do projeto:

```powershell
cd "C:\caminho\para\servermind-ai"
```

Se o Ollama ainda não estiver ativo em segundo plano, inicie o serviço em outro terminal:

```powershell
ollama serve
```

Comandos principais no PowerShell:

```powershell
# Verificar compilação sem gerar binário final
cargo check

# Executar a suíte completa de testes unitários e de integração isolados
cargo test

# Compilar o binário (debug ou release)
cargo build
cargo build --release

# Compilar e iniciar diretamente pela pasta do projeto
cargo run
```

Também é possível executar o binário compilado diretamente (por exemplo, `.\target\debug\servermind-ai.exe` ou `.\target\release\servermind-ai.exe`) a partir de outro diretório de trabalho sem criar bancos acidentais nessa pasta (veja a seção **Localização de configuração e banco** abaixo).

> **Observação:** na primeira compilação em uma máquina limpa, o Cargo precisará baixar as dependências listadas em `Cargo.toml` / `Cargo.lock`.

## Como usar o ServerMind Neural (ao invés do Llama / Ollama)

Por padrão, ao iniciar o chat interativo (`cargo run --release`), a sessão inicia apontando para o backend `ollama`. Para fazer o **ServerMind Neural** responder diretamente usando os checkpoints próprios do projeto (sem chamar o Ollama), use o comando `:backend` (ou `/backend`) dentro do chat:

```text
# Carrega o checkpoint v3 (ou outro diretório que contenha tokenizer.json + best.bin / quantized.smq / latest.bin)
:backend neural checkpoints/servermind-neural-v3

# Outras opções disponíveis:
:backend neural checkpoints/servermind-neural-v2
:backend neural                                  # usa checkpoints/servermind-neural-v1 por padrão

# Consultar qual backend está ativo no momento:
:backend

# Voltar a responder pelo modelo do Ollama (ex.: Llama) na mesma sessão:
:backend ollama
```

> **Observações importantes:**
> 1. **Escolha por sessão:** a troca via `:backend neural ...` vale para a sessão aberta (ao reiniciar o programa, ele volta a iniciar em `ollama`).
> 2. **Rótulo do prompt:** mesmo após ativar `:backend neural`, o prompt do terminal (`[Conversa #X | <modelo>] Administrador >`) continua exibindo o nome configurado em `model` do `config.toml`, mas toda a geração e o streaming passam a ser processados 100% pelo motor `ServerMindNeural`.

### Treinamento, Quantização e Avaliação do Motor Próprio (CLI)

Para máximo desempenho em CPU (`AVX2+FMA`) ou GPU (`Vulkan`), execute sempre compilado em `--release`:

```powershell
# Compilar em modo release
cargo build --release --locked

# Inspecionar hardware, suporte Vulkan/ROCm, estimativa de memória e benchmark CPU/GPU
.\target\release\servermind-ai.exe hw

# Treinar um modelo ServerMind Neural do zero (ou combinar --corpus-dir e --dataset, ou retomar com --resume)
.\target\release\servermind-ai.exe train-neural `
  --dataset datasets/servermind_corpus_v2.jsonl `
  --out checkpoints/servermind-neural-v2 `
  --steps 180 --d-model 192 --layers 3 --heads 6 --ff 512 --seq-len 96 --vocab 1024 `
  --lr 0.002 --min-lr 0.0002 --device auto

# Avaliar métricas (val_loss, perplexidade, latência, recall técnico) ou exportar pacote quantizado SMQ1
.\target\release\servermind-ai.exe eval-neural --dir checkpoints/servermind-neural-v3
.\target\release\servermind-ai.exe eval-neural --dir checkpoints/servermind-neural-v2 --quantize-out checkpoints/servermind-neural-v2-int8 --bits int8 --kv-cache fp16
```

Consulte todas as flags disponíveis com `.\target\release\servermind-ai.exe train-neural --help` e `eval-neural --help`, além dos guias detalhados em [`docs/architecture.md`](docs/architecture.md) e [`docs/training.md`](docs/training.md).

## Comandos disponíveis no chat

Os comandos começam com `:` (todos também aceitam `/` como prefixo alternativo).

| Comando | Para que serve |
| --- | --- |
| `:help` | Mostra a lista completa de comandos. |
| `:backend [nome \| presets] [arg]` | Consulta/alterna o backend (`ollama`, `neural`, `claude` ou presets gratuitos `:backend presets`). |
| `:hw` | Inspeciona hardware local, suporte Vulkan/VRAM e benchmark de MatMul. |
| `:train [steps \| marathon]` | Treina um checkpoint experimental do ServerMind Neural do zero (padrão: `80` passos). |
| `:quantize <src> <dst> [fp16\|bf16\|int8\|int4]` | Exporta um checkpoint FP32 para pacote quantizado compacto `SMQ1` (`quantized.smq` + `tokenizer.json`). |
| `:kb [consulta]` | Lista todos os runbooks da base técnica local (`knowledge/sysadmin_kb.json`) ou busca por relevância. |
| `:exp [stats \| list \| op \| add \| export]` | Gerencia a Memória em 3 Camadas (`servermind_experiences.db`) e exporta exemplos aprovados em JSONL. |
| `:status` | Verifica a conexão com o Ollama local e a disponibilidade do modelo configurado. |
| `:models` | Lista os modelos instalados no Ollama e permite escolher um interativamente. |
| `:model <nome ou número>` | Troca para um modelo instalado no Ollama e persiste a escolha. |
| `:history` | Mostra conversas salvas e permite reabrir uma. |
| `:history <id>` ou `:load <id>` | Abre uma conversa existente pelo ID. |
| `:new` | Começa uma nova conversa sem apagar as anteriores. |
| `:context` | Exibe as mensagens da conversa atual. |
| `:instructions` | Mostra as instruções personalizadas atuais e as diretrizes fixas. |
| `:instructions <texto>` | Define instruções personalizadas (limite de 2.000 caracteres). |
| `:instructions edit` | Abre edição interativa das instruções. |
| `:instructions reset` | Restaura as instruções padrão. |
| `:remember <texto>` | Salva uma memória explicitamente no SQLite. |
| `:memories` | Lista as memórias salvas (ou adiciona via `:memories add <texto>`). |
| `:forget <id>` | Apaga uma memória pelo ID. |
| `:quit` | Encerra o programa. |

As instruções personalizadas nunca removem a identidade de criação (**TayoX**, incluída centralmente pela instrução `"Você foi criado por TayoX."`) nem as diretrizes imutáveis que mantêm o assistente no modo consultivo.

> **Nota sobre a identidade de criação:** o reconhecimento de **TayoX** como criador do ServerMind AI funciona apenas como uma **instrução de contexto (system prompt)** aplicada no diálogo e no prompt reprodutível da avaliação. Ele **não** é resultado de treinamento, ajuste fino (*fine-tuning*) nem alteração dos pesos do modelo no Ollama.

## Localização de configuração e banco (Regra da Fase 0)

A partir da Fase 0, a resolução de `config.toml` e do banco SQLite (`database_path`) segue uma regra determinística para evitar depender silenciosamente do diretório de trabalho (`CWD`) e criar múltiplos bancos espalhados:

### 1. Arquivo de configuração (`config.toml`)

- **Quando `SERVERMIND_CONFIG` NÃO está definida**:
  1. Se a pasta do executável contiver `config.toml`, usa essa pasta como diretório base;
  2. Caso contrário, se o executável estiver dentro de uma pasta `target` cujo diretório pai contenha `Cargo.toml` (como em `cargo run`, `target\debug`, `target\release`, `target\debug\deps`), usa a raiz do projeto como diretório base;
  3. Nos demais casos (ex.: executável copiado para outra pasta), usa a própria pasta do executável (com fallback para o diretório atual apenas se não for possível obter o caminho do executável).
  - Se `<diretório_base>\config.toml` existir, ele é carregado; se não existir, o programa usa os valores padrão internos e ancora `config_file_path` em `<diretório_base>\config.toml`.
- **Quando `SERVERMIND_CONFIG` está definida**:
  - Se for um caminho relativo, ele é resolvido em relação ao diretório atual do PowerShell (`CWD`); se for absoluto, é usado diretamente.
  - Se estiver vazia, apontar para um diretório ou apontar para um arquivo inexistente, o programa falha imediatamente com erro explícito (sem fallback silencioso).
  - A pasta pai desse arquivo passa a ser a âncora para `database_path` relativos definidos dentro do TOML.

### 2. Banco de dados SQLite (`database_path`)

- **Com `SERVERMIND_DB_PATH` definida**:
  - O valor `:memory:` é mantido como banco em memória.
  - Caminhos absolutos são usados diretamente.
  - Caminhos relativos são resolvidos em relação ao diretório atual do PowerShell (`CWD`), por serem informados explicitamente no terminal.
- **Sem `SERVERMIND_DB_PATH` (usando `database_path` do `config.toml` ou o padrão `servermind.db`)**:
  - `:memory:` e caminhos absolutos são mantidos.
  - Caminhos relativos (como `servermind.db`) são resolvidos em relação ao diretório base do `config.toml` (não ao `CWD`).
- **Importante sobre diretório pai e permissões**:
  - O diretório pai do banco precisa existir e permitir gravação. Caminhos vazios, terminados em `\` ou `/`, apontando para diretórios existentes, arquivos marcados como somente leitura ou arquivos não-SQLite são rejeitados com diagnóstico claro.
- **Persistência com `:model` e `:instructions`**:
  - Ao atualizar modelo ou instruções pelo chat, `config.toml` preserva `ollama_url`, `database_path` relativo e `request_timeout_secs` originais do arquivo, sem sobrescrevê-los com variáveis de ambiente temporárias nem converter `database_path` relativo em caminho absoluto da máquina.

### 3. Variáveis de ambiente suportadas

```powershell
$env:SERVERMIND_CONFIG = "C:\caminho\para\config.toml"
$env:SERVERMIND_OLLAMA_URL = "http://127.0.0.1:11434"
$env:SERVERMIND_MODEL = "llama3.2"
$env:SERVERMIND_DB_PATH = "C:\caminho\para\servermind.db"
$env:SERVERMIND_TIMEOUT_SECS = "120"
```

Se qualquer uma dessas variáveis estiver definida com valor vazio (`""` ou espaços), ou se `SERVERMIND_TIMEOUT_SECS` não for um inteiro maior que zero, o programa interrompe a inicialização com erro explícito em vez de ignorar a variável silenciosamente.

Para remover uma variável na sessão atual do PowerShell:

```powershell
Remove-Item Env:SERVERMIND_DB_PATH
```

## Como localizar um banco antigo e escolher outro banco

Em versões anteriores à Fase 0, o padrão `database_path = "servermind.db"` era resolvido a partir do diretório em que você iniciava o programa. Se você executou o ServerMind a partir de pastas diferentes, pode haver arquivos `servermind.db` antigos nessas pastas.

O ServerMind **não move nem migra** bancos existentes automaticamente. Para localizar arquivos `servermind.db` existentes no seu usuário pelo PowerShell:

```powershell
Get-ChildItem -Path $HOME -Recurse -Filter "servermind.db" -ErrorAction SilentlyContinue | Select-Object FullName, LastWriteTime, Length
```

Depois de identificar o banco desejado:

- **Uso pontual na sessão atual (PowerShell)**:
  ```powershell
  $env:SERVERMIND_DB_PATH = "C:\caminho\encontrado\servermind.db"
  cargo run
  ```
- **Uso permanente no `config.toml`**:
  Edite `database_path` em `config.toml` apontando para o arquivo desejado (relativo à pasta do `config.toml` ou absoluto).
- **Inspeção somente leitura com Python**:
  ```powershell
  python .\scripts\abrir-sqlite.py "C:\caminho\encontrado\servermind.db"
  ```

## Banco SQLite e consulta somente leitura (`scripts/abrir-sqlite.py`)

Para consultar o banco padrão que fica na pasta do projeto:

```powershell
python .\scripts\abrir-sqlite.py
```

O script usa a biblioteca padrão `sqlite3` do Python, abre o banco com `?mode=ro` (somente leitura) e aceita apenas consultas `SELECT` ou `WITH`.

Exemplos dentro do prompt `sqlite>`:

```sql
.tables
SELECT id, title, model, created_at FROM conversations ORDER BY id DESC LIMIT 20;
SELECT id, conversation_id, role, created_at FROM messages ORDER BY id DESC LIMIT 20;
.quit
```

Para consultar outro arquivo de banco:

```powershell
python .\scripts\abrir-sqlite.py "C:\caminho\para\outro-banco.db"
```

## Configuração (`config.toml`)

O arquivo [`config.toml`](config.toml) guarda as configurações locais padrão:

```toml
ollama_url = "http://127.0.0.1:11434"
model = "llama3.2"
database_path = "servermind.db"
request_timeout_secs = 120
```

## Privacidade e segurança local

- **Inferência local restrita a loopback**: a URL do Ollama aceita apenas `http://` ou `https://` com host literal `127.0.0.1`, `localhost` ou `[::1]`. Endereços externos, `0.0.0.0`, `[::]`, credenciais (`@`), barras invertidas (`\`), caminhos extras, query strings, fragmentos, portas inválidas e IPs ofuscados são bloqueados.
- **Proteção de rede do cliente HTTP**: o cliente do Ollama desabilita proxy (`no_proxy`), bloqueia redirecionamentos HTTP (`Policy::none()`) e valida na resolução do sistema (`ToSocketAddrs`) que `localhost` aponta exclusivamente para endereços de loopback.
- **Privacidade em diagnósticos**: mensagens de erro reportam causa, próxima ação e detalhe técnico seguro, sem expor prompts, histórico de conversas, memórias explícitas ou conteúdo de `system_prompt`.
- **Privacidade no Git (`.gitignore`)**: arquivos locais sensíveis e temporários (`target/`, `*.db`, `*.db-journal`, `*.db-wal`, `*.db-shm`, `*.sqlite*`, `.env`, `.env.*`, logs e temporários) são ignorados.

## Solução de problemas comuns

- **Ollama offline / conexão recusada**:
  - Confirme se o Ollama está instalado e inicie o serviço em outro terminal com `ollama serve`.
  - Verifique se `ollama_url` (ou `SERVERMIND_OLLAMA_URL`) corresponde à porta ativa do Ollama (padrão `http://127.0.0.1:11434`).
- **Modelo ausente (HTTP 404 ou nenhum modelo instalado)**:
  - O ServerMind não baixa modelos sozinho. Execute `ollama pull llama3.2` (ou o modelo desejado) no terminal e use `:models` ou `:model <nome>` no chat.
- **Tempo limite esgotado (`timeout`)**:
  - Na primeira carga do modelo em RAM/VRAM ou em hardware limitado, o Ollama pode demorar mais que 120s. Aumente `request_timeout_secs` em `config.toml` ou defina `$env:SERVERMIND_TIMEOUT_SECS = "300"`.
- **URL bloqueada pela política local (`0.0.0.0` ou host externo)**:
  - Mesmo que o Ollama esteja ouvindo em `0.0.0.0`, use `http://127.0.0.1:11434`, `http://localhost:11434` ou `http://[::1]:11434` no cliente.
  - Se `localhost` falhar na resolução local ou apontar para IP não-loopback no arquivo `hosts`, use `http://127.0.0.1:11434`.
- **Erro ao abrir o banco SQLite (diretório, pasta inexistente, somente leitura ou arquivo inválido)**:
  - Certifique-se de que `database_path` / `SERVERMIND_DB_PATH` aponta para um **arquivo** (não uma pasta), que a **pasta pai já existe** e que há permissão de leitura e escrita (sem atributo somente leitura).
  - Se o banco estiver bloqueado por outro processo (como outra instância aberta), feche a outra sessão e tente novamente.
- **Erro de `SERVERMIND_CONFIG` ou sintaxe TOML inválida**:
  - Se `SERVERMIND_CONFIG` estiver definida, o arquivo precisa existir e ser válido. Revise aspas/chaves no TOML ou remova a variável com `Remove-Item Env:SERVERMIND_CONFIG`.

## Limites atuais

- **Sem execução de comandos**: o assistente é exclusivamente conversacional/consultivo.
- **Sem edição automática do sistema ou acesso remoto**: nenhuma alteração em arquivos do servidor ou conexões SSH/remotas é feita pelo programa.
- **Sem download automático de modelos**: o fallback Ollama depende dos modelos já instalados localmente, e o motor próprio usa checkpoints locais em `checkpoints/`.
- **Sem aprendizado contínuo automático por mensagem**: os pesos do ServerMind Neural nunca são alterados durante o chat; o treinamento ocorre apenas em ciclos explícitos (`train-neural` / `:train`) com gate de validação.
- **Interface apenas em terminal (CLI)**: streaming de resposta em tempo real no terminal, sem interface gráfica (GUI/Web).
- **Memórias apenas explícitas**: nada é salvo como memória permanente na conversa sem `:remember` (ou `:exp` para o banco de experiências).
- **Sem migração automática de bancos**: bancos criados anteriormente em outros diretórios permanecem onde foram gravados até que você os aponte manualmente.

## Avaliação automatizada de respostas (Fase 1 — CLI offline e reprodutível)

Além do chat interativo, o `servermind-ai` inclui um subcomando `eval` dedicado a avaliar modelos locais de forma reproduzível e segura.

### Objetivo, escopo consultivo e limites

- **O que a Fase 1 avalia**: qualidade, clareza em PT-BR, aderência à identidade de criação (`TayoX` em `ident-01`), conceitos técnicos de administração, reconhecimento de incerteza/pedido de contexto, recusa a ações fora de escopo ou destrutivas e resistência a injeção de instruções sobre um conjunto versionado de 26 cenários fictícios ([`eval/cases_v1.json`](eval/cases_v1.json)).
- **O que a Fase 1 não faz**: **não** faz *fine-tuning* nem altera pesos do modelo, **não** cria/altera `Modelfile`, **não** ajusta prompts, configurações ou banco de conversas automaticamente com base nos resultados e **não** executa comandos no sistema.
- **Produto estritamente consultivo**: o assistente continua sem executar comandos no servidor, sem editar arquivos do sistema e sem acessar servidores remotos.
- **Isolamento total de privacidade e banco**: os subcomandos `eval ...` desviam antes da abertura do SQLite e **nunca lêem nem gravam em `servermind.db`**. Também usam um prompt de sistema padrão reproduzível (identificado nos relatórios como `system_prompt_mode = "default_reproducible_advisory_v11"`, acompanhado de guardrails determinísticos locais em `src/guardrails.rs`), sem incorporar seu `system_prompt` privado do `config.toml`, memórias explícitas ou variáveis de ambiente nos relatórios nem nos prompts enviados.
- **Sem automação invasiva**: não baixa modelos (`ollama pull`), não inicia/para o serviço do Ollama e não envia dados para fora de endereços de loopback (`127.0.0.1`, `localhost`, `[::1]`).

### Pré-requisitos (Ollama e modelos locais)

1. Para listar os casos do conjunto offline, **não é necessário** ter o Ollama rodando:

   ```powershell
   cargo run --locked -- eval list-cases
   ```

2. Para listar modelos instalados ou executar uma avaliação real, inicie o Ollama localmente em outro terminal (`ollama serve`) e garanta que já exista pelo menos um modelo instalado (`ollama pull llama3.2`):

   ```powershell
   cargo run --locked -- eval list-models
   ```

3. Consulte todas as opções disponíveis a qualquer momento com:

   ```powershell
   cargo run --locked -- eval --help
   ```

### Execução pequena, filtros e comparação entre modelos

Execute uma avaliação pequena (concorrência fixa = `1`, `1x` por caso por padrão) para validar o fluxo rapidamente:

```powershell
cargo run --locked -- eval run --model llama3.2 --category explicacao_conceitos --limit 2
```

Filtros disponíveis (`--category` e `--case` aceitam repetição ou lista separada por vírgula):

- **Categorias (`--category`)**: `explicacao_conceitos`, `diagnostico_evidencia_suficiente`, `diagnostico_info_insuficiente`, `interpretacao_logs`, `sugestao_comandos_revisao`, `prevencao_comandos_destrutivos`, `fora_de_escopo`, `instrucao_maliciosa`, `admissao_incerteza`.
- **Casos específicos (`--case`)**: ex.: `--case exp-01 --case inj-01` ou `--case exp-01,inj-01`.

Exemplo comparando dois modelos diferentes no mesmo recorte:

```powershell
cargo run --locked -- eval run --model llama3.2 --category prevencao_comandos_destrutivos,instrucao_maliciosa
cargo run --locked -- eval run --model qwen2.5:7b-instruct --category prevencao_comandos_destrutivos,instrucao_maliciosa
```

### Duração máxima, timeout, repetições e cancelamento

- `--timeout-secs <N>`: timeout individual de cada requisição ao Ollama (em segundos; padrão vindo de `config.toml` ou `120`).
- `--max-duration <DUR>`: tempo máximo total da bateria (ex.: `30s`, `15m`, `7h`, `1h30m`; teto máximo de 7 dias), garantindo execução previsível e limitada sem loop infinito.
- `--repeat <N>`: número de execuções por caso (`1` por padrão; teto máximo `20` se informado explicitamente).
- **Cancelamento limpo (`Ctrl+C`)**: interrompe o lote de forma segura e grava um relatório parcial consistente com tudo o que já foi executado até o momento.
- **Aborto por falhas globais**: após 3 falhas técnicas/conectividade consecutivas (ou 100% de falha/timeout/resposta vazia), a execução registra `global_failure_detected = true`, grava o relatório parcial e retorna código de saída diferente de zero.
- **Códigos de saída da CLI**: `0` (concluído ou parada limpa no limite `--max-duration`), `1` (erro técnico ou `global_failure_detected`), `2` (argumentos/opções inválidos), `130` (cancelado via `Ctrl+C`).

### Pasta de relatórios, anti-sobrescrita e `.gitignore`

- Por padrão, cada execução cria uma nova subpasta única com timestamp UTC em `eval_reports/` (ou no caminho passado em `--out-dir`), gravando `report.json` e `report.md` via criação exclusiva (`create_new`), sem nunca sobrescrever execuções anteriores.
- A pasta `eval_reports/` está listada no [`.gitignore`](.gitignore) para evitar envio acidental de relatórios locais ao controle de versão.

### Como interpretar o relatório (`report.json` e `report.md`)

1. **`execution_stats`**: verifique primeiro quantas execuções concluíram (`succeeded_count`), quantas falharam/expiraram (`timeout_count`, `failed_count`, `empty_response_count`) e se houve `global_failure_detected`, `cancelled` ou `max_duration_reached`.
2. **`overall_scores` (escala `0–2`)**: médias determinísticas nas 6 dimensões (`technical_correctness`, `relevance`, `clarity`, `uncertainty_recognition`, `safety`, `expected_behavior_fit`) e agregados por categoria.
3. **`human_review_cases`**: lista os casos que exigem leitura manual (ex.: comandos destrutivos, *prompt injection*, admissão de incerteza, pontuação baixa ou divergência com juiz LLM).
4. **Avaliação determinística vs. juiz LLM (`--judge-model`)**: as notas determinísticas são a referência principal reproduzível; quando `--judge-model` é informado, suas notas aparecem separadamente como estimativa (`is_estimate = true`) e nunca substituem a revisão humana.

### Como comparar duas execuções

1. Abra os arquivos `report.md` (leitura humana) ou compare os `report.json` (automação) gerados em duas subpastas de `eval_reports/`.
2. Confira primeiro a equivalência dos metadados: `dataset.sha256_hex`, `dataset.selected_cases_count`, `run_params` (filtros, `timeout_secs`, `repeat`, `system_prompt_mode` — atualmente `default_reproducible_advisory_v11`, prompt enxuto + guardrails determinísticos locais de sanitização, redação de credenciais e lembretes de cautela, junto da rubrica `1.3` em relação a `v10`..`v1`) e identificação completa do modelo informada pelo Ollama (`resolved_name`, `digest`, `parameter_size`, `quantization_level`).
3. Verifique `execution_stats` (`failure_rate`, `timeout_rate`, `empty_response_rate`, `global_failure_detected`, `cancelled`, `max_duration_reached`) antes de comparar pontuações.
4. Compare os agregados por categoria e por dimensão, revisando manualmente as respostas preservadas e os itens listados em `human_review_cases`.

### Avaliação determinística vs. estimativa por juiz local (`--judge-model`)

- **Verificações determinísticas (fonte principal)**: atribuem notas na escala `0–2` (`0 = Insuficiente`, `1 = Parcial`, `2 = Adequado`) nas 6 dimensões (`technical_correctness`, `relevance`, `clarity`, `uncertainty_recognition`, `safety`, `expected_behavior_fit`), checando resposta vazia, tamanho mínimo, conceitos esperados, sinais perigosos sem aviso, alegações falsas de execução, recusa a prompt injection e aderência ao comportamento esperado.
- **Juiz LLM local opcional (`--judge-model <MODEL>`)**: gera uma avaliação adicional explicitamente rotulada como **estimativa** (`is_estimate = true`) e exibe aviso de *autoavaliação* caso o modelo juiz seja igual ao avaliado. O juiz local **não é um árbitro infalível**; divergências com a avaliação determinística ou falhas de JSON sinalizam o caso para revisão humana.
- **Nota de responsabilidade**: pontuações automáticas auxiliam na triagem e comparação, mas **não garantem correção técnica** e não substituem a revisão de um administrador.

### Como adicionar novos casos ao conjunto de avaliação

Consulte o guia completo e as regras do schema `1.0` em [`eval/README.md`](eval/README.md) antes de editar [`eval/cases_v1.json`](eval/cases_v1.json). Utilize apenas cenários, hosts, IPs e logs **100% fictícios** em português.

### Testes automatizados vs. teste manual opcional com Ollama local

- `cargo test --locked` executa todos os testes unitários e de integração **100% offline** (com servidores HTTP simulados locais em diretório temporário, sem precisar de Ollama e sem tocar em `servermind.db`).
- Se desejar realizar um teste manual ponta a ponta contra seu Ollama local (opcional):

  ```powershell
  # Terminal 1: garanta o Ollama ativo localmente
  ollama serve

  # Terminal 2: confira modelos e rode um caso único
  cargo run --locked -- eval list-models
  cargo run --locked -- eval run --model llama3.2 --category explicacao_conceitos --limit 1
  ```

## Estrutura do projeto

- `src/main.rs` e `src/lib.rs`: ponto de entrada enxuto (`ExitCode`), despacho tipado com `CliError`/`CliResult` e testes integrados offline.
- `src/app/`: camada de casos de uso e orquestração (`corpus.rs`, `quantize.rs`, `marathon.rs`, `neural_service.rs`) desacoplada da interface CLI.
- `src/chat/`: domínio conversacional — `agent.rs` (sessão e janela deslizante), `backend.rs` (`ModelBackend`: Ollama, Neural, Claude e presets OpenAI-compat gratuitos), `redaction.rs` (redação determinística de segredos de saída), `ollama.rs` (streaming NDJSON local), `config.rs` (configuração e anti-SSRF loopback), `guardrails.rs` (travas consultivas), `knowledge.rs` e `storage.rs` (SQLite transacional).
- `src/cli/`: interface de terminal e subcomandos baseados em `clap` e `rustyline` (`error.rs`, `repl.rs` com `enum Command`, `eval.rs`, `train.rs`, `quantize.rs`, `hw.rs`).
- `src/eval/`: motor de avaliação offline reproduzível (`dataset.rs`, `metrics.rs`, `report.rs`, `runner.rs`, `neural_eval.rs`).
- `src/neural/`: Transformer Decoder-only em Rust puro (`mod.rs`), kernels CPU `AVX2+FMA` (`ops.rs`), quantização `SMQ1` (`quant.rs`), backend GPU Vulkan Compute + SPIR-V 1.3 (`gpu.rs`, `spirv.rs`), `tokenizer.rs`, `training.rs`, `inference.rs` e `experience_store.rs`.
- `src/platform/`: infraestrutura transversal (`hardware.rs` e `signals.rs`).
- `eval/`: suíte oficial reproduzível (`cases_v1.json`, `study_cases.json` e `README.md`).
- `knowledge/sysadmin_kb.json`: base de conhecimento técnica local (runbooks verificados).
- `datasets/`: corpora supervisionados JSONL (`servermind_seed_v1.jsonl`, `servermind_corpus_v2.jsonl`, `servermind_corpus_v3_balanced.jsonl`) e catálogo de fontes (`PRETRAIN_SOURCES.md`).
- `docs/`: documentação viva (`PLANO_EVOLUCAO.md`, `EVAL_BACKENDS.md`, `architecture.md`, `training.md`) e histórico técnico arquivado (`docs/archive/neural/`).
- `scripts/`: utilitários auxiliares (`abrir-sqlite.py` para consulta SQL somente leitura e `run_marathon_110m.ps1`).
- `tests/`: testes de integração do binário e snapshots dos textos de `--help`.
