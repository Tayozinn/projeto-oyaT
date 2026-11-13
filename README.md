# ServerMind AI

Estou construindo o ServerMind AI por etapas. A ideia é chegar a um assistente local especializado em servidores, mas comecei pelo que precisa funcionar primeiro: conversar com um modelo instalado na própria máquina, guardar o histórico e permitir que eu controle como o assistente se comporta.

O projeto está na fase de assistente conversacional. Ele ainda não administra servidores nem executa comandos. Por enquanto, ajuda a tirar dúvidas, pensar em diagnósticos e preparar comandos para que o administrador revise e execute por conta própria.

## O que já funciona

- Conversa local com modelos servidos pelo [Ollama](https://ollama.com/).
- Seleção entre os modelos já instalados, sem downloads automáticos.
- Histórico de conversas que pode ser reaberto ou separado em uma nova conversa.
- Instruções personalizadas para ajustar o foco do assistente.
- Memórias explícitas, salvas apenas quando você usa o comando para isso.
- Persistência local em SQLite.
- Consulta ao banco por um script em modo somente leitura.

O programa foi escrito em Rust. O Ollama executa o modelo; o SQLite guarda as conversas, configurações e memórias no computador.

## Estado do projeto

Este é um projeto em desenvolvimento e um experimento que estou construindo para aprender e evoluir a ideia com calma. Quero avançar por etapas: primeiro ter uma base conversacional útil, depois estudar como acrescentar recursos de administração com limites claros e verificações cuidadosas.

Nesta versão, o ServerMind não executa comandos, não altera arquivos ou configurações do sistema e não acessa servidores. Ele oferece orientação; quem administra decide o que revisar e executar.

## O que você precisa

- Windows com PowerShell (as instruções abaixo usam esse ambiente).
- [Rust e Cargo](https://www.rust-lang.org/tools/install).
- [Ollama](https://ollama.com/) instalado e em execução.
- Pelo menos um modelo instalado no Ollama.

Um modelo pequeno, como `llama3.2`, pode servir para começar. O desempenho depende do modelo e dos recursos disponíveis na máquina. Para baixar esse modelo pelo Ollama:

```powershell
ollama pull llama3.2
```

O ServerMind não baixa modelos sozinho. O Ollama precisa estar disponível localmente em `http://127.0.0.1:11434`, que é o endereço padrão.

## Instalação e execução

Clone o repositório e entre na pasta do projeto:

```powershell
git clone <URL-DO-REPOSITORIO>
cd servermind-ai
```

Se o Ollama não estiver iniciando automaticamente, abra outro terminal e execute:

```powershell
ollama serve
```

Depois, na pasta do projeto, compile e inicie o ServerMind:

```powershell
cargo run
```

Para rodar a suíte de testes:

```powershell
cargo test
```

Na primeira compilação, o Cargo pode precisar baixar as dependências do projeto.

## Comandos disponíveis

Os comandos começam com `:`. Alguns também aceitam `/` como prefixo alternativo.

| Comando | Para que serve |
| --- | --- |
| `:help` | Mostra a lista de comandos. |
| `:status` | Verifica a conexão com o Ollama e a disponibilidade do modelo ativo. |
| `:models` | Lista os modelos instalados e permite escolher um. |
| `:model <nome ou número>` | Troca para um modelo instalado. |
| `:history` | Mostra conversas salvas e permite reabrir uma. |
| `:history <id>` ou `:load <id>` | Abre uma conversa pelo ID. |
| `:new` | Começa uma conversa nova sem apagar as anteriores. |
| `:context` | Exibe as mensagens da conversa atual. |
| `:instructions` | Mostra as instruções personalizadas atuais. |
| `:instructions <texto>` | Define instruções personalizadas. O limite é 2.000 caracteres. |
| `:instructions edit` | Abre a edição interativa das instruções. |
| `:instructions reset` | Restaura as instruções padrão. |
| `:remember <texto>` | Salva uma memória explicitamente. |
| `:memories` | Lista as memórias salvas. |
| `:forget <id>` | Apaga uma memória pelo ID. |
| `:quit` | Encerra o programa. |

As instruções personalizadas não removem as regras fixas que mantêm o assistente no modo consultivo.

## Banco SQLite

O banco padrão é `servermind.db`, criado no diretório de trabalho usado para iniciar o programa. Para consultar o banco que fica na pasta do projeto, execute no PowerShell:

```powershell
python .\abrir-sqlite.py
```

O script usa o Python e sua biblioteca SQLite padrão; não é preciso instalar outro cliente. Ele abre o banco em modo somente leitura e aceita consultas `SELECT` e `WITH`.

Dentro do prompt `sqlite>`:

```sql
.tables
SELECT id, title, model FROM conversations ORDER BY updated_at DESC LIMIT 20;
.quit
```

Para consultar outro arquivo de banco, passe seu caminho:

```powershell
python .\abrir-sqlite.py "C:\caminho\para\servermind.db"
```

## Configuração

O arquivo [`config.toml`](config.toml) guarda as opções locais. Os principais valores são:

```toml
ollama_url = "http://127.0.0.1:11434"
model = "llama3.2"
database_path = "servermind.db"
request_timeout_secs = 120
```

Também é possível definir opções pela sessão do PowerShell:

```powershell
$env:SERVERMIND_OLLAMA_URL = "http://127.0.0.1:11434"
$env:SERVERMIND_MODEL = "llama3.2"
$env:SERVERMIND_DB_PATH = "servermind.db"
$env:SERVERMIND_TIMEOUT_SECS = "120"
```

Como o banco usa um caminho relativo por padrão, inicie o programa sempre a partir da pasta do projeto para manter o arquivo SQLite no mesmo lugar.

## Privacidade e segurança

As conversas, configurações e memórias são guardadas localmente no SQLite. A inferência é feita pelo Ollama local. O cliente valida o endereço do Ollama para aceitar apenas endereços locais.

O ServerMind é consultivo nesta fase: não executa comandos nem faz alterações no sistema. As memórias também não são inferidas automaticamente da conversa; só são gravadas com `:remember`.

## Estrutura do projeto

- `src/main.rs`: inicialização e testes.
- `src/cli.rs`: interface de terminal e comandos.
- `src/agent.rs`: coordenação das conversas e do contexto.
- `src/ollama.rs`: comunicação com a API local do Ollama.
- `src/storage.rs`: armazenamento SQLite.
- `src/config.rs`: configuração e validações.
- `src/models.rs`: estruturas de dados.
- `abrir-sqlite.py`: consulta ao banco SQLite em modo somente leitura.

## Ideias para as próximas etapas

Quero continuar desenvolvendo o projeto aos poucos, ouvindo o que funciona na prática e reforçando a segurança antes de ampliar o que o agente pode fazer. A administração automatizada de servidores é uma direção futura, não uma capacidade desta versão.

Sugestões, dúvidas e contribuições são bem-vindas.
