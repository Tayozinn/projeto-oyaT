# Plano de Evolução — ServerMind AI

Nov 25, 2026 · @tayo

## Visão e foco

O ServerMind AI passa a ter um único objetivo: ser um agente que ajuda a administrar servidores e a programar, com segurança acima de velocidade.

- **Produto:** um agente que consulta, propõe comandos e, só com a sua confirmação, executa.
- **Cérebro trocável:** o agente fala com o modelo por uma interface única (`ModelBackend`). O modelo pode ser Claude via API, Ollama local ou o seu modelo neural.
- **Congelado:** o modelo neural próprio, a quantização e a maratona de treino. Entram só correções de bugs que perdem dados.
- **Fora de escopo por ora:** interface web, vários usuários, execução em vários servidores ao mesmo tempo.

**Como saber que deu certo**

- O `eval` mostra o modelo escolhido acertando mais casos que o llama3.2 atual (baseline medido na Fase 3).
- Nenhum comando roda sem passar pela política, e todo comando executado aparece no log de auditoria.
- `cargo fmt --check`, `cargo clippy` e `cargo test` passam no CI a cada commit.
- Você consegue explicar em 5 minutos onde fica cada parte do projeto.

## Regras de ouro

Sete regras mantêm o projeto sob controle. Quando travar, volte aqui antes de abrir código novo.

1. **Uma fase ativa por vez**, com no máximo 2 tarefas abertas ao mesmo tempo.
2. **Cada tarefa cabe em uma sessão de 2 a 3 horas.** Se não cabe, quebre em duas.
3. **Em bug, o teste vem primeiro.** Escreva o teste que falha e depois a correção.
4. **Não misture mover código com mudar comportamento** no mesmo commit.
5. **Testes verdes antes de commitar:** `cargo fmt`, `cargo clippy` e `cargo test`.
6. **Congelado é congelado.** Ideia nova vai para a Caixa de ideias, no fim do documento, e não para o código.
7. **Nada executa em servidor real** antes da Fase 6 e de um ensaio em VM.

Sinal de alerta: se você abriu mais de dois arquivos que não fazem parte da tarefa do dia, pare e releia a tarefa.

## Estado atual

O projeto reúne três produtos e uma interface de linha de comando, e é no `cli.rs` que a complexidade se concentra.

| Domínio | Módulos | Estado | O que dói |
| --- | --- | --- | --- |
| Chat consultivo | `agent`, `ollama`, `storage`, `guardrails`, `knowledge`, `models`, `config` | Estável e bem testado (rollback, privacidade, mocks de HTTP) | Respostas fracas do modelo local; o guardrail roda depois do streaming |
| Avaliação offline | `eval_dataset`, `eval_metrics`, `eval_report`, `eval_runner`, `evaluation` | Robusto: argumentos validados, códigos de saída documentados, testes ponta a ponta | O help manual diverge do parser (`--with-kb` ausente); o `CtrlCGuard` mora aqui e o treino depende dele |
| Modelo neural | `neural`, `tokenizer`, `training`, `inference`, `experience`, `infrastructure` | Funciona e tem testes de retomada de checkpoint e loss mascarada (`SM-011`) | Treino novo apaga checkpoints; loop de treino com cerca de 400 linhas; vai ficar congelado |
| Interface (CLI) | `cli.rs` (cerca de 3.900 linhas), `main.rs` | REPL e subcomandos funcionam | Mistura REPL, parsing, treino e quantização; três estilos de parsing; erros como `String` |

**Pontos fortes a preservar**

- Injeção de dependências: `load_with_overrides` e writers `out`/`err` tornam tudo testável.
- Testes de privacidade com "canário" e de rollback transacional.
- Segurança tratada como requisito, e não como detalhe.
- Avaliação reproduzível, com SHA do dataset e relatórios sem dados privados.

## Roadmap por fases

Sete fases levam do estado atual até o agente com ações, e a Fase 3 já entrega um chat com respostas melhores:

- **Fase 0:** Rede de segurança (`pre-refactor`, CI `fmt`/`clippy`/`test`, snapshots de `--help`).
- **Fase 1:** Correção dos bugs críticos (`A1`–`A17`) com testes de regressão.
- **Fase 2:** Organização arquitetural (`lib.rs`, separação por domínios e divisão do `cli.rs`).
- **Fase 3:** Cérebro trocável (`ModelBackend` com `OllamaBackend`, `ClaudeBackend` e `NeuralBackend` + baseline no `eval`).
- **Fase 4:** Consolidação de interface (`app/`, `CliError`, `clap`, `enum Command`).
- **Fase 5:** Agente somente leitura (`Tool`, política determinística, log de auditoria e defesa contra prompt injection).
- **Fase 6:** Agente com ações confirmadas (isolamento, dry-run, confirmação interativa, edição com diff e ensaio em VM).

Os prazos são estimativas para cerca de 8 horas por semana. Se uma fase passar do dobro do tempo previsto, reduza o escopo dela em vez de estender o prazo.

## Backlog A: bugs e riscos críticos

São 17 itens, todos da Fase 1 (P0 e P1) ou de manutenção (P2). P0 = perde dados ou age errado sem avisar; P1 = comportamento errado visível; P2 = polimento. Faça na ordem da tabela.

| ID | Problema | Onde | Prio. | Esforço | Pronto quando | Status |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | Treino novo apaga `best.bin`, `latest.bin` e `metrics.jsonl` do diretório, e a maratona apaga os logs | `training/mod.rs`; `cli.rs` maratona | P0 | 2h | Um segundo treino no mesmo diretório recusa ou move para backup, com teste | Concluído |
| A2 | `evaluate_loss` devolve 0.0 sem tokens e congela o `best.bin`; `val_loss` NaN é ignorada | `training/mod.rs`; `cli.rs` estágio 2 | P0 | 2h | Retorna `Option<f32>`, NaN gera aviso, com teste | Concluído |
| A3 | `train-neural` ignora `cancel_flag` e força o handler de Ctrl+C | `main.rs` L49 | P0 | 30min | Usa `run_train_neural_cli_with_cancel` com os parâmetros do dispatch | Concluído |
| A4 | Mensagem que começa com `/` vira comando (ex.: `/var/log/...`) | `cli.rs` L50 | P1 | 1h | `/` só é comando se a palavra for conhecida; teste com `/var/log` | Concluído |
| A5 | `eval-neural` engole erro de dataset e de serialização | `cli.rs` | P1 | 30min | Dataset ausente sai com código 1 e mensagem clara | Concluído |
| A6 | `load_seed_corpus` aceita linha sem `prompt` ou `response` | `cli.rs` | P1 | 1h | Erro com o número da linha, com teste | Concluído |
| A7 | Maratona com `--resume` sem checkpoint treina do zero em silêncio | `cli.rs` | P1 | 1h | Falha com erro claro, como o modo normal | Concluído |
| A8 | O spinner soma até 500 ms a cada resposta e disputa a tela com o primeiro trecho | `cli.rs` L60–82 | P1 | 2h | Spinner com canal e `recv_timeout`; join antes do primeiro print | Não iniciado |
| A9 | Erro no meio do treino perde o progresso, e falha ao gravar métrica é fatal | `training/mod.rs` (loop) | P1 | 3h | Checkpoint de emergência; falha de métrica só avisa | Não iniciado |
| A10 | Checkpoints não são gravados de forma atômica | `neural/mod.rs` (`save_checkpoint` / `save_quantized_checkpoint`) | P1 | 2h | Grava em `.tmp`, `sync_all` e `rename` | Já implementado no código (validar cobertura de teste na Fase 1) |
| A11 | O guardrail é aplicado depois de o texto já ter sido impresso | `cli.rs` L84–103 | P1 | 1–2 dias | Decisão tomada: buffer antes de imprimir, ou aviso a posteriori declarado no banner | Não iniciado |
| A12 | `--resume` sem `--steps` falha citando um `--steps` de 80 que você não digitou | `cli.rs` | P2 | 1h | Mensagem clara ou padrão calculado | Não iniciado |
| A13 | O texto "em andamento" aparece como dispositivo no relatório | `cli.rs` | P2 | 15min | Status e dispositivo em parâmetros separados | Não iniciado |
| A14 | `--with-kb` não consta do help do `eval` | `cli.rs` | P2 | 15min | Help atualizado e coberto por snapshot | Não iniciado |
| A15 | A queda da GPU para CPU acontece sem aviso | `training/mod.rs` | P2 | 1h | Aviso com a causa | Não iniciado |
| A16 | A avaliação conta os tokens da cauda em dobro | `training/mod.rs` | P2 | 2h | Alvos já contados são mascarados, com teste | Não iniciado |
| A17 | `:exp add` aceita `verified` autodeclarado e já envia ao treino | `cli.rs` | P2 | 2h | Aprovação exige uma revisão separada | Não iniciado |

## Backlog B: arquitetura e qualidade de código

Estas tarefas não mudam o que o programa faz; deixam o código mais fácil de manter. Faça cada uma na fase indicada, e não antes: o plano depende dessa ordem.

| ID | Tarefa | Por quê | Fase | Esforço | Status |
| --- | --- | --- | --- | --- | --- |
| B1 | Git com tag `pre-refactor` e CI com `fmt`, `clippy` e `test` | Rede de segurança para tudo o que vem depois | 0 | 0,5 dia | Concluído |
| B2 | Snapshots de `--help` e dos códigos de saída do CLI | Detecta regressão ao mexer no `cli.rs` | 0 | 0,5 dia | Concluído |
| B3 | `lib.rs` e `main.rs` fino com `ExitCode` | API pública, testes de integração e destructors que rodam | 2 | 0,5 dia | Não iniciado |
| B4 | Mover o `CtrlCGuard` para `platform/signals` | O treino deixa de depender da avaliação | 2 | 1h | Não iniciado |
| B5 | Agrupar módulos em `chat/`, `eval/`, `neural/` e `platform/` | Fronteiras visíveis entre os três domínios | 2 | 1 dia | Não iniciado |
| B6 | Renomear `evaluation`, `infrastructure` e `experience` | Os nomes passam a dizer o que fazem | 2 | 0,5 dia | Não iniciado |
| B7 | Dividir o `cli.rs` em `repl`, `eval`, `train`, `marathon`, `quantize` e `hw` | Um arquivo de 3.900 linhas vira partes legíveis | 2 | 1 dia | Não iniciado |
| B8 | Resolver caminhos pelo diretório da config, e não pelo CWD | Mesmo resultado em qualquer pasta de execução | 2 | 1 dia | Não iniciado |
| B9 | Limpeza rápida: banner "Fase 0", `allow(dead_code)` obsoletos, defaults de `:train` (60) e CLI (80) | Higiene que evita confusão | 2 | 1h | Não iniciado |
| B10 | Camada `app/` com o currículo da maratona, `load_seed_corpus`, o recorder e a quantização | O CLI passa a só traduzir argumentos | 4 | 3 a 5 dias | Não iniciado |
| B11 | `CliError` com código de saída dentro do enum | Acaba com `String` e `i32` mágicos | 4 | 2 a 3 dias | Não iniciado |
| B12 | `clap` em `train-neural`, depois `eval-neural` e por fim `eval` | Help gerado e erros consistentes | 4 | 3 a 5 dias | Não iniciado |
| B13 | `enum Command` no REPL, com o `/help` gerado da tabela de comandos | Remove a duplicação do texto de ajuda | 4 | 1 a 2 dias | Não iniciado |
| B14 | `rustyline` no REPL | Histórico e setas | 4 | 0,5 dia | Não iniciado |
| B15 | Constantes e `Default` para hiperparâmetros; fim dos sentinelas `usize::MAX / 2` | Configuração legível | Depois | 1 dia | Não iniciado |
| B16 | `enum Backend` (CPU e GPU) e `StopCondition` no treino | Remove os `.expect` e cinco blocos repetidos; só vale quando o neural voltar | Depois | 3 a 5 dias | Não iniciado |
| B17 | Workspace com crates separados, começando por `neural` | Opcional: compila e evolui em separado | Depois | 1 a 2 dias | Não iniciado |

## Backlog C: o agente

O agente nasce somente leitura (Fase 5) e só ganha ações, sempre com confirmação, na Fase 6. A Fase 3 troca o cérebro sem mexer ainda em execução.

| ID | Entrega | Por quê | Fase | Esforço | Status |
| --- | --- | --- | --- | --- | --- |
| C1 | Interface `ModelBackend` (entram mensagens e ferramentas, sai texto ou chamada de ferramenta) com adaptadores `OllamaBackend` e `NeuralBackend` | O agente deixa de depender de um modelo específico | 3 | 2 a 3 dias | Não iniciado |
| C2 | `ClaudeBackend` com chave em variável de ambiente, `max_tokens`, timeout e limite de histórico | Respostas melhores com custo sob controle | 3 | 2 a 3 dias | Não iniciado |
| C3 | Redação de segredos antes de enviar a um backend remoto (tokens, senhas, chaves) e banner "backend remoto" | Logs de servidor podem conter segredos | 3 | 2 dias | Não iniciado |
| C4 | Regra "só loopback" vale só para o Ollama; backend remoto exige ativação explícita | Mantém a promessa de privacidade onde ela se aplica | 3 | 0,5 dia | Não iniciado |
| C5 | Comparar Ollama e Claude no `eval` e registrar o baseline | Decidir com dados, e não por impressão | 3 | 1 dia | Não iniciado |
| C6 | Trait `Tool` e ferramentas só leitura: `df`, `free`, `uptime`, `ss`, `systemctl status`, `journalctl` limitado, `docker ps` | Primeiras capacidades sem risco de alterar estado | 5 | 3 a 5 dias | Não iniciado |
| C7 | Política de permissões: allowlist, argumentos validados, sem shell livre (nada de `sh -c`) | A segurança não depende de o modelo acertar | 5 | 3 dias | Não iniciado |
| C8 | Log de auditoria só de acréscimo: proposto, aprovado, negado, executado, saída truncada | Rastreia tudo o que o agente faz | 5 | 2 dias | Não iniciado |
| C9 | Loop do agente: propor, checar política, executar, observar, com limite de passos e de tempo | O coração do agente | 5 | 3 a 5 dias | Não iniciado |
| C10 | Saída de comandos e de arquivos tratada como dado, nunca como instrução | Defesa contra prompt injection | 5 | 1 dia | Não iniciado |
| C11 | `eval` do agente: cenários em VM com resultado esperado e lista de comandos proibidos | Mede o agente, e não só as respostas | 5 | 3 a 5 dias | Não iniciado |
| C12 | Execução isolada: usuário sem root, timeout, limite de saída e VM de teste | Conter o estrago de um erro | 6 | 3 dias | Não iniciado |
| C13 | Confirmação interativa e modo dry-run para ações que alteram estado | Você aprova o comando exato antes de rodar | 6 | 2 dias | Não iniciado |
| C14 | Assistente de programação: ler e escrever arquivos em um diretório restrito, com diff antes de aplicar | Segunda metade do objetivo | 6 | 5 dias ou mais | Não iniciado |
| C15 | Execução remota por SSH com hosts em allowlist | Só depois de todo o resto estar sólido | 6 | 5 dias ou mais | Não iniciado |

**Princípios de segurança do agente**

- Leitura por padrão; tudo que altera estado exige a sua confirmação.
- A política de permissões é código determinístico, independente do que o modelo diga.
- Menor privilégio: usuário sem root, e `sudo` apenas para comandos específicos, se for preciso.
- Tudo vai para o log, inclusive o que foi negado.
- Ensaie em VM antes de tocar em qualquer servidor real.

## Trilha de pesquisa: modelo neural (congelado)

O modelo neural próprio fica congelado até o agente chegar à Fase 5. Até lá, entram só as correções A1, A2, A3 e A10, que evitam perda de dados.

**Condições para retomar**

- Fase 5 concluída, com o agente somente leitura estável.
- `eval` do agente funcionando e com baseline registrado.
- Tempo reservado de no máximo uma sessão por semana, sem tirar horas do agente.

**O que fazer quando voltar, em ordem**

1. Aplicar B15 e B16 (hiperparâmetros, `Backend` e `StopCondition`) antes de qualquer feature nova.
2. ~~Máscara de loss no fine-tuning: calcular a loss só nos tokens depois de `<assistant>`.~~ *(Já implementado no `SM-011` / commit `84367bf` via `pack_chat_masked_sequences` e `evaluate_masked_loss`)*
3. Schedule de learning rate por passos, e não por relógio, para o treino ser reproduzível.
4. Plugar o modelo como `NeuralBackend` e medir no mesmo `eval` dos outros.

**Expectativa realista:** um modelo treinado na sua máquina tem de alguns milhões a algumas dezenas de milhões de parâmetros. É um ótimo laboratório para aprender, mas não deve ser o cérebro do agente enquanto não superar os outros backends no `eval`.

## Testes e CI

Cada bug corrigido ganha um teste, e os testes novos protegem as refatorações das Fases 2 e 4. Os testes que você já tem (rollback, privacidade, `eval` ponta a ponta) são o maior ativo do projeto: não os apague.

| ID | Tarefa | Por quê | Fase | Status |
| --- | --- | --- | --- | --- |
| T1 | CI com `cargo fmt --check`, `cargo clippy -D warnings` e `cargo test` | Rede de segurança automática | 0 | Concluído |
| T2 | Snapshots de `--help` (geral, `eval`, `train-neural`, `eval-neural`) | Pega divergência entre help e parser (A14) | 0 | Concluído |
| T3 | Um teste de regressão por bug, de A1 a A17 | Bug corrigido continua corrigido | 1 | Não iniciado |
| T4 | Testes de tabela para os parsers de argumentos, válidos e inválidos | Base segura para trocar por `clap` | 2 | Não iniciado |
| T5 | Testes do REPL com writers injetados (comando conhecido, desconhecido e `/var/log`) | Cobre A4 e o `enum Command` | 2 | Não iniciado |
| T6 | Padronizar o lock de CWD com `unwrap_or_else(\|e\| e.into_inner())` ou usar `serial_test` | Um teste que falha não derruba os outros | 2 | Não iniciado |
| T7 | Pular os testes de arquivo somente leitura quando rodar como root | CI em Docker costuma rodar como root | 2 | Não iniciado |
| T8 | Dar folga ao teste de `--max-duration` ou injetar um relógio | Evita falha intermitente em máquina lenta | 2 | Não iniciado |
| T9 | Módulo `test_support` com `TempDirGuard` e um único mock HTTP; renomear `_cancel_server` | Menos duplicação e nomes honestos | 2 | Não iniciado |
| T10 | Mover o teste da maratona de `training` para os testes do `cli` | Camadas na ordem certa | 4 | Não iniciado |
| T11 | Testes do agente: comando negado, confirmação, auditoria e injeção na saída de ferramenta | Segurança comprovada, e não só declarada | 5 | Não iniciado |

## Pontos de atenção e riscos

Os riscos mais caros são privacidade e execução em servidor real. O resto é ritmo e escopo.

| Risco | Por que importa | O que fazer |
| --- | --- | --- |
| Segredos saindo da máquina | Com backend remoto, logs de servidor podem levar senhas, tokens e IPs | Redação antes de enviar (C3), ativação explícita (C4), banner de backend remoto, chave só em variável de ambiente e `.gitignore` |
| Comando errado em servidor real | Um comando mal escolhido pode derrubar um serviço | Leitura por padrão, confirmação, VM de ensaio, usuário sem root e log de auditoria (C7, C8, C12, C13) |
| Prompt injection | Saída de comando ou conteúdo de arquivo pode trazer instruções disfarçadas | Tratar como dado (C10); quem decide é a política, e não o modelo |
| Modelo errando com confiança | Qualquer modelo, até os grandes, erra | Mostrar o comando exato, exigir confirmação e medir no `eval` do agente (C11) |
| Custo da API | A cobrança por token pode surpreender | `max_tokens`, limite de histórico, contador de gasto e um teto mensal definido antes de começar |
| Perda de checkpoints de treino | Horas de treino somem sem aviso | A1, A9 e A10, mais backup antes de qualquer treino novo |
| Escopo crescendo | Voltar ao neural ou abrir frentes novas trava o agente | "Congelado é congelado", Caixa de ideias e uma fase por vez |
| Refatorar demais sem entregar | Meses arrumando e nada de utilizável | Cada fase termina em algo usável, com limite de tempo e critério "bom o suficiente" |
| Esgotamento | Projeto grande, tocado sozinho | Metas semanais pequenas, sessões de 2 a 3 horas, pausas e revisão na sexta |
| Depender de um único provedor | Preço, política ou indisponibilidade | `ModelBackend` mantém o Ollama como alternativa offline |
| Caminhos relativos ao CWD | O resultado muda conforme a pasta de execução | B8 na Fase 2 |

## Rotina semanal

Cada semana tem três momentos: planejar, trabalhar e revisar. O ritmo abaixo supõe cerca de 8 horas por semana; se o seu tempo for outro, ajuste o cronograma e não a qualidade.

1. **Segunda, 30 minutos: planejar.** Escolha até 3 tarefas da fase ativa e escreva a meta da semana em uma frase.
2. **Dias de trabalho: sessões de 2 a 3 horas.** Uma tarefa por sessão, testes verdes e um commit no final.
3. **Sexta, 30 minutos: revisar.** Rode `fmt`, `clippy` e `test`, marque o que foi feito nas tabelas e anote o que travou e o que você aprendeu.

**Quando travar**

- 45 minutos sem avanço: escreva a dúvida em uma frase e mude de tarefa ou peça ajuda.
- Vontade de "arrumar só mais isso": anote na Caixa de ideias e volte à tarefa.
- Cansaço: pare. Um commit pequeno com testes verdes vale mais que uma sessão longa e confusa.

**Como acompanhar**

- Tarefas concluídas por semana. Uma semana boa tem de 2 a 4.
- Itens do Backlog A fechados.
- Marcos concluídos no checklist final.
- Duas semanas seguidas sem avanço: reduza a meta da semana pela metade, e não aumente as horas.

## Decisões em aberto e marcos

Seis decisões ainda dependem de você, e sete marcos mostram quando cada fase está de fato concluída.

**Decisões em aberto**

- Agora: quantas horas por semana você consegue dedicar? Isso calibra o cronograma.
- Fase 3: qual provedor e modelo remoto usar, e qual o teto mensal de gasto?
- Fase 3: o backend remoto fica ativo por padrão ou só sob comando?
- Fase 5: o agente roda no próprio servidor ou na sua máquina, conectando por SSH?
- Fase 5: quais comandos entram na allowlist inicial?
- Depois da Fase 6: o neural volta ou continua como laboratório separado?

**Marcos.** Cada um só fecha quando o critério vale.

- [x] Fase 0, Rede de segurança: repositório com tag `pre-refactor`, CI verde e snapshots de help.
- [ ] Fase 1, Bugs críticos: A1 a A11 corrigidos, cada um com teste.
- [ ] Fase 2, Organização: `lib.rs`, pastas por domínio e `cli` dividido, com todos os testes antigos passando.
- [ ] Fase 3, Cérebro trocável: `ModelBackend` com Ollama e Claude, e baseline do `eval` registrado.
- [ ] Fase 4, Consolidação: `cli` sem lógica de negócio, `CliError` e `clap` nos subcomandos.
- [ ] Fase 5, Agente somente leitura: ferramentas, política, auditoria e testes de negação passando.
- [ ] Fase 6, Agente com ações: confirmação, execução isolada e um cenário ponta a ponta em VM.

**Caixa de ideias.** Anote aqui as ideias novas e só revise quando fechar um marco.

- Em `ClaudeBackend` (`C2`), considerar crate HTTP síncrona leve (`ureq` com TLS nativo) para manter compatibilidade simples sem exigir migração global para `async`/`tokio`.
- Quando retomar a trilha neural pós-Fase 5: limitar a duração da Fase 2 (`Chat Fine-Tuning`) em modelos >=100M para evitar overfitting em corpora Q&A compactos.
