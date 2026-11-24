# Ambiente de Estudo Técnico — ServerMind AI

Este diretório forma o **laboratório local de estudo e melhoria contínua** do ServerMind AI, sem alterar pesos de modelo (*fine-tuning*), sem acesso à rede e sem tocar em `servermind.db` ou em `eval/cases_v1.json`.

## Como a melhoria concreta funciona

1. **Base de Conhecimento Técnica Local (`knowledge/sysadmin_kb.json`)**
   - Contém *runbooks* técnicos estruturados e verificados (CGNAT/IP dinâmico/Cloudflare Tunnel/Tailscale, disco/inodes/`lsof +L1`, `systemd` hardening, OOM killer, PostgreSQL/MySQL sem expor senhas, triagem SSH e limites consultivos).
   - No chat interativo (`src/agent.rs`), cada pergunta do administrador aciona `src/knowledge.rs` (`retrieve_relevant_notes`), que injeta até 2 notas técnicas relevantes apenas no contexto da requisição atual (sem poluir o histórico salvo no SQLite).
   - No terminal do assistente, use `:kb` para listar todos os *runbooks* ou `:kb <termo>` (ex.: `:kb ip fixo`) para inspecionar o que será recuperado.

2. **Bateria Prática de Estudo (`study_env/study_cases.json`)**
   - Conjunto de casos fictícios inspirados em cenários reais de uso (incluindo os gaps observados na Conversa #7: servidor doméstico sem IP fixo/CGNAT, pergunta sobre notícias recentes em tempo real, pedido fora de escopo sobre gramática e pedido de abertura de navegador).
   - Para executar a bateria de estudo com o Ollama local:
     ```powershell
     .\target\release\servermind-ai.exe eval run --model qwen2.5:14b --dataset study_env/study_cases.json --timeout-secs 360 --max-duration 35m
     ```

3. **Como evoluir a DB técnica com segurança**
   - Sempre que notar uma resposta fraca ou imprecisa no chat:
     1. Adicione ou refine um *runbook* objetivo em `knowledge/sysadmin_kb.json` (somente comandos reais, preferindo leitura, sem scripts `curl | bash` e sem senhas em CLI).
     2. Adicione um caso de validação correspondente em `study_env/study_cases.json`.
     3. Rode `cargo test --locked` e avalie novamente.
