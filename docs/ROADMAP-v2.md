# Motion Lab — Roadmap v2 (Sprints 12–16)

Roadmap v1 (Sprints 01–11) fechado em `ac5f523`. Este ciclo ataca três frentes,
na ordem em que uma desbloqueia a outra:

1. **Produto/UX** — gravar dataset pelo navegador (hoje só via janela OpenCV).
2. **Dataset real + treino** — sair do `--simulate`, medir generalização de verdade.
3. **ML temporal** — janela temporal no vetor de ML; `wave` volta a ser treinável
   (resolve a "Limitação conhecida" do README).

Princípios do v1 continuam lei: junta oculta vira `None`, nunca `0`; MediaPipe
preso em `backend/vision/`; nenhum pixel persistido; camadas opcionais degradam
sem derrubar o pipeline; cada sprint documenta suas "Decisões" no README.

---

## Orquestração

**Maestro (Claude, sessão principal):** escreve spec + contrato de cada sprint,
cria worktrees, despacha agentes, integra, revisa, abre PR. Nunca mergeia em
`main` — o merge é do Victor.

| Papel | Agente | Escopo | Como roda |
|-------|--------|--------|-----------|
| Arquiteto / integrador / revisor | Claude (maestro) | spec, contrato, integração, README | sessão principal |
| Backend + ML | Codex | `backend/`, `scripts/`, `tests/` | `codex exec --full-auto -C <worktree> "<prompt>"` |
| Frontend | Cursor Agent | `frontend/` | `cursor-agent -p --force "<prompt>"` (cwd = worktree) |
| Revisão independente | Claude subagente (`ecc:code-reviewer`) + `codex exec review` | diff da sprint | após integração |

### Ciclo por sprint

1. **Spec** — maestro escreve `docs/sprints/NN-<nome>.md`: objetivo, critérios de
   aceite, **contrato** (mensagens WebSocket / assinaturas Python / colunas) e
   arquivos que cada agente pode tocar.
2. **Branches** — `sprint-NN` (integração) + worktrees
   `../wt/sNN-backend` (`sprint-NN-backend`) e `../wt/sNN-frontend`
   (`sprint-NN-frontend`), ambos a partir de `sprint-NN`.
3. **Execução paralela** — Codex e Cursor rodam em background, cada um só no
   seu diretório. O contrato é a única interface entre eles.
4. **Integração** — maestro mergeia as duas branches em `sprint-NN`, resolve
   conflito, roda os gates.
5. **Revisão** — dois revisores independentes; achados bloqueadores voltam
   pro agente dono do arquivo.
6. **PR** — `sprint-NN` → `main`, com evidência dos gates. Victor aprova.

### Gates (obrigatórios antes do PR)

- `uv run pytest -q` verde
- `node --test frontend/` verde
- `check_*.py --simulate` (ou equivalente sem câmera) da sprint roda limpo
- README: linha na tabela de Status + seção "Decisões do Sprint NN"
- Nenhum arquivo fora do escopo do agente alterado (checado no diff)

### Limite honesto

Agentes não têm webcam. Qualquer critério que dependa de câmera real (gravação
real, latência ao vivo, acurácia em sessão real) é **checkpoint humano**:
maestro prepara o roteiro, Victor executa, maestro analisa o resultado.

---

## Sprint 12 — Studio de Gravação no Navegador

**Por quê:** gravar hoje exige `hello_dataset.py` (janela OpenCV, tecla
segurada, auto-repeat do SO). Dataset real (Sprint 13) depende de gravar ser fácil.

**Backend (Codex)**
- Mensagens WebSocket cliente→servidor: `record_start {label}`, `record_stop`.
- Servidor→cliente: `record_status {recording, label, session_id, written, skipped}`.
- `PipelineRunner` reusa `DatasetWriter` (sem nova lógica de validação);
  `session_id` gerado por sessão de gravação no servidor.
- Rótulos permitidos = `DEFAULT_LABELS` (recusa `wave` até Sprint 14 — mesmo
  `REMOVED_LABELS`). Múltiplas abas: só uma gravação ativa por vez (lock).
- Testes em `tests/test_web.py` com câmera/landmarker fakes.

**Frontend (Cursor)**
- Painel "studio": botões por rótulo, contador ao vivo (escritas / descartadas),
  contagem regressiva de 3 s antes de gravar, atalhos 1–4.
- Indicador claro de "GRAVANDO" (contrato de privacidade, igual ao LED).
- Testes `node --test` pra lógica pura (estado do painel), sem DOM real.

**Aceite:** gravar sessão de `arm_raised` pelo navegador gera linhas válidas no
`dataset.csv`, idênticas em schema às do `hello_dataset.py`.

## Sprint 13 — Dataset Real + Baseline Honesto

**Checkpoint humano:** Victor grava ≥3 sessões × 4 rótulos posturais
(`arm_raised`, `arms_crossed`, `arms_open`, `idle`), variando distância e luz.
Maestro entrega roteiro de gravação antes.

**Backend (Codex)**
- `scripts/dataset_stats.py`: classes × sessões, linhas descartadas por motivo,
  duração por sessão — falha se alguma classe tiver < 2 sessões.
- Validação cruzada agrupada por sessão (`GroupKFold`/leave-one-session-out)
  no `train()`, reportando média ± desvio de macro-F1, não só um split.
- Relatório (`report.py`) ganha seção de estatísticas do dataset.

**Frontend (Cursor)** — opcional: painel de "cobertura do dataset" no studio
(quantas sessões por rótulo já existem), lido de endpoint novo `GET /dataset/stats`.

**Aceite:** modelo treinado em dado real, macro-F1 por leave-one-session-out
documentado no README (número real, seja ele bom ou ruim).

## Sprint 14 — ML Temporal (janela) + `wave` treinável

**Decisão de abordagem:** features de janela + Random Forest, não LSTM.
Mantém a explicabilidade do Sprint 10 (`feature_importances_`), roda em CPU,
treina com dataset pequeno. LSTM só se RF com janela falhar no critério.

**Backend (Codex)**
- `backend/features/window.py`: `WindowFeatures` sobre os últimos N s
  (default 1.0 s, por segundo, nunca por frame — regra do Sprint 04):
  média/desvio/amplitude de `wrist_height`, nº de reversões de direção,
  velocidade máxima, fração do tempo em movimento. Janela com buraco >
  `max_gap` é inválida (`None`), igual ao `MotionTracker`.
- Dataset: janelas reconstruídas **do CSV existente** (`session_id`,
  `frame_index`, `timestamp` já existem) — sem mudar schema de gravação.
- Bundle `.joblib` ganha `schema_version: 2` + `window_seconds`;
  `load_model` recusa misturar v1/v2 com erro explícito.
- `wave` sai de `REMOVED_LABELS` **só** para o modelo v2; `PipelineRunner`
  mantém buffer de janela pra inferência ao vivo.
- Teste-chave: aceno sintético vs. braço subindo sem repetir — frames
  individuais idênticos, janelas separáveis.

**Frontend (Cursor)** — tecla 5 / botão `wave` no studio quando servidor
anunciar `model_schema: 2` no `hello`.

**Checkpoint humano:** gravar sessões de `wave`.

**Aceite:** RF v2 separa `wave` de `arm_raised` em leave-one-session-out;
relatório compara v1 × v2 lado a lado.

## Sprint 15 — Dashboard ao Vivo

**Frontend (Cursor)**
- Timeline de confiança do ML (últimos 30 s) + linha de concordância regra × ML.
- Relatório do Sprint 10 servido em `/report` (hoje HTML estático solto).
- Painel de saúde: FPS, latência por estágio, taxa de frames descartados.

**Backend (Codex)**
- Payload ganha `timings` por estágio (câmera, pose, features, ML) — medido,
  não estimado.
- `GET /report` serve o último relatório gerado; `404` limpo se não existir.

**Aceite:** latência por estágio visível ao vivo; nada recalculado no browser
(regra do Sprint 05: `app.js` nunca calcula feature).

## Sprint 16 — Hardening + Avaliação ao Vivo

- Sessão ao vivo roteirizada (checkpoint humano) medindo concordância v1 × v2 ×
  regra; números no README.
- Revisão de arquitetura do ciclo inteiro (mesma prática do pré-Sprint 11).
- Decisão documentada: aposentar `_WaveDetector` por regra ou manter os dois.

---

## Dependências

```
S12 (studio) ──> S13 (dataset real) ──> S14 (temporal) ──> S16 (avaliação)
                                    └──> S15 (dashboard) ──┘
```

S15 pode começar em paralelo com S14 (só toca payload/frontend).

## Riscos

| Risco | Mitigação |
|-------|-----------|
| Agentes editam fora do escopo | escopo de arquivos no prompt + checagem do diff no gate |
| Contrato WebSocket diverge entre backend e frontend | contrato escrito antes, testes dos dois lados contra o mesmo exemplo JSON em `docs/sprints/` |
| Dataset real pequeno demais → métricas sem sentido | `dataset_stats.py` bloqueia treino; reportar desvio, não só média |
| Python 3.13 no ambiente do agente | `uv venv --python python3.12` em todo worktree |
