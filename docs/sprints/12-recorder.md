# Sprint 12 — Recorder: gravação de dataset pelo navegador

Roadmap: `docs/ROADMAP-v2.md`. Integração: branch `sprint-12`.

## Objetivo

Gravar amostras rotuladas em `data/training/dataset.csv` direto do navegador,
sem `scripts/hello_dataset.py` (janela OpenCV + tecla segurada + auto-repeat do
SO). Linhas gravadas pelo navegador são **byte a byte o mesmo schema**
(`DATASET_COLUMNS`) que as do `hello_dataset.py` — mesma `DatasetWriter`, nenhuma
validação nova duplicada.

Nome: **Recorder** (o nome "Studio" já existe em `frontend/studio-controls.js`
para presets visuais — não confundir, não tocar nele).

## Contrato (fonte única entre backend e frontend)

Todas as mensagens são JSON texto no WebSocket `/ws` existente.

### `hello` (servidor → cliente) ganha um campo

```json
{ "type": "hello", "...campos existentes...": "...",
  "record_labels": ["arm_raised", "arms_crossed", "arms_open", "idle"] }
```

`record_labels` = `list(backend.dataset.writer.DEFAULT_LABELS)`, nunca
hardcoded no frontend.

### Cliente → servidor

```json
{ "type": "record_start", "label": "arm_raised" }
{ "type": "record_stop" }
```

Qualquer outra mensagem ou JSON inválido: ignorado (log em debug), conexão
continua aberta. Os toggles visuais seguem 100% client-side como hoje.

### Servidor → cliente: `record_status` (broadcast a todas as abas)

```json
{ "type": "record_status",
  "recording": true,
  "label": "arm_raised",
  "session_id": "web-20261119-153012-a1b2c3",
  "written": 42,
  "skipped": 3,
  "owner": true,
  "error": null }
```

- `owner`: `true` só na aba que é dona da gravação **ativa** (calculado por
  subscriber). Sem gravação ativa, `owner` é sempre `false` — inclusive numa
  rejeição (rótulo inválido/`wave` com o servidor ocioso:
  `{recording:false, label:null, session_id:null, owner:false, error:"..."}`).
  `recording`/`label`/`session_id` numa rejeição sempre refletem o estado real
  do servidor, nunca o pedido recusado.
- Rejeições (regras 1, 2, 4) vão **só** para o solicitante; o frontend trata
  `error != null` como mensagem para esta aba e **não** deve usar uma rejeição
  para apagar/alterar o estado de gravação de outra aba (o estado vem dos
  campos `recording`/`label`/`session_id`, que são verdadeiros).
- `label`/`session_id`: `null` quando `recording=false`, **exceto** na mensagem
  de parada, que mantém os valores da sessão que acabou de fechar + contagens
  finais (o painel mostra o resumo).
- `error`: `null`, ou texto legível quando a gravação foi recusada/abortada.
- Enviado: na conexão (estado atual, logo após o `hello`), em toda mudança de
  estado, e no máximo a cada 0.25 s enquanto gravando (contagens).
- A fila de controle tem limite de 64 mensagens por assinante. Contagens e
  rejeições repetidas podem ser agrupadas. Transições, finais e erros fatais
  têm prioridade. Se a fila estiver cheia apenas com mensagens prioritárias,
  o servidor fecha o WebSocket com 1013. A reconexão recebe o estado atual.

### Regras

1. **Uma gravação ativa por servidor.** `record_start` com outra gravação em
   curso → `record_status` com `error` só para o solicitante; a gravação
   ativa não é afetada.
2. **Só o dono para.** `record_stop` de quem não é dono → ignorado com `error`
   só para o solicitante.
3. **Parada automática:** dono desconecta; stream da câmera termina (idle,
   falha, retry); `DatasetWriterError` ao escrever. Em todos os casos o
   arquivo é fechado e um `record_status` final (`recording=false`) é
   enviado a quem ainda estiver conectado.
4. **Rótulo inválido / `wave`:** recusado *antes* de abrir o arquivo, com a
   mensagem do próprio `DatasetWriter`/`REMOVED_LABELS` em `error`.
5. **`session_id`** gerado no servidor por gravação:
   `web-YYYYMMDD-HHMMSS-<6 hex>`. Cada `record_start` = sessão nova (o split
   agrupado do Sprint 08 depende disso).
6. **Escrita na thread do pipeline**, no mesmo laço de `_stream`, com o mesmo
   `features`/`motion` daquele frame. Comandos chegam do event loop por
   estrutura thread-safe (lock/fila) — nunca bloquear o event loop com I/O
   de arquivo.
7. **Nunca derruba o pipeline.** Qualquer erro do Recorder vira `error` no
   `record_status` e para a gravação; pose/gestos/ML/face continuam.
8. Frame incompleto → `skipped` (já é o comportamento de `DatasetWriter.write`).
9. `WebConfig.dataset_path` (default `DEFAULT_DATASET_PATH`) — testes usam `tmp_path`.
10. `/health` ganha `"recording": {recording, label, session_id, written, skipped}`.

## Escopo por agente

### Backend — Codex (worktree `../wt/s12-backend`, branch `sprint-12-backend`)

Pode tocar: `backend/web/*.py`, `backend/dataset/*.py` (só se estritamente
necessário — preferir não), `tests/test_web.py`, novo `tests/test_recorder.py`,
novo `backend/web/recorder.py` (sugerido: lógica do Recorder isolada do
`PipelineRunner`, testável sem WebSocket).

Não tocar: `frontend/`, `README.md`, `docs/`.

Testes obrigatórios (câmera/landmarker fakes já existentes em `tests/test_web.py`):
- start → N frames → stop: CSV em `tmp_path` com header `DATASET_COLUMNS`,
  N linhas (menos skipped), todas com o mesmo `session_id` e o rótulo pedido.
- dois starts seguidos (stop no meio) → dois `session_id` distintos.
- `wave` e rótulo desconhecido → `error`, arquivo nem criado.
- segunda aba tenta start durante gravação → `error` só pra ela; contagens da
  primeira seguem.
- dono desconecta → gravação para, status final enviado à outra aba.
- `DatasetWriterError` forçado → gravação para, pipeline continua mandando `frame`.
- `hello.record_labels == list(DEFAULT_LABELS)`.

### Frontend — Cursor (worktree `../wt/s12-frontend`, branch `sprint-12-frontend`)

Pode tocar: `frontend/index.html`, `frontend/app.js`, `frontend/style.css`,
novo `frontend/recorder.js` + `frontend/recorder.test.js`.

Não tocar: `backend/`, `tests/`, `README.md`, `docs/`, `studio-controls.js`.

- `recorder.js`: lógica pura sem DOM (mesmo padrão IIFE de
  `studio-controls.js`/`face-history.js`): máquina de estados
  `idle → countdown(3s) → recording → idle`, reduzindo mensagens
  `record_status` e ações do usuário; cancelar contagem não envia nada.
- Painel "gravação": um botão por `hello.record_labels` (atalhos 1..N),
  botão parar (atalho espaço), contadores `written`/`skipped` ao vivo,
  resumo da última sessão, `error` visível.
- Indicador **GRAVANDO** inconfundível (vermelho, persistente) sempre que
  `recording=true` — inclusive nas abas que não são donas ("outra aba está
  gravando"; botões desabilitados).
- `app.js` nunca calcula feature (regra do Sprint 05).
- `recorder.test.js` (`node --test`) cobre a máquina de estados contra os
  exemplos JSON deste contrato.

## Aceite (gates)

- `uv run pytest -q` verde (baseline: 189 passed).
- `node --test frontend/` verde (baseline: 34 pass).
- Diff de cada agente só dentro do escopo acima.
- Checkpoint humano (Victor): `hello_web.py --open`, gravar 10 s de
  `arm_raised`, conferir linhas no CSV.
