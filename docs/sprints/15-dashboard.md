# Sprint 15 — Dashboard ao vivo

Roadmap: `docs/ROADMAP-v2.md`. Integração: branch `sprint-15` (sai de `sprint-12`;
reconcilia com `sprint-13` na integração). Groundwork já feito:
`frontend/confidence-timeline.js` (commit 239acde, branch `sprint-15-timeline`).

## Objetivo

Ver ao vivo, sem recalcular nada no navegador: confiança do ML nos últimos 30 s,
concordância regra × ML, saúde do pipeline (FPS, latência por estágio, frames
descartados) e o relatório do Sprint 10 servido pelo próprio servidor.

## Contrato (fonte única entre backend e frontend)

### `frame` ganha três campos

```json
{ "type": "frame", "...campos existentes...": "...",
  "rule_label": "arm_raised",
  "timings": { "capture_ms": 2.1, "pose_ms": 18.4, "features_ms": 0.3,
               "ml_ms": 1.2, "face_ms": null, "total_ms": 23.0 },
  "health": { "fps": 29.8, "dropped_frames": 3, "dropped_rate": 0.01 } }
```

- `rule_label`: calculado **no servidor** com a mesma função de
  `backend/ml/report.py::rule_label` (mover para módulo compartilhado se preciso,
  sem duplicar), usando o `open_threshold` do `GestureEngine`. `null` quando o
  frame não tem pose detectada. O navegador só compara strings.
- `timings`: medidos com `time.perf_counter()` em volta de cada estágio, no mesmo
  frame, em ms com 1 casa. Estágio que não rodou (ML sem modelo, face desligada)
  = `null`, nunca 0. `total_ms` = do frame capturado ao payload pronto.
- `health.dropped_frames`: frames que o subscriber coalesceu/descartou (contador
  desde a conexão daquele cliente; vem do mecanismo de coalescência do S12).
  `dropped_rate` = descartados / (enviados + descartados), 2 casas.
- `inference_ms` existente continua (compatibilidade).

### Decisões do maestro (dúvidas Bigorna × Pincel)

1. `rule_label = null` quando `detected=false` ou falta feature necessária ao
   predicado; `"idle"` só com pose detectada e nenhum predicado verdadeiro.
2. Fatias: `capture_ms` = leitura da câmera; `pose_ms` = landmarker (mesmo valor
   de `inference_ms`, que segue por compatibilidade); `features_ms` = features +
   motion + gestos; `ml_ms`; `face_ms`; `total_ms` = início da captura → payload
   montado (≥ soma das fatias não nulas).
3. `health.fps` repete `fps` de propósito: o bloco `health` é autocontido.
4. Primeiro frame de um cliente: `dropped_frames=0`, `dropped_rate=0.0` (nunca `null`).
5. `report_available` é snapshot só no `hello`; o link sempre tenta `/report` e
   o 404 mostra a mensagem — não há push de mudança.

### `hello` ganha `"report_available": true|false`

### `GET /report`

- 200 `text/html` com o último relatório gerado (`scripts/report.py`, caminho
  padrão `<model>.report.html`, configurável em `WebConfig.report_path`).
- 404 com corpo texto curto `"relatório ainda não gerado — rode scripts/report.py"`
  quando não existir. Nunca gera relatório na request. Leitura fora do event loop.

## Escopo por agente

### Backend — Bigorna (worktree `../wt/s15-backend`, branch `sprint-15-backend`)

Pode tocar: `backend/web/*.py` (exceto `recorder.py`), `backend/ml/report.py`
(só para extrair `rule_label` compartilhado), novo módulo compartilhado se
necessário, `tests/test_web.py`, `tests/test_report.py`, novo
`tests/test_dashboard.py`.

Não tocar: `frontend/`, `README.md`, `docs/`, `backend/web/recorder.py`.

Testes: `rule_label` no frame bate com `report.rule_label` nos casos do S10;
`timings` com `ml_ms`/`face_ms` `null` quando desligados; `health` coerente
com coalescência forçada; `/report` 200 e 404; `hello.report_available`.

### Frontend — Pincel (worktree `../wt/s15-timeline`, branch `sprint-15-timeline`)

Pode tocar: `frontend/index.html`, `app.js`, `style.css`,
`confidence-timeline.js` (+ test), novo `frontend/health-panel.js` (+ test).

Não tocar: `backend/`, `tests/`, `README.md`, `docs/`, `studio-controls.js`.

- Timeline: **uma** instância global em `app.js` (há uma câmera; não por aba).
  Linha = probabilidade do rótulo top do ML; faixa fina abaixo colorida por
  frame (concorda / discorda / sem ML), não um 0/1 desenhado como linha.
  Lacunas visíveis. Taxa de concordância numérica ao lado.
- Painel de saúde (`health-panel.js` puro): FPS, barras por estágio de
  `timings` (`null` = "—", nunca 0), `dropped_rate` com alerta acima de 0.10.
- Link "Relatório" que abre `/report` em nova aba; desabilitado com dica quando
  `hello.report_available=false`.
- `app.js` nunca calcula feature; só exibe números do servidor.

## Aceite (gates)

- `uv run pytest -q` e `node --test frontend/` verdes.
- Latência por estágio visível ao vivo (checkpoint com câmera: Victor).
- Diff de cada agente dentro do escopo.
