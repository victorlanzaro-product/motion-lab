# Sprint 13 — Dataset real + baseline honesto

Roadmap: `docs/ROADMAP-v2.md`. Integração: branch `sprint-13` (sai de `sprint-12`).

## Objetivo

Saber, com número honesto, quanto o modelo v1 acerta em dado real. Hoje
`train()` mede num único `GroupShuffleSplit`; com poucas sessões, um split
sortudo mente. Este sprint troca a métrica principal por validação cruzada
agrupada por sessão e torna o dataset inspecionável (script, endpoint,
relatório, painel de cobertura).

## Contrato (fonte única entre backend e frontend)

### `GET /dataset/stats` → 200 `application/json`

```json
{ "path": "data/training/dataset.csv",
  "exists": true,
  "total_rows": 1840,
  "usable_rows": 1795,
  "dropped": { "missing_feature": 40, "unknown_label": 5 },
  "labels": {
    "arm_raised":   { "rows": 512, "sessions": 3, "seconds": 41.2 },
    "arms_crossed": { "rows": 470, "sessions": 3, "seconds": 38.0 },
    "arms_open":    { "rows": 401, "sessions": 2, "seconds": 33.5 },
    "idle":         { "rows": 412, "sessions": 1, "seconds": 34.9 }
  },
  "sessions": [
    { "session_id": "web-20261119-153012-a1b2c3", "label": "arm_raised",
      "rows": 180, "seconds": 14.1 }
  ],
  "min_sessions_per_label": 2,
  "ready": false,
  "problems": ["idle: 1 sessão (mínimo 2)"] }
```

- `labels` sempre contém **todos** os `DEFAULT_LABELS` (zero se ausente:
  `{rows:0, sessions:0, seconds:0.0}`), na ordem de `DEFAULT_LABELS`.
- `exists:false` → arquivo ausente: `total_rows=0`, `labels` zerados,
  `sessions=[]`, `ready=false`, `problems=["dataset ausente"]`. Nunca 500.
- `seconds` = soma por sessão de `max(timestamp) - min(timestamp)`, 1 casa.
- `dropped`: chaves = motivos que o próprio loader de `train.py` já aplica
  (não inventar validação nova; reaproveitar a mesma função). Só motivos com
  contagem > 0.
- **Decisão (rótulo desconhecido):** a validação por linha do loader é
  extraída para uma função compartilhada que *classifica* a linha (ok ou
  motivo). `load_dataset`/`train()` mantêm o comportamento atual: rótulo
  desconhecido continua `DatasetError` (treino recusa). `dataset_stats()`
  usa a mesma função mas **conta** em `dropped.unknown_label`, nunca lança;
  qualquer `unknown_label > 0` força `ready=false` com problema
  `"N linhas com rótulo desconhecido (treino vai recusar)"`. Rótulos
  desconhecidos não entram em `labels` nem em `sessions`.
- `ready` = toda classe de `DEFAULT_LABELS` com ≥ `min_sessions_per_label`
  sessões. `problems`: frases legíveis em português, uma por violação.
- Lido do disco a cada request, fora do event loop (`run_in_threadpool`/
  `asyncio.to_thread`).
- O frontend **nunca** recalcula nada disso — só exibe.

### Fonte única

`backend/dataset/stats.py::dataset_stats(path) -> dict` produz exatamente o
JSON acima. `scripts/dataset_stats.py`, `GET /dataset/stats` e a seção do
relatório chamam essa mesma função.

## Escopo por agente

### Backend — Codex (worktree `../wt/s13-backend`, branch `sprint-13-backend`)

Pode tocar: novo `backend/dataset/stats.py`, novo `scripts/dataset_stats.py`,
`backend/ml/train.py`, `backend/ml/report.py`, `scripts/train_model.py`,
`scripts/report.py`, `backend/web/app.py` (só a rota nova), novo
`tests/test_dataset_stats.py`, e `tests/test_train.py`, `tests/test_report.py`,
`tests/test_web.py`.

Não tocar: `frontend/`, `README.md`, `docs/`, `backend/web/recorder.py`,
`backend/web/pipeline.py`.

1. `dataset_stats()` + `scripts/dataset_stats.py` (tabela legível; `--json`
   imprime o dict; exit code 1 se `ready=false`).
2. **Validação cruzada agrupada** em `train()`:
   - `GroupKFold(n_splits=min(5, n_sessões))`. Dobra que deixa alguma classe
     sem amostra de treino é registrada como inválida e excluída da média,
     com contagem explícita (`folds_used`, `folds_skipped`). Zero dobras
     válidas → `DatasetError` explícito.
   - `TrainingReport` ganha `cv: {n_splits, folds_used, folds_skipped,
     macro_f1_mean, macro_f1_std, per_fold: [...]}`.
   - Modelo final salvo continua treinado como hoje; métrica de manchete
     passa a ser `macro_f1_mean ± macro_f1_std`. O split único antigo segue
     no relatório como detalhe (matriz de confusão).
   - Determinístico: mesma entrada → mesmo `cv`.
   - **Decisão de integração:** mantenha GroupKFold como padrão do Sprint 13 e
     ofereça `--cv loso` no CLI de treino para o aceite do roadmap. Nesse
     modo, cada dobra segura uma sessão inteira. Registre o macro-F1 das
     predições reunidas fora da amostra no conjunto fixo de quatro rótulos,
     além do resumo e da contagem de dobras. Uma sessão pode ter só um rótulo,
     então a média de F1 das dobras responde a uma pergunta diferente do
     macro-F1 calculado sobre todas as sessões previstas.
3. `report.py`: seções "Dataset" (por rótulo, sessões, problemas) e
   "Validação cruzada" (média ± desvio, por dobra). Sem dependência nova.
4. `GET /dataset/stats` usando `WebConfig.dataset_path` (já existe no S12).
5. Testes: dataset sintético em `tmp_path` com N sessões; arquivo ausente;
   classe com 1 sessão → `ready=false` + problema; `dropped` coerente com o
   loader; CV determinístico; dobra inválida contada; rota HTTP.

### Frontend — Cursor (worktree `../wt/s13-frontend`, branch `sprint-13-frontend`)

Pode tocar: `frontend/index.html`, `frontend/app.js`, `frontend/style.css`,
novo `frontend/dataset-coverage.js` + `frontend/dataset-coverage.test.js`.

Não tocar: `backend/`, `tests/`, `README.md`, `docs/`, `studio-controls.js`.

- Painel "Cobertura do dataset" junto ao painel de gravação: por rótulo,
  sessões / mínimo e segundos; marcação clara de rótulo abaixo do mínimo;
  `problems` listados; estado `exists:false` amigável.
- Busca `GET /dataset/stats` ao carregar e após cada `record_status` de
  parada (`recording:false` com `session_id != null`). Falha de rede = aviso
  discreto, nunca quebra o resto da UI.
- `dataset-coverage.js`: lógica pura (IIFE, sem DOM) que transforma o JSON
  em view-model; testes contra os exemplos deste contrato (inclusive
  `exists:false` e rótulo com 0 sessões).

## Checkpoint humano (Victor)

Roteiro: `docs/sprints/13-roteiro-gravacao.md`.

## Aceite (gates)

- `uv run pytest -q` e `node --test frontend/` verdes.
- `uv run python scripts/dataset_stats.py` no dataset real → `ready=true`.
- `uv run python scripts/train_model.py --cv loso` → macro-F1 com uma sessão
  inteira de teste por dobra,
  documentado no README (número real, bom ou ruim).
- Diff de cada agente dentro do escopo.
