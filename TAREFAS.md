# Fila de tarefas

Cada tarefa usa estes campos fixos: **ID**, **objetivo**, **critério de pronto**, **status** e **branch**.

Status permitidos: `aberta`, `em andamento`, `pronta para revisão`, `aprovada`.

## T1

- **ID:** T1
- **objetivo:** cobrir com um teste determinístico o lado direito do extrator de features quando o punho direito está acima do ombro, usando uma pose sintética.
- **critério de pronto:** em `tests/test_features.py`, criar uma pose sintética baseada no fixture `standing()` com `RIGHT_WRIST` acima do ombro direito e verificar por `extract()` que `right_wrist_above_shoulder` é `True`, `right_wrist_height > 0`, enquanto o braço esquerdo pendente continua `False` e com altura negativa. `uv run pytest -q tests/test_features.py` passa sem câmera, modelo ou vídeo.
- **status:** aberta
- **branch:** `t1-feature-right-wrist`
