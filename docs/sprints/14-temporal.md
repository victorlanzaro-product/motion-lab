# Sprint 14 — ML temporal e gravação de wave

Integração em `sprint-14` sobre a S13. Objetivo do [Roadmap v2](../ROADMAP-v2.md): usar janelas temporais e Random Forest para distinguir um aceno de um braço que sobe. O motor de gestos por regras continua ativo durante esta sprint.

## Treino, bundle e inferência

- Schema 1 permanece o padrão e aceita bundles anteriores sem `schema_version`. O schema 2 usa `FEATURE_NAMES` do último frame, seguidas das 14 estatísticas de janela de `WINDOW_TRAINING_COLUMNS`, na ordem do builder de treino.
- Treino e inferência compartilham a montagem da linha temporal em `backend/dataset/windows.py` e o mesmo `WindowFeatures`. O bundle v2 registra `schema_version: 2`, `window_seconds`, `max_gap`, `min_samples` e `deadband` usados no treino. Os padrões são 1 s, 0,30 s, 4 amostras e 0,35 larguras de ombro por segundo, respectivamente. Valide valores no carregamento. A ordem de colunas do bundle é parte do contrato.
- Um tracker por stream, com o timestamp monotônico do frame e reinício na perda de pose e na reabertura do stream. A janela está indisponível durante aquecimento, lacuna inválida ou perda de pulso. O v2 só chama o modelo com o vetor completo dos dois braços. Sem modelo compatível ou vetor completo, o frame mantém `ml: null` e pose, gestos, face e gravação continuam.
- `hello_web.py --model-schema {1,2}` seleciona o schema. `--model` e `--dataset` permitem substituir os caminhos. Em schema 1, os padrões permanecem `backend/models/gesture_classifier.joblib` e `data/training/dataset.csv`. Em schema 2, use `backend/models/gesture_classifier_v2.joblib` e `data/training/dataset_v2.csv` por padrão. Use `WebConfig.dataset_path` para a gravação e a cobertura.
- `load_model(path, schema=...)` rejeita bundle de outro schema. O servidor anuncia o modo configurado mesmo sem bundle porque é preciso gravar `wave` antes do primeiro treino. `ml_classes` vem do bundle compatível ou é `[]` quando indisponível.

## WebSocket e cobertura

Trechos do `hello` sem modelo treinado. Os demais campos continuam como na S13.

```json
{"type":"hello","model_schema":1,"ml_classes":[],"record_labels":["arm_raised","arms_crossed","arms_open","idle"]}
```

```json
{"type":"hello","model_schema":2,"ml_classes":[],"record_labels":["arm_raised","arms_crossed","arms_open","idle","wave"]}
```

O frontend usa essa ordem, mostra `wave` e habilita a tecla 5 quando `model_schema === 2` e `record_labels` contém `wave`. Sem o campo de schema, preserve os quatro rótulos da S13. O Recorder e o `DatasetWriter` usam a mesma lista de rótulos habilitados pelo servidor. A validação no servidor recusa `wave` em modo 1, sem abrir arquivo. Em modo 2 a gravação é:

```json
{"type":"record_start","label":"wave"}
```

```json
{"type":"record_status","recording":true,"label":"wave","session_id":"web-20261119-223000-95ab23","written":0,"skipped":0,"owner":true,"error":null}
```

O CSV mantém exatamente `DATASET_COLUMNS`, incluindo timestamp, índice e sessão por gravação. Uma gravação continua a ter um dono. `GET /dataset/stats` mantém o formato da S13 e usa os rótulos do modo ativo. Em v2, `labels` tem a entrada adicional `"wave": {"rows":0,"sessions":0,"seconds":0.0}` enquanto não houver amostras. Cobertura mede frames gravados e sessões. O treino valida janelas completas e sua cobertura separadamente.

## Escopo

- Pendulo, treino e relatório: `backend/features/window.py`, `backend/dataset/windows.py`, `backend/ml/{__init__,train,report}.py`, `scripts/{train_model,report}.py`, testes correspondentes. Integração de inferência, recorder e cobertura: adicionalmente `backend/web/{app,payload,pipeline,recorder}.py`, `backend/dataset/{writer,stats}.py`, `scripts/hello_web.py` e os testes correspondentes.
- Pincel, frontend: `frontend/{app,index,style,recorder,dataset-coverage}` com as extensões existentes e testes da lógica modificada. Sem cálculo temporal no frontend.
- Maestro: spec, integração, resolução de conflitos, roteiro e registro de decisões no README.

## Aceite

- Gates Python e Node, diff no escopo e smoke sintético por CLI. Casos de teste: warm-up, lacuna, pose perdida, reconexão, compatibilidade v1, bundle incompatível, schema 2 sem modelo, gravação de `wave`, CSV de mesmo schema e teste em que o histórico distingue aceno de subida com o mesmo frame final.
- Victor grava pelo menos três sessões de ~15 s de `wave` em luz e distância variadas, com ambos os pulsos visíveis. Alterne o braço de uma sessão para outra e faça reversões repetidas com o braço ativo. Repita sessões de `arm_raised` se elas não contiverem subidas sem reversão. Use uma cópia do CSV postural da S13 como ponto de partida do dataset v2 e mantenha os artefatos de treino locais.
- Relatório compara v1 e v2 lado a lado com classes, quantidade de sessões e dobras utilizadas. Avalie `wave` e `arm_raised` no v2 com sessões inteiras fora do treino, uma por dobra. Registre F1 por classe e confusões com predições fora da amostra, sem atribuir ao v1 reconhecimento de uma classe ausente do seu vocabulário.
