# Motion Lab

Computer vision playground rodando 100% local: webcam → pose → features → gestos → ML → relatório.

O objetivo não é só funcionar, é deixar visível cada etapa do pipeline. Vídeo nunca é
armazenado — apenas landmarks temporários, dados de treino e o modelo treinado.

## Status

| Sprint | Entrega | Estado |
|--------|---------|--------|
| 01 | Camera Engine (webcam + FPS) | ✅ pronto |
| 02 | Pose Engine (MediaPipe) | ✅ pronto |
| 03 | Arm Geometry (ângulos, distâncias) | ✅ pronto |
| 04 | Motion (histórico, velocidade) | ✅ pronto |
| 05 | Web App (FastAPI + WebSocket) | ✅ pronto |
| 06 | Gesture Rules | ✅ pronto |
| 07 | Dataset Builder | ✅ pronto |
| 08 | ML Training (Random Forest) | ✅ pronto |
| 09 | Live ML (Sinal X em tempo real) | ✅ pronto |
| 10 | Motion Lab (métricas, explicabilidade) | ✅ pronto |
| 11 | Reações Faciais (sinais observáveis, opt-in) | ✅ pronto |

## Setup

Requer Python 3.10–3.12 (MediaPipe ainda não suporta 3.13+) e [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python python3.12
uv sync --group dev --extra vision --extra web --extra ml
```

O modelo de pose (`models/pose_landmarker_lite.task`, 5.5 MB) é baixado sozinho na
primeira execução, via `ensure_model()`. Fora do git.

O modelo facial (`models/face_landmarker.task`, ~3.6 MB, Sprint 11) só é baixado se
`--face` for passado — opt-in, nunca automático.

## Rodar

```bash
# Sprint 01 — janela com a webcam ao vivo (q ou ESC para sair)
uv run python scripts/hello_camera.py

# checagem sem interface: acessa a câmera e imprime métricas
uv run python scripts/check_camera.py --frames 60

# Sprint 02 — webcam com ombro/cotovelo/punho desenhados (L liga os rótulos)
uv run python scripts/hello_pose.py

# checagem de pose sem interface: taxa de detecção + visibilidade por junta
uv run python scripts/check_pose.py --frames 60

# Sprint 03 — ângulo do cotovelo desenhado ao vivo + painel de features
uv run python scripts/hello_geometry.py

# features sem interface (determinístico numa foto, ou ao vivo)
uv run python scripts/check_geometry.py --image foto.jpg
uv run python scripts/check_geometry.py --frames 30

# Sprint 04 — velocidade e direção ao vivo, com rastro do punho (t liga/desliga)
uv run python scripts/hello_motion.py

# direção sem interface: gesto roteirizado (determinístico) ou webcam
uv run python scripts/check_motion.py --simulate
uv run python scripts/check_motion.py --frames 90

# Sprint 05 — mesmo pipeline, no navegador (câmera abre só quando a página conecta)
uv run python scripts/hello_web.py --open

# checagem sem navegador: conecta no WebSocket de verdade e mede o payload
uv run python scripts/check_web.py --frames 30

# Sprint 06 — braço levantado / aceno / braços cruzados / abertos, ao vivo no navegador
# (hello_web.py já mostra os eventos; não precisa de script separado)

# checagem sem câmera: postura roteirizada, determinística
uv run python scripts/check_gestures.py --simulate
uv run python scripts/check_gestures.py --frames 300   # webcam

# Sprint 07 — grava amostras rotuladas (features, nunca pixel) em data/training/dataset.csv
# segure 1-4 pra gravar cada rótulo, SPACE pra parar, q/ESC pra sair
# ("wave" não é um rótulo gravável -- é movimento, não postura; ver "Limitação
# conhecida" no Sprint 11 e as Decisões do Sprint 07, mais abaixo)
uv run python scripts/hello_dataset.py

# Sprint 08 — treina o Random Forest a partir do dataset gravado
uv run python scripts/train_model.py

# sem dataset ainda? treina em cima de dados sintéticos, só pra provar o pipeline
uv run python scripts/train_model.py --simulate

# Sprint 09 — o mesmo hello_web.py, agora com o sinal do modelo ao vivo
uv run python scripts/hello_web.py --model backend/models/gesture_classifier.joblib
uv run python scripts/check_web.py --model backend/models/gesture_classifier.joblib

# Sprint 10 — relatório: matriz de confusão, importância de feature, saúde
# ao vivo e concordância regra x ML, tudo num HTML só
uv run python scripts/report.py
uv run python scripts/report.py --no-live   # sem câmera, só a parte do treino

# Sprint 11 — reações faciais (sorriso aparente, boca, piscar, sobrancelha,
# orientação da cabeça), opt-in — baixa o modelo facial na primeira vez
uv run python scripts/hello_web.py --face --open
uv run python scripts/check_web.py --face --frames 30

# testes (não precisam de webcam nem do modelo — ambos são falsificados)
uv run pytest -q
```

### Permissão de câmera no macOS

Na primeira execução o macOS pede acesso à câmera para o app que rodou o comando
(Terminal, iTerm, VS Code…). Se a permissão for negada, o erro é:

```
FAIL: Could not open camera index 0. On macOS, grant camera access to your
terminal/IDE in System Settings > Privacy & Security > Camera, then restart it.
```

Libere em **Ajustes do Sistema › Privacidade e Segurança › Câmera** e reinicie o
terminal — a permissão só passa a valer depois do restart do processo pai.

## Arquitetura

```
Camera Engine -> Pose Engine -> Feature Engine -> Motion Engine
   (OpenCV)       (MediaPipe)    (ângulos,          (velocidade,
                    |             distâncias)         direção)
                    |                                     |
                    |               Gesture Engine  <----+---->  Live ML
                    |               (regras, cooldown)   |      (Random Forest,
                    |                                    |       Sprint 07-09)
                    v                                    v
             Face Engine  ->  Face Features  ->  Face Motion    Web App
             (MediaPipe        (sinais            (variação    (FastAPI + WebSocket)
              Face Landmarker,  observáveis,        temporal,        |
              Sprint 11,        nunca emoção)        Sprint 11)  Motion Lab Report
              opcional)                                          (métricas, explicabilidade)
```

Gesture Engine e Live ML são dois sinais paralelos sobre o mesmo Motion Engine —
nenhum depende do outro, e o relatório do Sprint 10 compara os dois. Face Engine
(Sprint 11) é a mesma ideia de "camada opcional em paralelo", desta vez sobre o
mesmo `Frame` da câmera em vez do Motion Engine: reaproveita a captura, roda um
segundo MediaPipe (Face Landmarker), e nunca é dependência de pose/gestos/ML.

Regra de acoplamento: cada camada recebe o **dado** da anterior, nunca o handle dela.
`CameraEngine` entrega `Frame`, não um `cv2.VideoCapture` — por isso trocar webcam por
arquivo de vídeo não toca em nada do lado da visão.

### Decisões do Sprint 11: sinal facial observável, nunca emoção

`backend/vision/face_detector.py` isola o MediaPipe Face Landmarker (só arquivo
que o importa, mesma regra do Sprint 02), `backend/features/face_features.py`
extrai sinais nomeados de um `FaceSnapshot`, `backend/features/face_motion.py`
adiciona a variação temporal (`SignalTrack`, reaproveitado do Sprint 04, mais um
contador de piscadas por minuto). Cinco escolhas:

1. **Blendshape do próprio MediaPipe, não geometria refeita na mão.** O Face
   Landmarker já expõe 52 scores nomeados (`mouthSmileLeft`, `jawOpen`,
   `eyeBlinkLeft`, `browInnerUp`...) calibrados contra um rig facial real.
   Recalcular "sorriso" a partir dos 478 pontos da malha seria uma segunda
   fonte de verdade, pior, competindo com uma melhor que já existe — mesma
   lógica do Sprint 10 sobre `feature_importances_` não precisar de SHAP.
2. **Nunca emoção, nunca estado mental.** Todo campo de `FaceSignals` é um
   sinal facial observável (sorriso *aparente*, abertura de boca, piscar,
   elevação de sobrancelha, orientação da cabeça) ou a variação temporal de
   um desses — nunca uma etiqueta como "feliz" ou "surpreso". Isso é regra
   dura do briefing, não só estilo: `FACE_SIGNAL_NAMES` é a lista completa de
   nomes permitidos, e um teste (`test_extract_never_claims_an_emotion_label`
   em `tests/test_face.py`) garante que "emotion"/"mood"/"feeling" nunca
   aparecem ali.
3. **Sem confiança fabricada.** Pose Landmarker dá `visibility`/`presence`
   por junta; Face Landmarker não expõe um score de confiança equivalente —
   só ativação por blendshape e a malha. Inventar um número aqui pareceria
   tão confiável quanto um de verdade, medindo nada. `FaceSnapshot.detected`
   (veio uma lista de landmarks ou não) é o único sinal de disponibilidade
   honesto; `frame.face.available` (modelo carregou nesta sessão) e
   `frame.face.detected` (tem rosto neste frame) são dois booleanos
   separados no payload por causa disso — "indisponível" e "sem rosto agora"
   são fatos diferentes.
4. **Opcional de ponta a ponta, degrada sem derrubar pose/web.** Mesma
   disciplina do sinal de ML (Sprint 09): `WebConfig.face_enabled` é `False`
   por padrão (custo real de CPU — é uma segunda inferência MediaPipe por
   frame); ligado, um modelo ausente/corrompido vira `face.available=false`
   com o motivo, nunca uma exceção; um erro em tempo real dentro do loop
   (`face_detector.detect()`) é capturado por frame e desliga só o sinal
   facial pelo resto da sessão — pose, motion, gestos e ML continuam de pé
   (`tests/test_web.py:test_a_face_runtime_error_degrades_without_stopping_pose_or_the_pipeline`).
5. **Mesmo `Frame`, nunca uma segunda captura.** `_stream` em
   `backend/web/pipeline.py` chama `face_detector.detect(frame)` com o
   *mesmo* `Frame` que o `PoseDetector` já processou naquele laço — camera ->
   vision continua sendo uma captura, dois estimadores, exatamente a regra
   de acoplamento que abre esta seção.

Orientação da cabeça (`head_yaw/pitch/roll`) vem da matriz de transformação
facial do MediaPipe, decomposta em ângulos por
`backend/vision/face_landmarks.py:euler_from_matrix` — geometria pura (sem
tipo do MediaPipe vazando adiante), verificada com matrizes de rotação
sintéticas de um eixo só (`tests/test_face.py`), mas a ordem de composição
real do MediaPipe para uma rotação simultânea em vários eixos não foi
verificada contra captura ao vivo — documentado como limitação conhecida no
próprio docstring, não assumido como correto silenciosamente.

Toggle no frontend (`index.html`/`app.js`, tecla `f`) só esconde o painel
nesta aba — ligar/desligar o processamento de fato é `WebConfig.face_enabled`
no servidor (`--face` em `hello_web.py`/`check_web.py`), a mesma divisão que
o painel "ml ao vivo" já tem (o servidor decide o que roda, o browser decide
o que mostra).

Nenhum frame, recorte facial ou malha de pontos é persistido em lugar nenhum
— nem em disco, nem em `data/`. Ver Privacidade.

Nenhum evento facial (tipo "sorriso detectado") foi adicionado ao
`GestureEngine` nesta sprint — o pedido era sinais e a tendência deles, não
reconhecimento de gesto facial; manter o escopo em números observáveis evita
que o painel vire, sem querer, um classificador de expressão.

### Dívidas corrigidas antes do Sprint 11

Quatro dívidas bloqueadoras, levantadas na revisão de arquitetura antes de
começar o sinal facial, corrigidas nesta mesma leva:

1. **`PipelineRunner` não se recuperava de uma câmera que parava de entregar
   frame.** `camera.frames()` levanta `CameraError` depois de
   `max_read_failures` leituras seguidas falhas (cabo solto, o SO reclamando
   o dispositivo por um instante) — antes disso, `_run` tratava isso como
   falha definitiva (`state = "failed"`, sem nova tentativa) igual a uma
   permissão negada. Agora `_run` tenta de novo com backoff exponencial
   limitado (`max_camera_retries`, `retry_backoff_base/cap` em `WebConfig`);
   um frame de verdade entregue no meio zera a sequência de tentativas (é
   prova de recuperação real, não só um `open()` com sorte); cada tentativa
   manda `status_message("recovering", ...)` — mensagem não-fatal, o
   WebSocket continua aberto e o app.js mostra "reconectando à câmera…" em
   vez de derrubar a conexão. `tests/test_web.py:test_camera_recovers_after_a_transient_failure_and_resumes_streaming`
   cobre o caso "câmera falha e volta".
2. **Split de treino aleatório por frame, não por sessão.** `train()` usava
   `train_test_split` embaralhando linhas — como `hello_dataset.py` grava
   sem debounce (Sprint 07: cada frame com a tecla segurada é uma linha),
   frames vizinhos de uma mesma sessão são quase duplicatas, e um split
   aleatório por linha deixa cópias quase idênticas dos dois lados do
   split, inflando a acurácia reportada sem separar nada de verdade. Agora
   `train()` usa `GroupShuffleSplit` agrupado por `session_id` e recusa
   treinar (`DatasetError`) se o dataset tiver amostras de uma única sessão
   — não dá pra medir generalização sem pelo menos duas.
3. **Motion ausente virava `0` silenciosamente.** `_to_number` tratava
   célula vazia (junta oculta, ou `MotionTracker` ainda sem histórico
   suficiente — `min_samples`) como "velocidade 0", ensinando o modelo que
   "não sei" e "parado" são a mesma coisa — a mesma armadilha que o Sprint
   03 já evita para features estáticas, só que sem correção para motion.
   Agora `build_vector`/`load_dataset` levantam `MissingFeatureError` e
   descartam a linha (contada em `Dataset.dropped`), nunca inventam um
   número; e `PipelineRunner._predict` ganhou o mesmo gate
   (`MotionState.complete`, novo em `features/velocity.py`) do lado da
   inferência ao vivo — sem ele, o primeiro frame de toda sessão (antes do
   tracker esquentar) derrubava a thread inteira do pipeline com
   `MissingFeatureError` não capturado.
4. **`shoulder_width` treinava com o modelo, `predict()` ignorava
   `bundle["columns"]`.** `shoulder_width` é a régua de escala de todas as
   outras features (Sprint 03 já a mantém fora de `FEATURE_NAMES` por isso),
   mas `TRAINING_COLUMNS` a incluía explicitamente — um modelo treinado só
   funcionaria bem na distância de gravação. `predict()` também montava o
   vetor a partir do `TRAINING_COLUMNS` importado do módulo atual, não de
   `bundle["columns"]` — um bundle salvo por uma versão diferente do treino
   prediria em silêncio contra a ordem errada. Ambos corrigidos:
   `shoulder_width` fora de `TRAINING_COLUMNS`; `predict()` sempre reconstrói
   o vetor a partir de `bundle["columns"]`. `load_dataset` também passou a
   validar o header do CSV contra `DATASET_COLUMNS` (schema), descartar
   rótulo vazio e valor não-finito — tudo contado em `Dataset.dropped`, nunca
   escondido.

**Limitação conhecida — decisão tomada:** `wave` é uma classe de *movimento*,
não de postura, mas o vetor de ML (`TRAINING_COLUMNS`) só carrega uma linha de
motion por frame (velocidade e direção instantâneas), sem a janela de
reversões que `GestureEngine` (`_WaveDetector` em `gestures/engine.py`) usa
pra reconhecer um aceno de verdade — um frame no meio de um aceno pode ser
geometricamente idêntico a um frame de braço subindo sem repetir, e o
classificador não tem como saber a diferença sem uma janela temporal própria,
que este sprint não construiu para o vetor de ML (só para os sinais faciais,
via `FaceMotionTracker`).

Decisão do orquestrador: tirar `wave` do vocabulário treinável do Random
Forest por frame de vez, e deixar só a regra (`GestureEngine`) reconhecer
aceno, até o vetor de ML ganhar uma janela temporal de verdade. Aplicado em
toda a cadeia, não só no baseline sintético:

- `backend/dataset/writer.py:DEFAULT_LABELS` não inclui mais `wave` —
  `hello_dataset.py` não oferece mais a tecla pra ele (agora só
  `1 arm_raised  2 arms_crossed  3 arms_open  4 idle`).
- `DatasetWriter.write()` recusa a gravação de uma linha rotulada `wave`
  (`REMOVED_LABELS`), então nenhuma sessão nova consegue reintroduzir o
  rótulo mesmo chamando a API diretamente.
- `backend/ml/train.py:load_dataset` recusa carregar um `dataset.csv` que
  ainda tenha linhas `wave` de antes dessa decisão — erro explícito, com os
  passos de migração (filtrar as linhas `wave` do CSV, ou mover o arquivo de
  lado e regravar do zero), nunca um drop silencioso em `Dataset.dropped` nem
  um treino em cima de um sinal que o Forest não consegue separar de verdade.
- `scripts/train_model.py --simulate` já não incluía `wave` no baseline
  sintético (correção anterior); o comentário ali agora deixa claro que a
  decisão é definitiva, não uma omissão só do simulador.
- `backend/ml/report.py`'s `compute_agreement` mantém a exclusão de um
  `ml.label == "wave"` da concordância regra × ML, mas só como salvaguarda
  pra um `gesture_classifier.joblib` treinado antes desta decisão — o modelo
  atual nunca emite essa classe, porque ela não está mais no vocabulário que
  ele treina.

`ml_classes` (mandado ao cliente no `hello` do WebSocket) é, a partir de
agora, sempre só postura — o ML nunca reconhece aceno; reconhecer aceno
continua sendo só `GestureEngine`, ao vivo, via regra, não modelo.

### Correções pós-revisão (bloqueadores encontrados antes de fechar o Sprint 11)

Uma revisão de arquitetura sobre o próprio diff do Sprint 11 encontrou seis
bloqueadores adicionais, corrigidos nesta mesma leva:

1. **Corrida de subscriber depois de uma falha fatal.** Um cliente que
   conectava exatamente enquanto a thread do pipeline morria com
   `CameraOpenError`/`PoseModelError` (já tinha transmitido o erro pros
   inscritos anteriores) ficava esperando um frame que nunca viria —
   `_ensure_running()` corretamente recusa abrir uma segunda thread enquanto a
   antiga ainda está viva/terminando, mas ninguém reenviava o erro pro
   recém-chegado. `PipelineRunner.subscribe` agora reenvia o último erro
   fatal quando `_ensure_running()` não reiniciou nada.
2. **Erro obsoleto entre uma falha e a nova tentativa.** Entre
   `self._thread.start()` e a primeira linha da nova `_run()` (que zerava
   `self.error`), `/health` podia mostrar `state: "starting"` ao lado do erro
   da tentativa *anterior*. `_ensure_running()` agora zera `self.error` no
   mesmo lock que decide reiniciar.
3. **Retry infinito por flapping.** Uma câmera que falha logo acima de
   `healthy_camera_frames` a cada tentativa nunca acumulava `attempt`
   (resetado a cada "recuperação"), retentando pra sempre. `flap_window_seconds`/
   `max_flaps_per_window` (`WebConfig`) contam tentativas de reabertura numa
   janela deslizante, independente do streak saudável, e desistem
   (`state = "failed"`) se um dispositivo flapar demais.
4. **Tasks do WebSocket canceladas nunca eram aguardadas.** `stream()`
   cancelava `writer`/`reader` mas nunca dava `await` neles antes de retornar
   — corrigido com `asyncio.gather(*pending, return_exceptions=True)`. Isso
   abriu um segundo problema: o teardown do próprio ASGI (ou do `TestClient`)
   podia cancelar o `stream()` bem no meio desse `await`, vazando
   `CancelledError` pra fora do handler. `except (WebSocketDisconnect,
   asyncio.CancelledError)` mais um `finally` que cancela qualquer task ainda
   viva protege o fechamento sem esconder a causa raiz.
5. **`_disable_face` prefixava o nome da exceção até em erros já explicáveis.**
   `FaceModelError("modelo não encontrado")` virava
   `"FaceModelError: modelo não encontrado"` no payload — o prefixo agora só
   aparece para exceções inesperadas (mesma regra que `_run` já aplicava a
   `PoseModelError`/`CameraError` vs. exceção genérica).
6. **Toggle facial mentia sobre o que controlava, e a UI misturava fechamento
   de olho com evento de piscada.** `toggle-face` vinha marcado por padrão
   independentemente de `WebConfig.face_enabled` — agora `hello.face_enabled`
   desabilita e desmarca o checkbox quando o servidor nunca ligou o sinal
   facial (rótulo "mostrar painel facial", não "reações faciais", já que é
   só isso que ele faz). O painel mostrava só `eyeBlink*` (o valor do
   blendshape neste frame, 0-1) rotulado "piscada" — agora "fechamento olho"
   e "piscadas/min" (`blink_rate_left/right`, já calculado por
   `FaceMotionTracker` mas nunca exibido) aparecem como campos separados.

Dataset e treino também endureceram nesta leva: `DatasetWriter` agora rejeita
(`DatasetWriterError`) um header de CSV incompatível ao reabrir um arquivo
existente, `session_id`/`label` vazios, `features`/`motion` de frames
diferentes, e qualquer valor não-finito — em vez de escrever uma linha que só
seria pega (ou não) rio abaixo. `train()` passou a exigir que toda classe
apareça em pelo menos duas sessões distintas (não só o dataset como um todo):
sem isso, o split agrupado por `session_id` pode jogar 100% de uma classe pra
um lado só, e a precision/recall/f1 dela no relatório não mede nada.
`load_model`/`predict` validam o bundle (`model`/`columns`/`classes`
presentes, não vazios, `predict_proba` disponível, contagem de classes batendo
com a saída do modelo) em vez de deixar um bundle corrompido falhar em algum
ponto silencioso mais adiante. O relatório e o `.joblib` agora carregam
macro-F1 e acurácia balanceada ao lado da acurácia simples — importante porque
`idle` tende a dominar uma sessão real (é a classe negativa, Sprint 07), e
acurácia simples sozinha pode esconder um modelo que só acerta a maioria.

### Decisões do Sprint 10: explicação é o que o modelo já sabe dizer, não um método novo

`scripts/report.py` fecha o roadmap: lê o `.joblib` do Sprint 08, amostra uma
sessão ao vivo (opcional) e funde os dois num HTML autocontido. Tensão real
nas respostas do briefing: métricas pedidas eram todas *ao vivo* (saúde do
pipeline, concordância regra × ML), mas a interface pedida foi HTML *estático,
gerado no treino*. Resolução: o HTML nasce no treino (matriz de confusão,
precision/recall, importância de feature — tudo já calculado, nunca precisa
de câmera pra existir), e o script soma uma amostragem ao vivo por cima
quando a câmera está disponível.

1. **Explicabilidade é o que o Random Forest já expõe, não SHAP nem LIME.**
   `feature_importances_` do próprio modelo já diz "decido principalmente por
   `wrist_distance` e `left_wrist_height`" — inventar um método de explicação
   por cima seria uma segunda fonte de verdade competindo com a primeira, sem
   necessidade real neste tamanho de modelo.
2. **Concordância regra × ML reusa o mesmo threshold da regra, não um novo.**
   `rule_label()` em `backend/ml/report.py` decide "braço levantado" com a
   mesma lógica de três booleanos que `GestureEngine` já usa — lido direto do
   payload do frame, não recalculado. Um `ml.label == "wave"` é excluído da
   comparação, não contado como discordância — não porque o modelo atual
   reconheça aceno (`wave` está fora do vocabulário treinável, ver
   "Limitação conhecida" acima), mas como salvaguarda pra um modelo salvo
   antes dessa decisão.
3. **Sem câmera, o relatório ainda existe.** `run_live_session` captura
   `CameraError`/timeout e degrada pra "só treino" em vez de falhar —
   consistente com o Sprint 09: ML e suas métricas são camada opcional, nunca
   dependência do resto funcionar.

`backend/models/*.html` fica fora do git, mesma regra do `.joblib`.

### Decisões do Sprint 09: o sinal de ML é opcional, nunca dependência

`PipelineRunner` carrega o `.joblib` (se existir) uma vez por sessão de câmera
e manda `ml: {label, confidence, probabilities}` no payload — `null` quando
não há modelo ou quando o frame tem junta oculta. Três escolhas:

1. **Sem modelo, o app inteiro continua de pé.** Pose, motion, gestos por
   regra (Sprint 01-06) não sabem que ML existe. `load_model` falha com
   `ModelError`, capturado só ali, nunca propaga — ao contrário de câmera ou
   modelo de pose ausentes, que são erro fatal, um `.joblib` ausente é
   silêncio numa coluna do payload.
2. **Frame incompleto nunca chega ao classificador.** Mesma regra do Sprint
   07/08: uma junta oculta virando `0` pareceria um valor real e confiante pro
   Random Forest (cotovelo "a 0 grau" pareceria totalmente dobrado). O sinal
   fica `null` nesse frame em vez de arriscar um palpite fabricado.
3. **Sinal cru, sem suavização.** É "Sinal X em tempo real" por definição —
   pisca frame a frame do jeito que o modelo realmente prevê, sem debounce
   nem cooldown como o `GestureEngine` tem. O painel mostra o modelo pelado,
   inclusive as vezes que ele erra ou hesita — dado real pra decidir depois se
   precisa de mais tratamento.

`--model` é flag nova em `hello_web.py`/`check_web.py`, default aponta pro
mesmo `backend/models/gesture_classifier.joblib` que o Sprint 08 salva.

### Decisões do Sprint 08: o vetor de treino é a única fonte da verdade

`backend/ml/train.py` lê `data/training/dataset.csv`, treina um
`RandomForestClassifier` e salva em `backend/models/*.joblib` (fora do git).
Três escolhas:

1. **`TRAINING_COLUMNS` é uma tupla, não uma convenção.** É a ordem exata do
   vetor com que o modelo foi ajustado. O Sprint 09 vai montar esse mesmo
   vetor a partir de um frame ao vivo — se a ordem lá divergir da ordem aqui,
   o modelo não quebra, só erra com confiança, sem erro pra pegar. Fixar isso
   numa tupla importável é a mesma disciplina append-only de `FEATURE_NAMES`.
2. **Direção sai, velocidade fica.** `left_direction`/`right_direction` já são
   um resumo com perda da velocidade com sinal (`classify()` em `velocity.py`:
   positivo é subindo). Treinar com as duas seria treinar com uma cópia
   derivada da mesma coluna, sem ganho de separação.
3. **Modelo, ordem de coluna e lista de classes viajam juntos.** O `.joblib`
   salvo é um dict com os três — o Sprint 09 carrega tudo de um artefato só,
   nunca re-deriva a ordem, então o vetor montado na hora da inferência não
   tem como divergir silenciosamente do que treinou o modelo.

Dataset real ainda está vazio (ninguém gravou sessão com `hello_dataset.py`
até agora), então `train_model.py --simulate` existe pelo mesmo motivo que
`check_motion.py --simulate`: provar o pipeline inteiro — carregar, separar
treino/teste, ajustar, avaliar, salvar, recarregar, prever — funciona hoje,
deterministicamente, sem esperar uma sessão de gravação real. Rodar sem
`--simulate` com o dataset vazio falha limpo (`DatasetError`), não com stack
trace.

### Decisões do Sprint 07: o dataset não é a mesma tabela do gesto

`backend/dataset/` grava `FrameFeatures` rotulados em `data/training/dataset.csv`
pra alimentar o Random Forest do Sprint 08. Quatro escolhas:

1. **Motion entra na mesma linha das features estáticas.** `MOTION_COLUMNS`
   (velocidade, direção) foi adicionado pra um classificador distinguir punho
   no meio de um movimento de qualquer outra posição de passagem — o sinal
   que uma foto estática de postura não dá. O caso que motivou isso, "aceno",
   acabou precisando de mais que a velocidade instantânea de um frame pra
   significar algo, e foi tirado do vocabulário treinável de vez (escolha 4);
   as colunas ficam mesmo assim, porque não custam nada ao Sprint 08 ignorar
   pras classes puramente posturais e um rótulo futuro que realmente separe
   por velocidade não precisaria de uma segunda migração pra ganhá-las.
2. **Frame incompleto não vira amostra.** `FrameFeatures.complete` já diz se
   toda coluna de ML está preenchida (regra do Sprint 03: junta oculta é
   `None`, nunca um `0` inventado). `DatasetWriter` descarta e conta em vez de
   gravar — `skipped` no HUD mostra oclusão acontecendo, não depois do fato.
3. **Gravação é sem debounce, de propósito.** `GestureEngine` exige postura
   sustentada antes de contar; aqui cada frame com a tecla segurada vira uma
   linha, erro incluído — é matéria-prima pra treinar, uma amostra ruim é só
   uma linha a filtrar depois, não um veredito ao vivo que precisa acertar.
4. **`wave` não é um rótulo treinável.** É uma classe de movimento (uma
   reversão de direção repetida algumas vezes), não de postura, e este vetor
   por frame só carrega a velocidade/direção de um instante — não a janela de
   reversões que o `_WaveDetector` do `GestureEngine`
   (`backend/gestures/engine.py`) de fato usa pra separar um aceno real de um
   braço subindo sem repetir. `DatasetWriter.write()` recusa o rótulo direto
   (`REMOVED_LABELS`); reconhecer aceno continua sendo só trabalho do
   `GestureEngine` até o vetor de ML ganhar uma janela temporal de verdade
   (ver "Limitação conhecida" no Sprint 11, mais acima).

Segurar tecla no OpenCV não tem key-up nativo: `hello_dataset.py` infere
"ainda segurando" pelo auto-repeat do SO chegando mais rápido que
`RELEASE_TIMEOUT` (0.25s) — SPACE sempre para na hora, como rede de segurança.
`data/training/*` fica fora do git (só `.gitkeep`): dataset de gesto carrega
dado biométrico-ish, fica local por padrão.

### Decisões do Sprint 06: um gesto é um evento, não um estado

`backend/gestures/` fica logo depois da Motion Engine: transforma
`wrist_above_shoulder = True` sustentado, ou o punho oscilando pra cima e pra
baixo, num evento nomeado que sai uma vez, não a cada frame. Três escolhas:

1. **Regra pura, tempo é do engine.** `rules.py` só responde "essa postura
   bate agora?" (igual `angles.py`/`positions.py`); quanto tempo precisa
   segurar e de quanto em quanto tempo pode repetir é decisão de
   `GestureEngine` (`engine.py`). Um `_Debounce` exige a condição segurada por
   `hold_seconds` antes de virar estável — a mesma zona morta que o Sprint 04
   usou pra velocidade, aplicada a um booleano — e um `_EdgeCooldown` dispara
   só na borda de subida, com `cooldown_seconds` de carência.
2. **Junta oculta nunca dispara gesto.** `None` (junta não confiável) vira
   "condição não bate", nunca "bate". Um gesto é uma afirmação positiva —
   inventar um a partir de dado ausente seria pior que ficar calado.
3. **`reset()` existe pra sobreviver ao modo ocioso.** `PipelineRunner` para de
   chamar `update()` quando ninguém está olhando, mas o relógio do processo
   continua andando; sem resetar o estado, um braço meio-levantado antes de
   todo mundo sair apareceria como "levantado há vários minutos" no primeiro
   frame de volta. `tests/test_gestures.py` documenta esse bug lado a lado com
   a correção (`test_without_reset_...` / `test_reset_prevents_...`).

Seis gestos na V1 — `arm_raised` e `wave` por lado, mais as posturas de duas
mãos `arms_crossed` e `arms_open` — todos só no payload do WebSocket
(`backend/web/payload.py`): o HUD mostra "gesto detectado" ao vivo, nada é
persistido em `data/events/`. O roadmap final (Sprints 09-10) foi por outro
caminho — sinal de ML ao vivo e relatório de métricas — em vez de um Event
Engine dedicado; `data/events/` fica como diretório reservado, sem uso.

### Decisões do Sprint 05: uma câmera, muitos navegadores

O front end é HTML/CSS/JS puro, sem framework e sem build step — abrir `index.html`
servido pelo FastAPI já é o app inteiro. Quatro escolhas carregam o resto:

1. **O backend é dono da câmera, o navegador só desenha.** `getUserMedia` foi
   descartado de propósito: duplicaria o pipeline (uma pose engine em Python,
   outra implícita se o browser processasse vídeo) e a webcam do notebook fica
   presa a uma aba. Em vez disso o Sprint 01-04 roda como sempre, e cada frame
   vira uma mensagem JSON pelo WebSocket — pixels, landmarks, features e motion
   juntos, para o esqueleto nunca desenhar sobre um frame que não é o dele.
2. **A câmera abre no primeiro cliente e fecha depois do último.** Não no boot
   do servidor. O LED aceso é um contrato de privacidade: só acende com alguém
   de fato olhando. `PipelineRunner` roda o loop numa thread (câmera e MediaPipe
   bloqueiam, e bloquear o event loop do asyncio travaria todo WebSocket junto)
   e cada aba é um `Subscriber` com uma única vaga — se o navegador não
   acompanha, o frame velho é descartado, nunca vira fila. Ver
   `backend/web/pipeline.py`.
3. **A pré-visualização é redimensionada, a inferência não.** O JPEG mandado
   pro navegador encolhe para `preview_width` (640 px por padrão); a pose roda
   no frame cheio. Tamanho de imagem e qualidade de detecção viram dois botões
   independentes.
4. **O espelhamento acontece nas coordenadas, não no canvas.** Espelhar o
   `<canvas>` inteiro inverteria os rótulos junto — "cotovelo esquerdo" apontaria
   pro lado errado. `app.js` desenha a imagem sob uma transformação invertida e
   desfaz antes do overlay, igual a regra do Sprint 01: espelhamento é só
   exibição.

Testado sem hardware: `tests/test_web.py` sobe o FastAPI real com
`TestClient`, mas troca a câmera por uma fake que nunca para de entregar frame e
o MediaPipe por um landmarker fake que sempre "vê" um corpo — cobre handshake,
payload, ciclo de vida da câmera (abre/fecha/compartilha entre duas abas) e
falha (permissão negada chega como mensagem de erro, não como conexão
pendurada).

### Decisões do Sprint 04: o pipeline ganha memória

Até aqui tudo era sem memória — um frame entra, uma linha de números sai. Um frame
sozinho só sabe dizer "o punho está acima do ombro"; um histórico sabe dizer "o punho
está subindo". Três escolhas sustentam isso:

1. **Velocidade é por segundo, nunca por frame.** A webcam não entrega FPS fixo (medi
   ~22 aqui, e cai quando a CPU aperta). Delta por frame faria o mesmo gesto virar
   número diferente em máquina rápida e lenta — e o Sprint 08 treinaria em cima desse
   artefato. Teste `test_same_movement_at_half_the_frame_rate_gives_the_same_velocity`
   roda o mesmo movimento a 60 e a 15 fps e exige o mesmo resultado.
2. **O sinal é medido contra o corpo, não contra a imagem.** O que é derivado é
   `wrist_height` (larguras de ombro acima do próprio ombro), então andar na direção da
   câmera ou o notebook balançar não vira movimento de braço. "Parado" aqui significa
   "parado em relação ao seu tronco", que é o que um gesto de fato é.
3. **Landmark treme, então velocidade zero não existe.** Braço totalmente parado ainda
   mede alguns centésimos de largura por segundo de ruído — sem zona morta a direção
   piscaria UP/DOWN a cada frame. A inclinação é ajustada por mínimos quadrados sobre a
   janela inteira (0.4 s), não sobre dois frames consecutivos, e abaixo de
   `still_threshold` o veredito é STILL.

Buraco no sinal tem regra própria: frame perdido isolado é tolerado, blecaute maior que
`max_gap` (0.3 s) **apaga** o histórico em vez de ligar os dois lados. Não sabemos o que
o braço fez enquanto estava escondido, e uma reta atravessando o buraco produziria uma
velocidade confiante e errada.

`direction` e `moving` são campos separados de propósito: braço varrendo de lado é
`STILL` na vertical e `moving=True`. Juntar os dois faria o HUD mentir.

### Decisões do Sprint 03: features precisam ser invariantes

Três armadilhas que o código resolve explicitamente:

1. **Coordenada normalizada é espremida.** `x` é dividido por largura, `y` por altura.
   Num frame 16:9 um braço a 45° reais mede ~60° se você calcular direto de x/y.
   `PoseSnapshot.aspect` carrega a razão e todo cálculo desfaz a distorção.
2. **Distância crua não serve pra ML.** "Punhos a 0.28" significa coisas diferentes perto
   e longe da câmera. Tudo é dividido pela **largura dos ombros** → a feature vira a mesma
   você a 1 m ou a 3 m. `shoulder_width` fica fora de `FEATURE_NAMES` de propósito: é a
   régua, e um modelo que aprendesse com ela só funcionaria na distância do treino.
3. **`y` cresce pra baixo.** "Punho acima do ombro" é `shoulder.y - wrist.y`, não o inverso.

E a regra que atravessa tudo: junta não confiável vira `None`, nunca `0`. Zero grau
pareceria cotovelo totalmente dobrado para o classificador. `FrameFeatures.complete` diz
se a linha serve para treinar.

### Decisão do Sprint 02: MediaPipe fica preso em um arquivo

Só `backend/vision/pose_detector.py` importa MediaPipe. Ele devolve `PoseSnapshot`
(tipos nossos), nunca objetos da lib. As camadas de feature e de ML não sabem qual
estimador de pose existe — trocar de modelo mexe em um arquivo.

Modo `VIDEO` (não `LIVE_STREAM`): mantém estado de tracking entre frames e é síncrono,
então um frame lento atrasa o loop em vez de descartar resultado em silêncio.
`num_poses=1` — múltiplas pessoas exigiria resolver identidade entre frames, fora do MVP.

### Decisão do Sprint 01: espelho é assunto de display

`CameraEngine.read()` devolve a imagem **crua** do sensor. O espelhamento (selfie view)
é aplicado só na hora de desenhar, via `mirror()`. Espelhar antes da pose inverteria
esquerda/direita e corromperia silenciosamente todo label de landmark daí para frente.

## Estrutura

```
backend/
  camera/capture.py   CameraEngine, Frame, FpsMeter, CameraError
  vision/landmarks.py PoseLandmark (33 índices), Side, Point, PoseSnapshot
  vision/pose_detector.py  PoseDetector (único import de MediaPipe pose)
  vision/drawing.py   draw_arms, draw_hud, draw_joint_values, draw_trail
  vision/face_landmarks.py  FaceSnapshot, euler_from_matrix (Sprint 11)
  vision/face_detector.py   FaceDetector (único import de MediaPipe Face Landmarker)
  features/angles.py  angle_between, elbow_angle, shoulder_angle
  features/distances.py  distance, shoulder_width, wrist_distance, forearm_length
  features/positions.py  wrist_height, wrist_above_shoulder, wrists_crossed
  features/features.py   FrameFeatures, extract, FEATURE_NAMES (ordem do dataset)
  features/velocity.py   SignalTrack, MotionTracker, MotionState, Direction, Trail
  features/face_features.py  FaceSignals, extract_face, FACE_SIGNAL_NAMES (Sprint 11)
  features/face_motion.py    FaceMotionTracker, FaceMotionState (variação temporal, Sprint 11)
  gestures/rules.py   arm_raised, arms_crossed, arms_open (postura, sem tempo)
  gestures/engine.py  GestureEngine, GestureConfig, GestureEvent (hold + cooldown)
  dataset/writer.py   DatasetWriter, DATASET_COLUMNS — features+motion rotulados em CSV
  ml/train.py         TRAINING_COLUMNS, train, load_model, predict (Random Forest)
  ml/report.py        render_report, compute_agreement — o relatório do Sprint 10
  events/             reservado, sem uso (roadmap foi por ML ao vivo + relatório em vez de Event Engine)
  models/             modelos .joblib, .report.html e face_landmarker.task (fora do git)
  web/payload.py      formato de mensagem do WebSocket (hello/frame/status/error)
  web/pipeline.py     PipelineRunner, Subscriber — uma câmera, muitos clientes
  web/app.py          FastAPI: rotas /, /health, /ws
frontend/
  index.html          layout: canvas + painel de stream/motion/face/toggles
  style.css           cores do overlay, mesmas de vision/drawing.py
  app.js              desenha o payload no canvas; nunca calcula feature
data/training/        amostras de gestos (features + motion rotulados, nunca imagens)
data/events/          reservado, sem uso
scripts/              entrypoints de cada sprint
tests/
```

## Privacidade

- O JPEG de cada frame **trafega para o navegador** pelo WebSocket (Sprint 05,
  `backend/web/payload.py:encode_preview` -> `frame_message`) — isso é rede de
  verdade, não "tudo fica local": em `127.0.0.1` (padrão de `hello_web.py`) o
  tráfego nunca sai da própria máquina, mas se o servidor subir com
  `--host 0.0.0.0` esse mesmo JPEG passa a trafegar pela LAN até qualquer
  navegador que conecte (sem autenticação — ver o aviso em `scripts/hello_web.py`).
  O que **não existe** é chamada de rede para fora — nem um frame, recorte ou
  landmark é enviado a um serviço de terceiros/nuvem; o único tráfego de saída
  do processo é o download do modelo, uma vez, explícito via `ensure_model`/
  `ensure_face_model` (nunca automático a cada frame).
- Frames vivem em memória e são descartados; `.gitignore` bloqueia `*.mp4/*.mov/*.png/*.jpg`.
- O dataset de treino guarda landmarks + features + label — nunca pixels.
- **Sprint 11 (reações faciais):** nenhum frame, recorte de rosto ou malha de
  478 pontos é persistido em disco ou em `data/` — o `FaceSnapshot` vive só na
  memória do laço de captura, pelo tempo de um frame. Só os *sinais* (números
  como `smile: 0.7`) trafegam pelo WebSocket; o dataset de treino do Sprint 07
  não grava nenhuma coluna facial ainda. Todo campo é um sinal facial
  observável — sorriso aparente, abertura de boca, piscar, elevação de
  sobrancelha, orientação da cabeça, e a variação temporal de cada um —
  **nunca** uma emoção ou estado mental inferido; ver "Decisões do Sprint 11"
  para o porquê disso ser regra, não estilo. `WebConfig.face_enabled` é
  `False` por padrão: ligar exige `--face` explícito.
