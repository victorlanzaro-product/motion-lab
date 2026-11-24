# Motion Lab

> A local-first computer vision playground: webcam → pose landmarks → features →
> gestures → machine-learning experiments. Video frames stay in memory and are
> never written as recordings by the application.

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue)](#setup)
[![Local processing](https://img.shields.io/badge/processing-local-brightgreen)](#privacy)

Motion Lab makes each step of a real-time movement pipeline visible and testable.
It includes pose and optional facial landmarks, gesture rules, dataset tools,
frame-based and temporal ML experiments, and a local web interface. This is a
research/playground project; real-session validation is still in progress.

## Demo

There is no hosted demo or checked-in video. Run the local web app to see the
live pipeline. The browser preview receives frames from the local server over
WebSocket; by default the server binds to `127.0.0.1`.

## Quick install

Requires Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python python3.12
uv sync --group dev --extra vision --extra web --extra ml
uv run python scripts/hello_web.py --open
```

The pose model downloads on first use. Facial analysis is optional and requires
the explicit `--face` flag. See [Privacy](#privacy) and the setup guide below.

## Privacy

- Processing runs on the machine. Frames are held in memory and discarded, not
  saved as video or sent to a cloud service.
- The browser preview receives frame images over WebSocket. The default bind is
  localhost; binding to `0.0.0.0` exposes the unauthenticated stream to the LAN.
- The pose model downloads once when first needed. The face model downloads only
  when facial analysis is explicitly enabled.
- Training datasets store landmarks and derived features with labels, never
  image pixels. Facial signals are opt-in and are not emotion or mental-state
  predictions.

See the detailed [Privacy section](#privacidade) for implementation details.

## Project status

This repository is being prepared for open-source release. The MIT license is a
proposal in this pull request and takes effect only if the project owner merges
it. See [CONTRIBUTING.md](CONTRIBUTING.md), the [Code of Conduct](CODE_OF_CONDUCT.md),
and the [issue and pull request templates](.github/).

Suggested GitHub topics for the owner to consider, recorded without changing
repository settings: `computer-vision`, `human-pose-estimation`, `mediapipe`,
`opencv`, `gesture-recognition`, `local-first`, `fastapi`, `machine-learning`.

---

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
| 12 | Recorder (gravação de dataset pelo navegador) | ✅ pronto (checkpoint com câmera pendente) |
| 13 | Cobertura do dataset e validação por sessão | Em validação com dataset real pendente |
| 14 | ML temporal v2, gravação de wave e inferência ao vivo | Implementado; gravações reais de wave e LOSO pendentes |

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

### Decisões do Sprint 13 — Cobertura e validação por sessão

- O Recorder e `GET /dataset/stats` usam o mesmo `WebConfig.dataset_path`.
  O endpoint lê o CSV fora do event loop e o painel exibe a cobertura
  recebida, atualizando após o fim de uma sessão.
- A CLI, o endpoint e o relatório compartilham o cálculo da cobertura.
  Rótulos desconhecidos contam em `dropped.unknown_label`, impedem
  `ready=true` e continuam sendo recusados pelo treino.
- As dobras preservam sessões inteiras. O relatório mostra as dobras
  usadas e descartadas, a média e o desvio de macro-F1 e a matriz do
  split de diagnóstico.
- O aceite usa `uv run python scripts/train_model.py --cv loso`, com uma
  sessão por dobra. A métrica principal reúne as predições fora da amostra
  antes de calcular o macro-F1 dos quatro rótulos. O split de diagnóstico
  reserva ao menos uma sessão de cada rótulo de cada lado quando cada
  gravação tem um único rótulo.
- O resultado em dado real será preenchido após as 12 gravações do
  [roteiro da S13](docs/sprints/13-roteiro-gravacao.md).

### Decisões do Sprint 14 — ML temporal e gravação de wave

- O schema v1 mantém treino e inferência por frame e segue como padrão;
  bundles sem `schema_version` continuam sendo interpretados como v1.
- O schema v2 usa Random Forest com features estáticas do último frame e
  14 fe…9055 tokens truncated…00f3rico em vez de ligar os dois lados. Não sabemos o que
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
