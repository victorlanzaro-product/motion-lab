# Motion Lab

Computer vision playground rodando 100% local: webcam → pose → features → gestos → eventos.

O objetivo não é só funcionar, é deixar visível cada etapa do pipeline. Vídeo nunca é
armazenado — apenas landmarks temporários, dados de treino e eventos detectados.

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
| 10 | Motion Lab (métricas, explicabilidade) | ⬜ |

## Setup

Requer Python 3.10–3.12 (MediaPipe ainda não suporta 3.13+) e [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python python3.12
uv sync --group dev --extra vision --extra web --extra ml
```

O modelo de pose (`models/pose_landmarker_lite.task`, 5.5 MB) é baixado sozinho na
primeira execução, via `ensure_model()`. Fora do git.

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
# segure 1-5 pra gravar cada rótulo, SPACE pra parar, q/ESC pra sair
uv run python scripts/hello_dataset.py

# Sprint 08 — treina o Random Forest a partir do dataset gravado
uv run python scripts/train_model.py

# sem dataset ainda? treina em cima de dados sintéticos, só pra provar o pipeline
uv run python scripts/train_model.py --simulate

# Sprint 09 — o mesmo hello_web.py, agora com o sinal do modelo ao vivo
uv run python scripts/hello_web.py --model backend/models/gesture_classifier.joblib
uv run python scripts/check_web.py --model backend/models/gesture_classifier.joblib

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
Camera Engine  ->  Pose Engine  ->  Landmark Engine  ->  Feature Engine
   (OpenCV)        (MediaPipe)      (posições)           (ângulos,
                                                          distâncias,
                                                          velocidades)
                                                              |
                                       Event Engine  <-  Gesture Engine
                                       (eventos com      (regras + modelo
                                        cooldown)         Random Forest)
```

Regra de acoplamento: cada camada recebe o **dado** da anterior, nunca o handle dela.
`CameraEngine` entrega `Frame`, não um `cv2.VideoCapture` — por isso trocar webcam por
arquivo de vídeo não toca em nada do lado da visão.

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
pra alimentar o Random Forest do Sprint 08. Três escolhas:

1. **Motion entra na mesma linha das features estáticas.** `FEATURE_NAMES`
   descreve um instante; "aceno" não é postura, é movimento. Um classificador
   treinado só em posição não distingue punho no meio de um aceno de qualquer
   outra posição de passagem. `MOTION_COLUMNS` (velocidade, direção) dá esse
   sinal — sem ele o rótulo `wave` seria ruído puro no dataset.
2. **Frame incompleto não vira amostra.** `FrameFeatures.complete` já diz se
   toda coluna de ML está preenchida (regra do Sprint 03: junta oculta é
   `None`, nunca um `0` inventado). `DatasetWriter` descarta e conta em vez de
   gravar — `skipped` no HUD mostra oclusão acontecendo, não depois do fato.
3. **Gravação é sem debounce, de propósito.** `GestureEngine` exige postura
   sustentada antes de contar; aqui cada frame com a tecla segurada vira uma
   linha, erro incluído — é matéria-prima pra treinar, uma amostra ruim é só
   uma linha a filtrar depois, não um veredito ao vivo que precisa acertar.

Segurar tecla no OpenCV não tem key-up nativo: `hello_dataset.py` infere
"ainda segurando" pelo auto-repeat do SO chegando mais rápido que
`RELEASE_TIMEOUT` (0.25s) — SPACE sempre para na hora, como rede de segurança.
`data/training/*` fica fora do git (só `.gitkeep`): dataset de gesto carrega
dado biométrico-ish, fica local por padrão.

### Decisões do Sprint 06: um gesto é um evento, não um estado

`backend/gestures/` fica entre a Motion Engine e o Event Engine do Sprint 09:
transforma `wrist_above_shoulder = True` sustentado, ou o punho oscilando pra
cima e pra baixo, num evento nomeado que sai uma vez, não a cada frame. Três
escolhas:

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
persistido ainda. Gravar em `data/events/` com cooldown de verdade fica pro
Event Engine do Sprint 09.

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
  vision/pose_detector.py  PoseDetector (único import de MediaPipe)
  vision/drawing.py   draw_arms, draw_hud, draw_joint_values, draw_trail
  features/angles.py  angle_between, elbow_angle, shoulder_angle
  features/distances.py  distance, shoulder_width, wrist_distance, forearm_length
  features/positions.py  wrist_height, wrist_above_shoulder, wrists_crossed
  features/features.py   FrameFeatures, extract, FEATURE_NAMES (ordem do dataset)
  features/velocity.py   SignalTrack, MotionTracker, MotionState, Direction, Trail
  gestures/rules.py   arm_raised, arms_crossed, arms_open (postura, sem tempo)
  gestures/engine.py  GestureEngine, GestureConfig, GestureEvent (hold + cooldown)
  dataset/writer.py   DatasetWriter, DATASET_COLUMNS — features+motion rotulados em CSV
  ml/train.py         TRAINING_COLUMNS, train, load_model, predict (Random Forest)
  events/             (sprint 09) event store
  models/             modelos .joblib treinados (fora do git)
  web/payload.py      formato de mensagem do WebSocket (hello/frame/error)
  web/pipeline.py     PipelineRunner, Subscriber — uma câmera, muitos clientes
  web/app.py          FastAPI: rotas /, /health, /ws
frontend/
  index.html          layout: canvas + painel de stream/motion/toggles
  style.css           cores do overlay, mesmas de vision/drawing.py
  app.js              desenha o payload no canvas; nunca calcula feature
data/training/        amostras de gestos (landmarks + features, nunca imagens)
data/events/          eventos detectados
scripts/              entrypoints de cada sprint
tests/
```

## Privacidade

- Nenhuma imagem sai da máquina; não há chamada de rede no pipeline.
- Frames vivem em memória e são descartados; `.gitignore` bloqueia `*.mp4/*.mov/*.png/*.jpg`.
- O dataset de treino guarda landmarks + features + label — nunca pixels.
