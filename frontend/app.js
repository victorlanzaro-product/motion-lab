/* Motion Lab web client.
 *
 * The browser is a display, not a second pipeline: it never computes a feature,
 * it draws what the WebSocket already decided. Three rules follow from that.
 *
 * 1. The mirror is applied to COORDINATES, not to the canvas. Flipping the
 *    context would flip the labels too and every number would read backwards.
 *    So the image is drawn under a flipped transform, the transform is undone,
 *    and the overlay uses `1 - x`.
 * 2. The bone topology comes from the `hello` message. The landmark table is
 *    domain knowledge that lives in Python; a copy here would drift.
 * 3. Only the newest frame is drawn. If decoding falls behind, the pending
 *    frame is replaced rather than queued — the same drop-the-stale-one rule
 *    the server applies per client.
 */

const DIRECTION_LABEL = {
  up: "subindo",
  down: "descendo",
  still: "parado",
  unknown: "—",
};

const GESTURE_LABEL = {
  arm_raised: "braço levantado",
  wave: "aceno",
  arms_crossed: "braços cruzados",
  arms_open: "braços abertos",
};

// Same vocabulary as GESTURE_LABEL, plus "idle" -- a training class the
// rule-based GestureEngine has no equivalent event for.
const ML_LABEL = { ...GESTURE_LABEL, idle: "parado" };

const SIDE_LABEL = { left: "esquerdo", right: "direito" };

const GESTURE_FEED_LIMIT = 6;
let toastTimer = null;

const COLORS = {
  accent: "#74f7b4",
  weak: "#ffb66e",
  bone: "#68c9ff",
  trail: "#c88cff",
  outline: "#070a0f",
  brow: "#ffd66b",
  // Sprint 11 -- face overlay. Two more otherwise-unused hues: a cool indigo
  // for the base contour (close enough to `bone` to read as "the same kind
  // of line as the skeleton", far enough to tell face from arm at a glance),
  // and a warm rose the mouth/eye chains blend towards as their observable
  // score rises (see `face-overlay.js:faceGroupColor`) -- never a colour
  // that stands for an emotion, just a highlight tied to a named signal
  // already on the wire (`face.smile`, `face.mouth_open`, `face.eye_blink_left/right`).
  faceLine: "#91aaff",
  faceHighlight: "#ff7fa3",
};

const state = {
  config: null,
  pending: null,
  drawing: false,
  socket: null,
  retries: 0,
  streaming: false,
  mirror: true,
  bodyOverlay: true,
  trail: false,
  labels: false,
  video: true,
  faceOverlay: true,
  faceEnabled: false,
  smoothedFaceLandmarks: null,
  activeTab: "now",
  lastFace: null,
  reportAvailable: false,
  coverageView: null,
  coverageFetchWarning: null,
  recordLabels: [],
};

const el = (id) => document.getElementById(id);
const canvas = el("view");
const ctx = canvas.getContext("2d");

// --- connection --------------------------------------------------------

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/ws`);
  state.socket = socket;

  socket.onopen = () => {
    state.retries = 0;
    confidenceTimeline.clear();
    // Not "ao vivo" yet: opening the webcam and loading the pose model takes a
    // few seconds on the first connection, and a green dot over a black canvas
    // reads as a broken app.
    setStatus("off", "ligando câmera…");
    hide("banner");
    const replay = recorder.prepareOutboundReplay();
    flushRecorderOutbound();
    if (replay.discardedStart) flashToast(Recorder.START_FAILED_ERROR);
    renderRecorder();
  };
  socket.onmessage = (event) => handle(JSON.parse(event.data));
  socket.onerror = () => socket.close();
  socket.onclose = () => {
    state.streaming = false;
    setStatus("off", "reconectando…");
    stopRecorderCountdownTimer();
    recorder.onWebSocketReconnect();
    renderRecorder();
    // Exponential backoff, capped: the usual reason for a close is the server
    // restarting, and hammering it while it boots helps nobody.
    const delay = Math.min(5000, 500 * 2 ** state.retries);
    state.retries += 1;
    setTimeout(connect, delay);
  };
}

function handle(message) {
  if (message.type === "hello") {
    state.config = message;
    applyReportAvailable(message.report_available);
    applyFaceEnabled(message.face_enabled);
    setPresenceChip(
      "face",
      message.face_enabled ? "pending" : "off",
      message.face_enabled ? "procurando rosto" : "rosto desligado",
    );
    applyRecordLabelsFromHello(message);
    renderRecorder();
    return;
  }
  if (message.type === "record_status") {
    recorder.applyStatus(message);
    renderRecorder();
    if (!message.recording && message.session_id != null) {
      fetchDatasetStats();
    }
    return;
  }
  if (message.type === "error") {
    setStatus("error", "parado");
    show("banner", message.message);
    return;
  }
  if (message.type === "status") {
    // Non-fatal lifecycle notice (backend/web/payload.py:status_message) --
    // the socket stays open, frames may resume, so this must never read like
    // the "parado" state `error` shows.
    if (message.state === "recovering") {
      setStatus("off", "reconectando à câmera…");
      show("banner", message.message || "câmera instável, tentando de novo…");
    }
    return;
  }
  if (message.type === "frame") {
    hide("banner");
    if (!state.streaming) {
      state.streaming = true;
      setStatus("live", "ao vivo");
    }
    // Gestures are edge events tied to this exact message — handled here,
    // not inside the render loop, so a slow decode never delays or drops one.
    if (message.gestures.length) handleGestures(message.gestures);
    // Recorded on arrival, not inside the (throttled) render loop: if decoding
    // falls behind, `render()` drops the stale pending frame and only draws
    // the newest one, but the history still needs every message's timestamp
    // to keep the ~10s window accurate.
    recordFaceHistory(message);
    recordConfidenceTimeline(message);
    state.pending = message;
    render();
  }
}

function handleGestures(events) {
  const feed = el("gesture-feed");
  for (const event of events) {
    const label = GESTURE_LABEL[event.name] || event.name;
    const side = event.side ? ` (${SIDE_LABEL[event.side] || event.side})` : "";
    const item = document.createElement("li");
    item.textContent = `${label}${side}`;
    feed.prepend(item);
    while (feed.children.length > GESTURE_FEED_LIMIT) feed.lastChild.remove();
    flashToast(`${label}${side}`);
  }
}

function flashToast(text) {
  const toast = el("toast");
  toast.textContent = text;
  toast.classList.remove("hidden");
  // Restart the CSS fade-out animation even if a gesture fires again mid-fade.
  toast.classList.remove("flash");
  void toast.offsetWidth; // force reflow so removing+re-adding the class replays it
  toast.classList.add("flash");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add("hidden"), 2000);
}

// --- render loop -------------------------------------------------------

async function render() {
  if (state.drawing || !state.pending || !state.config) return;
  const message = state.pending;
  state.pending = null;
  state.drawing = true;
  try {
    const bitmap = await createImageBitmap(toBlob(message.image.base64));
    draw(message, bitmap);
    bitmap.close();
    updatePanel(message);
  } catch (error) {
    console.error("frame descartado:", error);
  } finally {
    state.drawing = false;
    if (state.pending) requestAnimationFrame(render);
  }
}

function toBlob(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: "image/jpeg" });
}

function draw(message, bitmap) {
  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
  }

  if (state.video) {
    ctx.save();
    if (state.mirror) {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(bitmap, 0, 0);
    ctx.restore();
  } else {
    ctx.fillStyle = COLORS.outline;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  if (state.trail) drawTrail(message.trail);
  if (state.bodyOverlay) {
    drawBones(message.landmarks);
    drawJoints(message.landmarks);
  }
  // Numeric velocity labels belong to the diagnostic layer. Keeping them
  // behind the existing labels switch makes the default Studio preset calm.
  if (state.labels && state.bodyOverlay) drawVelocities(message);
  // Sprint 11 -- reuses `point()` below for the mirror, exactly like the arm
  // skeleton: raw coordinates travel on the wire, the flip happens here.
  if (state.faceOverlay) drawFaceOverlay(message.face);

  toggle("hint", !message.detected);
}

function point(x, y) {
  const mx = state.mirror ? 1 - x : x;
  return [mx * (canvas.width - 1), y * (canvas.height - 1)];
}

function reliable(landmark) {
  return landmark && landmark.v >= state.config.visibility_threshold;
}

function drawBones(landmarks) {
  ctx.strokeStyle = COLORS.bone;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  for (const [from, to] of state.config.chains) {
    const a = landmarks[from];
    const b = landmarks[to];
    // Same rule as draw_arms: a bone needs both ends reliable, otherwise a
    // hidden wrist would snap the line onto a guess.
    if (!reliable(a) || !reliable(b)) continue;
    ctx.beginPath();
    ctx.moveTo(...point(a.x, a.y));
    ctx.lineTo(...point(b.x, b.y));
    ctx.stroke();
  }
}

function drawJoints(landmarks) {
  for (const index of state.config.arm_landmarks) {
    const landmark = landmarks[index];
    if (!landmark) continue;
    const [x, y] = point(landmark.x, landmark.y);
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = reliable(landmark) ? COLORS.accent : COLORS.weak;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLORS.outline;
    ctx.stroke();
    if (state.labels) {
      const name = state.config.landmark_names[index] || index;
      label(`${name} ${landmark.v.toFixed(2)}`, x + 10, y - 8, COLORS.accent);
    }
  }
}

// --- face overlay (Sprint 11) -------------------------------------------
//
// Same contract as `drawBones`/`drawJoints`: topology comes from the `hello`
// message (`state.config.face_chains`), coordinates are raw/unmirrored on
// the wire and only flipped here via the shared `point()` helper, and a
// chain is skipped rather than guessed when either endpoint is missing.
// Unlike the arm skeleton, there is no per-point reliability score to check
// (Face Landmarker gives none -- see `backend/vision/face_landmarks.py`'s
// `FaceSnapshot` docstring) -- the one honest gate is frame-level: no
// detected face means `face.landmarks` arrives empty and nothing is drawn.

// The actual topology walk, no-face gate, missing-endpoint skip and colour
// blend all live in face-overlay.js (`FaceOverlay.buildFaceOverlayLines`) so
// they run under `node --test` with no DOM — this function only turns the
// returned segments into `ctx.stroke()` calls, mirroring exactly like
// `drawBones` via the shared `point()` helper.
function drawFaceOverlay(face) {
  if (!face || !face.available || !face.detected || !face.landmarks) {
    state.smoothedFaceLandmarks = null;
    return;
  }
  const landmarks = FaceOverlay.smoothLandmarks(
    state.smoothedFaceLandmarks,
    face.landmarks,
    0.42,
  );
  state.smoothedFaceLandmarks = landmarks;
  const displayFace = { ...face, landmarks };
  const groups = state.config && state.config.face_chains;
  const lines = FaceOverlay.buildFaceOverlayLines(displayFace, groups, {
    line: COLORS.faceLine,
    highlight: COLORS.faceHighlight,
  });
  ctx.save();
  ctx.globalAlpha = 0.84;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const seg of lines) {
    ctx.strokeStyle = seg.color;
    ctx.lineWidth = seg.group === "face_oval" ? 1.35 : 1.65;
    ctx.beginPath();
    ctx.moveTo(...point(seg.from.x, seg.from.y));
    ctx.lineTo(...point(seg.to.x, seg.to.y));
    ctx.stroke();
  }
  ctx.restore();
}

function drawTrail(trail) {
  ctx.strokeStyle = COLORS.trail;
  ctx.lineCap = "round";
  for (const segments of Object.values(trail)) {
    for (const run of segments) {
      for (let i = 1; i < run.length; i += 1) {
        // Older points thinner: the taper reads as direction of travel without
        // needing an arrowhead, exactly like draw_trail does.
        ctx.lineWidth = 1 + (3 * i) / run.length;
        ctx.beginPath();
        ctx.moveTo(...point(run[i - 1][0], run[i - 1][1]));
        ctx.lineTo(...point(run[i][0], run[i][1]));
        ctx.stroke();
      }
    }
  }
}

function drawVelocities(message) {
  for (const side of ["left", "right"]) {
    const wrist = message.landmarks[wristIndex(side)];
    const velocity = message.motion[`${side}_velocity_y`];
    if (!reliable(wrist) || velocity === null || velocity === undefined) continue;
    const [x, y] = point(wrist.x, wrist.y);
    label(velocity.toFixed(2), x + 12, y + 5, COLORS.accent);
  }
}

function wristIndex(side) {
  const wanted = `${side.toUpperCase()}_WRIST`;
  return Object.keys(state.config.landmark_names).find(
    (index) => state.config.landmark_names[index] === wanted,
  );
}

function label(text, x, y, color) {
  ctx.font = "12px ui-monospace, Menlo, monospace";
  // Dark outline first, so the text survives a bright background.
  ctx.lineWidth = 3;
  ctx.strokeStyle = COLORS.outline;
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

// --- panel -------------------------------------------------------------

function fmt(value, digits = 2) {
  return value === null || value === undefined ? "—" : value.toFixed(digits);
}

// --- Sprint 15 dashboard ------------------------------------------------
//
// One global timeline (single camera / pipeline). The browser only stores and
// draws what the frame payload already decided -- no feature recompute.

const confidenceTimeline = ConfidenceTimeline.createTimeline();

const TIMING_VALUE_IDS = {
  capture_ms: "timing-capture",
  pose_ms: "timing-pose",
  features_ms: "timing-features",
  ml_ms: "timing-ml",
  face_ms: "timing-face",
};

const TIMING_BAR_IDS = {
  capture_ms: "timing-bar-capture",
  pose_ms: "timing-bar-pose",
  features_ms: "timing-bar-features",
  ml_ms: "timing-bar-ml",
  face_ms: "timing-bar-face",
};

const AGREEMENT_COLORS = {
  agree: COLORS.accent,
  disagree: "#ff747b",
  unknown: "rgba(141, 155, 173, 0.35)",
};

function applyReportAvailable(available) {
  state.reportAvailable = Boolean(available);
  const link = el("report-link");
  if (!link) return;
  // Maestro S15: the link always opens /report; 404 carries the short message.
  // `report_available` only drives the hint, never disables navigation.
  link.classList.toggle("pending", !state.reportAvailable);
  link.title = state.reportAvailable
    ? "Abre o último relatório HTML gerado no servidor"
    : "Relatório ainda não gerado — rode scripts/report.py";
}

function recordConfidenceTimeline(message) {
  confidenceTimeline.push(message);
}

function updateHealthPanel(message) {
  const view = HealthPanel.snapshot(message);
  el("health-fps").textContent = view.fpsDisplay;
  if (view.dropped_frames === null || view.dropped_frames === undefined) {
    el("health-dropped-frames").textContent = "—";
  } else {
    el("health-dropped-frames").textContent = String(view.dropped_frames);
  }
  const rateNode = el("health-dropped-rate");
  rateNode.textContent = view.droppedRateDisplay;
  rateNode.classList.toggle("alert", view.droppedRateAlert);
  const rateLabel = el("pipeline-health-panel")?.querySelector(".health-dropped-label");
  if (rateLabel) rateLabel.classList.toggle("alert", view.droppedRateAlert);
  el("health-total-ms").textContent =
    view.totalDisplay === "—" ? view.totalDisplay : `${view.totalDisplay} ms`;
  for (const stage of view.stages) {
    const valueId = TIMING_VALUE_IDS[stage.key];
    const barId = TIMING_BAR_IDS[stage.key];
    if (valueId) {
      el(valueId).textContent = stage.display === "—" ? "—" : `${stage.display} ms`;
    }
    if (barId) el(barId).style.width = `${Math.round(stage.barFraction * 100)}%`;
  }
}

const mlConfidenceCanvas = el("ml-confidence-chart");
const mlConfidenceCtx = mlConfidenceCanvas ? mlConfidenceCanvas.getContext("2d") : null;

function renderConfidenceTimeline() {
  const rateNode = el("ml-agreement-rate");
  if (!mlConfidenceCanvas || !mlConfidenceCtx) return;
  if (state.activeTab !== "history") return;

  clearChartCanvas(mlConfidenceCanvas, mlConfidenceCtx);
  drawChartGrid(mlConfidenceCanvas, mlConfidenceCtx);

  const now = confidenceTimeline.now();
  if (now === null) {
    if (rateNode) rateNode.textContent = "—";
    return;
  }

  const points = confidenceTimeline.series(now);
  const bandHeight = 10;
  const padTop = 6;
  const plotHeight = mlConfidenceCanvas.height - bandHeight - padTop - 4;
  const width = mlConfidenceCanvas.width;
  const slice =
    points.length > 1 ? Math.max(1, Math.ceil(width / Math.max(points.length - 1, 1))) : 1;

  for (const point of points) {
    const x = Math.round(point.x * (width - 1));
    let color = AGREEMENT_COLORS.unknown;
    if (point.agreement === true) color = AGREEMENT_COLORS.agree;
    else if (point.agreement === false) color = AGREEMENT_COLORS.disagree;
    mlConfidenceCtx.fillStyle = color;
    mlConfidenceCtx.fillRect(x, mlConfidenceCanvas.height - bandHeight - 2, slice, bandHeight);
  }

  mlConfidenceCtx.strokeStyle = COLORS.bone;
  mlConfidenceCtx.lineWidth = 1.5;
  mlConfidenceCtx.lineJoin = "round";
  for (const run of confidenceTimeline.probabilityRuns(now)) {
    mlConfidenceCtx.beginPath();
    run.forEach((point, index) => {
      const x = point.x * (width - 1);
      const y = padTop + (1 - point.mlProbability) * plotHeight;
      if (index === 0) mlConfidenceCtx.moveTo(x, y);
      else mlConfidenceCtx.lineTo(x, y);
    });
    mlConfidenceCtx.stroke();
  }

  const rate = confidenceTimeline.agreementRate();
  if (rateNode) {
    rateNode.textContent = rate === null ? "—" : `${Math.round(rate * 100)}% concorda`;
  }
}

function updatePanel(message) {
  state.lastFace = message.face;
  setPresenceChip(
    "body",
    message.detected ? "live" : "missing",
    message.detected ? "corpo detectado" : "procurando corpo",
  );
  updateFacePresence(message.face);
  el("fps").textContent = fmt(message.fps, 1);
  el("inference").textContent = fmt(message.inference_ms, 1);
  el("confidence").textContent = fmt(message.confidence, 2);
  el("frame-index").textContent = message.frame_index;
  updateMlPanel(message.ml);
  updateFacePanel(message.face);
  renderFaceCharts(message.face);
  updateHealthPanel(message);
  renderConfidenceTimeline();

  for (const side of ["left", "right"]) {
    const direction = message.motion[`${side}_direction`] || "unknown";
    const badge = el(`${side}-direction`);
    badge.textContent = DIRECTION_LABEL[direction];
    badge.className = `badge ${direction}`;
    el(`${side}-vy`).textContent = fmt(message.motion[`${side}_velocity_y`]);
    el(`${side}-speed`).textContent = fmt(message.motion[`${side}_speed`]);
    el(`${side}-elbow`).textContent = fmt(message.features[`${side}_elbow_angle`], 0);
    el(`${side}-height`).textContent = fmt(message.features[`${side}_wrist_height`]);
  }
}

// Sprint 11 — facial signals. Every field here is a named, observable
// signal (or its rate of change) straight off the wire; nothing here is an
// emotion or mental-state label, and none of this is ever computed client
// side — the browser only ever displays what `backend/web/payload.py`
// already decided.
const FACE_FIELDS = [
  "face-smile",
  "face-mouth",
  "face-blink-left",
  "face-blink-right",
  "face-blink-rate-left",
  "face-blink-rate-right",
  "face-brow",
  "face-yaw",
  "face-pitch",
  "face-roll",
];

function updateSignalMeter(id, value) {
  el(id).style.width = `${StudioControls.signalPercent(value)}%`;
}

function updateFacePresence(face) {
  if (!state.faceEnabled) {
    setPresenceChip("face", "off", "rosto desligado");
  } else if (!face || !face.available) {
    setPresenceChip("face", "error", "sinal facial indisponível");
  } else if (face.detected) {
    setPresenceChip("face", "live", "rosto detectado");
  } else {
    setPresenceChip("face", "missing", "procurando rosto");
  }
}

function updateFacePanel(face) {
  const status = el("face-status");
  if (!face || !face.available) {
    status.textContent = face && face.reason === "disabled" ? "desligado" : "indisponível";
    status.className = "badge";
    for (const id of FACE_FIELDS) el(id).textContent = "—";
    el("face-eyes-summary").textContent = "aguardando sinal";
    el("face-head-direction").textContent = "aguardando sinal";
    updateSignalMeter("face-smile-meter", null);
    return;
  }
  status.textContent = face.detected ? "detectado" : "sem rosto";
  status.className = `badge ${face.detected ? "up" : ""}`;
  el("face-smile").textContent = fmt(face.smile);
  el("face-mouth").textContent = fmt(face.mouth_open);
  updateSignalMeter("face-smile-meter", face.smile);
  // "fechamento olho" is this frame's eyeBlink* blendshape value (0 = open, 1
  // = closed) -- a snapshot, not an event. "piscadas/min" is the actual
  // blink-event rate `FaceMotionTracker._BlinkCounter` derives from rising
  // edges of that same signal (backend/features/face_motion.py). Showing
  // both side by side is the point: a slow deliberate blink can hold
  // "fechamento" near 1 for a while without a second blink event ever firing.
  el("face-blink-left").textContent = fmt(face.eye_blink_left);
  el("face-blink-right").textContent = fmt(face.eye_blink_right);
  el("face-blink-rate-left").textContent = fmt(face.blink_rate_left, 1);
  el("face-blink-rate-right").textContent = fmt(face.blink_rate_right, 1);
  el("face-brow").textContent = fmt(face.brow_raise);
  el("face-yaw").textContent = fmt(face.head_yaw, 0);
  el("face-pitch").textContent = fmt(face.head_pitch, 0);
  el("face-roll").textContent = fmt(face.head_roll, 0);
  el("face-eyes-summary").textContent = StudioControls.describeEyes(face);
  el("face-head-direction").textContent = StudioControls.describeHead(face);
}

// --- face charts (Sprint 11) --------------------------------------------
//
// Rolling ~10s line charts for the facial signals above, drawn from
// `FaceHistory` (face-history.js) -- a browser-memory-only buffer, never
// written to storage, cleared by a page reload like everything else here.
// Two charts because the two families live on different axes: smile/mouth/
// blink/brow are all the same 0..1 blendshape unit, head yaw/pitch/roll are
// degrees -- sharing one axis between them would misrepresent both.

const FACE_HISTORY_WINDOW_SECONDS = 10;

const FACE_LEVEL_SERIES = [
  { name: "smile", label: "sorriso", color: COLORS.accent, valueId: "face-legend-smile" },
  { name: "mouth_open", label: "boca aberta", color: COLORS.bone, valueId: "face-legend-mouth" },
  {
    name: "eye_blink_left",
    label: "olho esq. fechado",
    color: COLORS.trail,
    valueId: "face-legend-blink-left",
  },
  {
    name: "eye_blink_right",
    label: "olho dir. fechado",
    color: COLORS.weak,
    valueId: "face-legend-blink-right",
  },
  { name: "brow_raise", label: "sobrancelhas", color: COLORS.brow, valueId: "face-legend-brow" },
];

const FACE_HEAD_SERIES = [
  { name: "head_yaw", label: "guinada (yaw)", color: COLORS.accent, valueId: "face-legend-yaw" },
  {
    name: "head_pitch",
    label: "inclinação (pitch)",
    color: COLORS.bone,
    valueId: "face-legend-pitch",
  },
  { name: "head_roll", label: "rotação (roll)", color: COLORS.trail, valueId: "face-legend-roll" },
];

// One series name per store slot; the store itself only ever holds
// {t, v} pairs in plain arrays -- see face-history.js for the pruning and
// gap rules that keep this bounded and honest.
const faceHistory = FaceHistory.createStore(
  [...FACE_LEVEL_SERIES, ...FACE_HEAD_SERIES].map((s) => s.name),
  { windowSeconds: FACE_HISTORY_WINDOW_SECONDS, maxPoints: 1200, maxGapSeconds: 1.0 },
);

const levelsCanvas = el("face-chart-levels");
const levelsCtx = levelsCanvas.getContext("2d");
const headCanvas = el("face-chart-head");
const headCtx = headCanvas.getContext("2d");
const levelSeriesSelect = el("face-level-series-select");
const headSeriesSelect = el("face-head-series-select");

function selectedSeries(seriesDefs, select) {
  return [seriesDefs.find((def) => def.name === select.value) || seriesDefs[0]];
}

function buildChartLegend(containerId, seriesDefs) {
  const container = el(containerId);
  container.innerHTML = "";
  for (const def of seriesDefs) {
    const item = document.createElement("li");
    const dot = document.createElement("span");
    dot.className = "chart-dot";
    dot.style.background = def.color;
    const label = document.createElement("span");
    label.textContent = def.label;
    const value = document.createElement("span");
    value.className = "chart-value";
    value.id = def.valueId;
    value.textContent = "—";
    item.append(dot, label, value);
    container.appendChild(item);
  }
}

function refreshChartLegends() {
  buildChartLegend("face-legend-levels", selectedSeries(FACE_LEVEL_SERIES, levelSeriesSelect));
  buildChartLegend("face-legend-head", selectedSeries(FACE_HEAD_SERIES, headSeriesSelect));
}

refreshChartLegends();

for (const select of [levelSeriesSelect, headSeriesSelect]) {
  select.addEventListener("change", () => {
    refreshChartLegends();
    renderFaceCharts(state.lastFace, true);
  });
}

// Called on every incoming frame (backend/web/payload.py:face_payload),
// whether or not a face is available/detected -- pushing every frame,
// including the unavailable ones, is what lets `FaceHistory.buildSegments`
// break the line instead of bridging across the gap. When a face IS
// detected the individual fields are still independently nullable (a
// blendshape MediaPipe didn't return this frame); `createStore.push`
// already turns a non-finite/missing value into that same break, so no
// extra detected-flag check is needed here.
function recordFaceHistory(message) {
  const face = message.face;
  const available = face && face.available;
  const values = available
    ? {
        smile: face.smile,
        mouth_open: face.mouth_open,
        eye_blink_left: face.eye_blink_left,
        eye_blink_right: face.eye_blink_right,
        brow_raise: face.brow_raise,
        head_yaw: face.head_yaw,
        head_pitch: face.head_pitch,
        head_roll: face.head_roll,
      }
    : null;
  // `message.timestamp` (snapshot.timestamp, seconds) is the same clock the
  // arm trail/motion already key off of -- a time-based window, not a frame
  // count, so it stays "~10s" regardless of fps.
  faceHistory.push(message.timestamp, values);
}

function clearChartCanvas(canvas, ctx) {
  ctx.fillStyle = "#05070a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawChartGrid(canvas, ctx) {
  ctx.strokeStyle = COLORS.outline;
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 3]);
  for (const frac of [0, 0.5, 1]) {
    const y = canvas.height * (1 - frac);
    ctx.beginPath();
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(canvas.width, y + 0.5);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function drawHistoryChart(canvas, ctx, seriesDefs, range) {
  clearChartCanvas(canvas, ctx);
  drawChartGrid(canvas, ctx);
  const now = faceHistory.now();
  if (now === null) return; // nothing recorded yet this connection

  const layout = {
    now,
    windowSeconds: faceHistory.windowSeconds,
    min: range.min,
    max: range.max,
    width: canvas.width,
    height: canvas.height,
  };
  for (const def of seriesDefs) {
    ctx.strokeStyle = def.color;
    ctx.fillStyle = def.color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    for (const run of faceHistory.segments(def.name)) {
      if (run.length === 1) {
        // A single sample between two gaps still deserves to be visible --
        // draw it as a dot rather than silently dropping it.
        const p = FaceHistory.layoutPoint(run[0], layout);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }
      ctx.beginPath();
      run.forEach((sample, i) => {
        const p = FaceHistory.layoutPoint(sample, layout);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
    }
  }
}

function updateChartLegendValues(seriesDefs, face) {
  for (const def of seriesDefs) {
    el(def.valueId).textContent = face ? fmt(face[def.name], def.name.startsWith("head_") ? 0 : 2) : "—";
  }
}

// `face` here is `message.face`, already read once by `updateFacePanel` --
// reusing `#face-status`'s text keeps "indisponível/desligado/detectado/sem
// rosto" a single source of truth instead of a second copy of that logic.
function renderFaceCharts(face, force = false) {
  // The panel toggle (`#toggle-face`) only hides this tab's view of a signal
  // the server may still be computing; it never turns the drawing off from
  // the server's perspective, but it SHOULD stop this browser tab from
  // spending time painting charts nobody can see.
  if (!faceToggle.checked || (!force && state.activeTab !== "history")) return;

  const stateText = el("face-status").textContent;
  el("face-chart-levels-state").textContent = stateText;
  el("face-chart-head-state").textContent = stateText;

  const available = face && face.available;
  if (!available) {
    clearChartCanvas(levelsCanvas, levelsCtx);
    clearChartCanvas(headCanvas, headCtx);
    updateChartLegendValues(selectedSeries(FACE_LEVEL_SERIES, levelSeriesSelect), null);
    updateChartLegendValues(selectedSeries(FACE_HEAD_SERIES, headSeriesSelect), null);
    return;
  }

  const visibleLevels = selectedSeries(FACE_LEVEL_SERIES, levelSeriesSelect);
  const visibleHead = selectedSeries(FACE_HEAD_SERIES, headSeriesSelect);
  drawHistoryChart(levelsCanvas, levelsCtx, visibleLevels, { min: 0, max: 1 });
  drawHistoryChart(headCanvas, headCtx, visibleHead, { min: -90, max: 90 });
  // The legend shows THIS frame's reading, same as the numeric panel above
  // it; when nobody is in frame right now (`detected: false`) the fields are
  // already `null` on the wire (backend/features/face_features.py), so `fmt`
  // renders "—" here exactly like it does in `updateFacePanel`.
  updateChartLegendValues(visibleLevels, face);
  updateChartLegendValues(visibleHead, face);
}

function updateMlPanel(ml) {
  const label = el("ml-label");
  const confidence = el("ml-confidence");
  if (ml === null) {
    // Two reasons this frame has nothing to say, told apart by whether a
    // model exists at all: no model trained yet, or this exact frame had an
    // occluded joint and the backend skipped it rather than guessing.
    const hasModel = state.config && state.config.ml_classes && state.config.ml_classes.length;
    label.textContent = hasModel ? "sem sinal" : "sem modelo";
    label.className = "badge";
    confidence.textContent = "—";
    return;
  }
  label.textContent = ML_LABEL[ml.label] || ml.label;
  label.className = `badge ${ml.label === "idle" ? "still" : "up"}`;
  confidence.textContent = `${Math.round(ml.confidence * 100)}%`;
}

function setStatus(kind, text) {
  el("dot").className = `dot ${kind}`;
  el("status-text").textContent = text;
  const chipKind = kind === "live" ? "live" : kind === "error" ? "error" : "pending";
  const chipText = kind === "live" ? "câmera ativa" : text;
  setPresenceChip("camera", chipKind, chipText);
}

function setPresenceChip(name, kind, text) {
  const chip = el(`${name}-chip`);
  const copy = el(`${name}-chip-text`);
  if (!chip || !copy) return;
  chip.className = `presence-chip ${kind}`;
  copy.textContent = text;
}

function show(id, text) {
  const node = el(id);
  node.textContent = text;
  node.classList.remove("hidden");
}

function hide(id) {
  el(id).classList.add("hidden");
}

function toggle(id, visible) {
  el(id).classList.toggle("hidden", !visible);
}

// --- controls ----------------------------------------------------------

const TOGGLES = {
  mirror: { key: "m", id: "toggle-mirror" },
  video: { key: "v", id: "toggle-video" },
  bodyOverlay: { key: "b", id: "toggle-body-overlay" },
  trail: { key: "t", id: "toggle-trail" },
  labels: { key: "l", id: "toggle-labels" },
};

function typingTarget(target) {
  return ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target.tagName);
}

for (const [name, control] of Object.entries(TOGGLES)) {
  const box = el(control.id);
  state[name] = box.checked;
  box.addEventListener("change", () => {
    state[name] = box.checked;
    syncPresetButtons();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() !== control.key || typingTarget(event.target)) return;
    box.checked = !box.checked;
    box.dispatchEvent(new Event("change"));
  });
}

// The face panel's toggle only shows/hides the panel in this browser tab --
// it never enables or disables the facial signal processing itself, the same
// way the "ml ao vivo" panel has no client toggle to turn training off. That
// processing is `WebConfig.face_enabled` (server, `--face`), reported once in
// `hello.face_enabled` -- `applyFaceEnabled` below is what actually respects
// it: when the server never turned the signal on, checking this box would
// only ever reveal a panel permanently stuck on "desligado", so it stays
// unchecked and disabled instead of pretending there is something to show.
const faceToggle = el("toggle-face");
const faceToggleLabel = faceToggle.closest("label");

// Sprint 11 -- a SEPARATE toggle from `faceToggle` above on purpose: that one
// shows/hides the side panel + charts (numbers and history), this one
// shows/hides the on-video landmark drawing. A viewer who wants the overlay
// but not the panel (or vice versa) is a real, common case -- one checkbox
// controlling both would make that impossible. Key `o` ("overlay"), distinct
// from panel's `f` ("face"), same gating rule: disabled + unchecked when the
// server itself never turned facial tracking on (`--face`).
const faceOverlayToggle = el("toggle-face-overlay");
const faceOverlayToggleLabel = faceOverlayToggle.closest("label");

function applyFaceEnabled(faceEnabled) {
  const becameAvailable = !state.faceEnabled && faceEnabled;
  state.faceEnabled = Boolean(faceEnabled);
  faceToggle.disabled = !faceEnabled;
  faceOverlayToggle.disabled = !faceEnabled;
  if (!faceEnabled) {
    faceToggle.checked = false;
    faceOverlayToggle.checked = false;
    if (faceToggleLabel) faceToggleLabel.title = "desligado no servidor (rode com --face)";
    if (faceOverlayToggleLabel) {
      faceOverlayToggleLabel.title = "desligado no servidor (rode com --face)";
    }
  } else {
    if (becameAvailable) {
      faceToggle.checked = true;
      faceOverlayToggle.checked = true;
    }
    if (faceToggleLabel) faceToggleLabel.title = "";
    if (faceOverlayToggleLabel) faceOverlayToggleLabel.title = "";
  }
  state.faceOverlay = faceOverlayToggle.checked;
  updateFaceSections();
  syncPresetButtons();
}

function updateFaceSections() {
  const visible = state.faceEnabled && faceToggle.checked;
  toggle("face-panel", visible);
  toggle("face-history-panel", visible);
  toggle("face-ui-empty", !visible);
  toggle("face-history-empty", !visible);
  el("face-ui-empty-copy").textContent = state.faceEnabled
    ? "Ative os dados faciais no painel de diagnóstico."
    : "Reinicie o servidor com --face para habilitar esta camada.";
}

updateFaceSections();
faceToggle.addEventListener("change", () => {
  updateFaceSections();
  if (faceToggle.checked) renderFaceCharts(state.lastFace, true);
});
document.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() !== "f" || typingTarget(event.target) || faceToggle.disabled) return;
  faceToggle.checked = !faceToggle.checked;
  faceToggle.dispatchEvent(new Event("change"));
});

state.faceOverlay = faceOverlayToggle.checked;
faceOverlayToggle.addEventListener("change", () => {
  state.faceOverlay = faceOverlayToggle.checked;
  if (!state.faceOverlay) state.smoothedFaceLandmarks = null;
  syncPresetButtons();
});
document.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() !== "o" || typingTarget(event.target) || faceOverlayToggle.disabled) return;
  faceOverlayToggle.checked = !faceOverlayToggle.checked;
  faceOverlayToggle.dispatchEvent(new Event("change"));
});

// --- Live Studio tabs -------------------------------------------------

const tabButtons = Array.from(document.querySelectorAll("[data-tab]"));
const tabPanels = Array.from(document.querySelectorAll("[data-panel]"));

function activateTab(name, focus = false) {
  state.activeTab = name;
  for (const button of tabButtons) {
    const active = button.dataset.tab === name;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    if (active && focus) button.focus();
  }
  for (const panel of tabPanels) panel.hidden = panel.dataset.panel !== name;
  if (name === "history") {
    renderFaceCharts(state.lastFace, true);
    renderConfidenceTimeline();
  }
}

tabButtons.forEach((button, index) => {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
  button.addEventListener("keydown", (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const offset = event.key === 'ArrowRight' ? 1 : -1;
    const next = (index + offset + tabButtons.length) % tabButtons.length;
    activateTab(tabButtons[next].dataset.tab, true);
  });
});

activateTab("now");

// --- presentation presets --------------------------------------------

const presetButtons = Array.from(document.querySelectorAll("[data-preset]"));
const PRESET_TOGGLE_IDS = {
  bodyOverlay: "toggle-body-overlay",
  trail: "toggle-trail",
  labels: "toggle-labels",
  faceOverlay: "toggle-face-overlay",
};

function syncPresetButtons() {
  if (!presetButtons.length) return;
  const match = StudioControls.matchingPreset(state, state.faceEnabled);
  for (const button of presetButtons) {
    const active = button.dataset.preset === match;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.disabled = button.dataset.preset === "face" && !state.faceEnabled;
  }
}

function applyPreset(name) {
  const preset = StudioControls.presetFor(name, state.faceEnabled);
  if (!preset) return;
  for (const [key, value] of Object.entries(preset)) {
    const box = el(PRESET_TOGGLE_IDS[key]);
    box.checked = value;
    box.dispatchEvent(new Event("change"));
  }
  syncPresetButtons();
}

for (const button of presetButtons) {
  button.addEventListener("click", () => applyPreset(button.dataset.preset));
}

syncPresetButtons();

// --- Sprint 12 recorder ------------------------------------------------

const recorder = Recorder.createRecorder();
let recorderCountdownTimer = null;
let recorderLabels = [];

function recorderSocketOpen() {
  return Boolean(state.socket) && state.socket.readyState === WebSocket.OPEN;
}

function sendRecorderOutbound(messages) {
  if (!messages.length) return [];
  if (!recorderSocketOpen()) return messages;
  for (const payload of messages) {
    state.socket.send(JSON.stringify(payload));
  }
  return [];
}

function deliverRecorderOutbound(batch) {
  if (!batch.length) return;
  const payloads = recorder.outboundPayloads(batch);
  const unsent = sendRecorderOutbound(payloads);
  if (!unsent.length) return;
  const unsentKeys = new Set(unsent.map((m) => JSON.stringify(m)));
  const toRequeue = batch.filter((entry, index) => unsentKeys.has(JSON.stringify(payloads[index])));
  recorder.requeueOutbound(toRequeue);
}

function flushRecorderOutbound() {
  deliverRecorderOutbound(recorder.drainOutbound());
}

function stopRecorderCountdownTimer() {
  if (recorderCountdownTimer === null) return;
  clearInterval(recorderCountdownTimer);
  recorderCountdownTimer = null;
}

function ensureRecorderCountdownTimer() {
  if (recorderCountdownTimer !== null) return;
  recorderCountdownTimer = setInterval(() => {
    const outbound = recorder.tick();
    if (outbound.length) deliverRecorderOutbound(outbound);
    renderRecorder();
    if (recorder.snapshot().phase !== "countdown") stopRecorderCountdownTimer();
  }, 100);
}

function applyRecordLabelsFromHello(hello) {
  const labels = Recorder.recordLabelsForHello(hello);
  state.recordLabels = labels;
  recorder.setLabels(labels);
  buildRecorderButtons(labels, hello && hello.model_schema);
}

function recorderModelSchema() {
  return state.config && state.config.model_schema;
}

function buildRecorderButtons(labels, modelSchema) {
  recorderLabels = labels.slice();
  const host = el("recorder-label-buttons");
  host.replaceChildren();
  const schema = modelSchema ?? recorderModelSchema();
  for (const label of labels) {
    const shortcut = Recorder.shortcutForLabel(label, labels, schema);
    const human = ML_LABEL[label] || label;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "recorder-label-button";
    if (label === "wave") button.classList.add("recorder-label-button-wave");
    button.dataset.recordLabel = label;
    if (shortcut) button.dataset.shortcut = shortcut;
    button.textContent = human;
    if (shortcut) {
      const key = document.createElement("kbd");
      key.textContent = shortcut;
      button.append(key);
    }
    button.addEventListener("click", () => startRecorderLabel(label));
    host.append(button);
  }
}

function startRecorderLabel(label) {
  if (!recorder.beginStart(label).ok) return;
  ensureRecorderCountdownTimer();
  flushRecorderOutbound();
  renderRecorder();
}

function stopRecorder() {
  if (!recorder.beginStop().ok) return;
  flushRecorderOutbound();
  renderRecorder();
}

function cancelRecorderCountdown() {
  if (!recorder.cancelCountdown()) return;
  stopRecorderCountdownTimer();
  flushRecorderOutbound();
  renderRecorder();
}

function renderRecorder() {
  const view = recorder.snapshot();
  const banner = el("recording-banner");
  const bannerDetail = el("recording-banner-detail");
  toggle("recording-banner", view.showRecordingBanner);
  if (view.showRecordingBanner) {
    const human = ML_LABEL[view.status.label] || view.status.label || "—";
    bannerDetail.textContent = view.isRemoteRecording
      ? `outra aba · ${human}`
      : human;
  } else {
    bannerDetail.textContent = "";
  }

  toggle("recorder-live-badge", view.showRecordingBanner);
  toggle("recorder-remote-note", view.isRemoteRecording);
  toggle("recorder-cancel-hint", view.cancelCountdownEnabled);

  const countdown = el("recorder-countdown");
  if (view.phase === "countdown" && view.countdownRemaining !== null) {
    const human = ML_LABEL[view.countdownLabel] || view.countdownLabel;
    countdown.textContent = `Iniciando “${human}” em ${view.countdownRemaining}s…`;
    toggle("recorder-countdown", true);
  } else {
    countdown.textContent = "";
    toggle("recorder-countdown", false);
  }

  const errorBox = el("recorder-error");
  const errorText = view.status.error;
  if (errorText) {
    errorBox.textContent = errorText;
    toggle("recorder-error", true);
  } else {
    errorBox.textContent = "";
    toggle("recorder-error", false);
  }

  for (const button of el("recorder-label-buttons").querySelectorAll("[data-record-label]")) {
    button.disabled = !view.startEnabled;
  }
  const stopButton = el("recorder-stop");
  stopButton.disabled = !view.stopEnabled;

  if (view.status.recording) {
    el("recorder-written").textContent = String(view.status.written);
    el("recorder-skipped").textContent = String(view.status.skipped);
  } else if (view.lastSummary) {
    el("recorder-written").textContent = String(view.lastSummary.written);
    el("recorder-skipped").textContent = String(view.lastSummary.skipped);
  } else {
    el("recorder-written").textContent = "—";
    el("recorder-skipped").textContent = "—";
  }

  const last = view.lastSummary;
  if (last) {
    toggle("recorder-last", true);
    el("recorder-last-label").textContent = ML_LABEL[last.label] || last.label;
    el("recorder-last-session").textContent = last.session_id;
    el("recorder-last-written").textContent = String(last.written);
    el("recorder-last-skipped").textContent = String(last.skipped);
  } else {
    toggle("recorder-last", false);
  }
}

el("recorder-stop").addEventListener("click", () => stopRecorder());

// --- Sprint 13 dataset coverage ----------------------------------------

const COVERAGE_FETCH_WARNING =
  "Não foi possível atualizar a cobertura do dataset. A gravação e o ao vivo seguem normais.";

async function fetchDatasetStats() {
  try {
    const response = await fetch("/dataset/stats");
    if (!response.ok) throw new Error(String(response.status));
    const stats = await response.json();
    state.coverageView = DatasetCoverage.buildCoverageView(stats, {
      labelTitles: ML_LABEL,
      coverageLabels: state.recordLabels.length ? state.recordLabels : undefined,
    });
    state.coverageFetchWarning = null;
  } catch (_error) {
    state.coverageFetchWarning = COVERAGE_FETCH_WARNING;
  }
  renderDatasetCoverage();
}

function renderDatasetCoverage() {
  const warning = el("coverage-fetch-warning");
  if (state.coverageFetchWarning) {
    warning.textContent = state.coverageFetchWarning;
    toggle("coverage-fetch-warning", true);
  } else {
    warning.textContent = "";
    toggle("coverage-fetch-warning", false);
  }

  const view = state.coverageView;
  if (!view || !view.valid) {
    el("coverage-summary").textContent = state.coverageFetchWarning
      ? "Cobertura desatualizada — última leitura indisponível."
      : "Consultando estatísticas…";
    el("coverage-status-badge").textContent = state.coverageFetchWarning ? "indisponível" : "carregando";
    el("coverage-status-badge").className = "badge coverage-badge";
    toggle("coverage-path", false);
    toggle("coverage-problems", false);
    el("coverage-label-grid").replaceChildren();
    toggle("coverage-dropped", false);
    return;
  }

  const badge = el("coverage-status-badge");
  badge.textContent = view.statusText;
  badge.className = `badge coverage-badge coverage-${view.statusKind}`;

  el("coverage-summary").textContent = view.summaryLine;

  const pathNode = el("coverage-path");
  if (view.path) {
    pathNode.textContent = `Arquivo: ${view.path}`;
    toggle("coverage-path", true);
  } else {
    pathNode.textContent = "";
    toggle("coverage-path", false);
  }

  const problems = el("coverage-problems");
  problems.replaceChildren();
  if (view.problems.length) {
    for (const text of view.problems) {
      const item = document.createElement("li");
      item.textContent = text;
      problems.appendChild(item);
    }
    toggle("coverage-problems", true);
  } else {
    toggle("coverage-problems", false);
  }

  const grid = el("coverage-label-grid");
  grid.replaceChildren();
  for (const row of view.labels) {
    const card = document.createElement("article");
    card.className = "coverage-label-card";
    if (row.belowMinimum) card.classList.add("coverage-label-below");
    card.setAttribute("role", "listitem");

    const title = document.createElement("h3");
    title.className = "coverage-label-name";
    title.textContent = row.title;

    const meta = document.createElement("dl");
    meta.className = "coverage-label-meta";

    const sessionsDt = document.createElement("dt");
    sessionsDt.textContent = "sessões (mín.)";
    const sessionsDd = document.createElement("dd");
    sessionsDd.textContent = row.sessionsText;

    const secondsDt = document.createElement("dt");
    secondsDt.textContent = "segundos no CSV";
    const secondsDd = document.createElement("dd");
    secondsDd.textContent = row.secondsText;

    meta.append(sessionsDt, sessionsDd, secondsDt, secondsDd);
    card.append(title, meta);
    grid.appendChild(card);
  }

  const droppedRows = view.dropped;
  if (droppedRows.length) {
    el("coverage-dropped-detail").textContent = droppedRows
      .map((row) => `${row.title}: ${row.count}`)
      .join(" · ");
    toggle("coverage-dropped", true);
  } else {
    toggle("coverage-dropped", false);
  }
}

document.addEventListener("keydown", (event) => {
  if (typingTarget(event.target)) return;
  const digit = event.key.length === 1 && event.key >= "1" && event.key <= "9" ? event.key : null;
  if (digit !== null) {
    const labelIndex = Recorder.labelIndexForDigit(digit, recorderLabels, recorderModelSchema());
    if (labelIndex >= 0 && labelIndex < recorderLabels.length) {
      const view = recorder.snapshot();
      if (!view.startEnabled) {
        flashToast("Gravação indisponível agora");
        return;
      }
      event.preventDefault();
      startRecorderLabel(recorderLabels[labelIndex]);
      return;
    }
  }
  if (event.key === " " || event.code === "Space") {
    const view = recorder.snapshot();
    if (!view.stopEnabled) return;
    event.preventDefault();
    stopRecorder();
    return;
  }
  if (event.key === "Escape") {
    const view = recorder.snapshot();
    if (!view.cancelCountdownEnabled) return;
    event.preventDefault();
    cancelRecorderCountdown();
  }
});

connect();
fetchDatasetStats();
