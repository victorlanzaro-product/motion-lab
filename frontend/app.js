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
    // Not "ao vivo" yet: opening the webcam and loading the pose model takes a
    // few seconds on the first connection, and a green dot over a black canvas
    // reads as a broken app.
    setStatus("off", "ligando câmera…");
    hide("banner");
  };
  socket.onmessage = (event) => handle(JSON.parse(event.data));
  socket.onerror = () => socket.close();
  socket.onclose = () => {
    state.streaming = false;
    setStatus("off", "reconectando…");
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
    applyFaceEnabled(message.face_enabled);
    setPresenceChip(
      "face",
      message.face_enabled ? "pending" : "off",
      message.face_enabled ? "procurando rosto" : "rosto desligado",
    );
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
  if (name === "history") renderFaceCharts(state.lastFace, true);
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

connect();
