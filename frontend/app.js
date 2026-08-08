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

const SIDE_LABEL = { left: "esquerdo", right: "direito" };

const GESTURE_FEED_LIMIT = 6;
let toastTimer = null;

const COLORS = {
  accent: "#78ff78",
  weak: "#ffa050",
  bone: "#5ac8ff",
  trail: "#dc82ff",
  outline: "#0b0f14",
};

const state = {
  config: null,
  pending: null,
  drawing: false,
  socket: null,
  retries: 0,
  streaming: false,
  mirror: true,
  trail: true,
  labels: false,
  video: true,
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
    return;
  }
  if (message.type === "error") {
    setStatus("error", "parado");
    show("banner", message.message);
    return;
  }
  if (message.type === "frame") {
    if (!state.streaming) {
      state.streaming = true;
      setStatus("live", "ao vivo");
    }
    // Gestures are edge events tied to this exact message — handled here,
    // not inside the render loop, so a slow decode never delays or drops one.
    if (message.gestures.length) handleGestures(message.gestures);
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
  drawBones(message.landmarks);
  drawJoints(message.landmarks);
  drawVelocities(message);

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
  el("fps").textContent = fmt(message.fps, 1);
  el("inference").textContent = fmt(message.inference_ms, 1);
  el("confidence").textContent = fmt(message.confidence, 2);
  el("frame-index").textContent = message.frame_index;

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

function setStatus(kind, text) {
  el("dot").className = `dot ${kind}`;
  el("status-text").textContent = text;
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

const TOGGLES = { mirror: "m", trail: "t", labels: "l", video: "v" };

for (const [name, key] of Object.entries(TOGGLES)) {
  const box = el(`toggle-${name}`);
  state[name] = box.checked;
  box.addEventListener("change", () => {
    state[name] = box.checked;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== key || event.target.tagName === "INPUT") return;
    box.checked = !box.checked;
    box.dispatchEvent(new Event("change"));
  });
}

connect();
