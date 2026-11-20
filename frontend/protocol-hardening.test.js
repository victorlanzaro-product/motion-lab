const test = require("node:test");
const assert = require("node:assert/strict");
const {
  isFatalPipelineError,
  isRecorderRejection,
  clientRecorderReason,
  displayDegradationMessage,
  shouldToastRecorderRejection,
  setRecordingBannerHidden,
  createDegradationTracker,
} = require("./protocol-hardening.js");

test("fatal defaults true; fatal=false is degradable", () => {
  assert.equal(isFatalPipelineError({ type: "error", message: "câmera negada" }), true);
  assert.equal(
    isFatalPipelineError({ type: "error", message: "modelo facial indisponível", fatal: false }),
    false,
  );
  assert.equal(isFatalPipelineError({ type: "status", state: "recovering" }), false);
});

test("record_status rejections expose Portuguese server text", () => {
  const idle = {
    type: "record_status",
    recording: false,
    owner: false,
    error: "Informe um rótulo de gravação válido.",
  };
  assert.equal(isRecorderRejection(idle), true);
  assert.equal(isRecorderRejection({ type: "record_status", error: null }), false);
});

test("client-side recorder reasons are Portuguese", () => {
  assert.equal(
    clientRecorderReason("invalid_label"),
    "Escolha um rótulo de gravação oferecido pelo servidor.",
  );
  assert.equal(clientRecorderReason("busy"), "Aguarde: gravação ou contagem em andamento em outra aba.");
  assert.equal(clientRecorderReason("other"), null);
});

test("displayDegradationMessage passes through server copy", () => {
  const msg = displayDegradationMessage({
    type: "error",
    fatal: false,
    message: "  Sinal de ML temporariamente indisponível.  ",
  });
  assert.equal(msg, "Sinal de ML temporariamente indisponível.");
});

test("fatal absent or omitted is fatal; only explicit false degrades", () => {
  const absent = { type: "error", message: "falha grave" };
  assert.equal(isFatalPipelineError(absent), true);
  assert.equal(isFatalPipelineError({ type: "error", fatal: true, message: "x" }), true);
  assert.equal(isFatalPipelineError({ type: "error", fatal: false, message: "x" }), false);
});

test("recorder rejection toast only when error panel is hidden", () => {
  assert.equal(shouldToastRecorderRejection(false), true);
  assert.equal(shouldToastRecorderRejection(true), false);
});

test("recording banner uses hidden attribute and class together", () => {
  const toggles = [];
  const el = {
    hidden: true,
    classList: { toggle(cls, on) { toggles.push([cls, on]); } },
  };
  setRecordingBannerHidden(el, true);
  assert.equal(el.hidden, false);
  assert.deepEqual(toggles, [["hidden", false]]);
  setRecordingBannerHidden(el, false);
  assert.equal(el.hidden, true);
});

test("degradation tracker expires after ttl without a new error", () => {
  let clock = 1000;
  const tracker = createDegradationTracker({ ttlMs: 30_000, now: () => clock });
  const msg = { type: "error", fatal: false, message: "aviso temporário" };
  tracker.absorbError(msg);
  assert.equal(tracker.currentNote(), "aviso temporário");
  clock += 29_999;
  assert.equal(tracker.currentNote(), "aviso temporário");
  clock += 2;
  assert.equal(tracker.currentNote(), null);
});

test("degradation tracker refreshes ttl when a new degradable error arrives", () => {
  let clock = 0;
  const tracker = createDegradationTracker({ ttlMs: 30_000, now: () => clock });
  tracker.absorbError({ type: "error", fatal: false, message: "primeiro" });
  clock += 25_000;
  tracker.absorbError({ type: "error", fatal: false, message: "segundo" });
  clock += 25_000;
  assert.equal(tracker.currentNote(), "segundo");
  clock += 6_000;
  assert.equal(tracker.currentNote(), null);
});

test("degradation tracker clear on fatal resets state", () => {
  const tracker = createDegradationTracker();
  tracker.absorbError({ type: "error", fatal: false, message: "parcial" });
  tracker.absorbError({ type: "error", message: "pipeline parou" });
  assert.equal(tracker.currentNote(), null);
});
