"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {
  snapshot,
  formatMs,
  barFraction,
  droppedRateAlert,
  DROPPED_RATE_ALERT,
  buildStageRows,
} = require("./health-panel.js");

test("formatMs renders em dash for absent values, never zero", () => {
  assert.equal(formatMs(null), "—");
  assert.equal(formatMs(undefined), "—");
  assert.equal(formatMs(NaN), "—");
  assert.equal(formatMs(12.34), "12.3");
});

test("buildStageRows keeps null stages as em dash without a bar", () => {
  const rows = buildStageRows({
    capture_ms: 1.2,
    pose_ms: 18.0,
    features_ms: 0.4,
    ml_ms: null,
    face_ms: null,
    total_ms: 20.1,
  });
  const ml = rows.find((r) => r.key === "ml_ms");
  const face = rows.find((r) => r.key === "face_ms");
  assert.equal(ml.display, "—");
  assert.equal(ml.barFraction, 0);
  assert.equal(face.display, "—");
  assert.equal(face.barFraction, 0);
  assert.ok(rows.find((r) => r.key === "pose_ms").barFraction > 0.5);
});

test("droppedRateAlert flags rates strictly above the threshold", () => {
  assert.equal(DROPPED_RATE_ALERT, 0.1);
  assert.equal(droppedRateAlert(0.1), false);
  assert.equal(droppedRateAlert(0.1001), true);
  assert.equal(droppedRateAlert(null), false);
});

test("snapshot reads health and timings without inventing zeros", () => {
  const view = snapshot({
    health: { fps: 29.8, dropped_frames: 3, dropped_rate: 0.01 },
    timings: {
      capture_ms: 2.1,
      pose_ms: 18.4,
      features_ms: 0.3,
      ml_ms: 1.2,
      face_ms: null,
      total_ms: 23.0,
    },
  });
  assert.equal(view.fpsDisplay, "29.8");
  assert.equal(view.dropped_frames, 3);
  assert.equal(view.droppedRateDisplay, "1.00%");
  assert.equal(view.droppedRateAlert, false);
  assert.equal(view.totalDisplay, "23.0");
  assert.equal(view.stages.length, 5);
});

test("snapshot tolerates missing health and timings", () => {
  const view = snapshot({});
  assert.equal(view.fpsDisplay, "—");
  assert.equal(view.droppedRateDisplay, "—");
  assert.equal(view.totalDisplay, "—");
  assert.ok(view.stages.every((row) => row.display === "—"));
});