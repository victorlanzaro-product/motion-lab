"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { createRecorder, DEFAULT_COUNTDOWN_SECONDS } = require("./recorder.js");

const CONTRACT_LABELS = ["arm_raised", "arms_crossed", "arms_open", "idle"];

const OWNER_RECORDING = {
  type: "record_status",
  recording: true,
  label: "arm_raised",
  session_id: "web-20261119-153012-a1b2c3",
  written: 42,
  skipped: 3,
  owner: true,
  error: null,
};

const REMOTE_RECORDING = {
  type: "record_status",
  recording: true,
  label: "arm_raised",
  session_id: "web-20261119-153012-a1b2c3",
  written: 42,
  skipped: 3,
  owner: false,
  error: null,
};

const STOP_SUMMARY = {
  type: "record_status",
  recording: false,
  label: "arm_raised",
  session_id: "web-20261119-153012-a1b2c3",
  written: 42,
  skipped: 3,
  owner: true,
  error: null,
};

const IDLE_REJECTION = {
  type: "record_status",
  recording: false,
  label: null,
  session_id: null,
  written: 0,
  skipped: 0,
  owner: false,
  error: "gravacao ja em andamento",
};

const LIVE_STATE_REJECTION = {
  type: "record_status",
  recording: true,
  label: "arm_raised",
  session_id: "web-20261119-153012-a1b2c3",
  written: 42,
  skipped: 3,
  owner: false,
  error: "gravacao ja em andamento",
};

test("owner recording reflects live counters from record_status", () => {
  const recorder = createRecorder({ countdownSeconds: 0 });
  recorder.setLabels(CONTRACT_LABELS);
  const t0 = 1_000_000;
  recorder.beginStart("arm_raised", t0);
  const outbound = recorder.tick(t0);
  assert.deepEqual(outbound, [{ type: "record_start", label: "arm_raised" }]);
  recorder.applyStatus(OWNER_RECORDING);
  const view = recorder.snapshot();
  assert.equal(view.showRecordingBanner, true);
  assert.equal(view.status.written, 42);
  assert.equal(view.status.skipped, 3);
  assert.equal(view.stopEnabled, true);
  assert.equal(view.startEnabled, false);
});

test("non-owner tab sees remote recording and cannot start or stop", () => {
  const recorder = createRecorder();
  recorder.setLabels(CONTRACT_LABELS);
  recorder.applyStatus(REMOTE_RECORDING);
  const view = recorder.snapshot();
  assert.equal(view.showRecordingBanner, true);
  assert.equal(view.isRemoteRecording, true);
  assert.equal(view.startEnabled, false);
  assert.equal(view.stopEnabled, false);
  assert.equal(recorder.beginStart("idle").ok, false);
  assert.equal(recorder.beginStop().ok, false);
});

test("stop summary keeps label and session_id on recording=false", () => {
  const recorder = createRecorder();
  recorder.setLabels(CONTRACT_LABELS);
  recorder.applyStatus(OWNER_RECORDING);
  recorder.applyStatus(STOP_SUMMARY);
  const view = recorder.snapshot();
  assert.equal(view.showRecordingBanner, false);
  assert.deepEqual(view.lastSummary, {
    label: "arm_raised",
    session_id: "web-20261119-153012-a1b2c3",
    written: 42,
    skipped: 3,
    error: null,
  });
});

test("error from record_status is exposed in snapshot", () => {
  const recorder = createRecorder({ countdownSeconds: 0 });
  recorder.setLabels(CONTRACT_LABELS);
  recorder.beginStart("idle", 0);
  recorder.tick(0);
  recorder.applyStatus(IDLE_REJECTION);
  const view = recorder.snapshot();
  assert.equal(view.status.error, "gravacao ja em andamento");
  assert.equal(view.status.owner, false);
  assert.equal(view.showRecordingBanner, false);
});

test("cancel countdown sends nothing on the wire", () => {
  const recorder = createRecorder();
  recorder.setLabels(CONTRACT_LABELS);
  const t0 = 5_000;
  recorder.beginStart("arms_open", t0);
  assert.equal(recorder.snapshot(t0).countdownRemaining, DEFAULT_COUNTDOWN_SECONDS);
  assert.equal(recorder.cancelCountdown(), true);
  assert.deepEqual(recorder.drainOutbound(), []);
  assert.equal(recorder.tick(t0 + DEFAULT_COUNTDOWN_SECONDS * 1000).length, 0);
  assert.equal(recorder.snapshot().phase, "idle");
});

test("hello then record_status on reconnect keeps GRAVANDO without a gap", () => {
  const recorder = createRecorder();
  recorder.setLabels(CONTRACT_LABELS);
  recorder.applyStatus(REMOTE_RECORDING);
  recorder.onWebSocketReconnect();
  const afterHello = recorder.snapshot();
  assert.equal(afterHello.showRecordingBanner, true);
  assert.equal(afterHello.isRemoteRecording, true);
  recorder.applyStatus(REMOTE_RECORDING);
  const confirmed = recorder.snapshot();
  assert.equal(confirmed.showRecordingBanner, true);
  assert.equal(confirmed.status.written, 42);
  assert.equal(confirmed.status.label, "arm_raised");
});

test("idle rejection does not clear another tab's live recording", () => {
  const recorder = createRecorder();
  recorder.setLabels(CONTRACT_LABELS);
  recorder.applyStatus(REMOTE_RECORDING);
  const result = recorder.applyStatus(IDLE_REJECTION);
  assert.equal(result.ignored, true);
  const view = recorder.snapshot();
  assert.equal(view.showRecordingBanner, true);
  assert.equal(view.isRemoteRecording, true);
  assert.equal(view.status.label, "arm_raised");
  assert.equal(view.status.session_id, "web-20261119-153012-a1b2c3");
  assert.equal(view.status.error, null);
});

test("idle rejection applies on an idle requester tab with owner false", () => {
  const recorder = createRecorder({ countdownSeconds: 0 });
  recorder.setLabels(CONTRACT_LABELS);
  recorder.beginStart("idle", 0);
  recorder.tick(0);
  recorder.applyStatus(IDLE_REJECTION);
  const view = recorder.snapshot();
  assert.equal(view.status.error, "gravacao ja em andamento");
  assert.equal(view.status.owner, false);
  assert.equal(view.showRecordingBanner, false);
});

test("rejection mirrors live server state when a session is already recording", () => {
  const recorder = createRecorder({ countdownSeconds: 0 });
  recorder.setLabels(CONTRACT_LABELS);
  recorder.beginStart("idle", 0);
  recorder.tick(0);
  recorder.applyStatus(LIVE_STATE_REJECTION);
  const view = recorder.snapshot();
  assert.equal(view.status.error, "gravacao ja em andamento");
  assert.equal(view.showRecordingBanner, true);
  assert.equal(view.isRemoteRecording, true);
  assert.equal(view.status.written, 42);
});

test("remote recording aborts an in-flight countdown without outbound", () => {
  const recorder = createRecorder();
  recorder.setLabels(CONTRACT_LABELS);
  const t0 = 10_000;
  recorder.beginStart("idle", t0);
  recorder.applyStatus(REMOTE_RECORDING);
  assert.equal(recorder.snapshot().phase, "idle");
  assert.deepEqual(recorder.tick(t0 + DEFAULT_COUNTDOWN_SECONDS * 1000), []);
});