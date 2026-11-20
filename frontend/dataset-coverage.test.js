"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { buildCoverageView, DEFAULT_LABELS, coverageLabelOrder } = require("./dataset-coverage.js");

const CONTRACT_SAMPLE = {
  path: "data/training/dataset.csv",
  exists: true,
  total_rows: 1840,
  usable_rows: 1795,
  dropped: { missing_feature: 40, unknown_label: 5 },
  labels: {
    arm_raised: { rows: 512, sessions: 3, seconds: 41.2 },
    arms_crossed: { rows: 470, sessions: 3, seconds: 38.0 },
    arms_open: { rows: 401, sessions: 2, seconds: 33.5 },
    idle: { rows: 412, sessions: 1, seconds: 34.9 },
  },
  sessions: [
    {
      session_id: "web-20261119-153012-a1b2c3",
      label: "arm_raised",
      rows: 180,
      seconds: 14.1,
    },
  ],
  min_sessions_per_label: 2,
  ready: false,
  problems: ["idle: 1 sessão (mínimo 2)"],
};

const MISSING_DATASET = {
  path: "data/training/dataset.csv",
  exists: false,
  total_rows: 0,
  usable_rows: 0,
  dropped: {},
  labels: {
    arm_raised: { rows: 0, sessions: 0, seconds: 0.0 },
    arms_crossed: { rows: 0, sessions: 0, seconds: 0.0 },
    arms_open: { rows: 0, sessions: 0, seconds: 0.0 },
    idle: { rows: 0, sessions: 0, seconds: 0.0 },
  },
  sessions: [],
  min_sessions_per_label: 2,
  ready: false,
  problems: ["dataset ausente"],
};

test("contract sample maps labels in DEFAULT_LABELS order", () => {
  const view = buildCoverageView(CONTRACT_SAMPLE);
  assert.equal(view.exists, true);
  assert.equal(view.ready, false);
  assert.equal(view.minSessionsPerLabel, 2);
  assert.deepEqual(
    view.labels.map((row) => row.key),
    [...DEFAULT_LABELS],
  );
  assert.equal(view.labels[3].sessions, 1);
  assert.equal(view.labels[3].belowMinimum, true);
  assert.equal(view.labels[0].belowMinimum, false);
  assert.deepEqual(view.problems, ["idle: 1 sessão (mínimo 2)"]);
  assert.equal(view.statusKind, "incomplete");
  assert.equal(view.labels[0].secondsText, "41.2");
  assert.equal(view.dropped.length, 2);
});

test("exists:false is friendly and never marks labels below minimum", () => {
  const view = buildCoverageView(MISSING_DATASET);
  assert.equal(view.showMissingDataset, true);
  assert.equal(view.exists, false);
  assert.equal(view.ready, false);
  assert.equal(view.totalRows, 0);
  assert.deepEqual(view.problems, ["dataset ausente"]);
  assert.equal(view.statusKind, "missing");
  for (const row of view.labels) {
    assert.equal(row.sessions, 0);
    assert.equal(row.isEmpty, true);
    assert.equal(row.belowMinimum, false);
    assert.equal(row.secondsText, "0.0");
  }
});

test("missing label bucket renders as zero sessions", () => {
  const partial = {
    ...CONTRACT_SAMPLE,
    ready: false,
    labels: {
      arm_raised: { rows: 10, sessions: 2, seconds: 1.0 },
      arms_crossed: { rows: 0, sessions: 0, seconds: 0.0 },
      arms_open: { rows: 5, sessions: 2, seconds: 2.5 },
      idle: { rows: 0, sessions: 0, seconds: 0.0 },
    },
    problems: ["arms_crossed: 0 sessões (mínimo 2)", "idle: 0 sessões (mínimo 2)"],
  };
  const view = buildCoverageView(partial);
  const armsCrossed = view.labels.find((row) => row.key === "arms_crossed");
  assert.equal(armsCrossed.sessions, 0);
  assert.equal(armsCrossed.belowMinimum, true);
  assert.equal(armsCrossed.isEmpty, true);
  assert.equal(armsCrossed.sessionsText, "0 / 2");
});

test("unknown_label in dropped keeps ready false and surfaces the maestro problem", () => {
  const stats = {
    path: "data/training/dataset.csv",
    exists: true,
    total_rows: 505,
    usable_rows: 500,
    dropped: { unknown_label: 5, missing_feature: 0, missing_label: 0, non_finite: 0 },
    labels: {
      arm_raised: { rows: 125, sessions: 2, seconds: 10.0 },
      arms_crossed: { rows: 125, sessions: 2, seconds: 10.0 },
      arms_open: { rows: 125, sessions: 2, seconds: 10.0 },
      idle: { rows: 125, sessions: 2, seconds: 10.0 },
    },
    sessions: [],
    min_sessions_per_label: 2,
    ready: false,
    problems: ["5 linhas com rótulo desconhecido (treino vai recusar)"],
  };
  const view = buildCoverageView(stats);
  assert.equal(view.ready, false);
  assert.equal(view.statusKind, "incomplete");
  assert.deepEqual(
    view.problems,
    ["5 linhas com rótulo desconhecido (treino vai recusar)"],
  );
  const unknown = view.dropped.find((row) => row.reason === "unknown_label");
  assert.equal(unknown.reason, "unknown_label");
  assert.equal(unknown.count, 5);
  assert.equal(unknown.title, "rótulo desconhecido");
  assert.ok(view.labels.every((row) => row.rows > 0));
});

const V2_STATS_EMPTY_WAVE = {
  path: "data/training/dataset_v2.csv",
  exists: true,
  total_rows: 0,
  usable_rows: 0,
  dropped: {},
  labels: {
    arm_raised: { rows: 0, sessions: 0, seconds: 0.0 },
    arms_crossed: { rows: 0, sessions: 0, seconds: 0.0 },
    arms_open: { rows: 0, sessions: 0, seconds: 0.0 },
    idle: { rows: 0, sessions: 0, seconds: 0.0 },
    wave: { rows: 0, sessions: 0, seconds: 0.0 },
  },
  sessions: [],
  min_sessions_per_label: 2,
  ready: false,
  problems: ["wave: 0 sessões (mínimo 2)"],
};

test("v2 stats include wave label bucket from the endpoint", () => {
  const view = buildCoverageView(V2_STATS_EMPTY_WAVE);
  assert.deepEqual(
    view.labels.map((row) => row.key),
    [...DEFAULT_LABELS, "wave"],
  );
  const wave = view.labels.find((row) => row.key === "wave");
  assert.equal(wave.title, "aceno");
  assert.equal(wave.sessions, 0);
  assert.equal(wave.isEmpty, true);
  assert.deepEqual(coverageLabelOrder(V2_STATS_EMPTY_WAVE), [...DEFAULT_LABELS, "wave"]);
  assert.deepEqual(
    coverageLabelOrder(V2_STATS_EMPTY_WAVE, {
      coverageLabels: V2_STATS_EMPTY_WAVE.labels && Object.keys(V2_STATS_EMPTY_WAVE.labels),
    }),
    Object.keys(V2_STATS_EMPTY_WAVE.labels),
  );
});

test("ready dataset uses ready badge", () => {
  const readyStats = {
    ...CONTRACT_SAMPLE,
    labels: {
      arm_raised: { rows: 100, sessions: 3, seconds: 10 },
      arms_crossed: { rows: 100, sessions: 3, seconds: 10 },
      arms_open: { rows: 100, sessions: 2, seconds: 10 },
      idle: { rows: 100, sessions: 2, seconds: 10 },
    },
    ready: true,
    problems: [],
  };
  const view = buildCoverageView(readyStats);
  assert.equal(view.ready, true);
  assert.equal(view.statusKind, "ready");
  assert.equal(view.problems.length, 0);
  assert.ok(view.labels.every((row) => !row.belowMinimum));
});