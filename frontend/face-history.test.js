/* Tests for face-history.js — run with Node's built-in test runner, no new
 * dependency:
 *
 *   node --test frontend/face-history.test.js
 *
 * The project has no JS test harness (see tests/ — all pytest). Rather than
 * add one, the history/gap logic was pulled into a plain module with zero
 * DOM/canvas calls (face-history.js) specifically so `node:test` + `assert`,
 * both already shipped with Node, are enough to cover it.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { createStore, buildSegments, layoutPoint, clamp } = require("./face-history.js");

test("clamp bounds a value into [min, max]", () => {
  assert.equal(clamp(0.5, 0, 1), 0.5);
  assert.equal(clamp(-1, 0, 1), 0);
  assert.equal(clamp(5, 0, 1), 1);
});

test("buildSegments keeps one run when samples are contiguous", () => {
  const points = [
    { t: 0, v: 0.1 },
    { t: 0.1, v: 0.2 },
    { t: 0.2, v: 0.3 },
  ];
  const runs = buildSegments(points, 1.0);
  assert.equal(runs.length, 1);
  assert.equal(runs[0].length, 3);
});

test("buildSegments breaks on an explicit null (face/field unavailable)", () => {
  const points = [
    { t: 0, v: 0.1 },
    { t: 0.1, v: 0.2 },
    { t: 0.2, v: null }, // e.g. face not detected this frame
    { t: 0.3, v: null },
    { t: 0.4, v: 0.5 },
  ];
  const runs = buildSegments(points, 1.0);
  assert.equal(runs.length, 2);
  assert.deepEqual(runs[0].map((p) => p.t), [0, 0.1]);
  assert.deepEqual(runs[1].map((p) => p.t), [0.4]);
});

test("buildSegments breaks on a time gap even without a null in between", () => {
  const points = [
    { t: 0, v: 0.1 },
    { t: 0.1, v: 0.2 },
    { t: 5.0, v: 0.3 }, // frames simply stopped arriving for a while
  ];
  const runs = buildSegments(points, 1.0);
  assert.equal(runs.length, 2);
  assert.equal(runs[0].length, 2);
  assert.equal(runs[1].length, 1);
});

test("buildSegments keeps a single isolated sample as a one-point run", () => {
  const points = [
    { t: 0, v: 1 },
    { t: 5, v: 2 },
    { t: 10, v: 3 },
  ];
  const runs = buildSegments(points, 1.0);
  assert.equal(runs.length, 3);
  for (const run of runs) assert.equal(run.length, 1);
});

test("createStore.push records a value per series per frame", () => {
  const store = createStore(["smile", "mouth_open"], { windowSeconds: 10 });
  store.push(0, { smile: 0.1, mouth_open: 0.2 });
  store.push(1, { smile: 0.3, mouth_open: 0.4 });
  assert.deepEqual(
    store.raw("smile").map((p) => p.v),
    [0.1, 0.3],
  );
  assert.deepEqual(
    store.raw("mouth_open").map((p) => p.v),
    [0.2, 0.4],
  );
  assert.equal(store.now(), 1);
});

test("createStore.push stores null for a missing/non-numeric field", () => {
  const store = createStore(["smile"]);
  store.push(0, { smile: 0.5 });
  store.push(1, {}); // face detected, but this field absent this frame
  store.push(2, { smile: null }); // explicit unavailable
  store.push(3, { smile: "not-a-number" }); // defensive: never trust the wire blindly
  assert.deepEqual(
    store.raw("smile").map((p) => p.v),
    [0.5, null, null, null],
  );
});

test("createStore prunes points older than windowSeconds relative to the newest sample", () => {
  const store = createStore(["smile"], { windowSeconds: 10, maxGapSeconds: 100 });
  for (let t = 0; t <= 25; t += 1) store.push(t, { smile: t / 25 });
  const points = store.raw("smile");
  // Newest sample is t=25; nothing older than t=15 should survive.
  assert.ok(points.every((p) => p.t >= 15));
  assert.equal(points[points.length - 1].t, 25);
});

test("createStore enforces the maxPoints backstop even inside the time window", () => {
  const store = createStore(["smile"], { windowSeconds: 1000, maxPoints: 5 });
  for (let i = 0; i < 50; i += 1) store.push(i * 0.001, { smile: i });
  assert.equal(store.raw("smile").length, 5);
});

test("createStore resets on a backward time jump (reconnect / new pipeline run)", () => {
  const store = createStore(["smile"], { windowSeconds: 10 });
  store.push(5, { smile: 0.9 });
  store.push(6, { smile: 0.9 });
  store.push(0.1, { smile: 0.1 }); // clock went backward -> new run
  const points = store.raw("smile");
  assert.equal(points.length, 1);
  assert.equal(points[0].t, 0.1);
  assert.equal(store.now(), 0.1);
});

test("createStore.segments() reflects gaps end to end (push -> segments)", () => {
  const store = createStore(["blink_left"], { windowSeconds: 10, maxGapSeconds: 0.5 });
  store.push(0, { blink_left: 0.0 });
  store.push(0.1, { blink_left: 0.0 });
  store.push(0.2, {}); // face lost this frame
  store.push(0.3, {});
  store.push(0.4, { blink_left: 0.1 });
  const runs = store.segments("blink_left");
  assert.equal(runs.length, 2);
  assert.deepEqual(runs[0].map((p) => p.t), [0, 0.1]);
  assert.deepEqual(runs[1].map((p) => p.t), [0.4]);
});

test("layoutPoint maps the newest sample to the right edge and the window edge to the left", () => {
  const layout = { now: 10, windowSeconds: 10, min: 0, max: 1, width: 100, height: 50 };
  const right = layoutPoint({ t: 10, v: 0.5 }, layout);
  const left = layoutPoint({ t: 0, v: 0.5 }, layout);
  assert.ok(Math.abs(right.x - 100) < 1e-9);
  assert.ok(Math.abs(left.x - 0) < 1e-9);
});

test("layoutPoint maps min to the bottom and max to the top", () => {
  const layout = { now: 10, windowSeconds: 10, min: -90, max: 90, width: 100, height: 60 };
  const bottom = layoutPoint({ t: 10, v: -90 }, layout);
  const top = layoutPoint({ t: 10, v: 90 }, layout);
  assert.ok(Math.abs(bottom.y - 60) < 1e-9);
  assert.ok(Math.abs(top.y - 0) < 1e-9);
});

test("layoutPoint clamps an out-of-range value to the axis edge instead of drawing off-canvas", () => {
  const layout = { now: 10, windowSeconds: 10, min: 0, max: 1, width: 100, height: 50 };
  const above = layoutPoint({ t: 10, v: 5 }, layout);
  const below = layoutPoint({ t: 10, v: -5 }, layout);
  assert.equal(above.y, 0);
  assert.equal(below.y, 50);
});
