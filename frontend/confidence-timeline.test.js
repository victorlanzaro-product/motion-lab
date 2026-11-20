"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { createTimeline, extractMl, agreementBetween } = require("./confidence-timeline.js");

function frame(t, overrides) {
  return {
    type: "frame",
    timestamp: t,
    ml: null,
    rule_label: "idle",
    ...overrides,
  };
}

test("createTimeline evicts samples older than 30 s relative to the newest timestamp", () => {
  const tl = createTimeline({ windowSeconds: 30 });
  for (let t = 0; t <= 40; t += 1) {
    tl.push(frame(t, { rule_label: "idle", ml: { label: "idle", confidence: 0.5 } }));
  }
  const points = tl.series(40);
  assert.ok(points.every((p) => p.t >= 10));
  assert.equal(points[points.length - 1].t, 40);
  assert.equal(points.length, 31);
});

test("missing ML stores null probability, never zero", () => {
  const tl = createTimeline();
  tl.push(frame(1, { ml: null, rule_label: "idle" }));
  const points = tl.series(1);
  assert.equal(points.length, 1);
  assert.equal(points[0].mlLabel, null);
  assert.equal(points[0].mlProbability, null);
  assert.equal(points[0].agreement, null);
});

test("agreementRate counts label matches and skips wave / missing ML", () => {
  const tl = createTimeline({ windowSeconds: 100 });
  tl.push(
    frame(1, {
      rule_label: "arm_raised",
      ml: { label: "arm_raised", confidence: 0.9 },
    }),
  );
  tl.push(
    frame(2, {
      rule_label: "idle",
      ml: { label: "arm_raised", confidence: 0.8 },
    }),
  );
  tl.push(frame(3, { rule_label: "idle", ml: null }));
  tl.push(
    frame(4, {
      rule_label: "idle",
      ml: { label: "wave", confidence: 0.7 },
    }),
  );
  tl.push(
    frame(5, {
      rule_label: "idle",
      ml: { label: "idle", confidence: 0.6 },
    }),
  );
  assert.equal(tl.agreementRate(), 2 / 3);
});

test("out-of-order and duplicate timestamps are ignored", () => {
  const tl = createTimeline();
  tl.push(frame(5, { ml: { label: "idle", confidence: 0.5 }, rule_label: "idle" }));
  tl.push(frame(5, { ml: { label: "arm_raised", confidence: 0.9 }, rule_label: "arm_raised" }));
  tl.push(frame(4, { ml: { label: "idle", confidence: 0.1 }, rule_label: "idle" }));
  const points = tl.series(5);
  assert.equal(points.length, 1);
  assert.equal(points[0].mlLabel, "idle");
  assert.equal(tl.now(), 5);
});

test("series preserves gaps — no synthetic points between missing frames", () => {
  const tl = createTimeline({ windowSeconds: 30 });
  tl.push(frame(0, { ml: { label: "idle", confidence: 0.5 }, rule_label: "idle" }));
  tl.push(frame(0.1, { ml: { label: "idle", confidence: 0.5 }, rule_label: "idle" }));
  tl.push(frame(5.0, { ml: { label: "idle", confidence: 0.5 }, rule_label: "idle" }));
  const points = tl.series(5);
  assert.equal(points.length, 3);
  assert.deepEqual(
    points.map((p) => p.t),
    [0, 0.1, 5.0],
  );
  const gap = points[2].x - points[1].x;
  assert.ok(gap > 0.1, "a multi-second gap must span a wide normalized x delta");
});

test("series maps the newest sample to x=1 and the window edge to x=0", () => {
  const tl = createTimeline({ windowSeconds: 10 });
  tl.push(frame(0, { ml: { label: "idle", confidence: 0.5 }, rule_label: "idle" }));
  tl.push(frame(10, { ml: { label: "idle", confidence: 0.5 }, rule_label: "idle" }));
  const points = tl.series(10);
  assert.ok(Math.abs(points[0].x - 0) < 1e-9);
  assert.ok(Math.abs(points[1].x - 1) < 1e-9);
});

test("clear drops all samples", () => {
  const tl = createTimeline();
  tl.push(frame(1, { ml: { label: "idle", confidence: 0.5 }, rule_label: "idle" }));
  tl.clear();
  assert.deepEqual(tl.series(1), []);
  assert.equal(tl.now(), null);
});

test("extractMl never coerces a missing signal to 0", () => {
  assert.deepEqual(extractMl({ ml: null }), { mlLabel: null, mlProbability: null });
  assert.deepEqual(extractMl({}), { mlLabel: null, mlProbability: null });
});

test("agreementBetween is strict label equality", () => {
  assert.equal(agreementBetween("idle", "idle"), true);
  assert.equal(agreementBetween("arm_raised", "idle"), false);
  assert.equal(agreementBetween(null, "idle"), null);
});