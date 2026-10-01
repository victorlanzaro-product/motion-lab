"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {
  PRESETS,
  signalPercent,
  presetFor,
  matchingPreset,
  describeEyes,
  describeHead,
} = require("./studio-controls.js");

test("the default studio preset is readable without debug clutter", () => {
  assert.deepEqual(presetFor("studio"), {
    bodyOverlay: true,
    trail: false,
    labels: false,
    faceOverlay: true,
  });
  assert.equal(PRESETS.debug.trail, true);
  assert.equal(PRESETS.debug.labels, true);
});

test("face overlay is suppressed when face processing is unavailable", () => {
  assert.equal(presetFor("face", false).faceOverlay, false);
  assert.equal(presetFor("studio", false).faceOverlay, false);
});

test("matchingPreset identifies known configurations and rejects custom ones", () => {
  assert.equal(matchingPreset(presetFor("body")), "body");
  assert.equal(
    matchingPreset({ bodyOverlay: false, trail: true, labels: true, faceOverlay: false }),
    null,
  );
});

test("signalPercent clamps finite blendshape values and handles missing values", () => {
  assert.equal(signalPercent(-1), 0);
  assert.equal(signalPercent(0.456), 46);
  assert.equal(signalPercent(9), 100);
  assert.equal(signalPercent(null), 0);
});

test("eye summary stays observable and handles missing detection", () => {
  assert.equal(describeEyes(null), "aguardando sinal");
  assert.equal(describeEyes({ detected: true, eye_blink_left: 0.1, eye_blink_right: 0.2 }), "olhos abertos");
  assert.equal(describeEyes({ detected: true, eye_blink_left: 0.8, eye_blink_right: 0.1 }), "fechamento detectado");
});

test("head summary describes geometry without claiming a mental state", () => {
  assert.equal(describeHead({ detected: true, head_yaw: 2, head_pitch: -4, head_roll: 1 }), "centralizada");
  assert.equal(describeHead({ detected: true, head_yaw: 28, head_pitch: 2, head_roll: 1 }), "virada lateralmente");
  assert.equal(describeHead({ detected: true, head_yaw: 28, head_pitch: 20, head_roll: 1 }), "fora do centro");
  for (const forbidden of ["feliz", "triste", "nervoso", "emoção", "humor"]) {
    assert.ok(!describeHead({ detected: true, head_yaw: 30, head_pitch: 0, head_roll: 0 }).includes(forbidden));
  }
});
