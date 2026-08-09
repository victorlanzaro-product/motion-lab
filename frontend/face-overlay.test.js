/* Tests for face-overlay.js — run with Node's built-in test runner, no new
 * dependency (same convention as face-history.test.js):
 *
 *   node --test frontend/face-overlay.test.js
 *
 * The last block below checks index.html's markup as plain text — there is
 * no jsdom in this project (see face-history.test.js's own comment on why),
 * so "DOM" verification here means asserting the actual served HTML has the
 * elements/attributes app.js reads by id, and that the new toggle is a
 * distinct control from the existing one rather than a string check on
 * behaviour a browser would otherwise have to run to prove.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {
  hexToRgb,
  mixColor,
  faceGroupColor,
  smoothLandmarks,
  buildFaceOverlayLines,
} = require("./face-overlay.js");

// --- hexToRgb / mixColor ---------------------------------------------------

test("hexToRgb parses a hex triplet", () => {
  assert.deepEqual(hexToRgb("#7ea8ff"), { r: 0x7e, g: 0xa8, b: 0xff });
});

test("mixColor at t=0 is exactly the from-colour", () => {
  assert.equal(mixColor("#000000", "#ff6f91", 0), "rgb(0, 0, 0)");
});

test("mixColor at t=1 is exactly the to-colour", () => {
  assert.equal(mixColor("#000000", "#ff6f91", 1), "rgb(255, 111, 145)");
});

test("mixColor clamps t outside [0, 1]", () => {
  assert.equal(mixColor("#000000", "#ff6f91", -5), mixColor("#000000", "#ff6f91", 0));
  assert.equal(mixColor("#000000", "#ff6f91", 5), mixColor("#000000", "#ff6f91", 1));
});

// --- faceGroupColor ---------------------------------------------------------

const COLORS = { line: "#7ea8ff", highlight: "#ff6f91" };

test("faceGroupColor holds the base colour for non-highlighted regions", () => {
  assert.equal(faceGroupColor("face_oval", { smile: 1, mouth_open: 1 }, COLORS), COLORS.line);
  assert.equal(faceGroupColor("nose", {}, COLORS), COLORS.line);
});

test("faceGroupColor warms lips towards the highlight as smile/mouth_open rises", () => {
  const closed = faceGroupColor("lips", { smile: 0, mouth_open: 0 }, COLORS);
  const open = faceGroupColor("lips", { smile: 0.1, mouth_open: 0.9 }, COLORS);
  assert.equal(closed, mixColor(COLORS.line, COLORS.highlight, 0));
  assert.equal(open, mixColor(COLORS.line, COLORS.highlight, 0.9)); // max(smile, mouth_open)
});

test("faceGroupColor tracks each eye's own blink score independently", () => {
  const face = { eye_blink_left: 0.9, eye_blink_right: 0.1 };
  assert.equal(faceGroupColor("left_eye", face, COLORS), mixColor(COLORS.line, COLORS.highlight, 0.9));
  assert.equal(faceGroupColor("right_eye", face, COLORS), mixColor(COLORS.line, COLORS.highlight, 0.1));
});

// --- smoothLandmarks -------------------------------------------------------

test("smoothLandmarks reduces jitter without mutating either frame", () => {
  const previous = { 1: { x: 0.2, y: 0.4 } };
  const current = { 1: { x: 0.6, y: 0.8 } };
  const smoothed = smoothLandmarks(previous, current, 0.5);
  assert.equal(smoothed[1].x, 0.4);
  assert.ok(Math.abs(smoothed[1].y - 0.6) < 1e-12);
  assert.deepEqual(previous, { 1: { x: 0.2, y: 0.4 } });
  assert.deepEqual(current, { 1: { x: 0.6, y: 0.8 } });
});

test("smoothLandmarks never keeps a landmark missing from the current frame", () => {
  const previous = { 1: landmark(0.1, 0.1), 2: landmark(0.2, 0.2) };
  const current = { 2: landmark(0.4, 0.4) };
  assert.deepEqual(Object.keys(smoothLandmarks(previous, current)), ["2"]);
});

// --- buildFaceOverlayLines ---------------------------------------------------

const CHAINS = {
  face_oval: [[10, 338]],
  lips: [[61, 291]],
  left_eye: [[263, 249]],
};

function landmark(x, y) {
  return { x, y };
}

test("no face in frame -> no lines, ever", () => {
  assert.deepEqual(buildFaceOverlayLines(null, CHAINS, COLORS), []);
  assert.deepEqual(buildFaceOverlayLines({ available: false, detected: false }, CHAINS, COLORS), []);
  assert.deepEqual(
    buildFaceOverlayLines({ available: true, detected: false, landmarks: {} }, CHAINS, COLORS),
    [],
  );
});

test("a detected face with a full set of landmarks draws one line per chain", () => {
  const face = {
    available: true,
    detected: true,
    smile: 0.2,
    mouth_open: 0.1,
    eye_blink_left: 0.0,
    landmarks: {
      10: landmark(0.1, 0.1),
      338: landmark(0.2, 0.1),
      61: landmark(0.3, 0.4),
      291: landmark(0.35, 0.4),
      263: landmark(0.5, 0.2),
      249: landmark(0.52, 0.2),
    },
  };
  const lines = buildFaceOverlayLines(face, CHAINS, COLORS);
  assert.equal(lines.length, 3);
  assert.deepEqual(new Set(lines.map((l) => l.group)), new Set(["face_oval", "lips", "left_eye"]));
  const oval = lines.find((l) => l.group === "face_oval");
  assert.deepEqual(oval.from, landmark(0.1, 0.1));
  assert.deepEqual(oval.to, landmark(0.2, 0.1));
});

test("a chain is skipped, not guessed, when either endpoint landmark is missing", () => {
  const face = {
    available: true,
    detected: true,
    landmarks: {
      10: landmark(0.1, 0.1),
      // 338 missing this frame -- e.g. just outside the overlay subset sent
      61: landmark(0.3, 0.4),
      291: landmark(0.35, 0.4),
      263: landmark(0.5, 0.2),
      249: landmark(0.52, 0.2),
    },
  };
  const lines = buildFaceOverlayLines(face, CHAINS, COLORS);
  assert.equal(lines.length, 2); // face_oval dropped, lips + left_eye survive
  assert.ok(!lines.some((l) => l.group === "face_oval"));
});

// --- served markup (index.html) --------------------------------------------

const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");

test("index.html serves face-overlay.js before app.js", () => {
  const overlayIndex = html.indexOf('src="/static/face-overlay.js"');
  const appIndex = html.indexOf('src="/static/app.js"');
  assert.ok(overlayIndex > -1, "face-overlay.js is not linked");
  assert.ok(overlayIndex < appIndex, "face-overlay.js must load before app.js");
});

test("the face overlay toggle is a distinct control from the face panel toggle", () => {
  assert.match(html, /id="toggle-face-overlay"/);
  assert.match(html, /id="toggle-face"[^-]/); // the panel toggle, not swallowed by "toggle-face-overlay"
  const overlayLabel = html.match(/<label>[^<]*<input[^>]*id="toggle-face-overlay"[\s\S]*?<\/label>/);
  const panelLabel = html.match(/<label>[^<]*<input[^>]*id="toggle-face"\s[\s\S]*?<\/label>/);
  assert.ok(overlayLabel, "overlay toggle label not found");
  assert.ok(panelLabel, "panel toggle label not found");
  // Different keyboard shortcuts -- "o" for overlay, "f" for the panel --
  // and neither label's wording is a substring of the other's, so a user
  // scanning the toggles list cannot mistake one control for the other.
  assert.match(overlayLabel[0], /<kbd>o<\/kbd>/);
  assert.match(panelLabel[0], /<kbd>f<\/kbd>/);
  assert.notEqual(overlayLabel[0], panelLabel[0]);
});
