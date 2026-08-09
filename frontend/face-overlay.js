/* Sprint 11 — face overlay: pure line-building + colour logic.
 *
 * Same reasoning as face-history.js: kept out of app.js and free of
 * `window`/`canvas` calls so it runs under `node --test` with no DOM and no
 * new dependency. `app.js` calls `buildFaceOverlayLines` for the list of
 * segments to stroke, then does the actual `ctx.stroke()` — mirroring,
 * canvas sizing and drawing stay there, exactly as `layoutPoint` in
 * face-history.js hands back pixel coordinates without ever touching a
 * `CanvasRenderingContext2D`.
 *
 * Two rules enforced here rather than at draw time:
 *
 *   1. No detected face means no lines, ever. `buildFaceOverlayLines` returns
 *      `[]` whenever `face.available`/`face.detected` is false — there is no
 *      per-point reliability score to fall back on (the Face Landmarker
 *      exposes none; see backend/vision/face_landmarks.py), so the one
 *      honest gate is frame-level.
 *   2. A chain with either endpoint missing is skipped, not guessed. Mirrors
 *      `drawBones`' rule for the arm skeleton.
 */

(function (global) {
  "use strict";

  function hexToRgb(hex) {
    const v = hex.replace("#", "");
    return {
      r: parseInt(v.slice(0, 2), 16),
      g: parseInt(v.slice(2, 4), 16),
      b: parseInt(v.slice(4, 6), 16),
    };
  }

  /** Linear blend between two hex colours; `t` is clamped into [0, 1]. */
  function mixColor(fromHex, toHex, t) {
    const k = Math.max(0, Math.min(1, t));
    const a = hexToRgb(fromHex);
    const b = hexToRgb(toHex);
    const r = Math.round(a.r + (b.r - a.r) * k);
    const g = Math.round(a.g + (b.g - a.g) * k);
    const bl = Math.round(a.b + (b.b - a.b) * k);
    return `rgb(${r}, ${g}, ${bl})`;
  }

  /**
   * Picks a region's stroke colour. `lips`/`left_eye`/`right_eye` warm up
   * towards `colors.highlight` as their own already-on-the-wire observable
   * score rises (`face.smile`/`mouth_open`, `face.eye_blink_left/right`) —
   * every other region stays the fixed base colour. Never derives or
   * displays an emotion label, only blends a colour by a named number that
   * was already part of the payload.
   */
  function faceGroupColor(name, face, colors) {
    if (name === "lips") {
      return mixColor(colors.line, colors.highlight, Math.max(face.smile || 0, face.mouth_open || 0));
    }
    if (name === "left_eye") return mixColor(colors.line, colors.highlight, face.eye_blink_left || 0);
    if (name === "right_eye") return mixColor(colors.line, colors.highlight, face.eye_blink_right || 0);
    return colors.line;
  }

  /**
   * Exponential smoothing for the coordinates drawn on the canvas. Only
   * landmarks present in the current frame survive, so smoothing can never
   * invent a point or keep a stale face visible after detection is lost.
   */
  function smoothLandmarks(previous, current, amount = 0.42) {
    if (!current) return {};
    const alpha = Math.max(0, Math.min(1, amount));
    const result = {};
    for (const [index, point] of Object.entries(current)) {
      if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;
      const before = previous && previous[index];
      if (!before || !Number.isFinite(before.x) || !Number.isFinite(before.y)) {
        result[index] = { ...point };
        continue;
      }
      result[index] = {
        ...point,
        x: before.x + (point.x - before.x) * alpha,
        y: before.y + (point.y - before.y) * alpha,
      };
    }
    return result;
  }

  /**
   * `face` is `message.face` (backend/web/payload.py:face_payload).
   * `chainGroups` is `hello.face_chains` (group name -> `[[a, b], ...]`).
   * `colors` is `{ line, highlight }` hex strings.
   *
   * Returns `[{ group, from: {x, y}, to: {x, y}, color }, ...]` in raw,
   * unmirrored normalized coordinates — the caller mirrors/scales to canvas
   * pixels, same division of labour `trail_payload`/`drawTrail` already have.
   */
  function buildFaceOverlayLines(face, chainGroups, colors) {
    const lines = [];
    if (!face || !face.available || !face.detected) return lines;
    const landmarks = face.landmarks;
    if (!landmarks || !chainGroups) return lines;
    for (const [name, pairs] of Object.entries(chainGroups)) {
      const color = faceGroupColor(name, face, colors);
      for (const [a, b] of pairs) {
        const pa = landmarks[a];
        const pb = landmarks[b];
        if (!pa || !pb) continue; // either point missing this frame -> skip, don't guess
        lines.push({ group: name, from: pa, to: pb, color });
      }
    }
    return lines;
  }

  const FaceOverlay = {
    hexToRgb,
    mixColor,
    faceGroupColor,
    smoothLandmarks,
    buildFaceOverlayLines,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = FaceOverlay;
  } else {
    global.FaceOverlay = FaceOverlay;
  }
})(typeof window !== "undefined" ? window : globalThis);
