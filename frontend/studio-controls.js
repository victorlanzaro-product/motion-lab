/* Live Studio presentation helpers.
 *
 * Kept DOM-free so presets and human-readable summaries are testable with
 * Node's built-in runner. These labels describe only observable geometry and
 * blendshape values; they never infer emotion or mental state.
 */

(function (global) {
  "use strict";

  const PRESETS = Object.freeze({
    clean: Object.freeze({ bodyOverlay: false, trail: false, labels: false, faceOverlay: false }),
    studio: Object.freeze({ bodyOverlay: true, trail: false, labels: false, faceOverlay: true }),
    body: Object.freeze({ bodyOverlay: true, trail: true, labels: false, faceOverlay: false }),
    face: Object.freeze({ bodyOverlay: false, trail: false, labels: false, faceOverlay: true }),
    debug: Object.freeze({ bodyOverlay: true, trail: true, labels: true, faceOverlay: true }),
  });

  function clamp01(value) {
    if (!Number.isFinite(value)) return null;
    return Math.max(0, Math.min(1, value));
  }

  function signalPercent(value) {
    const normalized = clamp01(value);
    return normalized === null ? 0 : Math.round(normalized * 100);
  }

  function presetFor(name, faceEnabled = true) {
    const preset = PRESETS[name];
    if (!preset) return null;
    return {
      ...preset,
      faceOverlay: faceEnabled ? preset.faceOverlay : false,
    };
  }

  function matchingPreset(viewState, faceEnabled = true) {
    for (const name of Object.keys(PRESETS)) {
      const preset = presetFor(name, faceEnabled);
      if (Object.keys(preset).every((key) => Boolean(viewState[key]) === preset[key])) return name;
    }
    return null;
  }

  function describeEyes(face) {
    if (!face || !face.detected) return "aguardando sinal";
    const values = [face.eye_blink_left, face.eye_blink_right].filter(Number.isFinite);
    if (!values.length) return "sinal incompleto";
    const strongest = Math.max(...values);
    if (strongest >= 0.68) return "fechamento detectado";
    if (strongest >= 0.34) return "olhos em transição";
    return "olhos abertos";
  }

  function describeHead(face, threshold = 16) {
    if (!face || !face.detected) return "aguardando sinal";
    const yaw = Number.isFinite(face.head_yaw) ? Math.abs(face.head_yaw) : null;
    const pitch = Number.isFinite(face.head_pitch) ? Math.abs(face.head_pitch) : null;
    const roll = Number.isFinite(face.head_roll) ? Math.abs(face.head_roll) : null;
    if (yaw === null && pitch === null && roll === null) return "sinal incompleto";

    const lateral = yaw !== null && yaw >= threshold;
    const vertical = pitch !== null && pitch >= threshold;
    const tilted = roll !== null && roll >= threshold;
    if ((lateral && vertical) || (lateral && tilted) || (vertical && tilted)) return "fora do centro";
    if (lateral) return "virada lateralmente";
    if (vertical) return "inclinada verticalmente";
    if (tilted) return "cabeça inclinada";
    return "centralizada";
  }

  const StudioControls = {
    PRESETS,
    clamp01,
    signalPercent,
    presetFor,
    matchingPreset,
    describeEyes,
    describeHead,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = StudioControls;
  } else {
    global.StudioControls = StudioControls;
  }
})(typeof window !== "undefined" ? window : globalThis);
