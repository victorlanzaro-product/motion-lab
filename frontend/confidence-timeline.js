/* Sprint 15 groundwork — in-memory ML confidence + rule agreement timeline.
 *
 * Pure data logic (no DOM), same UMD pattern as face-history.js so
 * `node --test` can cover it with zero new dependencies.
 *
 * The window is 30 s of server `timestamp` values — never `Date.now()`.
 * Samples exist only where a frame arrived; nothing is interpolated across
 * gaps. Missing ML (`ml: null` on the wire) is stored with `mlProbability:
 * null`, never coerced to 0.
 */

(function (global) {
  "use strict";

  const DEFAULT_WINDOW_SECONDS = 30;
  const DEFAULT_MAX_POINTS = 3600;

  function extractMl(frame) {
    const ml = frame && frame.ml;
    if (ml == null || typeof ml !== "object") {
      return { mlLabel: null, mlProbability: null };
    }
    const mlLabel = typeof ml.label === "string" ? ml.label : null;
    if (mlLabel === null) {
      return { mlLabel: null, mlProbability: null };
    }
    let mlProbability = null;
    if (typeof ml.confidence === "number" && Number.isFinite(ml.confidence)) {
      mlProbability = ml.confidence;
    } else if (ml.probabilities && typeof ml.probabilities === "object") {
      const raw = ml.probabilities[mlLabel];
      if (typeof raw === "number" && Number.isFinite(raw)) {
        mlProbability = raw;
      }
    }
    return { mlLabel, mlProbability };
  }

  function extractRuleLabel(frame) {
    if (!frame || typeof frame !== "object") return null;
    const raw = frame.rule_label ?? frame.ruleLabel;
    return typeof raw === "string" ? raw : null;
  }

  function agreementBetween(mlLabel, ruleLabel) {
    if (mlLabel === null || ruleLabel === null) return null;
    return mlLabel === ruleLabel;
  }

  /**
   * Rolling buffer for the confidence timeline chart (wiring comes later).
   *
   * Agreement is strictly label equality on the two strings already on the
   * sample — never a re-read of `features` at draw or rate time.
   */
  function createTimeline(options) {
    const opts = options || {};
    const windowSeconds = opts.windowSeconds || DEFAULT_WINDOW_SECONDS;
    const maxPoints = opts.maxPoints || DEFAULT_MAX_POINTS;

    /** @type {{ t: number, mlLabel: string|null, mlProbability: number|null, ruleLabel: string|null, agreement: boolean|null }[]} */
    const samples = [];
    let lastT = null;

    function clear() {
      samples.length = 0;
      lastT = null;
    }

    function prune() {
      if (lastT === null) return;
      const cutoff = lastT - windowSeconds;
      let start = 0;
      while (start < samples.length && samples[start].t < cutoff) start += 1;
      if (start > 0) samples.splice(0, start);
      if (samples.length > maxPoints) {
        samples.splice(0, samples.length - maxPoints);
      }
    }

    function push(frameMessage) {
      if (!frameMessage || typeof frameMessage !== "object") return;
      const t = frameMessage.timestamp;
      if (typeof t !== "number" || !Number.isFinite(t)) return;
      if (lastT !== null && t <= lastT) return;

      const { mlLabel, mlProbability } = extractMl(frameMessage);
      const ruleLabel = extractRuleLabel(frameMessage);
      const agreement = agreementBetween(mlLabel, ruleLabel);

      lastT = t;
      samples.push({ t, mlLabel, mlProbability, ruleLabel, agreement });
      prune();
    }

    /**
     * Points ready to map to canvas pixels. `x` is 0 at the oldest edge of
     * the window and 1 at `now` (newest), matching face-history's time axis.
     */
    function series(now) {
      const end =
        typeof now === "number" && Number.isFinite(now)
          ? now
          : lastT;
      if (end === null) return [];
      const cutoff = end - windowSeconds;
      const span = windowSeconds;
      const out = [];
      for (const sample of samples) {
        if (sample.t < cutoff || sample.t > end) continue;
        const x = span === 0 ? 1 : (sample.t - cutoff) / span;
        out.push({
          t: sample.t,
          x,
          mlLabel: sample.mlLabel,
          mlProbability: sample.mlProbability,
          ruleLabel: sample.ruleLabel,
          agreement: sample.agreement,
        });
      }
      return out;
    }

    /** Mirrors Sprint 10 report: skip frames with no ML; exclude `wave`. */
    function agreementRate() {
      let compared = 0;
      let matches = 0;
      for (const sample of samples) {
        if (sample.mlLabel === null) continue;
        if (sample.mlLabel === "wave") continue;
        if (sample.ruleLabel === null) continue;
        compared += 1;
        if (sample.agreement) matches += 1;
      }
      return compared > 0 ? matches / compared : null;
    }

    return {
      push,
      series,
      agreementRate,
      clear,
      windowSeconds,
      now: () => lastT,
    };
  }

  const ConfidenceTimeline = {
    createTimeline,
    extractMl,
    extractRuleLabel,
    agreementBetween,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = ConfidenceTimeline;
  } else {
    global.ConfidenceTimeline = ConfidenceTimeline;
  }
})(typeof window !== "undefined" ? window : globalThis);