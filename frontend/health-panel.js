/* Sprint 15 — pipeline health view model (pure, no DOM).
 *
 * Numbers come straight off the WebSocket frame; null means absent, never 0.
 */

(function (global) {
  "use strict";

  const DROPPED_RATE_ALERT = 0.1;

  const TIMING_STAGES = [
    { key: "capture_ms", label: "captura" },
    { key: "pose_ms", label: "pose" },
    { key: "features_ms", label: "features" },
    { key: "ml_ms", label: "ML" },
    { key: "face_ms", label: "rosto" },
  ];

  function formatMs(value, digits = 1) {
    if (value === null || value === undefined) return "—";
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return value.toFixed(digits);
  }

  function barFraction(ms, totalMs) {
    if (ms === null || ms === undefined) return 0;
    if (typeof ms !== "number" || !Number.isFinite(ms) || ms <= 0) return 0;
    if (typeof totalMs !== "number" || !Number.isFinite(totalMs) || totalMs <= 0) return 0;
    return Math.min(1, ms / totalMs);
  }

  function extractHealth(frame) {
    const health = frame && frame.health;
    if (!health || typeof health !== "object") {
      return { fps: null, dropped_frames: null, dropped_rate: null };
    }
    const fps = typeof health.fps === "number" && Number.isFinite(health.fps) ? health.fps : null;
    const dropped_frames =
      typeof health.dropped_frames === "number" && Number.isFinite(health.dropped_frames)
        ? health.dropped_frames
        : null;
    const dropped_rate =
      typeof health.dropped_rate === "number" && Number.isFinite(health.dropped_rate)
        ? health.dropped_rate
        : null;
    return { fps, dropped_frames, dropped_rate };
  }

  function extractTimings(frame) {
    const timings = frame && frame.timings;
    if (!timings || typeof timings !== "object") return null;
    return timings;
  }

  function buildStageRows(timings) {
    const totalMs =
      timings && typeof timings.total_ms === "number" && Number.isFinite(timings.total_ms)
        ? timings.total_ms
        : null;
    return TIMING_STAGES.map(({ key, label }) => {
      const raw = timings ? timings[key] : undefined;
      const ms =
        raw === null || raw === undefined
          ? null
          : typeof raw === "number" && Number.isFinite(raw)
            ? raw
            : null;
      return {
        key,
        label,
        ms,
        display: formatMs(ms),
        barFraction: barFraction(ms, totalMs),
      };
    });
  }

  function droppedRateAlert(rate) {
    return typeof rate === "number" && Number.isFinite(rate) && rate > DROPPED_RATE_ALERT;
  }

  function formatDroppedRate(rate) {
    if (rate === null || rate === undefined) return "—";
    if (typeof rate !== "number" || !Number.isFinite(rate)) return "—";
    return `${(rate * 100).toFixed(2)}%`;
  }

  function snapshot(frame) {
    const health = extractHealth(frame);
    const timings = extractTimings(frame);
    return {
      fps: health.fps,
      fpsDisplay: formatMs(health.fps, 1),
      dropped_frames: health.dropped_frames,
      dropped_rate: health.dropped_rate,
      droppedRateDisplay: formatDroppedRate(health.dropped_rate),
      droppedRateAlert: droppedRateAlert(health.dropped_rate),
      stages: buildStageRows(timings),
      totalDisplay: formatMs(timings && timings.total_ms),
    };
  }

  const HealthPanel = {
    TIMING_STAGES,
    DROPPED_RATE_ALERT,
    formatMs,
    formatDroppedRate,
    barFraction,
    droppedRateAlert,
    snapshot,
    buildStageRows,
    extractHealth,
    extractTimings,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = HealthPanel;
  } else {
    global.HealthPanel = HealthPanel;
  }
})(typeof window !== "undefined" ? window : globalThis);