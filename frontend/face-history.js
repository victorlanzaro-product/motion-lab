/* Sprint 11 — in-memory history for the facial line charts.
 *
 * Pure data logic, deliberately kept out of app.js and free of `window`/
 * `canvas` calls so it can run under `node --test` with no test framework
 * added to the project. The UMD-style export at the bottom is what makes
 * that possible: a classic `<script>` tag sees `window.FaceHistory`, Node's
 * module loader sees `module.exports`, same file either way.
 *
 * Three rules the charts depend on, enforced here rather than at draw time:
 *
 *   1. Nothing is ever persisted. `createStore` only ever holds arrays in
 *      memory; there is no localStorage/sessionStorage/IndexedDB call in this
 *      file, on purpose — history must not outlive the tab.
 *   2. The window is time-based, not frame-count-based. Points older than
 *      `windowSeconds` (relative to the newest sample) are dropped every
 *      push, so the chart always shows "the last ~10s" regardless of fps.
 *   3. A gap breaks the line. `buildSegments` never draws a straight line
 *      across a stretch where the value was unknown (explicit `null`) or
 *      simply didn't arrive for a while (`maxGapSeconds`) — the same rule
 *      `Trail`/`trail_payload` already apply to the wrist path
 *      (backend/web/payload.py:trail_payload).
 */

(function (global) {
  "use strict";

  const DEFAULT_WINDOW_SECONDS = 10;
  const DEFAULT_MAX_POINTS = 1200; // backstop: bounds memory even if timestamps misbehave
  const DEFAULT_MAX_GAP_SECONDS = 1.0; // longer than one stalled frame, shorter than "gone"

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  /**
   * One rolling, in-memory buffer per series name.
   *
   * `push(t, values)` is meant to be called once per incoming WebSocket
   * frame, whether or not a face was detected — `values[name]` being
   * missing/non-finite is exactly how "unavailable this frame" is recorded,
   * and it is what makes the line break instead of connecting across it.
   */
  function createStore(seriesNames, options) {
    const opts = options || {};
    const windowSeconds = opts.windowSeconds || DEFAULT_WINDOW_SECONDS;
    const maxPoints = opts.maxPoints || DEFAULT_MAX_POINTS;
    const maxGapSeconds = opts.maxGapSeconds || DEFAULT_MAX_GAP_SECONDS;

    const names = seriesNames.slice();
    const series = {};
    for (const name of names) series[name] = [];
    let lastT = null;

    function reset() {
      for (const name of names) series[name] = [];
      lastT = null;
    }

    function prune() {
      if (lastT === null) return;
      const cutoff = lastT - windowSeconds;
      for (const name of names) {
        const points = series[name];
        let start = 0;
        while (start < points.length && points[start].t < cutoff) start += 1;
        if (start > 0) points.splice(0, start);
        if (points.length > maxPoints) points.splice(0, points.length - maxPoints);
      }
    }

    function push(t, values) {
      if (typeof t !== "number" || !Number.isFinite(t)) return;
      // Clock went backward — a new pipeline run reconnected (camera reopened,
      // server restarted). The old run's samples share no time origin with
      // the new ones, so keeping them would either draw off-canvas or, worse,
      // splice a fake line between two unrelated runs. Starting over is the
      // only reading that isn't misleading.
      if (lastT !== null && t < lastT) reset();
      lastT = t;
      for (const name of names) {
        const raw = values ? values[name] : undefined;
        const v = typeof raw === "number" && Number.isFinite(raw) ? raw : null;
        series[name].push({ t, v });
      }
      prune();
    }

    function segments(name) {
      return buildSegments(series[name] || [], maxGapSeconds);
    }

    function raw(name) {
      return series[name] || [];
    }

    return {
      push,
      segments,
      raw,
      reset,
      windowSeconds,
      maxGapSeconds,
      now: () => lastT,
    };
  }

  /**
   * Splits one series' points into continuous runs, breaking wherever the
   * line must not connect two real samples:
   *
   *   - an explicit `null` (value unknown this frame) always breaks;
   *   - a time gap wider than `maxGapSeconds` breaks, even with no `null` in
   *     between — frames that simply stopped arriving for a while.
   *
   * Runs of a single point are kept (they draw as a dot, not a line) rather
   * than dropped, so one isolated sample between two gaps is still visible.
   */
  function buildSegments(points, maxGapSeconds) {
    const runs = [];
    let current = [];
    let prevT = null;
    for (const point of points) {
      if (point.v === null || point.v === undefined) {
        if (current.length) runs.push(current);
        current = [];
        prevT = null;
        continue;
      }
      if (prevT !== null && point.t - prevT > maxGapSeconds) {
        if (current.length) runs.push(current);
        current = [];
      }
      current.push(point);
      prevT = point.t;
    }
    if (current.length) runs.push(current);
    return runs;
  }

  /**
   * Time+value -> canvas pixel, for a chart showing the last `windowSeconds`
   * up to `now`. Kept separate from any `CanvasRenderingContext2D` call so
   * the coordinate math is testable without a DOM.
   *
   * The value is clamped into [min, max] for the y position — the numeric
   * readout elsewhere on the page still shows the exact figure; the chart is
   * a trend, not the source of truth, so an out-of-range spike is drawn
   * pinned to the edge rather than off-canvas or rescaling the whole axis.
   */
  function layoutPoint(point, layout) {
    const { now, windowSeconds, min, max, width, height } = layout;
    const age = now - point.t; // seconds ago, >= 0 for anything in the window
    const x = width * (1 - clamp(age, 0, windowSeconds) / windowSeconds);
    const span = max - min;
    const value = clamp(point.v, min, max);
    const y = span === 0 ? height / 2 : height * (1 - (value - min) / span);
    return { x, y };
  }

  const FaceHistory = { createStore, buildSegments, layoutPoint, clamp };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = FaceHistory;
  } else {
    global.FaceHistory = FaceHistory;
  }
})(typeof window !== "undefined" ? window : globalThis);
