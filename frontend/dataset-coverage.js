/* Sprint 13 — dataset coverage view-model (no DOM).
 *
 * Transforms GET /dataset/stats JSON into display-ready structures. The browser
 * never recomputes rows, sessions, or readiness — it only formats what the
 * backend already decided (backend/dataset/stats.py).
 */

(function (global) {
  "use strict";

  const DEFAULT_LABELS = Object.freeze([
    "arm_raised",
    "arms_crossed",
    "arms_open",
    "idle",
  ]);

  const DEFAULT_LABEL_TITLES = Object.freeze({
    arm_raised: "braço levantado",
    arms_crossed: "braços cruzados",
    arms_open: "braços abertos",
    idle: "parado",
    wave: "aceno",
  });

  const DROPPED_REASON_TITLES = Object.freeze({
    missing_label: "rótulo ausente",
    missing_feature: "feature ausente",
    non_finite: "valor não finito",
    unknown_label: "rótulo desconhecido",
  });

  function formatSeconds(value) {
    if (!Number.isFinite(value)) return "0.0";
    return value.toFixed(1);
  }

  function normalizeDropped(dropped) {
    if (!dropped || typeof dropped !== "object") return [];
    return Object.entries(dropped)
      .filter(([, count]) => Number(count) > 0)
      .map(([reason, count]) => ({
        reason,
        count: Number(count),
        title: DROPPED_REASON_TITLES[reason] || reason.replaceAll("_", " "),
      }));
  }

  function coverageLabelOrder(stats, options) {
    const opts = options || {};
    if (Array.isArray(opts.coverageLabels) && opts.coverageLabels.length > 0) {
      return opts.coverageLabels.slice();
    }
    const keys = stats && stats.labels && typeof stats.labels === "object" ? Object.keys(stats.labels) : [];
    const ordered = DEFAULT_LABELS.filter((key) => keys.includes(key));
    if (keys.includes("wave") && !ordered.includes("wave")) ordered.push("wave");
    return ordered.length > 0 ? ordered : [...DEFAULT_LABELS];
  }

  function labelEntry(stats, key) {
    const bucket = stats && stats.labels && stats.labels[key];
    if (!bucket || typeof bucket !== "object") {
      return { rows: 0, sessions: 0, seconds: 0.0 };
    }
    return {
      rows: Number(bucket.rows) || 0,
      sessions: Number(bucket.sessions) || 0,
      seconds: Number.isFinite(Number(bucket.seconds)) ? Number(bucket.seconds) : 0.0,
    };
  }

  function buildCoverageView(stats, options) {
    const opts = options || {};
    const titles = { ...DEFAULT_LABEL_TITLES, ...(opts.labelTitles || {}) };

    if (!stats || typeof stats !== "object") {
      return {
        valid: false,
        exists: false,
        ready: false,
        path: "",
        minSessionsPerLabel: 0,
        totalRows: 0,
        usableRows: 0,
        dropped: [],
        problems: [],
        labels: [],
        showMissingDataset: true,
        statusKind: "missing",
        statusText: "sem dados",
        summaryLine: "Estatísticas indisponíveis.",
      };
    }

    const exists = Boolean(stats.exists);
    const minSessions = Number(stats.min_sessions_per_label) || 0;
    const ready = Boolean(stats.ready);
    const problems = Array.isArray(stats.problems) ? stats.problems.filter(Boolean) : [];

    const labelOrder = coverageLabelOrder(stats, opts);
    const labels = labelOrder.map((key) => {
      const entry = labelEntry(stats, key);
      const belowMinimum = exists && minSessions > 0 && entry.sessions < minSessions;
      return {
        key,
        title: titles[key] || key,
        rows: entry.rows,
        sessions: entry.sessions,
        seconds: entry.seconds,
        secondsText: formatSeconds(entry.seconds),
        sessionsText: `${entry.sessions} / ${minSessions}`,
        belowMinimum,
        isEmpty: entry.sessions === 0 && entry.rows === 0,
      };
    });

    let statusKind = "ready";
    let statusText = "pronto";
    if (!exists) {
      statusKind = "missing";
      statusText = "ausente";
    } else if (!ready) {
      statusKind = "incomplete";
      statusText = "incompleto";
    }

    const totalRows = Number(stats.total_rows) || 0;
    const usableRows = Number(stats.usable_rows) || 0;
    const summaryLine = !exists
      ? "Nenhum arquivo de treino encontrado ainda."
      : `${usableRows} linhas utilizáveis de ${totalRows} no CSV`;

    return {
      valid: true,
      exists,
      ready,
      path: typeof stats.path === "string" ? stats.path : "",
      minSessionsPerLabel: minSessions,
      totalRows,
      usableRows,
      dropped: normalizeDropped(stats.dropped),
      problems,
      labels,
      showMissingDataset: !exists,
      statusKind,
      statusText,
      summaryLine,
    };
  }

  const DatasetCoverage = {
    DEFAULT_LABELS,
    DEFAULT_LABEL_TITLES,
    DROPPED_REASON_TITLES,
    coverageLabelOrder,
    buildCoverageView,
    formatSeconds,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = DatasetCoverage;
  } else {
    global.DatasetCoverage = DatasetCoverage;
  }
})(typeof window !== "undefined" ? window : globalThis);