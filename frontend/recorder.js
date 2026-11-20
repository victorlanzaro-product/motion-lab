/* Sprint 12 — browser-side Recorder UI state (no DOM).
 *
 * Pure state machine: idle → countdown(3s) → (record_start on wire) → mirrors
 * server `record_status` while recording → idle. Cancelling the countdown sends
 * nothing. Outbound commands are queued for app.js to drain and send on the
 * WebSocket; inbound status never touches the socket here.
 */

(function (global) {
  "use strict";

  const DEFAULT_COUNTDOWN_SECONDS = 3;
  const START_REPLAY_WINDOW_MS = 2000;
  const START_FAILED_ERROR = "Gravação não iniciada: conexão caiu";

  function createRecorder(options) {
    const opts = options || {};
    const countdownSeconds = opts.countdownSeconds ?? DEFAULT_COUNTDOWN_SECONDS;

    let labels = [];
    let phase = "idle";
    let countdownLabel = null;
    let countdownDeadline = null;

    let status = emptyStatus();
    let lastSummary = null;
    const outbound = [];

    function emptyStatus() {
      return {
        recording: false,
        label: null,
        session_id: null,
        written: 0,
        skipped: 0,
        owner: false,
        error: null,
      };
    }

    function resetCountdown() {
      phase = "idle";
      countdownLabel = null;
      countdownDeadline = null;
    }

    function setLabels(next) {
      labels = Array.isArray(next) ? next.slice() : [];
    }

    function remoteRecording() {
      return status.recording && !status.owner;
    }

    function canStart() {
      if (remoteRecording()) return false;
      if (status.recording) return false;
      return phase === "idle";
    }

    function beginStart(label, now) {
      const t = typeof now === "number" ? now : Date.now();
      if (!labels.includes(label)) return { ok: false, reason: "invalid_label" };
      if (!canStart()) return { ok: false, reason: "busy" };
      phase = "countdown";
      countdownLabel = label;
      countdownDeadline = t + countdownSeconds * 1000;
      return { ok: true };
    }

    function cancelCountdown() {
      if (phase !== "countdown") return false;
      resetCountdown();
      return true;
    }

    function enqueueOutbound(payload, now) {
      const t = typeof now === "number" ? now : Date.now();
      outbound.push({ payload, requestedAt: t });
    }

    function outboundPayloads(entries) {
      return entries.map((entry) => entry.payload);
    }

    function beginStop(now) {
      if (!status.recording || !status.owner) return { ok: false, reason: "not_owner" };
      enqueueOutbound({ type: "record_stop" }, now);
      return { ok: true };
    }

    function tick(now) {
      const t = typeof now === "number" ? now : Date.now();
      if (phase !== "countdown" || countdownDeadline === null) return [];
      if (t < countdownDeadline) return [];
      const label = countdownLabel;
      resetCountdown();
      enqueueOutbound({ type: "record_start", label }, t);
      return drainOutbound();
    }

    function isIdleRejection(message) {
      return (
        message.error != null &&
        !message.recording &&
        message.label == null &&
        message.session_id == null
      );
    }

    function applyStatus(message) {
      if (!message || message.type !== "record_status") return { ok: false };

      // error != null => private to this tab; idle rejections must never stomp
      // another tab's live remote-recording snapshot (defensive misdelivery).
      if (isIdleRejection(message) && remoteRecording()) {
        return {
          ok: true,
          ignored: true,
          status: { ...status },
          lastSummary: lastSummary ? { ...lastSummary } : null,
        };
      }

      status = {
        recording: Boolean(message.recording),
        label: message.label ?? null,
        session_id: message.session_id ?? null,
        written: Number(message.written) || 0,
        skipped: Number(message.skipped) || 0,
        owner: Boolean(message.owner),
        error: message.error ?? null,
      };

      if (status.recording && !status.owner && phase === "countdown") {
        resetCountdown();
      }

      if (!status.recording && message.label != null && message.session_id != null) {
        lastSummary = {
          label: message.label,
          session_id: message.session_id,
          written: status.written,
          skipped: status.skipped,
          error: status.error,
        };
      }

      return { ok: true, status: { ...status }, lastSummary: lastSummary ? { ...lastSummary } : null };
    }

    function onWebSocketReconnect() {
      if (phase === "countdown") resetCountdown();
      return { ok: true };
    }

    function drainOutbound() {
      return outbound.splice(0);
    }

    function requeueOutbound(entries) {
      if (!Array.isArray(entries) || entries.length === 0) return;
      outbound.unshift(...entries);
    }

    function prepareOutboundReplay(now) {
      const t = typeof now === "number" ? now : Date.now();
      let discardedStart = false;
      const kept = [];
      for (const entry of outbound) {
        if (!entry || !entry.payload) continue;
        if (entry.payload.type === "record_stop") continue;
        if (entry.payload.type === "record_start") {
          if (t - entry.requestedAt <= START_REPLAY_WINDOW_MS) {
            kept.push(entry);
          } else {
            discardedStart = true;
          }
          continue;
        }
        kept.push(entry);
      }
      outbound.length = 0;
      outbound.push(...kept);
      if (discardedStart) {
        status = { ...emptyStatus(), error: START_FAILED_ERROR };
      }
      return { ok: true, discardedStart, error: discardedStart ? START_FAILED_ERROR : null };
    }

    function countdownRemaining(now) {
      const t = typeof now === "number" ? now : Date.now();
      if (phase !== "countdown" || countdownDeadline === null) return null;
      const seconds = Math.ceil((countdownDeadline - t) / 1000);
      return Math.max(0, seconds);
    }

    function snapshot(now) {
      return {
        labels: labels.slice(),
        phase,
        countdownLabel,
        countdownRemaining: countdownRemaining(now),
        status: { ...status },
        lastSummary: lastSummary ? { ...lastSummary } : null,
        showRecordingBanner: status.recording,
        isRemoteRecording: remoteRecording(),
        startEnabled: canStart(),
        stopEnabled: status.recording && status.owner,
        cancelCountdownEnabled: phase === "countdown",
      };
    }

    return {
      COUNTDOWN_SECONDS: countdownSeconds,
      setLabels,
      beginStart,
      cancelCountdown,
      beginStop,
      tick,
      applyStatus,
      onWebSocketReconnect,
      drainOutbound,
      requeueOutbound,
      outboundPayloads,
      prepareOutboundReplay,
      snapshot,
      canStart,
    };
  }

  const Recorder = {
    createRecorder,
    DEFAULT_COUNTDOWN_SECONDS,
    START_REPLAY_WINDOW_MS,
    START_FAILED_ERROR,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = Recorder;
  } else {
    global.Recorder = Recorder;
  }
})(typeof window !== "undefined" ? window : globalThis);
