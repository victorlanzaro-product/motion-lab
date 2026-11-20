/* Sprint 16 — WebSocket consumer rules (pure, no DOM).
 *
 * Server errors shown to the operator stay in Portuguese as received.
 * Technical detail belongs in `console` only (app.js decides what to log).
 */

(function (global) {
  "use strict";

  const CLIENT_RECORDER_ERRORS = {
    invalid_label: "Escolha um rótulo de gravação oferecido pelo servidor.",
    busy: "Aguarde: gravação ou contagem em andamento em outra aba.",
  };

  const DEFAULT_DEGRADATION_TTL_MS = 30_000;

  /** @param {{ fatal?: boolean } | null | undefined} message */
  function isFatalPipelineError(message) {
    if (!message || message.type !== "error") return false;
    if (message.fatal === false) return false;
    return true;
  }

  /** Idle or live rejections: show servidor text, never treat as pipeline stop. */
  function isRecorderRejection(message) {
    if (!message || message.type !== "record_status") return false;
    return typeof message.error === "string" && message.error.length > 0;
  }

  function clientRecorderReason(reason) {
    if (typeof reason !== "string") return null;
    return CLIENT_RECORDER_ERRORS[reason] || null;
  }

  function displayDegradationMessage(message) {
    if (!message || message.type !== "error") return null;
    if (typeof message.message !== "string" || !message.message.trim()) return null;
    return message.message.trim();
  }

  /** Toast only when the dedicated recorder panel cannot show the rejection. */
  function shouldToastRecorderRejection(recorderErrorPanelVisible) {
    return !recorderErrorPanelVisible;
  }

  function setRecordingBannerHidden(element, visible) {
    if (!element) return;
    const show = Boolean(visible);
    element.hidden = !show;
    if (element.classList && typeof element.classList.toggle === "function") {
      element.classList.toggle("hidden", !show);
    }
  }

  function createDegradationTracker(options) {
    const opts = options || {};
    const ttlMs =
      typeof opts.ttlMs === "number" && Number.isFinite(opts.ttlMs)
        ? opts.ttlMs
        : DEFAULT_DEGRADATION_TTL_MS;
    const now = typeof opts.now === "function" ? opts.now : () => Date.now();
    let note = null;
    let updatedAt = null;

    function clear() {
      note = null;
      updatedAt = null;
    }

    function absorbError(message) {
      if (isFatalPipelineError(message)) {
        clear();
        return { fatal: true, note: null };
      }
      const next = displayDegradationMessage(message);
      if (next) {
        note = next;
        updatedAt = now();
      }
      return { fatal: false, note: currentNote() };
    }

    function currentNote(at) {
      if (!note || updatedAt === null) return null;
      const t = at !== undefined ? at : now();
      if (t - updatedAt > ttlMs) {
        clear();
        return null;
      }
      return note;
    }

    return { clear, absorbError, currentNote, ttlMs };
  }

  const ProtocolHardening = {
    CLIENT_RECORDER_ERRORS,
    DEFAULT_DEGRADATION_TTL_MS,
    isFatalPipelineError,
    isRecorderRejection,
    clientRecorderReason,
    displayDegradationMessage,
    shouldToastRecorderRejection,
    setRecordingBannerHidden,
    createDegradationTracker,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = ProtocolHardening;
  } else {
    global.ProtocolHardening = ProtocolHardening;
  }
})(typeof window !== "undefined" ? window : globalThis);
