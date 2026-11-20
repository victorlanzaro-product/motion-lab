"""Thread-safe coordination for browser-driven dataset recording.

WebSocket handlers run on the asyncio loop while the camera pipeline owns the
CSV writer on its worker thread. This class reserves sessions synchronously so
two tabs cannot win a start race, then hands every file operation to
``record_frame``/``finish`` on the pipeline thread.
"""

from __future__ import annotations

import errno
import logging
import secrets
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from backend.dataset.writer import (
    DEFAULT_DATASET_PATH,
    DatasetWriter,
    DatasetWriterError,
    validate_label,
)
from backend.features.features import FrameFeatures
from backend.features.velocity import MotionState


logger = logging.getLogger(__name__)

INVALID_RECORD_LABEL = "Informe um rótulo de gravação válido."


def validate_record_label(label: object) -> str:
    if not isinstance(label, str):
        raise DatasetWriterError(INVALID_RECORD_LABEL)
    return validate_label(label)


def _file_error(message: str, exc: Exception) -> str:
    if isinstance(exc, PermissionError):
        return f"{message} Verifique a permissão de acesso ao arquivo de gravação."
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return f"{message} Não há espaço livre para gravar."
    return message


class Recorder:
    """Coordinate recording requests without doing file I/O on the event loop.

    Requests reserve or enqueue work under a short lock. The pipeline thread
    consumes that work and exclusively owns ``DatasetWriter`` lifecycle calls.
    """

    STATUS_INTERVAL = 0.25
    SESSION_SUFFIX_SPACE = 16**6
    SESSION_SUFFIX_ATTEMPTS = 128

    def __init__(
        self,
        path: Path = DEFAULT_DATASET_PATH,
        writer_factory: Callable[..., DatasetWriter] = DatasetWriter,
    ) -> None:
        self.path = Path(path)
        self.writer_factory = writer_factory
        self._lock = threading.Lock()
        self._commands: deque[tuple[str, Any]] = deque()
        self._recording = False
        self._label: Optional[str] = None
        self._session_id: Optional[str] = None
        self._owner: Any = None
        self._session_suffixes: set[str] = set()
        self._written = 0
        self._skipped = 0
        self._stop_requested = False
        self._last_status_at = time.monotonic()
        # Only the pipeline thread reads or writes this handle.
        self._writer: Optional[DatasetWriter] = None

    def request_start(self, label: object, owner: Any) -> dict[str, Any]:
        """Reserve a new session and enqueue its writer open for the pipeline.

        Label validation is pure and shared with ``DatasetWriter``. It happens
        before the reservation or any file operation, so rejected labels never
        create the dataset file.
        """
        try:
            clean_label = validate_record_label(label)
        except DatasetWriterError:
            with self._lock:
                return self._snapshot_locked(error=INVALID_RECORD_LABEL)

        with self._lock:
            if self._recording:
                return self._snapshot_locked(error="Já há uma gravação em andamento.")

            # Session generation runs on the event loop; keep even a broken
            # randomness source from escaping and closing this WebSocket.
            try:
                session_id = self._new_session_id()
            except Exception:  # noqa: BLE001
                logger.exception("Could not allocate recording session ID")
                return self._snapshot_locked(error="Não foi possível criar a sessão de gravação.")
            self._recording = True
            self._label = clean_label
            self._session_id = session_id
            self._owner = owner
            self._written = 0
            self._skipped = 0
            self._stop_requested = False
            self._last_status_at = time.monotonic()
            self._commands.append(("start", (clean_label, session_id)))
            return self._snapshot_locked()

    def request_stop(self, owner: Any) -> Optional[dict[str, Any]]:
        """Queue a stop only for the owning tab; return a requester-only error."""
        with self._lock:
            if not self._recording:
                return self._snapshot_locked(error="Nenhuma gravação está em andamento.")
            if owner is not self._owner:
                return self._snapshot_locked(
                    error="Somente a aba que iniciou a gravação pode encerrá-la."
                )
            self._queue_stop_locked()
            return None

    def owner_disconnected(self, owner: Any) -> bool:
        """Stop a session when its owning WebSocket disconnects."""
        with self._lock:
            if not self._recording or owner is not self._owner:
                return False
            self._queue_stop_locked()
            return True

    def is_owner(self, owner: Any) -> bool:
        """Whether a subscriber owns the active reservation."""
        with self._lock:
            return self._recording and owner is self._owner

    def record_frame(
        self, features: FrameFeatures, motion: MotionState
    ) -> Optional[dict[str, Any]]:
        """Apply pending commands and write this frame on the pipeline thread.

        Writer failures become a final status instead of escaping into the
        camera loop. That keeps this optional dataset layer from taking pose,
        gestures, or live ML down with it.
        """
        with self._lock:
            commands = list(self._commands)
            self._commands.clear()

        for command, value in commands:
            if command == "start":
                label, session_id = value
                writer = None
                try:
                    writer = self.writer_factory(self.path, session_id=session_id)
                    writer.open()
                    self._writer = writer
                except Exception as exc:  # noqa: BLE001 - optional layer boundary
                    logger.exception("Could not open dataset for recording")
                    if writer is not None:
                        try:
                            writer.close()
                        except Exception:  # noqa: BLE001 - contain teardown
                            logger.exception("Could not close dataset after open failed")
                    if isinstance(exc, DatasetWriterError):
                        error = (
                            "O arquivo de gravação tem colunas incompatíveis. "
                            "Mova ou renomeie o arquivo antes de tentar de novo."
                        )
                    else:
                        error = _file_error("Não foi possível abrir o arquivo de gravação.", exc)
                    return self._finish_on_pipeline_thread(error)
            elif command == "stop":
                return self._finish_on_pipeline_thread()

        with self._lock:
            recording = self._recording
            label = self._label
            session_id = self._session_id
        if not recording or self._writer is None or label is None or session_id is None:
            return None

        try:
            written = self._writer.write(label, features, motion)
        except Exception as exc:  # noqa: BLE001 - optional layer boundary
            logger.exception("Could not write dataset frame")
            return self._finish_on_pipeline_thread(
                _file_error("Não foi possível salvar o quadro da gravação.", exc)
            )

        with self._lock:
            # A frame has only one pipeline writer, so this session cannot
            # change until the next call. Requests on the event loop only queue.
            if self._recording and self._session_id == session_id:
                if written:
                    self._written += 1
                else:
                    self._skipped += 1
        return None

    def status(self) -> dict[str, Any]:
        """Return the current state; an idle snapshot has no old session data."""
        with self._lock:
            return self._snapshot_locked()

    def status_for(self, owner: Any, error: Optional[str] = None) -> dict[str, Any]:
        """Capture status and ownership together for a single subscriber."""
        with self._lock:
            return {
                **self._snapshot_locked(error=error),
                "owner": self._recording and owner is self._owner,
            }

    def status_if_due(self) -> Optional[dict[str, Any]]:
        """Return live counts no more often than the wire contract permits."""
        now = time.monotonic()
        with self._lock:
            if not self._recording or now - self._last_status_at < self.STATUS_INTERVAL:
                return None
            self._last_status_at = now
            return self._snapshot_locked()

    def finish(self, error: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Close the active writer on pipeline teardown and return its summary."""
        with self._lock:
            active = self._recording
            if not active:
                self._commands.clear()
                return None
            # A queued start may not yet have reached the pipeline. Teardown
            # finalizes the reservation without opening a file that cannot be
            # used for another frame.
            self._commands.clear()
        return self._finish_on_pipeline_thread(error)

    def _queue_stop_locked(self) -> None:
        if not self._stop_requested:
            self._stop_requested = True
            self._commands.append(("stop", None))

    def _new_session_id(self) -> str:
        """Return an unused suffix so sessions stay distinct for this process."""
        if len(self._session_suffixes) >= self.SESSION_SUFFIX_SPACE:
            raise RuntimeError("Recorder session ID suffix space is exhausted.")
        for _attempt in range(self.SESSION_SUFFIX_ATTEMPTS):
            suffix = secrets.token_hex(3)
            if suffix not in self._session_suffixes:
                self._session_suffixes.add(suffix)
                return f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{suffix}"
        raise RuntimeError(
            "Could not generate a unique Recorder session ID suffix after "
            f"{self.SESSION_SUFFIX_ATTEMPTS} attempts."
        )

    def _finish_on_pipeline_thread(self, error: Optional[str] = None) -> dict[str, Any]:
        """Close the writer, then detach session state while preserving summary."""
        writer = self._writer
        self._writer = None
        close_error: Optional[str] = None
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001 - contain optional teardown
                logger.exception("Could not close recording dataset")
                close_error = _file_error("Não foi possível fechar o arquivo de gravação.", exc)

        with self._lock:
            final_error = error or close_error
            summary = {
                "recording": False,
                "label": self._label,
                "session_id": self._session_id,
                "written": self._written,
                "skipped": self._skipped,
                "error": final_error,
            }
            self._recording = False
            self._label = None
            self._session_id = None
            self._owner = None
            self._written = 0
            self._skipped = 0
            self._stop_requested = False
            self._commands.clear()
            return summary

    def _snapshot_locked(self, error: Optional[str] = None) -> dict[str, Any]:
        if self._recording:
            return {
                "recording": True,
                "label": self._label,
                "session_id": self._session_id,
                "written": self._written,
                "skipped": self._skipped,
                "error": error,
            }
        return {
            "recording": False,
            "label": None,
            "session_id": None,
            "written": 0,
            "skipped": 0,
            "error": error,
        }
