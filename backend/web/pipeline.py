"""Stream Engine — one camera, one pipeline, many browsers.

The scripts of Sprints 01-04 owned the whole loop: open camera, read, detect,
draw, repeat. A web app cannot work that way. There is exactly one webcam, and
two open tabs must not fight over it — so the loop runs once, in a background
thread, and every connected client subscribes to its output.

Four decisions carry this file:

1. **The camera opens on the first client and closes after the last one.** Not
   at import, not at startup. The green LED is a privacy contract: it should be
   on only while somebody is actually watching. A short `idle_seconds` grace
   keeps a page reload — a disconnect immediately followed by a connect — from
   cycling the device.

2. **The pipeline runs in a thread, never on the event loop.** `camera.read()`
   and `detector.detect()` both block for tens of milliseconds. Running them on
   the loop would stall every WebSocket, and the app would deliver its worst
   latency exactly when inference is slowest.

3. **A slow client drops frames, it does not build a queue.** Each subscriber
   holds a single slot with the newest message. A browser that cannot keep up
   would otherwise accumulate a backlog and fall further behind every second —
   a live view that is thirty seconds late is worse than one that skipped
   thirty frames.

4. **Nobody watching means nothing inferred.** While no client is connected the
   loop keeps the device warm but skips pose entirely; there is no point
   spending CPU on landmarks nobody will see.

5. **A camera that stops delivering frames gets a bounded number of retries,
   not one shot.** `camera.frames()` raises `CameraError` once it decides the
   device is gone (Sprint 01's `max_read_failures`); a cable wiggled loose or
   macOS reclaiming the device for a second must not read the same as "camera
   permanently denied". `_run` reopens the camera with an exponential
   backoff, capped, and gives up only after `max_camera_retries` consecutive
   attempts produce zero frames — one attempt that does get a frame through
   resets the streak, so a device that flakes for a few seconds every hour is
   never one bad minute away from a permanent `failed`. Every retry is a
   non-fatal `error` message (`fatal=False`): the client's WebSocket stays
   open and the page can say "reconectando à câmera…" instead of the tab
   dropping and reconnecting from scratch.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from backend.camera import CameraConfig, CameraEngine, CameraOpenError, CameraReadError
from backend.dataset.writer import (
    DEFAULT_DATASET_PATH,
    DEFAULT_V2_DATASET_PATH,
    DEFAULT_LABELS,
    DatasetWriterError,
    V2_LABELS,
)
from backend.dataset.windows import build_window_input_row
from backend.features import (
    FEATURE_NAMES,
    FaceMotionTracker,
    MotionConfig,
    MotionTracker,
    Trail,
    extract,
)
from backend.features.face_features import extract_face
from backend.features.features import FrameFeatures
from backend.features.velocity import MotionState
from backend.features.window import WindowFeatures
from backend.gestures import GestureConfig, GestureEngine
from backend.ml import DEFAULT_MODEL_PATH, DEFAULT_V2_MODEL_PATH, ModelError
from backend.ml.rule_label import rule_label as compute_rule_label
from backend.ml import load_model as load_ml_model
from backend.ml import predict as ml_predict
from backend.vision import (
    DEFAULT_FACE_MODEL_PATH,
    DEFAULT_VISIBILITY_THRESHOLD,
    FaceConfig,
    FaceDetector,
    FaceModelError,
    PoseConfig,
    PoseDetector,
    PoseModelError,
    Side,
    ensure_face_model,
    ensure_model,
)
from backend.web.payload import (
    PreviewError,
    encode_preview,
    error_message,
    face_payload,
    frame_message,
    record_status_message,
    status_message,
)
from backend.web.recorder import INVALID_RECORD_LABEL, Recorder, validate_record_label

logger = logging.getLogger(__name__)
PROJECT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def default_report_path(model_path: Path) -> Path:
    return PROJECT_DATA_DIR / "training" / f"{model_path.stem}.report.html"

#: Wire names for the two arms, in the order the payload uses them.
SIDES: dict[str, Side] = {"left": Side.LEFT, "right": Side.RIGHT}


@dataclass(frozen=True)
class WebConfig:
    """Everything the web app tunes, in one object.

    The camera/pose/motion configs are the same ones the CLI scripts use — the
    web app is another front end for the same pipeline, not a second pipeline.
    """

    camera: CameraConfig = field(default_factory=CameraConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    #: Path to a trained Sprint 08 model. Optional on purpose: the whole
    #: pipeline (pose, motion, rule-based gestures) works with no model at
    #: all — Live ML is a signal riding on top, never a dependency.
    model_path: Optional[Path] = None
    #: Width of the JPEG sent to the browser. Inference still runs on the full
    #: frame, so this trades picture size against bandwidth, never accuracy.
    preview_width: int = 640
    jpeg_quality: int = 70
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD
    #: How long the camera stays open after the last client leaves. Long enough
    #: to survive a page reload, short enough that a closed tab frees the device.
    idle_seconds: float = 2.0
    #: Consecutive camera failures (open or mid-stream) tolerated before the
    #: runner gives up and reports `failed`. One attempt that gets at least
    #: one real frame through resets this streak.
    max_camera_retries: int = 3
    #: Backoff before the first retry, doubled each attempt up to the cap.
    retry_backoff_base: float = 1.0
    retry_backoff_cap: float = 8.0
    healthy_camera_frames: int = 30
    #: A run that serves >= `healthy_camera_frames` before failing again
    #: resets `attempt` to 0 (class docstring, point 5) -- proof of a real
    #: recovery, not just a lucky `open()`. Left unchecked that also means a
    #: device flapping just above that threshold forever would never reach
    #: `max_camera_retries` and would retry forever. `max_flaps_per_window`
    #: bounds that separately: however healthy each individual run looked,
    #: more than this many (re)opens inside `flap_window_seconds` is not "a
    #: device that flakes for a few seconds every hour" anymore, so the
    #: runner gives up instead of retrying forever.
    flap_window_seconds: float = 60.0
    max_flaps_per_window: int = 10
    #: Sprint 11 — facial signals. Off by default: it is a second MediaPipe
    #: inference per frame on top of pose (real CPU cost), and camera-adjacent
    #: capability the operator must opt into, never a surprise upgrade.
    face_enabled: bool = False
    face_model_path: Path = DEFAULT_FACE_MODEL_PATH
    #: Shared labeled dataset target. Tests and alternate deployments can
    #: redirect writes without changing the DatasetWriter schema.
    dataset_path: Optional[Path] = None
    #: Sprint 15 — last generated Sprint 10 HTML report. `None` follows the
    #: model path using the same `<model-stem>.report.html` convention as
    #: `scripts/report.py`; deployments can point the web app elsewhere.
    report_path: Optional[Path] = None
    #: Schema selected for live inference and recording labels. Appended to
    #: preserve the positional field order of the pre-S14 WebConfig API.
    model_schema: int = 1

    def __post_init__(self) -> None:
        if type(self.model_schema) is not int or self.model_schema not in (1, 2):
            raise ValueError("model_schema must be 1 or 2")
        model_default = DEFAULT_MODEL_PATH if self.model_schema == 1 else DEFAULT_V2_MODEL_PATH
        dataset_default = (
            DEFAULT_DATASET_PATH if self.model_schema == 1 else DEFAULT_V2_DATASET_PATH
        )
        object.__setattr__(
            self, "model_path", Path(self.model_path) if self.model_path is not None else model_default
        )
        object.__setattr__(
            self,
            "dataset_path",
            Path(self.dataset_path) if self.dataset_path is not None else dataset_default,
        )

    @property
    def record_labels(self) -> tuple[str, ...]:
        """Labels offered by the server for the selected recording schema."""
        return DEFAULT_LABELS if self.model_schema == 1 else V2_LABELS


class Subscriber:
    """One browser's slot: always the newest message, never a backlog.

    `offer` runs on the event loop thread; `offer_threadsafe` hands a message
    over from the pipeline thread, because `asyncio.Queue` is not thread-safe.
    """

    MAX_CONTROL_MESSAGES = 64

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        # Frames may be coalesced to keep latency bounded. Control messages
        # carry state transitions and must remain queued until the client sees
        # them, even if a new frame arrives in the meantime.
        self._latest_frame: Optional[dict[str, Any]] = None
        self._control: deque[dict[str, Any]] = deque()
        # Start/stop state changes are boundaries, not disposable count
        # updates. Keep their identities so queue pressure evicts rejections
        # and replaceable controls first.
        self._transition_ids: set[int] = set()
        self._last_record_state: Optional[tuple[bool, Optional[str]]] = None
        self._overloaded = False
        self._available = asyncio.Event()
        self.dropped = 0
        self.sent = 0

    def offer(self, message: dict[str, Any]) -> None:
        if self._overloaded:
            return
        if message.get("type") == "frame":
            if self._latest_frame is not None:
                self.dropped += 1
            self._latest_frame = message
        else:
            if self._coalesce_record_status(message):
                self._available.set()
                return
            transition = self._is_record_transition(message)
            if message.get("type") == "record_status" and (
                message.get("error") is None or self._is_final_record_status(message)
            ):
                self._last_record_state = (
                    bool(message.get("recording")), message.get("session_id")
                )
            self._append_control(message, preserve=transition)
        self._available.set()

    def _coalesce_record_status(self, message: dict[str, Any]) -> bool:
        """Replace duplicate rejections/count updates without hiding transitions."""
        if message.get("type") != "record_status":
            return False

        if message.get("error") is not None and not self._is_final_record_status(message):
            rejection_key = (
                message.get("error"),
                bool(message.get("recording")),
                message.get("label"),
                message.get("session_id"),
                bool(message.get("owner")),
            )
            for index, pending in enumerate(self._control):
                if pending.get("type") != "record_status" or pending.get("error") is None:
                    continue
                pending_key = (
                    pending.get("error"),
                    bool(pending.get("recording")),
                    pending.get("label"),
                    pending.get("session_id"),
                    bool(pending.get("owner")),
                )
                if pending_key == rejection_key:
                    self._control[index] = message
                    return True
            return False

        state = (bool(message.get("recording")), message.get("session_id"))
        is_transition = state != self._last_record_state
        if (
            not is_transition
            and self._control
            and self._control[-1].get("type") == "record_status"
            and self._control[-1].get("error") is None
            and (
                bool(self._control[-1].get("recording")),
                self._control[-1].get("session_id"),
            )
            == state
            and id(self._control[-1]) not in self._transition_ids
        ):
            self._control[-1] = message
            return True
        return False

    def _is_record_transition(self, message: dict[str, Any]) -> bool:
        if message.get("type") != "record_status":
            return False
        if message.get("error") is not None and not self._is_final_record_status(message):
            return False
        state = (bool(message.get("recording")), message.get("session_id"))
        return state != self._last_record_state

    def _append_control(self, message: dict[str, Any], *, preserve: bool = False) -> None:
        """Append within the bound, evicting rejections before state changes."""
        if len(self._control) >= self.MAX_CONTROL_MESSAGES:
            removable = self._find_removable_control()
            if removable is None:
                # The queue contains only protected transitions/finals/fatal
                # controls. Silently dropping the new state here would let a
                # slow socket drain stale statuses and never learn the real
                # state. Close it so the client reconnects and receives a
                # fresh snapshot (and fatal pipeline error replay if needed).
                self._close_for_resnapshot()
                return
            old = self._control[removable]
            self._transition_ids.discard(id(old))
            del self._control[removable]
        self._control.append(message)
        if preserve:
            self._transition_ids.add(id(message))

    def _find_removable_control(self) -> Optional[int]:
        """Choose the least important queued item without dropping finals."""
        for index, pending in enumerate(self._control):
            if (
                pending.get("type") == "record_status"
                and pending.get("error") is not None
                and not self._is_final_record_status(pending)
            ):
                return index
        for index, pending in enumerate(self._control):
            if (
                id(pending) not in self._transition_ids
                and not self._is_final_record_status(pending)
                and not self._is_fatal_control(pending)
            ):
                return index
        return None

    def _close_for_resnapshot(self) -> None:
        self._overloaded = True
        self._control.clear()
        self._transition_ids.clear()
        self._available.set()

    @staticmethod
    def _is_fatal_control(message: dict[str, Any]) -> bool:
        return message.get("type") == "error" and message.get("fatal", True) is True

    @staticmethod
    def _is_final_record_status(message: dict[str, Any]) -> bool:
        return (
            message.get("type") == "record_status"
            and message.get("recording") is False
            and message.get("session_id") is not None
        )

    def offer_threadsafe(self, message: dict[str, Any]) -> None:
        try:
            self._loop.call_soon_threadsafe(self.offer, message)
        except RuntimeError:
            # The loop is already closed: that client is gone, and its WebSocket
            # handler unsubscribes on its way out.
            pass

    async def get(self) -> dict[str, Any]:
        while True:
            if self._overloaded:
                return {"type": "subscriber_overloaded"}
            if self._control:
                message = self._control.popleft()
                self._transition_ids.discard(id(message))
                return message
            if self._latest_frame is not None:
                message = dict(self._latest_frame)
                self._latest_frame = None
                first_frame = self.sent == 0
                self.sent += 1
                health = dict(message.get("health") or {})
                if first_frame:
                    # Establish the per-connection baseline at the first
                    # frame even if startup coalesced before the writer ran.
                    health["dropped_frames"] = 0
                    health["dropped_rate"] = 0.0
                else:
                    health["dropped_frames"] = self.dropped
                    total = self.sent + self.dropped
                    health["dropped_rate"] = round(self.dropped / total, 2)
                message["health"] = health
                return message
            self._available.clear()
            await self._available.wait()


def _default_camera(config: WebConfig) -> CameraEngine:
    return CameraEngine(config=config.camera)


def _default_detector(config: WebConfig) -> PoseDetector:
    """Download the model if needed, then build the detector.

    `ensure_model` runs here — on the pipeline thread, when the first client
    arrives — so a first-run download never blocks import or server startup.
    """
    model_path = ensure_model(config.pose.model_path)
    return PoseDetector(config=replace(config.pose, model_path=model_path))


def _default_face_detector(config: WebConfig) -> FaceDetector:
    """Same lazy-download-then-build shape as `_default_detector`, for the
    face model. Only ever called when `config.face_enabled` is True."""
    model_path = ensure_face_model(config.face_model_path)
    return FaceDetector(config=FaceConfig(model_path=model_path))


class PipelineRunner:
    """Owns the camera loop and fans its output out to subscribers.

    The factories are injectable for the same reason `CameraEngine` takes a
    `capture_factory`: the tests drive the whole web app with a fake camera and
    a fake landmarker — no hardware, no model file.
    """

    def __init__(
        self,
        config: Optional[WebConfig] = None,
        camera_factory: Callable[[WebConfig], CameraEngine] = _default_camera,
        detector_factory: Callable[[WebConfig], PoseDetector] = _default_detector,
        face_detector_factory: Callable[[WebConfig], FaceDetector] = _default_face_detector,
    ) -> None:
        self.config = config or WebConfig()
        self.camera_factory = camera_factory
        self.detector_factory = detector_factory
        self.face_detector_factory = face_detector_factory
        self.recorder = Recorder(
            self.config.dataset_path,
            allowed_labels=self.config.record_labels,
        )

        self.state = "idle"  # idle | starting | live | failed
        self.error: Optional[str] = None
        #: Independent of `self.error`: the face signal is optional (Sprint
        #: 11) and can fail on its own (missing model, failed download, a
        #: runtime error mid-session) without taking pose/web down with it —
        #: see `_stream`'s face block and `backend/web/payload.py:face_payload`.
        self.face_error: Optional[str] = None
        self.frames_served = 0

        self._subscribers: list[Subscriber] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # The active session ID is the disconnect signal. A bare Event can be
        # set for session B and then cleared by completion of an older stop for
        # session A, leaving B's writer open after its owner disconnects.
        self._owner_disconnected_session: Optional[str] = None
        # A start request may reserve a session only while a pipeline thread
        # can consume it. This gate and its status delivery share `_lock` so a
        # start racing shutdown is either ordered before the final status or
        # rejected with the recorder's actual current snapshot.
        self._accept_record_starts = False

    # --- model introspection --------------------------------------------
    def ml_classes(self) -> list[str]:
        """Class vocabulary of the trained model, or `[]` if there is none.

        Called from the WebSocket handshake (`backend/web/app.py`), before
        any frame has flowed and possibly before the camera thread has even
        started — so it loads independently of `_stream`'s own bundle rather
        than reading a cached value that might not exist yet.
        """
        try:
            bundle = load_ml_model(
                self.config.model_path,
                schema=self.config.model_schema,
            )
            return list(bundle["classes"])
        except Exception:  # noqa: BLE001 - optional live ML
            logger.debug("ML model unavailable during WebSocket handshake", exc_info=True)
            return []

    @property
    def report_path(self) -> Path:
        return self.config.report_path or default_report_path(self.config.model_path)

    def safe_report_path(self) -> Optional[Path]:
        """Resolve a regular report within data, rejecting symlinks."""
        candidate = self.report_path.absolute()
        try:
            resolved = candidate.resolve()
            if resolved != candidate or not resolved.is_relative_to(PROJECT_DATA_DIR):
                logger.warning("Rejected report path outside the allowed location: %s", candidate)
                return None
            return resolved if resolved.is_file() else None
        except (OSError, RuntimeError):
            logger.exception("Could not resolve saved report")
            return None

    def report_available(self) -> bool:
        """Whether the report can be served over HTTP."""
        return self.safe_report_path() is not None

    # --- subscriptions -------------------------------------------------
    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> Subscriber:
        """Register a client and make sure the camera is running.

        A subscriber that joins after the pipeline has already given up
        (`state == "failed"`, a fatal `CameraOpenError`/`PoseModelError`/etc.)
        is a real race, not an edge case: the dying thread already broadcast
        that error to every subscriber that existed at the time, and
        `_ensure_running` correctly refuses to start a second thread while
        the first is still alive/tearing down. Without the replay below, this
        late subscriber would sit on an empty queue forever — no frame is
        ever coming, and no fresh attempt was made on their behalf either.
        """
        subscriber = Subscriber(loop)
        with self._lock:
            self._subscribers.append(subscriber)
        started = self._ensure_running()
        if not started and self.state == "failed" and self.error is not None:
            subscriber.offer(error_message(self.error))
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
            # The reader task and writer task can finish in either order;
            # tying ownership to the subscription guarantees a closed tab
            # cannot leave an open CSV session behind.
            if self.recorder.owner_disconnected(subscriber):
                self._owner_disconnected_session = self.recorder.status()["session_id"]

    def _clear_owner_disconnect_for(self, event: Optional[dict[str, Any]]) -> None:
        """Clear only the disconnect signal belonging to this finished session."""
        if event is None or event.get("session_id") is None:
            return
        with self._lock:
            if self._owner_disconnected_session == event["session_id"]:
                self._owner_disconnected_session = None

    def _process_owner_disconnect(self) -> None:
        """Finish a disconnected owner's session if its reservation is current."""
        with self._lock:
            disconnected_session = self._owner_disconnected_session
        if disconnected_session is None:
            return

        snapshot = self.recorder.status()
        if snapshot["recording"] and snapshot["session_id"] == disconnected_session:
            self._close_recorder(resume_record_starts=True)

        with self._lock:
            if self._owner_disconnected_session == disconnected_session:
                self._owner_disconnected_session = None

    def record_start(self, label: object, subscriber: Subscriber) -> dict[str, Any]:
        """Reserve a session only while a worker can consume its command."""
        with self._lock:
            try:
                label = validate_record_label(label, self.config.record_labels)
            except DatasetWriterError:
                return self._reject_record_start(subscriber, INVALID_RECORD_LABEL)

            if not self._accept_record_starts or self.state in ("recovering", "failed"):
                reason = self.error or "A câmera ainda não está disponível."
                return self._reject_record_start(
                    subscriber, f"Não foi possível iniciar a gravação. {reason}"
                )

            # request_start is validation/reservation only; file I/O remains
            # on the pipeline thread. Offer the accepted status under the
            # same lock so a concurrent finish cannot overtake it.
            event = self.recorder.request_start(label, subscriber)
            if event.get("error") is None:
                self._publish_record_event_locked(event)
            else:
                self._offer_record_status(subscriber, event)
            return event

    def _reject_record_start(self, subscriber: Subscriber, error: str) -> dict[str, Any]:
        snapshot = self.recorder.status_for(subscriber, error=error)
        owner = snapshot.pop("owner")
        subscriber.offer_threadsafe(record_status_message(**snapshot, owner=owner))
        return snapshot

    def record_stop(self, subscriber: Subscriber) -> Optional[dict[str, Any]]:
        """Queue a stop request after the recorder checks tab ownership."""
        with self._lock:
            event = self.recorder.request_stop(subscriber)
            if event is not None:
                self._offer_record_status(subscriber, event)
            return event

    def recording_snapshot_for(self, subscriber: Subscriber) -> dict[str, Any]:
        """Current recorder state with ownership scoped to one browser tab."""
        return record_status_message(**self.recorder.status_for(subscriber))

    def recording_health(self) -> dict[str, Any]:
        """Current recorder state without leaking the in-process owner token."""
        snapshot = self.recorder.status()
        return {
            key: snapshot[key]
            for key in ("recording", "label", "session_id", "written", "skipped")
        }

    def _record_wire_message(
        self, snapshot: dict[str, Any], subscriber: Subscriber
    ) -> dict[str, Any]:
        # Ownership is meaningful only while the recording remains active.
        owner = bool(snapshot["recording"] and self.recorder.is_owner(subscriber))
        return record_status_message(
            recording=snapshot["recording"],
            label=snapshot["label"],
            session_id=snapshot["session_id"],
            written=snapshot["written"],
            skipped=snapshot["skipped"],
            owner=owner,
            error=snapshot["error"],
        )

    def _offer_record_status(self, subscriber: Subscriber, event: dict[str, Any]) -> None:
        subscriber.offer_threadsafe(self._record_wire_message(event, subscriber))

    def _publish_record_events(self, event: Optional[dict[str, Any]]) -> None:
        if event is None:
            return
        with self._lock:
            self._publish_record_event_locked(event)
        if event["error"] is not None:
            self._broadcast(error_message(event["error"], fatal=False))

    def _publish_record_event_locked(self, event: dict[str, Any]) -> None:
        """Queue one shared status while the subscriber set is stable."""
        for subscriber in self._subscribers:
            subscriber.offer_threadsafe(self._record_wire_message(event, subscriber))

    def _publish_record_counts_if_due(self) -> None:
        snapshot = self.recorder.status_if_due()
        if snapshot is None:
            return
        self._broadcast_record_snapshot(snapshot)

    def _broadcast_record_snapshot(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.offer_threadsafe(self._record_wire_message(snapshot, subscriber))

    def _close_recorder(
        self, error: Optional[str] = None, *, resume_record_starts: bool = False
    ) -> None:
        # Release the lifecycle lock during close(): it can flush/close a CSV
        # and must never make an asyncio handler wait behind file I/O. Starts
        # still see the closed gate and receive the live Recorder snapshot.
        with self._lock:
            self._accept_record_starts = False
        event = self.recorder.finish(error=error)
        with self._lock:
            if event is not None:
                self._publish_record_event_locked(event)
            if resume_record_starts and self.state == "live" and not self._stop.is_set():
                self._accept_record_starts = True
        if error is None and event is not None and event["error"] is not None:
            self._broadcast(error_message(event["error"], fatal=False))

    # --- lifecycle -----------------------------------------------------
    def _ensure_running(self) -> bool:
        """Start the pipeline thread if it is not already running.

        Returns whether this call actually (re)started it -- `subscribe`
        uses that to tell "a fresh attempt is now underway" from "the
        existing thread, alive or already dead, owns this subscriber's fate".
        `self.error` is cleared here, synchronously, rather than left for the
        new thread's own first line (`_run`'s `self.error = None`): between
        `self._thread.start()` and that line actually executing, `/health`
        could otherwise report `state: "starting"` next to the *previous*
        run's stale error message.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._accept_record_starts = False
            self.state = "starting"
            self.error = None
            self._thread = threading.Thread(
                target=self._run, name="motion-lab-pipeline", daemon=True
            )
            self._thread.start()
            # The worker now exists and can either consume queued requests or
            # finalize them if device/model startup fails.
            self._accept_record_starts = True
            return True

    def stop(self, timeout: float = 3.0) -> None:
        """Ask the loop to finish, and wait for the device to be released."""
        with self._lock:
            self._accept_record_starts = False
            self._stop.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _broadcast(self, message: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.offer_threadsafe(message)

    def _disable_face(self, detector: Optional[FaceDetector], message: str) -> str:
        """Keep the public reason about face tracking and log the underlying failure."""
        logger.exception("Face tracking unavailable")
        self.face_error = message
        self._broadcast(error_message(message, fatal=False))
        if detector is not None:
            try:
                detector.close()
            except Exception:  # noqa: BLE001 - optional teardown
                logger.exception("Could not close face detector")
        return message

    def _run(self) -> None:
        """Thread body: open the devices, stream, and always release them.

        A camera read failure gets bounded retries with exponential backoff.
        Camera open and pose model failures are fatal.
        """
        self.error = None
        attempt = 0
        #: Every (re)open attempt's timestamp, regardless of how healthy the
        #: run in between looked -- `attempt` alone resets on a healthy
        #: streak (see below) and would let a device flapping just above
        #: `healthy_camera_frames` retry forever. This is the independent,
        #: never-reset backstop for that.
        flap_times: list[float] = []
        while True:
            served_before = self.frames_served
            try:
                camera = self.camera_factory(self.config)
                detector = self.detector_factory(self.config)
                with camera, detector:
                    with self._lock:
                        self.state = "live"
                        self.error = None
                        self._accept_record_starts = not self._stop.is_set()
                    self._stream(camera, detector)
                self._close_recorder()
                with self._lock:
                    self.state = "idle"
                break
            except CameraOpenError:
                logger.exception("Could not open camera")
                with self._lock:
                    self._accept_record_starts = False
                    self.state = "failed"
                    self.error = (
                        f"Não foi possível abrir a câmera (índice {self.config.camera.index}). "
                        "Verifique se ela está conectada, livre e se o aplicativo tem "
                        "permissão de acesso à câmera. No macOS, confira em Ajustes do Sistema "
                        "> Privacidade e Segurança > Câmera e reinicie o aplicativo."
                    )
                self._close_recorder(error=self.error)
                self._broadcast(error_message(self.error))
                break
            except CameraReadError:
                logger.warning("Camera stopped delivering frames", exc_info=True)
                with self._lock:
                    self._accept_record_starts = False
                # At least one real frame got through before it died: that is
                # proof of an actual recovery, not just a lucky `open()` — the
                # streak resets, so a device that flakes for a few seconds
                # every hour is never one bad minute away from `failed`.
                if self.frames_served - served_before >= self.config.healthy_camera_frames:
                    attempt = 0
                attempt += 1
                now = time.monotonic()
                flap_times.append(now)
                cutoff = now - self.config.flap_window_seconds
                while flap_times and flap_times[0] < cutoff:
                    flap_times.pop(0)
                flapping = len(flap_times) > self.config.max_flaps_per_window
                message = "A câmera parou de enviar imagens."
                if self._stop.is_set() or attempt > self.config.max_camera_retries or flapping:
                    if flapping:
                        message += (
                            f" Houve {len(flap_times)} falhas em "
                            f"{self.config.flap_window_seconds:.0f} s."
                        )
                    message += " As tentativas de reconexão foram encerradas."
                    with self._lock:
                        self.state = "failed"
                        self.error = message
                    self._close_recorder(error=message)
                    self._broadcast(error_message(message))
                    break
                with self._lock:
                    self.state = "recovering"
                    self.error = message
                self._close_recorder(error=message)
                self._broadcast(error_message(message, fatal=False))
                self._broadcast(status_message("recovering", message))
                delay = min(
                    self.config.retry_backoff_base * (2 ** (attempt - 1)),
                    self.config.retry_backoff_cap,
                )
                if self._stop.wait(timeout=delay):
                    with self._lock:
                        self.state = "idle"
                    break
            except Exception as exc:  # noqa: BLE001 - surface fatal pipeline failures
                logger.exception("Camera pipeline stopped")
                if isinstance(exc, PoseModelError):
                    message = (
                        "Não foi possível baixar o modelo de pose. Verifique a conexão com a internet."
                        if exc.__cause__ is not None
                        else "O modelo de pose está indisponível. Verifique se ele foi instalado."
                    )
                elif isinstance(exc, PreviewError):
                    message = str(exc)
                else:
                    message = "O processamento da câmera foi interrompido por uma falha interna."
                with self._lock:
                    self._accept_record_starts = False
                    self.state = "failed"
                    self.error = message
                self._close_recorder(error=message)
                self._broadcast(error_message(message))
                break

        with self._lock:
            self._accept_record_starts = False
            self._thread = None
            # A client can connect during teardown, between the loop's last
            # idle check and this line; without the restart it would wait
            # forever on a thread that just died.
            restart = (
                self.state == "idle" and bool(self._subscribers) and not self._stop.is_set()
            )
        if restart:
            self._ensure_running()

    def _stream(self, camera: CameraEngine, detector: PoseDetector) -> None:
        tracker = MotionTracker(self.config.motion)
        gestures = GestureEngine(self.config.gesture)
        trails = {name: Trail(self.config.motion.trail_length) for name in SIDES}
        idle_since: Optional[float] = None
        # Loaded once per stream start (mirrors PoseDetector's own lifecycle)
        # so training a model mid-session takes effect on the next reconnect
        # without a server restart. Missing or corrupt is not fatal — Live ML
        # is a signal on top of the pipeline, never a dependency of it.
        ml_bundle = None
        try:
            if self.config.model_path.exists():
                ml_bundle = load_ml_model(
                    self.config.model_path,
                    schema=self.config.model_schema,
                )
        except Exception:  # noqa: BLE001 - optional live ML
            logger.exception("Could not load model for live inference")
            self._broadcast(
                error_message(
                    "Não foi possível carregar o modelo de classificação ao vivo.", fatal=False
                )
            )

        window_tracker: Optional[WindowFeatures] = None
        if self.config.model_schema == 2 and ml_bundle is not None:
            try:
                window_tracker = WindowFeatures(
                    window_seconds=ml_bundle["window_seconds"],
                    max_gap=ml_bundle["max_gap"],
                    min_samples=ml_bundle["min_samples"],
                    deadband=ml_bundle["deadband"],
                )
            except (KeyError, TypeError, ValueError):
                # Loading the optional model must never make pose, gestures,
                # recording, or the web stream unavailable.
                ml_bundle = None

        # Sprint 11 — facial signals, same isolation rule as the ML bundle
        # above: opt-in (`face_enabled`) and, once on, a missing/broken model
        # degrades to "unavailable" rather than taking pose/gestures/ML/web
        # down with it. Built once per stream start, reusing the *same*
        # `Frame` the pose detector reads each iteration below — no second
        # camera open, no second capture.
        face_detector: Optional[FaceDetector] = None
        face_error: Optional[str] = None
        face_tracker = FaceMotionTracker()
        if self.config.face_enabled:
            try:
                face_detector = self.face_detector_factory(self.config)
                face_detector.open()
            except Exception as exc:  # noqa: BLE001 - optional subsystem boundary
                message = "Não foi possível iniciar o rastreamento facial."
                if isinstance(exc, FaceModelError):
                    message = (
                        "Não foi possível baixar o modelo facial. Verifique a conexão com a internet."
                        if exc.__cause__ is not None
                        else "O modelo facial está indisponível. Verifique se ele foi instalado."
                    )
                face_error = self._disable_face(face_detector, message)
                face_detector = None
        self.face_error = face_error

        def timed_frames():
            """Measure the camera generator's read operation per frame."""
            frames = iter(camera.frames())
            while True:
                capture_started = time.perf_counter()
                try:
                    captured = next(frames)
                except StopIteration:
                    return
                capture_ms = (time.perf_counter() - capture_started) * 1000
                yield captured, capture_started, capture_ms

        try:
            for frame, capture_started, capture_ms in timed_frames():
                if self._stop.is_set():
                    return

                if self.client_count == 0:
                    # Commands still need servicing during the camera's idle
                    # grace period, especially the owner's disconnect stop.
                    self._process_owner_disconnect()
                    if idle_since is None:
                        # First frame with nobody watching: drop the temporal state.
                        # Whatever the arms did while the tab was closed is not part
                        # of the next gesture, and a stale trail would be drawn as if
                        # it were.
                        idle_since = time.monotonic()
                        tracker.reset()
                        gestures.reset()
                        face_tracker.reset()
                        if window_tracker is not None:
                            window_tracker.reset()
                        for trail in trails.values():
                            trail.clear()
                    if time.monotonic() - idle_since >= self.config.idle_seconds:
                        return  # release the camera; the next client restarts us
                    continue  # keep the device warm, but spend no CPU on pose

                idle_since = None
                # `PoseSnapshot.inference_ms` is timed by PoseDetector with
                # perf_counter around the landmarker call itself.
                snapshot = detector.detect(frame)
                pose_ms = snapshot.inference_ms

                features_started = time.perf_counter()
                features = extract(snapshot, self.config.visibility_threshold)
                motion = tracker.update(features)
                fired = gestures.update(features, motion)
                features_ms = (time.perf_counter() - features_started) * 1000

                window_state = None
                if window_tracker is not None:
                    if snapshot.detected:
                        window_state = window_tracker.push(features.timestamp, features)
                    else:
                        # A missing pose breaks temporal continuity. Do not
                        # carry pre-loss movement into the first re-detected frame.
                        window_tracker.reset()

                ml_started = time.perf_counter()
                try:
                    prediction = self._predict(
                        ml_bundle,
                        features,
                        motion,
                        window_state=window_state,
                    )
                except Exception:  # noqa: BLE001 - optional live ML
                    logger.exception("Disabling live ML after inference failed")
                    self._broadcast(
                        error_message(
                            "A classificação ao vivo foi desativada após uma falha de inferência.",
                            fatal=False,
                        )
                    )
                    ml_bundle = None
                    window_tracker = None
                    prediction = None
                ml_ms = (
                    (time.perf_counter() - ml_started) * 1000
                    if prediction is not None
                    else None
                )
                for name, trail in trails.items():
                    trail.add_from(snapshot, SIDES[name], self.config.visibility_threshold)

                face_dict = None
                face_ms = None
                if self.config.face_enabled:
                    if face_detector is None:
                        face_dict = face_payload(enabled=True, error=face_error)
                    else:
                        face_started = time.perf_counter()
                        try:
                            face_snapshot = face_detector.detect(frame)
                            face_signals = extract_face(face_snapshot)
                            face_motion = face_tracker.update(face_signals)
                            face_dict = face_payload(
                                enabled=True,
                                error=None,
                                detected=face_snapshot.detected,
                                signals=face_signals,
                                motion=face_motion,
                                mesh=face_snapshot.mesh,
                            )
                        except Exception:  # noqa: BLE001 - optional layer,
                            # must never sink pose/web (requirement 4): one bad
                            # frame disables face for the rest of THIS session
                            # rather than crashing the whole pipeline thread —
                            # pose, motion, gestures, ML keep running untouched.
                            face_error = self._disable_face(
                                face_detector, "Falha ao analisar o rosto. O rastreamento facial foi desativado."
                            )
                            face_detector = None
                            face_dict = face_payload(enabled=True, error=face_error)
                        finally:
                            face_ms = (time.perf_counter() - face_started) * 1000

                # The CSV writer runs here, on the pipeline thread, with the
                # exact feature and motion objects derived from this frame.
                record_event = self.recorder.record_frame(features, motion)
                self._publish_record_events(record_event)
                self._clear_owner_disconnect_for(record_event)
                self._publish_record_counts_if_due()

                timings: dict[str, Optional[float]] = {
                    "capture_ms": capture_ms,
                    "pose_ms": pose_ms,
                    "features_ms": features_ms,
                    "ml_ms": ml_ms,
                    "face_ms": face_ms,
                    # Filled immediately after the payload is constructed.
                    "total_ms": None,
                }
                message = frame_message(
                    snapshot=snapshot,
                    features=features,
                    motion=motion,
                    preview=encode_preview(
                        frame.image,  # raw, unmirrored: the browser flips at draw time
                        width=self.config.preview_width,
                        quality=self.config.jpeg_quality,
                    ),
                    fps=camera.fps,
                    trails=trails,
                    gestures=fired,
                    ml=prediction,
                    face=face_dict,
                    rule_label=(
                        compute_rule_label(
                            features.to_dict(), self.config.gesture.open_threshold
                        )
                        if snapshot.detected
                        else None
                    ),
                    timings=timings,
                )

                # Includes the camera read and all work through payload
                # construction. Ensure rounded stage values never sum beyond
                # the rounded total due to one-decimal rounding.
                stage_total_ms = sum(
                    round(value, 1)
                    for name, value in timings.items()
                    if name != "total_ms" and value is not None
                )
                elapsed_ms = (time.perf_counter() - capture_started) * 1000
                message["timings"]["total_ms"] = round(
                    max(elapsed_ms, stage_total_ms), 1
                )
                self._broadcast(message)
                self.frames_served += 1
        finally:
            if face_detector is not None:
                try:
                    face_detector.close()
                except Exception:  # noqa: BLE001 - optional teardown
                    self._disable_face(None, "Não foi possível encerrar o rastreamento facial.")

    @staticmethod
    def _predict(
        bundle: Optional[dict],
        features: FrameFeatures,
        motion: MotionState,
        window_state=None,
    ) -> Optional[tuple[str, dict[str, float]]]:
        """Skip, don't guess: an incomplete frame never reaches the model.

        Same rule `DatasetWriter` applies at training time (Sprint 07) — a
        fabricated 0 for an occluded joint reads as a real, confident value
        to the Forest (an unseen elbow at "0 degrees" looks fully bent), so a
        frame that would have been rejected as a training row is rejected as
        an inference input too.

        Sprint 11: `backend/ml/train.py`'s current `TRAINING_COLUMNS` is
        `FEATURE_NAMES` alone — `DEFAULT_LABELS` is purely postural, so a
        bundle trained today never asks for a motion column, and gating on
        `motion.complete` would only delay inference until `MotionTracker`
        warms up for no reason. A pre-Sprint-11 bundle can still list a
        motion column in `bundle["columns"]` (`load_model` keeps those
        loadable, see `KNOWN_TRAINING_COLUMNS`), and *that* bundle's vector
        was fit with real velocities, so a `None` there is exactly the same
        fabricated-zero risk as an occluded joint — this frame must still be
        skipped rather than fed a stale/missing value or crashing on
        `build_vector`'s `MissingFeatureError`. The decision is read off the
        bundle's own `columns`, never assumed globally, so a legacy model on
        disk keeps behaving exactly as it did before this sprint while the
        current one predicts on the very first complete frame.
        """
        if bundle is None or not features.complete:
            return None
        if bundle.get("schema_version", 1) == 2:
            row = build_window_input_row(features, window_state)
            if row is None:
                return None
            return ml_predict(bundle, row)
        columns = bundle.get("columns") or ()
        needs_motion = any(name not in FEATURE_NAMES for name in columns)
        if needs_motion and not motion.complete:
            return None
        row = {**features.to_dict(), **motion.to_dict()}
        return ml_predict(bundle, row)
