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
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from backend.camera import CameraConfig, CameraEngine, CameraOpenError, CameraReadError
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
from backend.gestures import GestureConfig, GestureEngine
from backend.ml import DEFAULT_MODEL_PATH, ModelError
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
    encode_preview,
    error_message,
    face_payload,
    frame_message,
    status_message,
)

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
    model_path: Path = DEFAULT_MODEL_PATH
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


class Subscriber:
    """One browser's slot: always the newest message, never a backlog.

    `offer` runs on the event loop thread; `offer_threadsafe` hands a message
    over from the pipeline thread, because `asyncio.Queue` is not thread-safe.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        self.dropped = 0

    def offer(self, message: dict[str, Any]) -> None:
        if self._queue.full():
            self._queue.get_nowait()  # the stale frame is worthless, discard it
            self.dropped += 1
        self._queue.put_nowait(message)

    def offer_threadsafe(self, message: dict[str, Any]) -> None:
        try:
            self._loop.call_soon_threadsafe(self.offer, message)
        except RuntimeError:
            # The loop is already closed: that client is gone, and its WebSocket
            # handler unsubscribes on its way out.
            pass

    async def get(self) -> dict[str, Any]:
        return await self._queue.get()


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

    # --- model introspection --------------------------------------------
    def ml_classes(self) -> list[str]:
        """Class vocabulary of the trained model, or `[]` if there is none.

        Called from the WebSocket handshake (`backend/web/app.py`), before
        any frame has flowed and possibly before the camera thread has even
        started — so it loads independently of `_stream`'s own bundle rather
        than reading a cached value that might not exist yet.
        """
        try:
            return list(load_ml_model(self.config.model_path)["classes"])
        except ModelError:
            return []

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
            self.state = "starting"
            self.error = None
            self._thread = threading.Thread(
                target=self._run, name="motion-lab-pipeline", daemon=True
            )
            self._thread.start()
            return True

    def stop(self, timeout: float = 3.0) -> None:
        """Ask the loop to finish, and wait for the device to be released."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _broadcast(self, message: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.offer_threadsafe(message)

    def _disable_face(self, detector: Optional[FaceDetector], exc: Exception) -> str:
        """Contain every optional-face lifecycle failure at its boundary.

        `FaceModelError` already carries a clean, explainable message (missing
        file, failed download) -- the same rule `_run` applies to
        `PoseModelError`/`CameraError`. Only an *unexpected* exception (a
        MediaPipe runtime hiccup, `close()` blowing up) gets its type name
        prefixed, since there `str(exc)` alone (e.g. `"boom"`) would not be
        enough context to tell a client-visible bug apart from an unrelated
        message that happens to read the same.
        """

        def _describe(inner: Exception) -> str:
            if isinstance(inner, FaceModelError):
                return str(inner)
            return f"{type(inner).__name__}: {inner}"

        error = _describe(exc)
        self.face_error = error
        if detector is not None:
            try:
                detector.close()
            except Exception as close_exc:  # noqa: BLE001 - optional teardown
                error = f"{error}; close failed: {_describe(close_exc)}"
                self.face_error = error
        return error

    def _run(self) -> None:
        """Thread body: open the devices, stream, and always release them.

        Wrapped in a bounded retry loop (class docstring, point 5): a
        `CameraError` — whether it is the very first `open()` or the camera
        going silent mid-stream — gets `max_camera_retries` attempts with an
        exponential backoff before the runner gives up. A `PoseModelError` or
        any other exception is not retried: the pose model file is not going
        to appear on its own between one attempt and the next.
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
                    self.state = "live"
                    self.error = None
                    self._stream(camera, detector)
                self.state = "idle"
                break
            except CameraOpenError as exc:
                self.state = "failed"
                self.error = str(exc)
                self._broadcast(error_message(self.error))
                break
            except CameraReadError as exc:
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
                self.error = str(exc)
                if self._stop.is_set() or attempt > self.config.max_camera_retries or flapping:
                    self.state = "failed"
                    if flapping:
                        self.error = (
                            f"{self.error} (camera failed {len(flap_times)} times in "
                            f"{self.config.flap_window_seconds:.0f}s — giving up instead "
                            "of retrying forever)"
                        )
                    self._broadcast(error_message(self.error))
                    break
                self.state = "recovering"
                self._broadcast(status_message("recovering", self.error))
                delay = min(
                    self.config.retry_backoff_base * (2 ** (attempt - 1)),
                    self.config.retry_backoff_cap,
                )
                if self._stop.wait(timeout=delay):
                    self.state = "idle"
                    break
            except PoseModelError as exc:
                # An expected, explainable failure: the pose model is missing
                # or unusable. The user has to see it, so it goes to every
                # client — and it will not fix itself on a retry.
                self.state = "failed"
                self.error = str(exc)
                self._broadcast(error_message(self.error))
                break
            except Exception as exc:  # noqa: BLE001 - a dead thread must not fail silently
                self.state = "failed"
                self.error = f"{type(exc).__name__}: {exc}"
                self._broadcast(error_message(self.error))
                break

        with self._lock:
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
        try:
            ml_bundle = load_ml_model(self.config.model_path)
        except ModelError:
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
                face_error = self._disable_face(face_detector, exc)
                face_detector = None
        self.face_error = face_error

        try:
            for frame in camera.frames():
                if self._stop.is_set():
                    return

                if self.client_count == 0:
                    if idle_since is None:
                        # First frame with nobody watching: drop the temporal state.
                        # Whatever the arms did while the tab was closed is not part
                        # of the next gesture, and a stale trail would be drawn as if
                        # it were.
                        idle_since = time.monotonic()
                        tracker.reset()
                        gestures.reset()
                        face_tracker.reset()
                        for trail in trails.values():
                            trail.clear()
                    if time.monotonic() - idle_since >= self.config.idle_seconds:
                        return  # release the camera; the next client restarts us
                    continue  # keep the device warm, but spend no CPU on pose

                idle_since = None
                snapshot = detector.detect(frame)
                features = extract(snapshot, self.config.visibility_threshold)
                motion = tracker.update(features)
                fired = gestures.update(features, motion)
                prediction = self._predict(ml_bundle, features, motion)
                for name, trail in trails.items():
                    trail.add_from(snapshot, SIDES[name], self.config.visibility_threshold)

                face_dict = None
                if self.config.face_enabled:
                    if face_detector is None:
                        face_dict = face_payload(enabled=True, error=face_error)
                    else:
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
                        except Exception as exc:  # noqa: BLE001 - optional layer,
                            # must never sink pose/web (requirement 4): one bad
                            # frame disables face for the rest of THIS session
                            # rather than crashing the whole pipeline thread —
                            # pose, motion, gestures, ML keep running untouched.
                            face_error = self._disable_face(face_detector, exc)
                            face_detector = None
                            face_dict = face_payload(enabled=True, error=face_error)

                self._broadcast(
                    frame_message(
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
                    )
                )
                self.frames_served += 1
        finally:
            if face_detector is not None:
                try:
                    face_detector.close()
                except Exception as exc:  # noqa: BLE001 - optional teardown
                    self._disable_face(None, exc)

    @staticmethod
    def _predict(
        bundle: Optional[dict], features: FrameFeatures, motion: MotionState
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
        columns = bundle.get("columns") or ()
        needs_motion = any(name not in FEATURE_NAMES for name in columns)
        if needs_motion and not motion.complete:
            return None
        row = {**features.to_dict(), **motion.to_dict()}
        return ml_predict(bundle, row)
