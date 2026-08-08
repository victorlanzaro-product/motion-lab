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
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from backend.camera import CameraConfig, CameraEngine, CameraError
from backend.features import MotionConfig, MotionTracker, Trail, extract
from backend.gestures import GestureConfig, GestureEngine
from backend.vision import (
    DEFAULT_VISIBILITY_THRESHOLD,
    PoseConfig,
    PoseDetector,
    PoseModelError,
    Side,
    ensure_model,
)
from backend.web.payload import encode_preview, error_message, frame_message

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
    #: Width of the JPEG sent to the browser. Inference still runs on the full
    #: frame, so this trades picture size against bandwidth, never accuracy.
    preview_width: int = 640
    jpeg_quality: int = 70
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD
    #: How long the camera stays open after the last client leaves. Long enough
    #: to survive a page reload, short enough that a closed tab frees the device.
    idle_seconds: float = 2.0


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
    ) -> None:
        self.config = config or WebConfig()
        self.camera_factory = camera_factory
        self.detector_factory = detector_factory

        self.state = "idle"  # idle | starting | live | failed
        self.error: Optional[str] = None
        self.frames_served = 0

        self._subscribers: list[Subscriber] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # --- subscriptions -------------------------------------------------
    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> Subscriber:
        """Register a client and make sure the camera is running."""
        subscriber = Subscriber(loop)
        with self._lock:
            self._subscribers.append(subscriber)
        self._ensure_running()
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    # --- lifecycle -----------------------------------------------------
    def _ensure_running(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self.state = "starting"
            self._thread = threading.Thread(
                target=self._run, name="motion-lab-pipeline", daemon=True
            )
            self._thread.start()

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

    def _run(self) -> None:
        """Thread body: open the devices, stream, and always release them."""
        self.error = None
        try:
            camera = self.camera_factory(self.config)
            detector = self.detector_factory(self.config)
            with camera, detector:
                self.state = "live"
                self._stream(camera, detector)
        except (CameraError, PoseModelError) as exc:
            # An expected, explainable failure: camera permission denied, model
            # missing. The user has to see it, so it goes out to every client.
            self.state = "failed"
            self.error = str(exc)
            self._broadcast(error_message(str(exc)))
        except Exception as exc:  # noqa: BLE001 - a dead thread must not fail silently
            self.state = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            self._broadcast(error_message(self.error))
        else:
            self.state = "idle"
        finally:
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
            for name, trail in trails.items():
                trail.add_from(snapshot, SIDES[name], self.config.visibility_threshold)

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
                )
            )
            self.frames_served += 1
