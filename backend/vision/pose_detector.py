"""Pose Engine — the only file that imports MediaPipe.

Runs the Pose Landmarker in VIDEO mode: it keeps tracking state between frames
(cheaper and steadier than re-detecting from scratch on every image) while
staying synchronous, so a slow frame slows the loop instead of silently
dropping results the way LIVE_STREAM would.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2

from backend.camera import Frame
from backend.vision.landmarks import Point, PoseSnapshot

#: Lite model: ~5.5 MB, fastest of the three. Good enough for arm tracking on a
#: MacBook CPU; swap for _full or _heavy if accuracy ever becomes the bottleneck.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = Path("models/pose_landmarker_lite.task")


class PoseModelError(RuntimeError):
    """Raised when the pose model file is missing or unusable."""


def ensure_model(path: Path = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Download the model on first run. Explicit, never on import."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {path} ...")
    try:
        urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed, hardcoded https URL
    except OSError as exc:
        raise PoseModelError(f"Could not download the pose model from {url}: {exc}") from exc
    return path


@dataclass(frozen=True)
class PoseConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    #: V1 tracks one person. More would force us to solve identity across frames.
    num_poses: int = 1
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


def _build_landmarker(config: PoseConfig):
    """Import MediaPipe lazily: importing it costs ~1s and spins up a GL context."""
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    if not Path(config.model_path).exists():
        raise PoseModelError(
            f"Pose model not found at {config.model_path}. "
            "Run ensure_model() or `uv run python scripts/check_pose.py` once with network access."
        )

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=config.num_poses,
        min_pose_detection_confidence=config.min_detection_confidence,
        min_pose_presence_confidence=config.min_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _to_points(landmarks) -> dict[int, Point]:
    return {
        index: Point(
            x=lm.x,
            y=lm.y,
            z=lm.z,
            visibility=lm.visibility or 0.0,
            presence=lm.presence or 0.0,
        )
        for index, lm in enumerate(landmarks)
    }


@dataclass
class PoseDetector:
    """Frame in, `PoseSnapshot` out. Never returns MediaPipe objects.

    `landmarker_factory` is injectable so tests can run without the model file.
    """

    config: PoseConfig = field(default_factory=PoseConfig)
    landmarker_factory: Callable[[PoseConfig], object] = _build_landmarker

    _landmarker: Optional[object] = field(default=None, init=False, repr=False)
    _last_timestamp_ms: int = field(default=-1, init=False)

    # --- lifecycle -----------------------------------------------------
    def open(self) -> "PoseDetector":
        if self._landmarker is None:
            self._landmarker = self.landmarker_factory(self.config)
        return self

    def close(self) -> None:
        landmarker = self._landmarker
        self._landmarker = None
        if landmarker is not None and hasattr(landmarker, "close"):
            landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- inference -----------------------------------------------------
    def _timestamp_ms(self, frame: Frame) -> int:
        """VIDEO mode rejects a timestamp that is not strictly increasing.

        perf_counter is monotonic, but rounding to milliseconds can collapse two
        fast frames onto the same value, so we nudge duplicates forward.
        """
        stamp = int(frame.timestamp * 1000)
        if stamp <= self._last_timestamp_ms:
            stamp = self._last_timestamp_ms + 1
        self._last_timestamp_ms = stamp
        return stamp

    def detect(self, frame: Frame) -> PoseSnapshot:
        """Always returns a snapshot; `snapshot.detected` is False when no body."""
        import mediapipe as mp

        if self._landmarker is None:
            self.open()

        # OpenCV gives BGR, MediaPipe wants RGB. Getting this backwards does not
        # crash — it just quietly makes detection worse.
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        started = time.perf_counter()
        result = self._landmarker.detect_for_video(image, self._timestamp_ms(frame))
        inference_ms = (time.perf_counter() - started) * 1000

        aspect = frame.width / frame.height if frame.height else 1.0

        if not result.pose_landmarks:
            return PoseSnapshot.empty(frame.index, frame.timestamp, inference_ms, aspect)

        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else []
        return PoseSnapshot(
            frame_index=frame.index,
            timestamp=frame.timestamp,
            normalized=_to_points(result.pose_landmarks[0]),
            world=_to_points(world),
            inference_ms=inference_ms,
            aspect=aspect,
        )
