"""Face Engine — the only file that imports the MediaPipe Face Landmarker.

Same shape as `pose_detector.py`: runs in VIDEO mode (keeps tracking state
between frames, stays synchronous so a slow frame slows the loop instead of
silently dropping a result), and the model downloads once on first use, not
on import.

100% local: the model file is downloaded once from Google's public model
zoo (same as the pose model), then every inference runs on-device — no frame,
crop, or landmark is ever sent anywhere.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2

from backend.camera import Frame
from backend.vision.face_landmarks import FACE_OVERLAY_LANDMARKS, FaceSnapshot, euler_from_matrix
from backend.vision.landmarks import Point

#: Google's standard Face Landmarker task bundle (~3.6 MB): mesh + 52
#: blendshape categories + facial transformation matrix, same model zoo and
#: URL shape as `pose_detector.py`'s `MODEL_URL`.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")


class FaceModelError(RuntimeError):
    """Raised when the face model file is missing or unusable.

    Caught by `PipelineRunner` alone (never by pose/gestures/web): the face
    layer is opt-in and must degrade to "unavailable", not take the rest of
    the pipeline down with it.
    """


def ensure_model(path: Path = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Download the model on first run. Explicit, never on import."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face model to {path} ...")
    try:
        urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed, hardcoded https URL
    except OSError as exc:
        raise FaceModelError(f"Could not download the face model from {url}: {exc}") from exc
    return path


def _to_mesh_points(landmarks) -> dict[int, Point]:
    """Overlay-subset points, keyed by index — mirrors `pose_detector._to_points`,
    restricted to `FACE_OVERLAY_LANDMARKS` instead of every point MediaPipe
    returned. A landmark list shorter than expected (a fake in a test, or a
    future model with fewer points) is skipped index-by-index rather than
    raising: a partially built overlay beats a crashed pipeline thread.
    """
    return {
        index: Point(x=landmarks[index].x, y=landmarks[index].y, z=landmarks[index].z)
        for index in FACE_OVERLAY_LANDMARKS
        if index < len(landmarks)
    }


@dataclass(frozen=True)
class FaceConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    #: V1 tracks one face — same identity-across-frames reasoning as
    #: `PoseConfig.num_poses`.
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


def _build_landmarker(config: FaceConfig):
    """Import MediaPipe lazily: importing it costs ~1s and spins up a GL context."""
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    if not Path(config.model_path).exists():
        raise FaceModelError(
            f"Face model not found at {config.model_path}. "
            "Run ensure_model() once with network access."
        )

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=config.num_faces,
        min_face_detection_confidence=config.min_face_detection_confidence,
        min_face_presence_confidence=config.min_face_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


@dataclass
class FaceDetector:
    """Frame in, `FaceSnapshot` out. Never returns MediaPipe objects.

    `landmarker_factory` is injectable so tests can run without the model
    file, same pattern as `PoseDetector`.
    """

    config: FaceConfig = field(default_factory=FaceConfig)
    landmarker_factory: Callable[[FaceConfig], object] = _build_landmarker

    _landmarker: Optional[object] = field(default=None, init=False, repr=False)
    _last_timestamp_ms: int = field(default=-1, init=False)

    # --- lifecycle -----------------------------------------------------
    def open(self) -> "FaceDetector":
        if self._landmarker is None:
            self._landmarker = self.landmarker_factory(self.config)
        return self

    def close(self) -> None:
        landmarker = self._landmarker
        self._landmarker = None
        if landmarker is not None and hasattr(landmarker, "close"):
            landmarker.close()

    def __enter__(self) -> "FaceDetector":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- inference -----------------------------------------------------
    def _timestamp_ms(self, frame: Frame) -> int:
        """Same monotonic-nudge rule as `PoseDetector`: VIDEO mode rejects a
        timestamp that is not strictly increasing, and rounding to
        milliseconds can collapse two fast frames onto the same value."""
        stamp = int(frame.timestamp * 1000)
        if stamp <= self._last_timestamp_ms:
            stamp = self._last_timestamp_ms + 1
        self._last_timestamp_ms = stamp
        return stamp

    def detect(self, frame: Frame) -> FaceSnapshot:
        """Always returns a snapshot; `snapshot.detected` is False when no face.

        Reuses the same `Frame` the pose detector already received this
        iteration — no second camera read, no second capture.
        """
        import mediapipe as mp

        if self._landmarker is None:
            self.open()

        # OpenCV gives BGR, MediaPipe wants RGB — same conversion as pose.
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        started = time.perf_counter()
        result = self._landmarker.detect_for_video(image, self._timestamp_ms(frame))
        inference_ms = (time.perf_counter() - started) * 1000

        if not result.face_landmarks:
            return FaceSnapshot.empty(frame.index, frame.timestamp, inference_ms)

        categories = result.face_blendshapes[0] if result.face_blendshapes else []
        blendshapes = {category.category_name: category.score for category in categories}

        head_yaw = head_pitch = head_roll = None
        if result.facial_transformation_matrixes:
            matrix = result.facial_transformation_matrixes[0]
            head_yaw, head_pitch, head_roll = euler_from_matrix(matrix)

        # Same `result.face_landmarks[0]` MediaPipe already returned above for
        # this frame — reading a subset of it here is not a second inference,
        # just picking out the ~130 points the overlay draws (see
        # `FACE_OVERLAY_LANDMARKS`'s docstring) instead of all 478.
        mesh = _to_mesh_points(result.face_landmarks[0])

        return FaceSnapshot(
            frame_index=frame.index,
            timestamp=frame.timestamp,
            detected=True,
            blendshapes=blendshapes,
            mesh=mesh,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            head_roll=head_roll,
            inference_ms=inference_ms,
        )
