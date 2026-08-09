"""Camera Engine — the only place in Motion Lab that talks to the webcam.

Design rules (from the spec):
  * everything downstream (pose, features, gestures) receives frames, never a
    `cv2.VideoCapture` handle. Swapping webcam for a video file must not touch
    the vision layer;
  * frames are never written to disk here (privacy requirement);
  * the frame delivered is the RAW sensor image, not mirrored. Mirroring is a
    display concern: flipping the image would swap left/right arms and silently
    corrupt every landmark label downstream.
"""

from __future__ import annotations

import platform
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or stops delivering frames."""


class CameraOpenError(CameraError):
    """Camera could not be opened; operator action may be required."""


class CameraReadError(CameraError):
    """An opened camera stopped delivering frames; this may be transient."""


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    #: how many consecutive empty reads before we consider the camera dead
    max_read_failures: int = 30


@dataclass(frozen=True)
class Frame:
    """One captured image plus the metadata every later stage needs."""

    image: np.ndarray  # BGR, as OpenCV delivers it
    index: int  # 0-based frame counter since the camera opened
    timestamp: float  # seconds, monotonic clock (perf_counter)

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


class FpsMeter:
    """Rolling FPS over the last N frame timestamps.

    Instant FPS (1 / last delta) jitters too much to read on screen, so we
    average over a window.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=max(2, window))

    def tick(self, timestamp: Optional[float] = None) -> None:
        self._timestamps.append(time.perf_counter() if timestamp is None else timestamp)

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


def _default_backend() -> int:
    """AVFoundation is the native macOS capture backend.

    Without it OpenCV falls back to a generic path that is slower to open and
    reports bogus resolutions on some MacBooks.
    """
    return cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY


def _open_capture(config: CameraConfig) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(config.index, _default_backend())
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    capture.set(cv2.CAP_PROP_FPS, config.target_fps)
    return capture


@dataclass
class CameraEngine:
    """Opens the webcam and yields `Frame` objects.

    `capture_factory` exists so tests can inject a fake capture instead of
    requiring real hardware.
    """

    config: CameraConfig = field(default_factory=CameraConfig)
    capture_factory: Callable[[CameraConfig], cv2.VideoCapture] = _open_capture

    _capture: Optional[cv2.VideoCapture] = field(default=None, init=False, repr=False)
    _frame_index: int = field(default=0, init=False)
    _fps: FpsMeter = field(default_factory=FpsMeter, init=False, repr=False)

    # --- lifecycle -----------------------------------------------------
    def open(self) -> "CameraEngine":
        if self._capture is not None:
            return self
        capture = self.capture_factory(self.config)
        if not capture.isOpened():
            capture.release()
            raise CameraOpenError(
                f"Could not open camera index {self.config.index}. "
                "On macOS, grant camera access to your terminal/IDE in "
                "System Settings > Privacy & Security > Camera, then restart it."
            )
        self._capture = capture
        return self

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "CameraEngine":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- capture -------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._capture is not None

    @property
    def fps(self) -> float:
        return self._fps.fps

    @property
    def frame_count(self) -> int:
        return self._frame_index

    @property
    def resolution(self) -> tuple[int, int]:
        """Resolution the driver actually granted (may differ from the request)."""
        if self._capture is None:
            raise CameraError("Camera is not open.")
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read(self) -> Optional[Frame]:
        """Grab one frame. Returns None on a transient read failure."""
        if self._capture is None:
            raise CameraError("Camera is not open. Call open() first.")
        ok, image = self._capture.read()
        if not ok or image is None:
            return None
        frame = Frame(image=image, index=self._frame_index, timestamp=time.perf_counter())
        self._frame_index += 1
        self._fps.tick(frame.timestamp)
        return frame

    def frames(self, limit: Optional[int] = None) -> Iterator[Frame]:
        """Yield frames until `limit` is reached or the camera dies.

        Transient dropped frames are tolerated; `max_read_failures` in a row
        means the device is gone (unplugged, or macOS revoked permission).
        """
        self.open()
        failures = 0
        produced = 0
        while limit is None or produced < limit:
            frame = self.read()
            if frame is None:
                failures += 1
                if failures >= self.config.max_read_failures:
                    raise CameraReadError(
                        f"Camera stopped delivering frames after "
                        f"{self.config.max_read_failures} consecutive failed reads."
                    )
                continue
            failures = 0
            produced += 1
            yield frame


def mirror(image: np.ndarray) -> np.ndarray:
    """Selfie view. Display only — never feed this to the pose estimator."""
    return cv2.flip(image, 1)
