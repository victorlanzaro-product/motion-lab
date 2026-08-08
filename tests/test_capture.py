"""Camera Engine tests. No real webcam involved: the capture is faked."""

from __future__ import annotations

import numpy as np
import pytest

from backend.camera import CameraConfig, CameraEngine, CameraError, FpsMeter, mirror


class FakeCapture:
    """Stand-in for cv2.VideoCapture.

    `script` is a list of booleans: True = a good frame, False = a failed read.
    """

    def __init__(self, script: list[bool], opened: bool = True, size: tuple[int, int] = (640, 480)):
        self._script = list(script)
        self._opened = opened
        self._size = size
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return self._opened

    def read(self):
        if not self._script:
            return False, None
        ok = self._script.pop(0)
        if not ok:
            return False, None
        w, h = self._size
        return True, np.zeros((h, w, 3), dtype=np.uint8)

    def get(self, prop: int) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._size[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._size[1])
        return 0.0

    def set(self, *_args) -> bool:
        return True

    def release(self) -> None:
        self.released = True


def engine(script: list[bool], opened: bool = True, **config_kwargs) -> CameraEngine:
    capture = FakeCapture(script, opened=opened)
    config = CameraConfig(**config_kwargs)
    return CameraEngine(config=config, capture_factory=lambda _cfg: capture)


# --- FpsMeter ----------------------------------------------------------


def test_fps_is_zero_before_two_ticks():
    meter = FpsMeter()
    assert meter.fps == 0.0
    meter.tick(1.0)
    assert meter.fps == 0.0


def test_fps_uses_the_rolling_window():
    meter = FpsMeter(window=5)
    for i in range(5):
        meter.tick(i * 0.05)  # 20 fps
    assert meter.fps == pytest.approx(20.0)


def test_fps_window_forgets_old_frames():
    meter = FpsMeter(window=3)
    meter.tick(0.0)
    meter.tick(10.0)  # ancient, must fall out of the window
    meter.tick(10.1)
    meter.tick(10.2)
    assert meter.fps == pytest.approx(10.0)


# --- CameraEngine ------------------------------------------------------


def test_open_failure_raises_with_permission_hint():
    with pytest.raises(CameraError, match="System Settings"):
        engine([True], opened=False).open()


def test_read_before_open_raises():
    with pytest.raises(CameraError, match="not open"):
        engine([True]).read()


def test_frames_are_numbered_and_timestamped():
    with engine([True, True, True]) as camera:
        frames = list(camera.frames(limit=3))
    assert [f.index for f in frames] == [0, 1, 2]
    assert frames[0].timestamp <= frames[-1].timestamp
    assert camera.frame_count == 3


def test_frame_exposes_its_size():
    with engine([True]) as camera:
        frame = camera.read()
    assert (frame.width, frame.height) == (640, 480)


def test_transient_read_failures_are_skipped():
    with engine([True, False, False, True]) as camera:
        frames = list(camera.frames(limit=2))
    assert len(frames) == 2
    assert [f.index for f in frames] == [0, 1]  # dropped reads do not consume an index


def test_camera_death_raises_after_the_failure_budget():
    with pytest.raises(CameraError, match="stopped delivering frames"):
        with engine([False] * 5, max_read_failures=3) as camera:
            list(camera.frames(limit=1))


def test_context_manager_releases_the_device():
    capture = FakeCapture([True])
    with CameraEngine(capture_factory=lambda _cfg: capture) as camera:
        camera.read()
    assert capture.released is True
    assert camera.is_open is False


def test_resolution_reports_what_the_driver_granted():
    with engine([True], width=1920, height=1080) as camera:
        assert camera.resolution == (640, 480)  # fake device only offers 640x480


# --- mirror ------------------------------------------------------------


def test_mirror_flips_horizontally_only():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = 255  # top-left marker
    flipped = mirror(image)
    assert flipped[0, 2, 0] == 255  # moved to top-right
    assert flipped[0, 0, 0] == 0
