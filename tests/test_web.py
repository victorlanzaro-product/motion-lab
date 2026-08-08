"""Web App tests — the whole stack, with no webcam, no model and no browser.

The pipeline is driven by a fake capture and a fake landmarker, so these tests
assert what actually reaches a client: the handshake, the payload, and the
camera lifecycle the privacy rule depends on.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="install the 'web' extra: uv sync --extra web")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from backend.camera import CameraConfig, CameraEngine  # noqa: E402
from backend.features import FEATURE_NAMES  # noqa: E402
from backend.vision import ARM_CHAINS, ARM_LANDMARKS, PoseDetector, PoseLandmark  # noqa: E402
from backend.web import WebConfig, create_app  # noqa: E402
from backend.web.pipeline import PipelineRunner, Subscriber  # noqa: E402

LEFT_WRIST_X = 0.2  # off-centre on purpose: a mirrored copy would read 0.8


class FakeCapture:
    """An endless camera. `delay` keeps the loop from spinning at 10k fps."""

    def __init__(
        self,
        size: tuple[int, int] = (320, 240),
        opened: bool = True,
        delay: float = 0.005,
    ):
        self._size = size
        self._opened = opened
        self._delay = delay
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return self._opened

    def read(self):
        time.sleep(self._delay)
        width, height = self._size
        return True, np.full((height, width, 3), 60, dtype=np.uint8)

    def get(self, _prop: int) -> float:
        return 0.0

    def set(self, *_args) -> bool:
        return True

    def release(self) -> None:
        self.released = True


class FakeLandmarker:
    """Always sees a body: all 33 points, with a distinctive left wrist.

    Shoulders sit apart on purpose — `shoulder_width` is the ruler every other
    body-relative feature divides by, so collapsing left/right onto the same
    pixel would make every derived feature (`wrist_above_shoulder`, angles,
    distances) silently `None` instead of a usable value.
    """

    def detect_for_video(self, _image, _timestamp_ms: int):
        points = [
            SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9, presence=0.9) for _ in range(33)
        ]
        # Raw image space: the left shoulder sits at a larger x than the right
        # one (positions.py:90), same convention `wrists_crossed` relies on.
        points[int(PoseLandmark.LEFT_SHOULDER)] = SimpleNamespace(
            x=0.6, y=0.5, z=0.0, visibility=0.9, presence=0.9
        )
        points[int(PoseLandmark.RIGHT_SHOULDER)] = SimpleNamespace(
            x=0.4, y=0.5, z=0.0, visibility=0.9, presence=0.9
        )
        points[int(PoseLandmark.LEFT_WRIST)] = SimpleNamespace(
            x=LEFT_WRIST_X, y=0.4, z=0.0, visibility=0.9, presence=0.9
        )
        return SimpleNamespace(pose_landmarks=[points], pose_world_landmarks=[points])

    def close(self) -> None:
        pass


def build(opened: bool = True, **overrides) -> tuple[PipelineRunner, list[FakeCapture]]:
    """A runner wired to fakes, plus the list of capture devices it opened."""
    captures: list[FakeCapture] = []

    def camera_factory(config: WebConfig) -> CameraEngine:
        capture = FakeCapture(opened=opened)
        captures.append(capture)
        return CameraEngine(config=config.camera, capture_factory=lambda _cfg: capture)

    def detector_factory(_config: WebConfig) -> PoseDetector:
        return PoseDetector(landmarker_factory=lambda _cfg: FakeLandmarker())

    config = WebConfig(
        camera=CameraConfig(width=320, height=240),
        preview_width=160,
        idle_seconds=0.2,
        **overrides,
    )
    runner = PipelineRunner(
        config, camera_factory=camera_factory, detector_factory=detector_factory
    )
    return runner, captures


def client_for(runner: PipelineRunner) -> TestClient:
    return TestClient(create_app(runner=runner))


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# --- handshake ---------------------------------------------------------


def test_hello_arrives_before_any_frame():
    """The client cannot draw a bone before it knows the bone table."""
    runner, _ = build()
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        hello = socket.receive_json()
    runner.stop()

    assert hello["type"] == "hello"
    assert len(hello["chains"]) == len(ARM_CHAINS)
    assert hello["feature_names"] == list(FEATURE_NAMES)
    assert hello["arm_landmarks"] == [int(lm) for lm in ARM_LANDMARKS]
    assert "arm_raised" in hello["gesture_names"]


# --- payload -----------------------------------------------------------


def test_frame_carries_pixels_joints_and_numbers_together():
    runner, _ = build()
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        frame = socket.receive_json()
    runner.stop()

    assert frame["type"] == "frame"
    assert frame["detected"] is True
    assert frame["image"]["base64"]
    assert frame["motion"]["left_direction"] in ("up", "down", "still", "unknown")
    assert set(FEATURE_NAMES) <= set(frame["features"])
    assert frame["gestures"] == []  # a single frame is never enough to hold a gesture


def test_only_arm_landmarks_travel():
    """The estimator returns 33 points; V1 draws six."""
    runner, _ = build()
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()
        frame = socket.receive_json()
    runner.stop()

    assert set(frame["landmarks"]) == {str(int(lm)) for lm in ARM_LANDMARKS}


def test_coordinates_reach_the_browser_unmirrored():
    """Mirroring here would swap the arms — it stays a display concern."""
    runner, _ = build()
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()
        frame = socket.receive_json()
    runner.stop()

    wrist = frame["landmarks"][str(int(PoseLandmark.LEFT_WRIST))]
    assert wrist["x"] == pytest.approx(LEFT_WRIST_X)


def test_a_sustained_pose_fires_a_gesture_through_the_whole_pipeline():
    """End-to-end: the fake landmarker holds the left wrist above the shoulder
    continuously, so after `hold_seconds` the WebSocket must carry
    `arm_raised` — this is the wiring, not the already unit-tested engine."""
    runner, _ = build()
    fired = None
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            frame = socket.receive_json()
            if frame["gestures"]:
                fired = frame["gestures"]
                break
    runner.stop()

    assert fired is not None, "arm_raised never fired within 5s of a sustained pose"
    assert fired[0]["name"] == "arm_raised"
    assert fired[0]["side"] == "left"


def test_preview_is_downscaled_but_inference_is_not():
    """The camera gives 320 px; the browser gets 160 and the pose still ran."""
    runner, _ = build()
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()
        frame = socket.receive_json()
    runner.stop()

    assert frame["image"]["width"] == 160
    assert frame["image"]["height"] == 120
    assert frame["detected"] is True


# --- camera lifecycle --------------------------------------------------


def test_camera_stays_closed_until_a_client_connects():
    """The green LED is a promise: nobody watching, nothing captured."""
    runner, captures = build()
    with client_for(runner) as client:
        assert client.get("/health").json()["pipeline"] == "idle"
        assert captures == []

        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.receive_json()
            assert len(captures) == 1
    runner.stop()


def test_camera_is_released_after_the_last_client_leaves():
    runner, captures = build()
    with client_for(runner) as client:
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.receive_json()
        assert wait_until(lambda: captures[0].released), "the webcam was never released"
    runner.stop()

    assert runner.client_count == 0


def test_two_browsers_share_one_camera():
    """Two tabs must not race for the device: one loop, two subscribers."""
    runner, captures = build()
    with client_for(runner) as client:
        with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as second:
            first.receive_json()
            second.receive_json()
            assert first.receive_json()["type"] == "frame"
            assert second.receive_json()["type"] == "frame"
            assert runner.client_count == 2
    runner.stop()

    assert len(captures) == 1


# --- failures ----------------------------------------------------------


def test_camera_failure_reaches_the_browser_instead_of_hanging():
    runner, _ = build(opened=False)
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        message = socket.receive_json()
    runner.stop()

    assert message["type"] == "error"
    assert "System Settings" in message["message"]  # the macOS permission hint
    assert runner.state == "failed"


def test_health_reports_why_the_stream_is_not_running():
    runner, _ = build(opened=False)
    with client_for(runner) as client:
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.receive_json()
        health = client.get("/health").json()
    runner.stop()

    assert health["pipeline"] == "failed"
    assert "System Settings" in health["error"]


# --- backpressure ------------------------------------------------------


def test_a_slow_client_gets_the_newest_frame_not_a_backlog():
    """Falling behind must cost frames, never latency."""

    async def scenario():
        subscriber = Subscriber(asyncio.get_running_loop())
        for index in range(1, 4):
            subscriber.offer({"frame_index": index})
        return await subscriber.get(), subscriber.dropped

    message, dropped = asyncio.run(scenario())
    assert message["frame_index"] == 3
    assert dropped == 2


# --- static ------------------------------------------------------------


def test_the_page_and_its_assets_are_served():
    runner, _ = build()
    with client_for(runner) as client:
        page = client.get("/")
        script = client.get("/static/app.js")
    runner.stop()

    assert page.status_code == 200
    assert "MOTION LAB" in page.text
    assert script.status_code == 200
