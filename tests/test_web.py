"""Web App tests — the whole stack, with no webcam, no model and no browser.

The pipeline is driven by a fake capture and a fake landmarker, so these tests
assert what actually reaches a client: the handshake, the payload, and the
camera lifecycle the privacy rule depends on.
"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="install the 'web' extra: uv sync --extra web")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from backend.camera import CameraConfig, CameraEngine  # noqa: E402
from backend.dataset import DATASET_COLUMNS, DEFAULT_LABELS, V2_LABELS, DatasetWriter  # noqa: E402
from backend.dataset.stats import dataset_stats  # noqa: E402
from backend.features import (  # noqa: E402
    FEATURE_NAMES,
    ArmMotion,
    Direction,
    FrameFeatures,
    MotionState,
)
from backend.ml import MIN_SAMPLES_PER_CLASS, rule_label, train  # noqa: E402
from backend.vision import (  # noqa: E402
    ARM_CHAINS,
    ARM_LANDMARKS,
    FACE_CHAIN_GROUPS,
    FACE_OVERLAY_LANDMARKS,
    FaceDetector,
    FaceModelError,
    PoseDetector,
    PoseLandmark,
)
from backend.web import WebConfig, create_app  # noqa: E402
from backend.web.app import _write_frames  # noqa: E402
from backend.web.pipeline import PipelineRunner, Subscriber  # noqa: E402
from scripts import hello_web  # noqa: E402

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
        # The right wrist must differ from the right elbow's default point too
        # (0.5, 0.5): a zero-length elbow->wrist vector makes `angle_between`
        # return None (`angles.py`), which would make `FrameFeatures.complete`
        # false on every frame and silently skip ML inference (Sprint 09).
        points[int(PoseLandmark.RIGHT_WRIST)] = SimpleNamespace(
            x=0.55, y=0.6, z=0.0, visibility=0.9, presence=0.9
        )
        return SimpleNamespace(pose_landmarks=[points], pose_world_landmarks=[points])

    def close(self) -> None:
        pass


def build(
    opened: bool = True, face_detector_factory=None, landmarker_factory=None, **overrides
) -> tuple[PipelineRunner, list[FakeCapture]]:
    """A runner wired to fakes, plus the list of capture devices it opened.

    `face_detector_factory` is only ever consulted when a test also passes
    `face_enabled=True` — otherwise `PipelineRunner` never calls it, the same
    opt-in gate the live pipeline uses.
    """
    captures: list[FakeCapture] = []

    def camera_factory(config: WebConfig) -> CameraEngine:
        capture = FakeCapture(opened=opened)
        captures.append(capture)
        return CameraEngine(config=config.camera, capture_factory=lambda _cfg: capture)

    def detector_factory(_config: WebConfig) -> PoseDetector:
        factory = landmarker_factory or (lambda _cfg: FakeLandmarker())
        return PoseDetector(landmarker_factory=factory)

    config_values = {
        "camera": CameraConfig(width=320, height=240),
        "preview_width": 160,
        "idle_seconds": 0.2,
    }
    config_values.update(overrides)
    config = WebConfig(**config_values)
    kwargs = dict(camera_factory=camera_factory, detector_factory=detector_factory)
    if face_detector_factory is not None:
        kwargs["face_detector_factory"] = face_detector_factory
    runner = PipelineRunner(config, **kwargs)
    retur…11449 tokens truncated…e3o foi possível abrir a câmera."


class FlappingCapture:
    """Delivers exactly `healthy_frames` good reads on every open, then fails
    forever -- "healthy" by the `healthy_camera_frames` threshold every single
    time, so `attempt` resets to 0 on its own each cycle and never trips
    `max_camera_retries`. Only the independent flap-window cap should."""

    def __init__(self, healthy_frames: int, size: tuple[int, int] = (320, 240)):
        self._remaining = healthy_frames
        self._size = size
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def read(self):
        time.sleep(0.001)
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        width, height = self._size
        return True, np.full((height, width, 3), 60, dtype=np.uint8)

    def get(self, _prop: int) -> float:
        return 0.0

    def set(self, *_args) -> bool:
        return True

    def release(self) -> None:
        self.released = True


def test_a_camera_that_flaps_forever_eventually_gives_up():
    """Every individual run looks like a real recovery (>= healthy_camera_frames
    before failing again), so `attempt` alone would retry forever. This pins
    `max_flaps_per_window` as the backstop that still gives up."""

    def camera_factory(config: WebConfig) -> CameraEngine:
        return CameraEngine(
            config=replace(config.camera, max_read_failures=1),
            capture_factory=lambda _cfg: FlappingCapture(healthy_frames=2),
        )

    def detector_factory(_config: WebConfig) -> PoseDetector:
        return PoseDetector(landmarker_factory=lambda _cfg: FakeLandmarker())

    config = WebConfig(
        camera=CameraConfig(width=320, height=240),
        preview_width=160,
        idle_seconds=5.0,
        healthy_camera_frames=2,
        max_camera_retries=1000,  # would retry "forever" without the flap cap
        retry_backoff_base=0.001,
        retry_backoff_cap=0.002,
        flap_window_seconds=60.0,
        max_flaps_per_window=3,
    )
    runner = PipelineRunner(
        config, camera_factory=camera_factory, detector_factory=detector_factory
    )

    message = None
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            message = socket.receive_json()
            if message["type"] == "error" and message["fatal"]:
                break
    runner.stop()

    assert message is not None and message["type"] == "error"
    assert message["fatal"] is True
    assert "As tentativas de reconexão foram encerradas." in message["message"]
    assert runner.state == "failed"


# --- recovery ------------------------------------------------------------


class DeadCapture(FakeCapture):
    """Opens fine (`isOpened() is True`) but never actually delivers a frame --
    the mid-session failure this test is about, distinct from `opened=False`
    above (permission denied at `open()`, never recovers)."""

    def read(self):
        time.sleep(0.001)
        return False, None


def test_camera_recovers_after_a_transient_failure_and_resumes_streaming(tmp_path):
    """'camera falha e volta': a mid-stream failure is not a permanent
    disconnect. `PipelineRunner` retries with a bounded backoff, the socket
    stays open through a non-fatal `status` message, and frames resume once
    the camera (factory) comes back."""
    attempts: list[FakeCapture] = []

    def camera_factory(config: WebConfig) -> CameraEngine:
        capture = DeadCapture() if len(attempts) == 0 else FakeCapture()
        attempts.append(capture)
        return CameraEngine(
            config=replace(config.camera, max_read_failures=2),
            capture_factory=lambda _cfg: capture,
        )

    def detector_factory(_config: WebConfig) -> PoseDetector:
        return PoseDetector(landmarker_factory=lambda _cfg: FakeLandmarker())

    config = WebConfig(
        camera=CameraConfig(width=320, height=240),
        preview_width=160,
        idle_seconds=5.0,
        max_camera_retries=3,
        retry_backoff_base=0.01,
        retry_backoff_cap=0.02,
        dataset_path=tmp_path / "recovered.csv",
    )
    runner = PipelineRunner(
        config, camera_factory=camera_factory, detector_factory=detector_factory
    )

    saw_recovering_status = False
    saw_warning = False
    saw_frame_after_recovery = False
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not saw_frame_after_recovery:
            message = socket.receive_json()
            if message["type"] == "status":
                assert message["state"] == "recovering"
                saw_recovering_status = True
            elif message["type"] == "error":
                assert message["fatal"] is False
                assert message["message"] == "A câmera parou de enviar imagens."
                saw_warning = True
            elif message["type"] == "frame" and saw_recovering_status:
                saw_frame_after_recovery = True
        # Recovery opens a fresh stream path inside the same runner; the
        # lifecycle gate must accept commands again after the camera returns.
        socket.send_json({"type": "record_start", "label": "idle"})
        started = receive_until(
            socket,
            lambda item: item.get("type") == "record_status" and item["recording"],
        )
        assert started["owner"] is True
        socket.send_json({"type": "record_stop"})
        receive_until(
            socket,
            lambda item: item.get("type") == "record_status"
            and not item["recording"]
            and item["session_id"] == started["session_id"],
        )
    runner.stop()

    assert saw_warning
    assert saw_recovering_status, "expected a non-fatal 'recovering' status while retrying"
    assert saw_frame_after_recovery, "streaming never resumed after the camera came back"
    assert len(attempts) >= 2, "the runner should have reopened the camera after the failure"


# --- backpressure ------------------------------------------------------


def test_a_slow_client_gets_the_newest_frame_not_a_backlog():
    """Falling behind must cost frames, never latency."""

    async def scenario():
        subscriber = Subscriber(asyncio.get_running_loop())
        for index in range(1, 4):
            subscriber.offer({"type": "frame", "frame_index": index})
        return await subscriber.get(), subscriber.dropped

    message, dropped = asyncio.run(scenario())
    assert message["frame_index"] == 3
    assert dropped == 2


def test_control_flood_is_bounded_and_keeps_recording_transitions():
    """A rejected-command flood cannot hide the start or final stop status."""

    async def scenario():
        subscriber = Subscriber(asyncio.get_running_loop())
        started = {
            "type": "record_status",
            "recording": True,
            "label": "idle",
            "session_id": "session-1",
            "written": 0,
            "skipped": 0,
            "owner": True,
            "error": None,
        }
        subscriber.offer(started)

        for index in range(Subscriber.MAX_CONTROL_MESSAGES * 4):
            subscriber.offer(
                {
                    **started,
                    "owner": False,
                    "error": f"request {index} rejected while recording",
                }
            )

        repeated_rejection = {
            **started,
            "owner": False,
            "error": "another recording is already active",
        }
        for _ in range(1000):
            subscriber.offer(repeated_rejection)

        stopped = {
            "type": "record_status",
            "recording": False,
            "label": "idle",
            "session_id": "session-1",
            "written": 125,
            "skipped": 2,
            "owner": False,
            "error": "camera disconnected during active recording",
        }
        subscriber.offer(stopped)
        queued = len(subscriber._control)
        messages = [await subscriber.get() for _ in range(queued)]
        return queued, messages, started, stopped, repeated_rejection

    queued, messages, started, stopped, repeated_rejection = asyncio.run(scenario())

    assert queued <= Subscriber.MAX_CONTROL_MESSAGES
    assert messages[0] == started
    assert messages[-1] == stopped
    assert messages.count(repeated_rejection) == 1


@pytest.mark.parametrize(
    "overflow_message",
    [
        pytest.param(
            {"type": "error", "message": "camera permanently denied", "fatal": True},
            id="fatal-error",
        ),
        pytest.param(
            {
                "type": "record_status",
                "recording": True,
                "label": "idle",
                "session_id": "current-session",
                "written": 1,
                "skipped": 0,
                "owner": True,
                "error": None,
            },
            id="current-recording-state",
        ),
    ],
)
def test_final_only_control_overflow_closes_socket_for_fresh_snapshot(overflow_message):
    """A saturated final-only queue cannot leave a client on stale state."""

    async def scenario():
        subscriber = Subscriber(asyncio.get_running_loop())
        for index in range(Subscriber.MAX_CONTROL_MESSAGES):
            subscriber.offer(
                {
                    "type": "record_status",
                    "recording": False,
                    "label": "idle",
                    "session_id": f"session-{index}",
                    "written": index,
                    "skipped": 0,
                    "owner": False,
                    "error": None,
                }
            )
        assert len(subscriber._control) == Subscriber.MAX_CONTROL_MESSAGES
        assert all(subscriber._is_final_record_status(item) for item in subscriber._control)

        subscriber.offer(overflow_message)

        class Socket:
            def __init__(self):
                self.sent = []
                self.close_args = None

            async def send_json(self, message):
                self.sent.append(message)

            async def close(self, *, code, reason):
                self.close_args = (code, reason)

        socket = Socket()
        await _write_frames(socket, subscriber)
        return socket

    socket = asyncio.run(scenario())
    assert socket.sent == []
    assert socket.close_args is not None
    assert socket.close_args[0] == 1013
    assert "fresh state snapshot" in socket.close_args[1]


# --- static ------------------------------------------------------------


def test_the_page_and_its_assets_are_served():
    runner, _ = build()
    with client_for(runner) as client:
        page = client.get("/")
        script = client.get("/static/app.js")
        studio_controls = client.get("/static/studio-controls.js")
    runner.stop()

    assert page.status_code == 200
    assert "Motion Lab" in page.text
    assert "Live Studio" in page.text
    assert script.status_code == 200
    assert studio_controls.status_code == 200


# --- facial signals (Sprint 11) -----------------------------------------


class FakeFaceCategory:
    def __init__(self, category_name: str, score: float):
        self.category_name = category_name
        self.score = score


#: MediaPipe's real Face Landmarker always returns all 478 points; this fake
#: does too, so `FaceDetector.detect()` can index into `FACE_OVERLAY_LANDMARKS`
#: without going out of bounds -- same shape as `tests/test_face.py`'s
#: `_fake_mesh_landmarks`.
FULL_FACE_MESH_SIZE = 478


class FakeFaceLandmarker:
    """Always sees a face, with a distinctive smile/mouth-open reading."""

    def detect_for_video(self, _image, _timestamp_ms: int):
        categories = [FakeFaceCategory("mouthSmileLeft", 0.7), FakeFaceCategory("jawOpen", 0.1)]
        mesh = [
            SimpleNamespace(x=(i % 100) / 100, y=(i % 100) / 100, z=0.0)
            for i in range(FULL_FACE_MESH_SIZE)
        ]
        return SimpleNamespace(
            face_landmarks=[mesh],
            face_blendshapes=[categories],
            facial_transformation_matrixes=[],
        )

    def close(self):
        pass


def test_face_is_unavailable_by_default():
    """`face_enabled` defaults to `False`: the payload says so without ever
    touching a face model — no second inference, no surprise CPU cost."""
    runner, _ = build()
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        frame = receive_frame(socket)
    runner.stop()

    assert frame["face"] == {"available": False, "detected": False, "reason": "disabled"}


def test_face_signals_ride_along_when_enabled():
    runner, _ = build(
        face_enabled=True,
        face_detector_factory=lambda _config: FaceDetector(
            landmarker_factory=lambda _cfg: FakeFaceLandmarker()
        ),
    )
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        frame = receive_frame(socket)
    runner.stop()

    face = frame["face"]
    assert face["available"] is True
    assert face["detected"] is True
    assert face["smile"] == pytest.approx(0.7)
    assert face["mouth_open"] == pytest.approx(0.1)
    # Only named, observable signals travel — never an emotion/mood label.
    assert "emotion" not in face and "mood" not in face
    # Sprint 11 overlay: only the named-region subset travels, raw and
    # unmirrored (mirroring is a display concern, same rule as arm joints).
    assert set(face["landmarks"]) == {str(i) for i in FACE_OVERLAY_LANDMARKS}
    sample_index = FACE_OVERLAY_LANDMARKS[0]
    expected = (sample_index % 100) / 100
    assert face["landmarks"][str(sample_index)]["x"] == pytest.approx(expected)
    assert "v" not in face["landmarks"][str(sample_index)]  # no fabricated confidence


def test_face_overlay_landmarks_are_empty_when_no_face_is_in_frame():
    """'não desenhe linhas quando não há rosto': an empty `landmarks` dict is
    what tells the client that, distinct from `available`/`detected` above."""

    class NoFaceLandmarker:
        def detect_for_video(self, _image, _timestamp_ms: int):
            return SimpleNamespace(
                face_landmarks=[], face_blendshapes=[], facial_transformation_matrixes=[]
            )

        def close(self):
            pass

    runner, _ = build(
        face_enabled=True,
        face_detector_factory=lambda _config: FaceDetector(
            landmarker_factory=lambda _cfg: NoFaceLandmarker()
        ),
    )
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        frame = receive_frame(socket)
    runner.stop()

    assert frame["face"]["available"] is True
    assert frame["face"]["detected"] is False
    assert frame["face"]["landmarks"] == {}


def test_face_overlay_payload_cost_is_bounded_and_reported():
    """Sizes the actual per-frame byte cost of the Sprint 11 overlay so a
    regression that widened it (e.g. a future full-478-point mode enabled by
    accident) shows up here instead of only in a manual review of the wire."""
    runner, _ = build(
        face_enabled=True,
        face_detector_factory=lambda _config: FaceDetector(
            landmarker_factory=lambda _cfg: FakeFaceLandmarker()
        ),
    )
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        frame = receive_frame(socket)
    runner.stop()

    import json

    landmarks_json = json.dumps(frame["face"]["landmarks"])
    # ~130 points at `"NNN":{"x":0.NNNN,"y":0.NNNN}` each: a few KB, not the
    # ~3.5x-larger cost sending the full 478-point mesh would add (this test's
    # fake JPEG is flat-color and compresses to near nothing, so comparing
    # against `len(frame_json)` here would not mean what it means in
    # production -- the real preview JPEG is the dominant cost there, not
    # this overlay; see the report for the measured real-world proportion).
    assert len(landmarks_json) < 6000
    bytes_per_point = len(landmarks_json) / len(FACE_OVERLAY_LANDMARKS)
    assert bytes_per_point < 40  # no stray whitespace/precision bloat crept in


def test_face_degrades_to_unavailable_when_the_model_fails_to_load():
    """A broken/missing face model must not take pose, gestures or ML down
    with it — same isolation the ML bundle already has (Sprint 09)."""

    def failing_face_factory(_config):
        raise FaceModelError("face model not found")

    runner, _ = build(face_enabled=True, face_detector_factory=failing_face_factory)
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        frame = receive_frame(socket)
    runner.stop()

    assert frame["face"]["available"] is False
    assert frame["face"]["reason"] == (
        "O modelo facial está indisponível. Verifique se ele foi instalado."
    )
    # The rest of the pipeline is unaffected by the face model failing.
    assert frame["type"] == "frame"
    assert frame["detected"] is True


class ExplodingFaceLandmarker:
    """Opens fine, then blows up on the first real frame -- a MediaPipe
    runtime hiccup mid-session, distinct from the model never loading at all
    above."""

    def detect_for_video(self, _image, _timestamp_ms: int):
        raise RuntimeError("boom")

    def close(self) -> None:
        pass


def test_a_face_runtime_error_degrades_without_stopping_pose_or_the_pipeline(caplog):
    """Requirement: face is optional and fails degraded, never taking pose,
    gestures, ML or the WebSocket itself down with it (unlike a `CameraError`,
    which is a whole-pipeline concern -- this must stay scoped to `face`)."""
    runner, _ = build(
        face_enabled=True,
        face_detector_factory=lambda _config: FaceDetector(
            landmarker_factory=lambda _cfg: ExplodingFaceLandmarker()
        ),
    )
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        warning = receive_until(socket, lambda item: item["type"] == "error")
        frame1 = receive_frame(socket)
        frame2 = receive_frame(socket)
        health = client.get("/health").json()
    runner.stop()

    public_reason = "Falha ao analisar o rosto. O rastreamento facial foi desativado."
    assert warning == {"type": "error", "fatal": False, "message": public_reason}
    assert frame1["type"] == "frame" and frame1["detected"] is True
    assert frame2["type"] == "frame" and frame2["detected"] is True
    assert frame1["face"]["available"] is False
    assert frame1["face"]["reason"] == health["face_error"] == public_reason
    assert health["error"] is None
    assert "boom" in caplog.text


def test_corrupt_ml_model_does_not_stop_websocket_or_pose(tmp_path, caplog):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"")
    runner, _ = build(model_path=model_path)
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        assert socket.receive_json()["ml_classes"] == []
        receive_initial_record_status(socket)
        assert receive_until(socket, lambda item: item["type"] == "error") == {
            "type": "error",
            "fatal": False,
            "message": "Não foi possível carregar o modelo de classificação ao vivo.",
        }
        frame = receive_frame(socket)
        assert frame["detected"] is True
        assert frame["ml"] is None
    assert "EOFError" in caplog.text


def test_ml_inference_failure_warns_once_and_pose_keeps_streaming(tmp_path, monkeypatch, caplog):
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    monkeypatch.setattr(
        "backend.web.pipeline.load_ml_model",
        lambda _path, **_kwargs: {"classes": ["idle"], "columns": FEATURE_NAMES},
    )
    calls = 0

    def fail_inference(_bundle, _row):
        nonlocal calls
        calls += 1
        raise RuntimeError("inference test detail")

    monkeypatch.setattr("backend.web.pipeline.ml_predict", fail_inference)
    runner, _ = build(model_path=model_path)
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        assert socket.receive_json()["ml_classes"] == ["idle"]
        receive_initial_record_status(socket)
        assert receive_until(socket, lambda item: item["type"] == "error") == {
            "type": "error",
            "fatal": False,
            "message": "A classificação ao vivo foi desativada após uma falha de inferência.",
        }
        first = receive_frame(socket)
        second = receive_frame(socket)
        assert second["frame_index"] > first["frame_index"]
        assert second["detected"] is True
        assert first["ml"] is second["ml"] is None
    assert calls == 1
    assert "inference test detail" in caplog.text


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.10", "lan.example"])
def test_hello_web_blocks_non_loopback_host_without_consent(host, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["hello_web.py", "--host", host])
    monkeypatch.setattr(hello_web, "ensure_model", lambda _path: pytest.fail("loaded model"))
    monkeypatch.setattr(hello_web.uvicorn, "run", lambda *_args, **_kwargs: pytest.fail("bound server"))
    with pytest.raises(SystemExit) as exit_info:
        hello_web.main()
    assert exit_info.value.code == 2
    assert "exige --allow-lan" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("host", "bound", "allow_lan"),
    [("127.1.2.3", "127.1.2.3", False), ("::1", "::1", False),
     ("localhost", "127.0.0.1", False), ("0.0.0.0", "0.0.0.0", True)],
)
def test_hello_web_accepts_loopback_or_warns_after_lan_consent(
    host, bound, allow_lan, monkeypatch, capsys
):
    argv = ["hello_web.py", "--host", host]
    if allow_lan:
        argv.append("--allow-lan")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(hello_web, "ensure_model", lambda path: path)
    calls = []
    monkeypatch.setattr(hello_web.uvicorn, "run", lambda _app, **kwargs: calls.append(kwargs))
    assert hello_web.main() == 0
    assert [call["host"] for call in calls] == [bound]
    output = capsys.readouterr()
    assert ("sem autenticação" in output.err) is allow_lan
    if host == "::1":
        assert "http://[::1]:8000" in output.out
