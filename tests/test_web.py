"""Web App tests — the whole stack, with no webcam, no model and no browser.

The pipeline is driven by a fake capture and a fake landmarker, so these tests
assert what actually reaches a client: the handshake, the payload, and the
camera lifecycle the privacy rule depends on.
"""

from __future__ import annotations

import asyncio
import csv
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="install the 'web' extra: uv sync --extra web")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from backend.camera import CameraConfig, CameraEngine  # noqa: E402
from backend.dataset import DATASET_COLUMNS, DEFAULT_LABELS, DatasetWriter  # noqa: E402
from backend.dataset.stats import dataset_stats  # noqa: E402
from backend.features import (  # noqa: E402
    FEATURE_NAMES,
    ArmMotion,
    Direction,
    FrameFeatures,
    MotionState,
)
from backend.ml import MIN_SAMPLES_PER_CLASS, train  # noqa: E402
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
    opened: bool = True, face_detector_factory=None, **overrides
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
        return PoseDetector(landmarker_factory=lambda _cfg: FakeLandmarker())

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
    return runner, captures


def client_for(runner: PipelineRunner) -> TestClient:
    return TestClient(create_app(runner=runner))


def train_tiny_model(tmp_path):
    """A minimal real model, trained through the real Sprint 07/08 code —
    not a mock — so the wiring under test is the same as production.

    Two session_ids (the grouped train/test split needs more than one -- see
    backend/ml/train.py) and every motion column filled with a real value:
    `load_dataset` now drops a row it cannot fully populate rather than
    silently treating a missing column as 0."""
    dataset_path = tmp_path / "dataset.csv"
    still = ArmMotion(
        velocity_y=0.0, velocity_x=0.0, speed=0.0,
        elbow_velocity=0.0, direction=Direction.STILL, moving=False,
    )
    with DatasetWriter(dataset_path) as writer:
        for index in range(MIN_SAMPLES_PER_CLASS * 2):
            # Split by half, not by parity: label already alternates by
            # parity, so a session assignment on the same parity would give
            # each session only one label -- a grouped test fold with a
            # single class confuses sklearn's confusion_matrix.
            writer.session_id = f"test-{index // MIN_SAMPLES_PER_CLASS}"
            label = "arms_open" if index % 2 == 0 else "idle"
            wrist_distance = 2.8 if label == "arms_open" else 1.0
            values = {name: 0.0 for name in FEATURE_NAMES}
            for name in ("wrists_crossed", "left_wrist_above_shoulder", "right_wrist_above_shoulder"):
                values[name] = False
            values["wrist_distance"] = wrist_distance
            features = FrameFeatures(frame_index=index, timestamp=index / 30.0, **values)
            motion = MotionState(
                frame_index=index, timestamp=features.timestamp, left=still, right=still
            )
            writer.write(label, features, motion)

    model_path = tmp_path / "model.joblib"
    train(dataset_path=dataset_path, model_path=model_path)
    return model_path


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def receive_initial_record_status(socket) -> dict:
    """Read the mandatory recorder snapshot that follows every hello."""
    status = socket.receive_json()
    assert status["type"] == "record_status"
    return status


def receive_frame(socket) -> dict:
    """Skip any queued lifecycle messages and return the next camera frame."""
    while True:
        message = socket.receive_json()
        if message.get("type") == "frame":
            return message


def receive_until(
    socket, predicate, timeout: float = 5.0, observed: list | None = None
) -> dict:
    """Wait through coalesced frames/statuses for a specific wire event."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        message = socket.receive_json()
        last = message
        if observed is not None:
            observed.append(message)
        if predicate(message):
            return message
    raise AssertionError(f"expected WebSocket message did not arrive before timeout; last={last}")


def stub_preview_encoding(monkeypatch) -> None:
    """Keep recorder WebSocket tests focused on recording, not OpenCV codecs."""
    monkeypatch.setattr(
        "backend.web.pipeline.encode_preview",
        lambda _image, width=640, quality=70: {
            "base64": "ZmFrZQ==",
            "width": width,
            "height": 240,
        },
    )


# --- handshake ---------------------------------------------------------


def test_hello_arrives_before_any_frame():
    """The client cannot draw a bone before it knows the bone table."""
    runner, _ = build()
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        hello = socket.receive_json()
        record_status = receive_initial_record_status(socket)
    runner.stop()

    assert hello["type"] == "hello"
    assert len(hello["chains"]) == len(ARM_CHAINS)
    assert hello["feature_names"] == list(FEATURE_NAMES)
    assert hello["arm_landmarks"] == [int(lm) for lm in ARM_LANDMARKS]
    assert "arm_raised" in hello["gesture_names"]
    assert hello["record_labels"] == list(DEFAULT_LABELS)
    assert record_status == {
        "type": "record_status",
        "recording": False,
        "label": None,
        "session_id": None,
        "written": 0,
        "skipped": 0,
        "owner": False,
        "error": None,
    }


def test_hello_carries_the_face_overlay_topology_regardless_of_face_enabled():
    """Sprint 11: the face overlay's topology is static domain knowledge (like
    `chains` for arms) -- it travels even when `face_enabled` is False, the
    same way `chains`/`arm_landmarks` do not depend on the pose model having
    loaded yet. The client draws nothing without a `frame.face.landmarks`
    dict regardless, so shipping the topology up front is harmless."""
    runner, _ = build()  # face_enabled defaults to False
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        hello = socket.receive_json()
        receive_initial_record_status(socket)
    runner.stop()

    assert hello["face_enabled"] is False
    assert set(hello["face_landmarks"]) == set(FACE_OVERLAY_LANDMARKS)
    assert set(hello["face_chains"]) == set(FACE_CHAIN_GROUPS)
    for name, pairs in hello["face_chains"].items():
        assert pairs == [[a, b] for a, b in FACE_CHAIN_GROUPS[name]]
    assert isinstance(hello["face_mesh_version"], int) and hello["face_mesh_version"] >= 1


def test_failed_initial_record_status_send_unsubscribes_client():
    """A failed snapshot send must not leave a camera-owning subscriber."""
    runner, _ = build()
    app = create_app(runner=runner)
    websocket_route = next(route for route in app.routes if getattr(route, "path", None) == "/ws")

    class FailingSocket:
        async def accept(self):
            pass

        async def send_json(self, message):
            if message.get("type") == "record_status":
                raise RuntimeError("initial record status send failed")

    try:
        with pytest.raises(RuntimeError, match="initial record status send failed"):
            asyncio.run(websocket_route.endpoint(FailingSocket()))

        assert runner.client_count == 0
    finally:
        runner.stop()


def test_second_cancellation_during_child_cleanup_still_unsubscribes(monkeypatch):
    """Cancellation of gather in finally must not skip subscription cleanup."""

    async def scenario():
        runner, _ = build()
        app = create_app(runner=runner)
        websocket_route = next(
            route for route in app.routes if getattr(route, "path", None) == "/ws"
        )
        children_started = asyncio.Event()
        child_cleanup_started = asyncio.Event()
        started_count = 0

        async def stalled_child(*_args):
            nonlocal started_count
            started_count += 1
            if started_count == 2:
                children_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                child_cleanup_started.set()
                await asyncio.Future()

        monkeypatch.setattr("backend.web.app._write_frames", stalled_child)
        monkeypatch.setattr("backend.web.app._read_client_messages", stalled_child)

        class Socket:
            async def accept(self):
                pass

            async def send_json(self, _message):
                pass

        task = asyncio.create_task(websocket_route.endpoint(Socket()))
        try:
            await asyncio.wait_for(children_started.wait(), timeout=2.0)
            task.cancel()
            await asyncio.wait_for(child_cleanup_started.wait(), timeout=2.0)
            task.cancel()  # interrupts the gather inside stream()'s finally
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runner.client_count == 0
        finally:
            runner.stop()

    asyncio.run(scenario())


# --- browser recorder ---------------------------------------------------


def test_websocket_recording_appends_schema_rows_and_uses_a_new_session_id(tmp_path, monkeypatch):
    stub_preview_encoding(monkeypatch)
    dataset_path = tmp_path / "dataset.csv"
    runner, _ = build(dataset_path=dataset_path, preview_width=320)
    sessions = []
    session_counts = []
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        receive_frame(socket)  # let lazy pipeline setup finish before timing a recording

        for _ in range(2):
            socket.send_json({"type": "record_start", "label": "arm_raised"})
            started = receive_until(
                socket,
                lambda item: item.get("type") == "record_status" and item["recording"],
            )
            assert started["label"] == "arm_raised"
            assert started["owner"] is True
            written = receive_until(
                socket,
                lambda item: item.get("type") == "record_status"
                and item["recording"]
                and item["written"] >= 3,
            )
            socket.send_json({"type": "record_stop"})
            stopped = receive_until(
                socket,
                lambda item: item.get("type") == "record_status"
                and not item["recording"]
                and item["session_id"] == started["session_id"],
            )
            assert stopped["written"] >= written["written"]
            assert stopped["owner"] is False
            sessions.append(stopped["session_id"])
            session_counts.append(stopped["written"])

        health = client.get("/health").json()
        assert health["recording"] == {
            "recording": False,
            "label": None,
            "session_id": None,
            "written": 0,
            "skipped": 0,
        }
    runner.stop()

    assert sessions[0] != sessions[1]
    with dataset_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert tuple(rows[0]) == DATASET_COLUMNS
    assert {row["session_id"] for row in rows} == set(sessions)
    assert {row["label"] for row in rows} == {"arm_raised"}
    assert len(rows) == sum(session_counts)


def test_websocket_rejects_removed_and_unknown_labels_before_creating_dataset(tmp_path, monkeypatch):
    stub_preview_encoding(monkeypatch)
    dataset_path = tmp_path / "never-created.csv"
    runner, _ = build(dataset_path=dataset_path, preview_width=320)
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        for label in ("wave", "not-a-label"):
            socket.send_json({"type": "record_start", "label": label})
            rejected = receive_until(
                socket,
                lambda item: item.get("type") == "record_status" and item["error"] is not None,
            )
            assert rejected["recording"] is False
            assert rejected["label"] is None
            assert rejected["owner"] is False
            if label == "wave":
                assert "out of the trainable ML vocabulary" in rejected["error"]
            else:
                assert "not a recognized label" in rejected["error"]
        # Invalid messages and JSON do not close the live socket.
        socket.send_text("{")
        socket.send_json({"type": "mirror", "enabled": True})
        assert receive_frame(socket)["type"] == "frame"
    runner.stop()

    assert not dataset_path.exists()


def test_second_tab_conflict_error_is_private_and_active_owner_keeps_recording(tmp_path, monkeypatch):
    stub_preview_encoding(monkeypatch)
    dataset_path = tmp_path / "dataset.csv"
    runner, _ = build(dataset_path=dataset_path, preview_width=320)
    with client_for(runner) as client:
        with client.websocket_connect("/ws") as owner:
            owner.receive_json()  # hello
            receive_initial_record_status(owner)
            receive_frame(owner)
            owner.send_json({"type": "record_start", "label": "idle"})
            active = receive_until(
                owner,
                lambda item: item.get("type") == "record_status" and item["recording"],
            )
            assert active["owner"] is True

            with client.websocket_connect("/ws") as other:
                other.receive_json()  # hello
                joined = receive_initial_record_status(other)
                assert joined["recording"] is True
                assert joined["owner"] is False
                other.send_json({"type": "record_start", "label": "arms_open"})
                rejected = receive_until(
                    other,
                    lambda item: item.get("type") == "record_status" and item["error"] is not None,
                )
                assert rejected["recording"] is True
                assert rejected["label"] == "idle"
                assert rejected["owner"] is False

                other.send_json({"type": "record_stop"})
                refused_stop = receive_until(
                    other,
                    lambda item: item.get("type") == "record_status"
                    and item["error"] is not None,
                )
                assert refused_stop["recording"] is True
                assert refused_stop["label"] == "idle"
                assert refused_stop["owner"] is False

                owner_messages = []
                owner_update = receive_until(
                    owner,
                    lambda item: item.get("type") == "record_status"
                    and item["recording"]
                    and item["written"] > rejected["written"],
                    observed=owner_messages,
                )
                assert owner_update["owner"] is True
                assert owner_update["error"] is None
                assert not any(
                    item.get("type") == "record_status" and item["error"] is not None
                    for item in owner_messages
                )
                assert "already" in rejected["error"].lower() or "active" in rejected["error"].lower()

            owner.send_json({"type": "record_stop"})
            receive_until(
                owner,
                lambda item: item.get("type") == "record_status" and not item["recording"],
            )
    runner.stop()


def test_owner_disconnect_closes_session_and_notifies_other_tab(tmp_path, monkeypatch):
    stub_preview_encoding(monkeypatch)
    runner, _ = build(dataset_path=tmp_path / "dataset.csv", preview_width=320)
    with client_for(runner) as client:
        with client.websocket_connect("/ws") as other:
            other.receive_json()  # hello
            other_status = receive_initial_record_status(other)
            assert other_status["recording"] is False
            receive_frame(other)

            with client.websocket_connect("/ws") as owner:
                owner.receive_json()  # hello
                receive_initial_record_status(owner)
                owner.send_json({"type": "record_start", "label": "arms_crossed"})
                started = receive_until(
                    owner,
                    lambda item: item.get("type") == "record_status" and item["recording"],
                )
                assert started["owner"] is True

            stopped = receive_until(
                other,
                lambda item: item.get("type") == "record_status"
                and not item["recording"]
                and item["session_id"] == started["session_id"],
            )
            assert stopped["label"] == "arms_crossed"
            assert stopped["owner"] is False
    runner.stop()


def test_writer_error_stops_only_recording_and_pipeline_keeps_streaming(tmp_path, monkeypatch):
    stub_preview_encoding(monkeypatch)
    from backend.dataset import DatasetWriterError

    def fail_write(_self, _label, _features, _motion):
        raise DatasetWriterError("forced writer failure")

    monkeypatch.setattr("backend.dataset.writer.DatasetWriter.write", fail_write)
    runner, _ = build(dataset_path=tmp_path / "dataset.csv", preview_width=320)
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        receive_frame(socket)
        socket.send_json({"type": "record_start", "label": "idle"})
        failed = receive_until(
            socket,
            lambda item: item.get("type") == "record_status"
            and not item["recording"]
            and item["error"] is not None,
        )
        assert "forced writer failure" in failed["error"]
        frame = receive_frame(socket)
        assert frame["type"] == "frame"
        assert runner.state == "live"
    runner.stop()


# --- payload -----------------------------------------------------------


def test_frame_carries_pixels_joints_and_numbers_together():
    runner, _ = build()
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        frame = receive_frame(socket)
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
        receive_initial_record_status(socket)
        frame = receive_frame(socket)
    runner.stop()

    assert set(frame["landmarks"]) == {str(int(lm)) for lm in ARM_LANDMARKS}


def test_coordinates_reach_the_browser_unmirrored():
    """Mirroring here would swap the arms — it stays a display concern."""
    runner, _ = build()
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()
        receive_initial_record_status(socket)
        frame = receive_frame(socket)
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
        receive_initial_record_status(socket)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            frame = receive_frame(socket)
            if frame["gestures"]:
                fired = frame["gestures"]
                break
    runner.stop()

    assert fired is not None, "arm_raised never fired within 5s of a sustained pose"
    assert fired[0]["name"] == "arm_raised"
    assert fired[0]["side"] == "left"


def test_ml_signal_is_null_when_no_model_is_trained(tmp_path):
    runner, _ = build(model_path=tmp_path / "missing.joblib")
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        hello = socket.receive_json()
        receive_initial_record_status(socket)
        frame = receive_frame(socket)
    runner.stop()

    assert hello["ml_classes"] == []
    assert frame["ml"] is None


def test_ml_signal_carries_a_real_prediction_when_a_model_is_trained(tmp_path):
    """Sprint 11: `train_tiny_model` trains through the real `train()`, whose
    `TRAINING_COLUMNS` is `FEATURE_NAMES` alone -- purely postural, no motion.
    `PipelineRunner._predict` reads that off the bundle's own `columns` and no
    longer waits on `MotionTracker` to warm up, so the very first frame (which
    already has complete static features -- see `FakeLandmarker`) must already
    carry a prediction, not just "eventually once motion completes"."""
    model_path = train_tiny_model(tmp_path)
    runner, _ = build(model_path=model_path)
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        hello = socket.receive_json()
        receive_initial_record_status(socket)
        frame = receive_frame(socket)
    runner.stop()

    assert set(hello["ml_classes"]) == {"arms_open", "idle"}
    ml = frame["ml"]
    assert ml is not None, "static bundle should predict on the first frame, motion or not"
    assert ml["label"] in ("arms_open", "idle")
    assert 0.0 < ml["confidence"] <= 1.0
    assert sum(ml["probabilities"].values()) == pytest.approx(1.0, abs=1e-2)


# --- ML prediction gate (Sprint 11) -------------------------------------


def _complete_features(frame_index: int = 0, timestamp: float = 0.0) -> FrameFeatures:
    """A `FrameFeatures` with every `FEATURE_NAMES` column filled -- the
    shape `train_tiny_model` writes, minus the CSV/dataset plumbing."""
    values: dict = {name: 0.0 for name in FEATURE_NAMES}
    for name in ("wrists_crossed", "left_wrist_above_shoulder", "right_wrist_above_shoulder"):
        values[name] = False
    return FrameFeatures(frame_index=frame_index, timestamp=timestamp, **values)


def _complete_motion(frame_index: int = 0, timestamp: float = 0.0) -> MotionState:
    """A `MotionState` with every velocity column known -- `MotionTracker`
    fully warmed up, both arms still."""
    still = ArmMotion(
        velocity_y=0.0, velocity_x=0.0, speed=0.0,
        elbow_velocity=0.0, direction=Direction.STILL, moving=False,
    )
    return MotionState(frame_index=frame_index, timestamp=timestamp, left=still, right=still)


class _StubModel:
    """Minimal `predict`/`predict_proba` stand-in -- the gate under test
    never needs a real Random Forest, only something `ml_predict` can call."""

    def __init__(self, label: str, classes: tuple[str, ...]):
        self._label = label
        self._classes = classes

    def predict(self, vector):
        return [self._label]

    def predict_proba(self, vector):
        return np.array([[1.0 if name == self._label else 0.0 for name in self._classes]])


def _bundle(columns: tuple[str, ...]) -> dict:
    classes = ("arms_open", "idle")
    return {"model": _StubModel("idle", classes), "columns": columns, "classes": list(classes)}


def test_predict_returns_none_without_a_bundle():
    result = PipelineRunner._predict(None, _complete_features(), MotionState(0, 0.0))
    assert result is None


def test_predict_infers_on_a_static_bundle_even_with_incomplete_motion():
    """The current bundle only lists `FEATURE_NAMES` -- `motion.complete`
    must not gate it, so the very first frame (complete features, motion
    that has not warmed up yet) still gets a prediction."""
    bundle = _bundle(FEATURE_NAMES)
    incomplete_motion = MotionState(frame_index=0, timestamp=0.0)  # both arms default: None
    assert incomplete_motion.complete is False

    result = PipelineRunner._predict(bundle, _complete_features(), incomplete_motion)

    assert result is not None
    label, probabilities = result
    assert label == "idle"
    assert probabilities == {"arms_open": 0.0, "idle": 1.0}


def test_predict_returns_none_for_incomplete_static_features():
    """Bundle asking only for static columns, but the frame itself is
    missing one -- gated the same as before Sprint 11, motion irrelevant."""
    bundle = _bundle(FEATURE_NAMES)
    incomplete_features = FrameFeatures(frame_index=0, timestamp=0.0)  # every field None

    result = PipelineRunner._predict(bundle, incomplete_features, _complete_motion())

    assert result is None


def test_predict_still_gates_a_legacy_bundle_on_incomplete_motion():
    """A pre-Sprint-11 bundle can still list a motion column in its own
    `columns` (`backend/ml/train.py:KNOWN_TRAINING_COLUMNS` keeps those
    loadable) -- `_predict` must read that off the bundle, not assume every
    bundle is the current static-only shape, and keep skipping until motion
    warms up."""
    legacy_columns = FEATURE_NAMES + ("left_velocity_y",)
    bundle = _bundle(legacy_columns)
    incomplete_motion = MotionState(frame_index=0, timestamp=0.0)

    gated = PipelineRunner._predict(bundle, _complete_features(), incomplete_motion)
    assert gated is None, "legacy bundle asking for motion must stay gated until it warms up"

    ready = PipelineRunner._predict(bundle, _complete_features(), _complete_motion())
    assert ready is not None, "once motion completes, the legacy bundle predicts like before"


def test_preview_is_downscaled_but_inference_is_not():
    """The camera gives 320 px; the browser gets 160 and the pose still ran."""
    runner, _ = build()
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()
        receive_initial_record_status(socket)
        frame = receive_frame(socket)
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


def test_dataset_stats_route_returns_the_shared_stats_for_its_configured_path(tmp_path):
    """The endpoint exposes the same on-disk snapshot as the CLI/report code."""
    dataset_path = tmp_path / "missing.csv"

    with TestClient(create_app(WebConfig(dataset_path=dataset_path))) as client:
        response = client.get("/dataset/stats")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == dataset_stats(dataset_path)
    assert response.json()["exists"] is False


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
            receive_initial_record_status(first)
            receive_initial_record_status(second)
            assert receive_frame(first)["type"] == "frame"
            assert receive_frame(second)["type"] == "frame"
            assert runner.client_count == 2
    runner.stop()

    assert len(captures) == 1


# --- failures ----------------------------------------------------------


def test_camera_failure_reaches_the_browser_instead_of_hanging():
    """`max_camera_retries=0`: a permission-denied camera is never coming back
    on its own, so this pins the pre-retry behaviour -- one attempt, one
    fatal error. The retry/backoff path itself is `test_camera_recovers_...`
    below."""
    runner, _ = build(opened=False, max_camera_retries=0)
    with client_for(runner) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        receive_initial_record_status(socket)
        message = socket.receive_json()
    runner.stop()

    assert message["type"] == "error"
    assert "System Settings" in message["message"]  # the macOS permission hint
    assert runner.state == "failed"


def test_health_reports_why_the_stream_is_not_running():
    runner, _ = build(opened=False, max_camera_retries=0)
    with client_for(runner) as client:
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.receive_json()
        health = client.get("/health").json()
    runner.stop()

    assert health["pipeline"] == "failed"
    assert "System Settings" in health["error"]


def test_a_late_subscriber_gets_the_stored_fatal_error_instead_of_hanging():
    """A client that connects while the pipeline thread is still alive but has
    already given up (`state == "failed"`) must see that error too --
    `_ensure_running` correctly refuses to start a second thread here, so
    without this the late subscriber would sit on an empty queue forever."""
    runner, _ = build(opened=False, max_camera_retries=0)
    runner.state = "failed"
    runner.error = "camera permanently denied"
    # Stand-in for "the previous run's thread is still alive/tearing down" --
    # `_ensure_running` must see it as alive and refuse to start a new one.
    runner._thread = threading.Thread(target=time.sleep, args=(5,), daemon=True)
    runner._thread.start()

    async def scenario():
        subscriber = runner.subscribe(asyncio.get_running_loop())
        return await asyncio.wait_for(subscriber.get(), timeout=1.0)

    try:
        message = asyncio.run(scenario())
    finally:
        runner._thread = None  # the real background thread outlives this test harmlessly

    assert message["type"] == "error"
    assert message["message"] == "camera permanently denied"


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
            if message["type"] == "error":
                break
    runner.stop()

    assert message is not None and message["type"] == "error"
    assert "giving up" in message["message"]
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
                pytest.fail(f"a transient failure must not reach the browser as fatal: {message}")
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
    assert frame["face"]["reason"] == "face model not found"
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


def test_a_face_runtime_error_degrades_without_stopping_pose_or_the_pipeline():
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
        frame1 = receive_frame(socket)
        frame2 = receive_frame(socket)
    runner.stop()

    assert frame1["type"] == "frame" and frame1["detected"] is True
    assert frame2["type"] == "frame" and frame2["detected"] is True
    assert frame1["face"]["available"] is False
    assert "boom" in frame1["face"]["reason"]
    assert runner.face_error is not None and "boom" in runner.face_error
