"""Schema-2 live inference tests without a camera or pose model."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from backend.dataset.windows import WINDOW_TRAINING_COLUMNS
from backend.features import FEATURE_NAMES, FrameFeatures
from backend.features.window import WindowFeatures
from backend.ml import DEFAULT_MODEL_PATH, DEFAULT_V2_MODEL_PATH
from backend.dataset.writer import DEFAULT_DATASET_PATH, DEFAULT_V2_DATASET_PATH
from backend.web.pipeline import PipelineRunner, WebConfig


def _frame(index: int, timestamp: float, left_height: float = 0.5) -> FrameFeatures:
    values = {name: 0.0 for name in FEATURE_NAMES}
    for name in (
        "left_wrist_above_shoulder",
        "right_wrist_above_shoulder",
        "wrists_crossed",
    ):
        values[name] = False
    values.update(
        {
            "left_wrist_height": left_height,
            "right_wrist_height": 0.6,
            "left_wrist_offset_x": 0.1,
            "right_wrist_offset_x": -0.1,
        }
    )
    return FrameFeatures(frame_index=index, timestamp=timestamp, **values)


class _StubModel:
    def predict(self, _vector):
        return ["wave"]

    def predict_proba(self, _vector):
        return np.array([[0.1, 0.9]])


def _v2_bundle():
    return {
        "schema_version": 2,
        "model": _StubModel(),
        "columns": WINDOW_TRAINING_COLUMNS,
        "classes": ["arm_raised", "wave"],
    }


def _ready_window() -> tuple[FrameFeatures, object]:
    tracker = WindowFeatures()
    state = None
    for index in range(11):
        timestamp = index / 10
        frame = _frame(index, timestamp, left_height=0.5 + index / 20)
        state = tracker.push(timestamp, frame)
    assert state is not None and state.left is not None and state.right is not None
    return frame, state


def test_web_config_resolves_schema_paths_and_keeps_v1_as_default():
    v1 = WebConfig()
    v2 = WebConfig(model_schema=2)

    assert v1.model_schema == 1
    assert v1.model_path == DEFAULT_MODEL_PATH
    assert v1.dataset_path == DEFAULT_DATASET_PATH
    assert v2.model_path == DEFAULT_V2_MODEL_PATH
    assert v2.dataset_path == DEFAULT_V2_DATASET_PATH
    assert v2.record_labels[-1] == "wave"
    assert "wave" not in v1.record_labels


def test_web_config_allows_schema_specific_path_overrides_and_rejects_unknown_schema(tmp_path):
    model_path = tmp_path / "selected.joblib"
    dataset_path = tmp_path / "selected.csv"
    config = WebConfig(model_schema=2, model_path=model_path, dataset_path=dataset_path)

    assert config.model_path == model_path
    assert config.dataset_path == dataset_path

    try:
        WebConfig(model_schema=3)
    except ValueError as exc:
        assert "model_schema must be 1 or 2" in str(exc)
    else:
        raise AssertionError("unsupported model schema must be rejected")


def test_v2_prediction_waits_for_a_complete_two_arm_window():
    bundle = _v2_bundle()
    features, ready = _ready_window()

    assert PipelineRunner._predict(bundle, features, None, window_state=None) is None
    incomplete = replace(ready, right=None)
    assert PipelineRunner._predict(bundle, features, None, window_state=incomplete) is None

    result = PipelineRunner._predict(bundle, features, None, window_state=ready)
    assert result is not None
    label, probabilities = result
    assert label == "wave"
    assert probabilities == {"arm_raised": 0.1, "wave": 0.9}


def test_v2_prediction_returns_none_when_optional_bundle_is_missing():
    features, ready = _ready_window()

    assert PipelineRunner._predict(None, features, None, window_state=ready) is None


def test_stream_clears_v2_window_history_after_pose_loss(tmp_path, monkeypatch):
    import backend.web.pipeline as pipeline_module

    predictions = []
    resets = []
    current_frame_index = None
    window_seconds = 1.0
    pose_loss_frame_index = 11
    pose_loss_timestamp = pose_loss_frame_index / 10
    first_eligible_timestamp = pose_loss_timestamp + window_seconds

    class SpyWindowFeatures(WindowFeatures):
        instances = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reset_count = 0
            self.instances.append(self)

        def reset(self):
            resets.append(
                (current_frame_index, bool(self._samples["left"] or self._samples["right"]))
            )
            self.reset_count += 1
            super().reset()

    class AlwaysSubscribedRunner(PipelineRunner):
        @property
        def client_count(self):
            return 1

    class Camera:
        fps = 10.0

        def frames(self):
            # A valid window, a complete pose loss, and enough new frames to
            # warm up again. The lost-pose frame must break temporal history.
            flags = [True] * 11 + [False] + [True] * 12
            for index, detected in enumerate(flags):
                yield SimpleNamespace(
                    image=np.zeros((2, 2, 3), dtype=np.uint8),
                    detected=detected,
                    frame_index=index,
                    timestamp=index / 10,
                )

    class Detector:
        def detect(self, frame):
            nonlocal current_frame_index
            current_frame_index = frame.frame_index
            return SimpleNamespace(
                detected=frame.detected,
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                inference_ms=0.0,
            )

    class Trail:
        def __init__(self, _length):
            pass

        def add_from(self, *_args):
            pass

        def clear(self):
            pass

    monkeypatch.setattr(pipeline_module, "WindowFeatures", SpyWindowFeatures)
    monkeypatch.setattr(pipeline_module, "Trail", Trail)
    monkeypatch.setattr(
        pipeline_module,
        "extract",
        lambda snapshot, _threshold: (
            _frame(snapshot.frame_index, snapshot.timestamp)
            if snapshot.detected
            else FrameFeatures(
                frame_index=snapshot.frame_index,
                timestamp=snapshot.timestamp,
            )
        ),
    )
    monkeypatch.setattr(pipeline_module, "encode_preview", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        pipeline_module,
        "frame_message",
        lambda **kwargs: {"type": "frame", "timings": kwargs["timings"]},
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_ml_model",
        lambda *_args, **_kwargs: {
            **_v2_bundle(),
            "window_seconds": window_seconds,
            "max_gap": 0.30,
            "min_samples": 4,
            "deadband": 0.35,
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "ml_predict",
        lambda _bundle, row: predictions.append(row["timestamp"]) or ("wave", {}),
    )

    model_path = tmp_path / "v2-model.joblib"
    model_path.touch()
    runner = AlwaysSubscribedRunner(WebConfig(model_schema=2, model_path=model_path))
    try:
        runner._stream(Camera(), Detector())
    finally:
        runner.stop()

    assert (pose_loss_frame_index, True) in resets
    assert any(timestamp <= 1.0 for timestamp in predictions)
    # Pre-loss predictions are valid. After pose loss, there must be no
    # prediction until a full configured window has elapsed.
    assert not any(
        pose_loss_timestamp < timestamp < first_eligible_timestamp
        for timestamp in predictions
    )
    after_loss = [
        timestamp for timestamp in predictions if timestamp >= first_eligible_timestamp
    ]
    assert after_loss
    assert min(after_loss) >= first_eligible_timestamp
