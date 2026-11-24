"""Deterministic live evaluation tests — no camera, stub model bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.dataset.windows import WINDOW_TRAINING_COLUMNS
from backend.features import FEATURE_NAMES, FrameFeatures
from backend.features.velocity import Direction, MotionConfig, MotionTracker
from backend.features.window import WindowFeatures
from backend.gestures import GestureConfig, GestureEngine, GestureEvent
from backend.ml.live_eval import (
    COMMON_V1_LABELS,
    LiveEvaluationSession,
    SegmentSpec,
    WAVE_TEMPORAL_NOTE,
    evaluate_segments,
    summarize_segment,
    validate_bundles,
    _rate,
)
from backend.ml.train import ModelError, train
from backend.web.pipeline import PipelineRunner


def _complete_frame(index: int, timestamp: float, **overrides) -> FrameFeatures:
    values = {name: 0.0 for name in FEATURE_NAMES}
    for name in (
        "left_wrist_above_shoulder",
        "right_wrist_above_shoulder",
        "wrists_crossed",
    ):
        values[name] = False
    values.update(
        wrist_distance=1.0,
        left_wrist_height=0.2,
        right_wrist_height=0.2,
        left_wrist_offset_x=0.0,
        right_wrist_offset_x=0.0,
    )
    values.update(overrides)
    return FrameFeatures(frame_index=index, timestamp=timestamp, **values)


class _LabelModel:
    def __init__(self, label: str, classes: list[str]):
        self._label = label
        self.classes_ = np.array(classes)

    def predict(self, _row):
        return np.array([self._label])

    def predict_proba(self, _row):
        probs = np.zeros(len(self.classes_))
        idx = list(self.classes_).index(self._label)
        probs[idx] = 1.0
        return probs.reshape(1, -1)


def _v1_bundle(label: str = "idle"):
    return {
        "schema_version": 1,
        "model": _LabelModel(label, list(COMMON_V1_LABELS)),
        "columns": list(FEATURE_NAMES),
        "classes": list(COMMON_V1_LABELS),
    }


def _v2_bundle(label: str = "idle"):
    return {
        "schema_version": 2,
        "model": _LabelModel(label, ["arm_raised", "wave", "idle"]),
        "columns": WINDOW_TRAINING_COLUMNS,
        "classes": ["arm_raised", "wave", "idle"],
        "window_seconds": 1.0,
        "max_gap": 0.3,
        "min_samples": 4,
        "deadband": 0.35,
    }


def test_rate_null_when_denominator_zero():
    assert _rate(0, 0) is None


def test_validate_bundles_rejects_missing(tmp_path):
    missing = tmp_path / "nope.joblib"
    with pytest.raises(ModelError) as exc:
        validate_bundles(missing, missing)
    assert "inválido" in str(exc.value).lower() or "Modelo" in str(exc.value)


def test_validate_bundles_accepts_trained_pair(tmp_path):
    from tests.test_train import build_synthetic_dataset, build_temporal_wave_dataset

    v1_path, v2_path = tmp_path / "v1.joblib", tmp_path / "v2.joblib"
    v1_csv = build_synthetic_dataset(tmp_path, per_class=20)
    v2_csv = build_temporal_wave_dataset(tmp_path, sessions=4)
    train(dataset_path=v1_csv, model_path=v1_path, schema=1, n_estimators=10)
    train(dataset_path=v2_csv, model_path=v2_path, schema=2, n_estimators=10)
    v1, v2 = validate_bundles(v1_path, v2_path)
    assert v1.get("schema_version", 1) == 1
    assert v2["schema_version"] == 2


def test_session_aligns_v1_rule_on_idle_frames():
    session = LiveEvaluationSession(_v1_bundle("idle"), _v2_bundle("idle"))
    frames = [(_complete_frame(i, i / 10.0), True) for i in range(5)]
    evaluated = session.run_frames(frames)
    assert all(f.v1_label == "idle" for f in evaluated)
    assert all(f.rule_static == "idle" for f in evaluated)
    pair = summarize_segment(
        evaluated,
        SegmentSpec(label="idle", start_index=0, end_index=4),
    )
    assert pair["agreement"]["v1_rule"]["comparable"] == 5
    assert pair["agreement"]["v1_rule"]["rate"] == 1.0


def test_warmup_and_transition_excluded_from_agreement():
    session = LiveEvaluationSession(_v1_bundle("idle"), _v2_bundle("arm_raised"))
    frames = [(_complete_frame(i, i / 10.0), True) for i in range(6)]
    evaluated = session.run_frames(frames)
    evaluated[0].v1_label = "arm_raised"
    evaluated[-1].v1_label = "arm_raised"
    spec = SegmentSpec(label="idle", start_index=0, end_index=5, warmup=1, transition=1)
    summary = summarize_segment(evaluated, spec)
    assert summary["excluded"] == {"warmup": 1, "transition": 1}
    assert summary["agreement"]["v1_v2"]["comparable"] <= 4


def test_wave_coverage_uses_only_eligible_frames():
    session = LiveEvaluationSession(_v1_bundle("idle"), _v2_bundle("wave"))
    evaluated = session.run_frames(
        [(_complete_frame(i, i / 10.0), True) for i in range(5)]
    )
    for frame in evaluated:
        frame.v2_label = "wave"
        frame.gesture_events = [
            GestureEvent("wave", "left", frame.frame_index, frame.timestamp)
        ]

    summary = summarize_segment(
        evaluated,
        SegmentSpec(
            label="wave", start_index=0, end_index=4,
            warmup=1, transition=1, variant="wave_left",
        ),
    )

    assert summary["excluded"] == {"warmup": 1, "transition": 1}
    assert summary["wave"]["v2_labeled_frames"] == 3
    assert summary["wave"]["rule_events"] == 3
    assert [event["frame_index"] for event in summary["wave"]["rule_event_frames"]] == [1, 2, 3]


def test_agreement_null_when_no_overlap():
    session = LiveEvaluationSession(_v1_bundle("idle"), _v2_bundle("idle"))
    for f in session.run_frames([(_complete_frame(0, 0.0), False)]):
        assert f.v1_label is None
    summary = summarize_segment(
        session.run_frames([(_complete_frame(0, 0.0), False)]),
        SegmentSpec(label="idle", start_index=0, end_index=0),
    )
    assert summary["agreement"]["v1_v2"]["rate"] is None
    assert summary["agreement"]["v1_v2"]["comparable"] == 0


def test_wave_note_in_wave_segment():
    session = LiveEvaluationSession(_v1_bundle("idle"), _v2_bundle("wave"))
    frame = _complete_frame(0, 0.0)
    ev_frame = summarize_segment(
        [
            session.evaluate_frame(frame),
        ],
        SegmentSpec(label="wave", start_index=0, end_index=0),
    )
    assert "wave" in ev_frame
    assert WAVE_TEMPORAL_NOTE in ev_frame["wave"]["temporal_unit_note"]


def test_v2_waits_for_window_like_pipeline():
    bundle = _v2_bundle("wave")
    tracker = WindowFeatures(
        window_seconds=1.0,
        max_gap=0.3,
        min_samples=4,
        deadband=0.35,
    )
    frame = _complete_frame(0, 0.0)
    assert PipelineRunner._predict(bundle, frame, None, window_state=None) is None
    state = None
    for index in range(11):
        timestamp = index / 10
        frame = _complete_frame(index, timestamp, left_wrist_height=0.5 + index / 20)
        state = tracker.push(timestamp, frame)
    assert state is not None and state.left is not None and state.right is not None
    result = PipelineRunner._predict(bundle, frame, None, window_state=state)
    assert result is not None
    assert result[0] == "wave"
