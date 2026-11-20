"""Live capture helpers — fake camera/pose only."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from backend.camera import CameraConfig, CameraEngine
from backend.dataset.windows import WINDOW_TRAINING_COLUMNS
from backend.features import FEATURE_NAMES
from backend.gestures import GestureConfig
from backend.ml.live_capture import (
    LABEL_VARIANTS,
    ScriptedCapturePrompter,
    SegmentCaptureMeta,
    build_capture_report,
    parse_expected_label,
    run_capture_session,
    run_interactive_evaluation,
)
from backend.ml.live_eval import LiveEvaluationSession
from backend.vision import PoseDetector, PoseLandmark
from backend.vision.landmarks import PoseSnapshot
from tests.test_capture import FakeCapture
from tests.test_live_eval import _complete_frame, _v1_bundle, _v2_bundle


class FakeLandmarker:
    def detect_for_video(self, _image, _timestamp_ms: int):
        points = [
            SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9, presence=0.9) for _ in range(33)
        ]
        points[int(PoseLandmark.LEFT_WRIST)] = SimpleNamespace(
            x=0.35, y=0.4, z=0.0, visibility=0.9, presence=0.9
        )
        points[int(PoseLandmark.RIGHT_WRIST)] = SimpleNamespace(
            x=0.55, y=0.6, z=0.0, visibility=0.9, presence=0.9
        )
        return SimpleNamespace(pose_landmarks=[points], pose_world_landmarks=[points])

    def close(self) -> None:
        pass


def _fake_camera(total_frames: int) -> CameraEngine:
    script = [True] * total_frames
    capture = FakeCapture(script)
    return CameraEngine(config=CameraConfig(), capture_factory=lambda _cfg: capture)


def _fake_detector() -> PoseDetector:
    return PoseDetector(landmarker_factory=lambda _cfg: FakeLandmarker())


def test_parse_expected_label_variants():
    assert parse_expected_label("wave_left") == ("wave", "wave_left")
    assert parse_expected_label("arm_raised_no_reversal") == (
        "arm_raised",
        "arm_raised_no_reversal",
    )
    assert parse_expected_label("idle") == ("idle", "")
    with pytest.raises(ValueError):
        parse_expected_label("desconhecido")


def test_capture_session_single_stream_two_segments():
    plan = [
        SegmentCaptureMeta("idle_ref", "idle", frame_count=5, warmup=1),
        SegmentCaptureMeta(
            "wave_esq",
            "wave",
            variant="wave_left",
            frame_count=4,
            transition=1,
        ),
    ]
    prompter = ScriptedCapturePrompter(list(plan))
    session = LiveEvaluationSession(_v1_bundle(), _v2_bundle())
    camera = _fake_camera(9)
    evaluated, specs = run_capture_session(
        camera,
        _fake_detector(),
        session,
        prompter,
    )
    assert len(evaluated) == 9
    assert len(specs) == 2
    assert specs[0].label == "idle" and specs[0].warmup == 1
    assert specs[1].variant == "wave_left" and specs[1].transition == 1
    report = build_capture_report(evaluated, specs)
    assert report["schema_version"] == 1
    assert len(report["segments"]) == 2
    assert "agreement" in report["segments"][0]
    assert "expected_accuracy" in report["segments"][1]


def test_validate_bundles_runs_before_capture_interface(monkeypatch, tmp_path):
    order: list[str] = []

    def fake_validate(v1, v2):
        order.append("validate")
        return _v1_bundle(), _v2_bundle()

    def fake_run(*_args, **_kwargs):
        order.append("capture")
        return {"schema_version": 1, "segments": []}

    monkeypatch.setattr("scripts.evaluate_live.validate_bundles", fake_validate)
    monkeypatch.setattr("scripts.evaluate_live.run_interactive_evaluation", fake_run)

    import scripts.evaluate_live as mod

    code = mod.main(
        [
            "--model-v1",
            str(tmp_path / "v1.joblib"),
            "--model-v2",
            str(tmp_path / "v2.joblib"),
            "--capture",
        ]
    )
    assert code == 0
    assert order == ["validate", "capture"]


def test_capture_is_default_when_no_offline_files_are_given(monkeypatch, tmp_path):
    order: list[str] = []

    monkeypatch.setattr(
        "scripts.evaluate_live.validate_bundles",
        lambda *_args: (_v1_bundle(), _v2_bundle()),
    )
    monkeypatch.setattr(
        "scripts.evaluate_live.run_interactive_evaluation",
        lambda *_args, **_kwargs: order.append("capture") or {"schema_version": 1, "segments": []},
    )

    import scripts.evaluate_live as mod

    code = mod.main(
        ["--model-v1", str(tmp_path / "v1"), "--model-v2", str(tmp_path / "v2")]
    )
    assert code == 0
    assert order == ["capture"]


def test_camera_initialization_error_is_portuguese_and_releases_resources():
    from backend.camera import CameraOpenError
    from backend.ml.live_capture import init_camera_and_pose

    class BrokenCamera:
        closed = False

        def open(self):
            raise CameraOpenError("could not open")

        def close(self):
            self.closed = True

    camera = BrokenCamera()
    with pytest.raises(RuntimeError, match="Não foi possível iniciar a câmera"):
        init_camera_and_pose(camera, _fake_detector())
    assert camera.closed


def test_pose_initialization_error_is_portuguese_and_releases_camera():
    from backend.ml.live_capture import init_camera_and_pose

    class TrackingCamera:
        closed = False

        def open(self):
            return self

        def close(self):
            self.closed = True

    class BrokenDetector:
        def open(self):
            raise RuntimeError("missing pose model")

        def close(self):
            pass

    camera = TrackingCamera()
    with pytest.raises(RuntimeError, match="Não foi possível iniciar o modelo de pose"):
        init_camera_and_pose(camera, BrokenDetector())
    assert camera.closed


def test_offline_mode_rejects_capture_combo(tmp_path, monkeypatch):
    import scripts.evaluate_live as mod

    monkeypatch.setattr(
        "scripts.evaluate_live.validate_bundles",
        lambda v1, v2: (_v1_bundle(), _v2_bundle()),
    )

    frames = tmp_path / "f.json"
    segments = tmp_path / "s.json"
    frames.write_text("[]", encoding="utf-8")
    segments.write_text("[]", encoding="utf-8")

    code = mod.main(
        [
            "--model-v1",
            str(tmp_path / "v1.joblib"),
            "--model-v2",
            str(tmp_path / "v2.joblib"),
            "--capture",
            "--frames",
            str(frames),
        ]
    )
    assert code == 1
