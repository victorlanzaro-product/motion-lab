from __future__ import annotations

from pathlib import Path

import scripts.doctor as doctor
from scripts.doctor import collect_checks, supported_python


def _installed(_module: str) -> object:
    return object()


def _missing(_module: str) -> None:
    return None


def _optional_missing(module: str) -> object | None:
    return None if module in {"mediapipe", "fastapi", "uvicorn", "sklearn", "joblib"} else object()


def test_python_support_matches_project_range() -> None:
    assert supported_python((3, 10))
    assert supported_python((3, 12))
    assert not supported_python((3, 9))
    assert not supported_python((3, 13))


def test_missing_required_dependency_is_actionable_and_fails() -> None:
    checks = collect_checks(Path("."), (3, 12), _missing)
    numpy = next(check for check in checks if check.name == "numpy")
    assert numpy.status == "FAIL"
    assert "uv sync --group dev" in (numpy.action or "")


def test_absent_optional_models_and_extras_are_actionable_without_failure(tmp_path: Path) -> None:
    checks = collect_checks(tmp_path, (3, 12), _optional_missing)
    assert not any(check.status == "FAIL" for check in checks)
    pose = next(check for check in checks if check.name == "Pose model")
    face = next(check for check in checks if check.name == "Face model (opt-in)")
    assert pose.status == face.status == "INFO"
    assert "no download was attempted" in pose.detail
    assert "without opening a camera" in (pose.action or "")
    assert "uv sync --extra web" in next(
        check.action for check in checks if check.name == "fastapi"
    )


def test_empty_model_is_warning_and_valid_local_model_is_detected(tmp_path: Path) -> None:
    pose = tmp_path / "models/pose_landmarker_lite.task"
    pose.parent.mkdir()
    pose.touch()
    checks = collect_checks(tmp_path, (3, 12), _installed)
    assert next(check for check in checks if check.name == "Pose model").status == "WARN"

    pose.write_bytes(b"synthetic test fixture")
    checks = collect_checks(tmp_path, (3, 12), _installed)
    assert next(check for check in checks if check.name == "Pose model").status == "OK"


def test_main_returns_one_and_prints_action_for_required_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor,
        "collect_checks",
        lambda: [doctor.Check("numpy", "FAIL", "Required package is not installed.", "uv sync")],
    )

    assert doctor.main() == 1
    output = capsys.readouterr().out
    assert "FAIL numpy" in output
    assert "Next: uv sync" in output


def test_main_returns_zero_when_only_optional_items_are_absent(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor,
        "collect_checks",
        lambda: [doctor.Check("Pose model", "INFO", "Not present; no download attempted.")],
    )

    assert doctor.main() == 0
    output = capsys.readouterr().out
    assert "Required local checks passed" in output
