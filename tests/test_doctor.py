from pathlib import Path

from scripts.doctor import diagnose


def test_diagnose_reports_missing_modules_and_models_without_downloading(tmp_path):
    lookups = []

    def missing(module):
        lookups.append(module)
        return None

    checks = diagnose(python_version=(3, 12), spec_lookup=missing, root=tmp_path)

    assert checks[0].available is True
    assert checks[1].available is False
    assert "uv sync --group dev --extra vision" in checks[1].fix
    assert checks[-4].available is False
    assert "downloads" in checks[-4].fix
    assert "ensure_model()" in checks[-4].fix
    assert "requires network" in checks[-4].fix
    assert "scripts/check_pose.py" not in checks[-4].fix
    assert "doctor itself never downloads" in checks[-4].fix
    assert not (tmp_path / "models").exists()
    assert lookups == ["cv2", "numpy", "mediapipe", "fastapi", "uvicorn", "sklearn", "joblib"]


def test_diagnose_marks_present_files_and_supported_dependencies(tmp_path):
    modules = {name: object() for name in ("cv2", "numpy", "mediapipe", "fastapi", "uvicorn", "sklearn", "joblib")}
    lookup = lambda name: modules.get(name)
    pose = tmp_path / "models/pose_landmarker_lite.task"
    pose.parent.mkdir()
    pose.write_bytes(b"fixture")

    checks = diagnose(python_version=(3, 11), spec_lookup=lookup, root=tmp_path)

    assert all(check.available is True for check in checks[:4])
    assert checks[4].available is True
    assert checks[4].fix is None


def test_unsupported_python_has_actionable_fix():
    checks = diagnose(python_version=(3, 13), spec_lookup=lambda _: object())
    assert checks[0].available is False
    assert "python3.12" in checks[0].fix
