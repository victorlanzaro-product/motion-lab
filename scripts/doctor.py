"""Offline first-run checks for Motion Lab. Never opens a camera or downloads files."""
from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

@dataclass(frozen=True)
class Check:
    name: str
    available: bool | None
    detail: str
    fix: str | None = None

PACKAGES = {
    "core": ("cv2", "numpy"),
    "vision": ("mediapipe",),
    "web": ("fastapi", "uvicorn"),
    "ml": ("sklearn", "joblib"),
}
MODELS = (
    ("Pose model", Path("models/pose_landmarker_lite.task"), "First pose use downloads it. To prepare it, run `uv run python scripts/check_pose.py --frames 1` with network access."),
    ("Face model (optional)", Path("models/face_landmarker.task"), "Only needed with `--face`. The first facial-analysis run downloads it."),
    ("Gesture classifier (optional)", Path("backend/models/gesture_classifier.joblib"), "Train one with `uv run python scripts/train_model.py` after collecting labeled data."),
    ("Temporal classifier (optional)", Path("backend/models/gesture_classifier_v2.joblib"), "Train one with `uv run python scripts/train_model.py --schema v2` after collecting session data."),
)

def diagnose(*, python_version: tuple[int, int] | None = None,
              spec_lookup: Callable[[str], object | None] = importlib.util.find_spec,
              root: Path = Path.cwd()) -> list[Check]:
    version = python_version or sys.version_info[:2]
    supported = (3, 10) <= version < (3, 13)
    version_text = f"{version[0]}.{version[1]}"
    checks = [Check("Python", supported, f"Python {version_text}" + (" is supported." if supported else " is outside the supported range 3.10–3.12."), None if supported else "Use Python 3.10, 3.11, or 3.12, then recreate the environment with `uv venv --python python3.12`. ")]
    for extra, modules in PACKAGES.items():
        missing = [module for module in modules if spec_lookup(module) is None]
        checks.append(Check(f"Dependencies ({extra})", not missing,
                            "Installed: " + ", ".join(modules) if not missing else "Missing: " + ", ".join(missing),
                            None if not missing else ("Install core dependencies with `uv sync --group dev --extra vision --extra web --extra ml`." if extra == "core" else f"Install with `uv sync --group dev --extra {extra}`.")))
    for label, relative, fix in MODELS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(Check(label, present, str(relative) + (" is present." if present else " is absent."), None if present else fix))
    return checks

def main() -> int:
    print(f"Motion Lab doctor — {platform.system()} {platform.release()}")
    checks = diagnose()
    for check in checks:
        status = "OK" if check.available is True else "MISSING" if check.available is False else "UNKNOWN"
        print(f"[{status}] {check.name}: {check.detail}")
        if check.fix:
            print(f"  Fix: {check.fix}")
    return 1 if any(check.available is False and check.name in {"Python", "Dependencies (core)"} for check in checks) else 0

if __name__ == "__main__":
    raise SystemExit(main())
