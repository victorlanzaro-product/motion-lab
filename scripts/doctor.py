"""Check local Motion Lab prerequisites without opening hardware or network.

    python3 scripts/doctor.py

The diagnostic only checks Python support, whether dependency modules can be
found, and whether optional model files exist. It never imports those modules,
downloads models, or accesses a camera.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PYTHON = (3, 10), (3, 13)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    action: str | None = None


REQUIRED_MODULES = {
    "numpy": "Install core dependencies with `uv sync --group dev`.",
    "cv2": "Install core dependencies with `uv sync --group dev`.",
}
OPTIONAL_MODULES = {
    "mediapipe": ("vision", "Pose detection needs `uv sync --extra vision`."),
    "fastapi": ("web", "The web app needs `uv sync --extra web`."),
    "uvicorn": ("web", "The web app needs `uv sync --extra web`."),
    "sklearn": ("ml", "ML training/inference needs `uv sync --extra ml`."),
    "joblib": ("ml", "ML training/inference needs `uv sync --extra ml`."),
}
OPTIONAL_FILES = {
    "Pose model": (
        Path("models/pose_landmarker_lite.task"),
        "When using pose detection, run `uv run python -c \"from backend.vision.pose_detector import ensure_model; ensure_model()\"`; it needs network and saves the model locally, without opening a camera.",
    ),
    "Face model (opt-in)": (
        Path("models/face_landmarker.task"),
        "Only when enabling face detection, run `uv run python -c \"from backend.vision.face_detector import ensure_model; ensure_model()\"`; it needs network and saves the model locally.",
    ),
    "Trained v1 classifier": (
        Path("backend/models/gesture_classifier.joblib"),
        "Train a local demonstration model with `uv run python scripts/train_model.py --simulate`; it uses synthetic data and needs the `ml` extra.",
    ),
}


def supported_python(version: tuple[int, int]) -> bool:
    """Return whether a major/minor Python version is supported by pyproject."""
    return SUPPORTED_PYTHON[0] <= version < SUPPORTED_PYTHON[1]


def collect_checks(
    root: Path = PROJECT_ROOT,
    version: tuple[int, int] | None = None,
    module_finder: Callable[[str], object | None] | None = None,
) -> list[Check]:
    """Collect deterministic local checks; injectable inputs keep tests offline."""
    version = version or (sys.version_info.major, sys.version_info.minor)
    module_finder = module_finder or importlib.util.find_spec
    checks: list[Check] = []

    if supported_python(version):
        checks.append(Check("Python", "OK", f"{version[0]}.{version[1]} is supported."))
    else:
        checks.append(
            Check(
                "Python",
                "FAIL",
                f"{version[0]}.{version[1]} is outside the supported range 3.10–3.12.",
                "Run `uv python install 3.12`, then `uv sync --python 3.12 --group dev --extra vision --extra web --extra ml`.",
            )
        )

    for module, action in REQUIRED_MODULES.items():
        if module_finder(module) is None:
            checks.append(Check(module, "FAIL", "Required package is not installed.", action))
        else:
            checks.append(Check(module, "OK", "Required package is installed."))

    for module, (extra, action) in OPTIONAL_MODULES.items():
        if module_finder(module) is None:
            checks.append(
                Check(module, "INFO", f"Optional `{extra}` extra is not installed.", action)
            )
        else:
            checks.append(Check(module, "OK", f"Optional `{extra}` extra is installed."))

    for name, (relative_path, action) in OPTIONAL_FILES.items():
        path = root / relative_path
        if not path.exists():
            checks.append(
                Check(
                    name,
                    "INFO",
                    f"Not present: {relative_path}; no download was attempted.",
                    action,
                )
            )
        elif not path.is_file() or path.stat().st_size == 0:
            checks.append(Check(name, "WARN", f"Present but empty or not a file: {relative_path}."))
        else:
            checks.append(Check(name, "OK", f"Present locally: {relative_path}."))

    return checks


def main() -> int:
    print("Motion Lab local setup check (no camera or network access)")
    checks = collect_checks()
    for check in checks:
        print(f"{check.status:4} {check.name}: {check.detail}")
        if check.action:
            print(f"     Next: {check.action}")

    failures = sum(check.status == "FAIL" for check in checks)
    if failures:
        print(f"\n{failures} required check(s) failed. Optional files/extras may be absent.")
        return 1
    print("\nRequired local checks passed. INFO/WARN entries describe optional components.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
