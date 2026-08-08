"""Model Training — labeled CSV rows in, a Random Forest bundle out.

Three decisions carry this file:

1. **The input vector is one tuple, imported by both sides.** `TRAINING_COLUMNS`
   is the exact order the model is fit on. Sprint 09 will build the same
   vector from a live frame instead of a CSV row, and if the order there ever
   drifts from the order here the model still runs — it just predicts
   garbage, confidently, with no error to catch it. Fixing that order in one
   importable tuple is the same append-only discipline `FEATURE_NAMES`
   already established.
2. **Direction strings are dropped, velocity is kept.** `left_direction` /
   `right_direction` are already a lossy summary of the signed velocity
   columns (`classify()` in `velocity.py`: positive is up, negative is down).
   Feeding the Forest both would mean training on a derived copy of a raw
   column for no extra separating power, so only the numeric motion columns
   make it into `TRAINING_COLUMNS`.
3. **A dataset that cannot support a real train/test split fails loudly.**
   `MIN_SAMPLES_PER_CLASS` exists because a Random Forest "trained" on three
   examples of a class is not evidence of anything — it is `DatasetError`
   telling you to go record more with `hello_dataset.py`, not a model file
   that quietly can't be trusted.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from backend.dataset.writer import DEFAULT_DATASET_PATH, MOTION_COLUMNS
from backend.features.features import FEATURE_NAMES

DEFAULT_MODEL_PATH = Path("backend/models/gesture_classifier.joblib")

#: Below this many examples of a class, a train/test split cannot say anything
#: trustworthy about it.
MIN_SAMPLES_PER_CLASS = 10

#: Direction is excluded — see decision 2 above.
NUMERIC_MOTION_COLUMNS: tuple[str, ...] = tuple(
    name for name in MOTION_COLUMNS if not name.endswith("_direction")
)

#: The exact feature vector, in the exact order, every saved model commits to.
TRAINING_COLUMNS: tuple[str, ...] = FEATURE_NAMES + ("shoulder_width",) + NUMERIC_MOTION_COLUMNS


class DatasetError(RuntimeError):
    """Raised when there is not enough labeled data to train on."""


class ModelError(RuntimeError):
    """Raised when there is no trained model to load."""


def _to_number(value: Any) -> float:
    """One cell, any shape it might arrive in, to a float the Forest can use.

    Accepts both a CSV row's strings (loading `dataset.csv`) and a live
    Python dict's native types (what Sprint 09 will hand in from a frame) so
    the same function serves training and inference.
    """
    if value is None or value == "":
        return 0.0  # missing velocity: no signal yet, treated as "not moving"
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if value in ("True", "False"):
        return 1.0 if value == "True" else 0.0
    return float(value)


def build_vector(row: dict) -> list[float]:
    """One row (CSV dict or live feature dict) -> the model's input vector."""
    return [_to_number(row[name]) for name in TRAINING_COLUMNS]


@dataclass
class Dataset:
    X: list[list[float]]
    y: list[str]
    counts: dict[str, int]


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> Dataset:
    """Read `dataset.csv` into vectors + labels, or fail with a next step."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"No dataset at {path}. Record samples first: "
            "uv run python scripts/hello_dataset.py"
        )

    X: list[list[float]] = []
    y: list[str] = []
    counts: dict[str, int] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            X.append(build_vector(row))
            y.append(row["label"])
            counts[row["label"]] = counts.get(row["label"], 0) + 1

    if not counts:
        raise DatasetError(
            f"{path} has no rows yet. Record samples first: "
            "uv run python scripts/hello_dataset.py"
        )

    too_few = {label: n for label, n in counts.items() if n < MIN_SAMPLES_PER_CLASS}
    if too_few:
        shown = ", ".join(f"{label}={n}" for label, n in too_few.items())
        raise DatasetError(
            f"Not enough samples per class (need >= {MIN_SAMPLES_PER_CLASS}): {shown}. "
            "Record more with uv run python scripts/hello_dataset.py"
        )
    return Dataset(X=X, y=y, counts=counts)


@dataclass
class TrainingReport:
    model_path: Path
    n_samples: int
    counts: dict[str, int]
    accuracy: float
    report: str
    feature_importances: dict[str, float]
    trained_at: str
    #: Rows/cols both ordered by `class_labels` — the confusion matrix on its
    #: own is just a grid of integers otherwise.
    class_labels: list[str]
    confusion_matrix: list[list[int]]
    #: precision/recall/f1/support per class, from the same held-out split
    #: the confusion matrix and accuracy came from.
    per_class: dict[str, dict[str, float]]


def train(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 200,
) -> TrainingReport:
    """Load, split, fit, evaluate, save. One call, one artifact."""
    dataset = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.y,
    )

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    class_labels = model.classes_.tolist()
    matrix = confusion_matrix(y_test, predicted, labels=class_labels).tolist()
    per_class_report = classification_report(
        y_test, predicted, labels=class_labels, zero_division=0, output_dict=True
    )
    per_class = {label: per_class_report[label] for label in class_labels}
    feature_importances = dict(zip(TRAINING_COLUMNS, model.feature_importances_.tolist()))
    accuracy = accuracy_score(y_test, predicted)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trained_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    # Model, column order and class list travel together in one file — Sprint
    # 09 loads all three from the same artifact, never re-derives them, so
    # the vector it builds at inference time cannot silently drift from the
    # one this model was actually trained on. The evaluation numbers ride
    # along too (Sprint 10): `scripts/report.py` regenerates the HTML report
    # from `load_model()` alone, without re-running training.
    joblib.dump(
        {
            "model": model,
            "columns": TRAINING_COLUMNS,
            "classes": class_labels,
            "trained_at": trained_at,
            "n_samples": len(dataset.y),
            "counts": dataset.counts,
            "confusion_matrix": matrix,
            "per_class": per_class,
            "feature_importances": feature_importances,
            "accuracy": accuracy,
        },
        model_path,
    )

    return TrainingReport(
        model_path=model_path,
        n_samples=len(dataset.y),
        counts=dataset.counts,
        accuracy=accuracy,
        report=classification_report(y_test, predicted, labels=class_labels, zero_division=0),
        feature_importances=feature_importances,
        trained_at=trained_at,
        class_labels=class_labels,
        confusion_matrix=matrix,
        per_class=per_class,
    )


def load_model(path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ModelError(
            f"No trained model at {path}. Train one first: "
            "uv run python scripts/train_model.py"
        )
    return joblib.load(path)


def predict(bundle: dict[str, Any], row: dict) -> tuple[str, dict[str, float]]:
    """One row in (CSV dict or live feature dict), a label and the per-class
    probability distribution out. `row` must use the same keys `TRAINING_COLUMNS`
    names — `bundle["columns"]` is what a caller should build it against."""
    vector = [build_vector(row)]
    model: RandomForestClassifier = bundle["model"]
    label = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    return label, dict(zip(bundle["classes"], probabilities.tolist()))
