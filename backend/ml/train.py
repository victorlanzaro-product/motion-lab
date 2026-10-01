"""Model Training — labeled CSV rows in, a Random Forest bundle out.

Five decisions carry this file:

1. **The input vector is one tuple, imported by both sides.** `TRAINING_COLUMNS`
   is the exact order the model is fit on. Sprint 09 builds the same vector
   from a live frame instead of a CSV row — `predict()` always rebuilds it
   from `bundle["columns"]`, never from this module's own `TRAINING_COLUMNS`,
   so a model saved before a column change still predicts correctly against
   the vector it actually shipped with (see decision 4).
2. **Motion is out of the training vector entirely (Sprint 11).**
   `DEFAULT_LABELS` (`arm_raised`, `arms_crossed`, `arms_open`, `idle`) is
   purely postural -- every one of them is fully readable from one frame's
   static features -- so `TRAINING_COLUMNS` is exactly `FEATURE_NAMES`, no
   velocity, no direction. `MOTION_COLUMNS` still rides along in
   `dataset.csv` (`backend/dataset/writer.py`) for whatever future label
   genuinely needs arm speed, but the model committed to disk never sees it;
   this also means inference works on the very first complete frame
   (`FrameFeatures.complete`), without waiting for `MotionTracker` to warm up
   (`MotionState.complete`). `shoulder_width` is excluded for the same reason
   `features.py` keeps it out of `FEATURE_NAMES`: it is the ruler every other
   feature is already scaled by, so training on it directly would only teach
   the model the distance it happened to be recorded at.
3. **A dataset that cannot support a real train/test split fails loudly.**
   `MIN_SAMPLES_PER_CLASS` exists because a Random Forest "trained" on three
   examples of a class is not evidence of anything — it is `DatasetError`
   telling you to go record more with `hello_dataset.py`, not a model file
   that quietly can't be trusted. The same discipline applies to the split
   itself (decision 4).
4. **The held-out split is grouped by recording session, never by frame, and
   every class must land on both sides.** Consecutive frames of the same
   `hello_dataset.py` session are near duplicates (same pose held for a
   second, same lighting, same person) -- `train_test_split`'s random row
   shuffle would put near-identical frames on both sides of the split, so the
   reported accuracy would mostly measure "did the model memorise this exact
   session", not "does it generalise to a session it never saw".
   `GroupShuffleSplit` on `session_id` fixes that; a dataset with samples
   from only one session, or a class confined to a single session, cannot be
   split this way at all, so `train()` fails loudly instead of silently
   falling back to a per-frame split. Even past those checks, any *one*
   random split can still get unlucky and strand a whole class on a single
   side -- `_grouped_split` retries a small, deterministic sequence of seeds
   until every class lands on both sides, and fails loudly with an
   actionable message if none of the attempts do.
5. **"wave" cannot be trained here, so a legacy row for it is a hard error,
   not a silent drop.** It is motion, not posture, and this vector cannot
   reproduce the reversal window `GestureEngine`'s `_WaveDetector` actually
   needs to recognize one for real. `backend/dataset/writer.py:REMOVED_LABELS`
   blocks new recordings of it; `load_dataset` below raises `DatasetError`
   with migration steps if it finds the label anyway, in a `dataset.csv`
   written before that rule existed -- never silently trains on it or folds
   it into `dropped`, where a real problem would go unnoticed. The same
   `load_dataset` pass also rejects any label outside `DEFAULT_LABELS`
   entirely (a typo, a stray capitalization, a class that never went through
   that vocabulary) for the same reason.

A missing value is never a silent 0. `_to_number` used to treat an empty CSV
cell (a joint MediaPipe couldn't see, or a motion column with no history yet)
as "0 — not moving", which is a fabricated, confident answer to a question
the row never actually answered (Sprint 03's `None`-never-`0` rule, applied
here too). `load_dataset` now drops a row it cannot fully populate — counted
in `Dataset.dropped`, never fed to the Forest as a guess — and `build_vector`
raises `MissingFeatureError` instead of inventing a number, so a caller that
skips that check (there should not be one) fails loudly rather than training
or predicting on a lie.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

from backend.dataset.writer import (
    DATASET_COLUMNS,
    DEFAULT_DATASET_PATH,
    DEFAULT_LABELS,
    NUMERIC_MOTION_COLUMNS,
    REMOVED_LABELS,
)
from backend.features.features import FEATURE_NAMES

DEFAULT_MODEL_PATH = Path("backend/models/gesture_classifier.joblib")

#: Below this many examples of a class, a train/test split cannot say anything
#: trustworthy about it.
MIN_SAMPLES_PER_CLASS = 10

#: The exact feature vector, in the exact order, every saved model commits to.
#: `shoulder_width` is deliberately absent — see decision 2 above. Sprint 11:
#: motion is out entirely, not just `direction` — the current label set
#: (`DEFAULT_LABELS`) is purely postural, so a per-frame velocity column would
#: only add noise, and dropping it means inference works on the very first
#: complete frame (`FrameFeatures.complete`), without waiting for
#: `MotionTracker` to warm up (`MotionState.complete`).
TRAINING_COLUMNS: tuple[str, ...] = FEATURE_NAMES

#: Column names `load_model` will accept in a bundle's `columns` — the current
#: `TRAINING_COLUMNS` plus the numeric motion columns a pre-Sprint-11 bundle
#: may still carry, so an older model file stays loadable rather than being
#: treated as corrupt just because the trainable vector shrank.
KNOWN_TRAINING_COLUMNS: frozenset[str] = frozenset(FEATURE_NAMES) | frozenset(NUMERIC_MOTION_COLUMNS)


class DatasetError(RuntimeError):
    """Raised when there is not enough labeled data to train on."""


class ModelError(RuntimeError):
    """Raised when there is no trained model to load."""


class MissingFeatureError(RuntimeError):
    """Raised when a required column is missing/empty instead of a fabricated 0."""


def _to_number(value: Any) -> float:
    """One cell, any shape it might arrive in, to a float the Forest can use.

    Accepts both a CSV row's strings (loading `dataset.csv`) and a live
    Python dict's native types (what a frame hands in at inference time) so
    the same function serves training and inference. A missing value
    (`None`, or the empty string a CSV gives an unset column) is never a
    number here — see `MissingFeatureError`.
    """
    if value is None or value == "":
        raise MissingFeatureError("missing value — cannot convert to a training number")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if value in ("True", "False"):
        return 1.0 if value == "True" else 0.0
    return float(value)


def build_vector(row: dict, columns: tuple[str, ...] = TRAINING_COLUMNS) -> list[float]:
    """One row (CSV dict or live feature dict) -> the model's input vector.

    `columns` defaults to this module's own `TRAINING_COLUMNS`, but a caller
    holding a saved model bundle must pass `bundle["columns"]` instead — see
    `predict()`. Raises `MissingFeatureError` if `row` does not have every
    named column filled; it never pads a gap with 0.
    """
    return [_to_number(row.get(name)) for name in columns]


@dataclass
class Dataset:
    X: list[list[float]]
    y: list[str]
    #: One `session_id` per row, same order as `X`/`y` — what the grouped
    #: train/test split in `train()` splits on.
    groups: list[str]
    counts: dict[str, int]
    #: Rows excluded from `X`/`y`, by reason — never silently absorbed into a
    #: fabricated 0. `missing_label` and `missing_feature` are the two ways a
    #: row can be incomplete; `non_finite` catches a NaN/Inf that slipped
    #: through (e.g. a hand-edited cell) before it reaches the Forest.
    dropped: dict[str, int] = field(
        default_factory=lambda: {"missing_label": 0, "missing_feature": 0, "non_finite": 0}
    )


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
    groups: list[str] = []
    counts: dict[str, int] = {}
    dropped = {"missing_label": 0, "missing_feature": 0, "non_finite": 0}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or ())
        missing_columns = set(DATASET_COLUMNS) - header
        if missing_columns:
            raise DatasetError(
                f"{path} is missing column(s) {sorted(missing_columns)} — its header "
                "does not match backend/dataset/writer.py:DATASET_COLUMNS. Re-record "
                "with the current hello_dataset.py rather than hand-editing the CSV."
            )

        for row in reader:
            label = (row.get("label") or "").strip()
            if not label:
                dropped["missing_label"] += 1
                continue
            if label in REMOVED_LABELS:
                raise DatasetError(
                    f"{path} has row(s) labeled {label!r}, which is no longer a "
                    f"trainable class: {REMOVED_LABELS[label]} This CSV predates "
                    "that decision. Migrate before training: drop every row with "
                    f"label=={label!r} (e.g. `df[df.label != {label!r}]` in pandas, "
                    "or filter the CSV by hand) and save the result, or move this "
                    "file aside and record a fresh session with the current "
                    "hello_dataset.py, which no longer offers that label at all."
                )
            if label not in DEFAULT_LABELS:
                raise DatasetError(
                    f"{path} has row(s) labeled {label!r}, which is not one of the "
                    f"current trainable classes {DEFAULT_LABELS} -- a typo, a stray "
                    "capitalization, or a class that never went through "
                    "backend/dataset/writer.py:DEFAULT_LABELS. Migrate before "
                    f"training: drop every row with label=={label!r} and save the "
                    "result, or move this file aside and record a fresh session "
                    "with the current hello_dataset.py."
                )
            try:
                vector = build_vector(row)
            except MissingFeatureError:
                # An occluded joint, or a motion column with no history yet
                # (the first few frames of a held key, before the tracker has
                # enough samples) — a real "we don't know", not "value is 0".
                # Training on a fabricated 0 here would teach the Forest that
                # "unknown" and "not moving" are the same thing.
                dropped["missing_feature"] += 1
                continue
            if not all(math.isfinite(v) for v in vector):
                dropped["non_finite"] += 1
                continue
            X.append(vector)
            y.append(label)
            groups.append(row.get("session_id") or "")
            counts[label] = counts.get(label, 0) + 1

    if not counts:
        raise DatasetError(
            f"{path} has no usable rows (dropped: {dropped}). Record samples first: "
            "uv run python scripts/hello_dataset.py"
        )

    too_few = {label: n for label, n in counts.items() if n < MIN_SAMPLES_PER_CLASS}
    if too_few:
        shown = ", ".join(f"{label}={n}" for label, n in too_few.items())
        raise DatasetError(
            f"Not enough samples per class (need >= {MIN_SAMPLES_PER_CLASS}): {shown}. "
            "Record more with uv run python scripts/hello_dataset.py"
        )
    return Dataset(X=X, y=y, groups=groups, counts=counts, dropped=dropped)


#: Deterministic bound on how many alternate seeds `_grouped_split` tries
#: before giving up on a split with every class on both sides. Fixed and
#: small on purpose (decision 4b below) — a dataset that needs more than this
#: needs more sessions per class, not a cleverer search.
MAX_SPLIT_ATTEMPTS = 20


def _grouped_split(
    dataset: "Dataset", test_size: float, random_state: int
) -> tuple[list[int], list[int]]:
    """`GroupShuffleSplit`, retried over a deterministic sequence of seeds
    until every class has at least one sample on both sides of the split.

    Decision 4b: two sessions overall, and even two sessions per class
    (already enforced by `train()`'s callers before this runs), do not
    guarantee that any *one* random split lands a session from every class on
    both sides — with only a few sessions, one unlucky shuffle can still
    strand a whole class on a single side, making its precision/recall/f1
    meaningless rather than a real held-out measurement (the same problem
    decision 4's per-class-session check exists for, one level deeper).
    Retrying a small, fixed number of deterministic seeds (`random_state`,
    `random_state + 1`, ... — never `random.random()`/wall-clock, so the same
    call always searches the same sequence) finds a workable split when one
    exists, without pretending this is a more sophisticated search than that.
    """
    all_classes = set(dataset.y)
    last_train_classes: set[str] = set()
    last_test_classes: set[str] = set()
    for attempt in range(MAX_SPLIT_ATTEMPTS):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state + attempt
        )
        train_idx, test_idx = next(splitter.split(dataset.X, dataset.y, groups=dataset.groups))
        train_classes = {dataset.y[i] for i in train_idx}
        test_classes = {dataset.y[i] for i in test_idx}
        if train_classes == all_classes and test_classes == all_classes:
            return list(train_idx), list(test_idx)
        last_train_classes, last_test_classes = train_classes, test_classes

    missing_from_train = sorted(all_classes - last_train_classes)
    missing_from_test = sorted(all_classes - last_test_classes)
    raise DatasetError(
        "Could not find a grouped train/test split with every class present on "
        f"both sides after {MAX_SPLIT_ATTEMPTS} deterministic attempts (last "
        f"attempt: missing from train {missing_from_train or 'none'}, missing "
        f"from test {missing_from_test or 'none'}). Record more sessions per "
        "class with hello_dataset.py so a class does not depend on a single "
        "session landing on the right side of the split, or pass a different "
        "test_size."
    )


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
    #: Rows `load_dataset` excluded rather than fabricating — see `Dataset.dropped`.
    dropped: dict[str, int]
    #: Unweighted mean of per-class F1 -- unlike `accuracy`, a class with few
    #: samples counts as much as the biggest one, so a model that nails the
    #: common class and ignores a rare one cannot hide behind a high overall
    #: accuracy. `classification_report`'s own "macro avg" row, not re-derived.
    macro_f1: float
    #: Mean per-class recall -- `sklearn.metrics.balanced_accuracy_score`.
    #: `DEFAULT_LABELS` includes `idle` as the negative class on purpose
    #: (Sprint 07), so a real recording session is very likely class-imbalanced;
    #: plain `accuracy` on an imbalanced set can look good by mostly predicting
    #: the majority class, which balanced accuracy does not let it get away with.
    balanced_accuracy: float


def train(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 200,
) -> TrainingReport:
    """Load, split, fit, evaluate, save. One call, one artifact.

    The split is grouped by `session_id` (decision 4 above) — every frame of
    a given recording session lands entirely on one side, so the reported
    accuracy reflects generalising to an unseen session, not memorising this
    one's near-duplicate frames.
    """
    dataset = load_dataset(dataset_path)
    unique_sessions = sorted(set(dataset.groups))
    if len(unique_sessions) < 2:
        only = unique_sessions[0] if unique_sessions else "<empty>"
        raise DatasetError(
            f"All {len(dataset.y)} samples come from a single session ({only!r}). "
            "A random per-frame split would put near-duplicate frames from the same "
            "recording burst on both sides of train/test, inflating accuracy. Record "
            "at least one more session with hello_dataset.py so the held-out split is "
            "a different session, not just different frames of the same one."
        )

    # Two distinct sessions overall is not the same guarantee as two distinct
    # sessions *per class*. `GroupShuffleSplit` can still put every sample of
    # a class that only exists in one session entirely on one side of the
    # split -- that class then has zero support on the other side, and its
    # precision/recall/f1 in the report is either undefined (`zero_division`
    # papers over it as 0.0) or a meaningless 100%, not a real held-out
    # measurement of whether the model generalises to it.
    sessions_by_label: dict[str, set[str]] = {}
    for label, group in zip(dataset.y, dataset.groups):
        sessions_by_label.setdefault(label, set()).add(group)
    single_session_labels = {
        label: sorted(sessions)
        for label, sessions in sessions_by_label.items()
        if len(sessions) < 2
    }
    if single_session_labels:
        shown = ", ".join(
            f"{label} (only in {sessions})" for label, sessions in single_session_labels.items()
        )
        raise DatasetError(
            f"These classes only appear in a single recording session: {shown}. "
            "The grouped split (decision 4) could put every sample of one of "
            "these classes on the same side of train/test, making its "
            "precision/recall/f1 meaningless rather than a real held-out "
            "measurement. Record at least one more session with hello_dataset.py "
            "that also covers that label."
        )

    train_idx, test_idx = _grouped_split(dataset, test_size, random_state)
    X_train = [dataset.X[i] for i in train_idx]
    X_test = [dataset.X[i] for i in test_idx]
    y_train = [dataset.y[i] for i in train_idx]
    y_test = [dataset.y[i] for i in test_idx]

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
    macro_f1 = per_class_report["macro avg"]["f1-score"]
    balanced_accuracy = balanced_accuracy_score(y_test, predicted)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trained_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    # Model, column order and class list travel together in one file — the
    # live pipeline loads all three from the same artifact, never re-derives
    # them, so the vector it builds at inference time cannot silently drift
    # from the one this model was actually trained on (`predict()` below
    # enforces that by reading `bundle["columns"]`, not `TRAINING_COLUMNS`).
    # The evaluation numbers ride along too (Sprint 10): `scripts/report.py`
    # regenerates the HTML report from `load_model()` alone, without
    # re-running training.
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
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
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
        dropped=dataset.dropped,
        macro_f1=macro_f1,
        balanced_accuracy=balanced_accuracy,
    )


def load_model(path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ModelError(
            f"No trained model at {path}. Train one first: "
            "uv run python scripts/train_model.py"
        )
    bundle = joblib.load(path)
    # `columns`/`classes`/`model` are the three `predict()` actually reads
    # (module docstring, decision 1) -- missing or empty any of them means
    # this file was saved by an incompatible or corrupt version of `train()`,
    # and building a vector or reading a class list from it would silently
    # drift rather than fail where the mismatch actually is.
    missing = [key for key in ("model", "columns", "classes") if key not in bundle]
    if missing:
        raise ModelError(
            f"{path} bundle is missing {missing} — it was saved by an incompatible "
            "version of train(). Retrain with the current scripts/train_model.py."
        )
    if not bundle["columns"]:
        raise ModelError(
            f"{path} has an empty 'columns' in its bundle — it was saved by an "
            "incompatible version of train(). Retrain with the current scripts/train_model.py."
        )
    if not bundle["classes"]:
        raise ModelError(
            f"{path} has an empty 'classes' list in its bundle — a model with no "
            "class vocabulary cannot predict anything. Retrain with the current "
            "scripts/train_model.py."
        )
    if not hasattr(bundle["model"], "predict_proba"):
        raise ModelError(
            f"{path}'s 'model' does not support predict_proba() — `predict()` needs "
            "per-class probabilities, not just a label. Retrain with the current "
            "scripts/train_model.py."
        )
    # Post-Sprint-11 hardening: `columns`/`classes` are trusted verbatim by
    # `predict()` below, so a corrupt or hand-edited bundle has to be caught
    # here, once, rather than producing a wrong prediction that looks fine.
    columns = tuple(bundle["columns"])
    if len(set(columns)) != len(columns):
        raise ModelError(
            f"{path}'s 'columns' has duplicate name(s) — a repeated column would "
            "silently double-count that feature and shift every column after it "
            "out of alignment. Retrain with the current scripts/train_model.py."
        )
    unknown_columns = [name for name in columns if name not in KNOWN_TRAINING_COLUMNS]
    if unknown_columns:
        raise ModelError(
            f"{path}'s 'columns' has unrecognized name(s) {unknown_columns} — not in "
            "backend/features/features.py:FEATURE_NAMES or "
            "backend/dataset/writer.py:NUMERIC_MOTION_COLUMNS. Retrain with the "
            "current scripts/train_model.py."
        )
    model = bundle["model"]
    n_features_in = getattr(model, "n_features_in_", None)
    if n_features_in is not None and n_features_in != len(columns):
        raise ModelError(
            f"{path}'s 'columns' has {len(columns)} entries but the model itself "
            f"expects {n_features_in} — bundle is corrupt or mismatched. Retrain "
            "with the current scripts/train_model.py."
        )
    model_classes = getattr(model, "classes_", None)
    if model_classes is not None and list(bundle["classes"]) != list(model_classes):
        raise ModelError(
            f"{path}'s 'classes' {list(bundle['classes'])} does not match the "
            f"model's own classes_ {list(model_classes)} — bundle is corrupt or "
            "mismatched. Retrain with the current scripts/train_model.py."
        )
    return bundle


def predict(bundle: dict[str, Any], row: dict) -> tuple[str, dict[str, float]]:
    """One row in (CSV dict or live feature dict), a label and the per-class
    probability distribution out.

    The vector is built from `bundle["columns"]`, never this module's
    `TRAINING_COLUMNS` — a bundle saved by an older or newer version of
    `train()` (a different column set, a different order) still predicts
    against the vector it actually shipped with, instead of one that has
    silently drifted out of sync with what the model was fit on.
    """
    for key in ("model", "columns", "classes"):
        if key not in bundle:
            raise ModelError(
                f"model bundle is missing '{key}' — retrain with the current "
                "scripts/train_model.py rather than predicting off a stale/corrupt bundle."
            )
    columns = tuple(bundle["columns"])
    if not columns:
        raise ModelError("model bundle has an empty 'columns' list — retrain to fix it.")
    built = build_vector(row, columns)
    non_finite = [name for name, value in zip(columns, built) if not math.isfinite(value)]
    if non_finite:
        raise ModelError(
            f"non-finite value(s) in {non_finite} for this frame — refusing to feed "
            "NaN/Inf into the model as if it were a real number."
        )
    vector = [built]
    model: RandomForestClassifier = bundle["model"]
    label = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    classes = list(bundle["classes"])
    if len(classes) != len(probabilities):
        raise ModelError(
            f"model bundle's classes ({len(classes)}) do not match predict_proba's "
            f"output ({len(probabilities)}) — the bundle is corrupt or was saved by an "
            "incompatible version of train(). Retrain with the current scripts/train_model.py."
        )
    return label, dict(zip(classes, probabilities.tolist()))
