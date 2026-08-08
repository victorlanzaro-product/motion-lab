"""Model Training tests — a synthetic dataset built through DatasetWriter
itself (proving the Sprint 07 -> 08 handoff), no camera, no real recording.
"""

from __future__ import annotations

import random

import pytest

from backend.dataset import DatasetWriter
from backend.features import ArmMotion, Direction, FEATURE_NAMES, FrameFeatures, MotionState
from backend.ml import (
    MIN_SAMPLES_PER_CLASS,
    TRAINING_COLUMNS,
    DatasetError,
    load_dataset,
    load_model,
    predict,
    train,
)

#: Trivially separable prototypes, one per label, jittered per sample so the
#: Forest sees variation instead of N copies of the same row.
PROTOTYPES = {
    "arm_raised": dict(left_wrist_above_shoulder=True, left_wrist_height=1.2, wrist_distance=1.0),
    "arms_crossed": dict(wrists_crossed=True, wrist_distance=0.3),
    "arms_open": dict(wrist_distance=2.8, wrists_crossed=False),
    "idle": dict(wrist_distance=1.0, left_wrist_height=0.0, wrists_crossed=False),
}


def _jitter(rng: random.Random, value: float, spread: float = 0.05) -> float:
    return value + rng.uniform(-spread, spread)


def _complete_features(rng: random.Random, label: str, index: int) -> FrameFeatures:
    values = {name: 0.0 for name in FEATURE_NAMES}
    for name in ("wrists_crossed", "left_wrist_above_shoulder", "right_wrist_above_shoulder"):
        values[name] = False
    for name, value in PROTOTYPES[label].items():
        values[name] = value if isinstance(value, bool) else _jitter(rng, value)
    return FrameFeatures(frame_index=index, timestamp=index / 30.0, **values)


def build_synthetic_dataset(tmp_path, per_class: int = 40, seed: int = 7):
    """Record a synthetic session through the real `DatasetWriter`."""
    rng = random.Random(seed)
    path = tmp_path / "dataset.csv"
    index = 0
    with DatasetWriter(path, session_id="synthetic") as writer:
        for label in PROTOTYPES:
            for _ in range(per_class):
                features = _complete_features(rng, label, index)
                motion = MotionState(
                    frame_index=index,
                    timestamp=features.timestamp,
                    left=ArmMotion(direction=Direction.STILL),
                    right=ArmMotion(direction=Direction.STILL),
                )
                writer.write(label, features, motion)
                index += 1
    return path


# --- load_dataset --------------------------------------------------------


def test_load_dataset_fails_clearly_when_the_file_does_not_exist(tmp_path):
    with pytest.raises(DatasetError, match="hello_dataset.py"):
        load_dataset(tmp_path / "missing.csv")


def test_load_dataset_fails_clearly_when_a_class_is_too_small(tmp_path):
    path = build_synthetic_dataset(tmp_path, per_class=MIN_SAMPLES_PER_CLASS - 1)
    with pytest.raises(DatasetError, match="Not enough samples"):
        load_dataset(path)


def test_load_dataset_reads_every_row_and_counts_by_label(tmp_path):
    path = build_synthetic_dataset(tmp_path, per_class=MIN_SAMPLES_PER_CLASS)
    dataset = load_dataset(path)
    assert dataset.counts == {label: MIN_SAMPLES_PER_CLASS for label in PROTOTYPES}
    assert len(dataset.X) == len(dataset.y) == MIN_SAMPLES_PER_CLASS * len(PROTOTYPES)
    assert len(dataset.X[0]) == len(TRAINING_COLUMNS)


# --- train -----------------------------------------------------------------


def test_training_on_separable_data_reaches_high_accuracy(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path, per_class=40)
    model_path = tmp_path / "model.joblib"

    report = train(dataset_path=dataset_path, model_path=model_path)

    assert model_path.exists()
    assert report.accuracy >= 0.9  # prototypes are trivially separable by construction
    assert set(report.feature_importances) == set(TRAINING_COLUMNS)
    assert report.n_samples == 40 * len(PROTOTYPES)


def test_a_saved_model_round_trips_through_load_and_predict(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path, per_class=40)
    model_path = tmp_path / "model.joblib"
    train(dataset_path=dataset_path, model_path=model_path)

    bundle = load_model(model_path)
    assert bundle["columns"] == TRAINING_COLUMNS
    assert set(bundle["classes"]) == set(PROTOTYPES)

    row = {name: 0.0 for name in TRAINING_COLUMNS}
    row.update(PROTOTYPES["arms_open"])
    label, probabilities = predict(bundle, row)

    assert label == "arms_open"
    assert probabilities.keys() == set(bundle["classes"])
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)
