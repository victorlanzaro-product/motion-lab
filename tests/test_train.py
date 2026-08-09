"""Model Training tests — a synthetic dataset built through DatasetWriter
itself (proving the Sprint 07 -> 08 handoff), no camera, no real recording.
"""

from __future__ import annotations

import csv
import random

import joblib
import pytest

from backend.dataset import DatasetWriter
from backend.features import ArmMotion, Direction, FEATURE_NAMES, FrameFeatures, MotionState
from backend.ml import (
    MIN_SAMPLES_PER_CLASS,
    TRAINING_COLUMNS,
    DatasetError,
    ModelError,
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


def _complete_motion(rng: random.Random, frame_index: int, timestamp: float) -> MotionState:
    """Both arms fully populated -- a real (near-zero) velocity, not `None`.

    A bare `ArmMotion()` leaves every numeric motion column `None`; since
    `load_dataset` now drops a row it cannot fully populate rather than
    silently zeroing it (backend/ml/train.py), a fixture that wants its rows
    to actually reach the Forest has to supply real values on both sides.
    `frame_index` must match the `FrameFeatures` it is paired with --
    `DatasetWriter.write` now rejects a features/motion pair from two
    different frames (backend/dataset/writer.py).
    """

    def arm() -> ArmMotion:
        return ArmMotion(
            velocity_y=_jitter(rng, 0.0, 0.02),
            velocity_x=_jitter(rng, 0.0, 0.02),
            speed=abs(_jitter(rng, 0.02, 0.02)),
            elbow_velocity=_jitter(rng, 0.0, 0.5),
            direction=Direction.STILL,
            moving=False,
        )

    return MotionState(frame_index=frame_index, timestamp=timestamp, left=arm(), right=arm())


def build_synthetic_dataset(tmp_path, per_class: int = 40, seed: int = 7, sessions: int = 4):
    """Record synthetic rows through the real `DatasetWriter`, spread across
    multiple `session_id`s -- mirrors real recordings, where each run of
    `hello_dataset.py` stamps one session_id, and is what lets `train()`'s
    grouped split (backend/ml/train.py decision 4) hold out a whole session
    instead of failing for lack of a second group.
    """
    rng = random.Random(seed)
    path = tmp_path / "dataset.csv"
    index = 0
    with DatasetWriter(path) as writer:
        for label in PROTOTYPES:
            for i in range(per_class):
                writer.session_id = f"synthetic-{i % sessions}"
                features = _complete_features(rng, label, index)
                motion = _complete_motion(rng, index, features.timestamp)
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


# --- shoulder_width / grouped split (backend/ml/train.py decisions 2, 4) ---


def test_training_columns_exclude_shoulder_width():
    """Same rule as FEATURE_NAMES: shoulder_width is the scale reference, not
    a feature -- training on it directly would only work at the distance it
    was recorded at."""
    assert "shoulder_width" not in TRAINING_COLUMNS


def test_training_columns_are_exactly_feature_names_no_motion():
    """Sprint 11 decision 2: the current label vocabulary is purely
    postural, so the trained vector must be `FEATURE_NAMES` and nothing
    else -- no velocity, no direction, no waiting on `MotionState`."""
    assert TRAINING_COLUMNS == FEATURE_NAMES


def test_train_fails_when_dataset_has_only_one_session(tmp_path):
    path = build_synthetic_dataset(tmp_path, per_class=40, sessions=1)
    with pytest.raises(DatasetError, match="session"):
        train(dataset_path=path, model_path=tmp_path / "model.joblib")


# --- missing motion is dropped, never zeroed (backend/ml/train.py) --------


def test_load_dataset_keeps_rows_with_missing_motion_since_training_ignores_it(tmp_path):
    """Sprint 11: `TRAINING_COLUMNS` is `FEATURE_NAMES` only (decision 2) --
    motion no longer feeds the model at all, so a row with fully-populated
    static features but a MotionTracker that has not warmed up yet (every
    numeric motion column `None`) is a perfectly good training example, not
    a drop. This is what lets inference run on the very first complete
    frame, before `MotionState.complete` would ever be true."""
    path = tmp_path / "dataset.csv"
    rng = random.Random(3)
    with DatasetWriter(path) as writer:
        for session in range(2):
            writer.session_id = f"s{session}"
            for i in range(MIN_SAMPLES_PER_CLASS + 2):
                features = _complete_features(rng, "idle", i)
                motion = _complete_motion(rng, i, features.timestamp)
                writer.write("idle", features, motion)

        # One extra row with genuinely missing motion (no tracking history
        # yet on the right arm) -- irrelevant to training now.
        writer.session_id = "s0"
        incomplete = MotionState(
            frame_index=999,
            timestamp=999 / 30.0,
            left=ArmMotion(
                velocity_y=0.0, velocity_x=0.0, speed=0.0,
                elbow_velocity=0.0, direction=Direction.STILL, moving=False,
            ),
            right=ArmMotion(),  # no history yet: every numeric field is None
        )
        writer.write("idle", _complete_features(rng, "idle", 999), incomplete)

    dataset = load_dataset(path)
    assert dataset.dropped["missing_feature"] == 0
    assert dataset.counts["idle"] == 2 * (MIN_SAMPLES_PER_CLASS + 2) + 1


def test_load_dataset_still_drops_a_row_with_a_missing_static_feature(tmp_path):
    """The static-feature side of `MissingFeatureError` still applies --
    only motion became irrelevant (decision 2), not the "no fabricated 0"
    rule itself."""
    path = build_synthetic_dataset(tmp_path, per_class=MIN_SAMPLES_PER_CLASS + 5)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fieldnames = list(rows[0].keys())
    rows[0]["left_elbow_angle"] = ""  # a joint MediaPipe could not see
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dataset = load_dataset(path)
    assert dataset.dropped["missing_feature"] == 1


def test_load_dataset_rejects_a_csv_with_a_missing_column(tmp_path):
    """Schema drift (a hand-edited or stale CSV) must fail loudly, not train
    silently on a shifted column order."""
    path = tmp_path / "dataset.csv"
    path.write_text("session_id,frame_index,timestamp,label\ns1,0,0.0,idle\n")
    with pytest.raises(DatasetError, match="missing column"):
        load_dataset(path)


# --- "wave" removed from the trainable vocabulary (Sprint 11, product decision) --


def test_load_dataset_rejects_legacy_wave_rows_with_a_migration_message(tmp_path):
    """`DatasetWriter.write` refuses new "wave" rows outright, but a
    `dataset.csv` recorded before that decision could still have some on
    disk. `load_dataset` must fail loudly with migration steps -- never
    silently train on the label the Forest cannot actually separate, and
    never fold it into `dropped` where a real problem would go unnoticed."""
    path = build_synthetic_dataset(tmp_path, per_class=MIN_SAMPLES_PER_CLASS)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    legacy_row = dict(rows[0])
    legacy_row["label"] = "wave"
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writerow(legacy_row)

    with pytest.raises(DatasetError, match="no longer a trainable class"):
        load_dataset(path)


# --- per-class session coverage in the grouped split (post-Sprint-11) ------


def test_train_fails_when_a_class_only_appears_in_a_single_session(tmp_path):
    """Two sessions overall is not the same guarantee as two sessions *per
    class* -- GroupShuffleSplit could still put every sample of a
    single-session class on one side of the split, making its precision/
    recall/f1 meaningless rather than a real held-out measurement."""
    path = tmp_path / "dataset.csv"
    rng = random.Random(11)
    with DatasetWriter(path) as writer:
        for session in range(2):
            writer.session_id = f"s{session}"
            for i in range(MIN_SAMPLES_PER_CLASS + 5):
                features = _complete_features(rng, "idle", i)
                m = _complete_motion(rng, i, features.timestamp)
                writer.write("idle", features, m)
        # "arms_open" only ever recorded in session s0 -- the dataset as a
        # whole has 2 sessions, but this one class does not.
        writer.session_id = "s0"
        for i in range(MIN_SAMPLES_PER_CLASS + 5):
            features = _complete_features(rng, "arms_open", i)
            m = _complete_motion(rng, i, features.timestamp)
            writer.write("arms_open", features, m)

    with pytest.raises(DatasetError, match="single recording session"):
        train(dataset_path=path, model_path=tmp_path / "model.joblib")


# --- macro-F1 / balanced accuracy (post-Sprint-11) -------------------------


def test_train_reports_macro_f1_and_balanced_accuracy(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path, per_class=40)
    model_path = tmp_path / "model.joblib"
    report = train(dataset_path=dataset_path, model_path=model_path)

    # Trivially separable prototypes -> near-perfect on every axis, not just
    # accuracy (a model that only nailed the majority class would not clear
    # this on the rarer ones too).
    assert 0.9 <= report.macro_f1 <= 1.0
    assert 0.9 <= report.balanced_accuracy <= 1.0

    bundle = load_model(model_path)
    assert bundle["macro_f1"] == pytest.approx(report.macro_f1)
    assert bundle["balanced_accuracy"] == pytest.approx(report.balanced_accuracy)


# --- bundle / predict validation (post-Sprint-11) --------------------------


def test_load_model_rejects_a_bundle_missing_required_keys(tmp_path):
    path = tmp_path / "broken.joblib"
    joblib.dump({"columns": TRAINING_COLUMNS}, path)  # no 'model', no 'classes'
    with pytest.raises(ModelError, match="missing"):
        load_model(path)


def test_load_model_rejects_a_bundle_with_an_empty_class_list(tmp_path):
    path = tmp_path / "broken.joblib"
    joblib.dump({"model": object(), "columns": TRAINING_COLUMNS, "classes": []}, path)
    with pytest.raises(ModelError, match="classes"):
        load_model(path)


def test_predict_rejects_a_bundle_whose_classes_do_not_match_predict_proba(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path, per_class=40)
    model_path = tmp_path / "model.joblib"
    train(dataset_path=dataset_path, model_path=model_path)
    bundle = dict(load_model(model_path))
    bundle["classes"] = list(bundle["classes"]) + ["extra_ghost_class"]

    row = {name: 0.0 for name in TRAINING_COLUMNS}
    with pytest.raises(ModelError, match="do not match"):
        predict(bundle, row)


def test_load_model_rejects_duplicate_columns(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path, per_class=40)
    model_path = tmp_path / "model.joblib"
    train(dataset_path=dataset_path, model_path=model_path)
    bundle = dict(load_model(model_path))
    bundle["columns"] = (TRAINING_COLUMNS[0],) + TRAINING_COLUMNS  # duplicate first entry
    joblib.dump(bundle, model_path)

    with pytest.raises(ModelError, match="duplicate"):
        load_model(model_path)


def test_load_model_rejects_unrecognized_columns(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path, per_class=40)
    model_path = tmp_path / "model.joblib"
    train(dataset_path=dataset_path, model_path=model_path)
    bundle = dict(load_model(model_path))
    bundle["columns"] = TRAINING_COLUMNS + ("not_a_real_column",)
    joblib.dump(bundle, model_path)

    with pytest.raises(ModelError, match="unrecognized"):
        load_model(model_path)


def test_load_model_rejects_a_column_count_mismatch_with_the_model(tmp_path):
    """`columns` must match what the fitted model actually expects
    (`n_features_in_`) -- a bundle with an edited column list but the
    original model would otherwise build a vector of the wrong length."""
    dataset_path = build_synthetic_dataset(tmp_path, per_class=40)
    model_path = tmp_path / "model.joblib"
    train(dataset_path=dataset_path, model_path=model_path)
    bundle = dict(load_model(model_path))
    bundle["columns"] = TRAINING_COLUMNS[:-1]  # one short of what the model was fit on
    joblib.dump(bundle, model_path)

    with pytest.raises(ModelError, match="expects"):
        load_model(model_path)


def test_load_model_rejects_classes_that_do_not_match_the_models_own_classes(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path, per_class=40)
    model_path = tmp_path / "model.joblib"
    train(dataset_path=dataset_path, model_path=model_path)
    bundle = dict(load_model(model_path))
    bundle["classes"] = list(reversed(bundle["classes"]))  # same set, wrong order
    joblib.dump(bundle, model_path)

    with pytest.raises(ModelError, match="does not match the model's own classes_"):
        load_model(model_path)


def test_predict_rejects_a_non_finite_feature_value(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path, per_class=40)
    model_path = tmp_path / "model.joblib"
    train(dataset_path=dataset_path, model_path=model_path)
    bundle = load_model(model_path)

    row = {name: 0.0 for name in TRAINING_COLUMNS}
    row[TRAINING_COLUMNS[0]] = float("nan")
    with pytest.raises(ModelError, match="non-finite"):
        predict(bundle, row)


# --- unknown labels rejected at load time too (post-Sprint-11) ------------


def test_load_dataset_rejects_a_row_with_an_unknown_label(tmp_path):
    path = build_synthetic_dataset(tmp_path, per_class=MIN_SAMPLES_PER_CLASS)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    bad_row = dict(rows[0])
    bad_row["label"] = "sitting"
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writerow(bad_row)

    with pytest.raises(DatasetError, match="not one of the current trainable classes"):
        load_dataset(path)


# --- grouped split retries seeds until every class covers both sides ------


def test_train_fails_actionably_when_no_seed_gives_full_class_coverage(tmp_path):
    """Three sessions, three classes, each class spanning exactly two of the
    three sessions in a round-robin (A: s0/s1, B: s1/s2, C: s0/s2) -- every
    single-session holdout strands exactly one class from the test side, so
    no seed `_grouped_split` tries can ever produce full coverage. Every
    per-class-session precondition still passes (each class has 2 sessions),
    so this exercises the deterministic retry-and-fail path specifically."""
    path = tmp_path / "dataset.csv"
    rng = random.Random(5)
    coverage = {
        "arm_raised": ("s0", "s1"),
        "arms_crossed": ("s1", "s2"),
        "idle": ("s0", "s2"),
    }
    with DatasetWriter(path) as writer:
        for label, sessions in coverage.items():
            for session in sessions:
                writer.session_id = session
                for i in range(MIN_SAMPLES_PER_CLASS + 5):
                    features = _complete_features(rng, label, i)
                    m = _complete_motion(rng, i, features.timestamp)
                    writer.write(label, features, m)

    with pytest.raises(DatasetError, match="every class present on both sides"):
        train(dataset_path=path, model_path=tmp_path / "model.joblib")
