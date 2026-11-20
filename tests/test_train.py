"""Model Training tests — a synthetic dataset built through DatasetWriter
itself (proving the Sprint 07 -> 08 handoff), no camera, no real recording.
"""

from __future__ import annotations

import csv
import random
from collections import Counter

import joblib
import pytest

from backend.dataset import DatasetWriter
from backend.dataset.writer import DATASET_COLUMNS
from backend.dataset.windows import WINDOW_TRAINING_COLUMNS, build_window_rows
from backend.features import ArmMotion, Direction, FEATURE_NAMES, FrameFeatures, MotionState
from backend.ml import (
    MIN_SAMPLES_PER_CLASS,
    TRAINING_COLUMNS,
    Dataset,
    DatasetError,
    ModelError,
    load_dataset,
    load_model,
    predict,
    train,
)
from backend.ml.train import (
    V2_LABELS,
    _grouped_cv,
    build_vector,
    classify_dataset_row,
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


def build_temporal_wave_dataset(tmp_path, sessions: int = 5):
    """Make equal static-frame distributions with different temporal order.

    Each wave and arm-raised session has its own session id and the same
    wrist-height multiset. Wave repeats a triangle; arm-raise sees those same
    values sorted into one slow rise. Only ordering differs.
    """
    levels = [index / 20 for index in range(21)]
    wave_cycle = levels + list(reversed(levels[1:-1]))
    wave_values = wave_cycle * 8
    raised_values = sorted(wave_values)
    assert Counter(wave_values) == Counter(raised_values)

    path = tmp_path / "temporal-dataset.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        for session_index in range(sessions):
            for label, heights in (("wave", wave_values), ("arm_raised", raised_values)):
                session_id = f"{label}-{session_index}"
                for frame_index, height in enumerate(heights):
                    row = {name: "0" for name in DATASET_COLUMNS}
                    row.update(
                        {
                            "session_id": session_id,
                            "frame_index": str(frame_index),
                            "timestamp": f"{frame_index / 30:.6f}",
                            "label": label,
                            "left_wrist_height": f"{height:.6f}",
                            "right_wrist_height": "0.75",
                            "left_wrist_offset_x": "0",
                            "right_wrist_offset_x": "0",
                            "left_wrist_above_shoulder": "True",
                            "right_wrist_above_shoulder": "True",
                            "wrists_crossed": "False",
                        }
                    )
                    writer.writerow(row)
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


def test_shared_row_classifier_reports_loader_reasons_for_stats():
    valid = {"label": "idle", **{name: "0" for name in TRAINING_COLUMNS}}
    reason, vector = classify_dataset_row(valid)
    assert reason is None
    assert vector == [0.0] * len(TRAINING_COLUMNS)

    missing_label, _ = classify_dataset_row({**valid, "label": "  "})
    unknown_label, _ = classify_dataset_row({**valid, "label": "sitting"})
    missing_feature, _ = classify_dataset_row({**valid, TRAINING_COLUMNS[0]: ""})
    non_finite, _ = classify_dataset_row({**valid, TRAINING_COLUMNS[0]: "nan"})

    assert (missing_label, unknown_label, missing_feature, non_finite) == (
        "missing_label",
        "unknown_label",
        "missing_feature",
        "non_finite",
    )


# --- grouped cross-validation (Sprint 13) --------------------------------


def _six_session_cv_dataset() -> Dataset:
    """Six equal-size groups make GroupKFold's paired first fold explicit."""
    labels = []
    groups = []
    X = []
    for session_index in range(6):
        session = f"s{session_index}"
        # This class only appears in s0 and s5. GroupKFold(5) holds those two
        # sessions together in its first fold, so exactly one fold is invalid.
        session_labels = ["arm_raised", "arms_crossed", "idle"] * 3
        if session in {"s0", "s5"}:
            session_labels += ["arms_open"] * 11
        else:
            session_labels += ["arm_raised"] * 11
        for row_index, label in enumerate(session_labels):
            labels.append(label)
            groups.append(session)
            X.append([float(session_index), float(row_index)])
    return Dataset(
        X=X,
        y=labels,
        groups=groups,
        counts={label: labels.count(label) for label in set(labels)},
    )


def test_grouped_cv_counts_folds_missing_a_training_class():
    cv = _grouped_cv(_six_session_cv_dataset(), n_estimators=5, random_state=23)

    assert cv["n_splits"] == 5
    assert cv["folds_used"] == 4
    assert cv["folds_skipped"] == 1
    assert len(cv["per_fold"]) == 5
    assert cv["per_fold"][0]["valid"] is False
    assert cv["per_fold"][0]["macro_f1"] is None
    assert "arms_open" in cv["per_fold"][0]["reason"]
    assert all(fold["valid"] for fold in cv["per_fold"][1:])


@pytest.mark.parametrize("cv_mode", ["group-kfold", "loso"])
def test_grouped_cv_fails_when_every_training_fold_is_missing_a_class(cv_mode):
    dataset = Dataset(
        X=[[0.0], [1.0]],
        y=["arm_raised", "arms_open"],
        groups=["only-raised", "only-open"],
        counts={"arm_raised": 1, "arms_open": 1},
    )

    with pytest.raises(DatasetError, match="zero valid folds"):
        _grouped_cv(dataset, n_estimators=5, random_state=23, cv_mode=cv_mode)


def test_loso_training_on_recorders_single_label_sessions(tmp_path):
    path = tmp_path / "dataset.csv"
    rng = random.Random(7)
    frame = 0
    with DatasetWriter(path) as writer:
        for label in PROTOTYPES:
            for session in range(3):
                writer.session_id = f"{label}-{session}"
                for _ in range(MIN_SAMPLES_PER_CLASS):
                    features = _complete_features(rng, label, frame)
                    writer.write(label, features, _complete_motion(rng, frame, features.timestamp))
                    frame += 1

    report = train(dataset_path=path, model_path=tmp_path / "loso.joblib",
                   n_estimators=20, cv_mode="loso")

    cv = report.cv
    assert cv["mode"] == "loso"
    assert cv["folds_used"] == cv["n_splits"] == 12
    assert cv["folds_skipped"] == 0
    assert all(len(fold["test_sessions"]) == 1 for fold in cv["per_fold"])
    assert all(
        set(fold["train_sessions"]).isdisjoint(fold["test_sessions"])
        for fold in cv["per_fold"]
    )
    assert {fold["test_sessions"][0] for fold in cv["per_fold"]} == {
        f"{label}-{session}" for label in PROTOTYPES for session in range(3)
    }
    assert cv["macro_f1_oof"] > 0.9
    assert cv["macro_f1_mean"] <= 0.25  # Each fold contains one true class.
    assert all(metrics["support"] == MIN_SAMPLES_PER_CLASS for metrics in report.per_class.values())
    assert load_model(report.model_path)["cv"] == cv


def test_training_reports_deterministic_grouped_cv_and_saves_it(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path, per_class=40)
    first = train(
        dataset_path=dataset_path,
        model_path=tmp_path / "first.joblib",
        random_state=19,
    )
    second = train(
        dataset_path=dataset_path,
        model_path=tmp_path / "second.joblib",
        random_state=19,
    )

    assert first.cv == second.cv
    assert first.cv["n_splits"] == 4
    assert first.cv["folds_used"] == 4
    assert first.cv["folds_skipped"] == 0
    assert len(first.cv["per_fold"]) == 4
    assert load_model(first.model_path)["cv"] == first.cv
    assert load_model(first.model_path)["dataset_path"] == str(dataset_path)


def test_v2_temporal_grouped_cv_separates_wave_from_static_twins_and_bundles_schema(tmp_path):
    path = build_temporal_wave_dataset(tmp_path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert V2_LABELS == ("arm_raised", "arms_crossed", "arms_open", "idle", "wave")
    static_by_label = {
        label: Counter(
            tuple(build_vector(row, TRAINING_COLUMNS))
            for row in rows
            if row["label"] == label
        )
        for label in ("wave", "arm_raised")
    }
    # The full per-frame feature-vector histograms are exactly equal, so a
    # frame-only classifier has no label signal. Session groups remain intact.
    assert static_by_label["wave"] == static_by_label["arm_raised"]
    static_dataset = Dataset(
        X=[build_vector(row, TRAINING_COLUMNS) for row in rows],
        y=[row["label"] for row in rows],
        groups=[row["session_id"] for row in rows],
        counts=dict(Counter(row["label"] for row in rows)),
    )
    static_cv = _grouped_cv(static_dataset, n_estimators=20, random_state=17)
    assert static_cv["macro_f1_mean"] <= 0.6

    # v1 continues to reject wave; it must not silently treat a v2 dataset as
    # a v1-compatible per-frame model.
    with pytest.raises(DatasetError, match="no longer a trainable class"):
        train(dataset_path=path, model_path=tmp_path / "wrong-v1.joblib", schema=1)

    v2_path = tmp_path / "v2.joblib"
    report = train(
        dataset_path=path,
        model_path=v2_path,
        schema=2,
        n_estimators=20,
        random_state=17,
    )
    assert report.cv["folds_used"] == 5
    assert report.cv["macro_f1_mean"] >= 0.9
    assert set(report.feature_importances) == set(WINDOW_TRAINING_COLUMNS)

    bundle = load_model(v2_path, schema=2)
    assert bundle["schema_version"] == 2
    assert bundle["window_seconds"] == pytest.approx(1.0)
    assert bundle["max_gap"] == pytest.approx(0.30)
    assert bundle["min_samples"] == 4
    assert bundle["deadband"] == pytest.approx(0.35)
    assert tuple(bundle["columns"]) == WINDOW_TRAINING_COLUMNS
    with pytest.raises(ModelError, match="schema mismatch"):
        load_model(v2_path)  # default remains v1; no silent fallback

    wave_window = next(
        row for row in build_window_rows(rows) if row["label"] == "wave"
    )
    label, probabilities = predict(bundle, wave_window)
    assert label == "wave"
    assert probabilities["wave"] > probabilities["arm_raised"]

    # The inverse direction is rejected too: selecting schema 2 cannot load
    # an old untagged v1 bundle.
    v1_path = tmp_path / "v1.joblib"
    v1_dir = tmp_path / "v1-data"
    v1_dir.mkdir()
    v1_data = build_synthetic_dataset(v1_dir, per_class=20)
    train(dataset_path=v1_data, model_path=v1_path, n_estimators=10)
    v1_bundle = load_model(v1_path, schema=1)
    assert "schema_version" not in v1_bundle
    with pytest.raises(ModelError, match="schema mismatch"):
        load_model(v1_path, schema=2)


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
