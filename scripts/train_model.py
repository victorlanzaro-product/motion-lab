"""Sprint 08 — ML Training.

Acceptance criterion: "O sistema treina um Random Forest a partir do dataset
gravado e salva um modelo que separa as classes com boa acurácia."

    uv run python scripts/train_model.py             # real data/training/dataset.csv
    uv run python scripts/train_model.py --simulate  # synthetic, no recording needed

This is a batch job, not a live view — there is no camera in this sprint, so
it does not fit the hello_/check_ naming the earlier sprints use. `--simulate`
exists for the same reason `check_motion.py --simulate` does: it has to prove
the whole pipeline works today, deterministically, before anyone has actually
recorded a real session with `hello_dataset.py`.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
from pathlib import Path

from backend.dataset import DatasetWriter
from backend.dataset.writer import DATASET_COLUMNS, MOTION_COLUMNS
from backend.features import FEATURE_NAMES, ArmMotion, Direction, FrameFeatures, MotionState
from backend.ml import (
    DEFAULT_MODEL_PATH,
    DEFAULT_V2_MODEL_PATH,
    V2_LABELS,
    DatasetError,
    train,
)

#: Trivially separable prototypes, one per label. Not meant to resemble a real
#: recording session — only to prove load -> split -> fit -> save works.
#:
#: The v1 simulator uses the four default static postures. The v2 simulator
#: writes its temporal rows directly so it can include the v2-only `wave`
#: label without changing DatasetWriter's v1 recording contract.
PROTOTYPES = {
    "arm_raised": dict(left_wrist_above_shoulder=True, left_wrist_height=1.2, wrist_distance=1.0),
    "arms_crossed": dict(wrists_crossed=True, wrist_distance=0.3),
    "arms_open": dict(wrist_distance=2.8, wrists_crossed=False),
    "idle": dict(wrist_distance=1.0, left_wrist_height=0.0, wrists_crossed=False),
}
SAMPLES_PER_LABEL = 60


def _jitter(rng: random.Random, value: float, spread: float = 0.05) -> float:
    return value + rng.uniform(-spread, spread)


#: Multiple synthetic "sessions" so the grouped train/test split (backend/ml/
#: train.py decision 4) has more than one session_id to split on -- a single
#: session, however large, cannot demonstrate what a real held-out split does.
SIMULATED_SESSIONS = 4
SIMULATED_WINDOW_SESSIONS = 5


def _synthetic_row(
    rng: random.Random, label: str, index: int
) -> tuple[FrameFeatures, MotionState]:
    values = {name: 0.0 for name in FEATURE_NAMES}
    for name in ("wrists_crossed", "left_wrist_above_shoulder", "right_wrist_above_shoulder"):
        values[name] = False
    for name, value in PROTOTYPES[label].items():
        values[name] = value if isinstance(value, bool) else _jitter(rng, value)
    features = FrameFeatures(frame_index=index, timestamp=index / 30.0, **values)

    # Every label here is a static posture (see `PROTOTYPES`), so both arms
    # get a real, near-zero, STILL velocity -- a bare `ArmMotion()` leaves
    # every numeric motion column `None`, and `load_dataset` now drops a row
    # it cannot fully populate rather than letting a fabricated 0 through
    # (backend/ml/train.py).
    def arm() -> ArmMotion:
        return ArmMotion(
            velocity_y=_jitter(rng, 0.0, 0.05),
            velocity_x=_jitter(rng, 0.0, 0.05),
            speed=abs(_jitter(rng, 0.02, 0.02)),
            elbow_velocity=_jitter(rng, 0.0, 1.0),
            direction=Direction.STILL,
            moving=False,
        )

    motion = MotionState(
        frame_index=index,
        timestamp=features.timestamp,
        left=arm(),
        right=arm(),
    )
    return features, motion


def build_synthetic_dataset(
    path: Path, seed: int = 42, sessions: int = SIMULATED_SESSIONS
) -> None:
    rng = random.Random(seed)
    index = 0
    with DatasetWriter(path) as writer:
        for label in PROTOTYPES:
            for i in range(SAMPLES_PER_LABEL):
                writer.session_id = f"simulated-{i % sessions}"
                features, motion = _synthetic_row(rng, label, index)
                writer.write(label, features, motion)
                index += 1
    print(
        f"Simulated {writer.total} samples across {len(PROTOTYPES)} labels and "
        f"{sessions} synthetic sessions -> {path}"
    )


def build_synthetic_window_dataset(
    path: Path, seed: int = 42, sessions: int = SIMULATED_WINDOW_SESSIONS
) -> None:
    """Write a tiny synthetic CSV that exercises schema-v2 window training.

    `wave` and `arm_raised` share the arm-raised posture prototype while their
    synthetic wrist trajectories differ. This exercises the temporal training
    path; it is not evidence about real-world recognition quality.
    """
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        for label in V2_LABELS:
            for session_index in range(sessions):
                # A recording session has one label, matching the actual
                # dataset contract. Its own tracker starts with local time 0.
                session_id = f"simulated-{label}-{session_index}"
                prototype_label = "arm_raised" if label == "wave" else label
                for segment_frame in range(60):
                    frame_index = segment_frame
                    timestamp = segment_frame / 10.0
                    values = {name: 0.0 for name in FEATURE_NAMES}
                    for name in (
                        "wrists_crossed",
                        "left_wrist_above_shoulder",
                        "right_wrist_above_shoulder",
                    ):
                        values[name] = False
                    for name, value in PROTOTYPES[prototype_label].items():
                        # These two classes must have exactly the same frame
                        # feature distribution. The temporal ordering below is
                        # their only synthetic distinction.
                        if label in {"arm_raised", "wave"}:
                            values[name] = value
                        else:
                            values[name] = value if isinstance(value, bool) else _jitter(rng, value)

                    if label == "arm_raised":
                        # One rise followed by a hold: 30 low and 30 high rows.
                        values["left_wrist_height"] = 0.0 if segment_frame < 30 else 1.2
                    elif label == "wave":
                        # Reorder the exact same 30 low / 30 high observations
                        # into repeated reversals, preserving the v1 feature
                        # histogram while making the v2 window signal distinct.
                        values["left_wrist_height"] = 0.0 if segment_frame % 2 == 0 else 1.2
                    values["left_wrist_above_shoulder"] = values["left_wrist_height"] > 0.5

                    row = {
                        "session_id": session_id,
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "label": label,
                        **values,
                        "shoulder_width": 1.0,
                        **{name: None for name in MOTION_COLUMNS},
                    }
                    writer.writerow(row)
    print(
        f"Simulated {len(V2_LABELS)} v2 labels across {sessions} grouped sessions "
        f"-> {path}"
    )


def print_report(report, schema: int = 1) -> None:
    print(f"schema    v{schema}")
    print(f"\ntrained   {report.trained_at}")
    counts = ", ".join(f"{k}={v}" for k, v in report.counts.items())
    print(f"samples   {report.n_samples}  {counts}")
    cv = report.cv
    print(
        f"macro-F1  {cv['macro_f1_mean']:.3f} ± {cv['macro_f1_std']:.3f}  "
        f"(grouped CV; {cv['folds_used']}/{cv['n_splits']} folds; "
        f"{cv['folds_skipped']} skipped)"
    )
    print(f"accuracy  {report.accuracy * 100:.1f}%  (held-out test split)")
    print(f"split F1  {report.macro_f1:.3f}  (held-out split detail)")
    print(f"balanced  {report.balanced_accuracy * 100:.1f}%  (balanced accuracy)")
    print(f"saved to  {report.model_path}")
    print("\ntop features:")
    ranked = sorted(report.feature_importances.items(), key=lambda kv: kv[1], reverse=True)
    for name, importance in ranked[:6]:
        print(f"  {name:<24} {importance:.3f}")
    print("\n" + report.report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the gesture Random Forest.")
    parser.add_argument("--simulate", action="store_true", help="synthetic dataset, no recording")
    parser.add_argument("--dataset", type=Path, default=None, help="override the dataset CSV path")
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument(
        "--schema",
        type=int,
        choices=(1, 2),
        default=1,
        help="training schema (1: per-frame default, 2: temporal windows)",
    )
    args = parser.parse_args()
    model_path = args.model_out or (
        DEFAULT_V2_MODEL_PATH if args.schema == 2 else DEFAULT_MODEL_PATH
    )

    try:
        if args.simulate:
            print(
                "WARNING: --simulate trains on a synthetic dataset of trivially "
                "separable prototypes. It proves the pipeline (load -> split -> fit "
                "-> evaluate -> save -> reload -> predict) works end to end -- it "
                "proves NOTHING about real-world accuracy. Do not ship this model; "
                "record a real session with hello_dataset.py and train on that."
            )
            with tempfile.TemporaryDirectory() as tmp:
                dataset_path = Path(tmp) / "dataset.csv"
                if args.schema == 2:
                    build_synthetic_window_dataset(dataset_path)
                else:
                    build_synthetic_dataset(dataset_path)
                report = train(
                    dataset_path=dataset_path,
                    model_path=model_path,
                    schema=args.schema,
                )
        else:
            kwargs = {"model_path": model_path, "schema": args.schema}
            if args.dataset:
                kwargs["dataset_path"] = args.dataset
            report = train(**kwargs)
    except DatasetError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print_report(report, schema=args.schema)
    if args.simulate:
        print(
            "\nOK: the PIPELINE works (plumbing only -- synthetic data, not a real "
            "measurement of model quality). No image ever touched this file."
        )
    else:
        print("\nOK: model trained and saved. No image ever touched this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
