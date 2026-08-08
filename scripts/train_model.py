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
import random
import sys
import tempfile
from pathlib import Path

from backend.dataset import DatasetWriter
from backend.features import FEATURE_NAMES, ArmMotion, Direction, FrameFeatures, MotionState
from backend.ml import DEFAULT_MODEL_PATH, DatasetError, train

#: Trivially separable prototypes, one per label. Not meant to resemble a real
#: recording session — only to prove load -> split -> fit -> save works.
PROTOTYPES = {
    "arm_raised": dict(left_wrist_above_shoulder=True, left_wrist_height=1.2, wrist_distance=1.0),
    "wave": dict(wrist_distance=1.0),  # separated below by alternating velocity, not posture
    "arms_crossed": dict(wrists_crossed=True, wrist_distance=0.3),
    "arms_open": dict(wrist_distance=2.8, wrists_crossed=False),
    "idle": dict(wrist_distance=1.0, left_wrist_height=0.0, wrists_crossed=False),
}
SAMPLES_PER_LABEL = 60


def _jitter(rng: random.Random, value: float, spread: float = 0.05) -> float:
    return value + rng.uniform(-spread, spread)


def _synthetic_row(
    rng: random.Random, label: str, index: int
) -> tuple[FrameFeatures, MotionState]:
    values = {name: 0.0 for name in FEATURE_NAMES}
    for name in ("wrists_crossed", "left_wrist_above_shoulder", "right_wrist_above_shoulder"):
        values[name] = False
    for name, value in PROTOTYPES[label].items():
        values[name] = value if isinstance(value, bool) else _jitter(rng, value)
    features = FrameFeatures(frame_index=index, timestamp=index / 30.0, **values)

    # "wave" is motion, not posture: alternate direction with real velocity so
    # it is the motion columns, not the static ones, that separate it.
    direction = Direction.UP if label == "wave" and index % 2 == 0 else Direction.STILL
    velocity = 2.0 if direction is Direction.UP else 0.0
    left = ArmMotion(direction=direction, velocity_y=velocity)
    motion = MotionState(
        frame_index=index, timestamp=features.timestamp, left=left, right=ArmMotion()
    )
    return features, motion


def build_synthetic_dataset(path: Path, seed: int = 42) -> None:
    rng = random.Random(seed)
    index = 0
    with DatasetWriter(path, session_id="simulated") as writer:
        for label in PROTOTYPES:
            for _ in range(SAMPLES_PER_LABEL):
                features, motion = _synthetic_row(rng, label, index)
                writer.write(label, features, motion)
                index += 1
    print(f"Simulated {writer.total} samples across {len(PROTOTYPES)} labels -> {path}")


def print_report(report) -> None:
    print(f"\ntrained   {report.trained_at}")
    counts = ", ".join(f"{k}={v}" for k, v in report.counts.items())
    print(f"samples   {report.n_samples}  {counts}")
    print(f"accuracy  {report.accuracy * 100:.1f}%  (held-out test split)")
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
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    try:
        if args.simulate:
            with tempfile.TemporaryDirectory() as tmp:
                dataset_path = Path(tmp) / "dataset.csv"
                build_synthetic_dataset(dataset_path)
                report = train(dataset_path=dataset_path, model_path=args.model_out)
        else:
            kwargs = {"model_path": args.model_out}
            if args.dataset:
                kwargs["dataset_path"] = args.dataset
            report = train(**kwargs)
    except DatasetError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print_report(report)
    print("\nOK: model trained and saved. No image ever touched this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
