"""ML Training: dataset.csv rows in, a Random Forest bundle out."""

from backend.ml.report import compute_agreement, render_report, rule_label
from backend.ml.train import (
    DEFAULT_MODEL_PATH,
    MIN_SAMPLES_PER_CLASS,
    TRAINING_COLUMNS,
    Dataset,
    DatasetError,
    MissingFeatureError,
    ModelError,
    TrainingReport,
    build_vector,
    load_dataset,
    load_model,
    predict,
    train,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "MIN_SAMPLES_PER_CLASS",
    "TRAINING_COLUMNS",
    "Dataset",
    "DatasetError",
    "MissingFeatureError",
    "ModelError",
    "TrainingReport",
    "build_vector",
    "compute_agreement",
    "load_dataset",
    "load_model",
    "predict",
    "render_report",
    "rule_label",
    "train",
]
