"""Live evaluation — same inference path as the web pipeline, without a camera.

Feeds pre-recorded feature frames through `PipelineRunner._predict`, `WindowFeatures`,
`MotionTracker`, and `GestureEngine`, then aggregates segment metrics for Victor's
local hardening script (`scripts/evaluate_live.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from backend.dataset.writer import DEFAULT_LABELS
from backend.features import FEATURE_NAMES, FrameFeatures
from backend.features.velocity import MotionConfig, MotionTracker
from backend.features.window import WindowFeatures
from backend.gestures import GestureConfig, GestureEngine, GestureEvent
from backend.ml.report import rule_label
from backend.ml.train import ModelError, load_model

COMMON_V1_LABELS = frozenset(DEFAULT_LABELS)

WAVE_TEMPORAL_NOTE = (
    "A regra wave não é um rótulo por frame: o GestureEngine dispara um evento "
    "wave quando o _WaveDetector acumula reversões de direção do pulso dentro de "
    "wave_window_seconds (com debounce/cooldown), por braço. A cobertura rule "
    "conta esses eventos; v2 pode rotular wave em frames com janela temporal."
)

EVAL_SCHEMA_VERSION = 1


@dataclass
class EvaluatedFrame:
    frame_index: int
    timestamp: float
    detected: bool
    v1_label: Optional[str] = None
    v2_label: Optional[str] = None
    rule_static: Optional[str] = None
    gesture_events: list[GestureEvent] = field(default_factory=list)
    excluded: str = ""  # "" | "warmup" | "transition"


@dataclass(frozen=True)
class SegmentSpec:
    label: str
    start_index: int
    end_index: int
    warmup: int = 0
    transition: int = 0
    name: str = ""
    variant: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SegmentSpec":
        exclude = raw.get("exclude") or {}
        return cls(
            label=str(raw["label"]),
            start_index=int(raw["start_index"]),
            end_index=int(raw["end_index"]),
            warmup=int(exclude.get("warmup", 0)),
            transition=int(exclude.get("transition", 0)),
            name=str(raw.get("name", "")),
            variant=str(raw.get("variant", "")),
        )


def validate_bundles(v1_path: Path, v2_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load both model bundles or raise ModelError with a Portuguese message."""
    errors: list[str] = []
    v1_bundle: Optional[dict[str, Any]] = None
    v2_bundle: Optional[dict[str, Any]] = None
    for path, schema in ((v1_path, 1), (v2_path, 2)):
        try:
            bundle = load_model(path, schema=schema)
        except ModelError as exc:
            label = "v1" if schema == 1 else "v2"
            errors.append(f"Modelo {label} inválido ({path}): {exc}")
            continue
        if schema == 1:
            v1_bundle = bundle
        else:
            v2_bundle = bundle
    if errors:
        raise ModelError("; ".join(errors))
    assert v1_bundle is not None and v2_bundle is not None
    return v1_bundle, v2_bundle


def frame_from_dict(raw: dict[str, Any]) -> tuple[FrameFeatures, bool]:
    detected = bool(raw.get("detected", True))
    kwargs = {name: raw.get(name) for name in FEATURE_NAMES}
    features = FrameFeatures(
        frame_index=int(raw["frame_index"]),
        timestamp=float(raw["timestamp"]),
        **kwargs,
    )
    return features, detected


class LiveEvaluationSession:
    """Stateful evaluator mirroring `PipelineRunner._stream` feature processing."""

    def __init__(
        self,
        v1_bundle: dict[str, Any],
        v2_bundle: dict[str, Any],
        gesture_config: Optional[GestureConfig] = None,
        motion_config: Optional[MotionConfig] = None,
    ) -> None:
        self.v1_bundle = v1_bundle
        self.v2_bundle = v2_bundle
        self.gesture_config = gesture_config or GestureConfig()
        self._motion = MotionTracker(motion_config or MotionConfig())
        self._gestures = GestureEngine(self.gesture_config)
        self._window = WindowFeatures(
            window_seconds=v2_bundle["window_seconds"],
            max_gap=v2_bundle["max_gap"],
            min_samples=v2_bundle["min_samples"],
            deadband=v2_bundle["deadband"],
        )

    def reset(self) -> None:
        self._motion.reset()
        self._gestures.reset()
        self._window.reset()

    def evaluate_frame(self, features: FrameFeatures, *, detected: bool = True) -> EvaluatedFrame:
        if not detected:
            self._motion.reset()
            self._gestures.reset()
            self._window.reset()
            return EvaluatedFrame(
                frame_index=features.frame_index,
                timestamp=features.timestamp,
                detected=False,
            )

        motion = self._motion.update(features)
        events = self._gestures.update(features, motion)
        window_state = self._window.push(features.timestamp, features)

        from backend.web.pipeline import PipelineRunner

        v1_pred = PipelineRunner._predict(self.v1_bundle, features, motion, window_state=None)
        v2_pred = PipelineRunner._predict(self.v2_bundle, features, motion, window_state=window_state)

        feat_dict = features.to_dict()
        static = rule_label(feat_dict, self.gesture_config.open_threshold)

        return EvaluatedFrame(
            frame_index=features.frame_index,
            timestamp=features.timestamp,
            detected=True,
            v1_label=None if v1_pred is None else v1_pred[0],
            v2_label=None if v2_pred is None else v2_pred[0],
            rule_static=static,
            gesture_events=events,
        )

    def run_frames(self, frames: list[tuple[FrameFeatures, bool]]) -> list[EvaluatedFrame]:
        out: list[EvaluatedFrame] = []
        for features, detected in frames:
            out.append(self.evaluate_frame(features, detected=detected))
        return out


def _rate(matches: int, comparable: int) -> Optional[float]:
    if comparable <= 0:
        return None
    return round(matches / comparable, 4)


def _agreement_pair(
    frames: list[EvaluatedFrame],
    left: str,
    right: str,
    *,
    v1_vocab_only: bool = False,
) -> dict[str, Any]:
    comparable = 0
    matches = 0
    for frame in frames:
        a = getattr(frame, left, None)
        b = getattr(frame, right, None)
        if a is None or b is None:
            continue
        if left == "v1_label" or right == "v1_label":
            if a not in COMMON_V1_LABELS or b not in COMMON_V1_LABELS:
                continue
        if v1_vocab_only and (a not in COMMON_V1_LABELS or b not in COMMON_V1_LABELS):
            continue
        # Static rule never emits wave; skip v2 wave for rule pairing (same as report.py).
        if left == "v2_label" and a == "wave":
            continue
        if right == "v2_label" and b == "wave":
            continue
        comparable += 1
        if a == b:
            matches += 1
    return {"comparable": comparable, "matches": matches, "rate": _rate(matches, comparable)}


def _expected_accuracy(frames: list[EvaluatedFrame], expected: str) -> dict[str, Any]:
    sources = {
        "v1": "v1_label",
        "v2": "v2_label",
        "rule": "rule_static",
    }
    out: dict[str, Any] = {}
    for name, attr in sources.items():
        comparable = 0
        matches = 0
        for frame in frames:
            label = getattr(frame, attr)
            if label is None:
                continue
            if name == "v1" and label not in COMMON_V1_LABELS:
                continue
            comparable += 1
            if label == expected:
                matches += 1
        out[name] = {"comparable": comparable, "matches": matches, "rate": _rate(matches, comparable)}
    return out


def _wave_coverage(
    frames: list[EvaluatedFrame], variant: str = ""
) -> dict[str, Any]:
    v2_wave_frames = sum(1 for f in frames if f.v2_label == "wave")

    def _wave_event(ev: GestureEvent) -> bool:
        if ev.name != "wave":
            return False
        if variant == "wave_left":
            return ev.side == "left"
        if variant == "wave_right":
            return ev.side == "right"
        return True

    rule_events = sum(
        1 for f in frames for ev in f.gesture_events if _wave_event(ev)
    )
    event_sides = [
        {"side": ev.side, "frame_index": ev.frame_index, "timestamp": ev.timestamp}
        for f in frames
        for ev in f.gesture_events
        if _wave_event(ev)
    ]
    return {
        "v2_labeled_frames": v2_wave_frames,
        "rule_events": rule_events,
        "rule_event_frames": event_sides,
        "temporal_unit_note": WAVE_TEMPORAL_NOTE,
    }


def _apply_segment_exclusions(
    segment_frames: list[EvaluatedFrame], spec: SegmentSpec
) -> tuple[list[EvaluatedFrame], dict[str, int]]:
    """Mark warmup at segment start and transition at segment end; return comparable slice."""
    excluded = {"warmup": 0, "transition": 0}
    if not segment_frames:
        return [], excluded
    n = len(segment_frames)
    warmup = min(spec.warmup, n)
    transition = min(spec.transition, max(0, n - warmup))
    for i in range(warmup):
        segment_frames[i].excluded = "warmup"
        excluded["warmup"] += 1
    for i in range(n - transition, n):
        if segment_frames[i].excluded:
            continue
        segment_frames[i].excluded = "transition"
        excluded["transition"] += 1
    comparable = [f for f in segment_frames if not f.excluded]
    return comparable, excluded


def summarize_segment(
    segment_frames: list[EvaluatedFrame], spec: SegmentSpec
) -> dict[str, Any]:
    comparable, excluded = _apply_segment_exclusions(segment_frames, spec)
    available = {
        "v1": sum(1 for f in segment_frames if f.v1_label is not None),
        "v2": sum(1 for f in segment_frames if f.v2_label is not None),
        "rule": sum(1 for f in segment_frames if f.rule_static is not None and f.detected),
    }
    agreement = {
        "v1_v2": _agreement_pair(comparable, "v1_label", "v2_label"),
        "v1_rule": _agreement_pair(comparable, "v1_label", "rule_static"),
        "v2_rule": _agreement_pair(comparable, "v2_label", "rule_static"),
    }
    body: dict[str, Any] = {
        "label": spec.label,
        "name": spec.name,
        "variant": spec.variant,
        "expected": spec.label,
        "frames": len(segment_frames),
        "excluded": excluded,
        "available": available,
        "agreement": agreement,
        "expected_accuracy": _expected_accuracy(comparable, spec.label),
    }
    if (
        spec.label == "wave"
        or spec.variant.startswith("wave_")
        or any(f.v2_label == "wave" for f in segment_frames)
    ):
        body["wave"] = _wave_coverage(segment_frames, spec.variant)
    return body


def evaluate_segments(
    evaluated: list[EvaluatedFrame], segments: list[SegmentSpec]
) -> dict[str, Any]:
    by_index = {f.frame_index: f for f in evaluated}
    results: list[dict[str, Any]] = []
    for spec in segments:
        chunk: list[EvaluatedFrame] = []
        for index in range(spec.start_index, spec.end_index + 1):
            frame = by_index.get(index)
            if frame is not None:
                chunk.append(frame)
        results.append(summarize_segment(chunk, spec))
    return {"schema_version": EVAL_SCHEMA_VERSION, "segments": results}


def evaluate_from_json(
    v1_bundle: dict[str, Any],
    v2_bundle: dict[str, Any],
    frames_json: list[dict[str, Any]],
    segments_json: list[dict[str, Any]],
    *,
    gesture_config: Optional[GestureConfig] = None,
) -> dict[str, Any]:
    session = LiveEvaluationSession(v1_bundle, v2_bundle, gesture_config=gesture_config)
    parsed = [frame_from_dict(row) for row in frames_json]
    evaluated = session.run_frames(parsed)
    segments = [SegmentSpec.from_dict(row) for row in segments_json]
    return evaluate_segments(evaluated, segments)
