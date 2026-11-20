"""Interactive evaluation capture — one camera session, metrics only.

Opens CameraEngine + PoseDetector only after ML bundles validate. Each segment is
recorded in the same continuous frame stream and evaluated through
`LiveEvaluationSession` (no image/video/face persistence).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from backend.camera import CameraEngine
from backend.features import extract
from backend.gestures import GestureConfig
from backend.ml.live_eval import (
    EvaluatedFrame,
    LiveEvaluationSession,
    SegmentSpec,
    evaluate_segments,
)
from backend.vision import DEFAULT_VISIBILITY_THRESHOLD, PoseDetector

# Rótulos base do roteiro S16 e variantes nomeadas que Victor pode usar.
BASE_EVAL_LABELS: tuple[str, ...] = (
    "idle",
    "arm_raised",
    "arms_open",
    "arms_crossed",
    "wave",
)

LABEL_VARIANTS: dict[str, tuple[str, str]] = {
    "wave_left": ("wave", "wave_left"),
    "wave_right": ("wave", "wave_right"),
    "arm_raised_no_reversal": ("arm_raised", "arm_raised_no_reversal"),
}

PromptFn = Callable[[str], str]
PrintFn = Callable[[str], None]


@dataclass(frozen=True)
class SegmentCaptureMeta:
    """Metadata Victor supplies before each recorded segment."""

    name: str
    label: str
    variant: str = ""
    warmup: int = 0
    transition: int = 0
    frame_count: Optional[int] = None  # tests / scripted plans only


def parse_expected_label(raw: str) -> tuple[str, str]:
    """Resolve user text to canonical label + optional variant (Portuguese errors)."""
    text = raw.strip()
    if not text:
        raise ValueError("Informe um rótulo esperado.")
    lower = text.lower().replace(" ", "_")
    if lower in LABEL_VARIANTS:
        return LABEL_VARIANTS[lower]
    if lower in BASE_EVAL_LABELS:
        return lower, ""
    aliases = {
        "braco_esquerdo": ("wave", "wave_left"),
        "braco_direito": ("wave", "wave_right"),
        "subida_sem_reversao": ("arm_raised", "arm_raised_no_reversal"),
    }
    if lower in aliases:
        return aliases[lower]
    allowed = ", ".join([*BASE_EVAL_LABELS, *LABEL_VARIANTS])
    raise ValueError(f"Rótulo inválido {raw!r}. Use um de: {allowed}.")


def meta_to_segment_spec(
    meta: SegmentCaptureMeta, start_index: int, end_index: int
) -> SegmentSpec:
    return SegmentSpec(
        label=meta.label,
        start_index=start_index,
        end_index=end_index,
        warmup=meta.warmup,
        transition=meta.transition,
        name=meta.name,
        variant=meta.variant,
    )


class CapturePrompter(Protocol):
    def next_segment(self) -> Optional[SegmentCaptureMeta]:
        ...

    def wait_stop_recording(self) -> None:
        ...


@dataclass
class StdinCapturePrompter:
    """Interactive segment entry on the terminal (Victor)."""

    prompt: PromptFn = input
    emit: PrintFn = print
    default_warmup: int = 0
    default_transition: int = 0

    def next_segment(self) -> Optional[SegmentCaptureMeta]:
        self.emit(
            "Novo trecho: nome curto, rótulo esperado (ou variante wave_left/wave_right/"
            "arm_raised_no_reversal), warmup e transição. Linha vazia encerra."
        )
        name = self.prompt("nome do trecho: ").strip()
        if not name:
            return None
        label_raw = self.prompt("rótulo esperado: ").strip()
        try:
            label, variant = parse_expected_label(label_raw)
        except ValueError as exc:
            self.emit(str(exc))
            return self.next_segment()
        warmup_text = self.prompt(f"warmup frames [{self.default_warmup}]: ").strip()
        transition_text = self.prompt(
            f"transição frames [{self.default_transition}]: "
        ).strip()
        warmup = self.default_warmup if not warmup_text else int(warmup_text)
        transition = (
            self.default_transition if not transition_text else int(transition_text)
        )
        return SegmentCaptureMeta(
            name=name,
            label=label,
            variant=variant,
            warmup=warmup,
            transition=transition,
        )

    def wait_stop_recording(self) -> None:
        self.prompt("Gravando… Enter para encerrar o trecho.")


@dataclass
class ScriptedCapturePrompter:
    """Deterministic segment plan for tests (no stdin)."""

    _plan: list[SegmentCaptureMeta]

    def next_segment(self) -> Optional[SegmentCaptureMeta]:
        if not self._plan:
            return None
        return self._plan[0]

    def wait_stop_recording(self) -> None:
        return None

    def consume(self) -> None:
        if self._plan:
            self._plan.pop(0)


def run_capture_session(
    camera: CameraEngine,
    detector: PoseDetector,
    eval_session: LiveEvaluationSession,
    prompter: CapturePrompter,
    *,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> tuple[list[EvaluatedFrame], list[SegmentSpec]]:
    """Single continuous capture; segments are index ranges on the same session."""
    evaluated: list[EvaluatedFrame] = []
    segment_specs: list[SegmentSpec] = []

    with camera, detector:
        while True:
            meta = prompter.next_segment()
            if meta is None:
                break
            if isinstance(prompter, ScriptedCapturePrompter):
                prompter.consume()

            start_index: Optional[int] = None
            frames_in_segment = 0
            stop = threading.Event()

            if meta.frame_count is None:

                def _waiter() -> None:
                    prompter.wait_stop_recording()
                    stop.set()

                threading.Thread(target=_waiter, daemon=True).start()
            else:
                stop.clear()

            while True:
                if meta.frame_count is not None and frames_in_segment >= meta.frame_count:
                    break
                if meta.frame_count is None and stop.is_set() and frames_in_segment > 0:
                    break
                if meta.frame_count is None and stop.is_set() and frames_in_segment == 0:
                    # spurious early enter before first frame — keep going briefly
                    stop.clear()
                    threading.Thread(
                        target=lambda: (prompter.wait_stop_recording(), stop.set()),
                        daemon=True,
                    ).start()

                frame = camera.read()
                if frame is None:
                    continue

                snapshot = detector.detect(frame)
                features = extract(snapshot, visibility_threshold)
                row = eval_session.evaluate_frame(features, detected=snapshot.detected)
                evaluated.append(row)
                if start_index is None:
                    start_index = row.frame_index
                frames_in_segment += 1

                if meta.frame_count is not None and frames_in_segment >= meta.frame_count:
                    break

            if start_index is None:
                continue
            end_index = evaluated[-1].frame_index
            segment_specs.append(meta_to_segment_spec(meta, start_index, end_index))

    return evaluated, segment_specs


def build_capture_report(
    evaluated: list[EvaluatedFrame], segment_specs: list[SegmentSpec]
) -> dict[str, Any]:
    return evaluate_segments(evaluated, segment_specs)


def run_interactive_evaluation(
    v1_bundle: dict[str, Any],
    v2_bundle: dict[str, Any],
    camera: CameraEngine,
    detector: PoseDetector,
    prompter: CapturePrompter,
    *,
    gesture_config: Optional[GestureConfig] = None,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> dict[str, Any]:
    session = LiveEvaluationSession(v1_bundle, v2_bundle, gesture_config=gesture_config)
    init_camera_and_pose(camera, detector)
    evaluated, specs = run_capture_session(
        camera,
        detector,
        session,
        prompter,
        visibility_threshold=visibility_threshold,
    )
    return build_capture_report(evaluated, specs)


def init_camera_and_pose(
    camera: CameraEngine,
    detector: PoseDetector,
) -> None:
    """Fail fast with Portuguese guidance if hardware/model cannot start."""
    try:
        camera.open()
    except Exception as exc:  # noqa: BLE001 - boundary for CLI
        camera.close()
        raise RuntimeError(
            "Não foi possível iniciar a câmera. Verifique a conexão e a permissão "
            f"de câmera. Detalhe: {exc}"
        ) from exc
    try:
        detector.open()
    except Exception as exc:  # noqa: BLE001 - boundary for CLI
        detector.close()
        camera.close()
        raise RuntimeError(f"Não foi possível iniciar o modelo de pose: {exc}") from exc
