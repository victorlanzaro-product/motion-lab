"""Overlay drawing: the arm skeleton and the stats panel.

This lives in `vision/` rather than in the script because the bone topology is
domain knowledge, not presentation glue. In Sprint 05 the browser will draw the
same chains from the same landmark payload.
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.vision.landmarks import (
    ARM_CHAINS,
    ARM_LANDMARKS,
    DEFAULT_VISIBILITY_THRESHOLD,
    Point,
    PoseSnapshot,
    to_pixel,
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_ACCENT = (120, 255, 120)  # BGR: reliable landmark
_WEAK = (80, 160, 255)  # BGR: low visibility, MediaPipe is guessing
_DIM = (200, 200, 200)
_BONE = (255, 200, 90)
_TRAIL = (255, 130, 220)


def draw_arms(
    image: np.ndarray,
    snapshot: PoseSnapshot,
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    labels: bool = False,
) -> np.ndarray:
    """Draw shoulder->elbow->wrist over `image`, in place.

    A bone is only drawn when both of its endpoints are reliable, so a hidden
    wrist shows as a missing segment instead of a line snapping to a guess.
    """
    if not snapshot.detected:
        return image

    height, width = image.shape[:2]

    for start, end in ARM_CHAINS:
        a, b = snapshot.point(start), snapshot.point(end)
        if a is None or b is None:
            continue
        if not (a.is_reliable(threshold) and b.is_reliable(threshold)):
            continue
        cv2.line(
            image, to_pixel(a, width, height), to_pixel(b, width, height), _BONE, 3, cv2.LINE_AA
        )

    for landmark in ARM_LANDMARKS:
        point = snapshot.point(landmark)
        if point is None:
            continue
        center = to_pixel(point, width, height)
        reliable = point.is_reliable(threshold)
        color = _ACCENT if reliable else _WEAK
        cv2.circle(image, center, 8, color, -1, cv2.LINE_AA)
        cv2.circle(image, center, 8, (20, 20, 20), 1, cv2.LINE_AA)
        if labels:
            cv2.putText(
                image,
                f"{landmark.name} {point.visibility:.2f}",
                (center[0] + 12, center[1] - 8),
                _FONT,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    return image


def draw_joint_values(
    image: np.ndarray,
    snapshot: PoseSnapshot,
    values: dict[int, float],
    suffix: str = "",
) -> np.ndarray:
    """Print a number next to a joint — e.g. the live elbow angle.

    `values` is keyed by landmark index so this stays generic: Sprint 04 reuses
    it for wrist velocity without touching this function.
    """
    if not snapshot.detected:
        return image

    height, width = image.shape[:2]
    for index, value in values.items():
        point = snapshot.normalized.get(int(index))
        if point is None or value is None:
            continue
        x, y = to_pixel(point, width, height)
        text = f"{value:.0f}{suffix}"
        # Dark outline first so the text survives a bright background.
        cv2.putText(image, text, (x + 14, y + 6), _FONT, 0.7, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(image, text, (x + 14, y + 6), _FONT, 0.7, _ACCENT, 1, cv2.LINE_AA)
    return image


def draw_trail(
    image: np.ndarray,
    segments: list[list[Point]],
    color: tuple[int, int, int] = _TRAIL,
) -> np.ndarray:
    """Draw a wrist trajectory as one polyline per continuous run.

    Takes plain segments rather than the `Trail` object so `vision/` never has
    to import `features/` — the dependency only ever points one way.

    Older points are drawn thinner: the taper reads as direction of travel
    without needing an arrowhead.
    """
    height, width = image.shape[:2]
    for run in segments:
        pixels = [to_pixel(p, width, height) for p in run]
        for i in range(1, len(pixels)):
            weight = i / len(pixels)  # 0 = oldest, 1 = newest
            cv2.line(
                image,
                pixels[i - 1],
                pixels[i],
                color,
                max(1, int(round(1 + 3 * weight))),
                cv2.LINE_AA,
            )
    return image


def draw_hud(image: np.ndarray, lines: list[str], width_px: int = 340) -> np.ndarray:
    """Semi-transparent stats panel, top-left. First line is the title."""
    panel_h = 24 * len(lines) + 16
    overlay = image.copy()
    cv2.rectangle(overlay, (12, 12), (12 + width_px, 12 + panel_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, dst=image)

    y = 36
    for i, line in enumerate(lines):
        cv2.putText(image, line, (24, y), _FONT, 0.55, _ACCENT if i == 0 else _DIM, 1, cv2.LINE_AA)
        y += 24
    return image
