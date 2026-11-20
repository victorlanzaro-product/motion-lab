"""Shared static-rule label for live frames and report agreement."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def rule_label(features: Mapping[str, Any], open_threshold: float) -> Optional[str]:
    """Return the first known static-rule label, or ``None`` if uncertain.

    The rule order matches the report's S10 logic: crossed, open, raised,
    idle. A missing predicate value cannot be treated as a negative because
    that would turn an occluded joint into a confident ``idle`` verdict.
    """
    crossed = features.get("wrists_crossed")
    if crossed is True:
        return "arms_crossed"
    if crossed is not False:
        return None

    wrist_distance = features.get("wrist_distance")
    if wrist_distance is None:
        return None
    if wrist_distance > open_threshold:
        return "arms_open"

    left_raised = features.get("left_wrist_above_shoulder")
    right_raised = features.get("right_wrist_above_shoulder")
    if left_raised is True or right_raised is True:
        return "arm_raised"
    if left_raised is False and right_raised is False:
        return "idle"
    return None
