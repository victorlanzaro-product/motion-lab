"""Face landmark model — what MediaPipe's Face Landmarker returns, expressed in
Motion Lab's own types.

Same rule as `landmarks.py` (pose): the feature layer should not import
MediaPipe. If the face estimator is ever swapped, only this file and
`face_detector.py` change.

Three decisions carry this file:

1. **Blendshapes, not raw mesh geometry.** MediaPipe's Face Landmarker already
   exposes 52 named "how much" scores (0..1) — `mouthSmileLeft`, `jawOpen`,
   `eyeBlinkLeft`, `browInnerUp`, and so on — tuned against a real face rig.
   Re-deriving "smile" from the 478 raw mesh points ourselves would be a
   second, weaker source of truth competing with a better one MediaPipe
   already computed. `FaceSnapshot.blendshapes` carries the raw dict; the
   features layer (`features/face_features.py`) picks out the named entries
   it needs, the same division of labour `positions.py`/`angles.py` already
   have with `PoseSnapshot`.
2. **No fabricated confidence.** Pose Landmarker gives every joint a
   `visibility`/`presence` score, so `PoseSnapshot.confidence()` means
   something. Face Landmarker does not expose an equivalent single
   face-tracking confidence — only per-blendshape activation scores and the
   mesh itself. Inventing a number here would look exactly as trustworthy as
   the real thing while measuring nothing. `FaceSnapshot.detected` (whether a
   face landmarks list came back at all) is the one honest availability
   signal; `confidence` is deliberately not part of this type. See the
   Sprint 11 note in `README.md`.
3. **Head orientation is pure math, not a MediaPipe type.** The 4x4 facial
   transformation matrix is converted to yaw/pitch/roll degrees right here,
   so nothing downstream ever holds a matrix or a MediaPipe object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from backend.vision.landmarks import Point

#: Sprint 11 face overlay — the same "named chains, not the full mesh" call
#: `ARM_CHAINS` makes for the body. MediaPipe's Face Landmarker returns 478
#: raw points; only seven regions of that mesh actually read as a face to a
#: person looking at the overlay (jawline, eyes, eyebrows, nose bridge,
#: lips). The rest is the tesselation used for AR effects, not for a human to
#: read — drawing all of it would be `FACEMESH_TESSELATION`-dense noise on
#: top of the video, not a legible face.
#:
#: Index pairs are copied verbatim from MediaPipe's own topology table
#: (`mediapipe/python/solutions/face_mesh_connections.py`, Apache-2.0) —
#: reference data about which of the 478 points are anatomically adjacent,
#: the same kind of fixed lookup table `PoseLandmark`/`ARM_CHAINS` already
#: hardcode for BlazePose. A "malha completa" (full 478-point tesselation)
#: mode is deliberately not offered: it would roughly triple the per-frame
#: payload for a denser, harder-to-read picture, with no signal this sprint's
#: "reações faciais" feature needs that the seven named regions do not
#: already carry.
FACE_OVAL: tuple[tuple[int, int], ...] = (
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
)

LEFT_EYE: tuple[tuple[int, int], ...] = (
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
)

RIGHT_EYE: tuple[tuple[int, int], ...] = (
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
)

LEFT_EYEBROW: tuple[tuple[int, int], ...] = (
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334),
    (334, 296), (296, 336),
)

RIGHT_EYEBROW: tuple[tuple[int, int], ...] = (
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66),
    (66, 107),
)

NOSE: tuple[tuple[int, int], ...] = (
    (168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1), (1, 19),
    (19, 94), (94, 2), (98, 97), (97, 2), (2, 326), (326, 327), (327, 294),
    (294, 278), (278, 344), (344, 440), (440, 275), (275, 4), (4, 45),
    (45, 220), (220, 115), (115, 48), (48, 64), (64, 98),
)

LIPS: tuple[tuple[int, int], ...] = (
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
    (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270),
    (270, 409), (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
    (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312),
    (312, 311), (311, 310), (310, 415), (415, 308),
)

#: One group per overlay region, in the order the frontend draws them (outer
#: face first so the eyes/brows/nose/lips lines paint on top). The frontend
#: also uses these keys — `left_eye`/`right_eye`/`lips` — to pick which
#: region's color responds to an observable score (see `app.js:faceGroupColor`).
FACE_CHAIN_GROUPS: dict[str, tuple[tuple[int, int], ...]] = {
    "face_oval": FACE_OVAL,
    "left_eyebrow": LEFT_EYEBROW,
    "right_eyebrow": RIGHT_EYEBROW,
    "left_eye": LEFT_EYE,
    "right_eye": RIGHT_EYE,
    "nose": NOSE,
    "lips": LIPS,
}

#: Flat topology for anything that wants every bone regardless of region.
FACE_CHAINS: tuple[tuple[int, int], ...] = tuple(
    pair for group in FACE_CHAIN_GROUPS.values() for pair in group
)

#: Every point index any chain above touches — what actually needs to travel
#: on the wire. Roughly 130 of the mesh's 478 points, not all 478.
FACE_OVERLAY_LANDMARKS: tuple[int, ...] = tuple(
    sorted({index for pair in FACE_CHAINS for index in pair})
)

#: Bumped only if `FACE_CHAIN_GROUPS`/`FACE_OVERLAY_LANDMARKS` ever change
#: shape — lets a client tell "server sent a different topology than I
#: expect" apart from "server is just older and has no topology at all".
FACE_MESH_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class FaceSnapshot:
    """One frame's worth of face signal, or the absence of one.

    `blendshapes` is empty when no face was detected — never filled with
    fabricated zeros (the same `None`-means-missing discipline `PoseSnapshot`
    and `FrameFeatures` already use). `mesh` follows the same rule: empty
    when no face was detected, and — unlike `PoseSnapshot.normalized` — never
    carries a per-point visibility/presence score, because the Face
    Landmarker does not expose one (module docstring, point 2, applies to
    the raw mesh exactly as much as it does to blendshapes). A point is
    either in `mesh` because a face was detected this frame, or it is not in
    `mesh` at all — there is no fabricated "half-reliable" landmark.
    """

    frame_index: int
    timestamp: float
    detected: bool
    blendshapes: Mapping[str, float] = field(default_factory=dict)
    #: Raw, unmirrored normalized coordinates for `FACE_OVERLAY_LANDMARKS`
    #: only — not the full 478-point mesh (see that constant's docstring).
    mesh: Mapping[int, Point] = field(default_factory=dict)
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    head_roll: Optional[float] = None
    inference_ms: float = 0.0

    def blendshape(self, name: str) -> Optional[float]:
        return self.blendshapes.get(name)

    @classmethod
    def empty(
        cls, frame_index: int, timestamp: float, inference_ms: float = 0.0
    ) -> "FaceSnapshot":
        return cls(
            frame_index=frame_index,
            timestamp=timestamp,
            detected=False,
            blendshapes={},
            mesh={},
            inference_ms=inference_ms,
        )


def euler_from_matrix(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """4x4 (or 3x3) rotation matrix -> (yaw, pitch, roll) in degrees.

    Assumes a right-handed, Y-up camera frame (X right, Y up, Z towards the
    camera) — MediaPipe's own convention — so yaw is rotation about Y
    (turning the head left/right), pitch about X (nodding up/down), roll
    about Z (tilting side to side). Decomposes `R = Ry(yaw) @ Rx(pitch) @
    Rz(roll)` (row-major, `matrix[row][col]`):

        pitch = asin(-R[1][2])
        yaw   = atan2(R[0][2], R[2][2])
        roll  = atan2(R[1][0], R[1][1])

    Verified against synthetic single-axis rotation matrices (see
    `tests/test_face.py`) — a pure Y-axis rotation decodes to yaw alone, X to
    pitch alone, Z to roll alone. What is *not* independently verified against
    a live capture is that MediaPipe composes its own transformation matrix in
    this exact order — a different composition order would still decompose
    correctly for a rotation on a single axis (the three checks above), but
    could couple the reported values differently for a simultaneous
    multi-axis head turn. Documented as a known limitation, not silently
    assumed correct.
    """
    r = matrix
    pitch = math.asin(max(-1.0, min(1.0, -r[1][2])))
    if abs(r[1][2]) < 0.9999:
        yaw = math.atan2(r[0][2], r[2][2])
        roll = math.atan2(r[1][0], r[1][1])
    else:
        # Gimbal lock (looking straight up/down): yaw and roll become the same
        # rotation. Roll is reported as 0 rather than an arbitrary split.
        yaw = math.atan2(-r[2][0], r[0][0])
        roll = 0.0
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))
