"""Pose + Face Engines: MediaPipe in, Motion Lab landmark types out."""

from backend.vision.drawing import draw_arms, draw_hud, draw_joint_values, draw_trail
from backend.vision.face_detector import (
    DEFAULT_MODEL_PATH as DEFAULT_FACE_MODEL_PATH,
)
from backend.vision.face_detector import (
    MODEL_URL as FACE_MODEL_URL,
)
from backend.vision.face_detector import (
    FaceConfig,
    FaceDetector,
    FaceModelError,
)
from backend.vision.face_detector import ensure_model as ensure_face_model
from backend.vision.face_landmarks import (
    FACE_CHAIN_GROUPS,
    FACE_MESH_CONTRACT_VERSION,
    FACE_OVERLAY_LANDMARKS,
    FaceSnapshot,
    euler_from_matrix,
)
from backend.vision.landmarks import (
    ARM_CHAINS,
    ARM_LANDMARKS,
    DEFAULT_VISIBILITY_THRESHOLD,
    Point,
    PoseLandmark,
    PoseSnapshot,
    Side,
    to_pixel,
)
from backend.vision.pose_detector import (
    DEFAULT_MODEL_PATH,
    MODEL_URL,
    PoseConfig,
    PoseDetector,
    PoseModelError,
    ensure_model,
)

__all__ = [
    "ARM_CHAINS",
    "ARM_LANDMARKS",
    "DEFAULT_FACE_MODEL_PATH",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_VISIBILITY_THRESHOLD",
    "FACE_CHAIN_GROUPS",
    "FACE_MESH_CONTRACT_VERSION",
    "FACE_MODEL_URL",
    "FACE_OVERLAY_LANDMARKS",
    "FaceConfig",
    "FaceDetector",
    "FaceModelError",
    "FaceSnapshot",
    "MODEL_URL",
    "Point",
    "PoseConfig",
    "PoseDetector",
    "PoseLandmark",
    "PoseModelError",
    "PoseSnapshot",
    "Side",
    "draw_arms",
    "draw_hud",
    "draw_joint_values",
    "draw_trail",
    "ensure_face_model",
    "ensure_model",
    "euler_from_matrix",
    "to_pixel",
]
