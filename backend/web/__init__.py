"""Web App: the pipeline as a live stream, one camera shared by many browsers.

Importing this package pulls in FastAPI, which the Sprint 01-04 scripts do not
need — it lives behind the `web` extra.
"""

from backend.web.app import FRONTEND_DIR, create_app
from backend.web.payload import (
    encode_preview,
    error_message,
    frame_message,
    hello_message,
    landmarks_payload,
    status_message,
    trail_payload,
)
from backend.web.pipeline import PipelineRunner, Subscriber, WebConfig

__all__ = [
    "FRONTEND_DIR",
    "PipelineRunner",
    "Subscriber",
    "WebConfig",
    "create_app",
    "encode_preview",
    "error_message",
    "frame_message",
    "hello_message",
    "landmarks_payload",
    "status_message",
    "trail_payload",
]
