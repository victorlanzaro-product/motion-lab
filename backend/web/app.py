"""Web App — the FastAPI shell around the pipeline.

This layer is deliberately thin. It does not know what a landmark is: it opens a
WebSocket, subscribes to the `PipelineRunner`, and forwards whatever the runner
publishes. All the vision lives upstream, all the drawing lives in the browser.

Two things it does own:

  * **the disconnect.** A closed tab has to reach the runner promptly, or the
    camera stays on with nobody watching. Each connection therefore runs a
    reader alongside the writer — the browser sends nothing, but `receive` is
    what raises when the socket dies, and a send-only handler would not notice
    until the next frame failed to write;
  * **the failure.** A pipeline error is forwarded and the socket closed rather
    than left hanging, so the page can say why the video stopped.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.web.payload import hello_message
from backend.web.pipeline import PipelineRunner, Subscriber, WebConfig

#: backend/web/app.py -> backend/web -> backend -> project root
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
logger = logging.getLogger(__name__)
MAX_WS_MESSAGE_BYTES = 4 * 1024
REPORT_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; sandbox; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
}


def create_app(
    config: Optional[WebConfig] = None,
    runner: Optional[PipelineRunner] = None,
    frontend_dir: Path = FRONTEND_DIR,
) -> FastAPI:
    """Build the app. `runner` is injectable so tests can supply a fake camera."""
    pipeline = runner if runner is not None else PipelineRunner(config or WebConfig())

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        # Server shutting down: release the webcam even if a client is still
        # attached. A daemon thread would die anyway, but not necessarily before
        # the device handle is closed.
        pipeline.stop()

    app = FastAPI(title="Motion Lab", version="0.5.0", lifespan=lifespan)
    app.state.runner = pipeline

    if frontend_dir.is_dir():
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    @app.get("/health")
    async def health() -> dict:
        """Enough to tell "server up, camera off" from "camera failed"."""
        return {
            "status": "ok",
            "pipeline": pipeline.state,
            "clients": pipeline.client_count,
            "frames": pipeline.frames_served,
            "error": pipeline.error,
            # Independent of `error` on purpose: the face signal is optional
            # (Sprint 11) and can be unavailable while pose/web are perfectly
            # healthy -- `pipeline.state`/`error` must keep meaning "the whole
            # pipeline", not get muddied by an opt-in layer's own failure.
            "face_error": pipeline.face_error,
            "recording": pipeline.recording_health(),
        }

    @app.get("/report")
    async def report() -> Response:
        """Serve the last Sprint 10 report without generating it in a request."""
        try:
            html = await asyncio.to_thread(_read_saved_report, pipeline)
        except (FileNotFoundError, IsADirectoryError):
            return PlainTextResponse(
                "relatório ainda não gerado — rode scripts/report.py",
                status_code=404,
                headers=REPORT_HEADERS,
            )
        except (OSError, UnicodeError):
            logger.exception("Could not read the saved report")
            return PlainTextResponse(
                "Não foi possível ler o relatório salvo.", status_code=500, headers=REPORT_HEADERS
            )
        return HTMLResponse(html, headers=REPORT_HEADERS)

    @app.websocket("/ws")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        report_available = await asyncio.to_thread(pipeline.report_available)
        await websocket.send_json(
            hello_message(
                visibility_threshold=pipeline.config.visibility_threshold,
                still_threshold=pipeline.config.motion.still_threshold,
                preview_width=pipeline.config.preview_width,
                jpeg_quality=pipeline.config.jpeg_quality,
                ml_classes=pipeline.ml_classes(),
                face_enabled=pipeline.config.face_enabled,
                report_available=report_available,
            )
        )

        # Subscribing after the hello guarantees the client holds the landmark
        # table and the bone topology before the first frame that needs them.
        subscriber = pipeline.subscribe(asyncio.get_running_loop())
        # Every client receives the current shared recorder state immediately
        # after hello, including a second tab joining an active session.
        await websocket.send_json(pipeline.recording_snapshot_for(subscriber))
        writer = asyncio.create_task(_write_frames(websocket, subscriber))
        reader = asyncio.create_task(_read_client_messages(websocket, pipeline, subscriber))
        try:
            done, pending = await asyncio.wait(
                {writer, reader}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            # Always let a cancelled task finish unwinding before this handler
            # returns -- an un-awaited task is still mid-cleanup (or could
            # raise something other than `CancelledError`) after this
            # coroutine has already moved on, which used to surface as
            # "Task exception was never retrieved" on server shutdown.
            # `return_exceptions=True` means this call itself never raises.
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()  # surface a real bug instead of swallowing it
        except (WebSocketDisconnect, asyncio.CancelledError):
            # `CancelledError` here is the ASGI server tearing this same
            # coroutine down from outside (shutdown, or the test client's own
            # teardown) while we are still waiting on `pending` above -- the
            # same "client is gone" signal `WebSocketDisconnect` already
            # models, not a bug to propagate. Cleanup below still runs, and
            # the very next checkpoint re-raises into whatever cancelled us,
            # so the cancellation itself is never actually lost.
            pass
        finally:
            for task in (writer, reader):
                if not task.done():
                    task.cancel()
            pipeline.unsubscribe(subscriber)

    return app


def _read_saved_report(pipeline: PipelineRunner) -> str:
    path = pipeline.safe_report_path()
    if path is None:
        raise FileNotFoundError
    return path.read_text(encoding="utf-8")


async def _write_frames(websocket: WebSocket, subscriber: Subscriber) -> None:
    """Forward messages until the pipeline reports a fatal error."""
    while True:
        message = await subscriber.get()
        await websocket.send_json(message)
        if message.get("type") == "error" and message.get("fatal"):
            return


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


async def _read_client_messages(
    websocket: WebSocket, pipeline: PipelineRunner, subscriber: Subscriber
) -> None:
    """Handle recorder commands and ignore every other browser message.

    The visual toggles remain client-side. `receive()` lets malformed JSON,
    binary messages, and unknown message types be ignored without ending the
    connection that also carries the live camera stream.
    """
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return
        raw = message.get("text")
        if not isinstance(raw, str):
            binary = message.get("bytes")
            if binary is not None and len(binary) > MAX_WS_MESSAGE_BYTES:
                logger.warning("Ignoring oversized binary WebSocket message")
            else:
                logger.debug("Ignoring non-text WebSocket message")
            continue
        try:
            if len(raw) > MAX_WS_MESSAGE_BYTES or len(raw.encode("utf-8")) > MAX_WS_MESSAGE_BYTES:
                logger.warning("Ignoring oversized text WebSocket message")
                continue
            command = json.loads(raw, parse_constant=_reject_json_constant)
        except (ValueError, RecursionError):
            logger.debug("ignoring invalid JSON WebSocket message")
            continue
        if not isinstance(command, dict):
            logger.debug("ignoring non-object WebSocket JSON message")
            continue

        command_type = command.get("type")
        if command_type == "record_start":
            pipeline.record_start(command.get("label"), subscriber)
        elif command_type == "record_stop":
            pipeline.record_stop(subscriber)
        else:
            logger.debug("ignoring unknown WebSocket message type: %r", command_type)
            continue


#: Convenience target for `uvicorn backend.web.app:app` with default settings.
app = create_app()
