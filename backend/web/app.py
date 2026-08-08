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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.web.payload import hello_message
from backend.web.pipeline import PipelineRunner, Subscriber, WebConfig

#: backend/web/app.py -> backend/web -> backend -> project root
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


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
        }

    @app.websocket("/ws")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json(
            hello_message(
                visibility_threshold=pipeline.config.visibility_threshold,
                still_threshold=pipeline.config.motion.still_threshold,
                preview_width=pipeline.config.preview_width,
                jpeg_quality=pipeline.config.jpeg_quality,
            )
        )

        # Subscribing after the hello guarantees the client holds the landmark
        # table and the bone topology before the first frame that needs them.
        subscriber = pipeline.subscribe(asyncio.get_running_loop())
        writer = asyncio.create_task(_write_frames(websocket, subscriber))
        reader = asyncio.create_task(_watch_for_disconnect(websocket))
        try:
            done, pending = await asyncio.wait(
                {writer, reader}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()  # surface a real bug instead of swallowing it
        except WebSocketDisconnect:
            pass
        finally:
            pipeline.unsubscribe(subscriber)

    return app


async def _write_frames(websocket: WebSocket, subscriber: Subscriber) -> None:
    """Forward messages until the pipeline reports a fatal error."""
    while True:
        message = await subscriber.get()
        await websocket.send_json(message)
        if message.get("type") == "error" and message.get("fatal"):
            return


async def _watch_for_disconnect(websocket: WebSocket) -> None:
    """Read from a client that never writes, purely to notice when it leaves.

    Anything the browser does send is ignored on purpose: the UI toggles
    (mirror, trail, labels) are drawing choices, handled client-side.
    """
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        return


#: Convenience target for `uvicorn backend.web.app:app` with default settings.
app = create_app()
