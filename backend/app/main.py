"""RoomSpec AI FastAPI entrypoint."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.v1.routes import router as v1_router
from app.core.config import REPO_ROOT, get_settings
from app.core.metrics import registry
from app.db.bootstrap import bootstrap

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("roomspec")
logging.getLogger("httpx").setLevel(logging.WARNING)  # Qdrant client logs every HTTP call at INFO

SAMPLES_DIR = REPO_ROOT / "data" / "samples"


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.auto_seed:
        log.info("Bootstrap: %s", bootstrap())
    yield


app = FastAPI(
    title="RoomSpec AI",
    version=__version__,
    description=(
        "Computer vision + hybrid (vector + SQL) retrieval engine that turns a room photo and "
        "wall constraints into a dimensionally verified cabinet bill of materials."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST", "PUT", "PATCH"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.include_router(v1_router, prefix="/api/v1")


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    path = _route_template(request)
    if not path.startswith(("/static", "/samples")):
        registry.observe_request(request.method, path, response.status_code, time.perf_counter() - start)
    return response


def _route_template(request: Request) -> str:
    """Full route template, e.g. /api/v1/specs/{request_id}, so ids never become metric labels.

    Included routers are not flattened, so the matched route only knows its path relative to
    its router; the prefix is recovered from the concrete URL (no route uses slash-bearing params).
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if not template:
        return "/static" if request.url.path.startswith("/static") else "unmatched"
    actual = request.url.path.rstrip("/").split("/")
    depth = len(template.rstrip("/").split("/")) - 1
    return "/".join(actual[: len(actual) - depth]) + template


@app.get("/metrics", include_in_schema=False)
def metrics() -> PlainTextResponse:
    if not settings.metrics_enabled:
        return PlainTextResponse("metrics disabled\n", status_code=404)
    return PlainTextResponse(registry.render(), media_type="text/plain; version=0.0.4")


@app.get("/api/v1/samples", tags=["samples"])
def list_samples() -> list[dict[str, str]]:
    """Bundled sample room photos for trying the pipeline without an upload."""
    if not SAMPLES_DIR.exists():
        return []
    return [
        {"name": p.stem.replace("_", " ").title(), "url": f"/samples/{p.name}"}
        for p in sorted(SAMPLES_DIR.glob("*.jpg"))
        if not p.name.startswith("._")
    ]


if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")


app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")
