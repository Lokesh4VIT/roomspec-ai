"""RoomSpec AI FastAPI entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.v1.routes import router as v1_router
from app.core.config import REPO_ROOT, get_settings
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(v1_router, prefix="/api/v1")


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
