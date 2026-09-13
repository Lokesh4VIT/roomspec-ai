"""FastAPI v1 routers: health, catalog, search, spec (image + constraints) and BOM."""

from __future__ import annotations

import json
from html import escape
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, status
from pydantic import ValidationError

from app import __version__
from app.core.config import get_settings
from app.core.finishes import swatch_rgb
from app.db import postgres, qdrant
from app.schemas.spec import (
    DesignConstraints,
    HealthResponse,
    Module,
    SearchRequest,
    SearchResponse,
    SpecResponse,
)
from app.services.cv_preprocessor import ImageValidationError
from app.services.embedding_engine import get_embedder
from app.services.genai_bom import _provider
from app.services.hybrid_retriever import search_catalog
from app.services.pipeline import run_spec_pipeline

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


# --------------------------------------------------------------------------- health
@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    s = get_settings()
    try:
        rows = postgres.count_modules()
        db_state = postgres.get_engine().dialect.name
    except Exception as exc:  # pragma: no cover - surfaced to monitoring
        rows, db_state = 0, f"error: {type(exc).__name__}"
    try:
        points = qdrant.collection_count()
        vs_state = "qdrant-remote" if s.qdrant_url else "qdrant-embedded"
    except Exception as exc:  # pragma: no cover
        points, vs_state = 0, f"error: {type(exc).__name__}"
    ok = rows > 0 and points == rows and not db_state.startswith("error")
    return HealthResponse(
        status="ok" if ok else "degraded",
        version=__version__,
        database=db_state,
        catalog_rows=rows,
        vector_store=vs_state,
        vector_points=points,
        embedding_backend=get_embedder().name,
        llm_provider=_provider(),
    )


# --------------------------------------------------------------------------- catalog
@router.get("/catalog", response_model=list[Module], tags=["catalog"])
def list_catalog(
    category: Annotated[list[str] | None, Query()] = None,
    finish_style: str | None = None,
    max_width_cm: Annotated[float | None, Query(gt=0)] = None,
    max_depth_cm: Annotated[float | None, Query(gt=0)] = None,
    max_price_usd: Annotated[float | None, Query(gt=0)] = None,
    in_stock_only: bool = False,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[dict]:
    """Pure SQL constraint filter over the catalog (no vector search)."""
    return postgres.filter_modules(
        categories=category,
        finish_style=finish_style,
        max_width_cm=max_width_cm,
        max_depth_cm=max_depth_cm,
        max_price_usd=max_price_usd,
        in_stock_only=in_stock_only,
        limit=limit,
    )


@router.get("/catalog/facets", tags=["catalog"])
def catalog_facets() -> dict[str, list[str]]:
    return {
        "categories": postgres.distinct_values("category"),
        "finish_styles": postgres.distinct_values("finish_style"),
        "cabinet_finishes": postgres.distinct_values("finish_style", exclude_categories=["Countertop"]),
    }


@router.get("/catalog/{part_id}", response_model=Module, tags=["catalog"])
def get_module(part_id: str) -> dict:
    row = postgres.get_modules_by_ids([part_id]).get(part_id)
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown part_id {part_id!r}")
    return row


@router.get("/catalog/{part_id}/thumbnail.svg", tags=["catalog"], response_class=Response)
def module_thumbnail(part_id: str) -> Response:
    """Generated front-elevation drawing of a module, to scale, in its finish colour."""
    row = postgres.get_modules_by_ids([part_id]).get(part_id)
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown part_id {part_id!r}")
    rgb = swatch_rgb(row["finish_style"])
    fill = "#{:02x}{:02x}{:02x}".format(*rgb)
    stroke = "#1f2937" if sum(rgb) > 300 else "#e5e7eb"
    w, h = row["width_cm"], row["height_cm"]
    scale = 110 / max(w, h)
    sw, sh = w * scale, max(h * scale, 6)
    x, y = (120 - sw) / 2, (120 - sh) / 2
    inner = ""
    cat = row["category"]
    if cat == "Drawer Base":
        inner = "".join(
            f'<line x1="{x}" y1="{y + sh * i / 3}" x2="{x + sw}" y2="{y + sh * i / 3}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<rect x="{x + sw / 2 - 8}" y="{y + sh * i / 3 + 5}" width="16" height="3" rx="1.5" fill="{stroke}"/>'
            for i in range(3)
        )
    elif cat not in ("Countertop", "Filler Panel"):
        doors = 2 if w >= 60 else 1
        for d in range(doors):
            dx = x + sw * d / doors
            inner += f'<rect x="{dx + 3}" y="{y + 3}" width="{sw / doors - 6}" height="{sh - 6}" fill="none" stroke="{stroke}" stroke-width="1.2"/>'
            hx = dx + sw / doors - 9 if doors == 1 or d == 0 else dx + 6
            inner += (
                f'<rect x="{hx}" y="{y + sh * 0.2}" width="3" height="{min(18, sh * 0.25)}" rx="1.5" fill="{stroke}"/>'
            )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" role="img" aria-label="{escape(row["part_name"])}">'
        f'<rect x="{x}" y="{y}" width="{sw}" height="{sh}" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        f"{inner}</svg>"
    )
    return Response(svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


# --------------------------------------------------------------------------- search
@router.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest) -> SearchResponse:
    """Hybrid search: vector similarity for style, SQL for hard constraints."""
    results, hits, timings = search_catalog(
        req.query,
        categories=req.categories,
        finish_style=req.finish_style,
        max_width_cm=req.max_width_cm,
        max_depth_cm=req.max_depth_cm,
        max_price_usd=req.max_price_usd,
        in_stock_only=req.in_stock_only,
        top_k=req.top_k,
    )
    return SearchResponse(
        query=req.query, vector_hits=hits, results=results, timings_ms={k: round(v, 2) for k, v in timings.items()}
    )


# --------------------------------------------------------------------------- spec / BOM
@router.post("/spec", response_model=SpecResponse, tags=["spec"])
def create_spec(
    constraints: Annotated[
        str,
        Form(description='JSON DesignConstraints, e.g. {"max_width_cm": 240, "budget_usd": 1500}'),
    ] = "{}",
    image: Annotated[UploadFile | None, File(description="Room wall/corner photo (JPEG/PNG/WebP)")] = None,
) -> SpecResponse:
    """Full pipeline from a room photo plus constraints to a verified BOM."""
    try:
        parsed = DesignConstraints.model_validate(json.loads(constraints or "{}"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"constraints is not valid JSON: {exc}") from exc
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, json.loads(exc.json(include_url=False))) from exc

    data = None
    if image is not None and image.filename:
        if image.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported image type {image.content_type!r}"
            )
        limit = int(get_settings().max_upload_mb * 1024 * 1024)
        data = image.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Image exceeds {get_settings().max_upload_mb} MB")
    try:
        return run_spec_pipeline(parsed, data)
    except ImageValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


@router.post("/bom", response_model=SpecResponse, tags=["spec"])
def create_bom(constraints: DesignConstraints) -> SpecResponse:
    """Constraints-only BOM (no photo): style comes from `finish_style` / `style_prompt`."""
    return run_spec_pipeline(constraints)
