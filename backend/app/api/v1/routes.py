"""FastAPI v1 routers: health, catalog, search, spec (image + constraints), BOM and FIT."""

from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Path, Query, Response, UploadFile, status
from pydantic import ValidationError

from app import __version__
from app.core.config import get_settings
from app.core.metrics import registry
from app.core.security import rate_limit, require_admin_key, require_api_key
from app.core.thumbnails import render_module_svg
from app.db import postgres, qdrant, spec_store
from app.schemas.spec import (
    CatalogItem,
    CatalogPatch,
    CatalogUpsertResponse,
    DesignConstraints,
    HealthResponse,
    Module,
    ReindexResponse,
    SearchRequest,
    SearchResponse,
    SpecResponse,
    SpecRunSummary,
)
from app.services import fit_composer
from app.services.cv_preprocessor import ImageValidationError
from app.services.embedding_engine import get_embedder
from app.services.genai_bom import _provider
from app.services.hybrid_retriever import CABINET_CATEGORIES, COUNTERTOP_CATEGORIES, index_catalog, search_catalog
from app.services.pipeline import run_spec_pipeline

log = logging.getLogger(__name__)

router = APIRouter()
compute = [Depends(require_api_key), Depends(rate_limit)]
admin = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin_key)])

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
    svg = render_module_svg(row)
    return Response(svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


# --------------------------------------------------------------------------- search
@router.post("/search", response_model=SearchResponse, tags=["search"], dependencies=compute)
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
@router.post("/spec", response_model=SpecResponse, tags=["spec"], dependencies=compute)
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
        return _record(run_spec_pipeline(parsed, data))
    except ImageValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


@router.post("/bom", response_model=SpecResponse, tags=["spec"], dependencies=compute)
def create_bom(constraints: DesignConstraints) -> SpecResponse:
    """Constraints-only BOM (no photo): style comes from `finish_style` / `style_prompt`."""
    return _record(run_spec_pipeline(constraints))


def _record(spec: SpecResponse) -> SpecResponse:
    payload = spec.model_dump(mode="json")
    registry.observe_spec(payload)
    if get_settings().save_specs:
        try:
            spec_store.save_spec_run(payload)
        except Exception:  # a storage hiccup must not lose the user's result
            log.exception("Could not save spec run %s", spec.request_id)
    return spec


# --------------------------------------------------------------------------- saved specs
@router.get("/specs", response_model=list[SpecRunSummary], tags=["spec"])
def list_specs(limit: Annotated[int, Query(ge=1, le=100)] = 20) -> list[dict]:
    """Most recent saved specifications."""
    return spec_store.list_spec_runs(limit)


@router.get("/specs/{request_id}", response_model=SpecResponse, tags=["spec"])
def get_spec(request_id: Annotated[str, Path(pattern=r"^[0-9a-f]{12,32}$")]) -> dict:
    """A saved specification by id: the target of shareable `/?spec=<id>` links."""
    spec = spec_store.get_spec_run(request_id)
    if spec is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No saved specification {request_id!r}")
    return spec


# --------------------------------------------------------------------------- fit (new)
@router.post("/fit", tags=["fit"], dependencies=compute)
def fit(
    part_id: Annotated[str, Form(description="Catalog part_id to insert into the photo")],
    image: Annotated[UploadFile, File(description="The original room photo, re-sent by the browser")],
) -> Response:
    """Composite a chosen catalog module into the caller's room photo.

    The photo is NOT read from a saved spec: /spec never stores the original
    image (see `save_specs` / spec_store notes), so the browser re-sends the
    same file it already has in memory. See app/services/fit_composer.py for
    the current limitations (schematic reference image, unverified provider
    response shape).
    """
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported image type {image.content_type!r}")
    limit = int(get_settings().max_upload_mb * 1024 * 1024)
    room_bytes = image.file.read(limit + 1)
    if len(room_bytes) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Image exceeds {get_settings().max_upload_mb} MB")

    row = postgres.get_modules_by_ids([part_id]).get(part_id)
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown part_id {part_id!r}")

    try:
        result_png = fit_composer.compose_fit(room_bytes, row)
    except fit_composer.FitUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except fit_composer.FitError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    return Response(result_png, media_type="image/png")


# --------------------------------------------------------------------------- admin
@admin.patch("/catalog/{part_id}", response_model=Module)
def patch_module(part_id: str, changes: CatalogPatch) -> dict:
    """Change price or stock. Takes effect on the next request; no re-index needed."""
    row = postgres.update_module(part_id, changes.model_dump(exclude_none=True))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown part_id {part_id!r}")
    return row


@admin.put("/catalog", response_model=CatalogUpsertResponse)
def upsert_catalog(items: Annotated[list[CatalogItem], Body(min_length=1, max_length=5000)]) -> CatalogUpsertResponse:
    """Insert or update catalog rows and re-embed them so search sees the new text."""
    rows = []
    for item in items:
        row = item.model_dump()
        row["image_url"] = row["image_url"] or f"/api/v1/catalog/{item.part_id}/thumbnail.svg"
        rows.append(row)
    postgres.upsert_modules(rows)
    stored = postgres.get_modules_by_ids([r["part_id"] for r in rows])
    reindexed = index_catalog(list(stored.values()))
    known = set(CABINET_CATEGORIES + COUNTERTOP_CATEGORIES)
    return CatalogUpsertResponse(
        upserted=len(rows),
        reindexed=reindexed,
        catalog_rows=postgres.count_modules(),
        unknown_categories=sorted({r["category"] for r in rows} - known),
    )


@admin.post("/reindex", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    """Rebuild every vector from the SQL catalog (after bulk SQL edits made outside the API)."""
    indexed = index_catalog()
    return ReindexResponse(indexed=indexed, collection=qdrant.collection_name(), embedding_backend=get_embedder().name)


router.include_router(admin)
