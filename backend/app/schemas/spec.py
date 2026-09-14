"""Pydantic request/response models for the RoomSpec API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DesignConstraints(BaseModel):
    """What the user needs to fit on the wall."""

    model_config = ConfigDict(extra="forbid")

    max_width_cm: float = Field(240, gt=20, le=1200, description="Usable wall run width in cm")
    budget_usd: float = Field(1500, gt=0, le=250_000, description="Total budget for the BOM")
    finish_style: str | None = Field(None, max_length=64, description="e.g. 'Matte Walnut'")
    style_prompt: str | None = Field(
        None, max_length=300, description="Free-text style intent, e.g. 'warm scandinavian wood'"
    )
    max_depth_cm: float | None = Field(None, gt=0, le=120, description="Depth limit, e.g. a narrow galley")
    ceiling_height_cm: float | None = Field(None, gt=150, le=500)
    front_clearance_cm: float | None = Field(
        None, ge=0, le=300, description="Free floor space in front of the run (door / drawer swing)"
    )
    include_wall_cabinets: bool = True
    include_countertop: bool = True
    include_tall_units: bool = False
    layout_priority: Literal["fill_width", "complete_kitchen"] = Field(
        "fill_width",
        description=(
            "fill_width: use as much of the wall as possible, then add upper cabinets with the remaining budget. "
            "complete_kitchen: prefer a narrower run whose upper cabinets cover it when the budget cannot do both."
        ),
    )

    @field_validator("finish_style", "style_prompt")
    @classmethod
    def _blank_to_none(cls, v: str | None) -> str | None:
        return v.strip() or None if v is not None else None


class Module(BaseModel):
    part_id: str
    part_name: str
    category: str
    finish_style: str
    material: str
    price_usd: float
    width_cm: float
    height_cm: float
    depth_cm: float
    door_clearance_cm: float
    in_stock: bool
    description: str
    image_url: str


class ScoredModule(Module):
    style_score: float = Field(description="Cosine similarity from the vector search")


class ImageAnalysis(BaseModel):
    width_px: int
    height_px: int
    mean_luminance_before: float
    mean_luminance_after: float
    contrast_before: float
    contrast_after: float
    floor_line_y_ratio: float | None = Field(description="Wall/floor boundary as fraction of height")
    corner_x_ratio: float | None = Field(description="Dominant vertical wall corner as fraction of width")
    perspective_tilt_deg: float
    wall_roi_ratio: float = Field(description="Fraction of the image identified as usable wall")
    dominant_colors: list[str]
    suggested_finishes: list[str]
    warnings: list[str] = []


class BOMLine(BaseModel):
    line: int
    part_id: str
    part_name: str
    category: str
    finish_style: str
    quantity: int
    unit_price_usd: float
    line_total_usd: float
    width_cm: float
    height_cm: float
    depth_cm: float
    position_cm: float | None = Field(None, description="Left edge offset along the wall run")
    note: str | None = None


class LayoutPlacement(BaseModel):
    """One physical unit on the wall elevation (all values from the catalog row)."""

    part_id: str
    zone: Literal["base", "tall", "wall", "filler", "countertop"]
    x_cm: float = Field(description="Left edge along the wall run")
    bottom_cm: float = Field(description="Height of the unit's bottom edge above the floor")
    width_cm: float = Field(description="Installed width (cut length for countertops)")
    height_cm: float
    depth_cm: float
    swatch_hex: str


class ComplianceCheck(BaseModel):
    name: str
    passed: bool
    detail: str
    kind: Literal["constraint", "completeness"] = Field(
        "constraint",
        description="constraint = a hard physical/budget rule; completeness = a requested element is missing",
    )


class HallucinationReport(BaseModel):
    llm_used: bool
    provider: str
    passed: bool
    unknown_part_ids: list[str] = []
    unverified_numbers: list[str] = []
    fallback_reason: str | None = None


class AlternativePlan(BaseModel):
    """Best layout in another finish, solved under the same constraints for comparison."""

    finish_style: str
    swatch_hex: str
    style_score: float = Field(description="Fused style score of the finish, 0-1 (higher = closer to the brief)")
    status: Literal["ok", "partial"]
    total_usd: float
    run_width_cm: float
    wall_gap_cm: float
    units: int
    part_ids: list[str]


class SpecResponse(BaseModel):
    request_id: str
    constraints: DesignConstraints
    image_analysis: ImageAnalysis | None
    query_text: str
    finish_style_resolved: str | None
    finish_style_score: float | None = None
    candidates_considered: int
    candidates_compliant: int
    bom: list[BOMLine]
    layout: list[LayoutPlacement]
    total_usd: float
    run_width_cm: float
    wall_gap_cm: float
    compliance: list[ComplianceCheck]
    summary: str
    installation_notes: list[str]
    hallucination_check: HallucinationReport
    timings_ms: dict[str, float]
    status: Literal["ok", "partial", "infeasible"]
    alternatives: list[AlternativePlan] = []


class SpecRunSummary(BaseModel):
    request_id: str
    created_at: datetime | None
    status: str
    finish_style: str | None
    total_usd: float
    max_width_cm: float
    budget_usd: float
    had_image: bool


class CatalogPatch(BaseModel):
    """Fields an operator changes day to day; both are read from SQL per request (no reindex needed)."""

    model_config = ConfigDict(extra="forbid")

    price_usd: float | None = Field(None, ge=0, le=1_000_000)
    in_stock: bool | None = None


class CatalogItem(BaseModel):
    """A catalog row for bulk upsert. Text fields feed the embeddings, so upserts re-index."""

    model_config = ConfigDict(extra="forbid")

    part_id: str = Field(..., min_length=3, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    part_name: str = Field(..., min_length=1, max_length=255)
    category: str = Field(..., min_length=1, max_length=64)
    finish_style: str = Field(..., min_length=1, max_length=64)
    material: str = Field("", max_length=128)
    price_usd: float = Field(..., ge=0, le=1_000_000)
    width_cm: float = Field(..., gt=0, le=1000)
    height_cm: float = Field(..., gt=0, le=500)
    depth_cm: float = Field(..., gt=0, le=200)
    door_clearance_cm: float = Field(0, ge=0, le=300)
    in_stock: bool = True
    description: str = Field("", max_length=2000)
    image_url: str | None = Field(None, max_length=2000, description="Defaults to the generated thumbnail")


class CatalogUpsertResponse(BaseModel):
    upserted: int
    reindexed: int
    catalog_rows: int
    unknown_categories: list[str] = Field(description="Categories the layout solver does not use (still searchable)")


class ReindexResponse(BaseModel):
    indexed: int
    collection: str
    embedding_backend: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=2, max_length=300)
    categories: list[str] | None = None
    finish_style: str | None = None
    max_width_cm: float | None = Field(None, gt=0)
    max_depth_cm: float | None = Field(None, gt=0)
    max_price_usd: float | None = Field(None, gt=0)
    in_stock_only: bool = True
    top_k: int = Field(10, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    vector_hits: int
    results: list[ScoredModule]
    timings_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: str
    catalog_rows: int
    vector_store: str
    vector_points: int
    embedding_backend: str
    llm_provider: str
