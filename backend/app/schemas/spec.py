"""Pydantic request/response models for the RoomSpec API."""

from __future__ import annotations

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


class SpecResponse(BaseModel):
    request_id: str
    constraints: DesignConstraints
    image_analysis: ImageAnalysis | None
    query_text: str
    finish_style_resolved: str | None
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
