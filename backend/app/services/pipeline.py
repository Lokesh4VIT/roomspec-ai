"""End-to-end spec pipeline: CV -> embedding -> Qdrant -> SQL -> layout -> grounded BOM."""

from __future__ import annotations

import logging
import time
import uuid

from app.core.finishes import swatch_hex
from app.db import postgres
from app.schemas.spec import DesignConstraints, ImageAnalysis, LayoutPlacement, SpecResponse
from app.services import cv_preprocessor
from app.services.embedding_engine import get_embedder
from app.services.genai_bom import build_bom_lines, generate_narrative
from app.services.hybrid_retriever import build_style_query, retrieve_for_layout
from app.services.layout_solver import (
    BACKSPLASH_GAP_CM,
    PLINTH_CM,
    LayoutPlan,
    compliance_checks,
    solve_layout,
)

log = logging.getLogger(__name__)


def elevation(plan: LayoutPlan) -> list[LayoutPlacement]:
    """Front-elevation coordinates for every unit, derived from catalog heights."""
    base_top = PLINTH_CM + max((p.row["height_cm"] for p in plan.placements if p.zone == "base"), default=72.0)
    worktop = max((p.row["height_cm"] for p in plan.placements if p.zone == "countertop"), default=0.0)
    bottoms = {
        "base": PLINTH_CM,
        "tall": PLINTH_CM,
        "filler": PLINTH_CM,
        "countertop": base_top,
        "wall": base_top + worktop + BACKSPLASH_GAP_CM,
    }
    return [
        LayoutPlacement(
            part_id=p.row["part_id"],
            zone=p.zone,
            x_cm=p.position_cm,
            bottom_cm=bottoms[p.zone],
            width_cm=p.cut_length_cm if p.cut_length_cm is not None else p.row["width_cm"],
            height_cm=p.row["height_cm"],
            depth_cm=p.row["depth_cm"],
            swatch_hex=swatch_hex(p.row["finish_style"]),
        )
        for p in plan.placements
    ]


def run_spec_pipeline(constraints: DesignConstraints, image_bytes: bytes | None = None) -> SpecResponse:
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    analysis = None
    image_rgb = None
    if image_bytes:
        t0 = time.perf_counter()
        pre = cv_preprocessor.preprocess(image_bytes)
        image_rgb = pre.embedding_rgb
        analysis = ImageAnalysis(**pre.analysis)
        timings["cv_preprocess"] = (time.perf_counter() - t0) * 1000

    embedder = get_embedder()
    t0 = time.perf_counter()
    query = build_style_query(embedder, constraints, analysis.suggested_finishes if analysis else (), image_rgb)
    timings["embedding"] = (time.perf_counter() - t0) * 1000

    retrieval = retrieve_for_layout(constraints, query)
    timings.update(retrieval.timings_ms)

    t0 = time.perf_counter()
    plan = solve_layout(constraints, retrieval.modules)
    timings["layout_solver"] = (time.perf_counter() - t0) * 1000

    # Re-read every chosen part from SQL for verification, independent of the retrieval path.
    verified = postgres.get_modules_by_ids([p.row["part_id"] for p in plan.placements])
    checks = compliance_checks(constraints, plan, verified)
    lines = build_bom_lines(plan)

    t0 = time.perf_counter()
    narrative = generate_narrative(constraints, plan, lines)
    timings["bom_generation"] = (time.perf_counter() - t0) * 1000
    timings["total"] = (time.perf_counter() - t_start) * 1000

    status = plan.status
    if lines and not all(ch.passed for ch in checks if ch.kind == "completeness"):
        status = "partial"
    if lines and not all(ch.passed for ch in checks if ch.kind == "constraint"):
        # Should be unreachable: the solver only uses SQL-compliant rows. Never present it as a fit.
        log.error("Layout failed hard constraint checks: %s", [c.name for c in checks if not c.passed])
        status = "infeasible"

    return SpecResponse(
        request_id=uuid.uuid4().hex[:12],
        constraints=constraints,
        image_analysis=analysis,
        query_text=query.text,
        finish_style_resolved=retrieval.finish_style,
        candidates_considered=retrieval.candidates_considered,
        candidates_compliant=retrieval.candidates_compliant,
        bom=lines,
        layout=elevation(plan),
        total_usd=plan.total_usd,
        run_width_cm=plan.run_width_cm,
        wall_gap_cm=plan.wall_gap_cm,
        compliance=checks,
        summary=narrative.summary,
        installation_notes=narrative.installation_notes,
        hallucination_check=narrative.report,
        timings_ms={k: round(v, 2) for k, v in timings.items()},
        status=status,
    )
