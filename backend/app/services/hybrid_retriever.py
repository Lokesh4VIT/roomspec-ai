"""Hybrid retrieval: Qdrant style similarity + SQL hard dimensional constraints.

Vector search answers "what looks right"; SQL answers "what physically fits, is
priced and is in stock". A module is only ever returned if it passes both.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from app.core.config import get_settings
from app.db import postgres, qdrant
from app.schemas.spec import DesignConstraints
from app.services.embedding_engine import Embedder, get_embedder, lexical_document, lexical_vector, module_document

BASE_RUN_CATEGORIES = ("Base Cabinet", "Drawer Base", "Sink Base")
WALL_CATEGORIES = ("Wall Cabinet",)
TALL_CATEGORIES = ("Tall Pantry",)
FILLER_CATEGORIES = ("Filler Panel",)
COUNTERTOP_CATEGORIES = ("Countertop",)
CABINET_CATEGORIES = BASE_RUN_CATEGORIES + WALL_CATEGORIES + TALL_CATEGORIES + FILLER_CATEGORIES

RRF_K = 60  # reciprocal-rank-fusion damping constant (Cormack et al., 2009)


@dataclass
class StyleQuery:
    """Text intent plus optional photo, embedded separately and fused by rank."""

    text: str
    text_vector: np.ndarray
    lexical: tuple[list[int], list[float]] = ([], [])
    image_vector: np.ndarray | None = None
    text_weight: float = 1.0  # split evenly between dense and lexical text rankings
    image_weight: float = 0.0


@dataclass
class RetrievalResult:
    query_text: str
    finish_style: str | None
    countertop_finish: str | None
    candidates_considered: int
    modules: dict[str, list[dict]] = field(default_factory=dict)  # group -> compliant rows
    scores: dict[str, float] = field(default_factory=dict)  # part_id -> fused style score in [0, 1]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def candidates_compliant(self) -> int:
        return sum(len(v) for v in self.modules.values())


def build_query_text(c: DesignConstraints, suggested_finishes: Sequence[str] = ()) -> str:
    parts = []
    if c.style_prompt:
        parts.append(c.style_prompt)
    if c.finish_style:
        parts.append(f"{c.finish_style} finish")
    elif suggested_finishes:
        # Only the closest colour match: naming runners-up makes lexical search match them equally.
        parts.append(f"{suggested_finishes[0]} finish")
    parts.append("modular kitchen cabinets")
    return ". ".join(parts)


def build_style_query(
    embedder: Embedder,
    c: DesignConstraints,
    suggested_finishes: Sequence[str] = (),
    image_rgb: np.ndarray | None = None,
) -> StyleQuery:
    text = build_query_text(c, suggested_finishes)
    q = StyleQuery(text=text, text_vector=embedder.embed_texts([text])[0], lexical=lexical_vector(text))
    if image_rgb is not None and embedder.supports_images:
        q.image_vector = embedder.embed_image(image_rgb)
        # Explicit user intent outweighs the photo; colour-derived text is only a weak hint
        # because CLIP's image tower already sees the colours directly.
        user_intent = bool(c.style_prompt or c.finish_style)
        q.text_weight, q.image_weight = (0.6, 0.4) if user_intent else (0.15, 0.85)
    return q


def fused_search(q: StyleQuery, limit: int, categories: Sequence[str] | None = None) -> dict[str, float]:
    """Query Qdrant per signal (dense text, sparse lexical, image) and fuse with weighted RRF.

    Scores are not comparable across signals (CLIP text-text cosine sits near 0.8,
    image-text near 0.25, lexical scores are unbounded), so ranks are fused instead.
    """
    lists = [
        (q.text_weight / 2, qdrant.search(q.text_vector, limit, categories=categories)),
        (q.text_weight / 2, qdrant.search(q.lexical, limit, categories=categories)),
    ]
    if q.image_vector is not None:
        lists.append((q.image_weight, qdrant.search(q.image_vector, limit, categories=categories)))
    best = sum(w for w, _ in lists) / (RRF_K + 1)
    fused: dict[str, float] = defaultdict(float)
    for weight, hits in lists:
        for rank, hit in enumerate(hits, start=1):
            fused[hit.part_id] += weight / (RRF_K + rank)
    return {pid: round(score / best, 4) for pid, score in fused.items()}


def index_catalog(rows: list[dict] | None = None, embedder: Embedder | None = None) -> int:
    """(Re)build the Qdrant collection from the SQL catalog."""
    rows = rows if rows is not None else postgres.list_all_modules()
    embedder = embedder or get_embedder()
    qdrant.ensure_collection(embedder.dim)
    batch = 32
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        vectors = embedder.embed_texts([module_document(r) for r in chunk])
        sparse = [lexical_vector(lexical_document(r)) for r in chunk]
        payloads = [
            {"category": r["category"], "finish_style": r["finish_style"], "part_name": r["part_name"]} for r in chunk
        ]
        qdrant.upsert_vectors([r["part_id"] for r in chunk], vectors, sparse, payloads)
    return len(rows)


def _resolve_finish(rows: list[dict], scores: dict[str, float]) -> str | None:
    """Pick one coherent finish: the one whose best modules score highest on average."""
    by_finish: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_finish[r["finish_style"]].append(scores.get(r["part_id"], 0.0))
    if not by_finish:
        return None
    return max(by_finish, key=lambda f: float(np.mean(sorted(by_finish[f], reverse=True)[:5])))


def retrieve_for_layout(c: DesignConstraints, query: StyleQuery, top_k: int | None = None) -> RetrievalResult:
    top_k = top_k or get_settings().vector_top_k
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    groups = {
        "base": BASE_RUN_CATEGORIES + FILLER_CATEGORIES,
        "wall": WALL_CATEGORIES,
        "tall": TALL_CATEGORIES,
        "countertop": COUNTERTOP_CATEGORIES,
    }
    group_scores = {g: fused_search(query, top_k, categories=cats) for g, cats in groups.items()}
    timings["vector_search"] = (time.perf_counter() - t0) * 1000

    scores = {pid: sc for gs in group_scores.values() for pid, sc in gs.items()}
    considered = len(scores)

    t0 = time.perf_counter()
    common = dict(max_depth_cm=c.max_depth_cm, in_stock_only=True)
    # Hard SQL filter on the vector candidates. Finish is resolved on cabinets first so the
    # whole run shares one finish; countertops resolve their own material.
    cab_ids = [pid for g in ("base", "wall", "tall") for pid in group_scores[g]]
    cab_rows = postgres.filter_modules(
        part_ids=cab_ids,
        max_width_cm=c.max_width_cm,
        max_price_usd=c.budget_usd,
        finish_style=c.finish_style,
        **common,
    )
    if c.front_clearance_cm is not None:
        cab_rows = [r for r in cab_rows if r["door_clearance_cm"] <= c.front_clearance_cm]

    finish = c.finish_style or _resolve_finish([r for r in cab_rows if r["category"] in BASE_RUN_CATEGORIES], scores)
    if finish:
        # Canonical casing from the catalog; widen to every compliant width in that finish so
        # the layout solver is not limited by which widths happened to land in the top-k.
        finish = next((r["finish_style"] for r in cab_rows if r["finish_style"].lower() == finish.lower()), finish)
        cab_rows = postgres.filter_modules(
            categories=CABINET_CATEGORIES,
            finish_style=finish,
            max_width_cm=c.max_width_cm,
            max_price_usd=c.budget_usd,
            **common,
        )
        if c.front_clearance_cm is not None:
            cab_rows = [r for r in cab_rows if r["door_clearance_cm"] <= c.front_clearance_cm]

    ct_rows = (
        postgres.filter_modules(part_ids=list(group_scores["countertop"]), max_price_usd=c.budget_usd, **common)
        if c.include_countertop
        else []
    )
    ct_finish = _resolve_finish(ct_rows, scores)
    if ct_finish:
        ct_rows = postgres.filter_modules(
            categories=COUNTERTOP_CATEGORIES, finish_style=ct_finish, max_price_usd=c.budget_usd, **common
        )
    timings["sql_filter"] = (time.perf_counter() - t0) * 1000

    modules = {
        "base": [r for r in cab_rows if r["category"] in BASE_RUN_CATEGORIES],
        "filler": [r for r in cab_rows if r["category"] in FILLER_CATEGORIES],
        "wall": [r for r in cab_rows if r["category"] in WALL_CATEGORIES] if c.include_wall_cabinets else [],
        "tall": [r for r in cab_rows if r["category"] in TALL_CATEGORIES] if c.include_tall_units else [],
        "countertop": ct_rows,
    }
    return RetrievalResult(
        query_text=query.text,
        finish_style=finish,
        countertop_finish=ct_finish,
        candidates_considered=considered,
        modules=modules,
        scores=scores,
        timings_ms=timings,
    )


def search_catalog(
    query: str,
    *,
    categories: Sequence[str] | None = None,
    finish_style: str | None = None,
    max_width_cm: float | None = None,
    max_depth_cm: float | None = None,
    max_price_usd: float | None = None,
    in_stock_only: bool = True,
    top_k: int = 10,
) -> tuple[list[dict], int, dict[str, float]]:
    """Free-text hybrid search: dense + lexical top-k fused by rank, then hard SQL constraints."""
    emb = get_embedder()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    style = StyleQuery(text=query, text_vector=emb.embed_texts([query])[0], lexical=lexical_vector(query))
    timings["embed"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    score = fused_search(style, get_settings().vector_top_k, categories=categories)
    timings["vector_search"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    rows = postgres.filter_modules(
        part_ids=list(score),
        finish_style=finish_style,
        max_width_cm=max_width_cm,
        max_depth_cm=max_depth_cm,
        max_price_usd=max_price_usd,
        in_stock_only=in_stock_only,
    )
    timings["sql_filter"] = (time.perf_counter() - t0) * 1000

    rows.sort(key=lambda r: (-score[r["part_id"]], r["part_id"]))
    return [{**r, "style_score": score[r["part_id"]]} for r in rows[:top_k]], len(score), timings
