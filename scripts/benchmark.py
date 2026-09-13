"""Reproducible benchmarks for RoomSpec AI.

    python scripts/benchmark.py --backend hash      # fast, no model download
    python scripts/benchmark.py --backend clip      # CLIP ViT-B/32 on CPU

Writes benchmarks/RESULTS-<backend>.md and benchmarks/results-<backend>.json.

Metrics
-------
* Dimension compliance: 100 random wall scenarios (width, budget, depth limit, door
  clearance). A recommendation is compliant when every recommended part fits the wall,
  the depth limit, the swing clearance and is in stock, and the set fits the width and
  budget together. Baseline = top-5 keyword matches from the same catalog.
* Style Precision@5: labeled finish queries (paraphrased text + sample photos). A result
  is relevant when its finish matches the label. Baseline = keyword overlap ranking.
* Latency: P50/P95 over repeated runs on this machine (CPU, batch size 1).
* BOM hallucination rate: BOM lines whose part id, dimensions or price differ from the
  SQL row, plus narratives that fail grounding verification.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINISHES = [
    "Matte Walnut",
    "Natural Oak",
    "Gloss White",
    "Cream Shaker",
    "Charcoal Grey",
    "Matte Black",
    "Sage Green",
    "Navy Blue",
]
# Paraphrases deliberately avoid most of the catalog's own vocabulary.
TEXT_QUERIES = {
    "Matte Walnut": ["espresso toned timber kitchen", "rich brown hardwood mid century units", "walnut"],
    "Natural Oak": ["blonde nordic timber kitchen", "pale honey wooden cupboards", "oak"],
    "Gloss White": ["sleek shiny white handleless kitchen", "clean bright lacquer units", "white gloss"],
    "Cream Shaker": ["country cottage ivory panel doors", "traditional farmhouse cupboards", "shaker"],
    "Charcoal Grey": ["graphite urban loft kitchen", "slate coloured contemporary units", "grey"],
    "Matte Black": ["moody jet black kitchen", "dark noir cupboards", "black"],
    "Sage Green": ["soft olive botanical kitchen", "earthy muted green cupboards", "sage"],
    "Navy Blue": ["deep nautical blue kitchen", "bold indigo cupboards", "navy"],
}
IMAGE_LABELS = {
    "warm_walnut_kitchen": "Matte Walnut",
    "bright_minimal_white": "Gloss White",
    "industrial_grey_corner": "Charcoal Grey",
    "sage_cottage_wall": "Sage Green",
    "dim_navy_galley": "Navy Blue",
}


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def keyword_rank(rows: list[dict], query: str, k: int = 5, categories: tuple[str, ...] | None = None) -> list[dict]:
    tokens = set(re.findall(r"[a-z]+", query.lower())) - {"kitchen", "the", "a", "and"}

    def score(r: dict) -> int:
        text = f"{r['part_name']} {r['finish_style']} {r['material']} {r['description']}".lower()
        return sum(t in text for t in tokens)

    pool = [r for r in rows if not categories or r["category"] in categories]
    return sorted(pool, key=lambda r: (-score(r), r["part_id"]))[:k]


def item_ok(r: dict, s: dict, installed_width: float | None = None) -> bool:
    """Per-part hard constraints. Countertops are cut on site, so their installed width counts."""
    return (
        (installed_width if installed_width is not None else r["width_cm"]) <= s["max_width_cm"]
        and (s.get("max_depth_cm") is None or r["depth_cm"] <= s["max_depth_cm"])
        and (s.get("front_clearance_cm") is None or r["door_clearance_cm"] <= s["front_clearance_cm"])
        and r["in_stock"]
    )


def make_scenarios(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = {
            "finish": rng.choice(FINISHES),
            "max_width_cm": float(rng.randrange(60, 481, 5)),
            "budget_usd": float(rng.randrange(400, 6001, 50)),
        }
        if rng.random() < 0.3:
            s["max_depth_cm"] = float(rng.choice([35, 40, 58, 60, 65]))
        if rng.random() < 0.3:
            s["front_clearance_cm"] = float(rng.choice([35, 40, 45, 50, 60]))
        out.append(s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hash", "clip"], default="hash")
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--latency-runs", type=int, default=50)
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="roomspec-bench-"))
    os.environ.update(
        {
            "DATABASE_URL": f"sqlite:///{tmp / 'bench.db'}",
            "QDRANT_URL": "",
            "EMBEDDING_BACKEND": args.backend,
            "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "none"),
            "LOG_LEVEL": "WARNING",
        }
    )
    sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]

    from fastapi.testclient import TestClient

    from app.db import postgres
    from app.db.bootstrap import bootstrap
    from app.main import app
    from app.schemas.spec import DesignConstraints
    from app.services import cv_preprocessor
    from app.services.embedding_engine import get_embedder
    from app.services.hybrid_retriever import build_style_query, retrieve_for_layout, search_catalog
    from app.services.pipeline import run_spec_pipeline

    t0 = time.perf_counter()
    bootstrap()
    embedder = get_embedder()
    setup_s = time.perf_counter() - t0
    catalog = postgres.list_all_modules()
    by_id = {r["part_id"]: r for r in catalog}
    results: dict = {
        "backend": embedder.name,
        "machine": f"{platform.system()} {platform.machine()} · Python {platform.python_version()}",
        "catalog_rows": len(catalog),
        "setup_seconds": round(setup_s, 2),
    }

    # ---------------------------------------------------------------- dimension compliance
    scenarios = make_scenarios(args.scenarios)
    base_ok = base_item_total = base_item_ok = 0
    eng_ok = eng_items = eng_item_ok = infeasible = 0
    hallucinated_lines = narrative_failures = 0
    for s in scenarios:
        query = f"{s['finish']} base cabinet"
        picks = keyword_rank(catalog, query)
        items = [item_ok(r, s) for r in picks]
        base_item_total += len(items)
        base_item_ok += sum(items)
        floor = sum(r["width_cm"] for r in picks if r["category"] != "Wall Cabinet")
        base_ok += all(items) and floor <= s["max_width_cm"] and sum(r["price_usd"] for r in picks) <= s["budget_usd"]

        c = DesignConstraints(
            max_width_cm=s["max_width_cm"],
            budget_usd=s["budget_usd"],
            style_prompt=f"{s['finish']} kitchen cabinets",
            max_depth_cm=s.get("max_depth_cm"),
            front_clearance_cm=s.get("front_clearance_cm"),
        )
        res = run_spec_pipeline(c)
        if res.status == "infeasible":
            infeasible += 1
        parts_ok = [item_ok(by_id[p.part_id], s, p.width_cm) for p in res.layout]
        eng_items += len(parts_ok)
        eng_item_ok += sum(parts_ok)
        eng_ok += all(ch.passed for ch in res.compliance if ch.kind == "constraint") and all(parts_ok)
        for ln in res.bom:
            row = by_id.get(ln.part_id)
            if not row or (ln.width_cm, ln.height_cm, ln.depth_cm, ln.unit_price_usd) != (
                row["width_cm"],
                row["height_cm"],
                row["depth_cm"],
                row["price_usd"],
            ):
                hallucinated_lines += 1
        narrative_failures += not res.hallucination_check.passed

    results["dimension_compliance"] = {
        "scenarios": len(scenarios),
        "baseline_keyword_top5_set_compliant": base_ok / len(scenarios),
        "baseline_keyword_item_compliant": base_item_ok / max(base_item_total, 1),
        "engine_plan_compliant": eng_ok / len(scenarios),
        "engine_item_compliant": eng_item_ok / max(eng_items, 1),
        "engine_infeasible_scenarios": infeasible,
    }

    # ---------------------------------------------------------------- style precision@5
    def precision(parts: list[dict], label: str) -> float:
        return sum(p["finish_style"] == label for p in parts[:5]) / 5

    cab_categories = ("Base Cabinet", "Drawer Base", "Sink Base", "Wall Cabinet", "Tall Pantry")
    text_eng, text_base = [], []
    for label, queries in TEXT_QUERIES.items():
        for q in queries:
            found, _, _ = search_catalog(q, categories=cab_categories, in_stock_only=False, top_k=5)
            text_eng.append(precision(found, label))
            text_base.append(precision(keyword_rank(catalog, q, categories=cab_categories), label))

    img_eng = []
    for name, label in IMAGE_LABELS.items():
        data = (ROOT / "data" / "samples" / f"{name}.jpg").read_bytes()
        res = run_spec_pipeline(DesignConstraints(max_width_cm=240, budget_usd=10_000), data)
        img_eng.append(float(res.finish_style_resolved == label))
    results["style_precision_at_5"] = {
        "text_queries": len(text_eng),
        "baseline_keyword": sum(text_base) / len(text_base),
        "engine_text": sum(text_eng) / len(text_eng),
        "image_samples": len(img_eng),
        "engine_image_finish_accuracy": sum(img_eng) / len(img_eng),
    }

    # ---------------------------------------------------------------- latency
    sample = (ROOT / "data" / "samples" / "warm_walnut_kitchen.jpg").read_bytes()
    pre = cv_preprocessor.preprocess(sample)
    lat: dict[str, list[float]] = {"cv": [], "embed_image": [], "vector_sql": [], "e2e_api": []}
    c = DesignConstraints(max_width_cm=240, budget_usd=2500)
    query = build_style_query(embedder, c, pre.analysis["suggested_finishes"], pre.embedding_rgb)
    client = TestClient(app)
    payload = {"constraints": json.dumps(c.model_dump(exclude_none=True))}
    for _ in range(3):  # warm-up
        client.post("/api/v1/spec", data=payload, files={"image": ("r.jpg", sample, "image/jpeg")})
    for _ in range(args.latency_runs):
        t = time.perf_counter()
        cv_preprocessor.preprocess(sample)
        lat["cv"].append((time.perf_counter() - t) * 1000)
        if embedder.supports_images:
            t = time.perf_counter()
            embedder.embed_image(pre.embedding_rgb)
            lat["embed_image"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        retrieve_for_layout(c, query)
        lat["vector_sql"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        r = client.post("/api/v1/spec", data=payload, files={"image": ("r.jpg", sample, "image/jpeg")})
        lat["e2e_api"].append((time.perf_counter() - t) * 1000)
        assert r.status_code == 200
    results["latency_ms"] = {
        k: {"p50": round(pct(v, 50), 2), "p95": round(pct(v, 95), 2), "n": len(v)} for k, v in lat.items() if v
    }
    results["latency_notes"] = (
        f"LLM provider: {os.environ['LLM_PROVIDER']}; e2e excludes network LLM latency when 'none'. "
        "Vector store is embedded in-memory Qdrant and SQLite, so no network round-trips are included."
    )

    # ---------------------------------------------------------------- hallucination
    results["bom_hallucination"] = {
        "bom_lines_mismatching_sql": hallucinated_lines,
        "narratives_failing_grounding_check": narrative_failures,
        "scenarios": len(scenarios),
    }

    out_dir = ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"results-{args.backend}.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / f"RESULTS-{args.backend}.md").write_text(render_markdown(results))
    print(render_markdown(results))


def render_markdown(r: dict) -> str:
    d, s, lat, h = r["dimension_compliance"], r["style_precision_at_5"], r["latency_ms"], r["bom_hallucination"]
    p = lambda v: f"{v * 100:.1f}%"  # noqa: E731
    rows = [
        f"# Benchmark results · `{r['backend']}` embeddings",
        "",
        f"Machine: {r['machine']} · catalog rows: {r['catalog_rows']} · generated by `scripts/benchmark.py`",
        "",
        "| Metric | Keyword baseline | RoomSpec AI | Method |",
        "|---|---|---|---|",
        f"| Dimension compliance (whole recommendation) | {p(d['baseline_keyword_top5_set_compliant'])} "
        f"| {p(d['engine_plan_compliant'])} | {d['scenarios']} random wall/budget/depth/clearance scenarios |",
        f"| Dimension compliance (per part) | {p(d['baseline_keyword_item_compliant'])} "
        f"| {p(d['engine_item_compliant'])} | Same scenarios, each recommended part checked |",
        f"| Style Precision@5 (text) | {s['baseline_keyword']:.2f} | {s['engine_text']:.2f} "
        f"| {s['text_queries']} labeled finish queries, mostly paraphrased |",
        f"| Finish accuracy from photo | n/a | {p(s['engine_image_finish_accuracy'])} "
        f"| {s['image_samples']} synthetic sample rooms |",
    ]
    if "embed_image" in lat:
        rows.append(
            f"| CLIP image embedding | n/a | {lat['embed_image']['p95']} ms P95 "
            f"| ViT-B/32 forward pass, CPU, batch 1 (P50 {lat['embed_image']['p50']} ms) |"
        )
    rows += [
        f"| OpenCV preprocessing | n/a | {lat['cv']['p95']} ms P95 | 960×720 photo (P50 {lat['cv']['p50']} ms) |",
        f"| Vector + SQL retrieval | n/a | {lat['vector_sql']['p95']} ms P95 "
        f"| Qdrant (embedded) + SQL filters (P50 {lat['vector_sql']['p50']} ms) |",
        f"| End-to-end `/api/v1/spec` | n/a | {lat['e2e_api']['p95']} ms P95 "
        f"| Photo upload → BOM, in-process client (P50 {lat['e2e_api']['p50']} ms) |",
        f"| BOM lines not matching SQL | n/a | {h['bom_lines_mismatching_sql']} "
        f"| Every line re-checked against the catalog row |",
        f"| Narratives failing grounding check | n/a | {h['narratives_failing_grounding_check']} / {h['scenarios']} "
        f"| Part ids, cm and $ values verified |",
        "",
        f"Engine returned no plan (infeasible) in {d['engine_infeasible_scenarios']} of {d['scenarios']} scenarios; "
        "those count as compliant because nothing non-fitting was recommended.",
        "",
        f"_{r['latency_notes']}_",
        "",
    ]
    return "\n".join(rows)


if __name__ == "__main__":
    main()
