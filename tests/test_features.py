"""Alternatives, layout priority, saved specs, API keys, rate limiting, admin API and metrics."""

import pytest
from qdrant_client import models
from sqlalchemy import delete
from test_layout_solver import modules_for

from app.core.config import get_settings
from app.core.security import SlidingWindowLimiter, limiter
from app.db import postgres, qdrant
from app.schemas.spec import DesignConstraints
from app.services.layout_solver import solve_layout

BOM = "/api/v1/bom"
BASIC = {"max_width_cm": 240, "budget_usd": 2500, "style_prompt": "warm wood"}


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kv):
        for k, v in kv.items():
            monkeypatch.setattr(s, k, v)

    return set_


@pytest.fixture(autouse=True)
def _reset_limiter():
    limiter.reset()
    yield
    limiter.reset()


# --------------------------------------------------------------------------- layout priority
def test_complete_kitchen_trades_width_for_upper_cabinets():
    walnut = modules_for("Matte Walnut", countertop="Butcher Block Oak")
    fill = solve_layout(DesignConstraints(max_width_cm=300, budget_usd=1600), walnut)
    full = solve_layout(
        DesignConstraints(max_width_cm=300, budget_usd=1600, layout_priority="complete_kitchen"), walnut
    )

    def widths(plan, zone):
        return sum(p.row["width_cm"] for p in plan.placements if p.zone == zone)

    assert widths(fill, "base") > widths(full, "base")
    assert widths(full, "wall") > widths(fill, "wall")
    assert widths(full, "wall") >= widths(full, "base") - 30
    assert full.total_usd <= 1600 and fill.total_usd <= 1600


def test_priorities_agree_when_budget_allows_both():
    oak = modules_for("Natural Oak")
    a = solve_layout(DesignConstraints(max_width_cm=240, budget_usd=10_000), oak)
    b = solve_layout(DesignConstraints(max_width_cm=240, budget_usd=10_000, layout_priority="complete_kitchen"), oak)
    assert a.run_width_cm == b.run_width_cm == 240
    assert a.total_usd == b.total_usd


# --------------------------------------------------------------------------- alternatives
def test_alternatives_are_other_finishes_and_compliant(client):
    body = client.post(BOM, json=BASIC).json()
    alts = body["alternatives"]
    assert 1 <= len(alts) <= 3
    finishes = [a["finish_style"] for a in alts]
    assert body["finish_style_resolved"] not in finishes
    assert len(set(finishes)) == len(finishes)
    assert body["finish_style_score"] is not None
    for alt in alts:
        assert alt["total_usd"] <= BASIC["budget_usd"] and alt["run_width_cm"] <= BASIC["max_width_cm"]
        rows = postgres.get_modules_by_ids(alt["part_ids"])
        assert set(rows) == set(alt["part_ids"])
        assert {r["finish_style"] for r in rows.values() if r["category"] != "Countertop"} == {alt["finish_style"]}
        assert all(r["in_stock"] for r in rows.values())
        assert alt["swatch_hex"].startswith("#")


def test_alternatives_can_be_disabled(client, settings):
    settings(max_alternatives=0)
    body = client.post(BOM, json=BASIC).json()
    assert body["alternatives"] == []
    assert "alternatives" not in body["timings_ms"]


# --------------------------------------------------------------------------- saved specs
def test_spec_is_saved_and_shareable(client):
    body = client.post(BOM, json={**BASIC, "finish_style": "Sage Green"}).json()
    rid = body["request_id"]
    assert len(rid) == 16

    saved = client.get(f"/api/v1/specs/{rid}")
    assert saved.status_code == 200
    assert saved.json() == body

    recent = client.get("/api/v1/specs", params={"limit": 5}).json()
    assert recent[0]["request_id"] == rid
    assert recent[0]["finish_style"] == "Sage Green" and recent[0]["had_image"] is False


def test_saved_spec_errors(client):
    assert client.get("/api/v1/specs/0123456789abcdef").status_code == 404
    assert client.get("/api/v1/specs/not-an-id").status_code == 422


def test_saving_can_be_disabled(client, settings):
    settings(save_specs=False)
    rid = client.post(BOM, json=BASIC).json()["request_id"]
    assert client.get(f"/api/v1/specs/{rid}").status_code == 404


# --------------------------------------------------------------------------- API keys
def test_api_keys_protect_compute_endpoints_only(client, settings):
    settings(api_keys="alpha, beta")
    assert client.post(BOM, json=BASIC).status_code == 401
    assert client.post(BOM, json=BASIC, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post(BOM, json=BASIC, headers={"X-API-Key": "beta"}).status_code == 200
    search = {"query": "oak cabinets"}
    assert client.post("/api/v1/search", json=search).status_code == 401
    assert client.post("/api/v1/spec", data={"constraints": "{}"}).status_code == 401
    # Read-only endpoints and the UI stay public.
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/catalog/RS-BC-060-MWAL").status_code == 200
    assert client.get("/").status_code == 200


# --------------------------------------------------------------------------- rate limiting
def test_sliding_window_limiter():
    lim = SlidingWindowLimiter(window_s=60)
    assert lim.hit("a", 2, now=0) is None
    assert lim.hit("a", 2, now=10) is None
    assert lim.hit("a", 2, now=20) == pytest.approx(40)
    assert lim.hit("b", 2, now=20) is None  # clients are independent
    assert lim.hit("a", 2, now=60.5) is None  # first hit expired


def test_rate_limit_returns_429_with_retry_after(client, settings):
    settings(rate_limit_per_minute=2)
    assert client.post(BOM, json=BASIC).status_code == 200
    assert client.post(BOM, json=BASIC).status_code == 200
    limited = client.post(BOM, json=BASIC)
    assert limited.status_code == 429
    assert 1 <= int(limited.headers["Retry-After"]) <= 60
    # A different API key is a different client.
    assert client.post(BOM, json=BASIC, headers={"X-API-Key": "someone-else"}).status_code == 200


# --------------------------------------------------------------------------- admin API
ADMIN = {"X-API-Key": "admin-secret"}


def test_admin_api_disabled_without_keys(client):
    assert client.post("/api/v1/admin/reindex").status_code == 403


def test_admin_requires_valid_key(client, settings):
    settings(admin_api_keys="admin-secret")
    assert client.post("/api/v1/admin/reindex").status_code == 401
    assert client.post("/api/v1/admin/reindex", headers={"X-API-Key": "nope"}).status_code == 401


def test_admin_patch_price_and_stock_takes_effect(client, settings):
    settings(admin_api_keys="admin-secret")
    pid = "RS-DB-060-GWHT"
    original = client.get(f"/api/v1/catalog/{pid}").json()
    try:
        r = client.patch(f"/api/v1/admin/catalog/{pid}", json={"price_usd": 123.45, "in_stock": False}, headers=ADMIN)
        assert r.status_code == 200
        assert r.json()["price_usd"] == 123.45 and r.json()["in_stock"] is False
        rows = client.get("/api/v1/catalog", params={"finish_style": "Gloss White", "in_stock_only": True}).json()
        assert pid not in {row["part_id"] for row in rows}
    finally:
        client.patch(
            f"/api/v1/admin/catalog/{pid}",
            json={"price_usd": original["price_usd"], "in_stock": original["in_stock"]},
            headers=ADMIN,
        )
    assert client.patch("/api/v1/admin/catalog/NOPE-1", json={"in_stock": True}, headers=ADMIN).status_code == 404
    bad = client.patch(f"/api/v1/admin/catalog/{pid}", json={"width_cm": 1}, headers=ADMIN)
    assert bad.status_code == 422


def test_admin_upsert_makes_new_parts_searchable(client, settings):
    settings(admin_api_keys="admin-secret")
    item = {
        "part_id": "TEST-WC-045-TERRA",
        "part_name": "Terracotta Wall Cabinet 45 cm",
        "category": "Wall Cabinet",
        "finish_style": "Terracotta Clay",
        "price_usd": 199.0,
        "width_cm": 45,
        "height_cm": 72,
        "depth_cm": 33,
        "description": "burnt orange terracotta clay painted wall unit",
    }
    odd = {**item, "part_id": "TEST-XX-001-TERRA", "category": "Island Bench"}
    try:
        r = client.put("/api/v1/admin/catalog", json=[item, odd], headers=ADMIN)
        assert r.status_code == 200, r.text
        assert r.json()["upserted"] == 2 and r.json()["reindexed"] == 2
        assert r.json()["unknown_categories"] == ["Island Bench"]

        stored = client.get(f"/api/v1/catalog/{item['part_id']}").json()
        assert stored["image_url"].endswith("/thumbnail.svg")
        found = client.post("/api/v1/search", json={"query": "terracotta clay wall unit", "top_k": 3}).json()
        assert found["results"][0]["part_id"] == item["part_id"]
    finally:
        ids = [item["part_id"], odd["part_id"]]
        with postgres.get_engine().begin() as conn:
            conn.execute(delete(postgres.cabinet_modules).where(postgres.cabinet_modules.c.part_id.in_(ids)))
        qdrant.get_qdrant().delete(
            qdrant.collection_name(), models.PointIdsList(points=[qdrant.point_id(i) for i in ids])
        )
    assert client.get("/api/v1/health").json()["status"] == "ok"


def test_admin_upsert_validation(client, settings):
    settings(admin_api_keys="admin-secret")
    bad = [
        {
            "part_id": "X",
            "part_name": "",
            "category": "Wall Cabinet",
            "finish_style": "A",
            "price_usd": -1,
            "width_cm": 0,
            "height_cm": 72,
            "depth_cm": 33,
        }
    ]
    assert client.put("/api/v1/admin/catalog", json=bad, headers=ADMIN).status_code == 422
    assert client.put("/api/v1/admin/catalog", json=[], headers=ADMIN).status_code == 422


def test_admin_reindex(client, settings):
    settings(admin_api_keys="admin-secret")
    body = client.post("/api/v1/admin/reindex", headers=ADMIN).json()
    assert body == {"indexed": 176, "collection": "cabinet_modules_hash", "embedding_backend": "hash"}


# --------------------------------------------------------------------------- metrics
def test_metrics_exposes_requests_and_pipeline_stages(client):
    client.post(BOM, json=BASIC)
    rid = client.post(BOM, json=BASIC).json()["request_id"]
    client.get(f"/api/v1/specs/{rid}")
    text = client.get("/metrics").text
    assert 'roomspec_http_requests_total{method="POST",route="/api/v1/bom",status="200"}' in text
    assert 'route="/api/v1/specs/{request_id}"' in text  # templated, not the raw id
    assert rid not in text
    assert 'roomspec_pipeline_stage_seconds_count{stage="layout_solver"}' in text
    assert 'roomspec_specs_total{status="ok"}' in text
    assert 'roomspec_llm_narratives_total{provider="none",outcome="fallback"}' in text
