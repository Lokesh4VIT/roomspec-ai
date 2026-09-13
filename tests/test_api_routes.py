"""API contract tests over the full stack (SQLite + embedded Qdrant + hash embeddings)."""

import json

import pytest

from app.db import postgres


def test_health_reports_seeded_and_indexed(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["catalog_rows"] == body["vector_points"] == 176
    assert body["embedding_backend"] == "hash"
    assert body["llm_provider"] == "none"


def test_vector_collection_is_namespaced_by_embedding_backend(client):
    from app.db import qdrant

    assert qdrant.collection_name() == "cabinet_modules_hash"
    assert qdrant.get_qdrant().collection_exists("cabinet_modules_hash")


def test_frontend_and_docs_are_served(client):
    assert "RoomSpec AI" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/docs").status_code == 200
    assert "/api/v1/spec" in client.get("/openapi.json").json()["paths"]


def test_catalog_filters_are_hard_constraints(client):
    rows = client.get(
        "/api/v1/catalog",
        params={"category": "Wall Cabinet", "max_width_cm": 45, "in_stock_only": True},
    ).json()
    assert rows
    assert all(r["category"] == "Wall Cabinet" and r["width_cm"] <= 45 and r["in_stock"] for r in rows)


def test_catalog_item_and_thumbnail(client):
    item = client.get("/api/v1/catalog/RS-BC-060-MWAL").json()
    assert item["width_cm"] == 60
    svg = client.get(item["image_url"])
    assert svg.status_code == 200 and svg.headers["content-type"].startswith("image/svg+xml")
    assert client.get("/api/v1/catalog/NOPE").status_code == 404


def test_facets(client):
    f = client.get("/api/v1/catalog/facets").json()
    assert "Countertop" in f["categories"]
    assert "Quartz White" in f["finish_styles"] and "Quartz White" not in f["cabinet_finishes"]


def test_search_ranks_style_and_enforces_dimensions(client):
    res = client.post(
        "/api/v1/search", json={"query": "sage green cottage cabinets", "max_width_cm": 50, "top_k": 5}
    ).json()
    assert res["results"]
    assert all(r["width_cm"] <= 50 and r["in_stock"] for r in res["results"])
    assert res["results"][0]["finish_style"] == "Sage Green"
    scores = [r["style_score"] for r in res["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_validation(client):
    assert client.post("/api/v1/search", json={"query": "x"}).status_code == 422
    assert client.post("/api/v1/search", json={"query": "oak", "bogus": 1}).status_code == 422


def _assert_verified(body):
    catalog = postgres.get_modules_by_ids([ln["part_id"] for ln in body["bom"]])
    for ln in body["bom"]:
        row = catalog[ln["part_id"]]
        assert (ln["width_cm"], ln["height_cm"], ln["depth_cm"]) == (row["width_cm"], row["height_cm"], row["depth_cm"])
        assert ln["unit_price_usd"] == row["price_usd"]
        assert row["in_stock"]
    assert all(ch["passed"] for ch in body["compliance"] if ch["kind"] == "constraint"), body["compliance"]
    assert body["hallucination_check"]["passed"]


def test_spec_with_image(client, room_jpeg):
    constraints = {"max_width_cm": 240, "budget_usd": 3000, "style_prompt": "warm walnut wood"}
    r = client.post(
        "/api/v1/spec",
        data={"constraints": json.dumps(constraints)},
        files={"image": ("room.jpg", room_jpeg, "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_analysis"]["floor_line_y_ratio"] is not None
    assert body["finish_style_resolved"] == "Matte Walnut"
    assert body["run_width_cm"] <= 240 and body["total_usd"] <= 3000
    assert {"cv_preprocess", "embedding", "vector_search", "sql_filter", "total"} <= body["timings_ms"].keys()
    assert len(body["layout"]) == sum(ln["quantity"] for ln in body["bom"])
    _assert_verified(body)


def test_bom_without_image_respects_explicit_finish(client):
    r = client.post("/api/v1/bom", json={"max_width_cm": 180, "budget_usd": 2500, "finish_style": "navy blue"})
    body = r.json()
    assert r.status_code == 200
    assert body["image_analysis"] is None
    assert {ln["finish_style"] for ln in body["bom"] if ln["category"] != "Countertop"} == {"Navy Blue"}
    _assert_verified(body)


@pytest.mark.parametrize(
    "constraints",
    [
        {"max_width_cm": 150, "budget_usd": 1200, "max_depth_cm": 40},
        {"max_width_cm": 320, "budget_usd": 6000, "front_clearance_cm": 48, "ceiling_height_cm": 240},
        {"max_width_cm": 90, "budget_usd": 700, "include_countertop": False, "include_wall_cabinets": False},
    ],
)
def test_spec_constraint_matrix(client, constraints):
    body = client.post("/api/v1/bom", json=constraints).json()
    if body["status"] == "infeasible":
        assert body["bom"] == []
        return
    for ln in body["bom"]:
        row = postgres.get_modules_by_ids([ln["part_id"]])[ln["part_id"]]
        if "max_depth_cm" in constraints:
            assert row["depth_cm"] <= constraints["max_depth_cm"]
        if "front_clearance_cm" in constraints:
            assert row["door_clearance_cm"] <= constraints["front_clearance_cm"]
    _assert_verified(body)


def test_depth_limit_that_excludes_all_base_units_is_infeasible(client):
    body = client.post("/api/v1/bom", json={"max_width_cm": 240, "budget_usd": 3000, "max_depth_cm": 40}).json()
    assert body["status"] == "infeasible"
    assert body["bom"] == [] and body["total_usd"] == 0


def test_spec_rejects_bad_inputs(client):
    bad_json = client.post("/api/v1/spec", data={"constraints": "{not json"})
    assert bad_json.status_code == 422
    too_narrow = client.post("/api/v1/spec", data={"constraints": json.dumps({"max_width_cm": 5})})
    assert too_narrow.status_code == 422
    wrong_type = client.post(
        "/api/v1/spec", data={"constraints": "{}"}, files={"image": ("x.gif", b"GIF89a", "image/gif")}
    )
    assert wrong_type.status_code == 415
    corrupt = client.post(
        "/api/v1/spec", data={"constraints": "{}"}, files={"image": ("x.jpg", b"\xff\xd8garbage", "image/jpeg")}
    )
    assert corrupt.status_code == 422


def test_samples_listing(client):
    samples = client.get("/api/v1/samples").json()
    assert len(samples) == 5
    assert client.get(samples[0]["url"]).status_code == 200
