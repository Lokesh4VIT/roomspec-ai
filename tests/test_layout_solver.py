"""Layout solver and compliance checks against the real seed catalog."""

import json
from pathlib import Path

import pytest

from app.schemas.spec import DesignConstraints
from app.services.layout_solver import compliance_checks, solve_layout

SEED = json.loads((Path(__file__).resolve().parents[1] / "data" / "seed_modules.json").read_text())


def modules_for(finish: str, countertop: str = "Quartz White", **overrides) -> dict[str, list[dict]]:
    rows = [r for r in SEED if r["in_stock"]]
    cab = [r for r in rows if r["finish_style"] == finish]
    groups = {
        "base": [r for r in cab if r["category"] in ("Base Cabinet", "Drawer Base", "Sink Base")],
        "filler": [r for r in cab if r["category"] == "Filler Panel"],
        "wall": [r for r in cab if r["category"] == "Wall Cabinet"],
        "tall": [r for r in cab if r["category"] == "Tall Pantry"],
        "countertop": [r for r in rows if r["finish_style"] == countertop],
    }
    groups.update(overrides)
    return groups


def by_id() -> dict[str, dict]:
    return {r["part_id"]: r for r in SEED}


@pytest.mark.parametrize("width", [60, 95, 150, 240, 301, 420])
@pytest.mark.parametrize("budget", [600, 1500, 5000])
def test_plan_never_exceeds_width_or_budget(width, budget):
    c = DesignConstraints(max_width_cm=width, budget_usd=budget)
    plan = solve_layout(c, modules_for("Gloss White"))
    floor = sum(p.row["width_cm"] for p in plan.placements if p.zone in ("base", "tall", "filler"))
    assert floor <= width
    assert plan.total_usd <= budget
    assert plan.total_usd == pytest.approx(sum(p.row["price_usd"] for p in plan.placements), abs=0.01)
    if plan.placements:
        failed = [ch for ch in compliance_checks(c, plan, by_id()) if not ch.passed]
        assert not failed, failed


def test_generous_budget_fills_standard_wall_exactly():
    c = DesignConstraints(max_width_cm=240, budget_usd=10_000)
    plan = solve_layout(c, modules_for("Natural Oak"))
    assert plan.status == "ok"
    assert plan.wall_gap_cm == 0
    zones = {p.zone for p in plan.placements}
    assert {"base", "wall", "countertop"} <= zones
    assert sum(p.row["category"] == "Sink Base" for p in plan.placements) == 1


def test_countertop_is_cut_to_base_run():
    c = DesignConstraints(max_width_cm=200, budget_usd=10_000)
    plan = solve_layout(c, modules_for("Gloss White"))
    base_run = sum(p.row["width_cm"] for p in plan.placements if p.zone == "base")
    cut = sum(p.cut_length_cm for p in plan.placements if p.zone == "countertop")
    assert cut == pytest.approx(base_run)


def test_tiny_budget_is_infeasible_not_overspent():
    plan = solve_layout(DesignConstraints(max_width_cm=240, budget_usd=50), modules_for("Matte Walnut"))
    assert plan.status == "infeasible"
    assert plan.placements == []


def test_low_ceiling_excludes_wall_cabinets():
    c = DesignConstraints(max_width_cm=240, budget_usd=10_000, ceiling_height_cm=200)
    plan = solve_layout(c, modules_for("Gloss White"))
    assert not [p for p in plan.placements if p.zone == "wall"]
    assert any("ceiling" in n for n in plan.notes)


def test_tall_unit_is_used_when_requested():
    c = DesignConstraints(max_width_cm=300, budget_usd=10_000, include_tall_units=True, ceiling_height_cm=260)
    plan = solve_layout(c, modules_for("Cream Shaker"))
    tall = [p for p in plan.placements if p.zone == "tall"]
    assert len(tall) == 1 and tall[0].position_cm == 0


def test_no_base_modules_is_infeasible():
    plan = solve_layout(DesignConstraints(), modules_for("Gloss White", base=[], tall=[]))
    assert plan.status == "infeasible"


def test_compliance_flags_tampered_price():
    c = DesignConstraints(max_width_cm=240, budget_usd=3000)
    plan = solve_layout(c, modules_for("Gloss White"))
    catalog = by_id()
    victim = plan.placements[0].row["part_id"]
    catalog[victim] = {**catalog[victim], "price_usd": catalog[victim]["price_usd"] + 999}
    checks = {ch.name: ch for ch in compliance_checks(c, plan, catalog)}
    assert not checks["within_budget"].passed
