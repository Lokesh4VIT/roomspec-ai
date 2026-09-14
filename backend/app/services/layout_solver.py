"""Wall-run layout solver.

Given SQL-verified modules, choose quantities that fill the wall run as fully as
possible without ever exceeding the width or budget:

* base run      - exact-width unbounded knapsack (min cost), at most one sink base
* tall unit     - optional, placed at the left end
* wall cabinets - exact-width knapsack over the base run (excluding the tall unit)
* countertop    - cheapest single piece (or jointed pair) >= base run, cut to length
* fillers       - close a residual gap of 5-20 cm

Every dimension and price in the plan is copied from a database row.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from app.schemas.spec import ComplianceCheck, DesignConstraints

PLINTH_CM = 15.0
BACKSPLASH_GAP_CM = 55.0
MAX_FILLER_GAP_CM = 20
WALL_COVER_TOLERANCE_CM = 30  # upper run may stop this short of the base run and still count as covering it
INF = math.inf


@dataclass
class Placement:
    row: dict
    zone: str  # base | tall | wall | filler | countertop
    position_cm: float
    cut_length_cm: float | None = None


@dataclass
class LayoutPlan:
    placements: list[Placement] = field(default_factory=list)
    run_width_cm: float = 0.0
    wall_gap_cm: float = 0.0
    total_usd: float = 0.0
    status: str = "ok"  # ok | partial | infeasible
    notes: list[str] = field(default_factory=list)


def _w(row: dict) -> int:
    return int(round(row["width_cm"]))


def _exact_fill(items: list[dict], cap: int) -> tuple[list[float], list[int]]:
    """Unbounded knapsack: min cost to fill exactly w cm, with back-pointers."""
    cost = [INF] * (cap + 1)
    choice = [-1] * (cap + 1)
    cost[0] = 0.0
    for w in range(1, cap + 1):
        for i, it in enumerate(items):
            iw = _w(it)
            if iw <= w and cost[w - iw] + it["price_usd"] < cost[w]:
                cost[w] = cost[w - iw] + it["price_usd"]
                choice[w] = i
    return cost, choice


def _reconstruct(items: list[dict], choice: list[int], w: int) -> list[dict]:
    out = []
    while w > 0:
        it = items[choice[w]]
        out.append(it)
        w -= _w(it)
    return out


def _best_countertop(rows: list[dict], length: int) -> tuple[float, list[dict]]:
    if length <= 0 or not rows:
        return (0.0, []) if length <= 0 else (INF, [])
    singles = [r for r in rows if _w(r) >= length]
    best: tuple[float, list[dict]] = (INF, [])
    if singles:
        r = min(singles, key=lambda r: r["price_usd"])
        best = (r["price_usd"], [r])
    for i, a in enumerate(rows):
        for b in rows[i:]:
            if _w(a) + _w(b) >= length and a["price_usd"] + b["price_usd"] < best[0] and not singles:
                best = (a["price_usd"] + b["price_usd"], [a, b])
    return best


def _fillers(rows: list[dict], gap: int) -> tuple[float, list[dict]]:
    if gap < 5 or gap > MAX_FILLER_GAP_CM or not rows:
        return 0.0, []
    cost, choice = _exact_fill(rows, gap)
    for g in range(gap, 0, -1):  # largest coverable portion; remainder is scribed
        if cost[g] < INF:
            return cost[g], _reconstruct(rows, choice, g)
    return 0.0, []


def solve_layout(c: DesignConstraints, modules: dict[str, list[dict]]) -> LayoutPlan:
    cap = int(math.floor(c.max_width_cm))
    base = [r for r in modules.get("base", []) if r["category"] != "Sink Base"]
    sinks = [r for r in modules.get("base", []) if r["category"] == "Sink Base"]
    notes: list[str] = []

    ceiling = c.ceiling_height_cm
    walls = modules.get("wall", [])
    if walls and ceiling is not None:
        ok = [r for r in walls if PLINTH_CM + 72 + 3 + BACKSPLASH_GAP_CM + r["height_cm"] <= ceiling]
        if len(ok) < len(walls):
            notes.append(f"Wall cabinets taller than the {ceiling:.0f} cm ceiling allows were excluded.")
        walls = ok
    talls = modules.get("tall", [])
    if talls and ceiling is not None:
        talls = [r for r in talls if PLINTH_CM + r["height_cm"] <= ceiling]
    countertops = modules.get("countertop", []) if c.include_countertop else []
    fillers = modules.get("filler", [])

    if not base and not sinks and not talls:
        return LayoutPlan(
            status="infeasible",
            wall_gap_cm=c.max_width_cm,
            notes=notes
            + ["No in-stock base modules satisfy the width, depth, clearance, finish and budget constraints."],
        )

    base_cost, base_choice = _exact_fill(base, cap)
    wall_cost, wall_choice = _exact_fill(walls, cap) if walls else ([0.0] + [INF] * cap, [-1] * (cap + 1))
    wall_fills = [f for f in range(cap + 1) if wall_cost[f] < INF]
    ct_cache: dict[int, tuple[float, list[dict]]] = {}
    fill_cache: dict[int, tuple[float, list[dict]]] = {}

    best_key = None
    best = None
    tall_options: list[dict | None] = [None] + sorted(talls, key=lambda r: r["price_usd"])[:1]
    sink_options: list[dict | None] = [None] + sinks

    for tall in tall_options:
        tw = _w(tall) if tall else 0
        tcost = tall["price_usd"] if tall else 0.0
        if tw > cap:
            continue
        for sink in sink_options:
            sw = _w(sink) if sink else 0
            scost = sink["price_usd"] if sink else 0.0
            for bw in range(0, cap - tw - sw + 1):
                if base_cost[bw] == INF:
                    continue
                run_base = bw + sw
                run = run_base + tw
                if run == 0:
                    continue
                subtotal = tcost + scost + base_cost[bw]
                if subtotal > c.budget_usd:
                    continue

                if countertops and run_base > 0:
                    if run_base not in ct_cache:
                        ct_cache[run_base] = _best_countertop(countertops, run_base)
                    ct_cost, _ = ct_cache[run_base]
                else:
                    ct_cost = 0.0
                ct_ok = not c.include_countertop or run_base == 0 or ct_cost < INF
                if ct_cost < INF:
                    subtotal += ct_cost

                if cap - run not in fill_cache:
                    fill_cache[cap - run] = _fillers(fillers, cap - run)
                fill_cost, fill_rows = fill_cache[cap - run]
                if subtotal + fill_cost <= c.budget_usd:
                    subtotal += fill_cost
                else:
                    fill_rows = []
                if subtotal > c.budget_usd:
                    continue

                remaining = c.budget_usd - subtotal
                wall_fill = max((f for f in wall_fills if f <= run_base and wall_cost[f] <= remaining), default=0)
                total = subtotal + wall_cost[wall_fill]
                filled = run + sum(_w(r) for r in fill_rows)

                if c.layout_priority == "complete_kitchen":
                    uppers_cover = not walls or wall_fill >= run_base - WALL_COVER_TOLERANCE_CM
                    key = (ct_ok, uppers_cover, filled, tall is not None, sink is not None, wall_fill, -total)
                else:
                    key = (ct_ok, filled, tall is not None, sink is not None, wall_fill, -total)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (tall, sink, bw, ct_ok, fill_rows, wall_fill, total)

    if best is None:
        return LayoutPlan(
            status="infeasible",
            wall_gap_cm=c.max_width_cm,
            notes=notes + [f"No combination of compliant modules fits within the ${c.budget_usd:,.2f} budget."],
        )

    tall, sink, bw, ct_ok, fill_rows, wall_fill, total = best
    placements: list[Placement] = []
    x = 0.0
    if tall:
        placements.append(Placement(tall, "tall", x))
        x += tall["width_cm"]
    run_start = x
    base_rows = sorted(_reconstruct(base, base_choice, bw), key=lambda r: -r["width_cm"])
    if sink:  # keep the sink near the middle of the run for a practical work triangle
        base_rows.insert(len(base_rows) // 2, sink)
    for r in base_rows:
        placements.append(Placement(r, "base", x))
        x += r["width_cm"]
    run_base_width = x - run_start

    wx = run_start
    for r in sorted(_reconstruct(walls, wall_choice, wall_fill), key=lambda r: -r["width_cm"]) if wall_fill else []:
        placements.append(Placement(r, "wall", wx))
        wx += r["width_cm"]

    if countertops and run_base_width > 0 and ct_ok:
        _, pieces = ct_cache[int(round(run_base_width))]
        remaining_len = run_base_width
        cx = run_start
        for p in pieces:
            cut = min(p["width_cm"], remaining_len)
            placements.append(Placement(p, "countertop", cx, cut_length_cm=cut))
            cx += cut
            remaining_len -= cut

    for r in fill_rows:
        placements.append(Placement(r, "filler", x))
        x += r["width_cm"]

    gap = round(c.max_width_cm - x, 2)
    status = "ok"
    if c.include_countertop and run_base_width > 0 and not ct_ok:
        status = "partial"
        notes.append("No compliant countertop could be added within budget and stock.")
    if c.include_wall_cabinets and walls and wall_fill < run_base_width - WALL_COVER_TOLERANCE_CM:
        status = "partial"
        notes.append(
            f"Wall cabinets cover {wall_fill} of {run_base_width:.0f} cm; budget or stock limited the upper run."
        )
    if c.include_wall_cabinets and not walls:
        notes.append("No compliant wall cabinets available for this finish and constraints.")
    if gap > MAX_FILLER_GAP_CM:
        status = "partial"
        notes.append(f"{gap:.0f} cm of the wall remains open; budget or available widths prevented a full run.")
    elif gap > 0:
        notes.append(f"{gap:.1f} cm residual gap to be scribed on site.")

    return LayoutPlan(
        placements=placements,
        run_width_cm=round(x, 2),
        wall_gap_cm=gap,
        total_usd=round(total, 2),
        status=status,
        notes=notes,
    )


def compliance_checks(c: DesignConstraints, plan: LayoutPlan, verified: dict[str, dict]) -> list[ComplianceCheck]:
    """Programmatic checks against the database rows (never against LLM output)."""
    checks: list[ComplianceCheck] = []
    rows = [verified.get(p.row["part_id"]) for p in plan.placements]
    missing = [p.row["part_id"] for p, r in zip(plan.placements, rows, strict=True) if r is None]
    checks.append(
        ComplianceCheck(
            name="parts_exist_in_catalog",
            passed=not missing,
            detail="All part numbers verified against SQL catalog" if not missing else f"Unknown: {missing}",
        )
    )
    rows = [r for r in rows if r is not None]

    floor_run = sum(p.row["width_cm"] for p in plan.placements if p.zone in ("base", "tall", "filler"))
    checks.append(
        ComplianceCheck(
            name="width_fits_wall",
            passed=floor_run <= c.max_width_cm + 1e-6,
            detail=f"Floor run {floor_run:.1f} cm <= {c.max_width_cm:.1f} cm available",
        )
    )
    wall_run = sum(p.row["width_cm"] for p in plan.placements if p.zone == "wall")
    base_run = sum(p.row["width_cm"] for p in plan.placements if p.zone == "base")
    checks.append(
        ComplianceCheck(
            name="wall_cabinets_within_base_run",
            passed=wall_run <= base_run + 1e-6,
            detail=f"Upper run {wall_run:.1f} cm over base run {base_run:.1f} cm",
        )
    )
    total = round(sum(r["price_usd"] for r in rows), 2)
    checks.append(
        ComplianceCheck(
            name="within_budget",
            passed=total <= c.budget_usd + 1e-6 and abs(total - plan.total_usd) < 0.01,
            detail=f"${total:,.2f} <= ${c.budget_usd:,.2f}",
        )
    )
    out_of_stock = [r["part_id"] for r in rows if not r["in_stock"]]
    checks.append(
        ComplianceCheck(
            name="all_in_stock",
            passed=not out_of_stock,
            detail="All parts in stock" if not out_of_stock else f"Out of stock: {out_of_stock}",
        )
    )
    if c.max_depth_cm is not None:
        deep = [r["part_id"] for r in rows if r["depth_cm"] > c.max_depth_cm]
        checks.append(
            ComplianceCheck(
                name="depth_limit",
                passed=not deep,
                detail=f"All depths <= {c.max_depth_cm:.0f} cm" if not deep else f"Too deep: {deep}",
            )
        )
    if c.front_clearance_cm is not None:
        tight = [r["part_id"] for r in rows if r["door_clearance_cm"] > c.front_clearance_cm]
        checks.append(
            ComplianceCheck(
                name="door_swing_clearance",
                passed=not tight,
                detail=f"Door/drawer swing <= {c.front_clearance_cm:.0f} cm" if not tight else f"Blocked: {tight}",
            )
        )
    cabinet_finishes = {r["finish_style"] for r in rows if r["category"] != "Countertop"}
    if c.finish_style:
        ok = all(f.lower() == c.finish_style.lower() for f in cabinet_finishes)
        detail = f"Requested {c.finish_style}; plan uses {sorted(cabinet_finishes)}"
    else:
        ok = len(cabinet_finishes) <= 1
        detail = f"Single cabinet finish: {sorted(cabinet_finishes)}"
    checks.append(ComplianceCheck(name="finish_consistency", passed=ok, detail=detail))
    if c.include_countertop and base_run > 0:
        ct_len = sum(p.cut_length_cm or 0 for p in plan.placements if p.zone == "countertop")
        checks.append(
            ComplianceCheck(
                name="countertop_covers_base_run",
                passed=ct_len + 1e-6 >= base_run,
                detail=f"Worktop {ct_len:.1f} cm over base run {base_run:.1f} cm",
                kind="completeness",
            )
        )
    return checks


def summarize_lines(plan: LayoutPlan) -> list[tuple[dict, int, list[float], list[float]]]:
    """Group placements by part: (row, quantity, positions, cut lengths)."""
    order: list[str] = []
    groups: dict[str, list[Placement]] = {}
    for p in plan.placements:
        pid = p.row["part_id"]
        if pid not in groups:
            order.append(pid)
            groups[pid] = []
        groups[pid].append(p)
    counts = Counter(p.row["part_id"] for p in plan.placements)
    return [
        (
            groups[pid][0].row,
            counts[pid],
            [p.position_cm for p in groups[pid]],
            [p.cut_length_cm for p in groups[pid] if p.cut_length_cm is not None],
        )
        for pid in order
    ]
