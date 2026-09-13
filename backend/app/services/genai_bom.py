"""Grounded BOM + installation-notes generator.

The BOM table itself is built deterministically from SQL rows. The LLM (Groq or
Gemini) only writes the prose summary and installation notes from a JSON fact
sheet, and its output is then verified: every part number, centimetre value and
dollar amount it mentions must exist in the fact sheet. Anything unverifiable is
rejected and replaced by a deterministic template, so the response never
contains a hallucinated part, dimension or price.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

from app.core.config import get_settings
from app.schemas.spec import BOMLine, DesignConstraints, HallucinationReport
from app.services.layout_solver import BACKSPLASH_GAP_CM, PLINTH_CM, LayoutPlan, summarize_lines

log = logging.getLogger(__name__)

PART_ID_RE = re.compile(r"\b[A-Z]{2,}-[A-Z0-9]{2,}(?:-[A-Z0-9]+)+\b")
CM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:cm|centimet(?:er|re)s?)\b", re.IGNORECASE)
USD_RE = re.compile(r"(?:\$|USD\s?)\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)")

SYSTEM_PROMPT = """You are a kitchen installation specification writer.
You receive a JSON fact sheet containing a verified bill of materials.
Rules:
- Use ONLY part numbers, dimensions (cm) and prices ($) that appear in the fact sheet.
- Never invent parts, sizes, prices, brands or stock levels. If something is unknown, omit it.
- Write practical, ordered installation notes for a contractor.
Return strict JSON: {"summary": "<2-3 sentences>", "installation_notes": ["<step>", ...]} with 4-8 notes."""


@dataclass
class Narrative:
    summary: str
    installation_notes: list[str]
    report: HallucinationReport


def build_bom_lines(plan: LayoutPlan) -> list[BOMLine]:
    lines = []
    for i, (row, qty, positions, cuts) in enumerate(summarize_lines(plan), start=1):
        note_parts = []
        if len(positions) > 1:
            note_parts.append("positions " + ", ".join(f"{p:g}" for p in positions) + " cm")
        if cuts and any(abs(cut - row["width_cm"]) > 1e-6 for cut in cuts):
            note_parts.append("cut to " + " + ".join(f"{cut:g}" for cut in cuts) + " cm")
        lines.append(
            BOMLine(
                line=i,
                part_id=row["part_id"],
                part_name=row["part_name"],
                category=row["category"],
                finish_style=row["finish_style"],
                quantity=qty,
                unit_price_usd=row["price_usd"],
                line_total_usd=round(row["price_usd"] * qty, 2),
                width_cm=row["width_cm"],
                height_cm=row["height_cm"],
                depth_cm=row["depth_cm"],
                position_cm=positions[0],
                note="; ".join(note_parts) or None,
            )
        )
    return lines


def build_fact_sheet(c: DesignConstraints, plan: LayoutPlan, lines: list[BOMLine]) -> dict:
    return {
        "constraints": c.model_dump(exclude_none=True),
        "layout": {
            "status": plan.status,
            "floor_run_width_cm": plan.run_width_cm,
            "open_gap_cm": plan.wall_gap_cm,
            "total_usd": plan.total_usd,
            "plinth_height_cm": PLINTH_CM,
            "worktop_to_wall_cabinet_gap_cm": BACKSPLASH_GAP_CM,
            "notes": plan.notes,
        },
        "bill_of_materials": [
            {
                "part_id": ln.part_id,
                "name": ln.part_name,
                "category": ln.category,
                "finish": ln.finish_style,
                "qty": ln.quantity,
                "unit_price_usd": ln.unit_price_usd,
                "line_total_usd": ln.line_total_usd,
                "width_cm": ln.width_cm,
                "height_cm": ln.height_cm,
                "depth_cm": ln.depth_cm,
                "first_position_cm": ln.position_cm,
                "note": ln.note,
            }
            for ln in lines
        ],
    }


def _numbers_in(obj) -> set[float]:
    found: set[float] = set()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float)):
        found.add(round(float(obj), 2))
    elif isinstance(obj, str):
        found |= {round(float(m.replace(",", "")), 2) for m in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", obj)}
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= _numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _numbers_in(v)
    return found


def verify_text(text: str, facts: dict) -> tuple[list[str], list[str]]:
    """Return (unknown part ids, unverified cm/$ values) mentioned in `text`."""
    known_ids = {ln["part_id"] for ln in facts["bill_of_materials"]}
    unknown_ids = sorted({pid for pid in PART_ID_RE.findall(text) if pid not in known_ids})

    allowed = _numbers_in(facts)
    # Simple derived values a writer may legitimately state.
    lines = facts["bill_of_materials"]
    allowed |= {round(ln["height_cm"] + facts["layout"]["plinth_height_cm"], 2) for ln in lines}
    worktop_h = [ln["height_cm"] for ln in lines if ln["category"] == "Countertop"]
    base_h = [ln["height_cm"] for ln in lines if ln["category"] in ("Base Cabinet", "Drawer Base", "Sink Base")]
    for bh in base_h:
        top = PLINTH_CM + bh + (worktop_h[0] if worktop_h else 0)
        allowed |= {round(top, 2), round(top + BACKSPLASH_GAP_CM, 2)}

    unverified = []
    for m in CM_RE.finditer(text):
        if round(float(m.group(1)), 2) not in allowed:
            unverified.append(m.group(0))
    for m in USD_RE.finditer(text):
        if round(float(m.group(1).replace(",", "")), 2) not in allowed:
            unverified.append(m.group(0))
    return unknown_ids, sorted(set(unverified))


def template_narrative(c: DesignConstraints, plan: LayoutPlan, lines: list[BOMLine]) -> tuple[str, list[str]]:
    if not lines:
        return "No compliant layout could be produced. " + " ".join(plan.notes), []

    cabinet_finishes = sorted({ln.finish_style for ln in lines if ln.category != "Countertop"})
    units = sum(ln.quantity for ln in lines if ln.category != "Countertop")
    summary = (
        f"{units} catalog units in {', '.join(cabinet_finishes) or 'the selected finish'} fill "
        f"{plan.run_width_cm:g} cm of the {c.max_width_cm:g} cm wall for ${plan.total_usd:,.2f} "
        f"(budget ${c.budget_usd:,.2f}). All parts are in stock and verified against the catalog."
    )

    notes = [
        f"Confirm the wall run measures at least {plan.run_width_cm:g} cm at floor level and at "
        f"worktop height before unpacking; walls are rarely square.",
        f"Set plinth legs to {PLINTH_CM:g} cm and level the base run from the highest point of the floor.",
    ]
    tall = [ln for ln in lines if ln.category == "Tall Pantry"]
    if tall:
        notes.append(f"Install {tall[0].part_id} first at the left end and fix it to the wall with anti-tip brackets.")
    base = [ln for ln in lines if ln.category in ("Base Cabinet", "Drawer Base", "Sink Base")]
    if base:
        order = ", ".join(f"{ln.part_id} x{ln.quantity}" if ln.quantity > 1 else ln.part_id for ln in base)
        notes.append(f"Place and clamp base units left to right ({order}) before screwing carcasses together.")
    sink = [ln for ln in lines if ln.category == "Sink Base"]
    if sink:
        notes.append(f"Mark supply and waste positions before fixing {sink[0].part_id}; its back is open for plumbing.")
    worktops = [ln for ln in lines if ln.category == "Countertop"]
    for ln in worktops:
        cut = f" and {ln.note}" if ln.note else ""
        notes.append(f"Dry-fit worktop {ln.part_id} ({ln.depth_cm:g} cm deep){cut}; seal all cut edges.")
    walls = [ln for ln in lines if ln.category == "Wall Cabinet"]
    if walls:
        notes.append(
            f"Mount wall cabinets on a rail with a {BACKSPLASH_GAP_CM:g} cm gap above the worktop, "
            f"aligned to the left edge of the base run."
        )
    fillers = [ln for ln in lines if ln.category == "Filler Panel"]
    if fillers:
        notes.append(f"Scribe {fillers[0].part_id} to the wall at the right end of the run.")
    if c.front_clearance_cm is not None:
        notes.append(
            f"Keep the {c.front_clearance_cm:g} cm floor zone in front of the run free for door and drawer swing."
        )
    notes.extend(plan.notes)
    return summary, notes


def _provider() -> str:
    s = get_settings()
    if s.llm_provider == "none":
        return "none"
    if s.llm_provider in ("auto", "groq") and s.groq_api_key:
        return "groq"
    if s.llm_provider in ("auto", "gemini") and s.gemini_api_key:
        return "gemini"
    return "none"


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    data = json.loads(text)
    if not isinstance(data.get("summary"), str) or not isinstance(data.get("installation_notes"), list):
        raise ValueError("LLM JSON missing summary/installation_notes")
    return {"summary": data["summary"], "installation_notes": [str(n) for n in data["installation_notes"]]}


def _call_groq(facts: dict) -> dict:
    s = get_settings()
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {s.groq_api_key}"},
        json={
            "model": s.groq_model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(facts)},
            ],
        },
        timeout=s.llm_timeout_s,
    )
    resp.raise_for_status()
    return _parse_json(resp.json()["choices"][0]["message"]["content"])


def _call_gemini(facts: dict) -> dict:
    s = get_settings()
    resp = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{s.gemini_model}:generateContent",
        headers={"x-goog-api-key": s.gemini_api_key},
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(facts)}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        },
        timeout=s.llm_timeout_s,
    )
    resp.raise_for_status()
    return _parse_json(resp.json()["candidates"][0]["content"]["parts"][0]["text"])


def generate_narrative(c: DesignConstraints, plan: LayoutPlan, lines: list[BOMLine]) -> Narrative:
    facts = build_fact_sheet(c, plan, lines)
    provider = _provider()
    fallback_reason = None

    if provider != "none" and lines:
        try:
            out = (_call_groq if provider == "groq" else _call_gemini)(facts)
            text = out["summary"] + "\n" + "\n".join(out["installation_notes"])
            unknown_ids, unverified = verify_text(text, facts)
            if not unknown_ids and not unverified:
                return Narrative(
                    out["summary"],
                    out["installation_notes"],
                    HallucinationReport(llm_used=True, provider=provider, passed=True),
                )
            fallback_reason = "LLM output referenced unverified parts or values; replaced with template"
            log.warning("%s: ids=%s values=%s", fallback_reason, unknown_ids, unverified)
            report_ids, report_vals = unknown_ids, unverified
        except Exception as exc:
            fallback_reason = f"LLM call failed: {type(exc).__name__}"
            log.warning("LLM narrative failed: %s", exc)
            report_ids, report_vals = [], []
    else:
        report_ids, report_vals = [], []
        if provider == "none":
            fallback_reason = "No LLM API key configured; deterministic template used"

    summary, notes = template_narrative(c, plan, lines)
    # The template is held to the same verification standard as the LLM.
    t_ids, t_vals = verify_text(summary + "\n" + "\n".join(notes), facts)
    return Narrative(
        summary,
        notes,
        HallucinationReport(
            llm_used=False,
            provider=provider,
            passed=not t_ids and not t_vals,
            unknown_part_ids=report_ids + t_ids,
            unverified_numbers=report_vals + t_vals,
            fallback_reason=fallback_reason,
        ),
    )
