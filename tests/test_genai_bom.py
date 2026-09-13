"""Grounded narrative generation and hallucination verification."""

import pytest
from test_layout_solver import modules_for

from app.schemas.spec import DesignConstraints
from app.services import genai_bom
from app.services.layout_solver import solve_layout


@pytest.fixture
def spec():
    c = DesignConstraints(max_width_cm=240, budget_usd=4000)
    plan = solve_layout(c, modules_for("Matte Walnut"))
    lines = genai_bom.build_bom_lines(plan)
    return c, plan, lines, genai_bom.build_fact_sheet(c, plan, lines)


def test_bom_lines_aggregate_quantities(spec):
    _, plan, lines, _ = spec
    assert sum(ln.quantity for ln in lines) == len(plan.placements)
    assert sum(ln.line_total_usd for ln in lines) == pytest.approx(plan.total_usd, abs=0.02)
    assert [ln.line for ln in lines] == list(range(1, len(lines) + 1))


def test_verifier_accepts_grounded_text(spec):
    c, plan, lines, facts = spec
    pid = lines[0].part_id
    text = f"Install {pid} ({lines[0].width_cm:g} cm wide) first. Total ${plan.total_usd:,.2f}; plinth 15 cm."
    assert genai_bom.verify_text(text, facts) == ([], [])


def test_verifier_catches_invented_part_dimension_and_price(spec):
    _, _, _, facts = spec
    ids, values = genai_bom.verify_text("Add RS-XX-999-FAKE, a 73.5 cm unit for $12.34.", facts)
    assert ids == ["RS-XX-999-FAKE"]
    assert "73.5 cm" in values and "$12.34" in values


def test_template_narrative_passes_its_own_verification(spec):
    c, plan, lines, _ = spec
    result = genai_bom.generate_narrative(c, plan, lines)
    assert not result.report.llm_used
    assert result.report.passed
    assert result.installation_notes


def test_hallucinating_llm_is_replaced_by_template(spec, monkeypatch):
    c, plan, lines, _ = spec
    monkeypatch.setattr(genai_bom, "_provider", lambda: "groq")
    monkeypatch.setattr(
        genai_bom,
        "_call_groq",
        lambda facts: {"summary": "Uses RS-BC-999-GOLD at $9,999.00.", "installation_notes": ["Fit 13 cm spacer."]},
    )
    result = genai_bom.generate_narrative(c, plan, lines)
    assert not result.report.llm_used
    assert result.report.unknown_part_ids == ["RS-BC-999-GOLD"]
    assert "RS-BC-999-GOLD" not in result.summary + " ".join(result.installation_notes)
    assert result.report.fallback_reason


def test_grounded_llm_output_is_used(spec, monkeypatch):
    c, plan, lines, _ = spec
    monkeypatch.setattr(genai_bom, "_provider", lambda: "gemini")
    monkeypatch.setattr(
        genai_bom,
        "_call_gemini",
        lambda facts: {
            "summary": f"Walnut run for ${plan.total_usd:,.2f}.",
            "installation_notes": [f"Level {lines[0].part_id} on 15 cm legs."],
        },
    )
    result = genai_bom.generate_narrative(c, plan, lines)
    assert result.report.llm_used and result.report.passed
    assert result.summary.startswith("Walnut run")


def test_llm_failure_falls_back(spec, monkeypatch):
    c, plan, lines, _ = spec
    monkeypatch.setattr(genai_bom, "_provider", lambda: "groq")

    def boom(facts):
        raise TimeoutError("slow")

    monkeypatch.setattr(genai_bom, "_call_groq", boom)
    result = genai_bom.generate_narrative(c, plan, lines)
    assert not result.report.llm_used
    assert "TimeoutError" in result.report.fallback_reason


def test_parse_json_strips_code_fences():
    out = genai_bom._parse_json('```json\n{"summary": "s", "installation_notes": ["a"]}\n```')
    assert out == {"summary": "s", "installation_notes": ["a"]}
