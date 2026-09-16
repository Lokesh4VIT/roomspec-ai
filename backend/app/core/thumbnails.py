"""Shared SVG rendering for catalog module thumbnails.

Used by the `/catalog/{part_id}/thumbnail.svg` route AND by `fit_composer`,
which rasterizes the same drawing as the "product reference" image for the
FIT feature. Extracted so both call one implementation instead of the route
handler duplicating this logic (or fit_composer making an internal HTTP call
to itself).

IMPORTANT: this is a generated line-drawing, not a photo. Every seeded catalog
row's `image_url` currently points here (see data/seed_modules.json) — there
is no real product photography in this repo. FIT results will look schematic,
not photorealistic, until real photos/renders replace this per module.
"""

from __future__ import annotations

from html import escape

from app.core.finishes import swatch_rgb


def render_module_svg(row: dict) -> str:
    rgb = swatch_rgb(row["finish_style"])
    fill = "#{:02x}{:02x}{:02x}".format(*rgb)
    stroke = "#1f2937" if sum(rgb) > 300 else "#e5e7eb"
    w, h = row["width_cm"], row["height_cm"]
    scale = 110 / max(w, h)
    sw, sh = w * scale, max(h * scale, 6)
    x, y = (120 - sw) / 2, (120 - sh) / 2
    inner = ""
    cat = row["category"]
    if cat == "Drawer Base":
        inner = "".join(
            f'<line x1="{x}" y1="{y + sh * i / 3}" x2="{x + sw}" y2="{y + sh * i / 3}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<rect x="{x + sw / 2 - 8}" y="{y + sh * i / 3 + 5}" width="16" height="3" rx="1.5" fill="{stroke}"/>'
            for i in range(3)
        )
    elif cat not in ("Countertop", "Filler Panel"):
        doors = 2 if w >= 60 else 1
        for d in range(doors):
            dx = x + sw * d / doors
            inner += f'<rect x="{dx + 3}" y="{y + 3}" width="{sw / doors - 6}" height="{sh - 6}" fill="none" stroke="{stroke}" stroke-width="1.2"/>'
            hx = dx + sw / doors - 9 if doors == 1 or d == 0 else dx + 6
            inner += (
                f'<rect x="{hx}" y="{y + sh * 0.2}" width="3" height="{min(18, sh * 0.25)}" rx="1.5" fill="{stroke}"/>'
            )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" role="img" aria-label="{escape(row["part_name"])}">'
        f'<rect x="{x}" y="{y}" width="{sw}" height="{sh}" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        f"{inner}</svg>"
    )
