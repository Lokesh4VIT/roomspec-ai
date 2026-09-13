"""Regenerate `data/seed_modules.json` — a deterministic modular-kitchen catalog.

Run: python scripts/generate_seed.py
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "seed_modules.json"

CABINET_FINISHES = {
    # finish: (code, price multiplier, material, style words)
    "Matte Walnut": ("MWAL", 1.35, "Walnut veneer on MDF", "warm dark brown wood grain, mid-century, matte"),
    "Natural Oak": ("NOAK", 1.25, "Oak veneer on plywood", "light honey wood grain, scandinavian, natural"),
    "Gloss White": ("GWHT", 1.00, "High-gloss lacquered MDF", "bright white, glossy, minimalist, modern"),
    "Cream Shaker": ("CRSH", 1.10, "Painted solid-frame MDF", "cream off-white, shaker panel, farmhouse, classic"),
    "Charcoal Grey": ("CGRY", 1.05, "Super-matte laminate", "dark grey, industrial, contemporary, matte"),
    "Matte Black": ("MBLK", 1.15, "Fingerprint-resistant laminate", "black, dramatic, modern, matte"),
    "Sage Green": ("SGRN", 1.12, "Painted MDF", "muted green, organic, cottage, soft"),
    "Navy Blue": ("NAVY", 1.12, "Painted MDF", "deep blue, bold, coastal, classic"),
}

# category: (code, widths_cm, height_cm, depth_cm, base price per 60cm, door clearance, blurb)
CABINET_TYPES = {
    "Base Cabinet": (
        "BC",
        [30, 40, 45, 50, 60, 80, 90],
        72,
        58,
        180,
        45,
        "single/double door base unit with adjustable shelf",
    ),
    "Drawer Base": ("DB", [40, 60, 80], 72, 58, 290, 50, "three soft-close full-extension drawers"),
    "Sink Base": ("SB", [60, 80], 72, 58, 210, 45, "open-back base unit for sink and plumbing"),
    "Corner Base Cabinet": ("CB", [90], 72, 58, 340, 55, "blind corner base with pull-out carousel"),
    "Wall Cabinet": ("WC", [30, 40, 60, 80], 72, 33, 140, 35, "wall-hung unit with lift or hinged door"),
    "Tall Pantry": ("TP", [60], 200, 58, 520, 50, "full-height pantry with internal drawers"),
    "Filler Panel": ("FP", [5, 10], 72, 2, 25, 0, "scribe filler to close wall gaps"),
}

COUNTERTOPS = {
    # material: (code, price per metre, thickness_cm, words)
    "Quartz White": ("QWHT", 210, 2, "white engineered quartz, veined, bright"),
    "Butcher Block Oak": ("BBOK", 120, 4, "solid oak butcher block, warm wood"),
    "Black Granite": ("BGRN", 240, 3, "polished black granite, speckled, dramatic"),
    "Concrete Grey": ("CCGR", 150, 3, "grey concrete-effect laminate, industrial"),
}
COUNTERTOP_LENGTHS = [120, 180, 240, 300]


def main() -> None:
    rng = random.Random(42)
    rows = []

    for finish, (fcode, mult, material, words) in CABINET_FINISHES.items():
        for category, (ccode, widths, h, d, base_price, clearance, blurb) in CABINET_TYPES.items():
            for w in widths:
                price = base_price * mult * (0.55 + 0.45 * w / 60)
                rows.append(
                    {
                        "part_id": f"RS-{ccode}-{w:03d}-{fcode}",
                        "part_name": f"{finish} {category} {w} cm",
                        "category": category,
                        "finish_style": finish,
                        "material": material,
                        "price_usd": round(price, 2),
                        "width_cm": float(w),
                        "height_cm": float(h),
                        "depth_cm": float(d),
                        "door_clearance_cm": float(clearance),
                        "in_stock": rng.random() > 0.15,
                        "description": f"{w} cm {category.lower()}, {blurb}. Finish: {words}.",
                    }
                )

    for finish, (fcode, per_m, thick, words) in COUNTERTOPS.items():
        for length in COUNTERTOP_LENGTHS:
            rows.append(
                {
                    "part_id": f"RS-CT-{length:03d}-{fcode}",
                    "part_name": f"{finish} Countertop {length} cm",
                    "category": "Countertop",
                    "finish_style": finish,
                    "material": words.split(",")[0],
                    "price_usd": round(per_m * length / 100, 2),
                    "width_cm": float(length),
                    "height_cm": float(thick),
                    "depth_cm": 62.0,
                    "door_clearance_cm": 0.0,
                    "in_stock": rng.random() > 0.1,
                    "description": f"{length} cm x 62 cm worktop, cut-to-length on site. Surface: {words}.",
                }
            )

    for row in rows:
        row["image_url"] = f"/api/v1/catalog/{row['part_id']}/thumbnail.svg"

    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {len(rows)} modules to {OUT}")


if __name__ == "__main__":
    main()
