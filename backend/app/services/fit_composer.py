"""FIT feature: composite a chosen catalog module into the user's original room photo.

Uses an image-editing-capable model (Gemini's image-generation endpoint by
default) that accepts the room photo + a reference image + an edit instruction.

Known limitation this file does NOT solve: `_product_reference_png` rasterizes
the generated SVG line-drawing thumbnail (see app/core/thumbnails.py), because
that is the only image every catalog row currently has. This is a schematic,
not a photo — expect flat, diagrammatic FIT results until real product
photography is added per row (data/seed_modules.json `image_url`).

Also note: the exact Gemini image-generation request/response shape
(`generationConfig.responseModalities`, `inlineData` in the response parts)
should be re-verified against Google's current API docs before relying on
this in production — it follows the same request pattern as the existing
`genai_bom._call_gemini`, but that function only ever requests text back.
"""

from __future__ import annotations

import base64
import logging

import cairosvg
import httpx

from app.core.config import get_settings
from app.core.thumbnails import render_module_svg

log = logging.getLogger(__name__)


class FitUnavailable(RuntimeError):
    """No image-editing provider is configured."""


class FitError(RuntimeError):
    """The provider call failed or returned something unusable."""


def _provider() -> str:
    s = get_settings()
    if s.fit_provider == "none":
        return "none"
    if s.fit_provider in ("auto", "gemini") and s.gemini_api_key:
        return "gemini"
    return "none"


def _product_reference_png(row: dict) -> bytes:
    svg = render_module_svg(row)
    return cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=512, output_height=512)


def _call_gemini(room_png: bytes, product_png: bytes, row: dict) -> bytes:
    s = get_settings()
    prompt = (
        f"You are given two images: the first is a photo of a real room, the second is a "
        f"reference image of a {row['finish_style']} {row['category'].lower()} "
        f"({row['part_name']}). Edit the FIRST image only: insert the object shown in the "
        f"second image into the room, matching the room's perspective, scale and lighting. "
        f"Keep every other part of the room photo unchanged - walls, floor, windows, doors, "
        f"existing furniture, and camera angle must all be preserved. Return only the "
        f"edited photo, nothing else."
    )
    resp = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{s.gemini_image_model}:generateContent",
        headers={"x-goog-api-key": s.gemini_api_key},
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(room_png).decode()}},
                        {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(product_png).decode()}},
                    ],
                }
            ],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        },
        timeout=s.fit_timeout_s,
    )
    resp.raise_for_status()
    parts = resp.json()["candidates"][0]["content"]["parts"]
    for part in parts:
        data = part.get("inlineData", {}).get("data")
        if data:
            return base64.b64decode(data)
    raise FitError("Image-editing provider returned no image data")


def compose_fit(room_bytes: bytes, row: dict) -> bytes:
    provider = _provider()
    if provider == "none":
        raise FitUnavailable(
            "No image-editing provider is configured. Set GEMINI_API_KEY and "
            "FIT_PROVIDER=gemini (or auto) to enable the FIT feature."
        )
    product_png = _product_reference_png(row)
    try:
        return _call_gemini(room_bytes, product_png, row)
    except httpx.HTTPStatusError as exc:
        log.exception("FIT provider call failed")
        raise FitError(f"Image-editing provider returned {exc.response.status_code}") from exc
    except (KeyError, IndexError) as exc:
        log.exception("Unexpected FIT provider response shape")
        raise FitError("Image-editing provider returned an unexpected response") from exc
