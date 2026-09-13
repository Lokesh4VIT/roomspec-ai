"""OpenCV room-photo preprocessing.

Pipeline: decode + integrity check -> resize -> CLAHE on the LAB lightness
channel -> line analysis (floor boundary, wall corner, perspective tilt) ->
tilt correction + partial gray-world white balance -> wall ROI mask ->
dominant colours mapped to catalog finishes.

The geometry estimates are heuristics from Hough line statistics; they describe
the photo, they never override the user's measured dimensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.finishes import CABINET_SWATCHES

MAX_SIDE = 1024
MIN_SIDE = 64

FINISH_SWATCHES = CABINET_SWATCHES


class ImageValidationError(ValueError):
    """Raised for unreadable, corrupt or out-of-bounds images."""


@dataclass
class PreprocessResult:
    image_rgb: np.ndarray  # contrast/white-balance normalized, tilt-corrected RGB uint8
    embedding_rgb: np.ndarray  # colour-faithful, tilt-corrected, padded to square for CLIP
    wall_mask: np.ndarray  # uint8 {0,255}, same HxW
    analysis: dict = field(default_factory=dict)


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes to BGR and validate integrity."""
    if not data:
        raise ImageValidationError("Empty upload")
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageValidationError("File is not a decodable JPEG/PNG/WebP image")
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise ImageValidationError(f"Image too small ({w}x{h}); need at least {MIN_SIDE}px per side")
    return img


def resize_max_side(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def gray_world_balance(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Partially neutralize colour casts from warm/cool lighting.

    Full gray-world correction would also erase the room's real colour (a beige wall
    becomes grey), so only `strength` of the correction is applied.
    """
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1e-3)
    gain = 1 + strength * (np.clip(gain, 0.6, 1.6) - 1)
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    return cv2.cvtColor(cv2.merge((clahe.apply(l_chan), a, b)), cv2.COLOR_LAB2BGR)


def _luma_stats(img: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def detect_lines(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med) + 30))
    h, w = gray.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=int(0.12 * min(h, w)), maxLineGap=12)
    return np.empty((0, 4), dtype=np.int32) if lines is None else lines.reshape(-1, 4)


def analyze_geometry(lines: np.ndarray, h: int, w: int) -> dict:
    """Estimate floor boundary, wall corner and camera tilt from line segments."""
    floor_y = corner_x = None
    tilt_deg = 0.0
    if len(lines):
        x1, y1, x2, y2 = lines.T.astype(np.float64)
        length = np.hypot(x2 - x1, y2 - y1)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = (angle + 90) % 180 - 90  # fold into [-90, 90)

        horiz = np.abs(angle) < 25
        vert = np.abs(np.abs(angle) - 90) < 8
        mid_y = (y1 + y2) / 2
        mid_x = (x1 + x2) / 2

        # Floor boundary: longest near-horizontal structure in the lower 70% of the frame.
        floor_cand = horiz & (mid_y > 0.3 * h)
        if floor_cand.any():
            weights = length[floor_cand]
            ys = mid_y[floor_cand]
            # Length-weighted choice favouring the strongest long line band.
            idx = np.argmax(weights * (0.5 + ys / h))
            floor_y = float(ys[idx])

        # Wall corner: strongest long vertical segment away from the image border.
        corner_cand = vert & (mid_x > 0.08 * w) & (mid_x < 0.92 * w) & (length > 0.25 * h)
        if corner_cand.any():
            corner_x = float(mid_x[corner_cand][np.argmax(length[corner_cand])])

        # Tilt: robust (median) deviation of wall verticals from true vertical. Only segments
        # above the floor boundary count; floor seams and table legs are perspective lines.
        wall_vert = vert & (mid_y < (floor_y if floor_y is not None else 0.8 * h))
        if wall_vert.any():
            dev = np.where(angle[wall_vert] > 0, angle[wall_vert] - 90, angle[wall_vert] + 90)
            reps = np.repeat(dev, np.maximum(1, (length[wall_vert] / 20).astype(int)))
            tilt_deg = float(np.median(reps))

    return {
        "floor_line_y_ratio": None if floor_y is None else round(floor_y / h, 4),
        "corner_x_ratio": None if corner_x is None else round(corner_x / w, 4),
        "perspective_tilt_deg": round(tilt_deg, 2),
    }


def correct_tilt(img: np.ndarray, tilt_deg: float) -> np.ndarray:
    """Rotate so wall verticals are vertical (small-angle perspective normalization)."""
    if abs(tilt_deg) < 0.5 or abs(tilt_deg) > 15:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), tilt_deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def build_wall_mask(h: int, w: int, floor_y_ratio: float | None) -> np.ndarray:
    """Wall ROI: everything above the floor boundary, minus a ceiling band."""
    mask = np.zeros((h, w), dtype=np.uint8)
    bottom = int((floor_y_ratio if floor_y_ratio is not None else 0.8) * h)
    top = int(0.05 * h)
    if bottom - top < 0.2 * h:  # implausible boundary; fall back to central band
        top, bottom = int(0.1 * h), int(0.8 * h)
    mask[top:bottom, :] = 255
    return mask


def dominant_colors(img_bgr: np.ndarray, mask: np.ndarray, k: int = 4) -> list[tuple[int, int, int]]:
    """k-means dominant colours (RGB), sorted by pixel share."""
    pixels = img_bgr[mask > 0].reshape(-1, 3).astype(np.float32)
    if len(pixels) == 0:
        pixels = img_bgr.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 20_000:
        rng = np.random.default_rng(0)
        pixels = pixels[rng.choice(len(pixels), 20_000, replace=False)]
    k = max(1, min(k, len(np.unique(pixels, axis=0))))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    order = np.argsort(-counts)
    return [tuple(int(c) for c in centers[i][::-1]) for i in order]  # BGR -> RGB


def _lab(rgb: tuple[int, int, int]) -> np.ndarray:
    # float32 input in [0, 1] yields true CIELAB (L* 0-100), unlike the uint8 rescaled variant.
    px = np.float32([[list(rgb[::-1])]]) / 255.0
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0]


def nearest_finishes(colors: list[tuple[int, int, int]], top_n: int = 3) -> list[str]:
    """Rank catalog finishes by perceptual (CIELAB) distance to the dominant colours."""
    swatch_lab = {name: _lab(rgb) for name, rgb in FINISH_SWATCHES.items()}
    scores: dict[str, float] = {name: 0.0 for name in FINISH_SWATCHES}
    for rank, color in enumerate(colors):
        lab = _lab(color)
        # Painted walls are mostly neutral; saturated regions (wood, painted fronts) say more
        # about the intended finish, so chroma boosts a colour's vote.
        chroma = float(np.hypot(lab[1], lab[2]))
        weight = (0.35 + min(chroma, 30.0) / 30.0) / (rank + 1)
        for name, ref in swatch_lab.items():
            scores[name] += weight * math.exp(-float(np.linalg.norm(lab - ref)) / 25.0)
    return [n for n, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]]


def preprocess(data: bytes) -> PreprocessResult:
    raw = resize_max_side(decode_image(data))
    lum_before, con_before = _luma_stats(raw)

    contrast = apply_clahe(raw)
    h, w = contrast.shape[:2]

    geometry = analyze_geometry(detect_lines(contrast), h, w)
    normalized = correct_tilt(gray_world_balance(contrast), geometry["perspective_tilt_deg"])
    lum_after, con_after = _luma_stats(normalized)

    mask = build_wall_mask(h, w, geometry["floor_line_y_ratio"])
    # Colour analysis uses the un-balanced image: finishes are matched to what the room looks like.
    colors = dominant_colors(correct_tilt(contrast, geometry["perspective_tilt_deg"]), mask, k=5)

    warnings = []
    if lum_before < 50:
        warnings.append("Photo is very dark; colour-based finish suggestions may be unreliable.")
    if con_before < 18:
        warnings.append("Low contrast photo; wall/floor boundary detection may be unreliable.")
    if geometry["floor_line_y_ratio"] is None:
        warnings.append("No wall/floor boundary found; using default wall region.")
    if abs(geometry["perspective_tilt_deg"]) > 15:
        warnings.append("Strong camera tilt detected; retake the photo facing the wall squarely.")

    analysis = {
        "width_px": w,
        "height_px": h,
        "mean_luminance_before": round(lum_before, 2),
        "mean_luminance_after": round(lum_after, 2),
        "contrast_before": round(con_before, 2),
        "contrast_after": round(con_after, 2),
        **geometry,
        "wall_roi_ratio": round(float((mask > 0).mean()), 4),
        "dominant_colors": ["#{:02x}{:02x}{:02x}".format(*c) for c in colors],
        "suggested_finishes": nearest_finishes(colors),
        "warnings": warnings,
    }
    rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    faithful = cv2.cvtColor(correct_tilt(raw, geometry["perspective_tilt_deg"]), cv2.COLOR_BGR2RGB)
    return PreprocessResult(image_rgb=rgb, embedding_rgb=pad_to_square(faithful), wall_mask=mask, analysis=analysis)


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Letterbox to a square so CLIP's centre crop keeps the whole wall in view.

    Cropping to the wall ROI or colour-normalizing before CLIP both lowered finish
    retrieval precision on the sample set; CLIP expects natural photo colours.
    """
    h, w = img.shape[:2]
    side = max(h, w)
    out = np.empty((side, side, 3), dtype=img.dtype)
    out[:] = img.reshape(-1, 3).mean(axis=0).astype(img.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    out[top : top + h, left : left + w] = img
    return out
