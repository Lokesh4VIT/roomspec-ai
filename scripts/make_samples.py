"""Render synthetic room photos into data/samples/ for demos and tests.

Each scene has a wall, a floor boundary, a wall corner, some wooden or painted
surfaces and mild lighting gradient + sensor noise so the CV pipeline has real work to do.
"""

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"

SCENES = {
    # name: (wall RGB, floor RGB, accent RGB, tilt degrees, brightness gain)
    "warm_walnut_kitchen": ((210, 196, 176), (120, 84, 56), (92, 64, 45), 0.0, 0.9),
    "bright_minimal_white": ((236, 236, 232), (190, 186, 180), (242, 242, 240), 2.0, 1.05),
    "industrial_grey_corner": ((110, 112, 116), (70, 70, 72), (74, 76, 80), -3.0, 0.8),
    "sage_cottage_wall": ((196, 204, 182), (168, 136, 100), (150, 168, 138), 1.0, 0.95),
    "dim_navy_galley": ((70, 80, 104), (52, 46, 42), (40, 56, 92), 0.0, 0.55),
}


def render(wall, floor, accent, tilt, gain, w=960, h=720, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    floor_y = int(h * 0.72)
    corner_x = int(w * 0.62)
    bgr = lambda c: np.array(c[::-1], np.float32)  # noqa: E731

    img[:floor_y] = bgr(wall)
    img[:floor_y, corner_x:] = bgr(wall) * 0.82  # side wall in shade
    img[floor_y:] = bgr(floor)
    for x in range(0, w, 90):  # floor plank seams
        cv2.line(img, (x, floor_y), (x + 40, h), (bgr(floor) * 0.8).tolist(), 2)

    # Accent surfaces: an open shelf and a picture/cabinet block in the accent finish.
    cv2.rectangle(img, (int(w * 0.08), int(h * 0.18)), (int(w * 0.5), int(h * 0.24)), bgr(accent).tolist(), -1)
    # Existing lower cabinet run along the back wall.
    cv2.rectangle(img, (int(w * 0.04), int(h * 0.46)), (int(w * 0.58), floor_y), bgr(accent).tolist(), -1)
    for x in np.linspace(w * 0.04, w * 0.58, 5)[1:-1]:
        cv2.line(img, (int(x), int(h * 0.46)), (int(x), floor_y), (bgr(accent) * 0.7).tolist(), 2)
    cv2.line(img, (corner_x, 0), (corner_x, floor_y), (bgr(wall) * 0.6).tolist(), 3)
    cv2.line(img, (0, floor_y), (w, floor_y), (bgr(floor) * 0.55).tolist(), 4)

    # Window light gradient + noise.
    xs = np.linspace(1.15, 0.8, w, dtype=np.float32)[None, :, None]
    img = img * xs * gain + rng.normal(0, 4, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if tilt:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), tilt, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (name, params) in enumerate(SCENES.items()):
        cv2.imwrite(str(OUT / f"{name}.jpg"), render(*params, seed=i), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {len(SCENES)} samples to {OUT}")


if __name__ == "__main__":
    main()
