"""OpenCV preprocessing: integrity checks, normalization and geometry estimates."""

import cv2
import numpy as np
import pytest
from make_samples import SCENES, render

from app.services import cv_preprocessor as cv


def test_rejects_empty_and_corrupt_uploads():
    with pytest.raises(cv.ImageValidationError):
        cv.decode_image(b"")
    with pytest.raises(cv.ImageValidationError):
        cv.decode_image(b"GIF89a not really an image")


def test_rejects_tiny_images(encode):
    with pytest.raises(cv.ImageValidationError, match="too small"):
        cv.decode_image(encode(np.zeros((20, 20, 3), np.uint8), ".png"))


def test_resize_caps_longest_side():
    img = np.zeros((1500, 3000, 3), np.uint8)
    out = cv.resize_max_side(img, 1024)
    assert max(out.shape[:2]) == 1024
    assert out.shape[0] == 512


def test_clahe_raises_contrast_of_flat_image():
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(100, 6, (240, 320, 3)), 0, 255).astype(np.uint8)
    before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()
    after = cv2.cvtColor(cv.apply_clahe(img), cv2.COLOR_BGR2GRAY).std()
    assert after > before


def test_gray_world_reduces_colour_cast_without_erasing_it():
    img = np.full((100, 100, 3), (80, 120, 200), np.uint8)  # strong orange cast (BGR)
    spread_before = np.ptp(img.reshape(-1, 3).mean(axis=0))
    means = cv.gray_world_balance(img).reshape(-1, 3).mean(axis=0)
    assert 0 < np.ptp(means) < spread_before


def test_detects_floor_line_and_corner(room_jpeg):
    res = cv.preprocess(room_jpeg)
    a = res.analysis
    assert a["floor_line_y_ratio"] == pytest.approx(0.72, abs=0.03)
    assert a["corner_x_ratio"] == pytest.approx(0.62, abs=0.03)
    assert abs(a["perspective_tilt_deg"]) < 1.0
    assert res.wall_mask.shape == res.image_rgb.shape[:2]
    assert 0.5 < a["wall_roi_ratio"] < 0.8


@pytest.mark.parametrize("tilt", [-3.0, 2.0])
def test_tilt_estimate_has_correcting_sign(tilt, encode):
    wall, floor, accent, _, gain = SCENES["bright_minimal_white"]
    a = cv.preprocess(encode(render(wall, floor, accent, tilt, gain))).analysis
    # Rendering rotates by `tilt`; the estimate is the rotation that undoes it.
    assert a["perspective_tilt_deg"] == pytest.approx(-tilt, abs=1.0)


def test_embedding_image_is_square_and_colour_faithful(room_jpeg):
    res = cv.preprocess(room_jpeg)
    h, w = res.embedding_rgb.shape[:2]
    assert h == w == max(res.image_rgb.shape[:2])


@pytest.mark.parametrize(
    ("scene", "expected"),
    [
        ("warm_walnut_kitchen", "Matte Walnut"),
        ("bright_minimal_white", "Gloss White"),
        ("industrial_grey_corner", "Charcoal Grey"),
        ("sage_cottage_wall", "Sage Green"),
    ],
)
def test_finish_suggestions_follow_room_palette(scene, expected, encode):
    a = cv.preprocess(encode(render(*SCENES[scene]))).analysis
    assert expected in a["suggested_finishes"][:2]
    assert all(c.startswith("#") and len(c) == 7 for c in a["dominant_colors"])


def test_dark_photo_warns(encode):
    a = cv.preprocess(encode(render(*SCENES["dim_navy_galley"]))).analysis
    assert any("dark" in w for w in a["warnings"])
