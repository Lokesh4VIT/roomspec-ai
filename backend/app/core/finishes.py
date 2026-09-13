"""Finish colour swatches shared by CV matching, thumbnails and the UI elevation drawing."""

CABINET_SWATCHES: dict[str, tuple[int, int, int]] = {
    "Matte Walnut": (92, 64, 45),
    "Natural Oak": (196, 160, 112),
    "Gloss White": (242, 242, 240),
    "Cream Shaker": (232, 222, 196),
    "Charcoal Grey": (74, 76, 80),
    "Matte Black": (28, 28, 30),
    "Sage Green": (150, 168, 138),
    "Navy Blue": (40, 56, 92),
}

COUNTERTOP_SWATCHES: dict[str, tuple[int, int, int]] = {
    "Quartz White": (236, 236, 232),
    "Butcher Block Oak": (178, 132, 84),
    "Black Granite": (34, 34, 38),
    "Concrete Grey": (140, 140, 136),
}


def swatch_rgb(finish: str) -> tuple[int, int, int]:
    return CABINET_SWATCHES.get(finish) or COUNTERTOP_SWATCHES.get(finish) or (180, 180, 180)


def swatch_hex(finish: str) -> str:
    return "#{:02x}{:02x}{:02x}".format(*swatch_rgb(finish))
