"""Loading, validating and resizing images. Images are (height, width, 3) in [0, 1]."""
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image

SAMPLE_PACKAGE = f"{__package__}.samples"  # works whatever the package is imported as


def as_rgb(image) -> np.ndarray:
    """Validate an array and return it as float RGB in [0, 1]."""
    array = np.asarray(image)
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        raise ValueError("image must be a numeric array")
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"image must be (height, width) or (height, width, 1|3|4), got shape {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError(f"image must be at least 2 x 2 pixels, got {array.shape[:2]}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    array = array[:, :, :3]  # drop alpha
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(float) / np.iinfo(array.dtype).max
    array = array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError("image contains NaN or infinite values")
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def to_square(image, size_px: int) -> np.ndarray:
    """Centre-crop to a square and resize to (size_px, size_px, 3)."""
    array = as_rgb(image)
    height, width = array.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    crop = array[top:top + side, left:left + side]
    if side == size_px:
        return crop
    pil = Image.fromarray(np.round(crop * 255).astype(np.uint8))
    resized = pil.resize((size_px, size_px), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=float) / 255.0


def load_image(path) -> np.ndarray:
    """Read an image file as float RGB in [0, 1]."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no image at '{path}'")
    try:
        with Image.open(path) as pil:
            return as_rgb(np.asarray(pil.convert("RGB")))
    except OSError as error:
        raise ValueError(f"'{path}' is not a readable image: {error}") from None


def sample_names() -> list[str]:
    files = resources.files(SAMPLE_PACKAGE).iterdir()
    return sorted(f.name.rsplit(".", 1)[0] for f in files if f.name.endswith(".png"))


def load_sample(name: str) -> np.ndarray:
    if name not in sample_names():
        raise KeyError(f"unknown sample '{name}'; available: {', '.join(sample_names())}")
    with resources.as_file(resources.files(SAMPLE_PACKAGE) / f"{name}.png") as path:
        return load_image(path)
