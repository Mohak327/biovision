"""Write the bundled sample images from scikit-image's built-in photographs.

Run once from the repository root: python scripts/make_samples.py
"""
from pathlib import Path

from PIL import Image
from skimage import data

OUT = Path("src/biovision/samples")
SOURCES = {"astronaut": data.astronaut, "cat": data.chelsea, "coffee": data.coffee}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "__init__.py").touch()
for name, load in SOURCES.items():
    pixels = load()
    height, width = pixels.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = Image.fromarray(pixels[top:top + side, left:left + side])
    square.resize((256, 256), Image.Resampling.LANCZOS).save(OUT / f"{name}.png")
    print(f"wrote {OUT / f'{name}.png'}")
