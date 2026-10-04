"""A bank of Gabor receptive fields (V1 simple cells) over a mosaic."""
import numpy as np
from scipy.sparse import vstack

from .mosaic import Mosaic, all_types_at, square_lattice
from .receptive import gaussian
from .sparse import SparseStage, normalize_rows, pool


def gabor_kernel(dy: np.ndarray, dx: np.ndarray, sigma: float, wavelength: float,
                 theta: float, phase: float) -> np.ndarray:
    """A Gaussian envelope times a cosine grating at orientation `theta`."""
    along = dx * np.cos(theta) + dy * np.sin(theta)
    return gaussian(dy, dx, sigma) * np.cos(2.0 * np.pi * along / wavelength + phase)


def gabor_bank(mosaic: Mosaic, size_px: int, wavelengths_px, n_orientations: int = 4,
               sigma_ratio: float = 0.4, min_spacing_px: float = 1.0,
               density: float = 1.0, gains=None, name: str = "gabor") -> SparseStage:
    """Simple cells at several scales, orientations and two phases, per type.

    For each wavelength, cells sit on a square grid one envelope sigma apart.
    The coarsest scale also gets one non-oriented Gaussian cell per position,
    which carries the mean level that the oriented cells barely respond to.

    `density` multiplies the number of cells per unit area, so the grid
    spacing shrinks by its square root. `gains` gives one gain per wavelength,
    in the order the wavelengths are given; finer scales usually need more,
    because natural images have less contrast there.
    """
    wavelengths_px = list(wavelengths_px)
    gains = [1.0] * len(wavelengths_px) if gains is None else list(gains)
    if len(gains) != len(wavelengths_px):
        raise ValueError(f"gabor_bank needs one gain per wavelength "
                         f"({len(wavelengths_px)}), got {len(gains)}")
    gain_of = dict(zip(wavelengths_px, gains))
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")
    wavelengths = sorted(wavelengths_px, reverse=True)
    thetas = np.pi * np.arange(n_orientations) / n_orientations
    blocks = []
    for index, wavelength in enumerate(wavelengths):
        sigma = sigma_ratio * wavelength
        spacing = max(sigma, min_spacing_px) / np.sqrt(density)
        cells = all_types_at(square_lattice(size_px, spacing), mosaic.n_types)
        radius = 2.5 * sigma

        def build(kernel):
            return gain_of[wavelength] * normalize_rows(pool(
                cells.positions, cells.types, mosaic.positions, mosaic.types, radius, kernel))

        if index == 0:
            blocks.append(build(lambda dy, dx: gaussian(dy, dx, sigma)))
        for theta in thetas:
            for phase in (0.0, np.pi / 2.0):
                blocks.append(build(
                    lambda dy, dx: gabor_kernel(dy, dx, sigma, wavelength, theta, phase)))
    matrix = vstack(blocks).tocsr()
    return SparseStage(name, matrix, (len(mosaic),), (matrix.shape[0],))
