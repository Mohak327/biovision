"""Centre-surround receptive fields (difference of Gaussians) over a mosaic."""
import numpy as np

from .mosaic import Mosaic
from .sparse import SparseStage, normalize_rows, pool


def gaussian(dy: np.ndarray, dx: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(dy**2 + dx**2) / (2.0 * sigma**2))


def center_surround(mosaic: Mosaic, sigma_center_px: float, sigma_surround_px: float,
                    surround_weight: float, name: str = "center_surround") -> SparseStage:
    """One cell per receptor: a narrow centre minus a weighted wide surround.

    Centre and surround are each normalized to sum to 1 over the receptors
    they pool, so a uniform image gives a response of 1 - surround_weight.
    """
    pos, types = mosaic.positions, mosaic.types
    centre = normalize_rows(pool(pos, types, pos, types, 3.0 * sigma_center_px,
                                 lambda dy, dx: gaussian(dy, dx, sigma_center_px)))
    surround = normalize_rows(pool(pos, types, pos, types, 3.0 * sigma_surround_px,
                                   lambda dy, dx: gaussian(dy, dx, sigma_surround_px)))
    matrix = (centre - surround_weight * surround).tocsr()
    return SparseStage(name, matrix, (len(mosaic),), (len(mosaic),))
