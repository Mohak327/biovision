"""Retinal receptive fields over a mosaic: centre minus surround, per cell class."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import vstack

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


@dataclass(frozen=True)
class RetinaClass:
    """A class of retinal cell: how it weights the receptor types, and its gain.

    `weights` has one entry per receptor type. A class whose weights sum to
    zero ignores uniform grey and carries a colour difference. `gain` scales
    the response so the class fills its firing range. `surround_weight` is the
    strength of the spatial surround.
    """

    name: str
    weights: tuple[float, ...]
    gain: float
    surround_weight: float


def opponent_retina(mosaic: Mosaic, classes, sigma_center_px: float, sigma_surround_px: float,
                    name: str = "center_surround") -> tuple[SparseStage, Mosaic]:
    """Retinal cells that combine receptor types, one of each class at each position.

    For each receptor type, a normalized Gaussian pools that type's receptors
    around every position; a class adds those pools with its weights. The
    response is gain * (centre - surround_weight * surround).

    Returns the stage and the mosaic of the cells it made (their positions,
    with the class index as the type), which later stages pool from.
    """
    classes = tuple(classes)
    if not classes:
        raise ValueError("opponent_retina needs at least one class")
    for item in classes:
        if len(item.weights) != mosaic.n_types:
            raise ValueError(f"class '{item.name}' needs one weight per receptor type "
                             f"({mosaic.n_types}), got {len(item.weights)}")
    # Receptors of several types can share a position; cells sit once at each.
    _, first = np.unique(mosaic.positions, axis=0, return_index=True)
    positions = mosaic.positions[np.sort(first)]

    def pools(sigma):
        """For each receptor type: positions x receptors, rows summing to 1."""
        return [normalize_rows(pool(positions, np.full(len(positions), kind),
                                    mosaic.positions, mosaic.types, 3.0 * sigma,
                                    lambda dy, dx: gaussian(dy, dx, sigma)))
                for kind in range(mosaic.n_types)]

    centres, surrounds = pools(sigma_center_px), pools(sigma_surround_px)
    blocks = []
    for item in classes:
        block = None
        for kind, weight in enumerate(item.weights):
            if weight == 0.0:
                continue
            part = weight * (centres[kind] - item.surround_weight * surrounds[kind])
            block = part if block is None else block + part
        if block is None:
            raise ValueError(f"class '{item.name}' has no non-zero weight")
        blocks.append(item.gain * block)
    matrix = vstack(blocks).tocsr()
    cells = Mosaic(np.tile(positions, (len(classes), 1)),
                   np.repeat(np.arange(len(classes)), len(positions)), len(classes))
    return SparseStage(name, matrix, (len(mosaic),), (len(cells),)), cells
