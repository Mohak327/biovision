"""Multi-scale pooling: a wide cell reads a coarse layer that has summarized the fine one.

A cell with a large receptive field does not connect to every receptor under
it. It pools from cells that have each already averaged a small patch, as
retinal and cortical cells do. The number of inputs per cell then depends on
the shape of its receptive field and not on the number of pixels.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from .mosaic import Mosaic, all_types_at
from .sparse import gaussian, normalize_rows, pool

# Neighbour pairs (output cell, input cell within the radius, of any type) above
# which a pool goes through a coarse layer. Below it the direct matrix is cheap.
DIRECT_LIMIT = 5_000_000
POOL_SIGMA_RATIO = 0.5  # pooling Gaussian of a coarse cell, as a fraction of the grid spacing


def connections(out_pos: np.ndarray, in_pos: np.ndarray, radius: float) -> int:
    """How many (output, input) pairs lie within `radius` of each other."""
    return int(cKDTree(in_pos).query_ball_point(out_pos, r=radius, return_length=True).sum())


def coarse_layer(mosaic: Mosaic, spacing_px: float) -> tuple[Mosaic, csr_matrix]:
    """Pooling cells on a square grid over `mosaic`, one of each type at each node.

    Each takes the Gaussian-weighted mean of the cells of its own type around
    it, which removes detail finer than the grid before the grid samples it.
    Returns the pooling cells and the matrix that computes them.
    """
    low, high = mosaic.positions.min(axis=0), mosaic.positions.max(axis=0)
    axes = []
    for a, b in zip(low, high):
        n = int(np.floor((b - a) / spacing_px)) + 1
        axes.append(a + (b - a - (n - 1) * spacing_px) / 2.0 + np.arange(n) * spacing_px)
    rows, cols = np.meshgrid(*axes, indexing="ij")
    nodes = all_types_at(np.column_stack([rows.ravel(), cols.ravel()]), mosaic.n_types)
    sigma = POOL_SIGMA_RATIO * spacing_px
    matrix = normalize_rows(pool(nodes.positions, nodes.types, mosaic.positions, mosaic.types,
                                 3.0 * sigma, lambda dy, dx: gaussian(dy, dx, sigma)))
    return nodes, matrix


def summarize(out_pos: np.ndarray, mosaic: Mosaic, radius: float,
              spacing_px: float) -> tuple[Mosaic, list[csr_matrix]]:
    """The layer that cells at `out_pos` should pool from, and the matrices that make it.

    If connecting them directly to every cell of `mosaic` within `radius` is
    affordable, that is `mosaic` itself and no matrices. Otherwise it is a
    coarse layer `spacing_px` apart, provided that layer has at most half as
    many cells as `mosaic`; a grid nearly as fine as its input saves nothing
    and only blurs.
    """
    if connections(out_pos, mosaic.positions, radius) <= DIRECT_LIMIT:
        return mosaic, []
    nodes, matrix = coarse_layer(mosaic, spacing_px)
    if 2 * len(nodes) > len(mosaic):
        return mosaic, []
    return nodes, [matrix]
