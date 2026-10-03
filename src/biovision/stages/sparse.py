"""Linear stages stored as sparse matrices, and helpers to build them."""
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.spatial import cKDTree

from ..core.stage import LinearStage


class SparseStage(LinearStage):
    """A linear stage whose forward map is a sparse matrix on flattened arrays."""

    def __init__(self, name: str, matrix: csr_matrix,
                 in_shape: tuple[int, ...], out_shape: tuple[int, ...]):
        expected = (int(np.prod(out_shape)), int(np.prod(in_shape)))
        if matrix.shape != expected:
            raise ValueError(f"matrix shape {matrix.shape} does not match {expected}")
        self.name = name
        self.matrix = matrix.tocsr()
        self._transpose = self.matrix.T.tocsr()
        self.in_shape = tuple(in_shape)
        self.out_shape = tuple(out_shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (self.matrix @ x.ravel()).reshape(self.out_shape)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return (self._transpose @ y.ravel()).reshape(self.in_shape)


def pool(out_pos: np.ndarray, out_types: np.ndarray,
         in_pos: np.ndarray, in_types: np.ndarray,
         radius: float, kernel: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> csr_matrix:
    """Weights from input cells to output cells of the same type within `radius`.

    Positions are (n, 2) arrays of (row, col) in pixels. `kernel(dy, dx)` gives
    the weight for an input displaced by (dy, dx) from the output cell.
    """
    shape = (len(out_pos), len(in_pos))
    neighbours = cKDTree(in_pos).query_ball_point(out_pos, r=radius)
    counts = np.fromiter((len(n) for n in neighbours), dtype=int, count=len(out_pos))
    if counts.sum() == 0:
        return csr_matrix(shape)
    rows = np.repeat(np.arange(len(out_pos)), counts)
    cols = np.concatenate([np.asarray(n, dtype=int) for n in neighbours])
    same = in_types[cols] == out_types[rows]
    rows, cols = rows[same], cols[same]
    delta = in_pos[cols] - out_pos[rows]
    values = kernel(delta[:, 0], delta[:, 1])
    return csr_matrix((values, (rows, cols)), shape=shape)


def normalize_rows(matrix: csr_matrix) -> csr_matrix:
    """Scale each row so its absolute values sum to 1. Empty rows stay empty."""
    totals = np.asarray(abs(matrix).sum(axis=1)).ravel()
    scale = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    return (diags(scale) @ matrix).tocsr()
