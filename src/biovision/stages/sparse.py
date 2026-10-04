"""Linear stages stored as sparse matrices, and helpers to build them."""
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix, diags, vstack
from scipy.spatial import cKDTree

from ..core.stage import LinearStage

CHUNK_PAIRS = 2_000_000  # neighbour pairs handled at once while building a pool


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


class FactoredStage(LinearStage):
    """A linear stage that is a sum of products of sparse matrices.

    `terms` is a list of chains. A chain is a list of matrices applied first
    to last; the stage's output is the sum of the chains' outputs. Keeping the
    factors apart is what makes a wide receptive field cheap: a matrix to a
    coarse layer and a matrix from it are far smaller than their product.
    """

    def __init__(self, name: str, terms, in_shape: tuple[int, ...], out_shape: tuple[int, ...]):
        terms = [list(chain) for chain in terms]
        if not terms or not all(terms):
            raise ValueError("a factored stage needs at least one matrix in each term")
        for chain in terms:
            widths = [int(np.prod(in_shape))] + [m.shape[0] for m in chain]
            for matrix, width in zip(chain, widths):
                if matrix.shape[1] != width:
                    raise ValueError(f"matrix shape {matrix.shape} does not match {width} inputs")
            if widths[-1] != int(np.prod(out_shape)):
                raise ValueError(f"matrix shape {chain[-1].shape} does not match "
                                 f"{int(np.prod(out_shape))} outputs")
        self.name = name
        self.terms = [[m.tocsr() for m in chain] for chain in terms]
        self.in_shape = tuple(in_shape)
        self.out_shape = tuple(out_shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        total = 0.0
        for chain in self.terms:
            v = x.ravel()
            for matrix in chain:
                v = matrix @ v
            total = total + v
        return total.reshape(self.out_shape)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        total = 0.0
        for chain in self.terms:
            v = y.ravel()
            for matrix in reversed(chain):
                v = matrix.T @ v  # a view of the same numbers, so the transpose is exact
            total = total + v
        return total.reshape(self.in_shape)


def gaussian(dy: np.ndarray, dx: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(dy**2 + dx**2) / (2.0 * sigma**2))


def pool(out_pos: np.ndarray, out_types: np.ndarray,
         in_pos: np.ndarray, in_types: np.ndarray,
         radius: float, kernel: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> csr_matrix:
    """Weights from input cells to output cells of the same type within `radius`.

    Positions are (n, 2) arrays of (row, col) in pixels. `kernel(dy, dx)` gives
    the weight for an input displaced by (dy, dx) from the output cell.
    """
    tree = cKDTree(in_pos)
    counts = tree.query_ball_point(out_pos, r=radius, return_length=True)
    if counts.sum() == 0:
        return csr_matrix((len(out_pos), len(in_pos)))
    # Built a slice of output cells at a time, so memory follows the result
    # and not the lists of neighbours.
    edges = np.searchsorted(np.cumsum(counts), np.arange(0, counts.sum(), CHUNK_PAIRS))
    edges = np.unique(np.append(edges, len(out_pos)))
    parts = []
    for start, stop in zip(edges[:-1], edges[1:]):
        neighbours = tree.query_ball_point(out_pos[start:stop], r=radius)
        rows = np.repeat(np.arange(stop - start), counts[start:stop])
        cols = np.concatenate([np.asarray(n, dtype=int) for n in neighbours])
        same = in_types[cols] == out_types[start + rows]
        rows, cols = rows[same], cols[same]
        delta = in_pos[cols] - out_pos[start + rows]
        values = kernel(delta[:, 0], delta[:, 1])
        parts.append(csr_matrix((values, (rows, cols)), shape=(stop - start, len(in_pos))))
    return vstack(parts).tocsr()


def normalize_rows(matrix: csr_matrix) -> csr_matrix:
    """Scale each row so its absolute values sum to 1. Empty rows stay empty."""
    totals = np.asarray(abs(matrix).sum(axis=1)).ravel()
    scale = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    return (diags(scale) @ matrix).tocsr()
