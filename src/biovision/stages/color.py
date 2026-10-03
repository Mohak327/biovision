"""Projection of RGB onto a species' photoreceptor types."""
import numpy as np

from ..core.stage import LinearStage


class ColorProjection(LinearStage):
    """Applies a (types x 3) matrix to the channel axis of an RGB image."""

    def __init__(self, matrix, size_px: int, name: str = "color"):
        self.matrix = np.asarray(matrix, dtype=float)
        if self.matrix.ndim != 2 or self.matrix.shape[1] != 3:
            raise ValueError(f"matrix must be (types, 3), got {self.matrix.shape}")
        self.name = name
        self.in_shape = (3, size_px, size_px)
        self.out_shape = (self.matrix.shape[0], size_px, size_px)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tensordot(self.matrix, x, axes=1)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return np.tensordot(self.matrix.T, y, axes=1)
