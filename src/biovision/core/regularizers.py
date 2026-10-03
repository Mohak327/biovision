"""Smoothness penalty used by the decoder."""
import numpy as np


def laplacian(x: np.ndarray) -> np.ndarray:
    """Discrete Laplacian over the last two axes, periodic edges. Self-adjoint."""
    return (
        np.roll(x, 1, axis=-1) + np.roll(x, -1, axis=-1)
        + np.roll(x, 1, axis=-2) + np.roll(x, -1, axis=-2)
        - 4.0 * x
    )


def chroma(x: np.ndarray) -> np.ndarray:
    """Each channel minus the mean over channels. A self-adjoint projection."""
    return x - x.mean(axis=0, keepdims=True)
