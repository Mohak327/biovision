"""Spike counts in a time window."""
import numpy as np

from ..core.stage import PointwiseStage


class PoissonSpikes(PointwiseStage):
    """counts ~ Poisson(rate * window). Without an rng, returns the mean exactly."""

    def __init__(self, window_s: float, name: str = "spikes"):
        if window_s <= 0:
            raise ValueError(f"window_s must be positive, got {window_s}")
        self.name = name
        self.window_s = window_s

    def forward(self, x, rng=None):
        mean = np.maximum(x, 0.0) * self.window_s
        if rng is None:
            return mean
        return rng.poisson(mean).astype(float)

    def inverse(self, y):
        return np.asarray(y, dtype=float) / self.window_s
