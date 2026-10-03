"""Pointwise map from a signed linear response to a firing rate."""
import numpy as np

from ..core.stage import PointwiseStage


class LinearRectified(PointwiseStage):
    """rate = max(rest_hz * (1 + gain * x), 0), the output stage of an LNP neuron.

    `rest_hz` is the firing rate with no signal. It stands for an ON/OFF pair
    of cells: responses above rest are the ON cell, below rest the OFF cell.
    `gain` scales the response so natural images span the firing range.
    The inverse is exact wherever the rate is above zero.
    """

    def __init__(self, rest_hz: float, gain: float, name: str = "rate"):
        if rest_hz <= 0 or gain <= 0:
            raise ValueError("rest_hz and gain must be positive")
        self.name = name
        self.rest_hz = rest_hz
        self.gain = gain

    def forward(self, x, rng=None):
        return np.maximum(self.rest_hz * (1.0 + self.gain * x), 0.0)

    def inverse(self, y):
        return (np.asarray(y, dtype=float) / self.rest_hz - 1.0) / self.gain
