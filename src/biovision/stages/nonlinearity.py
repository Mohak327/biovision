"""Pointwise map from a signed linear response to a firing rate."""
import numpy as np

from ..core.stage import PointwiseStage


class LinearRectified(PointwiseStage):
    """rate = max(rest_hz * (1 + gain * x), 0), the output stage of an LNP neuron.

    `rest_hz` is the firing rate with no signal. It stands for an ON/OFF pair
    of cells: responses above rest are the ON cell, below rest the OFF cell
    (`OnOffPair` models the two cells themselves).
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


class OnOffPair(PointwiseStage):
    """Two rectified cells for each signal: an ON cell and an OFF cell.

    on = spontaneous_hz + swing_hz * gain * max(x, 0), and the OFF cell the same
    for -x. An increment fires the ON cell, a decrement the OFF cell, and
    neither is ever driven below its spontaneous rate. The rates have a last
    axis of two: ON, then OFF.

    `swing_hz` is the rate a cell adds when gain * x reaches 1. The spontaneous
    firing is the same in both cells, so it cancels in on - off, which is the
    signed signal again: the inverse is exact for every x, and it is linear in
    the two rates, so the decoder still solves one linear system. What the
    spontaneous firing costs is noise, since its spikes are counted too.
    """

    def __init__(self, swing_hz: float, gain: float, spontaneous_hz: float = 0.0,
                 name: str = "rate"):
        if swing_hz <= 0 or gain <= 0:
            raise ValueError("swing_hz and gain must be positive")
        if spontaneous_hz < 0:
            raise ValueError(f"spontaneous_hz must not be negative, got {spontaneous_hz}")
        self.name = name
        self.swing_hz = swing_hz
        self.gain = gain
        self.spontaneous_hz = spontaneous_hz

    def forward(self, x, rng=None):
        drive = self.swing_hz * self.gain * np.asarray(x, dtype=float)
        return self.spontaneous_hz + np.maximum(np.stack([drive, -drive], axis=-1), 0.0)

    def inverse(self, y):
        y = np.asarray(y, dtype=float)
        return (y[..., 0] - y[..., 1]) / (self.swing_hz * self.gain)
