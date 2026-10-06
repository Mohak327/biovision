"""Spike counts in a time window."""
import numpy as np

from ..core.stage import PointwiseStage


class PoissonSpikes(PointwiseStage):
    """Spike counts in a window. Without an rng, returns the mean exactly.

    With `fano` 1 (the default), counts ~ Poisson(rate * window): the variance
    of a count equals its mean.

    A real cell cannot fire twice within its refractory period, so its counts
    are more regular than that. `fano` below 1 is the ratio of a count's
    variance to its mean. The window is cut into `mean / (1 - fano)` slots,
    rounded up to a whole number n, each of which holds one spike with
    probability mean / n: counts ~ Binomial(n, mean / n). That is the simplest
    whole-number count with a ceiling, it comes from the same seeded generator,
    and its mean is exactly the Poisson mean. Its variance is
    mean * (1 - mean / n): `fano * mean` where the slots come out whole, and
    otherwise above that by less than (1 - fano)^2 of a spike. A cell that
    expects less than 1 - fano spikes has one slot and its variance is
    mean * (1 - mean), the least any whole-number count can have: at low rates
    a refractory period changes nothing, and firing is as irregular as Poisson.
    """

    def __init__(self, window_s: float, fano: float = 1.0, name: str = "spikes"):
        if window_s <= 0:
            raise ValueError(f"window_s must be positive, got {window_s}")
        if not 0.0 < fano <= 1.0:
            raise ValueError(f"fano must be above 0 and at most 1, got {fano}")
        self.name = name
        self.window_s = window_s
        self.fano = fano

    def _slots(self, mean):
        return np.maximum(np.ceil(mean / (1.0 - self.fano)), 1.0)

    def forward(self, x, rng=None):
        mean = np.maximum(x, 0.0) * self.window_s
        if rng is None:
            return mean
        if self.fano == 1.0:
            return rng.poisson(mean).astype(float)
        slots = self._slots(mean)
        return rng.binomial(slots.astype(np.int64), mean / slots).astype(float)

    def variance(self, counts):
        """An estimate of each count's variance, from the count itself.

        The count stands in for its mean, as it does for a Poisson count,
        which is its own variance. It reads a lone spike as a cell expecting
        one, so where cells expect far less than a spike it comes out low, by
        up to the factor `fano`.
        """
        if self.fano == 1.0:
            return counts
        return counts * (1.0 - counts / self._slots(counts))

    def lasting(self, window_s: float) -> "PoissonSpikes":
        """The same cells counted over another window."""
        return PoissonSpikes(window_s, self.fano, self.name)

    def inverse(self, y):
        return np.asarray(y, dtype=float) / self.window_s
