"""Photon noise: light arrives as photons, so a receptor's signal is a count."""
import numpy as np

from ..core.stage import LinearStage


class PhotonCatch(LinearStage):
    """Each receptor counts the photons it catches in the spike window.

    `photons` gives, for each receptor, the mean number it catches where the
    picture is white (signal 1). The count is Poisson with mean
    `photons * signal`, and the stage passes on `count / photons`: the signal
    again, with noise of variance `signal / photons`, so the dimmer the light
    the noisier the signal.

    As a linear map the stage is the identity, which is what the decoder
    inverts: the noise has zero mean and is added only while encoding, and
    only when a generator is given.
    """

    def __init__(self, photons, name: str = "photons"):
        self.photons = np.asarray(photons, dtype=float)
        if not np.all(self.photons > 0):
            raise ValueError("photons must be positive for every receptor")
        self.name = name
        self.in_shape = self.out_shape = self.photons.shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return y

    def encode(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            return x
        return rng.poisson(self.photons * np.maximum(x, 0.0)) / self.photons
