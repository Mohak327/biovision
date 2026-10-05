"""Fixational eye movements: several looks at one picture, each a little shifted."""
import numpy as np

from ..core.stage import LinearStage

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def look_offsets(looks: int, amplitude_px: float) -> np.ndarray:
    """Where the eye points at each look: (looks, 2) offsets as (row, col), in pixels.

    The offsets fill a disc of radius `amplitude_px` evenly (a sunflower
    spiral: equal areas, successive looks a golden angle apart). They are the
    same on every run, so a result depends only on its settings. Measured
    against a ring and against random positions: the same within 0.05 dB for
    the fly, and unlike a ring it does not put two looks on one line.
    """
    k = np.arange(looks)
    radius = amplitude_px * np.sqrt((k + 0.5) / looks)
    angle = k * GOLDEN_ANGLE
    return np.stack([radius * np.sin(angle), radius * np.cos(angle)], axis=1)


class EyeShifts(LinearStage):
    """One picture to several looks, each shifted by its own (row, col) offset.

    A shift is a phase ramp in the frequency domain, so it is exact for any
    fraction of a pixel and equals `np.roll` for whole pixels. Edges wrap
    around (periodic), as in `OpticalBlur`. The adjoint shifts every look back
    and adds them up.
    """

    def __init__(self, offsets_px, channels: int, size_px: int, name: str = "fixation"):
        offsets_px = np.asarray(offsets_px, dtype=float)
        if offsets_px.ndim != 2 or offsets_px.shape[1] != 2 or len(offsets_px) == 0:
            raise ValueError("offsets_px must be a non-empty list of (row, col) offsets")
        self.name = name
        self.offsets_px = offsets_px
        self.in_shape = (channels, size_px, size_px)
        self.out_shape = (len(offsets_px), channels, size_px, size_px)
        f = np.fft.fftfreq(size_px)
        cycles = (offsets_px[:, 0, None, None] * f[:, None]
                  + offsets_px[:, 1, None, None] * f[None, :])
        self._transfer = np.exp(-2j * np.pi * cycles)[:, None]  # (looks, 1, size, size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(np.fft.fft2(x) * self._transfer).real

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(np.fft.fft2(y) * self._transfer.conj()).real.sum(axis=0)


class PerLook(LinearStage):
    """One stage applied to each look in turn. The stage is shared, not copied."""

    def __init__(self, stage: LinearStage, looks: int):
        self.name = stage.name
        self.stage = stage
        self.in_shape = (looks, *stage.in_shape)
        self.out_shape = (looks, *stage.out_shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.stack([self.stage.forward(look) for look in x])

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return np.stack([self.stage.adjoint(look) for look in y])

    def encode(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return np.stack([self.stage.encode(look, rng) for look in x])
