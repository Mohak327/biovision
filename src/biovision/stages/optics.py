"""Blur by the eye's optics, modelled as a Gaussian point-spread function."""
import numpy as np

from ..core.stage import LinearStage


class OpticalBlur(LinearStage):
    """Gaussian blur of each channel, applied in the frequency domain.

    The transfer function is real and even, so the stage is its own adjoint.
    Edges wrap around (periodic), which keeps the adjoint exact.
    """

    def __init__(self, sigma_px: float, channels: int, size_px: int, name: str = "optics"):
        if sigma_px < 0:
            raise ValueError(f"sigma_px must not be negative, got {sigma_px}")
        self.name = name
        self.sigma_px = sigma_px
        self.in_shape = self.out_shape = (channels, size_px, size_px)
        f = np.fft.fftfreq(size_px)
        f2 = f[:, None] ** 2 + f[None, :] ** 2
        self._transfer = np.exp(-2.0 * np.pi**2 * sigma_px**2 * f2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(np.fft.fft2(x) * self._transfer).real

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return self.forward(y)
