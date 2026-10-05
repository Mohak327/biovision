"""Blur by the eye's optics, modelled as a Gaussian point-spread function."""
import numpy as np

from ..core.stage import LinearStage


def defocus_sigma_deg(pupil_mm: float, defocus_d: float) -> float:
    """The blur, as a Gaussian sigma in degrees, of light that is out of focus.

    A point `defocus_d` dioptres out of focus is spread over a circle whose
    angular diameter is the pupil's diameter in metres times the defocus (in
    radians, by geometric optics). A uniform disc has a standard deviation of
    a quarter of its diameter along each axis; the Gaussian with that sigma
    stands in for it. The blur grows in proportion to the pupil.
    """
    if pupil_mm < 0:
        raise ValueError(f"pupil_mm must not be negative, got {pupil_mm}")
    return float(np.degrees(pupil_mm * 1e-3 * abs(defocus_d)) / 4.0)


class OpticalBlur(LinearStage):
    """Gaussian blur of each channel, applied in the frequency domain.

    `sigma_px` is one sigma for every channel, or one per channel: an eye
    cannot focus every wavelength at once (chromatic aberration), so its
    receptor types see different blurs.

    The transfer function is real and even, so the stage is its own adjoint.
    Edges wrap around (periodic), which keeps the adjoint exact.
    """

    def __init__(self, sigma_px, channels: int, size_px: int, name: str = "optics"):
        sigma = np.asarray(sigma_px, dtype=float)
        if sigma.ndim > 1 or (sigma.ndim == 1 and len(sigma) != channels) or np.any(sigma < 0):
            raise ValueError(f"sigma_px must be one sigma that is not negative, or one for each "
                             f"of the {channels} channels, got {sigma_px}")
        self.name = name
        self.sigma_px = sigma_px
        self.in_shape = self.out_shape = (channels, size_px, size_px)
        f = np.fft.fftfreq(size_px)
        f2 = f[:, None] ** 2 + f[None, :] ** 2
        if sigma.ndim == 1:
            sigma = sigma[:, None, None]
        self._transfer = np.exp(-2.0 * np.pi**2 * sigma**2 * f2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(np.fft.fft2(x) * self._transfer).real

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return self.forward(y)
