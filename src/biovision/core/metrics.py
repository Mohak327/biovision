"""Image quality measures. Images are (height, width, 3) in [0, 1]."""
import numpy as np
from skimage.metrics import structural_similarity


def psnr(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """Peak signal-to-noise ratio in decibels. Infinite for identical images."""
    mse = float(np.mean((np.asarray(original) - np.asarray(reconstruction)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def ssim(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """Structural similarity, between -1 and 1."""
    return float(structural_similarity(original, reconstruction, channel_axis=-1, data_range=1.0))


def radial_power_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean power at each spatial frequency (cycles per image), averaged over angle.

    Returns (frequencies, power) for the luminance of `image`.
    """
    luminance = np.asarray(image).mean(axis=-1)
    size = luminance.shape[0]
    power = np.abs(np.fft.fftshift(np.fft.fft2(luminance - luminance.mean()))) ** 2
    rows, cols = np.indices(power.shape)
    radius = np.hypot(rows - size // 2, cols - size // 2).astype(int)
    limit = size // 2
    totals = np.bincount(radius.ravel(), weights=power.ravel())[:limit]
    counts = np.bincount(radius.ravel())[:limit]
    return np.arange(limit, dtype=float), totals / np.maximum(counts, 1)
