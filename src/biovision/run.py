"""The single entry point: encode an image with a species and decode it again."""
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import io
from .core.decoder import Decoder, Reconstruction
from .core.field import VisualField
from .core.metrics import psnr, ssim
from .core.pipeline import NeuralCode, Pipeline
from .core.registry import species
from .stages.spiking import PoissonSpikes

LAM_FLOOR = 1e-4  # regularization when there is no noise
NOISE_GAIN = 10.0  # lam = LAM_FLOOR + NOISE_GAIN * (noise variance of the linear drive)
CHROMA_WEIGHT = 0.1


@dataclass(frozen=True)
class Settings:
    species: str
    fov_deg: float
    size_px: int
    window_ms: float
    noise: bool
    lam: float
    chroma_weight: float
    seed: int


@dataclass(frozen=True)
class RunResult:
    original: np.ndarray  # (size, size, 3)
    reconstructed: np.ndarray  # (size, size, 3)
    pipeline: Pipeline
    code: NeuralCode
    reconstruction: Reconstruction
    metrics: dict[str, float]
    settings: Settings
    runtime_s: float


@lru_cache(maxsize=16)
def build_pipeline(species_name: str, size_px: int, fov_deg: float) -> Pipeline:
    """Build (and cache) a species' pipeline. Building the sparse stages is slow."""
    return species.get(species_name)(VisualField(size_px, fov_deg))


def run(image, species_name: str, *, fov_deg: float = 60.0, size_px: int = 128,
        window_ms: float = 100.0, noise: bool = True, lam: float | None = None,
        chroma_weight: float = CHROMA_WEIGHT, seed: int = 0) -> RunResult:
    """Encode `image` through a species' visual system and reconstruct it.

    `image` is any (height, width[, channels]) array; it is centre-cropped and
    resized to `size_px`. With `lam=None` the regularization is set from the
    spike noise.
    """
    if window_ms <= 0:
        raise ValueError(f"window_ms must be positive, got {window_ms}")
    if lam is not None and lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}")
    start = time.perf_counter()
    original = io.to_square(image, size_px)
    pipeline = build_pipeline(species_name, size_px, float(fov_deg))
    pipeline = pipeline.replace(PoissonSpikes(window_ms / 1000.0))
    rng = np.random.default_rng(seed) if noise else None
    code = pipeline.encode(original.transpose(2, 0, 1), rng)
    if lam is None:
        lam = LAM_FLOOR
        if noise:
            lam += NOISE_GAIN * Decoder(pipeline, 1.0).noise_variance(code)
    decoder = Decoder(pipeline, lam, chroma_weight)
    reconstruction = decoder.decode(code)
    reconstructed = reconstruction.image.transpose(1, 2, 0)
    metrics = {
        "psnr_db": psnr(original, reconstructed),
        "ssim": ssim(original, reconstructed),
        "neurons": float(pipeline.n_neurons),
        "compression_ratio": pipeline.n_neurons / original.size,
        "mean_spikes": float(np.mean(code.responses)),
    }
    settings = Settings(species_name, float(fov_deg), size_px, float(window_ms),
                        noise, float(lam), chroma_weight, seed)
    return RunResult(original, reconstructed, pipeline, code, reconstruction,
                     metrics, settings, time.perf_counter() - start)
