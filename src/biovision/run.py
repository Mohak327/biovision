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
from .species.eye import fixate, lit
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
    density: float
    neuron_density: float
    looks: int = 1
    photons_per_s: float | None = None


@dataclass(frozen=True)
class Progress:
    """One step of a run, for a live display.

    `stages` names every step in order, ending with "decoding"; `current`
    indexes the step just reached. `image` is (size, size, 3) in [0, 1]: the
    stage's output where that is an image, the current estimate while
    decoding, otherwise None. `iteration` is 0 until decoding starts.
    """

    stages: tuple[str, ...]
    current: int
    image: np.ndarray | None
    iteration: int


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
def build_pipeline(species_name: str, size_px: int, fov_deg: float,
                   density: float = 1.0, neuron_density: float = 1.0) -> Pipeline:
    """Build (and cache) a species' pipeline. Building the sparse stages is slow."""
    return species.get(species_name)(VisualField(size_px, fov_deg), density, neuron_density)


def _first_look(intermediates: dict[str, np.ndarray], looks: int) -> dict[str, np.ndarray]:
    return intermediates if looks == 1 else {name: output[0]
                                             for name, output in intermediates.items()}


def stage_outputs(result: RunResult) -> dict[str, np.ndarray]:
    """Each stage's output, for drawing. With several looks, those of the first look."""
    return _first_look(result.code.intermediates, result.settings.looks)


def spike_counts(result: RunResult) -> np.ndarray:
    """Each neuron's spikes over the whole spike window: the sum over its looks."""
    responses = result.code.responses
    return responses if result.settings.looks == 1 else responses.sum(axis=0)


def _photon_noise_variance(pipeline: Pipeline, code: NeuralCode,
                           rng: np.random.Generator) -> float:
    """Mean variance that photon noise adds to the linear drive.

    Estimated from one fresh draw of the receptors' photon noise, passed
    through the stages after them. Over thousands of neurons one draw is
    enough, and it tells the decoder how large the noise is, not what it was.
    """
    stages = pipeline.linear_stages
    index = [stage.name for stage in stages].index("photons")
    signal = code.intermediates[stages[index - 1].name]
    noise = stages[index].encode(signal, rng) - signal
    for stage in stages[index + 1:]:
        noise = stage.forward(noise)
    return float(np.mean(noise**2))


def _stage_image(output: np.ndarray) -> np.ndarray | None:
    """A stage output as a (size, size, 3) image, or None if it is not image-like."""
    if output.ndim != 3:
        return None
    gray = np.clip(output.mean(axis=0), 0.0, 1.0)
    return np.repeat(gray[:, :, None], 3, axis=2)


def run(image, species_name: str, *, fov_deg: float = 60.0, size_px: int = 128,
        window_ms: float = 100.0, noise: bool = True, lam: float | None = None,
        chroma_weight: float = CHROMA_WEIGHT, seed: int = 0,
        density: float = 1.0, neuron_density: float = 1.0, looks: int = 1,
        photons_per_s: float | None = None, on_progress=None) -> RunResult:
    """Encode `image` through a species' visual system and reconstruct it.

    `image` is any (height, width[, channels]) array; it is centre-cropped and
    resized to `size_px`. With `lam=None` the regularization is set from the
    spike noise. `density` multiplies the receptors per unit area and
    `neuron_density` the cortex cells per unit area (1 is the real animal). If given, `on_progress` is called with a `Progress` after
    each stage and after each decoding iteration.

    `looks` is how many times the eye looks during the spike window, moved a
    little each time (`fixate`). The looks share the window, `window_ms / looks`
    each, with independent spike noise, and one solve rebuilds the picture from
    all of them. The total looking time is the same for any number of looks.

    `photons_per_s` is the light level: the photons one real receptor catches
    each second where the picture is white. Light arrives as photons, so in dim
    light the receptors' own signal is noisy (`lit`). None is unlimited light,
    with no photon noise. For a human cone, 1 cd/m2 seen through a 3 mm pupil
    is about 900 photons a second: sunlight is about 1e7, a lit room 1e5.
    `noise=False` gives the noise-free code, with neither spike nor photon noise.
    """
    if window_ms <= 0:
        raise ValueError(f"window_ms must be positive, got {window_ms}")
    if lam is not None and lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}")
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")
    if neuron_density <= 0:
        raise ValueError(f"neuron_density must be positive, got {neuron_density}")
    if photons_per_s is not None and photons_per_s <= 0:
        raise ValueError(f"photons_per_s must be positive, got {photons_per_s}")
    if looks != int(looks) or looks < 1:
        raise ValueError(f"looks must be a whole number of at least 1, got {looks}")
    looks = int(looks)
    start = time.perf_counter()
    original = io.to_square(image, size_px)
    pipeline = build_pipeline(species_name, size_px, float(fov_deg), float(density),
                              float(neuron_density))
    look_s = window_ms / 1000.0 / looks
    if photons_per_s is not None:
        pipeline = lit(pipeline, photons_per_s, look_s)
    if looks > 1:
        pipeline = fixate(pipeline, looks)
    pipeline = pipeline.replace(PoissonSpikes(look_s))
    rng = np.random.default_rng(seed) if noise else None
    code = pipeline.encode(original.transpose(2, 0, 1), rng)
    on_iteration = None
    if on_progress is not None:
        stages = tuple(code.intermediates) + ("decoding",)
        for index, output in enumerate(_first_look(code.intermediates, looks).values()):
            on_progress(Progress(stages, index, _stage_image(output), 0))

        def on_iteration(k, estimate):
            on_progress(Progress(stages, len(stages) - 1, estimate.transpose(1, 2, 0), k))
    if lam is None:
        # The data term of K looks is K times one look's, so the floor grows alike.
        lam = looks * LAM_FLOOR
        if noise:
            variance = Decoder(pipeline, 1.0).noise_variance(code)
            if photons_per_s is not None:
                variance += _photon_noise_variance(pipeline, code, rng)
            lam += NOISE_GAIN * variance
    decoder = Decoder(pipeline, lam, chroma_weight)
    reconstruction = decoder.decode(code, on_iteration)
    reconstructed = reconstruction.image.transpose(1, 2, 0)
    neurons = code.responses.size // looks  # every cell that spikes: both cells of an ON/OFF pair
    metrics = {
        "psnr_db": psnr(original, reconstructed),
        "ssim": ssim(original, reconstructed),
        "neurons": float(neurons),
        "receptors": float(len(pipeline.metadata["mosaic"])),
        "compression_ratio": neurons / original.size,
        "mean_spikes": float(np.mean(code.responses)) * looks,  # over the whole window
    }
    settings = Settings(species_name, float(fov_deg), size_px, float(window_ms),
                        noise, float(lam), chroma_weight, seed, float(density),
                        float(neuron_density), looks,
                        None if photons_per_s is None else float(photons_per_s))
    return RunResult(original, reconstructed, pipeline, code, reconstruction,
                     metrics, settings, time.perf_counter() - start)
