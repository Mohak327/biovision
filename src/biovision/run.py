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
from .species.eye import fixate, lit, with_rods

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
    indexes the step just reached. `image` is (size, size, 3) in [0, 1]: a
    preview of the stage's output, or the current estimate while decoding.
    `iteration` is 0 until decoding starts, and `fraction` is then the share
    of the solve that is done, from 0 to 1. `plot` holds the numbers behind a
    stage whose output is better read as a graph (see `stage_plots`).
    """

    stages: tuple[str, ...]
    current: int
    image: np.ndarray
    iteration: int
    fraction: float = 0.0
    plot: dict | None = None


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


PLOT_POINTS = 41  # points along the rate curve; one fewer bins under it
SPIKE_BINS = 30


def stage_plots(pipeline: Pipeline, code: NeuralCode) -> dict[str, dict]:
    """The numbers behind the two pointwise stages of this run, for a graph of each.

    The rate stage: its curve, evaluated by the stage itself over the range of
    responses this picture produced (`response` against `rates`, one curve per
    kind of cell), and how many cells fall in each interval of that range
    (`cells`). The spike stage: a histogram of the counts its cells fired.
    """
    rate_stage, spike_stage = pipeline.pointwise_stages[0], pipeline.pointwise_stages[-1]
    drive = np.asarray(code.intermediates[pipeline.linear_stages[-1].name], dtype=float).ravel()
    reach = float(np.max(np.abs(np.percentile(drive, (0.5, 99.5))))) or 1.0
    response = np.linspace(-reach, reach, PLOT_POINTS)
    rates = np.asarray(rate_stage.forward(response), dtype=float).reshape(PLOT_POINTS, -1)
    kinds = ("ON cells", "OFF cells") if rates.shape[1] == 2 else ("cells",)
    in_range, _ = np.histogram(drive, bins=response)
    fired, edges = np.histogram(np.asarray(code.responses).ravel(), bins=SPIKE_BINS)
    return {
        rate_stage.name: {"response": response.tolist(),
                          "rates": {kind: rates[:, i].tolist() for i, kind in enumerate(kinds)},
                          "cells": in_range.tolist()},
        spike_stage.name: {"edges": edges.tolist(), "cells": fired.tolist()},
    }


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


def _stretched(picture: np.ndarray) -> np.ndarray:
    """A (height, width, channels) picture with each channel's 1st to 99th
    percentiles stretched to [0, 1], which also removes a colour cast."""
    low, high = np.percentile(picture, (1.0, 99.0), axis=(0, 1))
    span = np.where(high > low, high - low, 1.0)
    return np.clip((picture - low) / span, 0.0, 1.0)


def stage_previews(pipeline: Pipeline, code: NeuralCode, looks: int = 1) -> dict[str, np.ndarray]:
    """A (size, size, 3) picture in [0, 1] for every stage's output, for display.

    A stage whose output is already a picture per receptor type is shown as
    its mean over types. Any other output (receptor samples, retinal and
    cortical cells, rates, spikes) is carried back into picture space: the
    pointwise stages up to it are undone, then the transposes of the linear
    stages before it are applied, and the result is stretched for contrast.
    It shows where in the picture that stage's signal sits, not a reconstruction.
    """
    size = pipeline.field.size_px
    pictures = {}
    first_look = _first_look(code.intermediates, looks)
    for index, stage in enumerate(pipeline.stages):
        output = first_look[stage.name]
        if output.ndim == 3 and output.shape[1:] == (size, size):
            gray = np.clip(output.mean(axis=0), 0.0, 1.0)
            pictures[stage.name] = np.repeat(gray[:, :, None], 3, axis=2)
            continue
        back = np.asarray(code.intermediates[stage.name], dtype=float)
        linear = pipeline.linear_stages
        if stage in pipeline.pointwise_stages:
            reached = pipeline.pointwise_stages.index(stage) + 1
            for undone in reversed(pipeline.pointwise_stages[:reached]):
                back = undone.inverse(back)
        else:
            linear = linear[:linear.index(stage) + 1]
        for before in reversed(linear):
            back = before.adjoint(back)
        pictures[stage.name] = _stretched(back.transpose(1, 2, 0))
    return pictures


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
    An eye that has rods uses them at a light level, as far as that light leaves
    them unsaturated (`with_rods`): for the human eye, from about 1e4 down.
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
        pipeline = lit(with_rods(pipeline, photons_per_s), photons_per_s, look_s)
    if looks > 1:
        pipeline = fixate(pipeline, looks)
    pipeline = pipeline.replace(pipeline.pointwise_stages[-1].lasting(look_s))
    rng = np.random.default_rng(seed) if noise else None
    code = pipeline.encode(original.transpose(2, 0, 1), rng)
    on_iteration = None
    if on_progress is not None:
        stages = tuple(code.intermediates) + ("decoding",)
        plots = stage_plots(pipeline, code)
        for index, (name, picture) in enumerate(stage_previews(pipeline, code, looks).items()):
            on_progress(Progress(stages, index, picture, 0, plot=plots.get(name)))
        furthest = [0.0]  # the residual can rise for a step; the reported share never does

        def on_iteration(k, estimate, residual):
            furthest[0] = max(furthest[0], decoder.progress(residual))
            on_progress(Progress(stages, len(stages) - 1, estimate.transpose(1, 2, 0), k,
                                 furthest[0]))
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
