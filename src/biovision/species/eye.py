"""Shared assembly of an eye from parameters. Species files supply the numbers."""
from dataclasses import dataclass, replace

import numpy as np

from ..core.field import VisualField
from ..core.pipeline import Pipeline
from ..stages.color import ColorProjection
from ..stages.gabor import gabor_bank
from ..stages.movement import EyeShifts, PerLook, look_offsets
from ..stages.mosaic import (Mosaic, all_types_at, assign_types, foveated_lattice,
                             hex_lattice, mosaic_sampling, retype_inside, square_lattice)
from ..stages.nonlinearity import LinearRectified, OnOffPair
from ..stages.optics import OpticalBlur, defocus_sigma_deg
from ..stages.photons import PhotonCatch
from ..stages.receptive import RetinaClass, center_surround, opponent_retina
from ..stages.spiking import PoissonSpikes

DEFAULT_WINDOW_S = 0.1
MIN_SPACING_PX = 1.0  # an image carries no detail finer than a pixel
MIN_SIGMA_PX = 0.5


@dataclass(frozen=True)
class EyeParams:
    """Everything that distinguishes one species' eye. Angles are in degrees."""

    receptor_names: tuple[str, ...]
    color_matrix: tuple[tuple[float, float, float], ...]  # rows: receptor types, cols: RGB
    type_fractions: tuple[float, ...]
    colocated: bool  # True: every type at every position (fly ommatidia)
    blur_sigma_deg: float
    lattice: str  # "square", "hex" or "foveated"
    spacing_deg: float  # receptor spacing (at the centre, for "foveated")
    center_sigma_deg: float
    surround_sigma_deg: float
    surround_weight: float
    e2_deg: float = 0.0  # eccentricity at which spacing doubles ("foveated" only)
    cortex_sf_cpd: tuple[float, ...] = ()  # preferred spatial frequencies; empty = no cortex
    # Firing rate with no signal, of a cell that stands for an ON/OFF pair. With
    # separate ON and OFF cells, half the top of the range: the pair shares the
    # real cells, and each fires 2 * rest_hz above its spontaneous rate when
    # contrast_gain * signal reaches 1, where the one cell did.
    rest_hz: float = 100.0
    contrast_gain: float = 2.5  # scales responses to span the firing range
    mosaic_seed: int = 0
    # Retinal cell classes that combine receptor types; empty = one cell per receptor.
    retina_classes: tuple[RetinaClass, ...] = ()
    cortex_gains: tuple[float, ...] = ()  # one per cortical frequency; empty = all 1
    # The retinal classes each cortical frequency carries (None = all); empty = all, at all.
    cortex_types: tuple[tuple[int, ...] | None, ...] = ()
    spike_patch_deg: float = 0.625  # a cortex cell fires for the real receptors in this patch
    # How far the gaze strays from its mean position while the eye fixates: the
    # radius of the disc that several looks are spread over. 0 = an eye held still.
    fixation_deg: float = 0.0
    # Chromatic aberration: how far out of focus each receptor type's light is, in
    # dioptres, and the pupil's diameter in millimetres, which sets how much blur
    # that defocus makes. Empty = every receptor type sees the same blur.
    chromatic_defocus_d: tuple[float, ...] = ()
    pupil_mm: float = 0.0
    # The share of the light that reaches each receptor type through the eye's
    # own filters (lens, macular pigment), relative to the others. It scales the
    # photons a receptor catches, not its signal: a receptor adapts its gain to
    # the light it gets. Empty = 1 for every type.
    transmission: tuple[float, ...] = ()
    # For each receptor type, the radius in degrees of the zone round the centre of
    # gaze that has none of it (0 = found everywhere). Empty = all types everywhere.
    absent_within_deg: tuple[float, ...] = ()
    # How far receptors sit from a perfect lattice: the standard deviation of each
    # one's displacement, as a fraction of the local spacing. 0 = a perfect lattice.
    jitter: float = 0.0
    # Separate ON and OFF cells: the rate, in spikes/s, at which each fires with
    # no signal. None = one cell around `rest_hz` stands for the pair.
    spontaneous_hz: float | None = None
    # How regular the spikes are: the variance of a spike count over its mean
    # (no unit). 1 = Poisson; a refractory period makes it less (`PoissonSpikes`).
    fano: float = 1.0


def local_spacing_px(params: EyeParams, field: VisualField, positions: np.ndarray,
                     density: float = 1.0) -> np.ndarray:
    """The real eye's receptor spacing at each position, in pixels."""
    true_spacing = field.to_px(params.spacing_deg) / np.sqrt(density)
    if params.lattice != "foveated":
        return np.full(len(positions), true_spacing)
    eccentricity = np.linalg.norm(positions - (field.size_px - 1) / 2.0, axis=1)
    return true_spacing * (1.0 + eccentricity / field.to_px(params.e2_deg))


def build_mosaic(params: EyeParams, field: VisualField,
                 density: float = 1.0) -> tuple[Mosaic, float]:
    """Receptor positions and types at this image resolution, and cells per position.

    Where the eye's receptors are smaller than a pixel, many receptors of every
    type fall inside each pixel, so that position carries all types. Elsewhere
    each position holds one receptor of a randomly drawn type, moved off the
    lattice by `jitter`.

    A type is left out inside its zone of `absent_within_deg`: a single
    receptor there is drawn again from the other types, and a shared position
    loses the type if its whole pixel is inside the zone.

    The second value is the mean number of real cells that one model position
    stands for (1 when receptors are at least a pixel apart).

    `density` multiplies the number of receptors per unit area, so the spacing
    shrinks by its square root. 1 is the real animal.
    """
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")
    if params.jitter < 0:
        raise ValueError(f"jitter must not be negative, got {params.jitter}")
    size = field.size_px
    true_spacing = field.to_px(params.spacing_deg) / np.sqrt(density)
    spacing = max(true_spacing, MIN_SPACING_PX)
    if params.lattice == "square":
        positions = square_lattice(size, spacing)
    elif params.lattice == "hex":
        positions = hex_lattice(size, spacing)
    elif params.lattice == "foveated":
        positions = foveated_lattice(size, true_spacing, field.to_px(params.e2_deg),
                                     MIN_SPACING_PX)
    else:
        raise ValueError(f"unknown lattice '{params.lattice}'")
    local_spacing = local_spacing_px(params, field, positions, density)
    cells_per_position = float(np.mean(np.maximum(MIN_SPACING_PX / local_spacing, 1.0) ** 2))
    n_types = len(params.receptor_names)
    if params.colocated:
        return all_types_at(positions, n_types), cells_per_position
    rng = np.random.default_rng(params.mosaic_seed)
    dense = local_spacing < MIN_SPACING_PX
    shared = all_types_at(positions[dense], n_types)
    single = positions[~dense]
    single_types = assign_types(len(single), params.type_fractions, rng)
    if params.absent_within_deg:
        if len(params.absent_within_deg) != n_types:
            raise ValueError("absent_within_deg needs one value per receptor type")
        radius = field.to_px(np.asarray(params.absent_within_deg, dtype=float))

        def eccentricity(points):
            return np.linalg.norm(points - (size - 1) / 2.0, axis=1)

        single_types = retype_inside(single_types, eccentricity(single), radius,
                                     params.type_fractions, rng)
        reach = MIN_SPACING_PX / np.sqrt(2.0)  # from the centre of a pixel to its corner
        present = eccentricity(shared.positions) + reach > radius[shared.types]
        shared = Mosaic(shared.positions[present], shared.types[present], n_types)
    if params.jitter:
        moved = rng.standard_normal(single.shape) * params.jitter * local_spacing[~dense, None]
        single = np.clip(single + moved, 0.0, size - 1.0)
    mosaic = Mosaic(np.vstack([shared.positions, single]),
                    np.concatenate([shared.types, single_types]), n_types)
    return mosaic, cells_per_position


def receptors_each(params: EyeParams, field: VisualField, mosaic: Mosaic,
                   density: float = 1.0) -> np.ndarray:
    """How many real receptors each receptor of the model's mosaic stands for.

    1 where the eye's receptors are at least a pixel apart. Where they are
    smaller, a position stands for all the real receptors in its pixel, and
    its receptor of each type for that type's share of them (all of them in
    an eye whose positions carry every type, such as a fly's facets).
    """
    spacing = local_spacing_px(params, field, mosaic.positions, density)
    in_pixel = np.maximum(MIN_SPACING_PX / spacing, 1.0) ** 2
    if params.colocated:
        return in_pixel
    share = np.asarray(params.type_fractions, dtype=float)
    share = share / share.sum()
    return np.where(spacing < MIN_SPACING_PX, in_pixel * share[mosaic.types], 1.0)


def cells_per_neuron(params: EyeParams, field: VisualField, mosaic: Mosaic,
                     cells_per_position: float) -> float:
    """How many real cells' spikes one model output cell fires: the spike budget.

    The eye has a fixed number of real cells in a field of view, so this number
    must never depend on how many pixels the picture has.

    A cortex cell stands for the real receptors in a patch of field
    `spike_patch_deg` across (at least one). The patch is fixed in degrees, and
    so is the grid of cortex cells, so a cell fires at the same rate at every
    picture size. A scale too fine for a small picture is left out there: its
    real cells would fire, but tell nothing about a picture without that detail.

    Without a cortex the output cells are retinal, one at each model position,
    and each stands for the real receptors at its position. Positions follow
    the pixels where receptors are smaller than a pixel, so that number changes
    with the picture, but positions times receptors each, the total, does not.
    """
    if not params.cortex_sf_cpd:
        return cells_per_position
    receptors = cells_per_position * len(np.unique(mosaic.positions, axis=0))
    return max(receptors * (params.spike_patch_deg / field.fov_deg) ** 2, 1.0)


def blur_sigmas_deg(params: EyeParams):
    """The optical blur each receptor type sees: one sigma, or one per type.

    `blur_sigma_deg` is the blur of light in focus. Light a receptor type sees
    out of focus is blurred further, and the two blurs' variances add.
    """
    if not params.chromatic_defocus_d:
        return params.blur_sigma_deg
    if len(params.chromatic_defocus_d) != len(params.receptor_names):
        raise ValueError("chromatic_defocus_d needs one value per receptor type")
    return tuple(float(np.hypot(params.blur_sigma_deg,
                                defocus_sigma_deg(params.pupil_mm, defocus)))
                 for defocus in params.chromatic_defocus_d)


def assemble(name: str, field: VisualField, params: EyeParams,
             description: str, citations: tuple[str, ...], density: float = 1.0,
             neuron_density: float = 1.0) -> Pipeline:
    """Optics, mosaic, retina, optional cortex, then rate and spikes.

    `density` scales the receptors per unit area and `neuron_density` the
    cortex cells per unit area. 1 is the real animal for both.
    """
    if neuron_density <= 0:
        raise ValueError(f"neuron_density must be positive, got {neuron_density}")
    size = field.size_px
    n_types = len(params.receptor_names)
    if params.transmission and len(params.transmission) != n_types:
        raise ValueError("transmission needs one value per receptor type")
    mosaic, cells_per_position = build_mosaic(params, field, density)
    sigma_center = field.to_px(params.center_sigma_deg)
    sigma_surround = max(field.to_px(params.surround_sigma_deg), 2 * MIN_SIGMA_PX)
    if params.retina_classes:
        # Cells that mix receptor types must reach receptors of each type, so
        # no centre is narrower than half the spacing between positions.
        spacing = size / np.sqrt(len(np.unique(mosaic.positions, axis=0)))
        retina, cells = opponent_retina(mosaic, params.retina_classes, sigma_center,
                                        sigma_surround, max(MIN_SIGMA_PX, 0.5 * spacing))
    else:
        retina = center_surround(mosaic, max(sigma_center, MIN_SIGMA_PX), sigma_surround,
                                 params.surround_weight)
        cells = mosaic
    stages = [
        ColorProjection(params.color_matrix, size),
        OpticalBlur(field.to_px(np.asarray(blur_sigmas_deg(params))), n_types, size),
        mosaic_sampling(mosaic, size),
        retina,
    ]
    if params.cortex_sf_cpd:
        # A wavelength under two pixels is beyond what the image can carry.
        gains = params.cortex_gains or (1.0,) * len(params.cortex_sf_cpd)
        types = params.cortex_types or (None,) * len(params.cortex_sf_cpd)
        scales = [(field.to_px(1.0 / sf), gain, kept)
                  for sf, gain, kept in zip(params.cortex_sf_cpd, gains, types)]
        scales = [scale for scale in scales if scale[0] >= 2.0 * MIN_SPACING_PX]
        if scales:
            wavelengths, gains, types = zip(*scales)
            stages.append(gabor_bank(cells, size, wavelengths, min_spacing_px=MIN_SPACING_PX,
                                     density=neuron_density, gains=gains, types=types))
    # One model cell stands for several real cells, and their spikes add.
    real_cells = cells_per_neuron(params, field, mosaic, cells_per_position)
    if params.spontaneous_hz is None:
        rate = LinearRectified(params.rest_hz * real_cells, params.contrast_gain)
    else:
        # Half the real cells are ON and half OFF, and a cell's range is twice its
        # resting rate: (real_cells / 2) * (2 * rest_hz) is the same swing.
        rate = OnOffPair(params.rest_hz * real_cells, params.contrast_gain,
                         params.spontaneous_hz * real_cells / 2.0)
    stages += [rate, PoissonSpikes(DEFAULT_WINDOW_S, params.fano)]
    return Pipeline(name, field, tuple(stages), description, citations,
                    {"params": params, "mosaic": mosaic, "cells": cells,
                     "cells_per_position": cells_per_position,
                     "receptors_each": receptors_each(params, field, mosaic, density),
                     "cells_per_neuron": real_cells, "density": density,
                     "neuron_density": neuron_density})


def fixate(pipeline: Pipeline, looks: int) -> Pipeline:
    """An eye that looks at the picture `looks` times, moved a little each time.

    A real eye is never still. Each look lands the picture on a different part
    of the mosaic, so together the looks sample the scene where a single look
    has no receptor. The result is one pipeline: the picture is shifted once
    per look (`fixation`), then every look passes through the eye's own stages,
    which are shared, not copied. Its code has one row of spike counts per look.
    """
    params = pipeline.metadata["params"]
    channels, size, _ = pipeline.in_shape
    offsets = look_offsets(looks, pipeline.field.to_px(params.fixation_deg))
    stages = [EyeShifts(offsets, channels, size)]
    stages += [PerLook(stage, looks) for stage in pipeline.linear_stages]
    return replace(pipeline, stages=(*stages, *pipeline.pointwise_stages))


def lit(pipeline: Pipeline, photons_per_s: float, window_s: float) -> Pipeline:
    """An eye in light of a given level: its receptors count photons (`PhotonCatch`).

    `photons_per_s` is how many photons one real receptor catches each second
    where the picture is white. A model receptor catches that for every real
    receptor it stands for, over the window, less what the eye's own filters
    absorb before its type (`transmission`). The stage sits straight after the
    mosaic, and changes the code only through its noise.
    """
    if photons_per_s <= 0:
        raise ValueError(f"photons_per_s must be positive, got {photons_per_s}")
    params, mosaic = pipeline.metadata["params"], pipeline.metadata["mosaic"]
    transmission = np.asarray(params.transmission or (1.0,) * mosaic.n_types)
    photons = (photons_per_s * window_s * pipeline.metadata["receptors_each"]
               * transmission[mosaic.types])
    stages = []
    for stage in pipeline.stages:
        stages.append(stage)
        if stage.name == "mosaic":
            stages.append(PhotonCatch(photons))
    return replace(pipeline, stages=tuple(stages))
