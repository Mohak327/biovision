"""Shared assembly of an eye from parameters. Species files supply the numbers."""
from dataclasses import dataclass

import numpy as np

from ..core.field import VisualField
from ..core.pipeline import Pipeline
from ..stages.color import ColorProjection
from ..stages.gabor import gabor_bank
from ..stages.mosaic import (Mosaic, all_types_at, assign_types, foveated_lattice,
                             hex_lattice, mosaic_sampling, square_lattice)
from ..stages.nonlinearity import LinearRectified
from ..stages.optics import OpticalBlur
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
    rest_hz: float = 100.0  # firing rate with no signal
    contrast_gain: float = 2.5  # scales responses to span the firing range
    mosaic_seed: int = 0
    # Retinal cell classes that combine receptor types; empty = one cell per receptor.
    retina_classes: tuple[RetinaClass, ...] = ()
    cortex_gains: tuple[float, ...] = ()  # one per cortical frequency; empty = all 1


def build_mosaic(params: EyeParams, field: VisualField,
                 density: float = 1.0) -> tuple[Mosaic, float]:
    """Receptor positions and types at this image resolution, and cells per position.

    Where the eye's receptors are smaller than a pixel, many receptors of every
    type fall inside each pixel, so that position carries all types. Elsewhere
    each position holds one receptor of a randomly drawn type.

    The second value is the mean number of real cells that one model position
    stands for (1 when receptors are at least a pixel apart).

    `density` multiplies the number of receptors per unit area, so the spacing
    shrinks by its square root. 1 is the real animal.
    """
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")
    size = field.size_px
    true_spacing = field.to_px(params.spacing_deg) / np.sqrt(density)
    spacing = max(true_spacing, MIN_SPACING_PX)
    if params.lattice == "square":
        positions = square_lattice(size, spacing)
        local_spacing = np.full(len(positions), true_spacing)
    elif params.lattice == "hex":
        positions = hex_lattice(size, spacing)
        local_spacing = np.full(len(positions), true_spacing)
    elif params.lattice == "foveated":
        e2 = field.to_px(params.e2_deg)
        positions = foveated_lattice(size, true_spacing, e2, MIN_SPACING_PX)
        eccentricity = np.linalg.norm(positions - (size - 1) / 2.0, axis=1)
        local_spacing = true_spacing * (1.0 + eccentricity / e2)
    else:
        raise ValueError(f"unknown lattice '{params.lattice}'")
    cells_per_position = float(np.mean(np.maximum(MIN_SPACING_PX / local_spacing, 1.0) ** 2))
    n_types = len(params.receptor_names)
    if params.colocated:
        return all_types_at(positions, n_types), cells_per_position
    rng = np.random.default_rng(params.mosaic_seed)
    dense = local_spacing < MIN_SPACING_PX
    shared = all_types_at(positions[dense], n_types)
    single = positions[~dense]
    single_types = assign_types(len(single), params.type_fractions, rng)
    mosaic = Mosaic(np.vstack([shared.positions, single]),
                    np.concatenate([shared.types, single_types]), n_types)
    return mosaic, cells_per_position


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
    mosaic, cells_per_position = build_mosaic(params, field, density)
    sigma_center = max(field.to_px(params.center_sigma_deg), MIN_SIGMA_PX)
    sigma_surround = max(field.to_px(params.surround_sigma_deg), 2 * MIN_SIGMA_PX)
    if params.retina_classes:
        # Cells that mix receptor types must reach receptors of each type, so
        # their centre is at least half the spacing between positions.
        spacing = size / np.sqrt(len(np.unique(mosaic.positions, axis=0)))
        retina, cells = opponent_retina(mosaic, params.retina_classes,
                                        max(sigma_center, 0.5 * spacing), sigma_surround)
    else:
        retina = center_surround(mosaic, sigma_center, sigma_surround, params.surround_weight)
        cells = mosaic
    stages = [
        ColorProjection(params.color_matrix, size),
        OpticalBlur(field.to_px(params.blur_sigma_deg), n_types, size),
        mosaic_sampling(mosaic, size),
        retina,
    ]
    if params.cortex_sf_cpd:
        # A wavelength under two pixels is beyond what the image can carry.
        gains = params.cortex_gains or (1.0,) * len(params.cortex_sf_cpd)
        scales = [(field.to_px(1.0 / sf), gain) for sf, gain in zip(params.cortex_sf_cpd, gains)]
        scales = [(w, gain) for w, gain in scales if w >= 2.0 * MIN_SPACING_PX]
        if scales:
            stages.append(gabor_bank(cells, size, [w for w, _ in scales],
                                     min_spacing_px=MIN_SPACING_PX, density=neuron_density,
                                     gains=[gain for _, gain in scales]))
    stages += [
        # One model cell stands for every real cell at its position, so their
        # spikes add: the effective rate scales with the number of cells.
        LinearRectified(params.rest_hz * cells_per_position, params.contrast_gain),
        PoissonSpikes(DEFAULT_WINDOW_S),
    ]
    return Pipeline(name, field, tuple(stages), description, citations,
                    {"params": params, "mosaic": mosaic, "cells": cells,
                     "cells_per_position": cells_per_position, "density": density,
                     "neuron_density": neuron_density})
