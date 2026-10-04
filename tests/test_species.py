import numpy as np
import pytest

from biovision import io, species
from biovision.core.field import VisualField
from biovision.species import human, mouse
from biovision.species.eye import build_mosaic

NAMES = ["fly", "human", "mouse"]


def test_the_three_species_are_registered():
    assert species.names() == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_composed_operator_adjoint(name, field, rng):
    operator = species.get(name)(field).linear_operator()
    x = rng.standard_normal(operator.shape[1])
    y = rng.standard_normal(operator.shape[0])
    assert operator.matvec(x) @ y == pytest.approx(x @ operator.rmatvec(y), rel=1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_pipeline_carries_description_citations_and_metadata(name, field):
    pipeline = species.get(name)(field)
    assert pipeline.name == name
    assert len(pipeline.description) > 40 and len(pipeline.citations) >= 3
    assert {"params", "mosaic", "cells_per_position"} <= set(pipeline.metadata)
    assert [s.name for s in pipeline.pointwise_stages] == ["rate", "spikes"]


@pytest.mark.parametrize("name", NAMES)
def test_natural_image_rates_stay_above_zero(name, sample):
    """The contrast gain must not push many cells into rectification."""
    field = VisualField(64, 60.0)
    pipeline = species.get(name)(field)
    image = io.to_square(sample, field.size_px).transpose(2, 0, 1)
    rates = pipeline.encode(image).intermediates["rate"]
    assert np.mean(rates == 0.0) < 0.01


def test_fly_has_no_cortex_and_mammals_do():
    field = VisualField(64, 60.0)
    names = {n: [s.name for s in species.get(n)(field).linear_stages] for n in NAMES}
    assert "gabor" not in names["fly"]
    assert "gabor" in names["human"] and "gabor" in names["mouse"]


def test_cortex_is_dropped_when_no_wavelength_fits_the_image():
    """At 8 pixels across 60 degrees, every mouse and human wavelength under
    two pixels is beyond the image's Nyquist limit."""
    stages = [s.name for s in species.get("human")(VisualField(4, 170.0)).linear_stages]
    assert "gabor" not in stages


def test_receptor_counts_rank_human_mouse_fly():
    field = VisualField(64, 60.0)
    counts = {n: len(species.get(n)(field).metadata["mosaic"]) for n in NAMES}
    assert counts["human"] > counts["mouse"] > counts["fly"]


def test_sub_pixel_receptors_share_positions_and_pool_cells(field):
    mosaic, cells = build_mosaic(human.PARAMS, field)
    positions = {tuple(p) for p in np.round(mosaic.positions, 6)}
    assert len(mosaic) == 3 * len(positions)  # every type at every position
    assert cells > 1.0
    _, mouse_cells = build_mosaic(mouse.PARAMS, VisualField(128, 60.0))
    assert mouse_cells == 1.0


def test_narrow_field_gives_single_type_receptors_in_the_periphery():
    mosaic, _ = build_mosaic(human.PARAMS, VisualField(128, 1.5))
    positions = {tuple(p) for p in np.round(mosaic.positions, 6)}
    assert len(positions) < len(mosaic) < 3 * len(positions)


@pytest.mark.parametrize("name", NAMES)
def test_density_is_recorded_and_defaults_to_one(name, field):
    assert species.get(name)(field).metadata["density"] == 1.0
    assert species.get(name)(field, density=4.0).metadata["density"] == 4.0


def test_density_packs_receptors_closer():
    field = VisualField(64, 60.0)
    base = len(species.get("mouse")(field).metadata["mosaic"])
    sparse = len(species.get("mouse")(field, density=0.25).metadata["mosaic"])
    assert sparse < base / 3


def test_neuron_density_scales_cortex_cells_but_not_receptors():
    field = VisualField(64, 60.0)
    base = species.get("mouse")(field)
    dense = species.get("mouse")(field, neuron_density=4.0)
    assert dense.metadata["neuron_density"] == 4.0 and base.metadata["neuron_density"] == 1.0
    assert dense.n_neurons > 3 * base.n_neurons
    assert len(dense.metadata["mosaic"]) == len(base.metadata["mosaic"])


def test_neuron_density_does_nothing_without_a_cortex():
    field = VisualField(64, 60.0)
    assert (species.get("fly")(field, neuron_density=4.0).n_neurons
            == species.get("fly")(field).n_neurons)
