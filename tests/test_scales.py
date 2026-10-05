"""The spike budget and the cortex scales must follow the eye, not the picture's pixel count."""
import dataclasses

import pytest

from biovision import species
from biovision.core.field import VisualField
from biovision.species import fly, human
from biovision.species.eye import assemble


def rest_hz(pipeline):
    """The firing rate of a model cell with no signal: rest_hz times the real cells it stands for."""
    return pipeline.pointwise_stages[0].rest_hz


def build_human(field, **changes):
    params = dataclasses.replace(human.PARAMS, **changes)
    return assemble("human", field, params, human.DESCRIPTION, human.CITATIONS)


def test_a_cortex_cell_fires_at_the_same_rate_whatever_the_picture_size():
    """Before, the rate fell fourfold each time the picture's side doubled."""
    rates = [rest_hz(species.get("human")(VisualField(size, 60.0))) for size in (48, 64, 96)]
    assert rates == pytest.approx([rates[0]] * 3, rel=0.01)


def test_a_human_cortex_cell_stands_for_the_cones_in_its_patch_of_field():
    pipeline = species.get("human")(VisualField(96, 60.0))
    mosaic, cells = pipeline.metadata["mosaic"], pipeline.metadata["cells_per_position"]
    cones_per_square_degree = cells * len(mosaic) / 3 / 60.0**2  # three cone types a position
    expected = cones_per_square_degree * human.PARAMS.spike_patch_deg**2
    assert pipeline.metadata["cells_per_neuron"] == pytest.approx(expected)
    assert rest_hz(pipeline) == pytest.approx(human.PARAMS.rest_hz * expected)
    # At 96 px a pixel is one patch, so this is nearly the old rule's value there
    # (the lattice leaves a margin at the picture's edge).
    assert expected == pytest.approx(cells, rel=0.03)


def test_more_receptors_per_degree_give_a_cortex_cell_more_spikes():
    field = VisualField(48, 60.0)
    base = species.get("human")(field)
    dense = species.get("human")(field, density=4.0)
    assert rest_hz(dense) == pytest.approx(4.0 * rest_hz(base), rel=0.01)


@pytest.mark.parametrize("size", [24, 64, 128])
def test_a_mouse_cortex_cell_stands_for_one_real_cell_at_any_size(size):
    """Mouse receptors are a degree apart, wider than the patch, at every picture size."""
    pipeline = species.get("mouse")(VisualField(size, 60.0))
    assert pipeline.metadata["cells_per_neuron"] == 1.0
    assert rest_hz(pipeline) == pytest.approx(100.0)


def test_retinal_output_cells_keep_the_total_of_real_receptors():
    """Without a cortex the model's cells sit at its receptor positions, so the
    real receptors each stands for fall as the picture grows and their total stays."""
    params = dataclasses.replace(fly.PARAMS, spacing_deg=0.5)  # finer than these pixels
    totals = []
    for size in (32, 64):
        pipeline = assemble("fly", VisualField(size, 60.0), params, fly.DESCRIPTION, fly.CITATIONS)
        assert pipeline.metadata["cells_per_neuron"] == pipeline.metadata["cells_per_position"] > 1
        totals.append(pipeline.n_neurons * rest_hz(pipeline))
    assert totals[0] == pytest.approx(totals[1], rel=0.05)  # the lattice's edge rows


def test_human_cortex_scales_reach_as_fine_as_the_picture_can_carry():
    """A wavelength under two pixels is dropped; 64 px across 15 degrees is 256 px across 60."""
    assert human.PARAMS.cortex_sf_cpd == (0.2, 0.8, 1.6, 3.2)
    assert len(human.PARAMS.cortex_gains) == len(human.PARAMS.cortex_types) == 4
    neurons = [build_human(VisualField(64, 15.0), cortex_sf_cpd=human.PARAMS.cortex_sf_cpd[:n],
                           cortex_gains=human.PARAMS.cortex_gains[:n],
                           cortex_types=human.PARAMS.cortex_types[:n]).n_neurons
               for n in (2, 3, 4)]
    assert neurons[1] > 2 * neurons[0]  # the 1.6 cycles/degree scale fits
    assert neurons[2] == neurons[1]  # 3.2 cycles/degree would be 1.3 px a wave


def test_the_finest_human_scale_carries_luminance_only():
    """Colour is seen at lower resolution than brightness (Mullen 1985)."""
    assert human.PARAMS.cortex_types == (None, None, None, (0,))
    assert human.PARAMS.retina_classes[0].name == "luminance"
    field = VisualField(64, 7.5)  # the same degrees per pixel as 512 px across 60
    coarser = build_human(field, cortex_sf_cpd=(0.2, 0.8, 1.6), cortex_gains=(1.0, 2.0, 2.0),
                          cortex_types=()).n_neurons
    luminance = species.get("human")(field).n_neurons - coarser
    every_class = build_human(field, cortex_types=()).n_neurons - coarser
    assert luminance > 0 and every_class == 3 * luminance


def test_the_human_eye_with_fine_scales_keeps_an_exact_transpose(rng):
    operator = species.get("human")(VisualField(64, 7.5)).linear_operator()
    x = rng.standard_normal(operator.shape[1])
    y = rng.standard_normal(operator.shape[0])
    assert operator.matvec(x) @ y == pytest.approx(x @ operator.rmatvec(y), rel=1e-9)

