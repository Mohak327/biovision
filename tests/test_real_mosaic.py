"""The real mosaic (phase 13 of the human vision audit): no S cones at the centre,
an L:M ratio that differs between people, and receptors off a perfect lattice."""
from dataclasses import replace

import numpy as np
import pytest

from biovision import species
from biovision.core.field import VisualField
from biovision.species import fly, human, mouse
from biovision.species.eye import assemble, build_mosaic
from biovision.stages.mosaic import Mosaic, square_lattice
from biovision.stages.receptive import RetinaClass, opponent_retina

NARROW = VisualField(128, 1.0)  # a pixel is half an arcminute: one cone per position
MEDIUM = VisualField(128, 2.0)  # a pixel is an arcminute: every position still holds all types
ZONE = 0.175  # degrees


def eccentricity_deg(mosaic, field):
    return field.to_deg(np.linalg.norm(mosaic.positions - (field.size_px - 1) / 2.0, axis=1))


def test_the_human_fovea_has_no_s_cones_at_its_centre():
    """Curcio et al. 1991: a zone about 0.35 degrees across without S cones."""
    assert human.PARAMS.absent_within_deg == (0.0, 0.0, ZONE)
    mosaic, _ = build_mosaic(human.PARAMS, NARROW)
    short = eccentricity_deg(mosaic, NARROW)[mosaic.types == 2]
    assert len(short) > 100 and short.min() >= ZONE
    everywhere, _ = build_mosaic(replace(human.PARAMS, absent_within_deg=()), NARROW)
    inside = eccentricity_deg(everywhere, NARROW) < ZONE
    assert np.sum(everywhere.types[inside] == 2) > 50  # there were some to remove


def test_only_the_s_cones_of_the_zone_change_type():
    mosaic, _ = build_mosaic(human.PARAMS, NARROW)
    everywhere, _ = build_mosaic(replace(human.PARAMS, absent_within_deg=()), NARROW)
    np.testing.assert_array_equal(mosaic.positions, everywhere.positions)
    changed = mosaic.types != everywhere.types
    assert np.all(everywhere.types[changed] == 2) and np.all(mosaic.types[changed] < 2)
    assert np.all(eccentricity_deg(mosaic, NARROW)[changed] < ZONE)


def test_a_pixel_wholly_inside_the_zone_holds_no_s_cone():
    mosaic, _ = build_mosaic(human.PARAMS, MEDIUM)
    ecc = eccentricity_deg(mosaic, MEDIUM)
    half_pixel = MEDIUM.to_deg(np.sqrt(0.5))
    assert not np.any((mosaic.types == 2) & (ecc + half_pixel <= ZONE))
    for kind in range(3):  # everywhere else nothing changed
        outside = (mosaic.types == kind) & (ecc + half_pixel > ZONE)
        assert outside.sum() == np.sum(ecc[mosaic.types == 0] + half_pixel > ZONE)
    assert np.sum(mosaic.types == 2) < np.sum(mosaic.types == 0)


def test_across_a_wide_field_the_zone_is_smaller_than_a_pixel_and_changes_nothing():
    field = VisualField(64, 60.0)
    mosaic, _ = build_mosaic(human.PARAMS, field)
    everywhere, _ = build_mosaic(replace(human.PARAMS, absent_within_deg=()), field)
    np.testing.assert_array_equal(mosaic.positions, everywhere.positions)
    np.testing.assert_array_equal(mosaic.types, everywhere.types)


def test_a_cell_with_none_of_a_receptor_type_it_needs_is_silent(rng):
    """A blue-yellow cell where there are no S cones has nothing to compare."""
    size = 24
    positions = square_lattice(size, 1.0)
    far = np.linalg.norm(positions - (size - 1) / 2.0, axis=1) > 6.0
    mosaic = Mosaic(np.vstack([positions, positions, positions[far]]),
                    np.concatenate([np.zeros(len(positions), int), np.ones(len(positions), int),
                                    np.full(far.sum(), 2)]), 3)
    classes = (RetinaClass("luminance", (0.5, 0.5, 0.0), 1.5, 0.7),
               RetinaClass("blue_yellow", (-0.5, -0.5, 1.0), 3.0, 0.7))
    stage, cells = opponent_retina(mosaic, classes, 1.0, 8.0)  # the centre reaches 3 px
    out = stage.forward(rng.random(len(mosaic)))
    centre = np.linalg.norm(cells.positions - (size - 1) / 2.0, axis=1) < 2.5
    assert centre.sum() > 0
    np.testing.assert_array_equal(out[centre & (cells.types == 1)], 0.0)
    assert np.all(out[centre & (cells.types == 0)] != 0.0)  # luminance needs no S cone
    edge = np.linalg.norm(cells.positions - (size - 1) / 2.0, axis=1) > 9.5
    assert np.all(out[edge & (cells.types == 1)] != 0.0)
    y = rng.standard_normal(stage.out_shape)
    x = rng.standard_normal(stage.in_shape)
    assert np.vdot(stage.forward(x), y) == pytest.approx(np.vdot(x, stage.adjoint(y)), rel=1e-10)


def test_grey_stays_silent_for_the_colour_cells_around_the_zone():
    eye = species.get("human")(NARROW)
    stages = {s.name: s for s in eye.linear_stages}
    cells = eye.metadata["cells"]
    out = stages["center_surround"].forward(np.ones(len(eye.metadata["mosaic"])))
    np.testing.assert_allclose(out[cells.types == 2], 0.0, atol=1e-9)


def test_the_cone_fractions_follow_the_l_to_m_ratio():
    np.testing.assert_allclose(human.cone_fractions(2.0), (0.60, 0.30, 0.10), atol=1e-12)
    np.testing.assert_allclose(human.PARAMS.type_fractions, human.cone_fractions(human.LM_RATIO))
    low, high = human.LM_RATIO_RANGE
    assert low < human.LM_RATIO < high
    for ratio in (low, high):
        long, middle, short = human.cone_fractions(ratio)
        assert long / middle == pytest.approx(ratio) and short == pytest.approx(0.10)
        assert long + middle + short == pytest.approx(1.0)
    with pytest.raises(ValueError, match="lm_ratio"):
        human.cone_fractions(0.0)


def test_a_person_with_few_m_cones_has_a_mosaic_to_match():
    params = replace(human.PARAMS, type_fractions=human.cone_fractions(16.5))
    mosaic, _ = build_mosaic(params, NARROW)
    counts = np.bincount(mosaic.types, minlength=3)
    assert counts[0] / counts[1] == pytest.approx(16.5, rel=0.15)
    eye = assemble("human", VisualField(48, 0.5), params, "", ())
    assert eye.metadata["cells"].n_types == 3


@pytest.mark.parametrize("module", [human, mouse, fly])
def test_no_species_jitters_its_receptors_by_default(module):
    assert module.PARAMS.jitter == 0.0


def test_jitter_moves_single_receptors_by_a_fraction_of_their_spacing():
    regular, _ = build_mosaic(human.PARAMS, NARROW)
    jittered, _ = build_mosaic(replace(human.PARAMS, jitter=0.1), NARROW)
    np.testing.assert_array_equal(jittered.types, regular.types)
    moved = jittered.positions - regular.positions
    inner = np.all((regular.positions > 2) & (regular.positions < 125), axis=1)
    # Spacing here is 1 to 1.4 pixels, so a tenth of it is 0.10 to 0.14 pixels.
    assert 0.09 < moved[inner].std() < 0.15
    assert abs(moved[inner].mean()) < 0.01
    assert jittered.positions.min() >= 0.0 and jittered.positions.max() <= 127.0


def test_jitter_leaves_pooled_positions_where_they_are():
    field = VisualField(64, 60.0)
    regular, _ = build_mosaic(human.PARAMS, field)
    jittered, _ = build_mosaic(replace(human.PARAMS, jitter=0.2), field)
    np.testing.assert_array_equal(jittered.positions, regular.positions)
    with pytest.raises(ValueError, match="jitter"):
        build_mosaic(replace(human.PARAMS, jitter=-0.1), field)


def test_a_jittered_eye_still_has_an_exact_adjoint(rng):
    params = replace(human.PARAMS, jitter=0.15)
    operator = assemble("human", VisualField(32, 0.25), params, "", ()).linear_operator()
    x = rng.standard_normal(operator.shape[1])
    y = rng.standard_normal(operator.shape[0])
    assert operator.matvec(x) @ y == pytest.approx(x @ operator.rmatvec(y), rel=1e-9)
