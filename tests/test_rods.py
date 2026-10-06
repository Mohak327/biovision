"""Rods and night vision (phase 14 of the human vision audit)."""
import numpy as np
import pytest

from biovision.stages.mosaic import Mosaic
from biovision.stages.receptive import rod_pathway

ROD = 2  # the rod's type index in the small mosaic below


def _mosaic():
    """Two cone types on a 6 x 6 grid, and a rod at every position right of column 1."""
    rows, cols = np.meshgrid(np.arange(6.0), np.arange(6.0), indexing="ij")
    grid = np.column_stack([rows.ravel(), cols.ravel()])
    rods = grid[grid[:, 1] > 1]
    return Mosaic(np.vstack([grid, grid, rods]),
                  np.concatenate([np.zeros(36, int), np.ones(36, int), np.full(len(rods), ROD)]), 3)


def test_rod_pathway_gives_one_output_for_each_receptor_that_is_not_a_rod():
    stage = rod_pathway(_mosaic(), ROD, share=0.5, sigma_px=0.5)
    assert stage.name == "rods"
    assert stage.in_shape == (96,) and stage.out_shape == (72,)


def test_rod_pathway_leaves_a_uniform_field_as_it_is():
    """The rods' share replaces part of the cone's own signal; it does not add to it."""
    stage = rod_pathway(_mosaic(), ROD, share=0.3, sigma_px=0.5)
    np.testing.assert_allclose(stage.forward(np.ones(96)), 1.0)


def test_rod_pathway_mixes_the_rods_round_a_receptor_into_its_signal(rng):
    mosaic = _mosaic()
    cones, rods = rng.random(72), np.full(24, 0.25)
    out = rod_pathway(mosaic, ROD, share=0.4, sigma_px=0.5).forward(np.concatenate([cones, rods]))
    with_rods = mosaic.positions[:72, 1] > 1
    np.testing.assert_allclose(out[with_rods], 0.6 * cones[with_rods] + 0.4 * 0.25)


def test_a_receptor_with_no_rod_in_reach_keeps_its_own_signal(rng):
    mosaic = _mosaic()
    signal = rng.random(96)
    out = rod_pathway(mosaic, ROD, share=0.4, sigma_px=0.2).forward(signal)
    alone = mosaic.positions[:72, 1] <= 1  # 3 sigmas reach 0.6 px: no rod
    np.testing.assert_array_equal(out[alone], signal[:72][alone])


def test_saturated_rods_pass_nothing_on(rng):
    signal = rng.random(96)
    out = rod_pathway(_mosaic(), ROD, share=0.0, sigma_px=0.5).forward(signal)
    np.testing.assert_array_equal(out, signal[:72])


def test_rod_pathway_adjoint_is_the_exact_transpose(rng):
    stage = rod_pathway(_mosaic(), ROD, share=0.7, sigma_px=1.0)
    x, y = rng.standard_normal(stage.in_shape), rng.standard_normal(stage.out_shape)
    assert np.vdot(stage.forward(x), y) == pytest.approx(np.vdot(x, stage.adjoint(y)), rel=1e-10)


@pytest.mark.parametrize("share", [-0.1, 1.1])
def test_rod_pathway_rejects_a_share_outside_zero_to_one(share):
    with pytest.raises(ValueError, match="share"):
        rod_pathway(_mosaic(), ROD, share=share, sigma_px=0.5)
