"""Rods and night vision (phase 14 of the human vision audit)."""
from dataclasses import replace

import numpy as np
import pytest

import biovision.run as run_module
from biovision import species
from biovision.core.field import VisualField
from biovision.run import run
from biovision.species import fly, human, mouse
from biovision.species.eye import assemble, lit, rod_share, with_rods
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


# The eye: where its rods are, what they catch and when they work.

RODS = human.PARAMS.rods


def _stage(pipeline, name):
    return next(stage for stage in pipeline.linear_stages if stage.name == name)


def test_the_human_eye_has_rods_and_the_mouse_and_fly_models_do_not():
    assert RODS is not None
    assert mouse.PARAMS.rods is None and fly.PARAMS.rods is None
    assert sum(RODS.color) == pytest.approx(1.0)  # adapted to its light, like each cone


def test_an_eye_without_rods_is_returned_as_it_is(field):
    eye = species.get("fly")(field)
    assert with_rods(eye, 100.0) is eye


def test_with_rods_adds_rod_receptors_after_the_cones(field):
    eye = species.get("human")(field)
    seen = with_rods(eye, 100.0)
    cones, mosaic = eye.metadata["mosaic"], seen.metadata["mosaic"]
    assert seen.metadata["params"].receptor_names == ("L", "M", "S", "rod")
    assert mosaic.n_types == 4 and len(mosaic) > len(cones)
    np.testing.assert_array_equal(mosaic.positions[:len(cones)], cones.positions)
    np.testing.assert_array_equal(mosaic.types[:len(cones)], cones.types)
    assert np.all(mosaic.types[len(cones):] == 3)


def test_rods_join_before_the_retina_and_add_no_cells(field):
    eye = species.get("human")(field)
    seen = with_rods(eye, 100.0)
    names = [stage.name for stage in seen.linear_stages]
    assert names == ["color", "optics", "mosaic", "rods", "center_surround", "gabor"]
    assert seen.in_shape == eye.in_shape and seen.out_shape == eye.out_shape
    assert seen.pointwise_stages == eye.pointwise_stages
    lit_names = [stage.name for stage in lit(seen, 100.0, 0.1).linear_stages]
    assert lit_names[2:5] == ["mosaic", "photons", "rods"]  # a rod counts photons, then saturates


def test_there_are_no_rods_at_the_centre_of_gaze():
    field = VisualField(64, 10.0)  # the rod-free zone is 4 pixels in radius
    mosaic = with_rods(species.get("human")(field), 100.0).metadata["mosaic"]
    rods = mosaic.positions[mosaic.types == 3]
    eccentricity_deg = np.linalg.norm(rods - 31.5, axis=1) * 10.0 / 64.0
    pixel_deg = 10.0 / 64.0
    assert RODS.absent_within_deg - pixel_deg < eccentricity_deg.min() < RODS.absent_within_deg + pixel_deg
    assert eccentricity_deg.max() > 6.0  # and they reach the corners


def test_a_model_rod_stands_for_the_rods_in_its_pixel(field):
    eye = species.get("human")(field)
    each = with_rods(eye, 100.0).metadata["receptors_each"]
    cones = len(eye.metadata["mosaic"])
    np.testing.assert_array_equal(each[:cones], eye.metadata["receptors_each"])
    in_pixel = 1.0 / field.to_px(RODS.spacing_deg) ** 2
    np.testing.assert_allclose(each[cones:], in_pixel)
    dense = with_rods(species.get("human")(field, density=4.0), 100.0)
    np.testing.assert_allclose(dense.metadata["receptors_each"][-1], 4.0 * in_pixel)


def test_a_rod_catches_fewer_photons_than_a_cone(field):
    seen = with_rods(species.get("human")(field), 1000.0)
    photons = _stage(lit(seen, 1000.0, 0.1), "photons").photons
    each, types = seen.metadata["receptors_each"], seen.metadata["mosaic"].types
    np.testing.assert_allclose(photons, 100.0 * each * np.where(types == 3, RODS.catch, 1.0))


def test_the_rods_share_falls_as_the_light_saturates_them():
    half = RODS.saturation_photons_per_s / RODS.catch  # the cone's catch when a rod's is at its limit
    assert rod_share(RODS, half) == pytest.approx(0.5)
    assert rod_share(RODS, 1e-3 * half) > 0.99
    assert rod_share(RODS, 1e4 * half) < 1e-3
    assert rod_share(RODS, 1e5) < 0.05 < 0.5 < rod_share(RODS, 1e3)  # a lit room; 1 cd/m2


def test_the_rod_stage_carries_the_share_for_the_light(field):
    half = RODS.saturation_photons_per_s / RODS.catch
    seen = with_rods(species.get("human")(field), half)
    rods_only = (seen.metadata["mosaic"].types == 3).astype(float)
    np.testing.assert_allclose(_stage(seen, "rods").forward(rods_only), 0.5)


def test_in_bright_light_the_eye_with_rods_is_the_eye_without(field, rng):
    eye = species.get("human")(field)
    x = rng.random(int(np.prod(eye.in_shape)))
    np.testing.assert_allclose(with_rods(eye, 1e15).linear_operator().matvec(x),
                               eye.linear_operator().matvec(x), atol=1e-8)


def test_rods_carry_no_colour(field):
    """A rod's signal goes to every cone alike, so the colour cells cancel it."""
    seen = with_rods(species.get("human")(field), 1.0)  # nearly all rods
    red = np.zeros(seen.in_shape)
    red[0] = 1.0
    drive = seen.linear_operator().matvec(red.ravel())
    assert np.abs(drive).max() < 0.05  # rods do not see red, and little of the cones is left


def test_the_eye_with_rods_keeps_an_exact_adjoint(field, rng):
    seen = lit(with_rods(species.get("human")(field), 300.0), 300.0, 0.1)
    a = seen.linear_operator()
    x, y = rng.standard_normal(a.shape[1]), rng.standard_normal(a.shape[0])
    assert np.vdot(a.matvec(x), y) == pytest.approx(np.vdot(x, a.rmatvec(y)), rel=1e-9)


def test_an_eye_with_one_blur_gives_its_rods_that_blur(field):
    params = replace(human.PARAMS, chromatic_defocus_d=(), pupil_mm=0.0)
    eye = species.get("human")(field)
    seen = with_rods(replace(eye, metadata={**eye.metadata, "params": params}), 100.0)
    assert seen.metadata["params"].chromatic_defocus_d == ()


# The run: the light level decides who is working.


def test_a_run_brings_in_the_rods_only_when_it_has_a_light_level(sample):
    plain = run(sample, "human", size_px=32)
    dim = run(sample, "human", size_px=32, photons_per_s=100.0)
    assert "rods" not in plain.code.intermediates and "rods" in dim.code.intermediates
    assert plain.pipeline.metadata["params"] is human.PARAMS
    assert dim.metrics["receptors"] > plain.metrics["receptors"]
    assert dim.metrics["neurons"] == plain.metrics["neurons"]  # rods add no cells


def test_a_fly_in_dim_light_has_no_rods(sample):
    dim = run(sample, "fly", size_px=32, photons_per_s=100.0)
    assert "rods" not in dim.code.intermediates


def test_rods_help_in_the_dark(sample, monkeypatch):
    field = VisualField(48, 60.0)
    night = run(sample, "human", size_px=48, photons_per_s=3.0)
    cones_only = assemble("human", field, replace(human.PARAMS, rods=None), "", ())
    monkeypatch.setattr(run_module, "build_pipeline", lambda *args: cones_only)
    without = run(sample, "human", size_px=48, photons_per_s=3.0)
    assert "rods" not in without.code.intermediates
    assert night.metrics["psnr_db"] > without.metrics["psnr_db"] + 1.0


def test_rods_work_with_several_looks(sample):
    result = run(sample, "human", size_px=32, photons_per_s=100.0, looks=2)
    assert result.code.intermediates["rods"].shape[0] == 2
    assert result.reconstruction.converged
