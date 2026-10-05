"""Photon noise at the receptors (phase 10 of the human vision audit)."""
from dataclasses import replace

import numpy as np
import pytest

from biovision import species
from biovision.cli import build_parser
from biovision.core.field import VisualField
from biovision.run import run
from biovision.server import Settings
from biovision.species import human, mouse
from biovision.species.eye import assemble, fixate, lit, receptors_each
from biovision.stages.movement import PerLook
from biovision.stages.optics import OpticalBlur
from biovision.stages.photons import PhotonCatch

N = 20000


def test_photon_catch_is_the_identity_as_a_linear_map(rng):
    stage = PhotonCatch(np.full(50, 100.0))
    x = rng.random(50)
    np.testing.assert_array_equal(stage.forward(x), x)
    np.testing.assert_array_equal(stage.adjoint(x), x)
    assert stage.name == "photons" and stage.in_shape == stage.out_shape == (50,)


def test_without_a_generator_the_catch_is_exact(rng):
    x = rng.random(50)
    np.testing.assert_array_equal(PhotonCatch(np.full(50, 100.0)).encode(x), x)


def test_the_catch_is_a_poisson_count_scaled_back_to_the_signal(rng):
    photons = 200.0
    caught = PhotonCatch(np.full(N, photons)).encode(np.full(N, 0.5), rng)
    counts = caught * photons
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-9)  # whole photons
    assert caught.mean() == pytest.approx(0.5, abs=0.002)
    assert caught.var() == pytest.approx(0.5 / photons, rel=0.05)  # variance = signal / photons


def test_more_light_means_less_noise(rng):
    signal = np.full(N, 0.5)
    dim = PhotonCatch(np.full(N, 10.0)).encode(signal, rng)
    bright = PhotonCatch(np.full(N, 1000.0)).encode(signal, rng)
    assert dim.var() == pytest.approx(100.0 * bright.var(), rel=0.1)


def test_photon_catch_rejects_a_receptor_that_catches_nothing():
    with pytest.raises(ValueError, match="photons"):
        PhotonCatch(np.array([10.0, 0.0]))


def test_an_ordinary_stage_encodes_as_it_maps(rng):
    stage = OpticalBlur(1.5, 2, 16)
    x = rng.random(stage.in_shape)
    np.testing.assert_array_equal(stage.encode(x, rng), stage.forward(x))


def test_each_look_catches_its_own_photons(rng):
    stage = PerLook(PhotonCatch(np.full(N, 50.0)), 2)
    signal = np.full((2, N), 0.5)
    caught = stage.encode(signal, rng)
    assert caught.shape == (2, N) and not np.array_equal(caught[0], caught[1])
    np.testing.assert_array_equal(stage.encode(signal), signal)
    np.testing.assert_array_equal(stage.forward(signal), signal)


def test_a_model_receptor_stands_for_its_share_of_the_real_ones(field):
    """Across 60 degrees a pixel holds many cones: 60% L, 30% M, 10% S of them."""
    pipeline = species.get("human")(field)
    mosaic, each = pipeline.metadata["mosaic"], pipeline.metadata["receptors_each"]
    assert each.shape == (len(mosaic),)
    positions = len(mosaic) // 3
    assert each.sum() == pytest.approx(pipeline.metadata["cells_per_position"] * positions)
    shares = np.array([each[mosaic.types == t].sum() for t in range(3)]) / each.sum()
    np.testing.assert_allclose(shares, human.PARAMS.type_fractions, atol=1e-12)
    assert each.max() > 100 * each.min()  # far more cones in a pixel at the fovea


def test_a_receptor_at_least_a_pixel_wide_stands_for_itself():
    for name in ("mouse", "fly"):
        pipeline = species.get(name)(VisualField(64, 60.0))
        np.testing.assert_array_equal(pipeline.metadata["receptors_each"], 1.0)
    narrow = species.get("human")(VisualField(48, 0.25))
    np.testing.assert_array_equal(narrow.metadata["receptors_each"], 1.0)


def test_receptors_each_follows_the_density(field):
    mosaic = species.get("mouse")(field, density=16.0).metadata["mosaic"]
    each = receptors_each(mouse.PARAMS, field, mosaic, density=16.0)
    # 16 times the receptors: a quarter of the spacing, under a pixel, so every
    # position holds both types, half of that pixel's receptors each.
    spacing = field.to_px(mouse.PARAMS.spacing_deg) / 4.0
    np.testing.assert_allclose(each, 0.5 / spacing**2)


def test_lit_puts_a_photon_count_after_the_mosaic(field):
    eye = species.get("human")(field)
    seen = lit(eye, photons_per_s=1000.0, window_s=0.1)
    names = [stage.name for stage in seen.linear_stages]
    assert names.index("photons") == names.index("mosaic") + 1
    assert [n for n in names if n != "photons"] == [s.name for s in eye.linear_stages]
    stage = seen.linear_stages[names.index("photons")]
    np.testing.assert_allclose(stage.photons, 100.0 * eye.metadata["receptors_each"])
    assert seen.n_neurons == eye.n_neurons
    with pytest.raises(ValueError, match="photons_per_s"):
        lit(eye, photons_per_s=0.0, window_s=0.1)


def test_light_does_not_change_what_the_decoder_solves(field, rng):
    eye = species.get("fly")(field)
    x = rng.standard_normal(int(np.prod(eye.in_shape)))
    np.testing.assert_array_equal(lit(eye, 50.0, 0.1).linear_operator().matvec(x),
                                  eye.linear_operator().matvec(x))


def test_transmission_takes_photons_from_a_receptor_type(field):
    """Lens and macular pigment absorb short wavelengths before they reach the cones."""
    params = replace(human.PARAMS, transmission=(1.0, 1.0, 0.25))
    eye = assemble("human", field, params, "", ())
    stage = next(s for s in lit(eye, 1000.0, 0.1).linear_stages if s.name == "photons")
    each, types = eye.metadata["receptors_each"], eye.metadata["mosaic"].types
    np.testing.assert_allclose(stage.photons, 100.0 * each * np.array([1.0, 1.0, 0.25])[types])
    with pytest.raises(ValueError, match="transmission"):
        assemble("human", field, replace(human.PARAMS, transmission=(1.0, 0.5)), "", ())


def test_light_with_several_looks_is_shared_between_them(field, rng):
    eye = lit(species.get("fly")(field), 1000.0, 0.05)
    looking = fixate(eye, 2)
    image = rng.random(eye.in_shape)
    code = looking.encode(image, rng)
    assert code.intermediates["photons"].shape == (2, len(eye.metadata["mosaic"]))
    assert not np.array_equal(code.intermediates["photons"], code.intermediates["mosaic"])


def test_ideal_neurons_see_no_photon_noise(sample):
    """noise=False is the noise-free code: exact spikes and exact light."""
    plain = run(sample, "fly", size_px=32, noise=False)
    dim = run(sample, "fly", size_px=32, noise=False, photons_per_s=10.0)
    np.testing.assert_array_equal(dim.reconstructed, plain.reconstructed)
    assert dim.settings.photons_per_s == 10.0 and plain.settings.photons_per_s is None


def test_dim_light_costs_quality_and_bright_light_does_not(sample):
    plain = run(sample, "fly", size_px=64)
    bright = run(sample, "fly", size_px=64, photons_per_s=1e9)
    dim = run(sample, "fly", size_px=64, photons_per_s=30.0)
    assert bright.metrics["psnr_db"] == pytest.approx(plain.metrics["psnr_db"], abs=0.1)
    assert dim.metrics["psnr_db"] < plain.metrics["psnr_db"] - 0.5
    assert "photons" in dim.code.intermediates and "photons" not in plain.code.intermediates


def test_the_regularization_accounts_for_photon_noise(sample):
    plain = run(sample, "fly", size_px=64)
    bright = run(sample, "fly", size_px=64, photons_per_s=1e9)
    dim = run(sample, "fly", size_px=64, photons_per_s=30.0)
    assert bright.settings.lam == pytest.approx(plain.settings.lam, rel=0.02)  # other spikes drawn
    assert dim.settings.lam > 2.0 * plain.settings.lam


def test_a_longer_window_catches_more_photons(sample):
    short = run(sample, "fly", size_px=32, photons_per_s=100.0, window_ms=100.0)
    long = run(sample, "fly", size_px=32, photons_per_s=100.0, window_ms=1000.0)
    stage = next(s for s in long.pipeline.linear_stages if s.name == "photons")
    np.testing.assert_allclose(stage.photons, 100.0)
    assert long.metrics["psnr_db"] > short.metrics["psnr_db"]


def test_light_and_looks_run_together(sample):
    result = run(sample, "fly", size_px=32, photons_per_s=100.0, looks=2)
    stage = next(s for s in result.pipeline.linear_stages if s.name == "photons")
    np.testing.assert_allclose(stage.stage.photons, 5.0)  # 100 /s for 50 ms each look
    assert np.all(np.isfinite(result.reconstructed))


@pytest.mark.parametrize("level", [0.0, -5.0])
def test_run_rejects_light_that_is_not_positive(sample, level):
    with pytest.raises(ValueError, match="photons_per_s"):
        run(sample, "fly", size_px=32, photons_per_s=level)


def test_the_server_and_the_command_line_take_the_light_level():
    assert Settings(species="fly").run_options()["photons_per_s"] is None
    assert Settings(species="fly", photons_per_s=1e4).run_options()["photons_per_s"] == 1e4
    with pytest.raises(ValueError):
        Settings(species="fly", photons_per_s=0.0)
    args = build_parser().parse_args(["run", "--species", "fly", "--photons", "1e4"])
    assert args.photons == 1e4
    assert build_parser().parse_args(["run", "--species", "fly"]).photons is None
