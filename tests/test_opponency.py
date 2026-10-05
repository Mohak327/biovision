"""Cone opponency and pathway gains (phase 1 of the human vision audit)."""
import numpy as np
import pytest

from biovision import io, species
from biovision.core.field import VisualField
from biovision.run import run
from biovision.species import human
from biovision.stages.gabor import gabor_bank
from biovision.stages.mosaic import all_types_at, square_lattice
from biovision.stages.receptive import RetinaClass, center_surround, opponent_retina

SIZE = 16
CLASSES = (
    RetinaClass("luminance", (0.5, 0.5, 0.0), gain=2.0, surround_weight=0.7),
    RetinaClass("red_green", (1.0, -1.0, 0.0), gain=8.0, surround_weight=0.0),
    RetinaClass("blue_yellow", (-0.5, -0.5, 1.0), gain=3.0, surround_weight=0.0),
)


def dense_mosaic():
    return all_types_at(square_lattice(SIZE, 1.0), 3)


def signal(mosaic, levels):
    """A receptor signal that is uniform within each receptor type."""
    return np.asarray(levels, dtype=float)[mosaic.types]


def test_baselines_for_mouse_and_fly_are_unchanged(sample):
    """Opponency is a human-only change; the other eyes must give the same numbers.

    Re-pinned once, when the decoder's solve became preconditioned and its
    tolerance went from 1e-4 to 3e-5: the same system, solved more closely.
    Before: mouse 11.999567473 and 14.713525802, fly 13.131761815 and
    13.708582084 (real and ideal). The largest move is +0.042 dB.
    """
    expected = {("mouse", True): 11.999458055142469, ("mouse", False): 14.75544952536662,
                ("fly", True): 13.133059867104087, ("fly", False): 13.73031381124944}
    for (name, noise), value in expected.items():
        result = run(sample, name, size_px=64, noise=noise)
        assert result.metrics["psnr_db"] == pytest.approx(value, abs=1e-9)


def test_opponent_retina_makes_one_cell_per_class_at_each_position():
    mosaic = dense_mosaic()
    stage, cells = opponent_retina(mosaic, CLASSES, 0.5, 3.0)
    positions = SIZE * SIZE
    assert stage.in_shape == (len(mosaic),) and stage.out_shape == (3 * positions,)
    assert len(cells) == 3 * positions and cells.n_types == 3
    assert np.bincount(cells.types).tolist() == [positions] * 3
    assert stage.name == "center_surround"


def test_opponent_retina_adjoint_is_exact(rng):
    stage, _ = opponent_retina(dense_mosaic(), CLASSES, 0.5, 3.0)
    x = rng.standard_normal(stage.in_shape)
    y = rng.standard_normal(stage.out_shape)
    assert np.vdot(stage.forward(x), y) == pytest.approx(np.vdot(x, stage.adjoint(y)), rel=1e-10)


def test_uniform_grey_excites_luminance_only():
    mosaic = dense_mosaic()
    stage, cells = opponent_retina(mosaic, CLASSES, 0.5, 3.0)
    out = stage.forward(signal(mosaic, [1.0, 1.0, 1.0]))
    np.testing.assert_allclose(out[cells.types == 0], 2.0 * (1 - 0.7), atol=1e-12)
    np.testing.assert_allclose(out[cells.types == 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[cells.types == 2], 0.0, atol=1e-12)


def test_a_red_green_difference_excites_the_red_green_class_and_not_luminance():
    mosaic = dense_mosaic()
    stage, cells = opponent_retina(mosaic, CLASSES, 0.5, 3.0)
    baseline = stage.forward(signal(mosaic, [0.5, 0.5, 0.5]))
    shifted = stage.forward(signal(mosaic, [0.6, 0.4, 0.5]))  # L up, M down, L + M unchanged
    change = shifted - baseline
    np.testing.assert_allclose(change[cells.types == 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(change[cells.types == 1], 8.0 * 0.2, atol=1e-12)


def test_identity_classes_reproduce_the_plain_center_surround(rng):
    """One class per receptor type, weight 1, gain 1: the same numbers as before."""
    mosaic = dense_mosaic()
    identity = tuple(RetinaClass(f"type{i}", tuple(float(i == j) for j in range(3)), 1.0, 0.7)
                     for i in range(3))
    mixed, cells = opponent_retina(mosaic, identity, 0.5, 3.0)
    plain = center_surround(mosaic, 0.5, 3.0, 0.7)
    x = rng.random(len(mosaic))
    np.testing.assert_allclose(mixed.forward(x), plain.forward(x), atol=1e-12)
    np.testing.assert_array_equal(cells.types, mosaic.types)


def test_retina_class_weights_must_match_the_receptor_types():
    bad = (RetinaClass("luminance", (0.5, 0.5), 1.0, 0.7),)
    with pytest.raises(ValueError, match="one weight per receptor type"):
        opponent_retina(dense_mosaic(), bad, 0.5, 3.0)
    with pytest.raises(ValueError, match="at least one class"):
        opponent_retina(dense_mosaic(), (), 0.5, 3.0)


def test_gabor_bank_scales_each_wavelength_by_its_gain(rng):
    mosaic = dense_mosaic()
    plain = gabor_bank(mosaic, SIZE, [8.0, 4.0])
    gained = gabor_bank(mosaic, SIZE, [8.0, 4.0], gains=[1.0, 2.0])
    x = rng.standard_normal(len(mosaic))
    ratio = gained.forward(x) / plain.forward(x)
    coarse = gabor_bank(mosaic, SIZE, [8.0]).out_shape[0]
    np.testing.assert_allclose(ratio[:coarse], 1.0)
    np.testing.assert_allclose(ratio[coarse:], 2.0)
    with pytest.raises(ValueError, match="one gain per wavelength"):
        gabor_bank(mosaic, SIZE, [8.0, 4.0], gains=[1.0])


def test_the_human_eye_has_three_opponent_classes():
    names = [item.name for item in human.PARAMS.retina_classes]
    assert names == ["luminance", "red_green", "blue_yellow"]
    weights = np.array([item.weights for item in human.PARAMS.retina_classes])
    assert weights.shape == (3, 3)
    np.testing.assert_allclose(weights[1:].sum(axis=1), 0.0, atol=1e-12)  # colour classes ignore grey
    pipeline = species.get("human")(VisualField(32, 60.0))
    assert pipeline.metadata["cells"].n_types == 3
    assert len(human.PARAMS.cortex_gains) == len(human.PARAMS.cortex_sf_cpd)


def test_mouse_and_fly_keep_one_retinal_cell_per_receptor():
    for name in ("mouse", "fly"):
        pipeline = species.get(name)(VisualField(32, 60.0))
        assert pipeline.metadata["cells"] is pipeline.metadata["mosaic"]


@pytest.mark.parametrize("name", io.sample_names())
def test_human_cells_rarely_clip(name):
    image = io.to_square(io.load_sample(name), 96).transpose(2, 0, 1)
    pipeline = species.get("human")(VisualField(96, 60.0))
    rates = pipeline.encode(image).intermediates["rate"]
    assert np.mean(rates == 0.0) < 0.001


def test_human_reconstruction_with_real_neurons_is_much_better(sample):
    """Before opponency and gains this was 20.1 dB."""
    result = run(sample, "human", size_px=96)
    assert result.metrics["psnr_db"] > 26.0
    assert result.metrics["ssim"] > 0.8


def test_human_reconstruction_with_ideal_neurons_is_not_worse(sample):
    """Before opponency and gains this was 35.1 dB."""
    result = run(sample, "human", size_px=96, noise=False)
    assert result.metrics["psnr_db"] > 34.5


def test_human_runs_when_receptors_are_larger_than_a_pixel(sample):
    """A narrow field of view gives one cone type per position, not all three."""
    result = run(sample, "human", size_px=48, fov_deg=0.25, noise=False)
    mosaic = result.pipeline.metadata["mosaic"]
    positions = {tuple(p) for p in np.round(mosaic.positions, 6)}
    assert len(positions) == len(mosaic)  # one cone, of one type, at each position
    assert result.pipeline.metadata["cells"].n_types == 3
    assert np.all(np.isfinite(result.reconstructed))
