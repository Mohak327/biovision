import numpy as np
import pytest

from biovision.analysis import compare_species, sweep_density, sweep_lambda, sweep_window
from biovision.core.metrics import psnr, radial_power_spectrum, ssim
from biovision.run import run

SMALL = dict(size_px=32)


def test_metrics_on_known_images(rng):
    image = rng.random((16, 16, 3))
    assert psnr(image, image) == float("inf")
    assert ssim(image, image) == pytest.approx(1.0)
    assert psnr(np.zeros((4, 4, 3)), np.full((4, 4, 3), 0.1)) == pytest.approx(20.0)
    frequency, power = radial_power_spectrum(image)
    assert frequency.shape == power.shape == (8,)


def test_run_returns_images_metrics_and_settings(sample):
    result = run(sample, "mouse", noise=False, **SMALL)
    assert result.original.shape == result.reconstructed.shape == (32, 32, 3)
    assert set(result.metrics) == {"psnr_db", "ssim", "neurons", "receptors",
                                   "compression_ratio", "mean_spikes"}
    assert result.settings.species == "mouse" and result.settings.lam == 1e-4
    assert result.runtime_s > 0


def test_noise_free_quality_ranks_human_mouse_fly(sample):
    results = compare_species(sample, noise=False, size_px=64)
    quality = {name: r.metrics["psnr_db"] for name, r in results.items()}
    assert quality["human"] > quality["mouse"] > quality["fly"]
    assert quality["human"] > quality["fly"] + 5.0


def test_same_seed_reproduces_and_different_seed_differs(sample):
    first = run(sample, "fly", seed=3, **SMALL)
    again = run(sample, "fly", seed=3, **SMALL)
    other = run(sample, "fly", seed=4, **SMALL)
    np.testing.assert_array_equal(first.reconstructed, again.reconstructed)
    assert not np.array_equal(first.reconstructed, other.reconstructed)


def test_automatic_lambda_grows_with_noise(sample):
    quiet = run(sample, "mouse", window_ms=1000.0, **SMALL).settings.lam
    loud = run(sample, "mouse", window_ms=10.0, **SMALL).settings.lam
    clean = run(sample, "mouse", noise=False, **SMALL).settings.lam
    assert loud > quiet > clean == 1e-4


def test_longer_windows_do_not_reduce_quality(sample):
    rows = sweep_window(sample, "mouse", windows_ms=(10.0, 100.0, 1000.0),
                        seeds=(0, 1, 2), **SMALL)
    means = [row["psnr_mean"] for row in rows]
    assert means[0] < means[1] < means[2]
    assert all(row["psnr_std"] >= 0 for row in rows)


def test_sweep_lambda_returns_one_row_per_value(sample):
    rows = sweep_lambda(sample, "fly", lams=(1e-3, 1e-1), noise=False, **SMALL)
    assert [row["lam"] for row in rows] == [1e-3, 1e-1]
    assert all(np.isfinite(row["psnr_db"]) for row in rows)


def test_a_uniform_image_reconstructs_as_uniform():
    flat = np.full((20, 20, 3), 0.5)
    result = run(flat, "mouse", noise=False, **SMALL)
    assert np.all(np.isfinite(result.reconstructed))
    assert result.reconstructed.std() < 0.02
    assert result.reconstructed.mean() == pytest.approx(0.5, abs=0.02)


def test_a_black_image_does_not_break_the_metrics():
    result = run(np.zeros((20, 20, 3)), "fly", **SMALL)
    assert np.all(np.isfinite(result.reconstructed))
    assert np.isfinite(result.metrics["ssim"])


@pytest.mark.parametrize("name", ["fly", "human", "mouse"])
@pytest.mark.parametrize("fov", [5.0, 170.0])
def test_extreme_fields_of_view_still_run(sample, name, fov):
    result = run(sample, name, fov_deg=fov, noise=False, size_px=24)
    assert np.all(np.isfinite(result.reconstructed))
    assert result.metrics["neurons"] >= 1


def test_run_rejects_bad_settings(sample):
    with pytest.raises(KeyError, match="unknown species 'cat'; registered: fly, human, mouse"):
        run(sample, "cat", **SMALL)
    with pytest.raises(ValueError, match="window_ms must be positive"):
        run(sample, "fly", window_ms=0.0, **SMALL)
    with pytest.raises(ValueError, match="lam must be positive"):
        run(sample, "fly", lam=0.0, **SMALL)
    with pytest.raises(ValueError, match="density must be positive"):
        run(sample, "fly", density=0.0, **SMALL)
    with pytest.raises(ValueError, match="size_px must be greater than 1"):
        run(sample, "fly", size_px=1)


def test_density_scales_the_receptor_count_and_is_recorded(sample):
    base = run(sample, "fly", noise=False, **SMALL)
    dense = run(sample, "fly", noise=False, density=4.0, **SMALL)
    assert base.settings.density == 1.0 and dense.settings.density == 4.0
    assert dense.metrics["receptors"] > 3 * base.metrics["receptors"]
    assert dense.metrics["psnr_db"] > base.metrics["psnr_db"]


def test_more_cells_per_pixel_reduce_noise_for_the_human_eye(sample):
    """Human cones are already finer than a pixel, so density adds spikes, not detail."""
    base = run(sample, "human", **SMALL)
    dense = run(sample, "human", density=16.0, **SMALL)
    assert dense.metrics["receptors"] == base.metrics["receptors"]
    assert dense.metrics["mean_spikes"] > 10 * base.metrics["mean_spikes"]
    assert dense.metrics["psnr_db"] > base.metrics["psnr_db"]


def test_sweep_density_returns_one_row_per_value(sample):
    rows = sweep_density(sample, "fly", densities=(1.0, 4.0), noise=False, **SMALL)
    assert [row["density"] for row in rows] == [1.0, 4.0]
    assert rows[1]["receptors"] > rows[0]["receptors"]
    assert set(rows[0]) == {"species", "density", "receptors", "neurons", "psnr_db", "ssim"}


def test_run_reports_each_stage_then_the_decoding_frames(sample):
    events = []
    result = run(sample, "mouse", noise=False, on_progress=events.append, **SMALL)
    stages = ("color", "optics", "mosaic", "center_surround", "gabor", "rate", "spikes",
              "decoding")
    assert all(event.stages == stages for event in events)
    encode = [event for event in events if event.iteration == 0]
    decode = [event for event in events if event.iteration > 0]
    assert [event.current for event in encode] == list(range(7))
    assert all(event.current == 7 for event in decode)
    assert [event.iteration for event in decode] == list(range(1, len(decode) + 1))
    assert len(decode) == result.reconstruction.iterations
    assert all(event.image.shape == (32, 32, 3) for event in decode)
    np.testing.assert_array_equal(decode[-1].image, result.reconstructed)


def test_stage_events_carry_an_image_only_for_image_like_stages(sample):
    events = []
    run(sample, "fly", on_progress=events.append, **SMALL)
    by_stage = {event.stages[event.current]: event for event in events if event.iteration == 0}
    assert by_stage["color"].image.shape == (32, 32, 3)
    assert by_stage["optics"].image.shape == (32, 32, 3)
    assert by_stage["mosaic"].image is None and by_stage["spikes"].image is None


def test_progress_reporting_does_not_change_the_result(sample):
    plain = run(sample, "fly", seed=5, **SMALL)
    watched = run(sample, "fly", seed=5, on_progress=lambda event: None, **SMALL)
    np.testing.assert_array_equal(plain.reconstructed, watched.reconstructed)
