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
    with pytest.raises(ValueError, match="neuron_density must be positive"):
        run(sample, "mouse", neuron_density=0.0, **SMALL)
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


@pytest.mark.parametrize("name, options", [
    ("fly", {}), ("mouse", {}), ("human", {}),
    ("human", {"looks": 2}), ("human", {"photons_per_s": 1e4}),
])
def test_every_stage_reports_a_preview_picture(sample, name, options):
    """Stages whose output is not a picture are projected back into picture space."""
    events = []
    run(sample, name, on_progress=events.append, **options, **SMALL)
    stage_events = [event for event in events if event.iteration == 0]
    assert len(stage_events) == len(events[0].stages) - 1
    for event in stage_events:
        assert event.image is not None, event.stages[event.current]
        assert event.image.shape == (32, 32, 3)
        assert np.all(np.isfinite(event.image))
        assert event.image.min() >= 0.0 and event.image.max() <= 1.0
        assert event.fraction == 0.0


def test_stage_previews_differ_from_stage_to_stage(sample):
    """The mosaic, the retina and the spikes each show something of their own."""
    events = []
    run(sample, "mouse", on_progress=events.append, **SMALL)
    by_stage = {event.stages[event.current]: event.image for event in events if event.iteration == 0}
    assert not np.allclose(by_stage["mosaic"], by_stage["center_surround"])
    assert not np.allclose(by_stage["rate"], by_stage["spikes"])  # the spikes carry noise
    assert by_stage["mosaic"].std() > 0.05  # a picture, not a blank


def test_decoding_progress_only_grows_and_ends_done(sample):
    events = []
    result = run(sample, "mouse", noise=False, on_progress=events.append, **SMALL)
    fractions = [event.fraction for event in events if event.iteration > 0]
    assert fractions == sorted(fractions)
    assert 0.0 <= fractions[0] < fractions[-1]
    assert result.reconstruction.converged and fractions[-1] == 1.0


def test_progress_reporting_does_not_change_the_result(sample):
    plain = run(sample, "fly", seed=5, **SMALL)
    watched = run(sample, "fly", seed=5, on_progress=lambda event: None, **SMALL)
    np.testing.assert_array_equal(plain.reconstructed, watched.reconstructed)


def test_neuron_density_adds_neurons_and_is_recorded(sample):
    base = run(sample, "mouse", size_px=64)
    dense = run(sample, "mouse", neuron_density=4.0, size_px=64)
    assert base.settings.neuron_density == 1.0 and dense.settings.neuron_density == 4.0
    assert dense.metrics["neurons"] > 3 * base.metrics["neurons"]
    assert dense.metrics["receptors"] == base.metrics["receptors"]
    assert dense.metrics["psnr_db"] > base.metrics["psnr_db"]  # more spikes to average


def test_the_rate_stage_reports_its_real_curve_and_where_the_responses_fall(sample):
    events = []
    result = run(sample, "human", on_progress=events.append, **SMALL)
    by_stage = {event.stages[event.current]: event for event in events if event.iteration == 0}
    assert by_stage["optics"].plot is None and by_stage["center_surround"].plot is None
    plot = by_stage["rate"].plot
    stage = result.pipeline.pointwise_stages[0]
    response = np.array(plot["response"])
    rates = stage.forward(response)  # the curve is the stage itself, not a drawing
    np.testing.assert_allclose(plot["rates"]["ON cells"], rates[..., 0])
    np.testing.assert_allclose(plot["rates"]["OFF cells"], rates[..., 1])
    drive = result.code.intermediates[result.pipeline.linear_stages[-1].name]
    assert response[0] < 0 < response[-1]
    assert response[0] <= np.percentile(drive, 1) and response[-1] >= np.percentile(drive, 99)
    assert len(plot["cells"]) == len(response) - 1
    assert 0.95 * drive.size < sum(plot["cells"]) <= drive.size  # nearly every response is in range


def test_the_spike_stage_reports_the_counts_its_cells_fired(sample):
    events = []
    result = run(sample, "human", on_progress=events.append, **SMALL)
    plot = next(event.plot for event in events if event.stages[event.current] == "spikes")
    counts = np.asarray(result.code.responses)
    assert sum(plot["cells"]) == counts.size
    assert len(plot["edges"]) == len(plot["cells"]) + 1
    assert plot["edges"][0] == counts.min() and plot["edges"][-1] == counts.max()


def test_an_eye_with_one_cell_per_signal_reports_a_single_rate_curve(sample):
    events = []
    result = run(sample, "fly", on_progress=events.append, **SMALL)
    plot = next(event.plot for event in events if event.stages[event.current] == "rate")
    assert list(plot["rates"]) == ["cells"]
    stage = result.pipeline.pointwise_stages[0]
    np.testing.assert_allclose(plot["rates"]["cells"], stage.forward(np.array(plot["response"])))
