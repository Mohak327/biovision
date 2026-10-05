"""Real cell types (phase 6 of the human vision audit): separate ON and OFF cells."""
from dataclasses import replace

import numpy as np
import pytest

from biovision.core.decoder import Decoder
from biovision.core.field import VisualField
from biovision.core.pipeline import Pipeline
from biovision import run as run_module
from biovision import species
from biovision.run import run, spike_counts
from biovision.species import fly, mouse
from biovision.species.eye import assemble
from biovision.stages.color import ColorProjection
from biovision.stages.nonlinearity import LinearRectified, OnOffPair
from biovision.stages.optics import OpticalBlur
from biovision.stages.spiking import PoissonSpikes

SIZE = 8


def test_an_increment_fires_the_on_cell_and_a_decrement_the_off_cell():
    stage = OnOffPair(swing_hz=100.0, gain=2.0, spontaneous_hz=3.0)
    rates = stage.forward(np.array([0.25, 0.0, -0.5]))
    assert rates.shape == (3, 2)  # an ON and an OFF cell for each signal
    np.testing.assert_allclose(rates, [[53.0, 3.0], [3.0, 3.0], [3.0, 103.0]])


def test_the_pair_gives_back_any_signal_exactly(rng):
    """Unlike one cell around a resting rate, the pair has no signal it clips."""
    stage = OnOffPair(swing_hz=100.0, gain=2.5, spontaneous_hz=1.0)
    x = rng.uniform(-3.0, 3.0, (4, 50))
    np.testing.assert_allclose(stage.inverse(stage.forward(x)), x, atol=1e-12)


def test_the_pair_rejects_invalid_parameters():
    for bad in ((0.0, 1.0, 0.0), (100.0, 0.0, 0.0), (100.0, 1.0, -1.0)):
        with pytest.raises(ValueError):
            OnOffPair(*bad)


def pair_pipeline(window_s=0.2):
    matrix = [[0.5, 0.4, 0.1], [0.1, 0.6, 0.3], [0.0, 0.2, 0.8]]
    return Pipeline("pair", VisualField(SIZE, 60.0), (
        ColorProjection(matrix, SIZE), OpticalBlur(0.8, 3, SIZE),
        OnOffPair(100.0, 2.0, spontaneous_hz=5.0), PoissonSpikes(window_s)))


def test_noise_variance_adds_the_variances_of_the_two_cells(rng):
    """drive = (on - off) / (T * swing * gain): both cells' count variances add."""
    pipeline = pair_pipeline()
    image = rng.random((3, SIZE, SIZE))
    code = pipeline.encode(image, rng)
    assert code.responses.shape == (3, SIZE, SIZE, 2)
    decoder = Decoder(pipeline, 1.0)
    expected = code.responses.sum(axis=-1).mean() / (0.2 * 100.0 * 2.0) ** 2
    assert decoder.noise_variance(code) == pytest.approx(expected)
    drives = np.stack([decoder.linear_drive(pipeline.encode(image, rng).responses)
                       for _ in range(300)])
    assert drives.var(axis=0).mean() == pytest.approx(expected, rel=0.1)


def test_a_pair_decodes_to_what_one_cell_does_without_noise(rng):
    """The linear operator is the same, so ideal neurons rebuild the same picture."""
    image = rng.uniform(0.3, 0.7, (3, SIZE, SIZE))
    pair = pair_pipeline()
    single = pair.replace(LinearRectified(100.0, 2.0))
    rebuilt = [Decoder(p, 1e-3).decode(p.encode(image)).image for p in (pair, single)]
    np.testing.assert_allclose(rebuilt[0], rebuilt[1], atol=1e-9)


def eye(params, size=32):
    return assemble("eye", VisualField(size, 60.0), params, "", ())


def test_an_eye_with_a_spontaneous_rate_has_an_on_and_an_off_cell_for_each_signal():
    """Each cell of the pair stands for half the real cells one cell stood for,
    and uses the whole firing range for its own sign, so the spike budget and
    the rate a signal adds are the same."""
    single = eye(replace(mouse.PARAMS, spontaneous_hz=None))
    pair = eye(replace(mouse.PARAMS, spontaneous_hz=4.0))
    rest, rate = single.pointwise_stages[0], pair.pointwise_stages[0]
    assert isinstance(rest, LinearRectified) and isinstance(rate, OnOffPair)
    real_cells = pair.metadata["cells_per_neuron"]
    assert rate.swing_hz == rest.rest_hz and rate.gain == rest.gain
    assert rate.spontaneous_hz == pytest.approx(4.0 * real_cells / 2)
    assert pair.out_shape == single.out_shape  # the linear stages are the same


def test_a_run_counts_both_cells_of_each_pair(sample, monkeypatch):
    paired = replace(mouse.PARAMS, spontaneous_hz=1.0)
    alone = replace(mouse.PARAMS, spontaneous_hz=None)
    results = []
    for params in (alone, paired):
        monkeypatch.setattr(run_module, "build_pipeline",
                            lambda name, size, fov, *density, params=params: eye(params, size))
        results.append(run(sample, "mouse", size_px=32))
    one, two = results
    assert two.metrics["neurons"] == 2 * one.metrics["neurons"]
    assert two.code.responses.shape == (*one.code.responses.shape, 2)
    assert spike_counts(two).size == two.metrics["neurons"]
    assert two.metrics["compression_ratio"] == 2 * one.metrics["compression_ratio"]
    assert two.metrics["mean_spikes"] < one.metrics["mean_spikes"] / 2  # no resting rate to pay


def test_the_mammals_have_on_and_off_cells_and_the_fly_does_not():
    """Cortex simple cells come in pairs of opposite sign and are nearly silent
    at rest. A fly's lamina cells answer to both signs and do not spike."""
    field = VisualField(32, 60.0)
    for name in ("human", "mouse"):
        assert isinstance(species.get(name)(field).pointwise_stages[0], OnOffPair)
    assert isinstance(species.get("fly")(field).pointwise_stages[0], LinearRectified)
    assert fly.PARAMS.spontaneous_hz is None


def test_human_reconstruction_with_on_and_off_cells(sample):
    """With one cell around a resting rate this was 29.9 dB real and 34.9 ideal at 96 px."""
    assert run(sample, "human", size_px=96).metrics["psnr_db"] > 34.0
    assert run(sample, "human", size_px=96, noise=False).metrics["psnr_db"] > 36.5
