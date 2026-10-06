"""Realistic spike statistics (phase 8 of the human vision audit): counts more regular than Poisson."""
from dataclasses import replace

import numpy as np
import pytest

from biovision import run as run_module
from biovision import species
from biovision.run import run
from biovision.species import mouse
from biovision.species.eye import assemble
from biovision.core.decoder import Decoder
from biovision.core.field import VisualField
from biovision.core.pipeline import Pipeline
from biovision.stages.color import ColorProjection
from biovision.stages.nonlinearity import LinearRectified
from biovision.stages.spiking import PoissonSpikes

SIZE = 8


def test_a_fano_factor_below_one_makes_counts_more_regular_with_the_same_mean(rng):
    stage = PoissonSpikes(window_s=0.1, fano=0.5)
    rates = np.full(40000, 50.0)  # a mean of 5 spikes: 10 slots, each filled half the time
    np.testing.assert_allclose(stage.forward(rates), 5.0)  # without a generator, the mean
    counts = stage.forward(rates, rng)
    assert np.all(counts == np.round(counts)) and counts.min() >= 0
    assert counts.mean() == pytest.approx(5.0, abs=0.05)
    assert counts.var() == pytest.approx(0.5 * 5.0, rel=0.05)


def test_a_fano_factor_of_one_is_the_poisson_draw_bit_for_bit():
    rates = np.random.default_rng(1).uniform(0.0, 300.0, 5000)
    counts = PoissonSpikes(0.1, fano=1.0).forward(rates, np.random.default_rng(7))
    np.testing.assert_array_equal(counts, np.random.default_rng(7).poisson(rates * 0.1))
    assert PoissonSpikes(0.1).fano == 1.0


def test_a_cell_that_rarely_fires_is_as_irregular_as_poisson(rng):
    """Below 1 - fano expected spikes there is one slot: a spike or none."""
    counts = PoissonSpikes(0.1, fano=0.5).forward(np.full(40000, 2.0), rng)  # mean 0.2
    assert set(np.unique(counts)) == {0.0, 1.0}
    assert counts.mean() == pytest.approx(0.2, abs=0.01)
    assert PoissonSpikes(0.1, fano=0.5).forward(np.zeros(5), rng).tolist() == [0.0] * 5


@pytest.mark.parametrize("fano", [0.0, -0.5, 1.5])
def test_the_fano_factor_must_be_above_zero_and_at_most_one(fano):
    with pytest.raises(ValueError, match="fano"):
        PoissonSpikes(0.1, fano=fano)


@pytest.mark.parametrize("fano, mean", [(1.0, 5.0), (0.5, 5.0), (0.3, 7.3), (0.6, 20.4)])
def test_the_stage_gives_the_variance_of_its_counts(fano, mean, rng):
    stage = PoissonSpikes(0.1, fano=fano)
    counts = stage.forward(np.full(60000, mean / 0.1), rng)
    assert stage.variance(counts).mean() == pytest.approx(counts.var(), rel=0.05)
    if fano == 1.0:
        assert stage.variance(counts) is counts  # Poisson: the count is its own estimate


def test_another_window_keeps_the_regularity():
    longer = PoissonSpikes(0.1, fano=0.4, name="counts").lasting(0.5)
    assert (longer.window_s, longer.fano, longer.name) == (0.5, 0.4, "counts")


def test_the_decoder_counts_the_noise_regular_spikes_really_have(rng):
    matrix = [[0.5, 0.4, 0.1], [0.1, 0.6, 0.3], [0.0, 0.2, 0.8]]
    pipeline = Pipeline("regular", VisualField(SIZE, 60.0), (
        ColorProjection(matrix, SIZE), LinearRectified(100.0, 1.0), PoissonSpikes(0.2, fano=0.5)))
    flat = np.full((3, SIZE, SIZE), 0.5)  # 30 spikes a cell: 60 slots
    decoder = Decoder(pipeline, 1.0)
    code = pipeline.encode(flat, rng)
    expected = 0.5 * code.responses.mean() / (0.2 * 100.0 * 1.0) ** 2
    assert decoder.noise_variance(code) == pytest.approx(expected)
    drives = np.stack([decoder.linear_drive(pipeline.encode(flat, rng).responses)
                       for _ in range(300)])
    assert drives.var(axis=0).mean() == pytest.approx(expected, rel=0.1)


def test_a_run_keeps_the_eyes_fano_factor_at_any_window(sample, monkeypatch):
    regular = assemble("mouse", VisualField(32, 60.0), replace(mouse.PARAMS, fano=0.5), "", ())
    assert regular.pointwise_stages[-1].fano == 0.5
    monkeypatch.setattr(run_module, "build_pipeline", lambda *args: regular)
    result = run(sample, "mouse", size_px=32, window_ms=400.0, looks=2)
    spikes = result.pipeline.pointwise_stages[-1]
    assert (spikes.fano, spikes.window_s) == (0.5, 0.2)


@pytest.mark.parametrize("name", ["fly", "human", "mouse"])
def test_every_species_fires_poisson_spikes_by_default(name):
    """The cells whose spikes are counted are cortical for the mammals, and
    cortical counts are not more regular than Poisson; see the audit document."""
    pipeline = species.get(name)(VisualField(32, 60.0))
    assert pipeline.metadata["params"].fano == 1.0
    assert pipeline.pointwise_stages[-1].fano == 1.0
