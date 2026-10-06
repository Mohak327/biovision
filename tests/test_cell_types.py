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
from biovision.species import fly, human, mouse
from biovision.species.eye import assemble
from biovision.stages.color import ColorProjection
from biovision.stages.mosaic import all_types_at, square_lattice
from biovision.stages.nonlinearity import LinearRectified, OnOffPair
from biovision.stages.optics import OpticalBlur
from biovision.stages.receptive import RetinaClass, opponent_retina
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


# Midget and parasol classes: a class of retinal cell can have a centre of its own size.

MIDGET = RetinaClass("midget", (0.5, 0.5, 0.0), gain=1.0, surround_weight=0.0)
PARASOL = RetinaClass("parasol", (0.5, 0.5, 0.0), gain=1.0, surround_weight=0.0, center_scale=3.0)


def point_of_light(mosaic, row, col):
    return np.all(mosaic.positions == (row, col), axis=1).astype(float)


def test_a_class_with_a_larger_centre_pools_from_further_away():
    mosaic = all_types_at(square_lattice(16, 1.0), 3)
    stage, cells = opponent_retina(mosaic, (MIDGET, PARASOL), 0.5, 3.0)
    out = stage.forward(point_of_light(mosaic, 8, 8))
    three_away = np.all(cells.positions == (8, 11), axis=1)
    midget, parasol = out[three_away & (cells.types == 0)], out[three_away & (cells.types == 1)]
    assert midget[0] < 1e-6 < 1e-3 < parasol[0]  # 6 midget sigmas away, 2 parasol sigmas
    grey = stage.forward(np.ones(len(mosaic)))  # each centre is a mean, whatever its size
    np.testing.assert_allclose(grey, 1.0, atol=1e-12)


def test_a_retina_with_two_sizes_of_centre_keeps_an_exact_adjoint(rng):
    mosaic = all_types_at(square_lattice(16, 1.0), 3)
    surrounded = replace(PARASOL, surround_weight=0.7, gain=4.0)
    stage, _ = opponent_retina(mosaic, (MIDGET, surrounded), 0.5, 3.0)
    x, y = rng.standard_normal(stage.in_shape), rng.standard_normal(stage.out_shape)
    assert np.vdot(stage.forward(x), y) == pytest.approx(np.vdot(x, stage.adjoint(y)), rel=1e-10)
    with pytest.raises(ValueError, match="positive center_scale"):
        opponent_retina(mosaic, (replace(PARASOL, center_scale=0.0),), 0.5, 3.0)


def test_no_centre_is_narrower_than_the_smallest_the_picture_can_show():
    """Where both centres are under the floor, a parasol cell sees what a midget cell sees."""
    mosaic = all_types_at(square_lattice(16, 1.0), 3)
    stage, cells = opponent_retina(mosaic, (MIDGET, PARASOL), 0.1, 3.0, min_center_px=0.5)
    out = stage.forward(point_of_light(mosaic, 8, 8))
    np.testing.assert_allclose(out[cells.types == 1], out[cells.types == 0], atol=1e-15)
    wide, cells = opponent_retina(mosaic, (MIDGET, PARASOL), 0.4, 3.0, min_center_px=0.5)
    out = wide.forward(point_of_light(mosaic, 8, 8))
    assert out[cells.types == 1].max() < 0.5 * out[cells.types == 0].max()  # 1.2 px against 0.5


def test_the_parasol_class_is_luminance_with_a_larger_centre_and_more_gain():
    midget = human.PARAMS.retina_classes[0]
    assert human.PARASOL.weights == midget.weights == (0.5, 0.5, 0.0)
    assert human.PARASOL.center_scale == human.PARASOL_CENTER_RATIO == 3.0
    assert human.PARASOL.gain == human.PARASOL_GAIN_RATIO * midget.gain
    assert human.PARASOL not in human.PARAMS.retina_classes  # off: its gain fails the guards


def test_a_human_eye_with_parasol_cells_has_a_fourth_class_through_the_cortex(rng):
    params = replace(human.PARAMS,
                     retina_classes=human.PARAMS.retina_classes + (human.PARASOL,))
    plain, with_parasol = eye(human.PARAMS), eye(params)
    assert with_parasol.metadata["cells"].n_types == 4
    assert with_parasol.n_neurons == plain.n_neurons * 4 // 3  # 32 px: both scales carry all classes
    operator = with_parasol.linear_operator()
    x, y = rng.standard_normal(operator.shape[1]), rng.standard_normal(operator.shape[0])
    assert operator.matvec(x) @ y == pytest.approx(x @ operator.rmatvec(y), rel=1e-9)


def test_parasol_centres_are_wider_only_where_the_picture_can_show_it():
    params = replace(human.PARAMS, cortex_sf_cpd=(),
                     retina_classes=human.PARAMS.retina_classes + (human.PARASOL,))
    ratio = human.PARASOL.gain / human.PARAMS.retina_classes[0].gain
    for fov, same in ((60.0, True), (2.0, False)):  # a pixel is 1.9 degrees, then 3.75 arcminutes
        retina = assemble("eye", VisualField(32, fov), params, "", ()).linear_stages[-1]
        cells = retina.out_shape[0] // 4
        out = retina.forward(np.random.default_rng(0).random(retina.in_shape))
        midget, parasol = out[:cells], out[3 * cells:]
        assert np.allclose(parasol, ratio * midget, atol=1e-9) == same
