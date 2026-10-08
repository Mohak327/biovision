"""Fixational eye movements: several shifted looks combined in one solve (phase 3)."""
import base64
import json

import numpy as np
import pytest
from matplotlib.figure import Figure

from biovision import species
from biovision.analysis import sweep_looks
from biovision.cli import main
from biovision.core.decoder import Decoder
from biovision.core.field import VisualField
from biovision.core.pipeline import Pipeline
from biovision.report import figures
from biovision.run import run, spike_counts, stage_outputs
from biovision.species.eye import fixate
from biovision.stages.color import ColorProjection
from biovision.stages.movement import EyeShifts, PerLook, look_offsets
from biovision.stages.nonlinearity import LinearRectified
from biovision.stages.optics import OpticalBlur
from biovision.stages.spiking import PoissonSpikes

SIZE = 8
SMALL = dict(size_px=32)
NAMES = ["fly", "human", "mouse"]


def test_a_shift_by_whole_pixels_is_a_roll(rng):
    image = rng.standard_normal((3, SIZE, SIZE))
    looks = EyeShifts([(2.0, -3.0), (0.0, 0.0), (-1.0, 5.0)], 3, SIZE).forward(image)
    np.testing.assert_allclose(looks[0], np.roll(image, (2, -3), axis=(1, 2)), atol=1e-12)
    np.testing.assert_allclose(looks[1], image, atol=1e-12)
    np.testing.assert_allclose(looks[2], np.roll(image, (-1, 5), axis=(1, 2)), atol=1e-12)


def test_a_sub_pixel_shift_moves_a_smooth_picture_and_keeps_its_mean():
    rows = np.arange(SIZE)[:, None] * np.ones((1, SIZE))
    wave = np.cos(2 * np.pi * rows / SIZE)[None]
    moved = EyeShifts([(0.5, 0.0)], 1, SIZE).forward(wave)[0]
    np.testing.assert_allclose(moved, np.cos(2 * np.pi * (rows - 0.5) / SIZE)[None], atol=1e-12)
    assert moved.mean() == pytest.approx(wave.mean(), abs=1e-12)


def test_per_look_applies_one_shared_stage_to_every_look(rng):
    blur = OpticalBlur(1.0, 2, SIZE)
    stage = PerLook(blur, 3)
    assert stage.name == "optics" and stage.stage is blur
    x = rng.standard_normal(stage.in_shape)
    np.testing.assert_array_equal(stage.forward(x)[1], blur.forward(x[1]))


def test_look_offsets_are_deterministic_distinct_and_within_the_amplitude():
    offsets = look_offsets(8, 2.0)
    assert offsets.shape == (8, 2)
    np.testing.assert_array_equal(offsets, look_offsets(8, 2.0))
    assert np.all(np.linalg.norm(offsets, axis=1) <= 2.0)
    assert len(np.unique(offsets.round(9), axis=0)) == 8
    np.testing.assert_allclose(look_offsets(8, 4.0), 2.0 * offsets)


@pytest.mark.parametrize("name", NAMES)
def test_every_species_states_how_far_its_eye_moves(name, field):
    assert species.get(name)(field).metadata["params"].fixation_deg > 0


@pytest.mark.parametrize("name", NAMES)
def test_fixate_shares_the_stages_and_keeps_an_exact_adjoint(name, field, rng):
    eye = species.get(name)(field)
    stacked = fixate(eye, 3)
    assert stacked.out_shape == (3, *eye.out_shape) and stacked.in_shape == eye.in_shape
    assert [s.name for s in stacked.stages] == ["fixation"] + [s.name for s in eye.stages]
    assert all(wrapped.stage is stage
               for wrapped, stage in zip(stacked.linear_stages[1:], eye.linear_stages))
    assert stacked.pointwise_stages == eye.pointwise_stages
    operator = stacked.linear_operator()
    x = rng.standard_normal(operator.shape[1])
    y = rng.standard_normal(operator.shape[0])
    assert operator.matvec(x) @ y == pytest.approx(x @ operator.rmatvec(y), rel=1e-9)


def test_fixate_shifts_by_the_species_amplitude_in_pixels(field):
    eye = species.get("fly")(field)
    amplitude = field.to_px(eye.metadata["params"].fixation_deg)
    image = np.random.default_rng(0).random(eye.in_shape)
    expected = EyeShifts(look_offsets(4, amplitude), 3, field.size_px).forward(image)
    np.testing.assert_array_equal(fixate(eye, 4).linear_stages[0].forward(image), expected)


def small_eye(window_s):
    matrix = [[0.5, 0.4, 0.1], [0.1, 0.6, 0.3], [0.0, 0.2, 0.8]]
    return Pipeline("small", VisualField(SIZE, 60.0), (
        ColorProjection(matrix, SIZE), OpticalBlur(0.8, 3, SIZE),
        LinearRectified(100.0, 1.0), PoissonSpikes(window_s)))


def stack(eye, offsets_px):
    looks = len(offsets_px)
    stages = [EyeShifts(offsets_px, 3, SIZE)]
    stages += [PerLook(stage, looks) for stage in eye.linear_stages]
    return Pipeline(eye.name, eye.field, (*stages, *eye.pointwise_stages))


def test_looks_that_do_not_move_decode_like_one_look_of_the_whole_time(rng):
    """K unmoved looks of window / K hold what one look of the window holds: the
    stacked data term is K times the single one, so K times the prior gives the
    same solve, and the noise variance of the drive is K times larger."""
    image = rng.random((3, SIZE, SIZE))
    looks = 4
    one, many = small_eye(0.1), stack(small_eye(0.1 / looks), np.zeros((looks, 2)))
    single = Decoder(one, 1e-3, tol=1e-10).decode(one.encode(image))
    code = many.encode(image)
    assert code.responses.shape == (looks, 3, SIZE, SIZE)
    stacked = Decoder(many, looks * 1e-3, tol=1e-10).decode(code)
    np.testing.assert_allclose(stacked.image, single.image, atol=1e-6)
    assert Decoder(many, 1.0).noise_variance(code) == pytest.approx(
        looks * Decoder(one, 1.0).noise_variance(one.encode(image)))


def test_run_with_looks_stacks_the_code_and_keeps_the_neuron_count(sample):
    one = run(sample, "mouse", **SMALL)
    four = run(sample, "mouse", looks=4, **SMALL)
    assert one.settings.looks == 1 and four.settings.looks == 4
    assert four.code.responses.shape == (4, *one.code.responses.shape)
    assert four.metrics["neurons"] == one.metrics["neurons"]
    assert four.metrics["compression_ratio"] == one.metrics["compression_ratio"]
    assert four.reconstructed.shape == (32, 32, 3)
    # Each look lasts a quarter of the window and draws its own spike noise; a
    # neuron's spikes over the whole time are the sum over its looks.
    assert four.metrics["mean_spikes"] == pytest.approx(one.metrics["mean_spikes"], rel=0.05)
    assert spike_counts(four).shape == one.code.responses.shape
    assert spike_counts(one) is one.code.responses
    assert not np.array_equal(four.code.responses[0], four.code.responses[1])


def test_one_look_is_the_run_without_the_option(sample):
    plain = run(sample, "fly", seed=2, **SMALL)
    one = run(sample, "fly", seed=2, looks=1, **SMALL)
    np.testing.assert_array_equal(plain.reconstructed, one.reconstructed)
    assert [s.name for s in one.pipeline.stages][0] == "color"


def test_looks_are_reproducible_for_a_seed(sample):
    first = run(sample, "fly", seed=3, looks=2, **SMALL)
    again = run(sample, "fly", seed=3, looks=2, **SMALL)
    other = run(sample, "fly", seed=4, looks=2, **SMALL)
    np.testing.assert_array_equal(first.reconstructed, again.reconstructed)
    assert not np.array_equal(first.reconstructed, other.reconstructed)


def test_the_noise_free_regularization_keeps_its_balance_with_the_stack(sample):
    assert run(sample, "fly", noise=False, looks=4, **SMALL).settings.lam == 4e-4
    assert run(sample, "fly", noise=False, looks=4, lam=0.01, **SMALL).settings.lam == 0.01


@pytest.mark.parametrize("looks", [0, -1, 1.5])
def test_run_rejects_a_bad_number_of_looks(sample, looks):
    with pytest.raises(ValueError, match="looks must be a whole number of at least 1"):
        run(sample, "fly", looks=looks, **SMALL)


@pytest.mark.parametrize("name", ["fly", "mouse"])
def test_shifted_looks_add_detail_for_a_coarse_eye_with_ideal_neurons(sample, name):
    """Each shift samples the scene where one look has no receptor."""
    one = run(sample, name, noise=False)
    four = run(sample, name, noise=False, looks=4)
    assert four.metrics["psnr_db"] > one.metrics["psnr_db"] + 0.2
    assert four.metrics["ssim"] > one.metrics["ssim"] + 0.05


def test_progress_and_stage_outputs_show_the_first_look(sample):
    events = []
    result = run(sample, "fly", looks=2, on_progress=events.append, **SMALL)
    names = ("fixation", "color", "optics", "mosaic", "center_surround", "rate", "spikes",
             "decoding")
    assert events[0].stages == names
    by_stage = {event.stages[event.current]: event for event in events if event.iteration == 0}
    assert by_stage["fixation"].image.shape == (32, 32, 3)
    assert by_stage["optics"].image.shape == (32, 32, 3)
    assert by_stage["mosaic"].image.shape == (32, 32, 3)
    outputs = stage_outputs(result)
    assert tuple(outputs) == names[:-1]
    np.testing.assert_array_equal(outputs["mosaic"], result.code.intermediates["mosaic"][0])
    one = run(sample, "fly", **SMALL)
    assert stage_outputs(one) is one.code.intermediates


def test_figures_draw_a_run_with_several_looks(sample):
    result = run(sample, "mouse", looks=2, **SMALL)
    assert isinstance(figures.pipeline_panel(result), Figure)
    assert isinstance(figures.neural_code(result), Figure)


def test_sweep_looks_returns_one_row_per_value(sample):
    rows = sweep_looks(sample, "fly", looks=(1, 2), noise=False, **SMALL)
    assert [row["looks"] for row in rows] == [1, 2]
    assert set(rows[0]) == {"species", "looks", "psnr_db", "ssim"}


def test_the_command_line_takes_looks(tmp_path, capsys):
    out = tmp_path / "panel.png"
    assert main(["run", "--species", "fly", "--size", "32", "--looks", "2",
                 "--out", str(out)]) == 0
    assert out.stat().st_size > 1000
    assert main(["run", "--species", "fly", "--size", "32", "--looks", "0"]) == 2
    assert "looks must be a whole number of at least 1" in capsys.readouterr().err


def test_the_server_takes_looks():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from biovision.server import app

    client = TestClient(app)

    def post(settings):
        return client.post("/api/runs", data={"settings": json.dumps(settings),
                                              "sample": "astronaut"})

    stream = [json.loads(line) for line in post(
        {"species": "fly", "size_px": 32, "looks": 2}).text.splitlines() if line]
    result = stream[-1]
    assert result["type"] == "result" and result["settings"]["looks"] == 2
    # The retina view is lit by one look: one value per receptor.
    caught = np.frombuffer(base64.b64decode(result["responses"]), dtype="<f4")
    assert caught.shape == (stream[1]["count"],)
    assert sum(result["spike_histogram"]["counts"]) == result["metrics"]["neurons"]
    assert post({"species": "fly", "size_px": 32, "looks": 0}).status_code == 422
