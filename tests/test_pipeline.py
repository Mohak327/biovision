import numpy as np
import pytest

from biovision.core.field import VisualField
from biovision.core.pipeline import Pipeline
from biovision.stages.color import ColorProjection
from biovision.stages.nonlinearity import LinearRectified
from biovision.stages.optics import OpticalBlur
from biovision.stages.spiking import PoissonSpikes

SIZE = 8
FIELD = VisualField(SIZE, 60.0)
MATRIX = [[0.3, 0.6, 0.1], [0.0, 0.2, 0.8]]


def make(stages):
    return Pipeline("test", FIELD, tuple(stages))


def valid_stages():
    return [ColorProjection(MATRIX, SIZE), OpticalBlur(1.0, 2, SIZE),
            LinearRectified(100.0, 2.5), PoissonSpikes(0.1)]


def test_encode_records_every_stage(rng):
    pipeline = make(valid_stages())
    code = pipeline.encode(rng.random((3, SIZE, SIZE)))
    assert list(code.intermediates) == ["color", "optics", "rate", "spikes"]
    assert code.responses.shape == (2, SIZE, SIZE)
    assert pipeline.n_neurons == 2 * SIZE * SIZE
    assert code.pipeline_name == "test"


def test_linear_operator_matches_the_stages(rng):
    pipeline = make(valid_stages())
    operator = pipeline.linear_operator()
    x = rng.random((3, SIZE, SIZE))
    expected = OpticalBlur(1.0, 2, SIZE).forward(ColorProjection(MATRIX, SIZE).forward(x))
    np.testing.assert_allclose(operator.matvec(x.ravel()), expected.ravel())
    y = rng.random(operator.shape[0])
    assert operator.matvec(x.ravel()) @ y == pytest.approx(x.ravel() @ operator.rmatvec(y))


def test_linear_stage_after_pointwise_is_rejected():
    stages = [ColorProjection(MATRIX, SIZE), PoissonSpikes(0.1), OpticalBlur(1.0, 2, SIZE)]
    with pytest.raises(ValueError, match="follows a pointwise stage"):
        make(stages)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch: 'color' outputs"):
        make([ColorProjection(MATRIX, SIZE), OpticalBlur(1.0, 3, SIZE)])


def test_duplicate_names_and_missing_linear_stage_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        make([OpticalBlur(1.0, 3, SIZE), OpticalBlur(1.0, 3, SIZE)])
    with pytest.raises(ValueError, match="at least one linear stage"):
        make([PoissonSpikes(0.1)])


def test_encode_rejects_a_wrong_image_shape(rng):
    with pytest.raises(ValueError, match="expected image of shape"):
        make(valid_stages()).encode(rng.random((3, SIZE + 1, SIZE)))


def test_replace_swaps_one_stage_and_keeps_the_rest():
    pipeline = make(valid_stages())
    longer = pipeline.replace(PoissonSpikes(1.0))
    assert longer.stages[-1].window_s == 1.0
    assert longer.stages[:-1] == pipeline.stages[:-1]
    assert pipeline.stages[-1].window_s == 0.1
    with pytest.raises(KeyError, match="no stage named 'other'"):
        pipeline.replace(PoissonSpikes(1.0, name="other"))
