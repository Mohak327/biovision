import numpy as np
import pytest

from biovision.stages.color import ColorProjection
from biovision.stages.movement import EyeShifts, PerLook
from biovision.stages.nonlinearity import LinearRectified
from biovision.stages.optics import OpticalBlur
from biovision.stages.photons import PhotonCatch
from biovision.stages.spiking import PoissonSpikes

SIZE = 16


def linear_stages():
    return [
        ColorProjection([[0.2, 0.7, 0.1], [0.0, 0.1, 0.9]], SIZE),
        OpticalBlur(1.5, 2, SIZE),
        OpticalBlur([0.4, 2.5], 2, SIZE, name="optics_per_channel"),
        EyeShifts([(0.3, -1.7), (2.0, 0.5), (-0.25, 0.0)], 2, SIZE),
        PerLook(OpticalBlur(1.5, 2, SIZE), 3),
        PhotonCatch(np.full(40, 25.0)),
    ]


@pytest.mark.parametrize("stage", linear_stages(), ids=lambda s: s.name)
def test_adjoint_is_the_exact_transpose(stage, rng):
    x = rng.standard_normal(stage.in_shape)
    y = rng.standard_normal(stage.out_shape)
    assert np.vdot(stage.forward(x), y) == pytest.approx(np.vdot(x, stage.adjoint(y)), rel=1e-10)


@pytest.mark.parametrize("stage", linear_stages(), ids=lambda s: s.name)
def test_forward_output_has_the_declared_shape(stage, rng):
    assert stage.forward(rng.standard_normal(stage.in_shape)).shape == stage.out_shape


def test_color_projection_rejects_a_bad_matrix():
    with pytest.raises(ValueError, match=r"\(types, 3\)"):
        ColorProjection([[1.0, 0.0]], SIZE)


def test_blur_keeps_the_mean_and_zero_sigma_is_identity(rng):
    x = rng.random((2, SIZE, SIZE))
    assert OpticalBlur(2.0, 2, SIZE).forward(x).mean() == pytest.approx(x.mean())
    np.testing.assert_allclose(OpticalBlur(0.0, 2, SIZE).forward(x), x, atol=1e-12)


def test_linear_rectified_round_trip_and_rest_rate(rng):
    stage = LinearRectified(rest_hz=100.0, gain=2.5)
    x = rng.uniform(-0.39, 0.39, 100)
    np.testing.assert_allclose(stage.inverse(stage.forward(x)), x, atol=1e-12)
    assert stage.forward(np.array([0.0]))[0] == 100.0
    assert stage.forward(np.array([-1.0]))[0] == 0.0  # rectified, never negative


def test_poisson_spikes_mean_exact_without_rng_and_noisy_with(rng):
    stage = PoissonSpikes(window_s=0.1)
    rates = np.full(20000, 50.0)
    np.testing.assert_allclose(stage.forward(rates), 5.0)
    np.testing.assert_allclose(stage.inverse(stage.forward(rates)), rates)
    counts = stage.forward(rates, rng)
    assert np.all(counts == np.round(counts))
    assert counts.mean() == pytest.approx(5.0, abs=0.1)
    assert counts.var() == pytest.approx(5.0, abs=0.3)


@pytest.mark.parametrize("make", [
    lambda: LinearRectified(0.0, 1.0), lambda: LinearRectified(100.0, 0.0),
    lambda: PoissonSpikes(0.0), lambda: OpticalBlur(-1.0, 1, SIZE),
])
def test_stages_reject_invalid_parameters(make):
    with pytest.raises(ValueError):
        make()
