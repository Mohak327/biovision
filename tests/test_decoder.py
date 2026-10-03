import dataclasses

import numpy as np
import pytest

from biovision.core.decoder import Decoder, conjugate_gradient
from biovision.core.field import VisualField
from biovision.core.pipeline import Pipeline
from biovision.core.regularizers import chroma, laplacian
from biovision.stages.color import ColorProjection
from biovision.stages.nonlinearity import LinearRectified
from biovision.stages.optics import OpticalBlur
from biovision.stages.spiking import PoissonSpikes

SIZE = 8


def small_pipeline(window_s=0.1):
    matrix = [[0.5, 0.4, 0.1], [0.1, 0.6, 0.3], [0.0, 0.2, 0.8]]
    return Pipeline("small", VisualField(SIZE, 60.0), (
        ColorProjection(matrix, SIZE), OpticalBlur(0.8, 3, SIZE),
        LinearRectified(100.0, 1.0), PoissonSpikes(window_s)))


def test_regularizers_are_self_adjoint(rng):
    x, y = rng.standard_normal((2, 3, SIZE, SIZE))
    for operator in (laplacian, chroma):
        assert np.vdot(operator(x), y) == pytest.approx(np.vdot(x, operator(y)))
    assert np.allclose(laplacian(np.ones((3, SIZE, SIZE))), 0.0)
    assert np.allclose(chroma(np.ones((3, SIZE, SIZE))), 0.0)


def test_conjugate_gradient_solves_a_small_system(rng):
    m = rng.standard_normal((20, 20))
    spd = m @ m.T + 20.0 * np.eye(20)
    b = rng.standard_normal(20)
    x, residuals, converged = conjugate_gradient(lambda v: spd @ v, b, 1e-10, 200)
    assert converged
    np.testing.assert_allclose(x, np.linalg.solve(spd, b), atol=1e-8)
    assert residuals[-1] <= 1e-10 and len(residuals) <= 200


def test_conjugate_gradient_with_zero_right_hand_side():
    x, residuals, converged = conjugate_gradient(lambda v: v, np.zeros(5), 1e-6, 10)
    assert converged and residuals == [] and np.all(x == 0.0)


def test_decode_matches_the_dense_solution(rng):
    """On a tiny problem, build A and the prior as matrices and solve exactly."""
    pipeline = small_pipeline()
    image = rng.random((3, SIZE, SIZE))
    lam, weight = 1e-2, 0.5
    operator = pipeline.linear_operator()
    n = operator.shape[1]
    identity = np.eye(n)
    a = np.column_stack([operator.matvec(identity[:, i]) for i in range(n)])
    prior = np.column_stack([
        (-laplacian(identity[:, i].reshape(3, SIZE, SIZE))
         + weight * chroma(identity[:, i].reshape(3, SIZE, SIZE))).ravel() for i in range(n)])
    y = a @ image.ravel()
    exact = np.linalg.solve(a.T @ a + lam * prior, a.T @ y).reshape(3, SIZE, SIZE)
    decoder = Decoder(pipeline, lam, weight, max_iter=2000, tol=1e-10)
    result = decoder.decode(pipeline.encode(image))
    assert result.converged
    np.testing.assert_allclose(result.image, np.clip(exact, 0.0, 1.0), atol=1e-5)


def test_noise_free_decoding_recovers_the_image(rng):
    """With an invertible pipeline (no blur) and almost no prior, decoding is exact."""
    matrix = [[0.5, 0.4, 0.1], [0.1, 0.6, 0.3], [0.0, 0.2, 0.8]]
    pipeline = Pipeline("exact", VisualField(SIZE, 60.0), (
        ColorProjection(matrix, SIZE), LinearRectified(100.0, 1.0), PoissonSpikes(0.1)))
    image = rng.random((3, SIZE, SIZE))
    decoder = Decoder(pipeline, 1e-9, 0.0, max_iter=5000, tol=1e-12)
    result = decoder.decode(pipeline.encode(image))
    np.testing.assert_allclose(result.image, image, atol=1e-3)


def test_noise_variance_is_the_poisson_variance_of_the_drive(rng):
    pipeline = small_pipeline(window_s=0.2)
    flat = np.full((3, SIZE, SIZE), 0.5)
    code = pipeline.encode(flat, rng)
    decoder = Decoder(pipeline, 1.0)
    # drive = (count / T / rest - 1) / gain, so its slope against count is 1 / (T * rest * gain)
    expected = code.responses.mean() * (1.0 / (0.2 * 100.0 * 1.0)) ** 2
    assert decoder.noise_variance(code) == pytest.approx(expected)
    drives = np.stack([decoder.linear_drive(pipeline.encode(flat, rng).responses)
                       for _ in range(200)])
    assert drives.var(axis=0).mean() == pytest.approx(expected, rel=0.1)


def test_unconverged_decode_warns_and_still_returns_an_image(rng):
    pipeline = small_pipeline()
    code = pipeline.encode(rng.random((3, SIZE, SIZE)))
    with pytest.warns(RuntimeWarning, match="did not converge in 2 iterations"):
        result = Decoder(pipeline, 1e-6, max_iter=2, tol=1e-12).decode(code)
    assert not result.converged and result.iterations == 2
    assert result.image.shape == (3, SIZE, SIZE)
    assert result.image.min() >= 0.0 and result.image.max() <= 1.0


def test_all_zero_spikes_decode_to_a_finite_image():
    pipeline = small_pipeline()
    code = pipeline.encode(np.zeros((3, SIZE, SIZE)))
    silent = dataclasses.replace(code, responses=np.zeros_like(code.responses))
    result = Decoder(pipeline, 1e-2).decode(silent)
    assert np.all(np.isfinite(result.image))


@pytest.mark.parametrize("kwargs", [dict(lam=0.0), dict(lam=-1.0),
                                    dict(lam=1.0, chroma_weight=-0.1)])
def test_decoder_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Decoder(small_pipeline(), **kwargs)
