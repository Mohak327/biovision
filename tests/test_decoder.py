import dataclasses

import numpy as np
import pytest

from biovision import io, species
from biovision.core.decoder import Decoder, channel_coupling, conjugate_gradient, uncoupling
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


def test_conjugate_gradient_reports_every_iteration(rng):
    m = rng.standard_normal((12, 12))
    spd = m @ m.T + 12.0 * np.eye(12)
    b = rng.standard_normal(12)
    seen = []
    x, residuals, _ = conjugate_gradient(lambda v: spd @ v, b, 1e-10, 100,
                                         on_iteration=lambda k, xk, residual: seen.append((k, xk.copy())))
    assert [k for k, _ in seen] == list(range(1, len(residuals) + 1))
    np.testing.assert_array_equal(seen[-1][1], x)


def test_decode_reports_clipped_image_estimates(rng):
    pipeline = small_pipeline()
    code = pipeline.encode(rng.random((3, SIZE, SIZE)))
    frames = []
    result = Decoder(pipeline, 1e-2).decode(code, on_iteration=lambda k, image, residual: frames.append((k, image)))
    assert len(frames) == result.iterations
    assert all(image.shape == (3, SIZE, SIZE) for _, image in frames)
    assert all(image.min() >= 0.0 and image.max() <= 1.0 for _, image in frames)
    np.testing.assert_array_equal(frames[-1][1], result.image)


def normal_equations(pipeline, image, lam, weight):
    """The decoder's system built by hand: its data term, its prior and the right-hand side."""
    operator = pipeline.linear_operator()
    shape = pipeline.in_shape

    def data(v):
        return operator.rmatvec(operator.matvec(v))

    def prior(v):
        x = v.reshape(shape)
        return lam * (-laplacian(x) + weight * chroma(x)).ravel()

    return data, prior, operator.rmatvec(operator.matvec(image.ravel()))


def test_a_preconditioner_does_not_change_the_solution(rng):
    m = rng.standard_normal((20, 20))
    scale = np.diag(10.0 ** rng.uniform(-2, 2, 20))
    spd = scale @ (m @ m.T + 20.0 * np.eye(20)) @ scale
    b = rng.standard_normal(20)
    plain, plain_residuals, _ = conjugate_gradient(lambda v: spd @ v, b, 1e-12, 2000)
    x, residuals, converged = conjugate_gradient(lambda v: spd @ v, b, 1e-12, 2000,
                                                 precondition=lambda r: r / np.diag(spd))
    assert converged and residuals[-1] <= 1e-12
    np.testing.assert_allclose(x, np.linalg.solve(spd, b), rtol=1e-8)
    np.testing.assert_allclose(x, plain, rtol=1e-8)
    assert len(residuals) < len(plain_residuals)  # and a fitting one gets there sooner


def test_channel_coupling_recovers_a_mixing_of_the_channels(rng):
    m = rng.standard_normal((3, 3))
    mixing = m @ m.T + np.eye(3)
    shape = (3, SIZE, SIZE)
    coupling = channel_coupling(lambda v: (mixing @ v.reshape(3, -1)).ravel(), shape)
    np.testing.assert_allclose(coupling, mixing, atol=1e-12)


def test_channel_coupling_of_an_eye_is_symmetric_and_positive(rng):
    pipeline = small_pipeline()
    data, _, _ = normal_equations(pipeline, rng.random((3, SIZE, SIZE)), 1e-2, 0.5)
    coupling = channel_coupling(data, pipeline.in_shape)
    np.testing.assert_allclose(coupling, coupling.T, atol=1e-12)
    assert np.linalg.eigvalsh(coupling).min() > 0.0


def test_uncoupling_inverts_a_channel_mixing_plus_the_prior_exactly(rng):
    """When the data term is the same mixing at every pixel, nothing is approximated."""
    m = rng.standard_normal((3, 3))
    mixing = m @ m.T + np.eye(3)
    shape = (3, SIZE, SIZE + 1)  # an odd width too

    def data(v):
        return (mixing @ v.reshape(3, -1)).ravel()

    def prior(v):
        x = v.reshape(shape)
        return 0.3 * (-laplacian(x) + 0.5 * chroma(x)).ravel()

    x = rng.standard_normal(int(np.prod(shape)))
    precondition = uncoupling(data, prior, shape)
    np.testing.assert_allclose(precondition(data(x) + prior(x)), x, atol=1e-10)
    y = rng.standard_normal(x.size)
    assert precondition(x) @ y == pytest.approx(x @ precondition(y))  # symmetric
    assert precondition(x) @ x > 0.0


def test_decode_matches_the_unpreconditioned_solve(rng):
    pipeline = small_pipeline()
    image = rng.random((3, SIZE, SIZE))
    lam, weight = 1e-2, 0.5
    data, prior, b = normal_equations(pipeline, image, lam, weight)
    plain, _, converged = conjugate_gradient(lambda v: data(v) + prior(v), b, 1e-10, 2000)
    assert converged
    result = Decoder(pipeline, lam, weight, max_iter=2000, tol=1e-10).decode(pipeline.encode(image))
    assert result.converged
    np.testing.assert_allclose(result.image, np.clip(plain.reshape(3, SIZE, SIZE), 0.0, 1.0),
                               atol=1e-7)


def test_uncoupling_shortens_the_solve_for_an_eye(sample):
    """The mouse senses green and blue and no red, so its channels are far from alike."""
    pipeline = species.get("mouse")(VisualField(64, 60.0))
    image = io.to_square(sample, 64).transpose(2, 0, 1)
    data, prior, b = normal_equations(pipeline, image, 1e-2, 0.1)  # a noisy run's lam

    def apply(v):
        return data(v) + prior(v)

    plain, plain_residuals, _ = conjugate_gradient(apply, b, 1e-4, 1000)
    x, residuals, converged = conjugate_gradient(
        apply, b, 1e-4, 1000, precondition=uncoupling(data, prior, pipeline.in_shape))
    assert converged and len(residuals) < 0.5 * len(plain_residuals)
    np.testing.assert_allclose(x, plain, atol=5e-3)  # both stop at the same loose tolerance


def test_conjugate_gradient_reports_the_residual_of_each_iteration(rng):
    m = rng.standard_normal((12, 12))
    spd = m @ m.T + 12.0 * np.eye(12)
    seen = []
    _, residuals, _ = conjugate_gradient(lambda v: spd @ v, rng.standard_normal(12), 1e-10, 100,
                                         on_iteration=lambda k, x, residual: seen.append(residual))
    assert seen == residuals


def test_decoder_progress_runs_from_nothing_to_done_on_a_log_scale():
    decoder = Decoder(small_pipeline(), 1e-2, tol=1e-4)
    assert decoder.progress(1.0) == 0.0
    assert decoder.progress(1e-2) == pytest.approx(0.5)
    assert decoder.progress(1e-4) == 1.0
    assert decoder.progress(1e-9) == 1.0  # past the tolerance is still done
    assert decoder.progress(5.0) == 0.0  # a residual above its start is no progress
