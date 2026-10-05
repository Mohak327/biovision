"""Reconstruct an image from a neural code with a regularized linear inverse."""
import warnings
from dataclasses import dataclass

import numpy as np

from .pipeline import NeuralCode, Pipeline
from .regularizers import chroma, laplacian


@dataclass(frozen=True)
class Reconstruction:
    image: np.ndarray  # (channels, size, size), clipped to [0, 1]
    iterations: int
    converged: bool
    residuals: tuple[float, ...]  # relative residual after each iteration
    lam: float


PROBES_PER_SIDE = 4  # the channel coupling is averaged over a grid of this many pixels a side


def conjugate_gradient(apply, b: np.ndarray, tol: float, max_iter: int, on_iteration=None,
                       precondition=None):
    """Solve apply(x) = b for a symmetric positive-definite operator.

    Returns (x, relative residual after each iteration, converged). If given,
    `on_iteration(k, x)` is called after iteration k with the current estimate.

    `precondition(r)`, if given, applies a symmetric positive-definite
    approximation of the operator's inverse. It changes how many iterations
    the solution takes, not the solution: the system and the stopping rule
    (the residual of the system itself) are the same.
    """
    x = np.zeros_like(b)
    r = b.copy()
    z = r if precondition is None else precondition(r)
    p = z.copy()
    rz = float(r @ z)
    b_norm = float(np.sqrt(b @ b))
    residuals: list[float] = []
    if b_norm == 0.0:
        return x, residuals, True
    for _ in range(max_iter):
        ap = apply(p)
        alpha = rz / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        residuals.append(float(np.sqrt(r @ r)) / b_norm)
        if on_iteration is not None:
            on_iteration(len(residuals), x)
        if residuals[-1] <= tol:
            return x, residuals, True
        z = r if precondition is None else precondition(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, residuals, False


def channel_coupling(apply, shape: tuple[int, int, int]) -> np.ndarray:
    """How `apply` couples the channels at one pixel, averaged over a grid of pixels.

    `apply` acts on flattened (channels, height, width) arrays. Entry (a, b) is
    the response in channel a, at the probed pixels, to a unit input at those
    pixels in channel b: `channels` applications of the operator in all. It is
    a projection of the operator, so it is symmetric and positive
    (semi-)definite whenever the operator is.
    """
    channels, height, width = shape
    rows, cols = (np.arange(n) * size // n + size // (2 * n)
                  for size in (height, width) for n in [min(PROBES_PER_SIDE, size)])
    probed = np.zeros((height, width), dtype=bool)
    probed[np.ix_(rows, cols)] = True
    coupling = np.empty((channels, channels))
    for b in range(channels):
        probe = np.zeros(shape)
        probe[b] = probed
        coupling[:, b] = apply(probe.ravel()).reshape(shape)[:, probed].sum(axis=1)
    return 0.5 * (coupling + coupling.T) / probed.sum()


def uncoupling(data, prior, shape: tuple[int, int, int]):
    """A preconditioner for `data + prior`: the exact inverse of a simpler operator.

    The simpler operator keeps the prior whole and replaces the data term by
    its channel coupling, the same at every pixel. `prior` must act alike at
    every pixel with periodic edges and be symmetric, as the decoder's does;
    its transfer function is then real and is read from its response to one
    impulse per channel. The sum is inverted frequency by frequency, a
    channels x channels matrix each.
    """
    channels, height, width = shape
    transfer = np.empty((height, width // 2 + 1, channels, channels))
    for b in range(channels):
        impulse = np.zeros(shape)
        impulse[b, 0, 0] = 1.0
        response = np.fft.rfft2(prior(impulse.ravel()).reshape(shape))
        transfer[:, :, :, b] = np.moveaxis(response.real, 0, -1)
    inverse = np.linalg.inv(channel_coupling(data, shape) + transfer)

    def precondition(r):
        spectrum = np.moveaxis(np.fft.rfft2(r.reshape(shape)), 0, -1)[..., None]
        solved = np.moveaxis((inverse @ spectrum)[..., 0], -1, 0)
        return np.fft.irfft2(solved, s=(height, width)).ravel()

    return precondition


class Decoder:
    """Solves min ||A x - y||^2 + lam * prior(x) by conjugate gradients.

    The prior penalizes image gradients (natural images have about 1/f^2
    power) and, with `chroma_weight`, differences between colour channels
    (natural images have strongly correlated channels).

    The solve is preconditioned (`uncoupling`). An eye senses some mixtures
    of the colour channels far more strongly than others, and what it does not
    sense is left to the prior; undoing both first lets the solver work on
    every mixture at a similar rate.

    `tol` is the relative residual at which the solve stops. At 3e-5 the
    preconditioned solve ends closer to the exact solution than the plain one
    did at 1e-4, in fewer iterations (human, 96 px, ideal neurons: 40.1 dB
    in about 470 against 39.7 dB in about 650; the exact solution gives 40.2).
    """

    def __init__(self, pipeline: Pipeline, lam: float, chroma_weight: float = 0.1,
                 max_iter: int = 1000, tol: float = 3e-5):
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        if chroma_weight < 0:
            raise ValueError(f"chroma_weight must not be negative, got {chroma_weight}")
        self.pipeline = pipeline
        self.lam = lam
        self.chroma_weight = chroma_weight
        self.max_iter = max_iter
        self.tol = tol

    def linear_drive(self, responses: np.ndarray) -> np.ndarray:
        """Undo the pointwise stages: spike counts back to the linear response."""
        y = np.asarray(responses, dtype=float)
        for stage in reversed(self.pipeline.pointwise_stages):
            y = stage.inverse(y)
        return y

    def noise_variance(self, code: NeuralCode) -> float:
        """Mean variance of the linear drive, assuming Poisson spike counts.

        A Poisson count has variance equal to its mean. The pointwise inverses
        are affine, so each count's variance reaches the drive scaled by the
        square of the drive's slope against that count. Where several cells
        carry one signal (an ON and an OFF cell, on the code's last axis),
        their noise is independent and the scaled variances add.
        """
        signals = self.pipeline.out_shape
        cells = code.responses.shape[len(signals):]  # the cells that carry one signal
        n_cells = int(np.prod(cells))
        probes = np.vstack([np.zeros(n_cells), np.eye(n_cells)]).reshape(n_cells + 1, *cells)
        drive = self.linear_drive(probes)  # with no spikes, then with one spike in each cell
        slopes = drive[1:] - drive[0]
        counts = np.moveaxis(code.responses.reshape(*signals, n_cells), -1, 0)
        return float(sum(float(np.mean(count)) * slope**2
                         for count, slope in zip(counts, slopes)))

    def decode(self, code: NeuralCode, on_iteration=None) -> Reconstruction:
        """Reconstruct the image. If given, `on_iteration(k, image)` receives the
        estimate after each solver iteration, as (channels, size, size) in [0, 1]."""
        pipeline = self.pipeline
        a = pipeline.linear_operator()
        shape = pipeline.in_shape

        def data(v):
            return a.rmatvec(a.matvec(v))

        def prior(v):
            x = v.reshape(shape)
            return self.lam * (-laplacian(x) + self.chroma_weight * chroma(x)).ravel()

        def normal(v):
            return data(v) + prior(v)

        b = a.rmatvec(self.linear_drive(code.responses).ravel())
        def as_image(v):
            return np.clip(v.reshape(shape), 0.0, 1.0)

        report = None
        if on_iteration is not None:
            def report(k, v):
                on_iteration(k, as_image(v))

        x, residuals, converged = conjugate_gradient(normal, b, self.tol, self.max_iter, report,
                                                     uncoupling(data, prior, shape))
        if not converged:
            warnings.warn(
                f"decoder for '{pipeline.name}' did not converge in "
                f"{self.max_iter} iterations; returning the best estimate",
                RuntimeWarning, stacklevel=2,
            )
        return Reconstruction(as_image(x), len(residuals), converged, tuple(residuals),
                              self.lam)
