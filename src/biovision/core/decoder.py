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


def conjugate_gradient(apply, b: np.ndarray, tol: float, max_iter: int, on_iteration=None):
    """Solve apply(x) = b for a symmetric positive-definite operator.

    Returns (x, relative residual after each iteration, converged). If given,
    `on_iteration(k, x)` is called after iteration k with the current estimate.
    """
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = float(r @ r)
    b_norm = float(np.sqrt(b @ b))
    residuals: list[float] = []
    if b_norm == 0.0:
        return x, residuals, True
    for _ in range(max_iter):
        ap = apply(p)
        alpha = rr / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        rr_new = float(r @ r)
        residuals.append(float(np.sqrt(rr_new)) / b_norm)
        if on_iteration is not None:
            on_iteration(len(residuals), x)
        if residuals[-1] <= tol:
            return x, residuals, True
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x, residuals, False


class Decoder:
    """Solves min ||A x - y||^2 + lam * prior(x) by conjugate gradients.

    The prior penalizes image gradients (natural images have about 1/f^2
    power) and, with `chroma_weight`, differences between colour channels
    (natural images have strongly correlated channels).
    """

    def __init__(self, pipeline: Pipeline, lam: float, chroma_weight: float = 0.1,
                 max_iter: int = 1000, tol: float = 1e-4):
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
        are affine, so they scale that variance by their combined slope squared.
        """
        slope = float(np.diff(self.linear_drive(np.array([0.0, 1.0])))[0])
        return float(np.mean(code.responses)) * slope**2

    def decode(self, code: NeuralCode, on_iteration=None) -> Reconstruction:
        """Reconstruct the image. If given, `on_iteration(k, image)` receives the
        estimate after each solver iteration, as (channels, size, size) in [0, 1]."""
        pipeline = self.pipeline
        a = pipeline.linear_operator()
        shape = pipeline.in_shape

        def normal(v):
            x = v.reshape(shape)
            prior = -laplacian(x) + self.chroma_weight * chroma(x)
            return a.rmatvec(a.matvec(v)) + self.lam * prior.ravel()

        b = a.rmatvec(self.linear_drive(code.responses).ravel())
        def as_image(v):
            return np.clip(v.reshape(shape), 0.0, 1.0)

        report = None
        if on_iteration is not None:
            def report(k, v):
                on_iteration(k, as_image(v))

        x, residuals, converged = conjugate_gradient(normal, b, self.tol, self.max_iter, report)
        if not converged:
            warnings.warn(
                f"decoder for '{pipeline.name}' did not converge in "
                f"{self.max_iter} iterations; returning the best estimate",
                RuntimeWarning, stacklevel=2,
            )
        return Reconstruction(as_image(x), len(residuals), converged, tuple(residuals),
                              self.lam)
