"""Stage interfaces. A pipeline is linear stages followed by pointwise stages."""
from abc import ABC, abstractmethod

import numpy as np


class Stage(ABC):
    """One step of a visual pipeline."""

    name: str


class LinearStage(Stage):
    """A linear map with an exact transpose."""

    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def adjoint(self, y: np.ndarray) -> np.ndarray: ...


class PointwiseStage(Stage):
    """An elementwise function with an elementwise inverse."""

    @abstractmethod
    def forward(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray: ...

    @abstractmethod
    def inverse(self, y: np.ndarray) -> np.ndarray: ...
