"""Conversion between image pixels and degrees of visual angle."""
from dataclasses import dataclass


@dataclass(frozen=True)
class VisualField:
    """A square image of `size_px` pixels spanning `fov_deg` degrees."""

    size_px: int
    fov_deg: float

    def __post_init__(self):
        if self.size_px <= 1:
            raise ValueError(f"size_px must be greater than 1, got {self.size_px}")
        if self.fov_deg <= 0:
            raise ValueError(f"fov_deg must be positive, got {self.fov_deg}")

    @property
    def deg_per_px(self) -> float:
        return self.fov_deg / self.size_px

    def to_px(self, deg: float) -> float:
        return deg / self.deg_per_px

    def to_deg(self, px: float) -> float:
        return px * self.deg_per_px
