"""Photoreceptor mosaics: where receptors sit, and sampling an image at them."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .sparse import SparseStage


@dataclass(frozen=True)
class Mosaic:
    """Receptor positions as (row, col) pixels, and each receptor's type index."""

    positions: np.ndarray  # (n, 2)
    types: np.ndarray  # (n,) integers in [0, n_types)
    n_types: int

    def __len__(self) -> int:
        return len(self.positions)


def square_lattice(size_px: int, spacing_px: float) -> np.ndarray:
    """A centred square grid of points covering [0, size_px - 1]."""
    extent = size_px - 1
    n = int(np.floor(extent / spacing_px)) + 1
    axis = np.arange(n) * spacing_px + (extent - (n - 1) * spacing_px) / 2.0
    rows, cols = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([rows.ravel(), cols.ravel()])


def hex_lattice(size_px: int, spacing_px: float) -> np.ndarray:
    """A hexagonal grid: rows sqrt(3)/2 apart, alternate rows shifted by half."""
    extent = size_px - 1
    row_step = spacing_px * np.sqrt(3.0) / 2.0
    n_rows = int(np.floor(extent / row_step)) + 1
    row_offset = (extent - (n_rows - 1) * row_step) / 2.0
    n_cols = int(np.floor(extent / spacing_px)) + 1
    col_offset = (extent - (n_cols - 1) * spacing_px) / 2.0 - spacing_px / 4.0
    points = []
    for i in range(n_rows):
        shift = spacing_px / 2.0 if i % 2 else 0.0
        cols = col_offset + shift + np.arange(n_cols) * spacing_px
        cols = cols[(cols >= 0) & (cols <= extent)]
        points.append(np.column_stack([np.full(len(cols), row_offset + i * row_step), cols]))
    return np.vstack(points)


def foveated_lattice(size_px: int, center_spacing_px: float, e2_px: float,
                     min_spacing_px: float = 1.0) -> np.ndarray:
    """Rings whose spacing grows with eccentricity: s(r) = s0 * (1 + r / e2).

    Spacing never drops below `min_spacing_px`, because the image cannot carry
    detail finer than a pixel.
    """
    centre = (size_px - 1) / 2.0
    points = [np.array([[centre, centre]])]
    radius = 0.0
    limit = centre * np.sqrt(2.0)
    while True:
        spacing = max(center_spacing_px * (1.0 + radius / e2_px), min_spacing_px)
        radius += spacing
        if radius > limit:
            break
        spacing = max(center_spacing_px * (1.0 + radius / e2_px), min_spacing_px)
        count = max(int(round(2.0 * np.pi * radius / spacing)), 6)
        angles = 2.0 * np.pi * (np.arange(count) + 0.5 * (len(points) % 2)) / count
        points.append(np.column_stack([centre + radius * np.sin(angles),
                                       centre + radius * np.cos(angles)]))
    all_points = np.vstack(points)
    inside = np.all((all_points >= 0) & (all_points <= size_px - 1), axis=1)
    return all_points[inside]


def assign_types(n: int, fractions, rng: np.random.Generator) -> np.ndarray:
    """A random receptor type for each of `n` receptors, drawn with `fractions`."""
    fractions = np.asarray(fractions, dtype=float)
    return rng.choice(len(fractions), size=n, p=fractions / fractions.sum())


def retype_inside(types: np.ndarray, eccentricity: np.ndarray, absent_within,
                  fractions, rng: np.random.Generator) -> np.ndarray:
    """Receptor types with none of a type closer to the centre than its radius.

    `absent_within` gives one radius per type, in the units of `eccentricity`
    (0 = found everywhere). A receptor of a type that does not occur where it
    sits is drawn again from the types that do occur there, in their own
    proportions.
    """
    absent_within = np.asarray(absent_within, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    types = types.copy()
    for kind in np.argsort(-absent_within):  # the widest zone first
        wrong = (types == kind) & (eccentricity < absent_within[kind])
        if wrong.any():
            # Types whose own zone is no wider occur wherever this one is redrawn,
            # or are redrawn in their turn.
            allowed = fractions * (absent_within <= absent_within[kind])
            allowed[kind] = 0.0
            types[wrong] = assign_types(int(wrong.sum()), allowed, rng)
    return types


def all_types_at(positions: np.ndarray, n_types: int) -> Mosaic:
    """A mosaic with one receptor of every type at each position."""
    return Mosaic(np.tile(positions, (n_types, 1)),
                  np.repeat(np.arange(n_types), len(positions)), n_types)


def mosaic_sampling(mosaic: Mosaic, size_px: int, name: str = "mosaic") -> SparseStage:
    """Each receptor reads its own type's channel at its position (bilinear)."""
    n = len(mosaic)
    r = np.clip(mosaic.positions[:, 0], 0, size_px - 1)
    c = np.clip(mosaic.positions[:, 1], 0, size_px - 1)
    r0 = np.minimum(np.floor(r).astype(int), size_px - 2)
    c0 = np.minimum(np.floor(c).astype(int), size_px - 2)
    fr, fc = r - r0, c - c0
    base = mosaic.types * size_px * size_px
    rows = np.tile(np.arange(n), 4)
    cols = np.concatenate([
        base + r0 * size_px + c0,
        base + r0 * size_px + c0 + 1,
        base + (r0 + 1) * size_px + c0,
        base + (r0 + 1) * size_px + c0 + 1,
    ])
    values = np.concatenate([(1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc])
    matrix = csr_matrix((values, (rows, cols)), shape=(n, mosaic.n_types * size_px * size_px))
    return SparseStage(name, matrix, (mosaic.n_types, size_px, size_px), (n,))
