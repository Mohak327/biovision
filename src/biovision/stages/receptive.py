"""Retinal receptive fields over a mosaic: centre minus surround, per cell class."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags, identity, kron

from ..core.stage import LinearStage
from .mosaic import Mosaic, all_types_at
from .pyramid import POOL_SIGMA_RATIO, summarize
from .sparse import FactoredStage, SparseStage, gaussian, normalize_rows, pool


def smooth(out: Mosaic, mosaic: Mosaic, sigma_px: float) -> list[csr_matrix]:
    """Matrices, applied in order, giving each `out` cell the Gaussian mean of its type.

    One direct matrix where that is affordable. Otherwise the receptors are
    first pooled onto a coarse layer and the cell pools from that; the two
    Gaussians' variances add up to `sigma_px` squared.

    Where a type has no receptors, the coarse cells there are empty. A cell
    that pools from some of those would get less than a mean, so its weights
    are scaled back up to sum to 1 over the receptors it does reach.
    """
    spacing = sigma_px / np.sqrt(1.0 + POOL_SIGMA_RATIO**2)
    source, head = summarize(out.positions, mosaic, 3.0 * sigma_px, spacing)
    width = spacing if head else sigma_px
    last = normalize_rows(pool(out.positions, out.types, source.positions, source.types,
                               3.0 * width, lambda dy, dx: gaussian(dy, dx, width)))
    if head:
        total = np.ones(len(mosaic))
        for matrix in head + [last]:
            total = matrix @ total
        if np.any(total < 1.0 - 1e-9):  # only then, so an eye without gaps is left as it was
            scale = np.divide(1.0, total, out=np.zeros_like(total), where=total > 0)
            last = (diags(scale) @ last).tocsr()
    return head + [last]


def center_surround(mosaic: Mosaic, sigma_center_px: float, sigma_surround_px: float,
                    surround_weight: float, name: str = "center_surround") -> LinearStage:
    """One cell per receptor: a narrow centre minus a weighted wide surround.

    Centre and surround are each normalized to sum to 1 over the receptors
    they pool, so a uniform image gives a response of 1 - surround_weight.
    """
    shape = (len(mosaic),)
    centre = smooth(mosaic, mosaic, sigma_center_px)
    *head, surround = smooth(mosaic, mosaic, sigma_surround_px)
    if len(centre) == 1 and not head:
        return SparseStage(name, (centre[0] - surround_weight * surround).tocsr(), shape, shape)
    return FactoredStage(name, [centre, head + [-surround_weight * surround]], shape, shape)


@dataclass(frozen=True)
class RetinaClass:
    """A class of retinal cell: how it weights the receptor types, and its gain.

    `weights` has one entry per receptor type. A class whose weights sum to
    zero ignores uniform grey and carries a colour difference. `gain` scales
    the response so the class fills its firing range. `surround_weight` is the
    strength of the spatial surround.
    """

    name: str
    weights: tuple[float, ...]
    gain: float
    surround_weight: float


def opponent_retina(mosaic: Mosaic, classes, sigma_center_px: float, sigma_surround_px: float,
                    name: str = "center_surround") -> tuple[LinearStage, Mosaic]:
    """Retinal cells that combine receptor types, one of each class at each position.

    For each receptor type, a normalized Gaussian pools that type's receptors
    around every position; a class adds those pools with its weights. The
    response is gain * (centre - surround_weight * surround).

    A cell whose centre reaches no receptor of a type its class weights is
    silent: with one of its inputs missing it has nothing to compare (a
    blue-yellow cell where there are no S cones).

    Returns the stage and the mosaic of the cells it made (their positions,
    with the class index as the type), which later stages pool from.
    """
    classes = tuple(classes)
    if not classes:
        raise ValueError("opponent_retina needs at least one class")
    for item in classes:
        if len(item.weights) != mosaic.n_types:
            raise ValueError(f"class '{item.name}' needs one weight per receptor type "
                             f"({mosaic.n_types}), got {len(item.weights)}")
        if not any(item.weights):
            raise ValueError(f"class '{item.name}' has no non-zero weight")
    # Receptors of several types can share a position; cells sit once at each.
    _, first = np.unique(mosaic.positions, axis=0, return_index=True)
    positions = mosaic.positions[np.sort(first)]
    pools = all_types_at(positions, mosaic.n_types)  # one pool per receptor type per position
    centre = smooth(pools, mosaic, sigma_center_px)
    surround = smooth(pools, mosaic, sigma_surround_px)

    def mix(scale):
        """Classes x positions from types x positions: each class's weighted sum of pools."""
        weights = np.array([[scale(item) * w for w in item.weights] for item in classes])
        return kron(csr_matrix(weights), identity(len(positions)), format="csr")

    mix_centre = mix(lambda item: item.gain)
    mix_surround = mix(lambda item: -item.gain * item.surround_weight)
    reached = np.ones(len(mosaic))
    for matrix in centre:
        reached = matrix @ reached  # 1 for a pool that found receptors, 0 for an empty one
    reached = reached.reshape(mosaic.n_types, len(positions)) > 0
    if not reached.all():
        complete = diags(np.concatenate([reached[np.flatnonzero(item.weights)].all(axis=0)
                                         for item in classes]).astype(float))
        mix_centre, mix_surround = complete @ mix_centre, complete @ mix_surround
    cells = Mosaic(np.tile(positions, (len(classes), 1)),
                   np.repeat(np.arange(len(classes)), len(positions)), len(classes))
    shapes = (len(mosaic),), (len(cells),)
    if len(centre) == 1 and len(surround) == 1:
        matrix = mix_centre @ centre[0] + mix_surround @ surround[0]
        return SparseStage(name, matrix.tocsr(), *shapes), cells
    return FactoredStage(name, [centre + [mix_centre], surround + [mix_surround]], *shapes), cells
