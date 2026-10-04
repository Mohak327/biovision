import numpy as np
import pytest

from biovision.stages.gabor import gabor_bank, gabor_kernel
from biovision.stages.mosaic import (Mosaic, all_types_at, assign_types, foveated_lattice,
                                     hex_lattice, mosaic_sampling, square_lattice)
from biovision.stages.receptive import center_surround
from biovision.stages.sparse import SparseStage, normalize_rows, pool

SIZE = 16


def _mosaic(n_types=2):
    positions = hex_lattice(SIZE, 2.0)
    types = assign_types(len(positions), [1.0] * n_types, np.random.default_rng(0))
    return Mosaic(positions, types, n_types)


def linear_stages():
    mosaic = _mosaic()
    return [
        mosaic_sampling(mosaic, SIZE),
        center_surround(mosaic, 1.0, 3.0, 0.7),
        gabor_bank(mosaic, SIZE, [8.0, 4.0]),
    ]


@pytest.mark.parametrize("stage", linear_stages(), ids=lambda s: s.name)
def test_adjoint_is_the_exact_transpose(stage, rng):
    x = rng.standard_normal(stage.in_shape)
    y = rng.standard_normal(stage.out_shape)
    assert np.vdot(stage.forward(x), y) == pytest.approx(np.vdot(x, stage.adjoint(y)), rel=1e-10)


@pytest.mark.parametrize("stage", linear_stages(), ids=lambda s: s.name)
def test_forward_output_has_the_declared_shape(stage, rng):
    assert stage.forward(rng.standard_normal(stage.in_shape)).shape == stage.out_shape


def test_mosaic_sampling_reads_a_uniform_image_exactly():
    mosaic = _mosaic()
    image = np.stack([np.full((SIZE, SIZE), 0.25), np.full((SIZE, SIZE), 0.75)])
    samples = mosaic_sampling(mosaic, SIZE).forward(image)
    np.testing.assert_allclose(samples, np.where(mosaic.types == 0, 0.25, 0.75))


def test_center_surround_response_to_a_uniform_field():
    mosaic = _mosaic()
    out = center_surround(mosaic, 1.0, 3.0, 0.7).forward(np.ones(len(mosaic)))
    np.testing.assert_allclose(out, 0.3, atol=1e-12)


def test_lattices_have_the_requested_spacing_and_stay_inside():
    square = square_lattice(SIZE, 3.0)
    assert len(square) == 36
    hexagonal = hex_lattice(SIZE, 3.0)
    for points in (square, hexagonal):
        assert points.min() >= 0 and points.max() <= SIZE - 1
    distances = np.linalg.norm(hexagonal[:, None] - hexagonal[None], axis=2)
    nearest = np.sort(distances, axis=1)[:, 1]
    assert np.median(nearest) == pytest.approx(3.0, abs=1e-9)


def test_foveated_lattice_is_denser_at_the_centre():
    points = foveated_lattice(64, 0.5, 4.0)
    assert points.min() >= 0 and points.max() <= 63
    radius = np.linalg.norm(points - 31.5, axis=1)
    inner_density = (radius < 8).sum() / (np.pi * 8**2)
    outer_density = ((radius >= 24) & (radius < 31)).sum() / (np.pi * (31**2 - 24**2))
    assert inner_density > 2.0 * outer_density


def test_foveated_lattice_never_packs_tighter_than_the_minimum():
    points = foveated_lattice(32, 0.01, 2.0, min_spacing_px=1.0)
    distances = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() > 0.7


def test_all_types_at_repeats_each_position_per_type():
    mosaic = all_types_at(square_lattice(SIZE, 4.0), 3)
    assert len(mosaic) == 3 * 16
    assert sorted(set(mosaic.types.tolist())) == [0, 1, 2]


def test_pool_only_connects_cells_of_the_same_type():
    positions = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    types = np.array([0, 1, 0])
    matrix = pool(positions, types, positions, types, 5.0, lambda dy, dx: np.ones_like(dx))
    np.testing.assert_array_equal(matrix.toarray(), [[1, 0, 1], [0, 1, 0], [1, 0, 1]])


def test_pool_with_no_neighbours_gives_an_empty_matrix():
    far = np.array([[100.0, 100.0]])
    matrix = pool(far, np.array([0]), np.zeros((1, 2)), np.array([0]), 1.0,
                  lambda dy, dx: np.ones_like(dx))
    assert matrix.shape == (1, 1) and matrix.nnz == 0
    assert normalize_rows(matrix).nnz == 0


def test_gabor_bank_has_the_expected_number_of_cells():
    mosaic = _mosaic()
    stage = gabor_bank(mosaic, SIZE, [8.0], n_orientations=4)
    # sigma = 0.4 * 8 = 3.2 px, so the grid has 5 x 5 positions for each of 2 types;
    # each position has 1 non-oriented cell and 4 orientations x 2 phases.
    assert stage.out_shape == (5 * 5 * 2 * (1 + 8),)
    assert stage.in_shape == (len(mosaic),)


def test_gabor_kernel_is_even_or_odd_by_phase():
    x = np.linspace(-5, 5, 11)
    zero = np.zeros_like(x)
    even = gabor_kernel(zero, x, 2.0, 5.0, 0.0, 0.0)
    odd = gabor_kernel(zero, x, 2.0, 5.0, 0.0, np.pi / 2.0)
    np.testing.assert_allclose(even, even[::-1], atol=1e-12)
    np.testing.assert_allclose(odd, -odd[::-1], atol=1e-12)


def test_sparse_stage_rejects_a_matrix_of_the_wrong_shape():
    from scipy.sparse import identity
    with pytest.raises(ValueError, match="does not match"):
        SparseStage("bad", identity(4, format="csr"), (5,), (4,))


def test_gabor_bank_density_multiplies_the_cell_count(rng):
    mosaic = _mosaic()
    base = gabor_bank(mosaic, SIZE, [8.0])
    dense = gabor_bank(mosaic, SIZE, [8.0], density=4.0)
    assert 3 * base.out_shape[0] < dense.out_shape[0] < 6 * base.out_shape[0]
    x = rng.standard_normal(dense.in_shape)
    y = rng.standard_normal(dense.out_shape)
    assert np.vdot(dense.forward(x), y) == pytest.approx(np.vdot(x, dense.adjoint(y)), rel=1e-10)
    with pytest.raises(ValueError, match="density must be positive"):
        gabor_bank(mosaic, SIZE, [8.0], density=0.0)
