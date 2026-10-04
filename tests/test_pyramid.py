"""Multi-scale pooling (phase 2): wide cells pool from a coarse layer, not every receptor."""
import numpy as np
import pytest
from scipy.sparse import random as sparse_random

from biovision.run import build_pipeline, run
from biovision.stages import pyramid
from biovision.stages.gabor import gabor_bank
from biovision.stages.mosaic import Mosaic, all_types_at, square_lattice
from biovision.stages.pyramid import coarse_layer, summarize
from biovision.stages.receptive import RetinaClass, center_surround, opponent_retina
from biovision.stages.sparse import FactoredStage, SparseStage

SIZE = 32
CLASSES = (
    RetinaClass("luminance", (0.5, 0.5, 0.0), gain=2.0, surround_weight=0.7),
    RetinaClass("red_green", (1.0, -1.0, 0.0), gain=8.0, surround_weight=0.5),
    RetinaClass("blue_yellow", (-0.5, -0.5, 1.0), gain=3.0, surround_weight=0.0),
)


def dense_mosaic(size=SIZE, n_types=3):
    return all_types_at(square_lattice(size, 1.0), n_types)


def smooth_signal(mosaic, size=SIZE):
    """A slow wave across the image, different for each type."""
    rows, cols = mosaic.positions.T
    return np.sin(2 * np.pi * rows / size + mosaic.types) + np.cos(2 * np.pi * cols / size)


@pytest.fixture
def always_pool(monkeypatch):
    """Make every pool count as too large to connect directly."""
    monkeypatch.setattr(pyramid, "DIRECT_LIMIT", 0)


def assert_adjoint(stage, rng):
    x = rng.standard_normal(stage.in_shape)
    y = rng.standard_normal(stage.out_shape)
    assert np.vdot(stage.forward(x), y) == pytest.approx(np.vdot(x, stage.adjoint(y)), rel=1e-10)


def test_factored_stage_is_the_sum_of_its_products(rng):
    a = sparse_random(5, 7, 0.5, random_state=1, format="csr")
    b = sparse_random(4, 5, 0.5, random_state=2, format="csr")
    c = sparse_random(4, 7, 0.5, random_state=3, format="csr")
    stage = FactoredStage("sum", [[a, b], [c]], (7,), (2, 2))
    dense = (b @ a + c).toarray()
    x = rng.standard_normal(7)
    y = rng.standard_normal((2, 2))
    np.testing.assert_allclose(stage.forward(x), (dense @ x).reshape(2, 2), atol=1e-12)
    np.testing.assert_allclose(stage.adjoint(y), dense.T @ y.ravel(), atol=1e-12)
    assert_adjoint(stage, rng)


def test_factored_stage_rejects_factors_that_do_not_chain():
    a = sparse_random(5, 7, 0.5, random_state=1, format="csr")
    b = sparse_random(4, 6, 0.5, random_state=2, format="csr")
    with pytest.raises(ValueError, match="does not match"):
        FactoredStage("bad", [[a, b]], (7,), (4,))
    with pytest.raises(ValueError, match="does not match"):
        FactoredStage("bad", [[a]], (7,), (4,))
    with pytest.raises(ValueError, match="at least one"):
        FactoredStage("bad", [], (7,), (4,))


def test_coarse_layer_summarizes_each_type_separately():
    mosaic = dense_mosaic()
    nodes, matrix = coarse_layer(mosaic, 4.0)
    assert nodes.n_types == 3 and len(nodes) == 3 * 8 * 8
    assert matrix.shape == (len(nodes), len(mosaic))
    out = matrix @ np.array([0.2, 0.5, 0.9])[mosaic.types]
    np.testing.assert_allclose(out, np.array([0.2, 0.5, 0.9])[nodes.types], atol=1e-12)


def test_small_pools_stay_direct():
    mosaic = dense_mosaic()
    source, head = summarize(mosaic.positions, mosaic, 10.0, 4.0)
    assert source is mosaic and head == []


def test_large_pools_read_from_a_coarse_layer(always_pool):
    mosaic = dense_mosaic()
    source, head = summarize(mosaic.positions, mosaic, 10.0, 4.0)
    assert len(source) == 3 * 8 * 8 and len(head) == 1
    assert head[0].shape == (len(source), len(mosaic))


def test_a_coarse_layer_that_saves_nothing_is_not_used(always_pool):
    """A grid about as fine as the cells it would summarize only adds blur."""
    mosaic = dense_mosaic()
    source, head = summarize(mosaic.positions, mosaic, 10.0, 1.2)
    assert source is mosaic and head == []


def test_the_switch_follows_the_number_of_connections(monkeypatch):
    mosaic = dense_mosaic()
    pairs = pyramid.connections(mosaic.positions, mosaic.positions, 10.0)
    monkeypatch.setattr(pyramid, "DIRECT_LIMIT", pairs)
    assert summarize(mosaic.positions, mosaic, 10.0, 4.0)[1] == []
    monkeypatch.setattr(pyramid, "DIRECT_LIMIT", pairs - 1)
    assert len(summarize(mosaic.positions, mosaic, 10.0, 4.0)[1]) == 1


def test_pooled_gabor_bank_matches_the_direct_one(always_pool, monkeypatch, rng):
    mosaic = dense_mosaic()
    pooled = gabor_bank(mosaic, SIZE, [16.0], gains=[2.0])
    monkeypatch.setattr(pyramid, "DIRECT_LIMIT", 10**12)
    direct = gabor_bank(mosaic, SIZE, [16.0], gains=[2.0])
    assert isinstance(pooled, FactoredStage) and isinstance(direct, SparseStage)
    assert pooled.in_shape == direct.in_shape and pooled.out_shape == direct.out_shape
    assert_adjoint(pooled, rng)
    x = smooth_signal(mosaic)
    a, b = pooled.forward(x), direct.forward(x)
    assert np.corrcoef(a, b)[0, 1] > 0.99
    assert np.linalg.norm(a - b) < 0.2 * np.linalg.norm(b)


def test_pooled_gabor_bank_mixes_pooled_and_direct_scales(always_pool, rng):
    """The coarse scale pools; the fine one, whose grid would save nothing, stays direct."""
    mosaic = dense_mosaic()
    stage = gabor_bank(mosaic, SIZE, [16.0, 4.0], gains=[1.0, 2.0])
    assert isinstance(stage, FactoredStage)
    assert [len(chain) for chain in stage.terms] == [2, 1]
    coarse = gabor_bank(mosaic, SIZE, [16.0])
    fine = gabor_bank(mosaic, SIZE, [4.0], gains=[2.0])
    plain = fine.out_shape[0] // 9  # alone, the fine scale also gets non-oriented cells
    assert stage.out_shape == (coarse.out_shape[0] + fine.out_shape[0] - plain,)
    assert_adjoint(stage, rng)
    x = smooth_signal(mosaic)
    expected = np.concatenate([coarse.forward(x), fine.forward(x)[plain:]])
    np.testing.assert_allclose(stage.forward(x), expected, atol=1e-12)


def test_a_pooled_cortex_cell_has_the_same_inputs_at_any_image_size(always_pool):
    """Doubling the pixels across the same scene must not add inputs per cell."""
    def inputs_per_cell(size):
        stage = gabor_bank(dense_mosaic(size, 1), size, [size / 2.0])
        return np.diff(stage.terms[0][-1].indptr).max()

    assert inputs_per_cell(64) == inputs_per_cell(32)


def test_pooled_center_surround_matches_the_direct_one(always_pool, monkeypatch, rng):
    mosaic = dense_mosaic()
    pooled = center_surround(mosaic, 0.5, 4.0, 0.7)
    monkeypatch.setattr(pyramid, "DIRECT_LIMIT", 10**12)
    direct = center_surround(mosaic, 0.5, 4.0, 0.7)
    assert isinstance(pooled, FactoredStage) and isinstance(direct, SparseStage)
    assert_adjoint(pooled, rng)
    np.testing.assert_allclose(pooled.forward(np.ones(len(mosaic))), 0.3, atol=1e-12)
    x = smooth_signal(mosaic)
    a, b = pooled.forward(x), direct.forward(x)
    assert np.linalg.norm(a - b) < 0.1 * np.linalg.norm(b)


def test_pooled_opponent_retina_matches_the_direct_one(always_pool, monkeypatch, rng):
    mosaic = dense_mosaic()
    pooled, cells = opponent_retina(mosaic, CLASSES, 0.5, 4.0)
    monkeypatch.setattr(pyramid, "DIRECT_LIMIT", 10**12)
    direct, direct_cells = opponent_retina(mosaic, CLASSES, 0.5, 4.0)
    assert isinstance(pooled, FactoredStage) and isinstance(direct, SparseStage)
    assert pooled.out_shape == direct.out_shape == (len(cells),)
    np.testing.assert_array_equal(cells.positions, direct_cells.positions)
    assert_adjoint(pooled, rng)
    grey = pooled.forward(np.ones(len(mosaic)))
    np.testing.assert_allclose(grey[cells.types == 0], 2.0 * (1 - 0.7), atol=1e-12)
    np.testing.assert_allclose(grey[cells.types != 0], 0.0, atol=1e-12)
    x = smooth_signal(mosaic)
    a, b = pooled.forward(x), direct.forward(x)
    assert np.linalg.norm(a - b) < 0.1 * np.linalg.norm(b)


def test_pooled_identity_classes_reproduce_the_pooled_center_surround(always_pool, rng):
    mosaic = dense_mosaic()
    identity = tuple(RetinaClass(f"type{i}", tuple(float(i == j) for j in range(3)), 1.0, 0.7)
                     for i in range(3))
    mixed, _ = opponent_retina(mosaic, identity, 0.5, 4.0)
    plain = center_surround(mosaic, 0.5, 4.0, 0.7)
    x = rng.random(len(mosaic))
    np.testing.assert_allclose(mixed.forward(x), plain.forward(x), atol=1e-12)


def test_cells_without_any_input_do_not_break_the_layer():
    far = Mosaic(np.array([[0.0, 0.0], [0.0, 40.0]]), np.array([0, 0]), 1)
    nodes, matrix = coarse_layer(far, 4.0)
    assert np.all(np.isfinite(matrix.data))
    assert matrix.shape == (len(nodes), 2)


def test_human_eye_built_with_pooling_reconstructs_as_well(sample, monkeypatch):
    """The same eye through coarse layers must give about the same picture."""
    direct = run(sample, "human", size_px=96)
    build_pipeline.cache_clear()
    monkeypatch.setattr(pyramid, "DIRECT_LIMIT", 0)
    try:
        pooled = run(sample, "human", size_px=96)
    finally:
        build_pipeline.cache_clear()
    kinds = {s.name: type(s) for s in pooled.pipeline.linear_stages}
    assert kinds["gabor"] is FactoredStage
    assert pooled.metrics["neurons"] == direct.metrics["neurons"]
    assert abs(pooled.metrics["psnr_db"] - direct.metrics["psnr_db"]) < 0.5
