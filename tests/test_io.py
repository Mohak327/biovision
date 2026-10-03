import numpy as np
import pytest
from PIL import Image

from biovision import io


def test_samples_are_bundled():
    assert io.sample_names() == ["astronaut", "cat", "coffee"]
    image = io.load_sample("cat")
    assert image.ndim == 3 and image.shape[2] == 3
    assert 0.0 <= image.min() and image.max() <= 1.0
    with pytest.raises(KeyError, match="unknown sample 'dog'"):
        io.load_sample("dog")


def test_to_square_crops_the_centre_of_a_wide_image():
    wide = np.zeros((10, 30, 3))
    wide[:, 10:20] = 1.0
    square = io.to_square(wide, 10)
    assert square.shape == (10, 10, 3)
    np.testing.assert_allclose(square, 1.0)


@pytest.mark.parametrize("shape", [(40, 40), (40, 40, 1), (40, 40, 4), (37, 53, 3), (2, 2, 3)])
def test_to_square_accepts_grayscale_alpha_odd_and_tiny_sizes(shape, rng):
    out = io.to_square(rng.random(shape), 16)
    assert out.shape == (16, 16, 3)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_integer_images_are_scaled_by_their_type_range():
    eight = io.as_rgb(np.full((4, 4, 3), 255, dtype=np.uint8))
    sixteen = io.as_rgb(np.full((4, 4, 3), 65535, dtype=np.uint16))
    np.testing.assert_allclose(eight, 1.0)
    np.testing.assert_allclose(sixteen, 1.0)


def test_float_images_in_0_255_are_rescaled():
    np.testing.assert_allclose(io.as_rgb(np.full((4, 4, 3), 255.0)), 1.0)


@pytest.mark.parametrize("bad, message", [
    (np.zeros((4, 4, 2)), "must be"),
    (np.zeros((4,)), "must be"),
    (np.zeros((1, 9, 3)), "at least 2 x 2"),
    (np.full((4, 4, 3), np.nan), "NaN or infinite"),
    (np.array([["a", "b"], ["c", "d"]]), "numeric"),
])
def test_as_rgb_rejects_bad_arrays(bad, message):
    with pytest.raises(ValueError, match=message):
        io.as_rgb(bad)


def test_load_image_reads_files_and_reports_problems(tmp_path):
    good = tmp_path / "good.png"
    Image.fromarray(np.full((6, 6), 128, dtype=np.uint8)).save(good)
    image = io.load_image(good)
    assert image.shape == (6, 6, 3)
    assert image[0, 0, 0] == pytest.approx(128 / 255)
    with pytest.raises(FileNotFoundError, match="no image at"):
        io.load_image(tmp_path / "missing.png")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="not a readable image"):
        io.load_image(broken)
