"""Real optics (phase 9 of the human vision audit): a blur per receptor type."""
import numpy as np
import pytest

from biovision import species
from biovision.core.field import VisualField
from biovision.species import fly, human, mouse
from biovision.stages.optics import OpticalBlur, defocus_sigma_deg

SIZE = 16


def test_each_channel_is_blurred_by_its_own_sigma(rng):
    x = rng.random((3, SIZE, SIZE))
    out = OpticalBlur([0.0, 1.0, 2.5], 3, SIZE).forward(x)
    for channel, sigma in enumerate([0.0, 1.0, 2.5]):
        alone = OpticalBlur(sigma, 1, SIZE).forward(x[channel:channel + 1])[0]
        np.testing.assert_allclose(out[channel], alone, atol=1e-12)


def test_one_sigma_for_every_channel_is_the_blur_as_it_was(rng):
    x = rng.random((2, SIZE, SIZE))
    np.testing.assert_array_equal(OpticalBlur([1.5, 1.5], 2, SIZE).forward(x),
                                  OpticalBlur(1.5, 2, SIZE).forward(x))


@pytest.mark.parametrize("sigma", [[1.0, 2.0], [1.0, -2.0, 1.0]])
def test_blur_rejects_sigmas_that_do_not_fit_the_channels(sigma):
    with pytest.raises(ValueError, match="sigma_px"):
        OpticalBlur(sigma, 3, SIZE)


def test_defocus_blur_is_a_quarter_of_the_blur_circle():
    """A pupil of 3 mm, 1 dioptre out of focus: a blur circle 3 milliradians across."""
    assert defocus_sigma_deg(3.0, 1.0) == pytest.approx(np.degrees(0.003) / 4.0)
    assert defocus_sigma_deg(3.0, -1.0) == defocus_sigma_deg(3.0, 1.0)
    assert defocus_sigma_deg(6.0, 1.0) == pytest.approx(2.0 * defocus_sigma_deg(3.0, 1.0))
    assert defocus_sigma_deg(3.0, 0.0) == 0.0
    with pytest.raises(ValueError, match="pupil_mm"):
        defocus_sigma_deg(-1.0, 1.0)


def optics(name, size=64, fov=60.0):
    pipeline = species.get(name)(VisualField(size, fov))
    return next(s for s in pipeline.linear_stages if s.name == "optics")


def test_the_human_eye_blurs_blue_far_more_than_red_and_green():
    """Chromatic aberration: S cones see about a dioptre of defocus (Thibos et al. 1992)."""
    assert len(human.PARAMS.chromatic_defocus_d) == 3 and human.PARAMS.pupil_mm > 0
    long, middle, short = optics("human", fov=2.0).sigma_px
    assert short > 4 * long and short > 4 * middle
    # 0.95 D through 3 mm is a blur of 2.4 arcminutes; in focus it is under half of one.
    field = VisualField(64, 2.0)
    assert field.to_deg(short) == pytest.approx(0.041, abs=0.003)
    assert field.to_deg(long) == pytest.approx(human.PARAMS.blur_sigma_deg, rel=0.1)


@pytest.mark.parametrize("module", [mouse, fly])
def test_mouse_and_fly_keep_one_blur_for_every_receptor(module):
    assert module.PARAMS.chromatic_defocus_d == ()
    stage = optics(module.__name__.rsplit(".", 1)[-1])
    assert np.ndim(stage.sigma_px) == 0
