import pytest

from biovision.core.field import VisualField


def test_conversions_round_trip():
    field = VisualField(size_px=120, fov_deg=60.0)
    assert field.deg_per_px == 0.5
    assert field.to_px(5.0) == 10.0
    assert field.to_deg(field.to_px(3.3)) == pytest.approx(3.3)


@pytest.mark.parametrize("size, fov", [(0, 60.0), (1, 60.0), (64, 0.0), (64, -5.0)])
def test_rejects_invalid_values(size, fov):
    with pytest.raises(ValueError):
        VisualField(size, fov)
