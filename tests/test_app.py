from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_renders_the_default_view_without_errors():
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert not app.exception
    assert app.title[0].value == "biovision"
    assert [tab.label for tab in app.tabs] == ["Explore", "Stages", "Analysis", "Compare"]
    assert [metric.label for metric in app.metric] == ["PSNR", "SSIM", "Neurons", "Receptors"]


def test_app_switches_species_from_the_registry():
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    selector = next(box for box in app.sidebar.selectbox if box.label == "Species")
    assert list(selector.options) == ["fly", "human", "mouse"]
    selector.set_value("mouse").run()
    assert not app.exception


def test_app_density_slider_changes_the_receptor_count():
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    receptors = lambda: next(m.value for m in app.metric if m.label == "Receptors")
    before = receptors()
    slider = next(s for s in app.sidebar.select_slider if s.label == "Receptor density")
    assert slider.value == 1.0
    slider.set_value(4.0).run()
    assert not app.exception
    assert int(receptors().replace(",", "")) > 3 * int(before.replace(",", ""))


def test_app_returns_to_earlier_settings_without_error():
    """Going back to settings already run must reuse the result, not fail."""
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    first = [metric.value for metric in app.metric]
    selector = next(box for box in app.sidebar.selectbox if box.label == "Species")
    selector.set_value("mouse").run()
    selector = next(box for box in app.sidebar.selectbox if box.label == "Species")
    selector.set_value("fly").run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == first
