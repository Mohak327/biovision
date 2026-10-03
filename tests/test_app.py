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
    assert [metric.label for metric in app.metric] == ["PSNR", "SSIM", "Neurons"]


def test_app_switches_species_from_the_registry():
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    selector = next(box for box in app.sidebar.selectbox if box.label == "Species")
    assert list(selector.options) == ["fly", "human", "mouse"]
    selector.set_value("mouse").run()
    assert not app.exception
