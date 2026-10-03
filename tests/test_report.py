import csv
import json
import zipfile
from io import BytesIO

import pytest
from matplotlib.figure import Figure

from biovision.analysis import compare_species, sweep_lambda, sweep_window
from biovision.report import export, figures, tables
from biovision.report.export import Report, build_report, write_report, zip_report

SMALL = dict(size_px=32)


@pytest.fixture(scope="module")
def results(sample):
    return compare_species(sample, **SMALL)


@pytest.mark.parametrize("key", list(figures.PER_SPECIES))
@pytest.mark.parametrize("name", ["fly", "human", "mouse"])
def test_every_per_species_figure_draws(results, name, key):
    assert isinstance(figures.PER_SPECIES[key](results[name]), Figure)


def test_multi_species_figures_draw(results, sample):
    assert isinstance(figures.species_grid(results), Figure)
    assert isinstance(figures.convergence(results), Figure)
    windows = {"fly": sweep_window(sample, "fly", windows_ms=(30.0, 300.0), seeds=(0, 1), **SMALL)}
    lams = {"fly": sweep_lambda(sample, "fly", lams=(1e-3, 1e-1), **SMALL)}
    assert isinstance(figures.window_sweep(windows), Figure)
    assert isinstance(figures.lambda_sweep(lams), Figure)


def test_tables_have_the_expected_columns(results):
    rows = tables.results_table(results)
    assert [row["species"] for row in rows] == ["fly", "human", "mouse"]
    assert {"neurons", "receptors", "compression_ratio", "psnr_db", "ssim", "lambda",
            "iterations", "converged", "runtime_s", "mean_spikes"} <= set(rows[0])
    parameters = tables.parameters_table(results["mouse"].pipeline)
    by_name = {row["parameter"]: row for row in parameters}
    assert by_name["spacing_deg"]["value"] == "1.0"
    assert by_name["spacing_deg"]["unit"] == "degrees"
    settings = {row["setting"]: row["value"] for row in tables.settings_table(results["fly"])}
    assert settings["species"] == "fly" and settings["size_px"] == "32"
    assert "biovision_version" in settings


def test_write_report_creates_every_file(results, tmp_path):
    out = write_report(Report(results), tmp_path / "report")
    names = {path.name for path in out.iterdir()}
    expected = {"report.md", "results.json", "results.csv", "settings.csv", "parameters.csv",
                "species_grid.png", "species_grid.pdf", "convergence.png", "convergence.pdf"}
    for species_name in results:
        for key in figures.PER_SPECIES:
            expected |= {f"{species_name}_{key}.png", f"{species_name}_{key}.pdf"}
    assert expected <= names
    assert "window_sweep.png" not in names
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert [row["species"] for row in data["results"]] == ["fly", "human", "mouse"]
    with open(out / "results.csv", newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    text = (out / "report.md").read_text(encoding="utf-8")
    for heading in ("## Methods", "## Results", "## Figures", "## References", "## Limits"):
        assert heading in text
    assert "Land MF (1997)" in text and "![" in text


def test_build_report_with_sweeps_adds_sweep_outputs(sample, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "sweep_window", lambda image, name, **s: sweep_window(
        image, name, windows_ms=(30.0, 300.0), seeds=(0,), **s))
    monkeypatch.setattr(export, "sweep_lambda", lambda image, name, **s: sweep_lambda(
        image, name, lams=(1e-2,), **s))
    report = build_report(sample, ["fly"], **SMALL)
    out = write_report(report, tmp_path)
    names = {path.name for path in out.iterdir()}
    assert {"window_sweep.png", "lambda_sweep.pdf", "window_sweep.csv",
            "lambda_sweep.csv"} <= names


def test_zip_report_contains_the_written_files(results, tmp_path):
    out = write_report(Report({"fly": results["fly"]}), tmp_path)
    with zipfile.ZipFile(BytesIO(zip_report(out))) as archive:
        assert "report.md" in archive.namelist()
        assert "fly_pipeline.png" in archive.namelist()


def _titles_inside(figure):
    """True if every axes title and the figure title lie within the figure."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    texts = [axis.title for axis in figure.axes if axis.get_title()]
    if figure._suptitle is not None:
        texts.append(figure._suptitle)
    box = figure.bbox
    return all(box.x0 <= extent.x0 and extent.x1 <= box.x1 and extent.y1 <= box.y1
               for extent in (text.get_window_extent(renderer) for text in texts))


@pytest.mark.parametrize("key", list(figures.PER_SPECIES))
@pytest.mark.parametrize("name", ["fly", "human", "mouse"])
def test_per_species_figure_titles_are_not_cut_off(results, name, key):
    assert _titles_inside(figures.PER_SPECIES[key](results[name]))


def test_multi_species_figure_titles_are_not_cut_off(results):
    assert _titles_inside(figures.species_grid(results))
    assert _titles_inside(figures.convergence(results))
