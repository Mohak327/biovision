import numpy as np
from PIL import Image

from biovision.cli import main


def test_list_prints_the_species(capsys):
    assert main(["list"]) == 0
    assert capsys.readouterr().out.split() == ["fly", "human", "mouse"]


def test_run_writes_a_figure_and_prints_metrics(tmp_path, capsys):
    out = tmp_path / "panel.png"
    assert main(["run", "--species", "fly", "--size", "32", "--out", str(out)]) == 0
    assert out.stat().st_size > 1000
    printed = capsys.readouterr().out
    assert "fly: PSNR" in printed and "SSIM" in printed


def test_run_accepts_a_user_image_and_options(tmp_path):
    source = tmp_path / "in.jpg"
    pixels = (np.random.default_rng(0).random((50, 70, 3)) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(source)
    out = tmp_path / "panel.png"
    code = main(["run", "--species", "mouse", "--image", str(source), "--size", "32",
                 "--window", "300", "--no-noise", "--lam", "0.01", "--out", str(out)])
    assert code == 0 and out.exists()


def test_report_writes_a_directory(tmp_path):
    out = tmp_path / "results"
    assert main(["report", "--species", "fly", "--size", "32", "--no-sweeps",
                 "--out", str(out)]) == 0
    assert (out / "report.md").exists() and (out / "results.json").exists()


def test_errors_are_reported_without_a_traceback(tmp_path, capsys):
    assert main(["run", "--species", "cat", "--size", "32"]) == 2
    assert "unknown species 'cat'" in capsys.readouterr().err
    assert main(["run", "--species", "fly", "--image", str(tmp_path / "none.png")]) == 2
    assert "no image at" in capsys.readouterr().err
    assert main(["run", "--species", "fly", "--window", "0", "--size", "32"]) == 2
    assert "window_ms must be positive" in capsys.readouterr().err


def test_run_accepts_a_density(tmp_path, capsys):
    out = tmp_path / "panel.png"
    assert main(["run", "--species", "fly", "--size", "32", "--density", "4",
                 "--out", str(out)]) == 0
    assert "receptors" in capsys.readouterr().out
    assert main(["run", "--species", "fly", "--size", "32", "--density", "0"]) == 2
    assert "density must be positive" in capsys.readouterr().err
