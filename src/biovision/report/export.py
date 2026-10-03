"""Build a full report and write it to disk: figures, tables, numbers, Markdown."""
import csv
import io as std_io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..analysis import compare_species, sweep_density, sweep_lambda, sweep_window
from ..run import RunResult
from . import figures, tables

CAPTIONS = {
    "pipeline": "The image after each stage of the visual system, and the reconstruction.",
    "mosaic": "Receptor positions, coloured by receptor type.",
    "filters": "The centre-surround profile and the cortical Gabor kernels.",
    "colour": "Receptor sensitivity to red, green and blue, and the colour that survives.",
    "neural_code": "Spike counts across the output neurons.",
    "error": "Reconstruction error over the image and per colour channel.",
    "spectrum": "Power spectrum of the original and the reconstruction, with the "
                "eye's sampling limit.",
}


@dataclass(frozen=True)
class Report:
    results: dict[str, RunResult]
    window_sweeps: dict[str, list[dict]] = field(default_factory=dict)
    lambda_sweeps: dict[str, list[dict]] = field(default_factory=dict)
    density_sweeps: dict[str, list[dict]] = field(default_factory=dict)


def build_report(image, names=None, sweeps: bool = True, **settings) -> Report:
    """Run every species, and optionally the window, lambda and density sweeps."""
    results = compare_species(image, names, **settings)
    if not sweeps:
        return Report(results)
    return Report(
        results,
        {name: sweep_window(image, name, **settings) for name in results},
        {name: sweep_lambda(image, name, **settings) for name in results},
        {name: sweep_density(image, name, **settings) for name in results},
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict]) -> str:
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(row[key]) for key in header) + " |" for row in rows]
    return "\n".join(lines)


def _save(figure, out_dir: Path, name: str) -> None:
    figure.savefig(out_dir / f"{name}.png", dpi=300)
    figure.savefig(out_dir / f"{name}.pdf")


def write_report(report: Report, out_dir) -> Path:
    """Write figures (PNG and PDF), tables (CSV), results.json and report.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = report.results
    first = next(iter(results.values()))

    figure_names: list[tuple[str, str]] = []
    _save(figures.species_grid(results), out_dir, "species_grid")
    figure_names.append(("species_grid", "Reconstructions from every species."))
    for species_name, result in results.items():
        for key, draw in figures.PER_SPECIES.items():
            name = f"{species_name}_{key}"
            _save(draw(result), out_dir, name)
            figure_names.append((name, f"{species_name}: {CAPTIONS[key]}"))
    _save(figures.convergence(results), out_dir, "convergence")
    figure_names.append(("convergence", "Relative residual of the solver per iteration."))
    if report.window_sweeps:
        _save(figures.window_sweep(report.window_sweeps), out_dir, "window_sweep")
        figure_names.append(("window_sweep", "Quality against spike window (mean and "
                                             "standard deviation over noise seeds)."))
    if report.lambda_sweeps:
        _save(figures.lambda_sweep(report.lambda_sweeps), out_dir, "lambda_sweep")
        figure_names.append(("lambda_sweep", "Quality against regularization strength."))
    if report.density_sweeps:
        _save(figures.density_sweep(report.density_sweeps), out_dir, "density_sweep")
        figure_names.append(("density_sweep", "Quality against neuron density, with receptor "
                                              "density scaled from the real eye (1)."))

    table_rows = {
        "results": tables.results_table(results),
        "settings": tables.settings_table(first),
        "parameters": [row for r in results.values()
                       for row in tables.parameters_table(r.pipeline)],
    }
    if report.window_sweeps:
        table_rows["window_sweep"] = [r for rows in report.window_sweeps.values() for r in rows]
    if report.lambda_sweeps:
        table_rows["lambda_sweep"] = [r for rows in report.lambda_sweeps.values() for r in rows]
    if report.density_sweeps:
        table_rows["density_sweep"] = [r for rows in report.density_sweeps.values() for r in rows]
    for name, rows in table_rows.items():
        _write_csv(out_dir / f"{name}.csv", rows)
    (out_dir / "results.json").write_text(json.dumps(table_rows, indent=2), encoding="utf-8")

    citations = sorted({c for r in results.values() for c in r.pipeline.citations})
    settings = first.settings
    noise = (f"Poisson spike counts in a {settings.window_ms:g} ms window"
             if settings.noise else "noise-free spike counts")
    lines = [
        "# biovision report", "",
        "## Methods", "",
        f"Each image was centre-cropped, resized to {settings.size_px} x {settings.size_px} "
        f"pixels and treated as spanning {settings.fov_deg:g} degrees of visual angle. "
        "For each species the image was encoded by a model of the early visual system: "
        "projection onto the photoreceptor types, optical blur, sampling by the receptor "
        "mosaic, centre-surround filtering and, for mammals, a bank of Gabor filters, "
        f"followed by a threshold-linear firing rate and {noise}. The image was then "
        "reconstructed by regularized least squares, solved with conjugate gradients, "
        "using a prior that penalizes image gradients and differences between colour "
        "channels. No parameters were learned from data.", "",
        "## Results", "", _markdown_table(table_rows["results"]), "",
        "## Settings", "", _markdown_table(table_rows["settings"]), "",
        "## Figures", "",
    ]
    for index, (name, caption) in enumerate(figure_names, start=1):
        lines += [f"![{caption}]({name}.png)", "", f"**Figure {index}.** {caption}", ""]
    lines += ["## Species parameters", "", _markdown_table(table_rows["parameters"]), "",
              "## Limits", "",
              "- Ultraviolet is approximated from the blue channel of an RGB image.",
              "- Image values are treated as linear light.",
              "- Receptors smaller than a pixel are pooled: the image, not the eye, "
              "sets the resolution there.",
              "- The mouse model is its cone pathway in daylight.", "",
              "## References", ""]
    lines += [f"{index}. {citation}" for index, citation in enumerate(citations, start=1)]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir


def zip_report(out_dir) -> bytes:
    """The contents of a report directory as a zip archive."""
    out_dir = Path(out_dir)
    buffer = std_io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out_dir.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    return buffer.getvalue()
