"""One pure function per table. A table is a list of row dictionaries."""
import dataclasses

from .. import __version__
from ..core.pipeline import Pipeline
from ..run import RunResult

UNITS = {
    "blur_sigma_deg": "degrees", "spacing_deg": "degrees", "e2_deg": "degrees",
    "center_sigma_deg": "degrees", "surround_sigma_deg": "degrees",
    "cortex_sf_cpd": "cycles/degree", "rest_hz": "spikes/s",
}


def parameters_table(pipeline: Pipeline) -> list[dict]:
    """Every parameter of a species' eye, with its unit."""
    params = pipeline.metadata["params"]
    rows = [{"species": pipeline.name, "parameter": field.name,
             "value": str(getattr(params, field.name)), "unit": UNITS.get(field.name, "")}
            for field in dataclasses.fields(params)]
    rows.append({"species": pipeline.name, "parameter": "cells_per_position",
                 "value": f"{pipeline.metadata['cells_per_position']:.2f}", "unit": "cells"})
    rows.append({"species": pipeline.name, "parameter": "cells_per_neuron",
                 "value": f"{pipeline.metadata['cells_per_neuron']:.2f}", "unit": "cells"})
    return rows


def settings_table(result: RunResult) -> list[dict]:
    """The settings of a run, plus the package version."""
    rows = [{"setting": key, "value": str(value)}
            for key, value in dataclasses.asdict(result.settings).items()]
    rows.append({"setting": "biovision_version", "value": __version__})
    return rows


def results_table(results: dict[str, RunResult]) -> list[dict]:
    """One row of outcomes per species."""
    rows = []
    for name, result in results.items():
        metrics, rebuilt = result.metrics, result.reconstruction
        rows.append({
            "species": name,
            "neurons": int(metrics["neurons"]),
            "receptors": len(result.pipeline.metadata["mosaic"]),
            "compression_ratio": round(metrics["compression_ratio"], 4),
            "mean_spikes": round(metrics["mean_spikes"], 2),
            "psnr_db": round(metrics["psnr_db"], 2),
            "ssim": round(metrics["ssim"], 4),
            "lambda": float(f"{result.settings.lam:.4g}"),
            "iterations": rebuilt.iterations,
            "converged": rebuilt.converged,
            "runtime_s": round(result.runtime_s, 2),
        })
    return rows
