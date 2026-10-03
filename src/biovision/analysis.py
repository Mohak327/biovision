"""Experiments built on run(): comparisons across species and parameter sweeps."""
import numpy as np

from .core.registry import species
from .run import RunResult, run

DEFAULT_WINDOWS_MS = (10.0, 30.0, 100.0, 300.0, 1000.0)
DEFAULT_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def compare_species(image, names=None, **settings) -> dict[str, RunResult]:
    """One run per species, with the same image and settings."""
    return {name: run(image, name, **settings) for name in (names or species.names())}


def sweep_window(image, species_name: str, windows_ms=DEFAULT_WINDOWS_MS,
                 seeds=(0, 1, 2), **settings) -> list[dict]:
    """Quality against spike window, averaged over noise seeds."""
    settings = {k: v for k, v in settings.items() if k not in ("window_ms", "seed", "noise")}
    rows = []
    for window in windows_ms:
        runs = [run(image, species_name, window_ms=window, seed=seed, noise=True, **settings)
                for seed in seeds]
        psnrs = [r.metrics["psnr_db"] for r in runs]
        ssims = [r.metrics["ssim"] for r in runs]
        rows.append({
            "species": species_name, "window_ms": float(window),
            "psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
            "ssim_mean": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
        })
    return rows


def sweep_lambda(image, species_name: str, lams=DEFAULT_LAMBDAS, **settings) -> list[dict]:
    """Quality against the regularization strength."""
    settings = {k: v for k, v in settings.items() if k != "lam"}
    rows = []
    for lam in lams:
        result = run(image, species_name, lam=lam, **settings)
        rows.append({"species": species_name, "lam": float(lam),
                     "psnr_db": result.metrics["psnr_db"], "ssim": result.metrics["ssim"]})
    return rows
