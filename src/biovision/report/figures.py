"""One pure function per figure: results in, a Matplotlib Figure out."""
import numpy as np

from ..core.metrics import radial_power_spectrum
from ..run import RunResult
from ..stages.gabor import gabor_kernel
from ..stages.receptive import gaussian
from .style import PALETTE, new_figure, show_image


def _params(result: RunResult):
    return result.pipeline.metadata["params"]


def _mosaic(result: RunResult):
    return result.pipeline.metadata["mosaic"]


def _as_display(array: np.ndarray) -> np.ndarray:
    """A (types, size, size) stage output as a displayable image."""
    array = np.asarray(array)
    if array.shape[0] == 3:
        return np.clip(array.transpose(1, 2, 0), 0.0, 1.0)
    return np.clip(array.mean(axis=0), 0.0, 1.0)


def _scatter(axis, result: RunResult, values: np.ndarray, title: str):
    mosaic = _mosaic(result)
    size = result.settings.size_px
    marker = max(1.0, 12000.0 / len(mosaic))
    axis.scatter(mosaic.positions[:, 1], mosaic.positions[:, 0], c=values,
                 s=marker, cmap="gray", linewidths=0)
    axis.set_xlim(0, size - 1)
    axis.set_ylim(size - 1, 0)
    axis.set_aspect("equal")
    axis.set_facecolor("black")
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def pipeline_panel(result: RunResult):
    """The image after each stage, from the original to the reconstruction."""
    inter = result.code.intermediates
    n_receptors = len(_mosaic(result))
    panels = [("original", "image", result.original)]
    for name, output in inter.items():
        if output.ndim == 3:
            panels.append((name, "image", _as_display(output)))
        elif output.ndim == 1 and len(output) == n_receptors:
            panels.append((name, "scatter", output))
    panels.append(("reconstruction", "image", result.reconstructed))
    figure, axes = new_figure(1, len(panels), width=2.3 * len(panels), height=2.6)
    for axis, (name, kind, data) in zip(axes[0], panels):
        if kind == "image":
            show_image(axis, data, name.replace("_", " "), cmap="gray", vmin=0, vmax=1)
        else:
            _scatter(axis, result, data, name.replace("_", " "))
    figure.suptitle(f"{result.settings.species}: stage by stage", fontsize=11)
    return figure


def mosaic_map(result: RunResult):
    """Where the receptors sit, coloured by receptor type."""
    mosaic, params = _mosaic(result), _params(result)
    size = result.settings.size_px
    figure, axes = new_figure(1, 1, width=4.5, height=4.5)
    axis = axes[0, 0]
    axis.imshow(result.original, alpha=0.35)
    marker = max(1.5, 9000.0 / len(mosaic))
    for index, name in enumerate(params.receptor_names):
        chosen = mosaic.types == index
        axis.scatter(mosaic.positions[chosen, 1], mosaic.positions[chosen, 0],
                     s=marker, color=PALETTE[index], label=f"{name} ({chosen.sum()})",
                     linewidths=0)
    axis.set_xlim(0, size - 1)
    axis.set_ylim(size - 1, 0)
    axis.set_xlabel("pixels")
    axis.set_ylabel("pixels")
    axis.set_title(f"{result.settings.species}: receptor mosaic ({len(mosaic)} receptors)")
    axis.legend(loc="upper right", fontsize=8, markerscale=2, framealpha=0.9, frameon=True)
    return figure


def filter_gallery(result: RunResult):
    """The centre-surround profile and, if present, the cortical Gabor kernels."""
    params, field = _params(result), result.pipeline.field
    wavelengths = [1.0 / sf for sf in params.cortex_sf_cpd
                   if field.to_px(1.0 / sf) >= 2.0]
    figure, axes = new_figure(1, 1 + len(wavelengths), width=3.2 * (1 + len(wavelengths)),
                              height=2.9)
    reach = 3.0 * params.surround_sigma_deg
    x = np.linspace(-reach, reach, 401)
    zero = np.zeros_like(x)
    centre = gaussian(zero, x, params.center_sigma_deg)
    surround = gaussian(zero, x, params.surround_sigma_deg)
    profile = centre / centre.sum() - params.surround_weight * surround / surround.sum()
    axis = axes[0, 0]
    axis.plot(x, profile / np.abs(profile).max(), color=PALETTE[0])
    axis.axhline(0.0, color="gray", linewidth=0.5)
    axis.set_xlabel("position (degrees)")
    axis.set_ylabel("weight (normalized)")
    axis.set_title("centre-surround")
    for axis, wavelength in zip(axes[0, 1:], wavelengths):
        sigma = 0.4 * wavelength
        grid = np.linspace(-2.5 * sigma, 2.5 * sigma, 101)
        dx, dy = np.meshgrid(grid, grid)
        kernel = gabor_kernel(dy, dx, sigma, wavelength, np.pi / 4.0, 0.0)
        axis.imshow(kernel, cmap="RdBu_r", vmin=-1, vmax=1,
                    extent=(grid[0], grid[-1], grid[-1], grid[0]))
        axis.set_xlabel("degrees")
        axis.set_title(f"Gabor, {1.0 / wavelength:.2g} cycles/degree")
    figure.suptitle(f"{result.settings.species}: receptive fields", fontsize=11)
    return figure


def color_model(result: RunResult):
    """How each receptor type weights red, green and blue, and what survives."""
    params = _params(result)
    matrix = np.asarray(params.color_matrix)
    figure, axes = new_figure(1, 3, width=9.0, height=2.9)
    axis = axes[0, 0]
    width = 0.8 / len(matrix)
    for index, (name, row) in enumerate(zip(params.receptor_names, matrix)):
        axis.bar(np.arange(3) + index * width, row, width, label=name, color=PALETTE[index])
    axis.set_xticks(np.arange(3) + 0.4 - width / 2.0)
    axis.set_xticklabels(["red", "green", "blue"])
    axis.set_ylabel("weight")
    axis.set_title("receptor sensitivity")
    axis.legend(fontsize=8)
    visible = np.clip(result.original @ (np.linalg.pinv(matrix) @ matrix).T, 0.0, 1.0)
    show_image(axes[0, 1], result.original, "original")
    show_image(axes[0, 2], visible, "colour the receptors capture")
    figure.suptitle(f"{result.settings.species}: colour model", fontsize=11)
    return figure


def neural_code(result: RunResult):
    """The distribution of spike counts across the output neurons."""
    counts = np.asarray(result.code.responses).ravel()
    figure, axes = new_figure(1, 1, width=4.5, height=3.0)
    axis = axes[0, 0]
    axis.hist(counts, bins=40, color=PALETTE[0])
    axis.set_xlabel(f"spikes in {result.settings.window_ms:g} ms")
    axis.set_ylabel("neurons")
    axis.set_title(f"{result.settings.species}: {len(counts)} neurons, "
                   f"mean {counts.mean():.1f} spikes")
    return figure


def error_map(result: RunResult):
    """Where the reconstruction is wrong, and by how much per channel."""
    error = result.reconstructed - result.original
    figure, axes = new_figure(1, 2, width=7.5, height=3.2)
    image = axes[0, 0].imshow(error.mean(axis=-1), cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[0, 0].set_title("reconstruction minus original")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    figure.colorbar(image, ax=axes[0, 0], shrink=0.8, label="error")
    rmse = np.sqrt((error**2).mean(axis=(0, 1)))
    axes[0, 1].bar(["red", "green", "blue"], rmse, color=[PALETTE[1], PALETTE[2], PALETTE[0]])
    axes[0, 1].set_ylabel("root-mean-square error")
    axes[0, 1].set_title("error per channel")
    figure.suptitle(f"{result.settings.species}: reconstruction error", fontsize=11)
    return figure


def spectrum(result: RunResult):
    """Power against spatial frequency, with the eye's sampling limit marked."""
    params, settings = _params(result), result.settings
    frequency, original = radial_power_spectrum(result.original)
    _, rebuilt = radial_power_spectrum(result.reconstructed)
    cpd = frequency / settings.fov_deg
    figure, axes = new_figure(1, 1, width=5.0, height=3.3)
    axis = axes[0, 0]
    axis.loglog(cpd[1:], original[1:], color=PALETTE[6], label="original")
    axis.loglog(cpd[1:], rebuilt[1:], color=PALETTE[1], label="reconstruction")
    image_limit = settings.size_px / (2.0 * settings.fov_deg)
    eye_limit = min(1.0 / (2.0 * params.spacing_deg), image_limit)
    axis.axvline(eye_limit, color=PALETTE[0], linestyle="--",
                 label=f"sampling limit ({eye_limit:.2g} cycles/degree)")
    axis.set_xlabel("spatial frequency (cycles/degree)")
    axis.set_ylabel("power")
    axis.set_title(f"{settings.species}: power spectrum")
    axis.legend(fontsize=8)
    return figure


def window_sweep(rows_by_species: dict[str, list[dict]]):
    """Quality against spike window, with error bars over noise seeds."""
    figure, axes = new_figure(1, 2, width=8.0, height=3.2)
    for index, (name, rows) in enumerate(rows_by_species.items()):
        windows = [row["window_ms"] for row in rows]
        for axis, key in zip(axes[0], ("psnr", "ssim")):
            axis.errorbar(windows, [row[f"{key}_mean"] for row in rows],
                          yerr=[row[f"{key}_std"] for row in rows], marker="o",
                          capsize=3, color=PALETTE[index], label=name)
    for axis, label in zip(axes[0], ("PSNR (dB)", "SSIM")):
        axis.set_xscale("log")
        axis.set_xlabel("spike window (ms)")
        axis.set_ylabel(label)
        axis.legend(fontsize=8)
    figure.suptitle("Reconstruction quality against spike window", fontsize=11)
    return figure


def lambda_sweep(rows_by_species: dict[str, list[dict]]):
    """Quality against the regularization strength."""
    figure, axes = new_figure(1, 2, width=8.0, height=3.2)
    for index, (name, rows) in enumerate(rows_by_species.items()):
        lams = [row["lam"] for row in rows]
        axes[0, 0].plot(lams, [row["psnr_db"] for row in rows], marker="o",
                        color=PALETTE[index], label=name)
        axes[0, 1].plot(lams, [row["ssim"] for row in rows], marker="o",
                        color=PALETTE[index], label=name)
    for axis, label in zip(axes[0], ("PSNR (dB)", "SSIM")):
        axis.set_xscale("log")
        axis.set_xlabel("regularization strength (lambda)")
        axis.set_ylabel(label)
        axis.legend(fontsize=8)
    figure.suptitle("Reconstruction quality against regularization", fontsize=11)
    return figure


def density_sweep(rows_by_species: dict[str, list[dict]]):
    """Quality against receptor density. Points are labelled with receptor counts."""
    figure, axes = new_figure(1, 2, width=8.0, height=3.2)
    for index, (name, rows) in enumerate(rows_by_species.items()):
        densities = [row["density"] for row in rows]
        for axis, key in zip(axes[0], ("psnr_db", "ssim")):
            axis.plot(densities, [row[key] for row in rows], marker="o",
                      color=PALETTE[index], label=name)
    for axis, label in zip(axes[0], ("PSNR (dB)", "SSIM")):
        axis.set_xscale("log")
        axis.axvline(1.0, color="gray", linewidth=0.5, linestyle="--")
        axis.set_xlabel("receptor density (1 = the real eye)")
        axis.set_ylabel(label)
        axis.legend(fontsize=8)
    figure.suptitle("Reconstruction quality against neuron density", fontsize=11)
    return figure


def species_grid(results: dict[str, RunResult]):
    """The original and every species' reconstruction, side by side."""
    first = next(iter(results.values()))
    figure, axes = new_figure(1, 1 + len(results), width=2.6 * (1 + len(results)), height=3.5)
    show_image(axes[0, 0], first.original, "original")
    for axis, (name, result) in zip(axes[0, 1:], results.items()):
        metrics = result.metrics
        show_image(axis, result.reconstructed,
                   f"{name}\n{metrics['psnr_db']:.1f} dB, SSIM {metrics['ssim']:.2f}\n"
                   f"{int(metrics['neurons'])} neurons")
    return figure


def convergence(results: dict[str, RunResult]):
    """The solver's relative residual at each iteration."""
    figure, axes = new_figure(1, 1, width=5.0, height=3.2)
    axis = axes[0, 0]
    for index, (name, result) in enumerate(results.items()):
        residuals = result.reconstruction.residuals
        axis.semilogy(np.arange(1, len(residuals) + 1), residuals,
                      color=PALETTE[index], label=name)
    axis.set_xlabel("conjugate-gradient iteration")
    axis.set_ylabel("relative residual")
    axis.set_title("Solver convergence")
    axis.legend(fontsize=8)
    return figure


# Figures drawn once per species, by file name.
PER_SPECIES = {
    "pipeline": pipeline_panel, "mosaic": mosaic_map, "filters": filter_gallery,
    "colour": color_model, "neural_code": neural_code, "error": error_map,
    "spectrum": spectrum,
}
