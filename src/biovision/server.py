"""HTTP server for the web app: a thin layer over run(), with no mathematics.

Start it with `biovision serve`. Every POST takes a multipart form with
`settings` (JSON) and either `sample` (a bundled image name) or `image` (an
uploaded file). Runs are streamed as newline-delimited JSON events.
"""
import base64
import dataclasses
import io as std_io
import json
import queue
import tempfile
import threading
import time
import warnings
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

from . import io
from .analysis import (compare_species, sampling_limit_cpd, sweep_density, sweep_lambda,
                       sweep_window)
from .core.field import VisualField
from .core.metrics import radial_power_spectrum
from .core.registry import species
from .report import tables
from .report.export import build_report, write_report, zip_report
from .run import RunResult, build_pipeline, run, spike_counts, stage_outputs

FRAME_INTERVAL_S = 0.05  # send a decoding frame at most this often
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
HISTOGRAM_BINS = 30
WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
SWEEPS = {"window": sweep_window, "lambda": sweep_lambda, "density": sweep_density}

app = FastAPI(title="biovision", docs_url="/api/docs", openapi_url="/api/openapi.json")


class Settings(BaseModel):
    """The options of run(), validated."""

    species: str
    size_px: int = Field(128, ge=2, le=512)
    fov_deg: float = Field(60.0, gt=0, le=180)
    window_ms: float = Field(100.0, gt=0)
    noise: bool = True
    lam: float | None = Field(None, gt=0)
    seed: int = Field(0, ge=0)
    density: float = Field(1.0, gt=0, le=64)
    neuron_density: float = Field(1.0, gt=0, le=16)
    looks: int = Field(1, ge=1, le=16)
    photons_per_s: float | None = Field(None, gt=0)

    def run_options(self) -> dict:
        return self.model_dump(exclude={"species"})


class _Cancelled(Exception):
    """Raised inside a run when the client has gone away."""


def _parse_settings(text: str) -> Settings:
    try:
        settings = Settings.model_validate_json(text)
    except ValidationError as error:
        raise HTTPException(422, json.loads(error.json(include_url=False))) from None
    if settings.species not in species.names():
        known = ", ".join(species.names())
        raise HTTPException(400, f"unknown species '{settings.species}'; registered: {known}")
    return settings


async def _read_image(image: UploadFile | None, sample: str | None) -> np.ndarray:
    if image is not None:
        data = await image.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(400, "That image is larger than 25 MB.")
        try:
            with Image.open(std_io.BytesIO(data)) as picture:
                return io.as_rgb(np.asarray(picture.convert("RGB")))
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise HTTPException(400, f"Could not read that image: {error}") from None
    if sample is None:
        raise HTTPException(400, "Send an image file or the name of a sample.")
    if sample not in io.sample_names():
        raise HTTPException(400, f"unknown sample '{sample}'; "
                                 f"available: {', '.join(io.sample_names())}")
    return io.load_sample(sample)


def _png(image: np.ndarray) -> str:
    """A (height, width, 3) image in [0, 1] as base64 PNG."""
    buffer = std_io.BytesIO()
    Image.fromarray(np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).save(
        buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def _line(event: dict) -> bytes:
    return (json.dumps(event) + "\n").encode("utf-8")


def mosaic_event(pipeline) -> dict:
    """Receptor positions (x, y in [0, 1], interleaved) and their type indices."""
    mosaic = pipeline.metadata["mosaic"]
    extent = pipeline.field.size_px - 1
    xy = mosaic.positions[:, ::-1] / extent  # (row, col) -> (x, y)
    return {"type": "mosaic", "count": len(mosaic),
            "positions": _b64(xy.ravel(), "<f4"), "types": _b64(mosaic.types, "u1"),
            "receptors": list(pipeline.metadata["params"].receptor_names)}


def summarize(result: RunResult) -> dict:
    """Everything the page draws for a finished run, as JSON-ready values."""
    settings, pipeline = result.settings, result.pipeline
    params = pipeline.metadata["params"]
    frequency, original = radial_power_spectrum(result.original)
    _, rebuilt = radial_power_spectrum(result.reconstructed)
    counts, edges = np.histogram(np.asarray(spike_counts(result)).ravel(), bins=HISTOGRAM_BINS)
    error = result.reconstructed - result.original
    # One value per receptor, for the retina view: what each receptor caught.
    caught = stage_outputs(result)["mosaic"]
    low, high = float(caught.min()), float(caught.max())
    responses = (caught - low) / (high - low) if high > low else np.zeros_like(caught)
    return {
        "type": "result",
        "metrics": result.metrics,
        "settings": dataclasses.asdict(settings),
        "runtime_s": result.runtime_s,
        "iterations": result.reconstruction.iterations,
        "converged": result.reconstruction.converged,
        "residuals": list(result.reconstruction.residuals),
        "original": _png(result.original),
        "reconstructed": _png(result.reconstructed),
        "responses": _b64(responses, "<f4"),
        "channel_rmse": np.sqrt((error**2).mean(axis=(0, 1))).tolist(),
        "spectrum": {"cpd": (frequency[1:] / settings.fov_deg).tolist(),
                     "original": original[1:].tolist(), "reconstructed": rebuilt[1:].tolist(),
                     "limit_cpd": sampling_limit_cpd(result)},
        "spike_histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
        "color_matrix": [list(row) for row in params.color_matrix],
        "receptors": list(params.receptor_names),
        "parameters": tables.parameters_table(pipeline),
        "description": pipeline.description,
        "citations": list(pipeline.citations),
    }


@lru_cache(maxsize=None)
def _species_info(name: str) -> dict:
    pipeline = species.get(name)(VisualField(64, 60.0))
    params = pipeline.metadata["params"]
    return {"name": name, "description": pipeline.description,
            "receptors": list(params.receptor_names),
            "has_cortex": bool(params.cortex_sf_cpd), "citations": list(pipeline.citations),
            # The steps of a run, so the page can lay them out before one starts.
            "stages": [stage.name for stage in pipeline.stages] + ["decoding"]}


@app.get("/api/species")
def list_species() -> list[dict]:
    return [_species_info(name) for name in species.names()]


@app.get("/api/samples")
def list_samples() -> list[str]:
    return io.sample_names()


@app.get("/api/samples/{name}")
def get_sample(name: str) -> Response:
    if name not in io.sample_names():
        raise HTTPException(404, f"unknown sample '{name}'")
    data = (resources.files(io.SAMPLE_PACKAGE) / f"{name}.png").read_bytes()
    return Response(data, media_type="image/png")


def frame_event(progress) -> dict:
    """The decoder's current estimate and how far its solve has got."""
    return {"type": "frame", "iteration": progress.iteration, "image": _png(progress.image),
            "progress": progress.fraction}


def _stream_run(image: np.ndarray, settings: Settings):
    """Run in a worker thread and yield its events as NDJSON lines."""
    events: queue.Queue = queue.Queue()
    cancelled = threading.Event()
    done = object()

    def work():
        last_frame = [0.0]
        pending = [None]

        def on_progress(progress):
            if cancelled.is_set():
                raise _Cancelled
            if progress.iteration == 0:
                events.put({"type": "stage", "stages": list(progress.stages),
                            "current": progress.current, "image": _png(progress.image),
                            "plot": progress.plot})
                return
            pending[0] = progress
            now = time.monotonic()
            if now - last_frame[0] >= FRAME_INTERVAL_S:
                last_frame[0] = now
                pending[0] = None
                events.put(frame_event(progress))

        try:
            options = settings.run_options()
            # The picture at the size the eye will see it, before anything slow.
            events.put({"type": "original", "image": _png(io.to_square(image, settings.size_px))})
            pipeline = build_pipeline(settings.species, settings.size_px, settings.fov_deg,
                                      settings.density, settings.neuron_density)
            events.put(mosaic_event(pipeline))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = run(image, settings.species, on_progress=on_progress, **options)
            if pending[0] is not None:
                events.put(frame_event(pending[0]))
            events.put(summarize(result))
        except _Cancelled:
            pass
        except Exception as error:  # report it to the page instead of dropping the stream
            events.put({"type": "error", "message": str(error) or type(error).__name__})
        finally:
            events.put(done)

    threading.Thread(target=work, daemon=True).start()
    try:
        while (event := events.get()) is not done:
            yield _line(event)
    finally:
        cancelled.set()


@app.post("/api/runs")
async def post_run(settings: str = Form(...), sample: str | None = Form(None),
                   image: UploadFile | None = File(None)) -> StreamingResponse:
    parsed = _parse_settings(settings)
    picture = await _read_image(image, sample)
    return StreamingResponse(_stream_run(picture, parsed), media_type="application/x-ndjson")


@app.post("/api/compare")
async def post_compare(settings: str = Form(...), sample: str | None = Form(None),
                       image: UploadFile | None = File(None)) -> StreamingResponse:
    parsed = _parse_settings(settings)
    picture = await _read_image(image, sample)

    def stream():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for name in species.names():
                yield _line(summarize(run(picture, name, **parsed.run_options())))

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/sweeps")
async def post_sweep(settings: str = Form(...), kind: str = Form(...),
                     sample: str | None = Form(None),
                     image: UploadFile | None = File(None)) -> list[dict]:
    parsed = _parse_settings(settings)
    if kind not in SWEEPS:
        raise HTTPException(422, f"kind must be one of: {', '.join(SWEEPS)}")
    picture = await _read_image(image, sample)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return SWEEPS[kind](picture, parsed.species, **parsed.run_options())


@app.post("/api/report")
async def post_report(settings: str = Form(...), sample: str | None = Form(None),
                      image: UploadFile | None = File(None), all_species: bool = Form(False),
                      sweeps: bool = Form(False)) -> Response:
    parsed = _parse_settings(settings)
    picture = await _read_image(image, sample)
    names = None if all_species else [parsed.species]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        report = build_report(picture, names, sweeps=sweeps, **parsed.run_options())
    with tempfile.TemporaryDirectory() as folder:
        data = zip_report(write_report(report, folder))
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="biovision_report.zip"'})


@app.api_route("/api/{path:path}", methods=["GET", "POST"], include_in_schema=False)
def unknown_api(path: str):
    raise HTTPException(404, f"no such endpoint: /api/{path}")


if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def web_app(path: str) -> FileResponse:
        target = (WEB_DIST / path).resolve()
        inside = target.is_relative_to(WEB_DIST) and target.is_file()
        return FileResponse(target if path and inside else WEB_DIST / "index.html")
