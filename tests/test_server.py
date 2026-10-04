import base64
import io as std_io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from biovision.server import app

client = TestClient(app)
FLY = {"species": "fly", "size_px": 32}


def form(settings, **extra):
    return {"settings": json.dumps(settings), **extra}


def events(response):
    return [json.loads(line) for line in response.text.splitlines() if line]


def decode_png(text):
    return np.asarray(Image.open(std_io.BytesIO(base64.b64decode(text))))


def test_species_lists_every_registered_species():
    body = client.get("/api/species").json()
    assert [item["name"] for item in body] == ["fly", "human", "mouse"]
    fly, human, _ = body
    assert fly["receptors"] == ["UV", "blue", "green"] and fly["has_cortex"] is False
    assert human["receptors"] == ["L", "M", "S"] and human["has_cortex"] is True
    assert len(fly["description"]) > 40 and len(fly["citations"]) >= 3


def test_samples_are_listed_and_served_as_png():
    assert client.get("/api/samples").json() == ["astronaut", "cat", "coffee"]
    response = client.get("/api/samples/cat")
    assert response.headers["content-type"] == "image/png"
    assert Image.open(std_io.BytesIO(response.content)).size == (256, 256)
    assert client.get("/api/samples/dog").status_code == 404


def test_run_streams_mosaic_stages_frames_then_result():
    response = client.post("/api/runs", data=form(FLY, sample="astronaut"))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    stream = events(response)
    kinds = [event["type"] for event in stream]
    assert kinds[0] == "mosaic" and kinds[-1] == "result"
    assert "error" not in kinds
    stage_events = [event for event in stream if event["type"] == "stage"]
    assert stage_events[0]["stages"] == ["color", "optics", "mosaic", "center_surround",
                                         "rate", "spikes", "decoding"]
    assert [event["current"] for event in stage_events] == list(range(6))
    assert decode_png(stage_events[0]["image"]).shape == (32, 32, 3)
    assert stage_events[2]["image"] is None
    frames = [event for event in stream if event["type"] == "frame"]
    assert frames and frames == sorted(frames, key=lambda event: event["iteration"])
    assert kinds.index("frame") > kinds.index("stage")


def test_run_result_carries_everything_the_page_draws():
    result = events(client.post("/api/runs", data=form(FLY, sample="astronaut")))[-1]
    assert set(result["metrics"]) == {"psnr_db", "ssim", "neurons", "receptors",
                                      "compression_ratio", "mean_spikes"}
    assert result["settings"]["species"] == "fly" and result["settings"]["size_px"] == 32
    assert decode_png(result["original"]).shape == (32, 32, 3)
    assert decode_png(result["reconstructed"]).shape == (32, 32, 3)
    assert result["iterations"] == len(result["residuals"]) > 0
    assert len(result["channel_rmse"]) == 3
    assert len(result["spectrum"]["cpd"]) == len(result["spectrum"]["original"]) == 15
    assert result["spectrum"]["limit_cpd"] == pytest.approx(0.1)
    assert sum(result["spike_histogram"]["counts"]) == result["metrics"]["neurons"]
    assert len(result["spike_histogram"]["edges"]) == len(result["spike_histogram"]["counts"]) + 1
    assert {row["parameter"] for row in result["parameters"]} >= {"spacing_deg", "rest_hz"}
    assert len(result["description"]) > 40 and len(result["citations"]) >= 3


def test_mosaic_event_gives_positions_types_and_responses_arrive_with_the_result():
    stream = events(client.post("/api/runs", data=form(FLY, sample="astronaut")))
    mosaic = stream[0]
    count = mosaic["count"]
    positions = np.frombuffer(base64.b64decode(mosaic["positions"]), dtype="<f4")
    types = np.frombuffer(base64.b64decode(mosaic["types"]), dtype=np.uint8)
    assert positions.shape == (2 * count,) and types.shape == (count,)
    assert positions.min() >= 0.0 and positions.max() <= 1.0
    assert mosaic["receptors"] == ["UV", "blue", "green"] and set(types) == {0, 1, 2}
    responses = np.frombuffer(base64.b64decode(stream[-1]["responses"]), dtype="<f4")
    assert responses.shape == (count,)
    assert responses.min() >= 0.0 and responses.max() <= 1.0


def test_run_accepts_an_uploaded_image():
    buffer = std_io.BytesIO()
    pixels = (np.random.default_rng(0).random((40, 60, 3)) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(buffer, format="JPEG")
    response = client.post("/api/runs", data=form(FLY),
                           files={"image": ("photo.jpg", buffer.getvalue(), "image/jpeg")})
    assert events(response)[-1]["type"] == "result"


def test_run_is_deterministic_for_a_seed():
    first = events(client.post("/api/runs", data=form({**FLY, "seed": 7}, sample="cat")))[-1]
    again = events(client.post("/api/runs", data=form({**FLY, "seed": 7}, sample="cat")))[-1]
    assert first["reconstructed"] == again["reconstructed"]


@pytest.mark.parametrize("settings, status, message", [
    ({"species": "cat", "size_px": 32}, 400, "unknown species 'cat'"),
    ({"species": "fly", "size_px": 32, "window_ms": 0}, 422, "window_ms"),
    ({"species": "fly", "size_px": 1}, 422, "size_px"),
    ({"species": "fly", "size_px": 4096}, 422, "size_px"),
    ({"species": "fly", "size_px": 32, "density": -1}, 422, "density"),
    ({"size_px": 32}, 422, "species"),
])
def test_run_rejects_bad_settings_with_a_readable_message(settings, status, message):
    response = client.post("/api/runs", data=form(settings, sample="astronaut"))
    assert response.status_code == status
    assert message in json.dumps(response.json())


def test_run_rejects_bad_input_images():
    response = client.post("/api/runs", data=form(FLY),
                           files={"image": ("notes.png", b"not an image", "image/png")})
    assert response.status_code == 400 and "Could not read that image" in response.json()["detail"]
    assert client.post("/api/runs", data=form(FLY)).status_code == 400
    assert client.post("/api/runs", data=form(FLY, sample="dog")).status_code == 400
    assert client.post("/api/runs", data={"settings": "{not json", "sample": "cat"}).status_code == 422


def test_compare_streams_one_result_per_species():
    stream = events(client.post("/api/compare", data=form({"species": "fly", "size_px": 32},
                                                           sample="astronaut")))
    assert [event["settings"]["species"] for event in stream] == ["fly", "human", "mouse"]
    assert all(decode_png(event["reconstructed"]).shape == (32, 32, 3) for event in stream)


@pytest.mark.parametrize("kind, column", [("density", "density"), ("lambda", "lam"),
                                          ("window", "window_ms")])
def test_sweeps_return_rows(kind, column):
    response = client.post("/api/sweeps", data=form(FLY, sample="astronaut", kind=kind))
    rows = response.json()
    assert len(rows) >= 4 and all(column in row for row in rows)
    assert client.post("/api/sweeps", data=form(FLY, sample="astronaut", kind="nope")).status_code == 422


def test_report_returns_a_zip_with_the_report():
    response = client.post("/api/report", data=form(FLY, sample="astronaut"))
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(std_io.BytesIO(response.content)) as archive:
        names = archive.namelist()
    assert "report.md" in names and "fly_pipeline.png" in names and "results.json" in names


def test_the_api_reports_unknown_paths_as_json():
    assert client.get("/api/nope").status_code == 404
