"""Streamlit app: upload an image, pick a species, see what its eye keeps.

Run with: streamlit run app/streamlit_app.py
"""
import hashlib
import tempfile
import time
import warnings

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from biovision import io, species
from biovision.analysis import compare_species, sweep_density, sweep_lambda, sweep_window
from biovision.report import figures, tables
from biovision.report.export import Report, write_report, zip_report
from biovision.run import run

st.set_page_config(page_title="biovision", layout="wide")


LIVE_VIEW_PX = 384  # the live view is enlarged by whole pixels to about this width
WORKING_SIZES = [64, 96, 128, 192, 256, 384, 512]
LIGHT_SIZE_PX = 128  # above this, the human model needs gigabytes of memory
FRAME_INTERVAL_S = 0.05  # at most this often while decoding
STAGE_PAUSE_S = 0.15  # long enough to read each stage as it is ticked off
MAX_KEPT_RUNS = 12  # finished runs kept per browser session


def pixelated(frame: np.ndarray) -> np.ndarray:
    """Enlarge an image by whole pixels, so each one shows as a sharp block."""
    scale = max(1, LIVE_VIEW_PX // frame.shape[0])
    return np.kron(frame, np.ones((scale, scale, 1)))


def stage_line(progress) -> str:
    """The stages as one line: done ones ticked, the current one in bold."""
    parts = []
    for index, stage in enumerate(progress.stages):
        label = stage.replace("_", " ")
        if index < progress.current:
            parts.append(f"✓ {label}")
        elif index == progress.current:
            suffix = f" (iteration {progress.iteration})" if progress.iteration else ""
            parts.append(f"**{label}{suffix}**")
        else:
            parts.append(f"<span style='opacity:0.4'>{label}</span>")
    return " → ".join(parts)


def live_run(area, image: np.ndarray, name: str, settings: tuple):
    """Run with the reconstruction drawn in `area` as it forms. A repeated run returns at once.

    `area` is the st.empty() slot that later holds the results, so the live
    view replaces the previous results instead of appearing above them.

    Finished runs are kept in the session, not in st.cache_data: that cache
    records and replays screen updates made inside the function, which fails
    for updates to placeholders created outside it.
    """
    kept = st.session_state.setdefault("runs", {})
    key = (hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest(), name, settings)
    if key in kept:
        return kept[key]
    with area.container():
        left, right = st.columns(2)
        left.image(pixelated(io.to_square(image, dict(settings)["size_px"])),
                   caption="Original", width="stretch")
        frame_slot = right.empty()
        status_slot = st.empty()
    last_frame = [0.0]

    def show(progress):
        decoding = progress.iteration > 0
        now = time.monotonic()
        if decoding and now - last_frame[0] < FRAME_INTERVAL_S:
            return
        last_frame[0] = now
        if progress.image is not None:
            caption = "Reconstructing..." if decoding else progress.stages[progress.current]
            frame_slot.image(pixelated(progress.image), caption=caption, width="stretch")
        status_slot.markdown(stage_line(progress), unsafe_allow_html=True)
        if not decoding:
            time.sleep(STAGE_PAUSE_S)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run(image, name, on_progress=show, **dict(settings))
    if len(kept) >= MAX_KEPT_RUNS:
        kept.pop(next(iter(kept)))
    kept[key] = result
    return result


@st.cache_data(show_spinner="Running every species...")
def cached_compare(image: np.ndarray, settings: tuple):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return compare_species(image, **dict(settings))


@st.cache_data(show_spinner="Running sweeps (this can take a few minutes)...")
def cached_sweeps(image: np.ndarray, name: str, settings: tuple):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (sweep_window(image, name, **dict(settings)),
                sweep_lambda(image, name, **dict(settings)),
                sweep_density(image, name, **dict(settings)))


def load_input() -> np.ndarray | None:
    uploaded = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        return io.load_sample(st.sidebar.selectbox("Or pick a sample", io.sample_names()))
    try:
        return io.as_rgb(np.asarray(Image.open(uploaded).convert("RGB")))
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.sidebar.error(f"Could not read that image: {error}")
        return None


st.title("biovision")
st.caption("Encode an image through a species' visual system, then rebuild it "
           "from the neural code with pure mathematics.")

image = load_input()
name = st.sidebar.selectbox("Species", species.names())
window_ms = st.sidebar.select_slider("Spike window (ms)", [10, 30, 100, 300, 1000], value=100)
density = st.sidebar.select_slider(
    "Receptor density", [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], value=1.0,
    help="Receptors per unit area, relative to the real eye (1 is the real animal).")
neuron_density = st.sidebar.select_slider(
    "Neuron density", [0.25, 0.5, 1.0, 2.0, 4.0], value=1.0,
    help="Cortex cells per unit area, relative to the real animal. The fly has no "
         "cortex stage: its neurons follow the receptor density instead.")
noise = st.sidebar.toggle("Spike noise", value=True)
fov_deg = st.sidebar.slider("Field of view (degrees)", 10, 120, 60, step=10)
size_px = st.sidebar.select_slider(
    "Working size (pixels)", WORKING_SIZES, value=128,
    help="The image is resized to this many pixels across. Time and memory grow with "
         "its square; the human model is the heaviest.")
auto_lam = st.sidebar.toggle("Set regularization from the noise", value=True)
lam = None if auto_lam else 10.0 ** st.sidebar.slider("log10(lambda)", -5.0, 1.0, -3.0, 0.5)
seed = st.sidebar.number_input("Noise seed", min_value=0, value=0, step=1)

if image is None:
    st.stop()

if size_px > LIGHT_SIZE_PX and not st.sidebar.checkbox(
        f"Run above {LIGHT_SIZE_PX} pixels", value=False,
        help="Fly and mouse take up to a minute at 512 pixels. The human model needs "
             "about 2 GB of memory at 192 pixels and 5 GB at 256, and takes minutes."):
    st.info(f"Working sizes above {LIGHT_SIZE_PX} pixels are slow, and for the human model "
            f"need several gigabytes of memory. Tick \"Run above {LIGHT_SIZE_PX} pixels\" "
            "in the sidebar to go ahead.")
    st.stop()

settings = tuple(sorted(dict(size_px=size_px, fov_deg=float(fov_deg),
                             window_ms=float(window_ms), noise=noise, lam=lam,
                             seed=int(seed), density=float(density),
                             neuron_density=float(neuron_density)).items(),
                        key=lambda item: item[0]))
# One slot holds the live view during a run and the results after it.
main = st.empty()
result = live_run(main, image, name, settings)
results = main.container()
if not result.reconstruction.converged:
    results.warning("The solver stopped before fully converging; this is its best estimate.")

explore, stages, analysis, compare = results.tabs(["Explore", "Stages", "Analysis", "Compare"])

with explore:
    left, right = st.columns(2)
    left.image(result.original, caption=f"Original at {size_px} pixels, as the model sees it",
               width="stretch")
    right.image(result.reconstructed, caption=f"What the {name} code keeps",
                width="stretch")
    first, second, third, fourth = st.columns(4)
    first.metric("PSNR", f"{result.metrics['psnr_db']:.1f} dB")
    second.metric("SSIM", f"{result.metrics['ssim']:.2f}")
    third.metric("Neurons", f"{int(result.metrics['neurons']):,}",
                 help="Output cells whose spikes are decoded.")
    fourth.metric("Receptors", f"{int(result.metrics['receptors']):,}",
                  help="Photoreceptor positions sampling the image.")
    st.write(result.pipeline.description)

with stages:
    st.pyplot(figures.pipeline_panel(result))
    left, right = st.columns(2)
    left.pyplot(figures.mosaic_map(result))
    right.pyplot(figures.color_model(result))
    st.pyplot(figures.filter_gallery(result))

with analysis:
    left, right = st.columns(2)
    left.pyplot(figures.error_map(result))
    right.pyplot(figures.spectrum(result))
    left, right = st.columns(2)
    left.pyplot(figures.neural_code(result))
    right.pyplot(figures.convergence({name: result}))
    st.subheader("Results")
    st.dataframe(tables.results_table({name: result}), hide_index=True)
    st.subheader("Settings")
    st.dataframe(tables.settings_table(result), hide_index=True)
    st.subheader("Species parameters")
    st.dataframe(tables.parameters_table(result.pipeline), hide_index=True)
    st.subheader("References")
    for citation in result.pipeline.citations:
        st.markdown(f"- {citation}")
    st.subheader("Sweeps")
    if st.button("Run window, lambda and receptor-density sweeps"):
        window_rows, lambda_rows, density_rows = cached_sweeps(image, name, settings)
        st.pyplot(figures.density_sweep({name: density_rows}))
        st.dataframe(density_rows, hide_index=True)
        st.pyplot(figures.window_sweep({name: window_rows}))
        st.dataframe(window_rows, hide_index=True)
        st.pyplot(figures.lambda_sweep({name: lambda_rows}))
        st.dataframe(lambda_rows, hide_index=True)

with compare:
    if st.button("Compare every species"):
        results = cached_compare(image, settings)
        st.pyplot(figures.species_grid(results))
        st.dataframe(tables.results_table(results), hide_index=True)
        with tempfile.TemporaryDirectory() as folder:
            write_report(Report(results), folder)
            st.download_button("Download report (zip)", zip_report(folder),
                               file_name="biovision_report.zip", mime="application/zip")
