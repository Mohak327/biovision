"""Streamlit app: upload an image, pick a species, see what its eye keeps.

Run with: streamlit run app/streamlit_app.py
"""
import tempfile
import warnings

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from biovision import io, species
from biovision.analysis import compare_species, sweep_lambda, sweep_window
from biovision.report import figures, tables
from biovision.report.export import Report, write_report, zip_report
from biovision.run import run

st.set_page_config(page_title="biovision", layout="wide")


@st.cache_data(show_spinner="Encoding and reconstructing...")
def cached_run(image: np.ndarray, name: str, settings: tuple):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return run(image, name, **dict(settings))


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
                sweep_lambda(image, name, **dict(settings)))


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
noise = st.sidebar.toggle("Spike noise", value=True)
fov_deg = st.sidebar.slider("Field of view (degrees)", 10, 120, 60, step=10)
size_px = st.sidebar.select_slider("Working size (pixels)", [64, 96, 128], value=96)
auto_lam = st.sidebar.toggle("Set regularization from the noise", value=True)
lam = None if auto_lam else 10.0 ** st.sidebar.slider("log10(lambda)", -5.0, 1.0, -3.0, 0.5)
seed = st.sidebar.number_input("Noise seed", min_value=0, value=0, step=1)

if image is None:
    st.stop()

settings = tuple(sorted(dict(size_px=size_px, fov_deg=float(fov_deg),
                             window_ms=float(window_ms), noise=noise, lam=lam,
                             seed=int(seed)).items(), key=lambda item: item[0]))
result = cached_run(image, name, settings)
if not result.reconstruction.converged:
    st.warning("The solver stopped before fully converging; this is its best estimate.")

explore, stages, analysis, compare = st.tabs(["Explore", "Stages", "Analysis", "Compare"])

with explore:
    left, right = st.columns(2)
    left.image(result.original, caption="Original", width="stretch")
    right.image(result.reconstructed, caption=f"What the {name} code keeps",
                width="stretch")
    first, second, third = st.columns(3)
    first.metric("PSNR", f"{result.metrics['psnr_db']:.1f} dB")
    second.metric("SSIM", f"{result.metrics['ssim']:.2f}")
    third.metric("Neurons", f"{int(result.metrics['neurons']):,}")
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
    if st.button("Run window and lambda sweeps"):
        window_rows, lambda_rows = cached_sweeps(image, name, settings)
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
