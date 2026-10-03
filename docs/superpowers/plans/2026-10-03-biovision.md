# biovision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python package that encodes an image through a model of a species' early visual system (human, mouse, fruit fly) and reconstructs it from the neural code with a regularized linear inverse, with a CLI, a Streamlit app and report-grade figures and tables.

**Architecture:** A species is a `Pipeline` of stages: linear stages (each with an exact `adjoint`) followed by pointwise stages (each with an exact `inverse`). Species are selected by name from a `Registry`. One `Decoder` undoes the pointwise stages and solves a regularized least-squares problem over the composed linear operator by conjugate gradients. `run()` is the single entry point; the CLI, the app and the report layer only call it.

**Tech Stack:** Python 3.10+, NumPy, SciPy (sparse matrices, KD-tree), Matplotlib, Pillow, scikit-image (SSIM and sample images), Streamlit (app, optional), pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-biovision-design.md`

**Provenance of the code in this plan:** every source and test file below was run as a prototype before this plan was written. The full suite (134 tests) passed on Python 3.11, NumPy 2.4, SciPy 1.17, scikit-image 0.26, Streamlit 1.65. Type the code as written; if a test fails, look for a typo before changing the design. Measured results at 128 px, 60 degrees, astronaut sample: noise-free PSNR is about 35 dB (human), 15 dB (mouse), 13 dB (fly); with spike noise at 100 ms it is about 18, 12 and 13 dB.

## Global Constraints

- Python `>=3.10`. Runtime dependencies are exactly: `numpy>=1.26`, `scipy>=1.11`, `matplotlib>=3.8`, `pillow>=10`, `scikit-image>=0.22`. Streamlit (`>=1.36`) is an optional extra, used only by `app/`.
- No machine learning and no fitted parameters. Every stage is a fixed mathematical model; decoding is a regularized linear inverse.
- `src/biovision/stages/` holds mathematics and never imports from `species/`. `src/biovision/species/` holds parameters, citations and assembly and contains no mathematics. `core/` imports from neither.
- A pipeline is all linear stages followed by all pointwise stages. `Pipeline` rejects anything else when it is constructed.
- Every linear stage has an `adjoint` that is the exact transpose of `forward`. The parametrized adjoint tests are the guard; never weaken their tolerance.
- All randomness goes through an explicit `numpy.random.Generator`. No global seeds, no `np.random.*` module functions.
- Internal image layout is `(channels, size, size)` for pipelines and the decoder, and `(size, size, 3)` in `[0, 1]` everywhere a user or a figure sees an image (`io`, `run`, `report`).
- Stage parameters are in pixels. Species parameters are in degrees of visual angle and are converted with `VisualField`.
- Figures are built with `matplotlib.figure.Figure` directly (never `pyplot`), so they are safe inside Streamlit.
- The CLI, the app and the report layer call `run()` / `analysis` and contain no mathematics.
- Commit after every task. Do not push unless asked.
- Defaults that were tuned on the prototype and must not be changed without re-measuring: `LAM_FLOOR = 1e-4`, `NOISE_GAIN = 10.0`, `CHROMA_WEIGHT = 0.1`, `contrast_gain = 2.5`, `rest_hz = 100.0` (fly: `2500.0`), decoder `max_iter = 500`, `tol = 1e-4`.

## Review Focus

These are inputs a real user will supply that the spec does not spell out. Each has a test in the task named.

1. An image that is not a square RGB photograph: grayscale, with an alpha channel, 16-bit, non-square, or only 2 x 2 pixels. Expected: it is converted, centre-cropped and resized without error. (Task 7, `tests/test_io.py`)
2. A file that is missing or is not an image, given to the CLI or uploaded to the app. Expected: a one-line readable error and exit code 2, never a traceback. (Task 7 `tests/test_io.py`, Task 11 `tests/test_cli.py`; the app shows a sidebar error, Task 12)
3. A uniform or all-black image, where there is no signal and PSNR can be infinite. Expected: a finite, uniform reconstruction and finite metrics. (Task 9, `tests/test_run.py`)
4. Extreme settings: a field of view of 5 or 170 degrees, or a very small working size, where a lattice has only a few receptors or every cortical wavelength falls below two pixels. Expected: the run completes; the cortex stage is dropped when no wavelength fits. (Task 8 `tests/test_species.py`, Task 9 `tests/test_run.py`)
5. A spike window so short that every neuron fires zero spikes. Expected: the decoder returns a finite image. (Task 6, `tests/test_decoder.py`)

## File Structure

```
pyproject.toml                 package metadata, dependencies, pytest config
README.md                      user documentation
AGENTS.md, CLAUDE.md           project description for coding agents (already present)
scripts/make_samples.py        writes the bundled sample images
src/biovision/
  __init__.py                  version; importing it registers the species
  io.py                        load, validate, crop and resize images; bundled samples
  run.py                       run(): encode + decode + metrics; RunResult, Settings
  analysis.py                  compare_species, sweep_window, sweep_lambda
  cli.py                       `biovision list | run | report`
  core/
    field.py                   VisualField: pixels <-> degrees
    stage.py                   Stage, LinearStage, PointwiseStage
    registry.py                Registry[T] and the `species` registry
    pipeline.py                Pipeline, NeuralCode
    regularizers.py            laplacian, chroma
    decoder.py                 conjugate_gradient, Decoder, Reconstruction
    metrics.py                 psnr, ssim, radial_power_spectrum
  stages/
    color.py                   ColorProjection
    optics.py                  OpticalBlur
    nonlinearity.py            LinearRectified
    spiking.py                 PoissonSpikes
    sparse.py                  SparseStage, pool, normalize_rows
    mosaic.py                  Mosaic, lattices, mosaic_sampling
    receptive.py               gaussian, center_surround
    gabor.py                   gabor_kernel, gabor_bank
  species/
    __init__.py                imports every species module
    eye.py                     EyeParams, build_mosaic, assemble
    human.py, mouse.py, fly.py parameters, description, citations, registration
  report/
    style.py                   palette, new_figure, show_image
    figures.py                 one function per figure
    tables.py                  one function per table
    export.py                  Report, build_report, write_report, zip_report
  samples/                     astronaut.png, cat.png, coffee.png
app/streamlit_app.py           the upload app
tests/                         one test file per module group
```

Run every command from the repository root. On Windows use `.venv\Scripts\python`; elsewhere `.venv/bin/python`. The plan writes `python` for whichever applies once the virtual environment is active.

---

### Task 1: Project scaffold and VisualField

**Files:**
- Create: `pyproject.toml`
- Create: `src/biovision/__init__.py`, `src/biovision/core/__init__.py`, `src/biovision/stages/__init__.py`, `src/biovision/report/__init__.py`
- Create: `src/biovision/core/field.py`
- Create: `tests/conftest.py`
- Test: `tests/test_field.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `VisualField(size_px: int, fov_deg: float)` with `.deg_per_px -> float`, `.to_px(deg) -> float`, `.to_deg(px) -> float`; raises `ValueError` if `size_px <= 1` or `fov_deg <= 0`. Test fixtures `rng` (a `numpy.random.Generator` seeded with 0) and `field` (`VisualField(32, 60.0)`).

- [ ] **Step 1: Create the package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "biovision"
version = "0.1.0"
description = "Modular biological vision systems with mathematical image reconstruction"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "matplotlib>=3.8",
    "pillow>=10",
    "scikit-image>=0.22",
]

[project.optional-dependencies]
app = ["streamlit>=1.36"]
dev = ["pytest>=8", "streamlit>=1.36"]

[project.scripts]
biovision = "biovision.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"biovision.samples" = ["*.png"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create the package skeleton**

Create `src/biovision/__init__.py` (Task 8 replaces this file):

```python
"""Biological vision systems with mathematical image reconstruction."""

__version__ = "0.1.0"
```

Create three empty files: `src/biovision/core/__init__.py`, `src/biovision/stages/__init__.py`, `src/biovision/report/__init__.py`.

Create `README.md` containing the single line `# biovision` (Task 13 writes the real one; the build needs the file to exist).

Create `tests/conftest.py` (Task 7 adds the `sample` fixture):

```python
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from biovision.core.field import VisualField

SIZE = 32


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def field():
    return VisualField(SIZE, 60.0)
```

- [ ] **Step 3: Create the environment and install**

Run:

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

Expected: ends with `Successfully installed ... biovision-0.1.0 ...`.

- [ ] **Step 4: Write the failing test**

Create `tests/test_field.py`:

```python
import pytest

from biovision.core.field import VisualField


def test_conversions_round_trip():
    field = VisualField(size_px=120, fov_deg=60.0)
    assert field.deg_per_px == 0.5
    assert field.to_px(5.0) == 10.0
    assert field.to_deg(field.to_px(3.3)) == pytest.approx(3.3)


@pytest.mark.parametrize("size, fov", [(0, 60.0), (1, 60.0), (64, 0.0), (64, -5.0)])
def test_rejects_invalid_values(size, fov):
    with pytest.raises(ValueError):
        VisualField(size, fov)
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `python -m pytest tests/test_field.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'biovision.core.field'`.

- [ ] **Step 6: Implement VisualField**

Create `src/biovision/core/field.py`:

```python
"""Conversion between image pixels and degrees of visual angle."""
from dataclasses import dataclass


@dataclass(frozen=True)
class VisualField:
    """A square image of `size_px` pixels spanning `fov_deg` degrees."""

    size_px: int
    fov_deg: float

    def __post_init__(self):
        if self.size_px <= 1:
            raise ValueError(f"size_px must be greater than 1, got {self.size_px}")
        if self.fov_deg <= 0:
            raise ValueError(f"fov_deg must be positive, got {self.fov_deg}")

    @property
    def deg_per_px(self) -> float:
        return self.fov_deg / self.size_px

    def to_px(self, deg: float) -> float:
        return deg / self.deg_per_px

    def to_deg(self, px: float) -> float:
        return px * self.deg_per_px
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `python -m pytest tests/test_field.py -q`
Expected: `5 passed`.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src tests README.md
git commit -m "feat: project scaffold and VisualField"
```

---

### Task 2: Registry

**Files:**
- Create: `src/biovision/core/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Registry(kind: str)` with `.register(name: str)` (a decorator that returns the object unchanged), `.get(name: str)`, `.names() -> list[str]` (sorted). A duplicate name raises `ValueError("<kind> '<name>' is already registered")`. An unknown name raises `KeyError("unknown <kind> '<name>'; registered: a, b")`. A module-level instance `species = Registry("species")` holds functions `build(field: VisualField) -> Pipeline`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry.py`:

```python
import pytest

from biovision.core.registry import Registry


def test_register_get_and_names():
    registry = Registry("thing")

    @registry.register("b")
    def make_b():
        return "B"

    registry.register("a")(lambda: "A")
    assert registry.get("b") is make_b
    assert registry.names() == ["a", "b"]


def test_duplicate_name_is_rejected():
    registry = Registry("thing")
    registry.register("a")(object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("a")(object())


def test_unknown_name_lists_known_names():
    registry = Registry("species")
    registry.register("mouse")(object())
    with pytest.raises(KeyError, match="unknown species 'cat'; registered: mouse"):
        registry.get("cat")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_registry.py -q`
Expected: `ModuleNotFoundError: No module named 'biovision.core.registry'`.

- [ ] **Step 3: Implement the registry**

Create `src/biovision/core/registry.py`:

```python
"""A name-to-object registry, used to select species by name."""
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator that stores the decorated object under `name`."""

        def decorator(item: T) -> T:
            if name in self._items:
                raise ValueError(f"{self._kind} '{name}' is already registered")
            self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(self.names()) or "none"
            raise KeyError(f"unknown {self._kind} '{name}'; registered: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._items)


species: Registry = Registry("species")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_registry.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/biovision/core/registry.py tests/test_registry.py
git commit -m "feat: generic registry and the species registry"
```

---

### Task 3: Stage interfaces and the four simple stages

**Files:**
- Create: `src/biovision/core/stage.py`
- Create: `src/biovision/stages/color.py`, `src/biovision/stages/optics.py`, `src/biovision/stages/nonlinearity.py`, `src/biovision/stages/spiking.py`
- Test: `tests/test_stages_simple.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `LinearStage` (abstract): attributes `name: str`, `in_shape: tuple`, `out_shape: tuple`; methods `forward(x) -> ndarray`, `adjoint(y) -> ndarray`.
  - `PointwiseStage` (abstract): attribute `name: str`; methods `forward(x, rng=None) -> ndarray`, `inverse(y) -> ndarray`.
  - `ColorProjection(matrix, size_px, name="color")`: `(3, size, size) -> (types, size, size)`.
  - `OpticalBlur(sigma_px, channels, size_px, name="optics")`: shape-preserving, self-adjoint, periodic edges.
  - `LinearRectified(rest_hz, gain, name="rate")`: `rate = max(rest_hz * (1 + gain * x), 0)`; `inverse(y) = (y / rest_hz - 1) / gain`.
  - `PoissonSpikes(window_s, name="spikes")`: with `rng=None` returns `rate * window_s` exactly; otherwise Poisson counts as floats. Attribute `.window_s`.

**Why threshold-linear and not a saturating curve:** the prototype tried a saturating (Naka-Rushton) rate. Inverting a curved function on noisy spike counts is biased, and its tangent approximation distorted even noise-free reconstructions by 14 dB. A threshold-linear rate has an exact inverse that stays unbiased under Poisson noise. Do not reintroduce a curved nonlinearity.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stages_simple.py`:

```python
import numpy as np
import pytest

from biovision.stages.color import ColorProjection
from biovision.stages.nonlinearity import LinearRectified
from biovision.stages.optics import OpticalBlur
from biovision.stages.spiking import PoissonSpikes

SIZE = 16


def linear_stages():
    return [
        ColorProjection([[0.2, 0.7, 0.1], [0.0, 0.1, 0.9]], SIZE),
        OpticalBlur(1.5, 2, SIZE),
    ]


@pytest.mark.parametrize("stage", linear_stages(), ids=lambda s: s.name)
def test_adjoint_is_the_exact_transpose(stage, rng):
    x = rng.standard_normal(stage.in_shape)
    y = rng.standard_normal(stage.out_shape)
    assert np.vdot(stage.forward(x), y) == pytest.approx(np.vdot(x, stage.adjoint(y)), rel=1e-10)


@pytest.mark.parametrize("stage", linear_stages(), ids=lambda s: s.name)
def test_forward_output_has_the_declared_shape(stage, rng):
    assert stage.forward(rng.standard_normal(stage.in_shape)).shape == stage.out_shape


def test_color_projection_rejects_a_bad_matrix():
    with pytest.raises(ValueError, match=r"\(types, 3\)"):
        ColorProjection([[1.0, 0.0]], SIZE)


def test_blur_keeps_the_mean_and_zero_sigma_is_identity(rng):
    x = rng.random((2, SIZE, SIZE))
    assert OpticalBlur(2.0, 2, SIZE).forward(x).mean() == pytest.approx(x.mean())
    np.testing.assert_allclose(OpticalBlur(0.0, 2, SIZE).forward(x), x, atol=1e-12)


def test_linear_rectified_round_trip_and_rest_rate(rng):
    stage = LinearRectified(rest_hz=100.0, gain=2.5)
    x = rng.uniform(-0.39, 0.39, 100)
    np.testing.assert_allclose(stage.inverse(stage.forward(x)), x, atol=1e-12)
    assert stage.forward(np.array([0.0]))[0] == 100.0
    assert stage.forward(np.array([-1.0]))[0] == 0.0  # rectified, never negative


def test_poisson_spikes_mean_exact_without_rng_and_noisy_with(rng):
    stage = PoissonSpikes(window_s=0.1)
    rates = np.full(20000, 50.0)
    np.testing.assert_allclose(stage.forward(rates), 5.0)
    np.testing.assert_allclose(stage.inverse(stage.forward(rates)), rates)
    counts = stage.forward(rates, rng)
    assert np.all(counts == np.round(counts))
    assert counts.mean() == pytest.approx(5.0, abs=0.1)
    assert counts.var() == pytest.approx(5.0, abs=0.3)


@pytest.mark.parametrize("make", [
    lambda: LinearRectified(0.0, 1.0), lambda: LinearRectified(100.0, 0.0),
    lambda: PoissonSpikes(0.0), lambda: OpticalBlur(-1.0, 1, SIZE),
])
def test_stages_reject_invalid_parameters(make):
    with pytest.raises(ValueError):
        make()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_stages_simple.py -q`
Expected: `ModuleNotFoundError: No module named 'biovision.stages.color'`.

- [ ] **Step 3: Implement the stage interfaces**

Create `src/biovision/core/stage.py`:

```python
"""Stage interfaces. A pipeline is linear stages followed by pointwise stages."""
from abc import ABC, abstractmethod

import numpy as np


class Stage(ABC):
    """One step of a visual pipeline."""

    name: str


class LinearStage(Stage):
    """A linear map with an exact transpose."""

    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def adjoint(self, y: np.ndarray) -> np.ndarray: ...


class PointwiseStage(Stage):
    """An elementwise function with an elementwise inverse."""

    @abstractmethod
    def forward(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray: ...

    @abstractmethod
    def inverse(self, y: np.ndarray) -> np.ndarray: ...
```

- [ ] **Step 4: Implement the four stages**

Create `src/biovision/stages/color.py`:

```python
"""Projection of RGB onto a species' photoreceptor types."""
import numpy as np

from ..core.stage import LinearStage


class ColorProjection(LinearStage):
    """Applies a (types x 3) matrix to the channel axis of an RGB image."""

    def __init__(self, matrix, size_px: int, name: str = "color"):
        self.matrix = np.asarray(matrix, dtype=float)
        if self.matrix.ndim != 2 or self.matrix.shape[1] != 3:
            raise ValueError(f"matrix must be (types, 3), got {self.matrix.shape}")
        self.name = name
        self.in_shape = (3, size_px, size_px)
        self.out_shape = (self.matrix.shape[0], size_px, size_px)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tensordot(self.matrix, x, axes=1)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return np.tensordot(self.matrix.T, y, axes=1)
```

Create `src/biovision/stages/optics.py`:

```python
"""Blur by the eye's optics, modelled as a Gaussian point-spread function."""
import numpy as np

from ..core.stage import LinearStage


class OpticalBlur(LinearStage):
    """Gaussian blur of each channel, applied in the frequency domain.

    The transfer function is real and even, so the stage is its own adjoint.
    Edges wrap around (periodic), which keeps the adjoint exact.
    """

    def __init__(self, sigma_px: float, channels: int, size_px: int, name: str = "optics"):
        if sigma_px < 0:
            raise ValueError(f"sigma_px must not be negative, got {sigma_px}")
        self.name = name
        self.sigma_px = sigma_px
        self.in_shape = self.out_shape = (channels, size_px, size_px)
        f = np.fft.fftfreq(size_px)
        f2 = f[:, None] ** 2 + f[None, :] ** 2
        self._transfer = np.exp(-2.0 * np.pi**2 * sigma_px**2 * f2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(np.fft.fft2(x) * self._transfer).real

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return self.forward(y)
```

Create `src/biovision/stages/nonlinearity.py`:

```python
"""Pointwise map from a signed linear response to a firing rate."""
import numpy as np

from ..core.stage import PointwiseStage


class LinearRectified(PointwiseStage):
    """rate = max(rest_hz * (1 + gain * x), 0), the output stage of an LNP neuron.

    `rest_hz` is the firing rate with no signal. It stands for an ON/OFF pair
    of cells: responses above rest are the ON cell, below rest the OFF cell.
    `gain` scales the response so natural images span the firing range.
    The inverse is exact wherever the rate is above zero.
    """

    def __init__(self, rest_hz: float, gain: float, name: str = "rate"):
        if rest_hz <= 0 or gain <= 0:
            raise ValueError("rest_hz and gain must be positive")
        self.name = name
        self.rest_hz = rest_hz
        self.gain = gain

    def forward(self, x, rng=None):
        return np.maximum(self.rest_hz * (1.0 + self.gain * x), 0.0)

    def inverse(self, y):
        return (np.asarray(y, dtype=float) / self.rest_hz - 1.0) / self.gain
```

Create `src/biovision/stages/spiking.py`:

```python
"""Spike counts in a time window."""
import numpy as np

from ..core.stage import PointwiseStage


class PoissonSpikes(PointwiseStage):
    """counts ~ Poisson(rate * window). Without an rng, returns the mean exactly."""

    def __init__(self, window_s: float, name: str = "spikes"):
        if window_s <= 0:
            raise ValueError(f"window_s must be positive, got {window_s}")
        self.name = name
        self.window_s = window_s

    def forward(self, x, rng=None):
        mean = np.maximum(x, 0.0) * self.window_s
        if rng is None:
            return mean
        return rng.poisson(mean).astype(float)

    def inverse(self, y):
        return np.asarray(y, dtype=float) / self.window_s
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_stages_simple.py -q`
Expected: `12 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/biovision/core/stage.py src/biovision/stages tests/test_stages_simple.py
git commit -m "feat: stage interfaces and colour, optics, rate and spike stages"
```

---

### Task 4: Sparse stages: mosaic, centre-surround, Gabor bank

**Files:**
- Create: `src/biovision/stages/sparse.py`, `src/biovision/stages/mosaic.py`, `src/biovision/stages/receptive.py`, `src/biovision/stages/gabor.py`
- Test: `tests/test_stages_sparse.py`

**Interfaces:**
- Consumes: `LinearStage` from Task 3.
- Produces:
  - `SparseStage(name, matrix, in_shape, out_shape)`: a `LinearStage` whose forward map is a SciPy CSR matrix on flattened arrays; raises `ValueError` if the matrix shape does not match.
  - `pool(out_pos, out_types, in_pos, in_types, radius, kernel) -> csr_matrix` of shape `(len(out_pos), len(in_pos))`: `kernel(dy, dx)` for same-type pairs within `radius`. Positions are `(n, 2)` arrays of `(row, col)` pixels.
  - `normalize_rows(matrix) -> csr_matrix`: each row scaled to absolute sum 1; empty rows stay empty.
  - `Mosaic(positions, types, n_types)` (frozen dataclass, supports `len`).
  - `square_lattice(size_px, spacing_px)`, `hex_lattice(size_px, spacing_px)`, `foveated_lattice(size_px, center_spacing_px, e2_px, min_spacing_px=1.0)`: each returns `(n, 2)` positions inside `[0, size_px - 1]`.
  - `assign_types(n, fractions, rng) -> ndarray`, `all_types_at(positions, n_types) -> Mosaic`.
  - `mosaic_sampling(mosaic, size_px, name="mosaic") -> SparseStage`: `(n_types, size, size) -> (len(mosaic),)`.
  - `gaussian(dy, dx, sigma) -> ndarray`.
  - `center_surround(mosaic, sigma_center_px, sigma_surround_px, surround_weight, name="center_surround") -> SparseStage`: `(len(mosaic),) -> (len(mosaic),)`; a uniform input of 1 gives `1 - surround_weight`.
  - `gabor_kernel(dy, dx, sigma, wavelength, theta, phase) -> ndarray`.
  - `gabor_bank(mosaic, size_px, wavelengths_px, n_orientations=4, sigma_ratio=0.4, min_spacing_px=1.0, name="gabor") -> SparseStage`: `(len(mosaic),) -> (n_cells,)`.

**Design notes:**
- Cells only pool inputs of their own receptor type. This is what keeps colour through the retina and cortex stages.
- The coarsest Gabor scale gets one non-oriented Gaussian cell per position. Oriented Gabors respond to the mean level with a gain of only about 0.04, so without these cells the mean brightness is nearly lost.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stages_sparse.py`:

```python
import numpy as np
import pytest

from biovision.stages.gabor import gabor_bank, gabor_kernel
from biovision.stages.mosaic import (Mosaic, all_types_at, assign_types, foveated_lattice,
                                     hex_lattice, mosaic_sampling, square_lattice)
from biovision.stages.receptive import center_surround
from biovision.stages.sparse import SparseStage, normalize_rows, pool

SIZE = 16


def _mosaic(n_types=2):
    positions = hex_lattice(SIZE, 2.0)
    types = assign_types(len(positions), [1.0] * n_types, np.random.default_rng(0))
    return Mosaic(positions, types, n_types)


def linear_stages():
    mosaic = _mosaic()
    return [
        mosaic_sampling(mosaic, SIZE),
        center_surround(mosaic, 1.0, 3.0, 0.7),
        gabor_bank(mosaic, SIZE, [8.0, 4.0]),
    ]


@pytest.mark.parametrize("stage", linear_stages(), ids=lambda s: s.name)
def test_adjoint_is_the_exact_transpose(stage, rng):
    x = rng.standard_normal(stage.in_shape)
    y = rng.standard_normal(stage.out_shape)
    assert np.vdot(stage.forward(x), y) == pytest.approx(np.vdot(x, stage.adjoint(y)), rel=1e-10)


@pytest.mark.parametrize("stage", linear_stages(), ids=lambda s: s.name)
def test_forward_output_has_the_declared_shape(stage, rng):
    assert stage.forward(rng.standard_normal(stage.in_shape)).shape == stage.out_shape


def test_mosaic_sampling_reads_a_uniform_image_exactly():
    mosaic = _mosaic()
    image = np.stack([np.full((SIZE, SIZE), 0.25), np.full((SIZE, SIZE), 0.75)])
    samples = mosaic_sampling(mosaic, SIZE).forward(image)
    np.testing.assert_allclose(samples, np.where(mosaic.types == 0, 0.25, 0.75))


def test_center_surround_response_to_a_uniform_field():
    mosaic = _mosaic()
    out = center_surround(mosaic, 1.0, 3.0, 0.7).forward(np.ones(len(mosaic)))
    np.testing.assert_allclose(out, 0.3, atol=1e-12)


def test_lattices_have_the_requested_spacing_and_stay_inside():
    square = square_lattice(SIZE, 3.0)
    assert len(square) == 36
    hexagonal = hex_lattice(SIZE, 3.0)
    for points in (square, hexagonal):
        assert points.min() >= 0 and points.max() <= SIZE - 1
    distances = np.linalg.norm(hexagonal[:, None] - hexagonal[None], axis=2)
    nearest = np.sort(distances, axis=1)[:, 1]
    assert np.median(nearest) == pytest.approx(3.0, abs=1e-9)


def test_foveated_lattice_is_denser_at_the_centre():
    points = foveated_lattice(64, 0.5, 4.0)
    assert points.min() >= 0 and points.max() <= 63
    radius = np.linalg.norm(points - 31.5, axis=1)
    inner_density = (radius < 8).sum() / (np.pi * 8**2)
    outer_density = ((radius >= 24) & (radius < 31)).sum() / (np.pi * (31**2 - 24**2))
    assert inner_density > 2.0 * outer_density


def test_foveated_lattice_never_packs_tighter_than_the_minimum():
    points = foveated_lattice(32, 0.01, 2.0, min_spacing_px=1.0)
    distances = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() > 0.7


def test_all_types_at_repeats_each_position_per_type():
    mosaic = all_types_at(square_lattice(SIZE, 4.0), 3)
    assert len(mosaic) == 3 * 16
    assert sorted(set(mosaic.types.tolist())) == [0, 1, 2]


def test_pool_only_connects_cells_of_the_same_type():
    positions = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    types = np.array([0, 1, 0])
    matrix = pool(positions, types, positions, types, 5.0, lambda dy, dx: np.ones_like(dx))
    np.testing.assert_array_equal(matrix.toarray(), [[1, 0, 1], [0, 1, 0], [1, 0, 1]])


def test_pool_with_no_neighbours_gives_an_empty_matrix():
    far = np.array([[100.0, 100.0]])
    matrix = pool(far, np.array([0]), np.zeros((1, 2)), np.array([0]), 1.0,
                  lambda dy, dx: np.ones_like(dx))
    assert matrix.shape == (1, 1) and matrix.nnz == 0
    assert normalize_rows(matrix).nnz == 0


def test_gabor_bank_has_the_expected_number_of_cells():
    mosaic = _mosaic()
    stage = gabor_bank(mosaic, SIZE, [8.0], n_orientations=4)
    # sigma = 0.4 * 8 = 3.2 px, so the grid has 5 x 5 positions for each of 2 types;
    # each position has 1 non-oriented cell and 4 orientations x 2 phases.
    assert stage.out_shape == (5 * 5 * 2 * (1 + 8),)
    assert stage.in_shape == (len(mosaic),)


def test_gabor_kernel_is_even_or_odd_by_phase():
    x = np.linspace(-5, 5, 11)
    zero = np.zeros_like(x)
    even = gabor_kernel(zero, x, 2.0, 5.0, 0.0, 0.0)
    odd = gabor_kernel(zero, x, 2.0, 5.0, 0.0, np.pi / 2.0)
    np.testing.assert_allclose(even, even[::-1], atol=1e-12)
    np.testing.assert_allclose(odd, -odd[::-1], atol=1e-12)


def test_sparse_stage_rejects_a_matrix_of_the_wrong_shape():
    from scipy.sparse import identity
    with pytest.raises(ValueError, match="does not match"):
        SparseStage("bad", identity(4, format="csr"), (5,), (4,))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_stages_sparse.py -q`
Expected: `ModuleNotFoundError: No module named 'biovision.stages.gabor'`.

- [ ] **Step 3: Implement the sparse helpers**

Create `src/biovision/stages/sparse.py`:

```python
"""Linear stages stored as sparse matrices, and helpers to build them."""
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.spatial import cKDTree

from ..core.stage import LinearStage


class SparseStage(LinearStage):
    """A linear stage whose forward map is a sparse matrix on flattened arrays."""

    def __init__(self, name: str, matrix: csr_matrix,
                 in_shape: tuple[int, ...], out_shape: tuple[int, ...]):
        expected = (int(np.prod(out_shape)), int(np.prod(in_shape)))
        if matrix.shape != expected:
            raise ValueError(f"matrix shape {matrix.shape} does not match {expected}")
        self.name = name
        self.matrix = matrix.tocsr()
        self._transpose = self.matrix.T.tocsr()
        self.in_shape = tuple(in_shape)
        self.out_shape = tuple(out_shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (self.matrix @ x.ravel()).reshape(self.out_shape)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return (self._transpose @ y.ravel()).reshape(self.in_shape)


def pool(out_pos: np.ndarray, out_types: np.ndarray,
         in_pos: np.ndarray, in_types: np.ndarray,
         radius: float, kernel: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> csr_matrix:
    """Weights from input cells to output cells of the same type within `radius`.

    Positions are (n, 2) arrays of (row, col) in pixels. `kernel(dy, dx)` gives
    the weight for an input displaced by (dy, dx) from the output cell.
    """
    shape = (len(out_pos), len(in_pos))
    neighbours = cKDTree(in_pos).query_ball_point(out_pos, r=radius)
    counts = np.fromiter((len(n) for n in neighbours), dtype=int, count=len(out_pos))
    if counts.sum() == 0:
        return csr_matrix(shape)
    rows = np.repeat(np.arange(len(out_pos)), counts)
    cols = np.concatenate([np.asarray(n, dtype=int) for n in neighbours])
    same = in_types[cols] == out_types[rows]
    rows, cols = rows[same], cols[same]
    delta = in_pos[cols] - out_pos[rows]
    values = kernel(delta[:, 0], delta[:, 1])
    return csr_matrix((values, (rows, cols)), shape=shape)


def normalize_rows(matrix: csr_matrix) -> csr_matrix:
    """Scale each row so its absolute values sum to 1. Empty rows stay empty."""
    totals = np.asarray(abs(matrix).sum(axis=1)).ravel()
    scale = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    return (diags(scale) @ matrix).tocsr()
```

- [ ] **Step 4: Implement the mosaic**

Create `src/biovision/stages/mosaic.py`:

```python
"""Photoreceptor mosaics: where receptors sit, and sampling an image at them."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .sparse import SparseStage


@dataclass(frozen=True)
class Mosaic:
    """Receptor positions as (row, col) pixels, and each receptor's type index."""

    positions: np.ndarray  # (n, 2)
    types: np.ndarray  # (n,) integers in [0, n_types)
    n_types: int

    def __len__(self) -> int:
        return len(self.positions)


def square_lattice(size_px: int, spacing_px: float) -> np.ndarray:
    """A centred square grid of points covering [0, size_px - 1]."""
    extent = size_px - 1
    n = int(np.floor(extent / spacing_px)) + 1
    axis = np.arange(n) * spacing_px + (extent - (n - 1) * spacing_px) / 2.0
    rows, cols = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([rows.ravel(), cols.ravel()])


def hex_lattice(size_px: int, spacing_px: float) -> np.ndarray:
    """A hexagonal grid: rows sqrt(3)/2 apart, alternate rows shifted by half."""
    extent = size_px - 1
    row_step = spacing_px * np.sqrt(3.0) / 2.0
    n_rows = int(np.floor(extent / row_step)) + 1
    row_offset = (extent - (n_rows - 1) * row_step) / 2.0
    n_cols = int(np.floor(extent / spacing_px)) + 1
    col_offset = (extent - (n_cols - 1) * spacing_px) / 2.0 - spacing_px / 4.0
    points = []
    for i in range(n_rows):
        shift = spacing_px / 2.0 if i % 2 else 0.0
        cols = col_offset + shift + np.arange(n_cols) * spacing_px
        cols = cols[(cols >= 0) & (cols <= extent)]
        points.append(np.column_stack([np.full(len(cols), row_offset + i * row_step), cols]))
    return np.vstack(points)


def foveated_lattice(size_px: int, center_spacing_px: float, e2_px: float,
                     min_spacing_px: float = 1.0) -> np.ndarray:
    """Rings whose spacing grows with eccentricity: s(r) = s0 * (1 + r / e2).

    Spacing never drops below `min_spacing_px`, because the image cannot carry
    detail finer than a pixel.
    """
    centre = (size_px - 1) / 2.0
    points = [np.array([[centre, centre]])]
    radius = 0.0
    limit = centre * np.sqrt(2.0)
    while True:
        spacing = max(center_spacing_px * (1.0 + radius / e2_px), min_spacing_px)
        radius += spacing
        if radius > limit:
            break
        spacing = max(center_spacing_px * (1.0 + radius / e2_px), min_spacing_px)
        count = max(int(round(2.0 * np.pi * radius / spacing)), 6)
        angles = 2.0 * np.pi * (np.arange(count) + 0.5 * (len(points) % 2)) / count
        points.append(np.column_stack([centre + radius * np.sin(angles),
                                       centre + radius * np.cos(angles)]))
    all_points = np.vstack(points)
    inside = np.all((all_points >= 0) & (all_points <= size_px - 1), axis=1)
    return all_points[inside]


def assign_types(n: int, fractions, rng: np.random.Generator) -> np.ndarray:
    """A random receptor type for each of `n` receptors, drawn with `fractions`."""
    fractions = np.asarray(fractions, dtype=float)
    return rng.choice(len(fractions), size=n, p=fractions / fractions.sum())


def all_types_at(positions: np.ndarray, n_types: int) -> Mosaic:
    """A mosaic with one receptor of every type at each position."""
    return Mosaic(np.tile(positions, (n_types, 1)),
                  np.repeat(np.arange(n_types), len(positions)), n_types)


def mosaic_sampling(mosaic: Mosaic, size_px: int, name: str = "mosaic") -> SparseStage:
    """Each receptor reads its own type's channel at its position (bilinear)."""
    n = len(mosaic)
    r = np.clip(mosaic.positions[:, 0], 0, size_px - 1)
    c = np.clip(mosaic.positions[:, 1], 0, size_px - 1)
    r0 = np.minimum(np.floor(r).astype(int), size_px - 2)
    c0 = np.minimum(np.floor(c).astype(int), size_px - 2)
    fr, fc = r - r0, c - c0
    base = mosaic.types * size_px * size_px
    rows = np.tile(np.arange(n), 4)
    cols = np.concatenate([
        base + r0 * size_px + c0,
        base + r0 * size_px + c0 + 1,
        base + (r0 + 1) * size_px + c0,
        base + (r0 + 1) * size_px + c0 + 1,
    ])
    values = np.concatenate([(1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc])
    matrix = csr_matrix((values, (rows, cols)), shape=(n, mosaic.n_types * size_px * size_px))
    return SparseStage(name, matrix, (mosaic.n_types, size_px, size_px), (n,))
```

- [ ] **Step 5: Implement the receptive fields**

Create `src/biovision/stages/receptive.py`:

```python
"""Centre-surround receptive fields (difference of Gaussians) over a mosaic."""
import numpy as np

from .mosaic import Mosaic
from .sparse import SparseStage, normalize_rows, pool


def gaussian(dy: np.ndarray, dx: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(dy**2 + dx**2) / (2.0 * sigma**2))


def center_surround(mosaic: Mosaic, sigma_center_px: float, sigma_surround_px: float,
                    surround_weight: float, name: str = "center_surround") -> SparseStage:
    """One cell per receptor: a narrow centre minus a weighted wide surround.

    Centre and surround are each normalized to sum to 1 over the receptors
    they pool, so a uniform image gives a response of 1 - surround_weight.
    """
    pos, types = mosaic.positions, mosaic.types
    centre = normalize_rows(pool(pos, types, pos, types, 3.0 * sigma_center_px,
                                 lambda dy, dx: gaussian(dy, dx, sigma_center_px)))
    surround = normalize_rows(pool(pos, types, pos, types, 3.0 * sigma_surround_px,
                                   lambda dy, dx: gaussian(dy, dx, sigma_surround_px)))
    matrix = (centre - surround_weight * surround).tocsr()
    return SparseStage(name, matrix, (len(mosaic),), (len(mosaic),))
```

Create `src/biovision/stages/gabor.py`:

```python
"""A bank of Gabor receptive fields (V1 simple cells) over a mosaic."""
import numpy as np
from scipy.sparse import vstack

from .mosaic import Mosaic, all_types_at, square_lattice
from .receptive import gaussian
from .sparse import SparseStage, normalize_rows, pool


def gabor_kernel(dy: np.ndarray, dx: np.ndarray, sigma: float, wavelength: float,
                 theta: float, phase: float) -> np.ndarray:
    """A Gaussian envelope times a cosine grating at orientation `theta`."""
    along = dx * np.cos(theta) + dy * np.sin(theta)
    return gaussian(dy, dx, sigma) * np.cos(2.0 * np.pi * along / wavelength + phase)


def gabor_bank(mosaic: Mosaic, size_px: int, wavelengths_px, n_orientations: int = 4,
               sigma_ratio: float = 0.4, min_spacing_px: float = 1.0,
               name: str = "gabor") -> SparseStage:
    """Simple cells at several scales, orientations and two phases, per type.

    For each wavelength, cells sit on a square grid one envelope sigma apart.
    The coarsest scale also gets one non-oriented Gaussian cell per position,
    which carries the mean level that the oriented cells barely respond to.
    """
    wavelengths = sorted(wavelengths_px, reverse=True)
    thetas = np.pi * np.arange(n_orientations) / n_orientations
    blocks = []
    for index, wavelength in enumerate(wavelengths):
        sigma = sigma_ratio * wavelength
        cells = all_types_at(square_lattice(size_px, max(sigma, min_spacing_px)), mosaic.n_types)
        radius = 2.5 * sigma

        def build(kernel):
            return normalize_rows(pool(cells.positions, cells.types,
                                       mosaic.positions, mosaic.types, radius, kernel))

        if index == 0:
            blocks.append(build(lambda dy, dx: gaussian(dy, dx, sigma)))
        for theta in thetas:
            for phase in (0.0, np.pi / 2.0):
                blocks.append(build(
                    lambda dy, dx: gabor_kernel(dy, dx, sigma, wavelength, theta, phase)))
    matrix = vstack(blocks).tocsr()
    return SparseStage(name, matrix, (len(mosaic),), (matrix.shape[0],))
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_stages_sparse.py -q`
Expected: `17 passed`.

- [ ] **Step 7: Commit**

```bash
git add src/biovision/stages tests/test_stages_sparse.py
git commit -m "feat: sparse stages for mosaic sampling, centre-surround and Gabor bank"
```

---

### Task 5: Pipeline

**Files:**
- Create: `src/biovision/core/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `VisualField` (Task 1); `LinearStage`, `PointwiseStage`, `ColorProjection`, `OpticalBlur`, `LinearRectified`, `PoissonSpikes` (Task 3).
- Produces:
  - `NeuralCode(responses, intermediates, pipeline_name)` (frozen dataclass). `intermediates` maps each stage name to its output, in order.
  - `Pipeline(name, field, stages, description="", citations=(), metadata={})` (frozen dataclass). Properties `linear_stages`, `pointwise_stages`, `in_shape`, `out_shape`, `n_neurons`. Methods `encode(image, rng=None) -> NeuralCode`, `linear_operator() -> scipy.sparse.linalg.LinearOperator` (with `matvec` and `rmatvec`), `replace(stage) -> Pipeline` (swaps the stage with the same name; `KeyError` if none).
  - Construction raises `ValueError` for: duplicate stage names, a linear stage after a pointwise stage, no linear stage, or mismatched shapes between consecutive linear stages.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
import numpy as np
import pytest

from biovision.core.field import VisualField
from biovision.core.pipeline import Pipeline
from biovision.stages.color import ColorProjection
from biovision.stages.nonlinearity import LinearRectified
from biovision.stages.optics import OpticalBlur
from biovision.stages.spiking import PoissonSpikes

SIZE = 8
FIELD = VisualField(SIZE, 60.0)
MATRIX = [[0.3, 0.6, 0.1], [0.0, 0.2, 0.8]]


def make(stages):
    return Pipeline("test", FIELD, tuple(stages))


def valid_stages():
    return [ColorProjection(MATRIX, SIZE), OpticalBlur(1.0, 2, SIZE),
            LinearRectified(100.0, 2.5), PoissonSpikes(0.1)]


def test_encode_records_every_stage(rng):
    pipeline = make(valid_stages())
    code = pipeline.encode(rng.random((3, SIZE, SIZE)))
    assert list(code.intermediates) == ["color", "optics", "rate", "spikes"]
    assert code.responses.shape == (2, SIZE, SIZE)
    assert pipeline.n_neurons == 2 * SIZE * SIZE
    assert code.pipeline_name == "test"


def test_linear_operator_matches_the_stages(rng):
    pipeline = make(valid_stages())
    operator = pipeline.linear_operator()
    x = rng.random((3, SIZE, SIZE))
    expected = OpticalBlur(1.0, 2, SIZE).forward(ColorProjection(MATRIX, SIZE).forward(x))
    np.testing.assert_allclose(operator.matvec(x.ravel()), expected.ravel())
    y = rng.random(operator.shape[0])
    assert operator.matvec(x.ravel()) @ y == pytest.approx(x.ravel() @ operator.rmatvec(y))


def test_linear_stage_after_pointwise_is_rejected():
    stages = [ColorProjection(MATRIX, SIZE), PoissonSpikes(0.1), OpticalBlur(1.0, 2, SIZE)]
    with pytest.raises(ValueError, match="follows a pointwise stage"):
        make(stages)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch: 'color' outputs"):
        make([ColorProjection(MATRIX, SIZE), OpticalBlur(1.0, 3, SIZE)])


def test_duplicate_names_and_missing_linear_stage_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        make([OpticalBlur(1.0, 3, SIZE), OpticalBlur(1.0, 3, SIZE)])
    with pytest.raises(ValueError, match="at least one linear stage"):
        make([PoissonSpikes(0.1)])


def test_encode_rejects_a_wrong_image_shape(rng):
    with pytest.raises(ValueError, match="expected image of shape"):
        make(valid_stages()).encode(rng.random((3, SIZE + 1, SIZE)))


def test_replace_swaps_one_stage_and_keeps_the_rest():
    pipeline = make(valid_stages())
    longer = pipeline.replace(PoissonSpikes(1.0))
    assert longer.stages[-1].window_s == 1.0
    assert longer.stages[:-1] == pipeline.stages[:-1]
    assert pipeline.stages[-1].window_s == 0.1
    with pytest.raises(KeyError, match="no stage named 'other'"):
        pipeline.replace(PoissonSpikes(1.0, name="other"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -q`
Expected: `ModuleNotFoundError: No module named 'biovision.core.pipeline'`.

- [ ] **Step 3: Implement the pipeline**

Create `src/biovision/core/pipeline.py`:

```python
"""An ordered, validated list of stages that encodes an image into a neural code."""
from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .field import VisualField
from .stage import LinearStage, PointwiseStage, Stage


@dataclass(frozen=True)
class NeuralCode:
    """The output of a pipeline, plus every stage's output for plotting."""

    responses: np.ndarray
    intermediates: dict[str, np.ndarray]
    pipeline_name: str


@dataclass(frozen=True)
class Pipeline:
    name: str
    field: VisualField
    stages: tuple[Stage, ...]
    description: str = ""
    citations: tuple[str, ...] = ()
    metadata: dict = dc_field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "stages", tuple(self.stages))
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique, got {names}")
        seen_pointwise = False
        for stage in self.stages:
            if isinstance(stage, PointwiseStage):
                seen_pointwise = True
            elif isinstance(stage, LinearStage):
                if seen_pointwise:
                    raise ValueError(
                        f"linear stage '{stage.name}' follows a pointwise stage; "
                        "all linear stages must come first"
                    )
            else:
                raise ValueError(f"{stage!r} is not a LinearStage or PointwiseStage")
        if not self.linear_stages:
            raise ValueError("a pipeline needs at least one linear stage")
        for a, b in zip(self.linear_stages, self.linear_stages[1:]):
            if a.out_shape != b.in_shape:
                raise ValueError(
                    f"shape mismatch: '{a.name}' outputs {a.out_shape} "
                    f"but '{b.name}' expects {b.in_shape}"
                )

    @property
    def linear_stages(self) -> tuple[LinearStage, ...]:
        return tuple(s for s in self.stages if isinstance(s, LinearStage))

    @property
    def pointwise_stages(self) -> tuple[PointwiseStage, ...]:
        return tuple(s for s in self.stages if isinstance(s, PointwiseStage))

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self.linear_stages[0].in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self.linear_stages[-1].out_shape

    @property
    def n_neurons(self) -> int:
        return int(np.prod(self.out_shape))

    def replace(self, stage: Stage) -> "Pipeline":
        """A copy with the stage of the same name swapped for `stage`."""
        if stage.name not in [s.name for s in self.stages]:
            raise KeyError(f"no stage named '{stage.name}' in pipeline '{self.name}'")
        stages = tuple(stage if s.name == stage.name else s for s in self.stages)
        return Pipeline(self.name, self.field, stages, self.description,
                        self.citations, self.metadata)

    def encode(self, image: np.ndarray, rng: np.random.Generator | None = None) -> NeuralCode:
        """Run `image` (channels, size, size) through every stage."""
        image = np.asarray(image, dtype=float)
        if image.shape != self.in_shape:
            raise ValueError(f"expected image of shape {self.in_shape}, got {image.shape}")
        x = image
        intermediates = {}
        for stage in self.linear_stages:
            x = stage.forward(x)
            intermediates[stage.name] = x
        for stage in self.pointwise_stages:
            x = stage.forward(x, rng)
            intermediates[stage.name] = x
        return NeuralCode(x, intermediates, self.name)

    def linear_operator(self) -> LinearOperator:
        """All linear stages composed into one operator A on flattened arrays."""
        stages = self.linear_stages

        def matvec(v):
            x = v.reshape(self.in_shape)
            for stage in stages:
                x = stage.forward(x)
            return x.ravel()

        def rmatvec(v):
            y = v.reshape(self.out_shape)
            for stage in reversed(stages):
                y = stage.adjoint(y)
            return y.ravel()

        shape = (self.n_neurons, int(np.prod(self.in_shape)))
        return LinearOperator(shape, matvec=matvec, rmatvec=rmatvec, dtype=float)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/biovision/core/pipeline.py tests/test_pipeline.py
git commit -m "feat: validated pipeline with composed linear operator"
```

---

### Task 6: Decoder

**Files:**
- Create: `src/biovision/core/regularizers.py`, `src/biovision/core/decoder.py`
- Test: `tests/test_decoder.py`

**Interfaces:**
- Consumes: `Pipeline`, `NeuralCode` (Task 5); stages from Task 3 in tests.
- Produces:
  - `laplacian(x) -> ndarray` and `chroma(x) -> ndarray`, both self-adjoint, on `(channels, size, size)` arrays.
  - `conjugate_gradient(apply, b, tol, max_iter) -> (x, residuals: list[float], converged: bool)`.
  - `Reconstruction(image, iterations, converged, residuals, lam)` (frozen dataclass); `image` is `(channels, size, size)` clipped to `[0, 1]`.
  - `Decoder(pipeline, lam, chroma_weight=0.1, max_iter=500, tol=1e-4)` with `linear_drive(responses) -> ndarray`, `noise_variance(code) -> float`, `decode(code) -> Reconstruction`. Raises `ValueError` if `lam <= 0` or `chroma_weight < 0`. A decode that does not converge emits a `RuntimeWarning` and still returns the best estimate.

**The mathematics:** the decoder solves `(A^T A + lam * P) x = A^T y`, where `A` is `pipeline.linear_operator()`, `y` is the spike counts with the pointwise stages undone, and `P x = -laplacian(x) + chroma_weight * chroma(x)`. `-laplacian` is the gradient penalty `||grad x||^2`, the standard prior for natural images (power falling as 1/f^2). `chroma` penalizes differences between colour channels, which fills in colours a species has no receptor for (red, for mouse and fly). The prototype tried `laplacian(laplacian(x))` first; it barely penalizes low frequencies and let low-frequency noise through, so keep the gradient form.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_decoder.py`:

```python
import dataclasses

import numpy as np
import pytest

from biovision.core.decoder import Decoder, conjugate_gradient
from biovision.core.field import VisualField
from biovision.core.pipeline import Pipeline
from biovision.core.regularizers import chroma, laplacian
from biovision.stages.color import ColorProjection
from biovision.stages.nonlinearity import LinearRectified
from biovision.stages.optics import OpticalBlur
from biovision.stages.spiking import PoissonSpikes

SIZE = 8


def small_pipeline(window_s=0.1):
    matrix = [[0.5, 0.4, 0.1], [0.1, 0.6, 0.3], [0.0, 0.2, 0.8]]
    return Pipeline("small", VisualField(SIZE, 60.0), (
        ColorProjection(matrix, SIZE), OpticalBlur(0.8, 3, SIZE),
        LinearRectified(100.0, 1.0), PoissonSpikes(window_s)))


def test_regularizers_are_self_adjoint(rng):
    x, y = rng.standard_normal((2, 3, SIZE, SIZE))
    for operator in (laplacian, chroma):
        assert np.vdot(operator(x), y) == pytest.approx(np.vdot(x, operator(y)))
    assert np.allclose(laplacian(np.ones((3, SIZE, SIZE))), 0.0)
    assert np.allclose(chroma(np.ones((3, SIZE, SIZE))), 0.0)


def test_conjugate_gradient_solves_a_small_system(rng):
    m = rng.standard_normal((20, 20))
    spd = m @ m.T + 20.0 * np.eye(20)
    b = rng.standard_normal(20)
    x, residuals, converged = conjugate_gradient(lambda v: spd @ v, b, 1e-10, 200)
    assert converged
    np.testing.assert_allclose(x, np.linalg.solve(spd, b), atol=1e-8)
    assert residuals[-1] <= 1e-10 and len(residuals) <= 200


def test_conjugate_gradient_with_zero_right_hand_side():
    x, residuals, converged = conjugate_gradient(lambda v: v, np.zeros(5), 1e-6, 10)
    assert converged and residuals == [] and np.all(x == 0.0)


def test_decode_matches_the_dense_solution(rng):
    """On a tiny problem, build A and the prior as matrices and solve exactly."""
    pipeline = small_pipeline()
    image = rng.random((3, SIZE, SIZE))
    lam, weight = 1e-2, 0.5
    operator = pipeline.linear_operator()
    n = operator.shape[1]
    identity = np.eye(n)
    a = np.column_stack([operator.matvec(identity[:, i]) for i in range(n)])
    prior = np.column_stack([
        (-laplacian(identity[:, i].reshape(3, SIZE, SIZE))
         + weight * chroma(identity[:, i].reshape(3, SIZE, SIZE))).ravel() for i in range(n)])
    y = a @ image.ravel()
    exact = np.linalg.solve(a.T @ a + lam * prior, a.T @ y).reshape(3, SIZE, SIZE)
    decoder = Decoder(pipeline, lam, weight, max_iter=2000, tol=1e-10)
    result = decoder.decode(pipeline.encode(image))
    assert result.converged
    np.testing.assert_allclose(result.image, np.clip(exact, 0.0, 1.0), atol=1e-5)


def test_noise_free_decoding_recovers_the_image(rng):
    """With an invertible pipeline (no blur) and almost no prior, decoding is exact."""
    matrix = [[0.5, 0.4, 0.1], [0.1, 0.6, 0.3], [0.0, 0.2, 0.8]]
    pipeline = Pipeline("exact", VisualField(SIZE, 60.0), (
        ColorProjection(matrix, SIZE), LinearRectified(100.0, 1.0), PoissonSpikes(0.1)))
    image = rng.random((3, SIZE, SIZE))
    decoder = Decoder(pipeline, 1e-9, 0.0, max_iter=5000, tol=1e-12)
    result = decoder.decode(pipeline.encode(image))
    np.testing.assert_allclose(result.image, image, atol=1e-3)


def test_noise_variance_is_the_poisson_variance_of_the_drive(rng):
    pipeline = small_pipeline(window_s=0.2)
    flat = np.full((3, SIZE, SIZE), 0.5)
    code = pipeline.encode(flat, rng)
    decoder = Decoder(pipeline, 1.0)
    # drive = (count / T / rest - 1) / gain, so its slope against count is 1 / (T * rest * gain)
    expected = code.responses.mean() * (1.0 / (0.2 * 100.0 * 1.0)) ** 2
    assert decoder.noise_variance(code) == pytest.approx(expected)
    drives = np.stack([decoder.linear_drive(pipeline.encode(flat, rng).responses)
                       for _ in range(200)])
    assert drives.var(axis=0).mean() == pytest.approx(expected, rel=0.1)


def test_unconverged_decode_warns_and_still_returns_an_image(rng):
    pipeline = small_pipeline()
    code = pipeline.encode(rng.random((3, SIZE, SIZE)))
    with pytest.warns(RuntimeWarning, match="did not converge in 2 iterations"):
        result = Decoder(pipeline, 1e-6, max_iter=2, tol=1e-12).decode(code)
    assert not result.converged and result.iterations == 2
    assert result.image.shape == (3, SIZE, SIZE)
    assert result.image.min() >= 0.0 and result.image.max() <= 1.0


def test_all_zero_spikes_decode_to_a_finite_image():
    pipeline = small_pipeline()
    code = pipeline.encode(np.zeros((3, SIZE, SIZE)))
    silent = dataclasses.replace(code, responses=np.zeros_like(code.responses))
    result = Decoder(pipeline, 1e-2).decode(silent)
    assert np.all(np.isfinite(result.image))


@pytest.mark.parametrize("kwargs", [dict(lam=0.0), dict(lam=-1.0),
                                    dict(lam=1.0, chroma_weight=-0.1)])
def test_decoder_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Decoder(small_pipeline(), **kwargs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_decoder.py -q`
Expected: `ModuleNotFoundError: No module named 'biovision.core.decoder'`.

- [ ] **Step 3: Implement the regularizers**

Create `src/biovision/core/regularizers.py`:

```python
"""Smoothness penalty used by the decoder."""
import numpy as np


def laplacian(x: np.ndarray) -> np.ndarray:
    """Discrete Laplacian over the last two axes, periodic edges. Self-adjoint."""
    return (
        np.roll(x, 1, axis=-1) + np.roll(x, -1, axis=-1)
        + np.roll(x, 1, axis=-2) + np.roll(x, -1, axis=-2)
        - 4.0 * x
    )


def chroma(x: np.ndarray) -> np.ndarray:
    """Each channel minus the mean over channels. A self-adjoint projection."""
    return x - x.mean(axis=0, keepdims=True)
```

- [ ] **Step 4: Implement the decoder**

Create `src/biovision/core/decoder.py`:

```python
"""Reconstruct an image from a neural code with a regularized linear inverse."""
import warnings
from dataclasses import dataclass

import numpy as np

from .pipeline import NeuralCode, Pipeline
from .regularizers import chroma, laplacian


@dataclass(frozen=True)
class Reconstruction:
    image: np.ndarray  # (channels, size, size), clipped to [0, 1]
    iterations: int
    converged: bool
    residuals: tuple[float, ...]  # relative residual after each iteration
    lam: float


def conjugate_gradient(apply, b: np.ndarray, tol: float, max_iter: int):
    """Solve apply(x) = b for a symmetric positive-definite operator.

    Returns (x, relative residual after each iteration, converged).
    """
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = float(r @ r)
    b_norm = float(np.sqrt(b @ b))
    residuals: list[float] = []
    if b_norm == 0.0:
        return x, residuals, True
    for _ in range(max_iter):
        ap = apply(p)
        alpha = rr / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        rr_new = float(r @ r)
        residuals.append(float(np.sqrt(rr_new)) / b_norm)
        if residuals[-1] <= tol:
            return x, residuals, True
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x, residuals, False


class Decoder:
    """Solves min ||A x - y||^2 + lam * prior(x) by conjugate gradients.

    The prior penalizes image gradients (natural images have about 1/f^2
    power) and, with `chroma_weight`, differences between colour channels
    (natural images have strongly correlated channels).
    """

    def __init__(self, pipeline: Pipeline, lam: float, chroma_weight: float = 0.1,
                 max_iter: int = 500, tol: float = 1e-4):
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        if chroma_weight < 0:
            raise ValueError(f"chroma_weight must not be negative, got {chroma_weight}")
        self.pipeline = pipeline
        self.lam = lam
        self.chroma_weight = chroma_weight
        self.max_iter = max_iter
        self.tol = tol

    def linear_drive(self, responses: np.ndarray) -> np.ndarray:
        """Undo the pointwise stages: spike counts back to the linear response."""
        y = np.asarray(responses, dtype=float)
        for stage in reversed(self.pipeline.pointwise_stages):
            y = stage.inverse(y)
        return y

    def noise_variance(self, code: NeuralCode) -> float:
        """Mean variance of the linear drive, assuming Poisson spike counts.

        A Poisson count has variance equal to its mean. The pointwise inverses
        are affine, so they scale that variance by their combined slope squared.
        """
        slope = float(np.diff(self.linear_drive(np.array([0.0, 1.0])))[0])
        return float(np.mean(code.responses)) * slope**2

    def decode(self, code: NeuralCode) -> Reconstruction:
        pipeline = self.pipeline
        a = pipeline.linear_operator()
        shape = pipeline.in_shape

        def normal(v):
            x = v.reshape(shape)
            prior = -laplacian(x) + self.chroma_weight * chroma(x)
            return a.rmatvec(a.matvec(v)) + self.lam * prior.ravel()

        b = a.rmatvec(self.linear_drive(code.responses).ravel())
        x, residuals, converged = conjugate_gradient(normal, b, self.tol, self.max_iter)
        if not converged:
            warnings.warn(
                f"decoder for '{pipeline.name}' did not converge in "
                f"{self.max_iter} iterations; returning the best estimate",
                RuntimeWarning, stacklevel=2,
            )
        image = np.clip(x.reshape(shape), 0.0, 1.0)
        return Reconstruction(image, len(residuals), converged, tuple(residuals), self.lam)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_decoder.py -q`
Expected: `11 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/biovision/core/regularizers.py src/biovision/core/decoder.py tests/test_decoder.py
git commit -m "feat: conjugate-gradient decoder with gradient and chroma priors"
```

---

### Task 7: Image input and bundled samples

**Files:**
- Create: `scripts/make_samples.py`
- Create (generated): `src/biovision/samples/__init__.py`, `src/biovision/samples/astronaut.png`, `cat.png`, `coffee.png`
- Create: `src/biovision/io.py`
- Modify: `tests/conftest.py` (add the `sample` fixture)
- Test: `tests/test_io.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `as_rgb(image) -> ndarray`: validates any numeric `(h, w)` or `(h, w, 1|3|4)` array and returns float `(h, w, 3)` in `[0, 1]`. Integers are divided by their type's maximum; floats above 1 are divided by 255; alpha is dropped. Raises `ValueError` for non-numeric arrays, wrong dimensions, fewer than 2 x 2 pixels, or NaN/infinite values.
  - `to_square(image, size_px) -> ndarray`: centre-crop to a square and resize to `(size_px, size_px, 3)`.
  - `load_image(path) -> ndarray`: `FileNotFoundError("no image at '<path>'")` or `ValueError("'<path>' is not a readable image: ...")`.
  - `sample_names() -> list[str]` (`["astronaut", "cat", "coffee"]`), `load_sample(name) -> ndarray` (`KeyError` for an unknown name).
  - Test fixture `sample`: the astronaut image.

- [ ] **Step 1: Generate the sample images**

Create `scripts/make_samples.py`:

```python
"""Write the bundled sample images from scikit-image's built-in photographs.

Run once from the repository root: python scripts/make_samples.py
"""
from pathlib import Path

from PIL import Image
from skimage import data

OUT = Path("src/biovision/samples")
SOURCES = {"astronaut": data.astronaut, "cat": data.chelsea, "coffee": data.coffee}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "__init__.py").touch()
for name, load in SOURCES.items():
    pixels = load()
    height, width = pixels.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = Image.fromarray(pixels[top:top + side, left:left + side])
    square.resize((256, 256), Image.Resampling.LANCZOS).save(OUT / f"{name}.png")
    print(f"wrote {OUT / f'{name}.png'}")
```

Run: `python scripts/make_samples.py`
Expected: three lines, `wrote src\biovision\samples\astronaut.png` and the same for `cat.png` and `coffee.png`. The script also creates `src/biovision/samples/__init__.py`.

Then reinstall so the package data is picked up: `python -m pip install -e ".[dev]" -q`

- [ ] **Step 2: Write the failing tests**

Replace `tests/conftest.py` with:

```python
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from biovision import io
from biovision.core.field import VisualField

SIZE = 32


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def sample():
    """A bundled photograph, (height, width, 3) in [0, 1]."""
    return io.load_sample("astronaut")


@pytest.fixture(scope="session")
def field():
    return VisualField(SIZE, 60.0)
```

Create `tests/test_io.py`:

```python
import numpy as np
import pytest
from PIL import Image

from biovision import io


def test_samples_are_bundled():
    assert io.sample_names() == ["astronaut", "cat", "coffee"]
    image = io.load_sample("cat")
    assert image.ndim == 3 and image.shape[2] == 3
    assert 0.0 <= image.min() and image.max() <= 1.0
    with pytest.raises(KeyError, match="unknown sample 'dog'"):
        io.load_sample("dog")


def test_to_square_crops_the_centre_of_a_wide_image():
    wide = np.zeros((10, 30, 3))
    wide[:, 10:20] = 1.0
    square = io.to_square(wide, 10)
    assert square.shape == (10, 10, 3)
    np.testing.assert_allclose(square, 1.0)


@pytest.mark.parametrize("shape", [(40, 40), (40, 40, 1), (40, 40, 4), (37, 53, 3), (2, 2, 3)])
def test_to_square_accepts_grayscale_alpha_odd_and_tiny_sizes(shape, rng):
    out = io.to_square(rng.random(shape), 16)
    assert out.shape == (16, 16, 3)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_integer_images_are_scaled_by_their_type_range():
    eight = io.as_rgb(np.full((4, 4, 3), 255, dtype=np.uint8))
    sixteen = io.as_rgb(np.full((4, 4, 3), 65535, dtype=np.uint16))
    np.testing.assert_allclose(eight, 1.0)
    np.testing.assert_allclose(sixteen, 1.0)


def test_float_images_in_0_255_are_rescaled():
    np.testing.assert_allclose(io.as_rgb(np.full((4, 4, 3), 255.0)), 1.0)


@pytest.mark.parametrize("bad, message", [
    (np.zeros((4, 4, 2)), "must be"),
    (np.zeros((4,)), "must be"),
    (np.zeros((1, 9, 3)), "at least 2 x 2"),
    (np.full((4, 4, 3), np.nan), "NaN or infinite"),
    (np.array([["a", "b"], ["c", "d"]]), "numeric"),
])
def test_as_rgb_rejects_bad_arrays(bad, message):
    with pytest.raises(ValueError, match=message):
        io.as_rgb(bad)


def test_load_image_reads_files_and_reports_problems(tmp_path):
    good = tmp_path / "good.png"
    Image.fromarray(np.full((6, 6), 128, dtype=np.uint8)).save(good)
    image = io.load_image(good)
    assert image.shape == (6, 6, 3)
    assert image[0, 0, 0] == pytest.approx(128 / 255)
    with pytest.raises(FileNotFoundError, match="no image at"):
        io.load_image(tmp_path / "missing.png")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="not a readable image"):
        io.load_image(broken)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_io.py -q`
Expected: `ImportError: cannot import name 'io' from 'biovision'`.

- [ ] **Step 4: Implement image input**

Create `src/biovision/io.py`:

```python
"""Loading, validating and resizing images. Images are (height, width, 3) in [0, 1]."""
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image

SAMPLE_PACKAGE = "biovision.samples"


def as_rgb(image) -> np.ndarray:
    """Validate an array and return it as float RGB in [0, 1]."""
    array = np.asarray(image)
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        raise ValueError("image must be a numeric array")
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"image must be (height, width) or (height, width, 1|3|4), got shape {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError(f"image must be at least 2 x 2 pixels, got {array.shape[:2]}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    array = array[:, :, :3]  # drop alpha
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(float) / np.iinfo(array.dtype).max
    array = array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError("image contains NaN or infinite values")
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def to_square(image, size_px: int) -> np.ndarray:
    """Centre-crop to a square and resize to (size_px, size_px, 3)."""
    array = as_rgb(image)
    height, width = array.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    crop = array[top:top + side, left:left + side]
    if side == size_px:
        return crop
    pil = Image.fromarray(np.round(crop * 255).astype(np.uint8))
    resized = pil.resize((size_px, size_px), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=float) / 255.0


def load_image(path) -> np.ndarray:
    """Read an image file as float RGB in [0, 1]."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no image at '{path}'")
    try:
        with Image.open(path) as pil:
            return as_rgb(np.asarray(pil.convert("RGB")))
    except OSError as error:
        raise ValueError(f"'{path}' is not a readable image: {error}") from None


def sample_names() -> list[str]:
    files = resources.files(SAMPLE_PACKAGE).iterdir()
    return sorted(f.name.rsplit(".", 1)[0] for f in files if f.name.endswith(".png"))


def load_sample(name: str) -> np.ndarray:
    if name not in sample_names():
        raise KeyError(f"unknown sample '{name}'; available: {', '.join(sample_names())}")
    with resources.as_file(resources.files(SAMPLE_PACKAGE) / f"{name}.png") as path:
        return load_image(path)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_io.py -q`
Expected: `15 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts src/biovision/io.py src/biovision/samples tests/conftest.py tests/test_io.py
git commit -m "feat: image loading, validation and bundled samples"
```

---

### Task 8: Species

**Files:**
- Create: `src/biovision/species/__init__.py`, `src/biovision/species/eye.py`, `src/biovision/species/human.py`, `src/biovision/species/mouse.py`, `src/biovision/species/fly.py`
- Modify: `src/biovision/__init__.py` (replace the whole file)
- Test: `tests/test_species.py`

**Interfaces:**
- Consumes: `VisualField` (Task 1); `species` registry (Task 2); every stage (Tasks 3 and 4); `Pipeline` (Task 5); `io.to_square` and the `sample` fixture (Task 7).
- Produces:
  - `EyeParams` (frozen dataclass, all angles in degrees): `receptor_names`, `color_matrix`, `type_fractions`, `colocated`, `blur_sigma_deg`, `lattice` (`"square" | "hex" | "foveated"`), `spacing_deg`, `center_sigma_deg`, `surround_sigma_deg`, `surround_weight`, `e2_deg=0.0`, `cortex_sf_cpd=()`, `rest_hz=100.0`, `contrast_gain=2.5`, `mosaic_seed=0`.
  - `build_mosaic(params, field) -> (Mosaic, cells_per_position: float)`.
  - `assemble(name, field, params, description, citations) -> Pipeline`. Stage names are, in order: `color`, `optics`, `mosaic`, `center_surround`, `gabor` (only if a wavelength fits), `rate`, `spikes`. `pipeline.metadata` has keys `params`, `mosaic`, `cells_per_position`.
  - `species.get("human" | "mouse" | "fly")(field) -> Pipeline`. Each species module exposes `PARAMS`, `DESCRIPTION`, `CITATIONS`, `build`.
  - `from biovision import species, __version__` works and the three species are registered on import.

**Design notes (each was measured on the prototype):**
- *Receptors smaller than a pixel.* At 128 px across 60 degrees a pixel is 0.47 degrees, far larger than a human cone. Where the true spacing is under one pixel, the model puts every receptor type at that position (many real cones of each type fall in the pixel) and multiplies the firing rate by `cells_per_position`, the number of real cells one model cell stands for. Without the first, the human reconstruction was limited to about 20 dB by an artificial one-type-per-pixel mosaic. Without the second, human vision under noise was no better than the fly's.
- *Nyquist guard.* A cortical wavelength under two pixels cannot be represented in the image, so it is dropped; if none remain the pipeline has no `gabor` stage.
- *Fly rate.* The fly's output neurons (lamina cells) use graded potentials, not spikes, with far higher signal-to-noise. This is modelled as a Poisson rate of 2500 events per second.
- *Colour matrices.* Human rows are the linear-RGB-to-LMS matrix of Vienot et al. (1999), scaled to sum to 1. Mouse and fly rows are approximations, with ultraviolet taken from the blue channel.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_species.py`:

```python
import numpy as np
import pytest

from biovision import io, species
from biovision.core.field import VisualField
from biovision.species import human, mouse
from biovision.species.eye import build_mosaic

NAMES = ["fly", "human", "mouse"]


def test_the_three_species_are_registered():
    assert species.names() == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_composed_operator_adjoint(name, field, rng):
    operator = species.get(name)(field).linear_operator()
    x = rng.standard_normal(operator.shape[1])
    y = rng.standard_normal(operator.shape[0])
    assert operator.matvec(x) @ y == pytest.approx(x @ operator.rmatvec(y), rel=1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_pipeline_carries_description_citations_and_metadata(name, field):
    pipeline = species.get(name)(field)
    assert pipeline.name == name
    assert len(pipeline.description) > 40 and len(pipeline.citations) >= 3
    assert {"params", "mosaic", "cells_per_position"} <= set(pipeline.metadata)
    assert [s.name for s in pipeline.pointwise_stages] == ["rate", "spikes"]


@pytest.mark.parametrize("name", NAMES)
def test_natural_image_rates_stay_above_zero(name, sample):
    """The contrast gain must not push many cells into rectification."""
    field = VisualField(64, 60.0)
    pipeline = species.get(name)(field)
    image = io.to_square(sample, field.size_px).transpose(2, 0, 1)
    rates = pipeline.encode(image).intermediates["rate"]
    assert np.mean(rates == 0.0) < 0.01


def test_fly_has_no_cortex_and_mammals_do():
    field = VisualField(64, 60.0)
    names = {n: [s.name for s in species.get(n)(field).linear_stages] for n in NAMES}
    assert "gabor" not in names["fly"]
    assert "gabor" in names["human"] and "gabor" in names["mouse"]


def test_cortex_is_dropped_when_no_wavelength_fits_the_image():
    """At 8 pixels across 60 degrees, every mouse and human wavelength under
    two pixels is beyond the image's Nyquist limit."""
    stages = [s.name for s in species.get("human")(VisualField(4, 170.0)).linear_stages]
    assert "gabor" not in stages


def test_receptor_counts_rank_human_mouse_fly():
    field = VisualField(64, 60.0)
    counts = {n: len(species.get(n)(field).metadata["mosaic"]) for n in NAMES}
    assert counts["human"] > counts["mouse"] > counts["fly"]


def test_sub_pixel_receptors_share_positions_and_pool_cells(field):
    mosaic, cells = build_mosaic(human.PARAMS, field)
    positions = {tuple(p) for p in np.round(mosaic.positions, 6)}
    assert len(mosaic) == 3 * len(positions)  # every type at every position
    assert cells > 1.0
    _, mouse_cells = build_mosaic(mouse.PARAMS, VisualField(128, 60.0))
    assert mouse_cells == 1.0


def test_narrow_field_gives_single_type_receptors_in_the_periphery():
    mosaic, _ = build_mosaic(human.PARAMS, VisualField(128, 1.5))
    positions = {tuple(p) for p in np.round(mosaic.positions, 6)}
    assert len(positions) < len(mosaic) < 3 * len(positions)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_species.py -q`
Expected: `ImportError: cannot import name 'species' from 'biovision'`.

- [ ] **Step 3: Implement the shared eye assembly**

Create `src/biovision/species/eye.py`:

```python
"""Shared assembly of an eye from parameters. Species files supply the numbers."""
from dataclasses import dataclass

import numpy as np

from ..core.field import VisualField
from ..core.pipeline import Pipeline
from ..stages.color import ColorProjection
from ..stages.gabor import gabor_bank
from ..stages.mosaic import (Mosaic, all_types_at, assign_types, foveated_lattice,
                             hex_lattice, mosaic_sampling, square_lattice)
from ..stages.nonlinearity import LinearRectified
from ..stages.optics import OpticalBlur
from ..stages.receptive import center_surround
from ..stages.spiking import PoissonSpikes

DEFAULT_WINDOW_S = 0.1
MIN_SPACING_PX = 1.0  # an image carries no detail finer than a pixel
MIN_SIGMA_PX = 0.5


@dataclass(frozen=True)
class EyeParams:
    """Everything that distinguishes one species' eye. Angles are in degrees."""

    receptor_names: tuple[str, ...]
    color_matrix: tuple[tuple[float, float, float], ...]  # rows: receptor types, cols: RGB
    type_fractions: tuple[float, ...]
    colocated: bool  # True: every type at every position (fly ommatidia)
    blur_sigma_deg: float
    lattice: str  # "square", "hex" or "foveated"
    spacing_deg: float  # receptor spacing (at the centre, for "foveated")
    center_sigma_deg: float
    surround_sigma_deg: float
    surround_weight: float
    e2_deg: float = 0.0  # eccentricity at which spacing doubles ("foveated" only)
    cortex_sf_cpd: tuple[float, ...] = ()  # preferred spatial frequencies; empty = no cortex
    rest_hz: float = 100.0  # firing rate with no signal
    contrast_gain: float = 2.5  # scales responses to span the firing range
    mosaic_seed: int = 0


def build_mosaic(params: EyeParams, field: VisualField) -> tuple[Mosaic, float]:
    """Receptor positions and types at this image resolution, and cells per position.

    Where the eye's receptors are smaller than a pixel, many receptors of every
    type fall inside each pixel, so that position carries all types. Elsewhere
    each position holds one receptor of a randomly drawn type.

    The second value is the mean number of real cells that one model position
    stands for (1 when receptors are at least a pixel apart).
    """
    size = field.size_px
    true_spacing = field.to_px(params.spacing_deg)
    spacing = max(true_spacing, MIN_SPACING_PX)
    if params.lattice == "square":
        positions = square_lattice(size, spacing)
        local_spacing = np.full(len(positions), true_spacing)
    elif params.lattice == "hex":
        positions = hex_lattice(size, spacing)
        local_spacing = np.full(len(positions), true_spacing)
    elif params.lattice == "foveated":
        e2 = field.to_px(params.e2_deg)
        positions = foveated_lattice(size, true_spacing, e2, MIN_SPACING_PX)
        eccentricity = np.linalg.norm(positions - (size - 1) / 2.0, axis=1)
        local_spacing = true_spacing * (1.0 + eccentricity / e2)
    else:
        raise ValueError(f"unknown lattice '{params.lattice}'")
    cells_per_position = float(np.mean(np.maximum(MIN_SPACING_PX / local_spacing, 1.0) ** 2))
    n_types = len(params.receptor_names)
    if params.colocated:
        return all_types_at(positions, n_types), cells_per_position
    rng = np.random.default_rng(params.mosaic_seed)
    dense = local_spacing < MIN_SPACING_PX
    shared = all_types_at(positions[dense], n_types)
    single = positions[~dense]
    single_types = assign_types(len(single), params.type_fractions, rng)
    mosaic = Mosaic(np.vstack([shared.positions, single]),
                    np.concatenate([shared.types, single_types]), n_types)
    return mosaic, cells_per_position


def assemble(name: str, field: VisualField, params: EyeParams,
             description: str, citations: tuple[str, ...]) -> Pipeline:
    """Optics, mosaic, retina, optional cortex, then rate and spikes."""
    size = field.size_px
    n_types = len(params.receptor_names)
    mosaic, cells_per_position = build_mosaic(params, field)
    stages = [
        ColorProjection(params.color_matrix, size),
        OpticalBlur(field.to_px(params.blur_sigma_deg), n_types, size),
        mosaic_sampling(mosaic, size),
        center_surround(mosaic,
                        max(field.to_px(params.center_sigma_deg), MIN_SIGMA_PX),
                        max(field.to_px(params.surround_sigma_deg), 2 * MIN_SIGMA_PX),
                        params.surround_weight),
    ]
    if params.cortex_sf_cpd:
        # A wavelength under two pixels is beyond what the image can carry.
        wavelengths = [w for w in (field.to_px(1.0 / sf) for sf in params.cortex_sf_cpd)
                       if w >= 2.0 * MIN_SPACING_PX]
        if wavelengths:
            stages.append(gabor_bank(mosaic, size, wavelengths, min_spacing_px=MIN_SPACING_PX))
    stages += [
        # One model cell stands for every real cell at its position, so their
        # spikes add: the effective rate scales with the number of cells.
        LinearRectified(params.rest_hz * cells_per_position, params.contrast_gain),
        PoissonSpikes(DEFAULT_WINDOW_S),
    ]
    return Pipeline(name, field, tuple(stages), description, citations,
                    {"params": params, "mosaic": mosaic,
                     "cells_per_position": cells_per_position})
```

- [ ] **Step 4: Implement the three species**

Create `src/biovision/species/human.py`:

```python
"""Human: three cone types, a fovea, centre-surround retina, V1 simple cells."""
from ..core.field import VisualField
from ..core.pipeline import Pipeline
from ..core.registry import species
from .eye import EyeParams, assemble

DESCRIPTION = (
    "Humans have three cone types (L, M, S) packed most densely at the fovea, "
    "optics sharp to about one arcminute, and a cortex that analyses the image "
    "with oriented filters. At ordinary image sizes the human eye resolves more "
    "detail than the image contains."
)

CITATIONS = (
    "Curcio CA, Sloan KR, Kalina RE, Hendrickson AE (1990). Human photoreceptor "
    "topography. J Comp Neurol 292:497-523.",
    "Vienot F, Brettel H, Mollon JD (1999). Digital video colourmaps for checking "
    "the legibility of displays by dichromats. Color Res Appl 24:243-252.",
    "Hofer H, Carroll J, Neitz J, Neitz M, Williams DR (2005). Organization of the "
    "human trichromatic cone mosaic. J Neurosci 25:9669-9679.",
    "Croner LJ, Kaplan E (1995). Receptive fields of P and M ganglion cells across "
    "the primate retina. Vision Res 35:7-24.",
    "De Valois RL, Albrecht DG, Thorell LG (1982). Spatial frequency selectivity of "
    "cells in macaque visual cortex. Vision Res 22:545-559.",
)

PARAMS = EyeParams(
    receptor_names=("L", "M", "S"),
    # Linear RGB to LMS (Vienot et al. 1999), rows scaled to sum to 1.
    color_matrix=((0.2730, 0.6643, 0.0629),
                  (0.1002, 0.7876, 0.1122),
                  (0.0178, 0.1096, 0.8726)),
    type_fractions=(0.60, 0.30, 0.10),  # Hofer et al. 2005; S cones are sparse
    colocated=False,
    blur_sigma_deg=0.007,  # point spread about 1 arcmin wide
    lattice="foveated",
    spacing_deg=0.008,  # foveal cone spacing about 0.5 arcmin (Curcio et al. 1990)
    e2_deg=2.0,  # spacing doubles by about 2 degrees eccentricity
    center_sigma_deg=0.05,  # midget cell centre (Croner & Kaplan 1995)
    surround_sigma_deg=0.5,
    surround_weight=0.7,
    cortex_sf_cpd=(0.2, 0.8),  # within the range an image of this size can carry
)


@species.register("human")
def build(field: VisualField) -> Pipeline:
    return assemble("human", field, PARAMS, DESCRIPTION, CITATIONS)
```

Create `src/biovision/species/mouse.py`:

```python
"""Mouse: two cone types (UV and green), coarse uniform sampling, low-frequency V1."""
from ..core.field import VisualField
from ..core.pipeline import Pipeline
from ..core.registry import species
from .eye import EyeParams, assemble

DESCRIPTION = (
    "Mice see with about a hundred times less acuity than humans: roughly half a "
    "cycle per degree. They have two cone types, one tuned to ultraviolet and one "
    "to green, and almost no sensitivity to red. This models the cone pathway in "
    "daylight; the mouse retina is otherwise dominated by rods."
)

CITATIONS = (
    "Prusky GT, West PW, Douglas RM (2000). Behavioral assessment of visual acuity "
    "in mice and rats. Vision Res 40:2201-2209.",
    "Jacobs GH, Neitz J, Deegan JF (1991). Retinal receptors in rodents maximally "
    "sensitive to ultraviolet light. Nature 353:655-656.",
    "Niell CM, Stryker MP (2008). Highly selective receptive fields in mouse visual "
    "cortex. J Neurosci 28:7520-7536.",
    "Stone C, Pinto LH (1993). Response properties of ganglion cells in the isolated "
    "mouse retina. Vis Neurosci 10:31-39.",
)

PARAMS = EyeParams(
    receptor_names=("UV", "M"),
    # UV opsin (360 nm) is approximated from blue; M opsin (508 nm) from green.
    color_matrix=((0.00, 0.10, 0.90),
                  (0.05, 0.85, 0.10)),
    type_fractions=(0.5, 0.5),
    colocated=False,
    blur_sigma_deg=0.3,
    lattice="square",
    spacing_deg=1.0,  # Nyquist limit 0.5 cycles/degree (Prusky et al. 2000)
    center_sigma_deg=1.0,  # ganglion cell centres span several degrees (Stone & Pinto 1993)
    surround_sigma_deg=4.0,
    surround_weight=0.7,
    cortex_sf_cpd=(0.04, 0.16),  # Niell & Stryker 2008: preferred about 0.04 cycles/degree
)


@species.register("mouse")
def build(field: VisualField) -> Pipeline:
    return assemble("mouse", field, PARAMS, DESCRIPTION, CITATIONS)
```

Create `src/biovision/species/fly.py`:

```python
"""Fruit fly: a hexagonal compound eye and lateral inhibition in the lamina."""
from ..core.field import VisualField
from ..core.pipeline import Pipeline
from ..core.registry import species
from .eye import EyeParams, assemble

DESCRIPTION = (
    "A fruit fly's compound eye has about 750 facets, each looking at a patch of "
    "the world roughly five degrees wide, arranged in a hexagonal grid. Each facet "
    "holds receptors tuned to ultraviolet, blue and green. The first layer of the "
    "brain, the lamina, sharpens the picture by subtracting neighbouring facets. "
    "Flies see almost no red."
)

CITATIONS = (
    "Land MF (1997). Visual acuity in insects. Annu Rev Entomol 42:147-177.",
    "Gonzalez-Bellido PT, Wardill TJ, Juusola M (2011). Compound eyes and retinal "
    "information processing in miniature dipteran species match their specific "
    "ecological demands. PNAS 108:4224-4229.",
    "Salcedo E, Huber A, Henrich S, et al. (1999). Blue- and green-absorbing visual "
    "pigments of Drosophila. J Neurosci 19:10716-10726.",
    "Laughlin SB (1981). A simple coding procedure enhances a neuron's information "
    "capacity. Z Naturforsch C 36:910-912.",
)

PARAMS = EyeParams(
    receptor_names=("UV", "blue", "green"),
    # UV (Rh3/Rh4) is approximated from blue; blue is Rh5, green is Rh6/Rh1.
    color_matrix=((0.00, 0.00, 1.00),
                  (0.00, 0.20, 0.80),
                  (0.05, 0.80, 0.15)),
    type_fractions=(1.0, 1.0, 1.0),
    colocated=True,  # every facet carries all receptor types
    blur_sigma_deg=2.1,  # acceptance angle about 5 degrees full width at half maximum
    lattice="hex",
    spacing_deg=5.0,  # interommatidial angle (Land 1997)
    center_sigma_deg=1.0,  # a lamina cartridge is driven by one facet
    surround_sigma_deg=6.0,  # lateral inhibition from neighbouring cartridges
    surround_weight=0.6,
    rest_hz=2500.0,
)


@species.register("fly")
def build(field: VisualField) -> Pipeline:
    return assemble("fly", field, PARAMS, DESCRIPTION, CITATIONS)
```

Create `src/biovision/species/__init__.py`:

```python
"""Importing this package registers every species."""
from . import fly, human, mouse  # noqa: F401
```

- [ ] **Step 5: Register the species on import**

Replace `src/biovision/__init__.py` with:

```python
"""Biological vision systems with mathematical image reconstruction."""
from . import species as _species  # noqa: F401  (registers the species)
from .core.registry import species

__version__ = "0.1.0"
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_species.py -q`
Expected: `15 passed` (about 10 seconds).

- [ ] **Step 7: Run everything so far**

Run: `python -m pytest -q`
Expected: `85 passed`.

- [ ] **Step 8: Commit**

```bash
git add src/biovision/__init__.py src/biovision/species tests/test_species.py
git commit -m "feat: human, mouse and fruit fly visual systems"
```

---

### Task 9: Metrics, run() and analysis

**Files:**
- Create: `src/biovision/core/metrics.py`, `src/biovision/run.py`, `src/biovision/analysis.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `io.to_square` (Task 7); `species` registry and pipelines (Task 8); `Decoder` (Task 6); `PoissonSpikes` (Task 3); `Pipeline.replace` (Task 5).
- Produces:
  - `psnr(original, reconstruction) -> float` (infinite for identical images), `ssim(original, reconstruction) -> float`, `radial_power_spectrum(image) -> (frequencies, power)` in cycles per image, each of length `size // 2`.
  - `Settings(species, fov_deg, size_px, window_ms, noise, lam, chroma_weight, seed)` (frozen dataclass).
  - `RunResult(original, reconstructed, pipeline, code, reconstruction, metrics, settings, runtime_s)` (frozen dataclass). `original` and `reconstructed` are `(size, size, 3)`. `metrics` keys: `psnr_db`, `ssim`, `neurons`, `compression_ratio`, `mean_spikes`.
  - `build_pipeline(species_name, size_px, fov_deg) -> Pipeline` (cached).
  - `run(image, species_name, *, fov_deg=60.0, size_px=128, window_ms=100.0, noise=True, lam=None, chroma_weight=0.1, seed=0) -> RunResult`. With `lam=None`: `lam = 1e-4`, plus `10.0 * decoder.noise_variance(code)` when `noise` is true. Raises `ValueError` for a non-positive `window_ms` or `lam`, and `KeyError` for an unknown species.
  - `compare_species(image, names=None, **settings) -> dict[str, RunResult]` (all species, in sorted order, when `names` is None).
  - `sweep_window(image, species_name, windows_ms=(10, 30, 100, 300, 1000), seeds=(0, 1, 2), **settings) -> list[dict]` with keys `species`, `window_ms`, `psnr_mean`, `psnr_std`, `ssim_mean`, `ssim_std`.
  - `sweep_lambda(image, species_name, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), **settings) -> list[dict]` with keys `species`, `lam`, `psnr_db`, `ssim`.

**Why lambda follows the noise:** in a Wiener-style inverse the right regularization is the noise variance times the prior precision. `Decoder.noise_variance` measures the Poisson variance of the measurements in the units the solver works in, so one constant (`NOISE_GAIN = 10.0`) serves every species and every spike window.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run.py`:

```python
import numpy as np
import pytest

from biovision.analysis import compare_species, sweep_lambda, sweep_window
from biovision.core.metrics import psnr, radial_power_spectrum, ssim
from biovision.run import run

SMALL = dict(size_px=32)


def test_metrics_on_known_images(rng):
    image = rng.random((16, 16, 3))
    assert psnr(image, image) == float("inf")
    assert ssim(image, image) == pytest.approx(1.0)
    assert psnr(np.zeros((4, 4, 3)), np.full((4, 4, 3), 0.1)) == pytest.approx(20.0)
    frequency, power = radial_power_spectrum(image)
    assert frequency.shape == power.shape == (8,)


def test_run_returns_images_metrics_and_settings(sample):
    result = run(sample, "mouse", noise=False, **SMALL)
    assert result.original.shape == result.reconstructed.shape == (32, 32, 3)
    assert set(result.metrics) == {"psnr_db", "ssim", "neurons", "compression_ratio",
                                   "mean_spikes"}
    assert result.settings.species == "mouse" and result.settings.lam == 1e-4
    assert result.runtime_s > 0


def test_noise_free_quality_ranks_human_mouse_fly(sample):
    results = compare_species(sample, noise=False, size_px=64)
    quality = {name: r.metrics["psnr_db"] for name, r in results.items()}
    assert quality["human"] > quality["mouse"] > quality["fly"]
    assert quality["human"] > quality["fly"] + 5.0


def test_same_seed_reproduces_and_different_seed_differs(sample):
    first = run(sample, "fly", seed=3, **SMALL)
    again = run(sample, "fly", seed=3, **SMALL)
    other = run(sample, "fly", seed=4, **SMALL)
    np.testing.assert_array_equal(first.reconstructed, again.reconstructed)
    assert not np.array_equal(first.reconstructed, other.reconstructed)


def test_automatic_lambda_grows_with_noise(sample):
    quiet = run(sample, "mouse", window_ms=1000.0, **SMALL).settings.lam
    loud = run(sample, "mouse", window_ms=10.0, **SMALL).settings.lam
    clean = run(sample, "mouse", noise=False, **SMALL).settings.lam
    assert loud > quiet > clean == 1e-4


def test_longer_windows_do_not_reduce_quality(sample):
    rows = sweep_window(sample, "mouse", windows_ms=(10.0, 100.0, 1000.0),
                        seeds=(0, 1, 2), **SMALL)
    means = [row["psnr_mean"] for row in rows]
    assert means[0] < means[1] < means[2]
    assert all(row["psnr_std"] >= 0 for row in rows)


def test_sweep_lambda_returns_one_row_per_value(sample):
    rows = sweep_lambda(sample, "fly", lams=(1e-3, 1e-1), noise=False, **SMALL)
    assert [row["lam"] for row in rows] == [1e-3, 1e-1]
    assert all(np.isfinite(row["psnr_db"]) for row in rows)


def test_a_uniform_image_reconstructs_as_uniform():
    flat = np.full((20, 20, 3), 0.5)
    result = run(flat, "mouse", noise=False, **SMALL)
    assert np.all(np.isfinite(result.reconstructed))
    assert result.reconstructed.std() < 0.02
    assert result.reconstructed.mean() == pytest.approx(0.5, abs=0.02)


def test_a_black_image_does_not_break_the_metrics():
    result = run(np.zeros((20, 20, 3)), "fly", **SMALL)
    assert np.all(np.isfinite(result.reconstructed))
    assert np.isfinite(result.metrics["ssim"])


@pytest.mark.parametrize("name", ["fly", "human", "mouse"])
@pytest.mark.parametrize("fov", [5.0, 170.0])
def test_extreme_fields_of_view_still_run(sample, name, fov):
    result = run(sample, name, fov_deg=fov, noise=False, size_px=24)
    assert np.all(np.isfinite(result.reconstructed))
    assert result.metrics["neurons"] >= 1


def test_run_rejects_bad_settings(sample):
    with pytest.raises(KeyError, match="unknown species 'cat'; registered: fly, human, mouse"):
        run(sample, "cat", **SMALL)
    with pytest.raises(ValueError, match="window_ms must be positive"):
        run(sample, "fly", window_ms=0.0, **SMALL)
    with pytest.raises(ValueError, match="lam must be positive"):
        run(sample, "fly", lam=0.0, **SMALL)
    with pytest.raises(ValueError, match="size_px must be greater than 1"):
        run(sample, "fly", size_px=1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_run.py -q`
Expected: `ModuleNotFoundError: No module named 'biovision.analysis'`.

- [ ] **Step 3: Implement the metrics**

Create `src/biovision/core/metrics.py`:

```python
"""Image quality measures. Images are (height, width, 3) in [0, 1]."""
import numpy as np
from skimage.metrics import structural_similarity


def psnr(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """Peak signal-to-noise ratio in decibels. Infinite for identical images."""
    mse = float(np.mean((np.asarray(original) - np.asarray(reconstruction)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def ssim(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """Structural similarity, between -1 and 1."""
    return float(structural_similarity(original, reconstruction, channel_axis=-1, data_range=1.0))


def radial_power_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean power at each spatial frequency (cycles per image), averaged over angle.

    Returns (frequencies, power) for the luminance of `image`.
    """
    luminance = np.asarray(image).mean(axis=-1)
    size = luminance.shape[0]
    power = np.abs(np.fft.fftshift(np.fft.fft2(luminance - luminance.mean()))) ** 2
    rows, cols = np.indices(power.shape)
    radius = np.hypot(rows - size // 2, cols - size // 2).astype(int)
    limit = size // 2
    totals = np.bincount(radius.ravel(), weights=power.ravel())[:limit]
    counts = np.bincount(radius.ravel())[:limit]
    return np.arange(limit, dtype=float), totals / np.maximum(counts, 1)
```

- [ ] **Step 4: Implement run()**

Create `src/biovision/run.py`:

```python
"""The single entry point: encode an image with a species and decode it again."""
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import io
from .core.decoder import Decoder, Reconstruction
from .core.field import VisualField
from .core.metrics import psnr, ssim
from .core.pipeline import NeuralCode, Pipeline
from .core.registry import species
from .stages.spiking import PoissonSpikes

LAM_FLOOR = 1e-4  # regularization when there is no noise
NOISE_GAIN = 10.0  # lam = LAM_FLOOR + NOISE_GAIN * (noise variance of the linear drive)
CHROMA_WEIGHT = 0.1


@dataclass(frozen=True)
class Settings:
    species: str
    fov_deg: float
    size_px: int
    window_ms: float
    noise: bool
    lam: float
    chroma_weight: float
    seed: int


@dataclass(frozen=True)
class RunResult:
    original: np.ndarray  # (size, size, 3)
    reconstructed: np.ndarray  # (size, size, 3)
    pipeline: Pipeline
    code: NeuralCode
    reconstruction: Reconstruction
    metrics: dict[str, float]
    settings: Settings
    runtime_s: float


@lru_cache(maxsize=16)
def build_pipeline(species_name: str, size_px: int, fov_deg: float) -> Pipeline:
    """Build (and cache) a species' pipeline. Building the sparse stages is slow."""
    return species.get(species_name)(VisualField(size_px, fov_deg))


def run(image, species_name: str, *, fov_deg: float = 60.0, size_px: int = 128,
        window_ms: float = 100.0, noise: bool = True, lam: float | None = None,
        chroma_weight: float = CHROMA_WEIGHT, seed: int = 0) -> RunResult:
    """Encode `image` through a species' visual system and reconstruct it.

    `image` is any (height, width[, channels]) array; it is centre-cropped and
    resized to `size_px`. With `lam=None` the regularization is set from the
    spike noise.
    """
    if window_ms <= 0:
        raise ValueError(f"window_ms must be positive, got {window_ms}")
    if lam is not None and lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}")
    start = time.perf_counter()
    original = io.to_square(image, size_px)
    pipeline = build_pipeline(species_name, size_px, float(fov_deg))
    pipeline = pipeline.replace(PoissonSpikes(window_ms / 1000.0))
    rng = np.random.default_rng(seed) if noise else None
    code = pipeline.encode(original.transpose(2, 0, 1), rng)
    if lam is None:
        lam = LAM_FLOOR
        if noise:
            lam += NOISE_GAIN * Decoder(pipeline, 1.0).noise_variance(code)
    decoder = Decoder(pipeline, lam, chroma_weight)
    reconstruction = decoder.decode(code)
    reconstructed = reconstruction.image.transpose(1, 2, 0)
    metrics = {
        "psnr_db": psnr(original, reconstructed),
        "ssim": ssim(original, reconstructed),
        "neurons": float(pipeline.n_neurons),
        "compression_ratio": pipeline.n_neurons / original.size,
        "mean_spikes": float(np.mean(code.responses)),
    }
    settings = Settings(species_name, float(fov_deg), size_px, float(window_ms),
                        noise, float(lam), chroma_weight, seed)
    return RunResult(original, reconstructed, pipeline, code, reconstruction,
                     metrics, settings, time.perf_counter() - start)
```

- [ ] **Step 5: Implement the analysis helpers**

Create `src/biovision/analysis.py`:

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_run.py -q`
Expected: `16 passed` (about 10 seconds). One `RuntimeWarning` about the mouse decoder not converging may be printed at small image sizes; it is expected and harmless.

- [ ] **Step 7: Check the headline numbers by hand**

Run:

```
python -c "from biovision import io; from biovision.analysis import compare_species; [print(n, round(r.metrics['psnr_db'], 1), round(r.metrics['ssim'], 2), int(r.metrics['neurons'])) for n, r in compare_species(io.load_sample('astronaut'), noise=False).items()]"
```

Expected (within about 1 dB): `fly 13.3 0.3 504`, `human 34.8 0.98 369900`, `mouse 14.6 0.53 9864`. This takes about 30 seconds; the human decode dominates.

- [ ] **Step 8: Commit**

```bash
git add src/biovision/core/metrics.py src/biovision/run.py src/biovision/analysis.py tests/test_run.py
git commit -m "feat: run entry point, metrics and sweeps"
```

---

### Task 10: Report figures, tables and export

**Files:**
- Create: `src/biovision/report/style.py`, `src/biovision/report/figures.py`, `src/biovision/report/tables.py`, `src/biovision/report/export.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `RunResult` (Task 9); `compare_species`, `sweep_window`, `sweep_lambda` (Task 9); `radial_power_spectrum` (Task 9); `gabor_kernel`, `gaussian` (Task 4); `__version__` (Task 8).
- Produces:
  - `style.PALETTE` (Okabe-Ito colours), `style.new_figure(rows=1, cols=1, width=7.0, height=3.5) -> (Figure, axes[rows][cols])`, `style.show_image(axis, image, title, **kwargs)`.
  - `figures.pipeline_panel`, `mosaic_map`, `filter_gallery`, `color_model`, `neural_code`, `error_map`, `spectrum`: each `(result: RunResult) -> Figure`. `figures.PER_SPECIES` maps the file-name keys `pipeline`, `mosaic`, `filters`, `colour`, `neural_code`, `error`, `spectrum` to them.
  - `figures.species_grid(results: dict[str, RunResult])`, `figures.convergence(results)`, `figures.window_sweep(rows_by_species: dict[str, list[dict]])`, `figures.lambda_sweep(rows_by_species)`: each returns a `Figure`.
  - `tables.parameters_table(pipeline) -> list[dict]` (keys `species`, `parameter`, `value`, `unit`), `tables.settings_table(result) -> list[dict]` (keys `setting`, `value`), `tables.results_table(results) -> list[dict]`.
  - `export.Report(results, window_sweeps={}, lambda_sweeps={})`, `export.build_report(image, names=None, sweeps=True, **settings) -> Report`, `export.write_report(report, out_dir) -> Path`, `export.zip_report(out_dir) -> bytes`.
  - `write_report` writes, into `out_dir`: `<species>_<key>.png` and `.pdf` for every key in `PER_SPECIES`; `species_grid` and `convergence` (`.png`, `.pdf`); `window_sweep` and `lambda_sweep` figures and CSVs when the report has sweeps; `results.csv`, `settings.csv`, `parameters.csv`; `results.json`; `report.md` with sections Methods, Results, Settings, Figures, Species parameters, Limits, References.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_report.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_report.py -q`
Expected: `ImportError: cannot import name 'export' from 'biovision.report'`.

- [ ] **Step 3: Implement the shared style**

Create `src/biovision/report/style.py`:

```python
"""One look for every figure."""
from matplotlib.figure import Figure

# Okabe-Ito palette: distinguishable with any common colour-vision deficiency.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
RC = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300,
}


def new_figure(rows: int = 1, cols: int = 1, width: float = 7.0, height: float = 3.5):
    """A figure and a (rows, cols) array of axes, without pyplot's global state."""
    import matplotlib

    with matplotlib.rc_context(RC):
        figure = Figure(figsize=(width, height), layout="constrained")
        axes = figure.subplots(rows, cols, squeeze=False)
    for axis in axes.ravel():
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
    return figure, axes


def show_image(axis, image, title: str, **kwargs):
    axis.imshow(image, **kwargs)
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
```

- [ ] **Step 4: Implement the figures**

Create `src/biovision/report/figures.py`:

```python
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
    axes[0, 0].set_title("reconstruction minus original (mean of channels)")
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


def species_grid(results: dict[str, RunResult]):
    """The original and every species' reconstruction, side by side."""
    first = next(iter(results.values()))
    figure, axes = new_figure(1, 1 + len(results), width=2.6 * (1 + len(results)), height=3.0)
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
```

- [ ] **Step 5: Implement the tables**

Create `src/biovision/report/tables.py`:

```python
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
```

- [ ] **Step 6: Implement the export**

Create `src/biovision/report/export.py`:

```python
"""Build a full report and write it to disk: figures, tables, numbers, Markdown."""
import csv
import io as std_io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..analysis import compare_species, sweep_lambda, sweep_window
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


def build_report(image, names=None, sweeps: bool = True, **settings) -> Report:
    """Run every species, and optionally the window and lambda sweeps."""
    results = compare_species(image, names, **settings)
    if not sweeps:
        return Report(results)
    return Report(
        results,
        {name: sweep_window(image, name, **settings) for name in results},
        {name: sweep_lambda(image, name, **settings) for name in results},
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
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/test_report.py -q`
Expected: `26 passed` (about 25 seconds).

- [ ] **Step 8: Look at the figures**

Run:

```
python -c "from biovision import io; from biovision.report.export import build_report, write_report; write_report(build_report(io.load_sample('cat'), sweeps=False, size_px=96), 'results')"
```

Open `results/species_grid.png` and `results/mouse_pipeline.png`. Check by eye: the human reconstruction looks like the original; the mouse one is blurred with red missing; the fly one is a coarse blob image; no axis label or title is cut off. `results/` is git-ignored.

- [ ] **Step 9: Commit**

```bash
git add src/biovision/report tests/test_report.py
git commit -m "feat: report figures, tables and export"
```

---

### Task 11: Command line

**Files:**
- Create: `src/biovision/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `io.load_image`, `io.load_sample` (Task 7); `species` (Task 8); `run` (Task 9); `figures.pipeline_panel`, `build_report`, `write_report` (Task 10).
- Produces: `main(argv=None) -> int` and the console script `biovision`.
  - `biovision list` prints the species names, one per line.
  - `biovision run --species NAME [--image PATH | --sample NAME] [--size 128] [--fov 60] [--window 100] [--no-noise] [--lam X] [--seed 0] [--out biovision_run.png]` writes the pipeline panel and prints `NAME: PSNR x dB, SSIM y, n neurons, k iterations`.
  - `biovision report --species NAME|all [--no-sweeps] [--out results] [same options]` writes the full report directory.
  - Returns 0 on success. On `KeyError`, `ValueError` or `FileNotFoundError` it prints `error: <message>` to standard error and returns 2. Solver warnings are printed to standard error as `warning: <message>`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -q`
Expected: `ModuleNotFoundError: No module named 'biovision.cli'`.

- [ ] **Step 3: Implement the command line**

Create `src/biovision/cli.py`:

```python
"""Command line: `biovision list | run | report`."""
import argparse
import sys
import warnings

from . import io
from .core.registry import species
from .report import figures
from .report.export import build_report, write_report
from .run import run


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", help="path to an image; a bundled sample if omitted")
    parser.add_argument("--sample", default="astronaut", help="bundled sample to use")
    parser.add_argument("--size", type=int, default=128, help="working size in pixels")
    parser.add_argument("--fov", type=float, default=60.0, help="field of view in degrees")
    parser.add_argument("--window", type=float, default=100.0, help="spike window in ms")
    parser.add_argument("--no-noise", action="store_true", help="exact spike counts")
    parser.add_argument("--lam", type=float, default=None, help="regularization strength")
    parser.add_argument("--seed", type=int, default=0)


def _settings(args) -> dict:
    return dict(size_px=args.size, fov_deg=args.fov, window_ms=args.window,
                noise=not args.no_noise, lam=args.lam, seed=args.seed)


def _image(args):
    return io.load_image(args.image) if args.image else io.load_sample(args.sample)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biovision", description="Encode an image through a species' visual system "
                                      "and reconstruct it from the neural code.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list the registered species")
    run_parser = commands.add_parser("run", help="one species, one figure")
    run_parser.add_argument("--species", required=True)
    run_parser.add_argument("--out", default="biovision_run.png", help="figure to write")
    _add_common(run_parser)
    report_parser = commands.add_parser("report", help="full report for one or all species")
    report_parser.add_argument("--species", default="all", help="a species name, or 'all'")
    report_parser.add_argument("--out", default="results", help="directory to write")
    report_parser.add_argument("--no-sweeps", action="store_true",
                               help="skip the window and lambda sweeps (much faster)")
    _add_common(report_parser)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            print("\n".join(species.names()))
            return 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if args.command == "run":
                result = run(_image(args), args.species, **_settings(args))
                figures.pipeline_panel(result).savefig(args.out, dpi=200)
                metrics = result.metrics
                print(f"{args.species}: PSNR {metrics['psnr_db']:.2f} dB, "
                      f"SSIM {metrics['ssim']:.3f}, {int(metrics['neurons'])} neurons, "
                      f"{result.reconstruction.iterations} iterations")
                print(f"figure written to {args.out}")
            else:
                names = None if args.species == "all" else [args.species]
                report = build_report(_image(args), names, sweeps=not args.no_sweeps,
                                      **_settings(args))
                out_dir = write_report(report, args.out)
                print(f"report written to {out_dir}")
        for warning in caught:
            print(f"warning: {warning.message}", file=sys.stderr)
        return 0
    except (KeyError, ValueError, FileNotFoundError) as error:
        message = error.args[0] if error.args else str(error)
        print(f"error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Try the installed command**

Run: `biovision list`
Expected:

```
fly
human
mouse
```

Run: `biovision run --species fly --out results/fly.png`
Expected: `fly: PSNR 12.92 dB, SSIM 0.266, 504 neurons, 32 iterations` (the last digit may differ between NumPy versions) and `figure written to results/fly.png`.

- [ ] **Step 6: Commit**

```bash
git add src/biovision/cli.py tests/test_cli.py
git commit -m "feat: command line with list, run and report"
```

---

### Task 12: Streamlit app

**Files:**
- Create: `app/streamlit_app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `io` (Task 7); `species` (Task 8); `run`, `compare_species`, `sweep_window`, `sweep_lambda` (Task 9); `figures`, `tables`, `Report`, `write_report`, `zip_report` (Task 10).
- Produces: a Streamlit script with a sidebar (image upload or sample picker, species dropdown filled from the registry, spike window, noise toggle, field of view, working size, automatic or manual lambda, seed) and four tabs:
  - **Explore:** original beside reconstruction, three metrics (PSNR, SSIM, Neurons), and the species description.
  - **Stages:** pipeline panel, mosaic map, colour model, filter gallery.
  - **Analysis:** error map, spectrum, spike histogram, convergence, the three tables, references, and a button that runs the sweeps.
  - **Compare:** a button that runs every species, shows the grid and results table, and offers the report as a zip download.

**Notes:**
- Results are cached with `st.cache_data` on the image array and a tuple of settings, so moving a slider back to an earlier value is instant.
- The default working size in the app is 96 pixels, not 128, because the human decode takes about 10 seconds at 128.
- A file that cannot be read shows an error in the sidebar and the script stops; it must never raise.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_app.py`:

```python
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_renders_the_default_view_without_errors():
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert not app.exception
    assert app.title[0].value == "biovision"
    assert [tab.label for tab in app.tabs] == ["Explore", "Stages", "Analysis", "Compare"]
    assert [metric.label for metric in app.metric] == ["PSNR", "SSIM", "Neurons"]


def test_app_switches_species_from_the_registry():
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    selector = next(box for box in app.sidebar.selectbox if box.label == "Species")
    assert list(selector.options) == ["fly", "human", "mouse"]
    selector.set_value("mouse").run()
    assert not app.exception
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_app.py -q`
Expected: both tests fail; the app file does not exist, so `AppTest` reports an exception or `FileNotFoundError`.

- [ ] **Step 3: Implement the app**

Create `app/streamlit_app.py`:

```python
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
    left.image(result.original, caption="Original", use_container_width=True)
    right.image(result.reconstructed, caption=f"What the {name} code keeps",
                use_container_width=True)
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_app.py -q`
Expected: `2 passed` (about 15 seconds).

- [ ] **Step 5: Use the app by hand**

Run: `streamlit run app/streamlit_app.py`

In the browser, check each of these:
1. The page loads with the astronaut sample and the fly selected, and shows two images and three metrics.
2. Switching the species to `human` shows a spinner, then a reconstruction close to the original.
3. Uploading your own JPG replaces the sample.
4. Turning "Spike noise" off raises the PSNR.
5. Each of the four tabs renders with no red error box.
6. On the Compare tab, "Compare every species" shows three reconstructions and a "Download report (zip)" button whose zip contains `report.md`.
7. Uploading a text file renamed to `.png` shows "Could not read that image" in the sidebar and no traceback.

Stop the server with Ctrl+C.

- [ ] **Step 6: Commit**

```bash
git add app tests/test_app.py
git commit -m "feat: Streamlit app with explore, stages, analysis and compare tabs"
```

---

### Task 13: README and final verification

**Files:**
- Modify: `README.md` (replace the whole file)

**Interfaces:**
- Consumes: everything.
- Produces: user documentation and a verified, complete repository.

- [ ] **Step 1: Write the README**

Replace `README.md` with:

````markdown
# biovision

Encode an image through a model of a species' early visual system, then rebuild
the image from the resulting neural code with closed-form mathematics. No
machine learning: every stage is a published biological model, and decoding is
a regularized linear inverse.

The result shows what information about a scene survives each species' eye.

| Species | What its code keeps |
|---|---|
| Human | Almost everything an ordinary image contains, in full colour |
| Mouse | A blurred image (about half a cycle per degree) with no red |
| Fruit fly | A coarse hexagonal mosaic of about five-degree patches, no red |

## Install

```
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"
```

## Use

```
biovision list
biovision run --species mouse --image cat.jpg --out mouse.png
biovision report --species all --out results/
streamlit run app/streamlit_app.py
```

`run` writes one figure and prints the quality numbers. `report` writes every
figure (PNG and PDF), every table (CSV), `results.json` and `report.md`. Add
`--no-sweeps` to skip the slow parameter sweeps. With no `--image`, a bundled
sample is used.

From Python:

```python
from biovision import io
from biovision.run import run

result = run(io.load_image("cat.jpg"), "fly", window_ms=100)
print(result.metrics["psnr_db"], result.reconstructed.shape)
```

## How it works

Each species is a pipeline of stages. Linear stages come first, then pointwise
stages:

1. `color`: RGB projected onto the species' photoreceptor types.
2. `optics`: blur by the eye's point-spread function.
3. `mosaic`: sampling at the receptor positions (foveated, square or hexagonal).
4. `center_surround`: difference-of-Gaussians receptive fields.
5. `gabor` (mammals only): V1 simple cells at several scales and orientations.
6. `rate`: a threshold-linear firing rate around a resting rate.
7. `spikes`: Poisson spike counts in a time window.

Decoding undoes the pointwise stages, then solves

    minimize ||A x - y||^2 + lambda * (||grad x||^2 + w * ||chroma x||^2)

by conjugate gradients, where `A` is all the linear stages composed. The prior
says natural images are smooth (power falling as 1/f^2) and their colour
channels are correlated. With spike noise, `lambda` is set from the Poisson
variance of the measurements.

## Layout

```
src/biovision/
  core/      field, stage interfaces, pipeline, registry, decoder, metrics
  stages/    the mathematics; no species knowledge
  species/   parameters and citations; no mathematics
  report/    figures, tables, export
  run.py     the single entry point used by the CLI and the app
  analysis.py  species comparison and sweeps
app/streamlit_app.py
```

## Add a species

Create `src/biovision/species/<name>.py` with an `EyeParams`, a description,
citations and a registered `build` function, then import it in
`src/biovision/species/__init__.py`. The generic adjoint test covers it.

```python
@species.register("cat")
def build(field: VisualField) -> Pipeline:
    return assemble("cat", field, PARAMS, DESCRIPTION, CITATIONS)
```

## Limits

- Ultraviolet is approximated from the blue channel of an RGB image.
- Image values are treated as linear light.
- Where receptors are smaller than a pixel, the image sets the resolution, not
  the eye. One model cell then stands for all the real cells in that pixel.
- The mouse model is its cone pathway in daylight.
- Still images only; no motion pathways.

## Tests

```
pytest
```

## Credits

Sample images come from scikit-image: the astronaut photograph (NASA, public
domain), the cat (Stefan van der Walt, CC0) and the coffee cup (Rachel
Michetti, CC0).
````

- [ ] **Step 2: Run the whole suite**

Run: `python -m pytest -q`
Expected: `134 passed` (about one minute).

- [ ] **Step 3: Produce a full report end to end**

Run: `biovision report --species all --out results --size 96`
Expected: `report written to results` after a few minutes (the sweeps run 20 decodes per species). Open `results/report.md` and check it has the Methods, Results, Figures and References sections, that `results/window_sweep.png` shows quality rising with the spike window for all three species, and that `results/results.json` lists three species.

- [ ] **Step 4: Check the architecture rules**

Run: `python -c "import pathlib, re; bad = [str(p) for p in pathlib.Path('src/biovision/stages').glob('*.py') if re.search(r'^(from|import) .*species', p.read_text(), re.M)]; print(bad or 'stages do not import species')"`
Expected: `stages do not import species`.

Run: `python -c "import pathlib, re; bad = [str(p) for p in pathlib.Path('src').rglob('*.py') if re.search(r'^\s*(from|import) matplotlib.*pyplot', p.read_text(), re.M)]; print(bad or 'no pyplot imports')"`
Expected: `no pyplot imports`.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: README"
```
