# biovision: modular biological vision systems with mathematical image reconstruction

Design spec, 2026-10-03. Revised the same day after prototyping; see
"Revisions after prototyping" below.

## 1. Purpose

Build a system that encodes an image through a model of a species' early
visual system (human, mouse, fruit fly) and then reconstructs the image from
the resulting neural code using closed-form mathematics. No machine learning:
every stage is a published biological model, and decoding is a regularized
linear inverse.

The result shows, for each species, what information about a scene survives
its visual front end.

Audience: a portfolio piece for neuroscience and robotics work, usable by a
non-expert through a simple app, and able to produce every plot and number
needed for an academic report.

### Success criteria

- One command or one app interaction takes an image and a species and returns
  the neural code and a reconstructed image.
- Species are selected by name from a registry. Adding a species means adding
  one file and one import line.
- Species differences are clearly visible and match the biology (resolution,
  colour, sampling pattern).
- Every run can export report-grade figures and the numbers behind them.
- Results are reproducible from a seed.

### Decisions made

| Question | Decision |
|---|---|
| Meaning of "reconstruct" | Decode the neural code back to an image (invert the stack) |
| Neural code | Graded responses in early stages, Poisson spike counts in a time window at the output |
| Colour | RGB mapped to each species' photoreceptor types by a per-species matrix |
| Input | Still images. The stage interface leaves room for a time axis later |
| Decoding architecture | One global regularized inverse over a composed linear operator |
| Interface | Command line, Python API, and a Streamlit upload app |
| Repository | New standalone public GitHub repo, `Mohak327/biovision` |

### Out of scope for v1

- Video and motion pathways (fly elementary motion detectors, direction
  selectivity).
- Spike trains with timing (integrate-and-fire, GLMs).
- Hyperspectral input and true UV.
- Rod pathways and light adaptation.
- A browser-only (JavaScript) version.

### Revisions after prototyping

The first version of this spec was approved, then the whole design was run as
a prototype. These points changed, each for a measured reason:

1. **Firing-rate nonlinearity.** A saturating (Naka-Rushton) curve was replaced
   by a threshold-linear rate, the output stage of a linear-nonlinear-Poisson
   neuron. Inverting a curved function on noisy spike counts is biased (mean
   brightness error of about +0.5), and a tangent approximation cost 14 dB even
   without noise.
2. **Prior.** The smoothness penalty is the image gradient (`-laplacian`), not
   the squared Laplacian, which let low-frequency noise through. A second term
   penalizes differences between colour channels, which fills in the channel a
   species has no receptor for.
3. **Regularization strength.** By default `lambda` is set from the Poisson
   variance of the measurements, not from two fixed values.
4. **Receptors smaller than a pixel.** Where an eye's receptors are smaller
   than a pixel, that position carries every receptor type, and its firing
   rate is multiplied by the number of real cells it stands for. The earlier
   claim that the human reconstruction "degrades toward the edges" holds only
   when the field of view is narrow enough for peripheral receptors to exceed
   a pixel; at the default settings the human eye out-resolves the image
   everywhere.
5. **Fly output.** Fly lamina cells are graded, not spiking. This is modelled
   as a Poisson rate of 2500 events per second.
6. **Structure.** Species share one assembly function driven by an `EyeParams`
   dataclass. Mosaic, centre-surround and Gabor stages are sparse matrices
   built by one pooling helper. Sweeps and comparisons live in `analysis.py`.
   The solver is a small conjugate-gradient loop of our own.

## 2. Repository layout

```
biovision/
  pyproject.toml
  README.md
  AGENTS.md, CLAUDE.md
  scripts/make_samples.py
  src/biovision/
    __init__.py
    io.py              # load, validate, crop and resize images; bundled samples
    run.py             # run(): the single entry point used by CLI and app
    analysis.py        # compare_species, sweep_window, sweep_lambda
    cli.py
    core/
      field.py         # VisualField: pixels <-> degrees
      stage.py         # Stage, LinearStage, PointwiseStage
      pipeline.py      # Pipeline, NeuralCode
      registry.py      # Registry[T], the species registry
      regularizers.py  # laplacian, chroma
      decoder.py       # conjugate_gradient, Decoder, Reconstruction
      metrics.py       # psnr, ssim, radial_power_spectrum
    stages/
      color.py         # ColorProjection
      optics.py        # OpticalBlur
      nonlinearity.py  # LinearRectified
      spiking.py       # PoissonSpikes
      sparse.py        # SparseStage, pool, normalize_rows
      mosaic.py        # Mosaic, lattices, mosaic_sampling
      receptive.py     # center_surround
      gabor.py         # gabor_bank
    species/
      __init__.py      # imports each species module so it registers
      eye.py           # EyeParams, build_mosaic, assemble
      human.py
      mouse.py
      fly.py
    report/
      style.py         # shared Matplotlib style
      figures.py       # one pure function per figure
      tables.py        # one pure function per table
      export.py        # Report, build_report, write_report, zip_report
    samples/           # astronaut.png, cat.png, coffee.png
  app/
    streamlit_app.py
  tests/
```

Dependencies: NumPy, SciPy, Matplotlib, Pillow, scikit-image (SSIM and the
sample images), Streamlit (app only, optional extra), pytest (dev).

Rule of separation:

- `stages/` holds mathematics and knows nothing about species.
- `species/` holds parameters and assembly and contains no mathematics.
- `core/` knows nothing about either.
- `report/`, `cli.py` and `app/` are thin layers that call `run()` and
  `analysis`.

## 3. Core abstractions

### VisualField

`VisualField(size_px: int, fov_deg: float)`. Converts between pixels and
degrees of visual angle. Species parameters are given in degrees, so every
species sees the same image at the same field of view (default 60 degrees
across, working size 128 px, both configurable). Stages take pixels.

### Stage

Two kinds, both immutable after construction:

```python
class LinearStage(Stage):
    name: str
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    def forward(self, x: ndarray) -> ndarray: ...
    def adjoint(self, y: ndarray) -> ndarray: ...   # exact transpose of forward

class PointwiseStage(Stage):
    name: str
    def forward(self, x: ndarray, rng: Generator | None = None) -> ndarray: ...
    def inverse(self, y: ndarray) -> ndarray: ...   # elementwise inverse
```

A pipeline is all linear stages followed by all pointwise stages. This is the
biological order (filtering, firing rate, spikes) and it means the decoder
never has to handle a mixed sequence. A time axis can be added later without
changing the interface.

### Pipeline and NeuralCode

```python
@dataclass(frozen=True)
class Pipeline:
    name: str
    field: VisualField
    stages: tuple[Stage, ...]
    description: str = ""
    citations: tuple[str, ...] = ()
    metadata: dict = {}

    def encode(self, image, rng=None) -> NeuralCode: ...
    def linear_operator(self) -> scipy.sparse.linalg.LinearOperator: ...
    def replace(self, stage) -> Pipeline: ...   # swap the stage of the same name
    # properties: linear_stages, pointwise_stages, in_shape, out_shape, n_neurons
```

Construction validates: unique stage names, linear before pointwise, at least
one linear stage, and matching shapes between consecutive linear stages.
Invalid pipelines fail at construction.

`replace` lets `run()` change the spike window without rebuilding the sparse
stages.

`NeuralCode` is a frozen dataclass: `responses` (the spike counts),
`intermediates` (an ordered mapping from stage name to that stage's output,
kept for plotting), and `pipeline_name`.

### Registry

```python
class Registry(Generic[T]):
    def register(self, name: str) -> Callable[[T], T]: ...   # decorator
    def get(self, name: str) -> T: ...
    def names(self) -> list[str]: ...

species = Registry("species")   # holds build(field: VisualField) -> Pipeline
```

Duplicate names raise at import. Unknown names raise a `KeyError` whose
message lists the registered names.

## 4. Stages

| Stage | Kind | Forward | Adjoint or inverse |
|---|---|---|---|
| `ColorProjection` | linear | K x 3 matrix applied to RGB | transposed matrix |
| `OpticalBlur` | linear | Gaussian point-spread function, applied by FFT (periodic edges) | same operation (symmetric kernel) |
| `mosaic_sampling` | linear, sparse | each receptor reads its own type's channel at its position (bilinear) | matrix transpose |
| `center_surround` | linear, sparse | difference of Gaussians over the mosaic, one cell per receptor | matrix transpose |
| `gabor_bank` | linear, sparse | Gabor filters at several scales, 4 orientations and 2 phases, per receptor type | matrix transpose |
| `LinearRectified` | pointwise | rate = max(rest * (1 + gain * x), 0) | x = (rate / rest - 1) / gain |
| `PoissonSpikes` | pointwise | counts ~ Poisson(rate * window); with no rng, rate * window exactly | counts / window |

Notes:

- The three sparse stages are built by one helper, `pool`, which gives weights
  from input cells to output cells of the same receptor type within a radius.
  Pooling within a type is what keeps colour through the retina and cortex.
- Centre and surround are each normalized to sum to 1, so a uniform image
  gives a response of `1 - surround_weight`.
- Gabor rows are normalized to an absolute sum of 1. The coarsest scale also
  has one non-oriented Gaussian cell per position, which carries the mean
  level that oriented cells barely respond to.
- The resting rate stands for an ON/OFF pair of cells: responses above rest
  are the ON cell, below rest the OFF cell. `gain` (2.5) scales responses so
  natural images span the firing range without rectifying.
- Lattice generators: `square_lattice`, `hex_lattice` and `foveated_lattice`
  (spacing `s0 * (1 + r / e2)`, never below one pixel).

## 5. Species

Each species file holds an `EyeParams` (frozen dataclass, angles in degrees,
each value with a citation comment), a plain-language description, a list of
citations, and a registered `build(field)` that calls the shared `assemble`.

| | Human | Mouse | Fruit fly |
|---|---|---|---|
| Photoreceptor types | L, M, S cones (60:30:10) | UV and green cones | UV, blue, green in every facet |
| Optical blur (sigma) | 0.007 degrees | 0.3 degrees | 2.1 degrees |
| Mosaic | foveated, 0.008 degrees at centre, doubling by 2 degrees | square, 1 degree | hexagonal, 5 degrees |
| Resolution limit | about 60 cycles/degree | 0.5 cycles/degree | 0.1 cycles/degree |
| Centre / surround sigma | 0.05 / 0.5 degrees | 1 / 4 degrees | 1 / 6 degrees |
| Surround weight | 0.7 | 0.7 | 0.6 |
| Cortex (Gabor) frequencies | 0.2 and 0.8 cycles/degree | 0.04 and 0.16 cycles/degree | none |
| Resting rate | 100 /s per cell | 100 /s | 2500 /s (graded) |

Rules applied by `assemble`:

- Where receptor spacing is under one pixel, the position carries every
  receptor type and its rate is multiplied by the mean number of real cells
  per position.
- A cortical wavelength under two pixels is dropped. If none remain there is
  no cortex stage.

Stated limits (README and report):

- UV is approximated from the blue channel of an RGB image.
- Image values are treated as linear light.
- At ordinary image sizes the human eye out-resolves the image, so the image
  sets the resolution.
- The mouse retina is rod-dominated; this models its cone pathway in daylight.
- PSNR for mouse and fly is dominated by the red channel they cannot sense.

## 6. Decoding

```python
class Decoder:
    def __init__(self, pipeline, lam, chroma_weight=0.1, max_iter=500, tol=1e-4): ...
    def linear_drive(self, responses) -> ndarray: ...
    def noise_variance(self, code) -> float: ...
    def decode(self, code) -> Reconstruction: ...
```

Steps:

1. Undo the pointwise stages in reverse order to get the linear drive `y`.
2. Solve `(A^T A + lam * P) x = A^T y` by conjugate gradients, where `A` is
   `pipeline.linear_operator()` and
   `P x = -laplacian(x) + chroma_weight * chroma(x)`. The first term is the
   gradient penalty (natural images have power falling as 1/f^2). The second
   penalizes each channel's difference from the mean over channels.
3. Reshape to an image and clip to [0, 1].

`noise_variance` is the mean spike count (the variance of a Poisson count)
times the squared slope of the pointwise inverses: the variance of `y`.

`Reconstruction` is a frozen dataclass: `image`, `iterations`, `converged`,
`residuals` (relative residual per iteration), `lam`.

## 7. Run entry point

```python
def run(image, species_name, *, fov_deg=60.0, size_px=128, window_ms=100.0,
        noise=True, lam=None, chroma_weight=0.1, seed=0) -> RunResult
```

With `lam=None`, `lam = 1e-4`, plus `10 * noise_variance` when noise is on.
Pipelines are cached per (species, size, field of view).

`RunResult` holds `original`, `reconstructed` (both `(size, size, 3)`),
`pipeline`, `code`, `reconstruction`, `metrics`, `settings`, `runtime_s`.
Metrics: `psnr_db`, `ssim`, `neurons`, `compression_ratio`, `mean_spikes`.

## 8. Metrics and analysis

- `psnr`, `ssim`, `radial_power_spectrum` in `core/metrics.py`.
- `compare_species(image, names=None, **settings)`: one `RunResult` per species.
- `sweep_window(image, species_name, windows_ms, seeds, **settings)`: quality
  against spike window, mean and standard deviation over seeds.
- `sweep_lambda(image, species_name, lams, **settings)`: quality against
  regularization.

## 9. Reporting

The app must be friendly for a casual user and complete for an academic
report. Both needs are served by one reporting layer; the app and the CLI
only choose how much of it to show.

### Figures (`report/figures.py`)

Each is a pure function from results to a Matplotlib `Figure` (built without
`pyplot`), with a shared style: labelled axes with units, colourbars,
consistent fonts, and the Okabe-Ito colour-blind-safe palette.

1. Pipeline panel: original, output of each image-like stage, reconstruction.
2. Mosaic map: receptor positions over the image, coloured by type.
3. Filter gallery: centre-surround profile and Gabor kernels, in degrees.
4. Colour model: the RGB-to-receptor matrix and the colour that survives it.
5. Neural code: histogram of spike counts.
6. Reconstruction error: signed error map and per-channel error.
7. Spectrum comparison: radially averaged power spectrum of original and
   reconstruction, with the species' sampling limit marked. This links to the
   existing 2D DFT project.
8. Quality against spike window: PSNR and SSIM with error bars over seeds.
9. Quality against lambda.
10. Species comparison grid: reconstructions side by side with metrics.
11. Convergence: conjugate-gradient residual against iteration.

### Tables (`report/tables.py`)

- Species parameters with units.
- Run settings (field of view, size, window, lambda, seed, package version).
- Results per species: neurons, receptors, compression ratio, mean spikes,
  PSNR, SSIM, lambda, iterations, converged, runtime.
- Sweep data behind figures 8 and 9.

### Export (`report/export.py`)

`write_report(build_report(image, names, sweeps, **settings), out_dir)` writes:

- every figure as PNG (300 dpi) and PDF (vector),
- every table as CSV, plus one `results.json` with all numbers and settings,
- `report.md` with a methods paragraph, the results and settings tables, every
  figure with a numbered caption, the species parameters, the stated limits,
  and a reference list built from the species citations.

`zip_report(out_dir)` returns the directory as a zip for the app's download.

### Command line

```
biovision list
biovision run    --species mouse [--image cat.jpg] [--window 100] [--no-noise] [--lam 0.1] [--seed 0] [--out fig.png]
biovision report --species all --out results/ [--no-sweeps]
```

`run` saves the pipeline panel and prints the metrics. `report` writes the
full export. With no `--image`, a bundled sample is used. Errors print one
line and exit with code 2.

### Streamlit app

- Sidebar: image upload or sample picker, species dropdown filled from the
  registry, spike window, noise toggle, field of view, working size, automatic
  or manual lambda, seed.
- Tab "Explore": original beside reconstruction, three headline numbers
  (PSNR, SSIM, neurons), and a plain-language description of the species' eye.
- Tab "Stages": the pipeline panel, mosaic map, colour model, filter gallery.
- Tab "Analysis": error map, spectrum comparison, spike histogram,
  convergence, all tables, references, and a button that runs the sweeps.
- Tab "Compare": a button that runs every species, the comparison grid and
  results table, and a "Download report" button returning a zip.

Runs are cached on the image and settings so sliders stay responsive. The
app's default working size is 96 px, because the human decode takes about 10
seconds at 128 px.

## 10. Error handling

- Unknown species: `KeyError` listing registered names.
- Invalid pipeline (ordering, names or shape mismatch): `ValueError` at
  construction.
- Invalid image (non-numeric, wrong dimensions, under 2 x 2, NaN): `ValueError`
  with a readable message. Grayscale is promoted to RGB; alpha is dropped;
  integer types are scaled by their range.
- Missing or unreadable image file: `FileNotFoundError` or `ValueError`.
- Conjugate gradients not converging: return the best iterate with
  `converged=False` and emit a `RuntimeWarning`. The app shows a notice.
- Invalid settings (non-positive window, field of view, size or lambda):
  `ValueError`.

## 11. Testing

pytest, with all randomness through an explicit `numpy.random.Generator`.

- Adjoint test, parametrized over every linear stage and over the composed
  operator of every registered species: `<A x, y> == <x, A^T y>`.
- Pointwise round trip: `inverse(forward(x)) == x`.
- Dense oracle: on an 8 x 8 image, materialize `A` and the prior and check the
  conjugate-gradient solution against the exact solution.
- Noise variance: the formula matches the empirical variance over 200 draws.
- Registry, pipeline validation, field conversions, lattice spacing.
- Species: rates rarely rectify on a natural image; sub-pixel receptors are
  pooled; the cortex is dropped when no wavelength fits.
- Sanity ordering: with no noise, PSNR ranks human > mouse > fly.
- Noise: mean PSNR over seeds rises with the spike window.
- Reproducibility: same seed gives identical output.
- Robustness: uniform and black images, extreme fields of view, all-zero
  spikes, odd image shapes and types, unreadable files.
- Report: every figure draws for every species; `write_report` writes the
  expected files; `results.json` and the CSVs parse.
- CLI: `list`, `run`, `report` and the error paths.
- App: renders without exceptions and switches species (Streamlit `AppTest`).

## 12. Extension points

- New species: add `species/<name>.py` with an `EyeParams` and
  `@species.register`, and import it in `species/__init__.py`.
- New stage: subclass `LinearStage` (forward and adjoint) or `PointwiseStage`
  (forward and inverse); add it to the parametrized adjoint test.
- Motion and video: add a time axis to arrays and new stages; the pipeline,
  registry and decoder interfaces do not change for linear stages.
- New prior: another self-adjoint operator in `regularizers.py`.
