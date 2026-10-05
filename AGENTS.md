# biovision

Project description for coding agents. Read this first, then the spec and the
plan.

## What this is

A Python package that encodes an image through a model of a species' early
visual system (human, mouse, fruit fly) and reconstructs the image from the
resulting neural code with closed-form mathematics. The reconstruction shows
what information about a scene survives each species' eye.

This is not a machine-learning project. Every stage is a fixed, published
biological model. Decoding is a regularized linear inverse solved by conjugate
gradients. Nothing is trained or fitted.

It has two audiences: a casual user of the Streamlit app, and a reader of an
academic report who needs every figure and number.

## Documents

- Spec (what and why): `docs/superpowers/specs/2026-10-03-biovision-design.md`
- Plan (how, task by task, with full code): `docs/superpowers/plans/2026-10-03-biovision.md`

The code in the plan was run as a prototype before the plan was written, and
its 134 tests passed. The plan is now fully implemented; treat it as the
record of how the code was built.

## Status

The library, the command line, the web server and the React front end are
implemented on the `implementation` branch. The Streamlit app has been
removed. Added after the original plan: `density` and `neuron_density`
options, progress reporting from `run()`, `biovision.server` and `web/`.
The web app's design is recorded in
`docs/superpowers/specs/2026-10-04-react-frontend-design.md`.

## Commands

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"   # Windows
python -m pytest -q                               # whole suite, about one minute
python -m pytest tests/test_decoder.py -q         # one file
biovision list
biovision run --species mouse --out results/mouse.png
biovision report --species all --out results --no-sweeps
biovision serve                                   # web app at http://127.0.0.1:8000
cd web && npm install && npm run build            # build the front end (served by `serve`)
cd web && npm run dev                             # front-end dev server on :5173
cd web && npm test                                # front-end logic tests
python scripts/benchmark.py --size 128            # PSNR, SSIM, time and peak memory on the samples
```

## Architecture

```
image (size, size, 3)
  -> run()  ->  Pipeline.encode  ->  NeuralCode (spike counts)
            ->  Decoder.decode   ->  Reconstruction
            ->  metrics          ->  RunResult
```

- A **stage** is either a `LinearStage` (`forward` and an exact `adjoint`) or a
  `PointwiseStage` (`forward` and an exact `inverse`).
- A **pipeline** is all linear stages, then all pointwise stages. `Pipeline`
  validates this when it is constructed.
- A **species** is a registered function `build(field) -> Pipeline`. It lives in
  one file holding parameters and citations. `species/eye.py` assembles the
  stages from an `EyeParams`.
- The **decoder** undoes the pointwise stages, then solves
  `(A^T A + lam * P) x = A^T y`, where `A` is the composed linear stages and
  `P = -laplacian + chroma_weight * chroma`, by preconditioned conjugate
  gradients.
- `run()` is the single entry point. The CLI, the server and the report layer
  call it and contain no mathematics. The React app in `web/` only draws what
  the server sends.

Stage order for every species: `color`, `optics`, `mosaic`, `center_surround`,
`gabor` (mammals only, and only if a wavelength fits the image), `rate`,
`spikes`. For the human eye the `center_surround` stage makes three classes
of cell at each position (luminance, red-green, blue-yellow); the cells'
positions and classes are in `pipeline.metadata["cells"]`.

The retina and cortex stages are sparse matrices. Up to 128 px for the human
eye, and always for mouse and fly, each is one matrix (`SparseStage`). Above
that a wide cell pools from a coarse layer that has summarized the cells under
it (`stages/pyramid.py`), and the stage holds the factors and applies them in
turn (`FactoredStage`: a sum of products of sparse matrices, with the
transpose taken factor by factor).

## Rules

1. `stages/` holds mathematics and never imports from `species/`.
2. `species/` holds parameters, citations and assembly, and no mathematics.
3. `core/` imports from neither.
4. Every linear stage's `adjoint` is the exact transpose of `forward`. The
   parametrized adjoint tests guard this. Never loosen their tolerance.
5. All randomness goes through an explicit `numpy.random.Generator`.
6. Pipelines and the decoder use `(channels, size, size)`. Everything a user or
   a figure sees is `(size, size, 3)` in `[0, 1]`.
7. Stage parameters are in pixels. Species parameters are in degrees and are
   converted with `VisualField`.
8. Figures use `matplotlib.figure.Figure`, never `pyplot`.
9. Runtime dependencies are NumPy, SciPy, Matplotlib, Pillow and scikit-image.
   FastAPI and python-multipart are also core, because the hosted app installs
   only core dependencies; uvicorn is an optional extra (`server`) for running
   locally. Do not add others.
10. Follow test-driven development: write the failing test, see it fail,
    implement, see it pass, commit.
11. Keep it simple. Do not add features, options or abstractions the plan does
    not ask for.

## Engineering guidelines

These apply to every change. They combine Andrej Karpathy's agent guidelines
(github.com/forrestchang/andrej-karpathy-skills), Matt Pocock's agent skills
(github.com/mattpocock/skills), and the SRP, KISS and DRY principles.

### How to work

1. **Think before coding.** State assumptions. If a request has two readings,
   say so instead of picking one silently. If a simpler approach exists, say so.
2. **Simplicity first.** The minimum code that solves the problem. No feature,
   option or abstraction that was not asked for. If 200 lines could be 50,
   rewrite it.
3. **Surgical changes.** Touch only what the task needs. Match the existing
   style. Remove only the orphans your own change created; mention other dead
   code, do not delete it.
4. **Goal-driven.** Turn the task into checks that can be run (a test, a
   measurement, a screenshot) and loop until they pass.
5. **Small steps, red then green.** Write the failing test, watch it fail,
   make it pass, then tidy. "The rate of feedback is your speed limit."
6. **Use the domain's words.** Receptor, mosaic, retinal class, cortex cell,
   stage, pipeline, spike window. Code, tests and docs share one vocabulary.

### How to design

- **Deep modules.** A module offers a small interface over a lot of behaviour.
  `run()` is the model: one call, one result. Do not widen an interface to
  expose something only one caller needs.
- **Single responsibility.** One module, one reason to change. Mathematics in
  `stages/`, biology in `species/`, orchestration in `run.py`, presentation in
  `report/`, `server.py` and `web/`.
- **Do not repeat yourself.** Each fact lives in one place: a constant, a
  parameter on a frozen dataclass, or one function others call.
- **Classes where there is state with behaviour, or more than one
  implementation of one interface.** Otherwise a function. The patterns in use,
  to follow and extend:

  | Pattern | Where | Use it when |
  |---|---|---|
  | Strategy | `LinearStage`, `PointwiseStage` and their subclasses | Adding a computation to an eye: a new stage class, or a factory that returns one |
  | Composite | `Pipeline` holds stages and is used as one operator | Combining stages |
  | Registry | `core/registry.py`, `species` | Choosing an implementation by name |
  | Factory | `assemble()`, each species' `build()`, `gabor_bank()`, `opponent_retina()` | Building a configured object from parameters |
  | Value object | Frozen dataclasses: `VisualField`, `EyeParams`, `RetinaClass`, `Settings`, `RunResult`, `Progress` | Passing configuration or results; never mutate, use `dataclasses.replace` |
  | Facade | `run()` | One entry point over many parts |
  | Observer | `on_progress`, `on_iteration` callbacks | Reporting progress without the core knowing who listens |

- **A pattern must earn its place.** Introduce one when it removes real
  duplication or isolates a responsibility that already exists twice. A
  pattern added for a single use is complexity, not design.
- **Scale by composition.** A new eye is parameters plus existing stages. A new
  computation is one stage. Neither should require editing the decoder, the
  server or the front end.

## Decisions that were measured, not guessed

Each of these came out of the prototype. Do not undo one without re-measuring.

- **Threshold-linear firing rate.** A saturating curve cannot be inverted on
  noisy spike counts without bias, and its tangent approximation cost 14 dB
  even without noise.
- **Gradient prior.** `-laplacian` (the 1/f^2 natural-image prior). The squared
  Laplacian let low-frequency noise through.
- **Chroma prior.** Mouse and fly have no red-sensitive receptor. The prior
  that colour channels are correlated fills the missing channel.
- **Cells only pool their own receptor type.** This keeps colour through the
  retina and cortex stages.
- **Non-oriented cells at the coarsest cortical scale.** Oriented Gabors barely
  respond to the mean level.
- **Receptors smaller than a pixel are pooled.** Such a position carries every
  receptor type.
- **The spike budget is fixed in degrees, never in pixels.** A model cell
  fires for the real cells it stands for (`cells_per_neuron` in
  `species/eye.py`). A cortex cell stands for the real receptors in a patch of
  field 0.625 degrees across (`spike_patch_deg`; 82 for the human eye, 1 for
  the mouse), the same at every picture size. A retinal output cell (an eye
  without a cortex) stands for the real receptors at its position, whose total
  does not change with the picture. Before, a human cortex cell's rate fell
  fourfold each time the picture's side doubled. 0.625 degrees is one pixel of
  the 96 px picture the human gains were chosen at: a calibration.
- **Human cortex scales at 0.2, 0.8, 1.6 and 3.2 cycles/degree, gains 1, 2, 2,
  2; the 3.2 scale has luminance cells only.** A scale whose wavelength is
  under two pixels is left out, so 96 and 128 px use the first two. Measured at
  256 px: gain 3 at 1.6 clips cells and costs over 5 dB with ideal neurons;
  luminance only at 1.6 leaves ideal neurons at 36.3 dB against 43.6 with
  colour; a 0.4 scale adds 0.2 dB. The 3.2 scale's gain was not measured.
- **The solve is preconditioned by the channel coupling plus the exact
  prior** (`uncoupling` in `core/decoder.py`), with tolerance 3e-5. A diagonal
  preconditioner did nothing; a circulant fit to the whole operator failed to
  converge wherever cells sample on a grid coarser than the pixels. At 1e-4
  the preconditioned solve stopped 0.8 dB short of the old one with ideal
  neurons; at 3e-5 it is closer to the exact solution than the old one was, in
  fewer steps.
- **Regularization follows the noise.** `lam = 1e-4 + 10 * noise_variance`.
- **Cone opponency with a gain per channel (human).** Retinal cells combine
  the cones into luminance, red-green and blue-yellow, with gains 1.5, 8 and
  3, and the fine cortex scale has gain 2. Sending L and M separately made
  red versus green a difference of two noisy signals: 20 dB with real
  neurons against 31 dB now. The gains were chosen under two guards: under
  0.1% of cells clipped, and no loss with ideal neurons. See
  `docs/superpowers/plans/2026-10-04-human-vision-audit.md`, which also holds
  the roadmap of what is still missing.
- **Multi-scale pooling above five million connections.** A pool (one group of
  cells reading another) with more than `DIRECT_LIMIT` = 5 million neighbour
  pairs reads a coarse layer in place of every cell: a grid of pooling cells,
  each the Gaussian mean (sigma half the grid spacing) of its own type. Cortex
  cells read a grid 0.7 envelope sigmas apart (40 inputs per cell at any image
  size); the retinal surround is pooled to a grid and spread back. A coarse
  layer is used only if it has at most half the cells it summarizes. Smaller
  pools are built as before, so every result at 128 px and below, and mouse
  and fly at any size, is unchanged. Measured: human at 256 px went from
  5.5 GB and 206 s to 0.54 GB and under 60 s; 512 px, impossible before, takes
  1.15 GB. A grid 0.5 sigmas apart leaves the fine scale direct at 256 px: it
  is 1.4 dB better with ideal neurons there but takes 96 s a run. See the
  phase 2 entry in the audit document.
- **Own conjugate-gradient loop.** It records the residual at each iteration at
  no cost and does not depend on SciPy's changing `cg` arguments.

## Expected results

At 128 px across 60 degrees, astronaut sample:

| Species | Neurons | PSNR, no noise | PSNR, 100 ms spikes |
|---|---|---|---|
| Human | 369,900 | about 35 dB | about 29.5 dB |
| Mouse | 9,864 | about 15 dB | about 12 dB |
| Fly | 504 | about 13 dB | about 13 dB |

PSNR for mouse and fly is dominated by the missing red channel. The human
decode takes about 10 seconds at 128 px; mouse and fly take about 1 second.

Human eye at larger sizes, mean of the three samples (`scripts/benchmark.py`):

| Size | PSNR, no noise | PSNR, 100 ms spikes | Time per run | Peak memory |
|---|---|---|---|---|
| 96 px | 40.1 dB | 30.6 dB | 7 s real, 9 s ideal | 0.31 GB (before phase 2b) |
| 128 px | 40.7 dB | 30.0 dB | 13 s real, 20 s ideal | 0.67 GB (before phase 2b) |
| 256 px | 43.6 dB | 31.2 dB | 58 s real, 101 s ideal | 1.29 GB |
| 512 px | not measured | not measured | not measured | 3.3 GB to build, 2.3 GB to solve |

The human eye has 245,484 neurons at 96 px, 369,900 at 128, 1,752,300 at 256
and 3,595,500 at 512. 512 px is not finished: the eye builds, but a solve
stopped at the 1000-step limit and the benchmark was cut short by the
machine's memory. See phase 2b in the audit document.

## Adding a species

Add `src/biovision/species/<name>.py` with `PARAMS`, `DESCRIPTION`, `CITATIONS`
and a `@species.register("<name>")` build function, and import it in
`src/biovision/species/__init__.py`. Update the species lists in the tests.
