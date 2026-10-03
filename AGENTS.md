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
its 134 tests passed. Implement the plan task by task, in order. Type the code
as written. If a test fails, look for a typo before changing the design.

## Status

Nothing is implemented yet. The repository holds the spec, the plan and this
file. Start at Task 1 of the plan.

## Commands

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"   # Windows
python -m pytest -q                               # whole suite, about one minute
python -m pytest tests/test_decoder.py -q         # one file
biovision list
biovision run --species mouse --out results/mouse.png
biovision report --species all --out results --no-sweeps
streamlit run app/streamlit_app.py
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
  `P = -laplacian + chroma_weight * chroma`.
- `run()` is the single entry point. The CLI, the app and the report layer
  call it and contain no mathematics.

Stage order for every species: `color`, `optics`, `mosaic`, `center_surround`,
`gabor` (mammals only, and only if a wavelength fits the image), `rate`,
`spikes`.

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
   Streamlit is an optional extra. Do not add others.
10. Follow test-driven development: write the failing test, see it fail,
    implement, see it pass, commit.
11. Keep it simple. Do not add features, options or abstractions the plan does
    not ask for.

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
  receptor type, and its firing rate is multiplied by the number of real cells
  it stands for.
- **Regularization follows the noise.** `lam = 1e-4 + 10 * noise_variance`.
- **Own conjugate-gradient loop.** It records the residual at each iteration at
  no cost and does not depend on SciPy's changing `cg` arguments.

## Expected results

At 128 px across 60 degrees, astronaut sample:

| Species | Neurons | PSNR, no noise | PSNR, 100 ms spikes |
|---|---|---|---|
| Human | 369,900 | about 35 dB | about 18 dB |
| Mouse | 9,864 | about 15 dB | about 12 dB |
| Fly | 504 | about 13 dB | about 13 dB |

PSNR for mouse and fly is dominated by the missing red channel. The human
decode takes about 10 seconds at 128 px; mouse and fly take about 1 second.

## Adding a species

Add `src/biovision/species/<name>.py` with `PARAMS`, `DESCRIPTION`, `CITATIONS`
and a `@species.register("<name>")` build function, and import it in
`src/biovision/species/__init__.py`. Update the species lists in the tests.
