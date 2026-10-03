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

Add `--density 4` to give the eye four times as many receptors per unit area
(1 is the real animal).

`run` writes one figure and prints the quality numbers. `report` writes every
figure (PNG and PDF), every table (CSV), `results.json` and `report.md`. Add
`--no-sweeps` to skip the slow parameter sweeps (spike window, regularization
and receptor density). With no `--image`, a bundled
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
