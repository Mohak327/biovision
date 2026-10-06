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

### The web app

```
cd web
npm install
npm run build        # once, and again after changing the front end
cd ..
biovision serve      # then open http://127.0.0.1:8000
```

Pick a picture and an eye. The picture is rebuilt live from the eye's spikes;
drag the divider to compare it with the original. While working on the front
end, run `biovision serve` and `npm run dev` (in `web/`) side by side and open
http://localhost:5173.

### Hosting

`vercel.json` deploys the app to Vercel as two services in one project: the
front end in `web/` and the Python server, which answers everything under
`/api`. Deploy with `vercel deploy --prod`. A hosted function has less memory
than a laptop, so the human eye is limited to small picture sizes there.

### The command line

```
biovision list
biovision run --species mouse --image cat.jpg --out mouse.png
biovision report --species all --out results/
```

`run` writes one figure and prints the quality numbers. `report` writes every
figure (PNG and PDF), every table (CSV), `results.json` and `report.md`. Add
`--no-sweeps` to skip the slow parameter sweeps (spike window, regularization
and receptor density). Add `--density 4` or `--neuron-density 4` to give the
eye more receptors or cortex cells than the real animal. Add `--looks 4` to
let the eye look four times within the same spike window, moved a little each
time as a real eye is (fixational eye movements); the looks are combined in
one reconstruction. At 128 px this adds about 4 dB for the human eye with real
neurons, and detail for the mouse and fly once spike noise is low, and it
takes about as many times longer per solver step as there are looks. Add
`--photons 1e5` to set the light level: the photons a receptor catches each
second where the picture is white. Light arrives as photons, so in dim light
the receptors' own signal is noisy. For a human cone, sunlight is about 1e7,
a lit room 1e5 and dusk 1e4; the human reconstruction loses 0.1 dB in
sunlight, about 5 dB in a lit room and 11 dB at dusk. Without the option the
light is unlimited. With no `--image`, a bundled sample is used.

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
2. `optics`: blur by the eye's point-spread function. The human eye cannot
   focus every wavelength at once, so its S cones see a more blurred picture
   than its L and M cones (chromatic aberration).
3. `mosaic`: sampling at the receptor positions (foveated, square or hexagonal).
4. `center_surround`: difference-of-Gaussians receptive fields. In the human
   eye these cells combine the cone types into brightness, red-green and
   blue-yellow channels, each with its own gain.
5. `gabor` (mammals only): V1 simple cells at several scales and orientations.
6. `rate`: a threshold-linear firing rate. For the human and the mouse each
   signal has an ON cell, which fires for increments, and an OFF cell, which
   fires for decrements, both nearly silent at rest; the reconstruction uses
   their difference. The fly has one cell around a resting rate.
7. `spikes`: Poisson spike counts in a time window. An eye's parameters can
   make the counts more regular than Poisson (`EyeParams.fano`); no species
   does by default.

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
  server.py  the web app's HTTP server (FastAPI)
web/         the React front end (Vite, TypeScript, three.js)
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
- Image values are used as stored (sRGB-encoded), not converted to light. That
  encoding is itself compressive, and it stands in for the receptors' own
  compression (Weber's law); a separate cone compression was measured and not
  added (phase 5 in the audit document).
- Where receptors are smaller than a pixel, the image sets the resolution, not
  the eye. One model cell then stands for all the real cells in that pixel.
- The mouse model is its cone pathway in daylight.
- No rods. Photon noise (`--photons`) is for cones only, so the model's dusk
  is darker for it than for a real eye, which has switched to rods by then.
- The picture is three numbers per pixel, not a spectrum. Chromatic
  aberration uses one focus per cone type, and real spectra would need
  hyperspectral pictures (phase 12 in the audit document).
- The human eye is calibrated across 60 degrees. Across a degree or two the
  mosaic's detail shows (single cones, no S cones at the centre of gaze), but
  about 2% of retinal cells are driven to zero and quality is low.
- The lens and macular pigment, the jitter of real cone positions, and how the
  pupil opens in dim light have parameters or notes but no values: none could
  be supported from a source.
- A cell has no top firing rate: an ON or OFF cell can be driven past the
  range its gain was chosen for and loses nothing by it. A real cell saturates.
- The spikes that are counted are the cortex cells' for the human and the
  mouse. The retina's own cell types are not separate cells with their own
  spikes: a parasol class exists (`human.PARASOL`) and is off, because across
  60 degrees its larger field is smaller than a pixel; spike counts are
  Poisson, because the published regularity is for retinal cells; and noise
  shared between neighbouring cells is not modelled (measured: 0.2 dB or less).
  See phases 6 and 8 in the audit document.
- Still images only; no motion pathways.

## Tests

```
pytest                # the library and the server
cd web && npm test    # the front end's logic
```

## Credits

Sample images come from scikit-image: the astronaut photograph (NASA, public
domain), the cat (Stefan van der Walt, CC0) and the coffee cup (Rachel
Michetti, CC0).
