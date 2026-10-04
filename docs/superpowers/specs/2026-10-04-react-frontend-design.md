# biovision web app: React front end over a Python server

Design record, 2026-10-04. Replaces the Streamlit app. The `biovision` library
and its command line are unchanged; all mathematics stays in Python.

## Why

The Streamlit app put abstract sliders in a sidebar and hid the pipeline
behind tabs. Streamlit cannot do a draggable before/after wipe, a canvas
animation, or controls placed along the pipeline. A hand-built front end can.

## Shape

```
web/ (React, Vite, TypeScript)  <-- NDJSON stream -->  biovision.server (FastAPI)  -->  biovision.run()
```

`run()` already reports each stage and each decoding iteration through
`on_progress`. The server forwards those events to the browser as they
happen; the browser draws them.

## Server (`src/biovision/server.py`, optional extra `server`)

| Endpoint | Returns |
|---|---|
| `GET /api/species` | name, description, receptor names, whether it has a cortex stage, citations |
| `GET /api/samples`, `GET /api/samples/{name}` | sample names; a sample as PNG |
| `POST /api/runs` | NDJSON stream: `mosaic`, `stage`, `frame` events, then `result` or `error` |
| `POST /api/compare` | NDJSON stream: one `result` per species |
| `POST /api/sweeps` | rows for the `window`, `lambda` or `density` sweep |
| `POST /api/report` | the full report as a zip |

Every POST takes a multipart form: `settings` (JSON) plus either `sample` (a
name) or `image` (an uploaded file). Images in events are base64 PNG. Frames
are sent at most every 50 ms; the last one is always sent. If the client
disconnects, the run is cancelled at its next progress callback.

If `web/dist` exists the server also serves the built app at `/`, so one
process runs everything. `biovision serve` starts it.

## Front end (`web/`)

One screen, top to bottom:

1. **Species picker.** Three cards, each drawn with that eye's mosaic pattern
   (foveated rings, square grid, hexagons).
2. **Eyepiece.** A circular viewport with a draggable divider between the
   original and the reconstruction. During a run the reconstruction forms
   live as crisp pixels, with the stage line beneath.
3. **Controls as named presets**, not raw sliders: glance / look / stare for
   the spike window; ideal / real neurons; half / real / 4x / 16x receptors;
   cortex density; viewing distance for field of view; draft / standard / fine
   for working size. Regularization and seed sit under "Advanced".
4. **Signal path.** One station per stage with its output and a plain-language
   line. The mosaic station is a 3D retina (React Three Fiber) whose receptors
   are coloured by type and lit by their response.
5. **Measurements.** Power spectrum, error per channel, spike histogram and
   solver convergence as SVG charts; sweeps and species comparison on demand;
   report download.

### Libraries

- React, TypeScript, Vite.
- Plain canvas for photographs and live frames.
- three, @react-three/fiber, @react-three/drei for the retina.
- motion for the eyepiece drag and stage transitions.
- d3-scale, d3-shape, d3-array for chart geometry; charts are hand-drawn SVG.
- Native radio inputs and buttons for controls (accessible without a kit).
- PixiJS was considered for the mosaic and dropped: three covers it.

### Visual identity

- Ground: pale cool grey, like a microscope slide. Ink: deep blue-black.
- Accents are the receptor colours and always mean a receptor type:
  S violet, M green, L amber, UV magenta.
- Photographs sit in dark wells.
- Type: Atkinson Hyperlegible for text (designed for low-vision readers);
  Bricolage Grotesque for headings.
- One piece of motion: the reconstruction forming. Reduced-motion is honoured.

## Testing

- Server: pytest with FastAPI's TestClient, covering every endpoint, the
  event order of a run stream, uploads, and the error paths.
- Front end: Vitest for the stream parser, presets and chart geometry;
  `tsc` and `vite build` must pass; screenshots from a headless browser are
  checked by eye.

## Out of scope

- Running Python in the browser (Pyodide). The library needs no change for
  it, so it can follow later.
- Hosting. The app runs locally with `biovision serve`.
