# Human vision audit and plan

2026-10-04. An audit of what the human model computes against what the human
eye and early cortex compute, the additions worth making, their measured
gains, and the plan to build them. Phases 1 and 2 are implemented; see "Phase 1
result" and "Phase results" below. The later phases are not.

## The problem

With real (noisy) neurons the human reconstruction is poor: 18 dB at 128
pixels, with pink and green speckle. With ideal neurons the same eye reaches
38 dB. So the limit is how the model carries its signal through spike noise.

## Audit: what the model does today

| Stage | Today | In the real eye |
|---|---|---|
| Receptor colours | RGB to L, M, S cones | The same |
| Optics | Gaussian blur | The same, to first order |
| Mosaic | Cones at 60:30:10; at these sizes every type at every pixel | The same |
| Retina | Each cell pools one cone type: centre minus surround | Cells combine cone types into luminance, red-green and blue-yellow channels |
| Cortex | Edge detectors at two scales, per cone type | Edge detectors on the luminance and colour channels |
| Firing rate | One fixed gain for every cell | Each pathway adapts its gain to fill its firing range |
| Spikes | Poisson counts | The same |
| Decoder | Smoothness and colour-correlation priors | (no counterpart; this is our readout) |

## What is missing, and why it matters

**1. Cone opponency.** L and M cones respond to nearly the same light. Today
the model sends L and M separately, so red versus green has to be recovered
by subtracting two nearly equal noisy signals, which magnifies the noise. The
retina instead sends three combined channels (Derrington, Krauskopf & Lennie
1984; Dacey 2000):

- luminance: L + M
- red-green: L - M
- blue-yellow: S - (L + M) / 2

These channels are nearly uncorrelated for natural images (Buchsbaum &
Gottschalk 1983), which is why the eye uses them.

**2. Gain control per pathway.** The red-green signal is small. The retina
amplifies each channel so it fills the cell's firing range (Laughlin 1981;
Atick & Redlich 1992). Today one gain (2.5) is applied to everything, so the
colour channels and the fine-detail cortex cells use a sliver of their range
and drown in spike noise. Opponency without gain does nothing (see below);
the two belong together.

**3. Gain per cortex scale.** Natural images have less contrast at fine
scales (amplitude falls as 1/frequency; Field 1987). Fine-scale cortex cells
therefore respond weakly unless their gain is higher. This is the same
principle as item 2, applied across scale.

## Measured gains

Prototyped on a scratch copy and measured on all three sample images
(astronaut, cat, coffee), 60 degrees, real neurons, 100 ms. "Clipped" is the
share of cells driven to zero firing, which loses information.

### 96 pixels (standard)

| Configuration | PSNR | SSIM | Clipped | Ideal-neuron PSNR |
|---|---|---|---|---|
| Today | 20.4 dB | 0.68 | 0.00% | 38.0 dB |
| Opponent channels, no extra gain | 20.6 dB | 0.64 | 0.00% | |
| 1 + 2: opponent, gains 1 / 8 / 3 | 26.2 dB | 0.79 | 0.00% | 40.6 dB |
| 1 + 2 + 3: also fine cortex gain 2 | 29.4 dB | 0.88 | 0.00% | 38.8 dB |
| Fine cortex gain 3 (too much) | 30.9 dB | 0.91 | 0.04% | 33.4 dB |
| Luminance gain 2 in place of 1 (with 1 + 2) | 29.0 dB | 0.86 | 0.00% | |

### 128 pixels (fine): the configuration in the bug report

| Configuration | PSNR | SSIM | Ideal-neuron PSNR |
|---|---|---|---|
| Today | 18.2 dB | 0.59 | |
| 1 + 2 | 23.9 dB | 0.67 | |
| 1 + 2 + 3 | 27.0 dB | 0.79 | 39.5 dB |

**Expected result of this plan: about +9 dB with real neurons (20.4 to 29.4
at 96 pixels, 18.2 to 27.0 at 128), SSIM from about 0.6-0.7 to 0.8-0.9, and
no loss with ideal neurons.** The pink and green speckle goes because colour
is no longer recovered by subtraction.

The gains are fixed numbers chosen from natural-image statistics, in the same
spirit as the existing contrast gain. Nothing is fitted per image.

## Considered and not included

| Idea | Measured | Decision |
|---|---|---|
| Decoder smooths colour more than brightness | Alone: +1.8 dB, SSIM +0.15, but colours wash out. On top of 1-3: +0.4 dB. It also slows the solver for mouse and fly (the mouse run stops converging). | Leave out. Items 1-3 fix the cause; this treats the symptom. |
| Weight each neuron by its Poisson variance | -0.4 dB alone; no change on top of 1-3. | Leave out. |
| Colour only at the coarse cortex scale | Cuts neurons 2.5x (245,484 to 98,028), but costs 2 dB with noise and 8 dB with ideal neurons. | Leave out for now. It is true to biology and worth offering later as an option when speed matters. |
| Fine cortex gain of 3 | +1.5 dB with noise but 5 dB worse with ideal neurons, because cells clip. | Use 2. |
| Compressive cone response (Weber's law) | Not measured. It would put a nonlinearity before the linear stages, which the architecture forbids, or need a pre/post transform. | Defer. |
| Rods, eye movements, temporal integration | Out of scope for still images in daylight. "Stare" already lengthens the spike window. | Out of scope. |
| Opponency for mouse and fly | Not measured. Both have colour-opponent cells in reality. | Separate follow-up, after the human result is confirmed. |

## Design

The rule that keeps the decoder exact still holds: every new computation is a
linear stage with an exact transpose, or a fixed gain.

### 1. A retina stage that mixes receptor types (`stages/receptive.py`)

`center_surround` pools one receptor type per cell. It becomes a function that
takes a list of cell classes, each with:

- weights over the receptor types (for luminance: L +0.5, M +0.5, S 0),
- a gain,
- a surround weight (luminance keeps its surround; the colour classes have
  none, which makes them low-pass in space, as measured by Mullen 1985).

It returns the stage and the mosaic of the cells it created (positions and
class), which the cortex stage then pools from.

`stages/sparse.py`'s `pool` gains an optional table of weights between output
class and input type, replacing the built-in "same type only" rule. With the
identity table it behaves exactly as today.

### 2. Parameters (`species/eye.py`, `species/human.py`)

`EyeParams` gains:

- `retina_classes`: names, receptor weights, gains and surround weights. The
  default is one class per receptor type, weight 1, gain 1: today's behaviour.
- `cortex_gains`: one gain per cortical frequency, default all 1.

`human.py` sets three classes (luminance, red-green, blue-yellow) with gains
from the measurements, and `cortex_gains = (1, 2)`. Mouse and fly are not
changed and must produce the same numbers as before.

### 3. Cortex gain (`stages/gabor.py`)

`gabor_bank` takes a gain per wavelength and scales that wavelength's cells.

### 4. Choosing the final gains

Start from the measured values (luminance 1, red-green 8, blue-yellow 3, fine
cortex 2) and also try luminance 2 with the others. Keep the best setting
that satisfies both guards on the three samples:

- under 0.1% of cells clipped,
- ideal-neuron PSNR within 1 dB of today's 38.0 dB.

Record the chosen values and the measurements in `human.py` and `AGENTS.md`.

### 5. What else has to follow

- **Server:** the retina view lights receptors by the retinal cells' output,
  matched by position. With retinal cells no longer one per receptor, it
  switches to the receptors' own signal (the `mosaic` stage output).
- **Report figures:** the pipeline panel draws the retina stage's output at
  receptor positions; it switches to the retinal cells' positions. The filter
  gallery gains the three class profiles.
- **Web app:** the "retina" tile text changes to say cells combine cone types
  into brightness and two colour channels. No layout change.
- **Docs:** README limits, `AGENTS.md` decisions and expected results.

## Tests (written first)

1. Mouse and fly reconstructions are unchanged: PSNR on the astronaut sample
   recorded before the change and compared after.
2. `pool` with the identity table equals today's `pool`; with a mixing table
   it connects the stated types with the stated weights.
3. The new retina stage's transpose is exact (the existing adjoint test, now
   run on the human pipeline with mixing).
4. A uniform grey image gives zero response in both colour classes and
   `gain x (1 - surround)` in luminance.
5. A pure red-green difference with equal luminance excites the red-green
   class and not luminance.
6. Human, real neurons, 100 ms, 96 pixels, astronaut: PSNR at least 26 dB
   (today 20.1). Ideal neurons: at least 36 dB.
7. Under 0.1% of human cells clip on each sample.
8. Narrow field of view (receptors larger than a pixel, one type per
   position) still runs and reconstructs.
9. Server and report tests still pass; the retina responses match the
   receptor count.

## Order of work

1. Record mouse and fly baselines as tests.
2. `pool` with a mixing table.
3. The mixing retina stage, returning its cell mosaic.
4. Cortex gains.
5. `EyeParams`, `assemble` and `human.py`; choose the final gains.
6. Server, figures, web text.
7. Docs; full suites; screenshots of the web app on the bug-report settings.

## Sources

- Atick JJ, Redlich AN (1992). What does the retina know about natural
  scenes? Neural Comput 4:196-210.
- Buchsbaum G, Gottschalk A (1983). Trichromacy, opponent colours coding and
  optimum colour information transmission in the retina. Proc R Soc Lond B
  220:89-113.
- Dacey DM (2000). Parallel pathways for spectral coding in primate retina.
  Annu Rev Neurosci 23:743-775.
- Derrington AM, Krauskopf J, Lennie P (1984). Chromatic mechanisms in
  lateral geniculate nucleus of macaque. J Physiol 357:241-265.
- Field DJ (1987). Relations between the statistics of natural images and the
  response properties of cortical cells. J Opt Soc Am A 4:2379-2394.
- Laughlin SB (1981). A simple coding procedure enhances a neuron's
  information capacity. Z Naturforsch C 36:910-912.
- Mullen KT (1985). The contrast sensitivity of human colour vision to
  red-green and blue-yellow chromatic gratings. J Physiol 359:381-400.

## Phase 1 result

Built as planned, with these differences from the plan:

- **The colour classes keep a surround** (weight 0.7). The plan gave them
  none. Measured without it: 28.6 dB real and 32.9 dB ideal at gains 8 / 3,
  with 0.25% of cells clipped, which fails both guards.
- **`pool` was not changed.** The mixing retina stage builds one normalized
  pool per receptor type and adds them with the class weights, so no mixing
  table was needed.
- **The solver's step limit rose from 500 to 1000.** With the new gains the
  ideal-neuron solve on the astronaut needs 671 steps.

Chosen gains: luminance 1.5, red-green 8, blue-yellow 3, fine cortex 2.
Candidates measured on the three samples at 96 pixels:

| Luminance | Fine cortex | Real neurons | SSIM | Ideal neurons | Worst clipping |
|---|---|---|---|---|---|
| 1 | 2 | 29.3 dB | 0.87 | 38.9 dB | 0.006% |
| **1.5** | **2** | **30.7 dB** | **0.91** | **38.4 dB** | **0.055%** |
| 2 | 2 | 30.8 dB | 0.92 | 36.1 dB | 0.098% |
| 2 | 1.5 | 30.4 dB | 0.90 | 39.8 dB | 0.055% |
| 1.5 | 1.5 | 29.6 dB | 0.88 | 40.7 dB | 0.012% |

Result against the starting point (real neurons, 100 ms):

| | Before | After |
|---|---|---|
| 96 pixels, mean of three samples | 20.4 dB, SSIM 0.68 | 30.7 dB, SSIM 0.91 |
| 96 pixels, astronaut | 20.1 dB | 29.9 dB, SSIM 0.92 |
| 128 pixels, astronaut (the bug report) | 17.5 to 18.2 dB | 27.9 dB, SSIM 0.85 |
| Ideal neurons, 96 pixels, mean of three samples | 38.0 dB | 38.4 dB |
| Ideal neurons, 96 pixels, astronaut | 35.1 dB | 34.9 dB |
| Mouse and fly | | unchanged, to nine decimal places |

## Roadmap: every missing piece, by effect on reconstruction accuracy

Ordered by the size of the expected effect on how closely the rebuilt picture
matches the original, largest first. "Measured" numbers come from prototypes;
the rest are estimates to be measured before each phase is kept. Some pieces
make the model more faithful to the eye while lowering accuracy, because the
real eye loses that information; they are marked.

| Phase | Piece | Effect on accuracy | Basis | Keeps the exact inverse? |
|---|---|---|---|---|
| 1 | Cone opponency, gain per pathway, gain per cortex scale | +9 dB with real neurons | Measured | Yes |
| 2 | Multi-scale pooling, so the eye runs at 256 and 512 pixels: coarse cells pool from cells that already summarize a patch, as in the real retina and cortex, instead of from every receptor | Removes the largest remaining loss: today the picture is shrunk to 128 pixels before the eye sees it | Memory measured (5.5 GB at 256 today) | Yes |
| 3 | Fixational eye movements and looking time: several shifted looks combined | About +3 dB per doubling of looks; "stare" already gives +5.5 dB over "look" | Measured for looking time; estimate for shifts | Yes |
| 4 | Divisive normalization (contrast gain control in retina and cortex) | Estimated +2 to +4 dB on low-contrast pictures | Estimate | No: needs an iterative decoder |
| 5 | Cone light adaptation (Weber's law) | Estimated +1 to +3 dB in dark regions | Estimate | Yes, as an invertible step at the input |
| 6 | Real cell types: separate ON and OFF cells, midget and parasol classes, more cortex scales | Estimated +1 to +2 dB | Estimate | Mostly; rectified ON/OFF pairs need care |
| 7 | Foveation at high resolution: receptive fields growing with eccentricity | Lowers accuracy in the periphery, by design. Faithful. | Needs phase 2 | Yes |
| 8 | Realistic spikes: refractory period, bursts, correlated neighbours | Small either way: about +1 dB from more regular firing, about -1 dB from correlations | Estimate | The decoder stays linear; the noise model changes |
| 9 | Real optics: chromatic aberration, pupil size, lens and macular pigment | Lowers accuracy slightly, mostly in blue. Faithful. | Estimate | Yes |
| 10 | Photon noise at the receptors | Lowers accuracy, strongly only in dim light. Faithful. | Estimate | Yes |
| 11 | Cortex beyond simple cells: complex cells, depth, V2 and V4 | Lowers accuracy: these cells discard position detail. Faithful. | Estimate | No |
| 12 | Real spectra in place of RGB | Changes colour accuracy slightly; needs hyperspectral pictures | Estimate | Yes |
| 13 | The real mosaic: irregular positions, no S cones at the centre, personal L:M ratio | Negligible at these picture sizes | Estimate | Yes |
| 14 | Rods and night vision | None in daylight | Out of scope until dim light is modelled | Yes |

Phases 4 and 11 break the rule that every stage is linear or exactly
invertible. They need a decision of their own before they are built: the
decoder would become iterative, and "rebuilt with linear algebra" would no
longer be the whole story.

An exact copy of human vision is not reachable: the full circuitry is not
known, and the brain does not rebuild a picture from its spikes. The aim of
this roadmap is that each stage is as faithful as published measurements
allow.

## How the roadmap is being built

- **Phase 2 is multi-scale pooling, not FFT.** FFT is one way to compute
  "every cell applies the same filter". The eye saves work differently:
  coarse cells pool from cells that have already summarized a patch (a
  pyramid), and sampling is fine at the centre and coarse elsewhere
  (foveation, phase 7). Both are linear with exact transposes. Phase 7
  follows phase 2 directly.
- **One phase at a time**, each by an agent, in this order: 2, 7, 3, 5, 6, 8,
  9, 10, 13, 12, 14, then 4 and 11 (the two that need an iterative decoder).
- **Every phase is measured before it is kept**, on one benchmark: the three
  samples, human eye, 128 pixels, 60 degrees, real neurons at 100 ms and
  ideal neurons (`scripts/benchmark.py`). Results are appended below.
- **Defaults.** A feature that lowers the 128-pixel real-neuron PSNR by more
  than 0.5 dB is built, tested and documented, but left off by default behind
  a parameter, so the default eye stays the best-reconstructing one.
- **Guards for every phase:** mouse and fly results unchanged unless the
  phase is about them; every linear stage keeps an exact transpose; the full
  test suite passes.

## Phase results

(Each phase appends its measurements here.)

### Phase 2: multi-scale pooling (2026-10-04)

**What was built.** A pool is one group of cells reading another through a
sparse matrix. When a pool has more than five million neighbour pairs
(`DIRECT_LIMIT` in `stages/pyramid.py`), its cells no longer connect to every
cell under them. They read a coarse layer: a square grid of pooling cells,
each the Gaussian-weighted mean (sigma half the grid spacing) of the cells of
its own type around it.

- **Cortex** (`gabor_bank`): each scale reads a grid 0.7 envelope sigmas
  apart (3.6 samples per wavelength), so a cell has about 40 inputs at any
  image size. Direct, the coarse scale has 357 inputs per cell at 128 px,
  1,430 at 256 and 5,720 at 512.
- **Retinal surround** (`center_surround`, `opponent_retina`): receptors are
  pooled onto a grid 0.89 surround sigmas apart and each cell takes a Gaussian
  mean of the grid; the two Gaussians' variances add up to the surround's.
  The class mixing (luminance, red-green, blue-yellow) became a factor of its
  own. The narrow centre stays direct.
- **`FactoredStage`** (`stages/sparse.py`): a linear stage that is a sum of
  products of sparse matrices. The factors are kept apart, since multiplying
  them out would give back the large matrix. The transpose is taken factor by
  factor and is exact; the generic adjoint tests run on pooled versions of
  all three stage builders.
- **`pool`** builds its matrix a slice of output cells at a time, so building
  no longer needs the neighbour lists of the whole matrix in memory. The
  matrix is the same, bit for bit.
- **`scripts/benchmark.py`**: the measurements below.

**The switch.** A pool stays direct if it has at most five million neighbour
pairs, or if the coarse layer would have more than half as many cells as the
layer it summarizes (such a grid saves nothing and only blurs). Both tests
are exact counts, so the choice is deterministic. Consequences: the human eye
at 128 px and below is built exactly as before; mouse and fly are direct at
every size, because their receptor count does not grow with the picture
(checked for the mouse at 64 and 512 px); at
192 px the retinal surround and the coarse cortex scale are pooled; at 256 px
and above everything wide is pooled.

**Measurements.** Human eye, 60 degrees, mean of astronaut, cat and coffee;
real neurons are 100 ms of spikes with seed 0. Peak memory is from
`tracemalloc`. Times are without memory tracing, which slows a run two to
four times; the first run also builds the eye. The machine was shared with
other jobs, and the same run varied by up to 40%.

| Size | | Real PSNR | Real SSIM | Ideal PSNR | Ideal SSIM | Time per run | Peak memory |
|---|---|---|---|---|---|---|---|
| 96 px | before | 30.73 dB | 0.907 | 39.75 dB | 0.996 | not timed untraced | 308 MB |
| 96 px | after | 30.73 dB | 0.907 | 39.73 dB | 0.996 | not timed untraced | 314 MB |
| 128 px | before | 28.53 dB | 0.830 | 40.53 dB | 0.995 | 11 s to build; runs not timed untraced | 663 MB |
| 128 px | after | 28.53 dB | 0.830 | 40.53 dB | 0.995 | 13 s to build, 16 s real, 38 s ideal | 673 MB |
| 256 px | before | not run | | not run | | 206 s | 5.5 GB |
| 256 px | after | 23.11 dB | 0.600 | 32.72 dB | 0.952 | 30 s real (43 s with the build), 50 s ideal (55 s at worst) | 537 MB |
| 512 px | before | impossible | | | | | |
| 512 px | after | 17.16 dB | 0.478 | 16.11 dB | 0.706 | 65 s to build, 266 s real, 410 s ideal (astronaut) | 1,147 MB |

The "before" figures at 256 px (and 96 s, 1.7 GB at 192 px) are the ones
recorded when this phase was planned; they were not re-run, because 5.5 GB
does not fit on the machine used. The 128 px times are for the astronaut.
The 0.02 dB change at 96 px with ideal neurons comes from the retina's class
mixing being added up in a different order.

Per sample at 128 px, before and after alike: real 27.9 / 28.9 / 28.8 dB,
ideal 35.1 / 46.3 / 40.2 dB. Mouse and fly are unchanged to nine decimal
places; the pinned test was not edited.

**Does pooling itself cost accuracy?** Forcing the coarse cortex scale
through the pooled path at 128 px (astronaut) gives 27.93 dB real and
35.08 dB ideal against 27.93 and 35.08 direct (27.92 and 35.08 with the
grid 0.7 sigmas apart). A
test holds the pooled human eye within 0.5 dB of the direct one at 96 px.

**Deviations from the brief.**

- **Grid 0.7 sigmas apart, not 0.5.** At 0.5 the fine cortex scale at 256 px
  has a grid as fine as the pixels, stays direct with 30 million connections,
  and the ideal-neuron runs take 96 s on average (113 s at worst), over the
  90 s guard. At 0.7 it is pooled (13.5 million) and they take 50 s. The
  price at 256 px is 0.24 dB with real neurons (23.35 to 23.11) and 1.4 dB
  with ideal neurons (34.11 to 32.72): a grid 1.5 pixels apart cannot carry
  the finest detail the direct cells picked up.
- **Two code paths, switched by size**, in place of one. Pooling everything
  would have changed mouse and fly; the limit keeps them, and the human eye
  up to 128 px, exactly as they were.
- **`gaussian` moved** from `stages/receptive.py` to `stages/sparse.py`, so
  the pooling module can use it without a circular import.
- **`--no-memory` on the benchmark**, because tracing memory inflates the
  times it is meant to report.

**What 256 and 512 px show.** They run, but the picture is worse, not better:
28.5 dB at 128 px, 23.1 at 256, 17.2 at 512 with real neurons. Three causes,
none of them the pooling:

1. **The firing rate falls with picture size.** A model cell's rate is
   multiplied by the cones per pixel (47 at 128 px, 11.7 at 256), and that
   factor is applied to the cortex cells too, whose number does not change
   with size. So the whole code carries 4 times fewer spikes at 256 px and 16
   times fewer at 512. Giving the 256 px eye the rate it has at 128 px raises
   the astronaut from 22.4 to 25.4 dB (measured on a scratch copy, not
   built). A cortex cell should stand for a fixed number of real cells per
   square degree, whatever the pixel size.
2. **No cortex cell is tuned finer than 0.8 cycles per degree**, a wavelength
   of 5 pixels at 256 px and 11 at 512. Detail finer than that is not sent.
   Phase 6 (more cortex scales) and phase 7 (foveation) address this.
3. **The solver's 1000-step limit.** With ideal neurons at 512 px the
   astronaut does not converge in 1000 steps (the other two were not
   checked), and the estimate it stops at is poor (16.1 dB on average,
   below the real-neuron figure). The system gets harder to solve as the
   picture grows.

**Not done.** The nine cortex pools of one scale each search for the same
neighbours; sharing that search would cut the build time (65 s at 512 px)
several times over. 192 px was built and inspected but not benchmarked.


### Phase 2b: quality at 256 px and above (2026-10-05)

The three causes listed at the end of phase 2, each addressed and measured.
Foveation (phase 7) is not part of this step and was not built: across 60
degrees, up to 512 px, the picture's own pixel grid is coarser than the retina
everywhere, so receptive fields growing with eccentricity would change nothing
here.

**What was built.**

- **A spike budget fixed in degrees** (`cells_per_neuron` in `species/eye.py`,
  which holds the rule in its docstring). A cortex cell fires for the real
  receptors in a patch of field `spike_patch_deg` across (0.625 degrees, at
  least one receptor). The patch and the grid of cortex cells are both fixed in
  degrees, so a cell's rate is the same at every picture size: 82.2 to 82.8
  real cells per human cortex cell at 48, 64, 96, 128, 256 and 512 px (it was
  84.0 at 96 px, 47.0 at 128, 11.7 at 256, 3.2 at 512). 0.625 degrees is one pixel of the 96 px,
  60 degree picture at which the phase 1 gains were chosen; it is a
  calibration to that behaviour, not a published number. An eye without a
  cortex keeps the old rule, which was already independent of the picture:
  retinal cells sit at the model's positions, and positions times real
  receptors each is the eye's receptor count.
- **Human cortex scales at 0.2, 0.8, 1.6 and 3.2 cycles/degree**, gains 1, 2,
  2, 2. The existing rule drops a wavelength under two pixels, so across 60
  degrees the picture uses two scales up to 191 px, three from 192 px, four
  from 384 px. The 3.2 scale has luminance cells only (Mullen 1985); `gabor_bank`
  takes the cell types of each scale (`types`), `EyeParams` holds them
  (`cortex_types`).
- **A preconditioned solve** (`core/decoder.py`). `conjugate_gradient` takes an
  optional preconditioner. The decoder's (`uncoupling`) is the exact inverse of
  a simpler operator: the prior, kept whole, plus the data term's coupling of
  the colour channels (a 3 x 3 matrix, averaged over 16 probed pixels, three
  applications of the operator), inverted frequency by frequency. The system is
  the same and the solve is still linear; tests hold it to the unpreconditioned
  solve and to the dense oracle.
- **`scripts/benchmark.py`** also prints the most solver steps (inf if a run
  did not converge) and the largest share of clipped cells.

**Measurements.** Human eye, 60 degrees, mean of the three samples, real
neurons 100 ms with seed 0. Times are without memory tracing.

| Size | | Real PSNR | Real SSIM | Ideal PSNR | Ideal SSIM | Neurons | Time per run | Peak memory | Most clipped |
|---|---|---|---|---|---|---|---|---|---|
| 96 px | before | 30.73 dB | 0.907 | 39.73 dB | 0.996 | 245,484 | not timed | 314 MB | 0.055% |
| 96 px | after | 30.63 dB | 0.906 | 40.11 dB | 0.996 | 245,484 | 7 s real, 9 s ideal | not re-measured | 0.055% |
| 128 px | before | 28.53 dB | 0.830 | 40.53 dB | 0.995 | 369,900 | 16 s real, 38 s ideal | 673 MB | not measured |
| 128 px | after | 30.00 dB | 0.866 | 40.74 dB | 0.995 | 369,900 | 13 s real, 20 s ideal | not re-measured | 0.029% |
| 256 px | before | 23.11 dB | 0.600 | 32.72 dB | 0.952 | 369,900 | 30 s real, 50 s ideal | 537 MB | not measured |
| 256 px | after | 31.16 dB | 0.829 | 43.56 dB | 0.997 | 1,752,300 | about 58 s real, 101 s ideal (112 s at worst) | 1,291 MB | 0.017% |
| 512 px | before | 17.16 dB | 0.478 | 16.11 dB, not converged | 0.706 | 369,900 | 266 s real, 410 s ideal | 1,147 MB | not measured |
| 512 px | after | not measured | | not measured | | 3,595,500 | build 567 s with memory tracing | 3,327 MB building, 2,278 MB solving | not measured |

Per sample at 256 px after: real 30.8 / 31.4 / 31.3 dB, ideal 37.5 / 49.6 /
43.5 dB; at most 282 solver steps with real neurons and 553 with ideal ones.
The 96 px change with real neurons (-0.10 dB) is the patch holding 82.3 cells
where a pixel held 84.0. The 96 and 128 px memory is unchanged in kind (the
same matrices are built) but was not traced again.

**512 px is not finished.** The eye builds (3,595,500 neurons, 3.3 GB at the
peak of the build, 2.0 GB held afterwards). The benchmark was then started
and, before any result was printed, a solve stopped at the 1000-step limit
without converging; shortly afterwards the machine ran short of memory and the
run was stopped. So at 512 px there is no PSNR, the solver is known not to
converge within 1000 steps on at least one run, and which run that was (real
or ideal neurons) is not known. The gain of the 3.2 cycles/degree scale (2)
was carried over from the 1.6 scale and never measured at 512 px.

**The spike-budget rule, and a deviation.** The brief asked that the total
spike budget not depend on the pixel count. What was built makes each cell's
rate independent of the pixel count. The two agree for a fixed set of cells
(96 and 128 px), but a larger picture brings in finer scales, and their cells
fire as well: the model counts 4.7 times as many cells at 256 px as at 128,
all at the same rate. The reading behind this: the real cortex has those fine
cells at every picture size; on a small picture they are left out because
they would tell nothing, not because they are silent. The literal reading
(one fixed total shared out among however many cells are modelled) would give
each 256 px cell about a fifth of the rate. It was not measured.

**Scales and gains: candidates.** Astronaut, 256 px, measured on a scratch
copy with the budget at 84 cells and the old solver at tolerance 1e-4:

| Scales (cycles/degree) | Gains | Colour up to | Real | Ideal | Clipped | Decision |
|---|---|---|---|---|---|---|
| 0.2, 0.8 (before) | 1, 2 | 0.8 | not measured at this budget | 30.2 dB | | Baseline |
| 0.2, 0.8, 1.6 | 1, 2, 3 | 0.8 | 27.85 dB | 30.5 dB | 0.082% | Rejected: clipping |
| 0.2, 0.8, 1.6 | 1, 2, 2 | 0.8 | 28.47 dB | 33.19 dB | 0.036% | Rejected: colour too coarse |
| 0.2, 0.8, 1.6 | 1, 2, 1.5 | 0.8 | 28.35 dB | 33.88 dB | 0.013% | Rejected |
| 0.2, 0.8, 1.6 | 1, 2, 1 | 0.8 | 27.83 dB | 34.02 dB | 0.008% | Rejected |
| 0.2, 0.8, 1.6 | 1, 1.5, 2 | 0.8 | 28.06 dB | 33.19 dB | 0.029% | Rejected |
| 0.2, 0.8, 1.6 | 1, 3, 2 | 0.8 | 28.27 dB | 31.41 dB | 0.065% | Rejected |
| 0.2, 0.4, 0.8, 1.6 | 1, 1.4, 2, 3 | 0.8 | 28.02 dB | 30.63 dB | 0.074% | Rejected: the 0.4 scale adds 0.2 dB for 86,400 cells |
| 0.2, 0.8, 1.6 | 1, 2, 3 | 1.6 | 30.25 dB | 31.80 dB | 0.043% | Rejected: clipping |
| **0.2, 0.8, 1.6** | **1, 2, 2** | **1.6** | **30.87 dB** | **37.37 dB** | **0.017%** | **Chosen** |

With luminance only at 1.6 cycles/degree the three-sample means were 29.11 dB
real and 36.33 dB ideal (gains 1, 2, 2) and 28.85 and 36.63 (gains 1, 2, 1.5):
under the 37 dB guard, so colour is kept to 1.6 cycles/degree and dropped only
at 3.2. The scales are therefore not whole octaves from 0.2 to 0.8; the 0.4
scale was measured and left out.

**Solver: candidates.** Steps to the old tolerance of 1e-4:

| Preconditioner | Mouse 64 px, real / ideal | Fly 64 px, real / ideal | Human 96 px, real / ideal | Human 256 px (colour to 1.6), real / ideal |
|---|---|---|---|---|
| None | 86 / 308 | 33 / 90 | 304 / 680 | 375 / 709 |
| Diagonal (Jacobi) | | | 296 / 632 | |
| Circulant fitted to the whole operator | 10 / over 1000 | 9 / 341 | 174 / 210 | over 1000 with ideal neurons on two other 256 px configurations |
| **Channel coupling plus the exact prior** | **12 / 145** | **20 / 28** | **166 / 211** | **170 / 222** |

The diagonal does nothing because the operator's diagonal is nearly the same
at every pixel. The circulant fit fails where cells sample the picture on a
grid coarser than the pixels: the operator is then not the same at every
pixel, and the fit is confidently wrong about what is sensed.

**Tolerance 1e-4 became 3e-5.** The residual of a preconditioned solve is not
the same measure of error: at 1e-4 it stopped further from the exact solution
than before (human, 96 px, ideal: 38.93 dB against 39.73; the exact solution,
solved to 1e-9, gives 40.21). At 3e-5 it gives 40.12 dB in about 470 steps
against the old solve's 39.73 dB in about 650. So at equal accuracy the gain
for the human eye is about 1.4 times, far less than the step counts at 1e-4
suggest; for the mouse with real neurons it is about 7 times.

**Mouse and fly.** The spike-budget rule leaves them exactly as they were at
64 px (a mouse cortex cell stands for one real cell at every size; before, it
stood for 3.5 at 32 px). The solver moved the pinned values, and the test was
re-pinned with the old numbers in its docstring:

| | Before | After |
|---|---|---|
| Mouse, real | 11.999567473 | 11.999458055 |
| Mouse, ideal | 14.713525802 | 14.755449525 |
| Fly, real | 13.131761815 | 13.133059867 |
| Fly, ideal | 13.708582084 | 13.730313811 |

**Deviations from the brief.**

- The spike budget is per cell, not a fixed total (above).
- The preconditioner is not a diagonal one, and the tolerance changed (above).
- Scales are 0.2, 0.8, 1.6, 3.2, without 0.4.
- 512 px: built and measured for memory only; it does not meet the guard.

**Not done, not verified.** PSNR, time and clipping at 512 px. Convergence at
512 px, which failed at least once. The literal fixed-total budget. Sizes
between 128 and 256 px (at 192 px the 1.6 scale has a wavelength of exactly
two pixels). The web app's size hints still describe the old memory use; they
live in `web/`, which this step did not touch.
