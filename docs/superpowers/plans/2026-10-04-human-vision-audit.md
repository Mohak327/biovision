# Human vision audit and plan

2026-10-04. An audit of what the human model computes against what the human
eye and early cortex compute, the additions worth making, their measured
gains, and the plan to build them. Nothing here is implemented yet.

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
