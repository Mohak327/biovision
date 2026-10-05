"""Human: three cone types, a fovea, centre-surround retina, V1 simple cells."""
from ..core.field import VisualField
from ..core.pipeline import Pipeline
from ..core.registry import species
from ..stages.receptive import RetinaClass
from .eye import EyeParams, assemble

DESCRIPTION = (
    "Humans have three cone types (L, M, S) packed most densely at the fovea, "
    "optics sharp to about one arcminute, retinal cells that combine the cones into "
    "brightness, red-green and blue-yellow channels, and a cortex that analyses the image "
    "with oriented filters. At ordinary image sizes the human eye resolves more "
    "detail than the image contains."
)

CITATIONS = (
    "Derrington AM, Krauskopf J, Lennie P (1984). Chromatic mechanisms in lateral "
    "geniculate nucleus of macaque. J Physiol 357:241-265.",
    "Dacey DM (2000). Parallel pathways for spectral coding in primate retina. "
    "Annu Rev Neurosci 23:743-775.",
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
    "Mullen KT (1985). The contrast sensitivity of human colour vision to red-green "
    "and blue-yellow chromatic gratings. J Physiol 359:381-400.",
    "Rucci M, Poletti M (2015). Control and functions of fixational eye movements. "
    "Annu Rev Vis Sci 1:499-518.",
    "Curcio CA, Allen KA, Sloan KR, et al. (1991). Distribution and morphology of human "
    "cone photoreceptors stained with anti-blue opsin. J Comp Neurol 312:610-624.",
    "Thibos LN, Ye M, Zhang X, Bradley A (1992). The chromatic eye: a new reduced-eye "
    "model of ocular chromatic aberration in humans. Appl Opt 31:3594-3600.",
    "Wandell BA. Useful numbers in vision science (a table published with "
    "Foundations of Vision, Sinauer 1995).",
)

# L cones for each M cone. Two is near the average; among people with normal
# colour vision it runs from 1.1 to 16.5 (Hofer et al. 2005).
LM_RATIO = 2.0
LM_RATIO_RANGE = (1.1, 16.5)
S_FRACTION = 0.10  # S cones are sparse (Hofer et al. 2005)


def cone_fractions(lm_ratio: float) -> tuple[float, float, float]:
    """The shares of L, M and S cones for a person with `lm_ratio` L cones per M cone."""
    if lm_ratio <= 0:
        raise ValueError(f"lm_ratio must be positive, got {lm_ratio}")
    middle = (1.0 - S_FRACTION) / (1.0 + lm_ratio)
    return (lm_ratio * middle, middle, S_FRACTION)


PARAMS = EyeParams(
    receptor_names=("L", "M", "S"),
    # Linear RGB to LMS (Vienot et al. 1999), rows scaled to sum to 1.
    color_matrix=((0.2730, 0.6643, 0.0629),
                  (0.1002, 0.7876, 0.1122),
                  (0.0178, 0.1096, 0.8726)),
    type_fractions=cone_fractions(LM_RATIO),  # 0.60, 0.30, 0.10
    colocated=False,
    blur_sigma_deg=0.007,  # point spread about 1 arcmin wide
    lattice="foveated",
    spacing_deg=0.008,  # foveal cone spacing about 0.5 arcmin (Curcio et al. 1990)
    e2_deg=2.0,  # spacing doubles by about 2 degrees eccentricity
    center_sigma_deg=0.05,  # midget cell centre (Croner & Kaplan 1995)
    surround_sigma_deg=0.5,
    surround_weight=0.7,
    # V1 cells cover many octaves (De Valois et al. 1982). A scale whose wavelength
    # is under two pixels is left out, so a picture across 60 degrees uses the
    # first two up to 191 px, three from 192 px and all four from 384 px.
    cortex_sf_cpd=(0.2, 0.8, 1.6, 3.2),
    # Retinal cells combine the cones into three channels (Derrington, Krauskopf &
    # Lennie 1984; Dacey 2000). Each channel's gain lets it fill its firing range
    # (Laughlin 1981); the colour signals are small, so their gains are large.
    # Gains were chosen by measurement on the three samples at 96 px: the best
    # noisy reconstruction with under 0.1% of cells clipped and no loss with
    # ideal neurons (30.7 dB real, 38.4 dB ideal, 0.055% clipped at worst).
    retina_classes=(
        RetinaClass("luminance", (0.5, 0.5, 0.0), gain=1.5, surround_weight=0.7),
        RetinaClass("red_green", (1.0, -1.0, 0.0), gain=8.0, surround_weight=0.7),
        RetinaClass("blue_yellow", (-0.5, -0.5, 1.0), gain=3.0, surround_weight=0.7),
    ),
    # Natural images have less contrast at fine scales (Field 1987), so the
    # fine cortex cells get more gain. Measured on the three samples at 256 px:
    # 2 at every finer scale; 3 clips cells and costs over 5 dB with ideal neurons.
    cortex_gains=(1.0, 2.0, 2.0, 2.0),
    # Colour is seen at lower resolution than brightness (Mullen 1985): the
    # finest scale has luminance cells only (class 0), a third of the cells.
    cortex_types=(None, None, None, (0,)),
    # While fixating, the eye drifts and makes microsaccades of under a degree;
    # over a few seconds the gaze covers about the foveola, a degree across
    # (Rucci & Poletti 2015). A disc of half that width. Measured at 96 px with
    # four looks: 0.1, 0.25 and 0.5 degrees give 31.0, 31.7 and 32.2 dB with
    # real neurons against 30.6 with one look.
    fixation_deg=0.25,
    # The eye cannot focus every wavelength at once: its power changes by about
    # 2 dioptres across the visible spectrum. The chromatic eye of Thibos et al.
    # (1992) gives the refraction, in dioptres relative to 589 nm, as
    # 1.68524 - 0.63346 / (wavelength in micrometres - 0.21410). With 555 nm in
    # focus (the peak of daylight sensitivity; an assumption), the cones' peak
    # wavelengths of about 565, 545 and 440 nm are 0.05, 0.06 and 0.95 dioptres
    # out. A cone's signal from an RGB picture spans many wavelengths, so one
    # defocus per cone type is a first approximation. The formula's constants
    # are quoted from memory of the paper; they give 2.1 dioptres from 400 to
    # 700 nm, which agrees with the published total.
    chromatic_defocus_d=(0.05, 0.06, 0.95),
    # Pupil diameter in daylight; it ranges from 2 mm in bright light to 8 mm in
    # the dark (Wandell's table of useful numbers). 3 mm blurs the S cones' light
    # by 2.4 arcminutes.
    pupil_mm=3.0,
    # The centre of the fovea has no S cones: a zone about 100 micrometres, or
    # 0.35 degrees, across (Curcio et al. 1991). The radius, for each cone type.
    absent_within_deg=(0.0, 0.0, 0.175),
    # `jitter` is left at 0. Real cones are not on a perfect lattice (Hirsch &
    # Miller 1987), but no figure for the disorder could be checked for this
    # work. Measured at 0.1 and 0.2 of the spacing across 1 degree: no change.
    # The pupil also sets the light level for `run(photons_per_s=...)`: retinal
    # illuminance in trolands is luminance (cd/m2) times pupil area (mm2), and one
    # troland is about 137 photons absorbed per second by an L cone and 110 by an
    # M cone (Psychtoolbox, ComputePhotopigmentBleaching: 560 nm, CIE 10-degree
    # cone fundamentals). Through 3 mm, 1 cd/m2 is about 900 photons per cone per
    # second. No figure for S cones was found.
    # `transmission` is left at 1 for every cone type. The lens and the macular
    # pigment absorb short wavelengths before they reach the S cones, but no
    # value could be checked against a source for this work (from memory, a
    # macular density of 0.35 at 460 nm and a lens density of a few tenths, so
    # roughly 0.3 to 0.5 for S). Measured at 0.3: -0.2 dB in a lit room.
)


@species.register("human")
def build(field: VisualField, density: float = 1.0,
          neuron_density: float = 1.0) -> Pipeline:
    return assemble("human", field, PARAMS, DESCRIPTION, CITATIONS, density, neuron_density)
