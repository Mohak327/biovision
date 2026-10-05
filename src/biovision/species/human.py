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
    # 2 at every finer scale; 3 clips cells and costs 5 dB with ideal neurons.
    cortex_gains=(1.0, 2.0, 2.0, 2.0),
    # Colour is seen at lower resolution than brightness (Mullen 1985): the
    # finest scale has luminance cells only (class 0), a third of the cells.
    cortex_types=(None, None, None, (0,)),
)


@species.register("human")
def build(field: VisualField, density: float = 1.0,
          neuron_density: float = 1.0) -> Pipeline:
    return assemble("human", field, PARAMS, DESCRIPTION, CITATIONS, density, neuron_density)
