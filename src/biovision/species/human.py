"""Human: three cone types, a fovea, centre-surround retina, V1 simple cells."""
from ..core.field import VisualField
from ..core.pipeline import Pipeline
from ..core.registry import species
from .eye import EyeParams, assemble

DESCRIPTION = (
    "Humans have three cone types (L, M, S) packed most densely at the fovea, "
    "optics sharp to about one arcminute, and a cortex that analyses the image "
    "with oriented filters. At ordinary image sizes the human eye resolves more "
    "detail than the image contains."
)

CITATIONS = (
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
    cortex_sf_cpd=(0.2, 0.8),  # within the range an image of this size can carry
)


@species.register("human")
def build(field: VisualField) -> Pipeline:
    return assemble("human", field, PARAMS, DESCRIPTION, CITATIONS)
