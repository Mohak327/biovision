"""Fruit fly: a hexagonal compound eye and lateral inhibition in the lamina."""
from ..core.field import VisualField
from ..core.pipeline import Pipeline
from ..core.registry import species
from .eye import EyeParams, assemble

DESCRIPTION = (
    "A fruit fly's compound eye has about 750 facets, each looking at a patch of "
    "the world roughly five degrees wide, arranged in a hexagonal grid. Each facet "
    "holds receptors tuned to ultraviolet, blue and green. The first layer of the "
    "brain, the lamina, sharpens the picture by subtracting neighbouring facets. "
    "Flies see almost no red."
)

CITATIONS = (
    "Land MF (1997). Visual acuity in insects. Annu Rev Entomol 42:147-177.",
    "Gonzalez-Bellido PT, Wardill TJ, Juusola M (2011). Compound eyes and retinal "
    "information processing in miniature dipteran species match their specific "
    "ecological demands. PNAS 108:4224-4229.",
    "Salcedo E, Huber A, Henrich S, et al. (1999). Blue- and green-absorbing visual "
    "pigments of Drosophila. J Neurosci 19:10716-10726.",
    "Laughlin SB (1981). A simple coding procedure enhances a neuron's information "
    "capacity. Z Naturforsch C 36:910-912.",
)

PARAMS = EyeParams(
    receptor_names=("UV", "blue", "green"),
    # UV (Rh3/Rh4) is approximated from blue; blue is Rh5, green is Rh6/Rh1.
    color_matrix=((0.00, 0.00, 1.00),
                  (0.00, 0.20, 0.80),
                  (0.05, 0.80, 0.15)),
    type_fractions=(1.0, 1.0, 1.0),
    colocated=True,  # every facet carries all receptor types
    blur_sigma_deg=2.1,  # acceptance angle about 5 degrees full width at half maximum
    lattice="hex",
    spacing_deg=5.0,  # interommatidial angle (Land 1997)
    center_sigma_deg=1.0,  # a lamina cartridge is driven by one facet
    surround_sigma_deg=6.0,  # lateral inhibition from neighbouring cartridges
    surround_weight=0.6,
    rest_hz=2500.0,
)


@species.register("fly")
def build(field: VisualField) -> Pipeline:
    return assemble("fly", field, PARAMS, DESCRIPTION, CITATIONS)
