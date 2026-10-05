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
    "Joesch M, Schnell B, Raghu SV, Reiff DF, Borst A (2010). ON and OFF pathways in "
    "Drosophila motion vision. Nature 468:300-304.",
    "Juusola M, Dau A, Song Z, et al. (2017). Microsaccadic sampling of moving image "
    "information provides Drosophila hyperacute vision. eLife 6:e26117.",
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
    # A fly's eye is fixed to its head, but its photoreceptors twitch: light makes
    # them contract, which moves each receptive field by 0.5 to 4 degrees and lets
    # the fly resolve detail finer than its facet spacing (Juusola et al. 2017).
    # The middle of that range. Measured at 128 px with ideal neurons: 1.25, 2.5
    # and 5 degrees give the same gain within 0.05 dB.
    fixation_deg=2.0,
    # `spontaneous_hz` is left at None: one cell stands for the ON/OFF pair. The
    # lamina's L1 and L2 cells feed the ON and OFF motion pathways (Joesch et
    # al. 2010), but each answers to both signs with a graded potential; the
    # rectification comes later, in the medulla, which this model does not have.
    # Measured at 128 px, five seeds, with a rectified pair: 13.99 dB against
    # 13.75 with real neurons.
)


@species.register("fly")
def build(field: VisualField, density: float = 1.0,
          neuron_density: float = 1.0) -> Pipeline:
    return assemble("fly", field, PARAMS, DESCRIPTION, CITATIONS, density, neuron_density)
