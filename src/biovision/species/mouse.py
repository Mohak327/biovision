"""Mouse: two cone types (UV and green), coarse uniform sampling, low-frequency V1."""
from ..core.field import VisualField
from ..core.pipeline import Pipeline
from ..core.registry import species
from .eye import EyeParams, assemble

DESCRIPTION = (
    "Mice see with about a hundred times less acuity than humans: roughly half a "
    "cycle per degree. They have two cone types, one tuned to ultraviolet and one "
    "to green, and almost no sensitivity to red. This models the cone pathway in "
    "daylight; the mouse retina is otherwise dominated by rods."
)

CITATIONS = (
    "Prusky GT, West PW, Douglas RM (2000). Behavioral assessment of visual acuity "
    "in mice and rats. Vision Res 40:2201-2209.",
    "Jacobs GH, Neitz J, Deegan JF (1991). Retinal receptors in rodents maximally "
    "sensitive to ultraviolet light. Nature 353:655-656.",
    "Niell CM, Stryker MP (2008). Highly selective receptive fields in mouse visual "
    "cortex. J Neurosci 28:7520-7536.",
    "Stone C, Pinto LH (1993). Response properties of ganglion cells in the isolated "
    "mouse retina. Vis Neurosci 10:31-39.",
)

PARAMS = EyeParams(
    receptor_names=("UV", "M"),
    # UV opsin (360 nm) is approximated from blue; M opsin (508 nm) from green.
    color_matrix=((0.00, 0.10, 0.90),
                  (0.05, 0.85, 0.10)),
    type_fractions=(0.5, 0.5),
    colocated=False,
    blur_sigma_deg=0.3,
    lattice="square",
    spacing_deg=1.0,  # Nyquist limit 0.5 cycles/degree (Prusky et al. 2000)
    center_sigma_deg=1.0,  # ganglion cell centres span several degrees (Stone & Pinto 1993)
    surround_sigma_deg=4.0,
    surround_weight=0.7,
    cortex_sf_cpd=(0.04, 0.16),  # Niell & Stryker 2008: preferred about 0.04 cycles/degree
)


@species.register("mouse")
def build(field: VisualField) -> Pipeline:
    return assemble("mouse", field, PARAMS, DESCRIPTION, CITATIONS)
