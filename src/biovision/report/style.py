"""One look for every figure."""
from matplotlib.figure import Figure

# Okabe-Ito palette: distinguishable with any common colour-vision deficiency.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
RC = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300,
}


def new_figure(rows: int = 1, cols: int = 1, width: float = 7.0, height: float = 3.5):
    """A figure and a (rows, cols) array of axes, without pyplot's global state."""
    import matplotlib

    with matplotlib.rc_context(RC):
        figure = Figure(figsize=(width, height), layout="constrained")
        axes = figure.subplots(rows, cols, squeeze=False)
    for axis in axes.ravel():
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
    return figure, axes


def show_image(axis, image, title: str, **kwargs):
    axis.imshow(image, **kwargs)
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
