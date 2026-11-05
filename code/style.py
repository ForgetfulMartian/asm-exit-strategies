"""Shared figure style: validated categorical palette (validate_palette.js: all gates pass), recessive grid, thin marks, no top/right spines."""
from cycler import cycler
import matplotlib.pyplot as plt
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.prop_cycle": cycler(color=SERIES), "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 1.0,
    "axes.axisbelow": True, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
    "lines.linewidth": 1.8, "lines.markersize": 5, "patch.linewidth": 0.8, "patch.edgecolor": SURF,
    "legend.frameon": False, "legend.fontsize": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
})
