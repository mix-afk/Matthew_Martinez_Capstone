"""Shared chart style so every figure in the notebooks, report and decks looks the same."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GRAY, INK, GRID = "#52514e", "#0b0b0b", "#e6e5e0"
FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": GRAY,
            "axes.labelcolor": INK,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.prop_cycle": plt.cycler(color=SERIES),
            "legend.frameon": False,
        }
    )


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", facecolor="white")
