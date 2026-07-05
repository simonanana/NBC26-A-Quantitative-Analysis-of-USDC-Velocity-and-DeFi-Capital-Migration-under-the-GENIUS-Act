"""
Shared configuration: event dates, sample window, paths, plotting style,
and small statistical helpers used across all analysis modules.

Paper: "From Passive Yield to Active Utility: A Quantitative Analysis of
USDC Velocity and DeFi Capital Migration under the GENIUS Act"
"""

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Treatment dates and sample window
# ──────────────────────────────────────────────────────────────────────────────
T1 = pd.Timestamp("2025-07-18")        # T1: GENIUS Act signed into law (Pub. L. 119-27)
T2 = pd.Timestamp("2026-02-25")        # T2: OCC NPRM published (91 Fed. Reg. 10,202)
T_SENATE = pd.Timestamp("2025-06-17")  # Senate passage (anticipatory-window placebo)

SAMPLE_START = pd.Timestamp("2024-01-01")
SAMPLE_END = pd.Timestamp("2026-03-31")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "manual"       # place raw CSV/XLSX inputs here
API_CACHE_DIR = BASE_DIR / "data" / "api_cache"
FIG_DIR = BASE_DIR / "figures"
OUT_DIR = BASE_DIR / "output"

for _d in (DATA_DIR, API_CACHE_DIR, FIG_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib style and color palette
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.labelsize": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "legend.framealpha": 0.9, "legend.edgecolor": "#ccc",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

COLORS = dict(
    usdc="#2166AC", usdt="#B2182B", t1="#D6604D", t2="#F4A582",
    fit="#4393C3", ci="#92C5DE", lend="#2166AC", rwa="#1B7837",
    dex="#B2182B", grey="#969696",
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def star(p: float) -> str:
    """Significance stars: *** p<0.01, ** p<0.05, * p<0.10."""
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


def gini(x) -> float:
    """Gini coefficient of a positive-valued distribution."""
    x = np.sort(np.asarray(x, float))
    x = x[x > 0]
    if len(x) < 2:
        return np.nan
    n = len(x)
    return 2 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x)) - (n + 1) / n


def hhi(x) -> float:
    """Herfindahl-Hirschman Index of a positive-valued distribution."""
    x = np.asarray(x, float)
    x = x[x > 0]
    s = x.sum()
    return float(np.sum((x / s) ** 2)) if s > 0 else np.nan


def treatment_lines(ax, y1: float = 0.94, y2: float = 0.86) -> None:
    """Draw vertical lines and labels for T1 (GENIUS signing) and T2 (OCC NPRM)."""
    ax.axvline(T1, color=COLORS["t1"], ls="--", lw=1.4, alpha=0.85, zorder=2)
    ax.axvline(T2, color=COLORS["t2"], ls=":", lw=1.4, alpha=0.85, zorder=2)
    yl = ax.get_ylim()
    r = yl[1] - yl[0]
    ax.text(T1 + timedelta(4), yl[0] + r * y1, "T$_1$",
            fontsize=9, fontweight="bold", color=COLORS["t1"], va="top")
    ax.text(T2 + timedelta(4), yl[0] + r * y2, "T$_2$",
            fontsize=9, fontweight="bold", color=COLORS["t2"], va="top")
