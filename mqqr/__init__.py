"""
mqqr — Multivariate Quantile-on-Quantile Regression & Causality
================================================================

A comprehensive Python toolkit for:

* **QQR**   — bivariate Quantile-on-Quantile Regression (Sim & Zhou, 2015)
* **m-QQR** — Multivariate QQR with exogenous moderators
              (Sinha et al. 2023; Ozkan et al. 2023; Alola et al. 2023; Das et al. 2023)
* **m-QQ Causality** — Quantile Granger causality conditional on moderators
              (Troster 2018; Sinha et al., RisAnal 2024)

Visualizations replicate the MATLAB ``surf`` look-and-feel — Jet / Parula
colour-maps, journal-quality heat-maps, contour plots, causality heat-maps
with significance stars, and side-by-side QR vs m-QQR comparisons.

Author
------
Dr. Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane/qqrpy

Modules
-------
``qqr``        : Bivariate QQ regression (Sim & Zhou, 2015)
``mqqr``       : Multivariate QQ regression (m-QQR)
``causality``  : Quantile causality and multivariate quantile causality
``plotting``   : 3D surfaces, heat-maps, contours, causality plots
``tables``     : LaTeX / Markdown publication tables
``colors``     : MATLAB-style colour-maps (Jet, Parula, Turbo, …)
``utils``      : Quantile primitives, kernels, weighted quantile regression
"""

from ._version import __version__

# ── Core estimation ────────────────────────────────────────────────────────

from .qqr import qq_regression, QQResult
from .mqqr import mqq_regression, MQQResult
from .additive_mqqr import additive_mqq_regression, AdditiveMQQResult
from .causality import (
    qq_causality, QQCausalityResult,
    mqq_causality, MQQCausalityResult,
)

# ── Visualisation ──────────────────────────────────────────────────────────

from .plotting import (
    plot_qq_3d,
    plot_qq_heatmap,
    plot_qq_contour,
    plot_mqq_3d,
    plot_mqq_heatmap,
    plot_mqq_panel,
    plot_additive_mqq_3d,
    plot_additive_mqq_heatmap,
    plot_additive_mqq_panel,
    plot_qq_causality_heatmap,
    plot_qq_vs_mqq,
    plot_qq_3d_plotly,
)

# ── Tables ────────────────────────────────────────────────────────────────

from .tables import (
    descriptive_table,
    results_table,
    to_latex,
    to_markdown,
    significance_stars,
)

# ── Colours ───────────────────────────────────────────────────────────────

from .colors import (
    MATLAB_JET,
    MATLAB_PARULA,
    TURBO,
    BLUE_RED,
    RED_YELLOW_BLACK,
    list_colormaps,
    show_colormaps,
)

__all__ = [
    "__version__",
    # Core
    "qq_regression", "QQResult",
    "mqq_regression", "MQQResult",
    "additive_mqq_regression", "AdditiveMQQResult",
    "qq_causality", "QQCausalityResult",
    "mqq_causality", "MQQCausalityResult",
    # Plots
    "plot_qq_3d", "plot_qq_heatmap", "plot_qq_contour",
    "plot_mqq_3d", "plot_mqq_heatmap", "plot_mqq_panel",
    "plot_additive_mqq_3d", "plot_additive_mqq_heatmap",
    "plot_additive_mqq_panel",
    "plot_qq_causality_heatmap", "plot_qq_vs_mqq", "plot_qq_3d_plotly",
    # Tables
    "descriptive_table", "results_table",
    "to_latex", "to_markdown", "significance_stars",
    # Colours
    "MATLAB_JET", "MATLAB_PARULA", "TURBO", "BLUE_RED", "RED_YELLOW_BLACK",
    "list_colormaps", "show_colormaps",
]
