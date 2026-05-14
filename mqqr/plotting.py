"""
Publication-quality visualisations for mqqr.

* **3D surfaces** with MATLAB Jet / Parula / Turbo --
  the standard presentation in Sim & Zhou (2015), Sinha et al. (2023),
  Ozkan et al. (2023), Alola et al. (2023), Das et al. (2023).
* **Heat-maps** with significance stars, used for cross-quantile
  correlation panels in the Energy-Economics and SCED papers.
* **Causality heat-maps** with significance overlay.
* **QR vs m-QQR collapse plots** (Sim & Zhou Figure 5 style).
* **Side-by-side panels** for moderator interaction effects.

All static figures use ``matplotlib`` (publishable as PDF / EPS).
An optional ``plotly`` HTML 3D surface is provided when the package is
installed (``pip install mqqr[plotly]``).
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401  (enables 3D)

from .colors import get_cmap, MATLAB_JET, BLUE_RED, RED_YELLOW_BLACK


# ─────────────────────────────────────────────────────────────────────────────
#  Journal-quality matplotlib style
# ─────────────────────────────────────────────────────────────────────────────

def _journal_style():
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["DejaVu Serif", "Times New Roman",
                             "Times", "Computer Modern Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
        "axes.labelsize":   12,
        "axes.labelweight": "normal",
        "axes.linewidth":   0.9,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
        "legend.frameon":   True,
        "legend.framealpha": 0.95,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "axes.grid":        False,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  helpers
# ─────────────────────────────────────────────────────────────────────────────

def _star(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _matrix_from_result(result, value):
    """Resolve a (Y q × X q) matrix from a QQResult / MQQResult / Causality."""
    if hasattr(result, "to_matrix"):
        return result.to_matrix(value)
    if hasattr(result, "stat_matrix") and value == "t_value":
        return result.stat_matrix()
    if hasattr(result, "coef_matrix") and value == "beta1":
        return result.coef_matrix()
    raise TypeError("result object has no recognised matrix accessor")


# ═════════════════════════════════════════════════════════════════════════════
#  1.  QQR / m-QQR  --  3D MATLAB-style surface
# ═════════════════════════════════════════════════════════════════════════════

def plot_qq_3d(result, *, value: str = "beta1", cmap="jet",
               elev: float = 28, azim: float = -130,
               x_label: Optional[str] = None,
               y_label: Optional[str] = None,
               z_label: Optional[str] = None,
               title: Optional[str] = None,
               figsize=(11, 8),
               edge_color: str = "k",
               edge_alpha: float = 0.25,
               edge_lw: float = 0.25,
               save_path: Optional[str] = None):
    """
    MATLAB-``surf`` style 3D surface of a QQ / m-QQR matrix.

    Defaults to the principal slope ``β1(θ, τ)`` with the MATLAB Jet
    colour-map.  ``value`` can be ``beta1``, ``r_squared``, ``p_value``,
    ``t_value`` (causality) etc.

    Returns
    -------
    fig, ax
    """
    _journal_style()
    Z = _matrix_from_result(result, value)
    yq = np.asarray(result.y_quantiles, dtype=float)
    xq = np.asarray(result.x_quantiles, dtype=float)
    X, Y = np.meshgrid(xq, yq)

    if x_label is None:
        x_label = f"Quantile of {getattr(result, 'x_name', 'X')}  (τ)"
    if y_label is None:
        y_label = f"Quantile of {getattr(result, 'y_name', 'Y')}  (θ)"
    if z_label is None:
        z_label = {"beta1": r"$\hat{\beta}_1(\theta,\tau)$",
                   "beta0": r"$\hat{\beta}_0(\theta,\tau)$",
                   "r_squared": r"$R^2(\theta,\tau)$",
                   "p_value": r"$p(\theta,\tau)$",
                   "t_value": r"$t(\theta,\tau)$"}.get(value, value)
    if title is None:
        method = getattr(result, "method", "QQ Regression")
        title = method

    cmap_obj = get_cmap(cmap)

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Pad axis ranges slightly for visual breathing room
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cmap_obj,
        rstride=1, cstride=1,
        edgecolor=edge_color,
        linewidth=edge_lw,
        alpha=0.97,
        antialiased=True,
        shade=True,
    )

    ax.set_xlabel(x_label, labelpad=12)
    ax.set_ylabel(y_label, labelpad=12)
    ax.set_zlabel(z_label, labelpad=10)
    ax.set_title(title, pad=16)

    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")
    ax.xaxis.pane.set_edgecolor("#888888")
    ax.yaxis.pane.set_edgecolor("#888888")
    ax.zaxis.pane.set_edgecolor("#888888")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    cb = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=18, pad=0.10)
    cb.set_label(z_label, fontsize=11)
    cb.ax.tick_params(labelsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═════════════════════════════════════════════════════════════════════════════
#  2.  QQR / m-QQR  --  Heatmap (with stars)
# ═════════════════════════════════════════════════════════════════════════════

def plot_qq_heatmap(result, *, value: str = "beta1", cmap="jet",
                    x_label: Optional[str] = None,
                    y_label: Optional[str] = None,
                    title: Optional[str] = None,
                    figsize=(8.5, 7.2),
                    annotate: Union[bool, str] = "stars",
                    annot_fontsize: float = 8,
                    cell_lines: bool = True,
                    save_path: Optional[str] = None):
    """
    Journal-quality heat-map of a QQ / m-QQR matrix.

    ``annotate`` :

    * ``False``   -- no in-cell annotation,
    * ``'stars'`` -- significance markers (``*** / ** / *``),
    * ``'values'`` -- numeric values rounded to 2 dp.

    Returns
    -------
    fig, ax
    """
    _journal_style()
    Z = _matrix_from_result(result, value)
    yq = np.asarray(result.y_quantiles, dtype=float)
    xq = np.asarray(result.x_quantiles, dtype=float)

    if x_label is None:
        x_label = f"Quantile of {getattr(result, 'x_name', 'X')}  (τ)"
    if y_label is None:
        y_label = f"Quantile of {getattr(result, 'y_name', 'Y')}  (θ)"
    if title is None:
        label_map = {
            "beta1":     r"$\hat{\beta}_1$",
            "beta0":     r"$\hat{\beta}_0$",
            "r_squared": r"$R^2$",
            "p_value":   r"$p$-value",
            "t_value":   r"$t$-statistic",
        }
        title = f"{getattr(result, 'method', 'QQ regression')} -- {label_map.get(value, value)}"

    cmap_obj = get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    im = ax.imshow(
        Z, cmap=cmap_obj, aspect="auto", origin="lower",
        interpolation="nearest",
        extent=[-0.5, len(xq) - 0.5, -0.5, len(yq) - 0.5],
    )

    ax.set_xticks(range(len(xq)))
    ax.set_xticklabels([f"{q:.2f}" for q in xq], rotation=45, ha="right")
    ax.set_yticks(range(len(yq)))
    ax.set_yticklabels([f"{q:.2f}" for q in yq])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=10)

    # Cell-border grid (like the Sinha-paper heat-maps)
    if cell_lines:
        ax.set_xticks(np.arange(-0.5, len(xq), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(yq), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.6)
        ax.tick_params(which="minor", length=0)

    # Annotations
    df = result.results if hasattr(result, "results") else result.main_results
    pmat = df.pivot(index="y_quantile", columns="x_quantile",
                    values="p_value")
    pmat = pmat.sort_index(axis=0).sort_index(axis=1).values

    if annotate:
        vmax = np.nanmax(np.abs(Z))
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v = Z[i, j]
                if not np.isfinite(v):
                    continue
                # text colour : white on dark cells, black on light
                norm_v = (v - np.nanmin(Z)) / (np.nanmax(Z) - np.nanmin(Z) + 1e-12)
                txt_color = "white" if (norm_v < 0.25 or norm_v > 0.75) else "black"
                if annotate == "stars":
                    s = _star(pmat[i, j])
                    if s:
                        ax.text(j, i, s, ha="center", va="center",
                                fontsize=annot_fontsize, color=txt_color,
                                fontweight="bold")
                elif annotate == "values":
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=annot_fontsize - 1, color=txt_color)

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label({"beta1": r"$\hat{\beta}_1(\theta,\tau)$",
                  "r_squared": r"$R^2$",
                  "p_value": "p-value",
                  "t_value": r"$t$"}.get(value, value), fontsize=10)
    cb.ax.tick_params(labelsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═════════════════════════════════════════════════════════════════════════════
#  3.  QQR / m-QQR  --  Contour
# ═════════════════════════════════════════════════════════════════════════════

def plot_qq_contour(result, *, value: str = "beta1", cmap="jet",
                    levels: int = 22,
                    x_label: Optional[str] = None,
                    y_label: Optional[str] = None,
                    title: Optional[str] = None,
                    figsize=(8.5, 7.2),
                    save_path: Optional[str] = None):
    """Filled contour plot with labelled level curves."""
    _journal_style()
    Z = _matrix_from_result(result, value)
    yq = np.asarray(result.y_quantiles, dtype=float)
    xq = np.asarray(result.x_quantiles, dtype=float)
    X, Y = np.meshgrid(xq, yq)

    if x_label is None:
        x_label = f"Quantile of {getattr(result, 'x_name', 'X')}  (τ)"
    if y_label is None:
        y_label = f"Quantile of {getattr(result, 'y_name', 'Y')}  (θ)"
    if title is None:
        title = f"{getattr(result, 'method', 'QQ regression')} -- contour"

    cmap_obj = get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap_obj)
    cl = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.3,
                    alpha=0.55)
    ax.clabel(cl, inline=True, fontsize=7, fmt="%+.2f")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=10)
    plt.colorbar(cs, ax=ax, shrink=0.85, pad=0.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═════════════════════════════════════════════════════════════════════════════
#  4.  m-QQR specific plots
# ═════════════════════════════════════════════════════════════════════════════

def plot_mqq_3d(mqq_result, *, value: str = "beta1", cmap="jet",
                title: Optional[str] = None, **kwargs):
    """3-D surface of an m-QQR principal slope (alias for plot_qq_3d)."""
    if title is None:
        title = (f"m-QQR -- effect of {mqq_result.x_name} on {mqq_result.y_name}"
                 + (f"  (moderated by {', '.join(mqq_result.moderator_names)})"
                    if mqq_result.moderator_names else ""))
    return plot_qq_3d(mqq_result, value=value, cmap=cmap, title=title, **kwargs)


def plot_mqq_heatmap(mqq_result, *, value: str = "beta1", cmap="jet",
                     title: Optional[str] = None, **kwargs):
    """Heat-map of an m-QQR principal slope."""
    if title is None:
        title = (f"m-QQR -- effect of {mqq_result.x_name} on {mqq_result.y_name}"
                 + (f"  (cond. on {', '.join(mqq_result.moderator_names)})"
                    if mqq_result.moderator_names else ""))
    return plot_qq_heatmap(mqq_result, value=value, cmap=cmap,
                           title=title, **kwargs)


def plot_mqq_panel(mqq_result, *, cmap="jet", figsize=(15, 5),
                   save_path: Optional[str] = None):
    """
    Three-panel side-by-side comparison (Sinha 2023 figure style):

    +---------------+-----------------+----------------------+
    | β1 surface    | β1 heat-map     | per-moderator γ heat |
    | (3-D Jet)     | (with stars)    | (one per moderator)  |
    +---------------+-----------------+----------------------+
    """
    _journal_style()
    p = len(mqq_result.moderator_names)
    n_cols = 2 + max(1, p)
    fig = plt.figure(figsize=(figsize[0] * (n_cols / 3), figsize[1]),
                     facecolor="white")

    # Panel 1 : 3D surface
    ax1 = fig.add_subplot(1, n_cols, 1, projection="3d")
    Z = mqq_result.to_matrix("beta1")
    X, Y = np.meshgrid(mqq_result.x_quantiles, mqq_result.y_quantiles)
    cmap_obj = get_cmap(cmap)
    surf = ax1.plot_surface(X, Y, Z, cmap=cmap_obj, rstride=1, cstride=1,
                            edgecolor="k", linewidth=0.25, alpha=0.97)
    ax1.set_xlabel(f"τ  ({mqq_result.x_name})", labelpad=10)
    ax1.set_ylabel(f"θ  ({mqq_result.y_name})", labelpad=10)
    ax1.set_zlabel(r"$\hat{\beta}_1$")
    ax1.set_title(r"(a)  $\hat{\beta}_1(\theta,\tau)$ surface")
    ax1.view_init(elev=28, azim=-130)
    fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=18, pad=0.1)

    # Panel 2 : 2D heat-map
    ax2 = fig.add_subplot(1, n_cols, 2)
    im = ax2.imshow(Z, cmap=cmap_obj, aspect="auto", origin="lower",
                    interpolation="nearest")
    ax2.set_xticks(range(len(mqq_result.x_quantiles)))
    ax2.set_xticklabels([f"{q:.2f}" for q in mqq_result.x_quantiles],
                        rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(mqq_result.y_quantiles)))
    ax2.set_yticklabels([f"{q:.2f}" for q in mqq_result.y_quantiles],
                        fontsize=8)
    ax2.set_xlabel("τ"); ax2.set_ylabel("θ")
    ax2.set_title(r"(b)  $\hat{\beta}_1$ heat-map")
    fig.colorbar(im, ax=ax2, shrink=0.85)

    # Panels 3 … : moderator interaction γ heat-maps
    for i, name in enumerate(mqq_result.moderator_names):
        ax = fig.add_subplot(1, n_cols, 3 + i)
        try:
            G = mqq_result.interaction_matrix(name, "gamma")
        except Exception:
            ax.text(0.5, 0.5, "(no interactions)", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        im2 = ax.imshow(G, cmap=BLUE_RED, aspect="auto", origin="lower",
                        interpolation="nearest",
                        vmin=-np.nanmax(np.abs(G)),
                        vmax=np.nanmax(np.abs(G)))
        ax.set_xticks(range(len(mqq_result.x_quantiles)))
        ax.set_xticklabels([f"{q:.2f}" for q in mqq_result.x_quantiles],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(mqq_result.y_quantiles)))
        ax.set_yticklabels([f"{q:.2f}" for q in mqq_result.y_quantiles],
                           fontsize=8)
        ax.set_xlabel("τ"); ax.set_ylabel("θ")
        ax.set_title(f"(c{i+1})  γ -- {name}")
        fig.colorbar(im2, ax=ax, shrink=0.85)

    fig.suptitle(
        f"m-QQR :  {mqq_result.x_name}  ==>  {mqq_result.y_name}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  4b.  Additive m-QQR (Alola/Ozkan 2023) per-variable plots
# ═════════════════════════════════════════════════════════════════════════════

def plot_additive_mqq_3d(result, variable, *, value: str = "beta",
                          cmap="jet", elev: float = 28, azim: float = -130,
                          title: Optional[str] = None,
                          figsize=(11, 8),
                          save_path: Optional[str] = None):
    """
    MATLAB-style 3-D surface for one regressor from an
    :class:`AdditiveMQQResult` (Alola et al. 2023).

    Parameters
    ----------
    result : AdditiveMQQResult
    variable : str
        Name of the regressor whose surface to plot.
    value : str
        Column to render -- ``beta`` (default), ``r_squared``, ``p_value``...
    """
    _journal_style()
    Z = result.to_matrix(variable, value=value)
    yq = np.asarray(result.y_quantiles, dtype=float)
    xq = np.asarray(result.x_quantiles, dtype=float)
    X, Y = np.meshgrid(xq, yq)

    z_label = {"beta":      r"$\hat{\beta}_i(\theta,\Phi)$",
               "r_squared": r"$R^2(\theta,\Phi)$",
               "p_value":   r"$p(\theta,\Phi)$",
               "t_value":   r"$t(\theta,\Phi)$"}.get(value, value)
    if title is None:
        title = (f"Additive m-QQR -- effect of {variable} on "
                 f"{result.y_name}")

    cmap_obj = get_cmap(cmap)
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    surf = ax.plot_surface(
        X, Y, Z, cmap=cmap_obj, rstride=1, cstride=1,
        edgecolor="k", linewidth=0.25, alpha=0.97,
        antialiased=True, shade=True,
    )
    ax.set_xlabel(f"Quantile of {variable}  ($\\Phi$)", labelpad=12)
    ax.set_ylabel(f"Quantile of {result.y_name}  ($\\theta$)", labelpad=12)
    ax.set_zlabel(z_label, labelpad=10)
    ax.set_title(title, pad=16)
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    cb = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=18, pad=0.10)
    cb.set_label(z_label, fontsize=11)
    cb.ax.tick_params(labelsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_additive_mqq_heatmap(result, variable, *, value: str = "beta",
                               cmap="jet",
                               title: Optional[str] = None,
                               annotate: Union[bool, str] = "stars",
                               figsize=(8.5, 7.2),
                               save_path: Optional[str] = None):
    """
    Journal-style heat-map for one regressor of an
    :class:`AdditiveMQQResult`, with optional ``*/**/***``
    significance stars overlaid.
    """
    _journal_style()
    Z = result.to_matrix(variable, value=value)
    yq = np.asarray(result.y_quantiles, dtype=float)
    xq = np.asarray(result.x_quantiles, dtype=float)
    cmap_obj = get_cmap(cmap)

    if title is None:
        title = (f"Additive m-QQR -- {variable}  ==>  {result.y_name}")

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    vmax = np.nanmax(np.abs(Z)) if np.any(np.isfinite(Z)) else 1.0
    if value == "beta":
        im = ax.imshow(Z, cmap=cmap_obj, aspect="auto", origin="lower",
                       interpolation="nearest",
                       vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(Z, cmap=cmap_obj, aspect="auto", origin="lower",
                       interpolation="nearest")

    ax.set_xticks(range(len(xq)))
    ax.set_xticklabels([f"{q:.2f}" for q in xq],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(yq)))
    ax.set_yticklabels([f"{q:.2f}" for q in yq], fontsize=8)
    ax.set_xlabel(f"Quantile of {variable}  ($\\Phi$)")
    ax.set_ylabel(f"Quantile of {result.y_name}  ($\\theta$)")
    ax.set_title(title, pad=10)

    if annotate == "stars" and value == "beta":
        P = result.to_matrix(variable, "p_value")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                p = P[i, j]
                if not np.isfinite(p):
                    continue
                if p < 0.01:    s = "***"
                elif p < 0.05:  s = "**"
                elif p < 0.10:  s = "*"
                else:           s = ""
                if s:
                    ax.text(j, i, s, ha="center", va="center",
                            color="black", fontsize=7, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.ax.tick_params(labelsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_additive_mqq_panel(result, *, cmap="jet", value: str = "beta",
                             figsize_per_var=(5.5, 4.6),
                             save_path: Optional[str] = None):
    """
    Alola Fig. 3 style panel:  one column per regressor, with the
    3-D surface above and the heat-map below.

    Replicates the side-by-side "Traditional QQR vs Multivariate QQR"
    comparison layout of Alola et al. (2023) Energy Economics 120.
    """
    _journal_style()
    names = result.x_names
    n = len(names)
    fig = plt.figure(
        figsize=(figsize_per_var[0] * n, figsize_per_var[1] * 2),
        facecolor="white",
    )
    cmap_obj = get_cmap(cmap)

    for i, name in enumerate(names):
        Z = result.to_matrix(name, value=value)
        yq = np.asarray(result.y_quantiles, dtype=float)
        xq = np.asarray(result.x_quantiles, dtype=float)
        X, Y = np.meshgrid(xq, yq)

        # 3D surface (top row)
        ax3d = fig.add_subplot(2, n, i + 1, projection="3d")
        surf = ax3d.plot_surface(
            X, Y, Z, cmap=cmap_obj, rstride=1, cstride=1,
            edgecolor="k", linewidth=0.2, alpha=0.97,
        )
        ax3d.set_xlabel(f"$\\Phi$ ({name})", labelpad=8, fontsize=9)
        ax3d.set_ylabel(f"$\\theta$ ({result.y_name})", labelpad=8,
                        fontsize=9)
        ax3d.set_zlabel(r"$\hat{\beta}$", fontsize=9)
        ax3d.set_title(f"({chr(97+i)})  {name}  ==>  {result.y_name}",
                       fontsize=11, fontweight="bold")
        ax3d.view_init(elev=28, azim=-130)
        ax3d.tick_params(labelsize=7)
        fig.colorbar(surf, ax=ax3d, shrink=0.55, aspect=14, pad=0.10)

        # Heat-map (bottom row)
        ax2d = fig.add_subplot(2, n, n + i + 1)
        vmax = (np.nanmax(np.abs(Z))
                if np.any(np.isfinite(Z)) else 1.0)
        im = ax2d.imshow(Z, cmap=cmap_obj, aspect="auto",
                         origin="lower", interpolation="nearest",
                         vmin=-vmax if value == "beta" else None,
                         vmax=vmax if value == "beta" else None)
        ax2d.set_xticks(range(len(xq)))
        ax2d.set_xticklabels(
            [f"{q:.2f}" for q in xq], rotation=45, ha="right",
            fontsize=7,
        )
        ax2d.set_yticks(range(len(yq)))
        ax2d.set_yticklabels([f"{q:.2f}" for q in yq], fontsize=7)
        ax2d.set_xlabel(f"$\\Phi$ ({name})", fontsize=9)
        ax2d.set_ylabel(f"$\\theta$ ({result.y_name})", fontsize=9)
        ax2d.set_title(f"heat-map of $\\hat{{\\beta}}$  --  {name}",
                       fontsize=10)

        if value == "beta":
            P = result.to_matrix(name, "p_value")
            for r in range(Z.shape[0]):
                for c in range(Z.shape[1]):
                    p = P[r, c]
                    if not np.isfinite(p):
                        continue
                    if p < 0.01:    s = "***"
                    elif p < 0.05:  s = "**"
                    elif p < 0.10:  s = "*"
                    else:           s = ""
                    if s:
                        ax2d.text(c, r, s, ha="center", va="center",
                                  color="black", fontsize=6,
                                  fontweight="bold")
        fig.colorbar(im, ax=ax2d, shrink=0.85)

    fig.suptitle(
        f"Additive m-QQR (Alola et al. 2023) :  "
        f"effects on {result.y_name}",
        fontsize=14, fontweight="bold", y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  5.  QQ Causality heat-map
# ═════════════════════════════════════════════════════════════════════════════

def plot_qq_causality_heatmap(causality_result, *, cmap="red_yellow_black",
                              value: str = "t_value",
                              show_stars: bool = True,
                              title: Optional[str] = None,
                              figsize=(8.5, 7.2),
                              save_path: Optional[str] = None):
    """
    Causality heat-map across the (θ, τ) grid -- the standard
    presentation in Sinha-co-authored quantile-causality papers.

    By default plots the *absolute* test statistic with the
    red-yellow-black Sinha palette; ``*``, ``**``, ``***`` overlays
    flag significance.  Set ``cmap='jet'`` for the MATLAB Jet variant.
    """
    _journal_style()
    Z = causality_result.stat_matrix()
    Z_plot = np.abs(Z) if value == "t_value" else _matrix_from_result(
        causality_result, value)
    yq = np.asarray(causality_result.y_quantiles, dtype=float)
    xq = np.asarray(causality_result.x_quantiles, dtype=float)

    if title is None:
        title = (f"Quantile Granger Causality  "
                 f"({causality_result.cause_name} ==> "
                 f"{causality_result.effect_name})")
        if getattr(causality_result, "moderator_names", []):
            title += "  -- conditional on " + \
                ", ".join(causality_result.moderator_names)

    cmap_obj = get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    im = ax.imshow(Z_plot, cmap=cmap_obj, aspect="auto", origin="lower",
                   interpolation="nearest")

    ax.set_xticks(range(len(xq)))
    ax.set_xticklabels([f"{q:.2f}" for q in xq], rotation=45, ha="right")
    ax.set_yticks(range(len(yq)))
    ax.set_yticklabels([f"{q:.2f}" for q in yq])
    ax.set_xlabel(rf"Quantile of {causality_result.cause_name}  (τ)")
    ax.set_ylabel(rf"Quantile of {causality_result.effect_name}  (θ)")
    ax.set_title(title, pad=10)

    # Cell-border grid
    ax.set_xticks(np.arange(-0.5, len(xq), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(yq), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, alpha=0.6)
    ax.tick_params(which="minor", length=0)

    # Stars
    if show_stars:
        P = causality_result.pvalue_matrix()
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                s = _star(P[i, j])
                if s:
                    ax.text(j, i, s, ha="center", va="center",
                            fontsize=9, fontweight="bold", color="white")

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$|t(\theta, \tau)|$" if value == "t_value" else value,
                 fontsize=10)
    cb.ax.tick_params(labelsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═════════════════════════════════════════════════════════════════════════════
#  6.  Conventional QR vs m-QQR collapse plot  (Sim & Zhou Fig 5 style)
# ═════════════════════════════════════════════════════════════════════════════

def plot_qq_vs_mqq(qq_result, mqq_result=None, *, title: Optional[str] = None,
                   figsize=(11, 5), save_path: Optional[str] = None):
    """
    Validation panel: average over τ to recover QR-style slope and
    contrast with the QR-via-averaged-QQ check.

    If ``mqq_result`` is also supplied, its τ-averaged β1(θ) is overlaid
    -- this is the convenient way to visualise how moderation re-shapes
    the QR coefficient curve.
    """
    _journal_style()
    avg_qq = qq_result.average_over_tau()

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")

    ax = axes[0]
    ax.plot(avg_qq["y_quantile"], avg_qq["beta0"], color="#1f3a93",
            lw=2.0, marker="o", markersize=5, label=r"$\overline{\beta}_0$")
    if mqq_result is not None:
        avg_m = mqq_result.main_results.groupby("y_quantile")[["beta0", "beta1"]] \
            .mean().reset_index().sort_values("y_quantile")
        ax.plot(avg_m["y_quantile"], avg_m["beta0"], color="#c0392b",
                lw=2.0, linestyle="--", marker="s", markersize=5,
                label=r"m-QQR  $\overline{\beta}_0$")
    ax.set_xlabel(rf"Quantile of {qq_result.y_name}  (θ)")
    ax.set_ylabel(r"Intercept  $\beta_0(\theta)$")
    ax.set_title(r"(a)  $\beta_0(\theta)$")
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(avg_qq["y_quantile"], avg_qq["beta1"], color="#1f3a93",
            lw=2.0, marker="o", markersize=5, label=r"$\overline{\beta}_1$  (QQR)")
    if mqq_result is not None:
        ax.plot(avg_m["y_quantile"], avg_m["beta1"], color="#c0392b",
                lw=2.0, linestyle="--", marker="s", markersize=5,
                label=r"$\overline{\beta}_1$  (m-QQR)")
    ax.set_xlabel(rf"Quantile of {qq_result.y_name}  (θ)")
    ax.set_ylabel(r"Slope  $\beta_1(\theta)$")
    ax.set_title(r"(b)  $\beta_1(\theta)$")
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.grid(True, alpha=0.3); ax.legend()

    if title is None:
        title = "QQR vs m-QQR  --  τ-averaged coefficient curves"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


# ═════════════════════════════════════════════════════════════════════════════
#  7.  Plotly interactive 3-D (MATLAB-feel)
# ═════════════════════════════════════════════════════════════════════════════

def plot_qq_3d_plotly(result, *, value: str = "beta1", cmap: str = "Jet",
                       title: Optional[str] = None,
                       x_label: Optional[str] = None,
                       y_label: Optional[str] = None,
                       z_label: Optional[str] = None,
                       save_html: Optional[str] = None):
    """
    Interactive Plotly 3D surface (MATLAB-like Jet by default).

    Requires the optional ``plotly`` dependency::

        pip install mqqr[plotly]
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "plotly is required.  pip install mqqr[plotly]") from e

    Z = _matrix_from_result(result, value)
    yq = list(np.asarray(result.y_quantiles, dtype=float))
    xq = list(np.asarray(result.x_quantiles, dtype=float))

    if title is None:
        title = getattr(result, "method", "QQ Regression")
    if x_label is None:
        x_label = f"τ  ({getattr(result, 'x_name', 'X')})"
    if y_label is None:
        y_label = f"θ  ({getattr(result, 'y_name', 'Y')})"
    if z_label is None:
        z_label = {"beta1": "β1(θ,τ)", "r_squared": "R²",
                   "p_value": "p", "t_value": "t"}.get(value, value)

    # Plotly built-in scales (Jet / Parula not built-in; map appropriately)
    cmap_map = {"jet": "Jet", "parula": "Viridis", "turbo": "Turbo",
                "matlab_jet": "Jet", "matlab_parula": "Viridis"}
    plotly_scale = cmap_map.get(str(cmap).lower(), cmap)

    fig = go.Figure(data=[
        go.Surface(
            x=xq, y=yq, z=Z,
            colorscale=plotly_scale,
            colorbar=dict(title=z_label, len=0.7),
            contours=dict(
                x=dict(show=True, color="black", width=1,
                       start=min(xq), end=max(xq),
                       size=(max(xq) - min(xq)) / max(1, len(xq) - 1)),
                y=dict(show=True, color="black", width=1,
                       start=min(yq), end=max(yq),
                       size=(max(yq) - min(yq)) / max(1, len(yq) - 1)),
                z=dict(show=False),
            ),
            lighting=dict(ambient=0.55, diffuse=0.85, specular=0.15,
                          roughness=0.85),
            lightposition=dict(x=80, y=120, z=80),
            hovertemplate=(
                f"{x_label}: %{{x:.2f}}<br>"
                f"{y_label}: %{{y:.2f}}<br>"
                f"{z_label}: %{{z:.4f}}<extra></extra>"
            ),
        )
    ])
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        margin=dict(l=10, r=10, b=10, t=60),
        scene=dict(
            xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label,
            aspectratio=dict(x=1, y=1, z=0.75),
            camera=dict(eye=dict(x=1.5, y=1.7, z=1.1)),
        ),
        paper_bgcolor="white", plot_bgcolor="white",
    )

    if save_html:
        fig.write_html(save_html)
    return fig
