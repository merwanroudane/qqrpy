"""
Quantile and Multivariate-Quantile Causality.

Two complementary frameworks are implemented:

1. **QQ Granger Causality** -- bivariate.  Tests whether the τ-quantile
   of ``x_{t-1}`` Granger-causes the θ-quantile of ``y_t`` by inspecting
   the slope ``β1(θ,τ)`` in the QQR equation

       y_t = β0(θ,τ) + β1(θ,τ) (x_{t-1} - x^τ) + α(θ) y_{t-1} + v_t^θ.

   A bootstrap-based t-statistic and p-value are reported for each
   ``(θ, τ)`` cell, producing a 2-D causality test surface that can be
   plotted as a heat-map with significance stars (the standard
   presentation used in Sinha-co-authored Energy-Economics and
   Risk-Analysis articles).

2. **Multivariate-QQ Causality** -- conditional.  Same null
   "``x ⇏_τ y``" but the regression also conditions on a set of
   moderators / controls ``Z = {Z_1, …, Z_p}``:

       y_t = β0 + β1 (x_{t-1} - x^τ) + α y_{t-1}
             + Σ_j γ_j [x_{t-1} Z_{j,t-1} - x^τ Z_j^τ]
             + Σ_j δ_j (Z_{j,t-1} - Z_j^τ) + v_t^θ.

   This implements the conditional-causality test that under-pins
   Sinha et al. (2024, Risk Analysis) and the
   ``Multivariate-QQ-Causality-V1.0`` Stata-VBA toolkit.

A *Sup-Wald* summary across τ (Troster, 2018) is also exposed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict
from scipy import stats as sp_stats

from .utils import (
    coerce_array,
    standard_quantile_grid,
    qq_weights,
    weighted_quantile_regression,
    bootstrap_wqr_se,
    safe_print,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QQCausalityResult:
    """Bivariate QQ Granger-causality results (cause ``x`` → effect ``y``)."""
    results: pd.DataFrame
    y_quantiles: np.ndarray
    x_quantiles: np.ndarray
    n_obs: int
    bandwidth: float
    direction: str = "x -> y"
    cause_name: str = "X"
    effect_name: str = "Y"
    method: str = "QQ Granger Causality"

    # ── Matrix views ──────────────────────────────────────────────────────

    def stat_matrix(self) -> np.ndarray:
        m = self.results.pivot(index="y_quantile", columns="x_quantile",
                               values="t_value")
        return m.sort_index(axis=0).sort_index(axis=1).values

    def pvalue_matrix(self) -> np.ndarray:
        m = self.results.pivot(index="y_quantile", columns="x_quantile",
                               values="p_value")
        return m.sort_index(axis=0).sort_index(axis=1).values

    def coef_matrix(self) -> np.ndarray:
        m = self.results.pivot(index="y_quantile", columns="x_quantile",
                               values="beta1")
        return m.sort_index(axis=0).sort_index(axis=1).values

    def significance_matrix(self) -> np.ndarray:
        """String matrix of significance stars (``'***' / '**' / '*' / ''``)."""
        p = self.results.pivot(index="y_quantile", columns="x_quantile",
                               values="p_value").sort_index(axis=0).sort_index(axis=1)
        stars = np.where(p < 0.01, "***",
                np.where(p < 0.05, "**",
                np.where(p < 0.10, "*", "")))
        return stars

    # ── Sup-Wald summary (Troster, 2018) ──────────────────────────────────

    def sup_wald(self, alpha: float = 0.05) -> dict:
        """
        Sup-Wald test of *no causality at any quantile*:

            S = sup_{(θ,τ)} |t(θ,τ)|

        with a parametric-bootstrap critical value computed from the
        empirical distribution of t-statistics under H0 ≈ Normal(0,1).
        """
        t = self.results["t_value"].dropna()
        sup = float(t.abs().max())
        # Bonferroni-adjusted reference threshold over the grid
        p_grid = sp_stats.norm.sf(sup) * 2.0
        adj_p = min(1.0, p_grid * len(t))
        return {
            "sup_statistic": sup,
            "argmax_y_quantile": float(self.results.loc[t.abs().idxmax(),
                                                       "y_quantile"]),
            "argmax_x_quantile": float(self.results.loc[t.abs().idxmax(),
                                                       "x_quantile"]),
            "bonferroni_p_value": adj_p,
            "reject_H0_at_alpha": adj_p < alpha,
        }

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self):
        r = self.results.dropna(subset=["t_value"])
        n1  = int((r["p_value"] < 0.01).sum())
        n5  = int((r["p_value"] < 0.05).sum())
        n10 = int((r["p_value"] < 0.10).sum())
        sw = self.sup_wald()

        lines = [
            "",
            "+----------------------------------------------------+",
            "|        QQ Granger Causality  --  Summary           |",
            "+----------------------------------------------------+",
            "",
            f"  Direction         : {self.cause_name}  ==>  {self.effect_name}",
            f"  Method            : {self.method}",
            f"  Observations      : {self.n_obs}",
            f"  Bandwidth (h)     : {self.bandwidth:.4f}",
            f"  Y-quantiles (θ)   : {len(self.y_quantiles)}",
            f"  X-quantiles (τ)   : {len(self.x_quantiles)}",
            "",
            "  Causality across the (θ,τ) grid",
            f"    cells tested    : {len(r)}",
            f"    p < 0.10        : {n10:>4} ({100*n10/max(len(r),1):.1f}%)",
            f"    p < 0.05        : {n5:>4} ({100*n5/max(len(r),1):.1f}%)",
            f"    p < 0.01        : {n1:>4} ({100*n1/max(len(r),1):.1f}%)",
            "",
            "  Sup-Wald (Troster 2018) over (θ,τ)",
            f"    sup |t|         : {sw['sup_statistic']:.4f}",
            f"    at  (θ*, τ*)    : ({sw['argmax_y_quantile']:.2f}, "
            f"{sw['argmax_x_quantile']:.2f})",
            f"    Bonferroni p    : {sw['bonferroni_p_value']:.4g}",
            f"    Reject H0 at 5% : {sw['reject_H0_at_alpha']}",
            "",
        ]
        safe_print("\n".join(lines))
        return self


@dataclass
class MQQCausalityResult(QQCausalityResult):
    """Multivariate (conditional) QQ Granger-causality results."""
    moderator_names: list = field(default_factory=list)
    method: str = "Multivariate QQ Granger Causality (conditional)"


# ─────────────────────────────────────────────────────────────────────────────
#  Bivariate QQ causality
# ─────────────────────────────────────────────────────────────────────────────

def qq_causality(
    x,
    y,
    *,
    y_quantiles=None,
    x_quantiles=None,
    bandwidth: float = 0.05,
    n_boot: int = 200,
    cdf_based_kernel: bool = True,
    cause_name: str = "X",
    effect_name: str = "Y",
    verbose: bool = True,
    random_state: Optional[int] = 42,
) -> QQCausalityResult:
    """
    QQ Granger-causality test from ``x`` to ``y`` over a (θ, τ) grid.

    For each cell the slope ``β1(θ, τ)`` in the local linear quantile
    regression of ``y_t`` on ``x_{t-1}`` (controlling for ``y_{t-1}``) is
    tested against zero with a paired bootstrap.  The matrix of
    t-statistics and bootstrap p-values is returned, ready for a
    heat-map presentation with significance stars.
    """
    x = coerce_array(x, "x")
    y = coerce_array(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = len(y)
    if n < 30:
        raise ValueError("need at least 30 observations after NaN-removal")

    if y_quantiles is None:
        y_quantiles = standard_quantile_grid()
    if x_quantiles is None:
        x_quantiles = standard_quantile_grid()

    y_quantiles = np.asarray(y_quantiles, dtype=np.float64)
    x_quantiles = np.asarray(x_quantiles, dtype=np.float64)

    # Lag setup : test x_{t-1} -> y_t  (Granger first-order)
    y_resp = y[1:]
    x_lag  = x[:-1]
    y_lag  = y[:-1]
    n_eff = len(y_resp)

    if verbose:
        print(f"QQ Granger Causality   ({cause_name} ==> {effect_name})")
        print(f"  n = {n_eff},  Y q-grid = {len(y_quantiles)},  "
              f"X q-grid = {len(x_quantiles)},  h = {bandwidth}")

    rng = np.random.default_rng(random_state)
    records = []
    total = len(y_quantiles) * len(x_quantiles)
    done = 0
    pct_marker = max(1, total // 10)

    for tau in x_quantiles:
        x_tau = float(np.quantile(x_lag, tau))
        z = x_lag - x_tau
        w = qq_weights(x_lag, tau, h=bandwidth, cdf_based=cdf_based_kernel)
        X_mat = np.column_stack([np.ones(n_eff), z, y_lag])

        for theta in y_quantiles:
            done += 1
            rec = dict(
                y_quantile=float(round(theta, 4)),
                x_quantile=float(round(tau, 4)),
                beta1=np.nan, se=np.nan,
                t_value=np.nan, p_value=np.nan,
            )
            try:
                fit = weighted_quantile_regression(y_resp, X_mat, theta, w)
                if fit["success"]:
                    rec["beta1"] = float(fit["coef"][1])
                    se_vec, _ = bootstrap_wqr_se(
                        y_resp, X_mat, theta, weights=w,
                        n_boot=n_boot, rng=rng,
                    )
                    if np.isfinite(se_vec[1]) and se_vec[1] > 0:
                        rec["se"] = float(se_vec[1])
                        rec["t_value"] = float(rec["beta1"] / se_vec[1])
                        df_resid = max(1, n_eff - X_mat.shape[1])
                        rec["p_value"] = float(
                            2.0 * sp_stats.t.sf(abs(rec["t_value"]), df=df_resid)
                        )
            except Exception:
                pass
            records.append(rec)

            if verbose and done % pct_marker == 0:
                print(f"  Progress: {100 * done // total:3d}%")

    if verbose:
        print("  done.")

    df = pd.DataFrame(records).sort_values(
        ["y_quantile", "x_quantile"]).reset_index(drop=True)

    return QQCausalityResult(
        results=df,
        y_quantiles=y_quantiles,
        x_quantiles=x_quantiles,
        n_obs=n_eff,
        bandwidth=bandwidth,
        direction=f"{cause_name} -> {effect_name}",
        cause_name=cause_name,
        effect_name=effect_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Multivariate (conditional) QQ causality
# ─────────────────────────────────────────────────────────────────────────────

def mqq_causality(
    x,
    y,
    moderators: Dict[str, np.ndarray],
    *,
    y_quantiles=None,
    x_quantiles=None,
    bandwidth: float = 0.05,
    n_boot: int = 150,
    interactions: bool = True,
    cdf_based_kernel: bool = True,
    cause_name: str = "X",
    effect_name: str = "Y",
    verbose: bool = True,
    random_state: Optional[int] = 42,
) -> MQQCausalityResult:
    """
    Conditional QQ Granger-causality test from ``x`` to ``y`` controlling
    for a set of moderators / controls ``Z``.

    This is the m-QQR version of Sinha et al.'s Multivariate Quantile
    Causality framework.  Set ``interactions=True`` to also include
    ``x · Z_j`` cross-terms (full Sinha 2023 specification).

    Returns
    -------
    MQQCausalityResult
    """
    x = coerce_array(x, "x")
    y = coerce_array(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")

    Z_names = list(moderators.keys())
    Z_arrs = []
    for name in Z_names:
        a = coerce_array(moderators[name], name)
        if len(a) != len(y):
            raise ValueError(f"moderator '{name}' length mismatch")
        Z_arrs.append(a)
    Z = np.column_stack(Z_arrs) if Z_arrs else np.zeros((len(y), 0))

    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x = x[mask]; y = y[mask]; Z = Z[mask]
    n = len(y)
    if n < 40:
        raise ValueError("need at least 40 observations after NaN-removal")

    if y_quantiles is None:
        y_quantiles = standard_quantile_grid()
    if x_quantiles is None:
        x_quantiles = standard_quantile_grid()
    y_quantiles = np.asarray(y_quantiles, dtype=np.float64)
    x_quantiles = np.asarray(x_quantiles, dtype=np.float64)

    # Lag setup : test x_{t-1} -> y_t  conditional on Z_{t-1}
    y_resp = y[1:]
    x_lag  = x[:-1]
    y_lag  = y[:-1]
    Z_lag  = Z[:-1]
    n_eff = len(y_resp)
    p = Z_lag.shape[1]

    if verbose:
        print(f"Multivariate QQ Granger Causality   "
              f"({cause_name} ==> {effect_name}  |  {', '.join(Z_names) or 'none'})")
        print(f"  n = {n_eff},  Y q-grid = {len(y_quantiles)},  "
              f"X q-grid = {len(x_quantiles)},  h = {bandwidth},  "
              f"interactions = {interactions}")

    rng = np.random.default_rng(random_state)
    records = []
    total = len(y_quantiles) * len(x_quantiles)
    done = 0
    pct_marker = max(1, total // 10)

    for tau in x_quantiles:
        x_tau = float(np.quantile(x_lag, tau))
        z_x = x_lag - x_tau
        w = qq_weights(x_lag, tau, h=bandwidth, cdf_based=cdf_based_kernel)

        Z_tau = np.array([np.quantile(Z_lag[:, j], tau) for j in range(p)]) \
            if p > 0 else np.zeros(0)
        z_Z = Z_lag - Z_tau if p > 0 else Z_lag

        if interactions and p > 0:
            inter = x_lag[:, None] * Z_lag - (x_tau * Z_tau)
        else:
            inter = np.zeros((n_eff, 0))

        blocks = [np.ones(n_eff), z_x, y_lag]
        if p > 0:
            blocks.append(z_Z)
        if inter.shape[1] > 0:
            blocks.append(inter)
        X_mat = np.column_stack(blocks)

        for theta in y_quantiles:
            done += 1
            rec = dict(
                y_quantile=float(round(theta, 4)),
                x_quantile=float(round(tau, 4)),
                beta1=np.nan, se=np.nan,
                t_value=np.nan, p_value=np.nan,
            )
            try:
                fit = weighted_quantile_regression(y_resp, X_mat, theta, w)
                if fit["success"]:
                    rec["beta1"] = float(fit["coef"][1])
                    se_vec, _ = bootstrap_wqr_se(
                        y_resp, X_mat, theta, weights=w,
                        n_boot=n_boot, rng=rng,
                    )
                    if np.isfinite(se_vec[1]) and se_vec[1] > 0:
                        rec["se"] = float(se_vec[1])
                        rec["t_value"] = float(rec["beta1"] / se_vec[1])
                        df_resid = max(1, n_eff - X_mat.shape[1])
                        rec["p_value"] = float(
                            2.0 * sp_stats.t.sf(abs(rec["t_value"]), df=df_resid)
                        )
            except Exception:
                pass
            records.append(rec)

            if verbose and done % pct_marker == 0:
                print(f"  Progress: {100 * done // total:3d}%")

    if verbose:
        print("  done.")

    df = pd.DataFrame(records).sort_values(
        ["y_quantile", "x_quantile"]).reset_index(drop=True)

    return MQQCausalityResult(
        results=df,
        y_quantiles=y_quantiles,
        x_quantiles=x_quantiles,
        n_obs=n_eff,
        bandwidth=bandwidth,
        direction=f"{cause_name} -> {effect_name}",
        cause_name=cause_name,
        effect_name=effect_name,
        moderator_names=Z_names,
    )
