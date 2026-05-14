"""
Bivariate Quantile-on-Quantile Regression (Sim & Zhou, 2015).

Reference
---------
Sim, N. & Zhou, H. (2015).  "Oil prices, US stock return, and the
dependence between their quantiles."  Journal of Banking & Finance, 55, 1–12.
DOI: 10.1016/j.jbankfin.2015.01.013

Model
-----
For each (θ, τ) ∈ (0,1)² we estimate the local linear quantile-regression

    r_t = β₀(θ, τ) + β₁(θ, τ) · (x_t - x^τ) + α(θ) · r_{t-1} + v_t^θ

at the θ-quantile of r_t with Gaussian kernel weights based on the
empirical-CDF distance of x_t from its τ-quantile:

    w_t = K( (F_n(x_t) - τ) / h ).

The slope ``β₁(θ, τ)`` quantifies how the τ-quantile of x affects the
θ-quantile of y.  Sim & Zhou recommend a plug-in bandwidth h = 0.05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats as sp_stats

from .utils import (
    coerce_array,
    standard_quantile_grid,
    qq_weights,
    weighted_quantile_regression,
    bootstrap_wqr_se,
    pseudo_r2,
    safe_print,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QQResult:
    """
    Container for bivariate Quantile-on-Quantile regression results.

    Attributes
    ----------
    results : pandas.DataFrame
        Columns: ``y_quantile, x_quantile, beta0, beta1, se, t_value,
        p_value, r_squared``.
    y_quantiles, x_quantiles : ndarray
        Estimation grids.
    n_obs : int
    bandwidth : float
    method : str
    x_name, y_name : str
    """
    results: pd.DataFrame
    y_quantiles: np.ndarray
    x_quantiles: np.ndarray
    n_obs: int
    bandwidth: float
    x_name: str = "X"
    y_name: str = "Y"
    method: str = "Quantile-on-Quantile Regression (Sim & Zhou, 2015)"

    # ── Matrix views ──────────────────────────────────────────────────────

    def to_matrix(self, value: str = "beta1") -> np.ndarray:
        """
        Pivot results into a (len(y_quantiles), len(x_quantiles)) ndarray.
        Rows are sorted by θ ascending, columns by τ ascending.
        """
        if value not in self.results.columns:
            raise ValueError(f"unknown value '{value}'. "
                             f"available: {list(self.results.columns)}")
        m = self.results.pivot(index="y_quantile", columns="x_quantile",
                               values=value)
        return m.sort_index(axis=0).sort_index(axis=1).values

    def to_dataframe(self) -> pd.DataFrame:
        """Return full results DataFrame (copy)."""
        return self.results.copy()

    # ── QR-equivalent collapse (Sim & Zhou validation check) ──────────────

    def average_over_tau(self) -> pd.DataFrame:
        """
        Compute τ-averaged QQ coefficients.  These should approximately
        recover the conventional quantile-regression estimates (figure 5
        of Sim & Zhou, 2015).
        """
        return (
            self.results
                .groupby("y_quantile")[["beta0", "beta1"]]
                .mean()
                .reset_index()
                .sort_values("y_quantile")
        )

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self):
        r = self.results.dropna(subset=["beta1"])
        n5  = int((r["p_value"] < 0.05).sum())
        n10 = int((r["p_value"] < 0.10).sum())
        n1  = int((r["p_value"] < 0.01).sum())

        lines = [
            "",
            "+----------------------------------------------------+",
            "|   Quantile-on-Quantile Regression  --  Summary     |",
            "+----------------------------------------------------+",
            "",
            f"  Method            : {self.method}",
            f"  Dependent (Y)     : {self.y_name}",
            f"  Independent (X)   : {self.x_name}",
            f"  Observations      : {self.n_obs}",
            f"  Bandwidth (h)     : {self.bandwidth:.4f}",
            f"  Y-quantiles (θ)   : {len(self.y_quantiles)}  "
            f"[{self.y_quantiles[0]:.2f} … {self.y_quantiles[-1]:.2f}]",
            f"  X-quantiles (τ)   : {len(self.x_quantiles)}  "
            f"[{self.x_quantiles[0]:.2f} … {self.x_quantiles[-1]:.2f}]",
            f"  Total cells       : {len(self.results)}",
            f"  Successful fits   : {len(r)}",
            "",
            "  Slope β1(θ,τ)",
            f"    Mean   : {r['beta1'].mean():+.4f}",
            f"    Median : {r['beta1'].median():+.4f}",
            f"    Min    : {r['beta1'].min():+.4f}",
            f"    Max    : {r['beta1'].max():+.4f}",
            f"    Std    : {r['beta1'].std():.4f}",
            "",
            "  Pseudo R^2",
            f"    Mean   : {r['r_squared'].mean():.4f}",
            f"    Median : {r['r_squared'].median():.4f}",
            "",
            "  Significance of β1(θ,τ)",
            f"    p < 0.10 : {n10:>4} / {len(r)}  ({100*n10/max(len(r),1):.1f}%)",
            f"    p < 0.05 : {n5:>4} / {len(r)}  ({100*n5/max(len(r),1):.1f}%)",
            f"    p < 0.01 : {n1:>4} / {len(r)}  ({100*n1/max(len(r),1):.1f}%)",
            "",
        ]
        safe_print("\n".join(lines))
        return self

    # ── Export ────────────────────────────────────────────────────────────

    def export_csv(self, path: str, digits: int = 4):
        df = self.results.copy()
        for c in ["beta0", "beta1", "se", "t_value", "p_value", "r_squared"]:
            if c in df.columns:
                df[c] = df[c].round(digits)
        df.to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Main estimator
# ─────────────────────────────────────────────────────────────────────────────

def qq_regression(
    y,
    x,
    *,
    y_quantiles=None,
    x_quantiles=None,
    bandwidth: float = 0.05,
    include_lag: bool = True,
    se: str = "bootstrap",
    n_boot: int = 200,
    cdf_based_kernel: bool = True,
    x_name: str = "X",
    y_name: str = "Y",
    verbose: bool = True,
    random_state: Optional[int] = 42,
) -> QQResult:
    """
    Bivariate Quantile-on-Quantile regression (Sim & Zhou, 2015).

    Parameters
    ----------
    y, x : array_like
        Dependent and independent series of equal length.
    y_quantiles, x_quantiles : array_like or None
        Estimation grids on (0, 1).  Defaults to ``arange(0.05, 1, 0.05)``.
    bandwidth : float
        Kernel bandwidth on the empirical-CDF scale.  Default 0.05
        (Sim & Zhou plug-in).
    include_lag : bool
        If True the lagged dependent r_{t-1} enters the local model
        (Sim & Zhou's equation 6).
    se : {'bootstrap', 'none'}
        Standard-error procedure.  ``'bootstrap'`` uses paired bootstrap
        with ``n_boot`` replications.
    n_boot : int
        Bootstrap replications.
    cdf_based_kernel : bool
        Use F_n-distance kernel (default) or raw distance from x_τ.
    x_name, y_name : str
        Labels for printed/plotted output.
    verbose : bool
    random_state : int

    Returns
    -------
    QQResult
    """
    y = coerce_array(y, "y")
    x = coerce_array(x, "x")
    if len(y) != len(x):
        raise ValueError("y and x must have equal length")

    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    n = len(y)
    if n < 30:
        raise ValueError("need at least 30 observations after NaN-removal")

    if y_quantiles is None:
        y_quantiles = standard_quantile_grid()
    if x_quantiles is None:
        x_quantiles = standard_quantile_grid()
    y_quantiles = np.asarray(y_quantiles, dtype=np.float64)
    x_quantiles = np.asarray(x_quantiles, dtype=np.float64)

    if include_lag:
        y_resp = y[1:]
        x_main = x[1:]
        y_lag = y[:-1]
    else:
        y_resp = y
        x_main = x
        y_lag = None
    n_eff = len(y_resp)

    if verbose:
        print(f"Quantile-on-Quantile Regression  (Sim & Zhou, 2015)")
        print(f"  n = {n_eff},  Y q-grid = {len(y_quantiles)},  "
              f"X q-grid = {len(x_quantiles)},  h = {bandwidth}")

    rng = np.random.default_rng(random_state)
    records = []
    total = len(y_quantiles) * len(x_quantiles)
    done = 0
    pct_marker = max(1, total // 10)

    for tau in x_quantiles:
        x_tau = float(np.quantile(x_main, tau))
        z = x_main - x_tau
        w = qq_weights(x_main, tau, h=bandwidth, cdf_based=cdf_based_kernel)

        # Design matrix : [1, z, lag]  or [1, z]
        if include_lag:
            X_mat = np.column_stack([np.ones(n_eff), z, y_lag])
        else:
            X_mat = np.column_stack([np.ones(n_eff), z])

        for theta in y_quantiles:
            done += 1
            rec = dict(
                y_quantile=float(round(theta, 4)),
                x_quantile=float(round(tau, 4)),
                beta0=np.nan, beta1=np.nan,
                se=np.nan, t_value=np.nan, p_value=np.nan, r_squared=np.nan,
                n_eff=int((w > 1e-8).sum()),
            )
            try:
                fit = weighted_quantile_regression(y_resp, X_mat, theta, w)
                if fit["success"]:
                    coef = fit["coef"]
                    rec["beta0"] = float(coef[0])
                    rec["beta1"] = float(coef[1])

                    y_hat = X_mat @ coef
                    rec["r_squared"] = pseudo_r2(y_resp, y_hat, theta, w)

                    if se == "bootstrap":
                        se_vec, _ = bootstrap_wqr_se(
                            y_resp, X_mat, theta, weights=w,
                            n_boot=n_boot, rng=rng,
                        )
                        if np.isfinite(se_vec[1]) and se_vec[1] > 0:
                            rec["se"] = float(se_vec[1])
                            rec["t_value"] = float(coef[1] / se_vec[1])
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

    return QQResult(
        results=df,
        y_quantiles=y_quantiles,
        x_quantiles=x_quantiles,
        n_obs=n_eff,
        bandwidth=bandwidth,
        x_name=x_name,
        y_name=y_name,
    )
