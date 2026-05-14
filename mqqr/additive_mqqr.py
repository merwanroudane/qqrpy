"""
Additive Multivariate Quantile-on-Quantile Regression  (m-QQR, Type 1).

Reference
---------
Alola, A.A., Ozkan, O., Usman, O. (2023).  "Examining crude oil price
outlook amidst substitute energy price and household energy expenditure
in the USA: A novel nonparametric multivariate QQR approach."
*Energy Economics* 120, 106613.

Companion to Sinha-style m-QQR (interaction / moderator framework in
``mqqr.py``).  The two formulations differ:

* **Type 1 (Alola, additive)**  -- no interaction terms; each regressor
  has its OWN quantile Phi_i; output is *n* separate 3-D surfaces, one
  per regressor.
* **Type 2 (Sinha, moderation)** -- one principal regressor with quantile
  tau, exogenous moderators enter linearly AND as interaction terms
  x*Z; output is a single principal 3-D surface plus moderation
  surfaces.

Model
-----
For dependent variable y_t and n regressors (x_{1,t}, ..., x_{n,t}):

    y_t = beta_0(theta, Phi_1, ..., Phi_n)
          + sum_{i=1}^{n} beta_i(theta, Phi_i) * (x_{i,t} - x_i^{Phi_i})
          + alpha^theta * y_{t-1}
          + e_t^theta

Estimation for the i-th surface
-------------------------------
For each focal variable i and each (theta, Phi) cell:

1. Compute Gaussian kernel weights local in x_i at the Phi-quantile:
       w_t = K( (F_{n,i}(x_{i,t}) - Phi) / h )
2. Build design matrix with ALL regressors centred at their own
   Phi-quantile (so the slope on x_i has a "partialled-out" reading).
3. Solve the weighted theta-quantile regression and extract the
   coefficient on x_i.

Repeating across all i gives n surfaces  beta_i(theta, Phi)  that can be
plotted side-by-side (Alola Fig. 3 style).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
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


# ---------------------------------------------------------------------------
#  Result container
# ---------------------------------------------------------------------------

@dataclass
class AdditiveMQQResult:
    """
    Container for the Alola-style additive multivariate QQR.

    Attributes
    ----------
    surfaces : dict[str, DataFrame]
        For each regressor x_i, a long-format DataFrame with columns
        ``y_quantile, x_quantile, beta, se, t_value, p_value,
        r_squared, n_eff``.
    y_quantiles, x_quantiles : ndarray
        Quantile grids (theta, Phi).
    n_obs : int
    bandwidth : float
    y_name : str
    x_names : list[str]
    include_lag : bool
    method : str
    """
    surfaces: Dict[str, pd.DataFrame]
    y_quantiles: np.ndarray
    x_quantiles: np.ndarray
    n_obs: int
    bandwidth: float
    y_name: str = "Y"
    x_names: List[str] = field(default_factory=list)
    include_lag: bool = True
    method: str = "Additive Multivariate QQR (Alola et al. 2023)"

    # -- accessors -------------------------------------------------------

    def get(self, variable: str) -> pd.DataFrame:
        """Return long-format DataFrame for one regressor."""
        if variable not in self.surfaces:
            raise KeyError(
                f"unknown variable '{variable}'.  Available: {self.x_names}"
            )
        return self.surfaces[variable]

    def to_matrix(self, variable: str, value: str = "beta") -> np.ndarray:
        """Pivot the slope (or any column) for one regressor to (n_y, n_x)."""
        df = self.get(variable)
        m = df.pivot(index="y_quantile", columns="x_quantile", values=value)
        return m.sort_index(axis=0).sort_index(axis=1).values

    def significance_matrix(self, variable: str,
                            alpha: float = 0.05) -> np.ndarray:
        """Boolean significance mask (p_value < alpha)."""
        p = self.to_matrix(variable, "p_value")
        return (p < alpha) & np.isfinite(p)

    # -- summary ---------------------------------------------------------

    def summary(self):
        lines = [
            "",
            "+----------------------------------------------------+",
            "|  Additive m-QQR  (Alola et al. 2023)  - Summary    |",
            "+----------------------------------------------------+",
            "",
            f"  Dependent (Y)     : {self.y_name}",
            f"  Regressors        : {', '.join(self.x_names)}",
            f"  Observations      : {self.n_obs}",
            f"  Bandwidth (h)     : {self.bandwidth:.4f}",
            f"  Y-quantiles (theta): {len(self.y_quantiles)}",
            f"  X-quantiles (Phi) : {len(self.x_quantiles)}",
            f"  Y-lag included    : {self.include_lag}",
            "",
            "  Per-regressor slope summary",
        ]
        for name in self.x_names:
            df = self.get(name).dropna(subset=["beta"])
            if df.empty:
                lines.append(f"    {name:>14s} :  (no successful fits)")
                continue
            sig5 = int((df["p_value"] < 0.05).sum())
            lines.append(
                f"    {name:>14s} :  mean = {df['beta'].mean():+.4f}   "
                f"median = {df['beta'].median():+.4f}   "
                f"sig(5%) = {sig5}/{len(df)}"
            )
        lines.append("")
        safe_print("\n".join(lines))
        return self

    # -- export ----------------------------------------------------------

    def export_csv(self, prefix: str, digits: int = 4):
        """Write one CSV per regressor surface."""
        for name, df in self.surfaces.items():
            d = df.copy()
            num = [c for c in d.columns
                   if d[c].dtype.kind in "fc"
                   and c not in {"y_quantile", "x_quantile"}]
            d[num] = d[num].round(digits)
            safe_name = name.replace("/", "_").replace(" ", "_")
            d.to_csv(f"{prefix}_{safe_name}.csv", index=False)


# ---------------------------------------------------------------------------
#  Main estimator
# ---------------------------------------------------------------------------

def additive_mqq_regression(
    y,
    X: Dict[str, np.ndarray],
    *,
    y_quantiles=None,
    x_quantiles=None,
    bandwidth: float = 0.05,
    include_lag: bool = True,
    se: str = "bootstrap",
    n_boot: int = 150,
    cdf_based_kernel: bool = True,
    y_name: str = "Y",
    verbose: bool = True,
    random_state: Optional[int] = 42,
) -> AdditiveMQQResult:
    """
    Additive multivariate QQR (Alola, Ozkan & Usman, 2023).

    Each regressor x_i is given its own quantile axis Phi_i, producing
    n separate 3-D surfaces beta_i(theta, Phi).  There are NO interaction
    terms (contrast with the Sinha-style m-QQR in :func:`mqq_regression`).

    Parameters
    ----------
    y : array_like
        Dependent variable.
    X : dict[str, array_like]
        Mapping ``name -> series`` of n regressors.
    y_quantiles, x_quantiles : array_like or None
        Grids of theta and Phi in (0, 1).
    bandwidth : float
        Kernel bandwidth on the empirical-CDF scale (Alola's choice: 0.05).
    include_lag : bool
        Include y_{t-1} as a control.
    se : {'bootstrap', 'none'}
        Standard-error scheme.
    n_boot : int
        Bootstrap replications when ``se='bootstrap'``.
    cdf_based_kernel : bool
        Use F_n(x)-based kernel (default) or raw-distance kernel.
    y_name : str
    verbose : bool
    random_state : int or None

    Returns
    -------
    AdditiveMQQResult
    """
    y = coerce_array(y, "y")
    n0 = len(y)

    if not X:
        raise ValueError("X must contain at least one regressor")
    x_names = list(X.keys())
    arrs = []
    for name in x_names:
        a = coerce_array(X[name], name)
        if len(a) != n0:
            raise ValueError(
                f"regressor '{name}' has length {len(a)} != {n0}"
            )
        arrs.append(a)
    Xmat = np.column_stack(arrs)

    mask = np.isfinite(y) & np.all(np.isfinite(Xmat), axis=1)
    y = y[mask]
    Xmat = Xmat[mask]
    n = len(y)
    if n < 40:
        raise ValueError(
            "need at least 40 observations after NaN-removal"
        )

    if y_quantiles is None:
        y_quantiles = standard_quantile_grid()
    if x_quantiles is None:
        x_quantiles = standard_quantile_grid()
    y_quantiles = np.asarray(y_quantiles, dtype=np.float64)
    x_quantiles = np.asarray(x_quantiles, dtype=np.float64)

    if include_lag:
        y_resp = y[1:]
        Xt = Xmat[1:]
        y_lag = y[:-1]
    else:
        y_resp = y
        Xt = Xmat
        y_lag = None
    n_eff = len(y_resp)
    p = Xt.shape[1]

    if verbose:
        print("Additive Multivariate QQR  (Alola et al. 2023)")
        print(
            f"  n = {n_eff},  Y q-grid = {len(y_quantiles)},  "
            f"Phi-grid = {len(x_quantiles)},  h = {bandwidth},  "
            f"regressors = {x_names},  lag = {include_lag}"
        )

    rng = np.random.default_rng(random_state)
    surfaces: Dict[str, list] = {name: [] for name in x_names}
    total = p * len(y_quantiles) * len(x_quantiles)
    done = 0
    pct_marker = max(1, total // 10)

    for i, name in enumerate(x_names):
        x_focal = Xt[:, i]

        for phi in x_quantiles:
            w = qq_weights(
                x_focal, phi, h=bandwidth, cdf_based=cdf_based_kernel
            )

            # Centre EVERY regressor at its own Phi-quantile so the
            # multivariate slope on x_i has the "partial effect"
            # interpretation of Eq. (3).
            centres = np.array(
                [np.quantile(Xt[:, j], phi) for j in range(p)]
            )
            Z = Xt - centres                                    # (n_eff, p)

            blocks = [np.ones(n_eff), Z]
            if y_lag is not None:
                blocks.append(y_lag.reshape(-1, 1))
            design = np.column_stack(blocks)

            # column of beta_i is at index 1 + i  (after intercept)
            k_focal = 1 + i

            for theta in y_quantiles:
                done += 1
                rec = dict(
                    y_quantile=float(round(theta, 4)),
                    x_quantile=float(round(phi, 4)),
                    beta=np.nan, se=np.nan,
                    t_value=np.nan, p_value=np.nan,
                    r_squared=np.nan,
                    n_eff=int((w > 1e-8).sum()),
                )
                try:
                    fit = weighted_quantile_regression(
                        y_resp, design, theta, w
                    )
                    if fit["success"]:
                        coef = fit["coef"]
                        rec["beta"] = float(coef[k_focal])
                        y_hat = design @ coef
                        rec["r_squared"] = pseudo_r2(
                            y_resp, y_hat, theta, w
                        )

                        if se == "bootstrap":
                            se_vec, _ = bootstrap_wqr_se(
                                y_resp, design, theta, weights=w,
                                n_boot=n_boot, rng=rng,
                            )
                            s = se_vec[k_focal]
                            if np.isfinite(s) and s > 0:
                                rec["se"] = float(s)
                                t = coef[k_focal] / s
                                rec["t_value"] = float(t)
                                df_resid = max(
                                    1, n_eff - design.shape[1]
                                )
                                rec["p_value"] = float(
                                    2.0 * sp_stats.t.sf(
                                        abs(t), df=df_resid
                                    )
                                )
                except Exception:
                    pass

                surfaces[name].append(rec)

                if verbose and done % pct_marker == 0:
                    print(f"  Progress: {100 * done // total:3d}%")

    if verbose:
        print("  done.")

    out = {
        name: pd.DataFrame(rows)
        .sort_values(["y_quantile", "x_quantile"])
        .reset_index(drop=True)
        for name, rows in surfaces.items()
    }

    return AdditiveMQQResult(
        surfaces=out,
        y_quantiles=y_quantiles,
        x_quantiles=x_quantiles,
        n_obs=n_eff,
        bandwidth=bandwidth,
        y_name=y_name,
        x_names=x_names,
        include_lag=include_lag,
    )
