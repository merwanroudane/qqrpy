"""
Multivariate Quantile-on-Quantile Regression (m-QQR).

References
----------
* Sinha, A., Ghosh, V., Hussain, N., Nguyen, D.K., Das, N. (2023).
  "Green financing of renewable energy generation: Capturing the role
  of exogenous moderation for ensuring sustainable development."
  Energy Economics, 126, 107021.
* Ozkan, O., Haruna, R.A., Alola, A.A., Ghardallou, W., Usman, O.
  (2023).  Structural Change and Economic Dynamics, 65, 382–392.
* Alola, A.A., Özkan, O., Usman, O. (2023).  Energy Economics, 120, 106613.
* Das, N., Gangopadhyay, P., Bera, P., Hossain, M.E. (2023).
  Environmental Science & Pollution Research, 30, 45796–45814.

Model
-----
Let ``y`` be the dependent variable (e.g. renewable energy generation),
``x`` the principal regressor (e.g. green finance), and ``Z = {Z_1, …, Z_p}``
exogenous moderators / controls.  m-QQR fits

    y_t = β₀(θ,τ) + β₁(θ,τ) (x_t - x^τ)
          + Σ_j γ_j(θ,τ) [x_t Z_{j,t} - x^τ Z_j^τ]
          + Σ_j α_j(θ)    (Z_{j,t} - Z_j^τ)
          + e_t^θ

at the θ-quantile of y with Gaussian kernel weights

    w_t = K( (F_n(x_t) - τ) / h ).

The slope ``β₁(θ,τ)`` is the *moderated* marginal effect of the τ-quantile
of x on the θ-quantile of y, controlling for moderator interactions.
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
    pseudo_r2,
    safe_print,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MQQResult:
    """
    Container for Multivariate Quantile-on-Quantile regression results.

    Attributes
    ----------
    main_results : DataFrame
        One row per (θ, τ).  Columns: ``y_quantile, x_quantile, beta0,
        beta1, se, t_value, p_value, r_squared``.
    interactions : DataFrame
        Long-format slope coefficients γ_j(θ, τ) for each moderator
        interaction term.  Columns include ``moderator`` and the same
        statistical fields.
    moderator_direct : DataFrame
        α_j(θ) direct moderator effects (function only of θ).
    """
    main_results: pd.DataFrame
    interactions: pd.DataFrame
    moderator_direct: pd.DataFrame
    y_quantiles: np.ndarray
    x_quantiles: np.ndarray
    n_obs: int
    bandwidth: float
    y_name: str = "Y"
    x_name: str = "X"
    moderator_names: list = field(default_factory=list)
    method: str = "Multivariate Quantile-on-Quantile Regression (m-QQR)"

    # ── Matrix views ──────────────────────────────────────────────────────

    def to_matrix(self, value: str = "beta1") -> np.ndarray:
        """
        Pivot the main slope β1(θ,τ) to a 2-D matrix.  Use ``value``
        to pick another column (e.g. ``r_squared``, ``p_value``).
        """
        m = self.main_results.pivot(index="y_quantile", columns="x_quantile",
                                    values=value)
        return m.sort_index(axis=0).sort_index(axis=1).values

    def interaction_matrix(self, moderator: str,
                           value: str = "gamma") -> np.ndarray:
        sub = self.interactions[self.interactions["moderator"] == moderator]
        if sub.empty:
            raise KeyError(f"No moderator named '{moderator}'.")
        m = sub.pivot(index="y_quantile", columns="x_quantile", values=value)
        return m.sort_index(axis=0).sort_index(axis=1).values

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self):
        r = self.main_results.dropna(subset=["beta1"])
        n1  = int((r["p_value"] < 0.01).sum())
        n5  = int((r["p_value"] < 0.05).sum())
        n10 = int((r["p_value"] < 0.10).sum())

        lines = [
            "",
            "+----------------------------------------------------+",
            "|  Multivariate Quantile-on-Quantile Reg. - Summary  |",
            "+----------------------------------------------------+",
            "",
            f"  Method            : {self.method}",
            f"  Dependent (Y)     : {self.y_name}",
            f"  Principal (X)     : {self.x_name}",
            f"  Moderators        : {', '.join(self.moderator_names) or '(none)'}",
            f"  Observations      : {self.n_obs}",
            f"  Bandwidth (h)     : {self.bandwidth:.4f}",
            f"  Y-quantiles (θ)   : {len(self.y_quantiles)}",
            f"  X-quantiles (τ)   : {len(self.x_quantiles)}",
            f"  Total cells       : {len(self.main_results)}",
            f"  Successful fits   : {len(r)}",
            "",
            "  Principal slope β1(θ,τ)",
            f"    Mean   : {r['beta1'].mean():+.4f}",
            f"    Median : {r['beta1'].median():+.4f}",
            f"    Min    : {r['beta1'].min():+.4f}",
            f"    Max    : {r['beta1'].max():+.4f}",
            "",
            "  Significance",
            f"    p < 0.10 : {n10:>4} / {len(r)}  ({100*n10/max(len(r),1):.1f}%)",
            f"    p < 0.05 : {n5:>4} / {len(r)}  ({100*n5/max(len(r),1):.1f}%)",
            f"    p < 0.01 : {n1:>4} / {len(r)}  ({100*n1/max(len(r),1):.1f}%)",
            "",
        ]
        if self.moderator_names:
            lines.append("  Moderator interactions γ_j(θ,τ)")
            for m in self.moderator_names:
                sub = self.interactions[
                    (self.interactions["moderator"] == m) &
                    (self.interactions["gamma"].notna())
                ]
                if len(sub):
                    sig = int((sub["p_value"] < 0.05).sum())
                    lines.append(
                        f"    {m:>14s} :  mean γ = {sub['gamma'].mean():+.4f}   "
                        f"sig(5%) = {sig}/{len(sub)}"
                    )
            lines.append("")
        safe_print("\n".join(lines))
        return self

    # ── Export ────────────────────────────────────────────────────────────

    def export_csv(self, prefix: str, digits: int = 4):
        """Write main, interaction and direct-moderator tables as CSVs."""
        for name, df in [
            ("main", self.main_results),
            ("interactions", self.interactions),
            ("moderators", self.moderator_direct),
        ]:
            d = df.copy()
            num = [c for c in d.columns
                   if d[c].dtype.kind in "fc" and c not in {"y_quantile",
                                                            "x_quantile"}]
            d[num] = d[num].round(digits)
            d.to_csv(f"{prefix}_{name}.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Main estimator
# ─────────────────────────────────────────────────────────────────────────────

def mqq_regression(
    y,
    x,
    moderators: Dict[str, np.ndarray],
    *,
    y_quantiles=None,
    x_quantiles=None,
    bandwidth: float = 0.05,
    include_lag: bool = True,
    interactions: bool = True,
    se: str = "bootstrap",
    n_boot: int = 150,
    cdf_based_kernel: bool = True,
    x_name: str = "X",
    y_name: str = "Y",
    verbose: bool = True,
    random_state: Optional[int] = 42,
) -> MQQResult:
    """
    Multivariate Quantile-on-Quantile regression.

    Parameters
    ----------
    y : array_like
        Dependent variable.
    x : array_like
        Principal regressor whose τ-quantile drives the kernel weights.
    moderators : dict[str, array_like]
        Exogenous moderators / controls, e.g. ``{'EPU': z1, 'UNEMP': z2}``.
    y_quantiles, x_quantiles : array_like or None
        Grids of θ, τ in (0, 1).
    bandwidth : float
        Kernel bandwidth on the empirical-CDF scale.
    include_lag : bool
        Include y_{t-1} as a control (Sim & Zhou setting).
    interactions : bool
        If True, include x_t · Z_{j,t} interaction terms (full Sinha 2023
        specification).  Set to False for a simpler partial-effects model.
    se : {'bootstrap', 'none'}
    n_boot : int
    cdf_based_kernel : bool
    x_name, y_name : str
    verbose : bool
    random_state : int

    Returns
    -------
    MQQResult
    """
    y = coerce_array(y, "y")
    x = coerce_array(x, "x")
    if len(y) != len(x):
        raise ValueError("y and x must have equal length")

    Z_names = list(moderators.keys())
    Z_arrs = []
    for name in Z_names:
        a = coerce_array(moderators[name], name)
        if len(a) != len(y):
            raise ValueError(f"moderator '{name}' has length {len(a)} != {len(y)}")
        Z_arrs.append(a)
    Z = np.column_stack(Z_arrs) if Z_arrs else np.zeros((len(y), 0))

    mask = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(Z), axis=1)
    y = y[mask]
    x = x[mask]
    Z = Z[mask]
    n = len(y)
    if n < 40:
        raise ValueError("need at least 40 observations after NaN-removal")

    if y_quantiles is None:
        y_quantiles = standard_quantile_grid()
    if x_quantiles is None:
        x_quantiles = standard_quantile_grid()
    y_quantiles = np.asarray(y_quantiles, dtype=np.float64)
    x_quantiles = np.asarray(x_quantiles, dtype=np.float64)

    if include_lag:
        y_resp = y[1:]
        x_main = x[1:]
        Z_t = Z[1:]
        y_lag = y[:-1]
    else:
        y_resp = y
        x_main = x
        Z_t = Z
        y_lag = None
    n_eff = len(y_resp)
    p = Z_t.shape[1]

    if verbose:
        print("Multivariate Quantile-on-Quantile Regression  (m-QQR)")
        print(f"  n = {n_eff},  Y q-grid = {len(y_quantiles)},  "
              f"X q-grid = {len(x_quantiles)},  h = {bandwidth},  "
              f"moderators = {Z_names or '(none)'}, "
              f"interactions = {interactions}")

    rng = np.random.default_rng(random_state)
    main_records = []
    interaction_records = []
    moderator_records = []
    total = len(y_quantiles) * len(x_quantiles)
    done = 0
    pct_marker = max(1, total // 10)

    for tau in x_quantiles:
        x_tau = float(np.quantile(x_main, tau))
        z_x = x_main - x_tau                                           # (n,)
        w = qq_weights(x_main, tau, h=bandwidth, cdf_based=cdf_based_kernel)

        # Centred moderators at their own τ-quantiles
        Z_tau = np.array([np.quantile(Z_t[:, j], tau) for j in range(p)]) \
            if p > 0 else np.zeros(0)
        z_Z = Z_t - Z_tau if p > 0 else Z_t           # (n, p)

        # Interaction columns x_t * Z_{j,t}  centred at x^τ * Z_j^τ
        if interactions and p > 0:
            inter = x_main[:, None] * Z_t - (x_tau * Z_tau)            # (n, p)
        else:
            inter = np.zeros((n_eff, 0))

        # Build design matrix
        blocks = [np.ones(n_eff), z_x]
        if y_lag is not None:
            blocks.append(y_lag)
        if p > 0:
            blocks.append(z_Z)
        if inter.shape[1] > 0:
            blocks.append(inter)
        X_mat = np.column_stack(blocks)

        # Column-index map
        idx = {"intercept": 0, "beta1": 1}
        col = 2
        if y_lag is not None:
            idx["lag"] = col; col += 1
        if p > 0:
            for j, name in enumerate(Z_names):
                idx[f"alpha_{name}"] = col + j
            col += p
        if inter.shape[1] > 0:
            for j, name in enumerate(Z_names):
                idx[f"gamma_{name}"] = col + j
            col += p

        for theta in y_quantiles:
            done += 1
            base_rec = dict(
                y_quantile=float(round(theta, 4)),
                x_quantile=float(round(tau, 4)),
            )
            main = {**base_rec,
                    "beta0": np.nan, "beta1": np.nan, "se": np.nan,
                    "t_value": np.nan, "p_value": np.nan, "r_squared": np.nan,
                    "n_eff": int((w > 1e-8).sum())}
            try:
                fit = weighted_quantile_regression(y_resp, X_mat, theta, w)
                if fit["success"]:
                    coef = fit["coef"]
                    main["beta0"] = float(coef[idx["intercept"]])
                    main["beta1"] = float(coef[idx["beta1"]])
                    y_hat = X_mat @ coef
                    main["r_squared"] = pseudo_r2(y_resp, y_hat, theta, w)

                    if se == "bootstrap":
                        se_vec, _ = bootstrap_wqr_se(
                            y_resp, X_mat, theta, weights=w,
                            n_boot=n_boot, rng=rng,
                        )
                        s_b1 = se_vec[idx["beta1"]]
                        if np.isfinite(s_b1) and s_b1 > 0:
                            main["se"] = float(s_b1)
                            main["t_value"] = float(coef[idx["beta1"]] / s_b1)
                            df_resid = max(1, n_eff - X_mat.shape[1])
                            main["p_value"] = float(
                                2.0 * sp_stats.t.sf(abs(main["t_value"]), df=df_resid)
                            )

                        # Interaction γ_j and direct α_j fields
                        if interactions and p > 0:
                            for j, name in enumerate(Z_names):
                                k = idx[f"gamma_{name}"]
                                gamma = float(coef[k])
                                s_g = se_vec[k]
                                rec = {**base_rec, "moderator": name,
                                       "gamma": gamma}
                                if np.isfinite(s_g) and s_g > 0:
                                    rec["se"] = float(s_g)
                                    t = gamma / s_g
                                    rec["t_value"] = float(t)
                                    rec["p_value"] = float(
                                        2.0 * sp_stats.t.sf(abs(t), df=df_resid)
                                    )
                                else:
                                    rec["se"] = np.nan
                                    rec["t_value"] = np.nan
                                    rec["p_value"] = np.nan
                                interaction_records.append(rec)

                        if p > 0:
                            for j, name in enumerate(Z_names):
                                k = idx[f"alpha_{name}"]
                                alpha = float(coef[k])
                                s_a = se_vec[k]
                                rec = {**base_rec, "moderator": name,
                                       "alpha": alpha}
                                if np.isfinite(s_a) and s_a > 0:
                                    rec["se"] = float(s_a)
                                    t = alpha / s_a
                                    rec["t_value"] = float(t)
                                    rec["p_value"] = float(
                                        2.0 * sp_stats.t.sf(abs(t), df=df_resid)
                                    )
                                else:
                                    rec["se"] = np.nan
                                    rec["t_value"] = np.nan
                                    rec["p_value"] = np.nan
                                moderator_records.append(rec)
            except Exception:
                pass

            main_records.append(main)

            if verbose and done % pct_marker == 0:
                print(f"  Progress: {100 * done // total:3d}%")

    if verbose:
        print("  done.")

    main_df = pd.DataFrame(main_records).sort_values(
        ["y_quantile", "x_quantile"]).reset_index(drop=True)
    inter_df = (pd.DataFrame(interaction_records)
                .sort_values(["moderator", "y_quantile", "x_quantile"])
                .reset_index(drop=True)
                if interaction_records else
                pd.DataFrame(columns=["y_quantile", "x_quantile", "moderator",
                                      "gamma", "se", "t_value", "p_value"]))
    mod_df = (pd.DataFrame(moderator_records)
              .sort_values(["moderator", "y_quantile", "x_quantile"])
              .reset_index(drop=True)
              if moderator_records else
              pd.DataFrame(columns=["y_quantile", "x_quantile", "moderator",
                                    "alpha", "se", "t_value", "p_value"]))

    return MQQResult(
        main_results=main_df,
        interactions=inter_df,
        moderator_direct=mod_df,
        y_quantiles=y_quantiles,
        x_quantiles=x_quantiles,
        n_obs=n_eff,
        bandwidth=bandwidth,
        y_name=y_name,
        x_name=x_name,
        moderator_names=Z_names,
    )
