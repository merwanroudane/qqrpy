"""
Math primitives shared across the mqqr package.

* Empirical CDF
* Gaussian kernel weights
* Weighted quantile regression (LP via HiGHS)
* Bandwidth selection
* Check (rho) function for pseudo R²
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import linprog


# ─────────────────────────────────────────────────────────────────────────────
#  Empirical CDF
# ─────────────────────────────────────────────────────────────────────────────

def empirical_cdf(x: np.ndarray) -> np.ndarray:
    """
    Empirical CDF values F_n(x_i) ∈ (0, 1] for each observation.

    Uses the mid-rank convention to avoid the boundary value 1.0
    that would receive zero kernel weight at τ = 1.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    ranks = sp_stats.rankdata(x, method="average")
    return ranks / (n + 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Gaussian kernel weights
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Standard Gaussian kernel K(u) = (2π)^{-1/2} exp(-u²/2)."""
    return np.exp(-0.5 * np.asarray(u) ** 2) / np.sqrt(2.0 * np.pi)


def qq_weights(x: np.ndarray, tau: float, h: float = 0.05,
               cdf_based: bool = True) -> np.ndarray:
    """
    Kernel weights for the Sim & Zhou (2015) QQ estimator.

    By default uses the CDF-distance kernel
    ``K((F_n(x_t) - τ) / h)`` so the bandwidth ``h`` is interpretable
    on the unit interval (h = 0.05 is the Sim & Zhou plug-in).

    Set ``cdf_based=False`` for the alternative formulation that
    weights raw distance from ``x_τ`` (used by Sinha's R script).
    """
    x = np.asarray(x, dtype=np.float64)
    if cdf_based:
        Fn = empirical_cdf(x)
        u = (Fn - tau) / h
    else:
        x_tau = np.quantile(x, tau)
        sd = np.std(x, ddof=1)
        u = (x - x_tau) / (h * sd if sd > 0 else h)
    w = gaussian_kernel(u)
    # Normalise so weights sum to n (preserves problem scale in LP)
    s = w.sum()
    if s > 0:
        w = w * (len(x) / s)
    return w


# ─────────────────────────────────────────────────────────────────────────────
#  Weighted quantile regression  (LP formulation)
# ─────────────────────────────────────────────────────────────────────────────

def weighted_quantile_regression(y, X, tau, weights=None,
                                  fit_intercept_in_X=True):
    """
    Solve  min_β  Σ w_t · ρ_τ(y_t - X_t·β)

    via an exact linear-program reformulation:

        min   Σ w_t·τ·u_t⁺ + Σ w_t·(1-τ)·u_t⁻
        s.t.  y_t = X_t β + u_t⁺ - u_t⁻
              u_t⁺, u_t⁻ ≥ 0

    Solved with ``scipy.optimize.linprog(method='highs')`` — robust and
    typically << 100 ms for n ≤ 1000.

    Parameters
    ----------
    y : (n,) array_like
    X : (n, k) array_like           Design matrix.  Must already include
                                    intercept column if needed.
    tau : float                     Quantile level in (0, 1).
    weights : (n,) array_like or None
    fit_intercept_in_X : bool       Marker that intercept is in X.

    Returns
    -------
    dict with keys ``coef`` (k,), ``residuals`` (n,), ``loss`` (float),
    ``status`` (int), ``success`` (bool).
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}.")
    n, k = X.shape

    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=np.float64).copy()
        if w.shape != (n,):
            raise ValueError(f"weights must have shape ({n},).")
        w = np.maximum(w, 0.0)

    if not (0.0 < tau < 1.0):
        raise ValueError("tau must be in (0, 1).")

    # Variables : [β (k), u⁺ (n), u⁻ (n)]
    c = np.concatenate([np.zeros(k), tau * w, (1.0 - tau) * w])

    # A_eq @ var == y
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    b_eq = y

    bounds = (
        [(None, None)] * k +     # β unbounded
        [(0, None)] * n +        # u⁺ ≥ 0
        [(0, None)] * n          # u⁻ ≥ 0
    )

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res.success:
        return {
            "coef": np.full(k, np.nan),
            "residuals": np.full(n, np.nan),
            "loss": np.nan,
            "status": res.status,
            "success": False,
        }

    beta = res.x[:k]
    u_pos = res.x[k:k + n]
    u_neg = res.x[k + n:]
    residuals = u_pos - u_neg
    loss = float(c @ res.x)

    return {
        "coef": beta,
        "residuals": residuals,
        "loss": loss,
        "status": res.status,
        "success": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Bootstrap standard errors for weighted QR
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_wqr_se(y, X, tau, weights=None, n_boot=200, rng=None):
    """
    Pairs (xy-pair) bootstrap standard errors for weighted QR.

    Resamples (y_t, X_t, w_t) with replacement ``n_boot`` times and
    re-fits.  Returns per-coefficient std-error and t-stat.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    y = np.asarray(y, dtype=np.float64)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n, k = X.shape

    if weights is None:
        weights = np.ones(n)

    coefs = np.full((n_boot, k), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            r = weighted_quantile_regression(y[idx], X[idx], tau, weights[idx])
            if r["success"]:
                coefs[b] = r["coef"]
        except Exception:
            pass

    se = np.nanstd(coefs, axis=0, ddof=1)
    return se, coefs


# ─────────────────────────────────────────────────────────────────────────────
#  Check function and pseudo R²
# ─────────────────────────────────────────────────────────────────────────────

def check_function(u, tau):
    """Quantile regression check function ρ_τ(u) = u·(τ - I(u<0))."""
    u = np.asarray(u, dtype=np.float64)
    return u * (tau - (u < 0).astype(np.float64))


def pseudo_r2(y, y_pred, tau, weights=None):
    """Koenker–Machado pseudo R² for (weighted) quantile regression."""
    y = np.asarray(y, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(y)
    rho_model = float(np.sum(weights * check_function(y - y_pred, tau)))
    q_y = _weighted_quantile(y, tau, weights)
    rho_null = float(np.sum(weights * check_function(y - q_y, tau)))
    if rho_null <= 0.0:
        return 0.0
    return max(0.0, 1.0 - rho_model / rho_null)


def _weighted_quantile(x, tau, w):
    """Weighted τ-quantile of x with weights w."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    idx = np.argsort(x)
    xs = x[idx]
    ws = w[idx]
    cum = np.cumsum(ws) / ws.sum()
    return float(xs[np.searchsorted(cum, tau, side="left").clip(max=len(xs) - 1)])


# ─────────────────────────────────────────────────────────────────────────────
#  Bandwidth selection
# ─────────────────────────────────────────────────────────────────────────────

def silverman_h(x):
    """Silverman's rule for Gaussian kernel bandwidth."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    s = np.std(x, ddof=1)
    iqr = sp_stats.iqr(x)
    return 1.06 * min(s, iqr / 1.34) * n ** (-1.0 / 5.0)


def yu_jones_h(h_base, tau):
    """
    Yu & Jones (1998) quantile-adjusted bandwidth

        h(τ) = h_base · ((τ(1-τ)) / φ(Φ^{-1}(τ))²)^{1/5}
    """
    phi_val = sp_stats.norm.pdf(sp_stats.norm.ppf(tau))
    return h_base * ((tau * (1.0 - tau) / (phi_val ** 2)) ** 0.2)


# ─────────────────────────────────────────────────────────────────────────────
#  Embedding for Granger-causality lag matrices
# ─────────────────────────────────────────────────────────────────────────────

def embed(x, dim=2):
    """
    R-style ``embed`` — returns (n-dim+1, dim) matrix whose
    column 0 is x_t, column 1 is x_{t-1}, …
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < dim:
        raise ValueError("series shorter than embedding dimension")
    return np.column_stack([x[dim - 1 - i: n - i] for i in range(dim)])


def lag_matrix(x, lags):
    """Stacked lag matrix (without dropping initial NaN rows).  Returns (n, len(lags))."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.full((n, len(lags)), np.nan)
    for j, l in enumerate(lags):
        if l == 0:
            out[:, j] = x
        else:
            out[l:, j] = x[:-l]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Safe Unicode print (Windows cp1252 workaround)
# ─────────────────────────────────────────────────────────────────────────────

def safe_print(text):
    """Print, replacing characters that can't be encoded by the terminal."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc))


# ─────────────────────────────────────────────────────────────────────────────
#  Small numerical helpers
# ─────────────────────────────────────────────────────────────────────────────

def standard_quantile_grid(by: float = 0.05) -> np.ndarray:
    """Default quantile grid 0.05, 0.10, … 0.95  (length 19)."""
    return np.round(np.arange(by, 1.0, by), 4)


def coerce_array(a, name="x") -> np.ndarray:
    """Convert input to flat float ndarray and validate."""
    arr = np.asarray(a, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def add_intercept(X) -> np.ndarray:
    """Prepend a column of ones to X."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.column_stack([np.ones(X.shape[0]), X])
