"""
Publication-quality tables for mqqr results.

Generates:

* **Descriptive statistics**         -- table 1 of Sinha-style papers.
* **Results tables** (β1 / t / p)    -- the canonical 19 × 19 grids.
* **LaTeX** and **Markdown** export, with significance stars.
"""

from __future__ import annotations

from typing import Optional, Iterable, Dict, Union
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ─────────────────────────────────────────────────────────────────────────────
#  Descriptive statistics
# ─────────────────────────────────────────────────────────────────────────────

def descriptive_table(data: Union[Dict[str, np.ndarray], pd.DataFrame],
                      *,
                      include_normality: bool = True,
                      include_archlm: bool = True,
                      arch_lags: int = 1,
                      digits: int = 4) -> pd.DataFrame:
    """
    Table-1 style descriptive statistics.

    Parameters
    ----------
    data : dict or DataFrame
        Variable name → 1-D array, or a DataFrame.
    include_normality : bool
        Append Jarque–Bera statistic and p-value.
    include_archlm : bool
        Append ARCH-LM test (heteroskedasticity).
    arch_lags : int
        Lags for ARCH-LM.

    Returns
    -------
    pandas.DataFrame indexed by variable, with columns
    ``Mean, Median, Std, Min, Max, Skewness, Kurtosis, [JB, p-JB],
    [ARCH-LM, p-ARCH]``.
    """
    if isinstance(data, dict):
        df = pd.DataFrame({k: pd.Series(np.asarray(v, dtype=float).ravel())
                           for k, v in data.items()})
    else:
        df = pd.DataFrame(data).astype(float)

    rows = []
    for col in df.columns:
        x = df[col].dropna().values
        n = len(x)
        skew = float(sp_stats.skew(x))
        kurt = float(sp_stats.kurtosis(x, fisher=False))   # raw kurtosis
        row = {
            "N":        n,
            "Mean":     float(np.mean(x)),
            "Median":   float(np.median(x)),
            "Std":      float(np.std(x, ddof=1)),
            "Min":      float(np.min(x)),
            "Max":      float(np.max(x)),
            "Skewness": skew,
            "Kurtosis": kurt,
        }
        if include_normality:
            jb_stat, jb_p = sp_stats.jarque_bera(x)
            row["JB"]   = float(jb_stat)
            row["JB_p"] = float(jb_p)
        if include_archlm:
            arch_stat, arch_p = _arch_lm(x, lags=arch_lags)
            row["ARCH_LM"]   = float(arch_stat)
            row["ARCH_LM_p"] = float(arch_p)
        rows.append((col, row))

    out = pd.DataFrame.from_records([r for _, r in rows],
                                    index=[c for c, _ in rows])
    return out.round(digits)


def _arch_lm(x: np.ndarray, lags: int = 1):
    """Engle's ARCH-LM test (T·R² ~ χ²_lags under H0)."""
    x = np.asarray(x, dtype=float)
    e = x - x.mean()
    e2 = e ** 2
    n = len(e2)
    if n <= lags + 1:
        return np.nan, np.nan
    # Regress e2_t on a constant and lagged e2_{t-1}, …, e2_{t-lags}
    y = e2[lags:]                                      # length n - lags
    cols = [e2[lags - 1 - i : n - 1 - i] for i in range(lags)]
    X = np.column_stack([np.ones(n - lags)] + cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lm = (n - lags) * r2
    p = 1.0 - sp_stats.chi2.cdf(lm, df=lags)
    return lm, p


# ─────────────────────────────────────────────────────────────────────────────
#  Results tables (β1 grid)
# ─────────────────────────────────────────────────────────────────────────────

def results_table(result, *, value: str = "beta1",
                  stars: bool = True, digits: int = 3) -> pd.DataFrame:
    """
    Pretty 19 × 19 results table -- rows = θ, columns = τ.

    With ``stars=True`` cell entries become e.g. ``+0.142**`` (using the
    p-value from the same result object).  Empty for NaN cells.
    """
    if hasattr(result, "to_matrix"):
        V = result.to_matrix(value)
    else:
        raise TypeError("result has no to_matrix() method")

    yq = np.asarray(result.y_quantiles, dtype=float)
    xq = np.asarray(result.x_quantiles, dtype=float)
    df = result.results if hasattr(result, "results") else result.main_results

    P = df.pivot(index="y_quantile", columns="x_quantile",
                 values="p_value").sort_index(axis=0).sort_index(axis=1).values

    cells = np.full(V.shape, "", dtype=object)
    fmt = f"{{:+.{digits}f}}"
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            if np.isfinite(v):
                s = fmt.format(v)
                if stars:
                    p = P[i, j]
                    if np.isfinite(p):
                        if p < 0.01:
                            s += "***"
                        elif p < 0.05:
                            s += "**"
                        elif p < 0.10:
                            s += "*"
                cells[i, j] = s
            else:
                cells[i, j] = "--"

    out = pd.DataFrame(
        cells,
        index=[f"{q:.2f}" for q in yq],
        columns=[f"{q:.2f}" for q in xq],
    )
    # ASCII name keeps Windows-cp1252 terminals happy.  The LaTeX
    # exporter (`to_latex`) substitutes proper Greek letters.
    out.index.name   = "Y_q \\ X_q"
    out.columns.name = ""
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Significance-stars helper
# ─────────────────────────────────────────────────────────────────────────────

def significance_stars(p: float) -> str:
    """Return ``'***' / '**' / '*' / ''`` for usual thresholds."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Export
# ─────────────────────────────────────────────────────────────────────────────

def to_latex(table: pd.DataFrame, *, caption: str = "",
             label: str = "tab:mqqr",
             column_format: Optional[str] = None,
             notes: Optional[str] = None) -> str:
    """
    Convert a (results) DataFrame to a journal-grade LaTeX longtable
    snippet with optional caption, label, and notes.
    """
    if column_format is None:
        column_format = "l" + "r" * len(table.columns)

    # Render an ASCII row-name like "Y_q \\ X_q" with Greek symbols in LaTeX
    index_label = table.index.name or ""
    if index_label.startswith("Y_q"):
        index_label = r"$\theta \backslash \tau$"

    lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}" if caption else "",
        f"  \\label{{{label}}}" if label else "",
        f"  \\begin{{tabular}}{{{column_format}}}",
        r"    \toprule",
        "    " + " & ".join([index_label] +
                            list(map(str, table.columns))) + r" \\",
        r"    \midrule",
    ]
    for idx, row in table.iterrows():
        lines.append("    " + " & ".join([str(idx)] +
                                          [str(v) for v in row.values]) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
    ]
    if notes:
        lines.append(rf"  \par\medskip\footnotesize\textit{{Notes:}} {notes}")
    lines.append(r"\end{table}")
    return "\n".join(l for l in lines if l)


def to_markdown(table: pd.DataFrame, *, caption: str = "") -> str:
    """Markdown export (GFM-flavoured)."""
    out = []
    if caption:
        out.append(f"**{caption}**\n")
    header = "| " + (table.index.name or "") + " | " + \
             " | ".join(map(str, table.columns)) + " |"
    sep = "|" + "---|" * (len(table.columns) + 1)
    out.append(header)
    out.append(sep)
    for idx, row in table.iterrows():
        out.append("| " + str(idx) + " | " +
                   " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(out)
