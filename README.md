# mqqr — Multivariate Quantile-on-Quantile Regression & Causality

[![PyPI](https://img.shields.io/pypi/v/mqqr.svg)](https://pypi.org/project/mqqr/)
[![Python](https://img.shields.io/pypi/pyversions/mqqr.svg)](https://pypi.org/project/mqqr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/mqqr.svg)](https://pypi.org/project/mqqr/)

A comprehensive, **publication-grade** Python toolkit implementing the
Quantile-on-Quantile (QQR) family of methods used in top-tier energy,
finance, and environmental econometrics journals.

`mqqr` covers **both** flavours of multivariate QQR found in the
literature (additive *and* moderation), the bivariate Sim & Zhou (2015)
benchmark, and the matching quantile Granger-causality tests, with
MATLAB-style 3-D Jet visualisations and journal-ready figures &
LaTeX tables.

---

## Table of contents

- [Highlights](#highlights)
- [Methods](#methods)
- [The two m-QQR formulations](#the-two-m-qqr-formulations)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Detailed usage](#detailed-usage)
  - [1. Bivariate QQR (Sim & Zhou, 2015)](#1-bivariate-qqr--simzhou-2015-qq_regression)
  - [2. Additive m-QQR — Type 1 (Alola et al. 2023)](#2-additive-m-qqr--type-1-alola-et-al-2023-additive_mqq_regression)
  - [3. Moderated m-QQR — Type 2 (Sinha et al. 2023)](#3-moderated-m-qqr--type-2-sinha-et-al-2023-mqq_regression)
  - [4. Bivariate QQ Granger causality](#4-bivariate-qq-granger-causality-qq_causality)
  - [5. Multivariate (conditional) QQ causality](#5-multivariate-conditional-qq-causality-mqq_causality)
- [Visualisations](#visualisations)
- [Tables & LaTeX export](#tables--latex-export)
- [Colour-maps](#colour-maps)
- [Bandwidth, kernels & inference](#bandwidth-kernels--inference)
- [API reference](#api-reference)
- [Replicating published studies](#replicating-published-studies)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [References](#references)
- [Author](#author)
- [License](#license)

---

## Highlights

- ✅ **Faithful Sim & Zhou (2015) estimator** — empirical-CDF Gaussian
  kernel weights `K((F_n(x) − τ)/h)`, exact weighted-quantile regression
  via HiGHS LP, paired-bootstrap standard errors.
- ✅ **Both m-QQR formulations** — additive Alola/Özkan and moderated
  Sinha specifications under a single API.
- ✅ **QQ Granger causality** with bivariate and conditional (Sinha-style
  moderator-controlled) variants.
- ✅ **MATLAB-style 3-D surfaces** — bundled Jet, Parula (R2014b+),
  Turbo (Google 2019), Blue–Red diverging, and the Sinha
  red-yellow-black palette. 300 dpi, serif fonts, ready for
  Elsevier / Springer / Wiley submissions.
- ✅ **Heat-maps with `*`, `**`, `***` significance overlays**, contour
  plots, side-by-side QR-vs-mQQR comparison panels, Alola Fig. 3-style
  panels for the additive variant.
- ✅ **Publication-ready tables** — descriptive stats (Jarque-Bera,
  ARCH-LM, skewness, kurtosis), `19 × 19` results matrices with
  significance stars, LaTeX and Markdown export.
- ✅ **Optional interactive Plotly HTML 3-D surfaces** via
  `pip install mqqr[plotly]`.

---

## Methods

| # | Method                              | Function                       | Reference                                      |
|---|-------------------------------------|--------------------------------|------------------------------------------------|
| 1 | Bivariate QQR                       | `qq_regression()`              | Sim & Zhou (2015), *JBF*                       |
| 2 | **Additive m-QQR (Type 1)**         | `additive_mqq_regression()`    | Alola, Özkan & Usman (2023), *Energy Econ.*<br>Ozkan et al. (2023), *SCED* |
| 3 | **Moderated m-QQR (Type 2)**        | `mqq_regression()`             | Sinha et al. (2023), *Energy Econ.*<br>Das et al. (2023), *ESPR* |
| 4 | QQ Granger causality                | `qq_causality()`               | Troster (2018)                                  |
| 5 | Multivariate (conditional) causality | `mqq_causality()`              | Sinha et al. (2024), *Risk Analysis*           |

---

## The two m-QQR formulations

There are two distinct multivariate QQR specifications in the
literature.  They are **not** the same model — they answer different
questions and produce different output. `mqqr` implements both.

### Type 1 — Additive multivariate QQR (Alola, Özkan & Usman, 2023)

$$
y_t \;=\; \beta_0(\theta,\Phi_1,\dots,\Phi_n) \;+\; \sum_{i=1}^{n}
\beta_i(\theta,\Phi_i)\bigl(x_{i,t}-x_i^{\Phi_i}\bigr)
\;+\; \alpha^{\theta} y_{t-1} \;+\; \epsilon_t^{\theta}
$$

- Each regressor `x_i` has its **own** quantile axis `Φ_i`.
- No interaction terms — purely additive.
- Output: ***n* separate 3-D surfaces** `β̂_i(θ, Φ_i)`, one per
  regressor, plotted side-by-side (Alola 2023 Fig. 3 layout).
- Reads as the *partial effect* of `x_i` adjusted for the other
  regressors.

### Type 2 — Moderated m-QQR (Sinha et al., 2023)

$$
y_t \;=\; \beta_0(\theta,\tau) \;+\; \beta_1(\theta,\tau)(x_t-x^\tau)
\;+\; \sum_{j=1}^{p}\!\Bigl[\gamma_j(\theta,\tau)\bigl(x_t Z_{j,t}-x^\tau
Z_j^\tau\bigr) + \alpha_j(\theta)(Z_{j,t}-Z_j^\tau)\Bigr]
\;+\; \alpha^{\theta} y_{t-1} \;+\; \epsilon_t^{\theta}
$$

- One **principal** regressor `x` with quantile `τ`.
- Exogenous **moderators** `Z_j` enter linearly **and** through
  interaction terms `x · Z_j`.
- Output: a principal `β̂_1(θ, τ)` surface **plus** one moderation
  surface `γ̂_j(θ, τ)` per moderator.
- Reads as how the principal effect is *moderated* by `Z`.

### When to use which

| You want to …                                                | Use                            |
|--------------------------------------------------------------|--------------------------------|
| Compare the marginal effects of *several* regressors on `y`  | `additive_mqq_regression()`     |
| Study how *one* main effect is moderated by other covariates | `mqq_regression()`              |
| Replicate Alola, Özkan, Usman (2023) / Özkan et al. (2023)   | `additive_mqq_regression()`     |
| Replicate Sinha et al. (2023) / Das et al. (2023)            | `mqq_regression()`              |
| Just one explanatory variable                                | `qq_regression()` (bivariate)   |

---

## Installation

```bash
pip install mqqr                # core (numpy / pandas / scipy / matplotlib)
pip install mqqr[plotly]        # + interactive 3-D HTML
pip install mqqr[full]          # + plotly + seaborn + openpyxl + tabulate
```

For an editable install from source:

```bash
git clone https://github.com/merwanroudane/qqrpy.git
cd qqrpy
pip install -e .
```

Python 3.8+ supported.

---

## Quick start

```python
import numpy as np
from mqqr import (
    qq_regression, mqq_regression, additive_mqq_regression,
    qq_causality, mqq_causality,
    plot_qq_3d, plot_qq_heatmap,
    plot_mqq_panel, plot_additive_mqq_panel,
    plot_qq_causality_heatmap,
)

rng = np.random.default_rng(2026)
n   = 400
x   = np.cumsum(rng.normal(0, 1, n)) / 10        # principal regressor
epu = rng.normal(0, 1, n)                        # moderator 1
co2 = np.cumsum(rng.normal(0, 1, n)) / 12        # moderator 2
y   = -0.30 * x * (x < 0) + 0.15 * x * epu + rng.normal(0, 0.6, n)

# 1.  Bivariate QQR  (Sim & Zhou, 2015)
qq = qq_regression(y, x, x_name="GF", y_name="RE")
qq.summary()
plot_qq_3d(qq, cmap="jet")
plot_qq_heatmap(qq, annotate="stars")

# 2.  Additive m-QQR  (Type 1 - Alola 2023): one surface per regressor
amq = additive_mqq_regression(
    y, {"GF": x, "EPU": epu, "CO2": co2}, y_name="RE",
)
amq.summary()
plot_additive_mqq_panel(amq, cmap="jet")          # Alola Fig. 3 layout

# 3.  Moderated m-QQR  (Type 2 - Sinha 2023): principal + moderation surfaces
mq = mqq_regression(
    y, x, moderators={"EPU": epu, "CO2": co2},
    x_name="GF", y_name="RE",
)
mq.summary()
plot_mqq_panel(mq)                                # principal + γ panels

# 4.  Bivariate QQ Granger causality
cq = qq_causality(x, y, cause_name="GF", effect_name="RE")
plot_qq_causality_heatmap(cq)                     # red-yellow-black palette

# 5.  Conditional (m-QQ) causality
mcq = mqq_causality(x, y, moderators={"EPU": epu, "CO2": co2})
plot_qq_causality_heatmap(mcq)
```

---

## Detailed usage

### 1. Bivariate QQR — Sim & Zhou (2015) — `qq_regression()`

For each `(θ, τ)` on the unit square:

$$y_t = \beta_0(\theta,\tau) + \beta_1(\theta,\tau)\bigl(x_t-x^{\tau}\bigr) + \alpha^{\theta}y_{t-1} + v_t^{\theta}$$

solved as a *weighted* `θ`-quantile regression with Gaussian kernel
weights `w_t = K((F_n(x_t) − τ)/h)`.

```python
res = qq_regression(
    y, x,
    y_quantiles=np.arange(0.05, 1.0, 0.05),   # default 19-point grid
    x_quantiles=np.arange(0.05, 1.0, 0.05),
    bandwidth=0.05,                            # Sim & Zhou plug-in
    include_lag=True,                          # include y_{t-1}
    se="bootstrap",                            # or "none"
    n_boot=200,
    cdf_based_kernel=True,                     # CDF-distance kernel
    x_name="GF",  y_name="RE",
)
res.summary()                                  # tabular summary
B   = res.to_matrix("beta1")                   # (n_y, n_x) coefficient grid
P   = res.to_matrix("p_value")
R2  = res.to_matrix("r_squared")
qr  = res.average_over_tau()                   # collapsed-to-QR check
```

`QQResult` attributes:

| Attribute        | Type              | Description                                              |
|------------------|-------------------|----------------------------------------------------------|
| `results`        | `pandas.DataFrame`| `y_quantile, x_quantile, beta0, beta1, se, t, p, R²`     |
| `y_quantiles`    | `ndarray`         | θ-grid                                                   |
| `x_quantiles`    | `ndarray`         | τ-grid                                                   |
| `n_obs`          | `int`             | sample size after NaN-removal & lag                      |
| `bandwidth`      | `float`           | kernel bandwidth                                         |
| `x_name`, `y_name`| `str`            | display names                                            |

Visualise with `plot_qq_3d`, `plot_qq_heatmap`, `plot_qq_contour`,
`plot_qq_3d_plotly`.

---

### 2. Additive m-QQR — Type 1 (Alola et al. 2023) — `additive_mqq_regression()`

Implements

$$y_t \;=\; \beta_0(\theta,\Phi_1,\dots,\Phi_n) \;+\; \sum_{i=1}^{n} \beta_i(\theta,\Phi_i)\bigl(x_{i,t}-x_i^{\Phi_i}\bigr) \;+\; \alpha^{\theta}y_{t-1} \;+\; \epsilon_t^{\theta}.$$

For each focal regressor `x_i` and each cell `(θ, Φ)`:

1. Compute Gaussian kernel weights local in `x_i` at quantile `Φ`.
2. Build a design matrix with **all** regressors centred at their own
   `Φ`-quantile.
3. Solve the weighted `θ`-quantile regression and extract the slope on
   `x_i`.

```python
res = additive_mqq_regression(
    y,
    X={"GF": x1, "EPU": x2, "CO2": x3},      # dict of n regressors
    y_quantiles=np.arange(0.05, 1.0, 0.05),
    x_quantiles=np.arange(0.05, 1.0, 0.05),
    bandwidth=0.05,
    include_lag=True,
    se="bootstrap", n_boot=200,
    y_name="REN",
)
res.summary()

# Per-regressor matrix views
B_gf  = res.to_matrix("GF",  value="beta")
P_gf  = res.to_matrix("GF",  value="p_value")
mask  = res.significance_matrix("GF", alpha=0.05)
df_co2 = res.get("CO2")                       # long-format DataFrame

# Visualise — Alola Fig. 3 side-by-side panel
plot_additive_mqq_panel(res, cmap="jet",
                        save_path="additive_panel.pdf")
# or focus on one regressor
plot_additive_mqq_3d(res, "GF",   cmap="jet")
plot_additive_mqq_heatmap(res, "GF", annotate="stars")

# Export CSVs (one per regressor)
res.export_csv("alola_replication")           # → alola_replication_GF.csv etc.
```

`AdditiveMQQResult` attributes:

| Attribute     | Type                       | Description                                                |
|---------------|----------------------------|------------------------------------------------------------|
| `surfaces`    | `dict[str, DataFrame]`     | one long-format frame per regressor                        |
| `y_quantiles` | `ndarray`                  | θ-grid                                                     |
| `x_quantiles` | `ndarray`                  | Φ-grid                                                     |
| `n_obs`       | `int`                      | sample size after NaN-removal & lag                        |
| `bandwidth`   | `float`                    | kernel bandwidth                                           |
| `x_names`     | `list[str]`                | regressor names                                            |
| `y_name`      | `str`                      | dependent name                                             |
| `include_lag` | `bool`                     | whether `y_{t-1}` was included as a control                |

---

### 3. Moderated m-QQR — Type 2 (Sinha et al. 2023) — `mqq_regression()`

Implements

$$y_t \;=\; \beta_0(\theta,\tau) \;+\; \beta_1(\theta,\tau)(x_t-x^\tau) \;+\; \sum_{j}\!\bigl[\gamma_j(\theta,\tau)(x_t Z_{j,t}-x^\tau Z_j^\tau) + \alpha_j(\theta)(Z_{j,t}-Z_j^\tau)\bigr] \;+\; \alpha^{\theta}y_{t-1} \;+\; \epsilon_t^{\theta}.$$

```python
res = mqq_regression(
    y, x,
    moderators={"EPU": epu, "BUSCONF": bc, "UNEMP": un},
    y_quantiles=np.arange(0.05, 1.0, 0.05),
    x_quantiles=np.arange(0.05, 1.0, 0.05),
    bandwidth=0.05,
    include_lag=True,
    interactions=True,                  # set False for simple partial m-QQR
    se="bootstrap", n_boot=200,
    x_name="ETAX",  y_name="REN",
)
res.summary()

# Principal slope surface β1(θ, τ)
B1 = res.to_matrix("beta1")
# γ surface for one moderator
G_epu = res.interaction_matrix("EPU", value="gamma")

# Visualise
plot_mqq_3d(res, cmap="jet")                # principal 3-D surface
plot_mqq_heatmap(res, annotate="stars")
plot_mqq_panel(res)                         # 3-D + heat-map + γ-panels

# Exports
res.export_csv("sinha_replication")
```

`MQQResult` exposes three DataFrames:

| Attribute          | Description                                              |
|--------------------|----------------------------------------------------------|
| `main_results`     | `y_quantile, x_quantile, beta0, beta1, se, t, p, R²`     |
| `interactions`     | per-moderator `γ_j(θ,τ)` long-format                     |
| `moderator_direct` | per-moderator linear-term `α_j(θ)` long-format            |

---

### 4. Bivariate QQ Granger causality — `qq_causality()`

For each `(θ, τ)` tests whether `x_{t-1}` Granger-causes the `θ`-quantile
of `y_t`, using a paired bootstrap on the local weighted quantile
regression coefficient.

```python
res = qq_causality(
    x, y,                                # x → y
    bandwidth=0.05,
    n_boot=200,
    cause_name="OIL", effect_name="SP500",
)
res.summary()
T = res.stat_matrix()                    # t-statistic grid
P = res.pval_matrix()
sig = (T.__abs__() > 1.96)               # 5 %-level significance mask

plot_qq_causality_heatmap(res, cmap="red_yellow_black", show_stars=True)
```

---

### 5. Multivariate (conditional) QQ causality — `mqq_causality()`

The Sinha-style moderator-controlled version of the test.  `x` → `y`
conditional on a set of `Z` moderators, optionally with `x · Z`
interactions.

```python
res = mqq_causality(
    x, y, moderators={"EPU": epu, "UNEMP": un},
    bandwidth=0.05,
    n_boot=200,
    interactions=True,                   # full Sinha specification
    cause_name="GF", effect_name="RE",
)
res.summary()
plot_qq_causality_heatmap(res, cmap="red_yellow_black")
```

---

## Visualisations

All plotters return a matplotlib `Figure` (or `(fig, ax)`) and accept a
`save_path` argument writing PDF / PNG / EPS / SVG at 300 dpi.

| Plot                                 | Function                          | Typical use                              |
|--------------------------------------|-----------------------------------|------------------------------------------|
| 3-D surface (MATLAB Jet)             | `plot_qq_3d`                      | bivariate QQR / m-QQR principal slope    |
| 2-D heat-map with `*`/`**`/`***`     | `plot_qq_heatmap`                 | journal Table-Fig style                  |
| Filled contour                       | `plot_qq_contour`                 | publication-style contour panels         |
| m-QQR 3-D                            | `plot_mqq_3d`                     | Sinha Type 2 principal surface           |
| m-QQR heat-map                       | `plot_mqq_heatmap`                | Sinha Type 2 principal heat-map          |
| **Alola 3-panel m-QQR**              | **`plot_mqq_panel`**              | principal + heat-map + γ moderation     |
| **Additive m-QQR 3-D (per var)**     | **`plot_additive_mqq_3d`**        | one regressor surface                    |
| **Additive m-QQR heat-map (per var)**| **`plot_additive_mqq_heatmap`**   | one regressor heat-map                   |
| **Additive m-QQR panel**             | **`plot_additive_mqq_panel`**     | Alola Fig. 3 side-by-side (n columns)    |
| QQ causality heat-map                | `plot_qq_causality_heatmap`       | t-statistics with stars                  |
| QR vs m-QQR comparison               | `plot_qq_vs_mqq`                  | Sim & Zhou Fig. 5 collapse check         |
| Interactive 3-D HTML                 | `plot_qq_3d_plotly`               | requires `mqqr[plotly]`                  |

Each plotter accepts `cmap=` selecting from the bundled palettes
(`"jet"`, `"parula"`, `"turbo"`, `"blue_red"`, `"red_yellow_black"`) or
any matplotlib colormap name.

---

## Tables & LaTeX export

```python
from mqqr import descriptive_table, results_table, to_latex, to_markdown

# Descriptive statistics with Jarque-Bera & ARCH-LM
desc = descriptive_table({"GF": x, "EPU": epu, "CO2": co2, "RE": y})
print(desc)

# Results table with significance stars
tbl = results_table(res.main_results,
                    value_col="beta1", pval_col="p_value",
                    row_col="y_quantile", col_col="x_quantile",
                    digits=4)
print(tbl)

# LaTeX export (Elsevier / Springer template-compatible)
latex = to_latex(res.main_results,
                 caption="m-QQR results: GF → RE conditional on EPU and CO2",
                 label="tab:mqqr_main",
                 digits=4)

# Markdown export (for GitHub READMEs / Quarto / Jupyter)
md = to_markdown(res.main_results, digits=4)
```

---

## Colour-maps

Bundled MATLAB-faithful colormaps:

| Name                       | Style                                | Best for                              |
|----------------------------|--------------------------------------|---------------------------------------|
| `MATLAB_JET` (`"jet"`)     | classical rainbow                    | 3-D surfaces (Sim & Zhou, Alola)      |
| `MATLAB_PARULA` (`"parula"`)| R2014b+ default                     | accessible 3-D surfaces                |
| `TURBO` (`"turbo"`)        | Google 2019 perceptually-uniform     | sequential surfaces                    |
| `BLUE_RED` (`"blue_red"`)  | diverging blue ↔ red                 | signed coefficient heat-maps           |
| `RED_YELLOW_BLACK`         | Sinha-style red→yellow→black         | causality heat-maps                    |

Plus full access to every matplotlib colormap by name.  Inspect what
ships with:

```python
from mqqr import show_colormaps
show_colormaps()                            # renders a swatch panel
```

---

## Bandwidth, kernels & inference

- **Default bandwidth** `h = 0.05` follows Sim & Zhou (2015) and is used
  in every replication study in this package's reference set.
- **CDF-distance kernel** (`cdf_based_kernel=True`, default):
  `w_t = K((F_n(x_t) − τ)/h)`, so `h` is on the unit interval.
- **Raw-distance kernel** (`cdf_based_kernel=False`): `w_t = K((x_t −
  x_τ)/(h · sd(x)))`, matching the formulation in Sinha's reference R
  script `lprq()`.
- **Inference**: paired (xy-pair) bootstrap standard errors.  Set
  `n_boot=200` (default) for production, `n_boot=20–40` for fast
  prototyping.  Set `se="none"` to skip inference entirely (slope only).
- **Lag control**: `include_lag=True` (default) adds `y_{t-1}` as a
  covariate, matching the original Sim & Zhou specification.

---

## API reference

### Estimators

| Function                        | Returns                | Description                                              |
|---------------------------------|------------------------|----------------------------------------------------------|
| `qq_regression(y, x, ...)`      | `QQResult`             | Bivariate Sim & Zhou (2015) QQR                          |
| `additive_mqq_regression(y, X, ...)`  | `AdditiveMQQResult` | Type 1 — Alola/Özkan additive multivariate              |
| `mqq_regression(y, x, moderators, ...)`| `MQQResult`        | Type 2 — Sinha moderation + interactions                |
| `qq_causality(x, y, ...)`       | `QQCausalityResult`    | Bivariate QQ Granger causality                           |
| `mqq_causality(x, y, moderators, ...)` | `MQQCausalityResult` | Conditional (m-QQ) Granger causality                  |

### Plotting

| Function                             | Description                                          |
|--------------------------------------|------------------------------------------------------|
| `plot_qq_3d(result, ...)`            | 3-D surface (MATLAB Jet)                             |
| `plot_qq_heatmap(result, ...)`       | 2-D heat-map with significance stars                  |
| `plot_qq_contour(result, ...)`       | Filled contour with isolines                          |
| `plot_mqq_3d(result, ...)`           | Sinha Type 2 principal 3-D surface                    |
| `plot_mqq_heatmap(result, ...)`      | Sinha Type 2 principal heat-map                       |
| `plot_mqq_panel(result, ...)`        | 3-panel Sinha figure with γ moderation panels         |
| `plot_additive_mqq_3d(res, var, ...)`| One regressor surface from Type 1                     |
| `plot_additive_mqq_heatmap(res, var, ...)` | One regressor heat-map from Type 1              |
| `plot_additive_mqq_panel(res, ...)`  | Alola Fig. 3 side-by-side (n columns)                 |
| `plot_qq_causality_heatmap(res, ...)`| t-statistics heat-map with stars                      |
| `plot_qq_vs_mqq(res, ...)`           | Sim & Zhou Fig. 5 collapse comparison                 |
| `plot_qq_3d_plotly(result, ...)`     | Interactive HTML 3-D surface (requires `[plotly]`)     |

### Tables

| Function                  | Description                                          |
|---------------------------|------------------------------------------------------|
| `descriptive_table(data)` | Mean / sd / skew / kurt / JB / ARCH-LM               |
| `results_table(df, ...)`  | Pivot table with `*`/`**`/`***` formatting           |
| `to_latex(df, ...)`       | Journal-ready LaTeX                                  |
| `to_markdown(df, ...)`    | Markdown table                                       |

### Colours

| Object / function         | Description                                          |
|---------------------------|------------------------------------------------------|
| `MATLAB_JET`              | MATLAB Jet colormap                                  |
| `MATLAB_PARULA`           | MATLAB Parula colormap                               |
| `TURBO`                   | Google Turbo colormap                                |
| `BLUE_RED`                | Diverging blue-red colormap                          |
| `RED_YELLOW_BLACK`        | Sinha causality palette                              |
| `list_colormaps()`        | Bundled colormap names                               |
| `show_colormaps()`        | Render colormap swatches                             |

---

## Replicating published studies

### Alola, Özkan & Usman (2023), *Energy Economics* 120

```python
res = additive_mqq_regression(
    y=COP, X={"ED": ED, "REP": REP, "CO2": CO2},
    bandwidth=0.05, y_name="COP",
)
plot_additive_mqq_panel(res, cmap="jet",
                        save_path="alola2023_fig3.pdf")
```

### Sinha et al. (2023), *Energy Economics* 126

```python
res = mqq_regression(
    y=REN, x=ETAX,
    moderators={"BUSCONF": BUSCONF, "EPU": EPU, "UNEMP": UNEMP},
    bandwidth=0.05, y_name="REN", x_name="ETAX",
    interactions=True,
)
plot_mqq_panel(res, save_path="sinha2023_mqqr.pdf")
```

### Sim & Zhou (2015), *Journal of Banking & Finance* 55

```python
res = qq_regression(SP500_returns, OIL_returns,
                    bandwidth=0.05, x_name="OIL", y_name="SP500")
plot_qq_3d(res, cmap="jet", save_path="sim_zhou_2015_fig.pdf")
```

---

## Dependencies

### Required

| Package      | Min. version | Purpose                                    |
|--------------|--------------|--------------------------------------------|
| `numpy`      | `≥ 1.20`     | array maths                                |
| `pandas`     | `≥ 1.3`      | result frames & table export               |
| `scipy`      | `≥ 1.7`      | distributions, kernels, LP solver          |
| `statsmodels`| `≥ 0.13`     | OLS / robust QR primitives                 |
| `matplotlib` | `≥ 3.5`      | 3-D surfaces, heat-maps, contours          |

### Optional

| Extras                      | Install                          | Adds                                |
|-----------------------------|----------------------------------|-------------------------------------|
| `plotly` + `kaleido`        | `pip install mqqr[plotly]`       | interactive HTML 3-D surfaces       |
| `seaborn`, `openpyxl`, `tabulate` | `pip install mqqr[full]`   | richer table / Excel export         |

---

## Citation

If `mqqr` supports academic research, please cite **both** the relevant
methodology paper and the software:

```bibtex
@software{roudane_mqqr_2026,
  author    = {Merwan Roudane},
  title     = {{mqqr: Multivariate Quantile-on-Quantile Regression and Causality in Python}},
  year      = {2026},
  version   = {1.0.0},
  publisher = {PyPI},
  url       = {https://github.com/merwanroudane/qqrpy}
}
```

Methodology papers — please cite the formulation you used:

- **Bivariate QQR** → Sim & Zhou (2015).
- **Type 1 / additive m-QQR** → Alola, Özkan & Usman (2023); Ozkan et
  al. (2023).
- **Type 2 / moderated m-QQR** → Sinha et al. (2023); Das et al. (2023).
- **Multivariate QQ causality** → Sinha et al. (2024).

---

## References

1. **Sim, N. & Zhou, H. (2015).** *Oil prices, US stock return, and the
   dependence between their quantiles.* Journal of Banking & Finance,
   55, 1–12. [doi:10.1016/j.jbankfin.2015.01.013](https://doi.org/10.1016/j.jbankfin.2015.01.013)
2. **Alola, A. A., Özkan, O. & Usman, O. (2023).** *Examining crude oil
   price outlook amidst substitute energy price and household energy
   expenditure in the USA: A novel nonparametric multivariate QQR
   approach.* Energy Economics, 120, 106613.
   [doi:10.1016/j.eneco.2023.106613](https://doi.org/10.1016/j.eneco.2023.106613)
3. **Ozkan, O., Haruna, R. A., Alola, A. A., Ghardallou, W. & Usman, O.
   (2023).** *Investigating energy-related environmental risks of
   economic complexity: a multivariate QQR approach.* Structural Change
   and Economic Dynamics, 65, 382–392.
4. **Sinha, A., Ghosh, V., Hussain, N., Nguyen, D. K. & Das, N. (2023).**
   *Green financing of renewable energy generation: Capturing the role
   of exogenous moderation for ensuring sustainable development.*
   Energy Economics, 126, 107021.
   [doi:10.1016/j.eneco.2023.107021](https://doi.org/10.1016/j.eneco.2023.107021)
5. **Das, N., Gangopadhyay, P., Bera, P. & Hossain, M. E. (2023).**
   *Multivariate quantile-on-quantile regression and the Kaya identity:
   India.* Environmental Science and Pollution Research, 30, 45796–45814.
6. **Troster, V. (2018).** *Testing for Granger-causality in quantiles.*
   Econometric Reviews, 37(8), 850–866.
7. **Sinha, A. et al. (2024).** *Multivariate Quantile Causality.* Risk
   Analysis (in press).

---

## Author

**Dr. Merwan Roudane**
📧 `merwanroudane920@gmail.com`
🔗 [github.com/merwanroudane/qqrpy](https://github.com/merwanroudane/qqrpy)

Contributions, bug reports and feature requests are welcome via GitHub
Issues.

---

## License

MIT License © 2026 Dr. Merwan Roudane.  See [LICENSE](LICENSE).
