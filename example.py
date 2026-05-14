"""
End-to-end demonstration of the mqqr package.

Generates synthetic data that mimics the Sinha/Sim-Zhou setup
(asymmetric, moderated, tail-dependent relationship) and runs:

1. Bivariate QQR     (Sim & Zhou, 2015)
2. m-QQR             (Sinha et al., 2023)
3. QQ causality      (cause -> effect heat-map with stars)
4. m-QQ causality    (conditional on moderators)
5. Side-by-side panels + descriptive table + LaTeX export
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from mqqr import (
    qq_regression, mqq_regression,
    qq_causality, mqq_causality,
    plot_qq_3d, plot_qq_heatmap, plot_qq_contour,
    plot_mqq_panel, plot_qq_causality_heatmap, plot_qq_vs_mqq,
    descriptive_table, results_table, to_latex, to_markdown,
    show_colormaps,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic data set:   Y_t depends asymmetrically on X_{t-1} + moderators
# ─────────────────────────────────────────────────────────────────────────────

def generate(n=400, seed=2026):
    rng = np.random.default_rng(seed)
    # AR(1) "green-finance" shock
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.4 * x[t - 1] + rng.normal(0, 1)

    # Moderators
    epu  = 0.6 * rng.normal(0, 1, n) + 0.2 * x          # economic-policy uncertainty
    unemp = 0.5 * rng.normal(0, 1, n) + 0.15 * (-x)     # unemployment
    bconf = 0.6 * rng.normal(0, 1, n) - 0.1 * epu       # business confidence

    # Asymmetric, tail-dependent generator for Y :
    # large negative x AND high epu jointly amplify the negative impact.
    y = np.zeros(n)
    for t in range(1, n):
        base = 0.10 + 0.45 * y[t - 1]
        kicker = -0.30 * x[t - 1] * (x[t - 1] < 0)        # asymmetry on x⁻
        moderation = 0.20 * x[t - 1] * epu[t]              # interaction with EPU
        unemp_eff = -0.10 * unemp[t] * (x[t - 1] > 0)      # opposite-tail effect
        noise = rng.normal(0, 0.6)
        y[t] = base + kicker + moderation + unemp_eff + noise

    return dict(x=x, y=y, epu=epu, unemp=unemp, bconf=bconf)


# ─────────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print(" mqqr  --  end-to-end demo  (synthetic green-finance / RE data)")
    print("=" * 72)

    data = generate()

    # 1.  Descriptive statistics  ──────────────────────────────────────────
    desc = descriptive_table(
        {"Y (RE)": data["y"], "X (GF)": data["x"],
         "EPU":  data["epu"], "UNEMP": data["unemp"], "BCONF": data["bconf"]},
        digits=3,
    )
    print("\nTable 1  --  Descriptive statistics\n")
    print(desc.to_string())

    # 2.  Bivariate QQR  ──────────────────────────────────────────────────
    print("\n[1/4]   Bivariate Quantile-on-Quantile Regression")
    qq = qq_regression(
        y=data["y"], x=data["x"],
        x_name="GF (X)", y_name="RE (Y)",
        n_boot=120, verbose=False,
    )
    qq.summary()

    # 3.  m-QQR with moderators  ───────────────────────────────────────────
    print("\n[2/4]   Multivariate QQR with EPU, UNEMP, BCONF moderators")
    mq = mqq_regression(
        y=data["y"], x=data["x"],
        moderators={"EPU": data["epu"], "UNEMP": data["unemp"],
                    "BCONF": data["bconf"]},
        x_name="GF (X)", y_name="RE (Y)",
        n_boot=120, verbose=False,
    )
    mq.summary()

    # 4.  QQ Granger Causality  ────────────────────────────────────────────
    print("\n[3/4]   Bivariate QQ Granger Causality")
    cq = qq_causality(
        x=data["x"], y=data["y"],
        cause_name="GF", effect_name="RE",
        n_boot=120, verbose=False,
    )
    cq.summary()

    # 5.  m-QQ conditional Causality  ─────────────────────────────────────
    print("\n[4/4]   Multivariate (conditional) QQ Causality")
    mcq = mqq_causality(
        x=data["x"], y=data["y"],
        moderators={"EPU": data["epu"], "UNEMP": data["unemp"],
                    "BCONF": data["bconf"]},
        cause_name="GF", effect_name="RE",
        n_boot=120, verbose=False,
    )
    mcq.summary()

    # 6.  LaTeX export of the 19 × 19 β1 table  ───────────────────────────
    tab = results_table(qq, value="beta1", stars=True, digits=3)
    print("\nTable 2 (excerpt) -- QQR β1(θ,τ) coefficients with significance:")
    print(tab.iloc[::3, ::3].to_string())   # every 3rd row/col

    latex = to_latex(
        tab.iloc[::3, ::3],
        caption=r"QQR slope coefficients $\hat{\beta}_1(\theta,\tau)$ "
                r"(every third quantile shown).",
        label="tab:qqr_beta1",
        notes=r"$^{*},\,^{**},\,^{***}$ denote significance at the "
              r"10\%, 5\%, 1\% levels, respectively.",
    )
    with open("mqqr_qqr_beta1.tex", "w", encoding="utf-8") as f:
        f.write(latex)
    print("\nWrote LaTeX table → mqqr_qqr_beta1.tex")

    # 7.  Visualisations  ──────────────────────────────────────────────────
    print("\nGenerating figures (matplotlib backend = "
          f"{plt.get_backend()})  …")

    fig1, _ = plot_qq_3d(qq, value="beta1", cmap="jet",
                          save_path="fig1_qqr_3d_jet.png")
    fig2, _ = plot_qq_heatmap(qq, value="beta1", cmap="jet",
                               annotate="stars",
                               save_path="fig2_qqr_heatmap.png")
    fig3, _ = plot_qq_contour(qq, value="beta1", cmap="parula",
                               save_path="fig3_qqr_contour.png")
    fig4    = plot_mqq_panel(mq, cmap="jet",
                              save_path="fig4_mqqr_panel.png")
    fig5, _ = plot_qq_causality_heatmap(
        cq, cmap="red_yellow_black",
        save_path="fig5_qq_causality.png")
    fig6, _ = plot_qq_causality_heatmap(
        mcq, cmap="red_yellow_black",
        save_path="fig6_mqq_causality.png")
    fig7, _ = plot_qq_vs_mqq(qq, mq, save_path="fig7_qqr_vs_mqqr.png")
    fig8    = show_colormaps(save_path="fig8_colormaps.png")

    plt.close("all")
    print("Saved 8 figures (PNG) in the current directory.")
    print("\nDone.")


if __name__ == "__main__":
    main()
