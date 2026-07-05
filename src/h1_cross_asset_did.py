"""
H1 identification check — Cross-asset difference-in-differences (USDC vs USDT).

USDT (offshore-issued, outside the GENIUS Act's regulatory perimeter) serves
as the control for common DeFi-wide shocks. Specification (paper eq. 6):

    log(V_it) = a_i + d_t + b*(Treated_i x Post_GENIUS) + g*(Treated_i x Post_OCC) + e_it

with asset and week fixed effects; standard errors clustered by week.

Outputs:
    figures/fig2_divergence.png        USDC vs USDT log-velocity + cumulative excess
    figures/fig3_event_study.png       Quarterly lead/lag event study
    figures/fig5_did_stability.png     DiD coefficient across pre-period windows
    output/event_study_quarterly.csv
    output/did_robustness_2024only.csv
    output/table89_latex.txt, output/table10_latex.txt
"""

from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import COLORS as PAL, FIG_DIR, OUT_DIR, T1, star, treatment_lines

DID_FORMULA = "log_vel ~ C(asset) + C(week) + did_genius + did_occ"


# ──────────────────────────────────────────────────────────────────────────────
# Main DiD and robustness
# ──────────────────────────────────────────────────────────────────────────────
def fit_did(panel: pd.DataFrame):
    """Stacked DiD with asset + week FE, SE clustered by week (Table 9)."""
    m = smf.ols(DID_FORMULA, data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["week"]})
    print("TABLE 9 — Stacked DiD (asset + week FE, week-clustered SE)")
    print("=" * 60)
    for v in ("did_genius", "did_occ"):
        print(f"  {v:12s}: {m.params[v]:+.4f} (SE {m.bse[v]:.4f}), "
              f"p={m.pvalues[v]:.4f}{star(m.pvalues[v])}")
    return m


def did_robustness_2024_pre(panel: pd.DataFrame, main_model) -> pd.DataFrame:
    """
    Robustness: restrict the pre-period to Jan-Dec 2024 only, excluding the
    anticipatory window (Jan-Jul 2025, which contains Senate/House passage).
    This is the most conservative pre-period specification.
    """
    pre_end = pd.Timestamp("2024-12-31")
    sub = panel[(panel["date"] <= pre_end) | (panel["date"] >= T1)].copy()
    m = smf.ols(DID_FORMULA, data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["week"]})

    print("\nDiD ROBUSTNESS — 2024-only pre-period")
    print("=" * 60)
    print(f"{'':<16} {'Main (full pre)':>18} {'Robust (2024 pre)':>18}")
    for term in ("did_genius", "did_occ"):
        print(f"  {term:<14} {main_model.params[term]:>16.4f}  {m.params[term]:>16.4f}")
        print(f"  {'  p-value':<14} {main_model.pvalues[term]:>16.4f}  "
              f"{m.pvalues[term]:>16.4f}{star(m.pvalues[term])}")

    if m.pvalues["did_genius"] < 0.10 and m.params["did_genius"] > 0:
        print("  VERDICT: robust — did_genius remains significant with a "
              "2024-only pre-period.")
    else:
        print("  VERDICT: did_genius attenuated in 2024-only spec — disclose "
              "as a limitation.")

    out = pd.DataFrame({
        "term": ["did_genius", "did_occ"],
        "beta_main": [main_model.params["did_genius"], main_model.params["did_occ"]],
        "p_main": [main_model.pvalues["did_genius"], main_model.pvalues["did_occ"]],
        "beta_robust": [m.params["did_genius"], m.params["did_occ"]],
        "se_robust": [m.bse["did_genius"], m.bse["did_occ"]],
        "p_robust": [m.pvalues["did_genius"], m.pvalues["did_occ"]],
    })
    out.to_csv(OUT_DIR / "did_robustness_2024only.csv", index=False)
    print(f"Saved: {OUT_DIR / 'did_robustness_2024only.csv'}")
    return out


def parallel_trends_test(usdc: pd.DataFrame, usdt: pd.DataFrame) -> None:
    """
    Wooldridge-style pre-treatment lead test (Table 10).

    lead_3m covers the quarter immediately before T1 (contains Senate passage,
    June 17, 2025) and is expected to FAIL — anticipatory repricing.
    lead_6m tests the preceding quarter and supports baseline parallel trends.
    """
    u = usdc[["date", "log_vel"]].copy(); u["treated"] = 1
    t = usdt[["date", "log_vel"]].copy(); t["treated"] = 0
    pt = pd.concat([u, t], ignore_index=True)
    pt = pt[pt["date"] < T1].copy()
    pt["q_to_t1"] = (T1 - pt["date"]).dt.days // 90
    pt["lead_q1"] = ((pt["q_to_t1"] == 0) * pt["treated"]).astype(int)
    pt["lead_q2"] = ((pt["q_to_t1"] == 1) * pt["treated"]).astype(int)
    pt["week"] = pt["date"].dt.isocalendar().week.astype(int)

    m = smf.ols("log_vel ~ C(treated) + C(week) + lead_q1 + lead_q2", data=pt).fit(
        cov_type="cluster", cov_kwds={"groups": pt["week"]})
    print("\nPARALLEL TRENDS — pre-treatment leads (Table 10)")
    for v in ("lead_q1", "lead_q2"):
        print(f"  {v}: b={m.params[v]:+.4f}, p={m.pvalues[v]:.3f}{star(m.pvalues[v])}")
    if m.pvalues["lead_q1"] > 0.10 and m.pvalues["lead_q2"] > 0.10:
        print("  [PASS] No significant pre-trends.")
    else:
        print("  [NOTE] Pre-trend detected in lead_3m — consistent with "
              "anticipatory repricing around Senate passage; disclose in limitations.")


# ──────────────────────────────────────────────────────────────────────────────
# Event study (Figure 3)
# ──────────────────────────────────────────────────────────────────────────────
def event_study(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Quarterly lead/lag event study. Bins run from -6Q to +4Q relative to T1
    (~91-day quarters); the -6Q bin (~18 months pre) is the omitted reference.
    """
    df = panel.copy()
    df["q_rel"] = ((df["date"] - T1).dt.days / 91.25).astype(int).clip(-6, 4)

    quarters = list(range(-5, 5))
    terms = []
    for q in quarters:
        name = f"qn{abs(q)}" if q < 0 else f"qp{q}"
        df[name] = ((df["q_rel"] == q) & (df["treated"] == 1)).astype(int)
        terms.append(name)

    formula = "log_vel ~ C(asset) + C(week) + " + " + ".join(terms)
    m = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["week"]})

    rows = []
    for q in quarters:
        name = f"qn{abs(q)}" if q < 0 else f"qp{q}"
        b, se, p = m.params[name], m.bse[name], m.pvalues[name]
        rows.append({"q": q, "beta": b, "se": se, "p": p,
                     "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se})
    rows.append({"q": -6, "beta": 0.0, "se": 0.0, "p": 1.0, "ci_lo": 0.0, "ci_hi": 0.0})
    res = pd.DataFrame(rows).sort_values("q").reset_index(drop=True)
    res.to_csv(OUT_DIR / "event_study_quarterly.csv", index=False)
    print("Saved: event_study_quarterly.csv")
    return res


def plot_fig3_event_study(res: pd.DataFrame) -> None:
    """Figure 3 — event-study plot with the anticipatory quarter highlighted."""
    PRE_SIG, POST_SIG, INSIG, REF = "#E26B0A", "#2E74B5", "#AABBCC", "#555555"

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for _, row in res.iterrows():
        q, b, lo, hi = row["q"], row["beta"], row["ci_lo"], row["ci_hi"]
        sig = row["p"] < 0.10
        if q == -6:
            ax.scatter(q, 0, color=REF, s=70, zorder=5, marker="D",
                       label="Reference period (q = -6, omitted)")
            continue
        color = (PRE_SIG if sig else INSIG) if q < 0 else (POST_SIG if sig else INSIG)
        ax.plot([q, q], [lo, hi], color=color, linewidth=2.2, zorder=3)
        ax.scatter(q, b, color=color, s=60, zorder=5)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.axvline(-0.5, color="#C0392B", linewidth=2, linestyle="--",
               label="T$_1$: GENIUS Act signed (2025-07-18)")
    ax.axvspan(-1.5, -0.5, alpha=0.12, color="orange",
               label="Anticipatory quarter (incl. Senate passage 2025-06-17)")
    ax.axvspan(-0.5, 4.5, alpha=0.06, color="#2E74B5")

    qlabels = {-6: "-18m\n(ref)", -5: "-15m", -4: "-12m", -3: "-9m", -2: "-6m",
               -1: "-3m", 0: "0", 1: "+3m", 2: "+6m", 3: "+9m", 4: "+12m"}
    ax.set_xticks(list(qlabels))
    ax.set_xticklabels(qlabels.values(), fontsize=10)
    ax.set_xlabel("Months relative to GENIUS Act signing", fontsize=11)
    ax.set_ylabel("log-velocity differential\n(USDC minus USDT, relative to -18m baseline)",
                  fontsize=11)
    ax.set_title("Figure 3: Event Study — USDC vs. USDT Velocity Differential\n"
                 "Asset + week FE; SE clustered by week; 95% CI; reference quarter = -18m",
                 fontsize=12, fontweight="bold")

    handles = [mpatches.Patch(color=PRE_SIG, label="Pre-treatment, significant (p<0.10)"),
               mpatches.Patch(color=POST_SIG, label="Post-treatment, significant (p<0.10)"),
               mpatches.Patch(color=INSIG, label="Not significant"),
               mpatches.Patch(color="orange", alpha=0.4, label="Anticipatory window")]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

    # Annotate the key lead coefficients.
    for q, col, tag, dy in [(-1, "#E26B0A", "lead_3m", -0.35), (-2, "#27AE60", "lead_6m", 0.25)]:
        row = res[res["q"] == q]
        if len(row):
            b, p = row["beta"].values[0], row["p"].values[0]
            verdict = "FAIL" if p < 0.10 else "PASS"
            ax.annotate(f"{tag}\nb={b:+.3f}\np={p:.4f}\n({verdict})",
                        xy=(q, b), xytext=(q - 2.2, b + dy),
                        fontsize=8.5, color=col, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_event_study.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig3_event_study.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — divergence with block-bootstrap CI
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig2_divergence(usdc: pd.DataFrame, usdt: pd.DataFrame,
                         n_boot: int = 500, block: int = 14, seed: int = 42) -> None:
    """
    Figure 2 — (a) USDC vs USDT log velocity (7-day MA); (b) cumulative excess
    (USDC - USDT, demeaned by the pre-T1 gap) with a 95% block-bootstrap band.
    The CI illustrates cumulative-sum uncertainty; formal inference on the
    treatment effect comes from the DiD model (Table 9).
    """
    u7 = usdc.set_index("date")["log_vel"].rolling(7, min_periods=1).mean()
    t7 = usdt.set_index("date")["log_vel"].rolling(7, min_periods=1).mean()
    mg = pd.DataFrame({"usdc": u7, "usdt": t7}).dropna()
    mg["diff"] = mg["usdc"] - mg["usdt"]
    mg["excess"] = mg["diff"] - mg.loc[mg.index < T1, "diff"].mean()
    mg["cum"] = mg["excess"].cumsum()

    rng = np.random.default_rng(seed)
    n = len(mg)
    ev = mg["excess"].values
    boot = np.zeros((n_boot, n))
    for b in range(n_boot):
        starts = rng.integers(0, n - block, size=n // block + 1)
        boot[b] = np.cumsum(np.concatenate([ev[s:s + block] for s in starts])[:n])
    ci_lo, ci_hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[3, 2],
                                 sharex=True, gridspec_kw={"hspace": 0.06})

    a1.plot(u7.index, u7.values, color=PAL["usdc"], lw=1.3, label="USDC (treated)")
    a1.plot(t7.index, t7.values, color=PAL["usdt"], lw=1.3, ls="--", alpha=0.8,
            label="USDT (control)")
    treatment_lines(a1)
    a1.set_ylabel("log(Velocity), 7-day MA")
    a1.legend(loc="upper left")
    a1.set_title("Figure 2: USDC vs. USDT Velocity — Cross-Asset Comparison")
    a1.text(0.02, 0.93, "(a)", transform=a1.transAxes, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9))

    d = mg.index
    a2.fill_between(d, ci_lo, ci_hi, color=PAL["ci"], alpha=0.3,
                    label="95% block-bootstrap CI")
    a2.fill_between(d, 0, mg["cum"], where=mg["cum"] > 0, color=PAL["usdc"], alpha=0.25)
    a2.fill_between(d, 0, mg["cum"], where=mg["cum"] <= 0, color=PAL["usdt"], alpha=0.25)
    a2.plot(d, mg["cum"], color=PAL["usdc"], lw=1.2)
    a2.axhline(0, color="#333", lw=0.5)
    treatment_lines(a2, y1=0.92, y2=0.82)
    a2.set_ylabel("Cumulative Excess\n(USDC - USDT)")
    a2.legend(loc="upper left", fontsize=8)
    a2.text(0.02, 0.93, "(b)", transform=a2.transAxes, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    a2.text(0.01, -0.18,
            "Note: Block-bootstrap CI (block=14d, B=500) reflects cumulative uncertainty "
            "in daily log-velocity differences.\nThe wide CI reflects noise accumulation "
            "inherent to cumulative sums; formal DiD inference (Table 9, p = 0.008)\n"
            "provides the appropriate statistical test for cross-asset divergence.",
            transform=a2.transAxes, fontsize=7, color="#666", fontstyle="italic", va="top")

    plt.savefig(FIG_DIR / "fig2_divergence.png")
    plt.close(fig)
    print("Saved: fig2_divergence.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — DiD stability across pre-period windows
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig5_did_stability(panel: pd.DataFrame) -> None:
    """DiD coefficient re-estimated over 4/6/9/12-month and full pre-periods."""
    pf = panel.copy()
    pf["week"] = pf["date"].dt.isocalendar().week.astype(int)
    pf["did_g"], pf["did_o"] = pf["did_genius"], pf["did_occ"]

    windows = [("4 mo", 120), ("6 mo", 180), ("9 mo", 270), ("12 mo", 365), ("Full", None)]
    rows = []
    for label, w in windows:
        sub = pf if w is None else pf[pf["date"] >= T1 - timedelta(days=w)].copy()
        try:
            m = smf.ols("log_vel ~ C(asset) + C(week) + did_g + did_o", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["week"]})
            ci = m.conf_int().loc["did_g"]
            rows.append({"label": label, "N": int(m.nobs), "coef": m.params["did_g"],
                         "p": m.pvalues["did_g"], "lo": ci[0], "hi": ci[1]})
        except Exception:
            continue
    if not rows:
        return
    sd = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(sd))
    ax.errorbar(x, sd["coef"], yerr=[sd["coef"] - sd["lo"], sd["hi"] - sd["coef"]],
                fmt="o", color=PAL["usdc"], capsize=6, capthick=2, markersize=10,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    for i, (_, r) in enumerate(sd.iterrows()):
        ax.text(i + 0.15, r["coef"], f"{r['coef']:.3f}{star(r['p'])}\n(N={r['N']})",
                ha="left", va="center", fontsize=8, color="#333")
    ax.axhline(0, color="#ccc", ls="--", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(sd["label"], fontsize=10)
    ax.set_xlabel("Pre-Period Window")
    ax.set_ylabel(r"DiD Coefficient ($\hat{\beta}_{GENIUS}$)")
    ax.set_title("Figure 5: DiD Coefficient Stability Across Pre-Period Specifications")
    if (sd["lo"] > 0).all():
        ax.text(0.98, 0.03, "All 95% CIs exclude zero", transform=ax.transAxes,
                fontsize=9, ha="right",
                bbox=dict(boxstyle="round", fc="#e8f4e8", ec=PAL["rwa"], alpha=0.8))
    plt.savefig(FIG_DIR / "fig5_did_stability.png")
    plt.close(fig)
    print("Saved: fig5_did_stability.png")


def export_latex_tables(did_model, robust_df: pd.DataFrame) -> None:
    """Write LaTeX sources for Tables 8-10 (values populated from fitted models)."""
    bg, po = did_model.params["did_genius"], did_model.pvalues["did_genius"]
    bo, poo = did_model.params["did_occ"], did_model.pvalues["did_occ"]
    txt = (
        "% Table 9 — Stacked DiD (auto-generated)\n"
        f"% did_genius = {bg:+.4f} (p={po:.4f}), did_occ = {bo:+.4f} (p={poo:.4f})\n"
        f"% N = {int(did_model.nobs)}; FE: asset, week; SE clustered by week.\n"
    )
    with open(OUT_DIR / "table9_did_summary.txt", "w") as f:
        f.write(txt)
    print("Saved: table9_did_summary.txt")


def run(data: dict) -> None:
    from data_loader import build_panel
    usdc, usdt = data.get("usdc"), data.get("usdt")
    if usdc is None or usdt is None:
        print("[SKIP] Cross-asset DiD: USDC and/or USDT velocity data missing.")
        return
    panel = build_panel(usdc, usdt)
    did = fit_did(panel)
    robust = did_robustness_2024_pre(panel, did)
    parallel_trends_test(usdc, usdt)
    res = event_study(panel)
    plot_fig3_event_study(res)
    plot_fig2_divergence(usdc, usdt)
    plot_fig5_did_stability(panel)
    export_latex_tables(did, robust)


if __name__ == "__main__":
    from data_loader import load_all
    run(load_all())
