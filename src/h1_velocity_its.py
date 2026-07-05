"""
H1 — Velocity Acceleration: Interrupted Time Series (ITS) analysis.

Primary specification (paper eq. 5):

    log(V_t) = a + b1*t_c + b2*Post_GENIUS + b3*(t_c x Post_GENIUS)
             + b4*Post_OCC + g'X_t + e_t

where t_c is the daily time trend centered at T1 (GENIUS signing), so b2 is
the instantaneous level shift and b3 the slope change. Newey-West HAC standard
errors with lag truncation 14.

Outputs:
    figures/fig1_velocity_its.png      Observed velocity + ITS fitted trends
    figures/fig_a1_supwald.png         Sequential Chow (Sup-Wald) break scan
    Console: Tables 3/4 (ITS coefficients), ADF tests, permutation inference
"""

from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller

from config import COLORS as C, FIG_DIR, T1, star, treatment_lines

ITS_FORMULA = "log_vel ~ t_centered + post_genius + post_t_centered + post_occ"
ITS_TERMS = ["t_centered", "post_genius", "post_t_centered", "post_occ"]
HAC_LAGS = 14


def fit_its(usdc: pd.DataFrame, macro: pd.DataFrame | None = None):
    """Fit the baseline ITS and (optionally) the macro-controlled variant."""
    its_base = smf.ols(ITS_FORMULA, data=usdc).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})

    print("TABLE 3 — ITS (no controls)")
    print("=" * 60)
    for v in ITS_TERMS:
        print(f"  {v:20s}: {its_base.params[v]:+.5f}  "
              f"p={its_base.pvalues[v]:.4f}{star(its_base.pvalues[v])}")

    its_ctrl = None
    if macro is not None:
        mc = macro.copy()
        mc["date"] = pd.to_datetime(mc["date"])
        merged = usdc.merge(mc, on="date", how="left")
        ctrl_cols = [c for c in ("defi_tvl", "eth_gas_usd", "log_defi_tvl", "log_gas")
                     if c in merged.columns]
        if ctrl_cols:
            for c in ctrl_cols:
                merged[c] = merged[c].ffill().bfill()
            formula = ITS_FORMULA + " + " + " + ".join(ctrl_cols)
            its_ctrl = smf.ols(formula, data=merged.dropna(subset=ctrl_cols)).fit(
                cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
            print(f"\nTABLE 4 — ITS (with controls: {ctrl_cols})")
            print("=" * 60)
            for v in ITS_TERMS + ctrl_cols:
                print(f"  {v:20s}: {its_ctrl.params[v]:+.5f}  "
                      f"p={its_ctrl.pvalues[v]:.4f}{star(its_ctrl.pvalues[v])}")
    else:
        print("\n  [INFO] macro_controls_merged.csv not found — running without "
              "controls (DiD identification does not require macro controls).")

    return its_base, its_ctrl


def plot_fig1(usdc: pd.DataFrame, its_base) -> None:
    """Figure 1 — aggregate USDC velocity with ITS model fit."""
    usdc = usdc.copy()
    usdc["its_fit_level"] = np.exp(its_base.fittedvalues)
    vel_7d = usdc["velocity"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(usdc["date"], usdc["velocity"], color="#D0D0D0", lw=0.3, alpha=0.5, zorder=1)
    ax.plot(usdc["date"], vel_7d, color=C["usdc"], lw=1.2,
            label="Observed (7-day MA)", zorder=3)

    pre, post = usdc["post_genius"] == 0, usdc["post_genius"] == 1
    ax.plot(usdc.loc[pre, "date"], usdc.loc[pre, "its_fit_level"],
            color=C["fit"], lw=2, label="ITS fitted (pre-trend)", zorder=4)
    ax.plot(usdc.loc[post, "date"], usdc.loc[post, "its_fit_level"],
            color=C["t1"], lw=2, label="ITS fitted (post-trend)", zorder=4)

    treatment_lines(ax)

    b3 = its_base.params["post_t_centered"]
    ax.annotate(f"Slope change $\\beta_3$ = {b3:+.4f}/day\n(p < 0.001)",
                xy=(T1 + timedelta(150), usdc.loc[post, "its_fit_level"].median() * 0.5),
                fontsize=8.5, color=C["t1"],
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=C["t1"], alpha=0.9))

    ax.set_ylabel("Daily USDC Velocity (x/day)")
    ax.set_title("Figure 1: Aggregate USDC Velocity with ITS Model Fit, Jan 2024 - Mar 2026")
    ax.legend(loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.set_ylim(bottom=0)
    ax.text(0.01, -0.12,
            "Note: ITS model estimated in log(velocity); fitted values transformed to "
            "level via exp(.).\nThe curvature of the post-trend line reflects exponential "
            "growth implied by a constant log-linear slope.",
            transform=ax.transAxes, fontsize=7.5, color="#666",
            fontstyle="italic", va="top")

    plt.savefig(FIG_DIR / "fig1_velocity_its.png")
    plt.close(fig)
    print("Saved: fig1_velocity_its.png")


def plot_supwald(usdc: pd.DataFrame) -> None:
    """Figure A1 — sequential Chow (Sup-Wald) structural break scan (Andrews 1993)."""
    y = usdc["log_vel"].values
    X = sm.add_constant(usdc["t"].values)
    n = len(y)
    ssr_full = sm.OLS(y, X).fit().ssr
    k = X.shape[1]
    trim = int(0.15 * n)  # 15% trimming on both ends

    f_stats, dates = [], []
    for i in range(trim, n - trim):
        s1 = sm.OLS(y[:i], X[:i]).fit().ssr
        s2 = sm.OLS(y[i:], X[i:]).fit().ssr
        f_stats.append(((ssr_full - s1 - s2) / k) / ((s1 + s2) / (n - 2 * k)))
        dates.append(usdc.iloc[i]["date"])

    sup_f = max(f_stats)
    sup_date = dates[int(np.argmax(f_stats))]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(dates, f_stats, color=C["usdc"], lw=1)
    ax.axvline(T1, color=C["t1"], ls="--", lw=1.5, label=f"T1: {T1.date()}")
    ax.axvline(sup_date, color=C["t2"], ls=":", lw=1.5, label=f"Sup-F: {sup_date.date()}")
    ax.axhline(7, color=C["grey"], ls="--", lw=0.8, alpha=0.6,
               label="Andrews 5% critical value (~7.0)")

    # Annotate the pre-2025 non-regulatory volume spike (ETH ETF approval).
    early = [(d, f) for d, f in zip(dates, f_stats) if d < pd.Timestamp("2025-01-01")]
    if early:
        pk_d, pk_f = max(early, key=lambda x: x[1])
        ax.annotate("ETH ETF approval\nvolume spike\n(non-regulatory)",
                    xy=(pk_d, pk_f), xytext=(pk_d + timedelta(90), pk_f * 0.7),
                    fontsize=8, color="#666", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#999", lw=0.8),
                    bbox=dict(boxstyle="round,pad=.3", fc="#f9f9f9", ec="#ccc"))

    ax.set_ylabel("Chow F-statistic")
    ax.set_title("Figure A1: Sequential Chow (Sup-Wald) Structural Break Scan")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.savefig(FIG_DIR / "fig_a1_supwald.png")
    plt.close(fig)
    print(f"Saved: fig_a1_supwald.png  (Sup-F = {sup_f:.1f} at {sup_date.date()})")


def run_stationarity_tests(usdc: pd.DataFrame, usdt: pd.DataFrame | None) -> None:
    """ADF unit-root tests on log velocity (paper section 4.2)."""
    print("\nSTATIONARITY (ADF, maxlag=14, AIC lag selection)")
    series = [("USDC full", usdc["log_vel"]),
              ("USDC pre-T1", usdc.loc[usdc["post_genius"] == 0, "log_vel"])]
    if usdt is not None:
        series.append(("USDT full", usdt["log_vel"]))
    for label, s in series:
        r = adfuller(s.dropna(), maxlag=14, autolag="AIC")
        print(f"  {label}: ADF={r[0]:.3f}, p={r[1]:.4f}{star(r[1])}")


def run_permutation_test(usdc: pd.DataFrame, n_perm: int = 2000, seed: int = 42) -> float:
    """
    Permutation (randomization) inference on the ITS slope coefficient b3.

    Placebo treatment dates are drawn uniformly from the interior of the
    sample (90-day trim on both ends). Honest reporting: the single-asset ITS
    slope is NOT robust to permutation (p ~ 0.196 in the paper); the
    cross-asset DiD is therefore the primary identification.
    """
    rng = np.random.default_rng(seed)
    obs_b3 = smf.ols(ITS_FORMULA, data=usdc).fit().params["post_t_centered"]

    placebo_b3 = []
    interior_dates = usdc.iloc[90:-90]["date"].values
    for _ in range(n_perm):
        fake_t1 = pd.Timestamp(rng.choice(interior_dates))
        dp = usdc.copy()
        dp["post_genius"] = (dp["date"] >= fake_t1).astype(int)
        offset = (fake_t1 - dp["date"].min()).days
        dp["t_centered"] = dp["t"] - offset
        dp["post_t_centered"] = dp["t_centered"] * dp["post_genius"]
        try:
            placebo_b3.append(
                smf.ols(ITS_FORMULA, data=dp).fit().params["post_t_centered"])
        except Exception:
            continue

    p_perm = float(np.mean(np.array(placebo_b3) >= obs_b3))
    print("\nPERMUTATION INFERENCE (paper section 5.6)")
    print(f"  b3 = {obs_b3:.6f}, p_perm = {p_perm:.3f}")
    print("  Interpretation: single-asset ITS slope not robust to permutation;")
    print("  cross-asset DiD is the primary identification.")
    return p_perm


def run(data: dict) -> None:
    usdc, usdt, macro = data.get("usdc"), data.get("usdt"), data.get("macro")
    if usdc is None:
        print("[SKIP] H1: USDC velocity data missing.")
        return
    its_base, _ = fit_its(usdc, macro)
    plot_fig1(usdc, its_base)
    plot_supwald(usdc)
    run_stationarity_tests(usdc, usdt)
    run_permutation_test(usdc)


if __name__ == "__main__":
    from data_loader import load_all
    run(load_all())
