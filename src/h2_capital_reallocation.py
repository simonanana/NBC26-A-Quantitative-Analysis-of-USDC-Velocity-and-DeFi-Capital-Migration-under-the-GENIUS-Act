"""
H2 — Ternary Capital Reallocation: Lending vs. RWA vs. DEX shares.

Daily TVL for seven protocols (Aave V3; Uniswap V3/V4; BUIDL, Ondo, BENJI,
USYC, Mountain USDM) is grouped into three yield categories. Shares are
computed within the three-category panel (the intra-yield-seeking total).
Pre/post-T1 share changes are estimated with HAC(14) standard errors.

Data: local protocol_tvl_timeseries.csv if present, otherwise the public
DefiLlama API (responses cached under data/api_cache/).

Outputs:
    figures/fig4_h2_coef.png           Pre/post share-change coefficient plot
    figures/fig4_ternary_stacked.png   Stacked area chart of shares over time
"""

import json
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import (API_CACHE_DIR, COLORS as C, FIG_DIR, SAMPLE_END,
                    SAMPLE_START, T1, T2, star)

DEFI_LLAMA_SLUGS = {
    "aave-v3": "Lending",
    "uniswap-v3": "DEX",
    "uniswap-v4": "DEX",
    "blackrock-buidl": "RWA",
    "ondo-finance": "RWA",
    "mountain-protocol": "RWA",
}

CATEGORY_KEYWORDS = {
    "RWA": ["buidl", "ondo", "benji", "usyc", "mountain", "rwa"],
    "Lending": ["aave", "lending", "compound"],
    "DEX": ["uniswap", "dex", "sushi"],
}


def api_fetch(url: str, name: str, max_age_hours: int = 168) -> dict | None:
    """Fetch a JSON endpoint with a simple local file cache."""
    p = API_CACHE_DIR / f"{name}.json"
    if p.exists() and (time.time() - p.stat().st_mtime) / 3600 < max_age_hours:
        with open(p) as f:
            return json.load(f)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NBC26/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
        with open(p, "w") as f:
            json.dump(data, f)
        time.sleep(1)  # polite rate limiting
        return data
    except Exception as e:
        print(f"  [API] {e}")
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None


def build_ternary_from_local(tvl_local: pd.DataFrame) -> pd.DataFrame | None:
    """Build the Lending/RWA/DEX ternary panel from a local TVL export."""
    tvl = tvl_local.copy()
    tvl["date"] = pd.to_datetime(tvl["date"])
    tvl = tvl[(tvl["date"] >= SAMPLE_START) & (tvl["date"] <= SAMPLE_END)]

    cols = {cat: [c for c in tvl.columns
                  if any(k in c.lower() for k in kws)]
            for cat, kws in CATEGORY_KEYWORDS.items()}
    if not all(cols.values()):
        print("  [WARN] could not auto-detect category columns in local TVL file.")
        return None

    ternary = tvl[["date"]].copy()
    ternary["Lending"] = tvl[cols["Lending"]].sum(axis=1)
    ternary["RWA"] = tvl[cols["RWA"]].sum(axis=1)
    ternary["DEX"] = tvl[cols["DEX"]].sum(axis=1)
    ternary = ternary.set_index("date").ffill().dropna()
    print(f"  [OK] ternary built from local file: {len(ternary)} obs")
    return ternary


def build_ternary_from_api() -> pd.DataFrame | None:
    """Build the ternary panel from the DefiLlama protocol API."""
    print("  Using DefiLlama API for H2")
    series = {"Lending": [], "DEX": [], "RWA": []}
    for slug, cat in DEFI_LLAMA_SLUGS.items():
        data = api_fetch(f"https://api.llama.fi/protocol/{slug}", f"dl_{slug}")
        if data and "tvl" in data:
            rows = [{"date": pd.Timestamp(x["date"], unit="s"),
                     "tvl": x["totalLiquidityUSD"]}
                    for x in data["tvl"] if "totalLiquidityUSD" in x]
            df = pd.DataFrame(rows)
            df = df[(df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)]
            series[cat].append(df.sort_values("date").set_index("date")["tvl"].rename(slug))
            print(f"  [OK] {slug}")
    if not all(series.values()):
        return None
    cat_totals = {cat: pd.concat(s, axis=1).sum(axis=1) for cat, s in series.items()}
    return pd.DataFrame(cat_totals).ffill().dropna()


def estimate_share_changes(ternary: pd.DataFrame) -> pd.DataFrame:
    """Pre/post-T1 change in each category share, OLS on Post with HAC(14) SE."""
    ternary = ternary.copy()
    total = ternary[["Lending", "DEX", "RWA"]].sum(axis=1)
    for cat in ("Lending", "DEX", "RWA"):
        ternary[f"{cat}_sh"] = ternary[cat] / total
    ternary["post"] = (ternary.index >= T1).astype(int)

    rows = []
    for cat in ("Lending", "RWA", "DEX"):
        y = ternary[f"{cat}_sh"].values
        X = sm.add_constant(ternary["post"].values)
        r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 14})
        pre_m = ternary.loc[ternary["post"] == 0, f"{cat}_sh"].mean()
        post_m = ternary.loc[ternary["post"] == 1, f"{cat}_sh"].mean()
        rows.append({"cat": cat, "pre": pre_m, "post": post_m,
                     "dpp": (post_m - pre_m) * 100,
                     "b": r.params[1], "se": r.bse[1], "p": r.pvalues[1],
                     "ci_lo": r.conf_int()[1][0] * 100,
                     "ci_hi": r.conf_int()[1][1] * 100})
    return pd.DataFrame(rows)


def plot_fig4_coefficients(h2df: pd.DataFrame) -> None:
    """Figure 4 (coefficient version) — share changes with HAC 95% CIs."""
    labels = ["Lending\n(Aave V3)", "RWA\n(BUIDL, Ondo, Mt.)", "DEX\n(Uniswap V3/V4)"]
    colors = [C["lend"], C["rwa"], C["dex"]]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    yp = np.arange(len(h2df))
    ax.barh(yp, h2df["dpp"], height=0.5, color=colors, alpha=0.7,
            edgecolor=colors, lw=1.2)
    ax.errorbar(h2df["dpp"], yp,
                xerr=[h2df["dpp"] - h2df["ci_lo"], h2df["ci_hi"] - h2df["dpp"]],
                fmt="none", color="#333", capsize=5, capthick=1.5, zorder=5)
    for i, (_, r) in enumerate(h2df.iterrows()):
        x_pos = r["ci_hi"] + 0.8 if r["dpp"] > 0 else r["ci_lo"] - 0.8
        ha = "left" if r["dpp"] > 0 else "right"
        ax.text(x_pos, i,
                f"{r['dpp']:+.1f} pp ({r['pre']*100:.1f}%->{r['post']*100:.1f}%)\n"
                f"p = {r['p']:.3f}{star(r['p'])}",
                ha=ha, va="center", fontsize=8.5, color="#333")

    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(yp)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Change in Share (pp), HAC 95% CI")
    ax.set_title("Figure 4: Ternary Capital Reallocation — Pre vs. Post GENIUS Act")
    ax.text(0.98, 0.03, "Note: Protocol-level TVL.\nSee paper section 6.3 for limitations.",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            color="#999", fontstyle="italic")
    plt.savefig(FIG_DIR / "fig4_h2_coef.png")
    plt.close(fig)
    print("Saved: fig4_h2_coef.png")


def plot_fig4_stacked_area(ternary: pd.DataFrame) -> None:
    """Figure 4 (stacked-area version) — 14-day-smoothed category shares."""
    tvl = ternary.reset_index().rename(columns={"index": "date"})
    if "date" not in tvl.columns:
        tvl = tvl.rename(columns={tvl.columns[0]: "date"})
    total = tvl[["Lending", "RWA", "DEX"]].sum(axis=1)
    for cat in ("Lending", "RWA", "DEX"):
        tvl[f"s_{cat}"] = (tvl[cat] / total * 100).rolling(14, min_periods=1).mean()

    pre_means = {c: tvl.loc[tvl["date"] < T1, f"s_{c}"].mean() for c in ("Lending", "RWA", "DEX")}
    post_means = {c: tvl.loc[tvl["date"] >= T1, f"s_{c}"].mean() for c in ("Lending", "RWA", "DEX")}

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(tvl["date"], tvl["s_Lending"], tvl["s_RWA"], tvl["s_DEX"],
                 labels=["Passive lending / Aave V3 (H2a)",
                         "RWA protocols / BUIDL+Ondo+BENJI+USYC (H2b)",
                         "DEX passive liquidity / Uniswap V3+V4 (H2c)"],
                 colors=["#2E74B5", "#5FAD56", "#E26B0A"], alpha=0.82)
    ax.axvline(T1, color="#C0392B", linewidth=2, linestyle="--",
               label=f"T1: GENIUS signed ({T1.date()})")
    ax.axvline(T2, color="#8E44AD", linewidth=2, linestyle=":",
               label=f"T2: OCC NPRM ({T2.date()})")

    for when, means, xpos in (("Pre-GENIUS", pre_means, T1 - pd.Timedelta(days=180)),
                              ("Post-GENIUS", post_means, T1 + pd.Timedelta(days=90))):
        ax.annotate(f"{when}\nDEX: {means['DEX']:.1f}%\nLending: {means['Lending']:.1f}%\n"
                    f"RWA: {means['RWA']:.1f}%",
                    xy=(xpos, 50), fontsize=8.5, color="white", ha="center",
                    va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1F3864", alpha=0.7))

    ax.set_ylabel("Share of three-category TVL panel (%)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title("Figure 4: Ternary Capital Reallocation in USDC-bearing DeFi Yield "
                 "Categories\n(shares computed within three-protocol panel; "
                 "14-day rolling average)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_ternary_stacked.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig4_ternary_stacked.png")


def run(data: dict) -> None:
    ternary = None
    if data.get("tvl") is not None:
        print("Using local protocol_tvl_timeseries for H2")
        ternary = build_ternary_from_local(data["tvl"])
    if ternary is None:
        ternary = build_ternary_from_api()
    if ternary is None:
        print("[SKIP] H2: no TVL data available (local file missing and API failed).")
        return

    h2df = estimate_share_changes(ternary)
    print("\nH2 — share changes (pre vs post T1):")
    print(h2df[["cat", "pre", "post", "dpp", "p"]].round(4).to_string(index=False))
    plot_fig4_coefficients(h2df)
    plot_fig4_stacked_area(ternary)


if __name__ == "__main__":
    from data_loader import load_all
    run(load_all())
