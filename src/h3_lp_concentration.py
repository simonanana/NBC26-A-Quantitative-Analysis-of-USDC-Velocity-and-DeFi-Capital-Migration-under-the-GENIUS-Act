"""
H3 — Liquidity-Provider Concentration in the USDC/WETH 0.05% Uniswap V3 pool.

LP position snapshots (Dune Analytics) are aggregated to the wallet level per
snapshot date, and concentration is measured with the wallet-aggregated
Herfindahl-Hirschman Index (HHI), Gini coefficient, and Top-5 provider share.

The analysis is restricted to Uniswap V3 for architectural comparability:
V4's singleton architecture stores positions as ERC-6909 claim tokens, making
provider-level reconstruction infeasible over the sample window.

Outputs:
    figures/fig6_h3_concentration.png   HHI and Top-5 share across snapshots
    Console: per-snapshot HHI / Gini / Top-5 and pre/post means
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from config import COLORS as C, FIG_DIR, T1, gini, hhi

LIQ_CANDIDATES = ("liquidity", "liquidity_usd", "total_liquidity", "liq_usd")
WALLET_CANDIDATES = ("provider", "owner", "wallet", "address", "lp_address")


def detect_columns(lp: pd.DataFrame) -> dict:
    """Auto-detect the liquidity, wallet, date, and pool columns."""
    return {
        "liq": next((c for c in lp.columns if c in LIQ_CANDIDATES), None),
        "wallet": next((c for c in lp.columns if c in WALLET_CANDIDATES), None),
        "date": next((c for c in lp.columns if "date" in c or "snapshot" in c), None),
        "pool": next((c for c in lp.columns if "pool" in c or "fee" in c or "tier" in c), None),
    }


def compute_concentration(lp_raw: pd.DataFrame) -> pd.DataFrame | None:
    """Wallet-aggregated concentration metrics per snapshot date."""
    lp = lp_raw.copy()
    lp.columns = [c.strip().lower() for c in lp.columns]
    cols = detect_columns(lp)
    print(f"  Detected columns: {cols}")
    if not (cols["liq"] and cols["wallet"] and cols["date"]):
        print("  [WARN] required columns not found (need liquidity, wallet/address, date).")
        return None

    lp[cols["date"]] = pd.to_datetime(lp[cols["date"]])
    lp[cols["liq"]] = pd.to_numeric(lp[cols["liq"]], errors="coerce")
    lp = lp.dropna(subset=[cols["liq"], cols["wallet"]])

    # Restrict to the 0.05% fee tier if a pool/fee column exists.
    if cols["pool"]:
        pools = lp[cols["pool"]].unique()
        target = [p for p in pools
                  if "0.05" in str(p) or "500" in str(p) or "005" in str(p)]
        if target:
            lp = lp[lp[cols["pool"]].isin(target)]
            print(f"  Filtered to 0.05% fee tier: {len(lp)} rows")

    agg = lp.groupby([cols["date"], cols["wallet"]])[cols["liq"]].sum().reset_index()
    agg = agg[agg[cols["liq"]] > 0]

    rows = []
    for dt in sorted(agg[cols["date"]].unique()):
        vals = agg.loc[agg[cols["date"]] == dt, cols["liq"]].values
        total = vals.sum()
        rows.append({
            "date": dt,
            "n_lps": len(vals),
            "hhi": hhi(vals),
            "gini": gini(vals),
            "top5_share": np.sort(vals)[-5:].sum() / total if total > 0 else np.nan,
            "post": int(dt >= T1),
        })
    return pd.DataFrame(rows)


def report(cdf: pd.DataFrame) -> None:
    print(f"\nH3 concentration results ({len(cdf)} snapshots)")
    for _, r in cdf.iterrows():
        tag = "POST" if r["post"] else "PRE "
        print(f"  {tag} {r['date'].date()}: N={r['n_lps']:4d}  HHI={r['hhi']:.4f}  "
              f"Gini={r['gini']:.3f}  Top5={r['top5_share']*100:.1f}%")

    pre, post = cdf[cdf["post"] == 0], cdf[cdf["post"] == 1]
    if len(pre) and len(post):
        print(f"\n  HHI:  {pre['hhi'].mean():.4f} -> {post['hhi'].mean():.4f} "
              f"(+{(post['hhi'].mean() / pre['hhi'].mean() - 1) * 100:.0f}%)")
        print(f"  Gini: {pre['gini'].mean():.3f} -> {post['gini'].mean():.3f}")
        print(f"  Top5: {pre['top5_share'].mean()*100:.1f}% -> "
              f"{post['top5_share'].mean()*100:.1f}%")


def plot_fig6(cdf: pd.DataFrame) -> None:
    """Figure 6 — HHI and Top-5 share per snapshot, pre vs post coloring."""
    if len(cdf) < 3:
        print("  [SKIP] fewer than 3 snapshots — figure not generated.")
        return
    pre = cdf[cdf["post"] == 0]
    colors = [C["usdc"] if p else C["grey"] for p in cdf["post"]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"wspace": 0.35})

    a1.bar(range(len(cdf)), cdf["hhi"], color=colors, alpha=0.7,
           edgecolor=colors, lw=1.2)
    a1.set_xticks(range(len(cdf)))
    a1.set_xticklabels([d.strftime("%Y-%m") for d in cdf["date"]],
                       rotation=45, ha="right", fontsize=8)
    a1.set_ylabel("HHI")
    a1.set_title("(a) Herfindahl-Hirschman Index")
    if len(pre):
        a1.axhline(pre["hhi"].mean(), color=C["grey"], ls=":", lw=1, alpha=0.6)
        a1.text(0.3, pre["hhi"].mean() + 0.002,
                f"Pre mean: {pre['hhi'].mean():.4f}", fontsize=7.5, color=C["grey"])

    a2.bar(range(len(cdf)), cdf["top5_share"] * 100, color=colors, alpha=0.7,
           edgecolor=colors, lw=1.2)
    a2.set_xticks(range(len(cdf)))
    a2.set_xticklabels([d.strftime("%Y-%m") for d in cdf["date"]],
                       rotation=45, ha="right", fontsize=8)
    a2.set_ylabel("Top-5 Share (%)")
    a2.set_title("(b) Top-5 Liquidity Provider Share")
    a2.legend(handles=[Patch(facecolor=C["grey"], alpha=0.7, label="Pre-GENIUS"),
                       Patch(facecolor=C["usdc"], alpha=0.7, label="Post-GENIUS")],
              fontsize=8, loc="lower right")

    fig.suptitle("Figure 6: LP Concentration — USDC/WETH 0.05% Uniswap V3 Pool",
                 fontweight="bold", fontsize=12, y=1.02)
    plt.savefig(FIG_DIR / "fig6_h3_concentration.png")
    plt.close(fig)
    print("Saved: fig6_h3_concentration.png")


def run(data: dict) -> None:
    lp_raw = data.get("lp")
    if lp_raw is None:
        print("[SKIP] H3: dune_uniswap_lp_snapshot.csv not found "
              "(place it in data/manual/).")
        return
    cdf = compute_concentration(lp_raw)
    if cdf is None:
        return
    report(cdf)
    plot_fig6(cdf)


if __name__ == "__main__":
    from data_loader import load_all
    run(load_all())
