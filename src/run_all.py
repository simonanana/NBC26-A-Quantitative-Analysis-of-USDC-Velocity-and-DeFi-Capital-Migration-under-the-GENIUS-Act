"""
End-to-end reproduction pipeline.

Usage:
    cd src && python run_all.py

Runs, in order:
    1. Data ingestion (data_loader)
    2. H1 — ITS velocity analysis, Sup-Wald scan, ADF, permutation inference
    3. H1 — Cross-asset DiD, event study, parallel trends, robustness
    4. H2 — Ternary capital reallocation (local TVL file or DefiLlama API)
    5. H3 — LP concentration (HHI / Gini / Top-5)

Figures are written to figures/, tabular outputs to output/. Modules whose
input data is missing are skipped with a console notice rather than failing.
"""

import h1_cross_asset_did
import h1_velocity_its
import h2_capital_reallocation
import h3_lp_concentration
from config import FIG_DIR, OUT_DIR
from data_loader import load_all


def main() -> None:
    data = load_all()

    print("\n" + "=" * 70)
    print("H1 — VELOCITY ITS")
    print("=" * 70)
    h1_velocity_its.run(data)

    print("\n" + "=" * 70)
    print("H1 — CROSS-ASSET DiD (USDC vs USDT)")
    print("=" * 70)
    h1_cross_asset_did.run(data)

    print("\n" + "=" * 70)
    print("H2 — TERNARY CAPITAL REALLOCATION")
    print("=" * 70)
    h2_capital_reallocation.run(data)

    print("\n" + "=" * 70)
    print("H3 — LP CONCENTRATION")
    print("=" * 70)
    h3_lp_concentration.run(data)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Figures: {FIG_DIR}")
    print(f"  Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
