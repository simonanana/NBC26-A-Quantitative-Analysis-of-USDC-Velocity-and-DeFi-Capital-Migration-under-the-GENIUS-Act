# From Passive Yield to Active Utility

**A Quantitative Analysis of USDC Velocity and DeFi Capital Migration under the GENIUS Act**

Replication code and analysis pipeline for a quasi-experimental study of how Section 4(a)(11) of the GENIUS Act of 2025 — which prohibits permitted payment stablecoin issuers from paying yield to token holders — reshaped on-chain USDC behavior. The paper exploits the Act's enactment as a shock to the on-chain yield regime, using an interrupted time-series (ITS) design supplemented by a cross-asset difference-in-differences (DiD) with USDT (offshore-issued, outside the Act's regulatory perimeter) as the control.

## Key Findings

| Hypothesis | Result |
|---|---|
| **H1 — Velocity Acceleration** | Adjusted USDC velocity nearly doubled (0.750x → 1.484x/day). Signing (T₁) operates through a slope channel (β₃ = +0.0093 log-points/day, p < 0.001); the OCC NPRM (T₂) through a preliminary level channel. USDT shows no significant response (level p = 0.85; slope p = 0.48). Stacked DiD: USDC–USDT differential of +0.348 log-points (p = 0.008). |
| **H2 — Ternary Capital Reallocation** | Within the three-category yield panel: lending share +8.6 pp, tokenized RWA +5.1 pp, DEX liquidity share −13.8 pp (all p < 0.001). |
| **H3 — LP Concentration** | Wallet-aggregated HHI in the USDC/WETH 0.05% Uniswap V3 pool rises from 0.159 (Jun 2024) to 0.414 (Oct 2025), consistent with displacement of less-capitalized providers. |

**Mechanism.** A Baumol–Tobin extension to programmable assets: with passive yield *r* set to zero by statute while active on-chain deployment returns *ρ* remain positive, the opportunity cost of idle holding rises discretely — reversing the classical *r*-down → *V*-down prediction and producing a *forced redeployment* from passive storage to active utility.

**Timeline.** T₁ = 2025-07-18 (GENIUS Act signed, Pub. L. 119-27) · T₂ = 2026-02-25 (OCC NPRM, 91 Fed. Reg. 10,202) · Sample: 2024-01-01 → 2026-03-31 (daily).

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py                  # Event dates, sample window, paths, plot style, HHI/Gini helpers
│   ├── data_loader.py             # Robust ingestion of Artemis/DefiLlama/Dune exports
│   ├── h1_velocity_its.py         # ITS regressions, Fig 1, Sup-Wald scan (Fig A1), ADF, permutation
│   ├── h1_cross_asset_did.py      # Stacked DiD, event study (Fig 3), Figs 2 & 5, parallel trends
│   ├── h2_capital_reallocation.py # Ternary Lending/RWA/DEX shares, Fig 4 (coef + stacked area)
│   ├── h3_lp_concentration.py     # Wallet-aggregated HHI/Gini/Top-5, Fig 6
│   └── run_all.py                 # End-to-end pipeline
├── data/
│   ├── README.md                  # Data dictionary and acquisition instructions
│   ├── manual/                    # ← place raw CSV/XLSX inputs here (not committed)
│   └── api_cache/                 # DefiLlama API cache (auto-created)
├── figures/                       # Generated figures (Figs 1–6, A1)
└── output/                        # Event-study estimates, DiD robustness, table sources
```

## Reproduction

```bash
git clone https://github.com/simonanana/NBC26-A-Quantitative-Analysis-of-USDC-Velocity-and-DeFi-Capital-Migration-under-the-GENIUS-Act.git
cd NBC26-A-Quantitative-Analysis-of-USDC-Velocity-and-DeFi-Capital-Migration-under-the-GENIUS-Act
pip install -r requirements.txt

# Place input data in data/manual/ (see data/README.md), then:
cd src
python run_all.py
```

Each hypothesis module can also be run standalone (e.g. `python h2_capital_reallocation.py`). Modules degrade gracefully: if an optional input is missing, the corresponding analysis is skipped with a console notice, and H2 falls back to the public DefiLlama API.

## Data

Raw data files are **not** redistributed in this repository due to provider terms of service. See [`data/README.md`](data/README.md) for the full data dictionary and step-by-step acquisition instructions (Artemis Analytics, DefiLlama, Dune Analytics, Etherscan).

| File | Role | Required? |
|---|---|---|
| `artemis_usdc_velocity.csv` | H1 — USDC adjusted transfer volume + circulating supply | Yes |
| `artemis_usdt_velocity.csv` | H1 — USDT control arm for the DiD | Yes |
| `protocol_tvl_timeseries.csv` | H2 — 7-protocol daily TVL | Optional (API fallback) |
| `dune_uniswap_lp_snapshot.csv` | H3 — LP position snapshots, USDC/WETH 0.05% pool | Optional |
| `macro_controls_merged.csv` | ITS macro controls: log(DeFi TVL), log(ETH gas) | Optional |
| `aave_usdc_apy.csv` | Rate-compression discussion (paper §6.1) | Optional |

## Methodology Summary

- **ITS (eq. 5).** `log(V_t) = α + β₁t_c + β₂Post_GENIUS + β₃(t_c × Post_GENIUS) + β₄Post_OCC + γ'X_t + ε_t`, with the time trend centered at T₁; Newey–West HAC standard errors (lag 14).
- **Cross-asset DiD (eq. 6).** Asset + week fixed effects, standard errors clustered by week; USDT as control. Robustness: 2024-only pre-period (excluding the anticipatory window), DiD stability over 4/6/9/12-month pre-windows, quarterly event study, and Wooldridge-style lead tests (`lead_3m` fails as expected — anticipatory repricing around the June 17, 2025 Senate passage — while `lead_6m` passes).
- **Structural break.** Sequential Chow (Sup-Wald, Andrews 1993) scan with 15% trimming; data-driven break at 2025-08-28, 41 days after T₁.
- **Inference transparency.** The single-asset ITS slope is not robust to permutation inference (p ≈ 0.196, reported honestly); the cross-asset DiD is therefore the primary identification of the GENIUS-specific effect.
- **Concentration.** Wallet-aggregated HHI, Gini, and Top-5 share on Uniswap V3 LP snapshots (V4 excluded: ERC-6909 singleton architecture precludes provider-level reconstruction).

## Requirements

Python ≥ 3.10 with `pandas`, `numpy`, `statsmodels`, `scipy`, `matplotlib`, `openpyxl` (see `requirements.txt`).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{nbc26usdc,
  title     = {From Passive Yield to Active Utility: A Quantitative Analysis of
               {USDC} Velocity and {DeFi} Capital Migration under the {GENIUS} Act},
  author    = {[Author names]},
  year      = {2026},
  booktitle = {[Venue]}
}
```

## Disclaimer

This repository is provided for academic research and reproducibility. Nothing herein constitutes financial, investment, or legal advice.

## License

Code released under the MIT License. Raw data remain subject to the terms of their respective providers (Artemis Analytics, DefiLlama, Dune Analytics).
