"""
Robust data ingestion ("omni-reader").

Handles the messy realities of analytics exports: XLSX files disguised with a
.csv extension, legacy .xls, preamble/junk rows before the true header,
mixed encodings (UTF-8/BOM, GBK, Latin-1), and non-comma separators.

Expected input files (place in data/manual/ — see data/README.md):
    artemis_usdc_velocity.csv     (required, H1)
    artemis_usdt_velocity.csv     (required, H1 DiD control)
    protocol_tvl_timeseries.csv   (optional, H2; falls back to DefiLlama API)
    dune_uniswap_lp_snapshot.csv  (optional, H3)
    macro_controls_merged.csv     (optional, ITS controls)
    aave_usdc_apy.csv             (optional, rate-compression discussion)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, BASE_DIR, SAMPLE_START, SAMPLE_END, T1, T2

SEARCH_DIRS = [DATA_DIR, BASE_DIR / "data" / "raw", BASE_DIR]

HEADER_KEYWORDS = ["date", "time", "tvl", "supply", "volume", "apy", "pool"]


def load_generic(filename: str) -> pd.DataFrame | None:
    """
    Load a tabular file with automatic format detection.

    Strategy:
      1. Resolve the file across search directories and extension variants
         (.csv / .xlsx / .xls) if the exact name is not found.
      2. Sniff the magic bytes: 'PK\\x03\\x04' -> XLSX (openpyxl);
         D0 CF 11 E0 -> legacy XLS (xlrd).
      3. Otherwise treat as delimited text: scan the first 30 lines for a
         plausible header row (skipping export preambles), then try encoding x
         separator combinations until one parses into >1 column.
    """
    base_name, _ = os.path.splitext(filename)
    candidates = [filename, f"{base_name}.csv", f"{base_name}.xlsx", f"{base_name}.xls"]

    target = None
    for d in SEARCH_DIRS:
        for name in candidates:
            p = Path(d) / name
            if p.exists():
                target = p
                break
        if target:
            break

    if target is None:
        print(f"  [MISSING] {filename} (tried {candidates})")
        return None

    print(f"  Loading: {target}")

    with open(target, "rb") as f:
        header = f.read(4)

    # XLSX disguised as CSV (zip container)
    if header == b"PK\x03\x04":
        try:
            return pd.read_excel(target)
        except Exception as e:
            print(f"  [ERROR] XLSX parse failed: {e}")
            return None

    # Legacy XLS (OLE2 container)
    if header.startswith(b"\xd0\xcf\x11\xe0"):
        try:
            return pd.read_excel(target, engine="xlrd")
        except Exception as e:
            print(f"  [ERROR] legacy XLS parse failed: {e}")
            return None

    # Delimited text: locate the true header row, then brute-force parse.
    skip_rows, detected_enc = 0, "utf-8-sig"
    for enc in ["utf-8-sig", "gbk", "latin1"]:
        try:
            with open(target, "r", encoding=enc) as f:
                for i, line in enumerate(f):
                    if i > 30:
                        break
                    if any(kw in line.lower() for kw in HEADER_KEYWORDS):
                        skip_rows, detected_enc = i, enc
                        break
            break
        except Exception:
            continue

    for sep in [",", "\t", ";", r"\s+"]:
        for enc in {detected_enc, "utf-8-sig", "utf-8", "gbk"}:
            try:
                df = pd.read_csv(target, encoding=enc, sep=sep, skiprows=skip_rows,
                                 engine="python", on_bad_lines="skip")
                if len(df.columns) > 1:
                    print(f"  [OK] skipped {skip_rows} preamble row(s); "
                          f"encoding={enc}, sep={sep!r}, rows={len(df)}")
                    return df
            except Exception:
                continue

    print(f"  [ERROR] could not parse {target}")
    return None


def build_velocity_frame(raw: pd.DataFrame | None, label: str) -> pd.DataFrame | None:
    """
    Standardize a raw Artemis export into the velocity analysis frame.

    Adds: velocity (adjusted transfer volume / circulating supply), log_vel,
    post_genius / post_occ indicators, and centered time trends for the ITS
    specification (t_centered = 0 at T1).
    """
    if raw is None:
        return None

    label_lower = label.lower()
    rename = {}
    for c in raw.columns:
        cl = str(c).lower().strip().replace("\ufeff", "").replace('"', "")
        if cl in ("date", "datetime"):
            rename[c] = "date"
        elif f"{label_lower}_adj_vol" in cl or cl == "adj_vol":
            rename[c] = "adj_vol"
        elif f"{label_lower}_supply" in cl or cl == "supply":
            rename[c] = "supply"

    df = raw.rename(columns=rename)
    if "date" not in df.columns:
        print(f"  [WARN] {label}: no 'date' column. Columns: {list(df.columns)[:8]}")
        return None

    df = df.loc[:, ~df.columns.duplicated()]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)]
    df = df.sort_values("date").reset_index(drop=True)

    if not {"adj_vol", "supply"}.issubset(df.columns):
        print(f"  [WARN] {label}: missing adj_vol/supply in {list(df.columns)[:8]}")
        return None

    # Strip thousands separators before numeric coercion.
    for col in ("adj_vol", "supply"):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    df["velocity"] = df["adj_vol"] / df["supply"]
    df["log_vel"] = np.log(df["velocity"].clip(lower=1e-10))
    df["post_genius"] = (df["date"] >= T1).astype(int)
    df["post_occ"] = (df["date"] >= T2).astype(int)
    df["t"] = (df["date"] - df["date"].min()).dt.days
    t1_days = (T1 - df["date"].min()).days
    df["t_centered"] = df["t"] - t1_days
    df["post_t_centered"] = df["t_centered"] * df["post_genius"]
    df["week"] = df["date"].dt.to_period("W").astype(str)

    print(f"  [OK] {label}: {len(df)} obs, velocity range "
          f"[{df['velocity'].min():.3f}, {df['velocity'].max():.3f}]")
    return df


def load_all() -> dict:
    """Load every data source and return a dict of frames (None if missing)."""
    print("=== Loading velocity data ===")
    usdc = build_velocity_frame(load_generic("artemis_usdc_velocity"), "USDC")
    usdt = build_velocity_frame(load_generic("artemis_usdt_velocity"), "USDT")

    print("\n=== Loading supplementary data ===")
    data = {
        "usdc": usdc,
        "usdt": usdt,
        "tvl": load_generic("protocol_tvl_timeseries"),
        "lp": load_generic("dune_uniswap_lp_snapshot"),
        "macro": load_generic("macro_controls_merged"),
        "aave_apy": load_generic("aave_usdc_apy"),
    }

    print("\n=== Data loading summary ===")
    for k, v in data.items():
        print(f"  {'[OK]     ' if v is not None else '[MISSING]'} {k}")
    return data


def build_panel(usdc: pd.DataFrame, usdt: pd.DataFrame) -> pd.DataFrame | None:
    """Stack USDC (treated) and USDT (control) into a long DiD panel."""
    if usdc is None or usdt is None:
        return None
    u = usdc.copy(); u["asset"] = "USDC"; u["treated"] = 1
    t = usdt.copy(); t["asset"] = "USDT"; t["treated"] = 0
    panel = pd.concat([u, t], ignore_index=True)
    panel["did_genius"] = panel["treated"] * panel["post_genius"]
    panel["did_occ"] = panel["treated"] * panel["post_occ"]
    return panel.sort_values(["asset", "date"]).reset_index(drop=True)
