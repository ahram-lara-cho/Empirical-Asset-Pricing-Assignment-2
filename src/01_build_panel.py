"""
01_build_panel.py

Build GKX-style feature panel.

Output:
    data/processed/panel_1971_2025.parquet  (+ char_cols/macro_cols/interaction_cols csvs)

Panel key:
    permno, yyyymm

Target:
    ret_fwd = next-month return

Features:
    rank-normalized firm characteristics
    macro predictors (suffixed _macro to avoid name collisions)
    characteristic x macro interactions (float32)
"""

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Utility: next month key
# ------------------------------------------------------------
def next_yyyymm_array(x):
    """Convert yyyymm to next calendar month (201512 -> 201601)."""
    x = np.asarray(x).astype(int)
    y = x // 100
    m = x % 100
    return np.where(m == 12, (y + 1) * 100 + 1, y * 100 + m + 1)


# ------------------------------------------------------------
# 1. CRSP monthly returns
# ------------------------------------------------------------
def pull_crsp_returns(start_year, end_year):
    """
    CRSP monthly returns for NYSE common stocks (exchcd=1, shrcd in 10,11).
    Returns: permno, yyyymm, ret, me
    """
    import wrds

    print("[CRSP] Connecting to WRDS...")
    db = wrds.Connection()

    query = f"""
        SELECT
            a.permno,
            a.date,
            a.ret,
            a.prc,
            a.shrout
        FROM crsp.msf AS a
        JOIN crsp.msenames AS b
          ON a.permno = b.permno
         AND b.namedt <= a.date
         AND a.date <= b.nameendt
        WHERE b.exchcd = 1
          AND b.shrcd IN (10, 11)
          AND a.date BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
    """

    df = db.raw_sql(query, date_cols=["date"])
    db.close()

    print(f"[CRSP] Raw rows: {len(df):,}")

    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["prc"] = pd.to_numeric(df["prc"], errors="coerce")
    df["shrout"] = pd.to_numeric(df["shrout"], errors="coerce")

    df["yyyymm"] = df["date"].dt.year * 100 + df["date"].dt.month
    df["me"] = df["prc"].abs() * df["shrout"]

    df = df.dropna(subset=["ret"])
    df = df[["permno", "yyyymm", "ret", "me"]].copy()

    df["permno"] = df["permno"].astype(int)
    df["yyyymm"] = df["yyyymm"].astype(int)

    print(f"[CRSP] Clean rows: {len(df):,}")
    print(f"[CRSP] Months: {df['yyyymm'].min()} to {df['yyyymm'].max()}")

    return df


# ------------------------------------------------------------
# 2. Forward return target
# ------------------------------------------------------------
def add_forward_target(ret_df):
    """ret_fwd at month t equals ret at month t+1 (calendar-month aligned)."""
    ret_df = ret_df.copy()
    ret_df["yyyymm_next"] = next_yyyymm_array(ret_df["yyyymm"])

    fwd = ret_df[["permno", "yyyymm", "ret"]].copy()
    fwd = fwd.rename(columns={"yyyymm": "yyyymm_next", "ret": "ret_fwd"})

    out = ret_df.merge(fwd, on=["permno", "yyyymm_next"], how="left")
    out = out.drop(columns=["yyyymm_next"])

    print(f"[Target] Missing ret_fwd rate: {out['ret_fwd'].isna().mean():.2%}")
    return out


# ------------------------------------------------------------
# 3. OSAP characteristics
# ------------------------------------------------------------
def pull_characteristics(signals=None, max_signals=None):
    """
    Pull Open Source Asset Pricing firm characteristics.

    signals:
        None  -> pull all available signals (optionally capped by max_signals)
        list  -> pull exactly those signals (names must match OSAP exactly)
    max_signals:
        if set (and signals is None), keep the N signals with the broadest
        coverage (fewest missing) PLUS the OLS-3 core (bm, mom12m), so the
        canonical size/value/momentum benchmark is always available.

    Returns: permno, yyyymm, signal columns
    """
    import openassetpricing as oap

    print("[OSAP] Connecting to OpenAssetPricing...")
    openap = oap.OpenAP()

    if signals is None:
        print("[OSAP] Pulling all signals...")
        df = openap.dl_all_signals("pandas")
        df.columns = [str(c).lower() for c in df.columns]

        if max_signals is not None:
            key = ["permno", "yyyymm"]
            sig_cols = [c for c in df.columns if c not in key]
            coverage = df[sig_cols].notna().mean().sort_values(ascending=False)
            keep = coverage.index[:max_signals].tolist()
            # Always keep the OLS-3 core (value, momentum); size enters as size_crsp.
            for c in ["bm", "mom12m"]:
                if c in sig_cols and c not in keep:
                    keep.append(c)
                    print(f"[OSAP] force-kept core characteristic: {c}")
            df = df[key + keep]
            print(f"[OSAP] capped to {len(keep)} signals (coverage + core)")
    else:
        print(f"[OSAP] Pulling selected signals as one list: {signals}")
        try:
            df = openap.dl_signal("pandas", signals)
        except Exception as e:
            raise RuntimeError(
                f"OSAP dl_signal failed with signals={signals}. "
                "Check exact names via openap.dl_signal_doc('pandas')."
            ) from e
        df.columns = [str(c).lower() for c in df.columns]

    if "permno" not in df.columns or "yyyymm" not in df.columns:
        raise ValueError(f"OSAP output missing permno/yyyymm. Columns: {list(df.columns)[:20]}")

    df["permno"] = df["permno"].astype(int)
    df["yyyymm"] = df["yyyymm"].astype(int)

    non_key_cols = [c for c in df.columns if c not in ["permno", "yyyymm"]]
    print(f"[OSAP] Signal columns ({len(non_key_cols)}): {non_key_cols}")
    print(f"[OSAP] Final rows: {len(df):,} | Months: {df['yyyymm'].min()} to {df['yyyymm'].max()}")
    return df


# ------------------------------------------------------------
# 4. Macro predictors
# ------------------------------------------------------------
def load_macro(path):
    """
    Welch-Goyal macro predictors from Amit Goyal's PredictorData workbook.
    Monthly sheet; S&P level column 'price' (older files: 'index').
    Output (all suffixed _macro): dp, ep, bm, ntis, tbl, tms, dfy, svar
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Macro file not found: {path}\nPut Goyal's PredictorData.xlsx here."
        )

    if path.suffix.lower() in [".xlsx", ".xls"]:
        try:
            raw = pd.read_excel(path, sheet_name="Monthly")
        except ValueError:
            print("[Macro] No 'Monthly' sheet; reading first sheet instead.")
            raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)

    raw.columns = [str(c).strip().lower() for c in raw.columns]

    if "price" in raw.columns:
        index_col = "price"
    elif "index" in raw.columns:
        index_col = "index"
    else:
        raise ValueError(f"Macro file missing price/index column. Columns: {list(raw.columns)}")

    required = ["yyyymm", index_col, "d12", "e12", "b/m", "ntis", "tbl", "lty", "baa", "aaa", "svar"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Macro file missing columns: {missing}. Available: {list(raw.columns)}")

    m = pd.DataFrame()
    m["yyyymm"] = raw["yyyymm"].astype(int)
    idx = pd.to_numeric(raw[index_col], errors="coerce")

    m["dp"] = np.log(pd.to_numeric(raw["d12"], errors="coerce")) - np.log(idx)
    m["ep"] = np.log(pd.to_numeric(raw["e12"], errors="coerce")) - np.log(idx)
    m["bm"] = pd.to_numeric(raw["b/m"], errors="coerce")
    m["ntis"] = pd.to_numeric(raw["ntis"], errors="coerce")
    m["tbl"] = pd.to_numeric(raw["tbl"], errors="coerce")
    m["tms"] = pd.to_numeric(raw["lty"], errors="coerce") - pd.to_numeric(raw["tbl"], errors="coerce")
    m["dfy"] = pd.to_numeric(raw["baa"], errors="coerce") - pd.to_numeric(raw["aaa"], errors="coerce")
    m["svar"] = pd.to_numeric(raw["svar"], errors="coerce")

    # Suffix every macro predictor so it can't collide with an OSAP characteristic
    # name (the full set includes 'ep', 'bm', etc.).
    m = m.rename(columns={c: f"{c}_macro" for c in m.columns if c != "yyyymm"})
    m = m.sort_values("yyyymm").drop_duplicates("yyyymm")

    print(f"[Macro] Rows: {len(m):,} | Months: {m['yyyymm'].min()} to {m['yyyymm'].max()}")
    return m


def align_macro_to_panel_months(panel, macro):
    """Align macro to every panel month; forward-fill so panel months beyond the
    macro file (e.g. 2025 when macro ends 2024-12) aren't silently dropped."""
    macro = macro.sort_values("yyyymm").copy()
    macro_cols = [c for c in macro.columns if c != "yyyymm"]

    all_months = pd.DataFrame({"yyyymm": sorted(panel["yyyymm"].unique())})
    aligned = all_months.merge(macro, on="yyyymm", how="left")
    aligned[macro_cols] = aligned[macro_cols].ffill()

    miss = aligned[macro_cols].isna().mean().sort_values(ascending=False)
    if miss.max() > 0:
        print("[WARNING] Macro still missing after forward-fill:")
        print(miss[miss > 0])

    if panel["yyyymm"].max() > macro["yyyymm"].max():
        print(f"[WARNING] Panel extends beyond macro file "
              f"(macro max {macro['yyyymm'].max()}, panel max {panel['yyyymm'].max()}); "
              "using forward-filled macro for later months.")
    return aligned


# ------------------------------------------------------------
# 5. Cross-sectional rank normalization
# ------------------------------------------------------------
def rank_normalize(panel, char_cols):
    """Per month: rank pct * 2 - 1, missing -> 0 (cross-sectional median)."""
    panel = panel.copy()
    print(f"[Rank] Rank-normalizing {len(char_cols)} characteristics...")
    ranked = (panel.groupby("yyyymm", group_keys=False)[char_cols]
              .rank(pct=True).mul(2.0).sub(1.0).fillna(0.0))
    panel[char_cols] = ranked
    return panel


# ------------------------------------------------------------
# 6. Full panel construction
# ------------------------------------------------------------
def build_feature_matrix(start_year, end_year, signals, macro_path, output_path,
                         max_signals=None):
    """CRSP returns + OSAP characteristics + CRSP size + macro + interactions -> parquet."""
    print("=" * 80)
    print(f"Building panel: {start_year}-{end_year}")
    print("=" * 80)

    # 1. Returns and target
    ret = pull_crsp_returns(start_year, end_year)
    ret = add_forward_target(ret)

    # CRSP-based size (OSAP 'Size' isn't always in the dl_signal namespace)
    ret["size_crsp"] = np.log(pd.to_numeric(ret["me"], errors="coerce").replace(0, np.nan))

    # 2. Characteristics
    chars = pull_characteristics(signals=signals, max_signals=max_signals)
    osap_char_cols = [c for c in chars.columns if c not in ["permno", "yyyymm"]]

    # 3. Merge, then rank-normalize inside the final NYSE sample
    panel = ret.merge(chars, on=["permno", "yyyymm"], how="inner")
    char_cols = ["size_crsp"] + osap_char_cols
    print(f"[Merge] After CRSP x OSAP: {panel.shape}")

    panel = panel.dropna(subset=["size_crsp"])
    panel = rank_normalize(panel, char_cols)

    # 4. Macro
    macro = load_macro(macro_path)
    macro_aligned = align_macro_to_panel_months(panel, macro)
    macro_cols = [c for c in macro_aligned.columns if c != "yyyymm"]
    panel = panel.merge(macro_aligned, on="yyyymm", how="left")
    print(f"[Merge] After macro merge: {panel.shape}")

    # 5. Macro missing check
    print("[Macro] Missing rates:")
    print(panel[macro_cols].isna().mean().sort_values(ascending=False))
    before_macro_drop = len(panel)
    panel = panel.dropna(subset=macro_cols)
    if before_macro_drop != len(panel):
        print(f"[Macro] Dropped rows with unresolved missing macro: {before_macro_drop - len(panel):,}")

    # 6. Interactions (float32 to keep the wide matrix in memory)
    print("[Interactions] Creating characteristic x macro interactions (float32)...")
    interaction_cols = []
    new_cols = {}
    for cc in char_cols:
        for mc in macro_cols:
            name = f"{cc}__x__{mc}"
            new_cols[name] = (panel[cc] * panel[mc]).astype("float32")
            interaction_cols.append(name)
    panel = pd.concat([panel, pd.DataFrame(new_cols, index=panel.index)], axis=1)

    panel["year"] = panel["yyyymm"] // 100

    before = len(panel)
    panel = panel.dropna(subset=["ret_fwd"])
    print(f"[Target] Dropped rows without ret_fwd: {before - len(panel):,}")

    # 7. Sanity checks
    feature_cols = char_cols + macro_cols + interaction_cols
    print("=" * 80)
    print("[Final panel]")
    print(f"Shape: {panel.shape}")
    print(f"Years: {panel['year'].min()} to {panel['year'].max()}")
    print(f"Stocks: {panel['permno'].nunique():,}")
    print(f"Characteristics: {len(char_cols)}")
    print(f"Macro variables: {len(macro_cols)}")
    print(f"Interactions: {len(interaction_cols)}")
    print(f"Total features: {len(feature_cols)}")
    # confirm the OLS-3 core survived
    for c in ["size_crsp", "bm", "mom12m"]:
        print(f"  OLS-3 core '{c}' present: {c in char_cols}")
    print("=" * 80)

    # 8. Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    pd.Series(char_cols).to_csv(output_path.parent / "char_cols.csv", index=False, header=False)
    pd.Series(macro_cols).to_csv(output_path.parent / "macro_cols.csv", index=False, header=False)
    pd.Series(interaction_cols).to_csv(output_path.parent / "interaction_cols.csv", index=False, header=False)
    print(f"[Saved] Panel saved to: {output_path}")

    return panel, char_cols, macro_cols, interaction_cols


if __name__ == "__main__":

    MACRO_PATH = RAW_DIR / "PredictorData.xlsx"

    panel, char_cols, macro_cols, interaction_cols = build_feature_matrix(
        start_year=1971,
        end_year=2025,
        signals=None,        # pull the full OSAP set...
        max_signals=90,      # ...then keep the 90 best-covered + OLS-3 core
        macro_path=MACRO_PATH,
        output_path=PROCESSED_DIR / "panel_1971_2025.parquet",
    )