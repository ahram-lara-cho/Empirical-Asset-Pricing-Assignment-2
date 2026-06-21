"""
05_make_figure9.py

GKX (2020) Figure 9 -- "Cumulative return of machine learning portfolios."

Method (per model, per month t):
    1. sort stocks into deciles on the model's predicted return (pred at t),
    2. go LONG the top decile (10) and SHORT the bottom decile (1),
    3. value-weight within each leg by market equity (me at t),
    4. the realized long-short return is:
           VW return of decile 10  -  VW return of decile 1,
       earned over t -> t+1 (so it is realized in month t+1).
Cumulate the monthly long-short returns, plot on a log y-axis, overlay the
S&P 500 excess return (SP500 - Rf), and shade NBER recession months.

Models required by the assignment for Figure 9:
    OLS-3+H, PCR, ENet+H, RF, NN2, NN4

Two sets are produced in a single run (per-month portfolios are window-
independent, so Set 1 is just the early slice of the same series):
    Set 1: OOS formation years <= 2016 -> outputs/figure9_cumret_set1.png
    Set 2: OOS formation years <= 2024 -> outputs/figure9_cumret_set2.png

Benchmark (SP500 - Rf):
    Pulled from WRDS:
        crsp.msi.sprtrn          (S&P 500 composite monthly return)
        ff.factors_monthly.rf    (Fama-French monthly risk-free rate)
    If WRDS is unavailable, drop a CSV at data/benchmark.csv with columns
        yyyymm, sp500_ret, rf      (returns as decimals, e.g. 0.0123)
    and it is used instead.

Inputs (whichever exist are concatenated):
    outputs/predictions.parquet            (consolidated layout), and/or
    outputs/predictions_simple.parquet     (linear models from step 02), and/or
    outputs/predictions_ml.parquet         (ML models from step 03)
Each must carry: permno, yyyymm, year, ret_fwd, me, pred, model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # headless: write PNGs without a display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ------------------------------------------------------------
# Paths and configuration
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prediction files: existing ones are concatenated, so both the consolidated
# single-file layout and the older two-file layout work unchanged.
PRED_FILES = [
    OUTPUT_DIR / "predictions.parquet",
    OUTPUT_DIR / "predictions_simple.parquet",
    OUTPUT_DIR / "predictions_ml.parquet",
]

MODELS = ["OLS-3+H", "PCR", "ENet+H", "RF", "NN2", "NN4"]   # Figure-9 model set
N_DECILES = 10
WEIGHTING = "value"                 # "value" (GKX) or "equal"
BENCHMARK_CSV = DATA_DIR / "benchmark.csv"   # fallback if WRDS is not used

# Each set is defined by a formation-year cutoff.
SETS = {"set1": 2016, "set2": 2024}

# NBER business-cycle peaks -> troughs (monthly). Fixed historical dates.
NBER_RECESSIONS = [
    ("2001-03", "2001-11"),         # dot-com bust
    ("2007-12", "2009-06"),         # global financial crisis
    ("2020-02", "2020-04"),         # COVID-19
]

# Consistent per-model colors.
MODEL_COLORS = {
    "OLS-3+H": "#1f77b4",
    "PCR":     "#ff7f0e",
    "ENet+H":  "#2ca02c",
    "RF":      "#d62728",
    "NN2":     "#9467bd",
    "NN4":     "#8c564b",
}


# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------
def next_yyyymm(x):
    """yyyymm of the following calendar month (200112 -> 200201)."""
    x = np.asarray(x).astype(int)
    y, m = x // 100, x % 100
    return np.where(m == 12, (y + 1) * 100 + 1, y * 100 + m + 1)


def assign_deciles(pred, n=N_DECILES):
    """Rank-based deciles 1..n. Rank (not value) cuts so that the many TIED
    predictions a random forest produces still split into equal-count bins
    (plain qcut would raise on duplicate bin edges)."""
    r = pred.rank(method="first")
    d = np.ceil(r / len(pred) * n).astype(int)
    return d.clip(1, n)


def weighted_mean(ret, me, weighting):
    """Value- or equal-weighted mean return, robust to missing weights."""
    ret = pd.to_numeric(ret, errors="coerce").to_numpy()
    if weighting == "equal":
        return np.nanmean(ret)
    w = pd.to_numeric(me, errors="coerce").to_numpy()
    w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
    mask = np.isfinite(ret) & np.isfinite(w)
    if not mask.any() or w[mask].sum() <= 0:
        return np.nanmean(ret)      # fall back to equal weight if no valid me
    return np.average(ret[mask], weights=w[mask])


# ------------------------------------------------------------
# 1. Load predictions
# ------------------------------------------------------------
def load_predictions():
    frames = []
    for p in PRED_FILES:
        if p.exists():
            df = pd.read_parquet(p)
            print(f"[load] {p.name}: {df.shape}")
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No prediction files found among: {[str(p) for p in PRED_FILES]}"
        )

    preds = pd.concat(frames, ignore_index=True)
    preds = preds[preds["model"].isin(MODELS)].copy()
    # If a model appears in more than one file, keep one row per key.
    preds = preds.drop_duplicates(subset=["model", "permno", "yyyymm"])

    need = {"permno", "yyyymm", "year", "ret_fwd", "me", "pred", "model"}
    missing = need - set(preds.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    have = sorted(preds["model"].unique())
    print(f"[load] Figure-9 predictions: {preds.shape} | models present: {have}")
    for m in MODELS:
        if m not in have:
            print(f"[WARNING] model '{m}' not found in predictions; it will be skipped.")
    return preds


# ------------------------------------------------------------
# 2. Monthly long-short (decile 10 minus decile 1)
# ------------------------------------------------------------
def long_short_returns(preds, weighting=WEIGHTING):
    """Tidy frame: model, ret_month (realization yyyymm = formation + 1), ls_ret."""
    rows = []
    for (model, ym), g in preds.groupby(["model", "yyyymm"], sort=False):
        if len(g) < N_DECILES:               # need >=1 name to fill each decile
            continue
        d = assign_deciles(g["pred"])
        top = g[d.values == N_DECILES]
        bot = g[d.values == 1]
        if len(top) == 0 or len(bot) == 0:
            continue
        r_top = weighted_mean(top["ret_fwd"], top["me"], weighting)
        r_bot = weighted_mean(bot["ret_fwd"], bot["me"], weighting)
        rows.append((model, int(next_yyyymm(ym)), r_top - r_bot))

    ls = pd.DataFrame(rows, columns=["model", "ret_month", "ls_ret"])
    return ls


# ------------------------------------------------------------
# 3. Benchmark: SP500 - Rf
# ------------------------------------------------------------
def load_benchmark(start_yyyymm, end_yyyymm):
    """Monthly S&P 500 excess return indexed by realization yyyymm."""
    start_y, end_y = start_yyyymm // 100, end_yyyymm // 100

    try:
        import wrds
        db = wrds.Connection()
        sp = db.raw_sql(
            f"SELECT date, sprtrn FROM crsp.msi "
            f"WHERE date BETWEEN '{start_y}-01-01' AND '{end_y}-12-31'",
            date_cols=["date"],
        )
        ff = db.raw_sql(
            f"SELECT date, rf FROM ff.factors_monthly "
            f"WHERE date BETWEEN '{start_y}-01-01' AND '{end_y}-12-31'",
            date_cols=["date"],
        )
        db.close()
        for d in (sp, ff):
            d["yyyymm"] = d["date"].dt.year * 100 + d["date"].dt.month
        b = (sp[["yyyymm", "sprtrn"]]
             .merge(ff[["yyyymm", "rf"]], on="yyyymm", how="inner")
             .rename(columns={"sprtrn": "sp500_ret"}))
        print(f"[benchmark] WRDS: {len(b)} months")
    except Exception as e:
        print(f"[benchmark] WRDS unavailable ({type(e).__name__}); "
              f"trying CSV fallback {BENCHMARK_CSV}")
        if not BENCHMARK_CSV.exists():
            raise FileNotFoundError(
                "No benchmark available. Either run where WRDS is reachable, or "
                "provide data/benchmark.csv with columns [yyyymm, sp500_ret, rf] "
                "(returns as decimals)."
            )
        b = pd.read_csv(BENCHMARK_CSV)

    b["sp500_ret"] = pd.to_numeric(b["sp500_ret"], errors="coerce")
    b["rf"] = pd.to_numeric(b["rf"], errors="coerce")

    # Scale guard: convert to decimal if the file stored percent (e.g. rf ~ 0.4).
    if b["rf"].abs().median() > 0.02:
        print("[benchmark] rf looks like percent -> dividing by 100")
        b["rf"] = b["rf"] / 100.0
    if b["sp500_ret"].abs().median() > 0.05:
        print("[benchmark] sp500_ret looks like percent -> dividing by 100")
        b["sp500_ret"] = b["sp500_ret"] / 100.0

    b["sp500_exc"] = b["sp500_ret"] - b["rf"]
    b = b[(b["yyyymm"] >= start_yyyymm) & (b["yyyymm"] <= end_yyyymm)]
    return b[["yyyymm", "sp500_exc"]].sort_values("yyyymm").reset_index(drop=True)


# ------------------------------------------------------------
# 4. Cumulate + plot one set
# ------------------------------------------------------------
def make_figure_for_set(preds, set_name, end_year):
    sub = preds[preds["year"] <= end_year].copy()
    if sub.empty:
        print(f"[{set_name}] no predictions with formation year <= {end_year}; skipping.")
        return

    ls = long_short_returns(sub, WEIGHTING)
    if ls.empty:
        print(f"[{set_name}] no long-short returns computed; skipping.")
        return

    # month x model matrix of monthly long-short returns
    wide = (ls.pivot_table(index="ret_month", columns="model", values="ls_ret")
              .sort_index())
    wide = wide.reindex(columns=[m for m in MODELS if m in wide.columns])

    # benchmark over the realized window
    bstart, bend = int(wide.index.min()), int(wide.index.max())
    bench = load_benchmark(bstart, bend).set_index("yyyymm")["sp500_exc"]
    wide["SP500-Rf"] = bench.reindex(wide.index)
    if wide["SP500-Rf"].isna().any():
        n_missing = int(wide["SP500-Rf"].isna().sum())
        print(f"[{set_name}] benchmark missing for {n_missing} month(s); "
              "those months contribute a flat benchmark step.")

    # cumulative compounded growth of $1 (log scale matches GKX's Figure 9)
    cum = (1.0 + wide.fillna(0.0)).cumprod()

    # persist the underlying monthly returns for the write-up / Sharpe etc.
    ret_csv = OUTPUT_DIR / f"figure9_lsret_{set_name}.csv"
    wide.to_csv(ret_csv)
    print(f"[{set_name}] saved monthly long-short returns -> {ret_csv.name}")

    _plot(cum, set_name, end_year)


def _plot(cum, set_name, end_year):
    dates = pd.to_datetime(cum.index.astype(int).astype(str), format="%Y%m")
    lo, hi = dates.min(), dates.max()

    fig, ax = plt.subplots(figsize=(11, 6))

    # NBER recession shading (clipped to the plotted range)
    for s, e in NBER_RECESSIONS:
        s_d = pd.to_datetime(s)
        e_d = pd.to_datetime(e) + pd.offsets.MonthEnd(0)
        if e_d >= lo and s_d <= hi:
            ax.axvspan(max(s_d, lo), min(e_d, hi), color="grey", alpha=0.15, lw=0)

    for col in cum.columns:
        if col == "SP500-Rf":
            ax.plot(dates, cum[col].to_numpy(), color="black", lw=1.7, ls="--",
                    label="S&P 500 - Rf")
        else:
            ax.plot(dates, cum[col].to_numpy(), color=MODEL_COLORS.get(col),
                    lw=1.5, label=col)

    ax.set_yscale("log")
    ax.set_ylabel("Cumulative return ($1 start, log scale)")
    weight_tag = "value-weighted" if WEIGHTING == "value" else "equal-weighted"
    ax.set_title(f"Cumulative return of ML long-short portfolios "
                 f"({weight_tag}) -- OOS 2001-{end_year}")
    ax.legend(ncol=2, fontsize=9, frameon=False)
    ax.grid(True, which="both", axis="y", alpha=0.2)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    out = OUTPUT_DIR / f"figure9_cumret_{set_name}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[{set_name}] saved figure -> {out}")


# ------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------
def main():
    preds = load_predictions()
    for set_name, end_year in SETS.items():
        print("=" * 70)
        print(f"[{set_name}] formation years <= {end_year}")
        make_figure_for_set(preds, set_name, end_year)
    print("=" * 70)
    print("[done]")


if __name__ == "__main__":
    main()
