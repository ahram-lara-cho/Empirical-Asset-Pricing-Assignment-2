"""
06_make_final1_table.py

Compile GKX Table 1 -- "Monthly out-of-sample stock-level prediction
performance" -- as both a table and a figure, FOR BOTH REQUIRED SETS:

    Set 1: sample 1971-2016  -> OOS test years 2001-2016
    Set 2: sample 1971-2025  -> OOS test years 2001-2024 (panel max)

Because each per-year prediction depends only on its own rolling split
(train <= y-3, valid y-2..y-1, test y) and NOT on the sample-end year, the
Set 1 numbers are obtained by slicing the full prediction set to year <= 2016 --
identical to a separate Set 1 run, with no recomputation.

Inputs:
    outputs/predictions_simple.parquet   (OLS+H, OLS-3+H, PCR, ENet+H from step 02)
    outputs/predictions_ml.parquet       (RF, NN2, NN4 from step 03)
    -> these must hold the FULL 2001-2024 range for slicing to yield both sets.

Outputs (per set):
    outputs/predictions_all_<set>.parquet
    outputs/table1_final_<set>.csv
    outputs/table1_final_display_<set>.csv
    outputs/figure_table1_oos_r2_<set>.png
    outputs/figure_table1_monthly_r2_<set>.png

Models: OLS+H, OLS-3+H, PCR, ENet+H, RF, NN2, NN4   ("All" sample only)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # headless: no display needed
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["OLS+H", "OLS-3+H", "PCR", "ENet+H", "RF", "NN2", "NN4"]

# Each set is defined by a sample-period end year (OOS test years 2001..E).
SETS = {"set1": 2016, "set2": 2024}


# ------------------------------------------------------------
# 1. Load prediction files (loaded once; sliced per set)
# ------------------------------------------------------------
def load_predictions():
    simple_path = OUTPUT_DIR / "predictions_simple.parquet"
    ml_path = OUTPUT_DIR / "predictions_ml.parquet"
    for p in (simple_path, ml_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    simple = pd.read_parquet(simple_path)
    ml = pd.read_parquet(ml_path)
    print("=" * 80)
    print("[Load predictions]")
    print(f"Simple predictions: {simple.shape}")
    print(f"ML predictions:     {ml.shape}")

    preds = pd.concat([simple, ml], ignore_index=True)

    required = ["permno", "yyyymm", "year", "ret_fwd", "pred", "model"]
    missing = [c for c in required if c not in preds.columns]
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    preds["ret_fwd"] = pd.to_numeric(preds["ret_fwd"], errors="coerce")
    preds["pred"] = pd.to_numeric(preds["pred"], errors="coerce")
    preds["yyyymm"] = preds["yyyymm"].astype(int)
    preds["year"] = preds["year"].astype(int)
    preds = preds.dropna(subset=["ret_fwd", "pred"])
    preds = preds[preds["model"].isin(MODEL_ORDER)].copy()

    print("[Models found]")
    print(preds["model"].value_counts())

    missing_models = [m for m in MODEL_ORDER if m not in preds["model"].unique()]
    if missing_models:
        print(f"[WARNING] Missing models: {missing_models}")

    # Alignment check: every model should cover the same OOS span, otherwise the
    # pooled comparison mixes windows (e.g. one model run to 2016, another to 2024).
    span = preds.groupby("model")["year"].agg(["min", "max"])
    print("[Model OOS spans]")
    print(span)
    if span["min"].nunique() > 1 or span["max"].nunique() > 1:
        print("[WARNING] Models do not share the same OOS span -- re-run 02/03 so "
              "all seven cover the same test years before trusting the pooled table.")
    print("=" * 80)
    return preds


# ------------------------------------------------------------
# 2. Table 1 metrics
# ------------------------------------------------------------
def compute_oos_r2(y, yhat):
    """GKX stock-level OOS R2 = 1 - sum((r-rhat)^2)/sum(r^2); zero benchmark."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    sst = np.sum(y ** 2)
    if sst == 0:
        return np.nan
    return 1.0 - np.sum((y - yhat) ** 2) / sst


def compute_table1(preds):
    rows = []
    for model_name, g in preds.groupby("model"):
        r2_oos = compute_oos_r2(g["ret_fwd"].values, g["pred"].values)

        monthly_r2, monthly_obs = [], []
        for _, gm in g.groupby("yyyymm"):
            r2_m = compute_oos_r2(gm["ret_fwd"].values, gm["pred"].values)
            if not np.isnan(r2_m):
                monthly_r2.append(r2_m)
                monthly_obs.append(len(gm))

        rows.append({
            "model": model_name,
            "r2_oos": r2_oos,
            "r2_oos_percent": 100 * r2_oos,
            "mean_monthly_r2": np.mean(monthly_r2),
            "mean_monthly_r2_percent": 100 * np.mean(monthly_r2),
            "median_monthly_r2": np.median(monthly_r2),
            "median_monthly_r2_percent": 100 * np.median(monthly_r2),
            "n_obs": len(g),
            "n_months": g["yyyymm"].nunique(),
            "avg_obs_per_month": np.mean(monthly_obs),
            "start_month": int(g["yyyymm"].min()),
            "end_month": int(g["yyyymm"].max()),
        })

    table = pd.DataFrame(rows)
    table["model"] = pd.Categorical(table["model"], categories=MODEL_ORDER, ordered=True)
    return table.sort_values("model").reset_index(drop=True)


# ------------------------------------------------------------
# 3. Display table
# ------------------------------------------------------------
def make_display_table(table):
    display = table[[
        "model", "r2_oos_percent", "mean_monthly_r2_percent",
        "median_monthly_r2_percent", "n_obs", "n_months", "start_month", "end_month",
    ]].rename(columns={
        "model": "Model",
        "r2_oos_percent": "OOS R2 (%)",
        "mean_monthly_r2_percent": "Mean Monthly R2 (%)",
        "median_monthly_r2_percent": "Median Monthly R2 (%)",
        "n_obs": "Obs.", "n_months": "Months",
        "start_month": "Start", "end_month": "End",
    })
    for c in ["OOS R2 (%)", "Mean Monthly R2 (%)", "Median Monthly R2 (%)"]:
        display[c] = display[c].map(lambda x: f"{x:.4f}")
    display["Obs."] = display["Obs."].map(lambda x: f"{int(x):,}")
    display["Months"] = display["Months"].map(lambda x: f"{int(x):,}")
    return display


# ------------------------------------------------------------
# 4. Table 1 figures (per set)
# ------------------------------------------------------------
def make_table1_figures(table, set_name, end_year):
    plot_df = table.copy()
    plot_df["model"] = plot_df["model"].astype(str)

    for col, ylabel, title, fname in [
        ("r2_oos_percent", "OOS R² (%)",
         f"Monthly Out-of-Sample Stock-Level Prediction Performance (1971-{end_year})",
         f"figure_table1_oos_r2_{set_name}.png"),
        ("mean_monthly_r2_percent", "Mean Monthly OOS R² (%)",
         f"Mean Monthly Out-of-Sample R² by Model (1971-{end_year})",
         f"figure_table1_monthly_r2_{set_name}.png"),
    ]:
        plt.figure(figsize=(9, 5))
        plt.bar(plot_df["model"], plot_df[col])
        plt.axhline(0, linewidth=1)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel("Model")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out = OUTPUT_DIR / fname
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[{set_name}] saved figure -> {out.name}")


# ------------------------------------------------------------
# 5. Save outputs (per set)
# ------------------------------------------------------------
def save_outputs(preds, table, set_name, end_year):
    preds.to_parquet(OUTPUT_DIR / f"predictions_all_{set_name}.parquet", index=False)
    table.to_csv(OUTPUT_DIR / f"table1_final_{set_name}.csv", index=False)
    display = make_display_table(table)
    display.to_csv(OUTPUT_DIR / f"table1_final_display_{set_name}.csv", index=False)

    print(f"[{set_name}] Table 1 (1971-{end_year}):")
    print(display.to_string(index=False))
    make_table1_figures(table, set_name, end_year)


# ------------------------------------------------------------
# 6. Main -- loop over both sets
# ------------------------------------------------------------
def main():
    preds = load_predictions()
    for set_name, end_year in SETS.items():
        print("=" * 80)
        print(f"[{set_name}] slicing to OOS years 2001-{end_year}")
        sub = preds[preds["year"] <= end_year].copy()
        if sub.empty:
            print(f"[{set_name}] no predictions with year <= {end_year}; skipping.")
            continue
        table = compute_table1(sub)
        save_outputs(sub, table, set_name, end_year)
    print("=" * 80)
    print("[done]")


if __name__ == "__main__":
    main()