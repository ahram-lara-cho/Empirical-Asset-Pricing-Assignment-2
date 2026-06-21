"""
03_train_ml_models.py

Train RF, NN2, NN4 on the processed GKX-style panel.

Output:
    outputs/predictions_ml.parquet
    outputs/table1_ml.csv
    outputs/_ckpt_ml/ml_pred_<year>.parquet   (per-year checkpoints)

Models:
    RF
    NN2  hidden (32, 16)
    NN4  hidden (32, 16, 8, 4)   <- GKX geometric pyramid

SPEED / LAPTOP NOTES:
    * FIRM_SUBSAMPLE keeps a random fraction of *firms* (whole histories), the
      assignment's bias-free way to cut cost while keeping the full 1971 timeline.
    * N_JOBS = -1 uses all cores (fastest). Set -2 to leave one core free.
    * NN tuning is on a row-subsample (TUNE_MAX_ROWS); the ensemble is fit on
      full train. Larger NN_BATCH cuts per-epoch overhead.
    * CHECKPOINTING: each test year is saved as soon as it finishes and skipped
      on restart, so sleep/crash costs one year, not the whole run.
      >> If you change PANEL_NAME / FIRM_SUBSAMPLE / N_ENSEMBLE / NN_ALPHAS /
         any config, delete the outputs/_ckpt_ml folder first so stale years
         aren't reused. <<
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


warnings.filterwarnings("ignore")


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUTPUT_DIR / "_ckpt_ml"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Run configuration (the lines you change between runs) ------------------
PANEL_NAME      = "panel_1971_2025.parquet"  # match what you saved in step 01
START_TEST_YEAR = 2001                        # 1971 + 18y train + 12y val -> first OOS 2001
END_TEST_YEAR   = None                         # None = panel's last year; set 2016 for Set 1
FIRM_SUBSAMPLE  = 0.30                          # 0.30 = laptop-friendly; set None for full universe
SUBSAMPLE_SEED  = 42
N_ENSEMBLE      = 10                            # GKX use 10 (validated to fix the NN negatives)
N_JOBS          = -1                            # -1 = all cores (fastest); -2 = leave one free
NN_BATCH        = 2048                          # larger batch -> fewer Python-loop iters -> faster
NN_ALPHAS       = (1e-4, 1e-3, 1e-2)            # L2 grid; 1e-2 added (stronger reg, validated)
TUNE_MAX_ROWS   = 150_000                       # rows used to PICK the NN alpha (full train used for the ensemble)
RESUME          = True                          # skip test years already checkpointed
# -----------------------------------------------------------------------------


# ------------------------------------------------------------
# 1. Load data and feature columns
# ------------------------------------------------------------
def read_col_list(path):
    return pd.read_csv(path, header=None).iloc[:, 0].tolist()


def load_panel():
    panel_path = DATA_DIR / PANEL_NAME
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}")

    panel = pd.read_parquet(panel_path)

    # Optional bias-free firm subsample: pick permnos once so a firm is either
    # fully in or fully out (a random slice of the cross-section, not of rows).
    if FIRM_SUBSAMPLE is not None:
        permnos = panel["permno"].unique()
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        keep = rng.choice(permnos, size=int(len(permnos) * FIRM_SUBSAMPLE), replace=False)
        panel = panel[panel["permno"].isin(keep)]
        print(f"[Subsample] kept {len(keep):,}/{len(permnos):,} firms ({FIRM_SUBSAMPLE:.0%})")

    char_cols = read_col_list(DATA_DIR / "char_cols.csv")
    macro_cols = read_col_list(DATA_DIR / "macro_cols.csv")
    interaction_cols = read_col_list(DATA_DIR / "interaction_cols.csv")
    x_cols = char_cols + macro_cols + interaction_cols

    print("=" * 80)
    print("[Load panel]")
    print(f"Panel: {PANEL_NAME}")
    print(f"Panel shape: {panel.shape}")
    print(f"Years: {panel['year'].min()} to {panel['year'].max()}")
    print(f"Stocks: {panel['permno'].nunique():,}")
    print(f"Features: {len(x_cols)}")
    print("=" * 80)
    return panel, x_cols


# ------------------------------------------------------------
# 2. Rolling splits
# ------------------------------------------------------------
def make_rolling_splits(panel, start_test_year, end_test_year=None):
    """Expanding window: train years <= y-3, valid y-2..y-1, test y."""
    if end_test_year is None:
        end_test_year = int(panel["year"].max())

    years = sorted(panel["year"].unique())
    print("=" * 80)
    print("[Rolling split setup]")
    print(f"Available years: {min(years)} to {max(years)}")
    print(f"Test years: {start_test_year} to {end_test_year}")
    print("=" * 80)

    for y in range(start_test_year, end_test_year + 1):
        train = panel[panel["year"] <= y - 3]
        valid = panel[(panel["year"] >= y - 2) & (panel["year"] <= y - 1)]
        test = panel[panel["year"] == y]
        if len(train) == 0 or len(valid) == 0 or len(test) == 0:
            print(f"[Split {y}] skipped due to empty train/valid/test.")
            continue
        print(f"[Split {y}] train={train.shape}, valid={valid.shape}, test={test.shape}")
        yield y, train, valid, test


# ------------------------------------------------------------
# 3. Utilities (clean once, reuse across models)
# ------------------------------------------------------------
def prepare_block(df, x_cols):
    X = (df[x_cols]
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0)
         .astype(np.float32)
         .values)
    y = pd.to_numeric(df["ret_fwd"], errors="coerce").astype(np.float32).values
    return X, y


def _row_subsample(X, y, max_rows, seed=0):
    """Random row subsample (for cheap hyperparameter ranking only)."""
    if len(X) <= max_rows:
        return X, y
    idx = np.random.default_rng(seed).choice(len(X), size=max_rows, replace=False)
    return X[idx], y[idx]


def make_prediction_frame(test, pred, model_name):
    out = test[["permno", "yyyymm", "year", "ret_fwd", "ret", "me"]].copy()
    out["pred"] = pred
    out["model"] = model_name
    return out


# ------------------------------------------------------------
# 4. RF
# ------------------------------------------------------------
def fit_predict_rf(Xtr, ytr, Xva, yva, Xte, test_df):
    """Random Forest with a small validation-tuned grid."""
    candidates = [
        {"max_depth": 4, "min_samples_leaf": 200},
        {"max_depth": 6, "min_samples_leaf": 200},
        {"max_depth": 6, "min_samples_leaf": 100},
    ]

    best_model, best_params, best_mse = None, None, np.inf
    for params in candidates:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features="sqrt",
            n_jobs=N_JOBS,
            random_state=42,
        )
        model.fit(Xtr, ytr)
        mse = mean_squared_error(yva, model.predict(Xva))
        print(f"    [RF candidate] {params}, valid MSE={mse:.6g}")
        if mse < best_mse:
            best_mse, best_model, best_params = mse, model, params

    pred = best_model.predict(Xte)
    print(f"    [RF] best params={best_params}, valid MSE={best_mse:.6g}")
    return make_prediction_frame(test_df, pred, "RF")


# ------------------------------------------------------------
# 5. NN2 / NN4  (ensembled, GKX-style)
# ------------------------------------------------------------
def _make_mlp(hidden, alpha, lr, seed):
    return MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=lr,
        batch_size=NN_BATCH,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=seed,
    )


def fit_predict_nn(Xtr, ytr, Xva, yva, Xte, test_df, model_name):
    """
    NN with GKX geometric-pyramid width, ensembled over N_ENSEMBLE seeds.
    Tune alpha (over NN_ALPHAS) with ONE net per candidate on a row-subsample,
    then fit the ensemble on full train with the winning alpha and average.
    """
    if model_name == "NN2":
        hidden = (32, 16)
    elif model_name == "NN4":
        hidden = (32, 16, 8, 4)        # GKX pyramid
    else:
        raise ValueError(f"Unknown NN model name: {model_name}")

    scaler = StandardScaler().fit(Xtr)
    Ztr, Zva, Zte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    # --- tune alpha on a cheap row-subsample ---
    Ztune, ytune = _row_subsample(Ztr, ytr, TUNE_MAX_ROWS)
    best_alpha, best_mse = None, np.inf
    for alpha in NN_ALPHAS:
        net = _make_mlp(hidden, alpha, 1e-3, seed=0)
        net.fit(Ztune, ytune)
        mse = mean_squared_error(yva, net.predict(Zva))
        print(f"    [{model_name} alpha={alpha:g}] valid MSE={mse:.6g}")
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha

    # --- fit the ensemble on full train with the winning alpha ---
    preds = []
    for s in range(N_ENSEMBLE):
        net = _make_mlp(hidden, best_alpha, 1e-3, seed=s)
        net.fit(Ztr, ytr)
        preds.append(net.predict(Zte))
    pred = np.mean(preds, axis=0)

    print(f"    [{model_name}] best alpha={best_alpha:g}, ensemble={N_ENSEMBLE}, "
          f"valid MSE={best_mse:.6g}")
    return make_prediction_frame(test_df, pred, model_name)


# ------------------------------------------------------------
# 6. Table 1-style OOS performance
# ------------------------------------------------------------
def compute_table1(predictions):
    """R2_oos = 1 - sum((r - rhat)^2) / sum(r^2)."""
    rows = []
    for model_name, g in predictions.groupby("model"):
        y = pd.to_numeric(g["ret_fwd"], errors="coerce").astype(float)
        p = pd.to_numeric(g["pred"], errors="coerce").astype(float)
        r2_oos = 1.0 - np.sum((y - p) ** 2) / np.sum(y ** 2)

        monthly = []
        for _, gm in g.groupby("yyyymm"):
            ym = pd.to_numeric(gm["ret_fwd"], errors="coerce").astype(float)
            pm = pd.to_numeric(gm["pred"], errors="coerce").astype(float)
            sst_m = np.sum(ym ** 2)
            if sst_m > 0:
                monthly.append(1.0 - np.sum((ym - pm) ** 2) / sst_m)

        rows.append({
            "model": model_name,
            "r2_oos": r2_oos,
            "r2_oos_percent": r2_oos * 100,
            "mean_monthly_r2": np.mean(monthly),
            "mean_monthly_r2_percent": np.mean(monthly) * 100,
            "n_obs": len(g),
            "start_month": int(g["yyyymm"].min()),
            "end_month": int(g["yyyymm"].max()),
        })
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


# ------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------
def main():
    panel, x_cols = load_panel()
    end_test_year = END_TEST_YEAR if END_TEST_YEAR is not None else int(panel["year"].max())

    for test_year, train, valid, test in make_rolling_splits(
        panel, start_test_year=START_TEST_YEAR, end_test_year=end_test_year
    ):
        ckpt = CKPT_DIR / f"ml_pred_{test_year}.parquet"
        if RESUME and ckpt.exists():
            print(f"[year {test_year}] checkpoint exists -> skipping")
            continue

        print("=" * 80)
        print(f"[Training ML models for test year {test_year}]")
        print("=" * 80)
        t0 = time.time()

        # Clean the feature matrix ONCE and reuse across models.
        Xtr, ytr = prepare_block(train, x_cols)
        Xva, yva = prepare_block(valid, x_cols)
        Xte, _ = prepare_block(test, x_cols)

        print("  Fitting RF...")
        rf = fit_predict_rf(Xtr, ytr, Xva, yva, Xte, test)
        print("  Fitting NN2...")
        nn2 = fit_predict_nn(Xtr, ytr, Xva, yva, Xte, test, "NN2")
        print("  Fitting NN4...")
        nn4 = fit_predict_nn(Xtr, ytr, Xva, yva, Xte, test, "NN4")

        # Save the whole year atomically (only after all 3 models succeed).
        pd.concat([rf, nn2, nn4], ignore_index=True).to_parquet(ckpt, index=False)
        print(f"[year {test_year}] done in {time.time() - t0:.0f}s -> {ckpt.name}")

    # --- assemble all checkpointed years in range into the final outputs ---
    files = [CKPT_DIR / f"ml_pred_{y}.parquet"
             for y in range(START_TEST_YEAR, end_test_year + 1)]
    files = [f for f in files if f.exists()]
    if not files:
        print("No checkpoints found; nothing to assemble.")
        return

    predictions = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    pred_path = OUTPUT_DIR / "predictions_ml.parquet"
    predictions.to_parquet(pred_path, index=False)
    print("=" * 80)
    print(f"[Saved] ML predictions -> {pred_path}  (shape {predictions.shape})")
    print("=" * 80)

    table1 = compute_table1(predictions)
    table_path = OUTPUT_DIR / "table1_ml.csv"
    table1.to_csv(table_path, index=False)
    print("[Table 1 ML]")
    print(table1.to_string(index=False))
    print(f"[Saved] Table -> {table_path}")


if __name__ == "__main__":
    main()