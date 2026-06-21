"""
02_train_simple_models.py

Train simple GKX-style models on the processed panel.

Models:
    OLS+H      OLS with Huber loss (SGD)
    OLS-3+H    size/bm/momentum (+ macro + interactions), Huber (HuberRegressor)
    PCR        principal-component regression (no Huber, per GKX)
    ENet+H     elastic net with Huber loss (SGD)

To produce the two deliverable sets, run twice changing only END_TEST_YEAR:
    Set 1: START_TEST_YEAR=2001, END_TEST_YEAR=2016
    Set 2: START_TEST_YEAR=2001, END_TEST_YEAR=2025  (or None to use panel max)

FIRM_SUBSAMPLE must match the value AND seed used in 03 so the linear and ML
models share the exact same firms for the unified Table 1.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, SGDRegressor, HuberRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


warnings.filterwarnings("ignore")


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Run configuration (the only lines you change between runs) -------------
PANEL_NAME      = "panel_1971_2025.parquet"  # match whatever you saved in step 01
START_TEST_YEAR = 2001                        # 1971 + 18y train + 12y val -> first OOS 2001
END_TEST_YEAR   = None                        # None = panel's last year; set 2016 for Set 1
FIRM_SUBSAMPLE  = 0.30                         # MUST match 03 (value + seed) for a fair unified table
SUBSAMPLE_SEED  = 42                           # MUST match 03 so the SAME firms are kept
# -----------------------------------------------------------------------------

# Huber transition threshold for the SGD models, in RETURN units.
# (SGDRegressor's `epsilon` is in target units, unlike HuberRegressor's sigma
#  units.) ~0.15 keeps the bulk of monthly returns in the quadratic region and
# only caps the fat tails. Results are not very sensitive to it.
SGD_HUBER_EPS = 0.15


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

    # Bias-free firm subsample -- pick permnos once (same seed as 03) so the
    # linear and ML models run on the identical set of firms.
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
    print(f"Months: {panel['yyyymm'].min()} to {panel['yyyymm'].max()}")
    print(f"Stocks: {panel['permno'].nunique():,}")
    print(f"Features: {len(x_cols)}")
    print("=" * 80)

    return panel, char_cols, macro_cols, interaction_cols, x_cols


# ------------------------------------------------------------
# 2. Feature subsets
# ------------------------------------------------------------
def make_ols3_features(char_cols, macro_cols, interaction_cols):
    """GKX's OLS-3 uses size, book-to-market, momentum (+ macro + interactions)."""
    base3 = ["size_crsp", "bm", "mom12m"]
    base3 = [c for c in base3 if c in char_cols]

    inter3 = []
    for c in base3:
        for m in macro_cols:
            name = f"{c}__x__{m}"
            if name in interaction_cols:
                inter3.append(name)

    ols3_cols = base3 + macro_cols + inter3

    print("[OLS-3+H features]")
    print(f"Base characteristics: {base3}")   # <- confirm all THREE survive
    print(f"Macro variables: {len(macro_cols)}")
    print(f"Interactions: {len(inter3)}")
    print(f"Total OLS-3+H features: {len(ols3_cols)}")

    return ols3_cols


# ------------------------------------------------------------
# 3. Rolling splits
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
# 4. Utility: prepare arrays (clean once, reuse)
# ------------------------------------------------------------
def prepare_block(df, x_cols):
    """Clean a feature block to a float32 array and pull the float32 target."""
    X = (df[x_cols]
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0)
         .astype(np.float32)
         .values)
    y = pd.to_numeric(df["ret_fwd"], errors="coerce").astype(np.float32).values
    return X, y


def make_prediction_frame(test, pred, model_name):
    out = test[["permno", "yyyymm", "year", "ret_fwd", "ret", "me"]].copy()
    out["pred"] = pred
    out["model"] = model_name
    return out


# ------------------------------------------------------------
# 5. Models
# ------------------------------------------------------------
def fit_predict_ols_huber(Xtr, ytr, Xte, test_df, model_name="OLS+H"):
    """OLS with Huber loss (the '+H'), via SGD. Tiny L2 only for stability."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("huber", SGDRegressor(
            loss="huber", epsilon=SGD_HUBER_EPS,
            penalty="l2", alpha=1e-6,
            learning_rate="adaptive", eta0=1e-3,
            max_iter=2000, tol=1e-4,
            early_stopping=True, n_iter_no_change=5, validation_fraction=0.1,
            random_state=42,
        )),
    ])
    model.fit(Xtr, ytr)
    return make_prediction_frame(test_df, model.predict(Xte), model_name)


def fit_predict_ols3_huber(Xtr, ytr, Xte, test_df, model_name="OLS-3+H"):
    """OLS-3 with Huber loss. Small feature set -> HuberRegressor (sigma-units eps)."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("huber", HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500)),
    ])
    model.fit(Xtr, ytr)
    return make_prediction_frame(test_df, model.predict(Xte), model_name)


def fit_predict_pcr(Xtr, ytr, Xva, yva, Xte, test_df,
                    candidate_k=(5, 10, 20, 30, 40)):
    """PCR (no Huber, per GKX). PCA fit once at max k; ks slice nested components."""
    max_k = min(max(candidate_k), Xtr.shape[1])

    scaler = StandardScaler().fit(Xtr)
    Ztr, Zva, Zte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    pca = PCA(n_components=max_k, svd_solver="randomized", random_state=42).fit(Ztr)
    Ptr, Pva, Pte = pca.transform(Ztr), pca.transform(Zva), pca.transform(Zte)

    best_mse, best_k, best_ridge = np.inf, None, None
    for k in candidate_k:
        if k > max_k:
            continue
        ridge = Ridge(alpha=1e-6).fit(Ptr[:, :k], ytr)
        mse = mean_squared_error(yva, ridge.predict(Pva[:, :k]))
        if mse < best_mse:
            best_mse, best_k, best_ridge = mse, k, ridge

    pred = best_ridge.predict(Pte[:, :best_k])
    print(f"    [PCR] best n_components={best_k}, valid MSE={best_mse:.6g}")
    return make_prediction_frame(test_df, pred, "PCR")


def fit_predict_enet_huber(Xtr, ytr, Xva, yva, Xte, test_df,
                           alphas=(1e-5, 1e-4, 1e-3, 1e-2), l1_ratio=0.5):
    """Elastic net + Huber loss (the '+H'), via SGD. Tune alpha at rho=0.5."""
    best_mse, best_alpha, best_model = np.inf, None, None
    for alpha in alphas:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", SGDRegressor(
                loss="huber", epsilon=SGD_HUBER_EPS,
                penalty="elasticnet", alpha=alpha, l1_ratio=l1_ratio,
                learning_rate="adaptive", eta0=1e-3,
                max_iter=2000, tol=1e-4,
                early_stopping=True, n_iter_no_change=5, validation_fraction=0.1,
                random_state=42,
            )),
        ])
        model.fit(Xtr, ytr)
        mse = mean_squared_error(yva, model.predict(Xva))
        if mse < best_mse:
            best_mse, best_alpha, best_model = mse, alpha, model

    pred = best_model.predict(Xte)
    print(f"    [ENet+H] best alpha={best_alpha}, l1_ratio={l1_ratio}, "
          f"valid MSE={best_mse:.6g}")
    return make_prediction_frame(test_df, pred, "ENet+H")


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
    panel, char_cols, macro_cols, interaction_cols, x_cols = load_panel()

    ols3_cols = make_ols3_features(char_cols, macro_cols, interaction_cols)

    end_test_year = END_TEST_YEAR if END_TEST_YEAR is not None else int(panel["year"].max())

    all_preds = []
    for test_year, train, valid, test in make_rolling_splits(
        panel, start_test_year=START_TEST_YEAR, end_test_year=end_test_year
    ):
        print("=" * 80)
        print(f"[Training models for test year {test_year}]")
        print("=" * 80)

        # Clean the full feature matrix ONCE and reuse across models.
        Xtr, ytr = prepare_block(train, x_cols)
        Xva, yva = prepare_block(valid, x_cols)
        Xte, _ = prepare_block(test, x_cols)

        # OLS-3 uses its own (small) feature subset.
        Xtr3, ytr3 = prepare_block(train, ols3_cols)
        Xte3, _ = prepare_block(test, ols3_cols)

        print("  Fitting OLS+H...")
        all_preds.append(fit_predict_ols_huber(Xtr, ytr, Xte, test, "OLS+H"))

        print("  Fitting OLS-3+H...")
        all_preds.append(fit_predict_ols3_huber(Xtr3, ytr3, Xte3, test, "OLS-3+H"))

        print("  Fitting PCR...")
        all_preds.append(fit_predict_pcr(Xtr, ytr, Xva, yva, Xte, test))

        print("  Fitting ENet+H...")
        all_preds.append(fit_predict_enet_huber(Xtr, ytr, Xva, yva, Xte, test))

    predictions = pd.concat(all_preds, ignore_index=True)

    pred_path = OUTPUT_DIR / "predictions_simple.parquet"
    predictions.to_parquet(pred_path, index=False)

    print("=" * 80)
    print(f"[Saved] Predictions saved to: {pred_path}")
    print(f"Predictions shape: {predictions.shape}")
    print("=" * 80)

    table1 = compute_table1(predictions)
    table_path = OUTPUT_DIR / "table1_simple.csv"
    table1.to_csv(table_path, index=False)

    print("[Table 1 simple]")
    print(table1)
    print(f"[Saved] Table saved to: {table_path}")


if __name__ == "__main__":
    main()