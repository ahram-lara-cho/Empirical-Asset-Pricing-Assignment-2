"""
04_make_figure4.py

GKX (2020) Figure 4 -- "Variable importance by model."

Importance method (GKX): for each characteristic, set its column (and all of
its characteristic x macro interaction columns) to zero -- which, because the
characteristics are rank-normalized to [-1, 1] with 0 = cross-sectional median,
is exactly GKX's "set to the median" device -- re-predict, and record the drop
in out-of-sample R^2. Negative drops are clipped to zero; importances are then
normalized to shares (sum to 1) within each model so columns are comparable.

  >> SCOPE NOTE (acknowledge this in the results PDF) <<
  To avoid re-running the full rolling-window training (~4h), importance here is
  computed from ONE model per algorithm per set: each model is fit once on the
  full estimation window (1971-E), and zero-out importance is measured IN-SAMPLE
  over that same multi-decade window (on a capped random subsample for speed),
  rather than averaged over every rolling refit as in GKX. In-sample model-
  reliance is a standard importance notion -- it is what tree feature_importances_
  report -- and measuring it over decades (not a short out-of-sample slice) keeps
  the base R^2 stable and stops importance from being hijacked by corporate
  events that happen to cluster in any one short period. Magnitudes are therefore
  approximate, but the variable *ranking* -- what Figure 4 communicates -- is
  stable. NN importance uses a single net (set NN_IMP_ENSEMBLE>1 to stabilize).

Models required by the assignment for Figure 4:
    PCR, ENet+H, RF, NN2, NN4

Two sets (sample period end year E):
    Set 1: E = 2016 -> outputs/figure4_importance_set1.png  (+ .csv)
    Set 2: E = 2024 -> outputs/figure4_importance_set2.png  (+ .csv)

Inputs (first existing path wins):
    panel:     data/panel.parquet | data/processed/panel_1971_2025.parquet | data/panel_1971_2025.parquet
    col lists: char_cols.csv / macro_cols.csv / interaction_cols.csv next to the
               panel or under data/processed/. If absent, the feature groups are
               reconstructed from the column names ("__x__" = interaction,
               "_macro" suffix = macro, else characteristic).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# ------------------------------------------------------------
# Paths and configuration
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_CANDIDATES = [
    DATA_DIR / "panel.parquet",
    PROCESSED_DIR / "panel_1971_2025.parquet",
    DATA_DIR / "panel_1971_2025.parquet",
]

MODELS = ["PCR", "ENet+H", "RF", "NN2", "NN4"]   # Figure-4 model set
SETS = {"set1": 2016, "set2": 2024}              # sample-period end year E

# Keep consistent with steps 02/03 so importance reflects the same universe.
FIRM_SUBSAMPLE = 0.30
SUBSAMPLE_SEED = 42

# Single-fit hyperparameters (fixed; see scope note). Chosen to match the
# middle of the grids tuned in 02/03.
PCR_K = 30
ENET_ALPHA = 1e-3
ENET_L1 = 0.5
SGD_HUBER_EPS = 0.15
RF_MAX_DEPTH = 6
RF_MIN_LEAF = 100
NN_ALPHA = 1e-3
NN_IMP_ENSEMBLE = 1          # >1 averages nets to stabilize NN importance
N_JOBS = -1

EVAL_MAX_ROWS = 100_000      # rows used to MEASURE importance, sampled across the whole window
TRAIN_MAX_ROWS = None        # optional cap on training rows for the single fits
TOP_N = 20                   # characteristics shown in the heatmap
RANDOM_STATE = 42


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def read_col_list(path):
    return pd.read_csv(path, header=None).iloc[:, 0].tolist()


def get_feature_groups(panel, panel_dir):
    """Return (char_cols, macro_cols, interaction_cols), from csvs if present
    else reconstructed from column names."""
    csv_dirs = [panel_dir, PROCESSED_DIR, DATA_DIR]
    found = {}
    for name in ["char_cols", "macro_cols", "interaction_cols"]:
        hit = first_existing([d / f"{name}.csv" for d in csv_dirs])
        if hit is not None:
            found[name] = read_col_list(hit)
    if len(found) == 3:
        print(f"[cols] loaded char/macro/interaction lists from csv")
        return found["char_cols"], found["macro_cols"], found["interaction_cols"]

    # Reconstruct from column names.
    print("[cols] col-list csvs not all found -> reconstructing from column names")
    non_feature = {"permno", "yyyymm", "year", "ret", "ret_fwd", "me", "yyyymm_next"}
    feats = [c for c in panel.columns if c not in non_feature]
    interaction_cols = [c for c in feats if "__x__" in c]
    macro_cols = [c for c in feats if c.endswith("_macro") and "__x__" not in c]
    char_cols = [c for c in feats if "__x__" not in c and not c.endswith("_macro")]
    return char_cols, macro_cols, interaction_cols


def prepare_block(df, x_cols):
    """Clean feature block -> float32 array; float32 target."""
    X = (df[x_cols]
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0)
         .astype(np.float32)
         .values)
    y = pd.to_numeric(df["ret_fwd"], errors="coerce").astype(np.float32).values
    return X, y


# ------------------------------------------------------------
# Model fitting (one fit each); every fitter returns predict(X_original_space)
# ------------------------------------------------------------
def fit_pcr(Xtr, ytr):
    scaler = StandardScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    k = min(PCR_K, Ztr.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=RANDOM_STATE).fit(Ztr)
    ridge = Ridge(alpha=1e-6).fit(pca.transform(Ztr), ytr)
    return lambda X: ridge.predict(pca.transform(scaler.transform(X)))


def fit_enet(Xtr, ytr):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", SGDRegressor(
            loss="huber", epsilon=SGD_HUBER_EPS,
            penalty="elasticnet", alpha=ENET_ALPHA, l1_ratio=ENET_L1,
            learning_rate="adaptive", eta0=1e-3,
            max_iter=2000, tol=1e-4,
            early_stopping=True, n_iter_no_change=5, validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )),
    ]).fit(Xtr, ytr)
    return pipe.predict


def fit_rf(Xtr, ytr):
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=RF_MAX_DEPTH, min_samples_leaf=RF_MIN_LEAF,
        max_features="sqrt", n_jobs=N_JOBS, random_state=RANDOM_STATE,
    ).fit(Xtr, ytr)
    return rf.predict


def _make_mlp(hidden, seed):
    return MLPRegressor(
        hidden_layer_sizes=hidden, activation="relu", solver="adam",
        alpha=NN_ALPHA, learning_rate_init=1e-3, batch_size=2048,
        max_iter=100, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=10, random_state=seed,
    )


def fit_nn(Xtr, ytr, hidden):
    scaler = StandardScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    nets = [_make_mlp(hidden, seed=s).fit(Ztr, ytr) for s in range(NN_IMP_ENSEMBLE)]

    def predict(X):
        Z = scaler.transform(X)
        return np.mean([net.predict(Z) for net in nets], axis=0)

    return predict


def fit_model(name, Xtr, ytr):
    if name == "PCR":
        return fit_pcr(Xtr, ytr)
    if name == "ENet+H":
        return fit_enet(Xtr, ytr)
    if name == "RF":
        return fit_rf(Xtr, ytr)
    if name == "NN2":
        return fit_nn(Xtr, ytr, (32, 16))
    if name == "NN4":
        return fit_nn(Xtr, ytr, (32, 16, 8, 4))
    raise ValueError(name)


# ------------------------------------------------------------
# Zero-out importance
# ------------------------------------------------------------
def compute_importance(predict_fn, Xev, yev, char_cols, col_idx, inter_by_char):
    """Drop in R^2 when each characteristic (and its interactions) is zeroed.
    Mutates Xev in place then restores, to avoid copying the whole matrix."""
    Xev = np.array(Xev, copy=True)                 # one writable copy (not per-char)
    sst = float(np.sum(yev.astype(np.float64) ** 2))
    base_pred = predict_fn(Xev)
    base_r2 = 1.0 - np.sum((yev - base_pred) ** 2) / sst

    imps = {}
    for c in char_cols:
        cols = [col_idx[c]] + [col_idx[ic] for ic in inter_by_char.get(c, [])]
        saved = Xev[:, cols].copy()
        Xev[:, cols] = 0.0
        pred = predict_fn(Xev)
        Xev[:, cols] = saved                       # restore
        r2 = 1.0 - np.sum((yev - pred) ** 2) / sst
        imps[c] = max(base_r2 - r2, 0.0)           # GKX clip negatives to 0
    return imps, base_r2


# ------------------------------------------------------------
# One set -> importance matrix + heatmap
# ------------------------------------------------------------
def run_set(panel, set_name, end_year, char_cols, macro_cols, interaction_cols, x_cols):
    sample = panel[panel["year"] <= end_year]
    print(f"[{set_name}] estimation sample 1971-{end_year}: {len(sample):,} rows")
    if len(sample) == 0:
        print(f"[{set_name}] empty sample -> skipping")
        return

    Xtr, ytr = prepare_block(sample, x_cols)
    if TRAIN_MAX_ROWS and len(Xtr) > TRAIN_MAX_ROWS:
        idx = np.random.default_rng(0).choice(len(Xtr), TRAIN_MAX_ROWS, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]

    # Importance is measured IN-SAMPLE over the full window (many years -> stable
    # base R^2, no single-period event domination), on a capped random subsample.
    if len(Xtr) > EVAL_MAX_ROWS:
        eidx = np.random.default_rng(1).choice(len(Xtr), EVAL_MAX_ROWS, replace=False)
        Xev, yev = Xtr[eidx], ytr[eidx]
    else:
        Xev, yev = Xtr, ytr
    print(f"[{set_name}] fitting on {len(Xtr):,} rows, importance on {len(Xev):,} rows")

    col_idx = {c: i for i, c in enumerate(x_cols)}
    inter_by_char = {c: [ic for ic in interaction_cols if ic.startswith(f"{c}__x__")]
                     for c in char_cols}

    imp_matrix = pd.DataFrame(index=char_cols, columns=MODELS, dtype=float)
    for name in MODELS:
        print(f"  [{set_name}] fitting {name} ...")
        predict_fn = fit_model(name, Xtr, ytr)
        imps, base_r2 = compute_importance(predict_fn, Xev, yev,
                                           char_cols, col_idx, inter_by_char)
        s = pd.Series(imps)
        total = s.sum()
        imp_matrix[name] = (s / total) if total > 0 else 0.0   # normalize to shares
        print(f"  [{set_name}] {name}: base R^2={base_r2:+.4f}, "
              f"top char='{s.idxmax()}'")

    imp_matrix["__avg__"] = imp_matrix[MODELS].mean(axis=1)
    imp_matrix = imp_matrix.sort_values("__avg__", ascending=False)

    csv_out = OUTPUT_DIR / f"figure4_importance_{set_name}.csv"
    imp_matrix.drop(columns="__avg__").to_csv(csv_out)
    print(f"[{set_name}] saved importance matrix -> {csv_out.name}")

    _plot_heatmap(imp_matrix[MODELS].head(TOP_N), set_name, end_year)


def _plot_heatmap(mat, set_name, end_year):
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(mat) + 1.6))
    data = mat.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="Blues",
                   vmin=0.0, vmax=np.nanmax(data) if np.nanmax(data) > 0 else 1.0)

    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, fontsize=10)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_title(f"Variable importance by model (top {len(mat)}) -- sample 1971-{end_year}",
                 fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("importance share (column-normalized)", fontsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / f"figure4_importance_{set_name}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[{set_name}] saved figure -> {out}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    panel_path = first_existing(PANEL_CANDIDATES)
    if panel_path is None:
        raise FileNotFoundError(f"No panel found among: {[str(p) for p in PANEL_CANDIDATES]}")
    print(f"[load] panel: {panel_path}")
    panel = pd.read_parquet(panel_path)

    if FIRM_SUBSAMPLE is not None:
        permnos = panel["permno"].unique()
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        keep = rng.choice(permnos, size=int(len(permnos) * FIRM_SUBSAMPLE), replace=False)
        panel = panel[panel["permno"].isin(keep)]
        print(f"[subsample] kept {len(keep):,}/{len(permnos):,} firms ({FIRM_SUBSAMPLE:.0%})")

    char_cols, macro_cols, interaction_cols = get_feature_groups(panel, panel_path.parent)
    x_cols = char_cols + macro_cols + interaction_cols
    print(f"[cols] characteristics={len(char_cols)}, macro={len(macro_cols)}, "
          f"interactions={len(interaction_cols)}, total features={len(x_cols)}")

    for set_name, end_year in SETS.items():
        print("=" * 70)
        run_set(panel, set_name, end_year, char_cols, macro_cols, interaction_cols, x_cols)
    print("=" * 70)
    print("[done]")


if __name__ == "__main__":
    main()