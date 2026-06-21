# Empirical Asset Pricing Assignment 2

This repository contains the code for an empirical asset pricing replication based on Gu, Kelly, and Xiu (2020), *Empirical Asset Pricing via Machine Learning*. The project builds a stock-level monthly prediction panel, trains several linear and machine-learning models, and produces the required Table 1, Figure 4, and Figure 9 outputs.

## Repository

GitHub URL:

```text
https://github.com/ahram-lara-cho/Empirical-Asset-Pricing-Assignment-2
````

## Project Structure

```text
Empirical-Asset-Pricing-Assignment-2/
│
├── run_all.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── 01_build_panel.py
│   ├── 02_train_simple_models.py
│   ├── 03_train_ml_models.py
│   ├── 04_make_figure4.py
│   ├── 05_make_figure9.py
│   └── 06_make_final1_table.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── char_cols.csv
│       ├── macro_cols.csv
│       └── interaction_cols.csv
│
└── outputs/
    ├── table1_final_display_set1.csv
    ├── table1_final_display_set2.csv
    ├── table1_final_set1.csv
    ├── table1_final_set2.csv
    ├── figure_table1_oos_r2_set1.png
    ├── figure_table1_oos_r2_set2.png
    ├── figure4_importance_set1.png
    ├── figure4_importance_set2.png
    ├── figure9_cumret_set1.png
    └── figure9_cumret_set2.png
```

## Overview

The project replicates the following required empirical outputs:

1. **Table 1: Monthly out-of-sample stock-level prediction performance**

   * Models: OLS+H, OLS-3+H, PCR, ENet+H, RF, NN2, NN4
   * All sample only
   * Both table and figure are produced
   * Results are reported for:

     * Set 1: 1971–2016 sample
     * Set 2: 1971–2024 sample

2. **Figure 4: Variable importance by model**

   * Models: PCR, ENet+H, RF, NN2, NN4
   * Produced for both Set 1 and Set 2

3. **Figure 9: Cumulative return of machine-learning portfolios**

   * Models: OLS-3+H, PCR, ENet+H, RF, NN2, NN4
   * Includes SP500-Rf benchmark
   * NBER recession periods are shaded

## Data Sources

The code uses the following external data sources:

* **CRSP monthly stock returns** from WRDS
* **Open Source Asset Pricing** firm characteristics
* **Amit Goyal macro predictors** from `PredictorData.xlsx`
* **S&P 500 and risk-free rate** from WRDS for the Figure 9 benchmark

WRDS credentials are required to fully reproduce the data construction step.

This repository does **not** include raw WRDS data, processed WRDS-derived panels, or model prediction parquet files.

## Required External File

Before running the full pipeline, download Amit Goyal's monthly macro predictor file and place it at:

```text
data/raw/PredictorData.xlsx
```

The code expects this exact path.

## Environment Setup

The project was run using Python 3.11.

Create and activate the environment:

```bash
conda create -n gkx python=3.11 -y
conda activate gkx
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Full Pipeline

From the project root:

```bash
python run_all.py
```

This sequentially runs:

```text
01_build_panel.py
02_train_simple_models.py
03_train_ml_models.py
04_make_figure4.py
05_make_figure9.py
06_make_final1_table.py
```

## Running Selected Steps

To regenerate only figures and final tables after model predictions already exist:

```bash
python run_all.py --from-step 4
```

To regenerate only Table 1:

```bash
python run_all.py --from-step 6 --to-step 6
```

To run only up to model training:

```bash
python run_all.py --to-step 3
```

## Step Descriptions

### Step 01: Build Panel

```bash
python src/01_build_panel.py
```

This script:

* Downloads NYSE common stock monthly returns from CRSP
* Constructs market equity
* Constructs next-month return as the prediction target
* Downloads firm characteristics from Open Source Asset Pricing
* Loads macro predictors from `PredictorData.xlsx`
* Rank-normalizes firm characteristics cross-sectionally each month
* Creates characteristic-by-macro interaction terms
* Saves the processed panel and feature metadata

Main generated files:

```text
data/processed/panel_1971_2025.parquet
data/processed/char_cols.csv
data/processed/macro_cols.csv
data/processed/interaction_cols.csv
```

The large panel parquet file is not committed to GitHub.

### Step 02: Train Simple Models

```bash
python src/02_train_simple_models.py
```

This script trains:

* OLS+H
* OLS-3+H
* PCR
* ENet+H

Main generated files:

```text
outputs/predictions_simple.parquet
outputs/table1_simple.csv
```

Prediction parquet files are not committed to GitHub.

### Step 03: Train Machine-Learning Models

```bash
python src/03_train_ml_models.py
```

This script trains:

* RF
* NN2
* NN4

It uses rolling out-of-sample splits and saves checkpoint files by year.

Main generated files:

```text
outputs/predictions_ml.parquet
outputs/table1_ml.csv
outputs/_ckpt_ml/
```

Prediction parquet files and checkpoints are not committed to GitHub.

### Step 04: Make Figure 4

```bash
python src/04_make_figure4.py
```

This script creates variable-importance heatmaps for:

```text
PCR, ENet+H, RF, NN2, NN4
```

Main outputs:

```text
outputs/figure4_importance_set1.png
outputs/figure4_importance_set2.png
```

### Step 05: Make Figure 9

```bash
python src/05_make_figure9.py
```

This script creates cumulative long-short portfolio returns for:

```text
OLS-3+H, PCR, ENet+H, RF, NN2, NN4
```

The strategy sorts stocks each month by predicted return, goes long the top decile, short the bottom decile, and value-weights within each leg using market equity.

Main outputs:

```text
outputs/figure9_cumret_set1.png
outputs/figure9_cumret_set2.png
```

### Step 06: Make Final Table 1

```bash
python src/06_make_final1_table.py
```

This script combines predictions from Steps 02 and 03 and computes stock-level out-of-sample prediction performance.

The main metric is:

```text
R2_oos = 1 - sum((r - rhat)^2) / sum(r^2)
```

The benchmark is a zero-return prediction.

Main outputs:

```text
outputs/table1_final_display_set1.csv
outputs/table1_final_display_set2.csv
outputs/table1_final_set1.csv
outputs/table1_final_set2.csv
outputs/figure_table1_oos_r2_set1.png
outputs/figure_table1_oos_r2_set2.png
```

## Output Sets

The project reports two required sets.

### Set 1

```text
Sample: 1971–2016
Out-of-sample evaluation: 2001–2016
```

Main outputs:

```text
outputs/table1_final_display_set1.csv
outputs/figure_table1_oos_r2_set1.png
outputs/figure4_importance_set1.png
outputs/figure9_cumret_set1.png
```

### Set 2

```text
Sample: 1971–2024
Out-of-sample evaluation: 2001–2024
```

Although the assignment target is 2025, the WRDS CRSP monthly stock file available at the time of data collection returned observations only through December 2024. Since the prediction target is the next-month stock return, the final usable feature month is November 2024.

Main outputs:

```text
outputs/table1_final_display_set2.csv
outputs/figure_table1_oos_r2_set2.png
outputs/figure4_importance_set2.png
outputs/figure9_cumret_set2.png
```

## Reproducibility Notes

The full pipeline is reproducible for users with WRDS access. However, because WRDS-derived data cannot be redistributed, the repository excludes:

```text
data/processed/*.parquet
outputs/predictions*.parquet
outputs/predictions_all*.parquet
outputs/_ckpt_ml/
```

The small metadata files below are included because they only list feature names and help document the exact feature construction:

```text
data/processed/char_cols.csv
data/processed/macro_cols.csv
data/processed/interaction_cols.csv
```

The model scripts use a firm-level random subsample for computational feasibility. Firms are sampled once by `permno` using a fixed seed, so each retained firm enters with its full time series. The same subsample configuration is used across simple and machine-learning models.

## Dependencies

Main packages:

```text
numpy
pandas
scikit-learn
matplotlib
pyarrow
openpyxl
wrds
openassetpricing
```

See `requirements.txt` for the full dependency list.

## Author

AhRam Cho
BME.70034 Empirical Asset Pricing
Spring 2026

```
```
