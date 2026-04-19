# DMT Assignment 1

This repository contains the code for the Data Mining Techniques Assignment 1
(advanced track, smartphone mood dataset). It implements the full pipeline across
Tasks 1A, 1B, 1C, 2A, 4, and 5, organised as a set of self-contained entry scripts
that share a common library under `src/dmt_mood_pipeline/`.

---

## 1. Repository layout

```
.
├── dataset_mood_smartphone.csv       # Raw input data (do not modify)
├── task_1a.py                        # Task entry scripts (thin wrappers)
├── task_1b.py                        # forward-fill imputation (default)
├── task_1b_ewma.py                   # EWMA imputation
├── task_1c.py                        # feature engineering on forward-fill data
├── task_1c_ewma.py                   # feature engineering on EWMA data
├── task_2a_rf.py / task_2a_rf_ewma.py        # Random Forest classifier
├── task_2a_lstm.py / task_2a_lstm_ewma.py    # LSTM classifier
├── task_2a_gru.py / task_2a_gru_ewma.py      # GRU classifier
├── task_4_rf.py / task_4_rf_ewma.py          # Random Forest regressor
├── task_4_lstm.py / task_4_lstm_ewma.py      # LSTM regressor
├── task_4_gru.py / task_4_gru_ewma.py        # GRU regressor
├── _task_wrapper_bootstrap.py        # Internal dispatcher, do not run directly
├── src/dmt_mood_pipeline/            # Core library
│   ├── data.py                       # Daily aggregation, cleaning, imputation
│   ├── eda.py                        # EDA tables and plots (Task 1A)
│   ├── features.py                   # Window dataset construction (Task 1C)
│   ├── models.py                     # RF, GRU/LSTM, persistence baseline
│   ├── pipeline.py                   # Training, CV, hyperparameter search
│   ├── evaluation.py                 # MAE, MSE, RMSE, F1, confusion matrix
│   ├── splits.py                     # Patient-wise temporal train/val/test split
│   ├── tasks.py                      # Per-task runner functions
│   ├── task_entry.py                 # Argument parsing, config construction
│   ├── config.py                     # Hyperparameter grids, window size, etc.
│   └── utils.py                      # Helpers (gap flags, quantiles, slopes)
└── outputs/                          # Auto-generated; see Section 4
```

The 17 root-level `task_*.py` scripts are one-line wrappers that all delegate
to `src/dmt_mood_pipeline/task_entry.py`. You never need to call the internal
modules directly.

---

## 2. Installation

### Requirements

- Python 3.10 or newer
- Packages: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `torch`

### Setup

```bash
# (Optional but recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib scikit-learn torch
```

If `pip install` fails with an `externally-managed-environment` error (common
on recent Linux/macOS Python installs), either use the virtual-environment
approach above, or append `--break-system-packages` to the command.

For a lighter CPU-only PyTorch install (LSTM/GRU scripts still work on CPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. How to run

### 3.1 Minimal run (recommended)

Based on the Task 1B backtest (EWMA consistently outperformed forward-fill for
all three self-report signals), the downstream pipeline should use EWMA
imputation. Running the following seven scripts in order covers every
experimental result needed for the report:

```bash
python task_1a.py                  # EDA tables + plots
python task_1b_ewma.py             # Cleaning + imputation comparison
python task_1c_ewma.py             # Feature engineering for RF
python task_2a_rf_ewma.py          # Task 2A classification (RF)
python task_2a_lstm_ewma.py        # Task 2A classification (LSTM)
python task_4_rf_ewma.py           # Task 4 regression (RF)
python task_4_lstm_ewma.py         # Task 4 regression (LSTM)
```

Each script must be invoked from the repository root (the directory that
contains `dataset_mood_smartphone.csv`). Scripts are independent: each one
reads the raw CSV from scratch and reproduces upstream steps internally, so
you can run them in any order.

### 3.2 Alternative: GRU instead of LSTM

`task_2a_gru_ewma.py` and `task_4_gru_ewma.py` use the same pipeline as the
LSTM scripts but swap the recurrent cell type. Useful if you want to compare
GRU vs LSTM; not required by the assignment.

### 3.3 Forward-fill variants

The scripts without the `_ewma` suffix (e.g. `task_1c.py`, `task_2a_rf.py`) run
the same pipeline on forward-fill-imputed data. Given the Task 1B result you
will typically not need them, unless you want to perform a sensitivity
analysis showing how the imputation choice propagates to downstream models.

### 3.4 Expected runtime (single CPU)

| Script                    | Runtime        |
|---------------------------|----------------|
| `task_1a.py`              | ~10 seconds    |
| `task_1b_ewma.py`         | ~15 seconds    |
| `task_1c_ewma.py`         | ~20 seconds    |
| `task_2a_rf_ewma.py`      | 1–3 minutes    |
| `task_4_rf_ewma.py`       | 1–3 minutes    |
| `task_2a_lstm_ewma.py`    | 5–15 minutes   |
| `task_4_lstm_ewma.py`     | 5–15 minutes   |

LSTM/GRU runs are substantially faster on a GPU (CUDA) or on Apple Silicon
(MPS); the `GRUWrapper` in `models.py` auto-detects the available device.

---

## 4. Outputs

All artefacts are written under `outputs/<task_name>/`. The directory is
created automatically.

### 4.1 Task 1A — `outputs/task_1a/`

```
task_1a_tables/
├── table_dataset_overview.csv        # Per-variable count, min, max, mean, std, median, user count
└── table_user_summary.csv            # Per-patient date range, mood record counts, mood statistics

task_1a_plots/
├── plot1_mood_distribution.png       # Overall mood histogram + per-user boxplot
├── plot2_mood_over_time.png          # Daily mood trajectories for top 9 users
├── plot3_key_variable_distributions.png   # mood, arousal, valence, activity, screen
├── plot4_variable_coverage_heatmap.png    # % of days with data per user × variable
├── plot5_correlation_matrix.png           # Daily correlation of core variables
├── plot6_mood_temporal_patterns.png       # Mood by day-of-week and hour
├── plot7_records_per_variable.png         # Record counts per variable (log scale)
└── plot8_screen_activity_patterns.png     # Screen time trend and activity distribution
```

These are ready-to-include report assets.

### 4.2 Task 1B — `outputs/task_1b_ewma/`

```
daily_patient_table.csv       # Daily aggregated table (one row per patient-day),
                              #   with EWMA-imputed self-report columns (*_clean)
                              #   and 0/1 imputation flags (*_imputed)
imputation_backtest.json      # Side-by-side MAE/RMSE of forward_fill vs EWMA
                              #   on held-out mood / arousal / valence values
run_summary.json              # Config, data-quality audit, backtest results
```

Note: `imputation_backtest.json` is identical regardless of which imputation
method is selected — the backtest always compares both methods. Only the
`*_clean` columns inside `daily_patient_table.csv` differ between
`task_1b.py` and `task_1b_ewma.py`.

### 4.3 Task 1C — `outputs/task_1c_ewma/`

```
daily_patient_table.csv           # Same format as Task 1B output
tabular_window_dataset.csv        # Task 1C deliverable: one row per 7-day window
                                  #   Columns: sample_id, patient_id, anchor_date,
                                  #   target_date, baseline_mood_prediction,
                                  #   target_next_day_mood_mean, plus ~80 engineered
                                  #   features (3d/7d means, slopes, std, app
                                  #   category shares, coarse groupings, missing ratios)
sequence_window_metadata.csv      # Sample-level metadata for the sequence dataset
sequence_window_dataset.npz       # Shape (N, 7, F) float array for RNN input,
                                  #   accompanied by feature_names
run_summary.json                  # Config + data-quality audit
```

`tabular_window_dataset.csv` is what the Random Forest consumes.
`sequence_window_dataset.npz` is what LSTM / GRU consume. Both correspond to
the same set of 7-day windows (matched by `sample_id`).

### 4.4 Task 2A — `outputs/task_2a_rf_ewma/` and `outputs/task_2a_lstm_ewma/`

```
run_summary.json              # Full summary: config, classification thresholds,
                              #   best hyperparameters, train/val/test sizes,
                              #   macro-F1, balanced accuracy, confusion matrices
experiment_results.json       # Per-model results list (persistence baseline + classifier)
plots/
├── random_forest_classification_confusion_matrix.png       # (RF script)
├── lstm_classification_confusion_matrix.png                # (LSTM script)
└── persistence_baseline_classification_confusion_matrix.png
```

The mood target is bucketed into three classes (low / mid / high) using the
training-set tertiles; the thresholds are reported in `run_summary.json`
under `classification_thresholds`.

### 4.5 Task 4 — `outputs/task_4_rf_ewma/` and `outputs/task_4_lstm_ewma/`

```
run_summary.json              # Config, best hyperparameters, split sizes,
                              #   MAE, MSE, RMSE, residual summary
experiment_results.json       # Per-model results list
plots/
├── random_forest_regression_residuals.png       # (RF script)
├── lstm_regression_residuals.png                # (LSTM script)
└── persistence_baseline_regression_residuals.png
```

Task 4 scripts report MAE, MSE and RMSE simultaneously, so the output files
directly supply the evidence needed for Task 5B (impact of evaluation
metrics) without additional runs.

---

## 5. Minimal reproduction recipe for the report

```bash
# One-time setup
pip install pandas numpy matplotlib scikit-learn torch

# Full pipeline (run from repository root)
python task_1a.py
python task_1b_ewma.py
python task_1c_ewma.py
python task_2a_rf_ewma.py
python task_2a_lstm_ewma.py
python task_4_rf_ewma.py
python task_4_lstm_ewma.py
```

After these seven commands complete, every table, figure, and numerical
result referenced in the report is available under `outputs/`.
