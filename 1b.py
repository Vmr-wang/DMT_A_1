"""
=============================================================================
Data Mining Techniques - Assignment 1
Task 1B: Data Cleaning
  - 1B.1: Remove extreme and incorrect values
  - 1B.2: Impute missing values in the raw long-format time series
          (two methods compared, one selected)

Output: cleaned long-format dataset (same structure as input)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})
sns.set_style("whitegrid")

OUTPUT_DIR = 'task_1b_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 0. LOAD RAW DATA
# =============================================================================
print("=" * 70)
print("TASK 1B: DATA CLEANING")
print("=" * 70)

df = pd.read_csv('dataset_mood_smartphone.csv')
df = df.drop(columns=['Unnamed: 0'])
df['time'] = pd.to_datetime(df['time'])

variables = df['variable'].unique()
print(f"Loaded raw data: {len(df)} rows, {df['id'].nunique()} users, {len(variables)} variables")

# Check existing missing values in raw data
print("\nExisting NaN values in the raw data:")
for var in variables:
    n_na = df.loc[df['variable'] == var, 'value'].isna().sum()
    if n_na > 0:
        n_total = (df['variable'] == var).sum()
        print(f"  {var}: {n_na}/{n_total} ({n_na/n_total*100:.1f}%)")


# =============================================================================
# 1B.1: REMOVE EXTREME AND INCORRECT VALUES
# =============================================================================
print("\n" + "-" * 70)
print("1B.1: Removing Extreme and Incorrect Values")
print("-" * 70)

df_clean = df.copy()

# --- Step 1: Remove duplicates ---
before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['id', 'time', 'variable'], keep='first')
n_dupes = before - len(df_clean)
print(f"\n[Step 1] Removed {n_dupes} duplicate rows")

# --- Step 2: Remove values outside valid range ---
print("\n[Step 2] Removing out-of-range values:")
range_rules = {
    'mood': (1, 10),
    'circumplex.arousal': (-2, 2),
    'circumplex.valence': (-2, 2),
    'activity': (0, 1),
    'call': (1, 1),
    'sms': (1, 1),
}

removed_range = 0
for var, (vmin, vmax) in range_rules.items():
    mask = (df_clean['variable'] == var) & (
        (df_clean['value'] < vmin) | (df_clean['value'] > vmax)
    )
    n_removed = mask.sum()
    if n_removed > 0:
        print(f"  {var}: removed {n_removed} values outside [{vmin}, {vmax}]")
        removed_range += n_removed
    df_clean = df_clean[~mask]
print(f"  Total removed by range rules: {removed_range}")

# --- Step 3: Remove negative values for duration variables ---
print("\n[Step 3] Removing negative duration values:")
duration_vars = [v for v in variables if v.startswith('appCat.') or v == 'screen']
removed_neg = 0
for var in duration_vars:
    mask = (df_clean['variable'] == var) & (df_clean['value'] < 0)
    n_removed = mask.sum()
    if n_removed > 0:
        print(f"  {var}: removed {n_removed} negative values")
        removed_neg += n_removed
    df_clean = df_clean[~mask]
print(f"  Total removed: {removed_neg}")

# --- Step 4: Remove IQR outliers for continuous variables ---
print("\n[Step 4] Removing IQR outliers (3x IQR):")
iqr_vars = ['screen'] + [v for v in variables if v.startswith('appCat.')]
removed_iqr = 0
for var in iqr_vars:
    sub = df_clean[df_clean['variable'] == var]['value']
    Q1 = sub.quantile(0.25)
    Q3 = sub.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3.0 * IQR
    upper = Q3 + 3.0 * IQR
    mask = (df_clean['variable'] == var) & (
        (df_clean['value'] < lower) | (df_clean['value'] > upper)
    )
    n_removed = mask.sum()
    if n_removed > 0:
        print(f"  {var}: removed {n_removed} outliers (bounds: [{max(0,lower):.2f}, {upper:.2f}])")
        removed_iqr += n_removed
    df_clean = df_clean[~mask]
print(f"  Total removed by IQR: {removed_iqr}")

# --- Summary ---
total_removed = len(df) - len(df_clean)
print(f"\n{'='*50}")
print(f"  Rows before cleaning:  {len(df)}")
print(f"  Rows after cleaning:   {len(df_clean)}")
print(f"  Total removed:         {total_removed} ({total_removed/len(df)*100:.2f}%)")
print(f"    - Duplicates:        {n_dupes}")
print(f"    - Out-of-range:      {removed_range}")
print(f"    - Negative durations:{removed_neg}")
print(f"    - IQR outliers:      {removed_iqr}")
print(f"{'='*50}")

# ---- PLOT: Before vs After cleaning for screen ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
screen_before = df[df['variable'] == 'screen']['value']
screen_after = df_clean[df_clean['variable'] == 'screen']['value']

axes[0].hist(screen_before, bins=100, edgecolor='black', alpha=0.7, color='salmon')
axes[0].set_title(f'Screen Duration - Before Cleaning (n={len(screen_before)})')
axes[0].set_xlabel('Duration (s)')
axes[0].set_ylabel('Frequency')
axes[0].set_xlim(0, screen_before.quantile(0.99) * 1.5)

axes[1].hist(screen_after, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
axes[1].set_title(f'Screen Duration - After Cleaning (n={len(screen_after)})')
axes[1].set_xlabel('Duration (s)')
axes[1].set_ylabel('Frequency')
axes[1].set_xlim(0, screen_after.quantile(0.99) * 1.5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot_screen_before_after.png')
plt.close()
print("\nSaved: plot_screen_before_after.png")


# =============================================================================
# 1B.2: MISSING VALUE IMPUTATION (on long-format data)
# =============================================================================
print("\n" + "-" * 70)
print("1B.2: Missing Value Imputation")
print("-" * 70)

# The raw long-format data has NaN values in:
#   - circumplex.arousal: 46 NaN
#   - circumplex.valence: 156 NaN
# These are individual measurement points where the user submitted
# a mood report but left arousal/valence blank.
#
# We compare two time-series-appropriate imputation methods:
#   Method 1: Forward Fill (per user, per variable)
#   Method 2: Linear Interpolation (per user, per variable)

impute_vars = ['circumplex.arousal', 'circumplex.valence']
print(f"\nVariables with missing values to impute: {impute_vars}")

print("\nBefore imputation:")
for var in impute_vars:
    n_na = df_clean.loc[df_clean['variable'] == var, 'value'].isna().sum()
    n_total = (df_clean['variable'] == var).sum()
    print(f"  {var}: {n_na}/{n_total} missing ({n_na/n_total*100:.2f}%)")


# --- METHOD 1: Forward Fill + Backward Fill (per user, per variable) ---
print("\n--- Method 1: Forward Fill + Backward Fill ---")
df_ffill = df_clean.copy()
for var in impute_vars:
    mask = df_ffill['variable'] == var
    df_ffill.loc[mask, 'value'] = (
        df_ffill.loc[mask]
        .groupby('id')['value']
        .transform(lambda x: x.ffill().bfill())
    )
print("After imputation:")
for var in impute_vars:
    n_na = df_ffill.loc[df_ffill['variable'] == var, 'value'].isna().sum()
    print(f"  {var}: {n_na} missing remaining")


# --- METHOD 2: Linear Interpolation (per user, per variable) ---
print("\n--- Method 2: Linear Interpolation ---")
df_interp = df_clean.copy()
for var in impute_vars:
    mask = df_interp['variable'] == var
    var_data = df_interp.loc[mask].sort_values(['id', 'time'])
    var_data['value'] = (
        var_data
        .groupby('id')['value']
        .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    )
    df_interp.loc[var_data.index, 'value'] = var_data['value']
print("After imputation:")
for var in impute_vars:
    n_na = df_interp.loc[df_interp['variable'] == var, 'value'].isna().sum()
    print(f"  {var}: {n_na} missing remaining")


# =============================================================================
# COMPARE THE TWO METHODS
# =============================================================================
print("\n--- Quantitative Comparison ---")
print("Randomly masking known values, then comparing imputation accuracy...")

np.random.seed(42)
results = {}
for var in impute_vars:
    var_mask = (df_clean['variable'] == var) & df_clean['value'].notna()
    known = df_clean.loc[var_mask].copy()
    n_test = min(100, len(known) // 5)
    test_idx = np.random.choice(known.index, size=n_test, replace=False)

    # Mask them
    df_test = df_clean.copy()
    true_vals = df_test.loc[test_idx, 'value'].values
    df_test.loc[test_idx, 'value'] = np.nan

    # Method 1
    df_test_ff = df_test.copy()
    m = df_test_ff['variable'] == var
    df_test_ff.loc[m, 'value'] = (
        df_test_ff.loc[m]
        .groupby('id')['value']
        .transform(lambda x: x.ffill().bfill())
    )
    pred_ff = df_test_ff.loc[test_idx, 'value'].values

    # Method 2
    df_test_li = df_test.copy()
    m = df_test_li['variable'] == var
    var_data = df_test_li.loc[m].sort_values(['id', 'time'])
    var_data['value'] = (
        var_data
        .groupby('id')['value']
        .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    )
    df_test_li.loc[var_data.index, 'value'] = var_data['value']
    pred_li = df_test_li.loc[test_idx, 'value'].values

    # Evaluate
    valid = ~(np.isnan(pred_ff) | np.isnan(pred_li))
    true_v = true_vals[valid]
    mae_ff = np.mean(np.abs(true_v - pred_ff[valid]))
    mae_li = np.mean(np.abs(true_v - pred_li[valid]))
    rmse_ff = np.sqrt(np.mean((true_v - pred_ff[valid]) ** 2))
    rmse_li = np.sqrt(np.mean((true_v - pred_li[valid]) ** 2))

    results[var] = {
        'mae_ff': mae_ff, 'rmse_ff': rmse_ff,
        'mae_li': mae_li, 'rmse_li': rmse_li,
        'n_eval': valid.sum()
    }
    print(f"\n  {var} (evaluated on {valid.sum()} samples):")
    print(f"    Forward Fill:          MAE = {mae_ff:.4f},  RMSE = {rmse_ff:.4f}")
    print(f"    Linear Interpolation:  MAE = {mae_li:.4f},  RMSE = {rmse_li:.4f}")

avg_mae_ff = np.mean([r['mae_ff'] for r in results.values()])
avg_mae_li = np.mean([r['mae_li'] for r in results.values()])
avg_rmse_ff = np.mean([r['rmse_ff'] for r in results.values()])
avg_rmse_li = np.mean([r['rmse_li'] for r in results.values()])

print(f"\n  Average across variables:")
print(f"    Forward Fill:          MAE = {avg_mae_ff:.4f},  RMSE = {avg_rmse_ff:.4f}")
print(f"    Linear Interpolation:  MAE = {avg_mae_li:.4f},  RMSE = {avg_rmse_li:.4f}")

if avg_mae_li < avg_mae_ff:
    print("\n  >> Linear Interpolation performs better -> SELECTED")
    df_final = df_interp.copy()
    chosen_method = "Linear Interpolation"
else:
    print("\n  >> Forward Fill performs better -> SELECTED")
    df_final = df_ffill.copy()
    chosen_method = "Forward Fill"


# ---- PLOT: Imputation comparison for one user ----
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sample_user = 'AS14.01'

for i, var in enumerate(impute_vars):
    ax = axes[i]
    orig = df_clean[(df_clean['id'] == sample_user) & (df_clean['variable'] == var)].sort_values('time')
    ff = df_ffill[(df_ffill['id'] == sample_user) & (df_ffill['variable'] == var)].sort_values('time')
    li = df_interp[(df_interp['id'] == sample_user) & (df_interp['variable'] == var)].sort_values('time')

    orig_valid = orig[orig['value'].notna()]
    ax.plot(orig_valid['time'], orig_valid['value'], 'ko', markersize=3, label='Original', zorder=3)

    orig_missing = orig[orig['value'].isna()]
    if len(orig_missing) > 0:
        ff_vals = ff.loc[orig_missing.index, 'value']
        li_vals = li.loc[orig_missing.index, 'value']
        ax.plot(orig_missing['time'], ff_vals, 's', color='red', markersize=5,
                alpha=0.8, label='Forward Fill')
        ax.plot(orig_missing['time'], li_vals, '^', color='blue', markersize=5,
                alpha=0.8, label='Linear Interp.')

    ax.set_title(f'{var} - {sample_user}')
    ax.set_xlabel('Time')
    ax.set_ylabel(var)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.tick_params(axis='x', rotation=45)

plt.suptitle(f'Imputation Comparison for {sample_user}', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot_imputation_comparison.png')
plt.close()
print("\nSaved: plot_imputation_comparison.png")

# ---- PLOT: Distribution before vs after imputation ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, var in enumerate(impute_vars):
    ax = axes[i]
    before_vals = df_clean.loc[df_clean['variable'] == var, 'value'].dropna()
    after_vals = df_final.loc[df_final['variable'] == var, 'value'].dropna()
    ax.hist(before_vals, bins=30, alpha=0.5, color='gray', label=f'Before (n={len(before_vals)})',
            edgecolor='black', density=True)
    ax.hist(after_vals, bins=30, alpha=0.5, color='steelblue', label=f'After (n={len(after_vals)})',
            edgecolor='black', density=True)
    ax.set_title(f'{var}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

plt.suptitle(f'Distribution Before vs After Imputation ({chosen_method})', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot_imputation_distribution_impact.png')
plt.close()
print("Saved: plot_imputation_distribution_impact.png")


# =============================================================================
# SAVE CLEANED LONG-FORMAT DATASET
# =============================================================================
df_final.to_csv(f'cleaned_long_format.csv', index=False)
print(f"\nCleaned long-format dataset saved: {df_final.shape}")
print(f"Columns: {list(df_final.columns)}")

# Verify
print("\nRemaining NaN values:")
total_na = df_final['value'].isna().sum()
if total_na == 0:
    print("  None! All missing values have been imputed.")
else:
    for var in variables:
        n_na = df_final.loc[df_final['variable'] == var, 'value'].isna().sum()
        if n_na > 0:
            print(f"  {var}: {n_na}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("TASK 1B SUMMARY")
print("=" * 70)
print(f"""
1B.1 - Outlier Removal:
  Total rows removed: {total_removed} / {len(df)} ({total_removed/len(df)*100:.2f}%)
    - Duplicates:         {n_dupes}
    - Out-of-range:       {removed_range}
    - Negative durations: {removed_neg}
    - IQR outliers:       {removed_iqr}

1B.2 - Missing Value Imputation:
  Variables with NaN: {impute_vars}
  Method 1 (Forward Fill):         avg MAE = {avg_mae_ff:.4f},  avg RMSE = {avg_rmse_ff:.4f}
  Method 2 (Linear Interpolation): avg MAE = {avg_mae_li:.4f},  avg RMSE = {avg_rmse_li:.4f}
  Selected method: {chosen_method}

Output: cleaned_long_format.csv
  Shape: {df_final.shape}
  Format: long-format (same structure as raw input, ready for Task 1C)
""")