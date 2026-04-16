"""
=============================================================================
Data Mining Techniques - Assignment 1
Task 1C: Feature Engineering

  1. Aggregate long-format data to daily wide-format (per user per day)
  2. Define target: next-day average mood
  3. Define features using a 7-day sliding window (with 1d, 3d, 7d granularity)
  4. Build instance-based dataset as described in Figure 1

Input:  cleaned_long_format.csv (from Task 1B)
Output: feature_dataset.csv (ready for Task 2 classification & Task 4 regression)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

OUTPUT_DIR = 'task_1c_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW = 7  # sliding window size in days

# =============================================================================
# STEP 1: LOAD CLEANED LONG-FORMAT DATA
# =============================================================================
print("=" * 70)
print("TASK 1C: FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv('cleaned_long_format.csv')
df['time'] = pd.to_datetime(df['time'], format='mixed')
df['date'] = df['time'].dt.date
df['date'] = pd.to_datetime(df['date'])

print(f"Loaded cleaned data: {len(df)} rows, {df['id'].nunique()} users")


# =============================================================================
# STEP 2: AGGREGATE TO DAILY WIDE-FORMAT
# =============================================================================
print("\n" + "-" * 70)
print("Step 2: Aggregating to daily wide-format")
print("-" * 70)

variables = df['variable'].unique()

# --- Mean-aggregated: mood, arousal, valence, activity ---
mean_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
# --- Count-aggregated: call, sms ---
count_vars = ['call', 'sms']
# --- Sum-aggregated: screen, appCat.* ---
sum_vars = ['screen'] + [v for v in variables if v.startswith('appCat.')]

results = []
for var in mean_vars:
    sub = df[df['variable'] == var]
    agg = sub.groupby(['id', 'date'])['value'].mean().reset_index()
    agg.columns = ['id', 'date', var]
    results.append(agg)

for var in count_vars:
    sub = df[df['variable'] == var]
    agg = sub.groupby(['id', 'date'])['value'].sum().reset_index()
    agg.columns = ['id', 'date', var]
    results.append(agg)

for var in sum_vars:
    sub = df[df['variable'] == var]
    agg = sub.groupby(['id', 'date'])['value'].sum().reset_index()
    agg.columns = ['id', 'date', var]
    results.append(agg)

# Build full date range per user
date_ranges = []
for uid in df['id'].unique():
    user = df[df['id'] == uid]
    dates = pd.date_range(user['date'].min(), user['date'].max(), freq='D')
    date_ranges.append(pd.DataFrame({'id': uid, 'date': dates}))
full_index = pd.concat(date_ranges, ignore_index=True)

# Merge all
daily = full_index.copy()
for r in results:
    daily = daily.merge(r, on=['id', 'date'], how='left')

# call/sms/screen/appCat: 0 if no records on that day
daily['call'] = daily['call'].fillna(0)
daily['sms'] = daily['sms'].fillna(0)
for col in sum_vars:
    if col in daily.columns:
        daily[col] = daily[col].fillna(0)

# Mark which days have real mood data (not NaN)
daily['has_mood'] = daily['mood'].notna()

# Impute mood/arousal/valence/activity for feature computation
# (we need continuous values in the window to compute features,
#  but only days with real mood will be used as targets)
for col in mean_vars:
    daily[col] = daily.groupby('id')[col].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )

daily = daily.sort_values(['id', 'date']).reset_index(drop=True)
print(f"Daily wide-format: {daily.shape}")
print(f"Days with real mood data: {daily['has_mood'].sum()}")
print(f"Days without mood (interpolated for features only): {(~daily['has_mood']).sum()}")


# =============================================================================
# STEP 3: MERGE SMALL APP CATEGORIES
# =============================================================================
print("\n" + "-" * 70)
print("Step 3: Merging sparse app categories")
print("-" * 70)

# Combine low-coverage appCat columns into broader groups
daily['appCat.social_comm'] = daily['appCat.social'] + daily['appCat.communication']
daily['appCat.leisure'] = daily['appCat.entertainment'] + daily['appCat.game']
daily['appCat.productivity'] = daily['appCat.office'] + daily['appCat.finance']
daily['appCat.misc'] = (daily['appCat.other'] + daily['appCat.travel'] +
                        daily['appCat.unknown'] + daily['appCat.utilities'] +
                        daily['appCat.weather'])
daily['appCat.total'] = daily[[c for c in daily.columns if c.startswith('appCat.') and
                                c not in ['appCat.social_comm', 'appCat.leisure',
                                          'appCat.productivity', 'appCat.misc',
                                          'appCat.total']]].sum(axis=1)

print("Created merged app categories: social_comm, leisure, productivity, misc, total")


# =============================================================================
# STEP 4: DEFINE TARGET VARIABLE
# =============================================================================
print("\n" + "-" * 70)
print("Step 4: Defining target variable (next-day mood)")
print("-" * 70)

# Target = next day's average mood
# We shift mood by -1 within each user so row t gets the mood of day t+1
daily['target_mood'] = daily.groupby('id')['mood'].shift(-1)

# Also track if the target day has real mood data
daily['target_has_mood'] = daily.groupby('id')['has_mood'].shift(-1)

print(f"Target variable created (next-day mood)")
print(f"Rows with valid target (real mood on next day): {daily['target_has_mood'].sum()}")


# =============================================================================
# STEP 5: FEATURE ENGINEERING WITH SLIDING WINDOW
# =============================================================================
print("\n" + "-" * 70)
print("Step 5: Feature engineering (7-day sliding window)")
print("-" * 70)


def compute_slope(series):
    """Compute linear regression slope over a series."""
    y = series.values
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return np.nan
    x = np.arange(len(y))[valid]
    y = y[valid]
    if len(x) < 2:
        return np.nan
    slope = np.polyfit(x, y, 1)[0]
    return slope


# We will compute features using rolling windows on the daily data
# grouped by user. Features are computed for each row based on the
# WINDOW days ending at that row (i.e., day t uses days t-6 to t).

feature_dfs = []

for uid in sorted(daily['id'].unique()):
    user = daily[daily['id'] == uid].copy().sort_values('date').reset_index(drop=True)
    n = len(user)

    # --- 1. MOOD FEATURES ---
    user['mood_lag1'] = user['mood'].shift(1)
    user['mood_mean_3d'] = user['mood'].rolling(3, min_periods=1).mean().shift(1)
    user['mood_mean_7d'] = user['mood'].rolling(WINDOW, min_periods=1).mean().shift(1)
    user['mood_std_7d'] = user['mood'].rolling(WINDOW, min_periods=1).std().shift(1)
    user['mood_min_7d'] = user['mood'].rolling(WINDOW, min_periods=1).min().shift(1)
    user['mood_max_7d'] = user['mood'].rolling(WINDOW, min_periods=1).max().shift(1)
    user['mood_range_7d'] = user['mood_max_7d'] - user['mood_min_7d']
    # Slope: computed manually
    mood_slopes = []
    for i in range(n):
        start = max(0, i - WINDOW)
        end = i  # exclusive of current day (shift by 1)
        if end - start < 2:
            mood_slopes.append(np.nan)
        else:
            mood_slopes.append(compute_slope(user['mood'].iloc[start:end]))
    user['mood_slope_7d'] = mood_slopes

    # --- 2. AROUSAL FEATURES ---
    user['arousal_lag1'] = user['circumplex.arousal'].shift(1)
    user['arousal_mean_7d'] = user['circumplex.arousal'].rolling(WINDOW, min_periods=1).mean().shift(1)
    user['arousal_std_7d'] = user['circumplex.arousal'].rolling(WINDOW, min_periods=1).std().shift(1)

    # --- 3. VALENCE FEATURES ---
    user['valence_lag1'] = user['circumplex.valence'].shift(1)
    user['valence_mean_7d'] = user['circumplex.valence'].rolling(WINDOW, min_periods=1).mean().shift(1)
    user['valence_std_7d'] = user['circumplex.valence'].rolling(WINDOW, min_periods=1).std().shift(1)

    # --- 4. ACTIVITY FEATURES ---
    user['activity_lag1'] = user['activity'].shift(1)
    user['activity_mean_3d'] = user['activity'].rolling(3, min_periods=1).mean().shift(1)
    user['activity_mean_7d'] = user['activity'].rolling(WINDOW, min_periods=1).mean().shift(1)
    user['activity_std_7d'] = user['activity'].rolling(WINDOW, min_periods=1).std().shift(1)

    # --- 5. SCREEN FEATURES ---
    user['screen_lag1'] = user['screen'].shift(1)
    user['screen_mean_7d'] = user['screen'].rolling(WINDOW, min_periods=1).mean().shift(1)
    user['screen_std_7d'] = user['screen'].rolling(WINDOW, min_periods=1).std().shift(1)
    user['screen_total_7d'] = user['screen'].rolling(WINDOW, min_periods=1).sum().shift(1)

    # --- 6. SOCIAL INTERACTION FEATURES ---
    user['call_count_7d'] = user['call'].rolling(WINDOW, min_periods=1).sum().shift(1)
    user['call_mean_7d'] = user['call'].rolling(WINDOW, min_periods=1).mean().shift(1)
    user['sms_count_7d'] = user['sms'].rolling(WINDOW, min_periods=1).sum().shift(1)
    user['social_comm_7d'] = user['appCat.social_comm'].rolling(WINDOW, min_periods=1).mean().shift(1)

    # --- 7. APP USAGE FEATURES ---
    user['app_entertainment_7d'] = user['appCat.leisure'].rolling(WINDOW, min_periods=1).mean().shift(1)
    user['app_social_7d'] = user['appCat.social'].rolling(WINDOW, min_periods=1).mean().shift(1)
    user['app_total_7d'] = user['appCat.total'].rolling(WINDOW, min_periods=1).mean().shift(1)
    user['app_builtin_7d'] = user['appCat.builtin'].rolling(WINDOW, min_periods=1).mean().shift(1)

    # --- 8. TEMPORAL FEATURES ---
    user['day_of_week'] = user['date'].dt.dayofweek
    user['is_weekend'] = (user['day_of_week'] >= 5).astype(int)
    first_date = user['date'].min()
    user['day_index'] = (user['date'] - first_date).dt.days

    feature_dfs.append(user)

features_all = pd.concat(feature_dfs, ignore_index=True)
print(f"Features computed for all users: {features_all.shape}")


# =============================================================================
# STEP 6: BUILD FINAL INSTANCE-BASED DATASET
# =============================================================================
print("\n" + "-" * 70)
print("Step 6: Building final instance-based dataset")
print("-" * 70)

# Filter: only keep rows where
# 1. Target mood exists and is real (not interpolated)
# 2. At least WINDOW days of history available
# 3. The row itself is at least WINDOW days from the start

dataset = features_all.copy()

# Must have real target mood
dataset = dataset[dataset['target_has_mood'] == True]
print(f"After requiring real target mood: {len(dataset)} rows")

# Must have at least WINDOW days of history (day_index >= WINDOW)
dataset = dataset[dataset['day_index'] >= WINDOW]
print(f"After requiring {WINDOW}-day history: {len(dataset)} rows")

# Select feature columns and target
feature_cols = [
    # Mood features
    'mood_lag1', 'mood_mean_3d', 'mood_mean_7d', 'mood_std_7d',
    'mood_min_7d', 'mood_max_7d', 'mood_range_7d', 'mood_slope_7d',
    # Arousal features
    'arousal_lag1', 'arousal_mean_7d', 'arousal_std_7d',
    # Valence features
    'valence_lag1', 'valence_mean_7d', 'valence_std_7d',
    # Activity features
    'activity_lag1', 'activity_mean_3d', 'activity_mean_7d', 'activity_std_7d',
    # Screen features
    'screen_lag1', 'screen_mean_7d', 'screen_std_7d', 'screen_total_7d',
    # Social features
    'call_count_7d', 'call_mean_7d', 'sms_count_7d', 'social_comm_7d',
    # App usage features
    'app_entertainment_7d', 'app_social_7d', 'app_total_7d', 'app_builtin_7d',
    # Temporal features
    'day_of_week', 'is_weekend', 'day_index',
]

target_col = 'target_mood'

# Build final dataset
final = dataset[['id', 'date'] + feature_cols + [target_col]].copy()

# Drop rows with any NaN in features
before_drop = len(final)
final = final.dropna(subset=feature_cols + [target_col])
print(f"After dropping NaN rows: {len(final)} (dropped {before_drop - len(final)})")

print(f"\nFinal dataset shape: {final.shape}")
print(f"Number of features: {len(feature_cols)}")
print(f"Number of users: {final['id'].nunique()}")
print(f"Instances per user:")
for uid in sorted(final['id'].unique()):
    n = len(final[final['id'] == uid])
    print(f"  {uid}: {n} instances")


# =============================================================================
# STEP 7: SAVE AND VISUALIZE
# =============================================================================
print("\n" + "-" * 70)
print("Step 7: Saving and visualizing")
print("-" * 70)

final.to_csv(f'feature_dataset.csv', index=False)
print(f"Saved: feature_dataset.csv")

# --- Summary statistics of features ---
print(f"\nFeature summary statistics:")
print(final[feature_cols].describe().round(3).to_string())

# --- Target distribution ---
print(f"\nTarget (next-day mood) statistics:")
print(f"  Mean: {final[target_col].mean():.3f}")
print(f"  Std:  {final[target_col].std():.3f}")
print(f"  Min:  {final[target_col].min():.3f}")
print(f"  Max:  {final[target_col].max():.3f}")

# ---- PLOT 1: Target mood distribution ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(final[target_col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Next-Day Average Mood')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Target Variable (Next-Day Mood)')
axes[0].axvline(final[target_col].mean(), color='red', linestyle='--',
                label=f'Mean={final[target_col].mean():.2f}')
axes[0].legend()

# Instances per user
user_counts = final.groupby('id').size().sort_values()
user_counts.plot(kind='barh', ax=axes[1], color='steelblue', edgecolor='black')
axes[1].set_xlabel('Number of Instances')
axes[1].set_ylabel('User ID')
axes[1].set_title('Number of Training Instances per User')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot_1c_target_distribution.png')
plt.close()
print("Saved: plot_1c_target_distribution.png")

# ---- PLOT 2: Feature correlation with target ----
corr_with_target = final[feature_cols + [target_col]].corr()[target_col].drop(target_col)
corr_sorted = corr_with_target.abs().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 10))
colors = ['steelblue' if v >= 0 else 'salmon' for v in corr_with_target[corr_sorted.index]]
ax.barh(range(len(corr_sorted)), corr_with_target[corr_sorted.index], color=colors, edgecolor='black')
ax.set_yticks(range(len(corr_sorted)))
ax.set_yticklabels(corr_sorted.index)
ax.set_xlabel('Pearson Correlation with Target Mood')
ax.set_title('Feature Correlation with Next-Day Mood')
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot_1c_feature_correlation.png')
plt.close()
print("Saved: plot_1c_feature_correlation.png")

# ---- PLOT 3: Feature correlation heatmap (top features) ----
top_features = corr_sorted.tail(15).index.tolist()
corr_matrix = final[top_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            vmin=-1, vmax=1, square=True)
ax.set_title('Correlation Matrix of Top 15 Features')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot_1c_feature_heatmap.png')
plt.close()
print("Saved: plot_1c_feature_heatmap.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("TASK 1C SUMMARY")
print("=" * 70)
print(f"""
Feature Engineering Pipeline:
  1. Aggregated long-format -> daily wide-format (per user per day)
  2. Merged sparse app categories into 5 groups
  3. Target: next-day average mood
  4. Sliding window: {WINDOW} days, with multi-granularity (1d, 3d, 7d)
  5. Only days with real mood recordings used as targets

Feature Categories ({len(feature_cols)} features total):
  - Mood history:      8 features (lag1, mean_3d/7d, std, min, max, range, slope)
  - Arousal:           3 features (lag1, mean_7d, std_7d)
  - Valence:           3 features (lag1, mean_7d, std_7d)
  - Activity:          4 features (lag1, mean_3d/7d, std_7d)
  - Screen:            4 features (lag1, mean_7d, std_7d, total_7d)
  - Social:            4 features (call_count, call_mean, sms_count, social_comm)
  - App usage:         4 features (entertainment, social, total, builtin)
  - Temporal:          3 features (day_of_week, is_weekend, day_index)

Final Dataset:
  Shape: {final.shape}
  Users: {final['id'].nunique()}
  Instances: {len(final)}

Output: feature_dataset.csv (ready for Task 2 & Task 4)
""")