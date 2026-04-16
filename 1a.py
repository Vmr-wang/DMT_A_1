"""
DMT Assignment 1
Task 1A: Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
 
# Set plot style
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


# Load Data
df = pd.read_csv('dataset_mood_smartphone.csv')
df = df.drop(columns=['Unnamed: 0'])
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date
df['date'] = pd.to_datetime(df['date'])
 
print(f"Total records: {len(df)}")
print(f"Number of users: {df['id'].nunique()}")
print(f"Number of variables: {df['variable'].nunique()}")
print(f"Date range: {df['time'].min()} ~ {df['time'].max()}")


# Dataset overview table 
print("\n--- Dataset Overview ---")
variables = df['variable'].unique()
overview_rows = []
for var in variables:
    sub = df[df['variable'] == var]
    overview_rows.append({
        'Variable': var,
        'Count': len(sub),
        'Missing': sub['value'].isna().sum(),
        'Min': sub['value'].min(),
        'Max': sub['value'].max(),
        'Mean': sub['value'].mean(),
        'Std': sub['value'].std(),
        'Median': sub['value'].median(),
        'Users': sub['id'].nunique()
    })
overview_df = pd.DataFrame(overview_rows)
overview_df = overview_df.round(3)
print(overview_df.to_string(index=False))
overview_df.to_csv('task_1a_tables/table_dataset_overview.csv', index=False)
 
# Per-user summary
print("\n--- Per-User Summary ---")
user_summary = []
for uid in sorted(df['id'].unique()):
    user = df[df['id'] == uid]
    mood_data = user[user['variable'] == 'mood']
    user_summary.append({
        'User': uid,
        'Date Range': f"{user['date'].min().date()} ~ {user['date'].max().date()}",
        'Total Days': user['date'].dt.date.nunique(),
        'Mood Days': mood_data['date'].dt.date.nunique(),
        'Mood Records': len(mood_data),
        'Mood Mean': mood_data['value'].mean(),
        'Mood Std': mood_data['value'].std(),
        'Total Records': len(user)
    })
user_summary_df = pd.DataFrame(user_summary).round(2)
print(user_summary_df.to_string(index=False))
user_summary_df.to_csv('task_1a_tables/table_user_summary.csv', index=False)
 
# Duplicate check
print("\n--- Duplicate Check ---")
dupes = df.groupby(['id', 'time', 'variable']).size().reset_index(name='count')
n_dupes = (dupes['count'] > 1).sum()
print(f"Number of duplicate (id, time, variable) groups: {n_dupes}")


'''
Make Plots
'''

print("Plots")
 
# Plot 1: Mood distribution (overall + per user boxplot)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
mood = df[df['variable'] == 'mood']['value']
axes[0].hist(mood, bins=range(1, 12), edgecolor='black', alpha=0.7, color='steelblue', align='left')
axes[0].set_xlabel('Mood Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Overall Mood Distribution')
axes[0].set_xticks(range(1, 11))
axes[0].axvline(mood.mean(), color='red', linestyle='--', label=f'Mean={mood.mean():.2f}')
axes[0].legend()
 
mood_df = df[df['variable'] == 'mood'][['id', 'value']]
user_order = mood_df.groupby('id')['value'].mean().sort_values().index
sns.boxplot(data=mood_df, x='id', y='value', order=user_order, ax=axes[1],
            palette='coolwarm', fliersize=2)
axes[1].set_xlabel('User ID')
axes[1].set_ylabel('Mood Score')
axes[1].set_title('Mood Distribution per User')
axes[1].tick_params(axis='x', rotation=90)
 
plt.tight_layout()
plt.savefig('task_1a_plots/plot1_mood_distribution.png')
plt.close()
print("Saved: plot1_mood_distribution.png")
 
# Plot 2: Mood over time for selected users
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
mood_data = df[df['variable'] == 'mood'].copy()
top_users = mood_data.groupby('id').size().nlargest(9).index
for i, uid in enumerate(top_users):
    ax = axes[i // 3, i % 3]
    user_mood = mood_data[mood_data['id'] == uid].copy()
    daily_mood = user_mood.groupby('date')['value'].mean()
    ax.plot(daily_mood.index, daily_mood.values, 'o-', markersize=2, alpha=0.7, linewidth=1)
    ax.set_title(f'{uid} (n={len(user_mood)})')
    ax.set_ylim(0, 11)
    ax.set_ylabel('Mood')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.tick_params(axis='x', rotation=45)
plt.suptitle('Daily Average Mood Over Time (Top 9 Users by # Records)', fontsize=14)
plt.tight_layout()
plt.savefig('task_1a_plots/plot2_mood_over_time.png')
plt.close()
print("Saved: plot2_mood_over_time.png")
 
# Plot 3: Distribution of key variables
key_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen']
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for i, var in enumerate(key_vars):
    sub = df[(df['variable'] == var) & df['value'].notna()]['value']
    if var == 'screen':
        sub = sub[sub < sub.quantile(0.99)]  # clip for visualization
    axes[i].hist(sub, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].set_title(var)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
plt.suptitle('Distribution of Key Variables', fontsize=14)
plt.tight_layout()
plt.savefig('task_1a_plots/plot3_key_variable_distributions.png')
plt.close()
print("Saved: plot3_key_variable_distributions.png")
 
# Plot 4: Variable coverage heatmap (per-user)
coverage_pct = pd.DataFrame()
for uid in sorted(df['id'].unique()):
    user = df[df['id'] == uid]
    total_days = user['date'].dt.date.nunique()
    for var in variables:
        days_with_var = user[user['variable'] == var]['date'].dt.date.nunique()
        coverage_pct.loc[var, uid] = days_with_var / total_days * 100
 
fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(coverage_pct, annot=False, cmap='YlOrRd', ax=ax, vmin=0, vmax=100,
            cbar_kws={'label': 'Coverage (%)'})
ax.set_title('Variable Coverage per User (% of Days with Data)')
ax.set_xlabel('User ID')
ax.set_ylabel('Variable')
plt.tight_layout()
plt.savefig('task_1a_plots/plot4_variable_coverage_heatmap.png')
plt.close()
print("Saved: plot4_variable_coverage_heatmap.png")
 
# Plot 5: Correlation between daily aggregated variables
daily_pivot = df.groupby(['id', 'date', 'variable'])['value'].mean().reset_index()
daily_wide = daily_pivot.pivot_table(index=['id', 'date'], columns='variable', values='value')

core_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen',
             'appCat.communication', 'appCat.social', 'appCat.entertainment']
core_vars_available = [v for v in core_vars if v in daily_wide.columns]
corr = daily_wide[core_vars_available].corr()
 
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            mask=mask, vmin=-1, vmax=1, square=True)
ax.set_title('Correlation Matrix of Daily Aggregated Variables')
plt.tight_layout()
plt.savefig('task_1a_plots/plot5_correlation_matrix.png')
plt.close()
print("Saved: plot5_correlation_matrix.png")
 
# Plot 6: Mood by day of week
mood_data_full = df[df['variable'] == 'mood'].copy()
mood_data_full['dayofweek'] = mood_data_full['time'].dt.dayofweek
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
mood_dow = mood_data_full.copy()
mood_dow['day_name'] = mood_dow['dayofweek'].map(dict(enumerate(day_names)))
sns.boxplot(data=mood_dow, x='dayofweek', y='value', ax=axes[0], palette='Set2')
axes[0].set_xticklabels(day_names)
axes[0].set_xlabel('Day of Week')
axes[0].set_ylabel('Mood Score')
axes[0].set_title('Mood Distribution by Day of Week')
 
mood_data_full['hour'] = mood_data_full['time'].dt.hour
hourly_mood = mood_data_full.groupby('hour')['value'].agg(['mean', 'std'])
axes[1].errorbar(hourly_mood.index, hourly_mood['mean'], yerr=hourly_mood['std'],
                 fmt='o-', capsize=3, color='steelblue')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Mood Score')
axes[1].set_title('Mood by Time of Day (Mean ± Std)')
axes[1].set_xticks(range(0, 24, 2))
 
plt.tight_layout()
plt.savefig('task_1a_plots/plot6_mood_temporal_patterns.png')
plt.close()
print("Saved: plot6_mood_temporal_patterns.png")
 
# Plot 7: Number of records per variable
record_counts = df.groupby('variable').size().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
record_counts.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
ax.set_xlabel('Number of Records (log scale)')
ax.set_title('Number of Records per Variable')
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('task_1a_plots/plot7_records_per_variable.png')
plt.close()
print("Saved: plot7_records_per_variable.png")
 
# Plot 8: Screen and Activity daily patterns
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
screen_data = df[df['variable'] == 'screen'].copy()
screen_daily = screen_data.groupby(['id', 'date'])['value'].sum().reset_index()
screen_avg = screen_daily.groupby('date')['value'].mean()
axes[0].plot(screen_avg.index, screen_avg.values, '-', alpha=0.7, color='orange', linewidth=1)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Avg Daily Screen Time (s)')
axes[0].set_title('Average Daily Screen Time Across Users')
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
axes[0].tick_params(axis='x', rotation=45)

act_data = df[df['variable'] == 'activity']['value']
axes[1].hist(act_data, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].set_xlabel('Activity Level')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Activity Score Distribution')
 
plt.tight_layout()
plt.savefig('task_1a_plots/plot8_screen_activity_patterns.png')
plt.close()
print("Saved: plot8_screen_activity_patterns.png")