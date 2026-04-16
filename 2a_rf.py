"""
=============================================================================
Data Mining Techniques - Assignment 1
Task 2A - Algorithm 1: Random Forest Classification

Uses the instance-based feature dataset from Task 1C.
Target: next-day mood discretized into 3 classes (low / medium / high)

Evaluation pipeline:
  1. Split users into train (80%) and test (20%) sets
  2. On train set: GroupKFold CV to tune hyperparameters
  3. Retrain best model on full train set
  4. Report final performance on held-out test set

Input:  feature_dataset.csv (from Task 1C)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, ConfusionMatrixDisplay)
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 11
})
sns.set_style("whitegrid")

OUTPUT_DIR = 'task2a_rf_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 70)
print("TASK 2A - Algorithm 1: Random Forest Classification")
print("=" * 70)

df = pd.read_csv('feature_dataset.csv')
print(f"Loaded dataset: {df.shape}")

feature_cols = [
    'mood_lag1', 'mood_mean_3d', 'mood_mean_7d', 'mood_std_7d',
    'mood_min_7d', 'mood_max_7d', 'mood_range_7d', 'mood_slope_7d',
    'arousal_lag1', 'arousal_mean_7d', 'arousal_std_7d',
    'valence_lag1', 'valence_mean_7d', 'valence_std_7d',
    'activity_lag1', 'activity_mean_3d', 'activity_mean_7d', 'activity_std_7d',
    'screen_lag1', 'screen_mean_7d', 'screen_std_7d', 'screen_total_7d',
    'call_count_7d', 'call_mean_7d', 'sms_count_7d', 'social_comm_7d',
    'app_entertainment_7d', 'app_social_7d', 'app_total_7d', 'app_builtin_7d',
    'day_of_week', 'is_weekend', 'day_index',
]

# =============================================================================
# 2. DISCRETIZE TARGET INTO 3 CLASSES
# =============================================================================
print("\n--- Discretizing target into 3 classes ---")

q33 = df['target_mood'].quantile(0.33)
q67 = df['target_mood'].quantile(0.67)
print(f"Quantile thresholds: q33={q33:.2f}, q67={q67:.2f}")

def mood_to_class(val):
    if val <= q33:
        return 0  # low
    elif val <= q67:
        return 1  # medium
    else:
        return 2  # high

df['target_class'] = df['target_mood'].apply(mood_to_class)
class_names = ['low', 'medium', 'high']

print("Overall class distribution:")
for i, name in enumerate(class_names):
    n = (df['target_class'] == i).sum()
    print(f"  {name}: {n} ({n/len(df)*100:.1f}%)")

# =============================================================================
# 3. TRAIN / TEST SPLIT (by user)
# =============================================================================
print("\n--- Train / Test Split (by user) ---")

all_users = np.array(sorted(df['id'].unique()))
np.random.shuffle(all_users)

n_test_users = max(5, len(all_users) // 5)  # ~20% of users
test_users = set(all_users[:n_test_users])
train_users = set(all_users[n_test_users:])

print(f"Total users: {len(all_users)}")
print(f"Train users ({len(train_users)}): {sorted(train_users)}")
print(f"Test users  ({len(test_users)}):  {sorted(test_users)}")

train_df = df[df['id'].isin(train_users)]
test_df = df[df['id'].isin(test_users)]

X_train = train_df[feature_cols].values
y_train = train_df['target_class'].values
groups_train = train_df['id'].values

X_test = test_df[feature_cols].values
y_test = test_df['target_class'].values

print(f"\nTrain set: {len(train_df)} instances")
print(f"Test set:  {len(test_df)} instances")

print("\nTrain class distribution:")
for i, name in enumerate(class_names):
    n = (y_train == i).sum()
    print(f"  {name}: {n} ({n/len(y_train)*100:.1f}%)")
print("Test class distribution:")
for i, name in enumerate(class_names):
    n = (y_test == i).sum()
    print(f"  {name}: {n} ({n/len(y_test)*100:.1f}%)")

# =============================================================================
# 4. HYPERPARAMETER OPTIMIZATION (CV on train set only)
# =============================================================================
print("\n--- Hyperparameter Optimization (GridSearchCV on train set) ---")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5],
}

gkf = GroupKFold(n_splits=4)  # 4 folds on train users
rf_base = RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced')

grid_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=gkf,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=0,
    refit=True,
)

print("Searching over parameter grid...")
grid_search.fit(X_train, y_train, groups=groups_train)

best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_
print(f"\nBest parameters: {best_params}")
print(f"Best CV F1-macro (on train set): {best_cv_score:.4f}")

# =============================================================================
# 5. FINAL EVALUATION ON TEST SET
# =============================================================================
print("\n--- Final Evaluation on Test Set ---")

best_rf = grid_search.best_estimator_  # already refit on full train set
y_pred = best_rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\nTest Accuracy:    {acc:.4f}")
print(f"Test F1-macro:    {f1_macro:.4f}")
print(f"Test F1-weighted: {f1_weighted:.4f}")

print(f"\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred, target_names=class_names))

# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================
print("\n--- Feature Importance ---")

importances = best_rf.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print("Top 15 features:")
for _, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

# =============================================================================
# 7. PLOTS
# =============================================================================

# ---- PLOT 1: Confusion Matrix ----
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues')
ax.set_title(f'Random Forest - Test Set Confusion Matrix\nAccuracy={acc:.3f}, F1-macro={f1_macro:.3f}')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot_rf_confusion_matrix.png')
plt.close()
print("\nSaved: plot_rf_confusion_matrix.png")

# ---- PLOT 2: Feature Importance ----
fig, ax = plt.subplots(figsize=(10, 10))
top20 = importance_df.head(20).sort_values('importance', ascending=True)
ax.barh(range(len(top20)), top20['importance'], color='steelblue', edgecolor='black')
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['feature'])
ax.set_xlabel('Feature Importance (Gini)')
ax.set_title('Random Forest - Top 20 Feature Importances')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot_rf_feature_importance.png')
plt.close()
print("Saved: plot_rf_feature_importance.png")

# ---- PLOT 3: Class distribution (test set) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (vals, title) in enumerate([(y_test, 'Actual (Test)'), (y_pred, 'Predicted (Test)')]):
    counts = [np.sum(vals == c) for c in range(3)]
    axes[i].bar(class_names, counts, color=['salmon', 'lightyellow', 'steelblue'],
                edgecolor='black')
    axes[i].set_ylabel('Count')
    axes[i].set_title(f'{title} Class Distribution')
    for j, c in enumerate(counts):
        axes[i].text(j, c + 5, str(c), ha='center')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot_rf_class_distribution.png')
plt.close()
print("Saved: plot_rf_class_distribution.png")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("TASK 2A - RANDOM FOREST SUMMARY")
print("=" * 70)
print(f"""
Algorithm: Random Forest Classifier
  - Operates on: instance-based feature dataset (33 features from Task 1C)
  - Class definition: 3 classes (low/medium/high) via quantile binning
    Thresholds: low <= {q33:.2f}, medium <= {q67:.2f}, high > {q67:.2f}

Data Split:
  - Train: {len(train_users)} users ({len(train_df)} instances)
  - Test:  {len(test_users)} users ({len(test_df)} instances)
  - Split by user to prevent data leakage

Hyperparameter Optimization:
  - Method: GridSearchCV with GroupKFold (4 folds) on train set only
  - Scoring: F1-macro
  - Best parameters: {best_params}
  - Best CV F1-macro: {best_cv_score:.4f}

Test Set Results:
  - Accuracy:    {acc:.4f}
  - F1-macro:    {f1_macro:.4f}
  - F1-weighted: {f1_weighted:.4f}
  - Top 3 features: {', '.join(importance_df.head(3)['feature'].tolist())}
""")