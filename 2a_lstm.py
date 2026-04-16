"""
=============================================================================
Data Mining Techniques - Assignment 1
Task 2A - Algorithm 2: LSTM Classification (Inherently Temporal)

Uses the raw daily time series directly (NOT the feature-engineered dataset).
Feeds sequences of 7 consecutive days into an LSTM to predict next-day mood class.

Evaluation pipeline:
  1. Split users into train (80%) and test (20%) sets
  2. On train set: GroupKFold CV to select best epoch / validate
  3. Retrain best model on full train set
  4. Report final performance on held-out test set

Input:  cleaned_long_format.csv (from Task 1B)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 11
})
sns.set_style("whitegrid")

OUTPUT_DIR = 'task2a_lstm_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW = 7
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =============================================================================
# 1. LOAD AND AGGREGATE DATA TO DAILY WIDE-FORMAT
# =============================================================================
print("=" * 70)
print("TASK 2A - Algorithm 2: LSTM Classification")
print("=" * 70)

df = pd.read_csv('cleaned_long_format.csv')
df['time'] = pd.to_datetime(df['time'], format='mixed')
df['date'] = df['time'].dt.date
df['date'] = pd.to_datetime(df['date'])

variables = df['variable'].unique()
print(f"Loaded cleaned data: {len(df)} rows, {df['id'].nunique()} users")

# --- Aggregate to daily wide-format ---
print("\nAggregating to daily wide-format...")
mean_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
count_vars = ['call', 'sms']
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

# Full date range per user
date_ranges = []
for uid in df['id'].unique():
    user = df[df['id'] == uid]
    dates = pd.date_range(user['date'].min(), user['date'].max(), freq='D')
    date_ranges.append(pd.DataFrame({'id': uid, 'date': dates}))
full_index = pd.concat(date_ranges, ignore_index=True)

daily = full_index.copy()
for r in results:
    daily = daily.merge(r, on=['id', 'date'], how='left')

daily['call'] = daily['call'].fillna(0)
daily['sms'] = daily['sms'].fillna(0)
for col in sum_vars:
    if col in daily.columns:
        daily[col] = daily[col].fillna(0)

# Track real mood days
daily['has_mood'] = daily['mood'].notna()

# Interpolate for sequence continuity
for col in mean_vars:
    daily[col] = daily.groupby('id')[col].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )

daily = daily.sort_values(['id', 'date']).reset_index(drop=True)

# Feature columns for LSTM input
input_cols = mean_vars + count_vars + sum_vars
input_cols = list(dict.fromkeys(input_cols))
n_features = len(input_cols)
print(f"Daily data: {daily.shape}, Input features per day: {n_features}")


# =============================================================================
# 2. DISCRETIZE TARGET
# =============================================================================
print("\n--- Discretizing target ---")

real_mood = daily.loc[daily['has_mood'], 'mood']
q33 = real_mood.quantile(0.33)
q67 = real_mood.quantile(0.67)
print(f"Quantile thresholds: q33={q33:.2f}, q67={q67:.2f}")

class_names = ['low', 'medium', 'high']

def mood_to_class(val):
    if val <= q33:
        return 0
    elif val <= q67:
        return 1
    else:
        return 2


# =============================================================================
# 3. BUILD SEQUENCES
# =============================================================================
print("\n--- Building sequences ---")

def build_sequences(daily_df, input_cols, window=7):
    sequences = []
    targets = []
    user_ids = []

    for uid in sorted(daily_df['id'].unique()):
        user = daily_df[daily_df['id'] == uid].sort_values('date').reset_index(drop=True)
        n = len(user)

        for i in range(window, n):
            if not user.loc[i, 'has_mood']:
                continue

            seq = user.loc[i - window:i - 1, input_cols].values
            target_mood = user.loc[i, 'mood']

            if np.isnan(seq).any() or np.isnan(target_mood):
                continue

            sequences.append(seq)
            targets.append(mood_to_class(target_mood))
            user_ids.append(uid)

    return np.array(sequences), np.array(targets), np.array(user_ids)


X_seq, y_seq, user_ids = build_sequences(daily, input_cols, WINDOW)
print(f"Total sequences: {len(X_seq)}")
print(f"Sequence shape: {X_seq.shape}  (n_samples, window={WINDOW}, n_features={n_features})")
print(f"Class distribution:")
for i, name in enumerate(class_names):
    n = (y_seq == i).sum()
    print(f"  {name}: {n} ({n/len(y_seq)*100:.1f}%)")


# =============================================================================
# 4. TRAIN / TEST SPLIT (by user)
# =============================================================================
print("\n--- Train / Test Split (by user) ---")

all_users = np.array(sorted(set(user_ids)))
np.random.shuffle(all_users)

n_test_users = max(5, len(all_users) // 5)
test_users = set(all_users[:n_test_users])
train_users = set(all_users[n_test_users:])

print(f"Total users: {len(all_users)}")
print(f"Train users ({len(train_users)}): {sorted(train_users)}")
print(f"Test users  ({len(test_users)}):  {sorted(test_users)}")

train_mask = np.array([uid in train_users for uid in user_ids])
test_mask = ~train_mask

X_train_raw, y_train = X_seq[train_mask], y_seq[train_mask]
X_test_raw, y_test = X_seq[test_mask], y_seq[test_mask]
train_user_ids = user_ids[train_mask]

print(f"\nTrain set: {len(X_train_raw)} sequences")
print(f"Test set:  {len(X_test_raw)} sequences")

print("\nTrain class distribution:")
for i, name in enumerate(class_names):
    n = (y_train == i).sum()
    print(f"  {name}: {n} ({n/len(y_train)*100:.1f}%)")
print("Test class distribution:")
for i, name in enumerate(class_names):
    n = (y_test == i).sum()
    print(f"  {name}: {n} ({n/len(y_test)*100:.1f}%)")


# =============================================================================
# 5. STANDARDIZE FEATURES (fit on train only)
# =============================================================================
scaler = StandardScaler()
X_train_flat = X_train_raw.reshape(-1, n_features)
scaler.fit(X_train_flat)

X_train = scaler.transform(X_train_flat).reshape(X_train_raw.shape)
X_test = scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape)
print(f"\nFeatures standardized (fit on train set only)")


# =============================================================================
# 6. DEFINE LSTM MODEL
# =============================================================================

class MoodLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# 7. HYPERPARAMETER SELECTION VIA VALIDATION
# =============================================================================
print("\n--- Hyperparameter Selection (validation on train set) ---")

# Use one fold of train users as validation to pick best epoch
val_users_arr = np.array(sorted(train_users))
np.random.shuffle(val_users_arr)
n_val = max(4, len(val_users_arr) // 5)
val_user_set = set(val_users_arr[:n_val])

val_mask = np.array([uid in val_user_set for uid in train_user_ids])
tr_mask = ~val_mask

X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
X_val, y_val = X_train[val_mask], y_train[val_mask]

print(f"Sub-train: {len(X_tr)}, Validation: {len(X_val)}")

# Hyperparameters
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 0.001
EPOCHS = 80
BATCH_SIZE = 32

print(f"\nLSTM Hyperparameters:")
print(f"  Hidden size: {HIDDEN_SIZE}, Layers: {NUM_LAYERS}, Dropout: {DROPOUT}")
print(f"  Learning rate: {LR}, Max epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")

# Train on sub-train, monitor validation F1 to find best epoch
train_dataset = SequenceDataset(X_tr, y_tr)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model_val = MoodLSTM(n_features, HIDDEN_SIZE, NUM_LAYERS, 3, DROPOUT)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_val.parameters(), lr=LR)

best_val_f1 = 0
best_epoch = EPOCHS
val_f1_history = []

for epoch in range(EPOCHS):
    model_val.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model_val(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Evaluate on validation
    model_val.eval()
    with torch.no_grad():
        val_out = model_val(torch.FloatTensor(X_val))
        _, val_preds = torch.max(val_out, 1)
        val_f1 = f1_score(y_val, val_preds.numpy(), average='macro')
        val_f1_history.append(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1

print(f"\nBest validation F1-macro: {best_val_f1:.4f} at epoch {best_epoch}")


# =============================================================================
# 8. RETRAIN ON FULL TRAIN SET WITH BEST EPOCH
# =============================================================================
print(f"\n--- Retraining on full train set for {best_epoch} epochs ---")

train_dataset_full = SequenceDataset(X_train, y_train)
train_loader_full = DataLoader(train_dataset_full, batch_size=BATCH_SIZE, shuffle=True)

model_final = MoodLSTM(n_features, HIDDEN_SIZE, NUM_LAYERS, 3, DROPOUT)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_final.parameters(), lr=LR)

model_final.train()
for epoch in range(best_epoch):
    for batch_X, batch_y in train_loader_full:
        optimizer.zero_grad()
        outputs = model_final(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

print("Training complete.")


# =============================================================================
# 9. FINAL EVALUATION ON TEST SET
# =============================================================================
print("\n--- Final Evaluation on Test Set ---")

model_final.eval()
with torch.no_grad():
    test_out = model_final(torch.FloatTensor(X_test))
    _, y_pred = torch.max(test_out, 1)
    y_pred = y_pred.numpy()

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\nTest Accuracy:    {acc:.4f}")
print(f"Test F1-macro:    {f1_macro:.4f}")
print(f"Test F1-weighted: {f1_weighted:.4f}")

print(f"\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred, target_names=class_names))


# =============================================================================
# 10. PLOTS
# =============================================================================

# ---- PLOT 1: Confusion Matrix ----
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues')
ax.set_title(f'LSTM - Test Set Confusion Matrix\nAccuracy={acc:.3f}, F1-macro={f1_macro:.3f}')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot_lstm_confusion_matrix.png')
plt.close()
print("\nSaved: plot_lstm_confusion_matrix.png")

# ---- PLOT 2: Validation F1 over epochs ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(val_f1_history) + 1), val_f1_history, 'b-', linewidth=1)
ax.axvline(best_epoch, color='red', linestyle='--', label=f'Best epoch={best_epoch}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation F1-macro')
ax.set_title('LSTM - Validation F1 During Training')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot_lstm_training_curve.png')
plt.close()
print("Saved: plot_lstm_training_curve.png")

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
plt.savefig(f'{OUTPUT_DIR}/plot_lstm_class_distribution.png')
plt.close()
print("Saved: plot_lstm_class_distribution.png")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("TASK 2A - LSTM SUMMARY")
print("=" * 70)
print(f"""
Algorithm: LSTM (Long Short-Term Memory) - Inherently Temporal
  - Operates on: raw daily time series (NOT feature-engineered dataset)
  - Input: sequences of {WINDOW} consecutive days x {n_features} features per day
  - The LSTM learns temporal patterns directly from the sequence
  - Class definition: 3 classes (low/medium/high) via quantile binning
    Thresholds: low <= {q33:.2f}, medium <= {q67:.2f}, high > {q67:.2f}

Architecture:
  - LSTM: {NUM_LAYERS} layers, hidden_size={HIDDEN_SIZE}, dropout={DROPOUT}
  - FC layers: {HIDDEN_SIZE} -> 32 -> 3 (with ReLU and dropout)

Data Split:
  - Train: {len(train_users)} users ({len(X_train)} sequences)
  - Test:  {len(test_users)} users ({len(X_test)} sequences)
  - Split by user to prevent data leakage
  - Scaler fit on train set only

Hyperparameter Selection:
  - Validation split from train users to find best epoch
  - Best epoch: {best_epoch} (val F1-macro: {best_val_f1:.4f})

Test Set Results:
  - Accuracy:    {acc:.4f}
  - F1-macro:    {f1_macro:.4f}
  - F1-weighted: {f1_weighted:.4f}
""")