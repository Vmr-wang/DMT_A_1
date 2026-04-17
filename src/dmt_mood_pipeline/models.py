from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional

from .deps import require_dependency
from .evaluation import classification_metrics, regression_metrics


@dataclass
class TabularFeaturePreprocessor:
    base_feature_columns: list[str]
    patient_baseline_by_id: dict[str, float] = field(default_factory=dict)
    global_baseline: float = 0.0
    feature_columns_: list[str] = field(default_factory=list)
    feature_medians_: dict[str, float] = field(default_factory=dict)

    def fit(self, frame):
        baseline_source = frame["baseline_mood_prediction"].astype(float)
        baseline_mean = baseline_source.mean(skipna=True)
        self.global_baseline = 0.0 if baseline_mean != baseline_mean else float(baseline_mean)

        patient_baseline = (
            frame.groupby("patient_id")["baseline_mood_prediction"]
            .mean()
            .fillna(self.global_baseline)
        )
        self.patient_baseline_by_id = {str(key): float(value) for key, value in patient_baseline.to_dict().items()}

        augmented = self._augment(frame)
        self.feature_columns_ = [
            *self.base_feature_columns,
            "patient_baseline_mood",
            "mood_mean_clean_last_minus_patient_baseline",
            "mood_mean_clean_3d_mean_minus_patient_baseline",
            "mood_mean_clean_7d_mean_minus_patient_baseline",
        ]
        feature_frame = augmented[self.feature_columns_].copy()
        medians = feature_frame.median(numeric_only=True)
        self.feature_medians_ = {column: float(medians[column]) for column in self.feature_columns_}
        return self

    def fit_transform(self, frame):
        self.fit(frame)
        return self.transform(frame)

    def transform(self, frame):
        np = require_dependency("numpy")
        augmented = self._augment(frame)
        feature_frame = augmented[self.feature_columns_].copy()
        for column, median in self.feature_medians_.items():
            feature_frame[column] = feature_frame[column].fillna(median)
        return feature_frame.to_numpy(dtype=float)

    def _augment(self, frame):
        augmented = frame.copy()
        baseline = augmented["patient_id"].astype(str).map(self.patient_baseline_by_id).fillna(self.global_baseline)
        augmented["patient_baseline_mood"] = baseline
        augmented["mood_mean_clean_last_minus_patient_baseline"] = augmented["mood_mean_clean_last"] - baseline
        augmented["mood_mean_clean_3d_mean_minus_patient_baseline"] = augmented["mood_mean_clean_3d_mean"] - baseline
        augmented["mood_mean_clean_7d_mean_minus_patient_baseline"] = augmented["mood_mean_clean_7d_mean"] - baseline
        return augmented


@dataclass
class SequenceFeaturePreprocessor:
    mean_: Optional[object] = None
    std_: Optional[object] = None

    def fit(self, sequences):
        np = require_dependency("numpy")
        array = np.asarray(sequences, dtype=float)
        mean = np.nanmean(array, axis=(0, 1))
        std = np.nanstd(array, axis=(0, 1))
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
        std[std == 0.0] = 1.0
        self.mean_ = mean
        self.std_ = std
        return self

    def fit_transform(self, sequences):
        self.fit(sequences)
        return self.transform(sequences)

    def transform(self, sequences):
        np = require_dependency("numpy")
        array = np.asarray(sequences, dtype=float)
        normalized = (array - self.mean_) / self.std_
        normalized = np.nan_to_num(normalized, nan=0.0)
        return normalized.astype(float)


@dataclass
class PersistenceBaseline:
    fallback_value: float = 0.0

    def fit(self, frame):
        baseline_source = frame["baseline_mood_prediction"].astype(float)
        baseline_mean = baseline_source.mean(skipna=True)
        if baseline_mean == baseline_mean:
            self.fallback_value = float(baseline_mean)
        else:
            self.fallback_value = float(frame["target_next_day_mood_mean"].mean())
        return self

    def predict(self, frame) -> list[float]:
        predictions = frame["baseline_mood_prediction"].fillna(self.fallback_value).astype(float)
        return predictions.tolist()


@dataclass
class RandomForestWrapper:
    task: str
    params: dict[str, object]
    random_state: int = 42
    estimator: Optional[object] = None
    preprocessor: Optional[TabularFeaturePreprocessor] = None

    def fit(self, frame, feature_columns: list[str], y):
        np = require_dependency("numpy")
        sklearn_ensemble = require_dependency("sklearn.ensemble", "scikit-learn")
        self.preprocessor = TabularFeaturePreprocessor(feature_columns)
        X = self.preprocessor.fit_transform(frame)
        y_array = np.asarray(y)
        if self.task == "classification":
            self.estimator = sklearn_ensemble.RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
                **self.params,
            )
        else:
            self.estimator = sklearn_ensemble.RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                **self.params,
            )
        self.estimator.fit(X, y_array)
        return self

    def predict(self, frame) -> List[object]:
        X = self.preprocessor.transform(frame)
        predictions = self.estimator.predict(X)
        return predictions.tolist()


@dataclass
class GRUWrapper:
    task: str
    params: dict[str, object]
    cell_type: str = "gru"
    hidden_size: int = 32
    dropout: float = 0.2
    random_state: int = 42
    model: Optional[object] = None
    preprocessor: Optional[SequenceFeaturePreprocessor] = None
    num_classes: int = 3
    patience: int = 5
    device: str = "cpu"
    best_epoch: int = 0

    def __post_init__(self):
        normalized_cell_type = str(self.cell_type).lower()
        if normalized_cell_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported recurrent cell type: {self.cell_type}")
        self.cell_type = normalized_cell_type

    def fit(self, sequences, y, val_sequences=None, val_y=None):
        np = require_dependency("numpy")
        torch = require_dependency("torch")

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.device = _select_torch_device(torch)
        device = torch.device(self.device)

        train_sequences = np.asarray(sequences, dtype=float)
        self.preprocessor = SequenceFeaturePreprocessor()
        X_train = self.preprocessor.fit_transform(train_sequences)
        y_train = np.asarray(y)

        X_val = None
        y_val_array = None
        if val_sequences is not None and len(val_sequences):
            X_val = self.preprocessor.transform(val_sequences)
            y_val_array = np.asarray(val_y)

        input_size = X_train.shape[-1]
        output_size = self.num_classes if self.task == "classification" else 1

        class _SequenceNet(torch.nn.Module):
            def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float, cell_type: str):
                super().__init__()
                if cell_type == "gru":
                    self.recurrent = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                else:
                    self.recurrent = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                self.dropout = torch.nn.Dropout(dropout)
                self.output = torch.nn.Linear(hidden_size, output_size)

            def forward(self, batch):
                _, state = self.recurrent(batch)
                hidden = state[0] if isinstance(state, tuple) else state
                last_hidden = hidden[-1]
                return self.output(self.dropout(last_hidden))

        model = _SequenceNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            output_size=output_size,
            dropout=self.dropout,
            cell_type=self.cell_type,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params.get("weight_decay", 0.0)),
        )
        loss_fn = torch.nn.CrossEntropyLoss() if self.task == "classification" else torch.nn.MSELoss()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        if self.task == "classification":
            y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
        else:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device) if X_val is not None else None
        if y_val_array is not None:
            if self.task == "classification":
                y_val_tensor = torch.tensor(y_val_array, dtype=torch.long, device=device)
            else:
                y_val_tensor = torch.tensor(y_val_array, dtype=torch.float32, device=device).unsqueeze(-1)
        else:
            y_val_tensor = None

        best_state = copy.deepcopy(model.state_dict())
        best_score = float("-inf") if self.task == "classification" else float("inf")
        best_epoch = 0
        patience = 0
        batch_size = int(self.params.get("batch_size", 32))
        num_epochs = int(self.params.get("epochs", 25))

        for epoch_idx in range(num_epochs):
            model.train()
            for batch_indices in _iter_minibatches(len(X_train), batch_size=batch_size, random_state=self.random_state):
                optimizer.zero_grad()
                batch_index_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)
                batch_x = X_train_tensor.index_select(0, batch_index_tensor)
                batch_y = y_train_tensor.index_select(0, batch_index_tensor)
                predictions = model(batch_x)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()

            if X_val_tensor is None or y_val_tensor is None:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch_idx + 1
                continue

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
            if self.task == "classification":
                val_predictions = val_outputs.argmax(dim=1).cpu().tolist()
                val_metrics = classification_metrics(y_val_array.tolist(), val_predictions, labels=list(range(self.num_classes)))
                current_score = float(val_metrics["macro_f1"])
                improved = current_score > best_score
            else:
                val_predictions = val_outputs.squeeze(-1).cpu().tolist()
                val_metrics = regression_metrics(y_val_array.tolist(), val_predictions)
                current_score = float(val_metrics["mae"])
                improved = current_score < best_score

            if improved:
                best_score = current_score
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch_idx + 1
                patience = 0
            else:
                patience += 1
                if patience >= self.patience:
                    break

        model.load_state_dict(best_state)
        self.model = model
        self.best_epoch = best_epoch or num_epochs
        return self

    def predict(self, sequences) -> List[object]:
        np = require_dependency("numpy")
        torch = require_dependency("torch")
        transformed = self.preprocessor.transform(np.asarray(sequences, dtype=float))
        device = torch.device(self.device)
        with torch.no_grad():
            inputs = torch.tensor(transformed, dtype=torch.float32, device=device)
            outputs = self.model(inputs)
        if self.task == "classification":
            return outputs.argmax(dim=1).cpu().tolist()
        return outputs.squeeze(-1).cpu().tolist()


def _iter_minibatches(num_items: int, batch_size: int, random_state: int):
    indices = list(range(num_items))
    rng = random.Random(random_state)
    rng.shuffle(indices)
    for start in range(0, num_items, batch_size):
        yield indices[start : start + batch_size]


def _select_torch_device(torch) -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
