from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    csv_path: Path
    output_dir: Path
    window_size: int = 7
    selected_imputation_method: str = "forward_fill"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    inner_cv_folds: int = 3
    random_state: int = 42
    export_intermediate: bool = True
    generate_plots: bool = True
    sequence_cell_type: str = "gru"
    max_ffill_gap: int = 2
    ewma_alpha: float = 0.6
    ewma_history: int = 3
    rf_classifier_grid: list[dict] = field(
        default_factory=lambda: [
            {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 2},
            {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 2},
            {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
        ]
    )
    rf_regressor_grid: list[dict] = field(
        default_factory=lambda: [
            {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 2},
            {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 2},
            {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
        ]
    )
    gru_grid: list[dict] = field(
        default_factory=lambda: [
            {"learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 25, "batch_size": 32},
            {"learning_rate": 3e-3, "weight_decay": 1e-4, "epochs": 25, "batch_size": 32},
        ]
    )
    gru_hidden_size: int = 32
    gru_dropout: float = 0.2
    gru_patience: int = 5
    classification_labels: tuple[str, str, str] = ("low", "mid", "high")
