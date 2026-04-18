from __future__ import annotations

import json
from pathlib import Path

from .config import PipelineConfig
from .data import apply_imputation, backtest_imputation_strategies, build_daily_patient_table_with_audit
from .deps import MissingDependencyError, require_dependency
from .evaluation import classification_metrics, confusion_matrix, regression_metrics
from .features import SequenceWindowDataset, TabularWindowDataset, build_window_datasets
from .models import GRUWrapper, PersistenceBaseline, RandomForestWrapper
from .splits import assign_outer_splits, build_expanding_window_folds
from .utils import apply_thresholds, quantile_thresholds


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    daily_raw, raw_event_audit = build_daily_patient_table_with_audit(config.csv_path)
    imputation_backtest = backtest_imputation_strategies(daily_raw, config=config)
    daily_clean = apply_imputation(daily_raw, config=config, method=config.selected_imputation_method)

    tabular_dataset, sequence_dataset = build_window_datasets(daily_clean, window_size=config.window_size)
    tabular_dataset, sequence_dataset = _attach_outer_splits(tabular_dataset, sequence_dataset, config=config)
    thresholds = quantile_thresholds(
        tabular_dataset.frame.loc[tabular_dataset.frame["split"] == "train", tabular_dataset.target_column].tolist()
    )
    _attach_classification_targets(tabular_dataset, sequence_dataset, thresholds)

    results = []
    results.append(_run_persistence_baseline(tabular_dataset, thresholds, config))
    results.append(_run_random_forest(tabular_dataset, task="classification", config=config))
    results.append(_run_random_forest(tabular_dataset, task="regression", config=config))
    results.append(_run_gru(sequence_dataset, task="classification", config=config))
    results.append(_run_gru(sequence_dataset, task="regression", config=config))

    if config.export_intermediate:
        _export_intermediate_artifacts(
            daily_clean=daily_clean,
            tabular_dataset=tabular_dataset,
            sequence_dataset=sequence_dataset,
            output_dir=config.output_dir,
        )

    if config.generate_plots:
        for result in results:
            _write_optional_diagnostics(config.output_dir, result)

    summary = {
        "config": _to_builtin(config.__dict__),
        "data_quality": raw_event_audit,
        "classification_thresholds": {"lower": thresholds[0], "upper": thresholds[1]},
        "imputation_backtest": imputation_backtest,
        "results": results,
    }
    with (config.output_dir / "imputation_backtest.json").open("w", encoding="utf-8") as handle:
        json.dump(imputation_backtest, handle, indent=2, default=_json_default)
    with (config.output_dir / "experiment_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=_json_default)
    with (config.output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    return summary


def _attach_outer_splits(
    tabular_dataset: TabularWindowDataset,
    sequence_dataset: SequenceWindowDataset,
    config: PipelineConfig,
) -> tuple[TabularWindowDataset, SequenceWindowDataset]:
    split_metadata = assign_outer_splits(
        tabular_dataset.frame[["sample_id", "patient_id", "anchor_date"]].copy(),
        patient_col="patient_id",
        date_col="anchor_date",
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )[["sample_id", "split"]]
    tabular_dataset.frame = tabular_dataset.frame.merge(split_metadata, on="sample_id", how="left")
    sequence_dataset.metadata = sequence_dataset.metadata.merge(split_metadata, on="sample_id", how="left")
    return tabular_dataset, sequence_dataset


def _attach_classification_targets(
    tabular_dataset: TabularWindowDataset,
    sequence_dataset: SequenceWindowDataset,
    thresholds: tuple[float, float],
) -> None:
    tabular_dataset.frame["target_class"] = tabular_dataset.frame[tabular_dataset.target_column].apply(
        lambda value: apply_thresholds(float(value), thresholds)
    )
    sequence_dataset.metadata["target_class"] = sequence_dataset.metadata[sequence_dataset.target_column].apply(
        lambda value: apply_thresholds(float(value), thresholds)
    )


def _run_persistence_baseline(tabular_dataset: TabularWindowDataset, thresholds, config: PipelineConfig) -> dict[str, object]:
    train_frame = tabular_dataset.frame.loc[tabular_dataset.frame["split"] == "train"].reset_index(drop=True)
    val_frame = tabular_dataset.frame.loc[tabular_dataset.frame["split"] == "val"].reset_index(drop=True)
    test_frame = tabular_dataset.frame.loc[tabular_dataset.frame["split"] == "test"].reset_index(drop=True)

    baseline = PersistenceBaseline().fit(train_frame)
    val_reg_predictions = baseline.predict(val_frame)
    test_reg_predictions = baseline.predict(test_frame)
    val_cls_predictions = [apply_thresholds(float(value), thresholds) for value in val_reg_predictions]
    test_cls_predictions = [apply_thresholds(float(value), thresholds) for value in test_reg_predictions]

    return {
        "model_family": "persistence_baseline",
        "task": "shared",
        "device": "cpu",
        "best_params": {},
        "n_train": len(train_frame),
        "n_val": len(val_frame),
        "n_test": len(test_frame),
        "split_metrics": {
            "val_regression": regression_metrics(
                val_frame[tabular_dataset.target_column].astype(float).tolist(),
                [float(value) for value in val_reg_predictions],
            ),
            "test_regression": regression_metrics(
                test_frame[tabular_dataset.target_column].astype(float).tolist(),
                [float(value) for value in test_reg_predictions],
            ),
            "val_classification": classification_metrics(
                val_frame["target_class"].astype(int).tolist(),
                [int(value) for value in val_cls_predictions],
                labels=[0, 1, 2],
            ),
            "test_classification": classification_metrics(
                test_frame["target_class"].astype(int).tolist(),
                [int(value) for value in test_cls_predictions],
                labels=[0, 1, 2],
            ),
        },
        "diagnostics": {
            "test_regression_predictions": [float(value) for value in test_reg_predictions],
            "test_regression_truth": test_frame[tabular_dataset.target_column].astype(float).tolist(),
            "test_classification_predictions": [int(value) for value in test_cls_predictions],
            "test_classification_truth": test_frame["target_class"].astype(int).tolist(),
        },
    }


def _run_random_forest(tabular_dataset: TabularWindowDataset, task: str, config: PipelineConfig) -> dict[str, object]:
    train_frame = tabular_dataset.frame.loc[tabular_dataset.frame["split"] == "train"].reset_index(drop=True)
    val_frame = tabular_dataset.frame.loc[tabular_dataset.frame["split"] == "val"].reset_index(drop=True)
    test_frame = tabular_dataset.frame.loc[tabular_dataset.frame["split"] == "test"].reset_index(drop=True)

    target_column = "target_class" if task == "classification" else tabular_dataset.target_column
    param_grid = config.rf_classifier_grid if task == "classification" else config.rf_regressor_grid
    folds = build_expanding_window_folds(train_frame[["patient_id"]].copy(), patient_col="patient_id", n_folds=config.inner_cv_folds)

    best_params = None
    best_score = float("-inf") if task == "classification" else float("inf")
    for params in param_grid:
        fold_scores = []
        for fold_train_idx, fold_val_idx in folds:
            fold_train = train_frame.loc[fold_train_idx].reset_index(drop=True)
            fold_val = train_frame.loc[fold_val_idx].reset_index(drop=True)
            model = RandomForestWrapper(task=task, params=params, random_state=config.random_state)
            model.fit(fold_train, tabular_dataset.feature_columns, fold_train[target_column].tolist())
            predictions = model.predict(fold_val)
            if task == "classification":
                metrics = classification_metrics(
                    fold_val[target_column].astype(int).tolist(),
                    [int(value) for value in predictions],
                    labels=[0, 1, 2],
                )
                fold_scores.append(float(metrics["macro_f1"]))
            else:
                metrics = regression_metrics(
                    fold_val[target_column].astype(float).tolist(),
                    [float(value) for value in predictions],
                )
                fold_scores.append(float(metrics["mae"]))

        average_score = sum(fold_scores) / len(fold_scores)
        if task == "classification":
            if average_score > best_score:
                best_score = average_score
                best_params = params
        else:
            if average_score < best_score:
                best_score = average_score
                best_params = params

    final_model = RandomForestWrapper(task=task, params=best_params or param_grid[0], random_state=config.random_state)
    final_model.fit(train_frame, tabular_dataset.feature_columns, train_frame[target_column].tolist())
    val_predictions = final_model.predict(val_frame)
    test_predictions = final_model.predict(test_frame)

    if task == "classification":
        val_metrics = classification_metrics(
            val_frame[target_column].astype(int).tolist(),
            [int(value) for value in val_predictions],
            labels=[0, 1, 2],
        )
        test_metrics = classification_metrics(
            test_frame[target_column].astype(int).tolist(),
            [int(value) for value in test_predictions],
            labels=[0, 1, 2],
        )
    else:
        val_metrics = regression_metrics(
            val_frame[target_column].astype(float).tolist(),
            [float(value) for value in val_predictions],
        )
        test_metrics = regression_metrics(
            test_frame[target_column].astype(float).tolist(),
            [float(value) for value in test_predictions],
        )

    return {
        "model_family": "random_forest",
        "task": task,
        "device": "cpu",
        "best_params": best_params or param_grid[0],
        "inner_cv_score": best_score,
        "n_train": len(train_frame),
        "n_val": len(val_frame),
        "n_test": len(test_frame),
        "split_metrics": {"val": val_metrics, "test": test_metrics},
        "diagnostics": {
            "test_predictions": [int(value) if task == "classification" else float(value) for value in test_predictions],
            "test_truth": test_frame[target_column].astype(int if task == "classification" else float).tolist(),
        },
    }


def _run_gru(sequence_dataset: SequenceWindowDataset, task: str, config: PipelineConfig) -> dict[str, object]:
    train_mask = sequence_dataset.metadata["split"] == "train"
    val_mask = sequence_dataset.metadata["split"] == "val"
    test_mask = sequence_dataset.metadata["split"] == "test"

    train_metadata = sequence_dataset.metadata.loc[train_mask].reset_index(drop=True)
    val_metadata = sequence_dataset.metadata.loc[val_mask].reset_index(drop=True)
    test_metadata = sequence_dataset.metadata.loc[test_mask].reset_index(drop=True)

    train_sequences = sequence_dataset.sequences[train_mask.to_numpy()]
    val_sequences = sequence_dataset.sequences[val_mask.to_numpy()]
    test_sequences = sequence_dataset.sequences[test_mask.to_numpy()]

    target_column = "target_class" if task == "classification" else sequence_dataset.target_column
    folds = build_expanding_window_folds(train_metadata[["patient_id"]].copy(), patient_col="patient_id", n_folds=config.inner_cv_folds)

    best_params = None
    best_score = float("-inf") if task == "classification" else float("inf")

    for params in config.gru_grid:
        fold_scores = []
        for fold_train_idx, fold_val_idx in folds:
            fold_train_sequences = train_sequences[fold_train_idx]
            fold_val_sequences = train_sequences[fold_val_idx]
            fold_train_metadata = train_metadata.loc[fold_train_idx].reset_index(drop=True)
            fold_val_metadata = train_metadata.loc[fold_val_idx].reset_index(drop=True)
            model = GRUWrapper(
                task=task,
                params=params,
                cell_type=config.sequence_cell_type,
                hidden_size=config.gru_hidden_size,
                dropout=config.gru_dropout,
                random_state=config.random_state,
                patience=config.gru_patience,
            )
            model.fit(
                fold_train_sequences,
                fold_train_metadata[target_column].tolist(),
                val_sequences=fold_val_sequences,
                val_y=fold_val_metadata[target_column].tolist(),
            )
            predictions = model.predict(fold_val_sequences)
            if task == "classification":
                metrics = classification_metrics(
                    fold_val_metadata[target_column].astype(int).tolist(),
                    [int(value) for value in predictions],
                    labels=[0, 1, 2],
                )
                fold_scores.append(float(metrics["macro_f1"]))
            else:
                metrics = regression_metrics(
                    fold_val_metadata[target_column].astype(float).tolist(),
                    [float(value) for value in predictions],
                )
                fold_scores.append(float(metrics["mae"]))

        average_score = sum(fold_scores) / len(fold_scores)
        if task == "classification":
            if average_score > best_score:
                best_score = average_score
                best_params = params
        else:
            if average_score < best_score:
                best_score = average_score
                best_params = params

    epoch_train_idx, epoch_val_idx = folds[-1]
    epoch_train_sequences = train_sequences[epoch_train_idx]
    epoch_val_sequences = train_sequences[epoch_val_idx]
    epoch_train_metadata = train_metadata.loc[epoch_train_idx].reset_index(drop=True)
    epoch_val_metadata = train_metadata.loc[epoch_val_idx].reset_index(drop=True)
    epoch_selector = GRUWrapper(
        task=task,
        params=best_params or config.gru_grid[0],
        cell_type=config.sequence_cell_type,
        hidden_size=config.gru_hidden_size,
        dropout=config.gru_dropout,
        random_state=config.random_state,
        patience=config.gru_patience,
    )
    epoch_selector.fit(
        epoch_train_sequences,
        epoch_train_metadata[target_column].tolist(),
        val_sequences=epoch_val_sequences,
        val_y=epoch_val_metadata[target_column].tolist(),
    )
    selected_epoch = epoch_selector.best_epoch or int((best_params or config.gru_grid[0]).get("epochs", 25))
    final_params = dict(best_params or config.gru_grid[0])
    final_params["epochs"] = selected_epoch

    final_model = GRUWrapper(
        task=task,
        params=final_params,
        cell_type=config.sequence_cell_type,
        hidden_size=config.gru_hidden_size,
        dropout=config.gru_dropout,
        random_state=config.random_state,
        patience=config.gru_patience,
    )
    final_model.fit(
        train_sequences,
        train_metadata[target_column].tolist(),
    )
    val_predictions = final_model.predict(val_sequences)
    test_predictions = final_model.predict(test_sequences)

    if task == "classification":
        val_metrics = classification_metrics(
            val_metadata[target_column].astype(int).tolist(),
            [int(value) for value in val_predictions],
            labels=[0, 1, 2],
        )
        test_metrics = classification_metrics(
            test_metadata[target_column].astype(int).tolist(),
            [int(value) for value in test_predictions],
            labels=[0, 1, 2],
        )
    else:
        val_metrics = regression_metrics(
            val_metadata[target_column].astype(float).tolist(),
            [float(value) for value in val_predictions],
        )
        test_metrics = regression_metrics(
            test_metadata[target_column].astype(float).tolist(),
            [float(value) for value in test_predictions],
        )

    return {
        "model_family": final_model.cell_type,
        "task": task,
        "device": final_model.device,
        "best_params": best_params or config.gru_grid[0],
        "final_training_epochs": selected_epoch,
        "epoch_selection_metric": "macro_f1" if task == "classification" else "mae",
        "epoch_selection_split": {"n_train": len(epoch_train_metadata), "n_val": len(epoch_val_metadata)},
        "inner_cv_score": best_score,
        "n_train": len(train_metadata),
        "n_val": len(val_metadata),
        "n_test": len(test_metadata),
        "split_metrics": {"val": val_metrics, "test": test_metrics},
        "diagnostics": {
            "test_predictions": [int(value) if task == "classification" else float(value) for value in test_predictions],
            "test_truth": test_metadata[target_column].astype(int if task == "classification" else float).tolist(),
        },
    }


def _export_intermediate_artifacts(
    daily_clean,
    tabular_dataset: TabularWindowDataset,
    sequence_dataset: SequenceWindowDataset,
    output_dir: Path,
) -> None:
    np = require_dependency("numpy")
    daily_clean.to_csv(output_dir / "daily_patient_table.csv", index=False)
    tabular_dataset.frame.to_csv(output_dir / "tabular_window_dataset.csv", index=False)
    sequence_dataset.metadata.to_csv(output_dir / "sequence_window_metadata.csv", index=False)
    np.savez_compressed(
        output_dir / "sequence_window_dataset.npz",
        sequences=sequence_dataset.sequences,
        feature_names=np.asarray(sequence_dataset.feature_names, dtype=object),
    )


def _write_optional_diagnostics(output_dir: Path, result: dict[str, object]) -> None:
    try:
        plt = require_dependency("matplotlib.pyplot", "matplotlib")
    except MissingDependencyError:
        return

    diagnostics = result.get("diagnostics", {})
    task = result["task"]
    model_family = result["model_family"]
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if task == "classification" or task == "shared":
        truth = diagnostics.get("test_classification_truth") or diagnostics.get("test_truth")
        preds = diagnostics.get("test_classification_predictions") or diagnostics.get("test_predictions")
        if truth and preds:
            matrix = confusion_matrix(list(truth), list(preds), labels=[0, 1, 2])
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(matrix)
            ax.set_title(f"{model_family} classification")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for row_idx, row in enumerate(matrix):
                for col_idx, value in enumerate(row):
                    ax.text(col_idx, row_idx, str(value), ha="center", va="center")
            fig.tight_layout()
            fig.savefig(plots_dir / f"{model_family}_classification_confusion_matrix.png")
            plt.close(fig)

    if task == "regression" or task == "shared":
        truth = diagnostics.get("test_regression_truth") or diagnostics.get("test_truth")
        preds = diagnostics.get("test_regression_predictions") or diagnostics.get("test_predictions")
        if truth and preds:
            residuals = [pred - actual for actual, pred in zip(truth, preds)]
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(residuals, bins=20)
            ax.set_title(f"{model_family} regression residuals")
            ax.set_xlabel("Prediction - Truth")
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(plots_dir / f"{model_family}_regression_residuals.png")
            plt.close(fig)


def _json_default(value):
    return _to_builtin(value)


def _to_builtin(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
