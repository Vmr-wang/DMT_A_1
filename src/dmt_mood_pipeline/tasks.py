from __future__ import annotations

import json

from .config import PipelineConfig
from .data import apply_imputation, backtest_imputation_strategies, build_daily_patient_table_with_audit
from .eda import run_eda
from .evaluation import classification_metrics, regression_metrics
from .features import build_window_datasets
from .models import PersistenceBaseline
from .pipeline import (
    _attach_classification_targets,
    _attach_outer_splits,
    _export_intermediate_artifacts,
    _json_default,
    _run_gru,
    _run_random_forest,
    _to_builtin,
    _write_optional_diagnostics,
)
from .utils import apply_thresholds, quantile_thresholds


def run_task_1a(config: PipelineConfig) -> dict[str, object]:
    return run_eda(csv_path=config.csv_path, output_dir=config.output_dir)


def run_task_1b(config: PipelineConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    daily_raw, raw_event_audit = build_daily_patient_table_with_audit(config.csv_path)
    imputation_backtest = backtest_imputation_strategies(daily_raw, config=config)
    daily_clean = apply_imputation(daily_raw, config=config, method=config.selected_imputation_method)

    daily_clean.to_csv(config.output_dir / "daily_patient_table.csv", index=False)
    _write_json(config.output_dir / "imputation_backtest.json", imputation_backtest)
    summary = {
        "task": "task_1b",
        "config": _to_builtin(config.__dict__),
        "data_quality": raw_event_audit,
        "imputation_backtest": imputation_backtest,
        "generated_files": ["daily_patient_table.csv", "imputation_backtest.json"],
    }
    _write_json(config.output_dir / "run_summary.json", summary)
    return summary


def run_task_1c(config: PipelineConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    daily_raw, raw_event_audit = build_daily_patient_table_with_audit(config.csv_path)
    daily_clean = apply_imputation(daily_raw, config=config, method=config.selected_imputation_method)
    tabular_dataset, sequence_dataset = build_window_datasets(daily_clean, window_size=config.window_size)

    _export_intermediate_artifacts(
        daily_clean=daily_clean,
        tabular_dataset=tabular_dataset,
        sequence_dataset=sequence_dataset,
        output_dir=config.output_dir,
    )
    summary = {
        "task": "task_1c",
        "config": _to_builtin(config.__dict__),
        "data_quality": raw_event_audit,
        "generated_files": [
            "daily_patient_table.csv",
            "tabular_window_dataset.csv",
            "sequence_window_metadata.csv",
            "sequence_window_dataset.npz",
        ],
    }
    _write_json(config.output_dir / "run_summary.json", summary)
    return summary


def run_task_2a_rf(config: PipelineConfig) -> dict[str, object]:
    _, tabular_dataset, _, raw_event_audit, thresholds = _prepare_modeling_datasets(config, attach_classification_targets=True)
    results = [
        _run_persistence_baseline_for_classification(tabular_dataset, thresholds),
        _run_random_forest(tabular_dataset, task="classification", config=config),
    ]
    return _finalize_modeling_task(
        output_dir=config.output_dir,
        task_name="task_2a_rf",
        config=config,
        raw_event_audit=raw_event_audit,
        results=results,
        classification_thresholds=thresholds,
    )


def run_task_2a_sequence(config: PipelineConfig) -> dict[str, object]:
    _, tabular_dataset, sequence_dataset, raw_event_audit, thresholds = _prepare_modeling_datasets(
        config,
        attach_classification_targets=True,
    )
    results = [
        _run_persistence_baseline_for_classification(tabular_dataset, thresholds),
        _run_gru(sequence_dataset, task="classification", config=config),
    ]
    return _finalize_modeling_task(
        output_dir=config.output_dir,
        task_name=f"task_2a_{config.sequence_cell_type}",
        config=config,
        raw_event_audit=raw_event_audit,
        results=results,
        classification_thresholds=thresholds,
    )


def run_task_4_rf(config: PipelineConfig) -> dict[str, object]:
    _, tabular_dataset, _, raw_event_audit, _ = _prepare_modeling_datasets(config, attach_classification_targets=False)
    results = [
        _run_persistence_baseline_for_regression(tabular_dataset),
        _run_random_forest(tabular_dataset, task="regression", config=config),
    ]
    return _finalize_modeling_task(
        output_dir=config.output_dir,
        task_name="task_4_rf",
        config=config,
        raw_event_audit=raw_event_audit,
        results=results,
    )


def run_task_4_sequence(config: PipelineConfig) -> dict[str, object]:
    _, tabular_dataset, sequence_dataset, raw_event_audit, _ = _prepare_modeling_datasets(
        config,
        attach_classification_targets=False,
    )
    results = [
        _run_persistence_baseline_for_regression(tabular_dataset),
        _run_gru(sequence_dataset, task="regression", config=config),
    ]
    return _finalize_modeling_task(
        output_dir=config.output_dir,
        task_name=f"task_4_{config.sequence_cell_type}",
        config=config,
        raw_event_audit=raw_event_audit,
        results=results,
    )


def _prepare_modeling_datasets(
    config: PipelineConfig,
    *,
    attach_classification_targets: bool,
):
    daily_raw, raw_event_audit = build_daily_patient_table_with_audit(config.csv_path)
    daily_clean = apply_imputation(daily_raw, config=config, method=config.selected_imputation_method)
    tabular_dataset, sequence_dataset = build_window_datasets(daily_clean, window_size=config.window_size)
    tabular_dataset, sequence_dataset = _attach_outer_splits(tabular_dataset, sequence_dataset, config=config)

    thresholds = None
    if attach_classification_targets:
        thresholds = quantile_thresholds(
            tabular_dataset.frame.loc[tabular_dataset.frame["split"] == "train", tabular_dataset.target_column].tolist()
        )
        _attach_classification_targets(tabular_dataset, sequence_dataset, thresholds)
    return daily_clean, tabular_dataset, sequence_dataset, raw_event_audit, thresholds


def _run_persistence_baseline_for_classification(tabular_dataset, thresholds) -> dict[str, object]:
    train_frame, val_frame, test_frame = _split_tabular_frame(tabular_dataset.frame)
    baseline = PersistenceBaseline().fit(train_frame)
    val_predictions = baseline.predict(val_frame)
    test_predictions = baseline.predict(test_frame)
    val_classes = [apply_thresholds(float(value), thresholds) for value in val_predictions]
    test_classes = [apply_thresholds(float(value), thresholds) for value in test_predictions]

    return {
        "model_family": "persistence_baseline",
        "task": "classification",
        "device": "cpu",
        "best_params": {},
        "n_train": len(train_frame),
        "n_val": len(val_frame),
        "n_test": len(test_frame),
        "split_metrics": {
            "val": classification_metrics(
                val_frame["target_class"].astype(int).tolist(),
                [int(value) for value in val_classes],
                labels=[0, 1, 2],
            ),
            "test": classification_metrics(
                test_frame["target_class"].astype(int).tolist(),
                [int(value) for value in test_classes],
                labels=[0, 1, 2],
            ),
        },
        "diagnostics": {
            "test_predictions": [int(value) for value in test_classes],
            "test_truth": test_frame["target_class"].astype(int).tolist(),
        },
    }


def _run_persistence_baseline_for_regression(tabular_dataset) -> dict[str, object]:
    train_frame, val_frame, test_frame = _split_tabular_frame(tabular_dataset.frame)
    baseline = PersistenceBaseline().fit(train_frame)
    val_predictions = baseline.predict(val_frame)
    test_predictions = baseline.predict(test_frame)

    return {
        "model_family": "persistence_baseline",
        "task": "regression",
        "device": "cpu",
        "best_params": {},
        "n_train": len(train_frame),
        "n_val": len(val_frame),
        "n_test": len(test_frame),
        "split_metrics": {
            "val": regression_metrics(
                val_frame[tabular_dataset.target_column].astype(float).tolist(),
                [float(value) for value in val_predictions],
            ),
            "test": regression_metrics(
                test_frame[tabular_dataset.target_column].astype(float).tolist(),
                [float(value) for value in test_predictions],
            ),
        },
        "diagnostics": {
            "test_predictions": [float(value) for value in test_predictions],
            "test_truth": test_frame[tabular_dataset.target_column].astype(float).tolist(),
        },
    }


def _split_tabular_frame(frame):
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    val_frame = frame.loc[frame["split"] == "val"].reset_index(drop=True)
    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    return train_frame, val_frame, test_frame


def _finalize_modeling_task(
    *,
    output_dir,
    task_name: str,
    config: PipelineConfig,
    raw_event_audit: dict[str, object],
    results: list[dict[str, object]],
    classification_thresholds=None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.generate_plots:
        for result in results:
            _write_optional_diagnostics(output_dir, result)

    _write_json(output_dir / "experiment_results.json", results)
    summary = {
        "task": task_name,
        "config": _to_builtin(config.__dict__),
        "data_quality": raw_event_audit,
        "results": results,
    }
    if classification_thresholds is not None:
        summary["classification_thresholds"] = {
            "lower": classification_thresholds[0],
            "upper": classification_thresholds[1],
        }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def _write_json(path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
