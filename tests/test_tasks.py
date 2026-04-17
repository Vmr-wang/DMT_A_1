from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmt_mood_pipeline.config import PipelineConfig
from dmt_mood_pipeline.tasks import run_task_1b, run_task_2a_rf


class _FakeFrame:
    def __init__(self):
        self.written_paths = []

    def to_csv(self, path, index=False):
        self.written_paths.append((Path(path), index))
        Path(path).write_text("fake_frame\n", encoding="utf-8")


class TaskRunnerTestCase(unittest.TestCase):
    def test_run_task_1b_writes_daily_and_backtest(self):
        config = PipelineConfig(csv_path=Path("dataset.csv"), output_dir=Path(tempfile.mkdtemp()))
        fake_daily_raw = object()
        fake_daily_clean = _FakeFrame()
        fake_audit = {"duplicate_rows_removed": 1}
        fake_backtest = {"mood_mean": {"forward_fill": {"mae": 0.1}}}

        with patch("dmt_mood_pipeline.tasks.build_daily_patient_table_with_audit", return_value=(fake_daily_raw, fake_audit)), patch(
            "dmt_mood_pipeline.tasks.backtest_imputation_strategies",
            return_value=fake_backtest,
        ), patch(
            "dmt_mood_pipeline.tasks.apply_imputation",
            return_value=fake_daily_clean,
        ):
            summary = run_task_1b(config)

        self.assertEqual(summary["task"], "task_1b")
        self.assertTrue((config.output_dir / "daily_patient_table.csv").exists())
        self.assertTrue((config.output_dir / "imputation_backtest.json").exists())
        self.assertTrue((config.output_dir / "run_summary.json").exists())

    def test_run_task_2a_rf_only_persists_classification_results(self):
        config = PipelineConfig(csv_path=Path("dataset.csv"), output_dir=Path(tempfile.mkdtemp()))
        fake_results = [
            {"model_family": "persistence_baseline", "task": "classification"},
            {"model_family": "random_forest", "task": "classification"},
        ]

        with patch(
            "dmt_mood_pipeline.tasks._prepare_modeling_datasets",
            return_value=(None, object(), None, {"duplicate_rows_removed": 0}, (6.0, 7.0)),
        ), patch(
            "dmt_mood_pipeline.tasks._run_persistence_baseline_for_classification",
            return_value=fake_results[0],
        ), patch(
            "dmt_mood_pipeline.tasks._run_random_forest",
            return_value=fake_results[1],
        ), patch("dmt_mood_pipeline.tasks._write_optional_diagnostics") as write_plots:
            summary = run_task_2a_rf(config)

        self.assertEqual(summary["task"], "task_2a_rf")
        self.assertEqual([result["task"] for result in summary["results"]], ["classification", "classification"])
        self.assertEqual(summary["classification_thresholds"], {"lower": 6.0, "upper": 7.0})
        self.assertTrue((config.output_dir / "experiment_results.json").exists())
        self.assertTrue((config.output_dir / "run_summary.json").exists())
        self.assertEqual(write_plots.call_count, 2)

        stored_results = json.loads((config.output_dir / "experiment_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(stored_results), 2)


if __name__ == "__main__":
    unittest.main()
