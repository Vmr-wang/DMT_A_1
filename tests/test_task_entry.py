from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmt_mood_pipeline.task_entry import main_for_task


class TaskEntryTestCase(unittest.TestCase):
    def test_default_forward_fill_uses_script_stem_output(self):
        captured = {}

        def fake_runner(config):
            captured["config"] = config
            return {"task": "task_1b"}

        stdout = io.StringIO()
        original_argv = sys.argv
        try:
            sys.argv = ["task_1b.py"]
            with redirect_stdout(stdout):
                main_for_task("task_1b", generate_plots=False, task_runner=fake_runner)
        finally:
            sys.argv = original_argv

        config = captured["config"]
        self.assertEqual(config.selected_imputation_method, "forward_fill")
        self.assertEqual(config.output_dir.name, "task_1b")
        self.assertEqual(json.loads(stdout.getvalue())["task"], "task_1b")

    def test_ewma_wrapper_sets_ewma_output_dir(self):
        captured = {}

        def fake_runner(config):
            captured["config"] = config
            return {"task": "task_1b"}

        stdout = io.StringIO()
        original_argv = sys.argv
        try:
            sys.argv = ["task_1b_ewma.py"]
            with redirect_stdout(stdout):
                main_for_task("task_1b", imputation_method="ewma", generate_plots=False, task_runner=fake_runner)
        finally:
            sys.argv = original_argv

        config = captured["config"]
        self.assertEqual(config.selected_imputation_method, "ewma")
        self.assertEqual(config.output_dir.name, "task_1b_ewma")

    def test_lstm_wrapper_sets_sequence_cell_type(self):
        captured = {}

        def fake_runner(config):
            captured["config"] = config
            return {"task": "task_2a_lstm"}

        stdout = io.StringIO()
        original_argv = sys.argv
        try:
            sys.argv = ["task_2a_lstm.py"]
            with redirect_stdout(stdout):
                main_for_task("task_2a_sequence", sequence_cell_type="lstm", task_runner=fake_runner)
        finally:
            sys.argv = original_argv

        self.assertEqual(captured["config"].sequence_cell_type, "lstm")
        self.assertEqual(captured["config"].output_dir.name, "task_2a_lstm")


if __name__ == "__main__":
    unittest.main()
