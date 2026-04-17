from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PipelineConfig
from .tasks import (
    run_task_1a,
    run_task_1b,
    run_task_1c,
    run_task_2a_rf,
    run_task_2a_sequence,
    run_task_4_rf,
    run_task_4_sequence,
)

TASK_RUNNERS = {
    "task_1a": run_task_1a,
    "task_1b": run_task_1b,
    "task_1c": run_task_1c,
    "task_2a_rf": run_task_2a_rf,
    "task_2a_sequence": run_task_2a_sequence,
    "task_4_rf": run_task_4_rf,
    "task_4_sequence": run_task_4_sequence,
}


def main_for_task(
    task_name: str,
    *,
    imputation_method: str = "forward_fill",
    sequence_cell_type: str = "gru",
    generate_plots: bool = True,
    export_intermediate: bool = False,
    task_runner=None,
) -> None:
    script_stem = Path(sys.argv[0]).stem
    parser = build_parser(script_stem)
    args = parser.parse_args()

    config = PipelineConfig(
        csv_path=args.csv,
        output_dir=args.output_dir,
        selected_imputation_method=imputation_method,
        sequence_cell_type=sequence_cell_type,
        export_intermediate=export_intermediate,
        generate_plots=generate_plots,
    )
    runner = task_runner or TASK_RUNNERS[task_name]
    summary = runner(config)
    print(json.dumps(summary, indent=2))


def build_parser(script_stem: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {script_stem} using the new DMT pipeline.")
    parser.add_argument("--csv", type=Path, default=_repo_root() / "dataset_mood_smartphone.csv")
    parser.add_argument("--output-dir", type=Path, default=_repo_root() / "outputs" / script_stem)
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
