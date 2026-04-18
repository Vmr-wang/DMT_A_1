from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .eda import run_eda
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DMT advanced mood prediction pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full pipeline.")
    run_parser.add_argument("--csv", type=Path, required=True, help="Path to dataset_mood_smartphone.csv")
    run_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for exported artifacts.")
    run_parser.add_argument("--window-size", type=int, default=7)
    run_parser.add_argument("--imputation-method", choices=["forward_fill", "ewma"], default="forward_fill")
    run_parser.add_argument("--sequence-cell-type", choices=["gru", "lstm"], default="gru")
    run_parser.add_argument("--no-export-intermediate", action="store_true")
    run_parser.add_argument("--no-plots", action="store_true")

    eda_parser = subparsers.add_parser("eda", help="Export Task 1A EDA tables and plots.")
    eda_parser.add_argument("--csv", type=Path, required=True, help="Path to dataset_mood_smartphone.csv")
    eda_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for exported Task 1A artifacts.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        config = PipelineConfig(
            csv_path=args.csv,
            output_dir=args.output_dir,
            window_size=args.window_size,
            selected_imputation_method=args.imputation_method,
            sequence_cell_type=args.sequence_cell_type,
            export_intermediate=not args.no_export_intermediate,
            generate_plots=not args.no_plots,
        )
        summary = run_pipeline(config)
        print(json.dumps({"output_dir": str(config.output_dir), "num_results": len(summary["results"])}, indent=2))
        return

    if args.command == "eda":
        summary = run_eda(csv_path=args.csv, output_dir=args.output_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
