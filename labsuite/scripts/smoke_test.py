"""Run the first ESR single-file workflow against synthetic data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", type=Path, help="Input ESR file to analyze.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "smoke_test",
        help="Directory for workflow artifacts.",
    )
    return parser


def main() -> int:
    from labsuite.workflows.single_file import run_esr_single_file_workflow

    args = build_parser().parse_args()
    analysis, artifacts = run_esr_single_file_workflow(
        source_path=args.source_file.resolve(),
        recipe_path=PROJECT_ROOT / "recipes" / "esr" / "default.yaml",
        output_dir=args.output_dir.resolve(),
    )
    print(f"points={analysis.dataset.field_mT.size}")
    print(f"selected_mode={analysis.selected_mode}")
    if analysis.selected_mode == "single" and analysis.single_fit is not None:
        print(f"center_mT={analysis.single_fit.parameters['center_mT']:.4f}")
    else:
        print(f"peak_count={len(analysis.peak_fits)}")
    print(f"json={artifacts.json_path}")
    print(f"trace_csv={artifacts.csv_path}")
    print(f"summary_csv={artifacts.summary_csv_path}")
    print(f"figure={artifacts.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
