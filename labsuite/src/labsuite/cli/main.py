"""Command-line interface for the first LabSuite workflow slice."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Sequence

from labsuite import __version__
from labsuite.core.exceptions import LabSuiteError, WorkflowError
from labsuite.core.types import AnalysisResult
from labsuite.workflows.batch_folder import (
    BatchRunResult,
    build_default_batch_output_dir,
    discover_esr_source_files,
    run_esr_batch_workflow,
)
from labsuite.workflows.single_file import WorkflowArtifacts, run_esr_single_file_workflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECIPE = PROJECT_ROOT / "recipes" / "esr" / "default.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the ESR-only workflow."""

    parser = argparse.ArgumentParser(prog="labsuite")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_single_parser = subparsers.add_parser(
        "fit-single",
        help="Run one ESR fit using recipe-driven local resonance-window integration.",
    )
    _add_fit_single_arguments(fit_single_parser)

    fit_batch_parser = subparsers.add_parser(
        "fit-batch",
        help=(
            "Run ESR fits in batch using recipe-driven local resonance-window "
            "integration."
        ),
    )
    _add_fit_batch_arguments(fit_batch_parser)

    esr_parser = subparsers.add_parser(
        "esr-single",
        help="Run one ESR fit using recipe-driven local resonance-window integration.",
    )
    esr_parser.add_argument(
        "source_file",
        type=Path,
        help="Path to the Bruker ESR descriptor file (.dsc).",
    )
    esr_parser.add_argument(
        "--recipe",
        type=Path,
        default=DEFAULT_RECIPE,
        help="Path to the ESR preprocessing recipe.",
    )
    esr_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where JSON, CSV, and figure outputs will be written.",
    )
    esr_parser.add_argument(
        "--fit-mode",
        choices=("auto", "single", "split"),
        default=None,
        help="Select the fitting strategy for derivative resonances.",
    )
    esr_parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Display the exported raw/fit figure after saving it.",
    )

    return parser


def _add_shared_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recipe",
        type=Path,
        default=DEFAULT_RECIPE,
        help="Path to the ESR preprocessing recipe, including local integration settings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--fit-mode",
        choices=("auto", "single", "split"),
        default=None,
        help="Select the fitting strategy for derivative resonances.",
    )


def _add_fit_single_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        dest="input_path",
        required=True,
        type=Path,
        help=(
            "Path to a Bruker descriptor file or a folder containing exactly "
            "one .dsc file."
        ),
    )
    _add_shared_fit_arguments(parser)
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Display the exported raw/fit figure after saving it.",
    )


def _add_fit_batch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        dest="input_path",
        required=True,
        type=Path,
        help="Path to a Bruker descriptor file or a folder to scan for .dsc files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.dsc",
        help="Filename glob used when scanning folder inputs for descriptor files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan folder inputs for matching descriptor files.",
    )
    _add_shared_fit_arguments(parser)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by `python -m` and package scripts."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "fit-single":
            return _run_fit_single_command(
                input_path=args.input_path,
                recipe_path=args.recipe,
                output_dir=args.output_dir,
                fit_mode=args.fit_mode,
                show_raw=args.show_raw,
            )
        if args.command == "fit-batch":
            return _run_fit_batch_command(
                input_path=args.input_path,
                recipe_path=args.recipe,
                output_dir=args.output_dir,
                pattern=args.pattern,
                recursive=args.recursive,
                fit_mode=args.fit_mode,
            )
        if args.command == "esr-single":
            return _run_legacy_esr_single_command(
                source_file=args.source_file,
                recipe_path=args.recipe,
                output_dir=args.output_dir,
                fit_mode=args.fit_mode,
                show_raw=args.show_raw,
            )
        raise SystemExit(f"Unsupported command: {args.command}")
    except LabSuiteError as exc:
        raise SystemExit(str(exc)) from exc


def _run_fit_single_command(
    *,
    input_path: Path,
    recipe_path: Path,
    output_dir: Path | None,
    fit_mode: Literal["auto", "single", "split"] | None,
    show_raw: bool,
) -> int:
    source_file = _resolve_single_input_source(input_path)
    resolved_output_dir = (
        output_dir.resolve() if output_dir else DEFAULT_OUTPUT_ROOT / source_file.stem
    )
    analysis, artifacts = run_esr_single_file_workflow(
        source_path=source_file,
        recipe_path=recipe_path.resolve(),
        output_dir=resolved_output_dir,
        fit_mode=fit_mode,
        show_raw=show_raw,
    )
    _print_single_file_result(analysis, artifacts)
    return 0


def _run_fit_batch_command(
    *,
    input_path: Path,
    recipe_path: Path,
    output_dir: Path | None,
    pattern: str,
    recursive: bool,
    fit_mode: Literal["auto", "single", "split"] | None,
) -> int:
    resolved_input = input_path.resolve()
    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else build_default_batch_output_dir(DEFAULT_OUTPUT_ROOT, resolved_input)
    )
    batch_result = run_esr_batch_workflow(
        inputs=[resolved_input],
        recipe_path=recipe_path.resolve(),
        output_dir=resolved_output_dir,
        pattern=pattern,
        recursive=recursive,
        fit_mode=fit_mode,
    )
    _print_batch_result(batch_result)
    return 0


def _run_legacy_esr_single_command(
    *,
    source_file: Path,
    recipe_path: Path,
    output_dir: Path | None,
    fit_mode: Literal["auto", "single", "split"] | None,
    show_raw: bool,
) -> int:
    resolved_source_file = source_file.resolve()
    resolved_output_dir = (
        output_dir.resolve() if output_dir else DEFAULT_OUTPUT_ROOT / resolved_source_file.stem
    )
    analysis, artifacts = run_esr_single_file_workflow(
        source_path=resolved_source_file,
        recipe_path=recipe_path.resolve(),
        output_dir=resolved_output_dir,
        fit_mode=fit_mode,
        show_raw=show_raw,
    )
    _print_single_file_result(analysis, artifacts)
    return 0


def _resolve_single_input_source(input_path: Path) -> Path:
    discovered_sources = discover_esr_source_files([input_path.resolve()])
    if len(discovered_sources) != 1:
        raise WorkflowError(
            f"fit-single requires exactly one discovered .dsc file, found {len(discovered_sources)}"
        )
    return discovered_sources[0]


def _print_single_file_result(analysis: AnalysisResult, artifacts: WorkflowArtifacts) -> None:
    print(f"Loaded {analysis.dataset.source_path.name} with {analysis.dataset.field_mT.size} points")
    print(f"Selected fit mode: {analysis.selected_mode}")
    if analysis.selected_mode == "single" and analysis.single_fit is not None:
        print(f"Fit center: {analysis.single_fit.parameters['center_mT']:.4f} mT")
        print(f"Fit gamma: {analysis.single_fit.parameters['gamma_mT']:.4f} mT")
        print(f"R^2: {analysis.single_fit.metrics['r_squared']:.6f}")
    else:
        for peak_fit in analysis.peak_fits:
            print(
                f"{peak_fit.label}: center={peak_fit.fit.parameters['center_mT']:.4f} mT "
                f"gamma={peak_fit.fit.parameters['gamma_mT']:.4f} mT "
                f"R^2={peak_fit.fit.metrics['r_squared']:.6f}"
            )
        print(f"Split improvement: {analysis.fit_decision.split_improvement_ratio or 0.0:.6f}")
    total_area_integral = analysis.total_integral.area_integral
    diagnostic_area_integral = analysis.diagnostic_total_integral.area_integral
    total_label = "NA" if total_area_integral is None else f"{total_area_integral:.6f}"
    diagnostic_label = "NA" if diagnostic_area_integral is None else f"{diagnostic_area_integral:.6f}"
    print(f"Primary fit-derived area integral: {total_label}")
    print(f"Diagnostic full-span area integral: {diagnostic_label}")
    print(f"JSON: {artifacts.json_path}")
    print(f"Trace CSV: {artifacts.csv_path}")
    print(f"Summary CSV: {artifacts.summary_csv_path}")
    print(f"Figure: {artifacts.figure_path}")


def _print_batch_result(batch_result: BatchRunResult) -> None:
    print(f"Discovered {len(batch_result.discovered_sources)} source file(s)")
    print(f"Succeeded: {len(batch_result.succeeded_items)}")
    print(f"Failed: {len(batch_result.failed_items)}")
    print(f"Results folder: {batch_result.output_dir}")
    print(f"Batch summary: {batch_result.summary_csv_path}")
    print(f"Batch manifest: {batch_result.manifest_json_path}")


if __name__ == "__main__":
    raise SystemExit(main())
