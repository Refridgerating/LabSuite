"""Command-line interface for shared ESR and VSM workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Sequence

from labsuite import __version__
from labsuite.cli.registry import DEFAULT_OUTPUT_ROOT, MODALITY_SPECS, ModalityCliSpec
from labsuite.core.exceptions import LabSuiteError, WorkflowError
from labsuite.core.resonance_metrics import ResonanceMetricsConfig, parse_area_window_multipliers
from labsuite.workflows.batch_folder import run_esr_batch_workflow
from labsuite.workflows.measurement_batch import (
    BatchRunResult,
    build_default_batch_output_dir,
    resolve_single_source,
    run_batch_workflow,
)
from labsuite.workflows.single_file import WorkflowArtifacts, run_esr_single_file_workflow


def build_parser() -> argparse.ArgumentParser:
    """Build the shared CLI parser."""

    parser = argparse.ArgumentParser(prog="labsuite")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for modality_name, spec in MODALITY_SPECS.items():
        modality_parser = subparsers.add_parser(modality_name, help=f"{modality_name.upper()} workflows")
        _add_modality_subcommands(modality_parser, spec)

    fit_single_parser = subparsers.add_parser(
        "fit-single",
        help="Run one ESR fit using recipe-driven local resonance-window integration.",
    )
    _add_fit_single_arguments(fit_single_parser)

    fit_batch_parser = subparsers.add_parser(
        "fit-batch",
        help="Run ESR fits in batch using recipe-driven local resonance-window integration.",
    )
    _add_fit_batch_arguments(fit_batch_parser)

    esr_parser = subparsers.add_parser(
        "esr-single",
        help="Legacy ESR single-file command kept as a compatibility alias.",
    )
    esr_parser.add_argument("source_file", type=Path, help="Path to the Bruker ESR descriptor file (.dsc).")
    esr_parser.add_argument("--recipe", type=Path, default=MODALITY_SPECS["esr"].default_recipe)
    esr_parser.add_argument("--output-dir", type=Path, default=None)
    esr_parser.add_argument("--fit-mode", choices=("auto", "single", "split"), default=None)
    esr_parser.add_argument("--show-raw", action="store_true")
    _add_resonance_metrics_arguments(esr_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by `python -m` and package scripts."""

    args = build_parser().parse_args(argv)
    try:
        if args.command in MODALITY_SPECS:
            return _run_modality_command(args.command, args)
        if args.command == "fit-single":
            return _run_fit_single_command(
                input_path=args.input_path,
                recipe_path=args.recipe,
                output_dir=args.output_dir,
                fit_mode=args.fit_mode,
                show_raw=args.show_raw,
                resonance_metrics_config=_build_resonance_metrics_config(args),
            )
        if args.command == "fit-batch":
            return _run_fit_batch_command(
                input_path=args.input_path,
                recipe_path=args.recipe,
                output_dir=args.output_dir,
                pattern=args.pattern,
                recursive=args.recursive,
                fit_mode=args.fit_mode,
                resonance_metrics_config=_build_resonance_metrics_config(args),
            )
        if args.command == "esr-single":
            return _run_legacy_esr_single_command(
                source_file=args.source_file,
                recipe_path=args.recipe,
                output_dir=args.output_dir,
                fit_mode=args.fit_mode,
                show_raw=args.show_raw,
                resonance_metrics_config=_build_resonance_metrics_config(args),
            )
        raise SystemExit(f"Unsupported command: {args.command}")
    except LabSuiteError as exc:
        raise SystemExit(str(exc)) from exc


def _add_modality_subcommands(parser: argparse.ArgumentParser, spec: ModalityCliSpec) -> None:
    subparsers = parser.add_subparsers(dest="verb", required=True)

    single_parser = subparsers.add_parser("single", help=f"Run one {spec.name.upper()} analysis.")
    single_parser.add_argument("--input", dest="input_path", required=True, type=Path)
    single_parser.add_argument("--recipe", type=Path, default=spec.default_recipe)
    single_parser.add_argument("--output-dir", type=Path, default=None)
    if spec.name == "esr":
        single_parser.add_argument("--fit-mode", choices=("auto", "single", "split"), default=None)
        single_parser.add_argument("--show-raw", action="store_true")
    if spec.name in {"esr", "fmr"}:
        _add_resonance_metrics_arguments(single_parser)

    batch_parser = subparsers.add_parser("batch", help=f"Run {spec.name.upper()} analyses in batch.")
    batch_parser.add_argument("--input", dest="input_path", required=True, type=Path)
    batch_parser.add_argument("--pattern", default=spec.default_pattern)
    batch_parser.add_argument("--recursive", action="store_true")
    batch_parser.add_argument("--recipe", type=Path, default=spec.default_recipe)
    batch_parser.add_argument("--output-dir", type=Path, default=None)
    if spec.name == "esr":
        batch_parser.add_argument("--fit-mode", choices=("auto", "single", "split"), default=None)
    if spec.name in {"esr", "fmr"}:
        _add_resonance_metrics_arguments(batch_parser)

    config_parser = subparsers.add_parser("config", help=f"Print or write the default {spec.name.upper()} recipe.")
    config_parser.add_argument("--output", type=Path, default=None)

    export_parser = subparsers.add_parser("export", help="Regenerate CSV and figure outputs from saved JSON.")
    export_parser.add_argument("--input", dest="input_path", required=True, type=Path)
    export_parser.add_argument("--output-dir", type=Path, default=None)

    report_parser = subparsers.add_parser("report", help="Generate a Markdown report from saved JSON results.")
    report_parser.add_argument("--input", dest="input_path", required=True, type=Path)
    report_parser.add_argument("--output", type=Path, default=None)
    report_parser.add_argument("--recursive", action="store_true")


def _add_shared_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recipe", type=Path, default=MODALITY_SPECS["esr"].default_recipe)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fit-mode", choices=("auto", "single", "split"), default=None)
    _add_resonance_metrics_arguments(parser)


def _add_resonance_metrics_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compute-resonance-metrics",
        dest="compute_resonance_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--area-window-mode",
        choices=("side-aware", "symmetric"),
        default="side-aware",
    )
    parser.add_argument(
        "--area-window-multipliers",
        default="1,2,3",
        help="Comma-separated FWHM multipliers for windowed areas.",
    )
    parser.add_argument("--compute-full-area", action="store_true")
    parser.add_argument(
        "--report-asymmetry",
        dest="report_asymmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--halfmax-interp", choices=("linear",), default="linear")
    parser.add_argument("--metrics-from", choices=("reconstructed_absorption",), default="reconstructed_absorption")
    parser.add_argument("--export-resonance-metrics", action="store_true")
    parser.add_argument("--plot-halfmax-markers", action="store_true")
    parser.add_argument("--plot-area-windows", action="store_true")


def _add_fit_single_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", dest="input_path", required=True, type=Path)
    _add_shared_fit_arguments(parser)
    parser.add_argument("--show-raw", action="store_true")


def _add_fit_batch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", dest="input_path", required=True, type=Path)
    parser.add_argument("--pattern", default="*.dsc")
    parser.add_argument("--recursive", action="store_true")
    _add_shared_fit_arguments(parser)


def _run_modality_command(modality: str, args: argparse.Namespace) -> int:
    spec = MODALITY_SPECS[modality]
    if args.verb == "single":
        return _run_modality_single(spec, args)
    if args.verb == "batch":
        return _run_modality_batch(spec, args)
    if args.verb == "config":
        return _run_modality_config(spec, args.output)
    if args.verb == "export":
        return _run_modality_export(spec, args.input_path, args.output_dir)
    if args.verb == "report":
        return _run_modality_report(spec, args.input_path, args.output, args.recursive)
    raise SystemExit(f"Unsupported {modality} verb: {args.verb}")


def _run_modality_single(spec: ModalityCliSpec, args: argparse.Namespace) -> int:
    source_file = resolve_single_source(
        args.input_path,
        allowed_suffixes=spec.allowed_suffixes,
        pattern=spec.default_pattern,
        recursive=False,
        source_label=spec.source_label,
    )
    resolved_output_dir = args.output_dir.resolve() if args.output_dir else DEFAULT_OUTPUT_ROOT / source_file.stem
    workflow_options = _workflow_options(spec.name, args)
    analysis, artifacts = spec.run_single_workflow(
        source_path=source_file,
        recipe_path=args.recipe.resolve(),
        output_dir=resolved_output_dir,
        **workflow_options,
    )
    _print_single_result(spec.name, analysis, artifacts)
    return 0


def _run_modality_batch(spec: ModalityCliSpec, args: argparse.Namespace) -> int:
    resolved_input = args.input_path.resolve()
    resolved_output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else build_default_batch_output_dir(DEFAULT_OUTPUT_ROOT, resolved_input)
    )
    workflow_options = _workflow_options(spec.name, args, batch=True)
    if spec.run_batch_workflow is not None:
        batch_result = spec.run_batch_workflow(
            inputs=[resolved_input],
            recipe_path=args.recipe,
            output_dir=resolved_output_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            **workflow_options,
        )
    else:
        batch_result = run_batch_workflow(
            inputs=[resolved_input],
            recipe_path=args.recipe,
            output_dir=resolved_output_dir,
            allowed_suffixes=spec.allowed_suffixes,
            pattern=args.pattern,
            recursive=args.recursive,
            source_label=spec.source_label,
            run_single_workflow=spec.run_single_workflow,
            summarize_analysis=spec.summarize_analysis,
            export_batch_figure=spec.export_batch_figure,
            workflow_options=workflow_options,
        )
    _print_batch_result(batch_result)
    return 0


def _run_modality_config(spec: ModalityCliSpec, output: Path | None) -> int:
    recipe_text = spec.default_recipe.read_text(encoding="utf-8")
    if output is None:
        print(recipe_text.rstrip())
        return 0
    resolved_output = output.resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(recipe_text, encoding="utf-8")
    print(f"Wrote config to {resolved_output}")
    return 0


def _run_modality_export(spec: ModalityCliSpec, input_path: Path, output_dir: Path | None) -> int:
    artifacts = spec.export_from_json(input_path.resolve(), None if output_dir is None else output_dir.resolve())
    print(f"JSON: {artifacts['json_path']}")
    print(f"Trace CSV: {artifacts['csv_path']}")
    print(f"Summary CSV: {artifacts['summary_csv_path']}")
    print(f"Figure: {artifacts['figure_path']}")
    return 0


def _run_modality_report(
    spec: ModalityCliSpec,
    input_path: Path,
    output: Path | None,
    recursive: bool,
) -> int:
    resolved_input = input_path.resolve()
    effective_recursive = recursive or resolved_input.is_dir()
    report_path = spec.build_report(
        resolved_input,
        None if output is None else output.resolve(),
        effective_recursive,
    )
    print(f"Report: {report_path}")
    return 0


def _workflow_options(modality: str, args: argparse.Namespace, *, batch: bool = False) -> dict[str, object]:
    if modality not in {"esr", "fmr"}:
        return {}
    options: dict[str, object] = {"resonance_metrics_config": _build_resonance_metrics_config(args)}
    if modality == "esr":
        options["fit_mode"] = getattr(args, "fit_mode", None)
        if not batch:
            options["show_raw"] = bool(getattr(args, "show_raw", False))
    return options


def _build_resonance_metrics_config(args: argparse.Namespace) -> ResonanceMetricsConfig:
    return ResonanceMetricsConfig(
        compute_resonance_metrics=bool(getattr(args, "compute_resonance_metrics", True)),
        area_window_mode=getattr(args, "area_window_mode", "side-aware"),
        area_window_multipliers=parse_area_window_multipliers(getattr(args, "area_window_multipliers", "1,2,3")),
        compute_full_area=bool(getattr(args, "compute_full_area", False)),
        report_asymmetry=bool(getattr(args, "report_asymmetry", True)),
        halfmax_interp=getattr(args, "halfmax_interp", "linear"),
        metrics_from=getattr(args, "metrics_from", "reconstructed_absorption"),
        export_resonance_metrics=bool(getattr(args, "export_resonance_metrics", False)),
        plot_halfmax_markers=bool(getattr(args, "plot_halfmax_markers", False)),
        plot_area_windows=bool(getattr(args, "plot_area_windows", False)),
    )


def _run_fit_single_command(
    *,
    input_path: Path,
    recipe_path: Path,
    output_dir: Path | None,
    fit_mode: Literal["auto", "single", "split"] | None,
    show_raw: bool,
    resonance_metrics_config: ResonanceMetricsConfig,
) -> int:
    try:
        source_file = resolve_single_source(
            input_path,
            allowed_suffixes=MODALITY_SPECS["esr"].allowed_suffixes,
            pattern=MODALITY_SPECS["esr"].default_pattern,
            recursive=False,
            source_label=MODALITY_SPECS["esr"].source_label,
        )
    except WorkflowError as exc:
        message = str(exc)
        if message.startswith("single requires exactly one discovered source file, found "):
            count = message.rsplit(" ", maxsplit=1)[-1]
            raise WorkflowError(
                f"fit-single requires exactly one discovered .dsc file, found {count}"
            ) from exc
        raise
    resolved_output_dir = output_dir.resolve() if output_dir else DEFAULT_OUTPUT_ROOT / source_file.stem
    analysis, artifacts = run_esr_single_file_workflow(
        source_path=source_file,
        recipe_path=recipe_path.resolve(),
        output_dir=resolved_output_dir,
        fit_mode=fit_mode,
        show_raw=show_raw,
        resonance_metrics_config=resonance_metrics_config,
    )
    _print_single_result("esr", analysis, artifacts)
    return 0


def _run_fit_batch_command(
    *,
    input_path: Path,
    recipe_path: Path,
    output_dir: Path | None,
    pattern: str,
    recursive: bool,
    fit_mode: Literal["auto", "single", "split"] | None,
    resonance_metrics_config: ResonanceMetricsConfig,
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
        resonance_metrics_config=resonance_metrics_config,
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
    resonance_metrics_config: ResonanceMetricsConfig,
) -> int:
    resolved_source_file = source_file.resolve()
    resolved_output_dir = output_dir.resolve() if output_dir else DEFAULT_OUTPUT_ROOT / resolved_source_file.stem
    analysis, artifacts = run_esr_single_file_workflow(
        source_path=resolved_source_file,
        recipe_path=recipe_path.resolve(),
        output_dir=resolved_output_dir,
        fit_mode=fit_mode,
        show_raw=show_raw,
        resonance_metrics_config=resonance_metrics_config,
    )
    _print_single_result("esr", analysis, artifacts)
    return 0


def _print_single_result(modality: str, analysis, artifacts: WorkflowArtifacts) -> None:
    if modality == "esr":
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
    elif modality == "fmr":
        summary = analysis.summary_metrics
        print(f"Loaded {analysis.measurement.source_path.name} with {summary['trace_count']} trace(s)")
        print(f"Sample: {summary['sample_id']}  Replicate: {summary.get('replicate_id')}")
        print(
            f"Accepted traces: {summary.get('accepted_trace_count')} / {summary.get('trace_count')}  "
            f"Mode: {summary.get('measurement_mode')}"
        )
        print(
            "Physics: "
            f"Kittel={summary.get('kittel_success')}  "
            f"Linewidth={summary.get('linewidth_success')}  "
            f"g={summary.get('g')}  "
            f"alpha={summary.get('alpha')}"
        )
    else:
        summary = analysis.summary_metrics
        print(f"Loaded {analysis.measurement.source_path.name} with {summary['point_count']} points")
        print(f"Sample: {summary['sample_id']}  Replicate: {summary.get('replicate_id')}")
        print(f"Temperature: {summary.get('temperature_k'):.3f} K")
        print(
            "Loop metrics: "
            f"Hc-= {summary.get('coercive_field_negative_mT')} mT, "
            f"Hc+= {summary.get('coercive_field_positive_mT')} mT, "
            f"Mr+= {summary.get('remanence_positive_emu')} emu, "
            f"Mr-= {summary.get('remanence_negative_emu')} emu"
        )
        print(
            "Background: "
            f"mode={summary.get('background_mode')}  "
            f"accepted={summary.get('background_correction_accepted')}  "
            f"slope={summary.get('background_slope_emu_per_mT')} emu/mT, "
            f"intercept={summary.get('background_intercept_emu')} emu, "
            f"center_applied={summary.get('centering_applied')}"
        )
        print(f"Warnings: {summary.get('warning_count')}")
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
    if batch_result.resonance_metrics_csv_path is not None:
        print(f"Batch resonance metrics: {batch_result.resonance_metrics_csv_path}")
    for name, path in sorted(batch_result.batch_figure_paths.items()):
        print(f"Batch figure [{name}]: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
