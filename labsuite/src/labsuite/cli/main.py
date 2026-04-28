"""Command-line interface for shared ESR and VSM workflows."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml

from labsuite import __version__
from labsuite.cli.registry import DEFAULT_OUTPUT_ROOT, MODALITY_SPECS, PROJECT_ROOT, ModalityCliSpec
from labsuite.core.exceptions import LabSuiteError, WorkflowError
from labsuite.core.resonance_metrics import ResonanceMetricsConfig, parse_area_window_multipliers
from labsuite.core.sample_registry import (
    DEFAULT_MEASUREMENT_LEDGER_PATH,
    DEFAULT_PROCESSED_LEDGER_PATH,
    DEFAULT_SAMPLE_REGISTRY_PATH,
    AnalysisDefaults,
    DirectVolumeMetadata,
    QuantityMetadata,
    RegistryWorkflowOptions,
    VolumeMetadata,
    add_sample,
    empty_measurement_ledger,
    empty_registry,
    find_sample,
    load_measurement_ledger,
    load_registry,
    resolve_sample_magnetic_volume,
    sample_to_dict,
    save_measurement_ledger,
    save_registry,
    update_sample_magnetic_volume_fields,
    upsert_measurement_record,
    validate_registry,
)
from labsuite.core.sample_registry import (
    SampleRecord as RegistrySampleRecord,
)
from labsuite.sample_analysis.service import (
    analyze_sample,
    analyze_sample_batch,
    build_sample_readiness,
)
from labsuite.workflows.batch_folder import run_esr_batch_workflow
from labsuite.workflows.measurement_batch import (
    BatchRunResult,
    build_default_batch_output_dir,
    resolve_single_source,
    run_batch_workflow,
)
from labsuite.workflows.raw_import import RawImportResult, import_raw_for_ledger
from labsuite.workflows.single_file import WorkflowArtifacts, run_esr_single_file_workflow

DEFAULT_SAMPLE_REGISTRY_FILE = PROJECT_ROOT / DEFAULT_SAMPLE_REGISTRY_PATH
DEFAULT_MEASUREMENT_LEDGER_FILE = PROJECT_ROOT / DEFAULT_MEASUREMENT_LEDGER_PATH
DEFAULT_PROCESSED_LEDGER_FILE = PROJECT_ROOT / DEFAULT_PROCESSED_LEDGER_PATH
DEFAULT_RAW_IMPORT_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_SAMPLE_ANALYSIS_RECIPE = PROJECT_ROOT / "recipes" / "sample_analysis" / "default.yaml"
DEFAULT_DERIVED_OUTPUT_ROOT = PROJECT_ROOT / "data" / "derived"


def build_parser() -> argparse.ArgumentParser:
    """Build the shared CLI parser."""

    parser = argparse.ArgumentParser(prog="labsuite")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="Manage the project sample registry.")
    _add_sample_subcommands(sample_parser)

    for modality_name, spec in MODALITY_SPECS.items():
        modality_parser = subparsers.add_parser(
            modality_name, help=f"{modality_name.upper()} workflows"
        )
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
    esr_parser.add_argument(
        "source_file", type=Path, help="Path to the Bruker ESR descriptor file (.dsc)."
    )
    esr_parser.add_argument("--recipe", type=Path, default=MODALITY_SPECS["esr"].default_recipe)
    esr_parser.add_argument("--output-dir", type=Path, default=None)
    esr_parser.add_argument("--fit-mode", choices=("auto", "single", "split"), default=None)
    esr_parser.add_argument("--show-raw", action="store_true")
    _add_registry_analysis_arguments(esr_parser)
    _add_resonance_metrics_arguments(esr_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by `python -m` and package scripts."""

    args = build_parser().parse_args(argv)
    try:
        if args.command in MODALITY_SPECS:
            return _run_modality_command(args.command, args)
        if args.command == "sample":
            return _run_sample_command(args)
        if args.command == "fit-single":
            return _run_fit_single_command(
                input_path=args.input_path,
                recipe_path=args.recipe,
                output_dir=args.output_dir,
                fit_mode=args.fit_mode,
                show_raw=args.show_raw,
                resonance_metrics_config=_build_resonance_metrics_config(args),
                registry_options=_build_registry_workflow_options(args),
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
                registry_options=_build_registry_workflow_options(args),
            )
        if args.command == "esr-single":
            return _run_legacy_esr_single_command(
                source_file=args.source_file,
                recipe_path=args.recipe,
                output_dir=args.output_dir,
                fit_mode=args.fit_mode,
                show_raw=args.show_raw,
                resonance_metrics_config=_build_resonance_metrics_config(args),
                registry_options=_build_registry_workflow_options(args),
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
    _add_registry_analysis_arguments(single_parser)
    if spec.name == "esr":
        single_parser.add_argument("--fit-mode", choices=("auto", "single", "split"), default=None)
        single_parser.add_argument("--show-raw", action="store_true")
    if spec.name in {"esr", "fmr"}:
        _add_resonance_metrics_arguments(single_parser)
    if spec.name == "fmr":
        _add_fmr_field_polarity_arguments(single_parser)

    batch_parser = subparsers.add_parser(
        "batch", help=f"Run {spec.name.upper()} analyses in batch."
    )
    batch_parser.add_argument("--input", dest="input_path", required=True, type=Path)
    batch_parser.add_argument("--pattern", default=spec.default_pattern)
    batch_parser.add_argument("--recursive", action="store_true")
    batch_parser.add_argument("--recipe", type=Path, default=spec.default_recipe)
    batch_parser.add_argument("--output-dir", type=Path, default=None)
    _add_registry_analysis_arguments(batch_parser)
    if spec.name == "esr":
        batch_parser.add_argument("--fit-mode", choices=("auto", "single", "split"), default=None)
    if spec.name in {"esr", "fmr"}:
        _add_resonance_metrics_arguments(batch_parser)
    if spec.name == "fmr":
        _add_fmr_field_polarity_arguments(batch_parser)

    config_parser = subparsers.add_parser(
        "config", help=f"Print or write the default {spec.name.upper()} recipe."
    )
    config_parser.add_argument("--output", type=Path, default=None)

    export_parser = subparsers.add_parser(
        "export", help="Regenerate CSV and figure outputs from saved JSON."
    )
    export_parser.add_argument("--input", dest="input_path", required=True, type=Path)
    export_parser.add_argument("--output-dir", type=Path, default=None)

    report_parser = subparsers.add_parser(
        "report", help="Generate a Markdown report from saved JSON results."
    )
    report_parser.add_argument("--input", dest="input_path", required=True, type=Path)
    report_parser.add_argument("--output", type=Path, default=None)
    report_parser.add_argument("--recursive", action="store_true")


def _add_shared_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recipe", type=Path, default=MODALITY_SPECS["esr"].default_recipe)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fit-mode", choices=("auto", "single", "split"), default=None)
    _add_registry_analysis_arguments(parser)
    _add_resonance_metrics_arguments(parser)


def _add_registry_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--measurement-id", default=None)
    parser.add_argument("--geometry", choices=("ip", "oop", "angular", "unknown"), default=None)
    parser.add_argument("--branch-labels", default="")
    parser.add_argument("--sample-registry", type=Path, default=DEFAULT_SAMPLE_REGISTRY_FILE)
    parser.add_argument("--measurement-ledger", type=Path, default=DEFAULT_MEASUREMENT_LEDGER_FILE)
    parser.add_argument("--processed-ledger", type=Path, default=DEFAULT_PROCESSED_LEDGER_FILE)
    parser.add_argument("--update-ledger", action="store_true")
    parser.add_argument("--create-sample", action="store_true")
    parser.add_argument("--mark-canonical", action="store_true")
    parser.add_argument("--replace-canonical", action="store_true")
    parser.add_argument("--metadata-manifest", type=Path, default=None)
    parser.add_argument("--raw-import-root", type=Path, default=DEFAULT_RAW_IMPORT_ROOT)
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--g-mode", choices=("fixed", "float", "bounded"), default=None)
    parser.add_argument("--g-value", type=float, default=None)
    parser.add_argument("--interactive", action="store_true")


def _add_sample_subcommands(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="sample_verb", required=True)

    add_parser = subparsers.add_parser("add", help="Add a physical sample.")
    add_parser.add_argument("sample_id", nargs="?")
    _add_sample_registry_path(add_parser)
    add_parser.add_argument("--alias", action="append", default=[])
    add_parser.add_argument("--condition", default=None)
    add_parser.add_argument("--replicate", default=None)
    add_parser.add_argument("--stack", default=None)
    add_parser.add_argument("--area-value", type=float, default=None)
    add_parser.add_argument("--area-unit", default=None)
    add_parser.add_argument("--area-uncertainty", type=float, default=None)
    add_parser.add_argument("--thickness-value", type=float, default=None)
    add_parser.add_argument("--thickness-unit", default=None)
    add_parser.add_argument("--thickness-uncertainty", type=float, default=None)
    add_parser.add_argument("--vmag-value", type=float, default=None)
    add_parser.add_argument("--vmag-unit", default=None)
    add_parser.add_argument("--vmag-uncertainty", type=float, default=None)
    add_parser.add_argument("--vmag-method", default=None)
    add_parser.add_argument("--g-mode", choices=("fixed", "float", "bounded"), default="float")
    add_parser.add_argument("--g-value", type=float, default=None)
    add_parser.add_argument("--ms-source", default=None)
    add_parser.add_argument("--interactive", action="store_true")
    _add_sample_volume_arguments(add_parser)

    update_parser = subparsers.add_parser("update", help="Update a physical sample.")
    update_parser.add_argument("sample_id")
    _add_sample_registry_path(update_parser)
    update_parser.add_argument("--alias", action="append", default=[])
    update_parser.add_argument("--condition", default=None)
    update_parser.add_argument("--replicate", default=None)
    update_parser.add_argument("--stack", default=None)
    update_parser.add_argument("--area-value", type=float, default=None)
    update_parser.add_argument("--area-unit", default=None)
    update_parser.add_argument("--area-uncertainty", type=float, default=None)
    update_parser.add_argument("--thickness-value", type=float, default=None)
    update_parser.add_argument("--thickness-unit", default=None)
    update_parser.add_argument("--thickness-uncertainty", type=float, default=None)
    update_parser.add_argument("--vmag-value", type=float, default=None)
    update_parser.add_argument("--vmag-unit", default=None)
    update_parser.add_argument("--vmag-uncertainty", type=float, default=None)
    update_parser.add_argument("--vmag-method", default=None)
    update_parser.add_argument("--g-mode", choices=("fixed", "float", "bounded"), default=None)
    update_parser.add_argument("--g-value", type=float, default=None)
    update_parser.add_argument("--ms-source", default=None)
    update_parser.add_argument("--interactive", action="store_true")
    _add_sample_volume_arguments(update_parser)

    list_parser = subparsers.add_parser("list", help="List registered samples.")
    _add_sample_registry_path(list_parser)

    show_parser = subparsers.add_parser("show", help="Show one registered sample.")
    show_parser.add_argument("sample_id")
    _add_sample_registry_path(show_parser)

    register_parser = subparsers.add_parser("register-file", help="Register a measurement file.")
    register_parser.add_argument("path", type=Path)
    register_parser.add_argument("--type", required=True, choices=("fmr", "vsm", "esr"))
    register_parser.add_argument("--sample-id", default=None)
    register_parser.add_argument(
        "--geometry", choices=("ip", "oop", "angular", "unknown"), default="unknown"
    )
    register_parser.add_argument("--measurement-id", default=None)
    register_parser.add_argument("--branch-labels", default="")
    register_parser.add_argument("--instrument", default=None)
    register_parser.add_argument("--notes", default=None)
    register_parser.add_argument("--interactive", action="store_true")
    _add_sample_metadata_paths(register_parser, include_processed=False)

    validate_parser = subparsers.add_parser("validate", help="Validate the sample registry.")
    _add_sample_registry_path(validate_parser)

    readiness_parser = subparsers.add_parser(
        "readiness", help="Report sample-level analysis readiness."
    )
    readiness_parser.add_argument("sample_id")
    _add_sample_metadata_paths(readiness_parser)
    readiness_parser.add_argument("--recipe", type=Path, default=DEFAULT_SAMPLE_ANALYSIS_RECIPE)

    analyze_parser = subparsers.add_parser("analyze", help="Run sample-level derived analysis.")
    analyze_parser.add_argument("sample_id")
    _add_sample_metadata_paths(analyze_parser)
    analyze_parser.add_argument("--recipe", type=Path, default=DEFAULT_SAMPLE_ANALYSIS_RECIPE)
    analyze_parser.add_argument("--output-dir", type=Path, default=None)

    analyze_batch_parser = subparsers.add_parser(
        "analyze-batch", help="Run sample-level derived analysis for all registered samples."
    )
    _add_sample_metadata_paths(analyze_batch_parser)
    analyze_batch_parser.add_argument("--recipe", type=Path, default=DEFAULT_SAMPLE_ANALYSIS_RECIPE)
    analyze_batch_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DERIVED_OUTPUT_ROOT
    )


def _add_sample_registry_path(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-registry", type=Path, default=DEFAULT_SAMPLE_REGISTRY_FILE)


def _add_sample_volume_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--magnetic-volume-m3", type=float, default=None)
    parser.add_argument(
        "--geometry-shape",
        choices=("rectangle", "square", "circle", "custom_area", "array"),
        default=None,
    )
    parser.add_argument("--length", type=float, default=None)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--side", type=float, default=None)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--length-unit", default=None)
    parser.add_argument("--area", type=float, default=None)
    parser.add_argument("--layer", action="append", default=[])
    parser.add_argument("--estimate-magnetic-volume", action="store_true")
    parser.add_argument("--save-estimated-volume", action="store_true")
    parser.add_argument("--no-volume-prompt", action="store_true")


def _add_sample_metadata_paths(
    parser: argparse.ArgumentParser, *, include_processed: bool = True
) -> None:
    parser.add_argument("--sample-registry", type=Path, default=DEFAULT_SAMPLE_REGISTRY_FILE)
    parser.add_argument("--measurement-ledger", type=Path, default=DEFAULT_MEASUREMENT_LEDGER_FILE)
    if include_processed:
        parser.add_argument("--processed-ledger", type=Path, default=DEFAULT_PROCESSED_LEDGER_FILE)


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
    parser.add_argument(
        "--metrics-from", choices=("reconstructed_absorption",), default="reconstructed_absorption"
    )
    parser.add_argument("--export-resonance-metrics", action="store_true")
    parser.add_argument("--plot-halfmax-markers", action="store_true")
    parser.add_argument("--plot-area-windows", action="store_true")


def _add_fmr_field_polarity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--field-polarity-correction",
        choices=("none", "gonzalez-fuentes"),
        default=None,
    )
    parser.add_argument("--pair-field-polarities", action="store_true")
    parser.add_argument(
        "--fit-field", choices=("Hres", "Hres_avg", "Hres_pos", "Hres_neg"), default=None
    )
    parser.add_argument(
        "--require-polarity-pair",
        dest="require_polarity_pair",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--on-unpaired-polarity",
        choices=("warn_and_keep_raw", "drop", "fail"),
        default=None,
    )
    parser.add_argument("--polarity-column", default=None)
    parser.add_argument("--positive-polarity-labels", default=None)
    parser.add_argument("--negative-polarity-labels", default=None)
    parser.add_argument("--pair-by", default=None)
    parser.add_argument("--max-pair-frequency-tolerance-ghz", type=float, default=None)
    parser.add_argument("--max-pair-hres-split-mT", type=float, default=None)
    parser.add_argument("--compare-polarity-fits", action="store_true")
    parser.add_argument("--plot-polarity-diagnostics", action="store_true")


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


def _run_sample_command(args: argparse.Namespace) -> int:
    registry_path = args.sample_registry.resolve()
    if args.sample_verb == "add":
        return _run_sample_add(args, registry_path)
    if args.sample_verb == "update":
        return _run_sample_update(args, registry_path)
    if args.sample_verb == "list":
        return _run_sample_list(registry_path)
    if args.sample_verb == "show":
        return _run_sample_show(registry_path, args.sample_id)
    if args.sample_verb == "register-file":
        return _run_sample_register_file(args, registry_path)
    if args.sample_verb == "validate":
        return _run_sample_validate(registry_path)
    if args.sample_verb == "readiness":
        return _run_sample_readiness(args, registry_path)
    if args.sample_verb == "analyze":
        return _run_sample_analyze(args, registry_path)
    if args.sample_verb == "analyze-batch":
        return _run_sample_analyze_batch(args, registry_path)
    raise SystemExit(f"Unsupported sample verb: {args.sample_verb}")


def _run_sample_add(args: argparse.Namespace, registry_path: Path) -> int:
    registry = _load_or_empty_registry(registry_path)
    sample_id = args.sample_id or (_prompt("Sample ID") if args.interactive else None)
    if not sample_id:
        raise WorkflowError("sample add requires SAMPLE_ID unless --interactive is used.")
    sample = RegistrySampleRecord(
        sample_id=sample_id,
        aliases=list(args.alias or []),
        condition=args.condition,
        replicate=args.replicate,
        stack=args.stack,
        geometry=VolumeMetadata(
            area=QuantityMetadata(args.area_value, args.area_unit, args.area_uncertainty),
            magnetic_thickness=QuantityMetadata(
                args.thickness_value,
                args.thickness_unit,
                args.thickness_uncertainty,
            ),
            vmag=DirectVolumeMetadata(
                args.vmag_value,
                args.vmag_unit,
                args.vmag_uncertainty,
                args.vmag_method,
            ),
        ),
        defaults=AnalysisDefaults(
            g_mode=args.g_mode,
            g_value=args.g_value,
            ms_source=args.ms_source,
        ),
    )
    _apply_volume_args(sample, args, interactive=args.interactive)
    add_sample(registry, sample)
    save_registry(registry, registry_path)
    print(f"Added sample {sample.sample_id} to {registry_path}")
    return 0


def _run_sample_update(args: argparse.Namespace, registry_path: Path) -> int:
    registry = load_registry(registry_path)
    sample = find_sample(registry, args.sample_id)
    if sample is None:
        raise WorkflowError(f"Unknown sample_id or alias: {args.sample_id}")
    for alias in args.alias or []:
        if alias not in sample.aliases:
            sample.aliases.append(alias)
    if args.condition is not None:
        sample.condition = args.condition
    if args.replicate is not None:
        sample.replicate = args.replicate
    if args.stack is not None:
        sample.stack = args.stack
    if any(value is not None for value in (args.area_value, args.area_unit, args.area_uncertainty)):
        sample.geometry.area = QuantityMetadata(
            args.area_value,
            args.area_unit,
            args.area_uncertainty,
        )
    if any(
        value is not None
        for value in (args.thickness_value, args.thickness_unit, args.thickness_uncertainty)
    ):
        sample.geometry.magnetic_thickness = QuantityMetadata(
            args.thickness_value,
            args.thickness_unit,
            args.thickness_uncertainty,
        )
    if any(
        value is not None
        for value in (args.vmag_value, args.vmag_unit, args.vmag_uncertainty, args.vmag_method)
    ):
        sample.geometry.vmag = DirectVolumeMetadata(
            args.vmag_value,
            args.vmag_unit,
            args.vmag_uncertainty,
            args.vmag_method,
        )
    if args.g_mode is not None:
        sample.defaults.g_mode = args.g_mode
    if args.g_value is not None:
        sample.defaults.g_value = args.g_value
    if args.ms_source is not None:
        sample.defaults.ms_source = args.ms_source
    _apply_volume_args(sample, args, interactive=args.interactive)
    save_registry(registry, registry_path)
    print(f"Updated sample {sample.sample_id} in {registry_path}")
    return 0


def _run_sample_list(registry_path: Path) -> int:
    registry = _load_or_empty_registry(registry_path)
    if not registry.samples:
        print("No samples registered.")
        return 0
    for sample in sorted(registry.samples.values(), key=lambda item: item.sample_id.lower()):
        aliases = "" if not sample.aliases else f" aliases={','.join(sample.aliases)}"
        print(f"{sample.sample_id}{aliases}")
    return 0


def _run_sample_show(registry_path: Path, sample_id: str) -> int:
    registry = load_registry(registry_path)
    sample = find_sample(registry, sample_id)
    if sample is None:
        raise WorkflowError(f"Unknown sample_id or alias: {sample_id}")
    print(yaml.safe_dump(sample_to_dict(sample), sort_keys=False).rstrip())
    return 0


def _run_sample_register_file(args: argparse.Namespace, registry_path: Path) -> int:
    registry = _load_or_empty_registry(registry_path)
    measurement_ledger_path = args.measurement_ledger.resolve()
    sample_id = args.sample_id
    if not sample_id and args.interactive:
        sample_id = _prompt("Sample ID")
    if not sample_id:
        raise WorkflowError(
            "sample register-file requires --sample-id unless --interactive is used."
        )
    if find_sample(registry, sample_id) is None and args.interactive:
        add_sample(registry, RegistrySampleRecord(sample_id=sample_id))
        save_registry(registry, registry_path)
    if find_sample(registry, sample_id) is None:
        raise WorkflowError(f"Sample {sample_id!r} does not exist in {registry_path}.")
    branch_labels = [part.strip() for part in args.branch_labels.split(",") if part.strip()]
    ledger = (
        load_measurement_ledger(measurement_ledger_path)
        if measurement_ledger_path.exists()
        else empty_measurement_ledger()
    )
    measurement = upsert_measurement_record(
        ledger,
        measurement_id=args.measurement_id or f"{args.type}:{sample_id}:{args.path.stem}",
        sample_id=sample_id,
        measurement_type=args.type,
        raw_path=args.path.resolve(),
        geometry=args.geometry,
        branch_labels=branch_labels,
        instrument=args.instrument,
        notes=args.notes,
        base_dir=measurement_ledger_path.parent,
    )
    save_measurement_ledger(ledger, measurement_ledger_path)
    print(
        f"Registered {measurement.raw_path} as {measurement.measurement_id} "
        f"for {measurement.sample_id}"
    )
    return 0


def _run_sample_validate(registry_path: Path) -> int:
    registry = load_registry(registry_path)
    messages = validate_registry(registry, registry_base_dir=registry_path.parent)
    if not messages:
        print("Sample registry is valid.")
        return 0
    for message in messages:
        scope = message.sample_id or "registry"
        if message.measurement_id:
            scope = f"{scope}/{message.measurement_id}"
        print(f"{message.severity.upper()} {message.code} {scope}: {message.message}")
    return 1 if any(message.severity == "error" for message in messages) else 0


def _run_sample_readiness(args: argparse.Namespace, registry_path: Path) -> int:
    result = build_sample_readiness(
        sample_id=args.sample_id,
        registry_path=registry_path,
        measurement_ledger_path=args.measurement_ledger.resolve(),
        processed_ledger_path=args.processed_ledger.resolve(),
        recipe_path=args.recipe.resolve(),
    )
    summary = result["summary"]
    print(f"Sample: {summary['sample_id']}")
    print(f"Readiness: {summary['readiness']}")
    print(f"Usable processed inputs: {summary['usable_processed_inputs']}")
    print(f"Warnings: {len(result['warnings'])}")
    for label, ready in result["readiness_matrix"].items():
        print(f"{label}: {ready}")
    return 0


def _run_sample_analyze(args: argparse.Namespace, registry_path: Path) -> int:
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DEFAULT_DERIVED_OUTPUT_ROOT / args.sample_id
    )
    result = analyze_sample(
        sample_id=args.sample_id,
        registry_path=registry_path,
        measurement_ledger_path=args.measurement_ledger.resolve(),
        processed_ledger_path=args.processed_ledger.resolve(),
        recipe_path=args.recipe.resolve(),
        output_dir=output_dir,
    )
    print(f"Sample: {result.sample_id}")
    print(f"Readiness: {result.result['summary']['readiness']}")
    print(f"Warnings: {len(result.result['warnings'])}")
    print(f"Output folder: {result.output_dir}")
    print(f"Summary JSON: {result.artifacts['summary_json']}")
    return 0


def _run_sample_analyze_batch(args: argparse.Namespace, registry_path: Path) -> int:
    results = analyze_sample_batch(
        registry_path=registry_path,
        measurement_ledger_path=args.measurement_ledger.resolve(),
        processed_ledger_path=args.processed_ledger.resolve(),
        recipe_path=args.recipe.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"Analyzed samples: {len(results)}")
    print(f"Output folder: {args.output_dir.resolve()}")
    for result in results:
        readiness = result.result["summary"]["readiness"]
        warning_count = len(result.result["warnings"])
        print(f"{result.sample_id}: {readiness} warnings={warning_count}")
    return 0


def _load_or_empty_registry(path: Path):
    return load_registry(path) if path.exists() else empty_registry()


def _prompt(label: str) -> str:
    return input(f"{label}: ").strip()


def _prompt_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _prompt_yes_no(label: str, *, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    value = input(f"{label} [{suffix}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "true", "1"}


def _apply_volume_args(
    sample: RegistrySampleRecord,
    args: argparse.Namespace,
    *,
    interactive: bool,
) -> None:
    if _should_prompt_for_volume(args, interactive=interactive):
        _prompt_for_volume_metadata(sample)
        return

    if args.magnetic_volume_m3 is not None:
        if args.magnetic_volume_m3 <= 0.0:
            raise WorkflowError("--magnetic-volume-m3 must be greater than zero.")
        sample.magnetic_volume_m3 = args.magnetic_volume_m3
        sample.magnetic_volume_source = "manual"
        sample.magnetic_volume_method = "manual"

    if args.geometry_shape:
        sample.geometry.shape = args.geometry_shape
        sample.geometry.dimensions = _geometry_dimensions_from_args(args)

    if args.layer:
        sample.layer_stack = [_parse_layer_argument(item) for item in args.layer]

    if args.estimate_magnetic_volume or args.save_estimated_volume:
        resolution = resolve_sample_magnetic_volume(sample)
        _print_magnetic_volume_resolution(resolution)
        if args.save_estimated_volume:
            if not resolution.is_available:
                raise WorkflowError(
                    "Cannot save estimated magnetic volume because the estimate is unavailable: "
                    + "; ".join(resolution.warnings)
                )
            update_sample_magnetic_volume_fields(sample, resolution)
        elif resolution.is_available:
            sample.magnetic_volume_warnings = [
                *resolution.warnings,
                "Estimated magnetic volume was not saved.",
            ]


def _should_prompt_for_volume(args: argparse.Namespace, *, interactive: bool) -> bool:
    if not interactive or bool(getattr(args, "no_volume_prompt", False)):
        return False
    return not _has_volume_args(args)


def _has_volume_args(args: argparse.Namespace) -> bool:
    return any(
        [
            args.magnetic_volume_m3 is not None,
            args.geometry_shape is not None,
            args.length is not None,
            args.width is not None,
            args.side is not None,
            args.diameter is not None,
            args.radius is not None,
            args.area is not None,
            bool(args.layer),
            args.estimate_magnetic_volume,
            args.save_estimated_volume,
        ]
    )


def _geometry_dimensions_from_args(args: argparse.Namespace) -> dict[str, object]:
    shape = args.geometry_shape
    if shape == "rectangle":
        return _required_dimensions(
            {"length": args.length, "width": args.width, "unit": args.length_unit},
            "--length, --width, and --length-unit are required for rectangle geometry.",
        )
    if shape == "square":
        return _required_dimensions(
            {"side": args.side, "unit": args.length_unit},
            "--side and --length-unit are required for square geometry.",
        )
    if shape == "circle":
        if args.diameter is None and args.radius is None:
            raise WorkflowError("--diameter or --radius is required for circle geometry.")
        if not args.length_unit:
            raise WorkflowError("--length-unit is required for circle geometry.")
        dimensions: dict[str, object] = {"unit": args.length_unit}
        if args.diameter is not None:
            dimensions["diameter"] = args.diameter
        if args.radius is not None:
            dimensions["radius"] = args.radius
        return dimensions
    if shape == "custom_area":
        return _required_dimensions(
            {"area": args.area, "area_unit": args.area_unit},
            "--area and --area-unit are required for custom_area geometry.",
        )
    if shape == "array":
        raise WorkflowError(
            "Array geometry is supported in registry YAML but not yet by CLI flags."
        )
    return {}


def _required_dimensions(payload: dict[str, object], message: str) -> dict[str, object]:
    if any(value in {None, ""} for value in payload.values()):
        raise WorkflowError(message)
    return payload


def _parse_layer_argument(value: str) -> dict[str, object]:
    parts = value.split(":")
    if len(parts) != 4:
        raise WorkflowError(
            "--layer must use material:thickness:unit:magnetic, for example NiFe:7:nm:true."
        )
    material, thickness_text, unit, magnetic_text = [part.strip() for part in parts]
    if not material:
        raise WorkflowError("--layer material must not be empty.")
    try:
        thickness = float(thickness_text)
    except ValueError as exc:
        raise WorkflowError("--layer thickness must be numeric.") from exc
    if thickness <= 0.0:
        raise WorkflowError("--layer thickness must be greater than zero.")
    if not unit:
        raise WorkflowError("--layer unit must not be empty.")
    return {
        "material": material,
        "thickness": thickness,
        "thickness_unit": unit,
        "magnetic": _parse_layer_bool(magnetic_text),
    }


def _parse_layer_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in {"true", "yes", "y", "1", "magnetic"}:
        return True
    if text in {"false", "no", "n", "0", "non_magnetic", "non-magnetic"}:
        return False
    raise WorkflowError(
        "--layer magnetic field must be true/false, yes/no, 1/0, or magnetic/non_magnetic."
    )


def _prompt_for_volume_metadata(sample: RegistrySampleRecord) -> None:
    mode = (
        _prompt_default(
            "Magnetic volume is optional. Enter manually, estimate from geometry/layers, or skip?",
            "skip",
        )
        .strip()
        .lower()
    )
    if mode in {"", "skip", "s"}:
        return
    if mode in {"manual", "m"}:
        value_text = _prompt("magnetic_volume_m3")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise WorkflowError("magnetic_volume_m3 must be numeric.") from exc
        if value <= 0.0:
            raise WorkflowError("magnetic_volume_m3 must be greater than zero.")
        sample.magnetic_volume_m3 = value
        sample.magnetic_volume_source = "manual"
        sample.magnetic_volume_method = "manual"
        return
    if mode not in {"estimate", "e"}:
        raise WorkflowError("Volume entry must be manual, estimate, or skip.")
    _prompt_for_estimated_volume(sample)


def _prompt_for_estimated_volume(sample: RegistrySampleRecord) -> None:
    shape = _prompt_default(
        "Geometry shape [rectangle/square/circle/custom_area]",
        "rectangle",
    )
    sample.geometry.shape = shape
    if shape == "rectangle":
        sample.geometry.dimensions = {
            "length": float(_prompt("Length")),
            "width": float(_prompt("Width")),
            "unit": _prompt_default("Length unit", "mm"),
        }
    elif shape == "square":
        sample.geometry.dimensions = {
            "side": float(_prompt("Side")),
            "unit": _prompt_default("Length unit", "mm"),
        }
    elif shape == "circle":
        sample.geometry.dimensions = {
            "diameter": float(_prompt("Diameter")),
            "unit": _prompt_default("Length unit", "mm"),
        }
    elif shape == "custom_area":
        sample.geometry.dimensions = {
            "area": float(_prompt("Area")),
            "area_unit": _prompt_default("Area unit", "mm^2"),
        }
    else:
        raise WorkflowError(
            "Interactive volume estimate supports rectangle, square, circle, or custom_area."
        )
    layers: list[dict[str, object]] = []
    while True:
        layer = _prompt_default("Layer material:thickness:unit:magnetic (blank to finish)", "")
        if not layer:
            break
        layers.append(_parse_layer_argument(layer))
    sample.layer_stack = layers
    resolution = resolve_sample_magnetic_volume(sample)
    _print_magnetic_volume_resolution(resolution)
    if resolution.is_available and _prompt_yes_no("Save estimated magnetic volume?", default=True):
        update_sample_magnetic_volume_fields(sample, resolution)


def _print_magnetic_volume_resolution(resolution) -> None:
    if not resolution.is_available:
        print("Magnetic volume unavailable.")
        for warning in resolution.warnings:
            print(f"Warning: {warning}")
        return
    print(f"area_m2: {resolution.area_m2}")
    print(f"magnetic_thickness_total_m: {resolution.magnetic_thickness_total_m}")
    print(f"magnetic_volume_m3: {resolution.magnetic_volume_m3}")
    print(
        "included magnetic layers: "
        + ", ".join(str(layer.get("material")) for layer in resolution.included_layers)
    )
    print(
        "excluded nonmagnetic layers: "
        + ", ".join(str(layer.get("material")) for layer in resolution.excluded_layers)
    )
    for warning in resolution.warnings:
        print(f"Warning: {warning}")


def _run_modality_single(spec: ModalityCliSpec, args: argparse.Namespace) -> int:
    original_source_file = resolve_single_source(
        args.input_path,
        allowed_suffixes=spec.allowed_suffixes,
        pattern=spec.default_pattern,
        recursive=False,
        source_label=spec.source_label,
    )
    import_result = _maybe_import_raw_source(original_source_file, spec.name, args)
    source_file = import_result.imported_path
    _maybe_interactive_register_analysis_file(args, spec.name, source_file)
    resolved_output_dir = (
        args.output_dir.resolve() if args.output_dir else DEFAULT_OUTPUT_ROOT / source_file.stem
    )
    workflow_options = _workflow_options(spec.name, args)
    _apply_import_result_to_registry_options(workflow_options, import_result)
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
    workflow_options["raw_import_modality"] = spec.name
    workflow_options["raw_import_root"] = args.raw_import_root.resolve()
    if args.metadata_manifest is not None:
        workflow_options["metadata_by_source"] = _load_metadata_manifest(
            args.metadata_manifest.resolve()
        )
    elif getattr(args, "sample_id", None):
        print(
            "WARNING: --sample-id was passed to batch without --metadata-manifest; "
            "all files are assigned to the same sample."
        )
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


def _maybe_import_raw_source(
    source_file: Path, modality: str, args: argparse.Namespace
) -> RawImportResult:
    if not bool(getattr(args, "update_ledger", False)):
        return RawImportResult(
            original_path=source_file.resolve(),
            imported_path=source_file.resolve(),
            copied=False,
            status="unchanged",
            message="ledger update disabled",
        )
    return import_raw_for_ledger(
        source_file,
        modality=modality,
        raw_import_root=args.raw_import_root.resolve(),
    )


def _apply_import_result_to_registry_options(
    workflow_options: dict[str, object],
    import_result: RawImportResult,
) -> None:
    registry_options = workflow_options.get("registry_options")
    if not isinstance(registry_options, RegistryWorkflowOptions):
        return
    registry_options.original_source_path = (
        import_result.original_path
        if import_result.original_path != import_result.imported_path
        else None
    )
    registry_options.raw_import_copied = import_result.copied
    registry_options.raw_import_sidecars = list(import_result.copied_sidecars)
    registry_options.raw_import_message = import_result.message


def _load_metadata_manifest(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_path = Path(str(row.get("raw_path") or "")).resolve()
            branch_labels = [
                part.strip()
                for part in str(row.get("branch_labels") or "").split(",")
                if part.strip()
            ]
            rows[str(raw_path).lower()] = {
                "sample_id": row.get("sample_id") or None,
                "measurement_id": row.get("measurement_id") or None,
                "geometry": row.get("geometry") or None,
                "branch_labels": branch_labels,
            }
    return rows


def _maybe_interactive_register_analysis_file(
    args: argparse.Namespace,
    modality: str,
    source_file: Path,
) -> None:
    if not bool(getattr(args, "interactive", False)) or not bool(
        getattr(args, "update_ledger", False)
    ):
        return
    sample_id = getattr(args, "sample_id", None) or _prompt("Sample ID")
    if not sample_id:
        raise WorkflowError("--interactive analysis requires a sample ID.")
    args.sample_id = sample_id
    if not getattr(args, "measurement_id", None):
        args.measurement_id = _prompt_default(
            "Measurement ID", f"{modality}:{sample_id}:{source_file.stem}"
        )
    if modality in {"fmr", "esr"} and not getattr(args, "geometry", None):
        args.geometry = _prompt_default("Geometry [ip/oop/angular/unknown]", "unknown")
    if not getattr(args, "branch_labels", ""):
        args.branch_labels = _prompt_default("Branch labels (comma-separated, optional)", "")
    if not bool(getattr(args, "create_sample", False)):
        args.create_sample = _prompt_yes_no("Create sample if missing?", default=False)
    if not bool(getattr(args, "mark_canonical", False)):
        args.mark_canonical = _prompt_yes_no("Mark processed result canonical?", default=True)
    if not bool(getattr(args, "replace_canonical", False)):
        args.replace_canonical = _prompt_yes_no("Replace existing canonical result?", default=False)


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
    artifacts = spec.export_from_json(
        input_path.resolve(), None if output_dir is None else output_dir.resolve()
    )
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


def _workflow_options(
    modality: str, args: argparse.Namespace, *, batch: bool = False
) -> dict[str, object]:
    options: dict[str, object] = {"registry_options": _build_registry_workflow_options(args)}
    if modality not in {"esr", "fmr"}:
        return options
    options["resonance_metrics_config"] = _build_resonance_metrics_config(args)
    if modality == "esr":
        options["fit_mode"] = getattr(args, "fit_mode", None)
        if not batch:
            options["show_raw"] = bool(getattr(args, "show_raw", False))
    if modality == "fmr":
        options["fmr_recipe_overrides"] = _build_fmr_recipe_overrides(args)
    return options


def _build_fmr_recipe_overrides(args: argparse.Namespace) -> dict[str, object]:
    names = [
        "field_polarity_correction",
        "pair_field_polarities",
        "fit_field",
        "require_polarity_pair",
        "on_unpaired_polarity",
        "polarity_column",
        "positive_polarity_labels",
        "negative_polarity_labels",
        "pair_by",
        "max_pair_frequency_tolerance_ghz",
        "max_pair_hres_split_mT",
        "compare_polarity_fits",
        "plot_polarity_diagnostics",
    ]
    overrides: dict[str, object] = {}
    for name in names:
        value = getattr(args, name, None)
        if value is None:
            continue
        if (
            name in {"pair_field_polarities", "compare_polarity_fits", "plot_polarity_diagnostics"}
            and not value
        ):
            continue
        overrides[name] = value
    return overrides


def _build_registry_workflow_options(args: argparse.Namespace) -> RegistryWorkflowOptions:
    branch_labels = getattr(args, "branch_labels", "")
    if isinstance(branch_labels, str):
        branch_labels = [part.strip() for part in branch_labels.split(",") if part.strip()]
    return RegistryWorkflowOptions(
        sample_registry_path=getattr(
            args, "sample_registry", DEFAULT_SAMPLE_REGISTRY_FILE
        ).resolve(),
        measurement_ledger_path=getattr(
            args, "measurement_ledger", DEFAULT_MEASUREMENT_LEDGER_FILE
        ).resolve(),
        processed_ledger_path=getattr(
            args, "processed_ledger", DEFAULT_PROCESSED_LEDGER_FILE
        ).resolve(),
        sample_id=getattr(args, "sample_id", None),
        measurement_id=getattr(args, "measurement_id", None),
        geometry=getattr(args, "geometry", None),
        branch_labels=list(branch_labels),
        instrument=getattr(args, "instrument", None),
        notes=getattr(args, "notes", None),
        g_mode=getattr(args, "g_mode", None),
        g_value=getattr(args, "g_value", None),
        update_ledger=bool(getattr(args, "update_ledger", False)),
        create_sample=bool(getattr(args, "create_sample", False)),
        mark_canonical=bool(getattr(args, "mark_canonical", False)),
        replace_canonical=bool(getattr(args, "replace_canonical", False)),
        interactive=bool(getattr(args, "interactive", False)),
        raw_import_root=getattr(args, "raw_import_root", DEFAULT_RAW_IMPORT_ROOT).resolve(),
    )


def _build_resonance_metrics_config(args: argparse.Namespace) -> ResonanceMetricsConfig:
    return ResonanceMetricsConfig(
        compute_resonance_metrics=bool(getattr(args, "compute_resonance_metrics", True)),
        area_window_mode=getattr(args, "area_window_mode", "side-aware"),
        area_window_multipliers=parse_area_window_multipliers(
            getattr(args, "area_window_multipliers", "1,2,3")
        ),
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
    registry_options: RegistryWorkflowOptions,
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
    resolved_output_dir = (
        output_dir.resolve() if output_dir else DEFAULT_OUTPUT_ROOT / source_file.stem
    )
    interactive_args = argparse.Namespace(
        interactive=registry_options.interactive,
        update_ledger=registry_options.update_ledger,
        sample_id=registry_options.sample_id,
        measurement_id=registry_options.measurement_id,
        geometry=registry_options.geometry,
        branch_labels=",".join(registry_options.branch_labels),
        create_sample=registry_options.create_sample,
        mark_canonical=registry_options.mark_canonical,
        replace_canonical=registry_options.replace_canonical,
    )
    _maybe_interactive_register_analysis_file(interactive_args, "esr", source_file)
    registry_options.sample_id = interactive_args.sample_id
    registry_options.measurement_id = interactive_args.measurement_id
    registry_options.geometry = interactive_args.geometry
    registry_options.branch_labels = [
        part.strip() for part in interactive_args.branch_labels.split(",") if part.strip()
    ]
    registry_options.create_sample = interactive_args.create_sample
    registry_options.mark_canonical = interactive_args.mark_canonical
    registry_options.replace_canonical = interactive_args.replace_canonical
    analysis, artifacts = run_esr_single_file_workflow(
        source_path=source_file,
        recipe_path=recipe_path.resolve(),
        output_dir=resolved_output_dir,
        fit_mode=fit_mode,
        show_raw=show_raw,
        resonance_metrics_config=resonance_metrics_config,
        registry_options=registry_options,
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
    registry_options: RegistryWorkflowOptions,
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
        registry_options=registry_options,
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
    registry_options: RegistryWorkflowOptions,
) -> int:
    resolved_source_file = source_file.resolve()
    resolved_output_dir = (
        output_dir.resolve() if output_dir else DEFAULT_OUTPUT_ROOT / resolved_source_file.stem
    )
    interactive_args = argparse.Namespace(
        interactive=registry_options.interactive,
        update_ledger=registry_options.update_ledger,
        sample_id=registry_options.sample_id,
        measurement_id=registry_options.measurement_id,
        geometry=registry_options.geometry,
        branch_labels=",".join(registry_options.branch_labels),
        create_sample=registry_options.create_sample,
        mark_canonical=registry_options.mark_canonical,
        replace_canonical=registry_options.replace_canonical,
    )
    _maybe_interactive_register_analysis_file(interactive_args, "esr", resolved_source_file)
    registry_options.sample_id = interactive_args.sample_id
    registry_options.measurement_id = interactive_args.measurement_id
    registry_options.geometry = interactive_args.geometry
    registry_options.branch_labels = [
        part.strip() for part in interactive_args.branch_labels.split(",") if part.strip()
    ]
    registry_options.create_sample = interactive_args.create_sample
    registry_options.mark_canonical = interactive_args.mark_canonical
    registry_options.replace_canonical = interactive_args.replace_canonical
    analysis, artifacts = run_esr_single_file_workflow(
        source_path=resolved_source_file,
        recipe_path=recipe_path.resolve(),
        output_dir=resolved_output_dir,
        fit_mode=fit_mode,
        show_raw=show_raw,
        resonance_metrics_config=resonance_metrics_config,
        registry_options=registry_options,
    )
    _print_single_result("esr", analysis, artifacts)
    return 0


def _print_single_result(modality: str, analysis, artifacts: WorkflowArtifacts) -> None:
    if modality == "esr":
        point_count = analysis.dataset.field_mT.size
        print(f"Loaded {analysis.dataset.source_path.name} with {point_count} points")
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
        diagnostic_label = (
            "NA" if diagnostic_area_integral is None else f"{diagnostic_area_integral:.6f}"
        )
        print(f"Primary fit-derived area integral: {total_label}")
        print(f"Diagnostic full-span area integral: {diagnostic_label}")
    elif modality == "fmr":
        summary = analysis.summary_metrics
        print(
            f"Loaded {analysis.measurement.source_path.name} with {summary['trace_count']} trace(s)"
        )
        print(f"Sample: {summary['sample_id']}  Replicate: {summary.get('replicate_id')}")
        accepted_trace_count = summary.get("accepted_trace_count")
        trace_count = summary.get("trace_count")
        print(
            f"Accepted traces: {accepted_trace_count} / {trace_count}  "
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
        print(
            f"Loaded {analysis.measurement.source_path.name} with {summary['point_count']} points"
        )
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
    if batch_result.unresolved_items:
        print(f"Unresolved: {len(batch_result.unresolved_items)}")
    print(f"Results folder: {batch_result.output_dir}")
    print(f"Batch summary: {batch_result.summary_csv_path}")
    print(f"Batch manifest: {batch_result.manifest_json_path}")
    if batch_result.resonance_metrics_csv_path is not None:
        print(f"Batch resonance metrics: {batch_result.resonance_metrics_csv_path}")
    if batch_result.unresolved_csv_path is not None:
        print(f"Unresolved files: {batch_result.unresolved_csv_path}")
    if batch_result.raw_import_map_path is not None:
        print(f"Raw import map: {batch_result.raw_import_map_path}")
    for name, path in sorted(batch_result.batch_figure_paths.items()):
        print(f"Batch figure [{name}]: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
