"""ESR batch QC, duplicate grouping, and best-run selection helpers."""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from labsuite.core.recipes import EsrPreprocessingRecipe
from labsuite.core.types import AnalysisResult, FitResult

_ANGLE_TOKEN_PATTERN = re.compile(r"(-?\d+(?:[.,]\d+)?)deg$", flags=re.IGNORECASE)
_REPLICATE_TOKEN_PATTERN = re.compile(r"R\d+$", flags=re.IGNORECASE)
_TIMESTAMP_PREFIX_PATTERN = re.compile(r"^\d{8}_\d+_")


@dataclass(slots=True, frozen=True)
class EsrBatchIdentity:
    """Duplicate-grouping identity for one ESR source file."""

    sample_id: str
    replicate_id: str | None
    nominal_angle_deg: float | None
    frequency_bucket_GHz: float | None

    @property
    def grouping_key(self) -> tuple[str, str | None, float | None, float | None]:
        return (
            self.sample_id,
            self.replicate_id,
            self.nominal_angle_deg,
            self.frequency_bucket_GHz,
        )

    @property
    def sample_frequency_key(self) -> tuple[str, float | None]:
        return (self.sample_id, self.frequency_bucket_GHz)


@dataclass(slots=True)
class EsrBatchQcRecord:
    """QC and duplicate-selection outcome for one ESR batch item."""

    source_path: Path
    source_stem: str
    identity: EsrBatchIdentity
    fit_success: bool
    nrmse: float | None
    r_squared: float | None
    snr: float | None
    edge_margin_peak1: float | None
    edge_margin_peak2: float | None
    uncertainty_score: float | None
    acquisition_timestamp: datetime | None
    selected_as_best: bool = False
    accepted_for_plot: bool = False
    reject_reason: str | None = None
    notes: list[str] = field(default_factory=list)
    selected_mode: str | None = None
    candidate_peak_count: int | None = None
    successful_analysis: bool = False

    @property
    def worst_edge_margin(self) -> float:
        finite_margins = [
            value for value in (self.edge_margin_peak1, self.edge_margin_peak2) if value is not None
        ]
        if not finite_margins:
            return float("-inf")
        return min(finite_margins)

    def summary_metrics(self) -> dict[str, object]:
        return {
            "sample_id": self.identity.sample_id,
            "replicate_id": self.identity.replicate_id,
            "nominal_angle_deg": self.identity.nominal_angle_deg,
            "frequency_GHz": self.identity.frequency_bucket_GHz,
            "selected_as_best": self.selected_as_best,
            "accepted_for_plot": self.accepted_for_plot,
            "reject_reason": self.reject_reason,
            "fit_success": self.fit_success,
            "nRMSE": self.nrmse,
            "R2": self.r_squared,
            "SNR": self.snr,
            "edge_margin_peak1": self.edge_margin_peak1,
            "edge_margin_peak2": self.edge_margin_peak2,
            "notes": "|".join(self.notes),
        }


def parse_esr_batch_identity(path: Path, metadata: dict[str, object] | None) -> EsrBatchIdentity:
    """Parse ESR grouping metadata from the source path and parsed metadata."""

    stem = path.stem.split(" - ", maxsplit=1)[0].strip()
    normalized_stem = _TIMESTAMP_PREFIX_PATTERN.sub("", stem)
    tokens = [token for token in normalized_stem.split("-") if token]

    angle_index: int | None = None
    replicate_id: str | None = None
    nominal_angle_deg: float | None = None
    for index, token in enumerate(tokens):
        if nominal_angle_deg is None:
            angle_match = _ANGLE_TOKEN_PATTERN.fullmatch(token)
            if angle_match is not None:
                nominal_angle_deg = float(angle_match.group(1).replace(",", "."))
                angle_index = index
                continue
        if replicate_id is None and _REPLICATE_TOKEN_PATTERN.fullmatch(token):
            replicate_id = token.upper()

    sample_tokens: list[str]
    if angle_index is not None:
        sample_tokens = tokens[:angle_index]
    elif replicate_id is not None:
        replicate_index = next(
            (index for index, token in enumerate(tokens) if token.upper() == replicate_id),
            None,
        )
        sample_tokens = tokens[:replicate_index] if replicate_index is not None else tokens
    else:
        sample_tokens = tokens

    sample_id = "-".join(sample_tokens).strip("-_ ")
    if not sample_id:
        sample_id = normalized_stem or stem

    bruker_payload = metadata.get("bruker") if isinstance(metadata, dict) else None
    bruker_metadata = dict(bruker_payload) if isinstance(bruker_payload, dict) else {}
    frequency_GHz = bruker_metadata.get("frequency_GHz")
    frequency_bucket_GHz = None if frequency_GHz is None else round(float(frequency_GHz), 3)

    return EsrBatchIdentity(
        sample_id=sample_id,
        replicate_id=replicate_id,
        nominal_angle_deg=nominal_angle_deg,
        frequency_bucket_GHz=frequency_bucket_GHz,
    )


def compute_esr_qc_metrics(
    analysis: AnalysisResult | None,
    *,
    source_path: Path,
    recipe: EsrPreprocessingRecipe,
    error_message: str | None = None,
) -> EsrBatchQcRecord:
    """Compute ESR batch QC metrics and acceptance for one batch item."""

    metadata = None if analysis is None else analysis.dataset.metadata
    identity = parse_esr_batch_identity(source_path, metadata)
    record = EsrBatchQcRecord(
        source_path=source_path.resolve(),
        source_stem=source_path.stem,
        identity=identity,
        fit_success=False,
        nrmse=None,
        r_squared=None,
        snr=None,
        edge_margin_peak1=None,
        edge_margin_peak2=None,
        uncertainty_score=None,
        acquisition_timestamp=_parse_acquisition_timestamp(source_path, metadata),
        successful_analysis=analysis is not None,
    )

    if analysis is None:
        record.reject_reason = "fit_failed"
        if error_message:
            record.notes.append(error_message)
        return record

    record.selected_mode = analysis.selected_mode
    record.candidate_peak_count = analysis.fit_decision.candidate_peak_count
    component_fits = _selected_component_fits(analysis)
    record.fit_success = _selected_fit_success(analysis, component_fits)
    record.r_squared = _selected_r_squared(analysis)
    record.nrmse = _normalized_rmse(analysis)
    record.snr = _selected_snr(analysis, component_fits)
    edge_margins = [_edge_margin(analysis, fit) for fit in component_fits]
    if edge_margins:
        record.edge_margin_peak1 = edge_margins[0]
    if len(edge_margins) > 1:
        record.edge_margin_peak2 = edge_margins[1]
    record.uncertainty_score = _uncertainty_score(component_fits)

    fit_local_reason = analysis.fit_local_disagreement_reason
    if fit_local_reason:
        record.notes.append(f"fit_local={fit_local_reason}")
    if analysis.selected_mode == "split":
        record.notes.append("mode=split")
    if analysis.fit_decision.candidate_peak_count is not None:
        record.notes.append(f"candidate_peaks={analysis.fit_decision.candidate_peak_count}")

    reject_reason = _qc_reject_reason(
        analysis=analysis,
        component_fits=component_fits,
        record=record,
        recipe=recipe,
    )
    record.reject_reason = reject_reason
    if reject_reason is None:
        record.accepted_for_plot = True
    return record


def group_duplicate_runs(
    records: Iterable[EsrBatchQcRecord],
) -> dict[tuple[str, str | None, float | None, float | None], list[EsrBatchQcRecord]]:
    """Group ESR QC records into duplicate buckets."""

    grouped: dict[tuple[str, str | None, float | None, float | None], list[EsrBatchQcRecord]] = {}
    for record in records:
        grouped.setdefault(record.identity.grouping_key, []).append(record)
    for group_records in grouped.values():
        group_records.sort(key=lambda item: str(item.source_path).lower())
    return grouped


def rank_group_candidates(records: Sequence[EsrBatchQcRecord]) -> list[EsrBatchQcRecord]:
    """Rank accepted duplicate candidates from best to worst."""

    accepted_records = [record for record in records if record.reject_reason is None]
    return sorted(accepted_records, key=_candidate_rank_key)


def select_best_runs(records: Sequence[EsrBatchQcRecord]) -> list[EsrBatchQcRecord]:
    """Select the single best accepted run from each duplicate group."""

    for record in records:
        record.selected_as_best = False
        record.accepted_for_plot = record.reject_reason is None

    for group_records in group_duplicate_runs(records).values():
        ranked = rank_group_candidates(group_records)
        if not ranked:
            continue
        selected = ranked[0]
        selected.selected_as_best = True
        selected.accepted_for_plot = True
        for superseded in ranked[1:]:
            superseded.selected_as_best = False
            superseded.accepted_for_plot = False
            superseded.reject_reason = "superseded_by_better_duplicate"
            superseded.notes.append(f"superseded_by={selected.source_path.name}")
    return list(records)


def export_esr_batch_qc_csv(records: Sequence[EsrBatchQcRecord], destination: Path) -> Path:
    """Export ESR batch QC rows to CSV."""

    fieldnames = [
        "file",
        "sample",
        "replicate",
        "angle",
        "frequency_GHz",
        "selected_as_best",
        "accepted_for_plot",
        "reject_reason",
        "fit_success",
        "nRMSE",
        "R2",
        "SNR",
        "edge_margin_peak1",
        "edge_margin_peak2",
        "notes",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: str(item.source_path).lower()):
            writer.writerow(
                {
                    "file": str(record.source_path),
                    "sample": record.identity.sample_id,
                    "replicate": record.identity.replicate_id,
                    "angle": record.identity.nominal_angle_deg,
                    "frequency_GHz": record.identity.frequency_bucket_GHz,
                    "selected_as_best": record.selected_as_best,
                    "accepted_for_plot": record.accepted_for_plot,
                    "reject_reason": record.reject_reason,
                    "fit_success": record.fit_success,
                    "nRMSE": record.nrmse,
                    "R2": record.r_squared,
                    "SNR": record.snr,
                    "edge_margin_peak1": record.edge_margin_peak1,
                    "edge_margin_peak2": record.edge_margin_peak2,
                    "notes": "|".join(record.notes),
                }
            )
    return destination


def build_multi_bucket_slug(
    identities: Sequence[EsrBatchIdentity],
) -> dict[tuple[str, float | None], str]:
    """Build stable figure slugs for sample/frequency buckets."""

    sample_frequency_keys = sorted(
        {identity.sample_frequency_key for identity in identities},
        key=lambda item: (
            item[0].lower(),
            float("inf") if item[1] is None else float(item[1]),
        ),
    )
    if len(sample_frequency_keys) <= 1:
        return {key: "" for key in sample_frequency_keys}

    slugs: dict[tuple[str, float | None], str] = {}
    for sample_id, frequency in sample_frequency_keys:
        sample_slug = re.sub(r"[^A-Za-z0-9]+", "_", sample_id).strip("_").lower() or "sample"
        frequency_slug = "na" if frequency is None else f"{frequency:.3f}".replace(".", "p")
        slugs[(sample_id, frequency)] = f"{sample_slug}_{frequency_slug}GHz"
    return slugs


def _selected_component_fits(analysis: AnalysisResult) -> list[FitResult]:
    if analysis.selected_mode == "split":
        peak_fits = sorted(
            analysis.peak_fits,
            key=lambda item: item.fit.parameters.get("center_mT", float("inf")),
        )
        return [peak_fit.fit for peak_fit in peak_fits]
    if analysis.single_fit is None:
        return []
    return [analysis.single_fit]


def _selected_fit_success(analysis: AnalysisResult, component_fits: Sequence[FitResult]) -> bool:
    if analysis.selected_mode == "split":
        return len(component_fits) >= 2 and all(_fit_is_valid(fit) for fit in component_fits)
    if analysis.single_fit is None:
        return False
    return _fit_is_valid(analysis.single_fit)


def _selected_r_squared(analysis: AnalysisResult) -> float | None:
    if analysis.selected_mode == "split":
        return _optional_float(analysis.fit_decision.metrics.get("split_r_squared"))
    if analysis.single_fit is None:
        return None
    return _optional_float(analysis.single_fit.metrics.get("r_squared"))


def _normalized_rmse(analysis: AnalysisResult) -> float | None:
    signal = np.asarray(analysis.processed.signal, dtype=float)
    if signal.size == 0:
        return None
    signal_scale = float(np.max(np.abs(signal)))
    residual_rmse = float(
        np.sqrt(np.mean(np.asarray(analysis.selected_residual, dtype=float) ** 2))
    )
    if signal_scale <= 1e-12:
        return 0.0 if residual_rmse <= 1e-12 else float("inf")
    return residual_rmse / signal_scale


def _selected_snr(analysis: AnalysisResult, component_fits: Sequence[FitResult]) -> float | None:
    residual = np.asarray(analysis.selected_residual, dtype=float)
    if residual.size == 0:
        return None
    residual_std = float(np.std(residual))
    if residual_std <= 1e-12:
        return float("inf")

    amplitudes: list[float] = []
    for fit in component_fits:
        amplitude = fit.parameters.get("amplitude")
        if amplitude is None:
            continue
        amplitudes.append(abs(float(amplitude)))
    if not amplitudes:
        return None
    selected_amplitude = min(amplitudes) if analysis.selected_mode == "split" else max(amplitudes)
    return selected_amplitude / residual_std


def _edge_margin(analysis: AnalysisResult, fit: FitResult) -> float | None:
    feature = fit.feature_summary
    if feature is None:
        return None

    field = np.asarray(analysis.processed.field_mT, dtype=float)
    if field.size == 0:
        return None
    field_min = float(np.min(field))
    field_max = float(np.max(field))

    positive_distance = min(
        abs(feature.positive_extremum_field_mT - field_min),
        abs(field_max - feature.positive_extremum_field_mT),
    )
    negative_distance = min(
        abs(feature.negative_extremum_field_mT - field_min),
        abs(field_max - feature.negative_extremum_field_mT),
    )
    return min(positive_distance, negative_distance)


def _uncertainty_score(component_fits: Sequence[FitResult]) -> float | None:
    scores: list[float] = []
    for fit in component_fits:
        center_diagnostic = fit.parameter_diagnostics.get("center_mT")
        gamma_diagnostic = fit.parameter_diagnostics.get("gamma_mT")
        if center_diagnostic is None or gamma_diagnostic is None:
            return None
        if center_diagnostic.relative_stderr is None or gamma_diagnostic.relative_stderr is None:
            return None
        scores.append(
            float(center_diagnostic.relative_stderr) + float(gamma_diagnostic.relative_stderr)
        )
    if not scores:
        return None
    return float(np.mean(scores))


def _qc_reject_reason(
    *,
    analysis: AnalysisResult,
    component_fits: Sequence[FitResult],
    record: EsrBatchQcRecord,
    recipe: EsrPreprocessingRecipe,
) -> str | None:
    field = np.asarray(analysis.processed.field_mT, dtype=float)
    if field.size == 0:
        return "missing_sweep_axis"
    field_min = float(np.min(field))
    field_max = float(np.max(field))

    if analysis.selected_mode == "split" and len(component_fits) < 2:
        return "second_peak_missing"

    if not record.fit_success:
        return "fit_failed"

    for index, fit in enumerate(component_fits, start=1):
        prefix = "" if analysis.selected_mode == "single" else f"peak{index}_"
        center = fit.parameters.get("center_mT")
        gamma = fit.parameters.get("gamma_mT")
        if center is None or gamma is None:
            return f"{prefix}missing_parameters"
        if not np.isfinite(center) or not np.isfinite(gamma) or float(gamma) <= 0.0:
            return (
                "second_peak_poorly_constrained"
                if index == 2 and analysis.selected_mode == "split"
                else f"{prefix}nonphysical_parameters"
            )
        if not field_min <= float(center) <= field_max:
            return (
                "second_peak_center_outside_sweep"
                if index == 2 and analysis.selected_mode == "split"
                else f"{prefix}center_outside_sweep"
            )

        edge_margin = record.edge_margin_peak1 if index == 1 else record.edge_margin_peak2
        guard = max(
            recipe.batch_qc_edge_guard_min_mT,
            recipe.batch_qc_edge_guard_gamma_multiplier * float(gamma),
        )
        if edge_margin is None or not np.isfinite(edge_margin) or edge_margin < guard:
            return (
                "second_peak_edge_truncated"
                if index == 2 and analysis.selected_mode == "split"
                else f"{prefix}edge_truncated"
            )

        if not fit.convergence.errorbars:
            return (
                "second_peak_poorly_constrained"
                if index == 2 and analysis.selected_mode == "split"
                else f"{prefix}uncertainty_unusable"
            )
        center_diagnostic = fit.parameter_diagnostics.get("center_mT")
        gamma_diagnostic = fit.parameter_diagnostics.get("gamma_mT")
        if not _usable_uncertainty(center_diagnostic) or not _usable_uncertainty(gamma_diagnostic):
            return (
                "second_peak_poorly_constrained"
                if index == 2 and analysis.selected_mode == "split"
                else f"{prefix}uncertainty_unusable"
            )

    if record.nrmse is None or not np.isfinite(record.nrmse):
        return "nrmse_unusable"
    if record.nrmse > recipe.batch_qc_nrmse_max:
        return "nrmse_exceeds_threshold"
    return None


def _fit_is_valid(fit: FitResult) -> bool:
    return bool(fit.success and fit.convergence.success and fit.derived.get("fit_valid", True))


def _usable_uncertainty(diagnostic) -> bool:
    if diagnostic is None:
        return False
    stderr = diagnostic.stderr
    return stderr is not None and np.isfinite(stderr) and float(stderr) > 0.0


def _candidate_rank_key(record: EsrBatchQcRecord) -> tuple[float, float, float, float, float, str]:
    timestamp_value = (
        float(record.acquisition_timestamp.timestamp())
        if record.acquisition_timestamp is not None
        else float("-inf")
    )
    nrmse = float("inf") if record.nrmse is None else float(record.nrmse)
    uncertainty = (
        float("inf") if record.uncertainty_score is None else float(record.uncertainty_score)
    )
    snr = float("-inf") if record.snr is None else float(record.snr)
    return (
        -record.worst_edge_margin,
        nrmse,
        uncertainty,
        -snr,
        -timestamp_value,
        str(record.source_path).lower(),
    )


def _parse_acquisition_timestamp(path: Path, metadata: dict[str, object] | None) -> datetime | None:
    if isinstance(metadata, dict):
        bruker_metadata = metadata.get("bruker")
        if isinstance(bruker_metadata, dict):
            raw_timestamp = bruker_metadata.get("timestamp")
            if isinstance(raw_timestamp, str):
                try:
                    return datetime.fromisoformat(raw_timestamp)
                except ValueError:
                    pass

    match = re.match(r"^(\d{8})_(\d{6,})", path.stem)
    if match is None:
        return None
    date_token = match.group(1)
    time_token = match.group(2)[:6]
    try:
        return datetime.strptime(f"{date_token}{time_token}", "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
