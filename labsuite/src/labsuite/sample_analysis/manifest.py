"""Manifest resolution for processed-ledger sample analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labsuite.core.exceptions import WorkflowError
from labsuite.core.sample_registry import (
    MeasurementLedger,
    MeasurementRecord,
    ProcessedResultRecord,
    SampleRecord,
    canonical_results_for_sample,
    find_sample,
    load_measurement_ledger,
    load_processed_ledger,
    load_registry,
    measurement_ledger_to_dict,
    processed_ledger_to_dict,
    registry_to_dict,
    sample_to_dict,
)


@dataclass(slots=True)
class ProcessedInput:
    measurement_id: str
    sample_id: str
    modality: str
    geometry: str
    branch_labels: list[str]
    raw_path: Path | None
    processed_json_path: Path | None = None
    status: str = "missing_canonical"
    warning_code: str | None = None
    message: str | None = None
    payload: dict[str, Any] | None = None
    result_id: str | None = None
    processed_record: ProcessedResultRecord | None = None
    measurement_record: MeasurementRecord | None = None
    candidates: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "measurement_id": self.measurement_id,
            "sample_id": self.sample_id,
            "modality": self.modality,
            "geometry": self.geometry,
            "branch_labels": list(self.branch_labels),
            "raw_path": None if self.raw_path is None else str(self.raw_path),
            "processed_json_path": None
            if self.processed_json_path is None
            else str(self.processed_json_path),
            "status": self.status,
            "warning_code": self.warning_code,
            "message": self.message,
            "candidates": [str(path) for path in self.candidates],
        }


@dataclass(slots=True)
class SampleAnalysisManifest:
    registry_path: Path
    measurement_ledger_path: Path
    processed_ledger_path: Path
    sample: SampleRecord
    registry_snapshot: dict[str, Any]
    measurement_ledger_snapshot: dict[str, Any]
    processed_ledger_snapshot: dict[str, Any]
    processed_inputs: list[ProcessedInput]
    warnings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sample_id(self) -> str:
        return self.sample.sample_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_path": str(self.registry_path),
            "measurement_ledger_path": str(self.measurement_ledger_path),
            "processed_ledger_path": str(self.processed_ledger_path),
            "sample": sample_to_dict(self.sample),
            "processed_inputs": [item.to_dict() for item in self.processed_inputs],
            "warnings": list(self.warnings),
        }


def build_sample_manifest(
    *,
    sample_id: str,
    registry_path: Path,
    measurement_ledger_path: Path,
    processed_ledger_path: Path,
) -> SampleAnalysisManifest:
    resolved_registry = registry_path.resolve()
    resolved_measurement = measurement_ledger_path.resolve()
    resolved_processed = processed_ledger_path.resolve()
    registry = load_registry(resolved_registry)
    sample = find_sample(registry, sample_id)
    if sample is None:
        raise WorkflowError(f"Unknown sample_id or alias: {sample_id}")
    measurement_ledger = load_measurement_ledger(resolved_measurement)
    processed_ledger = load_processed_ledger(resolved_processed)
    canonical = canonical_results_for_sample(processed_ledger, sample.sample_id)
    measurements_for_sample = {
        measurement.measurement_id: measurement
        for measurement in measurement_ledger.measurements.values()
        if measurement.sample_id == sample.sample_id
    }
    inputs = [
        _input_from_processed_record(
            result,
            measurement_ledger=measurement_ledger,
            measurement_ledger_path=resolved_measurement,
            processed_ledger_path=resolved_processed,
        )
        for result in sorted(
            canonical, key=lambda item: (item.type, item.measurement_id, item.result_id)
        )
    ]
    warnings = _missing_canonical_warnings(sample.sample_id, measurements_for_sample, inputs)
    return SampleAnalysisManifest(
        registry_path=resolved_registry,
        measurement_ledger_path=resolved_measurement,
        processed_ledger_path=resolved_processed,
        sample=sample,
        registry_snapshot=registry_to_dict(registry),
        measurement_ledger_snapshot=measurement_ledger_to_dict(measurement_ledger),
        processed_ledger_snapshot=processed_ledger_to_dict(processed_ledger),
        processed_inputs=inputs,
        warnings=warnings,
    )


def _input_from_processed_record(
    result: ProcessedResultRecord,
    *,
    measurement_ledger: MeasurementLedger,
    measurement_ledger_path: Path,
    processed_ledger_path: Path,
) -> ProcessedInput:
    measurement = measurement_ledger.measurements.get(result.measurement_id)
    processed_path = _resolve_metadata_path(result.processed_path, processed_ledger_path.parent)
    payload = _read_payload(processed_path)
    geometry = str(
        result.summary.get("geometry") or (measurement.geometry if measurement else "unknown")
    )
    branch_labels = result.summary.get("branches") or (
        measurement.branch_labels if measurement else []
    )
    raw_path = (
        None
        if measurement is None
        else _resolve_metadata_path(measurement.raw_path, measurement_ledger_path.parent)
    )
    item = ProcessedInput(
        result_id=result.result_id,
        measurement_id=result.measurement_id,
        sample_id=result.sample_id,
        modality=result.type,
        geometry=geometry,
        branch_labels=[str(label) for label in branch_labels],
        raw_path=raw_path,
        processed_json_path=processed_path,
        status="usable",
        payload=payload,
        processed_record=result,
        measurement_record=measurement,
    )
    if payload is None:
        item.status = "missing_canonical"
        item.warning_code = "MISSING_CANONICAL_PROCESSED_RESULT"
        item.message = (
            f"Canonical processed JSON does not exist or cannot be read: {result.processed_path}"
        )
    return item


def _missing_canonical_warnings(
    sample_id: str,
    measurements: dict[str, MeasurementRecord],
    inputs: list[ProcessedInput],
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    canonical_measurements = {item.measurement_id for item in inputs if item.status == "usable"}
    for measurement in sorted(measurements.values(), key=lambda item: item.measurement_id):
        if measurement.measurement_id in canonical_measurements:
            continue
        warnings.append(
            {
                "code": "MISSING_CANONICAL_PROCESSED_RESULT",
                "sample_id": sample_id,
                "measurement_id": measurement.measurement_id,
                "modality": measurement.type,
                "message": (
                    f"No canonical processed result is recorded for {measurement.measurement_id}."
                ),
            }
        )
    if not inputs and not measurements:
        warnings.append(
            {
                "code": "MISSING_CANONICAL_PROCESSED_RESULT",
                "sample_id": sample_id,
                "measurement_id": None,
                "modality": None,
                "message": f"Sample {sample_id} has no canonical processed results.",
            }
        )
    for item in inputs:
        if item.warning_code is not None:
            warnings.append(
                {
                    "code": item.warning_code,
                    "sample_id": sample_id,
                    "measurement_id": item.measurement_id,
                    "modality": item.modality,
                    "message": item.message,
                }
            )
    return warnings


def _resolve_metadata_path(path: str, base_dir: Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (base_dir / resolved).resolve()


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, MemoryError, json.JSONDecodeError):
        return None
