"""Project metadata registries and ledgers.

The v2 metadata model intentionally separates physical samples, raw
measurements, and processed analysis results.  Sample-level analysis consumes
processed ledger entries only; raw paths remain provenance for reprocessing.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from labsuite.core.exceptions import LabSuiteError
from labsuite.core.magnetic_volume import (
    MagneticVolumeError,
    estimate_from_area_and_thickness,
    estimate_magnetic_volume,
    normalize_volume_m3,
)

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - PyYAML is a project dependency
    yaml = None

SAMPLE_SCHEMA_VERSION = 2
MEASUREMENT_LEDGER_SCHEMA_VERSION = 1
PROCESSED_LEDGER_SCHEMA_VERSION = 1

DEFAULT_SAMPLE_REGISTRY_PATH = Path("metadata") / "sample_registry.yaml"
DEFAULT_MEASUREMENT_LEDGER_PATH = Path("metadata") / "measurement_ledger.yaml"
DEFAULT_PROCESSED_LEDGER_PATH = Path("metadata") / "processed_ledger.yaml"

MeasurementType = Literal["vsm", "fmr", "esr"]
MeasurementGeometry = Literal["ip", "oop", "angular", "unknown"]
GMode = Literal["fixed", "float", "bounded"]
MeasurementStatus = Literal["active", "archived"]
ProcessedStatus = Literal["canonical", "superseded", "test", "archived"]
MagneticVolumeSource = Literal["manual", "estimated", "imported", "unknown"]


class RegistryError(LabSuiteError):
    """Base class for metadata errors."""


class MetadataSchemaError(RegistryError):
    """Raised when metadata does not match the required schema."""


class RegistryFormatError(MetadataSchemaError):
    """Raised when a metadata YAML file cannot be loaded."""


class RegistryResolutionError(RegistryError):
    """Raised when workflow metadata cannot be resolved."""


class MissingSampleError(RegistryResolutionError):
    """Raised when a referenced sample is absent."""


class MissingLedgerMetadataError(RegistryResolutionError):
    """Raised when ledger update metadata is incomplete."""


class DuplicateCanonicalProcessedResultError(RegistryError):
    """Raised when more than one canonical result exists for one measurement."""


class CanonicalResultExistsError(RegistryError):
    """Raised when canonical replacement was not explicitly requested."""


@dataclass(slots=True)
class QuantityMetadata:
    value: float | None = None
    unit: str | None = None
    uncertainty: float | None = None

    def complete(self) -> bool:
        return self.value is not None and bool(self.unit)


@dataclass(slots=True)
class DirectVolumeMetadata(QuantityMetadata):
    method: str | None = None

    def complete(self) -> bool:
        return QuantityMetadata.complete(self) and bool(self.method)

    def partially_filled(self) -> bool:
        return (
            any(
                value is not None and value != ""
                for value in (self.value, self.unit, self.uncertainty, self.method)
            )
            and not self.complete()
        )


@dataclass(slots=True)
class VolumeMetadata:
    area: QuantityMetadata = field(default_factory=QuantityMetadata)
    magnetic_thickness: QuantityMetadata = field(default_factory=QuantityMetadata)
    vmag: DirectVolumeMetadata = field(default_factory=DirectVolumeMetadata)
    shape: str | None = None
    dimensions: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def has_complete_derived_volume_inputs(self) -> bool:
        return self.area.complete() and self.magnetic_thickness.complete()

    def has_complete_direct_volume(self) -> bool:
        return self.vmag.complete()


@dataclass(slots=True)
class AnalysisDefaults:
    g_mode: GMode = "float"
    g_value: float | None = None
    ms_source: str | None = None


@dataclass(slots=True)
class SampleRecord:
    sample_id: str
    aliases: list[str] = field(default_factory=list)
    condition: str | None = None
    replicate: str | None = None
    stack: str | None = None
    geometry: VolumeMetadata = field(default_factory=VolumeMetadata)
    layer_stack: list[dict[str, Any]] = field(default_factory=list)
    magnetic_volume_m3: float | None = None
    magnetic_volume_source: MagneticVolumeSource | str | None = None
    magnetic_volume_method: str | None = None
    magnetic_volume_warnings: list[str] = field(default_factory=list)
    defaults: AnalysisDefaults = field(default_factory=AnalysisDefaults)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MagneticVolumeResolution:
    magnetic_volume_m3: float | None = None
    source: MagneticVolumeSource | str = "unknown"
    method: str | None = None
    area_m2: float | None = None
    magnetic_thickness_total_m: float | None = None
    included_layers: list[dict[str, Any]] = field(default_factory=list)
    excluded_layers: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_estimated: bool = False
    is_available: bool = False


@dataclass(slots=True)
class MeasurementRecord:
    measurement_id: str
    sample_id: str
    type: MeasurementType
    raw_path: str
    geometry: MeasurementGeometry = "unknown"
    branch_labels: list[str] = field(default_factory=list)
    instrument: str | None = None
    notes: str | None = None
    status: MeasurementStatus = "active"

    @property
    def path(self) -> str:
        """Compatibility alias for callers that only need display paths."""

        return self.raw_path


@dataclass(slots=True)
class ProcessedResultRecord:
    result_id: str
    measurement_id: str
    sample_id: str
    type: MeasurementType
    processed_path: str
    recipe_path: str
    recipe_hash: str | None = None
    created_at: str = ""
    status: ProcessedStatus = "test"
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SampleRegistry:
    schema_version: int = SAMPLE_SCHEMA_VERSION
    samples: dict[str, SampleRecord] = field(default_factory=dict)


@dataclass(slots=True)
class MeasurementLedger:
    schema_version: int = MEASUREMENT_LEDGER_SCHEMA_VERSION
    measurements: dict[str, MeasurementRecord] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessedLedger:
    schema_version: int = PROCESSED_LEDGER_SCHEMA_VERSION
    processed_results: dict[str, ProcessedResultRecord] = field(default_factory=dict)


@dataclass(slots=True)
class LedgerPruneSummary:
    sample_id: str
    archived_measurements: int = 0
    archived_processed_results: int = 0
    archived_canonical_results: int = 0
    untouched_measurements: int = 0
    untouched_processed_results: int = 0
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationMessage:
    severity: Literal["warning", "error"]
    code: str
    message: str
    sample_id: str | None = None
    measurement_id: str | None = None


@dataclass(slots=True)
class AnalysisSampleContext:
    sample_registry_path: Path | None = None
    measurement_ledger_path: Path | None = None
    processed_ledger_path: Path | None = None
    sample: SampleRecord | None = None
    measurement: MeasurementRecord | None = None
    geometry: MeasurementGeometry = "unknown"
    g_mode: GMode = "float"
    g_value: float | None = None
    branch_labels: list[str] = field(default_factory=list)
    original_source_path: Path | None = None
    raw_import_copied: bool = False
    raw_import_sidecars: list[Path] = field(default_factory=list)
    raw_import_message: str | None = None
    validation_warnings: list[ValidationMessage] = field(default_factory=list)
    sample_registry_snapshot: dict[str, Any] | None = None
    measurement_ledger_snapshot: dict[str, Any] | None = None
    processed_ledger_snapshot: dict[str, Any] | None = None

    @property
    def registry_path(self) -> Path | None:
        return self.sample_registry_path

    @property
    def registry_snapshot(self) -> dict[str, Any] | None:
        return self.sample_registry_snapshot

    @property
    def sample_id(self) -> str | None:
        return None if self.sample is None else self.sample.sample_id

    @property
    def measurement_id(self) -> str | None:
        return None if self.measurement is None else self.measurement.measurement_id

    def to_dict(self) -> dict[str, Any]:
        return to_serializable(
            {
                "sample_registry_path": self.sample_registry_path,
                "measurement_ledger_path": self.measurement_ledger_path,
                "processed_ledger_path": self.processed_ledger_path,
                "sample_id": self.sample_id,
                "measurement_id": self.measurement_id,
                "geometry": self.geometry,
                "g_mode": self.g_mode,
                "g_value": self.g_value,
                "branch_labels": self.branch_labels,
                "original_source_path": self.original_source_path,
                "raw_import_copied": self.raw_import_copied,
                "raw_import_sidecars": self.raw_import_sidecars,
                "raw_import_message": self.raw_import_message,
                "sample": self.sample,
                "measurement": self.measurement,
                "validation_warnings": self.validation_warnings,
            }
        )


@dataclass(slots=True)
class RegistryWorkflowOptions:
    sample_registry_path: Path | None = None
    measurement_ledger_path: Path | None = None
    processed_ledger_path: Path | None = None
    sample_id: str | None = None
    measurement_id: str | None = None
    geometry: MeasurementGeometry | None = None
    branch_labels: list[str] = field(default_factory=list)
    instrument: str | None = None
    notes: str | None = None
    g_mode: GMode | None = None
    g_value: float | None = None
    update_ledger: bool = False
    create_sample: bool = False
    mark_canonical: bool = False
    replace_canonical: bool = False
    interactive: bool = False
    raw_import_root: Path | None = None
    original_source_path: Path | None = None
    raw_import_copied: bool = False
    raw_import_sidecars: list[Path] = field(default_factory=list)
    raw_import_message: str | None = None

    @property
    def registry_path(self) -> Path | None:
        return self.sample_registry_path

    @registry_path.setter
    def registry_path(self, value: Path | None) -> None:
        self.sample_registry_path = value


class _UniqueKeyLoader(yaml.SafeLoader if yaml is not None else object):
    pass


def _construct_mapping(loader, node, deep=False):  # type: ignore[no-untyped-def]
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise MetadataSchemaError(f"Duplicate metadata key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


if yaml is not None:
    _UniqueKeyLoader.add_constructor(  # type: ignore[attr-defined]
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_mapping,
    )


def empty_registry() -> SampleRegistry:
    return SampleRegistry()


def empty_measurement_ledger() -> MeasurementLedger:
    return MeasurementLedger()


def empty_processed_ledger() -> ProcessedLedger:
    return ProcessedLedger()


def load_registry(path: Path) -> SampleRegistry:
    data = _load_yaml_mapping(path)
    return registry_from_dict(data)


def save_registry(registry: SampleRegistry, path: Path) -> Path:
    return _write_yaml(registry_to_dict(registry), path)


def load_measurement_ledger(path: Path) -> MeasurementLedger:
    data = _load_yaml_mapping(path)
    return measurement_ledger_from_dict(data)


def save_measurement_ledger(ledger: MeasurementLedger, path: Path) -> Path:
    return _write_yaml(measurement_ledger_to_dict(ledger), path)


def load_processed_ledger(path: Path) -> ProcessedLedger:
    data = _load_yaml_mapping(path)
    return processed_ledger_from_dict(data)


def save_processed_ledger(ledger: ProcessedLedger, path: Path) -> Path:
    return _write_yaml(processed_ledger_to_dict(ledger), path)


def registry_from_dict(data: dict[str, Any]) -> SampleRegistry:
    version = int(data.get("schema_version", 0))
    if version != SAMPLE_SCHEMA_VERSION:
        raise MetadataSchemaError(
            f"Sample registry schema_version {version!r} is not supported; "
            "schema_version: 2 is required."
        )
    samples_payload = data.get("samples", {})
    if not isinstance(samples_payload, dict):
        raise MetadataSchemaError(
            "Sample registry field 'samples' must be a mapping keyed by sample_id."
        )
    samples: dict[str, SampleRecord] = {}
    for key, payload in samples_payload.items():
        if not isinstance(payload, dict):
            raise MetadataSchemaError(f"Sample entry must be a mapping: {key}")
        if "measurements" in payload:
            raise MetadataSchemaError(
                "Old sample registry shape detected at samples.<sample_id>.measurements; "
                "metadata schema v2 requires measurement_ledger.yaml."
            )
        sample = sample_from_dict(str(key), payload)
        if sample.sample_id in samples:
            raise MetadataSchemaError(f"Duplicate sample_id in registry: {sample.sample_id}")
        samples[sample.sample_id] = sample
    return SampleRegistry(schema_version=version, samples=samples)


def registry_to_dict(registry: SampleRegistry) -> dict[str, Any]:
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "samples": {
            sample_id: sample_to_dict(sample)
            for sample_id, sample in sorted(
                registry.samples.items(), key=lambda item: item[0].lower()
            )
        },
    }


def measurement_ledger_from_dict(data: dict[str, Any]) -> MeasurementLedger:
    version = int(data.get("schema_version", 0))
    if version != MEASUREMENT_LEDGER_SCHEMA_VERSION:
        raise MetadataSchemaError(
            f"Measurement ledger schema_version {version!r} is not supported; "
            "schema_version: 1 is required."
        )
    payload = data.get("measurements", {})
    if not isinstance(payload, dict):
        raise MetadataSchemaError(
            "Measurement ledger field 'measurements' must be a mapping keyed by measurement_id."
        )
    measurements = {
        str(key): measurement_from_dict(str(key), value)
        for key, value in payload.items()
        if isinstance(value, dict)
    }
    return MeasurementLedger(schema_version=version, measurements=measurements)


def measurement_ledger_to_dict(ledger: MeasurementLedger) -> dict[str, Any]:
    return {
        "schema_version": MEASUREMENT_LEDGER_SCHEMA_VERSION,
        "measurements": {
            key: measurement_to_dict(record)
            for key, record in sorted(ledger.measurements.items(), key=lambda item: item[0].lower())
        },
    }


def processed_ledger_from_dict(data: dict[str, Any]) -> ProcessedLedger:
    version = int(data.get("schema_version", 0))
    if version != PROCESSED_LEDGER_SCHEMA_VERSION:
        raise MetadataSchemaError(
            f"Processed ledger schema_version {version!r} is not supported; "
            "schema_version: 1 is required."
        )
    payload = data.get("processed_results", {})
    if not isinstance(payload, dict):
        raise MetadataSchemaError(
            "Processed ledger field 'processed_results' must be a mapping keyed by result_id."
        )
    results = {
        str(key): processed_result_from_dict(str(key), value)
        for key, value in payload.items()
        if isinstance(value, dict)
    }
    ledger = ProcessedLedger(schema_version=version, processed_results=results)
    validate_processed_ledger_canonical_uniqueness(ledger)
    return ledger


def processed_ledger_to_dict(ledger: ProcessedLedger) -> dict[str, Any]:
    return {
        "schema_version": PROCESSED_LEDGER_SCHEMA_VERSION,
        "processed_results": {
            key: processed_result_to_dict(record)
            for key, record in sorted(
                ledger.processed_results.items(), key=lambda item: item[0].lower()
            )
        },
    }


def sample_from_dict(key: str, payload: dict[str, Any]) -> SampleRecord:
    sample_id = str(payload.get("sample_id") or key)
    geometry_payload = payload.get("geometry") if isinstance(payload.get("geometry"), dict) else {}
    defaults_payload = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else {}
    known_keys = {
        "sample_id",
        "aliases",
        "condition",
        "replicate",
        "stack",
        "geometry",
        "layer_stack",
        "magnetic_volume_m3",
        "magnetic_volume_source",
        "magnetic_volume_method",
        "magnetic_volume_warnings",
        "defaults",
    }
    return SampleRecord(
        sample_id=sample_id,
        aliases=[str(item) for item in payload.get("aliases", [])],
        condition=_optional_str(payload.get("condition")),
        replicate=_optional_str(payload.get("replicate")),
        stack=_optional_str(payload.get("stack")),
        geometry=volume_from_dict(geometry_payload),
        layer_stack=_layer_stack_from_payload(payload.get("layer_stack")),
        magnetic_volume_m3=_optional_float(payload.get("magnetic_volume_m3")),
        magnetic_volume_source=_optional_str(payload.get("magnetic_volume_source")),
        magnetic_volume_method=_optional_str(payload.get("magnetic_volume_method")),
        magnetic_volume_warnings=[
            str(item) for item in payload.get("magnetic_volume_warnings", []) if item is not None
        ],
        defaults=AnalysisDefaults(
            g_mode=_parse_g_mode(defaults_payload.get("g_mode", "float")),
            g_value=_optional_float(defaults_payload.get("g_value")),
            ms_source=_optional_str(defaults_payload.get("ms_source")),
        ),
        extra={str(name): value for name, value in payload.items() if name not in known_keys},
    )


def sample_to_dict(sample: SampleRecord) -> dict[str, Any]:
    payload = dict(sample.extra)
    payload.update(
        {
            "sample_id": sample.sample_id,
            "aliases": list(sample.aliases),
            "condition": sample.condition,
            "replicate": sample.replicate,
            "stack": sample.stack,
            "geometry": volume_to_dict(sample.geometry),
            "defaults": asdict(sample.defaults),
        }
    )
    if sample.layer_stack:
        payload["layer_stack"] = [dict(layer) for layer in sample.layer_stack]
    if sample.magnetic_volume_m3 is not None:
        payload["magnetic_volume_m3"] = sample.magnetic_volume_m3
    if sample.magnetic_volume_source:
        payload["magnetic_volume_source"] = sample.magnetic_volume_source
    if sample.magnetic_volume_method:
        payload["magnetic_volume_method"] = sample.magnetic_volume_method
    if sample.magnetic_volume_warnings:
        payload["magnetic_volume_warnings"] = list(sample.magnetic_volume_warnings)
    return payload


def measurement_from_dict(key: str, payload: dict[str, Any]) -> MeasurementRecord:
    measurement_type = _parse_measurement_type(payload.get("type"))
    return MeasurementRecord(
        measurement_id=str(payload.get("measurement_id") or key),
        sample_id=str(payload.get("sample_id") or ""),
        type=measurement_type,
        raw_path=str(payload.get("raw_path") or payload.get("path") or ""),
        geometry=_parse_geometry(payload.get("geometry", "unknown")),
        branch_labels=[str(item) for item in payload.get("branch_labels", [])],
        instrument=_optional_str(payload.get("instrument")),
        notes=_optional_str(payload.get("notes")),
        status=_parse_measurement_status(payload.get("status", "active")),
    )


def measurement_to_dict(measurement: MeasurementRecord) -> dict[str, Any]:
    return {
        "measurement_id": measurement.measurement_id,
        "sample_id": measurement.sample_id,
        "type": measurement.type,
        "raw_path": measurement.raw_path,
        "geometry": measurement.geometry,
        "branch_labels": list(measurement.branch_labels),
        "instrument": measurement.instrument,
        "notes": measurement.notes,
        "status": measurement.status,
    }


def processed_result_from_dict(key: str, payload: dict[str, Any]) -> ProcessedResultRecord:
    return ProcessedResultRecord(
        result_id=str(payload.get("result_id") or key),
        measurement_id=str(payload.get("measurement_id") or ""),
        sample_id=str(payload.get("sample_id") or ""),
        type=_parse_measurement_type(payload.get("type")),
        processed_path=str(payload.get("processed_path") or ""),
        recipe_path=str(payload.get("recipe_path") or ""),
        recipe_hash=_optional_str(payload.get("recipe_hash")),
        created_at=str(payload.get("created_at") or ""),
        status=_parse_processed_status(payload.get("status", "test")),
        summary=dict(payload.get("summary") or {}),
    )


def processed_result_to_dict(result: ProcessedResultRecord) -> dict[str, Any]:
    return {
        "result_id": result.result_id,
        "measurement_id": result.measurement_id,
        "sample_id": result.sample_id,
        "type": result.type,
        "processed_path": result.processed_path,
        "recipe_path": result.recipe_path,
        "recipe_hash": result.recipe_hash,
        "created_at": result.created_at,
        "status": result.status,
        "summary": dict(result.summary),
    }


def add_sample(registry: SampleRegistry, sample: SampleRecord) -> None:
    if sample.sample_id in registry.samples:
        raise MetadataSchemaError(f"Sample already exists: {sample.sample_id}")
    registry.samples[sample.sample_id] = sample


def find_sample(registry: SampleRegistry, sample_id_or_alias: str) -> SampleRecord | None:
    if sample_id_or_alias in registry.samples:
        return registry.samples[sample_id_or_alias]
    lowered = sample_id_or_alias.lower()
    for sample in registry.samples.values():
        if any(alias.lower() == lowered for alias in sample.aliases):
            return sample
    return None


def resolve_sample_magnetic_volume(
    sample: SampleRecord,
    *,
    strict: bool = False,
) -> MagneticVolumeResolution:
    warnings: list[str] = []
    has_estimate_inputs = bool(sample.geometry.shape and sample.layer_stack)

    if sample.magnetic_volume_m3 is not None:
        if sample.magnetic_volume_m3 > 0.0:
            source = sample.magnetic_volume_source or "manual"
            method = sample.magnetic_volume_method or (
                "manual" if source == "manual" else str(source)
            )
            warnings.extend(sample.magnetic_volume_warnings)
            if source != "estimated" and has_estimate_inputs:
                warnings.append(
                    "Manual magnetic_volume_m3 overrides geometry/layer-stack estimate."
                )
            return MagneticVolumeResolution(
                magnetic_volume_m3=sample.magnetic_volume_m3,
                source=source,
                method=method,
                warnings=warnings,
                is_estimated=source == "estimated",
                is_available=True,
            )
        message = "magnetic_volume_m3 must be greater than zero."
        if strict:
            raise MagneticVolumeError(message)
        return MagneticVolumeResolution(warnings=[message])

    vmag = sample.geometry.vmag
    if vmag.complete():
        try:
            return MagneticVolumeResolution(
                magnetic_volume_m3=normalize_volume_m3(vmag.value, vmag.unit),
                source="manual",
                method=vmag.method or "manual",
                warnings=["Legacy geometry.vmag was used as manual magnetic volume."],
                is_estimated=False,
                is_available=True,
            )
        except MagneticVolumeError as exc:
            if strict:
                raise
            return MagneticVolumeResolution(warnings=[str(exc)])

    if sample.geometry.shape and sample.layer_stack:
        try:
            estimate = estimate_magnetic_volume(
                _geometry_mapping_for_estimator(sample.geometry),
                sample.layer_stack,
            )
            return MagneticVolumeResolution(
                magnetic_volume_m3=estimate.magnetic_volume_m3,
                source="estimated",
                method=estimate.method,
                area_m2=estimate.area_m2,
                magnetic_thickness_total_m=estimate.magnetic_thickness_total_m,
                included_layers=estimate.included_layers,
                excluded_layers=estimate.excluded_layers,
                warnings=list(estimate.warnings),
                is_estimated=True,
                is_available=True,
            )
        except MagneticVolumeError as exc:
            if strict:
                raise
            return MagneticVolumeResolution(warnings=[str(exc)])

    if sample.geometry.has_complete_derived_volume_inputs():
        try:
            area = sample.geometry.area
            thickness = sample.geometry.magnetic_thickness
            estimate = estimate_from_area_and_thickness(
                area.value,
                area.unit,
                thickness.value,
                thickness.unit,
            )
            return MagneticVolumeResolution(
                magnetic_volume_m3=estimate.magnetic_volume_m3,
                source="estimated",
                method=estimate.method,
                area_m2=estimate.area_m2,
                magnetic_thickness_total_m=estimate.magnetic_thickness_total_m,
                included_layers=estimate.included_layers,
                excluded_layers=estimate.excluded_layers,
                warnings=list(estimate.warnings),
                is_estimated=True,
                is_available=True,
            )
        except MagneticVolumeError as exc:
            if strict:
                raise
            return MagneticVolumeResolution(warnings=[str(exc)])

    return MagneticVolumeResolution(
        warnings=[
            "Magnetic volume is unavailable; provide magnetic_volume_m3 or geometry "
            "plus layer_stack."
        ]
    )


def update_sample_magnetic_volume_fields(
    sample: SampleRecord,
    resolution: MagneticVolumeResolution,
) -> SampleRecord:
    if resolution.is_available and resolution.magnetic_volume_m3 is not None:
        sample.magnetic_volume_m3 = resolution.magnetic_volume_m3
        sample.magnetic_volume_source = resolution.source
        sample.magnetic_volume_method = resolution.method
        sample.magnetic_volume_warnings = list(resolution.warnings)
    else:
        sample.magnetic_volume_warnings = list(resolution.warnings)
        if sample.magnetic_volume_source is None:
            sample.magnetic_volume_source = "unknown"
    return sample


def _geometry_mapping_for_estimator(geometry: VolumeMetadata) -> dict[str, Any]:
    if not geometry.shape:
        raise MagneticVolumeError("Geometry requires a shape.")
    dimensions = dict(geometry.dimensions)
    shape = geometry.shape.strip().lower().replace("-", "_")
    if shape == "array":
        return _array_geometry_mapping(dimensions)
    return {"shape": shape, **dimensions}


def _array_geometry_mapping(dimensions: dict[str, Any]) -> dict[str, Any]:
    base_geometry = dimensions.get("base_geometry")
    if not isinstance(base_geometry, dict):
        element_shape = _optional_str(dimensions.get("element_shape"))
        element_dimensions = dimensions.get("element_dimensions")
        if not element_shape or not isinstance(element_dimensions, dict):
            raise MagneticVolumeError(
                "Array geometry requires base_geometry or element_shape and element_dimensions."
            )
        base_geometry = {"shape": element_shape, **element_dimensions}
    count = dimensions.get("element_count", dimensions.get("count"))
    payload = {"shape": "array", "base_geometry": base_geometry, "element_count": count}
    if "fill_factor" in dimensions:
        payload["fill_factor"] = dimensions["fill_factor"]
    return payload


def upsert_measurement_record(
    ledger: MeasurementLedger,
    *,
    measurement_id: str,
    sample_id: str,
    measurement_type: MeasurementType,
    raw_path: Path,
    geometry: MeasurementGeometry = "unknown",
    branch_labels: list[str] | None = None,
    instrument: str | None = None,
    notes: str | None = None,
    base_dir: Path | None = None,
) -> MeasurementRecord:
    record = MeasurementRecord(
        measurement_id=measurement_id,
        sample_id=sample_id,
        type=measurement_type,
        raw_path=_stored_path(raw_path, registry_base_dir=base_dir),
        geometry=geometry,
        branch_labels=list(branch_labels or []),
        instrument=instrument,
        notes=notes,
        status="active",
    )
    ledger.measurements[measurement_id] = record
    return record


def archive_sample_ledger_records(
    sample_id: str,
    measurement_ledger: MeasurementLedger,
    processed_ledger: ProcessedLedger,
    *,
    dry_run: bool = False,
) -> LedgerPruneSummary:
    """Archive all ledger records for a sample without deleting provenance."""

    summary = LedgerPruneSummary(sample_id=sample_id, dry_run=dry_run)
    matching_measurement_ids: set[str] = set()
    for measurement in measurement_ledger.measurements.values():
        if measurement.sample_id != sample_id:
            summary.untouched_measurements += 1
            continue
        matching_measurement_ids.add(measurement.measurement_id)
        if measurement.status == "archived":
            summary.untouched_measurements += 1
            continue
        summary.archived_measurements += 1
        if not dry_run:
            measurement.status = "archived"

    for result in processed_ledger.processed_results.values():
        if result.sample_id != sample_id and result.measurement_id not in matching_measurement_ids:
            summary.untouched_processed_results += 1
            continue
        if result.status == "archived":
            summary.untouched_processed_results += 1
            continue
        summary.archived_processed_results += 1
        if result.status == "canonical":
            summary.archived_canonical_results += 1
        if not dry_run:
            result.status = "archived"
    return summary


def add_processed_result_record(
    ledger: ProcessedLedger,
    *,
    result: ProcessedResultRecord,
    mark_canonical: bool,
    replace_canonical: bool,
) -> None:
    existing = canonical_results_for_measurement(ledger, result.measurement_id)
    if len(existing) > 1:
        raise DuplicateCanonicalProcessedResultError(
            "Multiple canonical processed results exist for measurement_id "
            f"{result.measurement_id}."
        )
    if mark_canonical:
        if existing and not replace_canonical:
            raise CanonicalResultExistsError(
                "Canonical processed result already exists for measurement_id "
                f"{result.measurement_id}; "
                "pass --replace-canonical to supersede it."
            )
        for previous in existing:
            previous.status = "superseded"
        result.status = "canonical"
    elif result.status == "canonical" and existing:
        raise CanonicalResultExistsError(
            f"Canonical processed result already exists for measurement_id {result.measurement_id}."
        )
    ledger.processed_results[result.result_id] = result


def canonical_results_for_measurement(
    ledger: ProcessedLedger,
    measurement_id: str,
) -> list[ProcessedResultRecord]:
    return [
        result
        for result in ledger.processed_results.values()
        if result.measurement_id == measurement_id and result.status == "canonical"
    ]


def canonical_results_for_sample(
    ledger: ProcessedLedger,
    sample_id: str,
) -> list[ProcessedResultRecord]:
    by_measurement: dict[str, list[ProcessedResultRecord]] = {}
    for result in ledger.processed_results.values():
        if result.sample_id == sample_id and result.status == "canonical":
            by_measurement.setdefault(result.measurement_id, []).append(result)
    duplicates = [
        measurement_id for measurement_id, rows in by_measurement.items() if len(rows) > 1
    ]
    if duplicates:
        raise DuplicateCanonicalProcessedResultError(
            "Multiple canonical processed results exist for measurement_id(s): "
            + ", ".join(sorted(duplicates))
        )
    return [rows[0] for rows in by_measurement.values()]


def validate_processed_ledger_canonical_uniqueness(ledger: ProcessedLedger) -> None:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for result in ledger.processed_results.values():
        if result.status != "canonical":
            continue
        previous = seen.get(result.measurement_id)
        if previous is not None:
            duplicates.append(result.measurement_id)
        seen[result.measurement_id] = result.result_id
    if duplicates:
        raise DuplicateCanonicalProcessedResultError(
            "Multiple canonical processed results exist for measurement_id(s): "
            + ", ".join(sorted(set(duplicates)))
        )


def resolve_analysis_context(
    *,
    source_path: Path,
    measurement_type: MeasurementType,
    options: RegistryWorkflowOptions | None,
) -> AnalysisSampleContext | None:
    options = options or RegistryWorkflowOptions()
    if (
        not options.update_ledger
        and not options.sample_id
        and not options.g_mode
        and not options.g_value
        and not options.geometry
    ):
        return None

    sample_registry_path = _defaulted_path(
        options.sample_registry_path, DEFAULT_SAMPLE_REGISTRY_PATH
    )
    measurement_ledger_path = _defaulted_path(
        options.measurement_ledger_path, DEFAULT_MEASUREMENT_LEDGER_PATH
    )
    processed_ledger_path = _defaulted_path(
        options.processed_ledger_path, DEFAULT_PROCESSED_LEDGER_PATH
    )
    registry = _load_registry_or_empty(sample_registry_path)
    sample: SampleRecord | None = None
    measurement: MeasurementRecord | None = None

    if options.sample_id:
        sample = find_sample(registry, options.sample_id)
        if sample is None and options.update_ledger:
            if not options.create_sample:
                raise MissingSampleError(
                    f"Sample {options.sample_id!r} does not exist in "
                    f"{sample_registry_path}; pass --create-sample to create it."
                )
            sample = SampleRecord(sample_id=options.sample_id)
            add_sample(registry, sample)
            save_registry(registry, sample_registry_path)
        elif sample is None:
            sample = SampleRecord(sample_id=options.sample_id)

    if options.update_ledger:
        _validate_ledger_update_options(options, measurement_type)
        assert sample is not None
        ledger = _load_measurement_ledger_or_empty(measurement_ledger_path)
        measurement = upsert_measurement_record(
            ledger,
            measurement_id=str(options.measurement_id),
            sample_id=sample.sample_id,
            measurement_type=measurement_type,
            raw_path=source_path,
            geometry=options.geometry or "unknown",
            branch_labels=options.branch_labels,
            instrument=options.instrument,
            notes=options.notes,
            base_dir=measurement_ledger_path.parent,
        )
        save_measurement_ledger(ledger, measurement_ledger_path)
        measurement_snapshot = measurement_ledger_to_dict(ledger)
    else:
        measurement_snapshot = None
        if options.measurement_id and options.sample_id:
            measurement = MeasurementRecord(
                measurement_id=options.measurement_id,
                sample_id=options.sample_id,
                type=measurement_type,
                raw_path=str(source_path),
                geometry=options.geometry or "unknown",
                branch_labels=list(options.branch_labels),
                instrument=options.instrument,
                notes=options.notes,
            )

    if sample is None and measurement is not None:
        sample = find_sample(registry, measurement.sample_id) or SampleRecord(
            sample_id=measurement.sample_id
        )
    if sample is None:
        return None

    return AnalysisSampleContext(
        sample_registry_path=sample_registry_path,
        measurement_ledger_path=measurement_ledger_path,
        processed_ledger_path=processed_ledger_path,
        sample=sample,
        measurement=measurement,
        geometry=options.geometry
        or (measurement.geometry if measurement is not None else "unknown"),
        g_mode=options.g_mode or sample.defaults.g_mode,
        g_value=options.g_value if options.g_value is not None else sample.defaults.g_value,
        branch_labels=list(
            options.branch_labels or (measurement.branch_labels if measurement else [])
        ),
        original_source_path=options.original_source_path,
        raw_import_copied=options.raw_import_copied,
        raw_import_sidecars=list(options.raw_import_sidecars),
        raw_import_message=options.raw_import_message,
        sample_registry_snapshot=registry_to_dict(registry),
        measurement_ledger_snapshot=measurement_snapshot,
    )


def record_processed_result(
    *,
    sample_context: AnalysisSampleContext | None,
    measurement_type: MeasurementType,
    processed_path: Path,
    recipe_path: Path,
    analysis: Any,
    options: RegistryWorkflowOptions | None,
) -> ProcessedResultRecord | None:
    options = options or RegistryWorkflowOptions()
    if not options.update_ledger:
        return None
    if (
        sample_context is None
        or sample_context.sample_id is None
        or sample_context.measurement_id is None
    ):
        raise MissingLedgerMetadataError(
            "--update-ledger requires sample_id and measurement_id before recording "
            "processed output."
        )
    processed_ledger_path = _defaulted_path(
        options.processed_ledger_path, DEFAULT_PROCESSED_LEDGER_PATH
    )
    ledger = _load_processed_ledger_or_empty(processed_ledger_path)
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    result_id = make_result_id(sample_context.measurement_id, recipe_path.stem, timestamp)
    status: ProcessedStatus = "canonical" if options.mark_canonical else "test"
    record = ProcessedResultRecord(
        result_id=result_id,
        measurement_id=sample_context.measurement_id,
        sample_id=sample_context.sample_id,
        type=measurement_type,
        processed_path=_stored_path(processed_path, registry_base_dir=processed_ledger_path.parent),
        recipe_path=_stored_path(recipe_path, registry_base_dir=processed_ledger_path.parent),
        recipe_hash=file_sha256(recipe_path),
        created_at=timestamp,
        status=status,
        summary=_processed_summary(sample_context, analysis),
    )
    add_processed_result_record(
        ledger,
        result=record,
        mark_canonical=options.mark_canonical,
        replace_canonical=options.replace_canonical,
    )
    save_processed_ledger(ledger, processed_ledger_path)
    sample_context.processed_ledger_snapshot = processed_ledger_to_dict(ledger)
    return record


def make_result_id(measurement_id: str, recipe_stem: str, timestamp: str) -> str:
    raw = f"{measurement_id}:{recipe_stem}:{timestamp}"
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", raw).strip("-")
    slug = slug.replace(":", "_")
    return slug or "processed-result"


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def validate_registry(
    registry: SampleRegistry, *, registry_base_dir: Path | None = None
) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    aliases: dict[str, str] = {}
    for key, sample in registry.samples.items():
        if key != sample.sample_id:
            messages.append(
                ValidationMessage(
                    "error",
                    "sample_key_mismatch",
                    f"Sample key {key!r} does not match sample_id {sample.sample_id!r}.",
                    sample.sample_id,
                )
            )
        _validate_quantity_units(messages, sample.sample_id, "area", sample.geometry.area)
        _validate_quantity_units(
            messages, sample.sample_id, "magnetic_thickness", sample.geometry.magnetic_thickness
        )
        _validate_quantity_units(messages, sample.sample_id, "vmag", sample.geometry.vmag)
        if sample.geometry.vmag.partially_filled():
            messages.append(
                ValidationMessage(
                    "warning",
                    "incomplete_vmag_metadata",
                    f"Sample {sample.sample_id} has incomplete direct vmag metadata.",
                    sample.sample_id,
                )
            )
        if sample.magnetic_volume_source and sample.magnetic_volume_source not in {
            "manual",
            "estimated",
            "imported",
            "unknown",
        }:
            messages.append(
                ValidationMessage(
                    "warning",
                    "invalid_magnetic_volume_source",
                    f"Sample {sample.sample_id} has unrecognized magnetic_volume_source "
                    f"{sample.magnetic_volume_source!r}.",
                    sample.sample_id,
                )
            )
        if sample.magnetic_volume_m3 is not None and sample.magnetic_volume_m3 <= 0.0:
            messages.append(
                ValidationMessage(
                    "warning",
                    "invalid_magnetic_volume",
                    f"Sample {sample.sample_id} has non-positive magnetic_volume_m3.",
                    sample.sample_id,
                )
            )
        resolution = resolve_sample_magnetic_volume(sample)
        for warning_text in resolution.warnings:
            if warning_text.startswith("Magnetic volume is unavailable"):
                continue
            messages.append(
                ValidationMessage(
                    "warning",
                    "magnetic_volume_warning",
                    f"Sample {sample.sample_id}: {warning_text}",
                    sample.sample_id,
                )
            )
        if not resolution.is_available:
            messages.append(
                ValidationMessage(
                    "warning",
                    "volume_metadata_incomplete",
                    f"Sample {sample.sample_id} cannot derive magnetic volume from "
                    "current metadata.",
                    sample.sample_id,
                )
            )
        for alias in sample.aliases:
            previous = aliases.get(alias.lower())
            if previous is not None and previous != sample.sample_id:
                messages.append(
                    ValidationMessage(
                        "warning",
                        "duplicate_alias",
                        f"Alias {alias!r} appears on both {previous} and {sample.sample_id}.",
                        sample.sample_id,
                    )
                )
            aliases[alias.lower()] = sample.sample_id
    return messages


def resolve_measurement_path(
    measurement: MeasurementRecord, *, registry_base_dir: Path | None = None
) -> Path:
    path = Path(measurement.raw_path)
    if path.is_absolute():
        return path
    if registry_base_dir is not None:
        return registry_base_dir / path
    return path


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def volume_from_dict(payload: dict[str, Any]) -> VolumeMetadata:
    known_keys = {"area", "magnetic_thickness", "vmag", "shape", "dimensions", "notes"}
    dimensions = payload.get("dimensions") if isinstance(payload.get("dimensions"), dict) else {}
    return VolumeMetadata(
        area=quantity_from_dict(payload.get("area")),
        magnetic_thickness=quantity_from_dict(payload.get("magnetic_thickness")),
        vmag=direct_volume_from_dict(payload.get("vmag")),
        shape=_optional_str(payload.get("shape")),
        dimensions=dict(dimensions),
        notes=_optional_str(payload.get("notes")),
        extra={str(name): value for name, value in payload.items() if name not in known_keys},
    )


def volume_to_dict(volume: VolumeMetadata) -> dict[str, Any]:
    payload = dict(volume.extra)
    payload.update(
        {
            "area": quantity_to_dict(volume.area),
            "magnetic_thickness": quantity_to_dict(volume.magnetic_thickness),
            "vmag": {
                **quantity_to_dict(volume.vmag),
                "method": volume.vmag.method,
            },
        }
    )
    if volume.shape:
        payload["shape"] = volume.shape
    if volume.dimensions:
        payload["dimensions"] = dict(volume.dimensions)
    if volume.notes:
        payload["notes"] = volume.notes
    return payload


def _layer_stack_from_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def quantity_from_dict(payload: Any) -> QuantityMetadata:
    payload = payload if isinstance(payload, dict) else {}
    return QuantityMetadata(
        value=_optional_float(payload.get("value")),
        unit=_optional_str(payload.get("unit")),
        uncertainty=_optional_float(payload.get("uncertainty")),
    )


def direct_volume_from_dict(payload: Any) -> DirectVolumeMetadata:
    payload = payload if isinstance(payload, dict) else {}
    return DirectVolumeMetadata(
        value=_optional_float(payload.get("value")),
        unit=_optional_str(payload.get("unit")),
        uncertainty=_optional_float(payload.get("uncertainty")),
        method=_optional_str(payload.get("method")),
    )


def quantity_to_dict(quantity: QuantityMetadata) -> dict[str, Any]:
    return {"value": quantity.value, "unit": quantity.unit, "uncertainty": quantity.uncertainty}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise MetadataSchemaError("PyYAML is required to load metadata files.")
    resolved = path.resolve()
    if not resolved.exists():
        raise MetadataSchemaError(f"Metadata file does not exist: {resolved}")
    try:
        data = yaml.load(resolved.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader) or {}
    except MetadataSchemaError:
        raise
    except Exception as exc:  # noqa: BLE001 - include YAML context
        raise MetadataSchemaError(f"Failed to parse metadata file {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise MetadataSchemaError(f"Metadata file must be a YAML mapping: {resolved}")
    return data


def _write_yaml(payload: dict[str, Any], path: Path) -> Path:
    if yaml is None:
        raise MetadataSchemaError("PyYAML is required to save metadata files.")
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(to_serializable(payload), sort_keys=False), encoding="utf-8")
    return resolved


def _load_registry_or_empty(path: Path) -> SampleRegistry:
    return load_registry(path) if path.exists() else empty_registry()


def _load_measurement_ledger_or_empty(path: Path) -> MeasurementLedger:
    return load_measurement_ledger(path) if path.exists() else empty_measurement_ledger()


def _load_processed_ledger_or_empty(path: Path) -> ProcessedLedger:
    return load_processed_ledger(path) if path.exists() else empty_processed_ledger()


def _defaulted_path(path: Path | None, default: Path) -> Path:
    return (path or default).resolve()


def _validate_ledger_update_options(
    options: RegistryWorkflowOptions, measurement_type: MeasurementType
) -> None:
    missing = []
    if not options.sample_id:
        missing.append("sample_id")
    if not options.measurement_id:
        missing.append("measurement_id")
    if measurement_type in {"fmr", "esr"} and not options.geometry:
        missing.append("geometry")
    if missing:
        raise MissingLedgerMetadataError("--update-ledger requires " + ", ".join(missing) + ".")


def _processed_summary(sample_context: AnalysisSampleContext, analysis: Any) -> dict[str, Any]:
    summary = getattr(analysis, "summary_metrics", {}) or {}
    provenance = getattr(analysis, "provenance", {}) or {}
    recipe_config = provenance.get("recipe_config", {}) if isinstance(provenance, dict) else {}
    branches = list(sample_context.branch_labels)
    payload = getattr(analysis, "analysis_payload", {}) or {}
    series = (payload.get("series_collection_result") or {}).get("series_by_label") or {}
    if series and not branches:
        branches = [str(key) for key in series]
    return {
        "geometry": sample_context.geometry,
        "g_mode": summary.get("g_mode") or sample_context.g_mode,
        "g_value": summary.get("g_value") or sample_context.g_value or summary.get("g"),
        "branches": branches,
        "field_polarity_correction": recipe_config.get("field_polarity_correction")
        if isinstance(recipe_config, dict)
        else None,
    }


def _validate_quantity_units(
    messages: list[ValidationMessage], sample_id: str, label: str, quantity: QuantityMetadata
) -> None:
    if (quantity.value is not None or quantity.uncertainty is not None) and not quantity.unit:
        messages.append(
            ValidationMessage(
                "warning",
                "missing_units",
                f"Sample {sample_id} has {label} value/uncertainty without units.",
                sample_id,
            )
        )


def _stored_path(path: Path, *, registry_base_dir: Path | None) -> str:
    resolved = path.resolve()
    if registry_base_dir is None:
        return str(resolved)
    try:
        return str(resolved.relative_to(registry_base_dir.resolve()))
    except ValueError:
        return str(resolved)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _parse_measurement_type(value: Any) -> MeasurementType:
    text = str(value or "").lower()
    if text not in {"vsm", "fmr", "esr"}:
        raise MetadataSchemaError(f"Measurement type must be one of vsm, fmr, esr: {value!r}")
    return text  # type: ignore[return-value]


def _parse_geometry(value: Any) -> MeasurementGeometry:
    text = str(value or "unknown").lower()
    if text not in {"ip", "oop", "angular", "unknown"}:
        raise MetadataSchemaError(f"Geometry must be one of ip, oop, angular, unknown: {value!r}")
    return text  # type: ignore[return-value]


def _parse_g_mode(value: Any) -> GMode:
    text = str(value or "float").lower()
    if text not in {"fixed", "float", "bounded"}:
        raise MetadataSchemaError(f"g_mode must be one of fixed, float, bounded: {value!r}")
    return text  # type: ignore[return-value]


def _parse_measurement_status(value: Any) -> MeasurementStatus:
    text = str(value or "active").lower()
    if text not in {"active", "archived"}:
        raise MetadataSchemaError(f"Measurement status must be active or archived: {value!r}")
    return text  # type: ignore[return-value]


def _parse_processed_status(value: Any) -> ProcessedStatus:
    text = str(value or "test").lower()
    if text not in {"canonical", "superseded", "test", "archived"}:
        raise MetadataSchemaError(
            f"Processed status must be canonical, superseded, test, or archived: {value!r}"
        )
    return text  # type: ignore[return-value]
