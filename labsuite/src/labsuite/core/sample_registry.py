"""Project-level sample registry models and helpers.

The registry is intentionally independent from CLI prompting and GUI state.  It
stores physical-sample metadata plus links to measurement files so analysis
workflows can resolve identity without relying only on filenames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

from labsuite.core.exceptions import LabSuiteError

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - PyYAML is a project dependency
    yaml = None

DEFAULT_SAMPLE_REGISTRY_PATH = Path("metadata") / "sample_registry.yaml"
SCHEMA_VERSION = 1
MeasurementType = Literal["vsm", "fmr", "esr"]
MeasurementGeometry = Literal["ip", "oop", "angular", "unknown"]
GMode = Literal["fixed", "float", "bounded"]


class RegistryError(LabSuiteError):
    """Base class for registry errors."""


class RegistryFormatError(RegistryError):
    """Raised when a registry file cannot be loaded as valid YAML."""


class RegistryResolutionError(RegistryError):
    """Raised when analysis cannot resolve required registry metadata."""


@dataclass(slots=True)
class QuantityMetadata:
    """Scalar value with units and optional uncertainty."""

    value: float | None = None
    unit: str | None = None
    uncertainty: float | None = None

    def complete(self) -> bool:
        return self.value is not None and bool(self.unit)


@dataclass(slots=True)
class DirectVolumeMetadata(QuantityMetadata):
    """Direct magnetic-volume entry with method provenance."""

    method: str | None = None

    def complete(self) -> bool:
        return QuantityMetadata.complete(self) and bool(self.method)

    def partially_filled(self) -> bool:
        return any(
            value is not None and value != ""
            for value in (self.value, self.unit, self.uncertainty, self.method)
        ) and not self.complete()


@dataclass(slots=True)
class VolumeMetadata:
    """Physical geometry needed to derive magnetic volume."""

    area: QuantityMetadata = field(default_factory=QuantityMetadata)
    magnetic_thickness: QuantityMetadata = field(default_factory=QuantityMetadata)
    vmag: DirectVolumeMetadata = field(default_factory=DirectVolumeMetadata)

    def has_complete_derived_volume_inputs(self) -> bool:
        return self.area.complete() and self.magnetic_thickness.complete()

    def has_complete_direct_volume(self) -> bool:
        return self.vmag.complete()


@dataclass(slots=True)
class MeasurementRecord:
    """One measurement-file link for a physical sample."""

    measurement_id: str
    sample_id: str
    type: MeasurementType
    path: str
    geometry: MeasurementGeometry = "unknown"
    branch_labels: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass(slots=True)
class AnalysisDefaults:
    """Default analysis settings associated with a sample."""

    g_mode: GMode = "float"
    g_value: float | None = None
    ms_source: str | None = None


@dataclass(slots=True)
class SampleRecord:
    """Project-level physical sample metadata."""

    sample_id: str
    aliases: list[str] = field(default_factory=list)
    condition: str | None = None
    replicate: str | None = None
    stack: str | None = None
    geometry: VolumeMetadata = field(default_factory=VolumeMetadata)
    defaults: AnalysisDefaults = field(default_factory=AnalysisDefaults)
    measurements: list[MeasurementRecord] = field(default_factory=list)


@dataclass(slots=True)
class ValidationMessage:
    """Validation message returned by registry checks."""

    severity: Literal["warning", "error"]
    code: str
    message: str
    sample_id: str | None = None
    measurement_id: str | None = None


@dataclass(slots=True)
class SampleRegistry:
    """Registry document."""

    schema_version: int = SCHEMA_VERSION
    samples: dict[str, SampleRecord] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisSampleContext:
    """Resolved registry metadata for one analysis run."""

    registry_path: Path | None = None
    sample: SampleRecord | None = None
    measurement: MeasurementRecord | None = None
    geometry: MeasurementGeometry = "unknown"
    g_mode: GMode = "float"
    g_value: float | None = None
    validation_warnings: list[ValidationMessage] = field(default_factory=list)
    registry_snapshot: dict[str, Any] | None = None

    @property
    def sample_id(self) -> str | None:
        return None if self.sample is None else self.sample.sample_id

    @property
    def measurement_id(self) -> str | None:
        return None if self.measurement is None else self.measurement.measurement_id

    def to_dict(self) -> dict[str, Any]:
        return to_serializable(
            {
                "registry_path": self.registry_path,
                "sample_id": self.sample_id,
                "measurement_id": self.measurement_id,
                "geometry": self.geometry,
                "g_mode": self.g_mode,
                "g_value": self.g_value,
                "sample": self.sample,
                "measurement": self.measurement,
                "validation_warnings": self.validation_warnings,
            }
        )


@dataclass(slots=True)
class RegistryWorkflowOptions:
    """Registry-related command options passed from CLI to workflows."""

    registry_path: Path | None = None
    sample_id: str | None = None
    geometry: MeasurementGeometry | None = None
    g_mode: GMode | None = None
    g_value: float | None = None
    interactive: bool = False


class _UniqueKeyLoader(yaml.SafeLoader if yaml is not None else object):
    pass


def _construct_mapping(loader, node, deep=False):  # type: ignore[no-untyped-def]
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise RegistryFormatError(f"Duplicate registry key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


if yaml is not None:
    _UniqueKeyLoader.add_constructor(  # type: ignore[attr-defined]
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_mapping,
    )


def empty_registry() -> SampleRegistry:
    return SampleRegistry()


def load_registry(path: Path) -> SampleRegistry:
    """Load a sample registry from YAML."""

    if yaml is None:
        raise RegistryFormatError("PyYAML is required to load sample registries.")
    resolved = path.resolve()
    if not resolved.exists():
        raise RegistryFormatError(f"Sample registry does not exist: {resolved}")
    try:
        data = yaml.load(resolved.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader) or {}
    except RegistryFormatError:
        raise
    except Exception as exc:  # noqa: BLE001 - include parser context from PyYAML
        raise RegistryFormatError(f"Failed to parse sample registry {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryFormatError(f"Sample registry must be a YAML mapping: {resolved}")
    return registry_from_dict(data)


def save_registry(registry: SampleRegistry, path: Path) -> Path:
    """Write a registry to YAML."""

    if yaml is None:
        raise RegistryFormatError("PyYAML is required to save sample registries.")
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(registry_to_dict(registry), sort_keys=False)
    resolved.write_text(text, encoding="utf-8")
    return resolved


def registry_from_dict(data: dict[str, Any]) -> SampleRegistry:
    samples_payload = data.get("samples", {})
    if not isinstance(samples_payload, dict):
        raise RegistryFormatError("Registry field 'samples' must be a mapping keyed by sample_id.")
    samples: dict[str, SampleRecord] = {}
    for key, payload in samples_payload.items():
        if not isinstance(payload, dict):
            raise RegistryFormatError(f"Sample entry must be a mapping: {key}")
        sample = sample_from_dict(str(key), payload)
        if sample.sample_id in samples:
            raise RegistryFormatError(f"Duplicate sample_id in registry: {sample.sample_id}")
        samples[sample.sample_id] = sample
    return SampleRegistry(schema_version=int(data.get("schema_version", SCHEMA_VERSION)), samples=samples)


def registry_to_dict(registry: SampleRegistry) -> dict[str, Any]:
    return {
        "schema_version": registry.schema_version,
        "samples": {
            sample_id: sample_to_dict(sample)
            for sample_id, sample in sorted(registry.samples.items(), key=lambda item: item[0].lower())
        },
    }


def sample_from_dict(key: str, payload: dict[str, Any]) -> SampleRecord:
    sample_id = str(payload.get("sample_id") or key)
    geometry_payload = payload.get("geometry") or {}
    defaults_payload = payload.get("defaults") or {}
    measurements_payload = payload.get("measurements") or []
    if not isinstance(measurements_payload, list):
        raise RegistryFormatError(f"Sample {sample_id} field 'measurements' must be a list.")
    return SampleRecord(
        sample_id=sample_id,
        aliases=[str(item) for item in payload.get("aliases", [])],
        condition=_optional_str(payload.get("condition")),
        replicate=_optional_str(payload.get("replicate")),
        stack=_optional_str(payload.get("stack")),
        geometry=volume_from_dict(geometry_payload if isinstance(geometry_payload, dict) else {}),
        defaults=AnalysisDefaults(
            g_mode=_parse_g_mode(defaults_payload.get("g_mode", "float")),
            g_value=_optional_float(defaults_payload.get("g_value")),
            ms_source=_optional_str(defaults_payload.get("ms_source")),
        ),
        measurements=[
            measurement_from_dict(sample_id, item)
            for item in measurements_payload
            if isinstance(item, dict)
        ],
    )


def sample_to_dict(sample: SampleRecord) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "aliases": list(sample.aliases),
        "condition": sample.condition,
        "replicate": sample.replicate,
        "stack": sample.stack,
        "geometry": {
            "area": quantity_to_dict(sample.geometry.area),
            "magnetic_thickness": quantity_to_dict(sample.geometry.magnetic_thickness),
            "vmag": {
                **quantity_to_dict(sample.geometry.vmag),
                "method": sample.geometry.vmag.method,
            },
        },
        "defaults": asdict(sample.defaults),
        "measurements": [measurement_to_dict(item) for item in sample.measurements],
    }


def volume_from_dict(payload: dict[str, Any]) -> VolumeMetadata:
    return VolumeMetadata(
        area=quantity_from_dict(payload.get("area")),
        magnetic_thickness=quantity_from_dict(payload.get("magnetic_thickness")),
        vmag=direct_volume_from_dict(payload.get("vmag")),
    )


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
    return {
        "value": quantity.value,
        "unit": quantity.unit,
        "uncertainty": quantity.uncertainty,
    }


def measurement_from_dict(sample_id: str, payload: dict[str, Any]) -> MeasurementRecord:
    measurement_type = _parse_measurement_type(payload.get("type"))
    return MeasurementRecord(
        measurement_id=str(payload.get("measurement_id") or _default_measurement_id(Path(str(payload.get("path", ""))), measurement_type)),
        sample_id=str(payload.get("sample_id") or sample_id),
        type=measurement_type,
        path=str(payload.get("path") or ""),
        geometry=_parse_geometry(payload.get("geometry", "unknown")),
        branch_labels=[str(item) for item in payload.get("branch_labels", [])],
        notes=_optional_str(payload.get("notes")),
    )


def measurement_to_dict(measurement: MeasurementRecord) -> dict[str, Any]:
    return {
        "measurement_id": measurement.measurement_id,
        "sample_id": measurement.sample_id,
        "type": measurement.type,
        "path": measurement.path,
        "geometry": measurement.geometry,
        "branch_labels": list(measurement.branch_labels),
        "notes": measurement.notes,
    }


def add_sample(registry: SampleRegistry, sample: SampleRecord) -> None:
    if sample.sample_id in registry.samples:
        raise RegistryFormatError(f"Sample already exists: {sample.sample_id}")
    registry.samples[sample.sample_id] = sample


def find_sample(registry: SampleRegistry, sample_id_or_alias: str) -> SampleRecord | None:
    if sample_id_or_alias in registry.samples:
        return registry.samples[sample_id_or_alias]
    lowered = sample_id_or_alias.lower()
    for sample in registry.samples.values():
        if any(alias.lower() == lowered for alias in sample.aliases):
            return sample
    return None


def find_measurement_by_path(
    registry: SampleRegistry,
    path: Path,
    *,
    registry_base_dir: Path | None = None,
) -> tuple[SampleRecord, MeasurementRecord] | None:
    target = path.resolve()
    for sample in registry.samples.values():
        for measurement in sample.measurements:
            if resolve_measurement_path(measurement, registry_base_dir=registry_base_dir).resolve() == target:
                return sample, measurement
    return None


def register_measurement(
    registry: SampleRegistry,
    *,
    sample_id: str,
    path: Path,
    measurement_type: MeasurementType,
    geometry: MeasurementGeometry = "unknown",
    measurement_id: str | None = None,
    branch_labels: list[str] | None = None,
    notes: str | None = None,
    registry_base_dir: Path | None = None,
) -> MeasurementRecord:
    sample = find_sample(registry, sample_id)
    if sample is None:
        raise RegistryResolutionError(f"Unknown sample_id: {sample_id}")
    existing = find_measurement_by_path(registry, path, registry_base_dir=registry_base_dir)
    if existing is not None:
        existing_sample, existing_measurement = existing
        if existing_sample.sample_id != sample.sample_id:
            raise RegistryResolutionError(
                f"Measurement path already registered to {existing_sample.sample_id}: {path}"
            )
        return existing_measurement
    record = MeasurementRecord(
        measurement_id=measurement_id or _default_measurement_id(path, measurement_type),
        sample_id=sample.sample_id,
        type=measurement_type,
        path=_stored_path(path, registry_base_dir=registry_base_dir),
        geometry=geometry,
        branch_labels=list(branch_labels or []),
        notes=notes,
    )
    sample.measurements.append(record)
    return record


def resolve_analysis_context(
    *,
    source_path: Path,
    measurement_type: MeasurementType,
    options: RegistryWorkflowOptions | None,
) -> AnalysisSampleContext | None:
    """Resolve registry metadata for a workflow source path.

    Missing registry files are ignored unless a sample id is requested.  If a
    registry exists but the file/sample is unresolved, the caller receives a
    RegistryResolutionError so batch workflows can write unresolved CSV rows.
    """

    options = options or RegistryWorkflowOptions()
    g_mode = options.g_mode or "float"
    g_value = options.g_value
    geometry = options.geometry or "unknown"
    registry_path = None if options.registry_path is None else options.registry_path.resolve()
    registry_exists = registry_path is not None and registry_path.exists()
    if not registry_exists:
        if options.sample_id:
            raise RegistryResolutionError(f"Sample registry does not exist: {registry_path}")
        if options.g_mode is not None or options.g_value is not None or options.geometry is not None:
            return AnalysisSampleContext(
                registry_path=registry_path,
                geometry=geometry,
                g_mode=g_mode,
                g_value=g_value,
            )
        return None

    assert registry_path is not None
    registry = load_registry(registry_path)
    registry_base_dir = registry_path.parent
    sample: SampleRecord | None = None
    measurement: MeasurementRecord | None = None
    if options.sample_id:
        sample = find_sample(registry, options.sample_id)
        if sample is None:
            raise RegistryResolutionError(f"Unknown sample_id in registry: {options.sample_id}")
        matched = find_measurement_by_path(registry, source_path, registry_base_dir=registry_base_dir)
        measurement = matched[1] if matched is not None and matched[0].sample_id == sample.sample_id else None
    else:
        matched = find_measurement_by_path(registry, source_path, registry_base_dir=registry_base_dir)
        if matched is not None:
            sample, measurement = matched
    if sample is None:
        raise RegistryResolutionError(f"No registry sample resolves source file: {source_path}")

    validation_warnings = [
        message for message in validate_registry(registry, registry_base_dir=registry_base_dir)
        if message.severity == "warning"
    ]
    resolved_geometry = options.geometry or (measurement.geometry if measurement is not None else "unknown")
    resolved_g_mode = options.g_mode or sample.defaults.g_mode
    resolved_g_value = options.g_value if options.g_value is not None else sample.defaults.g_value
    return AnalysisSampleContext(
        registry_path=registry_path,
        sample=sample,
        measurement=measurement,
        geometry=resolved_geometry,
        g_mode=resolved_g_mode,
        g_value=resolved_g_value,
        validation_warnings=validation_warnings,
        registry_snapshot=registry_to_dict(registry),
    )


def validate_registry(
    registry: SampleRegistry,
    *,
    registry_base_dir: Path | None = None,
) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    aliases: dict[str, str] = {}
    for key, sample in registry.samples.items():
        if key != sample.sample_id:
            messages.append(
                ValidationMessage(
                    severity="error",
                    code="sample_key_mismatch",
                    sample_id=sample.sample_id,
                    message=f"Sample key {key!r} does not match sample_id {sample.sample_id!r}.",
                )
            )
        _validate_quantity_units(messages, sample.sample_id, "area", sample.geometry.area)
        _validate_quantity_units(messages, sample.sample_id, "magnetic_thickness", sample.geometry.magnetic_thickness)
        _validate_quantity_units(messages, sample.sample_id, "vmag", sample.geometry.vmag)
        if sample.geometry.vmag.partially_filled():
            messages.append(
                ValidationMessage(
                    severity="warning",
                    code="incomplete_vmag_metadata",
                    sample_id=sample.sample_id,
                    message=f"Sample {sample.sample_id} has incomplete direct vmag metadata.",
                )
            )
        if not sample.geometry.has_complete_direct_volume() and not sample.geometry.has_complete_derived_volume_inputs():
            messages.append(
                ValidationMessage(
                    severity="warning",
                    code="volume_metadata_incomplete",
                    sample_id=sample.sample_id,
                    message=f"Sample {sample.sample_id} cannot derive magnetic volume from current metadata.",
                )
            )
        for alias in sample.aliases:
            normalized = alias.lower()
            previous = aliases.get(normalized)
            if previous is not None and previous != sample.sample_id:
                messages.append(
                    ValidationMessage(
                        severity="warning",
                        code="duplicate_alias",
                        sample_id=sample.sample_id,
                        message=f"Alias {alias!r} appears on both {previous} and {sample.sample_id}.",
                    )
                )
            aliases[normalized] = sample.sample_id
        for measurement in sample.measurements:
            if measurement.type in {"fmr", "esr"} and measurement.geometry == "unknown":
                messages.append(
                    ValidationMessage(
                        severity="warning",
                        code="missing_measurement_geometry",
                        sample_id=sample.sample_id,
                        measurement_id=measurement.measurement_id,
                        message=f"Measurement {measurement.measurement_id} is missing geometry.",
                    )
                )
            measurement_path = resolve_measurement_path(measurement, registry_base_dir=registry_base_dir)
            if not measurement_path.exists():
                messages.append(
                    ValidationMessage(
                        severity="warning",
                        code="missing_measurement_file",
                        sample_id=sample.sample_id,
                        measurement_id=measurement.measurement_id,
                        message=f"Measurement file does not exist: {measurement.path}",
                    )
                )
    return messages


def resolve_measurement_path(
    measurement: MeasurementRecord,
    *,
    registry_base_dir: Path | None = None,
) -> Path:
    path = Path(measurement.path)
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


def _validate_quantity_units(
    messages: list[ValidationMessage],
    sample_id: str,
    label: str,
    quantity: QuantityMetadata,
) -> None:
    if (quantity.value is not None or quantity.uncertainty is not None) and not quantity.unit:
        messages.append(
            ValidationMessage(
                severity="warning",
                code="missing_units",
                sample_id=sample_id,
                message=f"Sample {sample_id} has {label} value/uncertainty without units.",
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


def _default_measurement_id(path: Path, measurement_type: MeasurementType) -> str:
    stem = path.stem or "measurement"
    return f"{measurement_type}:{stem}"


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
        raise RegistryFormatError(f"Measurement type must be one of vsm, fmr, esr: {value!r}")
    return text  # type: ignore[return-value]


def _parse_geometry(value: Any) -> MeasurementGeometry:
    text = str(value or "unknown").lower()
    if text not in {"ip", "oop", "angular", "unknown"}:
        raise RegistryFormatError(f"Geometry must be one of ip, oop, angular, unknown: {value!r}")
    return text  # type: ignore[return-value]


def _parse_g_mode(value: Any) -> GMode:
    text = str(value or "float").lower()
    if text not in {"fixed", "float", "bounded"}:
        raise RegistryFormatError(f"g_mode must be one of fixed, float, bounded: {value!r}")
    return text  # type: ignore[return-value]
