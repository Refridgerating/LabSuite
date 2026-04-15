"""Parser for native Bruker ESR descriptor and binary data pairs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from labsuite.core.exceptions import ParseError
from labsuite.core.types import TraceDataset

_SUPPORTED_DESCRIPTOR_VALUES = {
    "IKKF": "REAL",
    "IRFMT": "D",
    "XFMT": "D",
    "XTYP": "IDX",
    "YTYP": "NODATA",
    "ZTYP": "NODATA",
}


def parse_esr_file(path: Path) -> TraceDataset:
    """Parse a native Bruker ESR dataset from a DSC descriptor path."""

    if not path.exists():
        raise ParseError(f"ESR file does not exist: {path}")
    if path.suffix.lower() != ".dsc":
        raise ParseError(f"Expected a Bruker descriptor file with .dsc suffix, got: {path.name}")

    descriptor_items = _parse_descriptor(path)
    descriptor_map = _descriptor_map(descriptor_items)
    _validate_descriptor(descriptor_map, path)

    point_count = _require_int(descriptor_map, "XPTS", path)
    field_start_gauss = _require_float(descriptor_map, "XMIN", path)
    field_width_gauss = _require_float(descriptor_map, "XWID", path)
    field_unit = descriptor_map.get("XUNI", "").strip("'")

    field_gauss = np.linspace(field_start_gauss, field_start_gauss + field_width_gauss, point_count)
    field_mT = _convert_field_to_mT(field_gauss, field_unit, path)
    dta_path = _resolve_dta_path(path)
    signal = np.fromfile(dta_path, dtype="<f8")
    if signal.size != point_count:
        raise ParseError(
            f"Bruker data length mismatch for {dta_path.name}: expected {point_count} points, got {signal.size}"
        )

    metadata = _build_metadata(descriptor_items, descriptor_map, dta_path, point_count)

    return TraceDataset(
        modality="esr",
        source_path=path,
        field_mT=np.asarray(field_mT, dtype=float),
        signal=np.asarray(signal, dtype=float),
        metadata=metadata,
    )


def _parse_descriptor(path: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("#"):
            continue

        parts = stripped.split(maxsplit=1)
        key = parts[0]
        value = parts[1].strip() if len(parts) > 1 else ""
        items.append((key, value))
    return items


def _descriptor_map(items: list[tuple[str, str]]) -> dict[str, str]:
    return {key: value for key, value in items}


def _validate_descriptor(descriptor_map: dict[str, str], path: Path) -> None:
    for key, expected in _SUPPORTED_DESCRIPTOR_VALUES.items():
        actual = descriptor_map.get(key)
        if actual != expected:
            raise ParseError(
                f"Unsupported Bruker descriptor in {path.name}: expected {key}={expected!r}, got {actual!r}"
            )


def _resolve_dta_path(dsc_path: Path) -> Path:
    for suffix in (".DTA", ".dta"):
        candidate = dsc_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise ParseError(f"Missing sibling Bruker data file for {dsc_path.name}")


def _convert_field_to_mT(field_gauss: np.ndarray, unit: str, path: Path) -> np.ndarray:
    normalized_unit = unit.upper()
    if normalized_unit == "G":
        return field_gauss / 10.0
    if normalized_unit == "MT":
        return field_gauss
    raise ParseError(f"Unsupported Bruker field unit {unit!r} in {path.name}")


def _build_metadata(
    descriptor_items: list[tuple[str, str]],
    descriptor_map: dict[str, str],
    dta_path: Path,
    point_count: int,
) -> dict[str, Any]:
    timestamp = _parse_timestamp(descriptor_map.get("DATE"), descriptor_map.get("TIME"))
    frequency_hz = _optional_float(descriptor_map.get("MWFQ"))
    microwave_power_watts = _optional_float(descriptor_map.get("MWPW"))
    modulation_amplitude_t = _optional_float(descriptor_map.get("B0MA"))
    modulation_frequency_hz = _optional_float(descriptor_map.get("B0MF"))
    temperature_k = _optional_float(descriptor_map.get("STMP"))
    sweep_start_gauss = _optional_float(descriptor_map.get("XMIN"))
    sweep_width_gauss = _optional_float(descriptor_map.get("XWID"))

    return {
        "parser": "bruker_esr_native_v1",
        "source_format": "bruker_dsc_dta",
        "signal_name": descriptor_map.get("IRNAM", "").strip("'"),
        "signal_unit": descriptor_map.get("IRUNI", "").strip("'"),
        "field_unit": "mT",
        "bruker": {
            "title": descriptor_map.get("TITL", "").strip("'"),
            "timestamp": timestamp,
            "frequency_hz": frequency_hz,
            "frequency_GHz": None if frequency_hz is None else frequency_hz / 1e9,
            "microwave_power_watts": microwave_power_watts,
            "microwave_power_mW": None if microwave_power_watts is None else microwave_power_watts * 1_000.0,
            "modulation_amplitude_t": modulation_amplitude_t,
            "modulation_amplitude_mT": None
            if modulation_amplitude_t is None
            else modulation_amplitude_t * 1_000.0,
            "modulation_frequency_hz": modulation_frequency_hz,
            "temperature_k": temperature_k,
            "temperature_c": None if temperature_k is None else temperature_k - 273.15,
            "sweep_start_mT": None if sweep_start_gauss is None else sweep_start_gauss / 10.0,
            "sweep_width_mT": None if sweep_width_gauss is None else sweep_width_gauss / 10.0,
            "point_count": point_count,
            "q_value": _optional_float(descriptor_map.get("QValue")),
            "dta_file": dta_path.name,
        },
        "raw_descriptor": [
            {
                "key": key,
                "value": value,
            }
            for key, value in descriptor_items
        ],
    }


def _parse_timestamp(raw_date: str | None, raw_time: str | None) -> str | None:
    if raw_date is None or raw_time is None:
        return None
    try:
        return datetime.strptime(f"{raw_date} {raw_time}", "%m/%d/%y %H:%M:%S").isoformat()
    except ValueError:
        return f"{raw_date} {raw_time}"


def _require_int(descriptor_map: dict[str, str], key: str, path: Path) -> int:
    value = descriptor_map.get(key)
    if value is None:
        raise ParseError(f"Missing Bruker descriptor key {key!r} in {path.name}")
    try:
        return int(value.strip("'"))
    except ValueError as exc:
        raise ParseError(f"Invalid integer value for {key!r} in {path.name}: {value!r}") from exc


def _require_float(descriptor_map: dict[str, str], key: str, path: Path) -> float:
    value = descriptor_map.get(key)
    if value is None:
        raise ParseError(f"Missing Bruker descriptor key {key!r} in {path.name}")
    try:
        return float(value.strip("'"))
    except ValueError as exc:
        raise ParseError(f"Invalid float value for {key!r} in {path.name}: {value!r}") from exc


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value.strip("'"))
    except ValueError:
        return None
