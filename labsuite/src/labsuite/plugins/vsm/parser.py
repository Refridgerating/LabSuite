"""Parser for Quantum Design VSM `.dat` loop files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from labsuite.core.exceptions import ParseError
from labsuite.plugins.vsm.models import VsmDataset

_FIELD_COLUMN = "Magnetic Field (Oe)"
_MOMENT_COLUMN = "Moment (emu)"
_MOMENT_STDERR_COLUMN = "M. Std. Err. (emu)"
_TEMPERATURE_COLUMN = "Temperature (K)"


def parse_vsm_file(path: Path) -> VsmDataset:
    """Parse one Quantum Design VSM `.dat` file."""

    if not path.exists():
        raise ParseError(f"VSM file does not exist: {path}")
    if path.suffix.lower() != ".dat":
        raise ParseError(f"Expected a Quantum Design .dat file, got: {path.name}")

    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        data_marker_index = lines.index("[Data]")
    except ValueError as exc:
        raise ParseError(f"Missing [Data] section in {path.name}") from exc
    if data_marker_index + 1 >= len(lines):
        raise ParseError(f"Missing data header row in {path.name}")

    raw_header_entries = _parse_header_entries(lines[:data_marker_index])
    data_columns = next(csv.reader([lines[data_marker_index + 1]]))
    column_map = {name: index for index, name in enumerate(data_columns)}
    for required_column in (_FIELD_COLUMN, _MOMENT_COLUMN, _TEMPERATURE_COLUMN):
        if required_column not in column_map:
            raise ParseError(f"Missing required VSM column {required_column!r} in {path.name}")

    acquisition_index: list[int] = []
    field_oe: list[float] = []
    moment_emu: list[float] = []
    moment_std_err_emu: list[float] = []
    temperature_k: list[float] = []

    raw_rows = 0
    valid_rows = 0
    for row_index, raw_line in enumerate(lines[data_marker_index + 2 :]):
        if not raw_line.strip():
            continue
        raw_rows += 1
        row = next(csv.reader([raw_line]))
        field_value = _optional_float(_get_row_value(row, column_map, _FIELD_COLUMN))
        moment_value = _optional_float(_get_row_value(row, column_map, _MOMENT_COLUMN))
        if field_value is None or moment_value is None:
            continue
        temperature_value = _optional_float(_get_row_value(row, column_map, _TEMPERATURE_COLUMN))
        std_err_value = _optional_float(_get_row_value(row, column_map, _MOMENT_STDERR_COLUMN))

        acquisition_index.append(row_index)
        field_oe.append(field_value)
        moment_emu.append(moment_value)
        temperature_k.append(float("nan") if temperature_value is None else temperature_value)
        moment_std_err_emu.append(float("nan") if std_err_value is None else std_err_value)
        valid_rows += 1

    if valid_rows < 3:
        raise ParseError(f"Insufficient valid VSM rows in {path.name}: found {valid_rows}")

    info_entries = {
        entry["values"][-1]: entry["values"][0]
        for entry in raw_header_entries
        if entry["key"] == "INFO" and len(entry["values"]) >= 2
    }

    return VsmDataset(
        source_path=path,
        acquisition_index=np.asarray(acquisition_index, dtype=int),
        field_oe=np.asarray(field_oe, dtype=float),
        field_mT=np.asarray(field_oe, dtype=float) / 10.0,
        moment_emu=np.asarray(moment_emu, dtype=float),
        moment_std_err_emu=np.asarray(moment_std_err_emu, dtype=float),
        temperature_k=np.asarray(temperature_k, dtype=float),
        metadata={
            "parser": "quantum_design_vsm_dat_v1",
            "source_format": "quantum_design_dat",
            "raw_header_entries": raw_header_entries,
            "data_columns": data_columns,
            "row_count": raw_rows,
            "valid_row_count": valid_rows,
            "info": info_entries,
        },
    )


def _parse_header_entries(lines: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("["):
            continue
        values = [value.strip() for value in next(csv.reader([raw_line]))]
        if not values:
            continue
        entries.append(
            {
                "key": values[0],
                "values": values[1:],
            }
        )
    return entries


def _get_row_value(row: list[str], column_map: dict[str, int], name: str) -> str | None:
    index = column_map.get(name)
    if index is None or index >= len(row):
        return None
    return row[index].strip()


def _optional_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None
