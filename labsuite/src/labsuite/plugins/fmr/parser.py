"""Parser for PhaseFMR field-swept log exports."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

from labsuite.core.exceptions import ParseError
from labsuite.plugins.fmr.models import FmrFileDataset, FmrTraceDataset


def parse_fmr_file(
    path: Path,
    *,
    polarity_column: str | None = None,
    positive_labels: list[str] | None = None,
    negative_labels: list[str] | None = None,
) -> FmrFileDataset:
    """Parse a PhaseFMR log file into standardized per-frequency FMR traces."""

    if not path.exists():
        raise ParseError(f"FMR file does not exist: {path}")
    if path.suffix.lower() != ".log":
        raise ParseError(f"Expected a PhaseFMR .log file, got: {path.name}")

    lines = path.read_text(encoding="utf-8").splitlines()
    sections = _split_sections(lines)
    instrument_settings = _parse_key_value_section(sections.get("Instrument Settings"))
    if instrument_settings.get("Instrument type") != "PhaseFMR":
        actual_type = instrument_settings.get("Instrument type")
        raise ParseError(f"Unsupported FMR instrument type in {path.name}: {actual_type!r}")

    sweep_header, sweep_rows = _parse_table_section(sections.get("Sweep settings"), path=path, name="Sweep settings")
    data_header, data_rows = _parse_table_section(sections.get("Data"), path=path, name="Data")
    if "Frequency" not in data_header or "Field" not in data_header:
        raise ParseError(f"Missing required FMR columns in {path.name}: expected Frequency and Field")

    file_metadata, filename_warnings = _parse_filename_metadata(path)
    measurement_mode = instrument_settings.get("FMR.MeasType")
    nominal_temperature_K = _extract_nominal_temperature(data_rows, data_header)

    sweep_rows_by_frequency = {
        _frequency_key(_parse_float(row.get("Frequency(GHz)"), path, "Frequency(GHz)")): row for row in sweep_rows
    }
    grouped_rows = _group_rows_by_frequency_and_polarity(
        data_rows,
        data_header,
        path,
        polarity_column=polarity_column,
        positive_labels=positive_labels or ["positive", "pos", "plus", "+H"],
        negative_labels=negative_labels or ["negative", "neg", "minus", "-H"],
    )
    ordered_frequencies = [frequency for frequency, _polarity, _raw_polarity, _rows in grouped_rows]

    traces: list[FmrTraceDataset] = []
    warnings = list(filename_warnings)
    if polarity_column is not None and polarity_column not in data_header:
        warnings.append(f"field_polarity_column_missing:{polarity_column}")
    for index, (frequency_GHz, field_polarity, raw_polarity_label, rows) in enumerate(grouped_rows, start=1):
        settings_row = sweep_rows_by_frequency.get(_frequency_key(frequency_GHz))
        if settings_row is None:
            warnings.append(f"missing_sweep_settings_for_frequency:{frequency_GHz:.6g}GHz")
        trace = _build_trace_dataset(
            path=path,
            index=index,
            frequency_GHz=frequency_GHz,
            rows=rows,
            data_header=data_header,
            instrument_settings=instrument_settings,
            sweep_settings=settings_row,
            sample_name=file_metadata["sample_name"],
            angle_deg=file_metadata["angle_deg"],
            nominal_temperature_K=nominal_temperature_K,
            field_polarity=field_polarity,
            raw_polarity_label=raw_polarity_label,
            polarity_column=polarity_column,
        )
        traces.append(trace)

    if not traces:
        raise ParseError(f"No valid FMR traces were parsed from {path.name}")

    return FmrFileDataset(
        source_path=path,
        sample_name=file_metadata["sample_name"],
        replicate_id=file_metadata["replicate_id"],
        angle_deg=file_metadata["angle_deg"],
        nominal_temperature_K=nominal_temperature_K,
        sweep_span_label=file_metadata["sweep_span_label"],
        measurement_mode=measurement_mode,
        traces=traces,
        metadata={
            "parser": "phasefmr_log_v1",
            "source_format": "phasefmr_log",
            "instrument_settings": instrument_settings,
            "qd_settings": _parse_key_value_section(sections.get("QD")),
            "sweep_settings_header": sweep_header,
            "sweep_settings_rows": sweep_rows,
            "data_header": data_header,
            "file_metadata": file_metadata,
            "section_names": list(sections),
            "trace_count": len(grouped_rows),
            "has_multiple_frequencies": len(grouped_rows) > 1,
            "frequency_GHz_values": ordered_frequencies,
            "polarity_column": polarity_column,
            "field_polarity_values": sorted(
                {polarity for _frequency, polarity, _raw, _rows in grouped_rows if polarity is not None}
            ),
        },
        warnings=warnings,
    )


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1]
            sections[current_section] = []
            continue
        if current_section is not None:
            sections[current_section].append(raw_line)
    return sections


def _parse_key_value_section(lines: list[str] | None) -> dict[str, str]:
    if not lines:
        return {}
    values: dict[str, str] = {}
    for raw_line in lines:
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", maxsplit=1)
        values[key.strip()] = value.strip().strip('"')
    return values


def _parse_table_section(
    lines: list[str] | None,
    *,
    path: Path,
    name: str,
) -> tuple[list[str], list[dict[str, str]]]:
    if not lines:
        raise ParseError(f"Missing [{name}] section in {path.name}")
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        columns = _split_table_line(raw_line)
        if header is None:
            header = columns
            continue
        if len(columns) < len(header):
            columns.extend([""] * (len(header) - len(columns)))
        rows.append({header[index]: columns[index] for index in range(len(header))})
    if header is None:
        raise ParseError(f"Missing header row in [{name}] section of {path.name}")
    return header, rows


def _split_table_line(line: str) -> list[str]:
    if "\t" in line:
        return next(csv.reader([line], delimiter="\t"))
    return re.split(r"\s{2,}", line.strip())


def _parse_filename_metadata(path: Path) -> tuple[dict[str, Any], list[str]]:
    stem = path.stem.split(" - ", maxsplit=1)[0].strip()
    parts = [part for part in stem.split("-") if part]

    replicate_id: str | None = None
    angle_deg: float | None = None
    sweep_span_label: str | None = None
    sample_parts: list[str] = []
    for part in parts:
        if replicate_id is None and re.fullmatch(r"R\d+", part, flags=re.IGNORECASE):
            replicate_id = part.upper()
            continue
        angle_match = re.fullmatch(r"(-?\d+(?:[.,]\d+)?)deg", part, flags=re.IGNORECASE)
        if angle_deg is None and angle_match is not None:
            angle_deg = float(angle_match.group(1).replace(",", "."))
            continue
        if sweep_span_label is None and re.fullmatch(r"\d+(?:[.,]\d+)?to\d+(?:[.,]\d+)?GHz", part, flags=re.IGNORECASE):
            sweep_span_label = part
            continue
        sample_parts.append(part)

    warnings: list[str] = []
    sample_name = "-".join(sample_parts).strip()
    if not sample_name:
        sample_name = stem
        warnings.append("filename_sample_name_fell_back_to_full_stem")
    if replicate_id is None:
        warnings.append("filename_replicate_id_missing")

    return (
        {
            "sample_name": sample_name,
            "replicate_id": replicate_id,
            "angle_deg": angle_deg,
            "sweep_span_label": sweep_span_label,
            "source_stem": stem,
        },
        warnings,
    )


def _extract_nominal_temperature(data_rows: list[dict[str, str]], data_header: list[str]) -> float | None:
    if "Temp" not in data_header:
        return None
    temperatures = [float(value) for value in (_safe_float(row.get("Temp")) for row in data_rows) if value is not None]
    if not temperatures:
        return None
    return float(round(float(np.median(np.asarray(temperatures, dtype=float)))))


def _group_rows_by_frequency_and_polarity(
    data_rows: list[dict[str, str]],
    data_header: list[str],
    path: Path,
    *,
    polarity_column: str | None,
    positive_labels: list[str],
    negative_labels: list[str],
) -> list[tuple[float, str | None, str | None, list[dict[str, str]]]]:
    grouped: dict[tuple[float, str | None, str | None], list[dict[str, str]]] = {}
    order: list[tuple[float, str | None, str | None]] = []
    use_polarity = polarity_column is not None and polarity_column in data_header
    for row in data_rows:
        frequency = _parse_float(row.get("Frequency"), path, "Frequency")
        field_polarity: str | None = None
        raw_polarity: str | None = None
        if use_polarity:
            raw_polarity = (row.get(polarity_column) or "").strip() or None
            field_polarity = _normalize_field_polarity(
                raw_polarity,
                positive_labels=positive_labels,
                negative_labels=negative_labels,
            )
        key = (frequency, field_polarity, raw_polarity)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)
    return [
        (frequency, polarity, raw_polarity, grouped[(frequency, polarity, raw_polarity)])
        for frequency, polarity, raw_polarity in order
    ]


def _build_trace_dataset(
    *,
    path: Path,
    index: int,
    frequency_GHz: float,
    rows: list[dict[str, str]],
    data_header: list[str],
    instrument_settings: dict[str, str],
    sweep_settings: dict[str, str] | None,
    sample_name: str,
    angle_deg: float | None,
    nominal_temperature_K: float | None,
    field_polarity: str | None,
    raw_polarity_label: str | None,
    polarity_column: str | None,
) -> FmrTraceDataset:
    field_oe = np.asarray([_parse_float(row.get("Field"), path, "Field") for row in rows], dtype=float)
    field_mT = field_oe / 10.0
    i_signal = _optional_column(rows, "I")
    q_signal = _optional_column(rows, "Q")
    fit_source_signal = _optional_column(rows, "Fit source")
    fit_signal = _optional_column(rows, "Fit")
    aux_signal = _optional_column(rows, "Aux")
    temp_signal = _optional_column(rows, "Temp")
    time_signal = _optional_column(rows, "Time")

    channel_priority = [
        ("fit_source", fit_source_signal),
        ("fit", fit_signal),
        ("i", i_signal),
        ("q", q_signal),
        ("aux", aux_signal),
    ]
    selected_name, selected_signal = next(
        ((name, signal) for name, signal in channel_priority if signal is not None),
        (None, None),
    )
    if selected_name is None or selected_signal is None:
        raise ParseError(f"Unable to determine a numeric FMR signal channel for {path.name} at {frequency_GHz:.6g} GHz")

    temperature_K = None
    if temp_signal is not None and np.any(np.isfinite(temp_signal)):
        temperature_K = float(round(float(np.nanmedian(temp_signal))))
    else:
        temperature_K = nominal_temperature_K

    sweep_direction: str | None = None
    if field_mT.size >= 2:
        if field_mT[-1] > field_mT[0]:
            sweep_direction = "ascending"
        elif field_mT[-1] < field_mT[0]:
            sweep_direction = "descending"

    polarity_suffix = "" if field_polarity is None else f"_{field_polarity}"
    trace_id = f"trace_{index:03d}_{frequency_GHz:.6f}GHz{polarity_suffix}"
    return FmrTraceDataset(
        trace_id=trace_id,
        source_file=path,
        sample_name=sample_name,
        frequency_GHz=frequency_GHz,
        angle_deg=angle_deg,
        temperature_K=temperature_K,
        field_mT=field_mT,
        signal=np.asarray(selected_signal, dtype=float),
        field_units="mT",
        signal_units="arb",
        sweep_direction=sweep_direction,
        metadata={
            "selected_signal_channel": selected_name,
            "measurement_mode": instrument_settings.get("FMR.MeasType"),
            "signal_fitting": instrument_settings.get("Signal.Signal fitting"),
            "signal_fit_source": instrument_settings.get("Signal.Fit.Src"),
            "field_control": instrument_settings.get("Field.Field Control"),
            "field_source": instrument_settings.get("Field.DC field source"),
            "raw_field_units": "Oe",
            "raw_signal_units": "arb",
            "sweep_settings": sweep_settings,
            "data_header": data_header,
            "point_count": int(field_mT.size),
            "field_polarity": field_polarity,
            "field_polarity_raw": raw_polarity_label,
            "field_polarity_column": polarity_column,
        },
        i_signal=i_signal,
        q_signal=q_signal,
        fit_source_signal=fit_source_signal,
        fit_signal=fit_signal,
        aux_signal=aux_signal,
        temp_K_signal=temp_signal,
        time_s_signal=time_signal,
    )


def _optional_column(rows: list[dict[str, str]], column_name: str) -> np.ndarray | None:
    if not rows or column_name not in rows[0]:
        return None
    values = [_safe_float(row.get(column_name)) for row in rows]
    if all(value is None for value in values):
        return None
    normalized = [float("nan") if value is None else value for value in values]
    return np.asarray(normalized, dtype=float)


def _safe_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_float(value: str | None, path: Path, label: str) -> float:
    parsed = _safe_float(value)
    if parsed is None:
        raise ParseError(f"Invalid numeric value for {label!r} in {path.name}: {value!r}")
    return parsed


def _frequency_key(value: float) -> str:
    return f"{value:.6f}"


def _normalize_field_polarity(
    value: str | None,
    *,
    positive_labels: list[str],
    negative_labels: list[str],
) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    positive = {label.strip().lower() for label in positive_labels}
    negative = {label.strip().lower() for label in negative_labels}
    if normalized in positive:
        return "positive"
    if normalized in negative:
        return "negative"
    return "unknown"
