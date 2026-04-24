from __future__ import annotations

import pytest

from labsuite.core.exceptions import ParseError
from labsuite.plugins.fmr.parser import parse_fmr_file


def test_parse_phasefmr_log_splits_multi_frequency_file(tmp_path, write_phasefmr_log) -> None:
    source_file = write_phasefmr_log(
        tmp_path / "Temp2-Co-A-2,5to17GHz-R1.log",
        frequencies_GHz=[4.0, 6.0, 8.0],
    )

    result = parse_fmr_file(source_file)

    assert result.sample_name == "Temp2-Co-A"
    assert result.replicate_id == "R1"
    assert result.sweep_span_label == "2,5to17GHz"
    assert len(result.traces) == 3
    assert [trace.frequency_GHz for trace in result.traces] == [4.0, 6.0, 8.0]
    assert result.metadata["trace_count"] == 3
    assert result.metadata["has_multiple_frequencies"] is True
    assert result.metadata["frequency_GHz_values"] == [4.0, 6.0, 8.0]
    assert all(trace.field_units == "mT" for trace in result.traces)
    assert all(trace.fit_source_signal is not None for trace in result.traces)
    assert all(trace.metadata["measurement_mode"] == "FMR IP - Dual" for trace in result.traces)


def test_parse_phasefmr_cryo_log_preserves_temperature_metadata(tmp_path, write_phasefmr_log) -> None:
    source_file = write_phasefmr_log(
        tmp_path / "NiFeStd1-03APR2026-R1.log",
        frequencies_GHz=[9.459],
        include_temp=True,
        temperature_K=121.4,
    )

    result = parse_fmr_file(source_file)

    assert result.sample_name == "NiFeStd1-03APR2026"
    assert result.nominal_temperature_K == 121.0
    assert result.metadata["trace_count"] == 1
    assert result.metadata["has_multiple_frequencies"] is False
    assert result.metadata["frequency_GHz_values"] == [9.459]
    assert result.metadata["qd_settings"]["Instrument Type"] == "VersaLab"
    assert result.traces[0].temperature_K == 121.0
    assert result.traces[0].temp_K_signal is not None


def test_parse_fmr_file_rejects_invalid_instrument_type(tmp_path) -> None:
    source_file = tmp_path / "invalid.log"
    source_file.write_text(
        "\n".join(
            [
                "[Instrument Settings]",
                'Instrument type = "NotFMR"',
                "[Sweep settings]",
                "Frequency(GHz)\tField Saturation (Oe)\tField Start (Oe)\tField Stop (Oe)\tField Step (Oe)",
                "9.0\t0\t1000\t2000\t25",
                "[Data]",
                "Frequency\tField\tFit source\tFit\tAux\tTime",
                "9.0\t1000\t1.0\t1.0\t0.0\t0.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ParseError, match="Unsupported FMR instrument type"):
        parse_fmr_file(source_file)
