from __future__ import annotations

from pathlib import Path

import pytest

from labsuite.cli.main import _parse_layer_argument, main
from labsuite.core.exceptions import WorkflowError
from labsuite.core.sample_registry import load_measurement_ledger, load_registry


def test_sample_cli_add_list_show_register_validate(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "metadata" / "sample_registry.yaml"
    measurement_ledger_path = tmp_path / "metadata" / "measurement_ledger.yaml"
    source = tmp_path / "sample.log"
    source.write_text("raw", encoding="utf-8")

    assert (
        main(
            [
                "sample",
                "add",
                "SAMPLE-A",
                "--sample-registry",
                str(registry_path),
                "--alias",
                "A1",
                "--area-value",
                "2.0",
                "--area-unit",
                "mm^2",
                "--thickness-value",
                "1.0",
                "--thickness-unit",
                "nm",
                "--g-mode",
                "fixed",
                "--g-value",
                "2.05",
            ]
        )
        == 0
    )

    assert main(["sample", "list", "--sample-registry", str(registry_path)]) == 0
    assert "SAMPLE-A" in capsys.readouterr().out

    assert main(["sample", "show", "A1", "--sample-registry", str(registry_path)]) == 0
    assert "g_mode: fixed" in capsys.readouterr().out

    assert (
        main(
            [
                "sample",
                "register-file",
                str(source),
                "--type",
                "fmr",
                "--sample-id",
                "SAMPLE-A",
                "--geometry",
                "ip",
                "--measurement-ledger",
                str(measurement_ledger_path),
                "--sample-registry",
                str(registry_path),
            ]
        )
        == 0
    )

    registry = load_registry(registry_path)
    ledger = load_measurement_ledger(measurement_ledger_path)
    assert "SAMPLE-A" in registry.samples
    assert next(iter(ledger.measurements.values())).type == "fmr"
    assert main(["sample", "validate", "--sample-registry", str(registry_path)]) == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("NiFe:7:nm:true", True),
        ("NiFe:7:nm:yes", True),
        ("NiFe:7:nm:y", True),
        ("NiFe:7:nm:1", True),
        ("NiFe:7:nm:magnetic", True),
        ("Ta:2:nm:false", False),
        ("Ta:2:nm:no", False),
        ("Ta:2:nm:n", False),
        ("Ta:2:nm:0", False),
        ("Ta:2:nm:non_magnetic", False),
    ],
)
def test_layer_argument_parser_boolean_variants(value: str, expected: bool) -> None:
    layer = _parse_layer_argument(value)

    assert layer["material"] in {"NiFe", "Ta"}
    assert layer["thickness"] in {2.0, 7.0}
    assert layer["thickness_unit"] == "nm"
    assert layer["magnetic"] is expected


def test_layer_argument_parser_rejects_bad_layer() -> None:
    with pytest.raises(WorkflowError, match="material:thickness:unit:magnetic"):
        _parse_layer_argument("NiFe:7:nm")


def test_sample_cli_add_manual_magnetic_volume(tmp_path: Path) -> None:
    registry_path = tmp_path / "metadata" / "sample_registry.yaml"

    assert (
        main(
            [
                "sample",
                "add",
                "SAMPLE-VOL",
                "--sample-registry",
                str(registry_path),
                "--magnetic-volume-m3",
                "1.23e-13",
            ]
        )
        == 0
    )

    sample = load_registry(registry_path).samples["SAMPLE-VOL"]
    assert sample.magnetic_volume_m3 == pytest.approx(1.23e-13)
    assert sample.magnetic_volume_source == "manual"


def test_sample_cli_update_saves_estimated_magnetic_volume(tmp_path: Path) -> None:
    registry_path = tmp_path / "metadata" / "sample_registry.yaml"
    assert main(["sample", "add", "SAMPLE-EST", "--sample-registry", str(registry_path)]) == 0

    assert (
        main(
            [
                "sample",
                "update",
                "SAMPLE-EST",
                "--sample-registry",
                str(registry_path),
                "--geometry-shape",
                "rectangle",
                "--length",
                "5",
                "--width",
                "5",
                "--length-unit",
                "mm",
                "--layer",
                "Ta:2:nm:false",
                "--layer",
                "Co:5:nm:true",
                "--layer",
                "NiFe:7:nm:true",
                "--estimate-magnetic-volume",
                "--save-estimated-volume",
            ]
        )
        == 0
    )

    sample = load_registry(registry_path).samples["SAMPLE-EST"]
    assert sample.geometry.shape == "rectangle"
    assert sample.layer_stack[0]["material"] == "Ta"
    assert sample.magnetic_volume_m3 == pytest.approx(25e-6 * 12e-9)
    assert sample.magnetic_volume_source == "estimated"


def test_sample_cli_estimate_without_save_does_not_store_volume(tmp_path: Path) -> None:
    registry_path = tmp_path / "metadata" / "sample_registry.yaml"
    assert main(["sample", "add", "SAMPLE-NOSAVE", "--sample-registry", str(registry_path)]) == 0

    assert (
        main(
            [
                "sample",
                "update",
                "SAMPLE-NOSAVE",
                "--sample-registry",
                str(registry_path),
                "--geometry-shape",
                "custom_area",
                "--area",
                "4",
                "--area-unit",
                "mm^2",
                "--layer",
                "NiFe:2:nm:true",
                "--estimate-magnetic-volume",
            ]
        )
        == 0
    )

    sample = load_registry(registry_path).samples["SAMPLE-NOSAVE"]
    assert sample.magnetic_volume_m3 is None
    assert "Estimated magnetic volume was not saved." in sample.magnetic_volume_warnings


def test_sample_cli_no_volume_prompt_skips_interactive_prompt(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "metadata" / "sample_registry.yaml"
    answers = iter(["SAMPLE-PROMPT"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert (
        main(
            [
                "sample",
                "add",
                "--sample-registry",
                str(registry_path),
                "--interactive",
                "--no-volume-prompt",
            ]
        )
        == 0
    )

    assert "SAMPLE-PROMPT" in load_registry(registry_path).samples
