from __future__ import annotations

from pathlib import Path

from labsuite.cli.main import main
from labsuite.core.sample_registry import load_registry


def test_sample_cli_add_list_show_register_validate(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "metadata" / "sample_registry.yaml"
    source = tmp_path / "sample.log"
    source.write_text("raw", encoding="utf-8")

    assert main(
        [
            "sample",
            "add",
            "SAMPLE-A",
            "--registry",
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
    ) == 0

    assert main(["sample", "list", "--registry", str(registry_path)]) == 0
    assert "SAMPLE-A" in capsys.readouterr().out

    assert main(["sample", "show", "A1", "--registry", str(registry_path)]) == 0
    assert "g_mode: fixed" in capsys.readouterr().out

    assert main(
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
            "--registry",
            str(registry_path),
        ]
    ) == 0

    registry = load_registry(registry_path)
    assert registry.samples["SAMPLE-A"].measurements[0].type == "fmr"
    assert main(["sample", "validate", "--registry", str(registry_path)]) == 0
