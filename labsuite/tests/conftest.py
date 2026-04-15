from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from labsuite.plugins.esr.fitters import derivative_lorentzian


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def write_bruker_esr_sample():
    def _write(
        path: Path,
        *,
        center_mT: float = 340.0,
        gamma_mT: float = 1.2,
        components: list[dict[str, float]] | None = None,
    ) -> Path:
        descriptor_path = path if path.suffix.lower() == ".dsc" else path.with_suffix(".dsc")
        data_path = descriptor_path.with_suffix(".DTA")

        field_mT = np.linspace(330.0, 350.0, 401)
        component_list = components or [
            {
                "amplitude": 1.3,
                "center_mT": center_mT,
                "gamma_mT": gamma_mT,
                "offset": 0.0,
            }
        ]
        signal = np.zeros_like(field_mT)
        for component in component_list:
            signal += derivative_lorentzian(field_mT, **component)
        signal += 0.018 * np.cos(np.linspace(0.0, 4.0 * np.pi, field_mT.size))
        signal += 0.03

        field_gauss = field_mT * 10.0
        descriptor_path.write_text(
            "\n".join(
                [
                    "#DESC\t1.2\t* DESCRIPTOR INFORMATION",
                    "DSRC\tEXP",
                    "BSEQ\tLIT",
                    "IKKF\tREAL",
                    "XTYP\tIDX",
                    "YTYP\tNODATA",
                    "ZTYP\tNODATA",
                    "IRFMT\tD",
                    "XFMT\tD",
                    f"XPTS\t{field_mT.size}",
                    f"XMIN\t{field_gauss[0]:.10g}",
                    f"XWID\t{field_gauss[-1] - field_gauss[0]:.10g}",
                    "TITL\t'pytest_fixture'",
                    "IRNAM\t'MW_Absorption'",
                    "XNAM\t'BField'",
                    "IRUNI\t''",
                    "XUNI\t'G'",
                    "DATE\t04/13/26",
                    "TIME\t12:00:00",
                    "MWFQ\t9498896736.34545",
                    "MWPW\t0.01",
                    "B0MA\t0.0001",
                    "B0MF\t100000",
                    "STMP\t305.15",
                    "QValue\t1000.0",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        np.asarray(signal, dtype="<f8").tofile(data_path)

        return descriptor_path

    return _write


@pytest.fixture
def bruker_raw_dir(project_root: Path) -> Path:
    return project_root / "data" / "raw"


@pytest.fixture
def bruker_sample_stem(bruker_raw_dir: Path) -> Path:
    matches = sorted(bruker_raw_dir.glob("*.dsc"))
    if not matches:
        pytest.skip("No Bruker .dsc file available in data/raw for acceptance testing.")
    return matches[0].with_suffix("")
