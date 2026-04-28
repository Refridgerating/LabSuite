"""Generate a synthetic Bruker-style ESR DSC+DTA pair for local testing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "esr_sample.dsc",
        help="Destination DSC file. A sibling .DTA will be written automatically.",
    )
    return parser


def main() -> int:
    from labsuite.plugins.esr.fitters import derivative_lorentzian

    args = build_parser().parse_args()
    descriptor_path = args.output.resolve()
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    if descriptor_path.suffix.lower() != ".dsc":
        descriptor_path = descriptor_path.with_suffix(".dsc")
    data_path = descriptor_path.with_suffix(".DTA")

    field_mT = np.linspace(330.0, 350.0, 401)
    signal = derivative_lorentzian(
        field_mT,
        amplitude=1.35,
        center_mT=340.2,
        gamma_mT=1.15,
        offset=0.0,
    )
    signal += 0.02 * np.sin(np.linspace(0.0, 3.0 * np.pi, field_mT.size))
    signal += 0.025

    field_gauss = field_mT * 10.0
    xwid = float(field_gauss[-1] - field_gauss[0])
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
                f"XWID\t{xwid:.10g}",
                "TITL\t'synthetic_esr_sample'",
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

    print(descriptor_path)
    print(data_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
