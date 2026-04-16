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
from labsuite.plugins.fmr.fitters import mixed_derivative_lorentzian


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
        field_start_mT: float = 330.0,
        field_end_mT: float = 350.0,
        point_count: int = 401,
    ) -> Path:
        descriptor_path = path if path.suffix.lower() == ".dsc" else path.with_suffix(".dsc")
        data_path = descriptor_path.with_suffix(".DTA")

        field_mT = np.linspace(field_start_mT, field_end_mT, point_count)
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


@pytest.fixture
def vsm_sample_files(project_root: Path) -> list[Path]:
    matches = sorted((project_root / "data" / "raw").glob("MTJ-B-*.dat"))
    if not matches:
        pytest.skip("No VSM .dat files available in data/raw for acceptance testing.")
    return matches


@pytest.fixture
def fmr_sample_files(project_root: Path) -> list[Path]:
    matches = sorted((project_root / "data" / "raw" / "FMR").glob("*.log"))
    if not matches:
        pytest.skip("No FMR .log files available in data/raw/FMR for acceptance testing.")
    return matches


@pytest.fixture
def cryo_fmr_sample_files(project_root: Path) -> list[Path]:
    matches = sorted((project_root / "data" / "raw" / "CryoFMR").glob("*.log"))
    if not matches:
        pytest.skip("No CryoFMR .log files available in data/raw/CryoFMR for acceptance testing.")
    return matches


@pytest.fixture
def write_phasefmr_log():
    def _write(
        path: Path,
        *,
        sample_stem: str = "Temp2-Co-A-2,5to17GHz-R1",
        frequencies_GHz: list[float] | None = None,
        angle_deg: float | None = None,
        include_temp: bool = False,
        temperature_K: float = 299.9,
        linewidth_mT: float = 22.0,
        resonance_offset_mT: float = 10.0,
        secondary_resonance_delta_mT: float | None = None,
        secondary_linewidth_mT: float | None = None,
        secondary_amplitude_scale: float = 0.55,
    ) -> Path:
        frequencies = frequencies_GHz or [6.0, 8.0, 10.0]
        base_stem = sample_stem
        if angle_deg is not None and "deg" not in base_stem:
            base_stem = f"{base_stem}-{angle_deg:g}deg"
        log_path = path if path.suffix.lower() == ".log" else path.with_suffix(".log")
        data_lines: list[str] = []
        sweep_lines: list[str] = [
            "Frequency(GHz)\tField Saturation (Oe)\tField Start (Oe)\tField Stop (Oe)\tField Step (Oe)\tInput gain\tOutput gain\tModul. Ampl.\tFMR ADC # of Samples\tFMR ADC Rate"
        ]
        for frequency in frequencies:
            resonance_mT = 25.0 * frequency + resonance_offset_mT
            field_mT = np.linspace(max(5.0, resonance_mT - 80.0), resonance_mT + 80.0, 41)
            signal = mixed_derivative_lorentzian(
                field_mT,
                H_res_mT=resonance_mT,
                DeltaH_mT=linewidth_mT,
                amplitude_symmetric=35.0,
                amplitude_antisymmetric=8.0,
                baseline_offset=0.02,
                baseline_slope=0.0006,
            )
            if secondary_resonance_delta_mT is not None:
                signal += mixed_derivative_lorentzian(
                    field_mT,
                    H_res_mT=resonance_mT + secondary_resonance_delta_mT,
                    DeltaH_mT=linewidth_mT if secondary_linewidth_mT is None else secondary_linewidth_mT,
                    amplitude_symmetric=35.0 * secondary_amplitude_scale,
                    amplitude_antisymmetric=8.0 * secondary_amplitude_scale,
                    baseline_offset=0.0,
                    baseline_slope=0.0,
                )
            signal += 0.01 * np.sin(np.linspace(0.0, 2.0 * np.pi, field_mT.size))
            i_signal = 0.4 * signal
            q_signal = 0.6 * signal
            fit_signal = mixed_derivative_lorentzian(
                field_mT,
                H_res_mT=resonance_mT,
                DeltaH_mT=linewidth_mT,
                amplitude_symmetric=34.0,
                amplitude_antisymmetric=7.5,
                baseline_offset=0.015,
                baseline_slope=0.0004,
            )
            if secondary_resonance_delta_mT is not None:
                fit_signal += mixed_derivative_lorentzian(
                    field_mT,
                    H_res_mT=resonance_mT + secondary_resonance_delta_mT,
                    DeltaH_mT=linewidth_mT if secondary_linewidth_mT is None else secondary_linewidth_mT,
                    amplitude_symmetric=34.0 * secondary_amplitude_scale,
                    amplitude_antisymmetric=7.5 * secondary_amplitude_scale,
                    baseline_offset=0.0,
                    baseline_slope=0.0,
                )
            aux_signal = np.linspace(0.2, -0.2, field_mT.size)
            time_signal = np.arange(field_mT.size, dtype=float) * 1.5

            field_oe = field_mT * 10.0
            field_start_oe = float(field_oe[0])
            field_stop_oe = float(field_oe[-1])
            field_step_oe = float(np.median(np.diff(field_oe)))
            sweep_lines.append(
                f"{frequency}\t0\t{field_start_oe:.6f}\t{field_stop_oe:.6f}\t{field_step_oe:.6f}\tx100\tx100\t0.3\t100\t1000"
            )
            for index in range(field_mT.size):
                common = [
                    f"{frequency:.6f}",
                    f"{field_oe[index]:.6f}",
                    f"{i_signal[index]:.6f}",
                    f"{q_signal[index]:.6f}",
                    f"{signal[index]:.6f}",
                    f"{fit_signal[index]:.6f}",
                    f"{aux_signal[index]:.6f}",
                ]
                if include_temp:
                    common.append(f"{temperature_K:.6f}")
                common.append(f"{time_signal[index]:.6f}")
                data_lines.append("\t".join(common))

        data_header = "Frequency\tField\tI\tQ\tFit source\tFit\tAux"
        if include_temp:
            data_header += "\tTemp"
        data_header += "\tTime"
        qd_block = ""
        if include_temp:
            qd_block = '\n[QD]\nInstrument Type = "VersaLab"\nDrive.Mode = "Driven"\n'

        log_path.write_text(
            "\n".join(
                [
                    "[Instrument Settings]",
                    'Instrument type = "PhaseFMR"',
                    'DAQ serial number = "01B11D33"',
                    'SW version = "2.9.0.13"',
                    'FMR.MeasType = "FMR IP - Dual"',
                    'Signal.Signal fitting = "Dual Derivated Lorentz."',
                    'Signal.Fit.Src = "IQ"',
                    'Field.DC field source = "PhaseFMR"',
                    'Field.Field Control = "Closed loop"',
                    qd_block.rstrip(),
                    "[Sweep settings]",
                    *sweep_lines,
                    "[Data]",
                    data_header,
                    *data_lines,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return log_path

    return _write
