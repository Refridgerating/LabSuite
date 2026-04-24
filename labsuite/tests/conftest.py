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
        frequency_GHz: float = 9.49889673634545,
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
                    f"MWFQ\t{frequency_GHz * 1e9:.12f}",
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
        matches = sorted((project_root / "data" / "raw" / "VSM").glob("MTJ-B-*.dat"))
    if not matches:
        pytest.skip("No VSM .dat files available in data/raw for acceptance testing.")
    return matches


@pytest.fixture
def build_vsm_loop_arrays():
    def _build(
        *,
        max_field_mT: float = 100.0,
        points_per_branch: int = 81,
        ms_emu: float = 3.5e-5,
        coercive_field_mT: float = 2.5,
        transition_width_mT: float = 4.5,
        background_slope_emu_per_mT: float = 0.0,
        positive_tail_curvature_emu: float = 0.0,
        negative_tail_curvature_emu: float = 0.0,
        increasing_scale: float = 1.0,
        decreasing_scale: float = 1.0,
        final_branch_offset_emu: float = 0.0,
        deterministic_noise_emu: float = 2.0e-7,
        moment_std_err_emu: float = 4.0e-7,
        temperature_K: float = 300.0,
    ) -> dict[str, np.ndarray]:
        branch_0 = np.linspace(-max_field_mT, max_field_mT, points_per_branch, dtype=float)
        branch_1 = np.linspace(max_field_mT, -max_field_mT, points_per_branch, dtype=float)
        branch_2 = np.linspace(-max_field_mT, max_field_mT, points_per_branch, dtype=float)
        field_mT = np.concatenate([branch_0, branch_1[1:], branch_2[1:]])

        branch_ids = np.concatenate(
            [
                np.zeros(branch_0.size, dtype=int),
                np.ones(branch_1.size - 1, dtype=int),
                np.full(branch_2.size - 1, 2, dtype=int),
            ]
        )
        decreasing_mask = branch_ids == 1
        increasing_mask = ~decreasing_mask

        moment_emu = np.zeros_like(field_mT)
        moment_emu[increasing_mask] = increasing_scale * ms_emu * np.tanh(
            (field_mT[increasing_mask] - coercive_field_mT) / transition_width_mT
        )
        moment_emu[decreasing_mask] = decreasing_scale * ms_emu * np.tanh(
            (field_mT[decreasing_mask] + coercive_field_mT) / transition_width_mT
        )

        normalized_field = field_mT / max(max_field_mT, 1e-9)
        positive_tail_component = positive_tail_curvature_emu * np.where(field_mT > 0.0, normalized_field**2, 0.0)
        negative_tail_component = negative_tail_curvature_emu * np.where(field_mT < 0.0, normalized_field**2, 0.0)
        deterministic_noise = deterministic_noise_emu * np.sin(np.linspace(0.0, 6.0 * np.pi, field_mT.size))

        moment_emu = (
            moment_emu
            + background_slope_emu_per_mT * field_mT
            + positive_tail_component
            + negative_tail_component
            + deterministic_noise
        )
        moment_emu[branch_ids == 2] += final_branch_offset_emu

        return {
            "field_mT": np.asarray(field_mT, dtype=float),
            "field_oe": np.asarray(field_mT * 10.0, dtype=float),
            "moment_emu": np.asarray(moment_emu, dtype=float),
            "moment_std_err_emu": np.full(field_mT.size, moment_std_err_emu, dtype=float),
            "temperature_k": np.full(field_mT.size, temperature_K, dtype=float),
            "branch_id": np.asarray(branch_ids, dtype=int),
        }

    return _build


@pytest.fixture
def write_vsm_sample(build_vsm_loop_arrays):
    def _write(
        path: Path,
        *,
        sample_stem: str = "Synthetic-300K-R1_00001",
        **loop_kwargs,
    ) -> Path:
        loop = build_vsm_loop_arrays(**loop_kwargs)
        dat_path = path if path.suffix.lower() == ".dat" else path / f"{sample_stem}.dat"
        dat_path.parent.mkdir(parents=True, exist_ok=True)

        header_lines = [
            "[Header]",
            "TITLE,",
            "INFO,Synthetic VSM Fixture,APPNAME",
            "BYAPP,VSM,2.0,1.0",
            "[Data]",
            "Comment,Time Stamp (sec),Temperature (K),Magnetic Field (Oe),Moment (emu),M. Std. Err. (emu)",
        ]
        data_lines = []
        for index in range(loop["field_mT"].size):
            data_lines.append(
                ",".join(
                    [
                        "",
                        f"{float(index):.6f}",
                        f"{float(loop['temperature_k'][index]):.6f}",
                        f"{float(loop['field_oe'][index]):.12g}",
                        f"{float(loop['moment_emu'][index]):.12g}",
                        f"{float(loop['moment_std_err_emu'][index]):.12g}",
                    ]
                )
            )
        dat_path.write_text("\n".join([*header_lines, *data_lines, ""]), encoding="utf-8")
        return dat_path

    return _write


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
        field_polarities: list[str] | None = None,
        polarity_field_offsets_mT: dict[str, float] | None = None,
    ) -> Path:
        frequencies = frequencies_GHz or [6.0, 8.0, 10.0]
        polarities = field_polarities or [""]
        polarity_offsets = polarity_field_offsets_mT or {}
        base_stem = sample_stem
        if angle_deg is not None and "deg" not in base_stem:
            base_stem = f"{base_stem}-{angle_deg:g}deg"
        log_path = path if path.suffix.lower() == ".log" else path.with_suffix(".log")
        data_lines: list[str] = []
        sweep_lines: list[str] = [
            "Frequency(GHz)\tField Saturation (Oe)\tField Start (Oe)\tField Stop (Oe)\tField Step (Oe)\tInput gain\tOutput gain\tModul. Ampl.\tFMR ADC # of Samples\tFMR ADC Rate"
        ]
        for frequency in frequencies:
            for polarity in polarities:
                polarity_label = polarity.strip()
                polarity_sign = -1.0 if polarity_label.lower() in {"negative", "neg", "minus", "-h"} else 1.0
                resonance_mT = polarity_sign * (
                    25.0 * frequency + resonance_offset_mT + polarity_offsets.get(polarity_label, 0.0)
                )
                lower = resonance_mT - 80.0
                upper = resonance_mT + 80.0
                if polarity_sign > 0:
                    lower = max(5.0, lower)
                else:
                    upper = min(-5.0, upper)
                field_mT = np.linspace(lower, upper, 41)
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
                        H_res_mT=resonance_mT + polarity_sign * secondary_resonance_delta_mT,
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
                        H_res_mT=resonance_mT + polarity_sign * secondary_resonance_delta_mT,
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
                    common = [f"{frequency:.6f}"]
                    if field_polarities is not None:
                        common.append(polarity_label)
                    common.extend(
                        [
                            f"{field_oe[index]:.6f}",
                            f"{i_signal[index]:.6f}",
                            f"{q_signal[index]:.6f}",
                            f"{signal[index]:.6f}",
                            f"{fit_signal[index]:.6f}",
                            f"{aux_signal[index]:.6f}",
                        ]
                    )
                    if include_temp:
                        common.append(f"{temperature_K:.6f}")
                    common.append(f"{time_signal[index]:.6f}")
                    data_lines.append("\t".join(common))

        data_header = "Frequency"
        if field_polarities is not None:
            data_header += "\tPolarity"
        data_header += "\tField\tI\tQ\tFit source\tFit\tAux"
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
