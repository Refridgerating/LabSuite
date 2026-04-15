from __future__ import annotations

import numpy as np
import pytest
from lmfit import Parameters

from labsuite.core.exceptions import ParseError, WorkflowError
from labsuite.core.preprocessing import scalar_integral
from labsuite.core.types import (
    ConvergenceSummary,
    FeatureSummary,
    FitResult,
    ResidualSummary,
)
from labsuite.plugins.esr.fitters import _build_bound_hits, _build_parameter_diagnostics
from labsuite.core.export.figure_export import _integrated_curve_series, _plotted_integrated_curves
from labsuite.plugins.esr.parser import parse_esr_file
from labsuite.plugins.esr import service as esr_service
from labsuite.plugins.esr.fitters import (
    absorption_lorentzian,
    fit_derivative_lorentzian_in_window as fit_derivative_lorentzian_in_window_impl,
)
from labsuite.plugins.esr.service import analyze_esr_file


def test_esr_service_analyzes_single_file(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_file = write_bruker_esr_sample(tmp_path / "service_trace.dsc", center_mT=339.8, gamma_mT=1.05)
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    result = analyze_esr_file(source_file, recipe_path)

    assert result.dataset.modality == "esr"
    assert result.recipe_name == "esr-default"
    assert len(result.processed.steps) == 3
    assert result.selected_mode == "single"
    assert result.single_fit is not None
    assert abs(result.single_fit.parameters["center_mT"] - 339.8) < 0.25
    assert abs(result.single_fit.parameters["gamma_mT"] - 1.05) < 0.25
    assert result.single_fit.metrics["r_squared"] > 0.98
    assert result.total_integral.label == "total"
    assert result.total_integral.integration_kind == "primary_fit_model"
    assert result.local_total_integral.integration_kind == "primary_local_window"
    assert result.total_integral.window_source == "fit_linewidth"
    assert result.diagnostic_total_integral.integration_kind == "diagnostic_full_span"
    assert result.primary_integrated is not None
    assert result.derivative_baseline.target == "derivative"
    assert result.absorption_baseline.target == "absorption"
    assert result.single_fit.convergence.success is True
    assert result.single_fit.residual_summary.rmse >= 0.0
    assert result.single_fit.feature_summary is not None
    assert result.single_fit.feature_summary.integrated_intensity_proxy is not None
    assert result.single_fit.feature_summary.integrated_intensity_proxy == pytest.approx(result.total_integral.area_integral)
    assert result.single_fit.derived["intensity_method"] == "fit_derived_lorentzian_area_integral"
    assert result.single_fit.derived["local_windowed_intensity_proxy"] == result.local_total_integral.area_integral
    assert result.diagnostic_total_integral.area_integral is not None
    expected_absorption = absorption_lorentzian(
        result.processed.field_mT,
        amplitude=result.single_fit.parameters["amplitude"],
        center_mT=result.single_fit.parameters["center_mT"],
        gamma_mT=result.single_fit.parameters["gamma_mT"],
    )
    assert np.allclose(result.primary_integrated.absorption_signal, expected_absorption)
    assert result.total_integral.area_integral == pytest.approx(
        scalar_integral(result.primary_integrated.field_mT, result.primary_integrated.absorption_signal)
    )
    assert result.fit_local_total_integral.integration_kind == "fit_local_windowed_model"
    if result.fit_local_integrated is not None:
        fit_local_mask = ~np.isnan(result.fit_local_integrated.absorption_signal)
        assert result.fit_local_integrated.start_field_mT == pytest.approx(result.fit_local_total_integral.start_field_mT)
        assert result.fit_local_integrated.end_field_mT == pytest.approx(result.fit_local_total_integral.end_field_mT)
        assert np.all(np.isnan(result.fit_local_integrated.absorption_signal[~fit_local_mask]))
        assert np.all(np.isnan(result.fit_local_integrated.area_signal[~fit_local_mask]))
        assert result.fit_local_total_integral.area_integral == pytest.approx(
            scalar_integral(
                result.fit_local_integrated.field_mT[fit_local_mask],
                result.fit_local_integrated.absorption_signal[fit_local_mask],
            )
        )
    if result.local_integrated is not None:
        local_mask = ~np.isnan(result.local_integrated.absorption_signal)
        assert result.local_integrated.start_field_mT == pytest.approx(result.local_total_integral.start_field_mT)
        assert result.local_integrated.end_field_mT == pytest.approx(result.local_total_integral.end_field_mT)
        assert np.all(np.isnan(result.local_integrated.absorption_signal[~local_mask]))
        assert np.all(np.isnan(result.local_integrated.area_signal[~local_mask]))
    assert result.fit_local_disagreement_flag is False
    assert "center_mT" in result.single_fit.parameter_diagnostics


def test_esr_service_selects_split_mode_for_double_peak(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_file = write_bruker_esr_sample(
        tmp_path / "double_trace.dsc",
        components=[
            {"amplitude": 1.15, "center_mT": 335.0, "gamma_mT": 0.95, "offset": 0.0},
            {"amplitude": 0.9, "center_mT": 345.0, "gamma_mT": 1.1, "offset": 0.0},
        ],
    )
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    result = analyze_esr_file(source_file, recipe_path)

    assert result.selected_mode == "split"
    assert len(result.peak_fits) == 2
    assert result.fit_decision.split_improvement_ratio is not None
    assert result.fit_decision.split_improvement_ratio > result.fit_decision.split_threshold
    assert len(result.peak_integrals) == 2
    assert len(result.fit_local_peak_integrals) == 2
    assert result.primary_integrated is not None
    assert result.fit_local_integrated is not None
    assert result.total_integral.area_integral == pytest.approx(
        sum(peak.area_integral for peak in result.peak_integrals if peak.area_integral is not None)
    )
    assert result.fit_local_total_integral.area_integral == pytest.approx(
        sum(peak.area_integral for peak in result.fit_local_peak_integrals if peak.area_integral is not None)
    )
    expected_absorption = sum(
        absorption_lorentzian(
            result.processed.field_mT,
            amplitude=peak.fit.parameters["amplitude"],
            center_mT=peak.fit.parameters["center_mT"],
            gamma_mT=peak.fit.parameters["gamma_mT"],
        )
        for peak in result.peak_fits
    )
    assert np.allclose(result.primary_integrated.absorption_signal, expected_absorption)
    assert all(peak.fit.metrics["r_squared"] > 0.9 for peak in result.peak_fits)
    assert all(peak.fit.feature_summary is not None for peak in result.peak_fits)
    assert all(peak.fit.residual_summary.rmse >= 0.0 for peak in result.peak_fits)
    assert all(peak.fit.feature_summary.integrated_intensity_proxy == pytest.approx(integral.area_integral) for peak, integral in zip(result.peak_fits, result.peak_integrals, strict=True))


def test_forced_split_mode_rejects_single_peak(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_file = write_bruker_esr_sample(tmp_path / "single_trace.dsc", center_mT=339.8, gamma_mT=1.05)
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    with pytest.raises(WorkflowError, match="two valid peak windows"):
        analyze_esr_file(source_file, recipe_path, fit_mode="split")


def test_bruker_parser_extracts_metadata_and_axis(bruker_sample_stem) -> None:
    dataset = parse_esr_file(bruker_sample_stem.with_suffix(".dsc"))

    assert dataset.field_mT.size == 60000
    assert dataset.signal.size == 60000
    assert dataset.field_mT[0] == pytest.approx(0.0)
    assert dataset.field_mT[-1] == pytest.approx(175.0)
    assert dataset.metadata["parser"] == "bruker_esr_native_v1"

    bruker = dataset.metadata["bruker"]
    assert bruker["point_count"] == 60000
    assert bruker["sweep_start_mT"] == pytest.approx(0.0)
    assert bruker["sweep_width_mT"] == pytest.approx(175.0)
    assert bruker["frequency_GHz"] == pytest.approx(9.49889673634545)
    assert bruker["microwave_power_mW"] == pytest.approx(10.0)
    assert bruker["modulation_amplitude_mT"] == pytest.approx(0.1)
    assert bruker["modulation_frequency_hz"] == pytest.approx(100000.0)
    assert bruker["q_value"] == pytest.approx(1062.78356933594)
    assert bruker["timestamp"] == "2026-02-18T15:26:36"
    assert len(dataset.metadata["raw_descriptor"]) >= 20


def test_bruker_parser_rejects_missing_sibling_dta(tmp_path, write_bruker_esr_sample) -> None:
    descriptor_path = write_bruker_esr_sample(tmp_path / "missing_pair.dsc")
    descriptor_path.with_suffix(".DTA").unlink()

    with pytest.raises(ParseError, match="Missing sibling Bruker data file"):
        parse_esr_file(descriptor_path)


def test_bruker_parser_rejects_point_count_mismatch(tmp_path, write_bruker_esr_sample) -> None:
    descriptor_path = write_bruker_esr_sample(tmp_path / "mismatch_pair.dsc")
    np.asarray([1.0, 2.0, 3.0], dtype="<f8").tofile(descriptor_path.with_suffix(".DTA"))

    with pytest.raises(ParseError, match="data length mismatch"):
        parse_esr_file(descriptor_path)


def test_bruker_csv_matches_native_axis_but_not_raw_signal(bruker_sample_stem) -> None:
    dataset = parse_esr_file(bruker_sample_stem.with_suffix(".dsc"))
    csv_path = bruker_sample_stem.with_suffix(".csv")
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip() == "BField [mT];MW_Absorption []") + 1
    csv_data = np.loadtxt(lines[start:], delimiter=";")

    assert csv_data.shape[0] == dataset.field_mT.size
    assert np.max(np.abs(csv_data[:, 0] - dataset.field_mT)) < 1e-9
    assert np.max(np.abs(csv_data[:, 1] - dataset.signal)) > 1.0


def test_actual_bruker_sample_prefers_split_mode(bruker_sample_stem, project_root) -> None:
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"
    result = analyze_esr_file(bruker_sample_stem.with_suffix(".dsc"), recipe_path)

    assert result.selected_mode == "split"
    assert len(result.peak_fits) == 2
    assert result.fit_decision.split_improvement_ratio is not None
    assert result.fit_decision.split_improvement_ratio > result.fit_decision.split_threshold
    assert result.total_integral.area_integral is not None
    assert result.primary_integrated is not None
    assert result.diagnostic_total_integral.area_integral is not None


def test_parameter_diagnostics_capture_bound_hits_and_missing_stderr() -> None:
    params = Parameters()
    params.add("gamma_mT", value=0.25, min=0.25, max=5.0)
    params.add("center_mT", value=100.0, min=0.0, max=200.0)
    params["gamma_mT"].stderr = None
    params["center_mT"].stderr = 0.5

    diagnostics = _build_parameter_diagnostics(params)
    bound_hits = _build_bound_hits(params)

    assert diagnostics["gamma_mT"].stderr_missing is True
    assert diagnostics["center_mT"].stderr == pytest.approx(0.5)
    assert bound_hits["gamma_mT"] is True
    assert bound_hits["center_mT"] is False


def test_esr_service_uses_fallback_single_fit_and_retains_rejected_global_attempt(
    tmp_path, project_root, write_bruker_esr_sample, monkeypatch
) -> None:
    source_file = write_bruker_esr_sample(tmp_path / "fallback_trace.dsc", center_mT=339.8, gamma_mT=1.0)
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    def _bad_global_fit(trace, *, integrated_intensity_proxy=None):
        del trace, integrated_intensity_proxy
        return _build_fit_result(
            center_mT=25.0,
            gamma_mT=400.0,
            amplitude=5.0,
            offset=0.0,
            hit_center_bound=True,
        )

    monkeypatch.setattr(esr_service, "fit_derivative_lorentzian", _bad_global_fit)
    monkeypatch.setattr(
        esr_service,
        "fit_derivative_lorentzian_in_window",
        fit_derivative_lorentzian_in_window_impl,
    )

    result = analyze_esr_file(source_file, recipe_path)

    assert result.single_fit is not None
    assert result.single_fit.derived["fit_scope"] == "detected_window_fallback"
    assert len(result.single_fit_attempts) == 2
    assert result.single_fit_attempts[0].scope == "global_full_trace"
    assert result.single_fit_attempts[0].accepted is False
    assert result.single_fit_attempts[1].scope == "detected_window_fallback"
    assert result.single_fit_attempts[1].accepted is True
    assert result.single_fit_attempts[1].selected_for_primary is True
    assert result.total_integral.area_integral is not None
    assert result.primary_integrated is not None
    if result.local_integrated is not None:
        assert result.local_integrated.start_field_mT == pytest.approx(result.local_total_integral.start_field_mT)

    plotted_curves, absorption_title, area_title = _plotted_integrated_curves(result)
    assert plotted_curves is result.primary_integrated
    assert absorption_title == "Primary Fit-Derived Absorption Curve"
    assert area_title == "Primary Fit-Derived Area Curve"
    labels = [item["label"] for item in _integrated_curve_series(result)]
    assert "Primary fit-derived" in labels
    assert "Window-matched fit-derived diagnostic" in labels


def test_esr_service_records_failed_fallback_attempt_and_unavailable_primary_integral(
    tmp_path, project_root, write_bruker_esr_sample, monkeypatch
) -> None:
    source_file = write_bruker_esr_sample(tmp_path / "failed_fallback_trace.dsc", center_mT=339.8, gamma_mT=1.0)
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    def _bad_fit(trace, *, integrated_intensity_proxy=None):
        del trace, integrated_intensity_proxy
        return _build_fit_result(
            center_mT=25.0,
            gamma_mT=400.0,
            amplitude=5.0,
            offset=0.0,
            hit_center_bound=True,
        )

    monkeypatch.setattr(esr_service, "fit_derivative_lorentzian", _bad_fit)
    monkeypatch.setattr(esr_service, "fit_derivative_lorentzian_in_window", lambda trace, window: _bad_fit(trace))

    result = analyze_esr_file(source_file, recipe_path)

    assert len(result.single_fit_attempts) == 2
    assert all(attempt.accepted is False for attempt in result.single_fit_attempts)
    assert result.total_integral.area_integral is None
    assert result.primary_integrated is None
    assert result.local_integrated is None
    assert result.single_fit_attempts[0].rejection_reason is not None
    assert result.single_fit_attempts[1].rejection_reason is not None

    plotted_curves, absorption_title, area_title = _plotted_integrated_curves(result)
    assert plotted_curves is result.integrated
    assert absorption_title == "Diagnostic/Fallback Full-Span Absorption Curve"
    assert area_title == "Diagnostic/Fallback Full-Span Area Curve"


def test_esr_service_flags_fit_local_disagreement_without_replacing_primary_output(
    tmp_path, project_root, write_bruker_esr_sample, monkeypatch
) -> None:
    source_file = write_bruker_esr_sample(tmp_path / "qc_trace.dsc", center_mT=339.8, gamma_mT=1.05)
    recipe_path = _write_recipe(
        tmp_path / "qc_recipe.yaml",
        project_root,
        integration_window_gamma_multiplier=4.0,
        integration_baseline_window_gamma_multiplier=8.0,
        integration_detected_window_padding_width_multiplier=0.5,
        fit_local_disagreement_ratio_threshold=0.2,
    )
    real_integrate = esr_service.integrate_local_resonance_with_curves

    def _distorted_local(*args, **kwargs):
        summary, curves = real_integrate(*args, **kwargs)
        summary.area_integral = None if summary.area_integral is None else summary.area_integral * 0.05
        return summary, curves

    monkeypatch.setattr(esr_service, "integrate_local_resonance_with_curves", _distorted_local)

    result = analyze_esr_file(source_file, recipe_path)

    assert result.primary_integrated is not None
    assert result.fit_local_integrated is not None
    assert result.local_integrated is not None
    assert result.fit_local_disagreement_flag is True
    assert result.fit_local_disagreement_ratio is not None
    assert result.fit_local_disagreement_reason is not None
    assert result.single_fit is not None
    assert result.single_fit.derived["fit_local_disagreement_flag"] is True
    assert result.single_fit.derived["fit_local_windowed_intensity_proxy"] == result.fit_local_total_integral.area_integral
    assert result.single_fit.feature_summary is not None
    assert result.single_fit.feature_summary.integrated_intensity_proxy == pytest.approx(result.total_integral.area_integral)


def _write_recipe(path, project_root, **overrides):
    recipe = (project_root / "recipes" / "esr" / "default.yaml").read_text(encoding="utf-8")
    payload: dict[str, object] = {}
    for line in recipe.splitlines():
        stripped = line.split("#", maxsplit=1)[0].strip()
        if not stripped:
            continue
        key, value = stripped.split(":", maxsplit=1)
        payload[key.strip()] = value.strip()
    payload.update({key: str(value).lower() if isinstance(value, bool) else value for key, value in overrides.items()})
    path.write_text("\n".join(f"{key}: {value}" for key, value in payload.items()) + "\n", encoding="utf-8")
    return path


def _build_fit_result(
    *,
    center_mT: float,
    gamma_mT: float,
    amplitude: float,
    offset: float,
    hit_center_bound: bool = False,
) -> FitResult:
    feature_summary = FeatureSummary(
        positive_extremum_field_mT=center_mT - gamma_mT / np.sqrt(3.0),
        negative_extremum_field_mT=center_mT + gamma_mT / np.sqrt(3.0),
        zero_crossing_field_mT=center_mT,
        peak_to_peak_separation_mT=2.0 * gamma_mT / np.sqrt(3.0),
        integrated_intensity_proxy=None,
    )
    return FitResult(
        model_name="derivative_lorentzian",
        parameters={
            "amplitude": amplitude,
            "center_mT": center_mT,
            "gamma_mT": gamma_mT,
            "offset": offset,
        },
        derived={},
        metrics={
            "r_squared": 0.1,
            "chi_square": 1.0,
            "reduced_chi_square": 1.0,
            "sum_squared_residuals": 1.0,
        },
        fitted_signal=np.zeros(10, dtype=float),
        residual=np.zeros(10, dtype=float),
        parameter_diagnostics={},
        convergence=ConvergenceSummary(
            success=True,
            message="mock fit",
            nfev=1,
            nvarys=4,
            errorbars=False,
        ),
        residual_summary=ResidualSummary(
            rss=1.0,
            rmse=1.0,
            mae=1.0,
            max_abs=1.0,
            mean=0.0,
            std=1.0,
        ),
        feature_summary=feature_summary,
        bound_hits={
            "amplitude": False,
            "center_mT": hit_center_bound,
            "gamma_mT": False,
            "offset": False,
        },
        success=True,
    )
