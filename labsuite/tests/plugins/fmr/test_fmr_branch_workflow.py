from __future__ import annotations

from pathlib import Path

import numpy as np

from labsuite.core.recipes import FmrRecipe
from labsuite.core.types import ConvergenceSummary, ResidualSummary
from labsuite.plugins.fmr.branch_tracking import BranchTrackingConfig, assign_branches
from labsuite.plugins.fmr.derived import build_fmr_series
from labsuite.plugins.fmr.fitters import (
    assess_trace_fit_quality,
    fit_fmr_trace,
    mixed_derivative_lorentzian,
)
from labsuite.plugins.fmr.kittel import fit_kittel_branch, gamma_over_2pi_from_g
from labsuite.plugins.fmr.linewidth import fit_linewidth_branch
from labsuite.plugins.fmr.models import FmrComponentFitResult, FmrTraceDataset, FmrTraceFitResult
from labsuite.plugins.fmr.polarity_matching import (
    PolarityMatchConfig,
    match_positive_negative_points,
)
from labsuite.plugins.fmr.preprocess import FmrProcessedTrace


def _trace_from_signal(
    field_mT: np.ndarray, signal: np.ndarray
) -> tuple[FmrTraceDataset, FmrProcessedTrace]:
    raw_trace = FmrTraceDataset(
        trace_id="trace_001",
        source_file=Path("synthetic.log"),
        sample_name="sample",
        frequency_GHz=9.5,
        angle_deg=None,
        temperature_K=300.0,
        field_mT=field_mT,
        signal=signal,
        field_units="mT",
        signal_units="arb",
        sweep_direction="ascending",
        metadata={"selected_signal_channel": "fit_source"},
        fit_source_signal=signal,
    )
    processed_trace = FmrProcessedTrace(
        trace_id="trace_001",
        field_mT=field_mT,
        signal=signal,
        steps=[{"name": "identity", "parameters": {}}],
    )
    return raw_trace, processed_trace


def test_auto_recovers_close_shoulder_peak_when_snr_is_sufficient() -> None:
    field_mT = np.linspace(20.0, 140.0, 1201)
    signal = mixed_derivative_lorentzian(
        field_mT,
        H_res_mT=76.0,
        DeltaH_mT=11.0,
        amplitude_symmetric=45.0,
        amplitude_antisymmetric=7.0,
        baseline_offset=0.0,
        baseline_slope=0.0,
    )
    signal += mixed_derivative_lorentzian(
        field_mT,
        H_res_mT=91.0,
        DeltaH_mT=8.0,
        amplitude_symmetric=22.0,
        amplitude_antisymmetric=3.0,
        baseline_offset=0.0,
        baseline_slope=0.0,
    )
    signal += 0.0005 * np.sin(np.linspace(0.0, 6.0 * np.pi, field_mT.size))
    raw_trace, processed_trace = _trace_from_signal(field_mT, signal)
    recipe = FmrRecipe(
        max_resonance_count=3,
        double_fit_min_improvement_ratio=0.03,
        min_peak_separation_mT=5.0,
        peak_min_prominence_ratio=0.08,
        amplitude_snr_min=3.0,
    )

    result = fit_fmr_trace(raw_trace, processed_trace, recipe)
    assess_trace_fit_quality(result, recipe=recipe)
    centers = sorted(
        component.H_res_mT for component in result.selected_components if component.accepted
    )

    assert result.n_peaks_selected >= 2
    assert abs(centers[0] - 76.0) < 2.0
    assert abs(centers[1] - 91.0) < 2.5


def test_noisy_trace_auto_does_not_select_extra_peak() -> None:
    field_mT = np.linspace(20.0, 140.0, 601)
    signal = 0.002 * np.sin(np.linspace(0.0, 11.0 * np.pi, field_mT.size))
    raw_trace, processed_trace = _trace_from_signal(field_mT, signal)
    recipe = FmrRecipe(max_resonance_count=3, amplitude_snr_min=5.0)

    result = fit_fmr_trace(raw_trace, processed_trace, recipe)

    assert result.n_peaks_selected == 1


def test_n_peaks_two_caps_selection_and_keeps_one_peak_when_extra_is_unjustified() -> None:
    field_mT = np.linspace(0.0, 45.0, 451)
    signal = mixed_derivative_lorentzian(
        field_mT,
        H_res_mT=7.0,
        DeltaH_mT=4.5,
        amplitude_symmetric=20.0,
        amplitude_antisymmetric=2.5,
        baseline_offset=0.0,
        baseline_slope=0.0,
    )
    signal += 0.002 * np.sin(np.linspace(0.0, 4.0 * np.pi, field_mT.size))
    raw_trace, processed_trace = _trace_from_signal(field_mT, signal)
    recipe = FmrRecipe(n_peaks="2", max_resonance_count=2)

    result = fit_fmr_trace(raw_trace, processed_trace, recipe)
    accepted, rejection_reason, _warnings = assess_trace_fit_quality(result, recipe=recipe)

    assert accepted is True
    assert rejection_reason is None
    assert result.n_peaks_selected == 1
    assert len([component for component in result.selected_components if component.accepted]) == 1


def test_branch_tracking_preserves_identity_with_missing_peak() -> None:
    traces = [
        _minimal_fit("t1", 8.0, [70.0, 105.0]),
        _minimal_fit("t2", 9.0, [80.0]),
        _minimal_fit("t3", 10.0, [90.0, 126.0]),
    ]
    assign_branches(traces)

    first_ids = [component.branch_id for component in traces[0].selected_components]
    third_ids = [component.branch_id for component in traces[2].selected_components]

    assert first_ids[0] == third_ids[0]
    assert third_ids[1] is not None


def test_branch_series_uses_only_two_eligible_branches_with_missing_low_frequency_peak() -> None:
    traces = [
        _minimal_fit("t1", 2.0, [10.0]),
        _minimal_fit("t2", 3.0, [20.0]),
        _minimal_fit("t3", 4.0, [30.0, 65.0]),
        _minimal_fit("t4", 5.0, [40.0, 85.0]),
        _minimal_fit("t5", 6.0, [50.0, 105.0]),
    ]
    recipe = FmrRecipe(n_peaks="2", enable_branch_tracking=True, kittel_min_points=3)

    assign_branches(traces, config=BranchTrackingConfig(max_branches=2))
    series = build_fmr_series(traces, recipe=recipe)

    assert sorted(series.series_by_label) == ["branch_1", "branch_2"]


def test_isolated_false_peak_does_not_create_branch_level_series() -> None:
    traces = [
        _minimal_fit("t1", 2.0, [10.0]),
        _minimal_fit("t2", 3.0, [20.0, 95.0]),
        _minimal_fit("t3", 4.0, [30.0]),
        _minimal_fit("t4", 5.0, [40.0]),
        _minimal_fit("t5", 6.0, [50.0]),
    ]
    recipe = FmrRecipe(n_peaks="2", enable_branch_tracking=True, kittel_min_points=3)

    assign_branches(traces, config=BranchTrackingConfig(max_branches=2))
    series = build_fmr_series(traces, recipe=recipe)

    assert sorted(series.series_by_label) == ["branch_1"]
    assert any("branch_2:branch_skipped_insufficient_points" in item for item in series.warnings)


def test_locked_g_kittel_recovers_known_meff() -> None:
    gamma = gamma_over_2pi_from_g(2.1)
    meff_t = 0.72
    h_t = np.linspace(0.08, 0.32, 8)
    frequency = gamma * np.sqrt(h_t * (h_t + meff_t))

    fit = fit_kittel_branch(
        frequency,
        h_t * 1000.0,
        model="ip_simple",
        g_locked=2.1,
        gamma_locked_GHz_per_T=None,
        fit_g=False,
    )

    assert fit.success is True
    assert abs(fit.parameters["mu0_Meff_T"] - meff_t) < 0.02
    assert abs(fit.parameters["g"] - 2.1) < 1e-9


def test_linewidth_fit_marks_too_few_points_diagnostic() -> None:
    fit = fit_linewidth_branch(
        np.asarray([8.0, 9.0, 10.0]),
        np.asarray([6.0, 7.0, 8.0]),
        gamma_over_2pi_GHz_per_T=28.0,
        min_points=4,
    )

    assert fit.success is True
    assert any("insufficient_points" in warning for warning in fit.warnings)
    assert fit.parameters["alpha_eff"] is not None


def test_cross_file_polarity_matching_uses_metadata_and_abs_fields() -> None:
    points = [
        {
            "sample_id": "S1",
            "replicate_id": "R1",
            "geometry": "IP",
            "branch_id": "branch_1",
            "measurement_id": "pos",
            "field_polarity": "positive",
            "frequency_GHz": 9.0,
            "component_id": "p",
            "trace_id": "tp",
            "Hres_raw_mT": 102.0,
            "DeltaH_raw_mT": 8.0,
        },
        {
            "sample_id": "S1",
            "replicate_id": "R1",
            "geometry": "IP",
            "branch_id": "branch_1",
            "measurement_id": "neg",
            "field_polarity": "negative",
            "frequency_GHz": 9.0004,
            "component_id": "n",
            "trace_id": "tn",
            "Hres_raw_mT": -98.0,
            "DeltaH_raw_mT": 10.0,
        },
    ]

    matched, warnings = match_positive_negative_points(
        points, config=PolarityMatchConfig(frequency_tolerance_GHz=0.001)
    )

    assert warnings == []
    assert matched[0]["polarity_pair_status"] == "paired"
    assert matched[0]["Hres_avg_mT"] == 100.0
    assert matched[0]["Hres_asymmetry_mT"] == 4.0


def _minimal_fit(trace_id: str, frequency_GHz: float, centers: list[float]) -> FmrTraceFitResult:
    field = np.linspace(40.0, 150.0, 11)
    components = [
        FmrComponentFitResult(
            component_id=f"{trace_id}:mode_{index}",
            component_label=f"mode_{index}",
            H_res_mT=center,
            DeltaH_mT=8.0,
            amplitude_symmetric=1.0,
            amplitude_antisymmetric=0.1,
            field_mT=field,
            component_signal=np.zeros_like(field),
            accepted=True,
            confidence="high",
        )
        for index, center in enumerate(centers, start=1)
    ]
    return FmrTraceFitResult(
        trace_id=trace_id,
        source_file=Path("synthetic.log"),
        sample_name="sample",
        frequency_GHz=frequency_GHz,
        angle_deg=None,
        temperature_K=None,
        model_name="synthetic",
        signal_channel="fit_source",
        field_mT=field,
        processed_signal=np.zeros_like(field),
        fitted_signal=np.zeros_like(field),
        residual=np.zeros_like(field),
        parameters={},
        parameter_diagnostics={},
        convergence=ConvergenceSummary(True, "ok", 1, 1, False),
        residual_summary=ResidualSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metrics={},
        bound_hits={},
        covariance=None,
        success=True,
        accepted=True,
        rejection_reason=None,
        selected_components=components,
    )
