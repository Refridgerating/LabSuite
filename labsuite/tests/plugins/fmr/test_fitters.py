from __future__ import annotations

from pathlib import Path

import numpy as np

from labsuite.core.recipes import FmrRecipe
from labsuite.plugins.fmr.fitters import assess_trace_fit_quality, detect_candidate_windows, fit_fmr_trace, mixed_derivative_lorentzian
from labsuite.plugins.fmr.models import FmrTraceDataset
from labsuite.plugins.fmr.preprocess import FmrProcessedTrace


def _build_trace(*, double: bool = False) -> tuple[FmrTraceDataset, FmrProcessedTrace]:
    field_mT = np.linspace(20.0, 150.0, 521)
    signal = mixed_derivative_lorentzian(field_mT, H_res_mT=65.0, DeltaH_mT=10.0, amplitude_symmetric=35.0, amplitude_antisymmetric=8.0, baseline_offset=0.02, baseline_slope=0.0004)
    if double:
        signal += mixed_derivative_lorentzian(field_mT, H_res_mT=118.0, DeltaH_mT=8.0, amplitude_symmetric=18.0, amplitude_antisymmetric=4.0, baseline_offset=0.0, baseline_slope=0.0)
    signal += 0.003 * np.cos(np.linspace(0.0, 4.0 * np.pi, field_mT.size))
    raw_trace = FmrTraceDataset(trace_id="trace_001", source_file=Path("synthetic.log"), sample_name="sample", frequency_GHz=9.5, angle_deg=None, temperature_K=300.0, field_mT=field_mT, signal=signal, field_units="mT", signal_units="arb", sweep_direction="ascending", metadata={"selected_signal_channel": "fit_source"}, fit_source_signal=signal)
    processed_trace = FmrProcessedTrace(trace_id="trace_001", field_mT=field_mT, signal=signal, steps=[{"name": "identity", "parameters": {}}])
    return raw_trace, processed_trace


def test_detect_candidate_windows_finds_two_features_for_double_trace() -> None:
    raw_trace, processed_trace = _build_trace(double=True)
    windows = detect_candidate_windows(processed_trace.field_mT, processed_trace.signal, FmrRecipe())
    assert len(windows) == 2
    assert windows[0].candidate_center_mT < windows[1].candidate_center_mT


def test_fit_fmr_trace_keeps_single_mode_for_single_resonance() -> None:
    raw_trace, processed_trace = _build_trace(double=False)
    result = fit_fmr_trace(raw_trace, processed_trace, FmrRecipe())
    accepted, rejection_reason, warnings = assess_trace_fit_quality(result, recipe=FmrRecipe())
    assert result.selected_mode == "single"
    assert len(result.selected_components) == 1
    assert result.selected_components[0].component_label == "single_unassigned"
    assert accepted is True
    assert rejection_reason is None
    assert warnings == []


def test_fit_fmr_trace_selects_double_mode_for_two_resonance_trace() -> None:
    raw_trace, processed_trace = _build_trace(double=True)
    result = fit_fmr_trace(raw_trace, processed_trace, FmrRecipe())
    accepted, rejection_reason, _warnings = assess_trace_fit_quality(result, recipe=FmrRecipe())
    centers = [component.H_res_mT for component in result.selected_components]
    assert result.selected_mode == "double"
    assert [component.component_label for component in result.selected_components] == ["mode_1", "mode_2"]
    assert centers[0] < centers[1]
    assert accepted is True
    assert rejection_reason is None


def test_auto_mode_stays_single_when_double_gain_is_below_threshold() -> None:
    raw_trace, processed_trace = _build_trace(double=True)
    recipe = FmrRecipe(double_fit_min_improvement_ratio=0.95)
    result = fit_fmr_trace(raw_trace, processed_trace, recipe)
    assert result.selected_mode == "single"


def test_partial_double_qc_keeps_valid_component() -> None:
    raw_trace, processed_trace = _build_trace(double=True)
    result = fit_fmr_trace(raw_trace, processed_trace, FmrRecipe())
    assert result.selected_mode == "double"
    result.selected_components[1].H_res_mT = result.selected_components[1].feature_center_mT + 25.0
    accepted, rejection_reason, warnings = assess_trace_fit_quality(result, recipe=FmrRecipe())
    assert accepted is True
    assert rejection_reason is None
    assert result.partial_component_qc is True
    assert result.selected_components[0].accepted is True
    assert result.selected_components[1].accepted is False
    assert any("partial_double_fit" in warning for warning in warnings)
