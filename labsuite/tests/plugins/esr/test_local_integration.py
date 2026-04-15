from __future__ import annotations

import numpy as np
import pytest

from labsuite.core.recipes import EsrPreprocessingRecipe, load_esr_recipe
from labsuite.core.types import PeakWindow, ProcessedTrace
from labsuite.plugins.esr.fitters import derivative_lorentzian
from labsuite.plugins.esr.preprocess import integrate_local_resonance, integrate_local_resonance_with_curves


def test_local_integration_is_robust_to_baseline_offset() -> None:
    reference = _integrate_trace(field_start=0.0, field_end=650.0)
    shifted = _integrate_trace(field_start=0.0, field_end=650.0, baseline_offset=0.04)

    assert reference.area_integral is not None
    assert shifted.area_integral is not None
    assert shifted.area_integral == pytest.approx(reference.area_integral, rel=0.05)


def test_local_integration_is_robust_to_baseline_slope() -> None:
    reference = _integrate_trace(field_start=0.0, field_end=650.0)
    sloped = _integrate_trace(field_start=0.0, field_end=650.0, baseline_slope=1.5e-4)

    assert reference.area_integral is not None
    assert sloped.area_integral is not None
    assert sloped.area_integral == pytest.approx(reference.area_integral, rel=0.08)


def test_local_integration_is_stable_across_field_span() -> None:
    short_span = _integrate_trace(field_start=80.0, field_end=150.0)
    long_span = _integrate_trace(field_start=0.0, field_end=650.0)

    assert short_span.area_integral is not None
    assert long_span.area_integral is not None
    assert long_span.area_integral == pytest.approx(short_span.area_integral, rel=0.05)


def test_local_integration_falls_back_to_constant_baseline_when_off_resonance_points_are_sparse() -> None:
    field = np.linspace(108.0, 122.0, 15)
    signal = derivative_lorentzian(field, amplitude=1.2, center_mT=115.0, gamma_mT=0.5, offset=0.0)
    trace = ProcessedTrace(field_mT=field, signal=np.asarray(signal, dtype=float))
    recipe = EsrPreprocessingRecipe()
    peak_window = PeakWindow(
        label="peak_1",
        start_index=4,
        end_index=10,
        start_field_mT=float(field[4]),
        end_field_mT=float(field[10]),
        peak_index=5,
        trough_index=9,
        peak_field_mT=float(field[5]),
        trough_field_mT=float(field[9]),
        width_mT=float(field[9] - field[5]),
        prominence=1.0,
    )

    integral = integrate_local_resonance(
        trace,
        recipe,
        label="peak_1",
        center_mT=115.0,
        gamma_mT=0.5,
        peak_window=peak_window,
    )

    assert integral.baseline_polyorder == 0
    assert integral.area_integral is not None


def test_local_integration_clips_to_detected_window_guard() -> None:
    field = np.linspace(80.0, 150.0, 8001)
    signal = derivative_lorentzian(field, amplitude=1.2, center_mT=115.0, gamma_mT=4.0, offset=0.0)
    trace = ProcessedTrace(field_mT=field, signal=np.asarray(signal, dtype=float))
    recipe = EsrPreprocessingRecipe()
    peak_window = PeakWindow(
        label="peak_1",
        start_index=3000,
        end_index=5000,
        start_field_mT=110.0,
        end_field_mT=120.0,
        peak_index=3400,
        trough_index=4600,
        peak_field_mT=112.0,
        trough_field_mT=118.0,
        width_mT=6.0,
        prominence=1.0,
    )

    integral = integrate_local_resonance(
        trace,
        recipe,
        label="single",
        center_mT=115.0,
        gamma_mT=4.0,
        peak_window=peak_window,
    )

    assert integral.integration_window_clipped_by_detected_window is True
    assert integral.start_field_mT == pytest.approx(92.0, abs=0.01)
    assert integral.end_field_mT == pytest.approx(138.0, abs=0.01)


def test_local_integration_without_detected_window_keeps_fit_linewidth_window() -> None:
    integral = _integrate_trace(field_start=80.0, field_end=150.0)

    assert integral.integration_window_clipped_by_detected_window is False
    assert integral.start_field_mT == pytest.approx(108.0, rel=1e-3)
    assert integral.end_field_mT == pytest.approx(122.0, rel=1e-3)


def test_default_esr_recipe_uses_wider_local_window_settings(project_root) -> None:
    recipe = load_esr_recipe(project_root / "recipes" / "esr" / "default.yaml")

    assert recipe.integration_window_gamma_multiplier > 0.0
    assert recipe.integration_baseline_window_gamma_multiplier > recipe.integration_window_gamma_multiplier
    assert recipe.integration_detected_window_padding_width_multiplier >= 0.0
    assert recipe.fit_local_disagreement_ratio_threshold > 0.0


def test_local_integration_with_curves_populates_only_selected_window() -> None:
    field = np.linspace(80.0, 150.0, 8001)
    signal = derivative_lorentzian(field, amplitude=1.2, center_mT=115.0, gamma_mT=1.0, offset=0.0)
    trace = ProcessedTrace(field_mT=field, signal=np.asarray(signal, dtype=float))
    recipe = EsrPreprocessingRecipe()

    integral, curves = integrate_local_resonance_with_curves(
        trace,
        recipe,
        label="single",
        center_mT=115.0,
        gamma_mT=1.0,
    )

    assert curves is not None
    inside_window = (field >= integral.start_field_mT) & (field <= integral.end_field_mT)
    outside_window = ~inside_window
    assert np.all(np.isnan(curves.absorption_signal[outside_window]))
    assert np.all(np.isnan(curves.area_signal[outside_window]))
    assert np.all(np.isfinite(curves.absorption_signal[inside_window]))
    assert np.all(np.isfinite(curves.area_signal[inside_window]))
    assert curves.area_signal[np.flatnonzero(inside_window)[-1]] == pytest.approx(integral.area_integral)


def _integrate_trace(
    *,
    field_start: float,
    field_end: float,
    baseline_offset: float = 0.0,
    baseline_slope: float = 0.0,
):
    field = np.linspace(field_start, field_end, 8001)
    signal = derivative_lorentzian(field, amplitude=1.2, center_mT=115.0, gamma_mT=1.0, offset=0.0)
    signal = signal + baseline_offset + baseline_slope * (field - 115.0)
    trace = ProcessedTrace(field_mT=field, signal=np.asarray(signal, dtype=float))
    recipe = EsrPreprocessingRecipe()
    return integrate_local_resonance(
        trace,
        recipe,
        label="single",
        center_mT=115.0,
        gamma_mT=1.0,
    )
