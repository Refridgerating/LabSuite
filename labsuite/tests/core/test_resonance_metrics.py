from __future__ import annotations

import numpy as np
import pytest

from labsuite.core.resonance_metrics import (
    ResonanceMetricsConfig,
    compute_absorption_mode_metrics,
)


def test_symmetric_absorption_metrics_return_expected_widths() -> None:
    field = np.linspace(90.0, 110.0, 4001)
    center = 100.0
    gamma = 1.75
    absorption = 1.0 / (1.0 + ((field - center) / gamma) ** 2)

    metrics = compute_absorption_mode_metrics(
        field,
        absorption,
        hres=center,
        config=ResonanceMetricsConfig(),
        support_start_field_mT=90.0,
        support_end_field_mT=110.0,
    )

    assert metrics.success is True
    assert metrics.peak_field_abs == pytest.approx(center, abs=0.02)
    assert metrics.hwhm_left == pytest.approx(gamma, abs=0.06)
    assert metrics.hwhm_right == pytest.approx(gamma, abs=0.06)
    assert metrics.fwhm == pytest.approx(2.0 * gamma, abs=0.12)
    assert metrics.asymmetry_ratio == pytest.approx(1.0, abs=0.03)


def test_asymmetric_mode_uses_side_aware_area_windows() -> None:
    field = np.linspace(90.0, 110.0, 4001)
    center = 100.0
    left_gamma = 1.1
    right_gamma = 2.4
    absorption = np.where(
        field <= center,
        1.0 / (1.0 + ((field - center) / left_gamma) ** 2),
        1.0 / (1.0 + ((field - center) / right_gamma) ** 2),
    )

    metrics = compute_absorption_mode_metrics(
        field,
        absorption,
        hres=center,
        config=ResonanceMetricsConfig(area_window_mode="side-aware"),
        support_start_field_mT=94.0,
        support_end_field_mT=108.0,
    )

    assert metrics.success is True
    assert metrics.hwhm_left < metrics.hwhm_right
    assert metrics.asymmetry_ratio is not None
    assert metrics.asymmetry_ratio > 1.5
    window = next(item for item in metrics.area_windows if item.multiplier == pytest.approx(1.0))
    assert window.start_field_mT == pytest.approx(center - metrics.hwhm_left)
    assert window.end_field_mT == pytest.approx(center + metrics.hwhm_right)


def test_linear_halfmax_interpolation_is_used() -> None:
    field = np.array([97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0], dtype=float)
    absorption = np.array([0.0, 0.0, 0.6, 1.0, 0.6, 0.0, 0.0], dtype=float)

    metrics = compute_absorption_mode_metrics(
        field,
        absorption,
        hres=100.0,
        config=ResonanceMetricsConfig(),
        support_start_field_mT=97.0,
        support_end_field_mT=103.0,
    )

    assert metrics.success is True
    assert metrics.h_left_half == pytest.approx(98.8333333333, abs=1e-6)
    assert metrics.h_right_half == pytest.approx(101.1666666667, abs=1e-6)


def test_missing_halfmax_crossings_fail_gracefully() -> None:
    field = np.linspace(99.0, 101.0, 401)
    absorption = np.exp(-((field - 100.0) / 0.02) ** 2)

    metrics = compute_absorption_mode_metrics(
        field,
        absorption,
        hres=100.0,
        config=ResonanceMetricsConfig(),
        support_start_field_mT=99.5,
        support_end_field_mT=100.0,
    )

    assert metrics.success is False
    assert metrics.failure_reason == "halfmax_crossing_not_found"
    assert metrics.peak_field_abs == pytest.approx(100.0, abs=0.02)
    assert metrics.fwhm is None
    assert metrics.area_pm_1fwhm is None


def test_compute_full_area_flag_controls_area_full() -> None:
    field = np.linspace(90.0, 110.0, 4001)
    absorption = 1.0 / (1.0 + ((field - 100.0) / 1.5) ** 2)

    disabled = compute_absorption_mode_metrics(
        field,
        absorption,
        hres=100.0,
        config=ResonanceMetricsConfig(compute_full_area=False),
        support_start_field_mT=94.0,
        support_end_field_mT=106.0,
    )
    enabled = compute_absorption_mode_metrics(
        field,
        absorption,
        hres=100.0,
        config=ResonanceMetricsConfig(compute_full_area=True),
        support_start_field_mT=94.0,
        support_end_field_mT=106.0,
    )

    assert disabled.area_full is None
    assert enabled.area_full is not None
    assert enabled.area_full > 0.0
