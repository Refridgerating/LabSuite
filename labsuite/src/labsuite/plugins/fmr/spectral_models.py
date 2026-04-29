"""Spectral models for field-modulated FMR derivative traces."""

from __future__ import annotations

import numpy as np


def derivative_lorentzian_component(
    field_mT: np.ndarray,
    H_res_mT: float,
    DeltaH_mT: float,
    amplitude_symmetric: float,
    amplitude_antisymmetric: float,
) -> np.ndarray:
    """Mixed derivative Lorentzian component used by PhaseFMR-style fits."""

    field = np.asarray(field_mT, dtype=float)
    delta = field - H_res_mT
    denominator = 4.0 * delta**2 + DeltaH_mT**2
    squared = denominator**2
    symmetric = (4.0 * DeltaH_mT * delta) / squared
    antisymmetric = (DeltaH_mT**2 - 4.0 * delta**2) / squared
    return amplitude_symmetric * symmetric - amplitude_antisymmetric * antisymmetric


def absorption_lorentzian_component(
    field_mT: np.ndarray,
    H_res_mT: float,
    DeltaH_mT: float,
    amplitude_symmetric: float,
    amplitude_antisymmetric: float,
) -> np.ndarray:
    """Absorption-like reconstruction corresponding to the derivative model."""

    field = np.asarray(field_mT, dtype=float)
    delta = field - H_res_mT
    denominator = 4.0 * delta**2 + DeltaH_mT**2
    absorption = (amplitude_symmetric * DeltaH_mT) / (2.0 * denominator)
    absorption += amplitude_antisymmetric * delta / denominator
    if np.max(absorption) < abs(np.min(absorption)):
        absorption = -absorption
    return np.asarray(absorption, dtype=float)


def background_signal(
    field_mT: np.ndarray,
    offset: float,
    slope: float,
    quadratic: float = 0.0,
    *,
    model: str = "linear",
) -> np.ndarray:
    """Shared background for all components in one trace."""

    field = np.asarray(field_mT, dtype=float)
    background = offset + slope * field
    if model == "quadratic":
        center = float(np.mean(field)) if field.size else 0.0
        background = background + quadratic * (field - center) ** 2
    return np.asarray(background, dtype=float)


def multi_component_derivative_lorentzian(
    field_mT: np.ndarray,
    components: list[dict[str, float]],
    *,
    baseline_offset: float,
    baseline_slope: float,
    baseline_quadratic: float = 0.0,
    background_model: str = "linear",
) -> np.ndarray:
    """Sum N derivative Lorentzian components plus one shared background."""

    signal = background_signal(
        field_mT,
        baseline_offset,
        baseline_slope,
        baseline_quadratic,
        model=background_model,
    )
    for component in components:
        signal = signal + derivative_lorentzian_component(
            field_mT,
            component["H_res_mT"],
            component["DeltaH_mT"],
            component["amplitude_symmetric"],
            component["amplitude_antisymmetric"],
        )
    return np.asarray(signal, dtype=float)


def mixed_derivative_lorentzian(
    field_mT: np.ndarray,
    H_res_mT: float,
    DeltaH_mT: float,
    amplitude_symmetric: float,
    amplitude_antisymmetric: float,
    baseline_offset: float,
    baseline_slope: float,
) -> np.ndarray:
    """Legacy one-component model API."""

    return multi_component_derivative_lorentzian(
        field_mT,
        [
            {
                "H_res_mT": H_res_mT,
                "DeltaH_mT": DeltaH_mT,
                "amplitude_symmetric": amplitude_symmetric,
                "amplitude_antisymmetric": amplitude_antisymmetric,
            }
        ],
        baseline_offset=baseline_offset,
        baseline_slope=baseline_slope,
    )


def double_mixed_derivative_lorentzian(
    field_mT: np.ndarray,
    H_res_1_mT: float,
    DeltaH_1_mT: float,
    amplitude_symmetric_1: float,
    amplitude_antisymmetric_1: float,
    H_res_2_mT: float,
    DeltaH_2_mT: float,
    amplitude_symmetric_2: float,
    amplitude_antisymmetric_2: float,
    baseline_offset: float,
    baseline_slope: float,
) -> np.ndarray:
    """Legacy two-component model API."""

    return multi_component_derivative_lorentzian(
        field_mT,
        [
            {
                "H_res_mT": H_res_1_mT,
                "DeltaH_mT": DeltaH_1_mT,
                "amplitude_symmetric": amplitude_symmetric_1,
                "amplitude_antisymmetric": amplitude_antisymmetric_1,
            },
            {
                "H_res_mT": H_res_2_mT,
                "DeltaH_mT": DeltaH_2_mT,
                "amplitude_symmetric": amplitude_symmetric_2,
                "amplitude_antisymmetric": amplitude_antisymmetric_2,
            },
        ],
        baseline_offset=baseline_offset,
        baseline_slope=baseline_slope,
    )


def mixed_absorption_lorentzian(
    field_mT: np.ndarray,
    H_res_mT: float,
    DeltaH_mT: float,
    amplitude_symmetric: float,
    amplitude_antisymmetric: float,
) -> np.ndarray:
    """Legacy absorption reconstruction API."""

    return absorption_lorentzian_component(
        field_mT,
        H_res_mT,
        DeltaH_mT,
        amplitude_symmetric,
        amplitude_antisymmetric,
    )
