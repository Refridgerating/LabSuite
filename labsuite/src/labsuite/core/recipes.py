"""Recipe loading utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from labsuite.core.exceptions import RecipeError

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML is absent
    yaml = None


@dataclass(slots=True)
class EsrPreprocessingRecipe:
    """Recipe for the first ESR preprocessing path."""

    name: str = "esr-default"
    derivative_baseline_edge_points: int = 64
    absorption_baseline_edge_points: int = 64
    savgol_window: int = 11
    savgol_polyorder: int = 3
    normalize: bool = False
    fit_mode: str = "auto"
    peak_min_prominence_ratio: float = 0.12
    peak_min_distance_mT: float = 8.0
    peak_min_pair_width_mT: float = 1.0
    split_min_improvement_ratio: float = 0.15
    integration_baseline_polyorder: int = 1
    integration_window_gamma_multiplier: float = 7.0
    integration_window_min_half_width_mT: float = 2.0
    integration_baseline_window_gamma_multiplier: float = 14.0
    integration_baseline_window_min_half_width_mT: float = 6.0
    integration_detected_window_padding_width_multiplier: float = 3.0
    fit_max_gamma_as_sweep_fraction: float = 0.5
    fit_local_disagreement_ratio_threshold: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VsmPreprocessingRecipe:
    """Recipe for the first VSM loop-processing path."""

    name: str = "vsm-default"
    background_tail_fraction: float = 0.12
    background_min_points_per_side: int = 24
    background_tail_flatness_ratio_tolerance: float = 0.08
    background_slope_disagreement_ratio_tolerance: float = 0.35
    center_loop: bool = False
    smoothing_enabled: bool = False
    smoothing_window: int = 0
    smoothing_polyorder: int = 0
    uncertainty_scale: float = 1.0
    uncertainty_zero_field_window_width_mT: float = 5.0
    uncertainty_zero_field_min_points: int = 5
    uncertainty_switching_half_width_mT: float = 3.0
    uncertainty_switching_min_points: int = 4
    uncertainty_loop_area_smoothing_window: int = 9
    uncertainty_loop_area_smoothing_polyorder: int = 2
    uncertainty_min_switching_slope_emu_per_mT: float = 1e-7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FmrRecipe:
    """Recipe for the first FMR field-swept analysis path."""

    name: str = "fmr-default"
    signal_channel: str = "fit_source"
    baseline_edge_points: int = 4
    baseline_enabled: bool = True
    smoothing_enabled: bool = False
    smoothing_window: int = 7
    smoothing_polyorder: int = 2
    fit_mode: str = "auto"
    trace_fit_model: str = "mixed_derivative_lorentzian"
    physics_model: str = "ip_field_swept_kittel"
    peak_min_prominence_ratio: float = 0.2
    peak_min_distance_mT: float = 8.0
    peak_min_pair_width_mT: float = 4.0
    candidate_window_padding_width_multiplier: float = 1.5
    double_fit_min_improvement_ratio: float = 0.15
    max_resonance_count: int = 2
    field_guard_fraction: float = 0.05
    linewidth_max_sweep_fraction: float = 0.35
    residual_rmse_max_signal_fraction: float = 0.12
    amplitude_snr_min: float = 5.0
    shape_center_tolerance_linewidth_fraction: float = 0.35
    shape_center_tolerance_min_mT: float = 1.5
    shape_pair_prominence_ratio: float = 0.20
    shape_consistency_policy: str = "reject"
    critical_bound_hit_policy: str = "reject"
    kittel_min_points: int = 3
    linewidth_min_points: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_esr_recipe(path: Path) -> EsrPreprocessingRecipe:
    """Load the ESR preprocessing recipe from YAML or a simple fallback mapping."""

    payload = _load_mapping(path)
    recipe = EsrPreprocessingRecipe(
        name=str(payload.get("name", "esr-default")),
        derivative_baseline_edge_points=int(
            payload.get(
                "derivative_baseline_edge_points",
                payload.get("baseline_edge_points", 64),
            )
        ),
        absorption_baseline_edge_points=int(payload.get("absorption_baseline_edge_points", 64)),
        savgol_window=int(payload.get("savgol_window", 11)),
        savgol_polyorder=int(payload.get("savgol_polyorder", 3)),
        normalize=bool(payload.get("normalize", False)),
        fit_mode=str(payload.get("fit_mode", "auto")),
        peak_min_prominence_ratio=float(payload.get("peak_min_prominence_ratio", 0.12)),
        peak_min_distance_mT=float(payload.get("peak_min_distance_mT", 8.0)),
        peak_min_pair_width_mT=float(payload.get("peak_min_pair_width_mT", 1.0)),
        split_min_improvement_ratio=float(payload.get("split_min_improvement_ratio", 0.15)),
        integration_baseline_polyorder=int(payload.get("integration_baseline_polyorder", 1)),
        integration_window_gamma_multiplier=float(payload.get("integration_window_gamma_multiplier", 7.0)),
        integration_window_min_half_width_mT=float(payload.get("integration_window_min_half_width_mT", 2.0)),
        integration_baseline_window_gamma_multiplier=float(
            payload.get("integration_baseline_window_gamma_multiplier", 14.0)
        ),
        integration_baseline_window_min_half_width_mT=float(
            payload.get("integration_baseline_window_min_half_width_mT", 6.0)
        ),
        integration_detected_window_padding_width_multiplier=float(
            payload.get("integration_detected_window_padding_width_multiplier", 3.0)
        ),
        fit_max_gamma_as_sweep_fraction=float(payload.get("fit_max_gamma_as_sweep_fraction", 0.5)),
        fit_local_disagreement_ratio_threshold=float(payload.get("fit_local_disagreement_ratio_threshold", 0.35)),
    )
    _validate_recipe(recipe)
    return recipe


def load_vsm_recipe(path: Path) -> VsmPreprocessingRecipe:
    """Load the VSM preprocessing recipe from YAML or a simple fallback mapping."""

    payload = _load_mapping(path)
    recipe = VsmPreprocessingRecipe(
        name=str(payload.get("name", "vsm-default")),
        background_tail_fraction=float(payload.get("background_tail_fraction", 0.12)),
        background_min_points_per_side=int(payload.get("background_min_points_per_side", 24)),
        background_tail_flatness_ratio_tolerance=float(
            payload.get("background_tail_flatness_ratio_tolerance", 0.08)
        ),
        background_slope_disagreement_ratio_tolerance=float(
            payload.get("background_slope_disagreement_ratio_tolerance", 0.35)
        ),
        center_loop=bool(payload.get("center_loop", False)),
        smoothing_enabled=bool(payload.get("smoothing_enabled", False)),
        smoothing_window=int(payload.get("smoothing_window", 0)),
        smoothing_polyorder=int(payload.get("smoothing_polyorder", 0)),
        uncertainty_scale=float(payload.get("uncertainty_scale", 1.0)),
        uncertainty_zero_field_window_width_mT=float(payload.get("uncertainty_zero_field_window_width_mT", 5.0)),
        uncertainty_zero_field_min_points=int(payload.get("uncertainty_zero_field_min_points", 5)),
        uncertainty_switching_half_width_mT=float(payload.get("uncertainty_switching_half_width_mT", 3.0)),
        uncertainty_switching_min_points=int(payload.get("uncertainty_switching_min_points", 4)),
        uncertainty_loop_area_smoothing_window=int(payload.get("uncertainty_loop_area_smoothing_window", 9)),
        uncertainty_loop_area_smoothing_polyorder=int(payload.get("uncertainty_loop_area_smoothing_polyorder", 2)),
        uncertainty_min_switching_slope_emu_per_mT=float(
            payload.get("uncertainty_min_switching_slope_emu_per_mT", 1e-7)
        ),
    )
    _validate_vsm_recipe(recipe)
    return recipe


def load_fmr_recipe(path: Path) -> FmrRecipe:
    """Load the FMR analysis recipe from YAML or a simple fallback mapping."""

    payload = _load_mapping(path)
    recipe = FmrRecipe(
        name=str(payload.get("name", "fmr-default")),
        signal_channel=str(payload.get("signal_channel", "fit_source")),
        baseline_edge_points=int(payload.get("baseline_edge_points", 4)),
        baseline_enabled=bool(payload.get("baseline_enabled", True)),
        smoothing_enabled=bool(payload.get("smoothing_enabled", False)),
        smoothing_window=int(payload.get("smoothing_window", 7)),
        smoothing_polyorder=int(payload.get("smoothing_polyorder", 2)),
        fit_mode=str(payload.get("fit_mode", "auto")),
        trace_fit_model=str(payload.get("trace_fit_model", "mixed_derivative_lorentzian")),
        physics_model=str(payload.get("physics_model", "ip_field_swept_kittel")),
        peak_min_prominence_ratio=float(payload.get("peak_min_prominence_ratio", 0.2)),
        peak_min_distance_mT=float(payload.get("peak_min_distance_mT", 8.0)),
        peak_min_pair_width_mT=float(payload.get("peak_min_pair_width_mT", 4.0)),
        candidate_window_padding_width_multiplier=float(
            payload.get("candidate_window_padding_width_multiplier", 1.5)
        ),
        double_fit_min_improvement_ratio=float(payload.get("double_fit_min_improvement_ratio", 0.15)),
        max_resonance_count=int(payload.get("max_resonance_count", 2)),
        field_guard_fraction=float(payload.get("field_guard_fraction", 0.05)),
        linewidth_max_sweep_fraction=float(payload.get("linewidth_max_sweep_fraction", 0.35)),
        residual_rmse_max_signal_fraction=float(payload.get("residual_rmse_max_signal_fraction", 0.12)),
        amplitude_snr_min=float(payload.get("amplitude_snr_min", 5.0)),
        shape_center_tolerance_linewidth_fraction=float(
            payload.get("shape_center_tolerance_linewidth_fraction", 0.35)
        ),
        shape_center_tolerance_min_mT=float(payload.get("shape_center_tolerance_min_mT", 1.5)),
        shape_pair_prominence_ratio=float(payload.get("shape_pair_prominence_ratio", 0.20)),
        shape_consistency_policy=str(payload.get("shape_consistency_policy", "reject")),
        critical_bound_hit_policy=str(payload.get("critical_bound_hit_policy", "reject")),
        kittel_min_points=int(payload.get("kittel_min_points", 3)),
        linewidth_min_points=int(payload.get("linewidth_min_points", 2)),
    )
    _validate_fmr_recipe(recipe)
    return recipe


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RecipeError(f"Recipe file does not exist: {path}")

    raw_text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(raw_text) or {}
        if not isinstance(data, dict):
            raise RecipeError(f"Recipe must deserialize to a mapping: {path}")
        return data

    data: dict[str, Any] = {}
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.split("#", maxsplit=1)[0].strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise RecipeError(f"Invalid recipe line {line_number} in {path}: {line!r}")
        key, raw_value = stripped.split(":", maxsplit=1)
        data[key.strip()] = _parse_scalar(raw_value.strip())
    return data


def _parse_scalar(value: str) -> Any:
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def _validate_recipe(recipe: EsrPreprocessingRecipe) -> None:
    if recipe.derivative_baseline_edge_points < 2:
        raise RecipeError("derivative_baseline_edge_points must be at least 2")
    if recipe.absorption_baseline_edge_points < 2:
        raise RecipeError("absorption_baseline_edge_points must be at least 2")
    if recipe.savgol_window < 3:
        raise RecipeError("savgol_window must be at least 3")
    if recipe.savgol_polyorder < 1:
        raise RecipeError("savgol_polyorder must be at least 1")
    if recipe.savgol_polyorder >= recipe.savgol_window:
        raise RecipeError("savgol_polyorder must be smaller than savgol_window")
    if recipe.integration_baseline_polyorder < 0:
        raise RecipeError("integration_baseline_polyorder must be zero or positive")
    if recipe.fit_mode not in {"auto", "single", "split"}:
        raise RecipeError("fit_mode must be one of: auto, single, split")
    if recipe.peak_min_prominence_ratio <= 0.0:
        raise RecipeError("peak_min_prominence_ratio must be positive")
    if recipe.peak_min_distance_mT <= 0.0:
        raise RecipeError("peak_min_distance_mT must be positive")
    if recipe.peak_min_pair_width_mT <= 0.0:
        raise RecipeError("peak_min_pair_width_mT must be positive")
    if not 0.0 <= recipe.split_min_improvement_ratio <= 1.0:
        raise RecipeError("split_min_improvement_ratio must be between 0 and 1")
    if recipe.integration_window_gamma_multiplier <= 0.0:
        raise RecipeError("integration_window_gamma_multiplier must be positive")
    if recipe.integration_window_min_half_width_mT <= 0.0:
        raise RecipeError("integration_window_min_half_width_mT must be positive")
    if recipe.integration_baseline_window_gamma_multiplier <= 0.0:
        raise RecipeError("integration_baseline_window_gamma_multiplier must be positive")
    if recipe.integration_baseline_window_min_half_width_mT <= 0.0:
        raise RecipeError("integration_baseline_window_min_half_width_mT must be positive")
    if recipe.integration_detected_window_padding_width_multiplier < 0.0:
        raise RecipeError("integration_detected_window_padding_width_multiplier must be zero or positive")
    if not 0.0 < recipe.fit_max_gamma_as_sweep_fraction <= 1.0:
        raise RecipeError("fit_max_gamma_as_sweep_fraction must be between 0 and 1")
    if recipe.fit_local_disagreement_ratio_threshold <= 0.0:
        raise RecipeError("fit_local_disagreement_ratio_threshold must be positive")


def _validate_vsm_recipe(recipe: VsmPreprocessingRecipe) -> None:
    if not 0.0 < recipe.background_tail_fraction < 0.5:
        raise RecipeError("background_tail_fraction must be between 0 and 0.5")
    if recipe.background_min_points_per_side < 2:
        raise RecipeError("background_min_points_per_side must be at least 2")
    if recipe.background_tail_flatness_ratio_tolerance <= 0.0:
        raise RecipeError("background_tail_flatness_ratio_tolerance must be positive")
    if recipe.background_slope_disagreement_ratio_tolerance <= 0.0:
        raise RecipeError("background_slope_disagreement_ratio_tolerance must be positive")
    if recipe.smoothing_window < 0:
        raise RecipeError("smoothing_window must be zero or positive")
    if recipe.smoothing_polyorder < 0:
        raise RecipeError("smoothing_polyorder must be zero or positive")
    if recipe.smoothing_enabled:
        if recipe.smoothing_window < 3:
            raise RecipeError("smoothing_window must be at least 3 when smoothing_enabled is true")
        if recipe.smoothing_polyorder >= recipe.smoothing_window:
            raise RecipeError("smoothing_polyorder must be smaller than smoothing_window")
    if recipe.uncertainty_scale <= 0.0:
        raise RecipeError("uncertainty_scale must be positive")
    if recipe.uncertainty_zero_field_window_width_mT <= 0.0:
        raise RecipeError("uncertainty_zero_field_window_width_mT must be positive")
    if recipe.uncertainty_zero_field_min_points < 2:
        raise RecipeError("uncertainty_zero_field_min_points must be at least 2")
    if recipe.uncertainty_switching_half_width_mT <= 0.0:
        raise RecipeError("uncertainty_switching_half_width_mT must be positive")
    if recipe.uncertainty_switching_min_points < 2:
        raise RecipeError("uncertainty_switching_min_points must be at least 2")
    if recipe.uncertainty_loop_area_smoothing_window < 3:
        raise RecipeError("uncertainty_loop_area_smoothing_window must be at least 3")
    if recipe.uncertainty_loop_area_smoothing_polyorder < 1:
        raise RecipeError("uncertainty_loop_area_smoothing_polyorder must be at least 1")
    if recipe.uncertainty_loop_area_smoothing_polyorder >= recipe.uncertainty_loop_area_smoothing_window:
        raise RecipeError(
            "uncertainty_loop_area_smoothing_polyorder must be smaller than uncertainty_loop_area_smoothing_window"
        )
    if recipe.uncertainty_min_switching_slope_emu_per_mT <= 0.0:
        raise RecipeError("uncertainty_min_switching_slope_emu_per_mT must be positive")


def _validate_fmr_recipe(recipe: FmrRecipe) -> None:
    if recipe.signal_channel not in {"i", "q", "fit_source", "fit", "aux"}:
        raise RecipeError("signal_channel must be one of: i, q, fit_source, fit, aux")
    if recipe.baseline_edge_points < 2:
        raise RecipeError("baseline_edge_points must be at least 2")
    if recipe.smoothing_window < 0:
        raise RecipeError("smoothing_window must be zero or positive")
    if recipe.smoothing_polyorder < 0:
        raise RecipeError("smoothing_polyorder must be zero or positive")
    if recipe.smoothing_enabled:
        if recipe.smoothing_window < 3:
            raise RecipeError("smoothing_window must be at least 3 when smoothing_enabled is true")
        if recipe.smoothing_polyorder >= recipe.smoothing_window:
            raise RecipeError("smoothing_polyorder must be smaller than smoothing_window")
    if recipe.fit_mode not in {"auto", "single", "double"}:
        raise RecipeError("fit_mode must be one of: auto, single, double")
    if recipe.trace_fit_model != "mixed_derivative_lorentzian":
        raise RecipeError("trace_fit_model must be 'mixed_derivative_lorentzian'")
    if recipe.physics_model != "ip_field_swept_kittel":
        raise RecipeError("physics_model must be 'ip_field_swept_kittel'")
    if recipe.peak_min_prominence_ratio <= 0.0:
        raise RecipeError("peak_min_prominence_ratio must be positive")
    if recipe.peak_min_distance_mT <= 0.0:
        raise RecipeError("peak_min_distance_mT must be positive")
    if recipe.peak_min_pair_width_mT <= 0.0:
        raise RecipeError("peak_min_pair_width_mT must be positive")
    if recipe.candidate_window_padding_width_multiplier < 0.0:
        raise RecipeError("candidate_window_padding_width_multiplier must be zero or positive")
    if not 0.0 <= recipe.double_fit_min_improvement_ratio <= 1.0:
        raise RecipeError("double_fit_min_improvement_ratio must be between 0 and 1")
    if recipe.max_resonance_count != 2:
        raise RecipeError("max_resonance_count must be 2 for FMR v1")
    if not 0.0 <= recipe.field_guard_fraction < 0.5:
        raise RecipeError("field_guard_fraction must be between 0 and 0.5")
    if not 0.0 < recipe.linewidth_max_sweep_fraction <= 1.0:
        raise RecipeError("linewidth_max_sweep_fraction must be between 0 and 1")
    if recipe.residual_rmse_max_signal_fraction <= 0.0:
        raise RecipeError("residual_rmse_max_signal_fraction must be positive")
    if recipe.amplitude_snr_min <= 0.0:
        raise RecipeError("amplitude_snr_min must be positive")
    if recipe.shape_center_tolerance_linewidth_fraction <= 0.0:
        raise RecipeError("shape_center_tolerance_linewidth_fraction must be positive")
    if recipe.shape_center_tolerance_min_mT <= 0.0:
        raise RecipeError("shape_center_tolerance_min_mT must be positive")
    if recipe.shape_pair_prominence_ratio <= 0.0:
        raise RecipeError("shape_pair_prominence_ratio must be positive")
    if recipe.shape_consistency_policy not in {"reject", "warn"}:
        raise RecipeError("shape_consistency_policy must be one of: reject, warn")
    if recipe.critical_bound_hit_policy not in {"reject", "warn"}:
        raise RecipeError("critical_bound_hit_policy must be one of: reject, warn")
    if recipe.kittel_min_points < 3:
        raise RecipeError("kittel_min_points must be at least 3")
    if recipe.linewidth_min_points < 2:
        raise RecipeError("linewidth_min_points must be at least 2")
