"""Recipe loading utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    batch_qc_nrmse_max: float = 0.12
    batch_qc_edge_guard_min_mT: float = 2.0
    batch_qc_edge_guard_gamma_multiplier: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VsmPreprocessingRecipe:
    """Recipe for the first VSM loop-processing path."""

    name: str = "vsm-default"
    vsm_quality_model: str = "simple"
    vsm_min_weight: float = 0.45
    vsm_accept_downweighted: bool = True
    vsm_hcut_fractions: list[float] = field(
        default_factory=lambda: [0.15, 0.20, 0.25, 0.30]
    )
    vsm_quality_slope_downweight_ratio: float = 0.20
    vsm_quality_slope_extreme_ratio: float = 2.0
    vsm_quality_symmetry_downweight_error: float = 0.20
    vsm_quality_symmetry_catastrophic_error: float = 0.80
    vsm_quality_cutoff_cv_downweight: float = 0.15
    vsm_quality_cutoff_cv_extreme: float = 0.60
    vsm_quality_tail_rmse_downweight_ratio: float = 0.08
    vsm_quality_tail_rmse_extreme_ratio: float = 0.50
    vsm_quality_near_zero_ms_emu: float = 1e-18
    background_tail_fraction: float = 0.12
    background_min_points_per_side: int = 24
    background_tail_fit_min_r_squared: float = 0.6
    background_tail_fit_catastrophic_r_squared: float = 0.2
    background_tail_fit_override_min_flatness_gain_score: float = 0.6
    background_tail_fit_override_min_flatness_gain_per_tail: float = 0.20
    background_tail_fit_override_min_gain_balance_score: float = 0.65
    background_tail_fit_override_min_switching_integrity_score: float = 0.85
    background_min_meaningful_slope_emu_per_mT: float = 5e-8
    background_tail_flatness_ratio_tolerance: float = 0.08
    background_slope_disagreement_ratio_tolerance: float = 0.35
    background_max_flatness_worsening: float = 0.02
    background_max_tail_flatness_regression: float = 0.05
    background_max_branch_asymmetry_worsening: float = 0.12
    background_max_loop_closure_worsening: float = 0.08
    background_max_zero_crossing_increase: int = 0
    background_max_switching_width_relative_change: float = 0.25
    background_max_coercive_ambiguity_worsening: int = 0
    background_min_flatness_gain_score: float = 0.10
    background_min_score_improvement: float = 0.02
    background_score_weight_flatness: float = 0.45
    background_score_weight_saturation_consistency: float = 0.30
    background_score_weight_closure_quality: float = 0.20
    background_score_weight_branch_asymmetry_penalty: float = 0.20
    background_score_weight_flatness_gain: float = 0.55
    background_score_weight_tail_slope_symmetry: float = 0.20
    background_score_weight_saturation_magnitude_symmetry: float = 0.15
    background_score_weight_switching_integrity: float = 0.10
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
class FmrFieldPolarityCorrectionRecipe:
    """Recipe controls for optional FMR field-polarity averaging."""

    enabled: bool = False
    method: str = "gonzalez_fuentes_average"
    polarity_column: str | None = None
    require_pair: bool = True
    group_by: list[str] = field(
        default_factory=lambda: [
            "sample_id",
            "replicate_id",
            "frequency",
            "geometry",
            "mode_id",
        ]
    )
    positive_labels: list[str] = field(default_factory=lambda: ["positive", "pos", "plus", "+H"])
    negative_labels: list[str] = field(default_factory=lambda: ["negative", "neg", "minus", "-H"])
    max_pair_frequency_tolerance_ghz: float = 0.001
    max_pair_hres_split_mT: float | None = None
    on_unpaired: str = "warn_and_keep_raw"
    fit_field: str = "Hres_avg"
    run_comparison_fits: bool = True
    plot_diagnostics: bool = False


@dataclass(slots=True)
class FmrGonzalezFuentesRequirements:
    """Measurement requirements for Gonzalez-Fuentes field-polarity averaging."""

    requires_positive_and_negative_field_sweeps: bool = True
    cannot_be_applied_to_single_polarity_data: bool = True


@dataclass(slots=True)
class FmrMeasurementRequirementsRecipe:
    """FMR measurement requirements documented in the recipe."""

    gonzalez_fuentes_average: FmrGonzalezFuentesRequirements = field(
        default_factory=FmrGonzalezFuentesRequirements
    )


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
    n_peaks: str = "auto"
    trace_fit_model: str = "mixed_derivative_lorentzian"
    physics_model: str = "ip_simple"
    background_model: str = "linear"
    multi_peak_selection: str = "bic"
    min_peak_separation_mT: float = 4.0
    min_linewidth_mT: float | None = None
    max_linewidth_mT: float | None = None
    enable_branch_tracking: bool = False
    plot_component_fits: bool = False
    fit_g: bool = False
    fit_g_diagnostic: bool = False
    compare_locked_vs_floating_g: bool = False
    fit_Hk: bool = False
    branch_locked_g: dict[str, float] = field(default_factory=dict)
    branch_locked_gamma_over_2pi_GHz_per_T: dict[str, float] = field(default_factory=dict)
    branch_locked_Hk_mT: dict[str, float] = field(default_factory=dict)
    floating_g_warning_percent_threshold: float = 2.0
    field_polarity: str = "unknown"
    geometry: str = "unknown"
    replicate_id: str | None = None
    measurement_id: str | None = None
    frequency_match_tolerance_GHz: float = 0.001
    allow_low_confidence_pos_neg_matching: bool = False
    prefer_pos_neg_matched_average: bool = False
    joint_pos_neg_matched_fit: bool = False
    fit_field_offset: bool = False
    linewidth_min_high_confidence_points: int = 4
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
    field_polarity_correction: FmrFieldPolarityCorrectionRecipe = field(
        default_factory=FmrFieldPolarityCorrectionRecipe
    )
    measurement_requirements: FmrMeasurementRequirementsRecipe = field(
        default_factory=FmrMeasurementRequirementsRecipe
    )

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
        integration_window_gamma_multiplier=float(
            payload.get("integration_window_gamma_multiplier", 7.0)
        ),
        integration_window_min_half_width_mT=float(
            payload.get("integration_window_min_half_width_mT", 2.0)
        ),
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
        fit_local_disagreement_ratio_threshold=float(
            payload.get("fit_local_disagreement_ratio_threshold", 0.35)
        ),
        batch_qc_nrmse_max=float(payload.get("batch_qc_nrmse_max", 0.12)),
        batch_qc_edge_guard_min_mT=float(payload.get("batch_qc_edge_guard_min_mT", 2.0)),
        batch_qc_edge_guard_gamma_multiplier=float(
            payload.get("batch_qc_edge_guard_gamma_multiplier", 2.0)
        ),
    )
    _validate_recipe(recipe)
    return recipe


def load_vsm_recipe(path: Path) -> VsmPreprocessingRecipe:
    """Load the VSM preprocessing recipe from YAML or a simple fallback mapping."""

    payload = _load_mapping(path)
    recipe = VsmPreprocessingRecipe(
        name=str(payload.get("name", "vsm-default")),
        vsm_quality_model=str(payload.get("vsm_quality_model", "simple")),
        vsm_min_weight=float(payload.get("vsm_min_weight", 0.45)),
        vsm_accept_downweighted=bool(payload.get("vsm_accept_downweighted", True)),
        vsm_hcut_fractions=_float_list(
            payload.get("vsm_hcut_fractions", [0.15, 0.20, 0.25, 0.30])
        ),
        vsm_quality_slope_downweight_ratio=float(
            payload.get("vsm_quality_slope_downweight_ratio", 0.20)
        ),
        vsm_quality_slope_extreme_ratio=float(
            payload.get("vsm_quality_slope_extreme_ratio", 2.0)
        ),
        vsm_quality_symmetry_downweight_error=float(
            payload.get("vsm_quality_symmetry_downweight_error", 0.20)
        ),
        vsm_quality_symmetry_catastrophic_error=float(
            payload.get("vsm_quality_symmetry_catastrophic_error", 0.80)
        ),
        vsm_quality_cutoff_cv_downweight=float(
            payload.get("vsm_quality_cutoff_cv_downweight", 0.15)
        ),
        vsm_quality_cutoff_cv_extreme=float(
            payload.get("vsm_quality_cutoff_cv_extreme", 0.60)
        ),
        vsm_quality_tail_rmse_downweight_ratio=float(
            payload.get("vsm_quality_tail_rmse_downweight_ratio", 0.08)
        ),
        vsm_quality_tail_rmse_extreme_ratio=float(
            payload.get("vsm_quality_tail_rmse_extreme_ratio", 0.50)
        ),
        vsm_quality_near_zero_ms_emu=float(payload.get("vsm_quality_near_zero_ms_emu", 1e-18)),
        background_tail_fraction=float(payload.get("background_tail_fraction", 0.12)),
        background_min_points_per_side=int(payload.get("background_min_points_per_side", 24)),
        background_tail_fit_min_r_squared=float(
            payload.get("background_tail_fit_min_r_squared", 0.6)
        ),
        background_tail_fit_catastrophic_r_squared=float(
            payload.get("background_tail_fit_catastrophic_r_squared", 0.2)
        ),
        background_tail_fit_override_min_flatness_gain_score=float(
            payload.get("background_tail_fit_override_min_flatness_gain_score", 0.6)
        ),
        background_tail_fit_override_min_flatness_gain_per_tail=float(
            payload.get("background_tail_fit_override_min_flatness_gain_per_tail", 0.20)
        ),
        background_tail_fit_override_min_gain_balance_score=float(
            payload.get("background_tail_fit_override_min_gain_balance_score", 0.65)
        ),
        background_tail_fit_override_min_switching_integrity_score=float(
            payload.get("background_tail_fit_override_min_switching_integrity_score", 0.85)
        ),
        background_min_meaningful_slope_emu_per_mT=float(
            payload.get("background_min_meaningful_slope_emu_per_mT", 5e-8)
        ),
        background_tail_flatness_ratio_tolerance=float(
            payload.get("background_tail_flatness_ratio_tolerance", 0.08)
        ),
        background_slope_disagreement_ratio_tolerance=float(
            payload.get("background_slope_disagreement_ratio_tolerance", 0.35)
        ),
        background_max_flatness_worsening=float(
            payload.get("background_max_flatness_worsening", 0.02)
        ),
        background_max_tail_flatness_regression=float(
            payload.get("background_max_tail_flatness_regression", 0.05)
        ),
        background_max_branch_asymmetry_worsening=float(
            payload.get("background_max_branch_asymmetry_worsening", 0.12)
        ),
        background_max_loop_closure_worsening=float(
            payload.get("background_max_loop_closure_worsening", 0.08)
        ),
        background_max_zero_crossing_increase=int(
            payload.get("background_max_zero_crossing_increase", 0)
        ),
        background_max_switching_width_relative_change=float(
            payload.get("background_max_switching_width_relative_change", 0.25)
        ),
        background_max_coercive_ambiguity_worsening=int(
            payload.get("background_max_coercive_ambiguity_worsening", 0)
        ),
        background_min_flatness_gain_score=float(
            payload.get("background_min_flatness_gain_score", 0.10)
        ),
        background_min_score_improvement=float(
            payload.get("background_min_score_improvement", 0.02)
        ),
        background_score_weight_flatness=float(
            payload.get("background_score_weight_flatness", 0.45)
        ),
        background_score_weight_saturation_consistency=float(
            payload.get("background_score_weight_saturation_consistency", 0.30)
        ),
        background_score_weight_closure_quality=float(
            payload.get("background_score_weight_closure_quality", 0.20)
        ),
        background_score_weight_branch_asymmetry_penalty=float(
            payload.get("background_score_weight_branch_asymmetry_penalty", 0.20)
        ),
        background_score_weight_flatness_gain=float(
            payload.get("background_score_weight_flatness_gain", 0.55)
        ),
        background_score_weight_tail_slope_symmetry=float(
            payload.get("background_score_weight_tail_slope_symmetry", 0.20)
        ),
        background_score_weight_saturation_magnitude_symmetry=float(
            payload.get("background_score_weight_saturation_magnitude_symmetry", 0.15)
        ),
        background_score_weight_switching_integrity=float(
            payload.get("background_score_weight_switching_integrity", 0.10)
        ),
        center_loop=bool(payload.get("center_loop", False)),
        smoothing_enabled=bool(payload.get("smoothing_enabled", False)),
        smoothing_window=int(payload.get("smoothing_window", 0)),
        smoothing_polyorder=int(payload.get("smoothing_polyorder", 0)),
        uncertainty_scale=float(payload.get("uncertainty_scale", 1.0)),
        uncertainty_zero_field_window_width_mT=float(
            payload.get("uncertainty_zero_field_window_width_mT", 5.0)
        ),
        uncertainty_zero_field_min_points=int(payload.get("uncertainty_zero_field_min_points", 5)),
        uncertainty_switching_half_width_mT=float(
            payload.get("uncertainty_switching_half_width_mT", 3.0)
        ),
        uncertainty_switching_min_points=int(payload.get("uncertainty_switching_min_points", 4)),
        uncertainty_loop_area_smoothing_window=int(
            payload.get("uncertainty_loop_area_smoothing_window", 9)
        ),
        uncertainty_loop_area_smoothing_polyorder=int(
            payload.get("uncertainty_loop_area_smoothing_polyorder", 2)
        ),
        uncertainty_min_switching_slope_emu_per_mT=float(
            payload.get("uncertainty_min_switching_slope_emu_per_mT", 1e-7)
        ),
    )
    _validate_vsm_recipe(recipe)
    return recipe


def load_fmr_recipe(path: Path) -> FmrRecipe:
    """Load the FMR analysis recipe from YAML or a simple fallback mapping."""

    payload = _load_mapping(path)
    fmr_payload = payload.get("fmr") if isinstance(payload.get("fmr"), dict) else {}
    kittel_payload = (
        fmr_payload.get("kittel") if isinstance(fmr_payload.get("kittel"), dict) else {}
    )
    field_polarity_payload = _mapping_value(
        payload.get("field_polarity_correction"),
        kittel_payload.get("field_polarity_correction"),
        {},
    )
    measurement_requirements_payload = _mapping_value(
        payload.get("measurement_requirements"),
        fmr_payload.get("measurement_requirements"),
        {},
    )
    recipe = FmrRecipe(
        name=str(payload.get("name", "fmr-default")),
        signal_channel=str(payload.get("signal_channel", "fit_source")),
        baseline_edge_points=int(payload.get("baseline_edge_points", 4)),
        baseline_enabled=bool(payload.get("baseline_enabled", True)),
        smoothing_enabled=bool(payload.get("smoothing_enabled", False)),
        smoothing_window=int(payload.get("smoothing_window", 7)),
        smoothing_polyorder=int(payload.get("smoothing_polyorder", 2)),
        fit_mode=str(payload.get("fit_mode", "auto")),
        n_peaks=str(payload.get("n_peaks", payload.get("n-peaks", "auto"))),
        trace_fit_model=str(payload.get("trace_fit_model", "mixed_derivative_lorentzian")),
        physics_model=str(payload.get("physics_model", "ip_simple")),
        background_model=str(payload.get("background_model", "linear")),
        multi_peak_selection=str(payload.get("multi_peak_selection", "bic")),
        min_peak_separation_mT=float(
            payload.get("min_peak_separation_mT", payload.get("peak_min_pair_width_mT", 4.0))
        ),
        min_linewidth_mT=_optional_float_value(payload.get("min_linewidth_mT")),
        max_linewidth_mT=_optional_float_value(payload.get("max_linewidth_mT")),
        enable_branch_tracking=bool(payload.get("enable_branch_tracking", False)),
        plot_component_fits=bool(payload.get("plot_component_fits", False)),
        fit_g=bool(payload.get("fit_g", False)),
        fit_g_diagnostic=bool(payload.get("fit_g_diagnostic", False)),
        compare_locked_vs_floating_g=bool(payload.get("compare_locked_vs_floating_g", False)),
        fit_Hk=bool(payload.get("fit_Hk", False)),
        branch_locked_g=_float_dict(payload.get("branch_locked_g")),
        branch_locked_gamma_over_2pi_GHz_per_T=_float_dict(
            payload.get("branch_locked_gamma_over_2pi_GHz_per_T")
        ),
        branch_locked_Hk_mT=_float_dict(payload.get("branch_locked_Hk_mT")),
        floating_g_warning_percent_threshold=float(
            payload.get("floating_g_warning_percent_threshold", 2.0)
        ),
        field_polarity=str(payload.get("field_polarity", "unknown")),
        geometry=str(payload.get("geometry", "unknown")),
        replicate_id=_optional_recipe_str(payload.get("replicate_id")),
        measurement_id=_optional_recipe_str(payload.get("measurement_id")),
        frequency_match_tolerance_GHz=float(
            payload.get("frequency_match_tolerance_GHz", 0.001)
        ),
        allow_low_confidence_pos_neg_matching=bool(
            payload.get("allow_low_confidence_pos_neg_matching", False)
        ),
        prefer_pos_neg_matched_average=bool(
            payload.get("prefer_pos_neg_matched_average", False)
        ),
        joint_pos_neg_matched_fit=bool(payload.get("joint_pos_neg_matched_fit", False)),
        fit_field_offset=bool(payload.get("fit_field_offset", False)),
        linewidth_min_high_confidence_points=int(
            payload.get("linewidth_min_high_confidence_points", 4)
        ),
        peak_min_prominence_ratio=float(payload.get("peak_min_prominence_ratio", 0.2)),
        peak_min_distance_mT=float(payload.get("peak_min_distance_mT", 8.0)),
        peak_min_pair_width_mT=float(payload.get("peak_min_pair_width_mT", 4.0)),
        candidate_window_padding_width_multiplier=float(
            payload.get("candidate_window_padding_width_multiplier", 1.5)
        ),
        double_fit_min_improvement_ratio=float(
            payload.get("double_fit_min_improvement_ratio", 0.15)
        ),
        max_resonance_count=int(payload.get("max_resonance_count", 2)),
        field_guard_fraction=float(payload.get("field_guard_fraction", 0.05)),
        linewidth_max_sweep_fraction=float(payload.get("linewidth_max_sweep_fraction", 0.35)),
        residual_rmse_max_signal_fraction=float(
            payload.get("residual_rmse_max_signal_fraction", 0.12)
        ),
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
        field_polarity_correction=_load_fmr_field_polarity_recipe(field_polarity_payload),
        measurement_requirements=_load_fmr_measurement_requirements(
            measurement_requirements_payload
        ),
    )
    _validate_fmr_recipe(recipe)
    return recipe


def _load_fmr_field_polarity_recipe(payload: Any) -> FmrFieldPolarityCorrectionRecipe:
    data = payload if isinstance(payload, dict) else {}
    max_split = data.get("max_pair_hres_split_mT")
    return FmrFieldPolarityCorrectionRecipe(
        enabled=bool(data.get("enabled", False)),
        method=str(data.get("method", "gonzalez_fuentes_average")),
        polarity_column=_optional_recipe_str(data.get("polarity_column")),
        require_pair=bool(data.get("require_pair", True)),
        group_by=_string_list(
            data.get(
                "group_by",
                ["sample_id", "replicate_id", "frequency", "geometry", "mode_id"],
            )
        ),
        positive_labels=_string_list(
            data.get("positive_labels", ["positive", "pos", "plus", "+H"])
        ),
        negative_labels=_string_list(
            data.get("negative_labels", ["negative", "neg", "minus", "-H"])
        ),
        max_pair_frequency_tolerance_ghz=float(data.get("max_pair_frequency_tolerance_ghz", 0.001)),
        max_pair_hres_split_mT=None if max_split in {None, ""} else float(max_split),
        on_unpaired=str(data.get("on_unpaired", "warn_and_keep_raw")),
        fit_field=str(data.get("fit_field", "Hres_avg")),
        run_comparison_fits=bool(data.get("run_comparison_fits", True)),
        plot_diagnostics=bool(data.get("plot_diagnostics", False)),
    )


def _load_fmr_measurement_requirements(payload: Any) -> FmrMeasurementRequirementsRecipe:
    data = payload if isinstance(payload, dict) else {}
    gf_payload = data.get("gonzalez_fuentes_average")
    gf_data = gf_payload if isinstance(gf_payload, dict) else {}
    return FmrMeasurementRequirementsRecipe(
        gonzalez_fuentes_average=FmrGonzalezFuentesRequirements(
            requires_positive_and_negative_field_sweeps=bool(
                gf_data.get("requires_positive_and_negative_field_sweeps", True)
            ),
            cannot_be_applied_to_single_polarity_data=bool(
                gf_data.get("cannot_be_applied_to_single_polarity_data", True)
            ),
        )
    )


def _mapping_value(*values: Any) -> Any:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def _float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [float(part) for part in value]
    return [float(value)]


def _optional_float_value(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _float_dict(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): float(item) for key, item in value.items()}
    if isinstance(value, str):
        output: dict[str, float] = {}
        for item in value.split(","):
            if not item.strip():
                continue
            key, raw_value = item.split(":", maxsplit=1)
            output[key.strip()] = float(raw_value)
        return output
    raise RecipeError(f"Expected branch value mapping, got: {value!r}")


def _optional_recipe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
        raise RecipeError(
            "integration_detected_window_padding_width_multiplier must be zero or positive"
        )
    if not 0.0 < recipe.fit_max_gamma_as_sweep_fraction <= 1.0:
        raise RecipeError("fit_max_gamma_as_sweep_fraction must be between 0 and 1")
    if recipe.fit_local_disagreement_ratio_threshold <= 0.0:
        raise RecipeError("fit_local_disagreement_ratio_threshold must be positive")
    if recipe.batch_qc_nrmse_max <= 0.0:
        raise RecipeError("batch_qc_nrmse_max must be positive")
    if recipe.batch_qc_edge_guard_min_mT <= 0.0:
        raise RecipeError("batch_qc_edge_guard_min_mT must be positive")
    if recipe.batch_qc_edge_guard_gamma_multiplier <= 0.0:
        raise RecipeError("batch_qc_edge_guard_gamma_multiplier must be positive")


def _validate_vsm_recipe(recipe: VsmPreprocessingRecipe) -> None:
    if recipe.vsm_quality_model not in {"simple", "legacy"}:
        raise RecipeError("vsm_quality_model must be one of: simple, legacy")
    if not 0.0 <= recipe.vsm_min_weight <= 1.0:
        raise RecipeError("vsm_min_weight must be between 0 and 1")
    if not recipe.vsm_hcut_fractions:
        raise RecipeError("vsm_hcut_fractions must contain at least one value")
    if any(not 0.0 < fraction < 0.5 for fraction in recipe.vsm_hcut_fractions):
        raise RecipeError("vsm_hcut_fractions values must be between 0 and 0.5")
    if recipe.vsm_quality_slope_downweight_ratio <= 0.0:
        raise RecipeError("vsm_quality_slope_downweight_ratio must be positive")
    if (
        recipe.vsm_quality_slope_extreme_ratio
        <= recipe.vsm_quality_slope_downweight_ratio
    ):
        raise RecipeError(
            "vsm_quality_slope_extreme_ratio must exceed "
            "vsm_quality_slope_downweight_ratio"
        )
    if recipe.vsm_quality_symmetry_downweight_error <= 0.0:
        raise RecipeError("vsm_quality_symmetry_downweight_error must be positive")
    if (
        recipe.vsm_quality_symmetry_catastrophic_error
        <= recipe.vsm_quality_symmetry_downweight_error
    ):
        raise RecipeError(
            "vsm_quality_symmetry_catastrophic_error must exceed "
            "vsm_quality_symmetry_downweight_error"
        )
    if recipe.vsm_quality_cutoff_cv_downweight <= 0.0:
        raise RecipeError("vsm_quality_cutoff_cv_downweight must be positive")
    if recipe.vsm_quality_cutoff_cv_extreme <= recipe.vsm_quality_cutoff_cv_downweight:
        raise RecipeError(
            "vsm_quality_cutoff_cv_extreme must exceed vsm_quality_cutoff_cv_downweight"
        )
    if recipe.vsm_quality_tail_rmse_downweight_ratio <= 0.0:
        raise RecipeError("vsm_quality_tail_rmse_downweight_ratio must be positive")
    if (
        recipe.vsm_quality_tail_rmse_extreme_ratio
        <= recipe.vsm_quality_tail_rmse_downweight_ratio
    ):
        raise RecipeError(
            "vsm_quality_tail_rmse_extreme_ratio must exceed "
            "vsm_quality_tail_rmse_downweight_ratio"
        )
    if recipe.vsm_quality_near_zero_ms_emu <= 0.0:
        raise RecipeError("vsm_quality_near_zero_ms_emu must be positive")
    if not 0.0 < recipe.background_tail_fraction < 0.5:
        raise RecipeError("background_tail_fraction must be between 0 and 0.5")
    if recipe.background_min_points_per_side < 2:
        raise RecipeError("background_min_points_per_side must be at least 2")
    if not 0.0 <= recipe.background_tail_fit_min_r_squared <= 1.0:
        raise RecipeError("background_tail_fit_min_r_squared must be between 0 and 1")
    if not 0.0 <= recipe.background_tail_fit_catastrophic_r_squared <= 1.0:
        raise RecipeError("background_tail_fit_catastrophic_r_squared must be between 0 and 1")
    if recipe.background_tail_fit_catastrophic_r_squared > recipe.background_tail_fit_min_r_squared:
        raise RecipeError(
            "background_tail_fit_catastrophic_r_squared must be less than or equal to "
            "background_tail_fit_min_r_squared"
        )
    if not 0.0 <= recipe.background_tail_fit_override_min_flatness_gain_score <= 1.0:
        raise RecipeError(
            "background_tail_fit_override_min_flatness_gain_score must be between 0 and 1"
        )
    if not 0.0 <= recipe.background_tail_fit_override_min_flatness_gain_per_tail <= 1.0:
        raise RecipeError(
            "background_tail_fit_override_min_flatness_gain_per_tail must be between 0 and 1"
        )
    if not 0.0 <= recipe.background_tail_fit_override_min_gain_balance_score <= 1.0:
        raise RecipeError(
            "background_tail_fit_override_min_gain_balance_score must be between 0 and 1"
        )
    if not 0.0 <= recipe.background_tail_fit_override_min_switching_integrity_score <= 1.0:
        raise RecipeError(
            "background_tail_fit_override_min_switching_integrity_score must be between 0 and 1"
        )
    if recipe.background_min_meaningful_slope_emu_per_mT <= 0.0:
        raise RecipeError("background_min_meaningful_slope_emu_per_mT must be positive")
    if recipe.background_tail_flatness_ratio_tolerance <= 0.0:
        raise RecipeError("background_tail_flatness_ratio_tolerance must be positive")
    if recipe.background_slope_disagreement_ratio_tolerance <= 0.0:
        raise RecipeError("background_slope_disagreement_ratio_tolerance must be positive")
    if recipe.background_max_flatness_worsening < 0.0:
        raise RecipeError("background_max_flatness_worsening must be zero or positive")
    if recipe.background_max_tail_flatness_regression < 0.0:
        raise RecipeError("background_max_tail_flatness_regression must be zero or positive")
    if recipe.background_max_branch_asymmetry_worsening < 0.0:
        raise RecipeError("background_max_branch_asymmetry_worsening must be zero or positive")
    if recipe.background_max_loop_closure_worsening < 0.0:
        raise RecipeError("background_max_loop_closure_worsening must be zero or positive")
    if recipe.background_max_zero_crossing_increase < 0:
        raise RecipeError("background_max_zero_crossing_increase must be zero or positive")
    if recipe.background_max_switching_width_relative_change < 0.0:
        raise RecipeError("background_max_switching_width_relative_change must be zero or positive")
    if recipe.background_max_coercive_ambiguity_worsening < 0:
        raise RecipeError("background_max_coercive_ambiguity_worsening must be zero or positive")
    if recipe.background_min_flatness_gain_score < 0.0:
        raise RecipeError("background_min_flatness_gain_score must be zero or positive")
    if recipe.background_min_score_improvement < 0.0:
        raise RecipeError("background_min_score_improvement must be zero or positive")
    if recipe.background_score_weight_flatness < 0.0:
        raise RecipeError("background_score_weight_flatness must be zero or positive")
    if recipe.background_score_weight_saturation_consistency < 0.0:
        raise RecipeError("background_score_weight_saturation_consistency must be zero or positive")
    if recipe.background_score_weight_closure_quality < 0.0:
        raise RecipeError("background_score_weight_closure_quality must be zero or positive")
    if recipe.background_score_weight_branch_asymmetry_penalty < 0.0:
        raise RecipeError(
            "background_score_weight_branch_asymmetry_penalty must be zero or positive"
        )
    if recipe.background_score_weight_flatness_gain < 0.0:
        raise RecipeError("background_score_weight_flatness_gain must be zero or positive")
    if recipe.background_score_weight_tail_slope_symmetry < 0.0:
        raise RecipeError("background_score_weight_tail_slope_symmetry must be zero or positive")
    if recipe.background_score_weight_saturation_magnitude_symmetry < 0.0:
        raise RecipeError(
            "background_score_weight_saturation_magnitude_symmetry must be zero or positive"
        )
    if recipe.background_score_weight_switching_integrity < 0.0:
        raise RecipeError("background_score_weight_switching_integrity must be zero or positive")
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
    if (
        recipe.uncertainty_loop_area_smoothing_polyorder
        >= recipe.uncertainty_loop_area_smoothing_window
    ):
        raise RecipeError(
            "uncertainty_loop_area_smoothing_polyorder must be smaller than "
            "uncertainty_loop_area_smoothing_window"
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
    if recipe.n_peaks not in {"auto", "1", "2", "3"}:
        raise RecipeError("n_peaks must be one of: auto, 1, 2, 3")
    if recipe.trace_fit_model != "mixed_derivative_lorentzian":
        raise RecipeError("trace_fit_model must be 'mixed_derivative_lorentzian'")
    if recipe.physics_model not in {
        "ip_simple",
        "ip_with_Hk",
        "oop_simple",
        "ip_field_swept_kittel",
    }:
        raise RecipeError(
            "physics_model must be one of: ip_simple, ip_with_Hk, oop_simple"
        )
    if recipe.background_model not in {"linear", "quadratic"}:
        raise RecipeError("background_model must be one of: linear, quadratic")
    if recipe.multi_peak_selection not in {"aic", "bic", "residual"}:
        raise RecipeError("multi_peak_selection must be one of: aic, bic, residual")
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
    if recipe.max_resonance_count not in {1, 2, 3}:
        raise RecipeError("max_resonance_count must be 1, 2, or 3")
    if recipe.min_peak_separation_mT < 0.0:
        raise RecipeError("min_peak_separation_mT must be zero or positive")
    if recipe.min_linewidth_mT is not None and recipe.min_linewidth_mT <= 0.0:
        raise RecipeError("min_linewidth_mT must be positive")
    if recipe.max_linewidth_mT is not None and recipe.max_linewidth_mT <= 0.0:
        raise RecipeError("max_linewidth_mT must be positive")
    if (
        recipe.min_linewidth_mT is not None
        and recipe.max_linewidth_mT is not None
        and recipe.min_linewidth_mT >= recipe.max_linewidth_mT
    ):
        raise RecipeError("min_linewidth_mT must be smaller than max_linewidth_mT")
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
    if recipe.linewidth_min_high_confidence_points < 2:
        raise RecipeError("linewidth_min_high_confidence_points must be at least 2")
    if str(recipe.field_polarity).lower() not in {"positive", "negative", "unknown"}:
        raise RecipeError("field_polarity must be positive, negative, or unknown")
    if str(recipe.geometry).lower() not in {"ip", "oop", "angular", "unknown"}:
        raise RecipeError("geometry must be IP, OOP, angular, or unknown")
    if recipe.frequency_match_tolerance_GHz < 0.0:
        raise RecipeError("frequency_match_tolerance_GHz must be zero or positive")
    correction = recipe.field_polarity_correction
    if correction.method != "gonzalez_fuentes_average":
        raise RecipeError("field_polarity_correction.method must be 'gonzalez_fuentes_average'")
    if correction.on_unpaired not in {"warn_and_keep_raw", "drop", "fail"}:
        raise RecipeError(
            "field_polarity_correction.on_unpaired must be one of: warn_and_keep_raw, drop, fail"
        )
    if correction.fit_field not in {"Hres", "Hres_avg", "Hres_pos", "Hres_neg"}:
        raise RecipeError(
            "field_polarity_correction.fit_field must be one of: Hres, Hres_avg, Hres_pos, Hres_neg"
        )
    if correction.max_pair_frequency_tolerance_ghz < 0.0:
        raise RecipeError(
            "field_polarity_correction.max_pair_frequency_tolerance_ghz must be zero or positive"
        )
    if correction.max_pair_hres_split_mT is not None and correction.max_pair_hres_split_mT < 0.0:
        raise RecipeError(
            "field_polarity_correction.max_pair_hres_split_mT must be zero or positive"
        )
    if not correction.positive_labels:
        raise RecipeError("field_polarity_correction.positive_labels must not be empty")
    if not correction.negative_labels:
        raise RecipeError("field_polarity_correction.negative_labels must not be empty")
