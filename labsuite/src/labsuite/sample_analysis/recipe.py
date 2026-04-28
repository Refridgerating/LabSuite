"""Recipe loading for sample-level derived analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from labsuite.core.exceptions import RecipeError

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


@dataclass(slots=True)
class SampleAnalysisThresholds:
    min_frequency_span_GHz: float = 1.0
    meff_mismatch_relative: float = 0.10
    field_offset_detected_mT: float = 1.0
    linewidth_nonlinear_r2_min: float = 0.95
    g_float_relative_stderr_max: float = 0.20
    min_kittel_points: int = 3
    min_linewidth_points: int = 2


@dataclass(slots=True)
class SampleAnalysisRecipe:
    name: str = "sample-analysis-default"
    allow_partial_results: bool = True
    ms_policy: str = "from_vsm_moment_and_volume"
    vmag_policy: str = "direct_or_area_times_thickness"
    fmr_g_mode: str = "inherit_registry"
    positive_negative_pairing_enabled: bool = True
    anisotropy_enabled: bool = True
    damping_enabled: bool = True
    esr_enabled: bool = True
    write_figures: bool = True
    thresholds: SampleAnalysisThresholds = field(default_factory=SampleAnalysisThresholds)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_sample_analysis_recipe(path: Path) -> SampleAnalysisRecipe:
    if yaml is None:
        raise RecipeError("PyYAML is required to load sample analysis recipes.")
    resolved = path.resolve()
    if not resolved.exists():
        raise RecipeError(f"Recipe file does not exist: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RecipeError(f"Recipe must deserialize to a mapping: {resolved}")
    sample_payload = (
        payload.get("sample_analysis") if isinstance(payload.get("sample_analysis"), dict) else {}
    )
    magnetization_payload = (
        payload.get("magnetization") if isinstance(payload.get("magnetization"), dict) else {}
    )
    thresholds_payload = (
        payload.get("quality")
        or payload.get("quality_thresholds")
        or payload.get("thresholds")
        or {}
    )
    if not isinstance(thresholds_payload, dict):
        thresholds_payload = {}
    fmr_payload = payload.get("fmr") if isinstance(payload.get("fmr"), dict) else {}
    pairing_payload = (
        payload.get("positive_negative_pairing")
        if isinstance(payload.get("positive_negative_pairing"), dict)
        else {}
    )
    output_payload = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
    recipe = SampleAnalysisRecipe(
        name=str(payload.get("name", "sample-analysis-default")),
        allow_partial_results=bool(
            sample_payload.get("allow_partial_results", payload.get("allow_partial_results", True))
        ),
        ms_policy=str(
            magnetization_payload.get("ms_policy", payload.get("ms_policy", "prefer_vsm"))
        ),
        vmag_policy=str(
            magnetization_payload.get(
                "vmag_policy", payload.get("vmag_policy", "direct_or_area_times_thickness")
            )
        ),
        fmr_g_mode=str(
            fmr_payload.get("g_mode", fmr_payload.get("preferred_g_source", "inherit_registry"))
        ),
        positive_negative_pairing_enabled=bool(pairing_payload.get("enabled", True)),
        anisotropy_enabled=bool(_enabled_mapping(payload, "anisotropy", True)),
        damping_enabled=bool(_enabled_mapping(payload, "damping", True)),
        esr_enabled=bool(_enabled_mapping(payload, "esr", True)),
        write_figures=bool(
            output_payload.get("write_figures", output_payload.get("figures", True))
        ),
        thresholds=SampleAnalysisThresholds(
            min_frequency_span_GHz=float(
                thresholds_payload.get(
                    "min_frequency_span_ghz", thresholds_payload.get("min_frequency_span_GHz", 1.0)
                )
            ),
            meff_mismatch_relative=float(thresholds_payload.get("meff_mismatch_relative", 0.10)),
            field_offset_detected_mT=float(thresholds_payload.get("field_offset_detected_mT", 1.0)),
            linewidth_nonlinear_r2_min=float(
                thresholds_payload.get("linewidth_nonlinear_r2_min", 0.95)
            ),
            g_float_relative_stderr_max=float(
                thresholds_payload.get("g_float_relative_stderr_max", 0.20)
            ),
            min_kittel_points=int(
                thresholds_payload.get(
                    "min_frequency_points", thresholds_payload.get("min_kittel_points", 3)
                )
            ),
            min_linewidth_points=int(thresholds_payload.get("min_linewidth_points", 2)),
        ),
    )
    _validate(recipe)
    return recipe


def _enabled_mapping(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key)
    if isinstance(value, dict):
        return bool(value.get("enabled", default))
    if value is None:
        return default
    return bool(value)


def _validate(recipe: SampleAnalysisRecipe) -> None:
    if recipe.fmr_g_mode == "processed_then_registry":
        recipe.fmr_g_mode = "inherit_registry"
    if recipe.fmr_g_mode not in {"inherit_registry", "float", "fixed", "bounded"}:
        raise RecipeError(
            "fmr.g_mode/preferred_g_source must be one of: inherit_registry, "
            "processed_then_registry, float, fixed, bounded"
        )
    if recipe.thresholds.min_kittel_points < 3:
        raise RecipeError("quality_thresholds.min_kittel_points must be at least 3")
    if recipe.thresholds.min_linewidth_points < 2:
        raise RecipeError("quality_thresholds.min_linewidth_points must be at least 2")
    if recipe.thresholds.min_frequency_span_GHz < 0.0:
        raise RecipeError("quality_thresholds.min_frequency_span_GHz must be non-negative")
