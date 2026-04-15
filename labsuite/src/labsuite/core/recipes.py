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
