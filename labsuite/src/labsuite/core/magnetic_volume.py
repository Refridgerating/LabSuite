"""Shared magnetic material volume estimation utilities.

This module is intentionally modality-agnostic.  It contains only geometry,
layer-stack, and unit-normalization logic so sample registration, CLI tools,
GUI forms, VSM, FMR, ESR, and sample-level analysis can all share the same
volume calculation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class MagneticVolumeError(ValueError):
    """Raised when magnetic volume inputs are invalid or incomplete."""


@dataclass(frozen=True, slots=True)
class MagneticLayer:
    """One physical layer in a sample stack."""

    material: str
    thickness: float
    thickness_unit: str
    magnetic: bool
    role: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class MagneticLayerBlock:
    """Repeated group of layers in a multilayer stack."""

    repeat_count: int
    layers: list[MagneticLayer]


@dataclass(frozen=True, slots=True)
class MagneticVolumeEstimate:
    """SI-normalized magnetic volume estimate with traceable layer decisions."""

    magnetic_volume_m3: float
    magnetic_thickness_total_m: float
    area_m2: float
    included_layers: list[dict[str, Any]] = field(default_factory=list)
    excluded_layers: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    method: str = "geometry_area_times_magnetic_layer_thickness"


_LENGTH_UNITS = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "nm": 1e-9,
    "a": 1e-10,
    "angstrom": 1e-10,
}

_AREA_UNITS = {
    "m^2": 1.0,
    "cm^2": 1e-4,
    "mm^2": 1e-6,
    "um^2": 1e-12,
    "nm^2": 1e-18,
}

_VOLUME_UNITS = {
    "m^3": 1.0,
    "cm^3": 1e-6,
    "mm^3": 1e-9,
    "um^3": 1e-18,
    "nm^3": 1e-27,
}


def normalize_length_m(value: Any, unit: str | None) -> float:
    """Convert a length or thickness value to meters."""

    return _convert_unit(value, unit, _LENGTH_UNITS, "length")


def normalize_area_m2(value: Any, unit: str | None) -> float:
    """Convert an area value to square meters."""

    return _convert_unit(value, unit, _AREA_UNITS, "area")


def normalize_volume_m3(value: Any, unit: str | None) -> float:
    """Convert a volume value to cubic meters."""

    return _convert_unit(value, unit, _VOLUME_UNITS, "volume")


def estimate_area_m2(geometry: Mapping[str, Any]) -> float:
    """Estimate the active sample area from a geometry mapping."""

    if not isinstance(geometry, Mapping):
        raise MagneticVolumeError("Geometry must be a mapping.")

    shape = _shape_name(geometry)
    if shape == "rectangle":
        width_m = _positive_length(geometry, "width")
        length_m = _positive_length(geometry, "length")
        return _positive_area(width_m * length_m, "rectangle area")

    if shape == "square":
        side_m = _positive_length(geometry, "side")
        return _positive_area(side_m * side_m, "square area")

    if shape == "circle":
        diameter_m = (
            _positive_length(geometry, "diameter")
            if geometry.get("diameter") not in {None, ""}
            else 2.0 * _positive_length(geometry, "radius")
        )
        return _positive_area(math.pi * (diameter_m / 2.0) ** 2, "circle area")

    if shape == "custom_area":
        area_m2 = normalize_area_m2(_required(geometry, "area"), _unit_for(geometry, "area"))
        return _positive_area(area_m2, "custom area")

    if shape == "array":
        base_geometry = geometry.get("base_geometry")
        if not isinstance(base_geometry, Mapping):
            raise MagneticVolumeError("Array geometry requires base_geometry mapping.")
        if _shape_name(base_geometry) == "array":
            raise MagneticVolumeError("Nested array geometries are not supported.")
        element_count = _positive_int(geometry.get("element_count"), "element_count")
        fill_factor = _fill_factor(geometry.get("fill_factor", 1.0))
        return _positive_area(
            estimate_area_m2(base_geometry) * element_count * fill_factor,
            "array area",
        )

    raise MagneticVolumeError(f"Unsupported geometry shape: {shape!r}.")


def estimate_magnetic_volume(
    geometry: Mapping[str, Any],
    layer_stack: Sequence[Any],
) -> MagneticVolumeEstimate:
    """Estimate magnetic material volume from geometry and layer-stack metadata."""

    area_m2 = estimate_area_m2(geometry)
    included_layers, excluded_layers, warnings = _evaluate_layer_stack(layer_stack)
    magnetic_thickness_total_m = sum(float(layer["thickness_m"]) for layer in included_layers)
    if not included_layers or magnetic_thickness_total_m <= 0.0:
        raise MagneticVolumeError("No magnetic layers with positive thickness were found.")

    magnetic_volume_m3 = area_m2 * magnetic_thickness_total_m
    warnings.extend(_scale_warnings(area_m2=area_m2, magnetic_volume_m3=magnetic_volume_m3))

    return MagneticVolumeEstimate(
        magnetic_volume_m3=magnetic_volume_m3,
        magnetic_thickness_total_m=magnetic_thickness_total_m,
        area_m2=area_m2,
        included_layers=included_layers,
        excluded_layers=excluded_layers,
        warnings=warnings,
        method="geometry_area_times_magnetic_layer_thickness",
    )


def estimate_from_area_and_thickness(
    area_value: Any,
    area_unit: str | None,
    thickness_value: Any,
    thickness_unit: str | None,
) -> MagneticVolumeEstimate:
    """Estimate volume from legacy area plus magnetic-thickness metadata."""

    area_m2 = _positive_area(normalize_area_m2(area_value, area_unit), "area")
    thickness_m = _thickness_m(thickness_value, thickness_unit, "magnetic_thickness")
    if thickness_m <= 0.0:
        raise MagneticVolumeError("Magnetic thickness must be greater than zero.")
    included = [
        {
            "material": "legacy_magnetic_thickness",
            "thickness": float(thickness_value),
            "thickness_unit": str(thickness_unit),
            "thickness_m": thickness_m,
            "magnetic": True,
            "role": None,
            "notes": "legacy area times magnetic_thickness metadata",
            "path": "legacy",
        }
    ]
    magnetic_volume_m3 = area_m2 * thickness_m
    return MagneticVolumeEstimate(
        magnetic_volume_m3=magnetic_volume_m3,
        magnetic_thickness_total_m=thickness_m,
        area_m2=area_m2,
        included_layers=included,
        excluded_layers=[],
        warnings=_scale_warnings(area_m2=area_m2, magnetic_volume_m3=magnetic_volume_m3),
        method="area_times_magnetic_thickness_legacy",
    )


def _scale_warnings(*, area_m2: float, magnetic_volume_m3: float) -> list[str]:
    warnings: list[str] = []
    if area_m2 < 1e-18:
        warnings.append("Estimated area is very small; confirm geometry units.")
    elif area_m2 > 1e-2:
        warnings.append("Estimated area is very large; confirm geometry units.")
    if magnetic_volume_m3 < 1e-27:
        warnings.append(
            "Estimated magnetic volume is very small; confirm dimensions and thickness units."
        )
    elif magnetic_volume_m3 > 1e-6:
        warnings.append(
            "Estimated magnetic volume is very large; confirm dimensions and thickness units."
        )
    return warnings


def _evaluate_layer_stack(
    layer_stack: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if not isinstance(layer_stack, Sequence) or isinstance(layer_stack, (str, bytes)):
        raise MagneticVolumeError("layer_stack must be a sequence of layers or repeat blocks.")
    if not layer_stack:
        raise MagneticVolumeError("layer_stack must not be empty.")

    included_layers: list[dict[str, Any]] = []
    excluded_layers: list[dict[str, Any]] = []
    warnings: list[str] = []

    for stack_index, item in enumerate(layer_stack):
        if _is_repeat_block(item):
            repeat_count, layers = _repeat_block_parts(item, stack_index)
            for repeat_index in range(1, repeat_count + 1):
                for layer_index, layer_item in enumerate(layers):
                    if _is_repeat_block(layer_item):
                        raise MagneticVolumeError("Nested repeat blocks are not supported.")
                    _classify_layer(
                        layer_item,
                        path=f"{stack_index}.repeat[{repeat_index}].layers[{layer_index}]",
                        included_layers=included_layers,
                        excluded_layers=excluded_layers,
                        warnings=warnings,
                    )
            continue

        _classify_layer(
            item,
            path=str(stack_index),
            included_layers=included_layers,
            excluded_layers=excluded_layers,
            warnings=warnings,
        )

    return included_layers, excluded_layers, warnings


def _classify_layer(
    item: Any,
    *,
    path: str,
    included_layers: list[dict[str, Any]],
    excluded_layers: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    layer = _layer_from_item(item, path)
    thickness_m = _thickness_m(layer.thickness, layer.thickness_unit, f"layer {path} thickness")
    report = {
        "material": layer.material,
        "thickness": float(layer.thickness),
        "thickness_unit": layer.thickness_unit,
        "thickness_m": thickness_m,
        "magnetic": layer.magnetic,
        "role": layer.role,
        "notes": layer.notes,
        "path": path,
    }
    if thickness_m == 0.0:
        excluded_layers.append({**report, "reason": "zero_thickness"})
        warnings.append(f"Layer {path} ({layer.material}) has zero thickness and was excluded.")
        return
    if not layer.magnetic:
        excluded_layers.append({**report, "reason": "non_magnetic"})
        return
    included_layers.append(report)


def _layer_from_item(item: Any, path: str) -> MagneticLayer:
    if isinstance(item, MagneticLayer):
        return item
    if not isinstance(item, Mapping):
        raise MagneticVolumeError(f"Layer {path} must be a mapping or MagneticLayer.")

    material = str(_required(item, "material")).strip()
    if not material:
        raise MagneticVolumeError(f"Layer {path} material must not be empty.")

    return MagneticLayer(
        material=material,
        thickness=_number(_required(item, "thickness"), f"layer {path} thickness"),
        thickness_unit=str(_required(item, "thickness_unit")).strip(),
        magnetic=_bool_value(_required(item, "magnetic"), f"layer {path} magnetic"),
        role=_optional_str(item.get("role")),
        notes=_optional_str(item.get("notes")),
    )


def _repeat_block_parts(item: Any, stack_index: int) -> tuple[int, Sequence[Any]]:
    if isinstance(item, MagneticLayerBlock):
        repeat_count = _positive_int(item.repeat_count, f"repeat block {stack_index} repeat_count")
        layers: Sequence[Any] = item.layers
    elif isinstance(item, Mapping):
        repeat_count = _positive_int(
            item.get("repeat_count"),
            f"repeat block {stack_index} repeat_count",
        )
        layers = item.get("layers")
    else:
        raise MagneticVolumeError(
            f"Repeat block {stack_index} must be a mapping or MagneticLayerBlock."
        )

    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        raise MagneticVolumeError(f"Repeat block {stack_index} layers must be a sequence.")
    if not layers:
        raise MagneticVolumeError(f"Repeat block {stack_index} layers must not be empty.")
    return repeat_count, layers


def _is_repeat_block(item: Any) -> bool:
    if isinstance(item, MagneticLayerBlock):
        return True
    return isinstance(item, Mapping) and "repeat_count" in item and "layers" in item


def _shape_name(geometry: Mapping[str, Any]) -> str:
    raw_shape = geometry.get("shape", geometry.get("type"))
    if raw_shape in {None, ""}:
        raise MagneticVolumeError("Geometry requires a shape or type field.")
    return str(raw_shape).strip().lower().replace("-", "_")


def _positive_length(geometry: Mapping[str, Any], key: str) -> float:
    value = _required(geometry, key)
    unit = _unit_for(geometry, key)
    return _positive_value(normalize_length_m(value, unit), key)


def _unit_for(geometry: Mapping[str, Any], key: str) -> str | None:
    return _optional_str(geometry.get(f"{key}_unit")) or _optional_str(geometry.get("unit"))


def _thickness_m(value: Any, unit: str | None, label: str) -> float:
    thickness_m = normalize_length_m(value, unit)
    if thickness_m < 0.0:
        raise MagneticVolumeError(f"{label} must not be negative.")
    return thickness_m


def _positive_area(value: float, label: str) -> float:
    return _positive_value(value, label)


def _positive_value(value: float, label: str) -> float:
    if value <= 0.0:
        raise MagneticVolumeError(f"{label} must be greater than zero.")
    return value


def _positive_int(value: Any, label: str) -> int:
    number = _number(value, label)
    if not float(number).is_integer():
        raise MagneticVolumeError(f"{label} must be an integer.")
    integer = int(number)
    if integer <= 0:
        raise MagneticVolumeError(f"{label} must be greater than zero.")
    return integer


def _fill_factor(value: Any) -> float:
    fill_factor = _number(value, "fill_factor")
    if fill_factor <= 0.0 or fill_factor > 1.0:
        raise MagneticVolumeError(
            "fill_factor must be greater than zero and less than or equal to one."
        )
    return fill_factor


def _convert_unit(value: Any, unit: str | None, table: dict[str, float], label: str) -> float:
    number = _number(value, label)
    key = _unit_key(unit)
    try:
        scale = table[key]
    except KeyError as exc:
        raise MagneticVolumeError(f"Unsupported {label} unit: {unit!r}.") from exc
    return number * scale


def _unit_key(unit: str | None) -> str:
    text = _optional_str(unit)
    if text is None:
        raise MagneticVolumeError("Unit is required.")
    return (
        text.lower()
        .replace("\u00b5", "u")
        .replace("\u03bc", "u")
        .replace("\u00b2", "^2")
        .replace("\u00b3", "^3")
        .replace(" ", "")
    )


def _number(value: Any, label: str) -> float:
    if value in {None, ""}:
        raise MagneticVolumeError(f"{label} is required.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MagneticVolumeError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise MagneticVolumeError(f"{label} must be finite.")
    return number


def _bool_value(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "magnetic"}:
            return True
        if text in {"false", "no", "n", "0", "non_magnetic", "non-magnetic"}:
            return False
    raise MagneticVolumeError(f"{label} must be boolean.")


def _required(payload: Mapping[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value in {None, ""}:
        raise MagneticVolumeError(f"Missing required field: {key}.")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
