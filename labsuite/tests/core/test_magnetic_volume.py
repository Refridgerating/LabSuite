from __future__ import annotations

import math

import pytest

from labsuite.core.magnetic_volume import (
    MagneticLayer,
    MagneticLayerBlock,
    MagneticVolumeError,
    estimate_area_m2,
    estimate_from_area_and_thickness,
    estimate_magnetic_volume,
    normalize_area_m2,
    normalize_length_m,
    normalize_volume_m3,
)


def test_rectangle_volume_counts_only_magnetic_layers() -> None:
    estimate = estimate_magnetic_volume(
        {"shape": "rectangle", "width": 10.0, "length": 20.0, "unit": "um"},
        [
            {"material": "Ta", "thickness": 2.0, "thickness_unit": "nm", "magnetic": False},
            {"material": "Co", "thickness": 5.0, "thickness_unit": "nm", "magnetic": True},
        ],
    )

    assert estimate.area_m2 == pytest.approx(2.0e-10)
    assert estimate.magnetic_thickness_total_m == pytest.approx(5.0e-9)
    assert estimate.magnetic_volume_m3 == pytest.approx(1.0e-18)
    assert [layer["material"] for layer in estimate.included_layers] == ["Co"]
    assert [layer["material"] for layer in estimate.excluded_layers] == ["Ta"]
    assert estimate.excluded_layers[0]["reason"] == "non_magnetic"
    assert estimate.method == "geometry_area_times_magnetic_layer_thickness"


def test_square_geometry() -> None:
    estimate = estimate_magnetic_volume(
        {"type": "square", "side": 2.0, "side_unit": "mm"},
        [{"material": "NiFe", "thickness": 10.0, "thickness_unit": "nm", "magnetic": True}],
    )

    assert estimate.area_m2 == pytest.approx(4.0e-6)
    assert estimate.magnetic_volume_m3 == pytest.approx(4.0e-14)


def test_circle_geometry() -> None:
    estimate = estimate_magnetic_volume(
        {"shape": "circle", "diameter": 100.0, "diameter_unit": "um"},
        [{"material": "CoFeB", "thickness": 1.5, "thickness_unit": "nm", "magnetic": True}],
    )

    assert estimate.area_m2 == pytest.approx(math.pi * (50.0e-6) ** 2)
    assert estimate.magnetic_volume_m3 == pytest.approx(math.pi * (50.0e-6) ** 2 * 1.5e-9)


def test_circle_radius_geometry_and_angstrom_thickness() -> None:
    estimate = estimate_magnetic_volume(
        {"shape": "circle", "radius": 50.0, "radius_unit": "um"},
        [{"material": "CoFeB", "thickness": 15.0, "thickness_unit": "angstrom", "magnetic": True}],
    )

    assert estimate.area_m2 == pytest.approx(math.pi * (50.0e-6) ** 2)
    assert estimate.magnetic_volume_m3 == pytest.approx(math.pi * (50.0e-6) ** 2 * 1.5e-9)


def test_custom_area_geometry() -> None:
    estimate = estimate_magnetic_volume(
        {"shape": "custom_area", "area": 2500.0, "area_unit": "um^2"},
        [{"material": "Co", "thickness": 2.0, "thickness_unit": "nm", "magnetic": True}],
    )

    assert estimate.area_m2 == pytest.approx(2500.0e-12)
    assert estimate.magnetic_volume_m3 == pytest.approx(5.0e-18)


def test_array_geometry_applies_element_count_and_fill_factor() -> None:
    estimate = estimate_magnetic_volume(
        {
            "shape": "array",
            "base_geometry": {"shape": "circle", "diameter": 100.0, "diameter_unit": "nm"},
            "element_count": 1_000_000,
            "fill_factor": 0.5,
        },
        [{"material": "NiFe", "thickness": 2.0, "thickness_unit": "nm", "magnetic": True}],
    )

    expected_area = math.pi * (50.0e-9) ** 2 * 1_000_000 * 0.5
    assert estimate.area_m2 == pytest.approx(expected_area)
    assert estimate.magnetic_volume_m3 == pytest.approx(expected_area * 2.0e-9)


def test_repeated_multilayer_blocks_are_expanded() -> None:
    estimate = estimate_magnetic_volume(
        {"shape": "square", "side": 1.0, "side_unit": "m"},
        [
            MagneticLayer("Ta", 3.0, "nm", False),
            MagneticLayerBlock(
                repeat_count=3,
                layers=[
                    MagneticLayer("Co", 0.8, "nm", True, role="free"),
                    MagneticLayer("Ru", 0.6, "nm", False, role="spacer"),
                    MagneticLayer("NiFe", 1.2, "nm", True, role="reference"),
                ],
            ),
        ],
    )

    assert estimate.magnetic_thickness_total_m == pytest.approx(6.0e-9)
    assert estimate.magnetic_volume_m3 == pytest.approx(6.0e-9)
    assert [layer["material"] for layer in estimate.included_layers] == [
        "Co",
        "NiFe",
        "Co",
        "NiFe",
        "Co",
        "NiFe",
    ]
    assert [layer["material"] for layer in estimate.excluded_layers] == ["Ta", "Ru", "Ru", "Ru"]


def test_zero_thickness_layers_are_excluded_with_warning() -> None:
    estimate = estimate_magnetic_volume(
        {"shape": "square", "side": 1.0, "side_unit": "um"},
        [
            {"material": "Co", "thickness": 0.0, "thickness_unit": "nm", "magnetic": True},
            {"material": "NiFe", "thickness": 1.0, "thickness_unit": "nm", "magnetic": True},
        ],
    )

    assert estimate.magnetic_thickness_total_m == pytest.approx(1.0e-9)
    assert estimate.excluded_layers[0]["reason"] == "zero_thickness"
    assert estimate.warnings == ["Layer 0 (Co) has zero thickness and was excluded."]


def test_unit_conversion_helpers_support_si_prefixes_and_micro_symbol() -> None:
    assert normalize_length_m(1.0, "m") == pytest.approx(1.0)
    assert normalize_length_m(2.0, "cm") == pytest.approx(2.0e-2)
    assert normalize_length_m(3.0, "mm") == pytest.approx(3.0e-3)
    assert normalize_length_m(4.0, "um") == pytest.approx(4.0e-6)
    assert normalize_length_m(5.0, "\u00b5m") == pytest.approx(5.0e-6)
    assert normalize_length_m(6.0, "nm") == pytest.approx(6.0e-9)
    assert normalize_area_m2(7.0, "\u00b5m^2") == pytest.approx(7.0e-12)
    assert normalize_volume_m3(8.0, "\u00b5m^3") == pytest.approx(8.0e-18)


def test_estimate_from_area_and_thickness_supports_legacy_registry_inputs() -> None:
    estimate = estimate_from_area_and_thickness(2.0, "mm^2", 5.0, "nm")

    assert estimate.area_m2 == pytest.approx(2.0e-6)
    assert estimate.magnetic_thickness_total_m == pytest.approx(5.0e-9)
    assert estimate.magnetic_volume_m3 == pytest.approx(1.0e-14)
    assert estimate.included_layers[0]["material"] == "legacy_magnetic_thickness"
    assert estimate.method == "area_times_magnetic_thickness_legacy"


@pytest.mark.parametrize(
    ("geometry", "match"),
    [
        ({"shape": "square", "side": 5.0}, "Unit is required"),
        ({"shape": "rectangle", "width": 5.0, "unit": "um"}, "Missing required field: length"),
        (
            {"shape": "custom_area", "area": 0.0, "area_unit": "m^2"},
            "custom area must be greater than zero",
        ),
        (
            {
                "shape": "array",
                "base_geometry": {"shape": "square", "side": 1.0, "side_unit": "um"},
                "element_count": 0,
            },
            "element_count must be greater than zero",
        ),
        (
            {
                "shape": "array",
                "base_geometry": {"shape": "square", "side": 1.0, "side_unit": "um"},
                "element_count": 1,
                "fill_factor": 1.5,
            },
            "fill_factor must be greater than zero",
        ),
        (
            {
                "shape": "array",
                "base_geometry": {
                    "shape": "array",
                    "base_geometry": {"shape": "square", "side": 1.0, "side_unit": "um"},
                    "element_count": 1,
                },
                "element_count": 1,
            },
            "Nested array geometries are not supported",
        ),
    ],
)
def test_geometry_validation_errors(geometry: dict[str, object], match: str) -> None:
    with pytest.raises(MagneticVolumeError, match=match):
        estimate_area_m2(geometry)


@pytest.mark.parametrize(
    ("layer_stack", "match"),
    [
        ([], "layer_stack must not be empty"),
        (
            [{"material": "Ta", "thickness": 2.0, "thickness_unit": "nm", "magnetic": False}],
            "No magnetic layers",
        ),
        (
            [{"material": "Co", "thickness": -1.0, "thickness_unit": "nm", "magnetic": True}],
            "must not be negative",
        ),
        (
            [{"material": "Co", "thickness": 1.0, "magnetic": True}],
            "Missing required field: thickness_unit",
        ),
        (
            [
                {
                    "material": "Co",
                    "thickness": 1.0,
                    "thickness_unit": "badunit",
                    "magnetic": True,
                }
            ],
            "Unsupported length unit",
        ),
        (
            [
                {
                    "repeat_count": 2,
                    "layers": [
                        {
                            "repeat_count": 2,
                            "layers": [
                                {
                                    "material": "Co",
                                    "thickness": 1.0,
                                    "thickness_unit": "nm",
                                    "magnetic": True,
                                }
                            ],
                        }
                    ],
                }
            ],
            "Nested repeat blocks are not supported",
        ),
    ],
)
def test_layer_stack_validation_errors(layer_stack: list[object], match: str) -> None:
    with pytest.raises(MagneticVolumeError, match=match):
        estimate_magnetic_volume(
            {"shape": "square", "side": 1.0, "side_unit": "um"},
            layer_stack,
        )
