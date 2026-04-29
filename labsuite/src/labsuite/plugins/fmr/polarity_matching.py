"""Cross-file positive/negative FMR branch matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PolarityMatchConfig:
    frequency_tolerance_GHz: float = 0.001
    allow_low_confidence: bool = False


def match_positive_negative_points(
    raw_points: list[dict[str, Any]],
    *,
    config: PolarityMatchConfig | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Match separate positive and negative branch points by metadata."""

    cfg = config or PolarityMatchConfig()
    warnings: list[str] = []
    positives = [point for point in raw_points if point.get("field_polarity") == "positive"]
    negatives = [point for point in raw_points if point.get("field_polarity") == "negative"]
    used_negative_ids: set[int] = set()
    output: list[dict[str, Any]] = []
    pair_index = 0
    for positive in positives:
        candidates = [
            (index, negative)
            for index, negative in enumerate(negatives)
            if index not in used_negative_ids
            and _metadata_compatible(positive, negative)
            and abs(float(positive["frequency_GHz"]) - float(negative["frequency_GHz"]))
            <= cfg.frequency_tolerance_GHz
        ]
        if len(candidates) != 1:
            status = "ambiguous_pair" if len(candidates) > 1 else "unpaired_positive"
            output.append(_unpaired_point(positive, status))
            warnings.append(f"polarity_matching_{status}:{positive.get('component_id')}")
            continue
        negative_index, negative = candidates[0]
        used_negative_ids.add(negative_index)
        pair_index += 1
        pair_id = _pair_id(positive, pair_index)
        output.append(_paired_point(pair_id, positive, negative))
    for index, negative in enumerate(negatives):
        if index in used_negative_ids:
            continue
        output.append(_unpaired_point(negative, "unpaired_negative"))
        warnings.append(f"polarity_matching_unpaired_negative:{negative.get('component_id')}")
    for point in raw_points:
        if point.get("field_polarity") not in {"positive", "negative"}:
            output.append(_unpaired_point(point, "unknown_polarity"))
            warnings.append(f"polarity_matching_unknown_polarity:{point.get('component_id')}")
    return sorted(
        output,
        key=lambda item: (float(item.get("frequency_GHz") or 0.0), str(item.get("component_id"))),
    ), warnings


def _metadata_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in ("sample_id", "geometry", "branch_id"):
        if left.get(key) and right.get(key) and left.get(key) != right.get(key):
            return False
    if left.get("replicate_id") and right.get("replicate_id"):
        return left.get("replicate_id") == right.get("replicate_id")
    return True


def _paired_point(
    pair_id: str,
    positive: dict[str, Any],
    negative: dict[str, Any],
) -> dict[str, Any]:
    pos_abs = abs(float(positive["Hres_raw_mT"]))
    neg_abs = abs(float(negative["Hres_raw_mT"]))
    hres_avg = (pos_abs + neg_abs) / 2.0
    hres_asymmetry = abs(pos_abs - neg_abs)
    confidence = (
        "high" if positive.get("replicate_id") and negative.get("replicate_id") else "medium"
    )
    item = dict(positive)
    item.update(
        {
            "matched_pair_id": pair_id,
            "polarity_pair_id": pair_id,
            "polarity_pair_status": "paired",
            "matching_confidence": confidence,
            "positive_measurement_id": positive.get("measurement_id"),
            "negative_measurement_id": negative.get("measurement_id"),
            "field_polarity": "paired",
            "field_polarity_raw": (
                f"{positive.get('field_polarity_raw')}|{negative.get('field_polarity_raw')}"
            ),
            "frequency_GHz": float(
                (float(positive["frequency_GHz"]) + float(negative["frequency_GHz"])) / 2.0
            ),
            "trace_id": f"{positive.get('trace_id')}|{negative.get('trace_id')}",
            "component_id": f"{positive.get('component_id')}|{negative.get('component_id')}",
            "Hres_pos_mT": float(positive["Hres_raw_mT"]),
            "Hres_neg_mT": float(negative["Hres_raw_mT"]),
            "Hres_positive_mT": float(positive["Hres_raw_mT"]),
            "Hres_negative_mT": float(negative["Hres_raw_mT"]),
            "Hres_avg_mT": hres_avg,
            "Hres_offset_mT": (pos_abs - neg_abs) / 2.0,
            "Hres_split_mT": hres_asymmetry,
            "Hres_asymmetry_mT": hres_asymmetry,
            "DeltaH_fit_mT": float(
                (float(positive["DeltaH_raw_mT"]) + float(negative["DeltaH_raw_mT"])) / 2.0
            ),
            "fit_field": "Hres_avg",
        }
    )
    return item


def _unpaired_point(point: dict[str, Any], status: str) -> dict[str, Any]:
    item = dict(point)
    item.setdefault("matched_pair_id", None)
    item.setdefault("matching_confidence", "low")
    item["polarity_pair_status"] = status
    item["fit_field"] = "Hres"
    item.setdefault("Hres_asymmetry_mT", None)
    return item


def _pair_id(point: dict[str, Any], pair_index: int) -> str:
    sample = point.get("sample_id") or "unknown"
    branch = point.get("branch_id") or point.get("mode_id") or "branch"
    return f"{sample}:{branch}:{pair_index:03d}"
