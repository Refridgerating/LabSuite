"""Branch assignment for per-trace FMR resonance components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from labsuite.plugins.fmr.models import FmrComponentFitResult, FmrTraceFitResult


@dataclass(slots=True)
class BranchTrackingConfig:
    max_hres_jump_mT: float = 75.0
    linewidth_jump_weight: float = 0.25
    missing_point_penalty: float = 20.0
    ambiguity_ratio: float = 1.25
    max_branches: int | None = None
    max_missing_traces: int = 4


def assign_branches(
    trace_fit_results: list[FmrTraceFitResult],
    *,
    config: BranchTrackingConfig | None = None,
) -> list[FmrTraceFitResult]:
    """Assign stable branch IDs by continuity in Hres(f)."""

    cfg = config or BranchTrackingConfig()
    ordered = sorted(trace_fit_results, key=lambda item: (float(item.frequency_GHz), item.trace_id))
    active: dict[str, FmrComponentFitResult] = {}
    missing_counts: dict[str, int] = {}
    branch_counter = 0
    for trace in ordered:
        components = [
            component
            for component in sorted(trace.selected_components, key=lambda item: item.H_res_mT)
            if component.accepted
        ]
        if not active:
            for component in components:
                if cfg.max_branches is not None and branch_counter >= cfg.max_branches:
                    _mark_unassigned(component, "branch_cap_reached")
                    continue
                branch_counter += 1
                _assign(component, f"branch_{branch_counter}", "high")
                active[component.branch_id or ""] = component
                missing_counts[component.branch_id or ""] = 0
            continue
        assignments = _greedy_assign(active, components, cfg)
        used_components: set[int] = set()
        matched_branches: set[str] = set()
        for branch_id, component, confidence in assignments:
            _assign(component, branch_id, confidence)
            used_components.add(id(component))
            active[branch_id] = component
            missing_counts[branch_id] = 0
            matched_branches.add(branch_id)
        for branch_id in list(active):
            if branch_id in matched_branches:
                continue
            missing_counts[branch_id] = missing_counts.get(branch_id, 0) + 1
            if missing_counts[branch_id] > cfg.max_missing_traces:
                active.pop(branch_id, None)
                missing_counts.pop(branch_id, None)
        for component in components:
            if id(component) in used_components:
                continue
            if cfg.max_branches is not None and branch_counter >= cfg.max_branches:
                _mark_unassigned(component, "branch_cap_reached")
                continue
            branch_counter += 1
            _assign(component, f"branch_{branch_counter}", "medium")
            active[component.branch_id or ""] = component
            missing_counts[component.branch_id or ""] = 0
    return trace_fit_results


def _greedy_assign(
    active: dict[str, FmrComponentFitResult],
    components: list[FmrComponentFitResult],
    config: BranchTrackingConfig,
) -> list[tuple[str, FmrComponentFitResult, str]]:
    candidates: list[tuple[float, str, FmrComponentFitResult]] = []
    for branch_id, previous in active.items():
        for component in components:
            h_jump = abs(float(component.H_res_mT) - float(previous.H_res_mT))
            linewidth_jump = abs(float(component.DeltaH_mT) - float(previous.DeltaH_mT))
            cost = h_jump + config.linewidth_jump_weight * linewidth_jump
            if h_jump <= config.max_hres_jump_mT:
                candidates.append((cost, branch_id, component))
    candidates.sort(key=lambda item: item[0])
    assigned_branches: set[str] = set()
    assigned_components: set[int] = set()
    output: list[tuple[str, FmrComponentFitResult, str]] = []
    for cost, branch_id, component in candidates:
        if branch_id in assigned_branches or id(component) in assigned_components:
            continue
        alternative_costs = [
            other_cost
            for other_cost, other_branch, other_component in candidates
            if other_branch == branch_id and id(other_component) != id(component)
            and id(other_component) not in assigned_components
        ]
        if alternative_costs and min(alternative_costs) <= cost * config.ambiguity_ratio:
            component.confidence = "low"
            component.metadata["branch_ambiguity"] = True
            continue
        confidence = "high" if cost <= config.max_hres_jump_mT * 0.5 else "medium"
        assigned_branches.add(branch_id)
        assigned_components.add(id(component))
        output.append((branch_id, component, confidence))
    return output


def _assign(component: FmrComponentFitResult, branch_id: str, confidence: str) -> None:
    component.branch_id = branch_id
    component.confidence = confidence
    component.metadata["branch_id"] = branch_id
    component.metadata["branch_confidence"] = confidence


def _mark_unassigned(component: FmrComponentFitResult, reason: str) -> None:
    component.branch_id = None
    component.confidence = "unassigned"
    component.metadata["branch_assignment_status"] = "unassigned"
    component.metadata["branch_rejection_reason"] = reason


def branch_ids(trace_fit_results: list[FmrTraceFitResult]) -> list[str]:
    """Return assigned branch IDs in natural order."""

    ids = {
        component.branch_id
        for trace in trace_fit_results
        for component in trace.selected_components
        if component.branch_id
    }
    return sorted(ids, key=_branch_sort_key)


def _branch_sort_key(value: str | None) -> tuple[int, str]:
    if value is None:
        return (10_000, "")
    parts = value.rsplit("_", maxsplit=1)
    if len(parts) == 2 and parts[1].isdigit():
        return (int(parts[1]), value)
    return (10_000, value)


def infer_branch_confidence(points: list[FmrComponentFitResult]) -> str:
    """Summarize branch confidence from point-level labels."""

    labels = [point.confidence for point in points if point.confidence != "unassigned"]
    if not labels:
        return "unassigned"
    if all(label == "high" for label in labels):
        return "high"
    if any(label == "low" for label in labels):
        return "low"
    return "medium"


def branch_span_quality(frequencies_GHz: np.ndarray) -> list[str]:
    """Warnings for branch-level fits that are weakly constrained by span."""

    if frequencies_GHz.size == 0:
        return ["no_branch_points"]
    span = float(np.nanmax(frequencies_GHz) - np.nanmin(frequencies_GHz))
    if span < 2.0:
        return ["frequency_span_too_narrow"]
    return []
