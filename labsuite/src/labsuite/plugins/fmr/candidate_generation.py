"""Initial-guess generation for FMR multi-resonance trace fits."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from labsuite.core.recipes import FmrRecipe
from labsuite.plugins.fmr.models import FmrCandidateWindow


def detect_candidate_windows(
    field: np.ndarray, signal: np.ndarray, recipe: FmrRecipe
) -> list[FmrCandidateWindow]:
    """Detect derivative peak/trough windows on the processed trace."""

    field = np.asarray(field, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if field.size < 3 or signal.size < 3:
        return []
    step = median_step(field)
    dist = max(1, int(round(recipe.peak_min_distance_mT / max(step, 1e-9))))
    prom = max(float(np.max(np.abs(signal))) * recipe.peak_min_prominence_ratio, 1e-9)
    pos_i, pos_p = find_peaks(signal, prominence=prom, distance=dist)
    neg_i, neg_p = find_peaks(-signal, prominence=prom, distance=dist)
    if pos_i.size == 0 or neg_i.size == 0:
        return []
    candidates: dict[tuple[int, int], FmrCandidateWindow] = {}
    for source_i, source_p, target_i, target_p, source_kind in (
        (pos_i, pos_p["prominences"], neg_i, neg_p["prominences"], "positive"),
        (neg_i, neg_p["prominences"], pos_i, pos_p["prominences"], "negative"),
    ):
        for idx, prominence in zip(source_i, source_p, strict=True):
            nearest = int(np.argmin(np.abs(field[target_i] - field[idx])))
            other = int(target_i[nearest])
            width = abs(float(field[other] - field[idx]))
            if width < recipe.peak_min_pair_width_mT:
                continue
            peak_idx, trough_idx = (
                (int(idx), other) if source_kind == "positive" else (other, int(idx))
            )
            left, right = min(peak_idx, trough_idx), max(peak_idx, trough_idx)
            padding = max(
                4,
                int(
                    round(
                        (width * recipe.candidate_window_padding_width_multiplier)
                        / max(step, 1e-9)
                    )
                ),
            )
            start = max(0, left - padding)
            end = min(signal.size - 1, right + padding)
            key = (left, right)
            item = FmrCandidateWindow(
                "",
                start,
                end,
                float(field[start]),
                float(field[end]),
                peak_idx,
                trough_idx,
                float(field[peak_idx]),
                float(field[trough_idx]),
                width,
                float(min(prominence, target_p[nearest])),
                float((field[peak_idx] + field[trough_idx]) / 2.0),
            )
            prev = candidates.get(key)
            if prev is None or item.prominence > prev.prominence:
                candidates[key] = item
    ranked = sorted(candidates.values(), key=lambda item: item.prominence, reverse=True)
    return _dedupe_and_label(ranked, recipe.max_resonance_count)


def candidate_guesses_for_n(
    field: np.ndarray,
    signal: np.ndarray,
    n_peaks: int,
    recipe: FmrRecipe,
    *,
    residual: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Build N component guesses from signal features and optional residual features."""

    step = median_step(field)
    windows = detect_candidate_windows(field, signal, recipe)
    guesses = [_window_guess(field, signal, window, step) for window in windows]
    if len(guesses) < n_peaks and residual is not None:
        residual_windows = residual_candidate_windows(field, residual, signal, recipe)
        guesses.extend(_window_guess(field, residual, window, step) for window in residual_windows)
    if not guesses:
        guesses.append(strongest_feature_guess(field, signal, recipe))
    guesses = _dedupe_guesses(guesses, min_separation=max(recipe.min_peak_separation_mT, step))
    while len(guesses) < n_peaks:
        center = float(np.mean(field)) + (len(guesses) - n_peaks / 2.0) * max(3.0 * step, 5.0)
        center = float(np.clip(center, np.min(field), np.max(field)))
        amplitude = max(float(np.max(np.abs(signal))), 1e-6) / (len(guesses) + 1)
        guesses.append(
            {
                "center": center,
                "linewidth": max(step * 4.0, recipe.min_linewidth_mT or 1e-4),
                "sym": amplitude,
                "asym": 0.1 * amplitude,
            }
        )
    guesses.sort(key=lambda item: item["center"])
    return guesses[:n_peaks]


def residual_candidate_windows(
    field: np.ndarray,
    residual: np.ndarray,
    signal: np.ndarray,
    recipe: FmrRecipe,
) -> list[FmrCandidateWindow]:
    """Find shoulder-like structured residual features without heavy smoothing."""

    residual = np.asarray(residual, dtype=float)
    if residual.size < 7:
        return []
    window = min(residual.size if residual.size % 2 else residual.size - 1, 7)
    if window >= 5:
        smoothed = savgol_filter(residual, window_length=window, polyorder=2)
    else:
        smoothed = residual
    residual_scale = max(float(np.std(residual)), 1e-12)
    signal_scale = max(float(np.max(np.abs(signal))), 1e-12)
    prom_ratio = max(recipe.peak_min_prominence_ratio * 0.35, 0.03)
    prom = max(residual_scale * 1.5, signal_scale * prom_ratio)
    step = median_step(field)
    dist = max(1, int(round(max(recipe.min_peak_separation_mT, step) / max(step, 1e-9))))
    pos_i, pos_p = find_peaks(smoothed, prominence=prom, distance=dist)
    neg_i, neg_p = find_peaks(-smoothed, prominence=prom, distance=dist)
    if pos_i.size == 0 or neg_i.size == 0:
        return []
    candidates: list[FmrCandidateWindow] = []
    for pos in pos_i:
        nearest = int(np.argmin(np.abs(field[neg_i] - field[pos])))
        neg = int(neg_i[nearest])
        width = abs(float(field[pos] - field[neg]))
        if width < max(step, recipe.peak_min_pair_width_mT * 0.35):
            continue
        left, right = min(int(pos), neg), max(int(pos), neg)
        pad = max(3, int(round(width / max(step, 1e-9))))
        start = max(0, left - pad)
        end = min(field.size - 1, right + pad)
        candidates.append(
            FmrCandidateWindow(
                "",
                start,
                end,
                float(field[start]),
                float(field[end]),
                int(pos),
                neg,
                float(field[pos]),
                float(field[neg]),
                width,
                float(
                    min(
                        pos_p["prominences"][np.where(pos_i == pos)[0][0]],
                        neg_p["prominences"][nearest],
                    )
                ),
                float((field[pos] + field[neg]) / 2.0),
            )
        )
    ranked = sorted(candidates, key=lambda item: item.prominence, reverse=True)
    return _dedupe_and_label(ranked, recipe.max_resonance_count)


def strongest_feature_guess(
    field: np.ndarray, signal: np.ndarray, recipe: FmrRecipe
) -> dict[str, float]:
    """Single strongest derivative feature guess."""

    windows = detect_candidate_windows(field, signal, recipe)
    if windows:
        return _window_guess(field, signal, windows[0], median_step(field))
    feature = detect_feature(field, signal, recipe.shape_pair_prominence_ratio)
    center = (
        float(field[int(np.argmax(np.abs(signal)))])
        if feature["feature_center_mT"] is None
        else float(feature["feature_center_mT"])
    )
    width = (
        0.0
        if feature["feature_peak_to_peak_mT"] is None
        else float(feature["feature_peak_to_peak_mT"])
    )
    step = median_step(field)
    amplitude = max(float(np.max(np.abs(signal))), 1e-6)
    max_i = int(np.argmax(signal))
    min_i = int(np.argmin(signal))
    sign = 1.0 if field[max_i] < field[min_i] else -1.0
    return {
        "center": center,
        "linewidth": max(width * math.sqrt(3.0) / 2.0, step * 2.0, 1e-4),
        "sym": sign * amplitude,
        "asym": 0.1 * amplitude,
    }


def detect_feature(
    field: np.ndarray, signal: np.ndarray, prominence_ratio: float
) -> dict[str, float | None]:
    """Return peak/trough landmarks for a derivative-like feature."""

    field = np.asarray(field, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if field.size == 0 or signal.size == 0:
        return _empty_feature()
    scale = (
        float(np.max(np.abs(signal[np.isfinite(signal)]))) if np.any(np.isfinite(signal)) else 0.0
    )
    prominence = max(scale * prominence_ratio, 1e-12)
    pos, _ = find_peaks(signal, prominence=prominence)
    neg, _ = find_peaks(-signal, prominence=prominence)
    if pos.size == 0:
        pos = np.asarray([int(np.argmax(signal))], dtype=int)
    if neg.size == 0:
        neg = np.asarray([int(np.argmin(signal))], dtype=int)
    best_pair: tuple[int, int] | None = None
    best_score = -np.inf
    for pos_i in pos:
        for neg_i in neg:
            if pos_i == neg_i:
                continue
            score = abs(float(signal[pos_i])) + abs(float(signal[neg_i]))
            if score > best_score:
                best_pair = (int(pos_i), int(neg_i))
                best_score = score
    if best_pair is None:
        return _empty_feature()
    pos_field = float(field[best_pair[0]])
    neg_field = float(field[best_pair[1]])
    return {
        "feature_center_mT": float((pos_field + neg_field) / 2.0),
        "feature_peak_to_peak_mT": float(abs(pos_field - neg_field)),
        "positive_extremum_mT": pos_field,
        "negative_extremum_mT": neg_field,
    }


def median_step(field: np.ndarray) -> float:
    diffs = np.diff(np.asarray(field, dtype=float))
    nz = np.abs(diffs[diffs != 0.0])
    return 1.0 if nz.size == 0 else float(np.median(nz))


def _window_guess(
    field: np.ndarray, signal: np.ndarray, window: FmrCandidateWindow, step: float
) -> dict[str, float]:
    sub_signal = signal[window.start_index : window.end_index + 1]
    sub_field = field[window.start_index : window.end_index + 1]
    max_i = int(np.argmax(sub_signal))
    min_i = int(np.argmin(sub_signal))
    width = abs(float(sub_field[max_i] - sub_field[min_i]))
    amplitude = max(float(np.max(np.abs(sub_signal))), 1e-6)
    sign = 1.0 if sub_field[max_i] < sub_field[min_i] else -1.0
    return {
        "center": window.candidate_center_mT,
        "linewidth": max(width * math.sqrt(3.0) / 2.0, window.width_mT, step * 2.0, 1e-4),
        "sym": sign * amplitude,
        "asym": 0.1 * amplitude,
    }


def _dedupe_and_label(
    ranked: list[FmrCandidateWindow], max_count: int
) -> list[FmrCandidateWindow]:
    selected: list[FmrCandidateWindow] = []
    for item in ranked:
        overlaps = any(
            not (item.end_index < current.start_index or item.start_index > current.end_index)
            for current in selected
        )
        if overlaps:
            continue
        selected.append(item)
        if len(selected) == max_count:
            break
    selected.sort(key=lambda item: item.candidate_center_mT)
    return [
        FmrCandidateWindow(
            f"candidate_{index}",
            item.start_index,
            item.end_index,
            item.start_field_mT,
            item.end_field_mT,
            item.peak_index,
            item.trough_index,
            item.peak_field_mT,
            item.trough_field_mT,
            item.width_mT,
            item.prominence,
            item.candidate_center_mT,
        )
        for index, item in enumerate(selected, start=1)
    ]


def _dedupe_guesses(
    guesses: list[dict[str, float]], *, min_separation: float
) -> list[dict[str, float]]:
    selected: list[dict[str, float]] = []
    for guess in sorted(guesses, key=lambda item: abs(item["sym"]), reverse=True):
        if any(abs(guess["center"] - item["center"]) < min_separation for item in selected):
            continue
        selected.append(guess)
    selected.sort(key=lambda item: item["center"])
    return selected


def _empty_feature() -> dict[str, float | None]:
    return {
        "feature_center_mT": None,
        "feature_peak_to_peak_mT": None,
        "positive_extremum_mT": None,
        "negative_extremum_mT": None,
    }
