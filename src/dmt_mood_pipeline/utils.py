from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import List, Optional, Tuple


def compute_outer_split_sizes(
    num_items: int, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> tuple[int, int, int]:
    if num_items < 3:
        raise ValueError("At least 3 items are required to build train/val/test splits.")
    train_size = max(1, int(num_items * train_ratio))
    val_size = max(1, int(num_items * val_ratio))
    test_size = num_items - train_size - val_size
    if test_size < 1:
        deficit = 1 - test_size
        if train_size >= val_size and train_size - deficit >= 1:
            train_size -= deficit
        else:
            val_size -= deficit
        test_size = 1
    return train_size, val_size, test_size


def long_gap_flags(observed_any: Sequence[bool], threshold: int = 2) -> List[bool]:
    flags = [False] * len(observed_any)
    gap_start: Optional[int] = None
    for idx, observed in enumerate(observed_any):
        if observed:
            if gap_start is not None and idx - gap_start > threshold:
                for gap_idx in range(gap_start, idx):
                    flags[gap_idx] = True
            gap_start = None
            continue
        if gap_start is None:
            gap_start = idx
    if gap_start is not None and len(observed_any) - gap_start > threshold:
        for gap_idx in range(gap_start, len(observed_any)):
            flags[gap_idx] = True
    return flags


def days_since_last_observation(observed: Sequence[bool]) -> List[Optional[int]]:
    result: List[Optional[int]] = []
    last_seen: Optional[int] = None
    for idx, value in enumerate(observed):
        if value:
            last_seen = idx
            result.append(0)
        elif last_seen is None:
            result.append(None)
        else:
            result.append(idx - last_seen)
    return result


def rolling_linear_slope(values: Sequence[Optional[float]]) -> Optional[float]:
    points = [(float(idx), float(value)) for idx, value in enumerate(values) if value is not None]
    if len(points) < 2:
        return None
    x_mean = sum(x for x, _ in points) / len(points)
    y_mean = sum(y for _, y in points) / len(points)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if math.isclose(denominator, 0.0):
        return 0.0
    return numerator / denominator


def quantile_thresholds(values: Sequence[float], lower_q: float = 1 / 3, upper_q: float = 2 / 3) -> Tuple[float, float]:
    if not values:
        raise ValueError("Cannot compute quantile thresholds from an empty sequence.")
    ordered = sorted(float(value) for value in values)
    lower_idx = min(len(ordered) - 1, max(0, int(lower_q * (len(ordered) - 1))))
    upper_idx = min(len(ordered) - 1, max(0, int(upper_q * (len(ordered) - 1))))
    lower = ordered[lower_idx]
    upper = ordered[upper_idx]
    if math.isclose(lower, upper):
        unique = sorted(set(ordered))
        if len(unique) < 3:
            eps = 1e-6
            return lower - eps, upper + eps
        lower = unique[max(0, len(unique) // 3 - 1)]
        upper = unique[min(len(unique) - 1, (2 * len(unique)) // 3)]
        if math.isclose(lower, upper):
            upper = lower + 1e-6
    return lower, upper


def apply_thresholds(value: float, thresholds: tuple[float, float]) -> int:
    lower, upper = thresholds
    if value < lower:
        return 0
    if value < upper:
        return 1
    return 2


def expanding_window_segments(indices: Sequence[int], n_folds: int) -> List[Tuple[List[int], List[int]]]:
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if len(indices) < n_folds + 1:
        raise ValueError("Not enough indices to build expanding-window folds.")
    boundaries = []
    for fold_idx in range(n_folds + 1):
        start = int(round(fold_idx * len(indices) / (n_folds + 1)))
        end = int(round((fold_idx + 1) * len(indices) / (n_folds + 1)))
        boundaries.append(indices[start:end])
    folds: List[Tuple[List[int], List[int]]] = []
    for fold_idx in range(n_folds):
        train_indices: list[int] = []
        for segment in boundaries[: fold_idx + 1]:
            train_indices.extend(segment)
        val_indices = list(boundaries[fold_idx + 1])
        if not train_indices or not val_indices:
            raise ValueError("Encountered an empty train or validation segment.")
        folds.append((train_indices, val_indices))
    return folds


def flatten(items: Iterable[Iterable[int]]) -> List[int]:
    result: List[int] = []
    for group in items:
        result.extend(group)
    return result
