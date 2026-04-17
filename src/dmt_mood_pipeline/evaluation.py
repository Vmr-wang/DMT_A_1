from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def residual_summary(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    residuals = [pred - true for true, pred in zip(y_true, y_pred)]
    mean_value = sum(residuals) / len(residuals)
    variance = sum((value - mean_value) ** 2 for value in residuals) / len(residuals)
    return {
        "mean": mean_value,
        "std": math.sqrt(variance),
        "min": min(residuals),
        "max": max(residuals),
    }


def regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, object]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "residual_summary": residual_summary(y_true, y_pred),
    }


def confusion_matrix(y_true: List[int], y_pred: List[int], labels: List[int]) -> List[List[int]]:
    table: List[List[int]] = []
    for true_label in labels:
        row: List[int] = []
        for pred_label in labels:
            row.append(sum(1 for a, b in zip(y_true, y_pred) if a == true_label and b == pred_label))
        table.append(row)
    return table


def balanced_accuracy(y_true: List[int], y_pred: List[int], labels: List[int]) -> float:
    recalls = []
    for label in labels:
        positives = [idx for idx, value in enumerate(y_true) if value == label]
        if not positives:
            continue
        correct = sum(1 for idx in positives if y_pred[idx] == label)
        recalls.append(correct / len(positives))
    return sum(recalls) / len(recalls)


def macro_f1(y_true: List[int], y_pred: List[int], labels: List[int]) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return sum(scores) / len(scores)


def classification_metrics(y_true: List[int], y_pred: List[int], labels: List[int]) -> Dict[str, object]:
    return {
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels),
        "true_class_counts": dict(Counter(y_true)),
        "predicted_class_counts": dict(Counter(y_pred)),
    }
