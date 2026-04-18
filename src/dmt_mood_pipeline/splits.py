from __future__ import annotations

from collections import defaultdict

from .utils import compute_outer_split_sizes, expanding_window_segments, flatten


def assign_outer_splits(metadata_frame, patient_col: str = "patient_id", date_col: str = "anchor_date",
                        train_ratio: float = 0.70, val_ratio: float = 0.15):
    frame = metadata_frame.sort_values([patient_col, date_col]).copy()
    split_labels: list[str] = []
    for _, patient_rows in frame.groupby(patient_col, sort=True):
        train_size, val_size, test_size = compute_outer_split_sizes(len(patient_rows), train_ratio, val_ratio)
        split_labels.extend(
            ["train"] * train_size +
            ["val"] * val_size +
            ["test"] * test_size
        )
    frame["split"] = split_labels
    return frame


def build_expanding_window_folds(train_metadata, patient_col: str = "patient_id", n_folds: int = 3) -> list[tuple[list[int], list[int]]]:
    patient_indices: dict[str, list[int]] = defaultdict(list)
    for row_idx, patient_id in zip(train_metadata.index.tolist(), train_metadata[patient_col].tolist()):
        patient_indices[patient_id].append(int(row_idx))

    grouped_folds: list[list[tuple[list[int], list[int]]]] = []
    for indices in patient_indices.values():
        grouped_folds.append(expanding_window_segments(indices, n_folds=n_folds))

    combined_folds: list[tuple[list[int], list[int]]] = []
    for fold_idx in range(n_folds):
        train_groups = [group[fold_idx][0] for group in grouped_folds]
        val_groups = [group[fold_idx][1] for group in grouped_folds]
        combined_folds.append((sorted(flatten(train_groups)), sorted(flatten(val_groups))))
    return combined_folds
