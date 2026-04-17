from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

from .data import APP_SUM_COLS, DURATION_SUM_BASES
from .deps import require_dependency
from .utils import rolling_linear_slope

CORE_TABULAR_SIGNAL_COLS = [
    "mood_mean_clean",
    "activity_mean",
    "screen_sum",
    "circumplex_arousal_mean_clean",
    "circumplex_valence_mean_clean",
    "call_sum",
    "sms_sum",
]

SELF_REPORT_CLEAN_TO_RAW = {
    "mood_mean_clean": "mood_mean",
    "circumplex_arousal_mean_clean": "circumplex_arousal_mean",
    "circumplex_valence_mean_clean": "circumplex_valence_mean",
}

COARSE_APP_GROUPS = {
    "appcat_social_comm": ("appcat_social_sum", "appcat_communication_sum"),
    "appcat_leisure": ("appcat_entertainment_sum", "appcat_game_sum"),
    "appcat_productivity": ("appcat_office_sum", "appcat_finance_sum"),
    "appcat_misc": (
        "appcat_other_sum",
        "appcat_travel_sum",
        "appcat_unknown_sum",
        "appcat_utilities_sum",
        "appcat_weather_sum",
    ),
}


@dataclass
class TabularWindowDataset:
    frame: object
    feature_columns: list[str]
    metadata_columns: list[str]
    target_column: str


@dataclass
class SequenceWindowDataset:
    metadata: object
    sequences: object
    feature_names: list[str]
    target_column: str


def build_window_datasets(daily_frame, window_size: int = 7) -> tuple[TabularWindowDataset, SequenceWindowDataset]:
    pd = require_dependency("pandas")
    np = require_dependency("numpy")

    transformed_daily = daily_frame.copy()
    for base_name in DURATION_SUM_BASES:
        column = f"{base_name}_sum"
        transformed_daily[column] = np.log1p(transformed_daily[column].clip(lower=0))

    sequence_feature_names = _sequence_feature_columns()
    tabular_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    sequences: list[object] = []
    sample_id = 0

    for patient_id, patient_rows in daily_frame.groupby("patient_id", sort=True):
        patient_rows = patient_rows.sort_values("date").reset_index(drop=True)
        patient_rows_transformed = transformed_daily[transformed_daily["patient_id"] == patient_id].sort_values("date").reset_index(drop=True)

        for anchor_idx in range(window_size - 1, len(patient_rows) - 1):
            target_value = patient_rows.loc[anchor_idx + 1, "mood_mean"]
            if pd.isna(target_value):
                continue

            raw_window = patient_rows.iloc[anchor_idx - window_size + 1 : anchor_idx + 1].copy()
            transformed_window = patient_rows_transformed.iloc[anchor_idx - window_size + 1 : anchor_idx + 1].copy()
            anchor_row = patient_rows.iloc[anchor_idx]
            target_row = patient_rows.iloc[anchor_idx + 1]

            tabular_row = {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "anchor_date": anchor_row["date"],
                "target_date": target_row["date"],
                "baseline_mood_prediction": anchor_row["last_observed_mood"],
                "target_next_day_mood_mean": float(target_value),
            }
            tabular_row.update(_build_tabular_features(raw_window, transformed_window, np))
            tabular_rows.append(tabular_row)

            sequence_rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "anchor_date": anchor_row["date"],
                    "target_date": target_row["date"],
                    "baseline_mood_prediction": anchor_row["last_observed_mood"],
                    "target_next_day_mood_mean": float(target_value),
                }
            )
            sequences.append(
                transformed_window[sequence_feature_names].to_numpy(dtype=float)
            )
            sample_id += 1

    tabular_frame = pd.DataFrame(tabular_rows)
    metadata_columns = [
        "sample_id",
        "patient_id",
        "anchor_date",
        "target_date",
        "baseline_mood_prediction",
    ]
    feature_columns = [
        column
        for column in tabular_frame.columns
        if column not in metadata_columns and column != "target_next_day_mood_mean"
    ]
    sequence_metadata = pd.DataFrame(sequence_rows)
    sequence_array = np.asarray(sequences, dtype=float)

    return (
        TabularWindowDataset(
            frame=tabular_frame,
            feature_columns=feature_columns,
            metadata_columns=metadata_columns,
            target_column="target_next_day_mood_mean",
        ),
        SequenceWindowDataset(
            metadata=sequence_metadata,
            sequences=sequence_array,
            feature_names=sequence_feature_names,
            target_column="target_next_day_mood_mean",
        ),
    )


def _build_tabular_features(raw_window, transformed_window, np) -> dict[str, float]:
    features: dict[str, float] = {}

    for column in CORE_TABULAR_SIGNAL_COLS:
        window_values = transformed_window[column]
        tail_3 = window_values.tail(3)
        tail_7 = window_values.tail(7)
        features[f"{column}_last"] = _safe_scalar(window_values.iloc[-1])
        features[f"{column}_3d_mean"] = _safe_scalar(tail_3.mean(skipna=True))
        features[f"{column}_7d_mean"] = _safe_scalar(tail_7.mean(skipna=True))
        features[f"{column}_7d_std"] = _safe_scalar(tail_7.std(skipna=True, ddof=0))
        features[f"{column}_7d_slope"] = _safe_scalar(
            rolling_linear_slope([_as_optional_float(value) for value in tail_7.tolist()])
        )
        features[f"{column}_missing_ratio_7d"] = _safe_scalar(
            raw_window[_missing_column_name(column)].mean(skipna=True)
        )
        imputed_column = _imputed_column_name(column)
        if imputed_column in raw_window.columns:
            features[f"{column}_imputed_ratio_7d"] = _safe_scalar(raw_window[imputed_column].mean(skipna=True))

    raw_app_sums = raw_window[APP_SUM_COLS].sum(skipna=True)
    total_app_sum = float(raw_app_sums.sum())
    for column in APP_SUM_COLS:
        raw_sum = float(raw_app_sums[column]) if column in raw_app_sums else 0.0
        features[f"{column}_7d_sum"] = float(np.log1p(max(raw_sum, 0.0)))
        features[f"{column}_7d_share"] = raw_sum / total_app_sum if total_app_sum > 0 else 0.0
        features[f"{column}_missing_ratio_7d"] = _safe_scalar(raw_window[f"{column}_missing"].mean(skipna=True))
    features.update(coarse_app_group_features(raw_app_sums.to_dict(), total_app_sum))

    features["mood_observation_density_7d"] = 1.0 - _safe_scalar(raw_window["mood_mean_missing"].mean(skipna=True))
    features["observed_any_ratio_7d"] = _safe_scalar(raw_window["observed_any"].mean(skipna=True))
    features["long_gap_ratio_7d"] = _safe_scalar(raw_window["long_gap_flag"].mean(skipna=True))
    features["weekend_ratio_7d"] = _safe_scalar(raw_window["is_weekend"].mean(skipna=True))
    features["days_since_last_mood_anchor"] = _safe_scalar(raw_window["days_since_last_mood"].iloc[-1])
    return features


def _sequence_feature_columns() -> list[str]:
    base_columns = [
        "mood_mean_clean",
        "circumplex_arousal_mean_clean",
        "circumplex_valence_mean_clean",
        "activity_mean",
        "activity_std",
        "screen_sum",
        "call_sum",
        "sms_sum",
        *APP_SUM_COLS,
        "observed_any",
        "is_weekend",
        "long_gap_flag",
        "days_since_last_mood",
    ]
    mask_columns = [
        "mood_mean_missing",
        "circumplex_arousal_mean_missing",
        "circumplex_valence_mean_missing",
        "activity_mean_missing",
        "activity_std_missing",
        "screen_sum_missing",
        "call_sum_missing",
        "sms_sum_missing",
        *[f"{column}_missing" for column in APP_SUM_COLS],
        "mood_mean_imputed",
        "circumplex_arousal_mean_imputed",
        "circumplex_valence_mean_imputed",
    ]
    return base_columns + mask_columns


def coarse_app_group_features(raw_app_sums: Mapping[str, float], total_app_sum: float) -> dict[str, float]:
    features: dict[str, float] = {}
    for group_name, columns in COARSE_APP_GROUPS.items():
        raw_sum = sum(float(raw_app_sums.get(column, 0.0)) for column in columns)
        features[f"{group_name}_7d_sum"] = math.log1p(max(raw_sum, 0.0))
        features[f"{group_name}_7d_share"] = raw_sum / total_app_sum if total_app_sum > 0 else 0.0
    return features


def _missing_column_name(feature_column: str) -> str:
    if feature_column in SELF_REPORT_CLEAN_TO_RAW:
        return f"{SELF_REPORT_CLEAN_TO_RAW[feature_column]}_missing"
    return f"{feature_column}_missing"


def _imputed_column_name(feature_column: str) -> str:
    if feature_column in SELF_REPORT_CLEAN_TO_RAW:
        return f"{SELF_REPORT_CLEAN_TO_RAW[feature_column]}_imputed"
    return ""


def _as_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _safe_scalar(value) -> float:
    numeric = _as_optional_float(value)
    return float("nan") if numeric is None else float(numeric)
