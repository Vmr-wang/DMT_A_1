from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .deps import require_dependency
from .evaluation import regression_metrics
from .utils import days_since_last_observation, long_gap_flags

RAW_TO_SAFE_VARIABLE = {
    "mood": "mood",
    "circumplex.arousal": "circumplex_arousal",
    "circumplex.valence": "circumplex_valence",
    "activity": "activity",
    "screen": "screen",
    "call": "call",
    "sms": "sms",
    "appCat.builtin": "appcat_builtin",
    "appCat.communication": "appcat_communication",
    "appCat.entertainment": "appcat_entertainment",
    "appCat.finance": "appcat_finance",
    "appCat.game": "appcat_game",
    "appCat.office": "appcat_office",
    "appCat.other": "appcat_other",
    "appCat.social": "appcat_social",
    "appCat.travel": "appcat_travel",
    "appCat.unknown": "appcat_unknown",
    "appCat.utilities": "appcat_utilities",
    "appCat.weather": "appcat_weather",
}

SELF_REPORT_BASES = ("mood", "circumplex_arousal", "circumplex_valence")
APP_USAGE_BASES = tuple(
    safe_name
    for raw_name, safe_name in RAW_TO_SAFE_VARIABLE.items()
    if raw_name.startswith("appCat.")
)
EVENT_SUM_BASES = ("screen", "call", "sms", *APP_USAGE_BASES)
DURATION_SUM_BASES = ("screen", *APP_USAGE_BASES)

SELF_REPORT_MEAN_COLS = [f"{name}_mean" for name in SELF_REPORT_BASES]
SELF_REPORT_COUNT_COLS = [f"{name}_count" for name in SELF_REPORT_BASES]
APP_SUM_COLS = [f"{name}_sum" for name in APP_USAGE_BASES]
EVENT_SUM_COLS = [f"{name}_sum" for name in EVENT_SUM_BASES]


def load_raw_events(csv_path):
    frame, _ = load_raw_events_with_audit(csv_path)
    return frame


def load_raw_events_with_audit(csv_path):
    pd = require_dependency("pandas")
    frame = pd.read_csv(
        csv_path,
        usecols=["id", "time", "variable", "value"],
        parse_dates=["time"],
        na_values=["NA", ""],
    )
    frame = frame.rename(columns={"id": "patient_id", "time": "timestamp"})
    num_rows_before_dedup = len(frame)
    frame = frame.drop_duplicates(subset=["patient_id", "timestamp", "variable"], keep="first")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    event_indicator_counts = {}
    warnings = []
    for variable_name in ("call", "sms"):
        invalid_mask = (
            frame["variable"].eq(variable_name)
            & frame["value"].notna()
            & frame["value"].ne(1.0)
        )
        invalid_count = int(invalid_mask.sum())
        event_indicator_counts[f"{variable_name}_non_one_count"] = invalid_count
        if invalid_count:
            warnings.append(
                f"Detected {invalid_count} {variable_name} rows with value != 1.0; kept unchanged for audit-only validation."
            )
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["safe_variable"] = frame["variable"].map(RAW_TO_SAFE_VARIABLE)
    frame = frame.sort_values(["patient_id", "timestamp", "variable"]).reset_index(drop=True)
    return frame, {
        "duplicate_rows_removed": int(num_rows_before_dedup - len(frame)),
        "event_indicator_non_one_counts": event_indicator_counts,
        "warnings": warnings,
    }


def build_daily_patient_table(csv_path):
    daily, _ = build_daily_patient_table_with_audit(csv_path)
    return daily


def build_daily_patient_table_with_audit(csv_path):
    pd = require_dependency("pandas")
    raw, raw_event_audit = load_raw_events_with_audit(csv_path)
    calendar = _build_patient_calendar(raw, pd)

    observed_any = raw.groupby(["patient_id", "date"]).size().rename("observed_event_count").to_frame()
    observed_any["observed_any"] = 1

    self_report = raw[raw["safe_variable"].isin(SELF_REPORT_BASES)].copy()
    self_report_means = self_report.pivot_table(
        index=["patient_id", "date"],
        columns="safe_variable",
        values="value",
        aggfunc="mean",
    )
    self_report_counts = self_report.pivot_table(
        index=["patient_id", "date"],
        columns="safe_variable",
        values="value",
        aggfunc="count",
    )
    self_report_means.columns = [f"{column}_mean" for column in self_report_means.columns]
    self_report_counts.columns = [f"{column}_count" for column in self_report_counts.columns]

    activity = raw[raw["safe_variable"] == "activity"].copy()
    activity_daily = activity.groupby(["patient_id", "date"])["value"].agg(
        activity_mean="mean",
        activity_std="std",
    )

    event_values = raw[raw["safe_variable"].isin(EVENT_SUM_BASES)].copy()
    event_daily = event_values.pivot_table(
        index=["patient_id", "date"],
        columns="safe_variable",
        values="value",
        aggfunc="sum",
    )
    event_daily.columns = [f"{column}_sum" for column in event_daily.columns]

    daily = (
        calendar.set_index(["patient_id", "date"])
        .join(observed_any, how="left")
        .join(self_report_means, how="left")
        .join(self_report_counts, how="left")
        .join(activity_daily, how="left")
        .join(event_daily, how="left")
        .reset_index()
        .sort_values(["patient_id", "date"])
        .reset_index(drop=True)
    )

    daily["observed_any"] = daily["observed_any"].fillna(0).astype(int)
    daily["observed_event_count"] = daily["observed_event_count"].fillna(0).astype(int)
    daily["is_weekend"] = daily["date"].dt.dayofweek.ge(5).astype(int)

    for column in SELF_REPORT_MEAN_COLS:
        daily[f"{column}_missing"] = daily[column].isna().astype(int)
    daily["activity_mean_missing"] = daily["activity_mean"].isna().astype(int)
    daily["activity_std_missing"] = daily["activity_mean"].isna().astype(int)
    for column in EVENT_SUM_COLS:
        daily[f"{column}_missing"] = daily[column].isna().astype(int)

    activity_observed_mask = daily["activity_mean"].notna() & daily["activity_std"].isna()
    daily.loc[activity_observed_mask, "activity_std"] = 0.0

    observed_mask = daily["observed_any"].eq(1)
    for column in EVENT_SUM_COLS:
        daily.loc[observed_mask & daily[column].isna(), column] = 0.0

    for column in SELF_REPORT_COUNT_COLS:
        daily[column] = daily[column].fillna(0).astype(int)

    daily["long_gap_flag"] = 0
    daily["days_since_last_mood"] = None
    daily["last_observed_mood"] = None

    for patient_id, patient_rows in daily.groupby("patient_id", sort=True):
        indices = patient_rows.index.tolist()
        observed_flags = [bool(value) for value in patient_rows["observed_any"].tolist()]
        mood_observed = [count > 0 for count in patient_rows["mood_count"].tolist()]

        patient_gap_flags = long_gap_flags(observed_flags, threshold=2)
        patient_days_since_mood = days_since_last_observation(mood_observed)
        patient_last_mood = patient_rows["mood_mean"].ffill().tolist()

        daily.loc[indices, "long_gap_flag"] = [int(value) for value in patient_gap_flags]
        daily.loc[indices, "days_since_last_mood"] = patient_days_since_mood
        daily.loc[indices, "last_observed_mood"] = patient_last_mood

    daily["days_since_last_mood"] = pd.to_numeric(daily["days_since_last_mood"], errors="coerce")
    daily["last_observed_mood"] = pd.to_numeric(daily["last_observed_mood"], errors="coerce")
    return daily, raw_event_audit


def apply_imputation(daily_frame, config: PipelineConfig, method: Optional[str] = None):
    pd = require_dependency("pandas")
    selected_method = method or config.selected_imputation_method
    if selected_method not in {"forward_fill", "ewma"}:
        raise ValueError(f"Unsupported imputation method: {selected_method}")

    result = daily_frame.copy()
    for imputation_method in ("forward_fill", "ewma"):
        for column in SELF_REPORT_MEAN_COLS:
            filled_parts = []
            imputed_parts = []
            for _, patient_rows in result.groupby("patient_id", sort=True):
                filled, imputed = _impute_self_report_series(
                    patient_rows[column],
                    patient_rows["long_gap_flag"],
                    pd=pd,
                    config=config,
                    method=imputation_method,
                )
                filled_parts.append(filled)
                imputed_parts.append(imputed)
            filled_series = pd.concat(filled_parts).sort_index()
            imputed_series = pd.concat(imputed_parts).sort_index()
            result[f"{column}_{imputation_method}"] = filled_series
            result[f"{column}_{imputation_method}_imputed"] = imputed_series.astype(int)

    for column in SELF_REPORT_MEAN_COLS:
        result[f"{column}_clean"] = result[f"{column}_{selected_method}"]
        result[f"{column}_imputed"] = result[f"{column}_{selected_method}_imputed"]
    return result


def backtest_imputation_strategies(daily_frame, config: PipelineConfig) -> dict[str, dict[str, dict[str, float]]]:
    pd = require_dependency("pandas")
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for column in SELF_REPORT_MEAN_COLS:
        masked_frame = daily_frame.copy()
        held_out_positions: dict[int, float] = {}
        for _, patient_rows in masked_frame.groupby("patient_id", sort=True):
            observed_indices = patient_rows.index[patient_rows[column].notna()].tolist()
            for held_out_idx in observed_indices[3::5]:
                held_out_positions[held_out_idx] = float(masked_frame.at[held_out_idx, column])
                masked_frame.at[held_out_idx, column] = float("nan")

        if not held_out_positions:
            continue

        column_result: dict[str, dict[str, float]] = {}
        for imputation_method in ("forward_fill", "ewma"):
            imputed = apply_imputation(masked_frame, config=config, method=imputation_method)
            y_true = [truth for _, truth in sorted(held_out_positions.items())]
            y_pred = [
                float(imputed.at[row_idx, f"{column}_clean"])
                for row_idx, _ in sorted(held_out_positions.items())
                if pd.notna(imputed.at[row_idx, f"{column}_clean"])
            ]
            y_eval_true = [
                truth
                for row_idx, truth in sorted(held_out_positions.items())
                if pd.notna(imputed.at[row_idx, f"{column}_clean"])
            ]
            metrics = regression_metrics(y_eval_true, y_pred) if y_pred else {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "residual_summary": {}}
            column_result[imputation_method] = {
                "num_masked": len(held_out_positions),
                "num_recovered": len(y_pred),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
            }
        summary[column] = column_result
    return summary


def _build_patient_calendar(raw_frame, pd):
    frames = []
    for patient_id, patient_rows in raw_frame.groupby("patient_id", sort=True):
        start_date = patient_rows["date"].min()
        end_date = patient_rows["date"].max()
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": patient_id,
                    "date": pd.date_range(start=start_date, end=end_date, freq="D"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _impute_self_report_series(series, long_gap_series, pd, config: PipelineConfig, method: str):
    values = series.tolist()
    long_gap_flags_list = [bool(flag) for flag in long_gap_series.tolist()]
    filled: list[float] = []
    imputed_flags: list[int] = []

    last_observed_value: Optional[float] = None
    short_gap_length = 0
    history: list[float] = []

    for value, is_long_gap in zip(values, long_gap_flags_list):
        if pd.notna(value):
            numeric_value = float(value)
            filled.append(numeric_value)
            imputed_flags.append(0)
            last_observed_value = numeric_value
            short_gap_length = 0
            history.append(numeric_value)
            history = history[-config.ewma_history :]
            continue

        if is_long_gap:
            filled.append(float("nan"))
            imputed_flags.append(0)
            last_observed_value = None
            short_gap_length = 0
            history.clear()
            continue

        if method == "forward_fill":
            short_gap_length += 1
            if last_observed_value is not None and short_gap_length <= config.max_ffill_gap:
                filled.append(last_observed_value)
                imputed_flags.append(1)
            else:
                filled.append(float("nan"))
                imputed_flags.append(0)
                if short_gap_length > config.max_ffill_gap:
                    last_observed_value = None
            continue

        recent_values = history[-config.ewma_history :]
        if recent_values:
            weights = [config.ewma_alpha ** (len(recent_values) - 1 - idx) for idx in range(len(recent_values))]
            denominator = sum(weights)
            weighted_average = sum(value * weight for value, weight in zip(recent_values, weights)) / denominator
            filled.append(float(weighted_average))
            imputed_flags.append(1)
        else:
            filled.append(float("nan"))
            imputed_flags.append(0)

    return (
        pd.Series(filled, index=series.index, dtype=float),
        pd.Series(imputed_flags, index=series.index, dtype=int),
    )
