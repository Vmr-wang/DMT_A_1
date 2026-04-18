from __future__ import annotations

from pathlib import Path

from .data import load_raw_events_with_audit
from .deps import require_dependency


def run_eda(csv_path, output_dir: Path) -> dict[str, object]:
    plt = require_dependency("matplotlib.pyplot", "matplotlib")
    mdates = require_dependency("matplotlib.dates", "matplotlib")
    pd = require_dependency("pandas")

    raw_events, raw_event_audit = load_raw_events_with_audit(csv_path)
    output_dir = Path(output_dir)
    tables_dir = output_dir / "task_1a_tables"
    plots_dir = output_dir / "task_1a_plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_overview = _dataset_overview_table(raw_events, pd)
    user_summary = _user_summary_table(raw_events, pd)
    dataset_overview.to_csv(tables_dir / "table_dataset_overview.csv", index=False)
    user_summary.to_csv(tables_dir / "table_user_summary.csv", index=False)

    _plot_mood_distribution(raw_events, plt, plots_dir / "plot1_mood_distribution.png")
    _plot_mood_over_time(raw_events, plt, mdates, plots_dir / "plot2_mood_over_time.png")
    _plot_key_variable_distributions(raw_events, plt, plots_dir / "plot3_key_variable_distributions.png")
    _plot_variable_coverage_heatmap(raw_events, plt, plots_dir / "plot4_variable_coverage_heatmap.png")
    _plot_daily_correlation_matrix(raw_events, plt, plots_dir / "plot5_correlation_matrix.png")
    _plot_mood_temporal_patterns(raw_events, plt, plots_dir / "plot6_mood_temporal_patterns.png")
    _plot_records_per_variable(raw_events, plt, plots_dir / "plot7_records_per_variable.png")
    _plot_screen_and_activity_patterns(raw_events, plt, mdates, plots_dir / "plot8_screen_activity_patterns.png")

    return {
        "output_dir": str(output_dir),
        "task": "task_1a",
        "num_tables": 2,
        "num_plots": 8,
        "data_quality": raw_event_audit,
    }


def _dataset_overview_table(raw_events, pd):
    overview_rows = []
    for variable, variable_rows in raw_events.groupby("variable", sort=True):
        overview_rows.append(
            {
                "Variable": variable,
                "Count": len(variable_rows),
                "Missing": int(variable_rows["value"].isna().sum()),
                "Min": variable_rows["value"].min(),
                "Max": variable_rows["value"].max(),
                "Mean": variable_rows["value"].mean(),
                "Std": variable_rows["value"].std(),
                "Median": variable_rows["value"].median(),
                "Users": int(variable_rows["patient_id"].nunique()),
            }
        )
    return pd.DataFrame(overview_rows).round(3)


def _user_summary_table(raw_events, pd):
    mood_events = raw_events[raw_events["variable"] == "mood"]
    summary_rows = []
    for patient_id, patient_rows in raw_events.groupby("patient_id", sort=True):
        patient_mood = mood_events[mood_events["patient_id"] == patient_id]
        summary_rows.append(
            {
                "User": patient_id,
                "Date Range": f"{patient_rows['date'].min().date()} ~ {patient_rows['date'].max().date()}",
                "Total Days": int(patient_rows["date"].nunique()),
                "Mood Days": int(patient_mood["date"].nunique()),
                "Mood Records": int(len(patient_mood)),
                "Mood Mean": patient_mood["value"].mean(),
                "Mood Std": patient_mood["value"].std(),
                "Total Records": int(len(patient_rows)),
            }
        )
    return pd.DataFrame(summary_rows).round(2)


def _plot_mood_distribution(raw_events, plt, output_path: Path) -> None:
    mood_events = raw_events[raw_events["variable"] == "mood"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mood_values = mood_events["value"].dropna()
    axes[0].hist(mood_values, bins=range(1, 12), edgecolor="black", alpha=0.7, color="steelblue", align="left")
    axes[0].set_xlabel("Mood Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Overall Mood Distribution")
    axes[0].set_xticks(range(1, 11))
    if not mood_values.empty:
        mood_mean = float(mood_values.mean())
        axes[0].axvline(mood_mean, color="red", linestyle="--", label=f"Mean={mood_mean:.2f}")
        axes[0].legend()

    patient_order = mood_events.groupby("patient_id")["value"].mean().sort_values().index.tolist()
    patient_values = [mood_events.loc[mood_events["patient_id"] == patient_id, "value"].dropna().tolist() for patient_id in patient_order]
    axes[1].boxplot(patient_values, labels=patient_order, patch_artist=True)
    axes[1].set_xlabel("User ID")
    axes[1].set_ylabel("Mood Score")
    axes[1].set_title("Mood Distribution per User")
    axes[1].tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_mood_over_time(raw_events, plt, mdates, output_path: Path) -> None:
    mood_events = raw_events[raw_events["variable"] == "mood"]
    top_users = mood_events.groupby("patient_id").size().nlargest(9).index.tolist()
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for axis, patient_id in zip(axes.flatten(), top_users):
        patient_mood = mood_events[mood_events["patient_id"] == patient_id]
        daily_mood = patient_mood.groupby("date")["value"].mean()
        axis.plot(daily_mood.index, daily_mood.values, "o-", markersize=2, alpha=0.7, linewidth=1)
        axis.set_title(f"{patient_id} (n={len(patient_mood)})")
        axis.set_ylim(0, 11)
        axis.set_ylabel("Mood")
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axis.tick_params(axis="x", rotation=45)

    for axis in axes.flatten()[len(top_users):]:
        axis.axis("off")

    fig.suptitle("Daily Average Mood Over Time (Top 9 Users by # Records)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_key_variable_distributions(raw_events, plt, output_path: Path) -> None:
    key_variables = ["mood", "circumplex.arousal", "circumplex.valence", "activity", "screen"]
    fig, axes = plt.subplots(1, len(key_variables), figsize=(18, 4))

    for axis, variable in zip(axes, key_variables):
        values = raw_events.loc[(raw_events["variable"] == variable) & raw_events["value"].notna(), "value"]
        if variable == "screen" and not values.empty:
            values = values[values < values.quantile(0.99)]
        axis.hist(values, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
        axis.set_title(variable)
        axis.set_xlabel("Value")
        axis.set_ylabel("Frequency")

    fig.suptitle("Distribution of Key Variables", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_variable_coverage_heatmap(raw_events, plt, output_path: Path) -> None:
    coverage = {}
    patient_ids = sorted(raw_events["patient_id"].unique().tolist())
    variables = sorted(raw_events["variable"].unique().tolist())

    for patient_id in patient_ids:
        patient_rows = raw_events[raw_events["patient_id"] == patient_id]
        total_days = max(1, int(patient_rows["date"].nunique()))
        coverage[patient_id] = []
        for variable in variables:
            days_with_variable = patient_rows.loc[patient_rows["variable"] == variable, "date"].nunique()
            coverage[patient_id].append((float(days_with_variable) / float(total_days)) * 100.0)

    matrix = [[coverage[patient_id][idx] for patient_id in patient_ids] for idx in range(len(variables))]
    fig, ax = plt.subplots(figsize=(16, 8))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_title("Variable Coverage per User (% of Days with Data)")
    ax.set_xlabel("User ID")
    ax.set_ylabel("Variable")
    ax.set_xticks(range(len(patient_ids)))
    ax.set_xticklabels(patient_ids, rotation=90)
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables)
    fig.colorbar(image, ax=ax, label="Coverage (%)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_daily_correlation_matrix(raw_events, plt, output_path: Path) -> None:
    daily_pivot = raw_events.groupby(["patient_id", "date", "variable"])["value"].mean().reset_index()
    daily_wide = daily_pivot.pivot_table(index=["patient_id", "date"], columns="variable", values="value")
    core_variables = [
        "mood",
        "circumplex.arousal",
        "circumplex.valence",
        "activity",
        "screen",
        "appCat.communication",
        "appCat.social",
        "appCat.entertainment",
    ]
    available_variables = [variable for variable in core_variables if variable in daily_wide.columns]
    correlation = daily_wide[available_variables].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlation.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix of Daily Aggregated Variables")
    ax.set_xticks(range(len(available_variables)))
    ax.set_xticklabels(available_variables, rotation=45, ha="right")
    ax.set_yticks(range(len(available_variables)))
    ax.set_yticklabels(available_variables)
    for row_idx, row in enumerate(correlation.values):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_mood_temporal_patterns(raw_events, plt, output_path: Path) -> None:
    mood_events = raw_events[raw_events["variable"] == "mood"].copy()
    mood_events["dayofweek"] = mood_events["timestamp"].dt.dayofweek
    mood_events["hour"] = mood_events["timestamp"].dt.hour
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    by_day = [mood_events.loc[mood_events["dayofweek"] == day_idx, "value"].dropna().tolist() for day_idx in range(7)]
    axes[0].boxplot(by_day, labels=day_names, patch_artist=True)
    axes[0].set_xlabel("Day of Week")
    axes[0].set_ylabel("Mood Score")
    axes[0].set_title("Mood Distribution by Day of Week")

    hourly_stats = mood_events.groupby("hour")["value"].agg(["mean", "std"])
    axes[1].errorbar(hourly_stats.index, hourly_stats["mean"], yerr=hourly_stats["std"], fmt="o-", capsize=3, color="steelblue")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Mood Score")
    axes[1].set_title("Mood by Time of Day (Mean ± Std)")
    axes[1].set_xticks(range(0, 24, 2))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_records_per_variable(raw_events, plt, output_path: Path) -> None:
    record_counts = raw_events.groupby("variable").size().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(record_counts.index.tolist(), record_counts.values.tolist(), color="steelblue", edgecolor="black")
    ax.set_xlabel("Number of Records (log scale)")
    ax.set_title("Number of Records per Variable")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_screen_and_activity_patterns(raw_events, plt, mdates, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    screen_events = raw_events[raw_events["variable"] == "screen"]
    screen_daily = screen_events.groupby(["patient_id", "date"])["value"].sum().reset_index()
    screen_average = screen_daily.groupby("date")["value"].mean()
    axes[0].plot(screen_average.index, screen_average.values, "-", alpha=0.7, color="orange", linewidth=1)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Avg Daily Screen Time (s)")
    axes[0].set_title("Average Daily Screen Time Across Users")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[0].tick_params(axis="x", rotation=45)

    activity_values = raw_events.loc[raw_events["variable"] == "activity", "value"].dropna()
    axes[1].hist(activity_values, bins=50, edgecolor="black", alpha=0.7, color="green")
    axes[1].set_xlabel("Activity Level")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Activity Score Distribution")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
