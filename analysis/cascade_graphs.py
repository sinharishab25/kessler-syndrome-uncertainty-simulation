"""Cascade time-series graphs regenerated only from saved table rows."""

from collections import defaultdict
from pathlib import Path
import csv
import statistics

import matplotlib.pyplot as plt

import config


GRAPH_SPECS = [
    (
        "N_active",
        "Active Population",
        "Active objects",
        "active objects",
        "CASCADE_ACTIVE_POPULATION_GRAPH_FILENAME",
    ),
    (
        "N_debris",
        "Debris Population",
        "Debris objects",
        "debris objects",
        "CASCADE_DEBRIS_POPULATION_GRAPH_FILENAME",
    ),
    (
        "collision_rate_lambda_per_model_year",
        "Collision Rate",
        "Collisions per model year",
        "collisions/model year",
        "CASCADE_COLLISION_RATE_GRAPH_FILENAME",
    ),
    (
        "cumulative_collisions",
        "Cumulative Collisions",
        "Cumulative true collisions",
        "collisions",
        "CASCADE_CUMULATIVE_COLLISIONS_GRAPH_FILENAME",
    ),
    (
        "maximum_cascade_depth_to_time",
        "Maximum Cascade Depth",
        "Maximum generation",
        "generation",
        "CASCADE_MAX_DEPTH_GRAPH_FILENAME",
    ),
    (
        "rolling_collision_rate_per_model_year",
        "Rolling Collision Rate",
        "Collisions per model year",
        "collisions/model year",
        "CASCADE_ROLLING_RATE_GRAPH_FILENAME",
    ),
    (
        "debris_growth_rate_per_model_year",
        "Debris Growth Rate",
        "Debris objects per model year",
        "debris/model year",
        "CASCADE_DEBRIS_GROWTH_RATE_GRAPH_FILENAME",
    ),
]


def plot_cascade_time_series_graphs(time_bin_csv_path: Path) -> list[Path]:
    """Regenerate cascade graphs from the saved time-bin CSV only."""
    rows = _read_csv_rows(time_bin_csv_path)
    output_paths = []
    for column_name, title, y_label, unit_label, filename_setting in GRAPH_SPECS:
        output_paths.append(
            _plot_metric(
                rows,
                column_name,
                title,
                y_label,
                unit_label,
                getattr(config, filename_setting),
            )
        )
    return output_paths


def _plot_metric(
    rows: list[dict[str, str]],
    column_name: str,
    title: str,
    y_label: str,
    unit_label: str,
    output_filename: str,
) -> Path:
    rows_by_run = defaultdict(list)
    for row in rows:
        rows_by_run[row["run_uid"]].append(row)

    run_count = len(rows_by_run)
    output_path = Path(config.OUTPUT_DIRECTORY) / output_filename
    figure, axes = plt.subplots()

    if run_count == 1:
        run_rows = next(iter(rows_by_run.values()), [])
        x_values = [float(row["sample_time_model_years"]) for row in run_rows]
        y_values = [float(row[column_name]) for row in run_rows]
        axes.plot(x_values, y_values, marker="o", label="run")
    else:
        _plot_individual_trajectories(axes, rows_by_run, column_name)
        grouped_by_time = defaultdict(list)
        for row in rows:
            grouped_by_time[float(row["sample_time_model_years"])].append(
                float(row[column_name])
            )
        x_values = sorted(grouped_by_time)
        medians = [statistics.median(grouped_by_time[x]) for x in x_values]
        q1_values = [_quantile(grouped_by_time[x], 0.25) for x in x_values]
        q3_values = [_quantile(grouped_by_time[x], 0.75) for x in x_values]
        axes.plot(x_values, medians, color="black", label="median")
        axes.fill_between(x_values, q1_values, q3_values, alpha=0.25, label="IQR")

    bin_width_years = config.model_time_config_value_to_years(
        config.CASCADE_TIME_BIN_WIDTH_MODEL_TIME
    )
    rolling_window_years = config.model_time_config_value_to_years(
        config.CASCADE_ROLLING_WINDOW_MODEL_TIME
    )
    axes.set_title(
        f"{title} (bin {bin_width_years:g} model years, "
        f"rolling W={rolling_window_years:g}, runs={run_count})"
    )
    axes.set_xlabel("Model time (years)")
    axes.set_ylabel(f"{y_label} ({unit_label})")
    axes.grid(True, alpha=0.3)
    if rows_by_run:
        axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_individual_trajectories(axes, rows_by_run, column_name: str) -> None:
    if not config.CASCADE_PLOT_INDIVIDUAL_TRAJECTORIES:
        return
    limit = int(config.CASCADE_MAX_INDIVIDUAL_TRAJECTORIES)
    for run_uid in sorted(rows_by_run)[:limit]:
        rows = rows_by_run[run_uid]
        axes.plot(
            [float(row["sample_time_model_years"]) for row in rows],
            [float(row[column_name]) for row in rows],
            alpha=0.25,
            linewidth=1.0,
        )


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))
