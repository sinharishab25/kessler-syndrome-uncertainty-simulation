"""Experiment-level analysis regenerated from saved v1.5.15 run rows."""

from collections import defaultdict
from pathlib import Path
import csv
import math
import statistics

import matplotlib.pyplot as plt

import config


PROBABILITY_METRICS = (
    "precision",
    "recall",
    "true_cascade_probability",
    "observed_cascade_probability",
    "false_cascade_probability",
    "missed_cascade_probability",
)


def generate_experiment_analysis_outputs(raw_csv_path: Path) -> list[Path]:
    """Write experiment summaries and graphs using only saved raw run rows."""
    rows = _read_csv_rows(raw_csv_path)
    probability_rows = build_probability_summary_rows(rows)
    onset_rows = build_onset_summary_rows(rows)
    diagnostic_rows = build_diagnostic_rows(rows)

    output_directory = Path(config.OUTPUT_DIRECTORY)
    probability_path = output_directory / config.EXPERIMENT_ANALYSIS_SUMMARY_FILENAME
    onset_path = output_directory / config.EXPERIMENT_ANALYSIS_ONSET_SUMMARY_FILENAME
    diagnostic_path = output_directory / config.EXPERIMENT_ANALYSIS_DIAGNOSTICS_FILENAME

    _write_csv(probability_path, probability_rows)
    _write_csv(onset_path, onset_rows)
    _write_csv(diagnostic_path, diagnostic_rows)

    graph_paths = [
        plot_precision_recall(probability_rows),
        plot_cascade_probabilities(probability_rows),
        plot_onset_summary(onset_rows),
        plot_onset_bias_distribution(rows),
        plot_false_missed_probabilities(probability_rows),
        plot_runtime_diagnostics(diagnostic_rows),
        plot_failure_cap_diagnostics(diagnostic_rows),
    ]
    if _supports_survival_curve(rows):
        graph_paths.append(plot_survival_onset_curve(rows))
    return [probability_path, onset_path, diagnostic_path, *graph_paths]


def build_probability_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Return probability metrics by N0 and sigma with Wilson intervals."""
    output_rows = []
    for key, group_rows in _successful_groups(rows).items():
        population_value, sigma_km = key
        counts = _cascade_counts(group_rows)
        metric_specs = [
            ("precision", counts["TP"], counts["TP"] + counts["FP"]),
            ("recall", counts["TP"], counts["TP"] + counts["FN"]),
            (
                "true_cascade_probability",
                counts["TP"] + counts["FN"],
                len(group_rows),
            ),
            (
                "observed_cascade_probability",
                counts["TP"] + counts["FP"],
                len(group_rows),
            ),
            ("false_cascade_probability", counts["FP"], len(group_rows)),
            ("missed_cascade_probability", counts["FN"], len(group_rows)),
        ]
        for metric, numerator, denominator in metric_specs:
            lower, upper = _wilson_interval(numerator, denominator)
            output_rows.append(
                {
                    "population_value": population_value,
                    "sigma_km": sigma_km,
                    "metric": metric,
                    "numerator": numerator,
                    "denominator": denominator,
                    "estimate": (
                        numerator / denominator if denominator > 0 else None
                    ),
                    "wilson_lower": lower,
                    "wilson_upper": upper,
                }
            )
    return output_rows


def build_onset_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Return onset summaries while retaining censored run counts."""
    output_rows = []
    for key, group_rows in _successful_groups(rows).items():
        population_value, sigma_km = key
        for layer, column in (
            ("true", "true_cascade_onset_time_model_years"),
            ("observed", "observed_cascade_onset_time_model_years"),
        ):
            detected_times = [_optional_float(row[column]) for row in group_rows]
            detected_times = [value for value in detected_times if value is not None]
            denominator = len(group_rows)
            detected_count = len(detected_times)
            output_rows.append(
                {
                    "population_value": population_value,
                    "sigma_km": sigma_km,
                    "onset_layer": layer,
                    "detected_count": detected_count,
                    "denominator": denominator,
                    "detection_proportion": (
                        detected_count / denominator if denominator else None
                    ),
                    "right_censored_count": denominator - detected_count,
                    "median_onset_model_years": _median_or_none(detected_times),
                    "iqr_low_model_years": _quantile_or_none(detected_times, 0.25),
                    "iqr_high_model_years": _quantile_or_none(detected_times, 0.75),
                }
            )
    return output_rows


def build_diagnostic_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Return runtime, failure, and cap-termination diagnostics by condition."""
    output_rows = []
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["population_value"]), float(row["position_uncertainty_sigma_km"]))].append(row)

    for (population_value, sigma_km), group_rows in sorted(groups.items()):
        runtimes = [
            float(row["execution_time_seconds"])
            for row in group_rows
            if row.get("execution_time_seconds", "") not in ("", None)
        ]
        failures = [row for row in group_rows if row.get("status") != "success"]
        caps = [
            row
            for row in group_rows
            if _to_bool(row.get("cap_termination", ""))
            or "cap" in row.get("cascade_termination_reason", "").lower()
        ]
        output_rows.append(
            {
                "population_value": population_value,
                "sigma_km": sigma_km,
                "run_count": len(group_rows),
                "failed_run_count": len(failures),
                "failure_probability": len(failures) / len(group_rows),
                "cap_termination_count": len(caps),
                "cap_termination_probability": len(caps) / len(group_rows),
                "median_runtime_seconds": _median_or_none(runtimes),
                "total_runtime_seconds": sum(runtimes),
                "estimated_runtime_seconds": sum(
                    _optional_float(row.get("estimated_runtime_seconds", ""))
                    or 0.0
                    for row in group_rows
                ),
                "estimated_output_size_bytes": sum(
                    int(float(row.get("estimated_output_size_bytes", "0") or 0))
                    for row in group_rows
                ),
            }
        )
    return output_rows


def plot_precision_recall(rows: list[dict[str, object]]) -> Path:
    return _plot_probability_metrics(
        rows,
        ("precision", "recall"),
        "Cascade Precision And Recall By N0",
        config.CASCADE_PRECISION_RECALL_GRAPH_FILENAME,
    )


def plot_cascade_probabilities(rows: list[dict[str, object]]) -> Path:
    return _plot_probability_metrics(
        rows,
        ("true_cascade_probability", "observed_cascade_probability"),
        "True And Observed Cascade Probability By N0",
        config.CASCADE_PROBABILITY_GRAPH_FILENAME,
    )


def plot_false_missed_probabilities(rows: list[dict[str, object]]) -> Path:
    return _plot_probability_metrics(
        rows,
        ("false_cascade_probability", "missed_cascade_probability"),
        "False And Missed Cascade Probability By N0",
        config.CASCADE_FALSE_MISSED_GRAPH_FILENAME,
    )


def plot_onset_summary(rows: list[dict[str, object]]) -> Path:
    output_path = Path(config.OUTPUT_DIRECTORY) / config.CASCADE_ONSET_SUMMARY_GRAPH_FILENAME
    figure, axes = plt.subplots()
    for (sigma_km, layer), group_rows in _group_summary_rows(rows, "onset_layer").items():
        valid_rows = [
            row
            for row in group_rows
            if row["median_onset_model_years"] not in ("", None)
        ]
        if not valid_rows:
            continue
        x_values = [int(row["population_value"]) for row in valid_rows]
        y_values = [float(row["median_onset_model_years"]) for row in valid_rows]
        axes.plot(x_values, y_values, marker="o", label=f"{layer}, sigma={sigma_km:g}")
    axes.set_title("Conditional Median Onset (detected runs only; proportions in CSV)")
    axes.set_xlabel("Initial population N0")
    axes.set_ylabel("Median onset time (model years)")
    axes.grid(True, alpha=0.3)
    if axes.lines:
        axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_onset_bias_distribution(rows: list[dict[str, str]]) -> Path:
    output_path = Path(config.OUTPUT_DIRECTORY) / config.CASCADE_ONSET_BIAS_GRAPH_FILENAME
    values_by_sigma = defaultdict(list)
    for row in rows:
        value = _optional_float(row.get("onset_time_bias_model_years", ""))
        if value is not None:
            values_by_sigma[float(row["position_uncertainty_sigma_km"])].append(value)

    figure, axes = plt.subplots()
    if values_by_sigma:
        labels = [f"sigma={sigma:g}" for sigma in sorted(values_by_sigma)]
        axes.boxplot(
            [values_by_sigma[sigma] for sigma in sorted(values_by_sigma)],
            tick_labels=labels,
        )
    axes.set_title("Observed Minus True Cascade Onset Bias")
    axes.set_xlabel("Uncertainty sigma (km)")
    axes.set_ylabel("Onset bias (model years)")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_runtime_diagnostics(rows: list[dict[str, object]]) -> Path:
    output_path = Path(config.OUTPUT_DIRECTORY) / config.EXPERIMENT_RUNTIME_DIAGNOSTIC_GRAPH_FILENAME
    figure, axes = plt.subplots()
    for sigma_km, group_rows in _diagnostic_rows_by_sigma(rows).items():
        axes.plot(
            [int(row["population_value"]) for row in group_rows],
            [float(row["median_runtime_seconds"] or 0.0) for row in group_rows],
            marker="o",
            label=f"sigma={sigma_km:g}",
        )
    axes.set_title("Median Runtime Diagnostics")
    axes.set_xlabel("Initial population N0")
    axes.set_ylabel("Runtime (seconds)")
    axes.grid(True, alpha=0.3)
    if axes.lines:
        axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_failure_cap_diagnostics(rows: list[dict[str, object]]) -> Path:
    output_path = Path(config.OUTPUT_DIRECTORY) / config.EXPERIMENT_FAILURE_CAP_DIAGNOSTIC_GRAPH_FILENAME
    figure, axes = plt.subplots()
    for sigma_km, group_rows in _diagnostic_rows_by_sigma(rows).items():
        axes.plot(
            [int(row["population_value"]) for row in group_rows],
            [float(row["failure_probability"]) for row in group_rows],
            marker="o",
            label=f"failure sigma={sigma_km:g}",
        )
        axes.plot(
            [int(row["population_value"]) for row in group_rows],
            [float(row["cap_termination_probability"]) for row in group_rows],
            marker="x",
            linestyle="--",
            label=f"cap sigma={sigma_km:g}",
        )
    axes.set_title("Failure And Cap-Termination Diagnostics")
    axes.set_xlabel("Initial population N0")
    axes.set_ylabel("Probability")
    axes.set_ylim(0.0, 1.0)
    axes.grid(True, alpha=0.3)
    if axes.lines:
        axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_survival_onset_curve(rows: list[dict[str, str]]) -> Path:
    output_path = Path(config.OUTPUT_DIRECTORY) / "cascade_onset_survival_curve_v1_5_15.png"
    figure, axes = plt.subplots()
    successful_rows = [row for row in rows if row.get("status") == "success"]
    groups = defaultdict(list)
    for row in successful_rows:
        groups[(int(row["population_value"]), float(row["position_uncertainty_sigma_km"]))].append(row)
    for (population_value, sigma_km), group_rows in sorted(groups.items()):
        horizon = max(float(row["actual_duration_model_years"] or 0.0) for row in group_rows)
        onset_values = [
            _optional_float(row["observed_cascade_onset_time_model_years"])
            for row in group_rows
        ]
        times = sorted(value for value in onset_values if value is not None)
        x_values = [0.0, *times, horizon]
        y_values = []
        for time_value in x_values:
            detected_by_time = sum(1 for onset in times if onset <= time_value)
            y_values.append(1.0 - detected_by_time / len(group_rows))
        axes.step(x_values, y_values, where="post", label=f"N={population_value}, sigma={sigma_km:g}")
    axes.set_title("Observed Onset Survival-Style Curve")
    axes.set_xlabel("Model time (years)")
    axes.set_ylabel("Fraction not yet observed-onset")
    axes.set_ylim(0.0, 1.0)
    axes.grid(True, alpha=0.3)
    if axes.lines:
        axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_probability_metrics(
    rows: list[dict[str, object]],
    metrics: tuple[str, ...],
    title: str,
    filename: str,
) -> Path:
    output_path = Path(config.OUTPUT_DIRECTORY) / filename
    figure, axes = plt.subplots()
    for metric in metrics:
        metric_rows = [row for row in rows if row["metric"] == metric]
        for sigma_km, group_rows in _probability_rows_by_sigma(metric_rows).items():
            valid_rows = [row for row in group_rows if row["estimate"] is not None]
            if not valid_rows:
                continue
            axes.plot(
                [int(row["population_value"]) for row in valid_rows],
                [float(row["estimate"]) for row in valid_rows],
                marker="o",
                label=f"{metric}, sigma={sigma_km:g}",
            )
    axes.set_title(f"{title} (Wilson intervals in CSV)")
    axes.set_xlabel("Initial population N0")
    axes.set_ylabel("Probability")
    axes.set_ylim(0.0, 1.0)
    axes.grid(True, alpha=0.3)
    if axes.lines:
        axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _successful_groups(rows: list[dict[str, str]]):
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") != "success":
            continue
        groups[(int(row["population_value"]), float(row["position_uncertainty_sigma_km"]))].append(row)
    return dict(sorted(groups.items()))


def _cascade_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for row in rows:
        label = row.get("cascade_level_classification", "TN")
        counts[label] += 1
    return counts


def _wilson_interval(numerator: int, denominator: int) -> tuple[float | None, float | None]:
    if denominator <= 0:
        return None, None
    z = 1.959963984540054
    p_hat = numerator / denominator
    factor = 1.0 + z * z / denominator
    center = (p_hat + z * z / (2.0 * denominator)) / factor
    half_width = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) / denominator) + (z * z / (4.0 * denominator * denominator)))
        / factor
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _supports_survival_curve(rows: list[dict[str, str]]) -> bool:
    successful_count = sum(1 for row in rows if row.get("status") == "success")
    return successful_count >= config.EXPERIMENT_SURVIVAL_MIN_RUNS


def _group_summary_rows(rows: list[dict[str, object]], layer_column: str):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(float(row["sigma_km"]), row[layer_column])].append(row)
    return dict(sorted(grouped.items()))


def _probability_rows_by_sigma(rows: list[dict[str, object]]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["sigma_km"])].append(row)
    return dict(sorted(grouped.items()))


def _diagnostic_rows_by_sigma(rows: list[dict[str, object]]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["sigma_km"])].append(row)
    return dict(sorted(grouped.items()))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ("" if value is None else value)
                    for key, value in row.items()
                }
            )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _optional_float(value: str | float | None) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _quantile_or_none(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")
