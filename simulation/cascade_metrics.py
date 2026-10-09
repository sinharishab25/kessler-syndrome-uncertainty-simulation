"""Cascade-level classification and aggregate metrics."""

from math import sqrt


CASCADE_LEVEL_LABELS = ("TP", "FP", "FN", "TN")


def classify_cascade_level(
    true_onset_time_model_years: float | None,
    observed_onset_time_model_years: float | None,
) -> str:
    """Classify one run-level cascade detection outcome."""
    true_detected = true_onset_time_model_years is not None
    observed_detected = observed_onset_time_model_years is not None
    if true_detected and observed_detected:
        return "TP"
    if not true_detected and observed_detected:
        return "FP"
    if true_detected and not observed_detected:
        return "FN"
    return "TN"


def onset_time_bias(
    true_onset_time_model_years: float | None,
    observed_onset_time_model_years: float | None,
) -> float | None:
    """Return observed-minus-true onset bias only when both onsets exist."""
    if true_onset_time_model_years is None or observed_onset_time_model_years is None:
        return None
    return observed_onset_time_model_years - true_onset_time_model_years


def calculate_cascade_level_metric_rows(results) -> list[dict[str, object]]:
    """Calculate aggregate cascade-level metrics with Wilson intervals."""
    counts = {label: 0 for label in CASCADE_LEVEL_LABELS}
    for result in results:
        counts[result.cascade_level_classification] += 1

    total = sum(counts.values())
    metric_specs = [
        ("precision", counts["TP"], counts["TP"] + counts["FP"]),
        ("recall", counts["TP"], counts["TP"] + counts["FN"]),
        ("true_cascade_probability", counts["TP"] + counts["FN"], total),
        ("observed_cascade_probability", counts["TP"] + counts["FP"], total),
        ("false_cascade_probability", counts["FP"], total),
        ("missed_cascade_probability", counts["FN"], total),
    ]
    return [
        {
            "metric": metric,
            "numerator": numerator,
            "denominator": denominator,
            "estimate": _safe_divide(numerator, denominator),
            "wilson_lower": _wilson_interval(numerator, denominator)[0],
            "wilson_upper": _wilson_interval(numerator, denominator)[1],
        }
        for metric, numerator, denominator in metric_specs
    ]


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _wilson_interval(
    numerator: int,
    denominator: int,
    z_value: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    """Return the two-sided Wilson score interval for a binomial proportion."""
    if denominator == 0:
        return (None, None)
    proportion = numerator / denominator
    z_squared = z_value * z_value
    denominator_term = 1.0 + z_squared / denominator
    center = (proportion + z_squared / (2.0 * denominator)) / denominator_term
    margin = (
        z_value
        * sqrt(
            (proportion * (1.0 - proportion) + z_squared / (4.0 * denominator))
            / denominator
        )
        / denominator_term
    )
    return (max(0.0, center - margin), min(1.0, center + margin))
