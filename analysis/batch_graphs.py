"""Post-batch Monte Carlo graphing from saved CSV tables."""

from collections import defaultdict
from pathlib import Path
import csv

import matplotlib.pyplot as plt

import config


def plot_monte_carlo_batch_graphs(
    raw_csv_path: Path,
    summary_csv_path: Path,
) -> list[Path]:
    """Create batch plots after CSV output exists, using only saved tables."""
    return [
        _plot_summary_column(
            summary_csv_path,
            "collision_probability",
            "Collision Probability By Population",
            "Collision probability",
            config.MONTE_CARLO_COLLISION_PROBABILITY_GRAPH_FILENAME,
        ),
        _plot_summary_column(
            summary_csv_path,
            "mean_collision_count",
            "Mean Collision Count By Population",
            "Mean collision count",
            config.MONTE_CARLO_MEAN_COLLISION_COUNT_GRAPH_FILENAME,
        ),
        _plot_cumulative_collision_probability(raw_csv_path),
    ]


def _plot_summary_column(
    summary_csv_path: Path,
    column_name: str,
    title: str,
    y_label: str,
    output_filename: str,
) -> Path:
    """Plot one population summary column from the saved summary CSV."""
    rows = _read_csv_rows(summary_csv_path)
    output_path = Path(config.OUTPUT_DIRECTORY) / output_filename

    figure, axes = plt.subplots()
    rows_by_sigma = defaultdict(list)
    for row in rows:
        rows_by_sigma[float(row.get("sigma_km", "0") or 0.0)].append(row)
    for sigma_km in sorted(rows_by_sigma):
        sigma_rows = sorted(rows_by_sigma[sigma_km], key=lambda row: int(row["population_value"]))
        axes.plot(
            [int(row["population_value"]) for row in sigma_rows],
            [float(row[column_name]) for row in sigma_rows],
            marker="o",
            label=f"sigma={sigma_km:g}",
        )
    axes.set_title(title)
    axes.set_xlabel("Population")
    axes.set_ylabel(y_label)
    axes.grid(True, alpha=0.3)
    if rows_by_sigma:
        axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return output_path


def _plot_cumulative_collision_probability(raw_csv_path: Path) -> Path:
    """Plot cumulative collision fraction by completed run from raw CSV rows."""
    rows_by_population = defaultdict(list)
    for row in _read_csv_rows(raw_csv_path):
        sigma_km = float(row.get("position_uncertainty_sigma_km", "0") or 0.0)
        rows_by_population[(int(row["population_value"]), sigma_km)].append(row)

    output_path = (
        Path(config.OUTPUT_DIRECTORY)
        / config.MONTE_CARLO_CUMULATIVE_COLLISION_PROBABILITY_GRAPH_FILENAME
    )
    figure, axes = plt.subplots()

    for population_value, sigma_km in sorted(rows_by_population):
        rows = rows_by_population[(population_value, sigma_km)]
        run_numbers = []
        probabilities = []
        collided_runs = 0
        for index, row in enumerate(rows, start=1):
            if int(row["total_particle_collisions"]) > 0:
                collided_runs += 1
            run_numbers.append(index)
            probabilities.append(collided_runs / index)

        axes.step(
            run_numbers,
            probabilities,
            where="post",
            label=f"N={population_value}, sigma={sigma_km:g}",
        )

    axes.set_title("Cumulative Collision Probability By Run")
    axes.set_xlabel("Completed run within population")
    axes.set_ylabel("Cumulative collision probability")
    axes.set_ylim(0.0, 1.0)
    axes.grid(True, alpha=0.3)
    if rows_by_population:
        axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return output_path


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read a saved CSV table into plain string rows."""
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))
