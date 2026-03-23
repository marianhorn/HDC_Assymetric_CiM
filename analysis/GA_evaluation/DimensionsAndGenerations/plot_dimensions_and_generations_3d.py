import argparse
import csv
import os
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install with 'python -m pip install matplotlib'."
    ) from exc

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_CSV = os.path.join(BASE_DIR, "results.csv")
DEFAULT_MANIFEST_CSV = os.path.join(BASE_DIR, "run_manifest.csv")
DEFAULT_SUMMARY_CSV = os.path.join(BASE_DIR, "dimensions_generations_postopt_test_summary.csv")
DEFAULT_PLOT_PNG = os.path.join(BASE_DIR, "dimensions_generations_postopt_test_3d.png")


def parse_info(info_text):
    info = {}
    if not info_text:
        return info
    for part in info_text.split(","):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        info[key.strip()] = value.strip()
    return info


def load_manifest_rows(manifest_csv):
    rows = []
    with open(manifest_csv, "r", encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        for row in reader:
            rows.append(
                {
                    "run_index": int(row["run_index"]),
                    "num_levels": int(row["num_levels"]),
                    "vector_dimension": int(row["vector_dimension"]),
                    "ga_generations": int(row["ga_generations"]),
                    "timestamp": row.get("timestamp", ""),
                    "log_file": row.get("log_file", ""),
                }
            )
    rows.sort(key=lambda item: item["run_index"])
    return rows


def load_overall_postopt_test_rows(results_csv):
    rows = []
    with open(results_csv, "r", encoding="utf-8", newline="") as results_file:
        reader = csv.DictReader(results_file)
        for row in reader:
            info = parse_info(row.get("info", ""))
            if info.get("scope") != "overall" or info.get("phase") != "postopt-test":
                continue
            rows.append(
                {
                    "num_levels": int(row["num_levels"]),
                    "vector_dimension": int(row["vector_dimension"]),
                    "accuracy": float(row["overall_accuracy"]),
                    "info": row["info"],
                }
            )
    return rows


def align_manifest_and_results(manifest_rows, result_rows):
    if not manifest_rows:
        raise RuntimeError("Manifest is empty.")
    if not result_rows:
        raise RuntimeError("No overall postopt-test rows found in results.csv.")

    if len(result_rows) < len(manifest_rows):
        raise RuntimeError(
            f"Not enough overall postopt-test rows in results.csv "
            f"({len(result_rows)}) for manifest entries ({len(manifest_rows)})."
        )

    if len(result_rows) > len(manifest_rows):
        print(
            f"Warning: results.csv contains {len(result_rows)} overall postopt-test rows, "
            f"manifest has {len(manifest_rows)} runs. Using the last {len(manifest_rows)} rows."
        )
        result_rows = result_rows[-len(manifest_rows):]

    aligned = []
    for manifest_row, result_row in zip(manifest_rows, result_rows):
        aligned.append(
            {
                "run_index": manifest_row["run_index"],
                "num_levels": manifest_row["num_levels"],
                "vector_dimension": manifest_row["vector_dimension"],
                "ga_generations": manifest_row["ga_generations"],
                "overall_postopt_test_accuracy": result_row["accuracy"],
                "timestamp": manifest_row["timestamp"],
                "log_file": manifest_row["log_file"],
            }
        )
    return aligned


def write_summary_csv(summary_rows, summary_csv):
    with open(summary_csv, "w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "run_index",
                "num_levels",
                "vector_dimension",
                "ga_generations",
                "overall_postopt_test_accuracy",
                "timestamp",
                "log_file",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def build_surface(summary_rows):
    vector_dimensions = sorted({row["vector_dimension"] for row in summary_rows})
    generations = sorted({row["ga_generations"] for row in summary_rows})

    grid = np.full((len(generations), len(vector_dimensions)), np.nan, dtype=float)
    dim_index = {value: idx for idx, value in enumerate(vector_dimensions)}
    gen_index = {value: idx for idx, value in enumerate(generations)}

    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[(row["vector_dimension"], row["ga_generations"])].append(
            row["overall_postopt_test_accuracy"]
        )

    for (vector_dimension, ga_generations), values in grouped.items():
        grid[gen_index[ga_generations], dim_index[vector_dimension]] = float(np.mean(values))

    return vector_dimensions, generations, grid


def plot_surface(summary_rows, output_png, show_plot):
    vector_dimensions, generations, z_grid = build_surface(summary_rows)
    x_grid, y_grid = np.meshgrid(
        np.array(vector_dimensions, dtype=float),
        np.array(generations, dtype=float),
        indexing="xy",
    )

    num_levels_values = sorted({row["num_levels"] for row in summary_rows})
    title_suffix = (
        f"NUM_LEVELS={num_levels_values[0]}"
        if len(num_levels_values) == 1
        else f"NUM_LEVELS in {num_levels_values}"
    )

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        x_grid,
        y_grid,
        np.ma.masked_invalid(z_grid * 100.0),
        cmap="viridis",
        edgecolor="none",
        antialiased=True,
    )
    fig.colorbar(surface, ax=ax, shrink=0.7, pad=0.1, label="Accuracy (%)")

    ax.set_xlabel("VECTOR_DIMENSION")
    ax.set_ylabel("GA Generations")
    ax.set_zlabel("Post-opt test accuracy (%)")
    ax.set_title(f"GA Evaluation 3D Surface\nOverall post-opt test accuracy, {title_suffix}")
    ax.set_xticks(vector_dimensions)
    ax.set_yticks(generations)
    ax.view_init(elev=28, azim=-135)

    fig.tight_layout()
    fig.savefig(output_png, dpi=180)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the overall post-opt test accuracy from DimensionsAndGenerations "
            "as a 3D surface over vector dimension and GA generations."
        )
    )
    parser.add_argument(
        "--results-csv",
        default=DEFAULT_RESULTS_CSV,
        help="Path to results.csv (default: analysis/GA_evaluation/DimensionsAndGenerations/results.csv).",
    )
    parser.add_argument(
        "--manifest-csv",
        default=DEFAULT_MANIFEST_CSV,
        help="Path to run_manifest.csv (default: analysis/GA_evaluation/DimensionsAndGenerations/run_manifest.csv).",
    )
    parser.add_argument(
        "--summary-csv",
        default=DEFAULT_SUMMARY_CSV,
        help="Path to write the aligned summary CSV.",
    )
    parser.add_argument(
        "--output-png",
        default=DEFAULT_PLOT_PNG,
        help="Path to save the 3D PNG plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively in addition to saving it.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        raise FileNotFoundError(f"Missing results CSV: {args.results_csv}")
    if not os.path.exists(args.manifest_csv):
        raise FileNotFoundError(f"Missing manifest CSV: {args.manifest_csv}")

    manifest_rows = load_manifest_rows(args.manifest_csv)
    result_rows = load_overall_postopt_test_rows(args.results_csv)
    summary_rows = align_manifest_and_results(manifest_rows, result_rows)
    write_summary_csv(summary_rows, args.summary_csv)
    plot_surface(summary_rows, args.output_png, args.show)

    print(f"Manifest rows: {len(manifest_rows)}")
    print(f"Overall postopt-test rows considered: {len(summary_rows)}")
    print(f"Saved summary CSV: {args.summary_csv}")
    print(f"Saved 3D plot: {args.output_png}")


if __name__ == "__main__":
    main()
