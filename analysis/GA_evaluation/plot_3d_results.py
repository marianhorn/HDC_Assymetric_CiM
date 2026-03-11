import argparse
import csv
import math
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


PHASES = [
    "preopt-validation",
    "preopt-test",
    "postopt-validation",
    "postopt-test",
]

PHASE_LABELS = {
    "preopt-validation": "Pre-opt validation",
    "preopt-test": "Pre-opt test",
    "postopt-validation": "Post-opt validation",
    "postopt-test": "Post-opt test",
}


def parse_info(info_str):
    info = {}
    if not info_str:
        return info
    for part in info_str.split(","):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        info[key.strip()] = value.strip()
    return info


def load_records(csv_path, mutation_rate=None):
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            info = parse_info(row.get("info", ""))
            scope = info.get("scope")
            phase = info.get("phase")
            if scope not in ("overall", "dataset") or phase not in PHASES:
                continue

            try:
                row_mut = float(row["ga_mutation_rate"])
                num_levels = int(row["num_levels"])
                vector_dimension = int(row["vector_dimension"])
                accuracy = float(row["overall_accuracy"])
            except (KeyError, ValueError):
                continue

            if mutation_rate is not None and abs(row_mut - mutation_rate) > 1e-12:
                continue

            dataset = None
            if scope == "dataset":
                if "dataset" not in info:
                    continue
                try:
                    dataset = int(info["dataset"])
                except ValueError:
                    continue

            records.append(
                {
                    "scope": scope,
                    "dataset": dataset,
                    "phase": phase,
                    "num_levels": num_levels,
                    "vector_dimension": vector_dimension,
                    "accuracy": accuracy,
                    "mutation_rate": row_mut,
                }
            )
    return records


def build_surface(records, levels, dims, scope, phase, dataset=None):
    cell_values = defaultdict(list)
    for rec in records:
        if rec["scope"] != scope:
            continue
        if rec["phase"] != phase:
            continue
        if scope == "dataset" and rec["dataset"] != dataset:
            continue
        key = (rec["num_levels"], rec["vector_dimension"])
        cell_values[key].append(rec["accuracy"])

    matrix = np.full((len(levels), len(dims)), np.nan, dtype=float)
    level_idx = {value: i for i, value in enumerate(levels)}
    dim_idx = {value: i for i, value in enumerate(dims)}
    for (level, dim), values in cell_values.items():
        if level in level_idx and dim in dim_idx:
            matrix[level_idx[level], dim_idx[dim]] = float(np.mean(values))
    return matrix


def plot_overall_phases(records, levels, dims, out_path):
    x, y = np.meshgrid(np.array(levels), np.array(dims), indexing="ij")
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("GA Results 3D Surfaces (overall)")

    for idx, phase in enumerate(PHASES, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        z = build_surface(records, levels, dims, "overall", phase)
        z_masked = np.ma.masked_invalid(z)
        ax.plot_surface(x, y, z_masked, cmap="viridis", edgecolor="none", antialiased=True)
        ax.set_title(PHASE_LABELS[phase])
        ax.set_xlabel("NUM_LEVELS")
        ax.set_ylabel("VECTOR_DIMENSION")
        ax.set_zlabel("Accuracy")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    return fig


def plot_dataset_postopt_test(records, levels, dims, out_path):
    datasets = sorted({rec["dataset"] for rec in records if rec["scope"] == "dataset"})
    if not datasets:
        return

    x, y = np.meshgrid(np.array(levels), np.array(dims), indexing="ij")
    cols = 2
    rows = math.ceil(len(datasets) / cols)
    fig = plt.figure(figsize=(14, 5 * rows))
    fig.suptitle("GA Results 3D Surfaces (post-opt test by dataset)")

    for idx, dataset in enumerate(datasets, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        z = build_surface(records, levels, dims, "dataset", "postopt-test", dataset=dataset)
        z_masked = np.ma.masked_invalid(z)
        ax.plot_surface(x, y, z_masked, cmap="coolwarm", edgecolor="none", antialiased=True)
        ax.set_title(f"Dataset {dataset}")
        ax.set_xlabel("NUM_LEVELS")
        ax.set_ylabel("VECTOR_DIMENSION")
        ax.set_zlabel("Accuracy")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    return fig


def plot_single_dataset_all_phases(records, levels, dims, dataset, out_path):
    x, y = np.meshgrid(np.array(levels), np.array(dims), indexing="ij")
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"GA Results 3D Surfaces (dataset {dataset})")

    for idx, phase in enumerate(PHASES, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        z = build_surface(records, levels, dims, "dataset", phase, dataset=dataset)
        z_masked = np.ma.masked_invalid(z)
        ax.plot_surface(x, y, z_masked, cmap="viridis", edgecolor="none", antialiased=True)
        ax.set_title(PHASE_LABELS[phase])
        ax.set_xlabel("NUM_LEVELS")
        ax.set_ylabel("VECTOR_DIMENSION")
        ax.set_zlabel("Accuracy")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    return fig


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Create 3D GA result surfaces over num_levels and vector_dimension."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(base_dir, "results", "results_all.csv"),
        help="Path to GA results CSV (default: analysis/GA_evaluation/results/results_all.csv).",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=None,
        help="Optional filter for ga_mutation_rate.",
    )
    parser.add_argument(
        "--out-dir",
        default=base_dir,
        help="Directory for output PNG files.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Only save plots to disk (do not open interactive windows).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Missing results CSV: {args.csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_records(args.csv, mutation_rate=args.mutation_rate)
    if not records:
        raise RuntimeError("No matching rows found in CSV.")

    levels = sorted({rec["num_levels"] for rec in records})
    dims = sorted({rec["vector_dimension"] for rec in records})
    mutations = sorted({rec["mutation_rate"] for rec in records})

    print(f"Loaded rows: {len(records)}")
    print(f"NUM_LEVELS values: {levels}")
    print(f"VECTOR_DIMENSION values: {dims}")
    print(f"Mutation rates in selection: {mutations}")

    overall_out = os.path.join(args.out_dir, "ga_3d_overall_phases.png")
    dataset_out = os.path.join(args.out_dir, "ga_3d_postopt_test_by_dataset.png")

    figures = []
    overall_fig = plot_overall_phases(records, levels, dims, overall_out)
    if overall_fig is not None:
        figures.append(overall_fig)

    dataset_fig = plot_dataset_postopt_test(records, levels, dims, dataset_out)
    if dataset_fig is not None:
        figures.append(dataset_fig)

    datasets = sorted({rec["dataset"] for rec in records if rec["scope"] == "dataset"})
    dataset_phase_outputs = []
    for dataset in datasets:
        per_dataset_out = os.path.join(
            args.out_dir, f"ga_3d_dataset_{dataset}_all_phases.png"
        )
        per_dataset_fig = plot_single_dataset_all_phases(
            records, levels, dims, dataset, per_dataset_out
        )
        if per_dataset_fig is not None:
            figures.append(per_dataset_fig)
            dataset_phase_outputs.append(per_dataset_out)

    print(f"Saved: {overall_out}")
    print(f"Saved: {dataset_out}")
    for path in dataset_phase_outputs:
        print(f"Saved: {path}")

    if args.no_show:
        for fig in figures:
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
