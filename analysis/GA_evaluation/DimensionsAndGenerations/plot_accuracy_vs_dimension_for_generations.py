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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_CSV = os.path.join(BASE_DIR, "results.csv")
DEFAULT_MANIFEST_CSV = os.path.join(BASE_DIR, "run_manifest.csv")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "generation_plots")


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


def parse_generations_arg(text):
    values = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("At least one generation value must be provided.")
    return values


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


def load_overall_phase_rows(results_csv):
    phase_rows = {"preopt-test": [], "postopt-test": []}
    with open(results_csv, "r", encoding="utf-8", newline="") as results_file:
        reader = csv.DictReader(results_file)
        for row in reader:
            info = parse_info(row.get("info", ""))
            if info.get("scope") != "overall":
                continue
            phase = info.get("phase")
            if phase not in phase_rows:
                continue
            phase_rows[phase].append(
                {
                    "num_levels": int(row["num_levels"]),
                    "vector_dimension": int(row["vector_dimension"]),
                    "accuracy": float(row["overall_accuracy"]),
                }
            )
    return phase_rows


def align_manifest_and_results(manifest_rows, phase_rows):
    aligned = []
    total_runs = len(manifest_rows)
    if total_runs == 0:
        raise RuntimeError("Manifest is empty.")

    for phase in ("preopt-test", "postopt-test"):
        if len(phase_rows[phase]) < total_runs:
            raise RuntimeError(
                f"Not enough '{phase}' rows in results.csv ({len(phase_rows[phase])}) "
                f"for manifest entries ({total_runs})."
            )
        if len(phase_rows[phase]) > total_runs:
            print(
                f"Warning: results.csv contains {len(phase_rows[phase])} rows for {phase}, "
                f"manifest has {total_runs} runs. Using the last {total_runs} rows."
            )
            phase_rows[phase] = phase_rows[phase][-total_runs:]

    for index, manifest_row in enumerate(manifest_rows):
        preopt = phase_rows["preopt-test"][index]
        postopt = phase_rows["postopt-test"][index]
        aligned.append(
            {
                "run_index": manifest_row["run_index"],
                "num_levels": manifest_row["num_levels"],
                "vector_dimension": manifest_row["vector_dimension"],
                "ga_generations": manifest_row["ga_generations"],
                "preopt_test_accuracy": preopt["accuracy"],
                "postopt_test_accuracy": postopt["accuracy"],
                "delta_accuracy": postopt["accuracy"] - preopt["accuracy"],
                "timestamp": manifest_row["timestamp"],
                "log_file": manifest_row["log_file"],
            }
        )
    return aligned


def write_summary_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "run_index",
                "num_levels",
                "vector_dimension",
                "ga_generations",
                "preopt_test_accuracy",
                "postopt_test_accuracy",
                "delta_accuracy",
                "timestamp",
                "log_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_generation_rows(aligned_rows, generations):
    rows_by_generation = defaultdict(list)
    for row in aligned_rows:
        if row["ga_generations"] in generations:
            rows_by_generation[row["ga_generations"]].append(row)

    missing = [gen for gen in generations if gen not in rows_by_generation]
    if missing:
        raise RuntimeError(f"No runs found for generations: {missing}")

    for generation in rows_by_generation:
        rows_by_generation[generation].sort(key=lambda item: item["vector_dimension"])
    return rows_by_generation


def plot_generation(rows, generation, output_path, show_plot):
    vector_dimensions = [row["vector_dimension"] for row in rows]
    preopt = np.array([row["preopt_test_accuracy"] * 100.0 for row in rows], dtype=float)
    postopt = np.array([row["postopt_test_accuracy"] * 100.0 for row in rows], dtype=float)
    delta = np.array([row["delta_accuracy"] * 100.0 for row in rows], dtype=float)
    num_levels_values = sorted({row["num_levels"] for row in rows})

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(
        f"GA Accuracy vs VECTOR_DIMENSION for {generation} Generations\n"
        f"NUM_LEVELS={num_levels_values[0] if len(num_levels_values) == 1 else num_levels_values}"
    )

    axes[0].plot(vector_dimensions, postopt, marker="o", color="tab:blue")
    axes[0].set_ylabel("Post-opt test (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Post-opt test accuracy")

    axes[1].plot(vector_dimensions, preopt, marker="o", color="tab:orange")
    axes[1].set_ylabel("Pre-opt test (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Pre-opt test accuracy")

    axes[2].plot(vector_dimensions, delta, marker="o", color="tab:green")
    axes[2].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[2].set_xlabel("VECTOR_DIMENSION")
    axes[2].set_ylabel("Post - pre (%)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Accuracy difference")

    axes[2].set_xticks(vector_dimensions)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot pre-opt test accuracy, post-opt test accuracy, and their difference "
            "over vector dimension for one or more specified GA generation counts."
        )
    )
    parser.add_argument(
        "--generations",
        required=True,
        help="Comma-separated GA generation counts to plot, e.g. 64 or 64,128,256.",
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
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV summaries and plot PNGs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively in addition to saving them.",
    )
    args = parser.parse_args()

    generations = parse_generations_arg(args.generations)
    if not os.path.exists(args.results_csv):
        raise FileNotFoundError(f"Missing results CSV: {args.results_csv}")
    if not os.path.exists(args.manifest_csv):
        raise FileNotFoundError(f"Missing manifest CSV: {args.manifest_csv}")

    os.makedirs(args.output_dir, exist_ok=True)

    manifest_rows = load_manifest_rows(args.manifest_csv)
    phase_rows = load_overall_phase_rows(args.results_csv)
    aligned_rows = align_manifest_and_results(manifest_rows, phase_rows)
    rows_by_generation = build_generation_rows(aligned_rows, generations)

    for generation in generations:
        generation_rows = rows_by_generation[generation]
        summary_path = os.path.join(
            args.output_dir,
            f"summary_gen{generation}.csv",
        )
        plot_path = os.path.join(
            args.output_dir,
            f"accuracy_vs_dimension_gen{generation}.png",
        )
        write_summary_csv(generation_rows, summary_path)
        plot_generation(generation_rows, generation, plot_path, args.show)
        print(f"Saved summary CSV: {summary_path}")
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
