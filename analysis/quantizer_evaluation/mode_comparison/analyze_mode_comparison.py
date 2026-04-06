import argparse
import csv
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "results.csv")
SUMMARY_PATH = os.path.join(BASE_DIR, "summary_mode_comparison.csv")
BEST_PATH = os.path.join(BASE_DIR, "best_advanced_test_by_dataset.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
FIXED_NUM_LEVELS_FOR_VECTOR_PLOTS = [20, 30, 40]

MODE_NAMES = {
    0: "uniform",
    1: "quantile",
    2: "kmeans_1d",
    3: "decision_tree_1d",
    4: "chimerge",
    5: "ga_refined",
}
PHASE_FIELDS = {
    "uniform-validation": "uniform_validation",
    "uniform-test": "uniform_test",
    "advanced-validation": "advanced_validation",
    "advanced-test": "advanced_test",
}


def parse_info(info_text):
    result = {}
    for chunk in info_text.split(","):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        result[key] = value
    return result


def load_summary_rows():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")

    grouped = {}
    with open(RESULTS_PATH, "r", encoding="utf-8", newline="") as results_file:
        reader = csv.DictReader(results_file)
        for row in reader:
            info = parse_info(row["info"])
            if info.get("scope") != "dataset":
                continue
            phase = info.get("phase")
            if phase not in PHASE_FIELDS:
                continue

            key = (
                int(row["binning_mode"]),
                int(row["num_levels"]),
                int(row["vector_dimension"]),
                int(info["dataset"]),
            )
            if key not in grouped:
                grouped[key] = {
                    "binning_mode": int(row["binning_mode"]),
                    "mode_name": MODE_NAMES.get(int(row["binning_mode"]), f"mode_{row['binning_mode']}"),
                    "num_levels": int(row["num_levels"]),
                    "vector_dimension": int(row["vector_dimension"]),
                    "dataset": int(info["dataset"]),
                    "uniform_validation": None,
                    "uniform_test": None,
                    "advanced_validation": None,
                    "advanced_test": None,
                }

            grouped[key][PHASE_FIELDS[phase]] = float(row["overall_accuracy"])

    summary_rows = []
    for _, row in sorted(grouped.items()):
        if row["binning_mode"] == 0:
            if row["uniform_validation"] is None:
                row["uniform_validation"] = row["advanced_validation"]
            if row["uniform_test"] is None:
                row["uniform_test"] = row["advanced_test"]
        summary_rows.append(row)

    return summary_rows


def write_summary_csv(summary_rows):
    with open(SUMMARY_PATH, "w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "binning_mode",
                "mode_name",
                "num_levels",
                "vector_dimension",
                "dataset",
                "uniform_validation",
                "uniform_test",
                "advanced_validation",
                "advanced_test",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def write_best_csv(summary_rows):
    best_rows = []
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["dataset"]].append(row)

    for dataset, rows in sorted(grouped.items()):
        best = max(rows, key=lambda item: item["advanced_test"] if item["advanced_test"] is not None else -1.0)
        best_rows.append(
            {
                "dataset": dataset,
                "mode_name": best["mode_name"],
                "num_levels": best["num_levels"],
                "vector_dimension": best["vector_dimension"],
                "advanced_test_accuracy": best["advanced_test"],
                "advanced_validation_accuracy": best["advanced_validation"],
            }
        )

    with open(BEST_PATH, "w", encoding="utf-8", newline="") as best_file:
        writer = csv.DictWriter(
            best_file,
            fieldnames=[
                "dataset",
                "mode_name",
                "num_levels",
                "vector_dimension",
                "advanced_test_accuracy",
                "advanced_validation_accuracy",
            ],
        )
        writer.writeheader()
        for row in best_rows:
            writer.writerow(row)

    print("Best advanced test accuracy per dataset")
    print("dataset | mode | num_levels | vector_dimension | test_acc (%) | val_acc (%)")
    print("--------+------+------------+------------------+--------------+------------")
    for row in best_rows:
        print(
            f"{row['dataset']:7d} | "
            f"{row['mode_name']:12s} | "
            f"{row['num_levels']:10d} | "
            f"{row['vector_dimension']:16d} | "
            f"{row['advanced_test_accuracy'] * 100.0:12.2f} | "
            f"{row['advanced_validation_accuracy'] * 100.0:10.2f}"
        )


def plot_mode_comparison_over_num_levels(summary_rows, show_plots):
    if plt is None:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    vector_dimensions = sorted({row["vector_dimension"] for row in summary_rows})
    datasets = sorted({row["dataset"] for row in summary_rows})
    mode_names = [MODE_NAMES[mode] for mode in sorted(MODE_NAMES) if mode in {row["binning_mode"] for row in summary_rows}]

    for dataset in datasets:
        for vector_dimension in vector_dimensions:
            subset = [
                row for row in summary_rows
                if row["dataset"] == dataset and row["vector_dimension"] == vector_dimension
            ]
            if not subset:
                continue

            for metric_key, metric_label in [
                ("advanced_validation", "Validation Accuracy"),
                ("advanced_test", "Test Accuracy"),
            ]:
                plt.figure(figsize=(9, 5))
                for mode_name in mode_names:
                    mode_rows = sorted(
                        [row for row in subset if row["mode_name"] == mode_name and row[metric_key] is not None],
                        key=lambda item: item["num_levels"],
                    )
                    if not mode_rows:
                        continue
                    xs = [row["num_levels"] for row in mode_rows]
                    ys = [row[metric_key] * 100.0 for row in mode_rows]
                    plt.plot(xs, ys, marker="o", label=mode_name)

                plt.xlabel("NUM_LEVELS")
                plt.ylabel(metric_label + " (%)")
                plt.title(f"Dataset {dataset} | VECTOR_DIMENSION={vector_dimension} | {metric_label}")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()

                filename = (
                    f"compare_modes_dataset{dataset}_vd{vector_dimension}_{metric_key}.png"
                )
                plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=160)
                if show_plots:
                    plt.show()
                plt.close()


def plot_mode_comparison_over_vector_dimension(summary_rows, show_plots, fixed_num_levels=30):
    if plt is None:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    datasets = sorted({row["dataset"] for row in summary_rows})
    mode_names = [MODE_NAMES[mode] for mode in sorted(MODE_NAMES) if mode in {row["binning_mode"] for row in summary_rows}]

    for dataset in datasets:
        subset = [
            row for row in summary_rows
            if row["dataset"] == dataset and row["num_levels"] == fixed_num_levels
        ]
        if not subset:
            continue

        for metric_key, metric_label in [
            ("advanced_validation", "Validation Accuracy"),
            ("advanced_test", "Test Accuracy"),
        ]:
            plt.figure(figsize=(9, 5))
            for mode_name in mode_names:
                mode_rows = sorted(
                    [row for row in subset if row["mode_name"] == mode_name and row[metric_key] is not None],
                    key=lambda item: item["vector_dimension"],
                )
                if not mode_rows:
                    continue
                xs = [row["vector_dimension"] for row in mode_rows]
                ys = [row[metric_key] * 100.0 for row in mode_rows]
                plt.plot(xs, ys, marker="o", label=mode_name)

            plt.xlabel("VECTOR_DIMENSION")
            plt.ylabel(metric_label + " (%)")
            plt.title(f"Dataset {dataset} | NUM_LEVELS={fixed_num_levels} | {metric_label}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            filename = (
                f"compare_modes_dataset{dataset}_nl{fixed_num_levels}_{metric_key}.png"
            )
            plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=160)
            if show_plots:
                plt.show()
            plt.close()


def plot_baseline_vs_advanced(summary_rows, show_plots, fixed_num_levels=20):
    if plt is None:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    datasets = sorted({row["dataset"] for row in summary_rows})
    mode_names = sorted({row["mode_name"] for row in summary_rows if row["mode_name"] != "uniform"})

    for dataset in datasets:
        for mode_name in mode_names:
            subset = [
                row for row in summary_rows
                if row["dataset"] == dataset and row["mode_name"] == mode_name and row["num_levels"] == fixed_num_levels
            ]
            subset = sorted(subset, key=lambda item: item["vector_dimension"])
            if not subset:
                continue

            plt.figure(figsize=(9, 5))
            plt.plot(
                [row["vector_dimension"] for row in subset],
                [row["uniform_validation"] * 100.0 for row in subset],
                marker="o",
                linestyle="--",
                label="uniform validation",
            )
            plt.plot(
                [row["vector_dimension"] for row in subset],
                [row["uniform_test"] * 100.0 for row in subset],
                marker="o",
                linestyle="--",
                label="uniform test",
            )
            plt.plot(
                [row["vector_dimension"] for row in subset],
                [row["advanced_validation"] * 100.0 for row in subset],
                marker="o",
                linestyle="-",
                label=f"{mode_name} validation",
            )
            plt.plot(
                [row["vector_dimension"] for row in subset],
                [row["advanced_test"] * 100.0 for row in subset],
                marker="o",
                linestyle="-",
                label=f"{mode_name} test",
            )
            plt.xlabel("VECTOR_DIMENSION")
            plt.ylabel("Accuracy (%)")
            plt.title(f"Dataset {dataset} | {mode_name} vs uniform | NUM_LEVELS={fixed_num_levels}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            filename = f"baseline_vs_advanced_dataset{dataset}_{mode_name}_nl{fixed_num_levels}.png"
            plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=160)
            if show_plots:
                plt.show()
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze non-GA quantizer mode comparison runs."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively in addition to saving them.",
    )
    args = parser.parse_args()

    summary_rows = load_summary_rows()
    if not summary_rows:
        raise RuntimeError("No dataset-level mode comparison rows found in results.csv.")

    write_summary_csv(summary_rows)
    write_best_csv(summary_rows)
    if plt is None:
        print("")
        print("matplotlib is not installed. Wrote CSV summaries but skipped plot generation.")
    else:
        plot_mode_comparison_over_num_levels(summary_rows, args.show)
        for fixed_num_levels in FIXED_NUM_LEVELS_FOR_VECTOR_PLOTS:
            plot_mode_comparison_over_vector_dimension(
                summary_rows,
                args.show,
                fixed_num_levels=fixed_num_levels,
            )
            plot_baseline_vs_advanced(
                summary_rows,
                args.show,
                fixed_num_levels=fixed_num_levels,
            )

    print("")
    print(f"Summary CSV: {SUMMARY_PATH}")
    print(f"Best-per-dataset CSV: {BEST_PATH}")
    if plt is not None:
        print(f"Plots directory: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
