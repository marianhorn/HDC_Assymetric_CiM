import argparse
import csv
import os
import re


GEN_RE = re.compile(r"GA generation\s+(\d+)/(\d+)")
IND_RE = re.compile(
    r"individual\s+\d+/\d+\s+accuracy:\s*([0-9.]+)%(?:,\s*similarity:\s*([0-9.]+))?"
)
NEW_SELECTED_RE = re.compile(r"new selected individuals:\s*(\d+)/(\d+)")
SEED_RE = re.compile(r"(?:seed|s)[_-]?(\d+)", re.IGNORECASE)


def mean(values):
    return sum(values) / len(values)


def parse_log(path):
    generations = {}
    current_generation = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            gen_match = GEN_RE.search(line)
            if gen_match:
                current_generation = int(gen_match.group(1))
                generations.setdefault(
                    current_generation,
                    {
                        "accuracies": [],
                        "similarities": [],
                        "new_selected": "",
                        "population": "",
                    },
                )
                continue

            if current_generation is None:
                continue

            ind_match = IND_RE.search(line)
            if ind_match:
                generations[current_generation]["accuracies"].append(float(ind_match.group(1)))
                if ind_match.group(2) is not None:
                    generations[current_generation]["similarities"].append(float(ind_match.group(2)))
                continue

            new_selected_match = NEW_SELECTED_RE.search(line)
            if new_selected_match:
                generations[current_generation]["new_selected"] = int(new_selected_match.group(1))
                generations[current_generation]["population"] = int(new_selected_match.group(2))

    rows = []
    for generation, data in sorted(generations.items()):
        accuracies = data["accuracies"]
        similarities = data["similarities"]
        rows.append(
            {
                "generation": generation,
                "mean_accuracy_percent": float(mean(accuracies)) if accuracies else "",
                "max_accuracy_percent": float(max(accuracies)) if accuracies else "",
                "num_individuals_with_accuracy": len(accuracies),
                "mean_class_vector_similarity": float(mean(similarities)) if similarities else "",
                "max_class_vector_similarity": float(max(similarities)) if similarities else "",
                "num_individuals_with_similarity": len(similarities),
                "new_selected": data["new_selected"],
                "population": data["population"],
            }
        )
    return rows


def infer_seed(path, fallback):
    match = SEED_RE.search(path)
    if match:
        return int(match.group(1))
    return fallback


def write_csv(rows, path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_array(rows, field):
    import numpy as np

    return np.array([float(row[field]) if row[field] != "" else np.nan for row in rows], dtype=float)


def plot_metric_rows(rows, output_path, show, metric_prefix, metric_label, y_label):
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        raise RuntimeError("No generation rows found in log.")

    x = np.array([row["generation"] for row in rows], dtype=int)
    mean_metric = numeric_array(rows, f"mean_{metric_prefix}")
    max_metric = numeric_array(rows, f"max_{metric_prefix}")
    new_selected = numeric_array(rows, "new_selected")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x, max_metric, color="#1f77b4", linewidth=2.0, label=f"Maximum {metric_label}")
    axes[0].plot(x, mean_metric, color="#ff7f0e", linewidth=2.0, label=f"Mean {metric_label}")
    axes[0].set_ylabel(y_label)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(
        x,
        new_selected,
        color="#2ca02c",
        linewidth=2.0,
        label="Newly accepted individuals per generation",
    )
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Individuals")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    axes[1].set_xlim(0, int(np.nanmax(x)))

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def build_aggregate_rows(seed_rows):
    import numpy as np

    generations = sorted({row["generation"] for _, rows in seed_rows for row in rows})
    aggregate_rows = []
    for generation in generations:
        values_by_metric = {
            "mean_accuracy_percent": [],
            "max_accuracy_percent": [],
            "mean_class_vector_similarity": [],
            "max_class_vector_similarity": [],
            "new_selected": [],
        }
        population_values = []
        for _, rows in seed_rows:
            match = next((row for row in rows if row["generation"] == generation), None)
            if not match:
                continue
            for metric in values_by_metric:
                if match[metric] != "":
                    values_by_metric[metric].append(float(match[metric]))
            if match["population"] != "":
                population_values.append(float(match["population"]))

        row = {"generation": generation}
        for metric, values in values_by_metric.items():
            arr = np.array(values, dtype=float)
            row[f"{metric}_mean_across_seeds"] = float(np.mean(arr)) if len(arr) else ""
            row[f"{metric}_std_across_seeds"] = float(np.std(arr)) if len(arr) else ""
            row[f"{metric}_num_seeds"] = len(arr)
        row["population"] = int(population_values[0]) if population_values else ""
        aggregate_rows.append(row)
    return aggregate_rows


def plot_aggregate_metric_rows(rows, output_path, show, metric_prefix, metric_label, y_label):
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        raise RuntimeError("No aggregate rows to plot.")

    x = np.array([row["generation"] for row in rows], dtype=int)
    mean_metric = numeric_array(rows, f"mean_{metric_prefix}_mean_across_seeds")
    mean_metric_std = numeric_array(rows, f"mean_{metric_prefix}_std_across_seeds")
    max_metric = numeric_array(rows, f"max_{metric_prefix}_mean_across_seeds")
    max_metric_std = numeric_array(rows, f"max_{metric_prefix}_std_across_seeds")
    new_selected = numeric_array(rows, "new_selected_mean_across_seeds")
    new_selected_std = numeric_array(rows, "new_selected_std_across_seeds")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x, max_metric, color="#1f77b4", linewidth=2.0, label=f"Maximum {metric_label}")
    axes[0].fill_between(
        x, max_metric - max_metric_std, max_metric + max_metric_std, color="#1f77b4", alpha=0.18
    )
    axes[0].plot(x, mean_metric, color="#ff7f0e", linewidth=2.0, label=f"Mean {metric_label}")
    axes[0].fill_between(
        x,
        mean_metric - mean_metric_std,
        mean_metric + mean_metric_std,
        color="#ff7f0e",
        alpha=0.18,
    )
    axes[0].set_ylabel(y_label)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(
        x,
        new_selected,
        color="#2ca02c",
        linewidth=2.0,
        label="Newly accepted individuals per generation",
    )
    axes[1].fill_between(x, new_selected - new_selected_std, new_selected + new_selected_std, color="#2ca02c", alpha=0.18)
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Individuals")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    axes[1].set_xlim(0, int(np.nanmax(x)))

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GA convergence from modelFoot OUTPUT_MODE=2 logs.")
    parser.add_argument("log", nargs="?", help="Path to one modelFoot output log.")
    parser.add_argument(
        "--logs",
        nargs="+",
        help="Paths to multiple modelFoot output logs. Produces per-log outputs plus aggregate mean/std over logs.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("analysis", "CiM_evaluation", "plots", "ga_run_convergence"),
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively.")
    args = parser.parse_args()

    log_paths = args.logs if args.logs else ([args.log] if args.log else [])
    if not log_paths:
        raise RuntimeError("Provide one log path or --logs path1 path2 ...")

    per_seed_fields = [
        "seed",
        "log",
        "generation",
        "mean_accuracy_percent",
        "max_accuracy_percent",
        "num_individuals_with_accuracy",
        "mean_class_vector_similarity",
        "max_class_vector_similarity",
        "num_individuals_with_similarity",
        "new_selected",
        "population",
    ]
    aggregate_fields = [
        "generation",
        "mean_accuracy_percent_mean_across_seeds",
        "mean_accuracy_percent_std_across_seeds",
        "mean_accuracy_percent_num_seeds",
        "max_accuracy_percent_mean_across_seeds",
        "max_accuracy_percent_std_across_seeds",
        "max_accuracy_percent_num_seeds",
        "mean_class_vector_similarity_mean_across_seeds",
        "mean_class_vector_similarity_std_across_seeds",
        "mean_class_vector_similarity_num_seeds",
        "max_class_vector_similarity_mean_across_seeds",
        "max_class_vector_similarity_std_across_seeds",
        "max_class_vector_similarity_num_seeds",
        "new_selected_mean_across_seeds",
        "new_selected_std_across_seeds",
        "new_selected_num_seeds",
        "population",
    ]

    seed_rows = []
    skipped_logs = []
    for index, log_path in enumerate(log_paths, start=1):
        rows = parse_log(log_path)
        if not rows:
            skipped_logs.append(log_path)
            print(f"{log_path}:")
            print("  skipped: no GA generation data found")
            print("  reason: log is empty/incomplete or was not produced with OUTPUT_MODE=2")
            continue

        seed = infer_seed(log_path, index)
        for row in rows:
            row["seed"] = seed
            row["log"] = log_path

        base_name = os.path.splitext(os.path.basename(log_path))[0]
        csv_path = os.path.join(args.output_dir, f"{base_name}_generation_metrics.csv")
        png_path = os.path.join(args.output_dir, f"{base_name}_accuracy_convergence.png")
        similarity_png_path = os.path.join(
            args.output_dir, f"{base_name}_class_vector_similarity_convergence.png"
        )

        write_csv(rows, csv_path, per_seed_fields)
        plot_metric_rows(
            rows,
            png_path,
            args.show and len(log_paths) == 1,
            "accuracy_percent",
            "validation accuracy",
            "Accuracy in %",
        )
        if any(row["mean_class_vector_similarity"] != "" for row in rows):
            plot_metric_rows(
                rows,
                similarity_png_path,
                args.show and len(log_paths) == 1,
                "class_vector_similarity",
                "class-vector similarity",
                "Class-vector similarity",
            )

        valid_acc_rows = [row for row in rows if row["mean_accuracy_percent"] != ""]
        valid_similarity_rows = [row for row in rows if row["mean_class_vector_similarity"] != ""]
        valid_selected_rows = [row for row in rows if row["new_selected"] != ""]
        print(f"{log_path}:")
        print(f"  parsed generations: {len(rows)}")
        print(f"  generations with individual accuracies: {len(valid_acc_rows)}")
        print(f"  generations with individual similarities: {len(valid_similarity_rows)}")
        print(f"  generations with new-selected counts: {len(valid_selected_rows)}")
        print(f"  saved csv: {csv_path}")
        print(f"  saved accuracy plot: {png_path}")
        if valid_similarity_rows:
            print(f"  saved similarity plot: {similarity_png_path}")

        seed_rows.append((seed, rows))

    if not seed_rows:
        raise RuntimeError("No usable GA generation data found in any provided log.")

    if len(seed_rows) > 1:
        all_rows = [row for _, rows in seed_rows for row in rows]
        all_csv_path = os.path.join(args.output_dir, "all_seed_generation_metrics.csv")
        write_csv(all_rows, all_csv_path, per_seed_fields)

        aggregate_rows = build_aggregate_rows(seed_rows)
        aggregate_csv_path = os.path.join(args.output_dir, "aggregate_generation_metrics.csv")
        aggregate_png_path = os.path.join(args.output_dir, "aggregate_accuracy_convergence.png")
        aggregate_similarity_png_path = os.path.join(
            args.output_dir, "aggregate_class_vector_similarity_convergence.png"
        )
        write_csv(aggregate_rows, aggregate_csv_path, aggregate_fields)
        plot_aggregate_metric_rows(
            aggregate_rows,
            aggregate_png_path,
            args.show,
            "accuracy_percent",
            "validation accuracy",
            "Accuracy in %",
        )
        if any(row["mean_class_vector_similarity_mean_across_seeds"] != "" for row in aggregate_rows):
            plot_aggregate_metric_rows(
                aggregate_rows,
                aggregate_similarity_png_path,
                args.show,
                "class_vector_similarity",
                "class-vector similarity",
                "Class-vector similarity",
            )
        print("aggregate:")
        print(f"  saved all-seed csv: {all_csv_path}")
        print(f"  saved aggregate csv: {aggregate_csv_path}")
        print(f"  saved aggregate accuracy plot: {aggregate_png_path}")
        if any(row["mean_class_vector_similarity_mean_across_seeds"] != "" for row in aggregate_rows):
            print(f"  saved aggregate similarity plot: {aggregate_similarity_png_path}")

    if skipped_logs:
        print("skipped logs:")
        for log_path in skipped_logs:
            print(f"  {log_path}")


if __name__ == "__main__":
    main()
