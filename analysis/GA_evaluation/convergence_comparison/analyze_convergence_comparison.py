import argparse
import csv
import os
import re
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
METRICS_CSV = os.path.join(BASE_DIR, "generation_metrics.csv")
SUMMARY_CSV = os.path.join(BASE_DIR, "dataset_accuracy_summary.csv")

np = None
plt = None

DATASET_RE = re.compile(r"Model for dataset #(\d+)")
GEN_RE = re.compile(r"GA generation (\d+)/(\d+)")
IND_RE = re.compile(r"individual \d+/\d+ accuracy: ([0-9.]+)%, similarity: ([0-9.]+)")
NEW_SEL_RE = re.compile(r"new selected individuals:\s*(\d+)/(\d+)")
POST_TEST_RE = re.compile(r"Dataset\s+(\d+)\s+post-opt test accuracy:\s*([0-9.]+)%")
TEST_ACC_RE = re.compile(r"Testing accuracy:\s*([0-9.]+)%")
POST_VAL_START_RE = re.compile(r"Evaluating post-optimization model on validation set\.")
SEED_FROM_NAME_RE = re.compile(r"run_seed_(\d+)\.txt$")


def parse_seed_from_filename(path):
    name = os.path.basename(path)
    match = SEED_FROM_NAME_RE.search(name)
    if not match:
        raise ValueError(f"Unexpected log filename format (expected run_seed_<n>.txt): {name}")
    return int(match.group(1))


def parse_single_log(path):
    seed = parse_seed_from_filename(path)
    run = {
        "seed": seed,
        "datasets": defaultdict(
            lambda: {
                "generations": defaultdict(lambda: {"acc": [], "sim": [], "new_selected": None, "population": None}),
                "post_test_acc": None,
                "post_val_acc": None,
            }
        ),
    }

    current_dataset = None
    current_generation = None
    awaiting_post_val_for_dataset = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m = DATASET_RE.search(line)
            if m:
                current_dataset = int(m.group(1))
                current_generation = None
                awaiting_post_val_for_dataset = None
                continue

            m = GEN_RE.search(line)
            if m and current_dataset is not None:
                current_generation = int(m.group(1))
                _ = run["datasets"][current_dataset]["generations"][current_generation]
                continue

            m = IND_RE.search(line)
            if m and current_dataset is not None and current_generation is not None:
                acc = float(m.group(1))
                sim = float(m.group(2))
                gen = run["datasets"][current_dataset]["generations"][current_generation]
                gen["acc"].append(acc)
                gen["sim"].append(sim)
                continue

            m = NEW_SEL_RE.search(line)
            if m and current_dataset is not None and current_generation is not None:
                gen = run["datasets"][current_dataset]["generations"][current_generation]
                gen["new_selected"] = int(m.group(1))
                gen["population"] = int(m.group(2))
                continue

            m = POST_TEST_RE.search(line)
            if m:
                ds = int(m.group(1))
                run["datasets"][ds]["post_test_acc"] = float(m.group(2))
                continue

            if POST_VAL_START_RE.search(line):
                if current_dataset is not None:
                    awaiting_post_val_for_dataset = current_dataset
                continue

            m = TEST_ACC_RE.search(line)
            if m and awaiting_post_val_for_dataset is not None:
                run["datasets"][awaiting_post_val_for_dataset]["post_val_acc"] = float(m.group(1))
                awaiting_post_val_for_dataset = None
                continue

    return run


def collect_runs(log_dir):
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    paths = []
    for name in os.listdir(log_dir):
        if not name.endswith(".txt"):
            continue
        if not SEED_FROM_NAME_RE.search(name):
            continue
        paths.append(os.path.join(log_dir, name))
    paths.sort()

    if not paths:
        raise FileNotFoundError(
            f"No run logs found in {log_dir}. Expected files like run_seed_<n>.txt"
        )

    return [parse_single_log(p) for p in paths]


def build_generation_rows(runs):
    rows = []
    for run in runs:
        seed = run["seed"]
        for dataset, ds_data in run["datasets"].items():
            generations = ds_data["generations"]
            for gen, gen_data in sorted(generations.items()):
                if not gen_data["acc"]:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "generation": gen,
                        "avg_accuracy": float(np.mean(gen_data["acc"])),
                        "max_accuracy": float(np.max(gen_data["acc"])),
                        "avg_similarity": float(np.mean(gen_data["sim"])),
                        "max_similarity": float(np.max(gen_data["sim"])),
                        "new_selected": gen_data["new_selected"],
                        "population": gen_data["population"],
                    }
                )
    return rows


def write_generation_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "seed",
        "dataset",
        "generation",
        "avg_accuracy",
        "max_accuracy",
        "avg_similarity",
        "max_similarity",
        "new_selected",
        "population",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["seed"], r["dataset"], r["generation"])):
            writer.writerow(row)


def aggregate_metric_by_dataset_generation(rows, metric_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["generation"])].append(row[metric_key])

    dataset_to_xy = defaultdict(lambda: {"x": [], "y": []})
    for (dataset, generation), values in sorted(grouped.items()):
        dataset_to_xy[dataset]["x"].append(generation)
        dataset_to_xy[dataset]["y"].append(float(np.mean(values)))
    return dataset_to_xy


def plot_metric(rows, metric_key, ylabel, title, filename, show):
    series = aggregate_metric_by_dataset_generation(rows, metric_key)
    if not series:
        print(f"No data for plot: {title}")
        return

    plt.figure(figsize=(9, 5))
    for dataset in sorted(series.keys()):
        x = series[dataset]["x"]
        y = series[dataset]["y"]
        plt.plot(x, y, marker="o", linewidth=1.8, markersize=3, label=f"Dataset {dataset}")

    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def summarize_dataset_maxima(runs):
    summary_rows = []
    all_datasets = sorted(
        {
            dataset
            for run in runs
            for dataset in run["datasets"].keys()
        }
    )

    print("\nMax achieved accuracy per dataset (across seeds)")
    print("dataset | max_post_test_acc (%) [seed] | max_post_val_acc (%) [seed]")
    print("--------+-------------------------------+-----------------------------")

    for dataset in all_datasets:
        test_candidates = []
        val_candidates = []
        for run in runs:
            ds = run["datasets"].get(dataset)
            if not ds:
                continue
            if ds["post_test_acc"] is not None:
                test_candidates.append((ds["post_test_acc"], run["seed"]))
            if ds["post_val_acc"] is not None:
                val_candidates.append((ds["post_val_acc"], run["seed"]))

        if test_candidates:
            max_test_acc, max_test_seed = max(test_candidates, key=lambda x: x[0])
            test_text = f"{max_test_acc:.3f} [seed {max_test_seed}]"
        else:
            max_test_acc = None
            max_test_seed = None
            test_text = "n/a"

        if val_candidates:
            max_val_acc, max_val_seed = max(val_candidates, key=lambda x: x[0])
            val_text = f"{max_val_acc:.3f} [seed {max_val_seed}]"
        else:
            max_val_acc = None
            max_val_seed = None
            val_text = "n/a"

        print(f"{dataset:7d} | {test_text:29s} | {val_text}")
        summary_rows.append(
            {
                "dataset": dataset,
                "max_post_test_acc": max_test_acc,
                "max_post_test_seed": max_test_seed,
                "max_post_val_acc": max_val_acc,
                "max_post_val_seed": max_val_seed,
            }
        )

    return summary_rows


def write_summary_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "max_post_test_acc",
                "max_post_test_seed",
                "max_post_val_acc",
                "max_post_val_seed",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    global np, plt
    parser = argparse.ArgumentParser(
        description="Analyze GA convergence comparison logs and generate plots."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively in addition to saving them.",
    )
    args = parser.parse_args()

    try:
        import numpy as np_mod
        import matplotlib.pyplot as plt_mod
    except Exception as exc:
        raise RuntimeError(
            "Missing Python dependencies for analysis script. "
            "Install them with: pip install numpy matplotlib"
        ) from exc
    np = np_mod
    plt = plt_mod

    runs = collect_runs(LOG_DIR)
    rows = build_generation_rows(runs)
    if not rows:
        raise RuntimeError("No generation metrics could be extracted from logs.")

    write_generation_csv(rows, METRICS_CSV)

    plot_metric(
        rows,
        metric_key="avg_accuracy",
        ylabel="Average accuracy (%)",
        title="Average Accuracy vs Generation",
        filename="avg_accuracy_vs_generation.png",
        show=args.show,
    )
    plot_metric(
        rows,
        metric_key="max_accuracy",
        ylabel="Max accuracy (%)",
        title="Max Accuracy vs Generation",
        filename="max_accuracy_vs_generation.png",
        show=args.show,
    )
    plot_metric(
        rows,
        metric_key="avg_similarity",
        ylabel="Average class vector distance (reported similarity)",
        title="Average Class Vector Distance vs Generation",
        filename="avg_class_vector_distance_vs_generation.png",
        show=args.show,
    )
    plot_metric(
        rows,
        metric_key="max_similarity",
        ylabel="Max class vector distance (reported similarity)",
        title="Max Class Vector Distance vs Generation",
        filename="max_class_vector_distance_vs_generation.png",
        show=args.show,
    )

    summary_rows = summarize_dataset_maxima(runs)
    write_summary_csv(summary_rows, SUMMARY_CSV)

    print("\nWrote:")
    print(f"- {METRICS_CSV}")
    print(f"- {SUMMARY_CSV}")
    print(f"- {PLOTS_DIR}")


if __name__ == "__main__":
    main()
