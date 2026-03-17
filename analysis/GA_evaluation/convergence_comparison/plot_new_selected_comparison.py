import argparse
import csv
import math
import os
import re
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADAPTIVE_DIR = os.path.join(BASE_DIR, "myModel5")
DEFAULT_NAIVE_DIR = os.path.join(BASE_DIR, "myModel6")
OUTPUT_DIR = os.path.join(BASE_DIR, "comparison_plots")

np = None
plt = None

SEED_FROM_NAME_RE = re.compile(r"run_seed_(\d+)\.txt$")
DATASET_RE = re.compile(r"Model for dataset #(\d+)")
POST_TEST_RE = re.compile(r"Dataset\s+(\d+)\s+post-opt test accuracy:\s*([0-9.]+)%")
POST_VAL_START_RE = re.compile(r"Evaluating post-optimization model on validation set\.")
TEST_ACC_RE = re.compile(r"Testing accuracy:\s*([0-9.]+)%")


def read_rows(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing metrics file: {csv_path}")
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "dataset": int(row["dataset"]),
                    "generation": int(row["generation"]),
                    "new_selected": float(row["new_selected"]),
                    "population": float(row["population"]) if row["population"] else float("nan"),
                }
            )
    if not rows:
        raise RuntimeError(f"No rows found in: {csv_path}")
    return rows


def aggregate_metric(rows, use_ratio):
    grouped = defaultdict(list)  # (dataset, generation) -> [value]
    for row in rows:
        if use_ratio:
            pop = row["population"]
            if not math.isfinite(pop) or pop <= 0:
                continue
            value = row["new_selected"] / pop
        else:
            value = row["new_selected"]
        grouped[(row["dataset"], row["generation"])].append(value)

    out = defaultdict(lambda: {"x": [], "mean": [], "std": []})
    for (dataset, generation), values in sorted(grouped.items()):
        arr = np.array(values, dtype=float)
        out[dataset]["x"].append(generation)
        out[dataset]["mean"].append(float(np.mean(arr)))
        out[dataset]["std"].append(float(np.std(arr)))
    return out


def write_comparison_csv(out_path, adaptive_series, naive_series):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "generation",
                "adaptive_mean",
                "adaptive_std",
                "naive_mean",
                "naive_std",
            ]
        )

        datasets = sorted(set(adaptive_series.keys()) | set(naive_series.keys()))
        for dataset in datasets:
            gen_to_ad = {}
            if dataset in adaptive_series:
                for i, gen in enumerate(adaptive_series[dataset]["x"]):
                    gen_to_ad[gen] = (
                        adaptive_series[dataset]["mean"][i],
                        adaptive_series[dataset]["std"][i],
                    )
            gen_to_nv = {}
            if dataset in naive_series:
                for i, gen in enumerate(naive_series[dataset]["x"]):
                    gen_to_nv[gen] = (
                        naive_series[dataset]["mean"][i],
                        naive_series[dataset]["std"][i],
                    )

            generations = sorted(set(gen_to_ad.keys()) | set(gen_to_nv.keys()))
            for gen in generations:
                ad = gen_to_ad.get(gen, (float("nan"), float("nan")))
                nv = gen_to_nv.get(gen, (float("nan"), float("nan")))
                writer.writerow([dataset, gen, ad[0], ad[1], nv[0], nv[1]])


def plot_dataset_subplots(adaptive_series, naive_series, use_ratio, show, adaptive_label, naive_label):
    datasets = sorted(set(adaptive_series.keys()) | set(naive_series.keys()))
    if not datasets:
        raise RuntimeError("No overlapping datasets found to plot.")

    cols = 2
    rows = int(math.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows), sharex=False)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    ylabel = "New selected / population" if use_ratio else "New selected individuals"

    for idx, dataset in enumerate(datasets):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        if dataset in adaptive_series:
            x = np.array(adaptive_series[dataset]["x"], dtype=float)
            y = np.array(adaptive_series[dataset]["mean"], dtype=float)
            s = np.array(adaptive_series[dataset]["std"], dtype=float)
            ax.plot(x, y, label=adaptive_label, color="#1f77b4", linewidth=2.0)
            ax.fill_between(x, y - s, y + s, color="#1f77b4", alpha=0.2)

        if dataset in naive_series:
            x = np.array(naive_series[dataset]["x"], dtype=float)
            y = np.array(naive_series[dataset]["mean"], dtype=float)
            s = np.array(naive_series[dataset]["std"], dtype=float)
            ax.plot(x, y, label=naive_label, color="#d62728", linewidth=2.0)
            ax.fill_between(x, y - s, y + s, color="#d62728", alpha=0.2)

        ax.set_title(f"Dataset {dataset}")
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    total_axes = rows * cols
    for idx in range(len(datasets), total_axes):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    fig.suptitle("Convergence Comparison: New Individuals from Offspring", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix = "ratio" if use_ratio else "count"
    out_path = os.path.join(OUTPUT_DIR, f"new_selected_comparison_by_dataset_{suffix}.png")
    fig.savefig(out_path, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def parse_accuracies_from_log(log_path):
    run_data = defaultdict(lambda: {"post_test_acc": None, "post_val_acc": None})
    current_dataset = None
    awaiting_post_val_for_dataset = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m = DATASET_RE.search(line)
            if m:
                current_dataset = int(m.group(1))
                awaiting_post_val_for_dataset = None
                continue

            m = POST_TEST_RE.search(line)
            if m:
                ds = int(m.group(1))
                run_data[ds]["post_test_acc"] = float(m.group(2))
                continue

            if POST_VAL_START_RE.search(line):
                if current_dataset is not None:
                    awaiting_post_val_for_dataset = current_dataset
                continue

            m = TEST_ACC_RE.search(line)
            if m and awaiting_post_val_for_dataset is not None:
                run_data[awaiting_post_val_for_dataset]["post_val_acc"] = float(m.group(1))
                awaiting_post_val_for_dataset = None
                continue

    return run_data


def collect_accuracy_runs(model_dir):
    logs_dir = os.path.join(model_dir, "logs")
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(f"Missing logs directory: {logs_dir}")

    out = {}
    for name in sorted(os.listdir(logs_dir)):
        m = SEED_FROM_NAME_RE.search(name)
        if not m:
            continue
        seed = int(m.group(1))
        log_path = os.path.join(logs_dir, name)
        out[seed] = parse_accuracies_from_log(log_path)

    if not out:
        raise RuntimeError(f"No run_seed_<n>.txt logs found in: {logs_dir}")
    return out


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())


def write_accuracy_diff_csv(path, rows, model_a_label, model_b_label):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "dataset",
                f"{model_a_label}_test_acc",
                f"{model_b_label}_test_acc",
                f"test_diff_{model_a_label}_minus_{model_b_label}",
                f"{model_a_label}_val_acc",
                f"{model_b_label}_val_acc",
                f"val_diff_{model_a_label}_minus_{model_b_label}",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["seed"],
                    r["dataset"],
                    r["model_a_test"],
                    r["model_b_test"],
                    r["diff_test"],
                    r["model_a_val"],
                    r["model_b_val"],
                    r["diff_val"],
                ]
            )


def print_accuracy_diff_table(rows, model_a_label, model_b_label):
    print(f"\nAccuracy Differences ({model_a_label} - {model_b_label}) per run and dataset")
    print(
        f"seed | dataset | {model_a_label}_test | {model_b_label}_test | diff_test | "
        f"{model_a_label}_val | {model_b_label}_val | diff_val"
    )
    print(
        "-----+---------+----------+------------+-----------+---------+-----------+---------"
    )
    for r in rows:
        model_a_test = f"{r['model_a_test']:.3f}" if r["model_a_test"] is not None else "n/a"
        model_b_test = f"{r['model_b_test']:.3f}" if r["model_b_test"] is not None else "n/a"
        diff_test = f"{r['diff_test']:+.3f}" if r["diff_test"] is not None else "n/a"
        model_a_val = f"{r['model_a_val']:.3f}" if r["model_a_val"] is not None else "n/a"
        model_b_val = f"{r['model_b_val']:.3f}" if r["model_b_val"] is not None else "n/a"
        diff_val = f"{r['diff_val']:+.3f}" if r["diff_val"] is not None else "n/a"
        print(
            f"{r['seed']:4d} | {r['dataset']:7d} | {model_a_test:8s} | {model_b_test:10s} | "
            f"{diff_test:9s} | {model_a_val:7s} | {model_b_val:9s} | {diff_val}"
        )


def main():
    global np, plt
    parser = argparse.ArgumentParser(
        description="Compare newly added individuals per generation between two convergence runs."
    )
    parser.add_argument(
        "--adaptive-dir",
        default=DEFAULT_ADAPTIVE_DIR,
        help="Directory of first model run (default: myModel5).",
    )
    parser.add_argument(
        "--naive-dir",
        default=DEFAULT_NAIVE_DIR,
        help="Directory of second model run (default: myModel6).",
    )
    parser.add_argument(
        "--ratio",
        action="store_true",
        help="Plot fraction new_selected/population instead of absolute count.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively in addition to saving.",
    )
    args = parser.parse_args()

    try:
        import numpy as np_mod
        import matplotlib.pyplot as plt_mod
    except Exception as exc:
        raise RuntimeError(
            "Missing Python dependencies. Install with: pip install numpy matplotlib"
        ) from exc
    np = np_mod
    plt = plt_mod

    adaptive_csv = os.path.join(args.adaptive_dir, "generation_metrics.csv")
    naive_csv = os.path.join(args.naive_dir, "generation_metrics.csv")

    adaptive_rows = read_rows(adaptive_csv)
    naive_rows = read_rows(naive_csv)

    adaptive_series = aggregate_metric(adaptive_rows, use_ratio=args.ratio)
    naive_series = aggregate_metric(naive_rows, use_ratio=args.ratio)
    adaptive_label = os.path.basename(os.path.normpath(args.adaptive_dir)) or "Model A"
    naive_label = os.path.basename(os.path.normpath(args.naive_dir)) or "Model B"

    plot_path = plot_dataset_subplots(
        adaptive_series,
        naive_series,
        use_ratio=args.ratio,
        show=args.show,
        adaptive_label=adaptive_label,
        naive_label=naive_label,
    )

    comparison_csv = os.path.join(
        OUTPUT_DIR,
        "new_selected_comparison_by_dataset.csv",
    )
    write_comparison_csv(comparison_csv, adaptive_series, naive_series)

    model_a_runs = collect_accuracy_runs(args.adaptive_dir)
    model_b_runs = collect_accuracy_runs(args.naive_dir)
    model_a_label = sanitize_name(adaptive_label)
    model_b_label = sanitize_name(naive_label)
    rows = []
    for seed in sorted(set(model_a_runs.keys()) | set(model_b_runs.keys())):
        model_a_ds = model_a_runs.get(seed, {})
        model_b_ds = model_b_runs.get(seed, {})
        datasets = sorted(set(model_a_ds.keys()) | set(model_b_ds.keys()))
        for dataset in datasets:
            model_a_test = model_a_ds.get(dataset, {}).get("post_test_acc")
            model_b_test = model_b_ds.get(dataset, {}).get("post_test_acc")
            model_a_val = model_a_ds.get(dataset, {}).get("post_val_acc")
            model_b_val = model_b_ds.get(dataset, {}).get("post_val_acc")
            rows.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "model_a_test": model_a_test,
                    "model_b_test": model_b_test,
                    "diff_test": (model_a_test - model_b_test)
                    if model_a_test is not None and model_b_test is not None
                    else None,
                    "model_a_val": model_a_val,
                    "model_b_val": model_b_val,
                    "diff_val": (model_a_val - model_b_val)
                    if model_a_val is not None and model_b_val is not None
                    else None,
                }
            )
    rows.sort(key=lambda r: (r["seed"], r["dataset"]))
    print_accuracy_diff_table(rows, adaptive_label, naive_label)
    accuracy_diff_csv = os.path.join(
        OUTPUT_DIR,
        f"accuracy_difference_{model_a_label}_vs_{model_b_label}_by_seed_dataset.csv",
    )
    write_accuracy_diff_csv(accuracy_diff_csv, rows, model_a_label, model_b_label)

    print(f"Saved plot: {plot_path}")
    print(f"Saved comparison CSV: {comparison_csv}")
    print(f"Saved accuracy-diff CSV: {accuracy_diff_csv}")


if __name__ == "__main__":
    main()
