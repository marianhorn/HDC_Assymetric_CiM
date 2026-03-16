import argparse
import csv
import os
import re
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "run_manifest.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_outputs", "new_selected_by_rate")

DATASET_RE = re.compile(r"Model for dataset #(\d+)")
GEN_RE = re.compile(r"GA generation (\d+)/(\d+)")
NEW_SEL_RE = re.compile(r"new selected individuals:\s*(\d+)/(\d+)")


def round_rate(value):
    return round(float(value) + 1e-12, 1)


def rate_matches(rate_a, rate_b, tol=1e-9):
    return abs(float(rate_a) - float(rate_b)) <= tol


def format_rate_tag(value):
    return f"{value:.1f}".replace(".", "p")


def load_matching_runs(manifest_path, mutation_rate, crossover_rate):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    wanted_mut = round_rate(mutation_rate)
    wanted_cx = round_rate(crossover_rate)

    matches = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_mut = round_rate(row["mutation_rate"])
            run_cx = round_rate(row["crossover_rate"])
            if rate_matches(run_mut, wanted_mut) and rate_matches(run_cx, wanted_cx):
                matches.append(
                    {
                        "run_index": int(row["run_index"]),
                        "seed": int(row["seed"]),
                        "crossover_rate": run_cx,
                        "mutation_rate": run_mut,
                        "log_file": row["log_file"],
                    }
                )

    matches.sort(key=lambda r: r["run_index"])
    if not matches:
        raise RuntimeError(
            f"No runs found for crossover={wanted_cx:.1f}, mutation={wanted_mut:.1f} "
            f"in {manifest_path}"
        )
    return matches


def parse_new_selected_from_log(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Missing log file: {log_path}")

    current_dataset = None
    current_generation = None
    rows = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m = DATASET_RE.search(line)
            if m:
                current_dataset = int(m.group(1))
                current_generation = None
                continue

            m = GEN_RE.search(line)
            if m and current_dataset is not None:
                current_generation = int(m.group(1))
                continue

            m = NEW_SEL_RE.search(line)
            if m and current_dataset is not None and current_generation is not None:
                new_selected = int(m.group(1))
                population = int(m.group(2))
                ratio = float(new_selected) / float(population) if population > 0 else float("nan")
                rows.append(
                    {
                        "dataset": current_dataset,
                        "generation": current_generation,
                        "new_selected": new_selected,
                        "population": population,
                        "new_selected_ratio": ratio,
                    }
                )

    if not rows:
        raise RuntimeError(f"No 'new selected individuals' entries found in: {log_path}")
    return rows


def collect_rows_for_rate(runs, log_dir):
    rows = []
    missing_logs = []

    for run in runs:
        log_path = os.path.join(log_dir, run["log_file"])
        if not os.path.exists(log_path):
            missing_logs.append(run["log_file"])
            continue

        parsed = parse_new_selected_from_log(log_path)
        for item in parsed:
            rows.append(
                {
                    "run_index": run["run_index"],
                    "seed": run["seed"],
                    "crossover_rate": run["crossover_rate"],
                    "mutation_rate": run["mutation_rate"],
                    **item,
                }
            )

    if not rows:
        raise RuntimeError("No rows parsed from matching runs.")
    return rows, missing_logs


def aggregate_rows(rows, crossover_rate, mutation_rate):
    grouped = defaultdict(lambda: {"count_values": [], "ratio_values": [], "population_values": []})
    for row in rows:
        key = (row["dataset"], row["generation"])
        grouped[key]["count_values"].append(float(row["new_selected"]))
        grouped[key]["ratio_values"].append(float(row["new_selected_ratio"]))
        grouped[key]["population_values"].append(float(row["population"]))

    summary = []
    for (dataset, generation), values in sorted(grouped.items()):
        count_arr = np.array(values["count_values"], dtype=float)
        ratio_arr = np.array(values["ratio_values"], dtype=float)
        pop_arr = np.array(values["population_values"], dtype=float)
        summary.append(
            {
                "crossover_rate": crossover_rate,
                "mutation_rate": mutation_rate,
                "dataset": dataset,
                "generation": generation,
                "count": int(count_arr.size),
                "mean_new_selected": float(np.mean(count_arr)),
                "std_new_selected": float(np.std(count_arr)),
                "min_new_selected": float(np.min(count_arr)),
                "max_new_selected": float(np.max(count_arr)),
                "mean_new_selected_ratio": float(np.mean(ratio_arr)),
                "std_new_selected_ratio": float(np.std(ratio_arr)),
                "min_new_selected_ratio": float(np.min(ratio_arr)),
                "max_new_selected_ratio": float(np.max(ratio_arr)),
                "mean_population": float(np.mean(pop_arr)),
            }
        )
    return summary


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_new_selected(summary_rows, crossover_rate, mutation_rate, num_runs, use_ratio, show, out_path):
    if not summary_rows:
        raise RuntimeError("No aggregated rows to plot.")

    per_dataset = defaultdict(lambda: {"x": [], "mean": [], "std": []})
    for row in summary_rows:
        ds = row["dataset"]
        per_dataset[ds]["x"].append(row["generation"])
        if use_ratio:
            per_dataset[ds]["mean"].append(row["mean_new_selected_ratio"])
            per_dataset[ds]["std"].append(row["std_new_selected_ratio"])
        else:
            per_dataset[ds]["mean"].append(row["mean_new_selected"])
            per_dataset[ds]["std"].append(row["std_new_selected"])

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for dataset in sorted(per_dataset.keys()):
        x = np.array(per_dataset[dataset]["x"], dtype=float)
        mean = np.array(per_dataset[dataset]["mean"], dtype=float)
        std = np.array(per_dataset[dataset]["std"], dtype=float)
        ax.plot(x, mean, marker="o", linewidth=1.8, markersize=3, label=f"Dataset {dataset}")
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)

    ylabel = "New selected / population" if use_ratio else "New selected individuals"
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(
        "Newly Accepted Individuals vs Generation "
        f"(cx={crossover_rate:.1f}, mut={mutation_rate:.1f}, runs={num_runs})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot new selected individuals vs generation for one specific "
            "(crossover_rate, mutation_rate) pair from crossMutRatesOptimization logs."
        )
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        required=True,
        help="Crossover rate to filter (e.g. 0.7).",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        required=True,
        help="Mutation rate to filter (e.g. 0.2).",
    )
    parser.add_argument(
        "--ratio",
        action="store_true",
        help="Plot new_selected/population instead of absolute new_selected.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively in addition to saving.",
    )
    args = parser.parse_args()

    global np, plt
    try:
        import numpy as np_mod
        import matplotlib.pyplot as plt_mod
    except Exception as exc:
        raise RuntimeError(
            "Missing Python dependencies. Install with: pip install numpy matplotlib"
        ) from exc
    np = np_mod
    plt = plt_mod

    crossover_rate = round_rate(args.crossover_rate)
    mutation_rate = round_rate(args.mutation_rate)

    runs = load_matching_runs(
        manifest_path=MANIFEST_PATH,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
    )
    rows, missing_logs = collect_rows_for_rate(runs, LOG_DIR)
    summary_rows = aggregate_rows(
        rows=rows,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
    )

    tag = f"cx{format_rate_tag(crossover_rate)}_mut{format_rate_tag(mutation_rate)}"
    mode = "ratio" if args.ratio else "count"

    detailed_csv = os.path.join(OUTPUT_DIR, f"new_selected_rows_{tag}.csv")
    summary_csv = os.path.join(OUTPUT_DIR, f"new_selected_summary_{tag}.csv")
    plot_path = os.path.join(OUTPUT_DIR, f"new_selected_plot_{tag}_{mode}.png")

    write_csv(
        detailed_csv,
        fieldnames=[
            "run_index",
            "seed",
            "crossover_rate",
            "mutation_rate",
            "dataset",
            "generation",
            "new_selected",
            "population",
            "new_selected_ratio",
        ],
        rows=rows,
    )
    write_csv(
        summary_csv,
        fieldnames=[
            "crossover_rate",
            "mutation_rate",
            "dataset",
            "generation",
            "count",
            "mean_new_selected",
            "std_new_selected",
            "min_new_selected",
            "max_new_selected",
            "mean_new_selected_ratio",
            "std_new_selected_ratio",
            "min_new_selected_ratio",
            "max_new_selected_ratio",
            "mean_population",
        ],
        rows=summary_rows,
    )

    plot_new_selected(
        summary_rows=summary_rows,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        num_runs=len(runs),
        use_ratio=args.ratio,
        show=args.show,
        out_path=plot_path,
    )

    seeds = sorted({r["seed"] for r in runs})
    print(f"Matched runs: {len(runs)}")
    print(f"Seeds: {seeds}")
    if missing_logs:
        print(f"Warning: missing {len(missing_logs)} logs (skipped).")
    print(f"Wrote detailed rows: {detailed_csv}")
    print(f"Wrote summary rows: {summary_csv}")
    print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
