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


def parse_info(info_str):
    info = {}
    if not info_str:
        return info
    for token in info_str.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        info[key.strip()] = value.strip()
    return info


def detect_scope(info):
    if "scope" in info:
        return info["scope"]
    if "dataset" in info:
        return "dataset"
    return "overall"


def load_phase_rows(csv_path, phase, metric, num_levels_filter=None):
    grouped = defaultdict(list)
    meta = {
        "n_gram_size": set(),
        "validation_ratio": set(),
        "use_genetic_item_memory": set(),
        "mutation_rate": set(),
    }

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            info = parse_info(row.get("info", ""))
            row_phase = info.get("phase")
            if row_phase != phase:
                continue

            scope = detect_scope(info)
            dataset = None
            if scope == "dataset":
                if "dataset" not in info:
                    continue
                try:
                    dataset = int(info["dataset"])
                except ValueError:
                    continue

            try:
                num_levels = int(row["num_levels"])
                vector_dimension = int(row["vector_dimension"])
                accuracy = float(row[metric])
            except (KeyError, ValueError):
                continue

            if num_levels_filter is not None and num_levels != num_levels_filter:
                continue

            key = (scope, dataset, num_levels, vector_dimension)
            grouped[key].append(accuracy)

            if row.get("n_gram_size"):
                meta["n_gram_size"].add(row["n_gram_size"])
            if row.get("validation_ratio"):
                meta["validation_ratio"].add(row["validation_ratio"])
            if row.get("use_genetic_item_memory"):
                meta["use_genetic_item_memory"].add(row["use_genetic_item_memory"])
            if row.get("ga_mutation_rate"):
                meta["mutation_rate"].add(row["ga_mutation_rate"])

    averaged = {key: float(np.mean(values)) for key, values in grouped.items()}
    return averaged, meta


def compute_equivalent_dim_records(ga_map, baseline_map, tolerance):
    baseline_by_group = defaultdict(list)
    for (scope, dataset, num_levels, dim), acc in baseline_map.items():
        group_key = (scope, dataset, num_levels)
        baseline_by_group[group_key].append((dim, acc))

    for group_key in baseline_by_group:
        baseline_by_group[group_key].sort(key=lambda x: x[0])

    records = []
    for (scope, dataset, num_levels, ga_dim), ga_acc in ga_map.items():
        group_key = (scope, dataset, num_levels)
        candidates = baseline_by_group.get(group_key)
        if not candidates:
            continue

        eq_dim = None
        eq_acc = None
        threshold = ga_acc - tolerance
        for dim, acc in candidates:
            if acc >= threshold:
                eq_dim = dim
                eq_acc = acc
                break

        if eq_dim is None:
            continue

        reduction = 1.0 - (float(ga_dim) / float(eq_dim))
        records.append(
            {
                "scope": scope,
                "dataset": dataset,
                "num_levels": num_levels,
                "ga_dim": ga_dim,
                "ga_acc": ga_acc,
                "baseline_eq_dim": eq_dim,
                "baseline_eq_acc": eq_acc,
                "reduction": reduction,
            }
        )

    return records


def aggregate_by_dim(records):
    grouped = defaultdict(list)
    for rec in records:
        key = (rec["scope"], rec["dataset"], rec["ga_dim"])
        grouped[key].append(rec)

    agg = []
    for key, vals in grouped.items():
        scope, dataset, ga_dim = key
        agg.append(
            {
                "scope": scope,
                "dataset": dataset,
                "ga_dim": ga_dim,
                "mean_eq_dim": float(np.mean([v["baseline_eq_dim"] for v in vals])),
                "mean_reduction": float(np.mean([v["reduction"] for v in vals])),
                "n": len(vals),
            }
        )

    agg.sort(key=lambda x: (x["scope"], -1 if x["dataset"] is None else x["dataset"], x["ga_dim"]))
    return agg


def write_summary_csv(records, agg, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scope",
                "dataset",
                "num_levels",
                "ga_dim",
                "ga_acc",
                "baseline_eq_dim",
                "baseline_eq_acc",
                "reduction_fraction",
                "reduction_percent",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec["scope"],
                    rec["dataset"] if rec["dataset"] is not None else "",
                    rec["num_levels"],
                    rec["ga_dim"],
                    f"{rec['ga_acc']:.6f}",
                    rec["baseline_eq_dim"],
                    f"{rec['baseline_eq_acc']:.6f}",
                    f"{rec['reduction']:.6f}",
                    f"{rec['reduction'] * 100.0:.3f}",
                ]
            )

        writer.writerow([])
        writer.writerow(["scope", "dataset", "ga_dim", "mean_eq_dim", "mean_reduction_percent", "n"])
        for row in agg:
            writer.writerow(
                [
                    row["scope"],
                    row["dataset"] if row["dataset"] is not None else "",
                    row["ga_dim"],
                    f"{row['mean_eq_dim']:.3f}",
                    f"{row['mean_reduction'] * 100.0:.3f}",
                    row["n"],
                ]
            )


def plot_equivalent_dim(agg_rows, out_path):
    overall = [r for r in agg_rows if r["scope"] == "overall"]
    datasets = [r for r in agg_rows if r["scope"] == "dataset"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if overall:
        xs = sorted({r["ga_dim"] for r in overall})
        ys = [next(r["mean_eq_dim"] for r in overall if r["ga_dim"] == x) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label="Overall")

    dataset_ids = sorted({r["dataset"] for r in datasets})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, dataset_id in enumerate(dataset_ids):
        rows = [r for r in datasets if r["dataset"] == dataset_id]
        xs = sorted({r["ga_dim"] for r in rows})
        ys = [next(r["mean_eq_dim"] for r in rows if r["ga_dim"] == x) for x in xs]
        color = colors[idx % len(colors)]
        ax.plot(xs, ys, marker="o", linewidth=1.8, linestyle="--", color=color, label=f"Dataset {dataset_id}")

    all_dims = sorted({r["ga_dim"] for r in agg_rows})
    if all_dims:
        ax.plot(all_dims, all_dims, color="black", linewidth=1.2, alpha=0.7, label="y = x")

    ax.set_title("Equivalent Non-GA Dimension")
    ax.set_xlabel("GA vector dimension")
    ax.set_ylabel("Non-GA dimension needed for same accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if overall:
        xs = sorted({r["ga_dim"] for r in overall})
        ys = [next(r["mean_reduction"] for r in overall if r["ga_dim"] == x) * 100.0 for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label="Overall")

    for idx, dataset_id in enumerate(dataset_ids):
        rows = [r for r in datasets if r["dataset"] == dataset_id]
        xs = sorted({r["ga_dim"] for r in rows})
        ys = [next(r["mean_reduction"] for r in rows if r["ga_dim"] == x) * 100.0 for x in xs]
        color = colors[idx % len(colors)]
        ax.plot(xs, ys, marker="o", linewidth=1.8, linestyle="--", color=color, label=f"Dataset {dataset_id}")

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_title("Dimension Reduction vs Non-GA")
    ax.set_xlabel("GA vector dimension")
    ax.set_ylabel("Reduction [%] (1 - GA_dim / NonGA_eq_dim)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    return fig


def print_meta_warning(name, meta):
    print(
        f"{name}: n_gram_size={sorted(meta['n_gram_size'])}, "
        f"validation_ratio={sorted(meta['validation_ratio'])}, "
        f"use_genetic_item_memory={sorted(meta['use_genetic_item_memory'])}, "
        f"ga_mutation_rate={sorted(meta['mutation_rate'])}"
    )


def resolve_default_csv(base_dir):
    candidates = [
        os.path.join(base_dir, "results", "results_all.csv"),
        os.path.join(base_dir, "results_all.csv"),
        os.path.join(base_dir, "results.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = resolve_default_csv(base_dir)
    parser = argparse.ArgumentParser(
        description="Plot how much vector dimension GA can save vs preopt-test baseline from the same CSV."
    )
    parser.add_argument(
        "--csv",
        default=default_csv,
        help=(
            "Input CSV path containing both preopt-test and postopt-test rows "
            "(auto-default: results/results_all.csv if present)."
        ),
    )
    parser.add_argument(
        "--ga-phase",
        default="postopt-test",
        help="Phase to read as GA-optimized results (default: postopt-test).",
    )
    parser.add_argument(
        "--baseline-phase",
        default="preopt-test",
        help="Phase to read as non-GA baseline (default: preopt-test).",
    )
    parser.add_argument(
        "--metric",
        default="overall_accuracy",
        help="Accuracy metric column (default: overall_accuracy).",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=91,
        help="Use only this NUM_LEVELS value (default: 61).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Accuracy tolerance when matching equivalent baseline dim (default: 0.0).",
    )
    parser.add_argument(
        "--out-dir",
        default=base_dir,
        help="Output directory for plots and summary CSV.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Only save plot to disk (do not open interactive window).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Missing input CSV: {args.csv}")

    os.makedirs(args.out_dir, exist_ok=True)

    ga_map, ga_meta = load_phase_rows(
        args.csv, args.ga_phase, args.metric, num_levels_filter=args.num_levels
    )
    baseline_map, baseline_meta = load_phase_rows(
        args.csv, args.baseline_phase, args.metric, num_levels_filter=args.num_levels
    )

    print(f"Input CSV: {args.csv}")
    print(f"NUM_LEVELS filter: {args.num_levels}")
    print(f"GA rows used: {len(ga_map)} (phase={args.ga_phase})")
    print(f"Baseline rows used: {len(baseline_map)} (phase={args.baseline_phase})")
    print_meta_warning("GA meta", ga_meta)
    print_meta_warning("Baseline meta", baseline_meta)

    records = compute_equivalent_dim_records(ga_map, baseline_map, tolerance=args.tolerance)
    if not records:
        raise RuntimeError(
            "No matched records found. Check phase names and whether num_levels/dim grids overlap."
        )

    agg = aggregate_by_dim(records)
    plot_out = os.path.join(args.out_dir, "ga_dimension_reduction_vs_no_ga.png")
    summary_out = os.path.join(args.out_dir, "ga_dimension_reduction_vs_no_ga.csv")

    fig = plot_equivalent_dim(agg, plot_out)
    write_summary_csv(records, agg, summary_out)

    print(f"Matched records: {len(records)}")
    print(f"Saved plot: {plot_out}")
    print(f"Saved summary: {summary_out}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
