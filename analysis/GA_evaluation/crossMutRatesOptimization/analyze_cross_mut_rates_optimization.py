import argparse
import csv
import math
import os
import re
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(BASE_DIR, "results.csv")
MANIFEST_CSV = os.path.join(BASE_DIR, "run_manifest.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_outputs")

np = None
plt = None

DATASET_RE = re.compile(r"Model for dataset #(\d+)")
GEN_RE = re.compile(r"GA generation (\d+)/(\d+)")
IND_RE = re.compile(r"individual \d+/\d+ accuracy:\s*([0-9.]+)%, similarity:\s*([0-9.]+)")
NEW_SEL_RE = re.compile(r"new selected individuals:\s*(\d+)/(\d+)")


def parse_info_field(info_text):
    out = {}
    for part in info_text.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def round_rate(value):
    return round(float(value) + 1e-12, 1)


def nanmean(values):
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(np.array(vals, dtype=float)))


def nanstd(values):
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.std(np.array(vals, dtype=float)))


def load_manifest(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing manifest: {path}")
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "run_index": int(row["run_index"]),
                    "seed": int(row["seed"]),
                    "crossover_rate": round_rate(row["crossover_rate"]),
                    "mutation_rate": round_rate(row["mutation_rate"]),
                    "log_file": row["log_file"],
                }
            )
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    rows.sort(key=lambda r: r["run_index"])
    return rows


def load_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing results CSV: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Results CSV is empty: {path}")
    return rows


def enrich_results_with_manifest(manifest_rows, result_rows):
    run_count = len(manifest_rows)
    if len(result_rows) % run_count != 0:
        raise RuntimeError(
            "Could not align results to manifest by run order: "
            f"{len(result_rows)} rows for {run_count} runs."
        )
    rows_per_run = len(result_rows) // run_count
    enriched = []
    mutation_mismatch = 0

    for run_pos, run in enumerate(manifest_rows):
        base = run_pos * rows_per_run
        for local_i in range(rows_per_run):
            row = result_rows[base + local_i]
            info = parse_info_field(row.get("info", ""))
            scope = info.get("scope")
            phase = info.get("phase")
            dataset = info.get("dataset")
            dataset_int = int(dataset) if dataset is not None and dataset.isdigit() else None
            metric_value = safe_float(row.get("overall_accuracy", "nan"))
            class_avg_value = safe_float(row.get("class_average_accuracy", "nan"))
            class_sim = safe_float(row.get("class_vector_similarity", "nan"))
            ga_mut = safe_float(row.get("ga_mutation_rate", "nan"))
            if math.isfinite(ga_mut) and abs(ga_mut - run["mutation_rate"]) > 1e-6:
                mutation_mismatch += 1

            enriched.append(
                {
                    "run_index": run["run_index"],
                    "seed": run["seed"],
                    "crossover_rate": run["crossover_rate"],
                    "mutation_rate": run["mutation_rate"],
                    "scope": scope,
                    "phase": phase,
                    "dataset": dataset_int,
                    "overall_accuracy": metric_value,
                    "class_average_accuracy": class_avg_value,
                    "class_vector_similarity": class_sim,
                }
            )

    if mutation_mismatch > 0:
        raise RuntimeError(
            "Mutation-rate mismatch between manifest and results rows. "
            f"Mismatched rows: {mutation_mismatch}"
        )
    return enriched, rows_per_run


def aggregate_accuracy(enriched_rows, metric_key, scope, phase, dataset=None):
    grouped = defaultdict(list)
    for row in enriched_rows:
        if row["scope"] != scope:
            continue
        if row["phase"] != phase:
            continue
        if dataset is not None and row["dataset"] != dataset:
            continue
        value = row.get(metric_key, float("nan"))
        if not math.isfinite(value):
            continue
        grouped[(row["crossover_rate"], row["mutation_rate"])].append(100.0 * value)

    summary = {}
    for key, values in grouped.items():
        arr = np.array(values, dtype=float)
        summary[key] = {
            "count": int(arr.size),
            "mean_accuracy_pct": float(np.mean(arr)),
            "std_accuracy_pct": float(np.std(arr)),
            "min_accuracy_pct": float(np.min(arr)),
            "max_accuracy_pct": float(np.max(arr)),
        }
    return summary


def sorted_rate_grid(manifest_rows):
    crossover_rates = sorted({r["crossover_rate"] for r in manifest_rows})
    mutation_rates = sorted({r["mutation_rate"] for r in manifest_rows})
    return crossover_rates, mutation_rates


def summary_to_grid(summary, crossover_rates, mutation_rates, value_key):
    z = np.full((len(crossover_rates), len(mutation_rates)), np.nan, dtype=float)
    for iy, cx in enumerate(crossover_rates):
        for ix, mut in enumerate(mutation_rates):
            record = summary.get((cx, mut))
            if record is None:
                continue
            value = record.get(value_key, float("nan"))
            if math.isfinite(value):
                z[iy, ix] = value
    return z


def write_accuracy_summary_csv(path, summary):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "crossover_rate",
                "mutation_rate",
                "count",
                "mean_accuracy_pct",
                "std_accuracy_pct",
                "min_accuracy_pct",
                "max_accuracy_pct",
            ]
        )
        for (cx, mut), rec in sorted(summary.items()):
            writer.writerow(
                [
                    f"{cx:.1f}",
                    f"{mut:.1f}",
                    rec["count"],
                    f"{rec['mean_accuracy_pct']:.6f}",
                    f"{rec['std_accuracy_pct']:.6f}",
                    f"{rec['min_accuracy_pct']:.6f}",
                    f"{rec['max_accuracy_pct']:.6f}",
                ]
            )


def write_rows_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_3d_surface(crossover_rates, mutation_rates, z_grid, title, z_label, out_path, show):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = np.array(mutation_rates, dtype=float)
    y = np.array(crossover_rates, dtype=float)
    xx, yy = np.meshgrid(x, y)
    zz = np.ma.masked_invalid(z_grid)

    fig = plt.figure(figsize=(9, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, zz, cmap="viridis", edgecolor="none", antialiased=True)

    ax.set_xlabel("Mutation rate")
    ax.set_ylabel("Crossover rate")
    ax.set_zlabel(z_label)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label=z_label)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_heatmap(crossover_rates, mutation_rates, z_grid, title, cbar_label, out_path, show):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    im = ax.imshow(z_grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Mutation rate")
    ax.set_ylabel("Crossover rate")
    ax.set_xticks(range(len(mutation_rates)))
    ax.set_xticklabels([f"{v:.1f}" for v in mutation_rates], rotation=45)
    ax.set_yticks(range(len(crossover_rates)))
    ax.set_yticklabels([f"{v:.1f}" for v in crossover_rates])
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_log_generation_metrics(path):
    per_dataset = defaultdict(
        lambda: defaultdict(
            lambda: {
                "acc": [],
                "sim": [],
                "new_selected": None,
                "population": None,
            }
        )
    )

    current_dataset = None
    current_generation = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
                _ = per_dataset[current_dataset][current_generation]
                continue

            m = IND_RE.search(line)
            if m and current_dataset is not None and current_generation is not None:
                acc = float(m.group(1))
                sim = float(m.group(2))
                gen = per_dataset[current_dataset][current_generation]
                gen["acc"].append(acc)
                gen["sim"].append(sim)
                continue

            m = NEW_SEL_RE.search(line)
            if m and current_dataset is not None and current_generation is not None:
                gen = per_dataset[current_dataset][current_generation]
                gen["new_selected"] = int(m.group(1))
                gen["population"] = int(m.group(2))
                continue

    parsed_rows = []
    for dataset, generations in per_dataset.items():
        for generation, gen in sorted(generations.items()):
            if not gen["acc"]:
                continue
            pop = gen["population"]
            new_sel = gen["new_selected"]
            new_ratio = float("nan")
            if pop is not None and pop > 0 and new_sel is not None:
                new_ratio = float(new_sel) / float(pop)
            parsed_rows.append(
                {
                    "dataset": dataset,
                    "generation": generation,
                    "avg_accuracy": float(np.mean(np.array(gen["acc"], dtype=float))),
                    "max_accuracy": float(np.max(np.array(gen["acc"], dtype=float))),
                    "avg_similarity": float(np.mean(np.array(gen["sim"], dtype=float))),
                    "max_similarity": float(np.max(np.array(gen["sim"], dtype=float))),
                    "new_selected_ratio": new_ratio,
                }
            )
    return parsed_rows


def compute_dataset_convergence(gen_rows):
    gen_rows = sorted(gen_rows, key=lambda r: r["generation"])
    generations = np.array([r["generation"] for r in gen_rows], dtype=float)
    max_curve = np.array([r["max_accuracy"] for r in gen_rows], dtype=float)
    best_so_far = np.maximum.accumulate(max_curve)

    initial_best = float(best_so_far[0])
    final_best = float(best_so_far[-1])

    if final_best <= initial_best + 1e-12:
        t95_generation = int(generations[0])
    else:
        target = initial_best + 0.95 * (final_best - initial_best)
        hit_indices = np.where(best_so_far >= target)[0]
        t95_generation = int(generations[int(hit_indices[0])]) if hit_indices.size else int(generations[-1])

    last_generation = int(generations[-1])
    t95_fraction = float(t95_generation) / float(last_generation) if last_generation > 0 else float("nan")

    if len(best_so_far) == 1:
        best_auc = float(best_so_far[0])
    else:
        span = float(generations[-1] - generations[0])
        if span <= 0:
            best_auc = float(best_so_far[-1])
        else:
            best_auc = float(np.trapz(best_so_far, generations) / span)

    new_ratios = [r["new_selected_ratio"] for r in gen_rows if math.isfinite(r["new_selected_ratio"])]
    mean_new_ratio = float(np.mean(np.array(new_ratios, dtype=float))) if new_ratios else float("nan")
    final_new_ratio = float(gen_rows[-1]["new_selected_ratio"])

    return {
        "first_generation": int(generations[0]),
        "last_generation": last_generation,
        "initial_best_accuracy": initial_best,
        "final_best_accuracy": final_best,
        "t95_generation": t95_generation,
        "t95_fraction": t95_fraction,
        "best_accuracy_auc": best_auc,
        "mean_new_selected_ratio": mean_new_ratio,
        "final_new_selected_ratio": final_new_ratio,
    }


def build_convergence_rows(manifest_rows, log_dir):
    per_dataset_rows = []
    missing_logs = []

    for run in manifest_rows:
        log_path = os.path.join(log_dir, run["log_file"])
        if not os.path.exists(log_path):
            missing_logs.append(run["log_file"])
            continue

        parsed = parse_log_generation_metrics(log_path)
        by_dataset = defaultdict(list)
        for row in parsed:
            by_dataset[row["dataset"]].append(row)

        for dataset, gen_rows in by_dataset.items():
            conv = compute_dataset_convergence(gen_rows)
            per_dataset_rows.append(
                {
                    "run_index": run["run_index"],
                    "seed": run["seed"],
                    "crossover_rate": run["crossover_rate"],
                    "mutation_rate": run["mutation_rate"],
                    "dataset": dataset,
                    **conv,
                }
            )

    return per_dataset_rows, missing_logs


def aggregate_convergence(per_dataset_rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in per_dataset_rows:
        key = (row["crossover_rate"], row["mutation_rate"])
        grouped[key]["t95_generation"].append(row["t95_generation"])
        grouped[key]["t95_fraction"].append(row["t95_fraction"])
        grouped[key]["best_accuracy_auc"].append(row["best_accuracy_auc"])
        grouped[key]["final_best_accuracy"].append(row["final_best_accuracy"])
        grouped[key]["mean_new_selected_ratio"].append(row["mean_new_selected_ratio"])
        grouped[key]["final_new_selected_ratio"].append(row["final_new_selected_ratio"])

    summary = {}
    for key, vals in grouped.items():
        summary[key] = {
            "count": len(vals["t95_generation"]),
            "mean_t95_generation": nanmean(vals["t95_generation"]),
            "std_t95_generation": nanstd(vals["t95_generation"]),
            "mean_t95_fraction": nanmean(vals["t95_fraction"]),
            "std_t95_fraction": nanstd(vals["t95_fraction"]),
            "mean_best_accuracy_auc": nanmean(vals["best_accuracy_auc"]),
            "mean_final_best_accuracy": nanmean(vals["final_best_accuracy"]),
            "mean_new_selected_ratio": nanmean(vals["mean_new_selected_ratio"]),
            "final_new_selected_ratio": nanmean(vals["final_new_selected_ratio"]),
        }
    return summary


def build_tradeoff_rows(accuracy_summary, convergence_summary):
    rows = []
    common_keys = sorted(set(accuracy_summary.keys()) & set(convergence_summary.keys()))
    for key in common_keys:
        a = accuracy_summary[key]
        c = convergence_summary[key]
        rows.append(
            {
                "crossover_rate": key[0],
                "mutation_rate": key[1],
                "mean_accuracy_pct": a["mean_accuracy_pct"],
                "std_accuracy_pct": a["std_accuracy_pct"],
                "mean_t95_generation": c["mean_t95_generation"],
                "mean_t95_fraction": c["mean_t95_fraction"],
                "mean_best_accuracy_auc": c["mean_best_accuracy_auc"],
                "mean_new_selected_ratio": c["mean_new_selected_ratio"],
                "final_new_selected_ratio": c["final_new_selected_ratio"],
            }
        )
    return rows


def pareto_front_accuracy_vs_t95(rows):
    front = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            acc_not_worse = other["mean_accuracy_pct"] >= row["mean_accuracy_pct"]
            conv_not_worse = other["mean_t95_fraction"] <= row["mean_t95_fraction"]
            strictly_better = (
                other["mean_accuracy_pct"] > row["mean_accuracy_pct"]
                or other["mean_t95_fraction"] < row["mean_t95_fraction"]
            )
            if acc_not_worse and conv_not_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    front.sort(key=lambda r: (-r["mean_accuracy_pct"], r["mean_t95_fraction"]))
    return front


def plot_accuracy_vs_convergence(rows, out_path, show):
    x = np.array([100.0 * r["mean_t95_fraction"] for r in rows], dtype=float)
    y = np.array([r["mean_accuracy_pct"] for r in rows], dtype=float)
    mut = np.array([r["mutation_rate"] for r in rows], dtype=float)
    cx = np.array([r["crossover_rate"] for r in rows], dtype=float)
    sizes = 40.0 + 120.0 * cx

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    sc = ax.scatter(
        x,
        y,
        c=mut,
        s=sizes,
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xlabel("Convergence speed metric: t95 (% of total generations)")
    ax.set_ylabel("Mean post-opt test accuracy (%)")
    ax.set_title("Accuracy vs Convergence Speed (point size ~ crossover rate)")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mutation rate")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_top_accuracy(summary, top_k):
    rows = []
    for (cx, mut), rec in summary.items():
        rows.append(
            {
                "crossover_rate": cx,
                "mutation_rate": mut,
                **rec,
            }
        )
    rows.sort(key=lambda r: (-r["mean_accuracy_pct"], r["std_accuracy_pct"], r["mutation_rate"], r["crossover_rate"]))

    print("\nTop rate settings by mean achieved accuracy (post-opt test, averaged over seeds)")
    print("rank | crossover | mutation | mean_acc (%) | std_acc | n")
    print("-----+-----------+----------+--------------+---------+---")
    for i, row in enumerate(rows[:top_k], start=1):
        print(
            f"{i:4d} | {row['crossover_rate']:9.1f} | {row['mutation_rate']:8.1f} | "
            f"{row['mean_accuracy_pct']:12.4f} | {row['std_accuracy_pct']:7.4f} | {row['count']}"
        )
    return rows


def print_dataset_best(enriched_rows, metric_key, phase):
    datasets = sorted({r["dataset"] for r in enriched_rows if r["scope"] == "dataset" and r["dataset"] is not None})
    print(f"\nBest rates per dataset ({phase}, averaged over seeds)")
    print("dataset | crossover | mutation | mean_acc (%) | std_acc")
    print("--------+-----------+----------+--------------+--------")
    out_rows = []
    for dataset in datasets:
        summary = aggregate_accuracy(
            enriched_rows,
            metric_key=metric_key,
            scope="dataset",
            phase=phase,
            dataset=dataset,
        )
        if not summary:
            continue
        rows = []
        for (cx, mut), rec in summary.items():
            rows.append(
                {
                    "dataset": dataset,
                    "crossover_rate": cx,
                    "mutation_rate": mut,
                    **rec,
                }
            )
        rows.sort(key=lambda r: (-r["mean_accuracy_pct"], r["std_accuracy_pct"]))
        best = rows[0]
        print(
            f"{dataset:7d} | {best['crossover_rate']:9.1f} | {best['mutation_rate']:8.1f} | "
            f"{best['mean_accuracy_pct']:12.4f} | {best['std_accuracy_pct']:.4f}"
        )
        out_rows.append(best)
    return out_rows


def print_convergence_summary(convergence_summary, top_k):
    rows = []
    for (cx, mut), rec in convergence_summary.items():
        rows.append(
            {
                "crossover_rate": cx,
                "mutation_rate": mut,
                **rec,
            }
        )
    rows.sort(key=lambda r: (r["mean_t95_fraction"], -r["mean_final_best_accuracy"]))

    print("\nFastest convergence settings (lower t95 is faster)")
    print("rank | crossover | mutation | mean_t95_gen | t95_frac | final_best_acc")
    print("-----+-----------+----------+--------------+----------+---------------")
    for i, row in enumerate(rows[:top_k], start=1):
        print(
            f"{i:4d} | {row['crossover_rate']:9.1f} | {row['mutation_rate']:8.1f} | "
            f"{row['mean_t95_generation']:12.2f} | {row['mean_t95_fraction']:8.4f} | "
            f"{row['mean_final_best_accuracy']:13.4f}"
        )
    return rows


def print_pareto_front(front_rows):
    print("\nAccuracy/Convergence Pareto front (maximize accuracy, minimize t95)")
    print("crossover | mutation | mean_acc (%) | t95_frac | mean_t95_gen | mean_new_selected_ratio")
    print("----------+----------+--------------+----------+--------------+------------------------")
    for row in front_rows:
        print(
            f"{row['crossover_rate']:9.1f} | {row['mutation_rate']:8.1f} | "
            f"{row['mean_accuracy_pct']:12.4f} | {row['mean_t95_fraction']:8.4f} | "
            f"{row['mean_t95_generation']:12.2f} | {row['mean_new_selected_ratio']:.4f}"
        )


def main():
    global np, plt
    parser = argparse.ArgumentParser(
        description=(
            "Analyze cross/mutation-rate GA runs: average across seeds, plot 3D accuracy "
            "surface, and summarize best settings plus convergence behavior."
        )
    )
    parser.add_argument(
        "--metric",
        default="overall_accuracy",
        choices=["overall_accuracy", "class_average_accuracy"],
        help="Accuracy metric to plot/rank from results.csv (default: overall_accuracy).",
    )
    parser.add_argument(
        "--phase",
        default="postopt-test",
        choices=["preopt-test", "preopt-validation", "postopt-test", "postopt-validation"],
        help="Phase to evaluate from results.csv (default: postopt-test).",
    )
    parser.add_argument(
        "--scope",
        default="overall",
        choices=["overall", "dataset"],
        help="Scope to evaluate from results.csv (default: overall).",
    )
    parser.add_argument(
        "--dataset",
        type=int,
        default=None,
        help="Dataset id if --scope dataset (default: all datasets not allowed, must provide).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top rows to print in each ranking (default: 10).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively in addition to saving them.",
    )
    args = parser.parse_args()

    if args.scope == "dataset" and args.dataset is None:
        raise ValueError("--dataset is required when --scope dataset.")

    try:
        import numpy as np_mod
        import matplotlib.pyplot as plt_mod
    except Exception as exc:
        raise RuntimeError(
            "Missing Python dependencies. Install with: pip install numpy matplotlib"
        ) from exc
    np = np_mod
    plt = plt_mod

    manifest_rows = load_manifest(MANIFEST_CSV)
    result_rows = load_results(RESULTS_CSV)
    enriched_rows, rows_per_run = enrich_results_with_manifest(manifest_rows, result_rows)
    crossover_rates, mutation_rates = sorted_rate_grid(manifest_rows)

    accuracy_summary = aggregate_accuracy(
        enriched_rows,
        metric_key=args.metric,
        scope=args.scope,
        phase=args.phase,
        dataset=args.dataset,
    )
    if not accuracy_summary:
        raise RuntimeError(
            f"No matching rows for scope={args.scope}, phase={args.phase}, dataset={args.dataset}."
        )

    suffix_dataset = f"_dataset{args.dataset}" if args.dataset is not None else ""
    tag = f"{args.scope}_{args.phase}_{args.metric}{suffix_dataset}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    accuracy_csv_path = os.path.join(OUTPUT_DIR, f"accuracy_summary_{tag}.csv")
    write_accuracy_summary_csv(accuracy_csv_path, accuracy_summary)

    z_accuracy = summary_to_grid(
        accuracy_summary,
        crossover_rates=crossover_rates,
        mutation_rates=mutation_rates,
        value_key="mean_accuracy_pct",
    )
    acc_surface_path = os.path.join(OUTPUT_DIR, f"accuracy_surface_{tag}.png")
    acc_heatmap_path = os.path.join(OUTPUT_DIR, f"accuracy_heatmap_{tag}.png")

    plot_3d_surface(
        crossover_rates,
        mutation_rates,
        z_accuracy,
        title=f"Mean Accuracy vs Mutation/Crossover ({args.scope}, {args.phase})",
        z_label="Mean accuracy (%)",
        out_path=acc_surface_path,
        show=args.show,
    )
    plot_heatmap(
        crossover_rates,
        mutation_rates,
        z_accuracy,
        title=f"Mean Accuracy Heatmap ({args.scope}, {args.phase})",
        cbar_label="Mean accuracy (%)",
        out_path=acc_heatmap_path,
        show=args.show,
    )

    # Always produce an additional heatmap for postopt-validation.
    postopt_val_phase = "postopt-validation"
    postopt_val_tag = f"{args.scope}_{postopt_val_phase}_{args.metric}{suffix_dataset}"
    postopt_val_csv_path = os.path.join(OUTPUT_DIR, f"accuracy_summary_{postopt_val_tag}.csv")
    postopt_val_heatmap_path = os.path.join(OUTPUT_DIR, f"accuracy_heatmap_{postopt_val_tag}.png")
    if args.phase == postopt_val_phase:
        write_accuracy_summary_csv(postopt_val_csv_path, accuracy_summary)
        if postopt_val_heatmap_path != acc_heatmap_path:
            plot_heatmap(
                crossover_rates,
                mutation_rates,
                z_accuracy,
                title=f"Mean Accuracy Heatmap ({args.scope}, {postopt_val_phase})",
                cbar_label="Mean accuracy (%)",
                out_path=postopt_val_heatmap_path,
                show=args.show,
            )
        else:
            postopt_val_heatmap_path = acc_heatmap_path
    else:
        postopt_val_summary = aggregate_accuracy(
            enriched_rows,
            metric_key=args.metric,
            scope=args.scope,
            phase=postopt_val_phase,
            dataset=args.dataset,
        )
        if postopt_val_summary:
            write_accuracy_summary_csv(postopt_val_csv_path, postopt_val_summary)
            z_postopt_val = summary_to_grid(
                postopt_val_summary,
                crossover_rates=crossover_rates,
                mutation_rates=mutation_rates,
                value_key="mean_accuracy_pct",
            )
            plot_heatmap(
                crossover_rates,
                mutation_rates,
                z_postopt_val,
                title=f"Mean Accuracy Heatmap ({args.scope}, {postopt_val_phase})",
                cbar_label="Mean accuracy (%)",
                out_path=postopt_val_heatmap_path,
                show=args.show,
            )
        else:
            postopt_val_csv_path = None
            postopt_val_heatmap_path = None
            print(
                f"Warning: no rows for extra heatmap (scope={args.scope}, "
                f"phase={postopt_val_phase}, dataset={args.dataset})."
            )

    top_accuracy_rows = print_top_accuracy(accuracy_summary, args.top_k)
    dataset_best_rows = print_dataset_best(enriched_rows, metric_key=args.metric, phase=args.phase)

    per_dataset_convergence_rows, missing_logs = build_convergence_rows(manifest_rows, LOG_DIR)
    if not per_dataset_convergence_rows:
        raise RuntimeError("No convergence metrics could be extracted from logs.")
    if missing_logs:
        print(f"\nWarning: missing {len(missing_logs)} log files; convergence summary is partial.")

    if args.scope == "dataset" and args.dataset is not None:
        per_dataset_convergence_rows = [
            r for r in per_dataset_convergence_rows if r["dataset"] == args.dataset
        ]
        if not per_dataset_convergence_rows:
            raise RuntimeError(
                f"No convergence metrics found for dataset={args.dataset} in logs."
            )

    convergence_summary = aggregate_convergence(per_dataset_convergence_rows)
    top_convergence_rows = print_convergence_summary(convergence_summary, args.top_k)

    tradeoff_rows = build_tradeoff_rows(accuracy_summary, convergence_summary)
    pareto_rows = pareto_front_accuracy_vs_t95(tradeoff_rows)
    print_pareto_front(pareto_rows)

    convergence_detail_csv = os.path.join(OUTPUT_DIR, "convergence_per_run_dataset.csv")
    write_rows_csv(
        convergence_detail_csv,
        fieldnames=[
            "run_index",
            "seed",
            "crossover_rate",
            "mutation_rate",
            "dataset",
            "first_generation",
            "last_generation",
            "initial_best_accuracy",
            "final_best_accuracy",
            "t95_generation",
            "t95_fraction",
            "best_accuracy_auc",
            "mean_new_selected_ratio",
            "final_new_selected_ratio",
        ],
        rows=per_dataset_convergence_rows,
    )

    convergence_summary_csv = os.path.join(OUTPUT_DIR, "convergence_summary_by_rate.csv")
    conv_rows_out = []
    for (cx, mut), rec in sorted(convergence_summary.items()):
        conv_rows_out.append(
            {
                "crossover_rate": cx,
                "mutation_rate": mut,
                **rec,
            }
        )
    write_rows_csv(
        convergence_summary_csv,
        fieldnames=[
            "crossover_rate",
            "mutation_rate",
            "count",
            "mean_t95_generation",
            "std_t95_generation",
            "mean_t95_fraction",
            "std_t95_fraction",
            "mean_best_accuracy_auc",
            "mean_final_best_accuracy",
            "mean_new_selected_ratio",
            "final_new_selected_ratio",
        ],
        rows=conv_rows_out,
    )

    tradeoff_csv = os.path.join(OUTPUT_DIR, f"accuracy_convergence_tradeoff_{tag}.csv")
    write_rows_csv(
        tradeoff_csv,
        fieldnames=[
            "crossover_rate",
            "mutation_rate",
            "mean_accuracy_pct",
            "std_accuracy_pct",
            "mean_t95_generation",
            "mean_t95_fraction",
            "mean_best_accuracy_auc",
            "mean_new_selected_ratio",
            "final_new_selected_ratio",
        ],
        rows=tradeoff_rows,
    )

    top_accuracy_csv = os.path.join(OUTPUT_DIR, f"top_accuracy_rates_{tag}.csv")
    write_rows_csv(
        top_accuracy_csv,
        fieldnames=[
            "crossover_rate",
            "mutation_rate",
            "count",
            "mean_accuracy_pct",
            "std_accuracy_pct",
            "min_accuracy_pct",
            "max_accuracy_pct",
        ],
        rows=top_accuracy_rows,
    )

    dataset_best_csv = os.path.join(OUTPUT_DIR, f"best_rates_per_dataset_{args.metric}.csv")
    write_rows_csv(
        dataset_best_csv,
        fieldnames=[
            "dataset",
            "crossover_rate",
            "mutation_rate",
            "count",
            "mean_accuracy_pct",
            "std_accuracy_pct",
            "min_accuracy_pct",
            "max_accuracy_pct",
        ],
        rows=dataset_best_rows,
    )

    z_t95 = summary_to_grid(
        convergence_summary,
        crossover_rates=crossover_rates,
        mutation_rates=mutation_rates,
        value_key="mean_t95_fraction",
    )
    t95_surface_path = os.path.join(OUTPUT_DIR, "convergence_t95_surface.png")
    plot_3d_surface(
        crossover_rates,
        mutation_rates,
        z_t95,
        title="Convergence Speed vs Mutation/Crossover (lower is faster)",
        z_label="Mean t95 fraction",
        out_path=t95_surface_path,
        show=args.show,
    )

    tradeoff_plot_path = os.path.join(OUTPUT_DIR, f"accuracy_vs_convergence_{tag}.png")
    plot_accuracy_vs_convergence(tradeoff_rows, tradeoff_plot_path, args.show)

    print("\nWrote:")
    print(f"- {accuracy_csv_path}")
    print(f"- {top_accuracy_csv}")
    print(f"- {dataset_best_csv}")
    print(f"- {acc_surface_path}")
    print(f"- {acc_heatmap_path}")
    if postopt_val_csv_path:
        print(f"- {postopt_val_csv_path}")
    if postopt_val_heatmap_path:
        print(f"- {postopt_val_heatmap_path}")
    print(f"- {convergence_detail_csv}")
    print(f"- {convergence_summary_csv}")
    print(f"- {t95_surface_path}")
    print(f"- {tradeoff_csv}")
    print(f"- {tradeoff_plot_path}")
    print(f"\nInput alignment:")
    print(f"- manifest runs: {len(manifest_rows)}")
    print(f"- results rows: {len(result_rows)}")
    print(f"- inferred rows per run: {rows_per_run}")
    print(f"- convergence log files parsed: {len(manifest_rows) - len(missing_logs)}")


if __name__ == "__main__":
    main()
