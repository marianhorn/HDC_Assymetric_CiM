#!/usr/bin/env python3
"""Visualize exported learned quantizer thresholds across seeds and methods."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
THRESHOLD_DIR = SCRIPT_DIR.parent
DEFAULT_EXPORTS_DIR = THRESHOLD_DIR / "exports"
DEFAULT_RESULTS_DIR = THRESHOLD_DIR / "results"
DEFAULT_PLOTS_DIR = THRESHOLD_DIR / "plots"

DEFAULT_METHODS = ["uniform", "quantile", "kmeans_1d", "decision_tree_1d", "chimerge"]
METHOD_LABELS = {
    "uniform": "uniform",
    "quantile": "quantile",
    "kmeans_1d": "k-means 1D",
    "decision_tree_1d": "decision tree 1D",
    "chimerge": "ChiMerge",
}
METHOD_COLORS = {
    "uniform": "#222222",
    "quantile": "#1f77b4",
    "kmeans_1d": "#ff7f0e",
    "decision_tree_1d": "#2ca02c",
    "chimerge": "#d62728",
}
METHOD_ALIASES = {
    "0": "uniform",
    "uniform": "uniform",
    "1": "quantile",
    "quantile": "quantile",
    "2": "kmeans_1d",
    "kmeans": "kmeans_1d",
    "kmeans_1d": "kmeans_1d",
    "3": "decision_tree_1d",
    "decision_tree": "decision_tree_1d",
    "decision_tree_1d": "decision_tree_1d",
    "4": "chimerge",
    "chimerge": "chimerge",
}


def get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: matplotlib. Use the same Python 3 environment as the other analysis scripts."
        ) from exc
    return plt


def parse_list(text: str, cast=str) -> list:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def parse_methods(text: str) -> list[str]:
    if text.strip().lower() == "all":
        return list(DEFAULT_METHODS)
    methods = []
    for item in text.split(","):
        key = item.strip().lower().replace("-", "_")
        if not key:
            continue
        if key not in METHOD_ALIASES:
            valid = ", ".join(DEFAULT_METHODS + ["0", "1", "2", "3", "4"])
            raise ValueError(f"Unknown method {item!r}. Valid: {valid}")
        method = METHOD_ALIASES[key]
        if method not in methods:
            methods.append(method)
    if not methods:
        raise ValueError("--methods must select at least one method")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot exported quantizer thresholds across methods and seeds."
    )
    parser.add_argument("--num-levels", type=int, default=40)
    parser.add_argument("--vector-dimension", type=int, default=10000)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--datasets", default="0,1,2,3")
    parser.add_argument("--methods", default="quantile")
    parser.add_argument(
        "--compare",
        choices=["seeds", "methods"],
        default="seeds",
        help=(
            "seeds: one plot per dataset/method with one line per seed. "
            "methods: one plot per dataset/seed with one line per method."
        ),
    )
    parser.add_argument(
        "--view",
        choices=["raw", "diff-uniform", "std-heatmap"],
        default="diff-uniform",
        help="raw thresholds, learned-minus-uniform curves, or seed std heatmaps.",
    )
    parser.add_argument(
        "--feature",
        default="0",
        help="Feature index to plot. Features are never averaged. Default: 0.",
    )
    parser.add_argument("--exports-dir", type=Path, default=DEFAULT_EXPORTS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    args.seeds = parse_list(args.seeds, int)
    args.datasets = parse_list(args.datasets, int)
    args.methods = parse_methods(args.methods)
    if args.view == "diff-uniform" and "uniform" not in args.methods:
        args.methods = ["uniform"] + args.methods
    try:
        args.feature = int(args.feature)
    except ValueError as exc:
        raise ValueError("--feature must be an integer feature index") from exc
    if args.feature < 0:
        raise ValueError("--feature must be non-negative")
    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    if not args.datasets:
        raise ValueError("--datasets must not be empty")
    return args


def cuts_path(exports_dir: Path, num_levels: int, vector_dimension: int, seed: int, method: str, dataset: int) -> Path:
    return (
        exports_dir
        / f"levels_{num_levels:03d}_dim_{vector_dimension:05d}"
        / f"seed_{seed:02d}"
        / method
        / f"cuts_dataset{dataset:02d}.csv"
    )


def read_cuts_file(path: Path) -> dict[int, list[float]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as f:
        data_lines = [line for line in f if not line.startswith("#") and line.strip()]

    reader = csv.DictReader(data_lines)
    if reader.fieldnames is None:
        raise ValueError(f"No CSV header found in {path}")

    cut_columns = [name for name in reader.fieldnames if name.startswith("cut_")]
    if not cut_columns:
        raise ValueError(f"No cut columns found in {path}")

    by_feature: dict[int, list[float]] = {}
    for row in reader:
        feature = int(row["feature"])
        by_feature[feature] = [float(row[col]) for col in cut_columns]
    return by_feature


def curve_for_feature(cuts: dict[int, list[float]], feature: int) -> list[float]:
    if feature not in cuts:
        raise KeyError(f"feature {feature} not present")
    return list(cuts[feature])


def load_curve_data(args: argparse.Namespace) -> dict[tuple[int, str], list[dict[str, object]]]:
    curves: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    missing = []

    for dataset in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                path = cuts_path(args.exports_dir, args.num_levels, args.vector_dimension, seed, method, dataset)
                if not path.exists():
                    missing.append(path)
                    continue
                cuts = read_cuts_file(path)
                curve = curve_for_feature(cuts, args.feature)
                curves[(dataset, method)].append({"seed": seed, "curve": curve, "path": path})

    if missing:
        print("Missing exported threshold files:")
        for path in missing[:20]:
            print(f"  {path}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")
        print("Continue with available files.")

    return curves


def summarize_curves(curves: list[dict[str, object]]) -> list[dict[str, float]]:
    if not curves:
        return []
    cut_count = len(curves[0]["curve"])
    summary = []
    for cut_idx in range(cut_count):
        values = [entry["curve"][cut_idx] for entry in curves]
        summary.append(
            {
                "cut_index": cut_idx,
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "seed_count": len(values),
            }
        )
    return summary


def write_curve_summary(args: argparse.Namespace, curve_data: dict[tuple[int, str], list[dict[str, object]]]) -> Path:
    feature_label = f"feature_{args.feature}"
    output_path = (
        args.results_dir
        / f"threshold_{args.view}_levels_{args.num_levels:03d}_dim_{args.vector_dimension:05d}_{feature_label}.csv"
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "view",
                "dataset",
                "method",
                "seed",
                "feature",
                "cut_index",
                "value",
                "source_path",
            ]
        )
        for (dataset, method), curves in sorted(curve_data.items()):
            for entry in sorted(curves, key=lambda item: item["seed"]):
                for cut_index, value in enumerate(entry["curve"]):
                    writer.writerow(
                        [
                            args.view,
                            dataset,
                            method,
                            entry["seed"],
                            args.feature,
                            cut_index,
                            f"{value:.17g}",
                            entry["path"],
                        ]
                    )
    return output_path


def subtract_uniform(curve_data: dict[tuple[int, str], list[dict[str, object]]]) -> dict[tuple[int, str], list[dict[str, object]]]:
    uniform_by_dataset_seed = {}
    for (dataset, method), curves in curve_data.items():
        if method != "uniform":
            continue
        for entry in curves:
            uniform_by_dataset_seed[(dataset, entry["seed"])] = entry["curve"]

    diff_data: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for (dataset, method), curves in curve_data.items():
        if method == "uniform":
            continue
        for entry in curves:
            key = (dataset, entry["seed"])
            if key not in uniform_by_dataset_seed:
                raise RuntimeError(
                    f"diff-uniform requires uniform export for dataset {dataset}, seed {entry['seed']}"
                )
            uniform_curve = uniform_by_dataset_seed[key]
            curve = entry["curve"]
            if len(curve) != len(uniform_curve):
                raise RuntimeError(f"cut count mismatch for dataset {dataset}, method {method}, seed {entry['seed']}")
            diff_data[(dataset, method)].append(
                {
                    "seed": entry["seed"],
                    "curve": [value - uniform_curve[idx] for idx, value in enumerate(curve)],
                    "path": entry["path"],
                }
            )
    return diff_data


def has_curve_entries(curve_data: dict[tuple[int, str], list[dict[str, object]]]) -> bool:
    return any(entries for entries in curve_data.values())


def plot_curves(args: argparse.Namespace, curve_data: dict[tuple[int, str], list[dict[str, object]]]) -> list[Path]:
    plt = get_pyplot()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    feature_label = f"feature {args.feature}"

    if args.compare == "seeds":
        for dataset in args.datasets:
            for method in args.methods:
                if args.view == "diff-uniform" and method == "uniform":
                    continue
                curves = sorted(curve_data.get((dataset, method), []), key=lambda item: item["seed"])
                if not curves:
                    continue
                fig, ax = plt.subplots(figsize=(10, 6))
                xs = list(range(len(curves[0]["curve"])))
                color = METHOD_COLORS.get(method)
                for entry in curves:
                    ax.plot(xs, entry["curve"], label=f"seed {entry['seed']}", linewidth=1.8)
                title_suffix = f"{METHOD_LABELS.get(method, method)}, one line per seed"
                output_suffix = f"{method}_seeds"
                _finish_curve_plot(args, fig, ax, dataset, feature_label, title_suffix)
                output_path = _save_curve_plot(args, fig, dataset, output_suffix)
                output_paths.append(output_path)
                if not args.no_show:
                    plt.show()
                else:
                    plt.close(fig)

    else:
        for dataset in args.datasets:
            for seed in args.seeds:
                fig, ax = plt.subplots(figsize=(10, 6))
                plotted = False
                for method in args.methods:
                    if args.view == "diff-uniform" and method == "uniform":
                        continue
                    entries = [entry for entry in curve_data.get((dataset, method), []) if entry["seed"] == seed]
                    if not entries:
                        continue
                    plotted = True
                    entry = entries[0]
                    xs = list(range(len(entry["curve"])))
                    ax.plot(
                        xs,
                        entry["curve"],
                        label=METHOD_LABELS.get(method, method),
                        color=METHOD_COLORS.get(method),
                        linewidth=1.8,
                    )
                if not plotted:
                    plt.close(fig)
                    continue
                title_suffix = f"seed {seed}, one line per method"
                output_suffix = f"seed_{seed:02d}_methods"
                _finish_curve_plot(args, fig, ax, dataset, feature_label, title_suffix)
                output_path = _save_curve_plot(args, fig, dataset, output_suffix)
                output_paths.append(output_path)
                if not args.no_show:
                    plt.show()
                else:
                    plt.close(fig)

    return output_paths


def _finish_curve_plot(args: argparse.Namespace, fig, ax, dataset: int, feature_label: str, title_suffix: str) -> None:
    ylabel = "threshold"
    if args.view == "diff-uniform":
        ylabel = "threshold - uniform threshold"
        ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--")

    ax.set_title(
        f"{args.view}: dataset {dataset}, {feature_label}, {title_suffix}, "
        f"levels={args.num_levels}, dim={args.vector_dimension}"
    )
    ax.set_xlabel("cut index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()


def _save_curve_plot(args: argparse.Namespace, fig, dataset: int, suffix: str) -> Path:
    filename = (
        f"threshold_{args.view}_dataset_{dataset}_levels_{args.num_levels:03d}_"
        f"dim_{args.vector_dimension:05d}_feature_{args.feature:03d}_{suffix}.png"
    )
    output_path = args.plots_dir / filename
    fig.savefig(output_path, dpi=180)
    return output_path


def load_heatmap_values(args: argparse.Namespace, method: str, dataset: int) -> tuple[list[int], list[int], list[list[float]]]:
    by_seed = []
    for seed in args.seeds:
        path = cuts_path(args.exports_dir, args.num_levels, args.vector_dimension, seed, method, dataset)
        if path.exists():
            by_seed.append(read_cuts_file(path))
    if not by_seed:
        return [], [], []

    features = sorted(set.intersection(*(set(cuts.keys()) for cuts in by_seed)))
    if not features:
        return [], [], []
    cut_count = len(by_seed[0][features[0]])
    cuts = list(range(cut_count))
    matrix = []
    for feature in features:
        row = []
        for cut_idx in cuts:
            values = [seed_cuts[feature][cut_idx] for seed_cuts in by_seed]
            row.append(statistics.pstdev(values) if len(values) > 1 else 0.0)
        matrix.append(row)
    return features, cuts, matrix


def write_heatmap_summary(args: argparse.Namespace) -> Path:
    output_path = (
        args.results_dir
        / f"threshold_std_heatmap_levels_{args.num_levels:03d}_dim_{args.vector_dimension:05d}.csv"
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "feature", "cut_index", "std_across_seeds"])
        for dataset in args.datasets:
            for method in args.methods:
                features, cuts, matrix = load_heatmap_values(args, method, dataset)
                for feature, row in zip(features, matrix):
                    for cut_idx, value in zip(cuts, row):
                        writer.writerow([dataset, method, feature, cut_idx, f"{value:.17g}"])
    return output_path


def plot_heatmaps(args: argparse.Namespace) -> list[Path]:
    plt = get_pyplot()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for dataset in args.datasets:
        for method in args.methods:
            features, cuts, matrix = load_heatmap_values(args, method, dataset)
            if not matrix:
                continue

            fig, ax = plt.subplots(figsize=(11, 6))
            image = ax.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest")
            fig.colorbar(image, ax=ax, label="std across seeds")
            ax.set_title(
                f"threshold seed stability: {METHOD_LABELS.get(method, method)}, dataset {dataset}, "
                f"levels={args.num_levels}, dim={args.vector_dimension}"
            )
            ax.set_xlabel("cut index")
            ax.set_ylabel("feature")
            if cuts:
                ax.set_xticks([0, len(cuts) - 1])
                ax.set_xticklabels([str(cuts[0]), str(cuts[-1])])
            if features:
                ax.set_yticks([0, len(features) - 1])
                ax.set_yticklabels([str(features[0]), str(features[-1])])
            fig.tight_layout()

            output_path = (
                args.plots_dir
                / f"threshold_std_heatmap_{method}_dataset_{dataset}_levels_{args.num_levels:03d}_"
                f"dim_{args.vector_dimension:05d}.png"
            )
            fig.savefig(output_path, dpi=180)
            output_paths.append(output_path)
            if not args.no_show:
                plt.show()
            else:
                plt.close(fig)

    return output_paths


def main() -> None:
    args = parse_args()
    args.exports_dir = args.exports_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    args.plots_dir = args.plots_dir.resolve()

    print(f"Exports: {args.exports_dir}")
    print(f"Results: {args.results_dir}")
    print(f"Plots:   {args.plots_dir}")
    print(f"View: {args.view}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Datasets: {', '.join(str(v) for v in args.datasets)}")
    print(f"Seeds: {', '.join(str(v) for v in args.seeds)}")

    if args.view == "std-heatmap":
        summary_path = write_heatmap_summary(args)
        plot_paths = plot_heatmaps(args)
    else:
        curve_data = load_curve_data(args)
        if args.view == "diff-uniform":
            curve_data = subtract_uniform(curve_data)
        if not has_curve_entries(curve_data):
            raise RuntimeError(
                "No threshold curves found for the selected grid. Export thresholds first with "
                "analysis/benchmarking/quantizer_only/scripts/export_quantizer_thresholds.py. "
                "If you are using Bash/WSL, use backslash line continuations, not PowerShell backticks."
            )
        summary_path = write_curve_summary(args, curve_data)
        plot_paths = plot_curves(args, curve_data)

    print(f"Wrote summary: {summary_path}")
    for path in plot_paths:
        print(f"Wrote plot: {path}")
    if not plot_paths:
        print("No plots were generated. Check that exported threshold CSVs exist for the selected grid.")


if __name__ == "__main__":
    main()
