#!/usr/bin/env python3
"""Combined threshold-boundary and feature-distribution diagnostic plot."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_feature_histograms import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_NUM_CLASSES,
    DEFAULT_NUM_FEATURES,
    DEFAULT_PLOTS_DIR,
    DEFAULT_VALIDATION_RATIO,
    DEFAULT_DOWNSAMPLE,
    empirical_cdf,
    fixed_bins,
    get_pyplot,
    histogram,
    load_feature_values_and_labels,
    value_stats,
)
from plot_thresholds import (  # noqa: E402
    DEFAULT_EXPORTS_DIR,
    METHOD_COLORS,
    METHOD_LABELS,
    cuts_path,
    parse_methods,
    read_cuts_file,
)


CLASS_COLORS = [
    "#b95f0a",
    "#af007b",
    "#342d74",
    "#8b8b00",
    "#0099aa",
    "#aec7e8",
    "#f7b467",
    "#71d45d",
    "#ff6360",
    "#d59eff",
]

THESIS_METHOD_LABELS = {
    "uniform": "Uniform binning",
    "quantile": "Quantile binning",
    "kmeans_1d": "k-means binning",
    "decision_tree_1d": "Decision-tree binning",
    "chimerge": "ChiMerge binning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one figure with threshold curves above and the matching "
            "feature histogram/CDF below. Defaults to dataset 0, feature 15."
        )
    )
    parser.add_argument("--dataset", type=int, default=0)
    parser.add_argument("--feature", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-levels", type=int, default=40)
    parser.add_argument("--vector-dimension", type=int, default=10000)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--scope", choices=["quantizer-train", "downsampled-train", "raw-train", "raw-test"], default="quantizer-train")
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--num-features", type=int, default=DEFAULT_NUM_FEATURES)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--exports-dir", type=Path, default=DEFAULT_EXPORTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    args.methods = parse_methods(args.methods)
    if args.num_levels <= 0:
        raise ValueError("--num-levels must be positive")
    if args.feature < 0 or args.feature >= args.num_features:
        raise ValueError(f"--feature must be in [0, {args.num_features})")
    if args.x_min >= args.x_max:
        raise ValueError("--x-min must be < --x-max")
    return args


def load_threshold_curve(args: argparse.Namespace, method: str) -> list[float]:
    path = cuts_path(args.exports_dir, args.num_levels, args.vector_dimension, args.seed, method, args.dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing threshold export: {path}\n"
            "Run export_quantizer_thresholds.py first for this seed/method/grid."
        )
    cuts = read_cuts_file(path)
    if args.feature not in cuts:
        raise KeyError(f"feature {args.feature} not found in {path}")
    return cuts[args.feature]


def method_label(method: str) -> str:
    return THESIS_METHOD_LABELS.get(method, METHOD_LABELS.get(method, method))


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def distribution_shape(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= 0.0:
        return 0.0, 0.0
    std = math.sqrt(variance)
    skewness = sum(((value - mean) / std) ** 3 for value in values) / len(values)
    kurtosis_excess = sum(((value - mean) / std) ** 4 for value in values) / len(values) - 3.0
    return skewness, kurtosis_excess


def count_in_range(values: list[float], lo: float, hi: float) -> int:
    return sum(1 for value in values if lo <= value <= hi)


def plot_threshold_panel(args: argparse.Namespace, ax) -> None:
    for method in args.methods:
        curve = load_threshold_curve(args, method)
        cut_indices = list(range(len(curve)))
        ax.plot(
            curve,
            cut_indices,
            label=method_label(method),
            color=METHOD_COLORS.get(method),
            linewidth=1.8,
        )
    ax.set_ylabel(r"Transition $i$")
    ax.set_title("Threshold boundaries")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def plot_histogram_panel(args: argparse.Namespace, ax) -> None:
    values, labels = load_feature_values_and_labels(args, args.dataset, args.feature)
    bins = fixed_bins(args)
    stats = value_stats(values)
    classes = sorted(set(labels))
    ax2 = ax.twinx()

    hist_all = ax.hist(
        values,
        bins=bins,
        density=args.density,
        color="lightgray",
        edgecolor="none",
        alpha=0.55,
        label="All samples",
    )
    cdf_x_all, cdf_y_all = empirical_cdf(values)
    cdf_all, = ax2.plot(
        cdf_x_all,
        cdf_y_all,
        color="black",
        linewidth=1.4,
        label="Overall CDF",
    )

    legend_handles = [hist_all[2][0], cdf_all]
    legend_labels = ["All samples", "Overall CDF"]

    for color_idx, cls in enumerate(classes):
        class_values = [value for value, label in zip(values, labels) if label == cls]
        if not class_values:
            continue
        hist_values, edges = histogram(class_values, bins, args.density)
        centers = [0.5 * (edges[idx] + edges[idx + 1]) for idx in range(len(edges) - 1)]
        color = CLASS_COLORS[color_idx % len(CLASS_COLORS)]
        hist_line, = ax.plot(centers, hist_values, color=color, linewidth=1.0, alpha=0.95)
        cdf_x, cdf_y = empirical_cdf(class_values)
        cdf_line, = ax2.plot(cdf_x, cdf_y, color=color, linewidth=0.9, linestyle="--", alpha=0.95)
        legend_handles.extend([hist_line, cdf_line])
        legend_labels.extend([f"Class {cls} samples", f"Class {cls} CDF"])

    if args.log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Density" if args.density else "Number of samples")
    ax2.set_ylabel("CDF")
    ax2.set_ylim(0.0, 1.0)
    ax.set_title("Feature distribution")
    ax.grid(True, alpha=0.2)
    ax.legend(legend_handles, legend_labels, loc="upper left", ncol=2, fontsize=8, frameon=True)


def print_summary(args: argparse.Namespace) -> None:
    values, labels = load_feature_values_and_labels(args, args.dataset, args.feature)
    bins = fixed_bins(args)
    stats = value_stats(values)
    hist_values, _ = histogram(values, bins, args.density)
    classes = sorted(set(labels))
    sorted_values = sorted(values)
    q01 = quantile(sorted_values, 0.01)
    q05 = quantile(sorted_values, 0.05)
    q25 = quantile(sorted_values, 0.25)
    q50 = quantile(sorted_values, 0.50)
    q75 = quantile(sorted_values, 0.75)
    q95 = quantile(sorted_values, 0.95)
    q99 = quantile(sorted_values, 0.99)
    skewness, kurtosis_excess = distribution_shape(values)
    nonzero_bins = sum(1 for count in hist_values if count > 0.0)
    empty_bins = len(hist_values) - nonzero_bins
    max_bin_count = max(hist_values) if hist_values else 0.0
    min_nonzero_bin_count = min((count for count in hist_values if count > 0.0), default=0.0)
    outside_plot_range = sum(1 for value in values if value < args.x_min or value > args.x_max)

    print("Threshold histogram summary")
    print(f"  dataset: {args.dataset}")
    print(f"  feature: {args.feature}")
    print(f"  seed: {args.seed}")
    print(f"  num_levels: {args.num_levels}")
    print(f"  vector_dimension: {args.vector_dimension}")
    print(f"  scope: {args.scope}")
    print(f"  samples: {stats['count']}")
    print(f"  unique feature values: {stats['unique_count']}")
    print(f"  feature range: {stats['min']:.8g} .. {stats['max']:.8g}")
    print(f"  feature mean/std: {stats['mean']:.8g} / {stats['std']:.8g}")
    print(f"  feature quantiles p01/p05/p25/p50/p75/p95/p99: {q01:.8g} / {q05:.8g} / {q25:.8g} / {q50:.8g} / {q75:.8g} / {q95:.8g} / {q99:.8g}")
    print(f"  feature skewness/excess kurtosis: {skewness:.8g} / {kurtosis_excess:.8g}")
    print(
        f"  most common value/count: {stats['most_common_value']:.8g} / "
        f"{stats['most_common_count']}"
    )
    print(f"  histogram bins: {len(bins) - 1} equally spaced from {args.x_min:g} to {args.x_max:g}")
    print(
        f"  occupied/empty bins: {nonzero_bins} / {empty_bins}; "
        f"min nonzero/max bin count: {min_nonzero_bin_count:.8g} / {max_bin_count:.8g}"
    )
    print(f"  samples outside histogram range: {outside_plot_range}")
    print()
    print("Class-wise feature distribution")
    print("class samples min p25 median p75 max mean std")
    for cls in classes:
        class_values = [value for value, label in zip(values, labels) if label == cls]
        class_stats = value_stats(class_values)
        sorted_class_values = sorted(class_values)
        print(
            f"{cls:5d} "
            f"{int(class_stats['count']):7d} "
            f"{float(class_stats['min']): .8g} "
            f"{quantile(sorted_class_values, 0.25): .8g} "
            f"{quantile(sorted_class_values, 0.50): .8g} "
            f"{quantile(sorted_class_values, 0.75): .8g} "
            f"{float(class_stats['max']): .8g} "
            f"{float(class_stats['mean']): .8g} "
            f"{float(class_stats['std']): .8g}"
        )

    print()
    print("Threshold boundaries")
    print("method cuts min_cut max_cut first_cut last_cut min_step median_step max_step samples_below_first samples_above_last")
    for method in args.methods:
        curve = load_threshold_curve(args, method)
        steps = [curve[idx + 1] - curve[idx] for idx in range(len(curve) - 1)]
        min_step = min(steps) if steps else 0.0
        median_step = quantile(sorted(steps), 0.50) if steps else 0.0
        max_step = max(steps) if steps else 0.0
        below_first = sum(1 for value in values if value <= curve[0])
        above_last = sum(1 for value in values if value > curve[-1])
        print(
            f"{method_label(method):>22s} "
            f"{len(curve):4d} "
            f"{min(curve): .8g} "
            f"{max(curve): .8g} "
            f"{curve[0]: .8g} "
            f"{curve[-1]: .8g} "
            f"{min_step: .8g} "
            f"{median_step: .8g} "
            f"{max_step: .8g} "
            f"{below_first:19d} "
            f"{above_last:18d}"
        )


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.exports_dir = args.exports_dir.resolve()
    args.plots_dir = args.plots_dir.resolve()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    print_summary(args)

    plt = get_pyplot()
    fig, (ax_threshold, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    plot_threshold_panel(args, ax_threshold)
    plot_histogram_panel(args, ax_hist)
    ax_threshold.set_xlim(args.x_min, args.x_max)
    fig.tight_layout()

    output_path = (
        args.plots_dir
        / f"threshold_hist_dataset_{args.dataset:02d}_feature_{args.feature:03d}_"
        f"seed_{args.seed:02d}_levels_{args.num_levels:03d}_dim_{args.vector_dimension:05d}.png"
    )
    fig.savefig(output_path, dpi=220)
    print(f"Wrote plot: {output_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
