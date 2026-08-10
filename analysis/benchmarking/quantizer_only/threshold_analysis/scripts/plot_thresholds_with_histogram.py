#!/usr/bin/env python3
"""Combined threshold-boundary and feature-distribution diagnostic plot."""

from __future__ import annotations

import argparse
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


def plot_threshold_panel(args: argparse.Namespace, ax) -> None:
    for method in args.methods:
        curve = load_threshold_curve(args, method)
        cut_indices = list(range(len(curve)))
        ax.plot(
            curve,
            cut_indices,
            label=f"method: {METHOD_LABELS.get(method, method)}",
            color=METHOD_COLORS.get(method),
            linewidth=1.8,
        )
    ax.set_ylabel("cut index")
    ax.set_title(
        f"Threshold boundaries: dataset {args.dataset}, feature {args.feature}, "
        f"seed {args.seed}, levels={args.num_levels}, dim={args.vector_dimension}"
    )
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
        label="all-hist",
    )
    cdf_x_all, cdf_y_all = empirical_cdf(values)
    cdf_all, = ax2.plot(cdf_x_all, cdf_y_all, color="black", linewidth=1.4, label="all-cdf")

    legend_handles = [hist_all[2][0], cdf_all]
    legend_labels = ["distribution: all histogram", "distribution: all CDF"]

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
        legend_labels.extend([f"class {cls}: histogram", f"class {cls}: CDF"])

    if args.log_y:
        ax.set_yscale("log")
    ax.set_xlabel("feature value / threshold")
    ax.set_ylabel("Hist density" if args.density else "Hist count")
    ax2.set_ylabel("CDF")
    ax2.set_ylim(0.0, 1.0)
    ax.set_title(
        f"Feature distribution: {args.scope}, samples={stats['count']}, unique={stats['unique_count']}, "
        f"bins={args.num_levels} from {args.x_min:g} to {args.x_max:g}"
    )
    ax.grid(True, alpha=0.2)
    ax.legend(legend_handles, legend_labels, loc="upper left", ncol=2, fontsize=8, frameon=True)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.exports_dir = args.exports_dir.resolve()
    args.plots_dir = args.plots_dir.resolve()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

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
