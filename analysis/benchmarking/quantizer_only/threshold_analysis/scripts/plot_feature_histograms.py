#!/usr/bin/env python3
"""Plot per-feature Foot EMG value histograms and CDFs per dataset."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
THRESHOLD_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_DATA_ROOT = REPO_ROOT / "foot" / "data"
DEFAULT_RESULTS_DIR = THRESHOLD_DIR / "results"
DEFAULT_PLOTS_DIR = THRESHOLD_DIR / "plots"
DEFAULT_NUM_CLASSES = 5
DEFAULT_NUM_FEATURES = 32
DEFAULT_VALIDATION_RATIO = 0.3
DEFAULT_DOWNSAMPLE = 1


def get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: matplotlib. Use the same Python 3 environment as the other analysis scripts."
        ) from exc
    return plt


def parse_csv_list(text: str, cast=int) -> list:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(cast(part))
    if not values:
        raise ValueError("List argument must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one feature-value distribution figure per dataset. The default "
            "scope mirrors quantizer fitting: downsampled training data after "
            "validation split removal."
        )
    )
    parser.add_argument("--features", default="15", help="Comma-separated feature indices. Default: 15.")
    parser.add_argument("--datasets", default="0,1,2,3", help="Comma-separated dataset ids. Default: 0,1,2,3.")
    parser.add_argument("--num-levels", type=int, default=40, help="Number of equal bins from -1 to 1. Default: 40.")
    parser.add_argument(
        "--scope",
        choices=["quantizer-train", "downsampled-train", "raw-train", "raw-test"],
        default="quantizer-train",
        help=(
            "quantizer-train: downsampled training values after validation split; "
            "downsampled-train: all downsampled training values; raw-train/raw-test: raw CSV values."
        ),
    )
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--density", action="store_true", help="Use density-normalized histograms.")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--num-features", type=int, default=DEFAULT_NUM_FEATURES)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    args.features = parse_csv_list(args.features, int)
    args.datasets = parse_csv_list(args.datasets, int)
    if args.num_levels <= 0:
        raise ValueError("--num-levels must be positive")
    if args.downsample <= 0:
        raise ValueError("--downsample must be positive")
    if args.x_min >= args.x_max:
        raise ValueError("--x-min must be < --x-max")
    if not 0.0 <= args.validation_ratio <= 1.0:
        raise ValueError("--validation-ratio must be in [0, 1]")
    for feature in args.features:
        if feature < 0 or feature >= args.num_features:
            raise ValueError(f"feature {feature} outside [0, {args.num_features})")
    return args


def dataset_file(data_root: Path, dataset: int, split: str, kind: str) -> Path:
    return data_root / f"dataset{dataset:02d}" / f"{split}_{kind}.csv"


def read_matrix(path: Path, num_features: int) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        for raw in reader:
            if not raw:
                continue
            if len(raw) < num_features:
                raise ValueError(f"{path}: row has {len(raw)} columns, expected at least {num_features}")
            rows.append([float(raw[idx]) for idx in range(num_features)])
    return rows


def read_labels(path: Path) -> list[int]:
    labels: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        for raw in reader:
            if raw:
                labels.append(int(float(raw[0])))
    return labels


def downsample_rows(data: list[list[float]], labels: list[int], factor: int) -> tuple[list[list[float]], list[int]]:
    length = min(len(data), len(labels)) // factor
    return [data[idx * factor] for idx in range(length)], [labels[idx * factor] for idx in range(length)]


def split_quantizer_training(
    data: list[list[float]],
    labels: list[int],
    validation_ratio: float,
    num_classes: int,
) -> tuple[list[list[float]], list[int]]:
    class_counts = [0] * num_classes
    for label in labels:
        if 0 <= label < num_classes:
            class_counts[label] += 1

    class_targets = []
    for count in class_counts:
        target = int(count * validation_ratio + 0.5)
        class_targets.append(min(target, count))

    class_assigned = [0] * num_classes
    train_data: list[list[float]] = []
    train_labels: list[int] = []
    for row, label in zip(data, labels):
        to_validation = 0 <= label < num_classes and class_assigned[label] < class_targets[label]
        if to_validation:
            class_assigned[label] += 1
        else:
            train_data.append(row)
            train_labels.append(label)
    return train_data, train_labels


def load_feature_values_and_labels(args: argparse.Namespace, dataset: int, feature: int) -> tuple[list[float], list[int]]:
    if args.scope == "raw-test":
        data = read_matrix(dataset_file(args.data_root, dataset, "testing", "emg"), args.num_features)
        labels = read_labels(dataset_file(args.data_root, dataset, "testing", "labels"))
        length = min(len(data), len(labels))
        return [data[idx][feature] for idx in range(length)], labels[:length]

    data = read_matrix(dataset_file(args.data_root, dataset, "training", "emg"), args.num_features)
    labels = read_labels(dataset_file(args.data_root, dataset, "training", "labels"))
    length = min(len(data), len(labels))
    data = data[:length]
    labels = labels[:length]

    if args.scope == "raw-train":
        return [row[feature] for row in data], labels

    data, labels = downsample_rows(data, labels, args.downsample)
    if args.scope == "downsampled-train":
        return [row[feature] for row in data], labels

    data, labels = split_quantizer_training(data, labels, args.validation_ratio, args.num_classes)
    return [row[feature] for row in data], labels


def value_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "unique_count": 0,
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "most_common_value": math.nan,
            "most_common_count": 0,
        }
    counts = Counter(values)
    most_common_value, most_common_count = counts.most_common(1)[0]
    return {
        "count": len(values),
        "unique_count": len(counts),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "most_common_value": most_common_value,
        "most_common_count": most_common_count,
    }


def empirical_cdf(values: list[float]) -> tuple[list[float], list[float]]:
    sorted_values = sorted(values)
    count = len(sorted_values)
    if count == 0:
        return [], []
    return sorted_values, [(idx + 1) / count for idx in range(count)]


def fixed_bins(args: argparse.Namespace) -> list[float]:
    return [
        args.x_min + (args.x_max - args.x_min) * idx / args.num_levels
        for idx in range(args.num_levels + 1)
    ]


def histogram(values: list[float], bins: list[float], density: bool) -> tuple[list[float], list[float]]:
    counts = [0.0] * (len(bins) - 1)
    if not counts:
        return counts, bins

    for value in values:
        if value < bins[0] or value > bins[-1]:
            continue
        if value == bins[-1]:
            idx = len(counts) - 1
        else:
            idx = int((value - bins[0]) / (bins[-1] - bins[0]) * len(counts))
            idx = max(0, min(idx, len(counts) - 1))
        counts[idx] += 1.0

    if density:
        total = sum(counts)
        if total > 0.0:
            widths = [bins[idx + 1] - bins[idx] for idx in range(len(counts))]
            counts = [count / (total * width) for count, width in zip(counts, widths)]
    return counts, bins


def write_summary(args: argparse.Namespace, records: dict[tuple[int, int], tuple[list[float], list[int]]]) -> Path:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    feature_label = "_".join(str(feature) for feature in args.features)
    output_path = args.results_dir / f"feature_histogram_summary_{args.scope}_features_{feature_label}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "scope",
                "dataset",
                "feature",
                "count",
                "unique_count",
                "min",
                "max",
                "mean",
                "std",
                "most_common_value",
                "most_common_count",
            ]
        )
        for (dataset, feature), (values, _) in sorted(records.items()):
            stats = value_stats(values)
            writer.writerow(
                [
                    args.scope,
                    dataset,
                    feature,
                    stats["count"],
                    stats["unique_count"],
                    f"{stats['min']:.17g}",
                    f"{stats['max']:.17g}",
                    f"{stats['mean']:.17g}",
                    f"{stats['std']:.17g}",
                    f"{stats['most_common_value']:.17g}",
                    stats["most_common_count"],
                ]
            )
    return output_path


def plot_dataset_feature(args: argparse.Namespace, dataset: int, feature: int, values: list[float], labels: list[int]) -> Path:
    plt = get_pyplot()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    bins = fixed_bins(args)
    stats = value_stats(values)
    classes = sorted(set(labels))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax2 = ax.twinx()
    cmap = plt.get_cmap("tab10")
    class_colors = {cls: cmap(idx % 10) for idx, cls in enumerate(classes)}

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
    legend_labels = ["All histogram", "All CDF"]

    for cls in classes:
        class_values = [value for value, label in zip(values, labels) if label == cls]
        if not class_values:
            continue
        hist_values, edges = histogram(class_values, bins, args.density)
        centers = [0.5 * (edges[idx] + edges[idx + 1]) for idx in range(len(edges) - 1)]
        hist_line, = ax.plot(
            centers,
            hist_values,
            color=class_colors[cls],
            linewidth=1.0,
            alpha=0.95,
            label=f"class {cls} hist",
        )
        cdf_x, cdf_y = empirical_cdf(class_values)
        cdf_line, = ax2.plot(
            cdf_x,
            cdf_y,
            color=class_colors[cls],
            linewidth=0.9,
            linestyle="--",
            alpha=0.95,
            label=f"class {cls} CDF",
        )
        legend_handles.extend([hist_line, cdf_line])
        legend_labels.extend([f"Class {cls} histogram", f"Class {cls} CDF"])

    if args.log_y:
        ax.set_yscale("log")

    ax.set_title(
        f"dataset {dataset}: feature {feature} value distribution and CDFs "
        f"({args.scope}, samples={stats['count']}, unique={stats['unique_count']}, "
        f"bins={args.num_levels} from {args.x_min:g} to {args.x_max:g})"
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Hist density" if args.density else "Hist count")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("CDF")
    ax.grid(True, alpha=0.2)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=min(4, len(legend_handles)),
        fontsize=9,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.955])

    output_path = (
        args.plots_dir
        / f"feature_hist_cdf_{args.scope}_dataset_{dataset:02d}_feature_{feature:03d}_levels_{args.num_levels:03d}.png"
    )
    fig.savefig(output_path, dpi=220)
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    return output_path


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.results_dir = args.results_dir.resolve()
    args.plots_dir = args.plots_dir.resolve()

    records: dict[tuple[int, int], tuple[list[float], list[int]]] = {}
    for dataset in args.datasets:
        for feature in args.features:
            records[(dataset, feature)] = load_feature_values_and_labels(args, dataset, feature)

    summary_path = write_summary(args, records)
    print(f"Wrote summary: {summary_path}")

    for feature in args.features:
        for dataset in args.datasets:
            values, labels = records[(dataset, feature)]
            plot_path = plot_dataset_feature(args, dataset, feature, values, labels)
            print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
