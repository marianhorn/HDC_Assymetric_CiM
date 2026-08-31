#!/usr/bin/env python3
"""Plot quantizer-only test accuracy over NUM_LEVELS against uniform baseline."""

from __future__ import annotations

import argparse
import csv
import html
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency: matplotlib. Run this script with the Python 3 "
        "environment used by the existing analysis scripts."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
QUANTIZER_DIR = SCRIPT_DIR.parent
PLOTTING_DIR = QUANTIZER_DIR.parent
DEFAULT_RUNS_ROOT = PLOTTING_DIR.parent / "experiment_runners" / "quantizer" / "runs"
DEFAULT_BASELINE_RUNS_DIR = PLOTTING_DIR.parent / "experiment_runners" / "baseline" / "runs"
DEFAULT_PLOTS_DIR = QUANTIZER_DIR / "plots"
DEFAULT_RESULTS_DIR = QUANTIZER_DIR / "results"
REFERENCE_DIMENSION = 10000
REFERENCE_NUM_LEVELS = 40
INFO_RE = re.compile(
    r"scope=dataset,dataset=(?P<dataset>\d+),"
    r"phase=(?P<phase>preopt|postopt)-(?P<split>validation|test)"
)
DEFAULT_MODES = ["quantile", "kmeans_1d", "decision_tree_1d", "chimerge"]
MODE_ALIASES = {
    "1": "quantile",
    "2": "kmeans_1d",
    "3": "decision_tree_1d",
    "4": "chimerge",
    "quantile": "quantile",
    "kmeans": "kmeans_1d",
    "kmeans_1d": "kmeans_1d",
    "decision_tree": "decision_tree_1d",
    "decision_tree_1d": "decision_tree_1d",
    "chimerge": "chimerge",
}
LABELS = {
    "baseline": "Uniform binning",
    "reference": "Reference accuracy of baseline model with $D=10{,}000$ and $L=40$",
    "quantile": "Quantile binning",
    "kmeans_1d": "k-means binning",
    "decision_tree_1d": "Decision-tree binning",
    "chimerge": "ChiMerge binning",
}
COLORS = {
    "baseline": "#0b3d91",
    "reference": "#000000",
    "quantile": "#d62728",
    "kmeans_1d": "#ff7f0e",
    "decision_tree_1d": "#2ca02c",
    "chimerge": "#9467bd",
}


def parse_modes(text: str) -> list[str]:
    modes = []
    for item in text.split(","):
        key = item.strip().lower().replace("-", "_")
        if not key:
            continue
        if key not in MODE_ALIASES:
            valid = ", ".join(DEFAULT_MODES + ["1", "2", "3", "4"])
            raise ValueError(f"Unknown quantizer mode {item!r}. Valid: {valid}")
        mode = MODE_ALIASES[key]
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("--binning-modes must select at least one mode")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot learned quantizer preopt-test accuracy against uniform "
            "baseline preopt-test accuracy over NUM_LEVELS."
        )
    )
    parser.add_argument("--vector-dimension", type=int, default=10000)
    parser.add_argument(
        "--binning-modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated modes. Default: quantile,kmeans_1d,decision_tree_1d,chimerge.",
    )
    parser.add_argument("--metric", default="overall_accuracy")
    parser.add_argument("--min-levels", type=int)
    parser.add_argument("--max-levels", type=int)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="Plot every Nth NUM_LEVELS value after filtering. Default: 1.",
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--baseline-runs-dir", type=Path, default=DEFAULT_BASELINE_RUNS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    args.binning_modes = parse_modes(args.binning_modes)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.min_levels is not None
        and args.max_levels is not None
        and args.min_levels > args.max_levels
    ):
        raise ValueError("--min-levels must be <= --max-levels")
    if args.y_min is not None and args.y_max is not None and args.y_min >= args.y_max:
        raise ValueError("--y-min must be < --y-max")
    if args.sample_step < 1:
        raise ValueError("--sample-step must be >= 1")


def load_rows(
    runs_dir: Path,
    source: str,
    metric: str,
    vector_dimension: int,
    min_levels: int | None,
    max_levels: int | None,
) -> list[dict[str, object]]:
    result_paths = sorted(runs_dir.resolve().glob("seed_*/results.csv"))
    if not result_paths:
        raise FileNotFoundError(f"No result files found under {runs_dir}")

    rows = []
    for path in result_paths:
        seed_match = re.fullmatch(r"seed_(\d+)", path.parent.name)
        if seed_match is None:
            continue
        seed = int(seed_match.group(1))
        with path.open(newline="", encoding="utf-8") as result_file:
            reader = csv.DictReader(result_file)
            if metric not in (reader.fieldnames or []):
                raise KeyError(f"Metric column not found in {path}: {metric}")
            for row in reader:
                info_match = INFO_RE.search(row["info"])
                if info_match is None:
                    continue
                levels = int(row["num_levels"])
                if (
                    int(row["vector_dimension"]) != vector_dimension
                    or info_match.group("phase") != "preopt"
                    or info_match.group("split") != "test"
                ):
                    continue
                if min_levels is not None and levels < min_levels:
                    continue
                if max_levels is not None and levels > max_levels:
                    continue
                rows.append(
                    {
                        "source": source,
                        "seed": seed,
                        "dataset": int(info_match.group("dataset")),
                        "num_levels": levels,
                        "value": float(row[metric]),
                    }
                )
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        key = (str(row["source"]), int(row["dataset"]), int(row["num_levels"]))
        groups[key].append((int(row["seed"]), float(row["value"])))

    averaged = []
    for (source, dataset, levels), values in sorted(groups.items()):
        measurements = [value for _, value in values]
        seeds = sorted({seed for seed, _ in values})
        averaged.append(
            {
                "source": source,
                "dataset": dataset,
                "num_levels": levels,
                "mean": statistics.fmean(measurements),
                "std": statistics.stdev(measurements) if len(measurements) > 1 else 0.0,
                "seed_count": len(seeds),
                "seeds": seeds,
            }
        )
    return averaged


def add_dataset_average(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["dataset"]) < 0:
            continue
        groups[(str(row["source"]), int(row["num_levels"]))].append(row)

    with_average = list(rows)
    for (source, levels), values in sorted(groups.items()):
        means = [float(row["mean"]) for row in values]
        if not means:
            continue
        seed_sets = [set(row["seeds"]) for row in values]
        common_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
        with_average.append(
            {
                "source": source,
                "dataset": -1,
                "num_levels": levels,
                "mean": statistics.fmean(means),
                "std": statistics.stdev(means) if len(means) > 1 else 0.0,
                "seed_count": len(common_seeds),
                "seeds": common_seeds,
            }
        )
    return sorted(
        with_average,
        key=lambda row: (
            int(row["dataset"]) if int(row["dataset"]) >= 0 else 999,
            str(row["source"]),
            int(row["num_levels"]),
        ),
    )


def sample_levels(rows: list[dict[str, object]], sample_step: int) -> list[dict[str, object]]:
    if sample_step == 1:
        return rows
    sampled = []
    for dataset in sorted({int(row["dataset"]) for row in rows}):
        levels = sorted({int(row["num_levels"]) for row in rows if int(row["dataset"]) == dataset})
        keep = set(levels[::sample_step])
        sampled.extend(
            row
            for row in rows
            if int(row["dataset"]) == dataset and int(row["num_levels"]) in keep
        )
    return sampled


def keep_common_levels(rows: list[dict[str, object]], sources: list[str]) -> list[dict[str, object]]:
    filtered = []
    for dataset in sorted({int(row["dataset"]) for row in rows}):
        dataset_rows = [row for row in rows if int(row["dataset"]) == dataset]
        levels_by_source = {
            source: {
                int(row["num_levels"])
                for row in dataset_rows
                if str(row["source"]) == source
            }
            for source in sources
        }
        if any(not levels for levels in levels_by_source.values()):
            continue
        common_levels = set.intersection(*levels_by_source.values())
        filtered.extend(
            row
            for row in dataset_rows
            if int(row["num_levels"]) in common_levels
        )
    return sorted(
        filtered,
        key=lambda row: (
            int(row["dataset"]) if int(row["dataset"]) >= 0 else 999,
            str(row["source"]),
            int(row["num_levels"]),
        ),
    )


def reference_values(rows: list[dict[str, object]]) -> dict[int, float]:
    return {
        int(row["dataset"]): float(row["mean"])
        for row in rows
        if str(row["source"]) == "reference"
    }


def dataset_label(dataset: int) -> str:
    return "Dataset average" if dataset < 0 else f"Dataset {dataset}"


def format_percent(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{100.0 * value:.2f}%"


def format_pp(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{100.0 * value:+.2f}pp"


def summarize_grid(values: list[int]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return str(values[0])

    ranges = []
    start = values[0]
    prev = values[0]
    step = None
    for current in values[1:]:
        current_step = current - prev
        if step is None:
            step = current_step
        elif current_step != step:
            ranges.append((start, prev, step))
            start = prev
            step = current_step
        prev = current
    ranges.append((start, prev, step))

    parts = []
    for lo, hi, interval in ranges:
        if lo == hi:
            parts.append(str(lo))
        else:
            parts.append(f"{lo}..{hi} step {interval}")
    return "; ".join(parts)


def print_level_grid(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    print()
    print("NUM_LEVELS grid by plotted source after common-level filtering")
    print("source levels range_summary")
    ordered_sources = ["baseline"] + args.binning_modes
    for source in ordered_sources:
        levels = sorted(
            {
                int(row["num_levels"])
                for row in rows
                if str(row["source"]) == source and int(row["dataset"]) >= 0
            }
        )
        print(
            f"{LABELS.get(source, source):>22s} "
            f"{len(levels):6d} "
            f"{summarize_grid(levels)}"
        )


def print_main_results(
    rows: list[dict[str, object]],
    references: dict[int, float],
    args: argparse.Namespace,
) -> None:
    print()
    print("Main results")
    print(
        "dataset source endpoint_levels endpoint_acc diff_to_uniform "
        "reference_acc diff_to_reference best_levels best_acc"
    )

    sources = ["baseline"] + args.binning_modes
    datasets = sorted({int(row["dataset"]) for row in rows})
    for dataset in datasets:
        dataset_rows = [row for row in rows if int(row["dataset"]) == dataset]
        endpoint_levels = max(int(row["num_levels"]) for row in dataset_rows)
        baseline_endpoint = next(
            (
                float(row["mean"])
                for row in dataset_rows
                if str(row["source"]) == "baseline"
                and int(row["num_levels"]) == endpoint_levels
            ),
            None,
        )
        reference = references.get(dataset)

        for source in sources:
            source_rows = [row for row in dataset_rows if str(row["source"]) == source]
            if not source_rows:
                continue
            endpoint_row = next(
                (
                    row
                    for row in source_rows
                    if int(row["num_levels"]) == endpoint_levels
                ),
                max(source_rows, key=lambda row: int(row["num_levels"])),
            )
            endpoint_acc = float(endpoint_row["mean"])
            best_row = max(source_rows, key=lambda row: float(row["mean"]))
            diff_to_uniform = (
                None
                if baseline_endpoint is None
                else endpoint_acc - baseline_endpoint
            )
            diff_to_reference = None if reference is None else endpoint_acc - reference
            print(
                f"{dataset_label(dataset):>15s} "
                f"{LABELS.get(source, source):>22s} "
                f"{int(endpoint_row['num_levels']):15d} "
                f"{format_percent(endpoint_acc):>12s} "
                f"{format_pp(diff_to_uniform):>15s} "
                f"{format_percent(reference):>13s} "
                f"{format_pp(diff_to_reference):>17s} "
                f"{int(best_row['num_levels']):11d} "
                f"{format_percent(float(best_row['mean'])):>9s}"
            )


def print_configurations(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    levels = sorted({int(row["num_levels"]) for row in rows})
    datasets = sorted({int(row["dataset"]) for row in rows})
    sources = sorted({str(row["source"]) for row in rows})
    seeds = sorted({seed for row in rows for seed in row["seeds"]})

    print("Selected quantizer-vs-baseline configurations")
    print(f"  VECTOR_DIMENSION: {args.vector_dimension}")
    print("  phase: preopt-test")
    print(f"  metric: {args.metric}")
    print(f"  sources: {', '.join(LABELS.get(source, source) for source in sources)}")
    print(
        "  reference: baseline model preopt-test at "
        f"D={REFERENCE_DIMENSION}, L={REFERENCE_NUM_LEVELS}"
    )
    dataset_text = ", ".join("average" if dataset < 0 else str(dataset) for dataset in datasets)
    print(f"  datasets: {dataset_text}")
    print(f"  seeds: {', '.join(map(str, seeds))}")
    if args.min_levels is not None or args.max_levels is not None:
        print(f"  NUM_LEVELS range: {args.min_levels}..{args.max_levels}")
    if args.y_min is not None or args.y_max is not None:
        print(f"  y range: {args.y_min}..{args.y_max}")
    if args.sample_step != 1:
        print(f"  sample step: every {args.sample_step} NUM_LEVELS value")
    print(f"  NUM_LEVELS ({len(levels)}): {', '.join(map(str, levels))}")


def output_suffix(args: argparse.Namespace) -> str:
    parts = [f"dim_{args.vector_dimension:05d}", "modes_" + "_".join(args.binning_modes)]
    if args.min_levels is not None or args.max_levels is not None:
        lo = "min" if args.min_levels is None else str(args.min_levels)
        hi = "max" if args.max_levels is None else str(args.max_levels)
        parts.append(f"levels_{lo}_{hi}")
    if args.y_min is not None or args.y_max is not None:
        lo = "min" if args.y_min is None else f"{args.y_min:g}"
        hi = "max" if args.y_max is None else f"{args.y_max:g}"
        parts.append(f"y_{lo}_{hi}")
    if args.sample_step != 1:
        parts.append(f"step_{args.sample_step}")
    return "_".join(parts).replace(".", "p")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["source", "dataset", "num_levels", "mean", "std", "seed_count", "seeds"],
        )
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["seeds"] = ",".join(map(str, row["seeds"]))
            writer.writerow(record)


def plot_png(
    path: Path,
    rows: list[dict[str, object]],
    references: dict[int, float],
    args: argparse.Namespace,
) -> None:
    datasets = sorted({int(row["dataset"]) for row in rows})
    sources = ["baseline"] + args.binning_modes
    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(11, max(3.2, 2.7 * len(datasets))),
        sharex=True,
        squeeze=False,
    )

    for ax, dataset in zip(axes[:, 0], datasets):
        for source in sources:
            values = [
                row
                for row in rows
                if int(row["dataset"]) == dataset and str(row["source"]) == source
            ]
            if not values:
                continue
            ax.plot(
                [int(row["num_levels"]) for row in values],
                [float(row["mean"]) for row in values],
                linewidth=1.8,
                linestyle="--" if source == "baseline" else "-",
                color=COLORS.get(source),
                label=LABELS.get(source, source),
            )
        if dataset in references:
            ax.axhline(
                references[dataset],
                color=COLORS["reference"],
                linestyle=":",
                linewidth=1.8,
                label=LABELS["reference"],
            )
        ax.set_title("Dataset average" if dataset < 0 else f"Dataset {dataset}")
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right" if dataset == 2 else "lower right")

    for ax in axes[:, 0]:
        ax.set_xlim(
            args.min_levels if args.min_levels is not None else None,
            args.max_levels if args.max_levels is not None else None,
        )
        ax.set_ylim(
            args.y_min if args.y_min is not None else 0.0,
            args.y_max if args.y_max is not None else 1.0,
        )
    axes[-1, 0].set_xlabel("Number of levels")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(fig)


def write_html(
    path: Path,
    rows: list[dict[str, object]],
    references: dict[int, float],
    args: argparse.Namespace,
) -> None:
    datasets = sorted({int(row["dataset"]) for row in rows})
    sources = ["baseline"] + args.binning_modes
    levels = sorted({int(row["num_levels"]) for row in rows})
    means = {
        dataset: {
            source: {
                int(row["num_levels"]): float(row["mean"])
                for row in rows
                if int(row["dataset"]) == dataset and str(row["source"]) == source
            }
            for source in sources
        }
        for dataset in datasets
    }
    title = f"Quantizer-only comparison at dimension {args.vector_dimension}"
    y_min = 0.0 if args.y_min is None else args.y_min
    y_max = 1.0 if args.y_max is None else args.y_max
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div id="plot" style="width: 1200px; height: {max(420, 300 * len(datasets))}px;"></div>
  <script>
    const datasets = {datasets!r};
    const sources = {sources!r};
    const labels = {LABELS!r};
    const colors = {COLORS!r};
    const levels = {levels!r};
    const means = {means!r};
    const references = {references!r};
    const traces = [];
    datasets.forEach((dataset, datasetIndex) => {{
      sources.forEach(source => {{
        const x = levels.filter(level => means[dataset][source][level] !== undefined);
        if (x.length === 0) return;
        traces.push({{
          x,
          y: x.map(level => means[dataset][source][level]),
          mode: "lines",
          name: `${{labels[source] || source}} / ${{dataset < 0 ? "dataset average" : "dataset " + dataset}}`,
          xaxis: `x${{datasetIndex + 1}}`,
          yaxis: `y${{datasetIndex + 1}}`,
          line: {{ color: colors[source], dash: source === "baseline" ? "dash" : "solid" }}
        }});
      }});
      if (references[dataset] !== undefined) {{
        traces.push({{
          x: levels,
          y: levels.map(_ => references[dataset]),
          mode: "lines",
          name: `${{labels.reference}} / ${{dataset < 0 ? "dataset average" : "dataset " + dataset}}`,
          xaxis: `x${{datasetIndex + 1}}`,
          yaxis: `y${{datasetIndex + 1}}`,
          line: {{ color: colors.reference, dash: "dot", width: 1.8 }}
        }});
      }}
    }});
    const layout = {{
      grid: {{ rows: datasets.length, columns: 1, pattern: "independent" }},
      showlegend: true,
      margin: {{ l: 75, r: 30, t: 30, b: 70 }}
    }};
    datasets.forEach((dataset, idx) => {{
      const suffix = idx === 0 ? "" : String(idx + 1);
      layout[`xaxis${{suffix}}`] = {{ title: idx === datasets.length - 1 ? "Number of levels" : "" }};
      layout[`yaxis${{suffix}}`] = {{ title: "Accuracy", tickformat: ".0%", range: [{y_min}, {y_max}] }};
    }});
    Plotly.newPlot("plot", traces, layout);
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(
        args.baseline_runs_dir,
        "baseline",
        args.metric,
        args.vector_dimension,
        args.min_levels,
        args.max_levels,
    )
    for mode in args.binning_modes:
        rows.extend(
            load_rows(
                args.runs_root / f"binning_mode_{mode}",
                mode,
                args.metric,
                args.vector_dimension,
                args.min_levels,
                args.max_levels,
            )
        )
    reference_rows = load_rows(
        args.baseline_runs_dir,
        "reference",
        args.metric,
        REFERENCE_DIMENSION,
        REFERENCE_NUM_LEVELS,
        REFERENCE_NUM_LEVELS,
    )
    if not rows:
        raise RuntimeError("No rows selected")

    sources = ["baseline"] + args.binning_modes
    averaged = add_dataset_average(aggregate(rows))
    averaged = keep_common_levels(averaged, sources)
    averaged = sample_levels(averaged, args.sample_step)
    references = reference_values(add_dataset_average(aggregate(reference_rows)))
    print_configurations(averaged, args)
    print_level_grid(averaged, args)
    print_main_results(averaged, references, args)

    suffix = output_suffix(args)
    output = args.output or DEFAULT_PLOTS_DIR / f"quantizer_vs_baseline_by_levels_{suffix}.png"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = (DEFAULT_RESULTS_DIR / f"quantizer_vs_baseline_by_levels_{suffix}.csv").resolve()
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    html_output = output.with_suffix(".html")

    plot_png(output, averaged, references, args)
    write_csv(csv_output, averaged)
    write_html(html_output, averaged, references, args)
    print()
    print(f"Saved PNG plot:      {output}")
    print(f"Saved HTML plot:     {html_output}")
    print(f"Saved averaged data: {csv_output}")


if __name__ == "__main__":
    main()
