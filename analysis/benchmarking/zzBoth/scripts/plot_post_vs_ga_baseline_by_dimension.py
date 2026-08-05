#!/usr/bin/env python3
"""Plot quantizer+GA test accuracy over dimensions against GA-only baseline."""

from __future__ import annotations

import argparse
import csv
import html
import re
import statistics
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency: matplotlib. Run this script with the Python 3 "
        "environment used by the existing analysis scripts."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
ZZBOTH_DIR = SCRIPT_DIR.parent
BENCHMARKING_DIR = ZZBOTH_DIR.parent
DEFAULT_RUNS_ROOT = BENCHMARKING_DIR.parent / "resource_saving" / "quantizer_and_cim" / "runs"
DEFAULT_BASELINE_RUNS_DIR = BENCHMARKING_DIR.parent / "resource_saving" / "cim_only" / "runs"
DEFAULT_NOGA_BASELINE_RUNS_DIR = BENCHMARKING_DIR.parent / "resource_saving" / "baseline" / "runs"
DEFAULT_PLOTS_DIR = ZZBOTH_DIR / "plots"
DEFAULT_RESULTS_DIR = ZZBOTH_DIR / "results"
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
    "baseline": "GA-only uniform",
    "no_ga_baseline": "no-GA uniform",
    "reference": "no-opt reference @ 10k/40",
    "quantile": "quantile",
    "kmeans_1d": "k-means 1D",
    "decision_tree_1d": "decision tree 1D",
    "chimerge": "ChiMerge",
}
COLORS = {
    "baseline": "#222222",
    "no_ga_baseline": "#8c8c8c",
    "reference": "#9467bd",
    "quantile": "#1f77b4",
    "kmeans_1d": "#ff7f0e",
    "decision_tree_1d": "#2ca02c",
    "chimerge": "#d62728",
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
            "Plot quantizer+GA test accuracy against GA-only uniform "
            "baseline accuracy."
        )
    )
    parser.add_argument(
        "--phase",
        choices=["preopt", "postopt"],
        default="postopt",
        help="Phase to plot. Default: postopt.",
    )
    parser.add_argument("--num-levels", type=int, default=40)
    parser.add_argument(
        "--binning-modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated modes. Default: quantile,kmeans_1d,decision_tree_1d,chimerge.",
    )
    parser.add_argument("--metric", default="overall_accuracy")
    parser.add_argument("--min-dimension", type=int)
    parser.add_argument("--max-dimension", type=int)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="Plot every Nth dimension after filtering. Default: 1.",
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--baseline-runs-dir", type=Path, default=DEFAULT_BASELINE_RUNS_DIR)
    parser.add_argument("--noga-baseline-runs-dir", type=Path, default=DEFAULT_NOGA_BASELINE_RUNS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    args.binning_modes = parse_modes(args.binning_modes)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.min_dimension is not None
        and args.max_dimension is not None
        and args.min_dimension > args.max_dimension
    ):
        raise ValueError("--min-dimension must be <= --max-dimension")
    if args.y_min is not None and args.y_max is not None and args.y_min >= args.y_max:
        raise ValueError("--y-min must be < --y-max")
    if args.sample_step < 1:
        raise ValueError("--sample-step must be >= 1")


def load_rows(
    runs_dir: Path,
    source: str,
    phase: str,
    metric: str,
    num_levels: int,
    min_dimension: int | None,
    max_dimension: int | None,
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
                dimension = int(row["vector_dimension"])
                if (
                    int(row["num_levels"]) != num_levels
                    or info_match.group("phase") != phase
                    or info_match.group("split") != "test"
                ):
                    continue
                if min_dimension is not None and dimension < min_dimension:
                    continue
                if max_dimension is not None and dimension > max_dimension:
                    continue
                rows.append(
                    {
                        "source": source,
                        "seed": seed,
                        "dataset": int(info_match.group("dataset")),
                        "vector_dimension": dimension,
                        "value": float(row[metric]),
                    }
                )
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        key = (str(row["source"]), int(row["dataset"]), int(row["vector_dimension"]))
        groups[key].append((int(row["seed"]), float(row["value"])))

    averaged = []
    for (source, dataset, dimension), values in sorted(groups.items()):
        measurements = [value for _, value in values]
        seeds = sorted({seed for seed, _ in values})
        averaged.append(
            {
                "source": source,
                "dataset": dataset,
                "vector_dimension": dimension,
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
        groups[(str(row["source"]), int(row["vector_dimension"]))].append(row)

    with_average = list(rows)
    for (source, dimension), values in sorted(groups.items()):
        means = [float(row["mean"]) for row in values]
        if not means:
            continue
        seed_sets = [set(row["seeds"]) for row in values]
        common_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
        with_average.append(
            {
                "source": source,
                "dataset": -1,
                "vector_dimension": dimension,
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
            int(row["vector_dimension"]),
        ),
    )


def sample_dimensions(rows: list[dict[str, object]], sample_step: int) -> list[dict[str, object]]:
    if sample_step == 1:
        return rows
    sampled = []
    for dataset in sorted({int(row["dataset"]) for row in rows}):
        dimensions = sorted(
            {int(row["vector_dimension"]) for row in rows if int(row["dataset"]) == dataset}
        )
        keep = set(dimensions[::sample_step])
        sampled.extend(
            row
            for row in rows
            if int(row["dataset"]) == dataset and int(row["vector_dimension"]) in keep
        )
    return sampled


def keep_complete_dimensions(rows: list[dict[str, object]], sources: list[str]) -> list[dict[str, object]]:
    complete_rows = []
    for dataset in sorted({int(row["dataset"]) for row in rows}):
        available_by_source = {
            source: {
                int(row["vector_dimension"])
                for row in rows
                if int(row["dataset"]) == dataset and str(row["source"]) == source
            }
            for source in sources
        }
        if any(not dimensions for dimensions in available_by_source.values()):
            continue
        complete_dimensions = set.intersection(*available_by_source.values())
        complete_rows.extend(
            row
            for row in rows
            if int(row["dataset"]) == dataset
            and int(row["vector_dimension"]) in complete_dimensions
        )
    return complete_rows


def reference_values(rows: list[dict[str, object]]) -> dict[int, float]:
    return {
        int(row["dataset"]): float(row["mean"])
        for row in rows
        if str(row["source"]) == "reference"
    }


def print_configurations(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    dimensions = sorted({int(row["vector_dimension"]) for row in rows})
    datasets = sorted({int(row["dataset"]) for row in rows})
    sources = sorted({str(row["source"]) for row in rows})
    seeds = sorted({seed for row in rows for seed in row["seeds"]})

    print("Selected quantizer+GA-vs-GA-only configurations")
    print(f"  NUM_LEVELS: {args.num_levels}")
    print(f"  phase: {args.phase}-test")
    print(f"  metric: {args.metric}")
    print(f"  sources: {', '.join(LABELS.get(source, source) for source in sources)}")
    print(f"  reference: no optimization preopt-test at dim={REFERENCE_DIMENSION}, levels={REFERENCE_NUM_LEVELS}")
    dataset_text = ", ".join("average" if dataset < 0 else str(dataset) for dataset in datasets)
    print(f"  datasets: {dataset_text}")
    print(f"  seeds: {', '.join(map(str, seeds))}")
    if args.min_dimension is not None or args.max_dimension is not None:
        print(f"  dimension range: {args.min_dimension}..{args.max_dimension}")
    if args.y_min is not None or args.y_max is not None:
        print(f"  y range: {args.y_min}..{args.y_max}")
    if args.sample_step != 1:
        print(f"  sample step: every {args.sample_step} dimension value")
    print(f"  dimensions ({len(dimensions)}): {', '.join(map(str, dimensions))}")


def output_suffix(args: argparse.Namespace) -> str:
    parts = [f"levels_{args.num_levels:03d}", "modes_" + "_".join(args.binning_modes)]
    if args.min_dimension is not None or args.max_dimension is not None:
        lo = "min" if args.min_dimension is None else str(args.min_dimension)
        hi = "max" if args.max_dimension is None else str(args.max_dimension)
        parts.append(f"dim_{lo}_{hi}")
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
            fieldnames=[
                "source",
                "dataset",
                "vector_dimension",
                "mean",
                "std",
                "seed_count",
                "seeds",
            ],
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
    sources = ["baseline", "no_ga_baseline"] + args.binning_modes
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
                [int(row["vector_dimension"]) for row in values],
                [float(row["mean"]) for row in values],
                marker="o" if source not in {"baseline", "no_ga_baseline"} else None,
                markersize=3.0,
                linewidth=1.8,
                linestyle=":" if source in {"baseline", "no_ga_baseline"} else "-",
                color=COLORS.get(source),
                label=LABELS.get(source, source),
            )
        if dataset in references:
            ax.axhline(
                references[dataset],
                color=COLORS["reference"],
                linestyle="-",
                linewidth=1.3,
                label=LABELS["reference"],
            )
        ax.set_title("Dataset average" if dataset < 0 else f"Dataset {dataset}")
        ax.set_ylabel("Mean accuracy")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[:, 0]:
        ax.set_xlim(
            args.min_dimension if args.min_dimension is not None else None,
            args.max_dimension if args.max_dimension is not None else None,
        )
        ax.set_ylim(
            args.y_min if args.y_min is not None else 0.0,
            args.y_max if args.y_max is not None else 1.0,
        )
    axes[-1, 0].set_xlabel("Vector dimension")
    fig.suptitle(
        f"Quantizer+GA vs GA-only uniform test {args.metric.replace('_', ' ')} "
        f"at {args.num_levels} levels"
    )
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
    sources = ["baseline", "no_ga_baseline"] + args.binning_modes
    dimensions = sorted({int(row["vector_dimension"]) for row in rows})
    means = {
        dataset: {
            source: {
                int(row["vector_dimension"]): float(row["mean"])
                for row in rows
                if int(row["dataset"]) == dataset and str(row["source"]) == source
            }
            for source in sources
        }
        for dataset in datasets
    }
    title = (
        f"Quantizer+GA vs GA-only uniform test {args.metric.replace('_', ' ')} "
        f"at {args.num_levels} levels"
    )
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
  <h1>{html.escape(title)}</h1>
  <div id="plot" style="width: 1200px; height: {max(420, 300 * len(datasets))}px;"></div>
  <script>
    const datasets = {datasets!r};
    const sources = {sources!r};
    const labels = {LABELS!r};
    const colors = {COLORS!r};
    const dimensions = {dimensions!r};
    const means = {means!r};
    const references = {references!r};
    const traces = [];
    datasets.forEach((dataset, datasetIndex) => {{
      sources.forEach(source => {{
        const x = dimensions.filter(d => means[dataset][source][d] !== undefined);
        if (x.length === 0) return;
        traces.push({{
          x,
          y: x.map(d => means[dataset][source][d]),
          mode: (source === "baseline" || source === "no_ga_baseline") ? "lines" : "lines+markers",
          name: `${{labels[source] || source}} / ${{dataset < 0 ? "dataset average" : "dataset " + dataset}}`,
          xaxis: `x${{datasetIndex + 1}}`,
          yaxis: `y${{datasetIndex + 1}}`,
          line: {{ color: colors[source], dash: (source === "baseline" || source === "no_ga_baseline") ? "dot" : "solid" }},
          marker: {{ size: 5 }}
        }});
      }});
      if (references[dataset] !== undefined) {{
        traces.push({{
          x: dimensions,
          y: dimensions.map(_ => references[dataset]),
          mode: "lines",
          name: `${{labels.reference}} / ${{dataset < 0 ? "dataset average" : "dataset " + dataset}}`,
          xaxis: `x${{datasetIndex + 1}}`,
          yaxis: `y${{datasetIndex + 1}}`,
          line: {{ color: colors.reference, dash: "solid", width: 1.3 }}
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
      layout[`xaxis${{suffix}}`] = {{ title: idx === datasets.length - 1 ? "Vector dimension" : "" }};
      layout[`yaxis${{suffix}}`] = {{ title: dataset < 0 ? "Dataset average" : `Dataset ${{dataset}}`, tickformat: ".0%", range: [{y_min}, {y_max}] }};
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
        args.phase,
        args.metric,
        args.num_levels,
        args.min_dimension,
        args.max_dimension,
    )
    rows.extend(
        load_rows(
            args.noga_baseline_runs_dir,
            "no_ga_baseline",
            "preopt",
            args.metric,
            args.num_levels,
            args.min_dimension,
            args.max_dimension,
        )
    )
    reference_rows = load_rows(
        args.noga_baseline_runs_dir,
        "reference",
        "preopt",
        args.metric,
        REFERENCE_NUM_LEVELS,
        REFERENCE_DIMENSION,
        REFERENCE_DIMENSION,
    )
    for mode in args.binning_modes:
        rows.extend(
            load_rows(
                args.runs_root / f"binning_mode_{mode}",
                mode,
                args.phase,
                args.metric,
                args.num_levels,
                args.min_dimension,
                args.max_dimension,
            )
        )
    if not rows:
        raise RuntimeError("No rows selected")

    sources = ["baseline", "no_ga_baseline"] + args.binning_modes
    averaged = sample_dimensions(
        keep_complete_dimensions(add_dataset_average(aggregate(rows)), sources),
        args.sample_step,
    )
    references = reference_values(add_dataset_average(aggregate(reference_rows)))
    print_configurations(averaged, args)

    suffix = output_suffix(args)
    output = args.output or DEFAULT_PLOTS_DIR / f"zzboth_vs_ga_baseline_by_dimension_{args.phase}_{suffix}.png"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = (DEFAULT_RESULTS_DIR / f"zzboth_vs_ga_baseline_by_dimension_{args.phase}_{suffix}.csv").resolve()
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
