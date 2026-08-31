#!/usr/bin/env python3
"""Plot seed-averaged baseline accuracy over dimensions at fixed levels."""

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
BASELINE_DIR = SCRIPT_DIR.parent
PLOTTING_DIR = BASELINE_DIR.parent
DEFAULT_RUNS_DIR = PLOTTING_DIR.parent / "experiment_runners" / "baseline" / "runs"
DEFAULT_PLOTS_DIR = BASELINE_DIR / "plots"
DEFAULT_RESULTS_DIR = BASELINE_DIR / "results"
INFO_RE = re.compile(
    r"scope=dataset,dataset=(?P<dataset>\d+),"
    r"phase=(?P<phase>preopt|postopt)-(?P<split>validation|test)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one baseline accuracy line per dataset over dimensions."
    )
    parser.add_argument("--num-levels", type=int, default=40)
    parser.add_argument("--metric", default="overall_accuracy")
    parser.add_argument("--phase", choices=["preopt", "postopt"], default="preopt")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--min-dimension", type=int)
    parser.add_argument("--max-dimension", type=int)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument(
        "--dimension-grid",
        choices=["thesis", "all"],
        default="thesis",
        help=(
            "Dimension grid for plotting. 'thesis' keeps 100-step points below "
            "1000, 500-step points from 1000 to 10000, and 1000-step points "
            "from 10000 upward. 'all' keeps every measured point."
        ),
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="Plot every Nth dimension after filtering and averaging. Default: 1.",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save without opening the interactive plot.",
    )
    args = parser.parse_args()
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


def load_selected_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    result_paths = sorted(args.runs_dir.resolve().glob("seed_*/results.csv"))
    if not result_paths:
        raise FileNotFoundError(f"No baseline results found under {args.runs_dir}")

    for path in result_paths:
        seed_match = re.fullmatch(r"seed_(\d+)", path.parent.name)
        if seed_match is None:
            continue
        seed = int(seed_match.group(1))
        with path.open(newline="", encoding="utf-8") as result_file:
            reader = csv.DictReader(result_file)
            if args.metric not in (reader.fieldnames or []):
                raise KeyError(f"Metric column not found in {path}: {args.metric}")
            for row in reader:
                info_match = INFO_RE.search(row["info"])
                if info_match is None:
                    continue
                if (
                    int(row["num_levels"]) != args.num_levels
                    or info_match.group("phase") != args.phase
                    or info_match.group("split") != args.split
                ):
                    continue
                vector_dimension = int(row["vector_dimension"])
                if args.min_dimension is not None and vector_dimension < args.min_dimension:
                    continue
                if args.max_dimension is not None and vector_dimension > args.max_dimension:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "dataset": int(info_match.group("dataset")),
                        "vector_dimension": vector_dimension,
                        "value": float(row[args.metric]),
                    }
                )

    if not rows:
        raise RuntimeError(
            f"No rows for levels={args.num_levels}, phase={args.phase}, "
            f"split={args.split}"
        )
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        key = (int(row["dataset"]), int(row["vector_dimension"]))
        groups[key].append((int(row["seed"]), float(row["value"])))

    averaged = []
    for (dataset, dimension), values in sorted(groups.items()):
        measurements = [value for _, value in values]
        seeds = sorted({seed for seed, _ in values})
        averaged.append(
            {
                "dataset": dataset,
                "vector_dimension": dimension,
                "mean": statistics.fmean(measurements),
                "std": statistics.stdev(measurements)
                if len(measurements) > 1
                else 0.0,
                "seed_count": len(seeds),
                "seeds": seeds,
            }
        )
    return averaged


def sample_dimensions(
    averaged: list[dict[str, object]], sample_step: int
) -> list[dict[str, object]]:
    if sample_step == 1:
        return averaged

    sampled = []
    for dataset in sorted({int(row["dataset"]) for row in averaged}):
        dataset_rows = [
            row for row in averaged if int(row["dataset"]) == dataset
        ]
        sampled.extend(dataset_rows[::sample_step])
    return sampled


def apply_dimension_grid(
    averaged: list[dict[str, object]], grid: str
) -> list[dict[str, object]]:
    if grid == "all":
        return averaged

    dimensions = sorted({int(row["vector_dimension"]) for row in averaged})
    low_dims = [dimension for dimension in dimensions if dimension < 1000]
    low_anchor = min(low_dims) if low_dims else 0

    def keep_dimension(dimension: int) -> bool:
        if dimension < 1000:
            return (dimension - low_anchor) % 100 == 0
        if dimension < 10000:
            return dimension % 500 == 0
        return dimension % 1000 == 0

    return [
        row
        for row in averaged
        if keep_dimension(int(row["vector_dimension"]))
    ]


def print_configurations(
    averaged: list[dict[str, object]], args: argparse.Namespace
) -> None:
    dimensions = sorted({int(row["vector_dimension"]) for row in averaged})
    seeds = sorted({seed for row in averaged for seed in row["seeds"]})
    datasets = sorted({int(row["dataset"]) for row in averaged})

    print("Selected baseline configurations")
    print(f"  NUM_LEVELS: {args.num_levels}")
    print(f"  phase/split: {args.phase}-{args.split}")
    print(f"  metric: {args.metric}")
    if args.min_dimension is not None or args.max_dimension is not None:
        print(f"  dimension range: {args.min_dimension}..{args.max_dimension}")
    if args.y_min is not None or args.y_max is not None:
        print(f"  y range: {args.y_min}..{args.y_max}")
    print(f"  dimension grid: {args.dimension_grid}")
    if args.sample_step != 1:
        print(f"  sample step: every {args.sample_step} dimension value")
    print(f"  seeds: {', '.join(map(str, seeds))}")
    print(f"  datasets: {', '.join(map(str, datasets))}")
    print(f"  dimensions ({len(dimensions)}): {', '.join(map(str, dimensions))}")
    print()
    print("All configurations used and their seed coverage:")
    print("dataset dimension seed_count seeds")
    for row in averaged:
        seed_text = ",".join(map(str, row["seeds"]))
        print(
            f"{int(row['dataset']):7d} "
            f"{int(row['vector_dimension']):9d} "
            f"{int(row['seed_count']):10d} {seed_text}"
        )


def write_csv(path: Path, averaged: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "dataset",
                "vector_dimension",
                "mean",
                "std",
                "seed_count",
                "seeds",
            ],
        )
        writer.writeheader()
        for row in averaged:
            record = dict(row)
            record["seeds"] = ",".join(map(str, row["seeds"]))
            writer.writerow(record)


def write_html(path: Path, averaged: list[dict[str, object]], args: argparse.Namespace) -> None:
    datasets = sorted({int(row["dataset"]) for row in averaged})
    dimensions = sorted({int(row["vector_dimension"]) for row in averaged})
    means_by_dataset = {
        dataset: {
            int(row["vector_dimension"]): float(row["mean"])
            for row in averaged
            if int(row["dataset"]) == dataset
        }
        for dataset in datasets
    }
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    title = (
        f"Baseline {args.metric.replace('_', ' ')} at "
        f"{args.num_levels} quantization levels"
    )
    subtitle = f"{args.phase}-{args.split}, averaged over available seeds"
    y_min = 0.0 if args.y_min is None else args.y_min
    y_max = 1.0 if args.y_max is None else args.y_max

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: Segoe UI, Calibri, sans-serif;
      margin: 28px;
      color: #1f2933;
      background: #f8fafc;
    }}
    h1 {{
      font-size: 22px;
      margin: 0 0 4px;
      font-weight: 650;
    }}
    p {{
      margin: 0 0 18px;
      color: #52606d;
    }}
    svg {{
      background: white;
      border: 1px solid #d9e2ec;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
    }}
    .grid {{
      stroke: #d9e2ec;
      stroke-width: 1;
    }}
    .axis {{
      stroke: #334e68;
      stroke-width: 1.4;
    }}
    .tick {{
      fill: #52606d;
      font-size: 11px;
    }}
    .label {{
      fill: #334e68;
      font-size: 13px;
      font-weight: 600;
    }}
    .legend text {{
      fill: #334e68;
      font-size: 13px;
    }}
    .series {{
      fill: none;
      stroke-width: 2.4;
    }}
    circle {{
      stroke: white;
      stroke-width: 1.2;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>{html.escape(subtitle)}</p>
  <svg id="chart" width="1180" height="700" role="img" aria-label="{html.escape(title)}"></svg>
  <script>
    const dimensions = {dimensions!r};
    const datasets = {datasets!r};
    const meansByDataset = {means_by_dataset!r};
    const colors = {colors!r};
    const svg = document.getElementById("chart");
    const width = 1180;
    const height = 700;
    const margin = {{ left: 78, right: 180, top: 30, bottom: 70 }};
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const minDim = Math.min(...dimensions);
    const maxDim = Math.max(...dimensions);
    const yMin = {y_min!r};
    const yMax = {y_max!r};
    const xScale = x => margin.left + ((x - minDim) / Math.max(1, maxDim - minDim)) * plotWidth;
    const yScale = y => margin.top + ((yMax - y) / (yMax - yMin)) * plotHeight;
    const makeSvg = (name, attrs, text) => {{
      const element = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key, value] of Object.entries(attrs || {{}})) {{
        element.setAttribute(key, value);
      }}
      if (text !== undefined) element.textContent = text;
      svg.appendChild(element);
      return element;
    }};
    for (let step = 0; step <= 10; step += 1) {{
      const y = yMin + ((yMax - yMin) * step / 10);
      const py = yScale(y);
      makeSvg("line", {{ x1: margin.left, y1: py, x2: width - margin.right, y2: py, class: "grid" }});
      makeSvg("text", {{ x: margin.left - 10, y: py + 4, "text-anchor": "end", class: "tick" }}, `${{Math.round(y * 100)}}%`);
    }}
    const xTicks = dimensions.filter((_, idx) => idx % Math.ceil(dimensions.length / 12) === 0);
    if (!xTicks.includes(maxDim)) xTicks.push(maxDim);
    for (const x of xTicks) {{
      const px = xScale(x);
      makeSvg("line", {{ x1: px, y1: margin.top, x2: px, y2: height - margin.bottom, class: "grid" }});
      makeSvg("text", {{ x: px, y: height - margin.bottom + 22, "text-anchor": "middle", class: "tick" }}, x);
    }}
    makeSvg("line", {{ x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: "axis" }});
    makeSvg("line", {{ x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: "axis" }});
    makeSvg("text", {{ x: margin.left + plotWidth / 2, y: height - 22, "text-anchor": "middle", class: "label" }}, "Vector dimension");
    const ylabel = makeSvg("text", {{ x: 22, y: margin.top + plotHeight / 2, "text-anchor": "middle", class: "label" }}, "Mean accuracy");
    ylabel.setAttribute("transform", `rotate(-90 22 ${{margin.top + plotHeight / 2}})`);
    datasets.forEach((dataset, datasetIndex) => {{
      const color = colors[datasetIndex % colors.length];
      const points = dimensions
        .filter(dimension => meansByDataset[dataset][dimension] !== undefined)
        .map(dimension => [xScale(dimension), yScale(meansByDataset[dataset][dimension]), dimension, meansByDataset[dataset][dimension]]);
      const pathData = points.map((point, index) => `${{index === 0 ? "M" : "L"}} ${{point[0].toFixed(2)}} ${{point[1].toFixed(2)}}`).join(" ");
      makeSvg("path", {{ d: pathData, stroke: color, class: "series" }});
      for (const [px, py, dimension, value] of points) {{
        const circle = makeSvg("circle", {{ cx: px, cy: py, r: 3.5, fill: color }});
        circle.appendChild(document.createElementNS("http://www.w3.org/2000/svg", "title")).textContent =
          `Dataset ${{dataset}}, dim ${{dimension}}: ${{(value * 100).toFixed(2)}}%`;
      }}
      const legendY = margin.top + 22 + datasetIndex * 24;
      makeSvg("line", {{ x1: width - margin.right + 25, y1: legendY, x2: width - margin.right + 55, y2: legendY, stroke: color, "stroke-width": 3 }});
      makeSvg("text", {{ x: width - margin.right + 65, y: legendY + 4, class: "legend" }}, `Dataset ${{dataset}}`);
    }});
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def plot_results(
    path: Path, averaged: list[dict[str, object]], args: argparse.Namespace
) -> None:
    datasets = sorted({int(row["dataset"]) for row in averaged})
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for dataset in datasets:
        values = [row for row in averaged if int(row["dataset"]) == dataset]
        ax.plot(
            [int(row["vector_dimension"]) for row in values],
            [float(row["mean"]) for row in values],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=f"Dataset {dataset}",
        )

    ax.set_xlabel("Vector dimension")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(
        args.min_dimension if args.min_dimension is not None else 0,
        args.max_dimension if args.max_dimension is not None else None,
    )
    ax.set_ylim(
        args.y_min if args.y_min is not None else 0.7,
        args.y_max if args.y_max is not None else 1.0,
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Dataset")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")

    if not args.no_show:
        plt.show()
    plt.close(fig)


def output_range_suffix(args: argparse.Namespace) -> str:
    parts = []
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
    if args.dimension_grid != "all":
        parts.append(args.dimension_grid)
    return "" if not parts else "_" + "_".join(parts).replace(".", "p")


def main() -> None:
    args = parse_args()
    averaged = aggregate(load_selected_rows(args))
    averaged = apply_dimension_grid(averaged, args.dimension_grid)
    averaged = sample_dimensions(averaged, args.sample_step)
    print_configurations(averaged, args)

    output = args.output or (
        DEFAULT_PLOTS_DIR
        / (
            f"baseline_levels_{args.num_levels:03d}_{args.phase}_{args.split}"
            f"{output_range_suffix(args)}.png"
        )
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = (
        DEFAULT_RESULTS_DIR
        / (
            f"baseline_levels_{args.num_levels:03d}_{args.phase}_{args.split}"
            f"{output_range_suffix(args)}.csv"
        )
    ).resolve()
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    html_output = output.with_suffix(".html")

    plot_results(output, averaged, args)
    write_csv(csv_output, averaged)
    write_html(html_output, averaged, args)
    print()
    print(f"Saved PNG plot:      {output}")
    print(f"Saved HTML plot:     {html_output}")
    print(f"Saved averaged data: {csv_output}")


if __name__ == "__main__":
    main()
