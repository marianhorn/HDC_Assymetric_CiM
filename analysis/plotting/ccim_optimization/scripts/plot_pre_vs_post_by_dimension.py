#!/usr/bin/env python3
"""Plot GA-only preopt vs postopt test accuracy over dimensions."""

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
    from matplotlib.ticker import MultipleLocator
    from matplotlib.ticker import PercentFormatter
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency: matplotlib. Run this script with the Python 3 "
        "environment used by the existing analysis scripts."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
CCIM_OPTIMIZATION_DIR = SCRIPT_DIR.parent
PLOTTING_DIR = CCIM_OPTIMIZATION_DIR.parent
DEFAULT_RUNS_DIR = PLOTTING_DIR.parent / "experiment_runners" / "ccim_optimization" / "runs"
DEFAULT_BASELINE_RUNS_DIR = (
    PLOTTING_DIR.parent / "experiment_runners" / "baseline" / "runs"
)
DEFAULT_PLOTS_DIR = CCIM_OPTIMIZATION_DIR / "plots"
DEFAULT_RESULTS_DIR = CCIM_OPTIMIZATION_DIR / "results"
INFO_RE = re.compile(
    r"scope=dataset,dataset=(?P<dataset>\d+),"
    r"phase=(?P<phase>preopt|postopt)-(?P<split>validation|test)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GA-only preopt-test vs postopt-test accuracy over dimensions."
    )
    parser.add_argument("--num-levels", type=int, default=40)
    parser.add_argument("--metric", default="overall_accuracy")
    parser.add_argument("--min-dimension", type=int)
    parser.add_argument("--max-dimension", type=int)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="Plot every Nth dimension after filtering and averaging. Default: 1.",
    )
    parser.add_argument(
        "--reference-dimension",
        type=int,
        default=10000,
        help="Draw a horizontal reference line at this dimension's averaged accuracy. Default: 10000.",
    )
    parser.add_argument(
        "--reference-phase",
        choices=["preopt", "postopt"],
        default="preopt",
        help="Baseline phase used for the reference line. Default: preopt.",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--reference-runs-dir",
        type=Path,
        default=DEFAULT_BASELINE_RUNS_DIR,
        help="Baseline runs directory used for the horizontal reference lines.",
    )
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
        raise FileNotFoundError(f"No GA-only results found under {args.runs_dir}")

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
                    or info_match.group("split") != "test"
                    or info_match.group("phase") not in {"preopt", "postopt"}
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
                        "phase": info_match.group("phase"),
                        "vector_dimension": vector_dimension,
                        "value": float(row[args.metric]),
                    }
                )

    if not rows:
        raise RuntimeError(f"No rows for levels={args.num_levels}, split=test")
    return rows


def load_reference_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    result_paths = sorted(args.reference_runs_dir.resolve().glob("seed_*/results.csv"))
    if not result_paths:
        raise FileNotFoundError(
            f"No baseline reference results found under {args.reference_runs_dir}"
        )

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
                    or int(row["vector_dimension"]) != args.reference_dimension
                    or info_match.group("split") != "test"
                    or info_match.group("phase") != args.reference_phase
                ):
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "dataset": int(info_match.group("dataset")),
                        "phase": args.reference_phase,
                        "vector_dimension": args.reference_dimension,
                        "value": float(row[args.metric]),
                    }
                )

    if not rows:
        raise RuntimeError(
            f"No baseline reference rows for levels={args.num_levels}, "
            f"dimension={args.reference_dimension}, phase={args.reference_phase}-test"
        )
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["dataset"]),
            str(row["phase"]),
            int(row["vector_dimension"]),
        )
        groups[key].append((int(row["seed"]), float(row["value"])))

    averaged = []
    for (dataset, phase, dimension), values in sorted(groups.items()):
        measurements = [value for _, value in values]
        seeds = sorted({seed for seed, _ in values})
        averaged.append(
            {
                "dataset": dataset,
                "phase": phase,
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
        dimensions = sorted(
            {int(row["vector_dimension"]) for row in averaged if int(row["dataset"]) == dataset}
        )
        keep = set(dimensions[::sample_step])
        sampled.extend(
            row
            for row in averaged
            if int(row["dataset"]) == dataset and int(row["vector_dimension"]) in keep
        )
    return sampled


def add_dataset_average(
    averaged: list[dict[str, object]]
) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in averaged:
        if int(row["dataset"]) < 0:
            continue
        groups[(str(row["phase"]), int(row["vector_dimension"]))].append(row)

    with_average = list(averaged)
    for (phase, dimension), values in sorted(groups.items()):
        means = [float(row["mean"]) for row in values]
        if not means:
            continue
        seed_sets = [set(row["seeds"]) for row in values]
        common_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
        with_average.append(
            {
                "dataset": -1,
                "phase": phase,
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
            int(row["vector_dimension"]),
            str(row["phase"]),
        ),
    )


def reference_values(
    averaged: list[dict[str, object]], args: argparse.Namespace
) -> dict[int, float]:
    values = {}
    for row in averaged:
        if (
            str(row["phase"]) == args.reference_phase
            and int(row["vector_dimension"]) == args.reference_dimension
        ):
            values[int(row["dataset"])] = float(row["mean"])
    return values


def dataset_label(dataset: int) -> str:
    return "dataset average" if dataset < 0 else str(dataset)


def phase_label(phase: str) -> str:
    return {
        "preopt": "Baseline model",
        "postopt": "GA-optimized model",
    }.get(phase, phase)


def reference_label(args: argparse.Namespace) -> str:
    dimension = f"{args.reference_dimension:,}".replace(",", "{,}")
    return (
        "Reference accuracy of baseline model "
        f"with $D={dimension}$ and $L={args.num_levels}$"
    )


def print_configurations(
    averaged: list[dict[str, object]],
    args: argparse.Namespace,
    references: dict[int, float],
) -> None:
    dimensions = sorted({int(row["vector_dimension"]) for row in averaged})
    seeds = sorted({seed for row in averaged for seed in row["seeds"]})
    datasets = sorted({int(row["dataset"]) for row in averaged})

    print("Selected GA-only pre/post configurations")
    print(f"  NUM_LEVELS: {args.num_levels}")
    print(f"  split: test")
    print(f"  metric: {args.metric}")
    if args.min_dimension is not None or args.max_dimension is not None:
        print(f"  dimension range: {args.min_dimension}..{args.max_dimension}")
    if args.y_min is not None or args.y_max is not None:
        print(f"  y range: {args.y_min}..{args.y_max}")
    if args.sample_step != 1:
        print(f"  sample step: every {args.sample_step} dimension value")
    print(
        f"  reference: baseline {args.reference_phase}-test at "
        f"VECTOR_DIMENSION={args.reference_dimension}"
    )
    print(f"  seeds: {', '.join(map(str, seeds))}")
    print(f"  datasets: {', '.join(dataset_label(dataset) for dataset in datasets)}")
    print(f"  dimensions ({len(dimensions)}): {', '.join(map(str, dimensions))}")
    print()
    print("dataset phase dimension seed_count seeds mean")
    for row in averaged:
        seed_text = ",".join(map(str, row["seeds"]))
        print(
            f"{dataset_label(int(row['dataset'])):>15s} "
            f"{str(row['phase']):7s} "
            f"{int(row['vector_dimension']):9d} "
            f"{int(row['seed_count']):10d} {seed_text:9s} "
            f"{float(row['mean']):.8f}"
        )
    if references:
        print()
        print("Reference values:")
        for dataset in sorted(references):
            print(
                f"  {dataset_label(dataset)}: "
                f"{references[dataset]:.8f}"
            )


def print_main_results(
    averaged: list[dict[str, object]],
    references: dict[int, float],
) -> None:
    datasets = sorted({int(row["dataset"]) for row in averaged})

    print()
    print("Main results")
    print(
        "dataset endpoint_dim baseline_acc ga_acc ga_minus_baseline "
        "reference_acc ga_minus_reference best_ga_dim best_ga_acc"
    )

    for dataset in datasets:
        dataset_rows = [row for row in averaged if int(row["dataset"]) == dataset]
        dimensions = sorted({int(row["vector_dimension"]) for row in dataset_rows})
        endpoint = None
        for dimension in reversed(dimensions):
            phases = {
                str(row["phase"]): float(row["mean"])
                for row in dataset_rows
                if int(row["vector_dimension"]) == dimension
            }
            if "preopt" in phases and "postopt" in phases:
                endpoint = (dimension, phases)
                break
        if endpoint is None:
            continue

        endpoint_dimension, endpoint_phases = endpoint
        postopt_rows = [row for row in dataset_rows if str(row["phase"]) == "postopt"]
        best_ga = max(postopt_rows, key=lambda row: float(row["mean"]))
        reference = references.get(dataset)

        baseline_acc = endpoint_phases["preopt"]
        ga_acc = endpoint_phases["postopt"]
        reference_text = "" if reference is None else f"{100.0 * reference:.2f}%"
        ga_minus_reference = (
            "" if reference is None else f"{100.0 * (ga_acc - reference):+.2f}pp"
        )

        print(
            f"{dataset_label(dataset):>15s} "
            f"{endpoint_dimension:12d} "
            f"{100.0 * baseline_acc:12.2f}% "
            f"{100.0 * ga_acc:6.2f}% "
            f"{100.0 * (ga_acc - baseline_acc):+17.2f}pp "
            f"{reference_text:>13s} "
            f"{ga_minus_reference:>18s} "
            f"{int(best_ga['vector_dimension']):11d} "
            f"{100.0 * float(best_ga['mean']):10.2f}%"
        )


def write_csv(path: Path, averaged: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "dataset",
                "phase",
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


def write_html(
    path: Path,
    averaged: list[dict[str, object]],
    args: argparse.Namespace,
    references: dict[int, float],
) -> None:
    datasets = sorted({int(row["dataset"]) for row in averaged})
    dimensions = sorted({int(row["vector_dimension"]) for row in averaged})
    means = {
        dataset: {
            phase: {
                int(row["vector_dimension"]): float(row["mean"])
                for row in averaged
                if int(row["dataset"]) == dataset and str(row["phase"]) == phase
            }
            for phase in ["preopt", "postopt"]
        }
        for dataset in datasets
    }
    colors = {
        "preopt": "#0b3d91",
        "postopt": "#d62728",
        "reference": "#000000",
    }
    labels = {
        "preopt": "Baseline model",
        "postopt": "GA-optimized model",
    }
    ref_label = reference_label(args)
    title = f"GA-only pre vs post test {args.metric.replace('_', ' ')} at {args.num_levels} levels"
    y_min = 0.0 if args.y_min is None else args.y_min
    y_max = 1.0 if args.y_max is None else args.y_max

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Segoe UI, Calibri, sans-serif; margin: 28px; color: #1f2933; background: #f8fafc; }}
    p {{ margin: 0 0 18px; color: #52606d; }}
    svg {{ background: white; border: 1px solid #d9e2ec; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.08); margin-bottom: 22px; }}
    .grid {{ stroke: #d9e2ec; stroke-width: 1; }}
    .axis {{ stroke: #334e68; stroke-width: 1.4; }}
    .tick {{ fill: #52606d; font-size: 11px; }}
    .label {{ fill: #334e68; font-size: 13px; font-weight: 600; }}
    .series {{ fill: none; stroke-width: 2.4; }}
  </style>
</head>
<body>
  <p>Each subplot is one dataset; values are averaged over available seeds.</p>
  <div id="charts"></div>
  <script>
    const datasets = {datasets!r};
    const dimensions = {dimensions!r};
    const means = {means!r};
    const colors = {colors!r};
    const labels = {labels!r};
    const references = {references!r};
    const referenceDimension = {args.reference_dimension!r};
    const referencePhase = {args.reference_phase!r};
    const referenceLabel = {ref_label!r};
    const yMin = {y_min!r};
    const yMax = {y_max!r};
        const width = 1050;
    const height = 420;
        const margin = {{ left: 76, right: 145, top: 36, bottom: 62 }};
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const minDim = Math.min(...dimensions);
    const maxDim = Math.max(...dimensions);
    const xScale = x => margin.left + ((x - minDim) / Math.max(1, maxDim - minDim)) * plotWidth;
    const yScale = y => margin.top + ((yMax - y) / (yMax - yMin)) * plotHeight;
    const tickStart = Math.ceil(minDim / 1000) * 1000;
    const xTicks = [];
    for (let tick = tickStart; tick <= maxDim; tick += 1000) xTicks.push(tick);
    if (!xTicks.includes(minDim)) xTicks.unshift(minDim);
    if (!xTicks.includes(maxDim)) xTicks.push(maxDim);
    const makeSvg = (svg, name, attrs, text) => {{
      const element = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key, value] of Object.entries(attrs || {{}})) element.setAttribute(key, value);
      if (text !== undefined) element.textContent = text;
      svg.appendChild(element);
      return element;
    }};
    for (const dataset of datasets) {{
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("width", width);
      svg.setAttribute("height", height);
      document.getElementById("charts").appendChild(svg);
      makeSvg(svg, "text", {{ x: margin.left, y: 24, class: "label" }}, dataset < 0 ? "Dataset average" : `Dataset ${{dataset}}`);
      for (let step = 0; step <= 5; step += 1) {{
        const y = yMin + ((yMax - yMin) * step / 5);
        const py = yScale(y);
        makeSvg(svg, "line", {{ x1: margin.left, y1: py, x2: width - margin.right, y2: py, class: "grid" }});
        makeSvg(svg, "text", {{ x: margin.left - 10, y: py + 4, "text-anchor": "end", class: "tick" }}, `${{Math.round(y * 100)}}%`);
      }}
      for (const x of xTicks) {{
        const px = xScale(x);
        makeSvg(svg, "line", {{ x1: px, y1: margin.top, x2: px, y2: height - margin.bottom, class: "grid" }});
        makeSvg(svg, "text", {{ x: px, y: height - margin.bottom + 22, "text-anchor": "middle", class: "tick" }}, x);
      }}
      makeSvg(svg, "line", {{ x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: "axis" }});
      makeSvg(svg, "line", {{ x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: "axis" }});
      if (references[dataset] !== undefined) {{
        const refY = yScale(references[dataset]);
        makeSvg(svg, "line", {{
          x1: margin.left,
          y1: refY,
          x2: width - margin.right,
          y2: refY,
          stroke: colors.reference,
          "stroke-width": 2,
          "stroke-dasharray": "3 5"
        }});
        makeSvg(svg, "text", {{
          x: width - margin.right - 8,
          y: refY - 6,
          "text-anchor": "end",
          class: "tick",
          fill: colors.reference
        }}, referenceLabel);
      }}
      const legendBaseY = dataset === 2 ? margin.top + 20 : height - margin.bottom - 42;
      ["preopt", "postopt"].forEach((phase, index) => {{
        const phasePoints = dimensions
          .filter(dimension => means[dataset][phase][dimension] !== undefined)
          .map(dimension => [xScale(dimension), yScale(means[dataset][phase][dimension]), dimension, means[dataset][phase][dimension]]);
        const pathData = phasePoints.map((point, pointIndex) => `${{pointIndex === 0 ? "M" : "L"}} ${{point[0].toFixed(2)}} ${{point[1].toFixed(2)}}`).join(" ");
        makeSvg(svg, "path", {{ d: pathData, stroke: colors[phase], class: "series", "stroke-dasharray": phase === "preopt" ? "7 4" : "" }});
        const legendY = legendBaseY + index * 24;
        makeSvg(svg, "line", {{ x1: width - margin.right + 24, y1: legendY, x2: width - margin.right + 54, y2: legendY, stroke: colors[phase], "stroke-width": 3, "stroke-dasharray": phase === "preopt" ? "7 4" : "" }});
        makeSvg(svg, "text", {{ x: width - margin.right + 64, y: legendY + 4, class: "tick" }}, labels[phase]);
      }});
    }}
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def plot_results(
    path: Path,
    averaged: list[dict[str, object]],
    args: argparse.Namespace,
    references: dict[int, float],
) -> None:
    datasets = sorted({int(row["dataset"]) for row in averaged})
    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(11, max(3.2, 2.7 * len(datasets))),
        sharex=True,
        squeeze=False,
    )

    colors = {"preopt": "#0b3d91", "postopt": "#d62728"}
    styles = {"preopt": "--", "postopt": "-"}
    for ax, dataset in zip(axes[:, 0], datasets):
        for phase in ["preopt", "postopt"]:
            values = [
                row
                for row in averaged
                if int(row["dataset"]) == dataset and str(row["phase"]) == phase
            ]
            ax.plot(
                [int(row["vector_dimension"]) for row in values],
                [float(row["mean"]) for row in values],
                linewidth=1.7,
                linestyle=styles[phase],
                color=colors[phase],
                label=phase_label(phase),
            )
        if dataset in references:
            ax.axhline(
                references[dataset],
                color="#000000",
                linestyle=":",
                linewidth=1.8,
                label=reference_label(args),
            )
        ax.set_title("Dataset average" if dataset < 0 else f"Dataset {dataset}")
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right" if dataset == 2 else "lower right")

    axes[-1, 0].set_xlabel("Vector dimension")
    for ax in axes[:, 0]:
        ax.set_xlim(
            args.min_dimension if args.min_dimension is not None else None,
            args.max_dimension if args.max_dimension is not None else None,
        )
        ax.set_ylim(
            args.y_min if args.y_min is not None else 0.0,
            args.y_max if args.y_max is not None else 1.0,
        )
        ax.xaxis.set_major_locator(MultipleLocator(1000))
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
    if args.reference_dimension != 10000 or args.reference_phase != "preopt":
        parts.append(f"ref_{args.reference_phase}_{args.reference_dimension}")
    return "" if not parts else "_" + "_".join(parts).replace(".", "p")


def main() -> None:
    args = parse_args()
    full_averaged = add_dataset_average(aggregate(load_selected_rows(args)))
    baseline_reference = add_dataset_average(aggregate(load_reference_rows(args)))
    references = reference_values(baseline_reference, args)
    averaged = sample_dimensions(full_averaged, args.sample_step)
    print_configurations(averaged, args, references)
    print_main_results(averaged, references)

    output = args.output or (
        DEFAULT_PLOTS_DIR
        / (
            f"ccim_optimization_pre_vs_post_levels_{args.num_levels:03d}_test"
            f"{output_range_suffix(args)}.png"
        )
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = (
        DEFAULT_RESULTS_DIR
        / (
            f"ccim_optimization_pre_vs_post_levels_{args.num_levels:03d}_test"
            f"{output_range_suffix(args)}.csv"
        )
    ).resolve()
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    html_output = output.with_suffix(".html")

    plot_results(output, averaged, args, references)
    write_csv(csv_output, averaged)
    write_html(html_output, averaged, args, references)
    print()
    print(f"Saved PNG plot:      {output}")
    print(f"Saved HTML plot:     {html_output}")
    print(f"Saved averaged data: {csv_output}")


if __name__ == "__main__":
    main()
