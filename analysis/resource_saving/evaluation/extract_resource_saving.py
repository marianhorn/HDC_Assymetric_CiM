#!/usr/bin/env python3
"""Extract resource-saving sweep CSVs into Parquet and an xarray tensor."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pandas. Install evaluation dependencies with:\n"
        "  python -m pip install -r analysis/resource_saving/evaluation/requirements.txt"
    ) from exc


METRIC_COLUMNS = [
    "overall_accuracy",
    "class_average_accuracy",
    "class_vector_similarity",
    "correct",
    "not_correct",
    "transition_error",
    "total",
    "duration_sec",
]

CONFIG_COLUMNS = [
    "num_features",
    "binning_mode",
    "bipolar_mode",
    "precomputed_item_memory",
    "use_genetic_item_memory",
    "ga_selection_mode",
    "ga_mutation_rate",
    "n_gram_size",
    "window",
    "downsample",
    "validation_ratio",
]

BINNING_MODE_NAMES = {
    0: "uniform",
    1: "quantile",
    2: "kmeans_1d",
    3: "decision_tree_1d",
    4: "chimerge",
}

INFO_RE = re.compile(
    r"scope=(?P<scope>dataset|overall)"
    r"(?:,dataset=(?P<dataset>\d+))?"
    r",phase=(?P<phase>preopt|postopt)-(?P<split>validation|test)"
)


def parse_info(info: str) -> tuple[str, str, str, str]:
    match = INFO_RE.search(str(info))
    if not match:
        raise ValueError(f"Could not parse result info field: {info!r}")

    scope = match.group("scope")
    dataset = "overall"
    if scope == "dataset":
        dataset = f"dataset_{match.group('dataset')}"

    return scope, dataset, match.group("phase"), match.group("split")


def read_manifest(path: Path) -> pd.DataFrame:
    manifest_path = path.parent / "run_manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame(columns=["num_levels", "vector_dimension", "run_index", "duration_sec"])

    manifest = pd.read_csv(manifest_path)
    return manifest[["num_levels", "vector_dimension", "run_index", "duration_sec"]]


def read_result_file(path: Path, variant: str, method: str, seed: int, cim_enabled: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    manifest = read_manifest(path)
    if not manifest.empty:
        frame = frame.merge(manifest, on=["num_levels", "vector_dimension"], how="left")
    else:
        frame["run_index"] = pd.NA
        frame["duration_sec"] = pd.NA

    parsed = frame["info"].map(parse_info)

    frame.insert(0, "method", method)
    frame.insert(1, "variant", variant)
    frame.insert(2, "seed", seed)
    frame.insert(3, "cim_enabled", cim_enabled)
    frame.insert(4, "scope", parsed.map(lambda item: item[0]))
    frame.insert(5, "dataset", parsed.map(lambda item: item[1]))
    frame.insert(6, "phase", parsed.map(lambda item: item[2]))
    frame.insert(7, "split", parsed.map(lambda item: item[3]))
    frame.insert(8, "binning_mode_name", frame["binning_mode"].map(BINNING_MODE_NAMES))
    frame["synthetic_phase_copy"] = False

    return frame


def add_non_cim_postopt_copy(frame: pd.DataFrame) -> pd.DataFrame:
    """For non-CiM variants, copy preopt rows to postopt instead of leaving NaNs."""
    preopt = frame[(~frame["cim_enabled"]) & (frame["phase"] == "preopt")].copy()
    preopt["phase"] = "postopt"
    preopt["synthetic_phase_copy"] = True
    return pd.concat([frame, preopt], ignore_index=True)


def existing_seed_result(path: Path, seed: int) -> Path | None:
    result_path = path / f"seed_{seed:02d}" / "results.csv"
    return result_path if result_path.exists() else None


def collect_sources(resource_root: Path) -> list[tuple[Path, str, str, int, bool]]:
    sources: list[tuple[Path, str, str, int, bool]] = []

    for seed in range(1, 6):
        path = existing_seed_result(resource_root / "baseline" / "runs", seed)
        if path:
            sources.append((path, "baseline_uniform", "baseline", seed, False))

    quantizer_root = resource_root / "quantizer_only" / "runs"
    for mode_dir in sorted(quantizer_root.glob("binning_mode_*")):
        mode_name = mode_dir.name.removeprefix("binning_mode_")
        for seed in range(1, 6):
            path = existing_seed_result(mode_dir, seed)
            if path:
                sources.append((path, f"quantizer_{mode_name}", "quantizer_only", seed, False))

    for seed in range(1, 6):
        path = existing_seed_result(resource_root / "cim_only" / "runs", seed)
        if path:
            sources.append((path, "cim_uniform", "cim_only", seed, True))

    quantizer_cim_root = resource_root / "quantizer_and_cim" / "runs"
    for mode_dir in sorted(quantizer_cim_root.glob("binning_mode_*")):
        mode_name = mode_dir.name.removeprefix("binning_mode_")
        for seed in range(1, 6):
            path = existing_seed_result(mode_dir, seed)
            if path:
                sources.append((path, f"quantizer_cim_{mode_name}", "quantizer_and_cim", seed, True))

    return sources


def build_tensor(frame: pd.DataFrame):
    try:
        import xarray as xr  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: xarray. Install evaluation dependencies with:\n"
            "  python -m pip install -r analysis/resource_saving/evaluation/requirements.txt"
        ) from exc

    index_columns = [
        "num_levels",
        "vector_dimension",
        "variant",
        "dataset",
        "seed",
        "phase",
        "split",
    ]

    tensor_frame = frame[index_columns + METRIC_COLUMNS].copy()
    tensor_frame = tensor_frame.groupby(index_columns, as_index=False)[METRIC_COLUMNS].first()
    long_frame = tensor_frame.melt(
        id_vars=index_columns,
        value_vars=METRIC_COLUMNS,
        var_name="metric",
        value_name="value",
    )

    data = long_frame.set_index(index_columns + ["metric"])["value"].to_xarray()
    return data.transpose(
        "num_levels",
        "vector_dimension",
        "variant",
        "dataset",
        "seed",
        "phase",
        "split",
        "metric",
    )


def write_report(output_path: Path, frame: pd.DataFrame, sources: list[tuple[Path, str, str, int, bool]]) -> None:
    lines = []
    lines.append("Resource-saving extraction report")
    lines.append("")
    lines.append(f"source files: {len(sources)}")
    lines.append(f"rows: {len(frame)}")
    lines.append(f"variants: {', '.join(sorted(frame['variant'].unique()))}")
    lines.append(f"seeds: {', '.join(str(seed) for seed in sorted(frame['seed'].unique()))}")
    lines.append(f"num_levels: {len(frame['num_levels'].unique())} unique")
    lines.append(f"vector_dimension: {len(frame['vector_dimension'].unique())} unique")
    lines.append("")
    lines.append("Rows by variant:")
    for variant, count in frame.groupby("variant").size().sort_index().items():
        lines.append(f"  {variant}: {count}")
    lines.append("")
    lines.append("Note: non-CiM variants copy preopt rows into postopt rows by request.")
    lines.append("Note: current CSV files contain validation/test metrics, not train metrics.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract resource-saving CSV results into Parquet and an xarray tensor."
    )
    parser.add_argument(
        "--resource-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to analysis/resource_saving.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "extracted"),
        help="Output directory for extracted files.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write combined.csv for debugging. Parquet remains the primary table format.",
    )
    args = parser.parse_args()

    resource_root = Path(args.resource_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = collect_sources(resource_root)
    if not sources:
        raise SystemExit(f"No results.csv files found under {resource_root}")

    frames = []
    for path, variant, method, seed, cim_enabled in sources:
        print(f"Reading {variant} seed={seed}: {path}")
        frames.append(read_result_file(path, variant, method, seed, cim_enabled))

    combined = pd.concat(frames, ignore_index=True)
    combined = add_non_cim_postopt_copy(combined)

    ordered_columns = [
        "method",
        "variant",
        "seed",
        "cim_enabled",
        "binning_mode_name",
        "num_levels",
        "vector_dimension",
        "scope",
        "dataset",
        "phase",
        "split",
        "synthetic_phase_copy",
        "run_index",
    ] + CONFIG_COLUMNS + METRIC_COLUMNS + ["info"]
    combined = combined[ordered_columns]

    parquet_path = output_dir / "combined.parquet"
    combined.to_parquet(parquet_path, index=False)

    if args.write_csv:
        combined.to_csv(output_dir / "combined.csv", index=False)

    tensor = build_tensor(combined)
    tensor.name = "resource_saving"
    tensor.attrs["description"] = (
        "Resource-saving sweep metrics. Non-CiM postopt values are copied from preopt."
    )
    tensor.to_netcdf(output_dir / "resource_saving.nc")

    write_report(output_dir / "extraction_report.txt", combined, sources)

    print(f"Wrote {parquet_path}")
    print(f"Wrote {output_dir / 'resource_saving.nc'}")
    print(f"Wrote {output_dir / 'extraction_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
