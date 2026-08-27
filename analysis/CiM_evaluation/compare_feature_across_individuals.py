import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from analyze_ga_cim import OUTPUT_DIR, is_binary_vectors, load_cim, resolve_run_dir


def generation_index(path: Path) -> int:
    return int(path.name.split("_")[-1])


def resolve_generation_dir(run_dir: Path, generation: str) -> Path:
    generation_dirs = sorted(
        p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("generation_")
    )
    if not generation_dirs:
        raise FileNotFoundError(f"No generation folders found in {run_dir}")
    if generation == "latest":
        return generation_dirs[-1]
    candidate = run_dir / f"generation_{int(generation):04d}"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Generation folder not found: {candidate}")
    return candidate


def load_individuals(generation_dir: Path):
    individuals = []
    for cim_path in sorted(generation_dir.glob("cim_*.csv")):
        header, mode, V = load_cim(cim_path)
        if mode != "precomputed":
            continue
        if not is_binary_vectors(V):
            raise ValueError(f"Expected binary CCIM vectors in {cim_path}")
        individuals.append(
            {
                "path": cim_path,
                "candidate": int(header.get("candidate", cim_path.stem.split("_")[-1])),
                "accuracy": float(header.get("accuracy", "nan")),
                "similarity": float(header.get("similarity", "nan")),
                "V": V,
            }
        )
    if not individuals:
        raise RuntimeError(f"No precomputed CCIM files found in {generation_dir}")
    return individuals


def feature_adjacent_distances(V: np.ndarray, feature: int) -> np.ndarray:
    return (V[:-1, feature, :] != V[1:, feature, :]).mean(axis=1)


def sorted_individuals(individuals, sort_by: str):
    if sort_by == "candidate":
        return sorted(individuals, key=lambda item: item["candidate"])
    if sort_by == "accuracy":
        return sorted(individuals, key=lambda item: (-item["accuracy"], item["candidate"]))
    if sort_by == "similarity":
        return sorted(individuals, key=lambda item: (-item["similarity"], item["candidate"]))
    raise ValueError(f"Unsupported sort key: {sort_by}")


def write_summary(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "candidate",
                "accuracy",
                "similarity",
                "mean_adjacent_distance",
                "std_adjacent_distance",
                "min_adjacent_distance",
                "max_adjacent_distance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare one CCIM feature across all individuals in one GA generation."
    )
    parser.add_argument("--run", default="latest")
    parser.add_argument("--generation", default="latest")
    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument(
        "--sort-by",
        choices=["candidate", "accuracy", "similarity"],
        default="candidate",
        help="Order individuals in plots. Default: candidate.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install with: pip install matplotlib")

    run_dir = resolve_run_dir(args.run)
    generation_dir = resolve_generation_dir(run_dir, args.generation)
    individuals = sorted_individuals(load_individuals(generation_dir), args.sort_by)

    V0 = individuals[0]["V"]
    num_levels, num_features, _ = V0.shape
    if args.feature < 0 or args.feature >= num_features:
        raise ValueError(f"--feature out of range: {args.feature}, valid 0..{num_features - 1}")

    matrix = np.stack(
        [feature_adjacent_distances(item["V"], args.feature) for item in individuals],
        axis=0,
    )
    labels = [str(item["candidate"]) for item in individuals]

    out_dir = (
        OUTPUT_DIR
        / "feature_across_individuals"
        / f"{run_dir.name}_{generation_dir.name}_feature_{args.feature:02d}_{args.sort_by}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for item, distances in zip(individuals, matrix):
        summary_rows.append(
            {
                "candidate": item["candidate"],
                "accuracy": f"{item['accuracy']:.10f}",
                "similarity": f"{item['similarity']:.10f}",
                "mean_adjacent_distance": f"{float(distances.mean()):.10f}",
                "std_adjacent_distance": f"{float(distances.std()):.10f}",
                "min_adjacent_distance": f"{float(distances.min()):.10f}",
                "max_adjacent_distance": f"{float(distances.max()):.10f}",
            }
        )
    write_summary(out_dir / "feature_summary.csv", summary_rows)

    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(
        f"{run_dir.name} {generation_dir.name}: feature {args.feature}, "
        f"one row per individual, sorted by {args.sort_by}"
    )
    ax.set_xlabel("Level transition l -> l+1")
    ax.set_ylabel("Individual candidate")
    tick_step = max(1, len(labels) // 20)
    ticks = list(range(0, len(labels), tick_step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=8)
    fig.colorbar(im, ax=ax, label="Hamming distance")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_adjacent_distance_heatmap.png", dpi=180)
    if args.show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    x = np.arange(num_levels - 1)
    for item, distances in zip(individuals, matrix):
        ax.plot(x, distances, alpha=0.18, linewidth=0.9, color="#1f77b4")
    ax.plot(x, matrix.mean(axis=0), linewidth=2.4, color="#d62728", label="population mean")
    ax.fill_between(
        x,
        matrix.mean(axis=0) - matrix.std(axis=0),
        matrix.mean(axis=0) + matrix.std(axis=0),
        color="#d62728",
        alpha=0.18,
        label="population +/- std",
    )
    ax.set_title(f"{run_dir.name} {generation_dir.name}: feature {args.feature} adjacent distances")
    ax.set_xlabel("Level transition l -> l+1")
    ax.set_ylabel("Hamming distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "feature_adjacent_distance_lines.png", dpi=180)
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"Run: {run_dir.name}")
    print(f"Generation: {generation_dir.name}")
    print(f"Feature: {args.feature}")
    print(f"Individuals: {len(individuals)}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
