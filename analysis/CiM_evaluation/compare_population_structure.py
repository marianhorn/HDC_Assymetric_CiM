import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from analyze_ga_cim import OUTPUT_DIR, is_binary_vectors, load_cim, resolve_run_dir


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


def adjacent_distance_map(V: np.ndarray) -> np.ndarray:
    # returns [feature, level_transition], normalized Hamming distance
    return (V[:-1] != V[1:]).mean(axis=2).T


def load_population(generation_dir: Path):
    population = []
    for cim_path in sorted(generation_dir.glob("cim_*.csv")):
        header, mode, V = load_cim(cim_path)
        if mode != "precomputed":
            continue
        if not is_binary_vectors(V):
            raise ValueError(f"Expected binary CCIM vectors in {cim_path}")
        flip_map = adjacent_distance_map(V)
        population.append(
            {
                "path": cim_path,
                "candidate": int(header.get("candidate", cim_path.stem.split("_")[-1])),
                "accuracy": float(header.get("accuracy", "nan")),
                "similarity": float(header.get("similarity", "nan")),
                "flip_map": flip_map,
                "mean_adjacent_distance": float(flip_map.mean()),
                "std_adjacent_distance": float(flip_map.std()),
                "max_adjacent_distance": float(flip_map.max()),
            }
        )
    if not population:
        raise RuntimeError(f"No precomputed CCIM files found in {generation_dir}")
    return population


def write_summary(path: Path, population) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "candidate",
                "accuracy",
                "similarity",
                "mean_adjacent_distance",
                "std_adjacent_distance",
                "max_adjacent_distance",
                "file",
            ],
        )
        writer.writeheader()
        for item in sorted(population, key=lambda row: row["candidate"]):
            writer.writerow(
                {
                    "candidate": item["candidate"],
                    "accuracy": f"{item['accuracy']:.10f}",
                    "similarity": f"{item['similarity']:.10f}",
                    "mean_adjacent_distance": f"{item['mean_adjacent_distance']:.10f}",
                    "std_adjacent_distance": f"{item['std_adjacent_distance']:.10f}",
                    "max_adjacent_distance": f"{item['max_adjacent_distance']:.10f}",
                    "file": item["path"].name,
                }
            )


def group_mean_map(population) -> np.ndarray:
    return np.stack([item["flip_map"] for item in population], axis=0).mean(axis=0)


def plot_heatmap(matrix, title: str, out_path: Path, label: str, show: bool, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Level transition l -> l+1")
    ax.set_ylabel("Feature")
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CCIM population structure: accuracy/similarity tradeoff and "
            "top-accuracy vs top-similarity adjacent-distance maps."
        )
    )
    parser.add_argument("--run", default="latest")
    parser.add_argument("--generation", default="latest")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install with: pip install matplotlib")
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")

    run_dir = resolve_run_dir(args.run)
    generation_dir = resolve_generation_dir(run_dir, args.generation)
    population = load_population(generation_dir)
    top_n = min(args.top_n, len(population))

    by_accuracy = sorted(population, key=lambda item: (-item["accuracy"], item["candidate"]))
    by_similarity = sorted(population, key=lambda item: (-item["similarity"], item["candidate"]))
    top_accuracy = by_accuracy[:top_n]
    top_similarity = by_similarity[:top_n]

    out_dir = OUTPUT_DIR / "population_structure" / f"{run_dir.name}_{generation_dir.name}_top{top_n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(out_dir / "population_summary.csv", population)

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    scatter = ax.scatter(
        [item["similarity"] for item in population],
        [item["accuracy"] for item in population],
        c=[item["mean_adjacent_distance"] for item in population],
        cmap="viridis",
        s=38,
        alpha=0.9,
    )
    for item in top_accuracy[:5]:
        ax.text(item["similarity"], item["accuracy"], f"a{item['candidate']}", fontsize=8)
    for item in top_similarity[:5]:
        ax.text(item["similarity"], item["accuracy"], f"s{item['candidate']}", fontsize=8)
    ax.set_title(f"{run_dir.name} {generation_dir.name}: accuracy vs robustness proxy")
    ax.set_xlabel("Exported similarity field")
    ax.set_ylabel("Exported accuracy")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="Mean adjacent Hamming distance")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_similarity_scatter.png", dpi=180)
    if args.show:
        plt.show()
    plt.close(fig)

    accuracy_map = group_mean_map(top_accuracy)
    similarity_map = group_mean_map(top_similarity)
    difference_map = accuracy_map - similarity_map
    population_std_map = np.stack([item["flip_map"] for item in population], axis=0).std(axis=0)

    plot_heatmap(
        accuracy_map,
        f"Top {top_n} by accuracy: mean adjacent Hamming distance",
        out_dir / "top_accuracy_mean_adjacent_distance.png",
        "Hamming distance",
        args.show,
    )
    plot_heatmap(
        similarity_map,
        f"Top {top_n} by similarity: mean adjacent Hamming distance",
        out_dir / "top_similarity_mean_adjacent_distance.png",
        "Hamming distance",
        args.show,
    )
    plot_heatmap(
        difference_map,
        f"Top accuracy minus top similarity adjacent-distance map",
        out_dir / "top_accuracy_minus_top_similarity.png",
        "Distance difference",
        args.show,
        cmap="coolwarm",
    )
    plot_heatmap(
        population_std_map,
        "Population std of adjacent Hamming distance",
        out_dir / "population_adjacent_distance_std.png",
        "Std of Hamming distance",
        args.show,
    )

    print(f"Run: {run_dir.name}")
    print(f"Generation: {generation_dir.name}")
    print(f"Individuals: {len(population)}")
    print(f"Top-N: {top_n}")
    print("Top accuracy candidates:", ", ".join(str(item["candidate"]) for item in top_accuracy))
    print("Top similarity candidates:", ", ".join(str(item["candidate"]) for item in top_similarity))
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
