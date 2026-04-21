import argparse
import re
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ModuleNotFoundError:
    plt = None

from analyze_ga_cim import OUTPUT_DIR, load_cim


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CIMS_ROOT = REPO_ROOT / "CiMs"
GEN_RE = re.compile(r"GA generation (\d+)/(\d+)")
NEW_SELECTED_RE = re.compile(r"new selected individuals:\s+(\d+)/(\d+)")


def resolve_run_dir(run_id: str) -> Path:
    run_dir = CIMS_ROOT / f"ga_{run_id}"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    return run_dir


def parse_new_selected_series(output_path: Path):
    generations = []
    selected_counts = []
    population_size = None
    current_generation = None

    with output_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            gen_match = GEN_RE.search(line)
            if gen_match:
                current_generation = int(gen_match.group(1))
                continue

            selected_match = NEW_SELECTED_RE.search(line)
            if selected_match and current_generation is not None:
                generations.append(current_generation)
                selected_counts.append(int(selected_match.group(1)))
                population_size = int(selected_match.group(2))

    if not generations:
        raise RuntimeError(f"No 'new selected individuals' data found in {output_path}")

    return np.array(generations), np.array(selected_counts), population_size


def get_final_generation_dir(run_dir: Path) -> Path:
    generation_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("generation_")])
    if not generation_dirs:
        raise FileNotFoundError(f"No generation folders found in {run_dir}")
    return generation_dirs[-1]


def compute_transition_flip_totals(V: np.ndarray, feature: int | None = None) -> np.ndarray:
    # V shape: [num_levels, num_features, dimension]
    adjacent_diff = V[:-1] != V[1:]
    if feature is None:
        return adjacent_diff.sum(axis=(1, 2)).astype(float)
    if feature < 0 or feature >= V.shape[1]:
        raise ValueError(f"Feature index {feature} out of range [0, {V.shape[1] - 1}]")
    return adjacent_diff[:, feature, :].sum(axis=1).astype(float)


def load_final_generation_individuals(run_dir: Path, feature: int | None = None):
    generation_dir = get_final_generation_dir(run_dir)
    cim_paths = sorted(generation_dir.glob("cim_*.csv"))
    if not cim_paths:
        raise FileNotFoundError(f"No CiM files found in {generation_dir}")

    individuals = []
    for cim_path in cim_paths:
        header, mode, V = load_cim(cim_path)
        if mode != "precomputed":
            raise ValueError(f"Only precomputed CiMs are supported, got mode={mode} in {cim_path}")

        similarity = float(header.get("similarity", 0.0))
        accuracy = float(header.get("accuracy", 0.0))
        robustness = 1.0 - similarity

        individuals.append(
            {
                "path": cim_path,
                "candidate": int(header.get("candidate", -1)),
                "generation": int(header.get("generation", -1)),
                "num_levels": int(header.get("num_levels", V.shape[0])),
                "accuracy": accuracy,
                "similarity": similarity,
                "robustness": robustness,
                "transition_flips": compute_transition_flip_totals(V, feature),
            }
        )

    return individuals


def plot_new_selected(generations, selected_counts, population_size, out_path: Path, run_id: str, show: bool):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(generations, selected_counts, marker="o", linewidth=2.0, color="#1f77b4")
    ax.set_title(f"GA run {run_id}: newly selected individuals per generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("New selected individuals")
    ax.grid(True, alpha=0.3)
    if population_size is not None:
        ax.set_ylim(0, max(population_size, int(np.max(selected_counts))) + 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_flip_surface(individuals, sort_key: str, out_path: Path, run_id: str, feature: int | None, show: bool):
    sorted_individuals = sorted(individuals, key=lambda item: item[sort_key], reverse=True)
    flip_matrix = np.stack([item["transition_flips"] for item in sorted_individuals], axis=0)

    num_individuals, num_transitions = flip_matrix.shape
    x = np.arange(1, num_transitions + 1)
    y = np.arange(1, num_individuals + 1)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(11.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, flip_matrix, cmap="viridis", edgecolor="none", antialiased=True)

    if sort_key == "accuracy":
        sort_label = "accuracy"
        y_tick_labels = [f"c{item['candidate']:02d} ({item['accuracy'] * 100.0:.1f}%)" for item in sorted_individuals]
    else:
        sort_label = "robustness"
        y_tick_labels = [f"c{item['candidate']:02d} ({item['robustness']:.3f})" for item in sorted_individuals]

    title_suffix = "all features" if feature is None else f"feature {feature}"
    ax.set_title(f"GA run {run_id}: adjacent bit-flips sorted by {sort_label} ({title_suffix})")
    ax.set_xlabel("Level transition")
    ax.set_ylabel("Individual rank")
    ax.set_zlabel("Bit-flips")
    ax.set_yticks(y)
    ax.set_yticklabels(y_tick_labels, fontsize=8)
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.08, label="Bit-flips")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def save_individual_summary(individuals, out_path: Path):
    sorted_by_candidate = sorted(individuals, key=lambda item: item["candidate"])
    with out_path.open("w", encoding="utf-8") as f:
        f.write("candidate,generation,accuracy,similarity,robustness,total_flips\n")
        for item in sorted_by_candidate:
            f.write(
                f"{item['candidate']},{item['generation']},{item['accuracy']:.10f},"
                f"{item['similarity']:.10f},{item['robustness']:.10f},"
                f"{int(item['transition_flips'].sum())}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze one final GA CiM run and create selection and 3D bit-flip plots."
    )
    parser.add_argument("--run", required=True, choices=["16", "32", "64", "128"], help="Run folder suffix under CiMs, e.g. 16, 32, or 64.")
    parser.add_argument("--feature", type=int, default=None, help="Optional feature index to analyze only that feature in the 3D plots.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively in addition to saving them.")
    args = parser.parse_args()

    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install with: pip install matplotlib")

    run_dir = resolve_run_dir(args.run)
    output_txt = run_dir / "output.txt"
    if not output_txt.is_file():
        raise FileNotFoundError(f"Missing output.txt in {run_dir}")

    individuals = load_final_generation_individuals(run_dir, args.feature)
    generations, selected_counts, population_size = parse_new_selected_series(output_txt)

    feature_tag = "all_features" if args.feature is None else f"feature_{args.feature:02d}"
    out_dir = OUTPUT_DIR / f"final_run_analysis_{run_dir.name}_{feature_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_new_selected(
        generations,
        selected_counts,
        population_size,
        out_dir / "new_selected_per_generation.png",
        args.run,
        args.show,
    )
    plot_flip_surface(
        individuals,
        "accuracy",
        out_dir / "flip_surface_sorted_by_accuracy.png",
        args.run,
        args.feature,
        args.show,
    )
    plot_flip_surface(
        individuals,
        "robustness",
        out_dir / "flip_surface_sorted_by_robustness.png",
        args.run,
        args.feature,
        args.show,
    )
    save_individual_summary(individuals, out_dir / "individual_summary.csv")

    print(f"Run: {run_dir.name}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
