import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from analyze_ga_cim import OUTPUT_DIR, is_binary_vectors, load_cim, resolve_run_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
FOOT_DATA_ROOT = REPO_ROOT / "foot" / "data"
NUM_FEATURES = 32


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
    return (V[:-1] != V[1:]).mean(axis=2).T


def load_population(generation_dir: Path):
    population = []
    for cim_path in sorted(generation_dir.glob("cim_*.csv")):
        header, mode, V = load_cim(cim_path)
        if mode != "precomputed":
            continue
        if not is_binary_vectors(V):
            raise ValueError(f"Expected binary CCIM vectors in {cim_path}")
        population.append(
            {
                "candidate": int(header.get("candidate", cim_path.stem.split("_")[-1])),
                "accuracy": float(header.get("accuracy", "nan")),
                "similarity": float(header.get("similarity", "nan")),
                "flip_map": adjacent_distance_map(V),
            }
        )
    if not population:
        raise RuntimeError(f"No precomputed CCIM files found in {generation_dir}")
    return population


def pearson_corr_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    denom = np.sqrt((Xc * Xc).sum(axis=0) * float((yc * yc).sum()))
    out = np.zeros(X.shape[1], dtype=float)
    valid = denom > 0.0
    out[valid] = (Xc[:, valid] * yc[:, None]).sum(axis=0) / denom[valid]
    return out


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        rank = 0.5 * (i + j - 1)
        ranked[order[i:j]] = rank
        i = j
    return ranked


def spearman_corr_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xr = np.apply_along_axis(ranks, 0, X)
    yr = ranks(y)
    return pearson_corr_matrix(Xr, yr)


def load_csv_matrix(path: Path, expected_cols: int) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            values = [float(x) for x in line.strip().split(",")[:expected_cols]]
            if len(values) != expected_cols:
                raise ValueError(f"Unexpected column count in {path}: {len(values)}")
            rows.append(values)
    return np.asarray(rows, dtype=float)


def load_csv_labels(path: Path) -> np.ndarray:
    labels = []
    with path.open("r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            if line.strip():
                labels.append(int(float(line.strip().split(",")[0])))
    return np.asarray(labels, dtype=int)


def split_training_validation(data: np.ndarray, labels: np.ndarray, validation_ratio: float):
    classes = sorted(int(x) for x in set(labels.tolist()) if x >= 0)
    class_counts = {cls: int((labels == cls).sum()) for cls in classes}
    class_targets = {
        cls: min(class_counts[cls], int(class_counts[cls] * validation_ratio + 0.5))
        for cls in classes
    }
    assigned = {cls: 0 for cls in classes}
    train_indices = []
    val_indices = []
    for idx, label in enumerate(labels):
        label = int(label)
        if label in assigned and assigned[label] < class_targets[label]:
            val_indices.append(idx)
            assigned[label] += 1
        else:
            train_indices.append(idx)
    return data[train_indices], labels[train_indices], data[val_indices], labels[val_indices]


def uniform_level(value: float, num_levels: int) -> int:
    cut_count = num_levels - 1
    for cut in range(cut_count):
        numerator = 20000 * (cut + 1) - 10000
        threshold_scaled = (numerator + cut_count - 1) // cut_count
        boundary = ((threshold_scaled - 1) - 10000.0) / 10000.0
        if value <= boundary:
            return cut
    return num_levels - 1


def occupancy_by_feature_level(data: np.ndarray, num_levels: int) -> np.ndarray:
    occ = np.zeros((data.shape[1], num_levels), dtype=int)
    for row in data:
        for feature, value in enumerate(row):
            level = uniform_level(float(value), num_levels)
            occ[feature, level] += 1
    return occ


def transition_occupancy(level_occupancy: np.ndarray) -> np.ndarray:
    return level_occupancy[:, :-1] + level_occupancy[:, 1:]


def write_ranked_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature",
                "transition",
                "pearson_accuracy",
                "spearman_accuracy",
                "pearson_similarity",
                "spearman_similarity",
                "mean_flip_distance",
                "std_flip_distance",
                "train_transition_occupancy",
                "validation_transition_occupancy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(matrix: np.ndarray, title: str, output: Path, label: str, show: bool):
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Level transition l -> l+1")
    ax.set_ylabel("Feature")
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate per-feature/per-transition CCIM flip distances with GA accuracy "
            "and similarity, then compare those positions with dataset occupancy."
        )
    )
    parser.add_argument("--run", default="ga_128")
    parser.add_argument("--generation", default="latest")
    parser.add_argument("--dataset", type=int, default=1)
    parser.add_argument("--validation-ratio", type=float, default=0.3)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install with: pip install matplotlib")

    run_dir = resolve_run_dir(args.run)
    generation_dir = resolve_generation_dir(run_dir, args.generation)
    population = load_population(generation_dir)
    num_features, transitions = population[0]["flip_map"].shape
    num_levels = transitions + 1

    X = np.stack([item["flip_map"].reshape(-1) for item in population], axis=0)
    accuracy = np.asarray([item["accuracy"] for item in population], dtype=float)
    similarity = np.asarray([item["similarity"] for item in population], dtype=float)

    pearson_accuracy = pearson_corr_matrix(X, accuracy).reshape(num_features, transitions)
    spearman_accuracy = spearman_corr_matrix(X, accuracy).reshape(num_features, transitions)
    pearson_similarity = pearson_corr_matrix(X, similarity).reshape(num_features, transitions)
    spearman_similarity = spearman_corr_matrix(X, similarity).reshape(num_features, transitions)
    mean_flip = X.mean(axis=0).reshape(num_features, transitions)
    std_flip = X.std(axis=0).reshape(num_features, transitions)

    data_dir = FOOT_DATA_ROOT / f"dataset{args.dataset:02d}"
    training_data = load_csv_matrix(data_dir / "training_emg.csv", num_features)
    training_labels = load_csv_labels(data_dir / "training_labels.csv")
    train_data, _, val_data, _ = split_training_validation(
        training_data, training_labels, args.validation_ratio
    )
    train_occ = transition_occupancy(occupancy_by_feature_level(train_data, num_levels))
    val_occ = transition_occupancy(occupancy_by_feature_level(val_data, num_levels))

    rows = []
    for feature in range(num_features):
        for transition in range(transitions):
            rows.append(
                {
                    "feature": feature,
                    "transition": transition,
                    "pearson_accuracy": f"{pearson_accuracy[feature, transition]:.10f}",
                    "spearman_accuracy": f"{spearman_accuracy[feature, transition]:.10f}",
                    "pearson_similarity": f"{pearson_similarity[feature, transition]:.10f}",
                    "spearman_similarity": f"{spearman_similarity[feature, transition]:.10f}",
                    "mean_flip_distance": f"{mean_flip[feature, transition]:.10f}",
                    "std_flip_distance": f"{std_flip[feature, transition]:.10f}",
                    "train_transition_occupancy": int(train_occ[feature, transition]),
                    "validation_transition_occupancy": int(val_occ[feature, transition]),
                }
            )

    out_dir = OUTPUT_DIR / "flipcount_effects" / f"{run_dir.name}_{generation_dir.name}_dataset{args.dataset:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ranked_csv(out_dir / "feature_transition_correlations.csv", rows)

    plot_heatmap(
        pearson_accuracy,
        "Pearson correlation: adjacent flip distance vs accuracy",
        out_dir / "pearson_accuracy_heatmap.png",
        "Pearson r",
        args.show,
    )
    plot_heatmap(
        pearson_similarity,
        "Pearson correlation: adjacent flip distance vs similarity",
        out_dir / "pearson_similarity_heatmap.png",
        "Pearson r",
        args.show,
    )

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    ax.scatter(val_occ.reshape(-1), np.abs(pearson_accuracy).reshape(-1), alpha=0.65, s=20)
    ax.set_title("Do high-impact accuracy transitions sit in populated bins?")
    ax.set_xlabel("Validation occupancy of adjacent level pair")
    ax.set_ylabel("|Pearson r with accuracy|")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_correlation_vs_validation_occupancy.png", dpi=180)
    if args.show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    ax.scatter(val_occ.reshape(-1), np.abs(pearson_similarity).reshape(-1), alpha=0.65, s=20)
    ax.set_title("Do high-impact similarity transitions sit in populated bins?")
    ax.set_xlabel("Validation occupancy of adjacent level pair")
    ax.set_ylabel("|Pearson r with similarity|")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "similarity_correlation_vs_validation_occupancy.png", dpi=180)
    if args.show:
        plt.show()
    plt.close(fig)

    def top(metric, reverse=True):
        return sorted(
            rows,
            key=lambda row: float(row[metric]),
            reverse=reverse,
        )[: args.top_n]

    print(f"Run: {run_dir.name}")
    print(f"Generation: {generation_dir.name}")
    print(f"Dataset: {args.dataset}")
    print(f"Individuals: {len(population)}")
    print(f"Accuracy range: {accuracy.min()*100:.3f}%..{accuracy.max()*100:.3f}%")
    print(f"Similarity range: {similarity.min():.6f}..{similarity.max():.6f}")
    print(f"Output directory: {out_dir}")
    print()
    for title, metric in [
        ("Top positive accuracy correlations", "pearson_accuracy"),
        ("Top negative accuracy correlations", "pearson_accuracy"),
        ("Top positive similarity correlations", "pearson_similarity"),
        ("Top negative similarity correlations", "pearson_similarity"),
    ]:
        reverse = "negative" not in title
        print(title)
        for row in top(metric, reverse=reverse):
            print(
                f"  f={row['feature']:>2} t={row['transition']:>2} "
                f"{metric}={float(row[metric]): .3f} "
                f"mean_flip={float(row['mean_flip_distance']):.3f} "
                f"val_occ={row['validation_transition_occupancy']} "
                f"train_occ={row['train_transition_occupancy']}"
            )
        print()


if __name__ == "__main__":
    main()
