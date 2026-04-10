import argparse
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Python dependency: numpy. Install with: python -m pip install numpy matplotlib"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from analyze_ga_cim import OUTPUT_DIR, list_run_dirs, load_cim, resolve_run_dir


def parse_generation_index(path: Path) -> int:
    return int(path.name.split("_")[-1])


def get_last_generation_dir(run_dir: Path) -> Path:
    generation_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("generation_")])
    if not generation_dirs:
        raise FileNotFoundError(f"No generation folders found in {run_dir}")
    return generation_dirs[-1]


def compute_adjacent_flips(V: np.ndarray) -> np.ndarray:
    # returns [feature, transition]
    return (V[:-1] != V[1:]).sum(axis=2).T


def safe_normalize_rows(X: np.ndarray) -> np.ndarray:
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return X / row_sums


def cosine_similarity_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T


def pearson_similarity_rows(X: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(X)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    basis = vt[:2].T
    return Xc @ basis


def load_all_last_generation_individuals(run_dirs):
    individuals = []
    skipped_files = []
    for run_dir in run_dirs:
        generation_dir = get_last_generation_dir(run_dir)
        generation_index = parse_generation_index(generation_dir)
        cim_files = sorted(generation_dir.glob("cim_*.csv"))
        if not cim_files:
            continue

        for cim_path in cim_files:
            try:
                header, mode, V = load_cim(cim_path)
            except (ValueError, OSError) as exc:
                skipped_files.append((cim_path, str(exc)))
                continue
            if mode != "precomputed":
                raise ValueError(f"Only precomputed CiMs are supported, got {mode} in {cim_path}")

            flips = compute_adjacent_flips(V)
            individuals.append(
                {
                    "run": run_dir.name,
                    "generation": generation_index,
                    "candidate": int(header.get("candidate", -1)),
                    "accuracy": float(header.get("accuracy", 0.0)),
                    "similarity": float(header.get("similarity", 0.0)),
                    "flips": flips,
                    "num_features": V.shape[1],
                    "num_transitions": V.shape[0] - 1,
                }
            )
    if not individuals:
        raise RuntimeError("No final-generation CiMs found.")
    return individuals, skipped_files


def save_metadata_csv(individuals, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("index,run,generation,candidate,accuracy,similarity,total_flips\n")
        for idx, item in enumerate(individuals):
            f.write(
                f"{idx},{item['run']},{item['generation']},{item['candidate']},"
                f"{item['accuracy']:.10f},{item['similarity']:.10f},{int(item['flips'].sum())}\n"
            )


def save_position_summary_csv(mean_flips, std_flips, out_path: Path):
    num_features, num_transitions = mean_flips.shape
    with out_path.open("w", encoding="utf-8") as f:
        f.write("feature,transition,mean_flips,std_flips\n")
        for feature in range(num_features):
            for transition in range(num_transitions):
                f.write(
                    f"{feature},{transition},{mean_flips[feature, transition]:.10f},"
                    f"{std_flips[feature, transition]:.10f}\n"
                )


def plot_heatmap(matrix, title: str, out_path: Path, colorbar_label: str, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Transition")
    ax.set_ylabel("Feature")
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_individual_similarity(matrix, labels, title: str, out_path: Path, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(9.2, 7.8))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Individual")
    ax.set_ylabel("Individual")
    tick_step = max(1, len(labels) // 16)
    ticks = list(range(0, len(labels), tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=7)
    fig.colorbar(im, ax=ax, label="Similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_embedding(points, run_names, labels, title: str, out_path: Path):
    unique_runs = sorted(set(run_names))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(2, len(unique_runs))))
    color_map = {run: colors[i % len(colors)] for i, run in enumerate(unique_runs)}

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    for idx, (xy, run_name) in enumerate(zip(points, run_names)):
        ax.scatter(xy[0], xy[1], color=color_map[run_name], s=36, alpha=0.85)
        ax.text(xy[0], xy[1], labels[idx], fontsize=7, alpha=0.8)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[run], label=run) for run in unique_runs]
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze similarities and differences across all last-generation CiMs from all runs."
    )
    parser.add_argument("--run", default="all", help='Specific run folder, "latest", or "all" (default).')
    args = parser.parse_args()

    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install with: pip install matplotlib")

    if args.run == "all":
        run_dirs = list_run_dirs()
    else:
        run_dirs = [resolve_run_dir(args.run)]

    individuals, skipped_files = load_all_last_generation_individuals(run_dirs)
    flip_stack = np.stack([item["flips"] for item in individuals], axis=0)
    flat_flips = flip_stack.reshape(flip_stack.shape[0], -1).astype(float)
    flat_flips_norm = safe_normalize_rows(flat_flips)

    mean_flips = flip_stack.mean(axis=0)
    std_flips = flip_stack.std(axis=0)

    cosine_sim = cosine_similarity_rows(flat_flips_norm)
    pearson_sim = pearson_similarity_rows(flat_flips_norm)
    embedding = pca_2d(flat_flips_norm)

    unique_runs = []
    for item in individuals:
        if item["run"] not in unique_runs:
            unique_runs.append(item["run"])
    run_to_idx = {run_name: idx + 1 for idx, run_name in enumerate(unique_runs)}

    labels = [f"r{run_to_idx[item['run']]}-i{item['candidate']:02d}" for item in individuals]
    run_names = [item["run"] for item in individuals]

    run_tag = "all_runs" if args.run == "all" else run_dirs[0].name
    out_dir = OUTPUT_DIR / f"final_generation_similarity_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_metadata_csv(individuals, out_dir / "individual_metadata.csv")
    save_position_summary_csv(mean_flips, std_flips, out_dir / "flip_position_summary.csv")

    plot_heatmap(
        mean_flips,
        "Mean adjacent flips across all last-generation individuals",
        out_dir / "mean_flip_heatmap.png",
        "Mean flips",
    )
    plot_heatmap(
        std_flips,
        "Std of adjacent flips across all last-generation individuals",
        out_dir / "std_flip_heatmap.png",
        "Std flips",
    )
    plot_individual_similarity(
        cosine_sim,
        labels,
        "Cosine similarity across last-generation individuals",
        out_dir / "individual_cosine_similarity.png",
        vmin=0.0,
        vmax=1.0,
    )
    plot_individual_similarity(
        pearson_sim,
        labels,
        "Pearson correlation across last-generation individuals",
        out_dir / "individual_pearson_similarity.png",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_embedding(
        embedding,
        run_names,
        labels,
        "PCA of normalized final-generation flip maps",
        out_dir / "individual_pca.png",
    )

    print(f"Runs analyzed: {len(run_dirs)}")
    print(f"Individuals analyzed: {len(individuals)}")
    if skipped_files:
        print(f"Skipped invalid CiM files: {len(skipped_files)}")
        for path, reason in skipped_files[:10]:
            print(f"  {path}: {reason}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
