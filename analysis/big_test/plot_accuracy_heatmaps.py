import argparse
import csv
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install with 'python -m pip install matplotlib'."
    ) from exc


def parse_info_field(info_value):
    result = {}
    if not info_value:
        return result
    for token in info_value.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def load_accuracy_map(csv_path):
    grouped = defaultdict(list)
    levels = set()
    dimensions = set()
    scopes = set()

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            info = parse_info_field(row.get("info", ""))
            scope = info.get("scope", "overall")
            dataset = int(info["dataset"]) if "dataset" in info else -1

            level = int(row["num_levels"])
            dimension = int(row["vector_dimension"])
            acc = float(row["overall_accuracy"])

            key = (level, dimension, scope, dataset)
            grouped[key].append(acc)
            levels.add(level)
            dimensions.add(dimension)
            scopes.add((scope, dataset))

    avg_map = {key: mean(values) for key, values in grouped.items()}
    return avg_map, sorted(levels), sorted(dimensions), sorted(scopes)


def build_scope_order(scopes):
    scope_keys = []
    if ("overall", -1) in scopes:
        scope_keys.append(("overall", -1))
    dataset_scopes = sorted([s for s in scopes if s[0] == "dataset"], key=lambda x: x[1])
    scope_keys.extend(dataset_scopes)
    for scope in scopes:
        if scope not in scope_keys:
            scope_keys.append(scope)
    return scope_keys


def scope_label(scope, dataset):
    if scope == "overall":
        return "overall"
    if scope == "dataset":
        return f"dataset_{dataset}"
    return f"{scope}_{dataset}"


def plot_model_heatmaps(csv_path, title, out_path, vmin=0.0, vmax=1.0):
    avg_map, levels, dimensions, scopes = load_accuracy_map(csv_path)
    if not levels or not dimensions or not scopes:
        print(f"No plottable rows in: {csv_path}")
        return

    ordered_scopes = build_scope_order(scopes)
    n = len(ordered_scopes)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = [ax for line in axes for ax in line]

    image = None
    for i, (scope, dataset) in enumerate(ordered_scopes):
        matrix = []
        for level in levels:
            row = []
            for dimension in dimensions:
                row.append(avg_map.get((level, dimension, scope, dataset), float("nan")))
            matrix.append(row)

        ax = axes[i]
        image = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        ax.set_xticks(range(len(dimensions)))
        ax.set_xticklabels([str(d) for d in dimensions], rotation=45, ha="right")
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([str(l) for l in levels])
        ax.set_xlabel("Vector Dimension")
        ax.set_ylabel("Num Levels")
        ax.set_title(scope_label(scope, dataset))

    for i in range(n, len(axes)):
        axes[i].axis("off")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes[:n], shrink=0.9)
        cbar.set_label("Accuracy")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    parser = argparse.ArgumentParser(
        description="Plot accuracy heatmaps for your model and Krischan's model (separate figures)."
    )
    parser.add_argument(
        "--mine",
        default=os.path.join(script_dir, "repeats_results_model.csv"),
        help="CSV path for your model results.",
    )
    parser.add_argument(
        "--krischan",
        default=os.path.join(repo_root, "krischans_model", "results", "repeats_results.csv"),
        help="CSV path for Krischan model results.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(script_dir, "plots"),
        help="Directory to store output images.",
    )
    parser.add_argument("--vmin", type=float, default=0.0, help="Heatmap minimum value.")
    parser.add_argument("--vmax", type=float, default=1.0, help="Heatmap maximum value.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mine_out = os.path.join(args.out_dir, "accuracy_heatmaps_mine.png")
    kr_out = os.path.join(args.out_dir, "accuracy_heatmaps_krischan.png")

    plot_model_heatmaps(args.mine, "Mine: Accuracy Heatmaps", mine_out, args.vmin, args.vmax)
    plot_model_heatmaps(args.krischan, "Krischan: Accuracy Heatmaps", kr_out, args.vmin, args.vmax)


if __name__ == "__main__":
    main()
