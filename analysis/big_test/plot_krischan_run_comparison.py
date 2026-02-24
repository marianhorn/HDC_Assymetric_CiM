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


def parse_info(info_value):
    parsed = {}
    if not info_value:
        return parsed
    for token in info_value.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def load_accuracy_grid(csv_path, excluded_datasets=None):
    if excluded_datasets is None:
        excluded_datasets = set()
    grouped = defaultdict(list)
    levels = set()
    dims = set()
    scopes = set()

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            info = parse_info(row.get("info", ""))
            scope = info.get("scope", "overall")
            dataset_id = int(info["dataset"]) if "dataset" in info else -1
            if scope == "dataset" and dataset_id in excluded_datasets:
                continue
            level = int(row["num_levels"])
            dim = int(row["vector_dimension"])
            acc = float(row["overall_accuracy"])

            key = (level, dim, scope, dataset_id)
            grouped[key].append(acc)
            levels.add(level)
            dims.add(dim)
            scopes.add((scope, dataset_id))

    avg = {k: mean(v) for k, v in grouped.items()}
    return avg, sorted(levels), sorted(dims), scopes


def ordered_scopes(scopes):
    out = []
    if ("overall", -1) in scopes:
        out.append(("overall", -1))
    out.extend(sorted([s for s in scopes if s[0] == "dataset"], key=lambda s: s[1]))
    out.extend(sorted([s for s in scopes if s[0] != "overall" and s[0] != "dataset"]))
    return out


def scope_label(scope, dataset_id):
    if scope == "overall":
        return "overall"
    if scope == "dataset":
        return f"dataset_{dataset_id}"
    return f"{scope}_{dataset_id}"


def value_range(a_map, b_map):
    values = []
    for key in set(a_map.keys()) | set(b_map.keys()):
        a = a_map.get(key, float("nan"))
        b = b_map.get(key, float("nan"))
        if a == a and b == b:
            values.append(a - b)
    values = [v for v in values if v == v]  # filter NaN
    if not values:
        return -0.1, 0.1
    vmax = max(abs(min(values)), abs(max(values)))
    if vmax == 0:
        vmax = 0.1
    return -vmax, vmax


def draw_comparison_heatmap(current_map, previous_map, levels, dims, scopes, title, out_path, vmin, vmax):
    n = len(scopes)
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
    for i, (scope, dataset_id) in enumerate(scopes):
        matrix = []
        for level in levels:
            row = []
            for dim in dims:
                key = (level, dim, scope, dataset_id)
                current = current_map.get(key, float("nan"))
                previous = previous_map.get(key, float("nan"))
                row.append(current - previous if current == current and previous == previous else float("nan"))
            matrix.append(row)

        ax = axes[i]
        image = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
        ax.set_title(scope_label(scope, dataset_id))
        ax.set_xlabel("Vector Dimension")
        ax.set_ylabel("Num Levels")
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels([str(d) for d in dims], rotation=45, ha="right")
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([str(l) for l in levels])

    for i in range(n, len(axes)):
        axes[i].axis("off")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes[:n], shrink=0.9)
        cbar.set_label("Accuracy delta (current - previous)")
    fig.text(
        0.5,
        0.02,
        "Green = current run better, Red = previous run better",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def format_case(key):
    level, dim, scope, dataset_id = key
    if scope == "dataset":
        return f"L={level} D={dim} dataset={dataset_id}"
    return f"L={level} D={dim} scope={scope}"


def print_text_analysis(current_map, previous_map, eps, top_n):
    rows = []
    for key in sorted(set(current_map.keys()) & set(previous_map.keys())):
        current_acc = current_map[key]
        previous_acc = previous_map[key]
        if current_acc != current_acc or previous_acc != previous_acc:
            continue
        delta = current_acc - previous_acc
        if delta > eps:
            winner = "current"
        elif delta < -eps:
            winner = "previous"
        else:
            winner = "tie"
        rows.append((key, current_acc, previous_acc, delta, winner))

    current_better = [r for r in rows if r[4] == "current"]
    previous_better = [r for r in rows if r[4] == "previous"]
    ties = [r for r in rows if r[4] == "tie"]

    print(f"Compared cases: {len(rows)}")
    print(f"Current better: {len(current_better)}")
    print(f"Previous better: {len(previous_better)}")
    print(f"Ties: {len(ties)}")

    if not rows:
        return

    top_n = max(0, top_n)
    if top_n == 0:
        return

    current_better.sort(key=lambda x: x[3], reverse=True)
    previous_better.sort(key=lambda x: x[3])

    if current_better:
        print(f"\nTop {min(top_n, len(current_better))} cases where current is better:")
        for key, current_acc, previous_acc, delta, _ in current_better[:top_n]:
            print(
                f"  {format_case(key)} | current={current_acc:.4f} "
                f"previous={previous_acc:.4f} delta={delta:+.4f}"
            )

    if previous_better:
        print(f"\nTop {min(top_n, len(previous_better))} cases where previous is better:")
        for key, current_acc, previous_acc, delta, _ in previous_better[:top_n]:
            print(
                f"  {format_case(key)} | current={current_acc:.4f} "
                f"previous={previous_acc:.4f} delta={delta:+.4f}"
            )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Create heatmaps for current and previous Krischan run results."
    )
    parser.add_argument(
        "--current",
        default=os.path.join(script_dir, "..", "..", "krischans_model", "results", "repeats_results.csv"),
        help="CSV path of current Krischan run.",
    )
    parser.add_argument(
        "--previous",
        default=os.path.join(script_dir, "plots", "krischanVSadapted", "results_krischan.csv"),
        help="CSV path of previous Krischan run.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(script_dir, "plots", "krischanD"),
        help="Output directory for heatmap image.",
    )
    parser.add_argument("--eps", type=float, default=1e-9, help="Tie tolerance for delta.")
    parser.add_argument("--top", type=int, default=10, help="Top-N cases to print for each side.")
    args = parser.parse_args()

    excluded_datasets = {1}
    current_map, current_levels, current_dims, current_scopes = load_accuracy_grid(
        args.current, excluded_datasets
    )
    previous_map, prev_levels, prev_dims, prev_scopes = load_accuracy_grid(
        args.previous, excluded_datasets
    )

    levels = sorted(set(current_levels) | set(prev_levels))
    dims = sorted(set(current_dims) | set(prev_dims))
    scopes = ordered_scopes(set(current_scopes) | set(prev_scopes))
    vmin, vmax = value_range(current_map, previous_map)

    os.makedirs(args.out_dir, exist_ok=True)
    compare_out = os.path.join(args.out_dir, "accuracy_heatmap_krischan_current_vs_previous.png")

    draw_comparison_heatmap(
        current_map,
        previous_map,
        levels,
        dims,
        scopes,
        "Krischan Run Comparison (Current - Previous)",
        compare_out,
        vmin,
        vmax,
    )
    print_text_analysis(current_map, previous_map, args.eps, args.top)


if __name__ == "__main__":
    main()
