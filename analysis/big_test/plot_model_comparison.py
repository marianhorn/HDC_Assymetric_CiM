import argparse
import csv
import os

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install with 'python -m pip install matplotlib'."
    ) from exc


def load_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "num_levels": int(row["num_levels"]),
                    "vector_dimension": int(row["vector_dimension"]),
                    "scope": row["scope"],
                    "dataset_id": int(row["dataset_id"]),
                    "delta": float(row["delta_mine_minus_krischan"]),
                    "winner": row["winner"],
                }
            )
    return rows


def build_scope_key(row):
    if row["scope"] == "overall":
        return "overall"
    return f"dataset_{row['dataset_id']}"


def build_delta_matrix(rows, scope_key, levels, dimensions):
    delta_lookup = {}
    for row in rows:
        if build_scope_key(row) != scope_key:
            continue
        delta_lookup[(row["num_levels"], row["vector_dimension"])] = row["delta"]

    matrix = []
    for level in levels:
        line = []
        for dimension in dimensions:
            line.append(delta_lookup.get((level, dimension), float("nan")))
        matrix.append(line)
    return matrix


def winner_counts(rows, scope_key):
    counts = {"mine": 0, "krischan": 0, "tie": 0}
    for row in rows:
        if build_scope_key(row) != scope_key:
            continue
        counts[row["winner"]] = counts.get(row["winner"], 0) + 1
    return counts


def plot_heatmaps(rows, out_path):
    levels = sorted({row["num_levels"] for row in rows})
    dimensions = sorted({row["vector_dimension"] for row in rows})

    scope_keys = ["overall"] + sorted(
        {build_scope_key(row) for row in rows if build_scope_key(row) != "overall"}
    )
    if not scope_keys:
        print("No data found for heatmap.")
        return

    n = len(scope_keys)
    cols = 3
    rows_count = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(5.5 * cols, 4.5 * rows_count))
    if rows_count == 1 and cols == 1:
        axes = [axes]
    elif rows_count == 1:
        axes = list(axes)
    else:
        axes = [ax for line in axes for ax in line]

    v_abs = max(abs(row["delta"]) for row in rows) if rows else 1.0
    if v_abs == 0:
        v_abs = 1.0

    for idx, scope_key in enumerate(scope_keys):
        ax = axes[idx]
        matrix = build_delta_matrix(rows, scope_key, levels, dimensions)
        image = ax.imshow(matrix, cmap="RdYlGn", vmin=-v_abs, vmax=v_abs, aspect="auto", origin="lower")

        ax.set_xticks(range(len(dimensions)))
        ax.set_xticklabels([str(d) for d in dimensions], rotation=45, ha="right")
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([str(l) for l in levels])
        ax.set_xlabel("Vector Dimension")
        ax.set_ylabel("Num Levels")
        ax.set_title(f"Delta (mine - krischan): {scope_key}")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    cbar = fig.colorbar(image, ax=axes[:n], shrink=0.9)
    cbar.set_label("Accuracy difference")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_winner_bars(rows, out_path):
    scope_keys = ["overall"] + sorted(
        {build_scope_key(row) for row in rows if build_scope_key(row) != "overall"}
    )
    if not scope_keys:
        print("No data found for winner bars.")
        return

    mine_vals = []
    krischan_vals = []
    tie_vals = []
    for scope_key in scope_keys:
        counts = winner_counts(rows, scope_key)
        mine_vals.append(counts.get("mine", 0))
        krischan_vals.append(counts.get("krischan", 0))
        tie_vals.append(counts.get("tie", 0))

    x = list(range(len(scope_keys)))
    width = 0.27

    fig, ax = plt.subplots(figsize=(max(9, len(scope_keys) * 1.4), 4.5))
    ax.bar([i - width for i in x], mine_vals, width=width, label="Mine better")
    ax.bar(x, krischan_vals, width=width, label="Krischan better")
    ax.bar([i + width for i in x], tie_vals, width=width, label="Tie")

    ax.set_xticks(x)
    ax.set_xticklabels(scope_keys, rotation=30, ha="right")
    ax.set_ylabel("Number of configs")
    ax.set_title("Win/Loss counts per scope")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(script_dir, "comparison_model_vs_krischan.csv")
    default_out_dir = os.path.join(script_dir, "plots")

    parser = argparse.ArgumentParser(description="Plot visual comparison between your model and Krischan's model.")
    parser.add_argument("--input", default=default_in, help="Input CSV created by compare_model_runs.py")
    parser.add_argument("--out-dir", default=default_out_dir, help="Output folder for plots")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        print("Input CSV has no rows.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    heatmap_path = os.path.join(args.out_dir, "comparison_heatmaps.png")
    bar_path = os.path.join(args.out_dir, "comparison_winner_counts.png")

    plot_heatmaps(rows, heatmap_path)
    plot_winner_bars(rows, bar_path)

    print(f"Saved heatmaps to: {heatmap_path}")
    print(f"Saved winner-count chart to: {bar_path}")


if __name__ == "__main__":
    main()
