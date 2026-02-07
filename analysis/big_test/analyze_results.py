import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------------
# CONFIG
# -------------------------
CSV_NAME = "results.csv"
ACC_FIELD = "overall_accuracy"  # or "class_average_accuracy"

# Expected phases in info field
PHASES = {
    "preopt-val": "preopt_val",
    "preopt-test": "preopt_test",
    "postopt-val": "postopt_val",
    "postopt-test": "postopt_test",
}

# -------------------------
# HELPERS
# -------------------------

def parse_info(info_str):
    info = {}
    if not info_str:
        return info
    parts = [p.strip() for p in info_str.split(",") if p.strip()]
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def print_stats(name, arr, datasets, levels, dims):
    valid = ~np.isnan(arr)
    count = int(np.sum(valid))
    print(f"\n{name}:")
    print(f"  shape: {arr.shape} (datasets={len(datasets)}, levels={len(levels)}, dims={len(dims)})")
    print(f"  filled: {count} / {arr.size}")
    if count == 0:
        return
    vals = arr[valid]
    print(f"  mean: {np.mean(vals):.6f}")
    print(f"  std:  {np.std(vals):.6f}")
    print(f"  min:  {np.min(vals):.6f}")
    print(f"  max:  {np.max(vals):.6f}")

def plot_dataset_surfaces(dataset_id, levels, dims, arrays, acc_field):
    levels_arr = np.array(levels)
    dims_arr = np.array(dims)
    X, Y = np.meshgrid(levels_arr, dims_arr, indexing="ij")

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(f"Dataset {dataset_id} - {acc_field}")

    for idx, (title, data) in enumerate(arrays, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        Z = np.ma.masked_invalid(data)
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", antialiased=True)
        ax.set_title(title)
        ax.set_xlabel("NUM_LEVELS")
        ax.set_ylabel("VECTOR_DIMENSION")
        ax.set_zlabel("Accuracy")

    fig.tight_layout()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, CSV_NAME)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    rows = []
    datasets = set()
    num_levels_set = set()
    vector_dims_set = set()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            info = parse_info(row.get("info", ""))
            dataset = info.get("dataset", None)
            phase = info.get("phase", None)
            if dataset is None or phase not in PHASES:
                continue
            try:
                dataset = int(dataset)
                num_levels = int(row["num_levels"])
                vector_dim = int(row["vector_dimension"])
                acc = float(row[ACC_FIELD])
            except (KeyError, ValueError):
                continue
            datasets.add(dataset)
            num_levels_set.add(num_levels)
            vector_dims_set.add(vector_dim)
            rows.append((dataset, num_levels, vector_dim, phase, acc))

    datasets = sorted(datasets)
    levels = sorted(num_levels_set)
    dims = sorted(vector_dims_set)

    if not rows:
        print("No usable rows found. Check CSV header and info field.")
        return

    d_idx = {d: i for i, d in enumerate(datasets)}
    l_idx = {l: i for i, l in enumerate(levels)}
    v_idx = {v: i for i, v in enumerate(dims)}

    shape = (len(datasets), len(levels), len(dims))
    preopt_val = np.full(shape, np.nan)
    preopt_test = np.full(shape, np.nan)
    postopt_val = np.full(shape, np.nan)
    postopt_test = np.full(shape, np.nan)

    phase_to_array = {
        "preopt-val": preopt_val,
        "preopt-test": preopt_test,
        "postopt-val": postopt_val,
        "postopt-test": postopt_test,
    }

    for dataset, num_levels, vector_dim, phase, acc in rows:
        i = d_idx[dataset]
        j = l_idx[num_levels]
        k = v_idx[vector_dim]
        phase_to_array[phase][i, j, k] = acc

    print(f"Loaded {len(rows)} rows from {CSV_NAME}")
    print(f"Datasets: {datasets}")
    print(f"Num_levels: {levels}")
    print(f"Vector_dimension: {dims}")
    print(f"Accuracy field: {ACC_FIELD}")

    print_stats("Pre-optimization validation", preopt_val, datasets, levels, dims)
    print_stats("Pre-optimization test", preopt_test, datasets, levels, dims)
    print_stats("Post-optimization validation", postopt_val, datasets, levels, dims)
    print_stats("Post-optimization test", postopt_test, datasets, levels, dims)

    # Optional: report missing counts per phase
    for label, arr in [
        ("preopt-val", preopt_val),
        ("preopt-test", preopt_test),
        ("postopt-val", postopt_val),
        ("postopt-test", postopt_test),
    ]:
        missing = int(np.sum(np.isnan(arr)))
        print(f"Missing entries for {label}: {missing}")

    # Accuracy delta (post - pre) on test set
    delta_test = postopt_test - preopt_test
    print_stats("Post-pre test accuracy delta", delta_test, datasets, levels, dims)

    # 3D plots per dataset
    for dataset in datasets:
        i = d_idx[dataset]
        plot_dataset_surfaces(
            dataset,
            levels,
            dims,
            [
                ("Pre-opt val", preopt_val[i]),
                ("Pre-opt test", preopt_test[i]),
                ("Post-opt val", postopt_val[i]),
                ("Post-opt test", postopt_test[i]),
            ],
            ACC_FIELD,
        )

    # Delta plots (post - pre) for test set in one figure
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Post-Pre Test Accuracy Delta (All Datasets)")
    cols = 2
    rows = (len(datasets) + cols - 1) // cols
    X, Y = np.meshgrid(np.array(levels), np.array(dims), indexing="ij")
    for idx, dataset in enumerate(datasets, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        i = d_idx[dataset]
        Z = np.ma.masked_invalid(delta_test[i])
        ax.plot_surface(X, Y, Z, cmap="coolwarm", edgecolor="none", antialiased=True)
        ax.set_title(f"Dataset {dataset}")
        ax.set_xlabel("NUM_LEVELS")
        ax.set_ylabel("VECTOR_DIMENSION")
        ax.set_zlabel("Accuracy Delta")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
