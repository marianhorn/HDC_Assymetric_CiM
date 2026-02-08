import csv
import os
import numpy as np
import matplotlib.pyplot as plt

CSV_NAME = "intermed_results.csv"
ACC_FIELD = "overall_accuracy"  # or "class_average_accuracy"
TARGET_DATASET = 1
TARGET_LEVELS = {41, 61}

PHASES = ["preopt-test", "postopt-test"]


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


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, CSV_NAME)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    rows = []
    level_set = set()
    dim_set = set()
    dataset_set = set()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            info = parse_info(row.get("info", ""))
            phase = info.get("phase", None)
            if phase not in PHASES:
                continue
            dataset = info.get("dataset", None)
            try:
                dataset = int(dataset)
            except (TypeError, ValueError):
                continue
            try:
                num_levels = int(row["num_levels"])
                vector_dim = int(row["vector_dimension"])
                acc = float(row[ACC_FIELD])
            except (KeyError, ValueError):
                continue

            level_set.add(num_levels)
            dim_set.add(vector_dim)
            dataset_set.add(dataset)
            rows.append((dataset, num_levels, vector_dim, phase, acc))

    levels = sorted(level_set)
    if TARGET_LEVELS:
        levels = [lvl for lvl in levels if lvl in TARGET_LEVELS]
    dims = sorted(dim_set)

    if not rows:
        print("No usable rows found. Check CSV header and info field.")
        return

    lvl_idx = {l: i for i, l in enumerate(levels)}
    dim_idx = {d: i for i, d in enumerate(dims)}

    data = {phase: np.full((len(levels), len(dims)), np.nan) for phase in PHASES}
    datasets = sorted(dataset_set)
    ds_idx = {d: i for i, d in enumerate(datasets)}
    data_all = {phase: np.full((len(datasets), len(levels), len(dims)), np.nan) for phase in PHASES}

    for dataset, num_levels, vector_dim, phase, acc in rows:
        if TARGET_LEVELS and num_levels not in TARGET_LEVELS:
            continue
        i = lvl_idx[num_levels]
        j = dim_idx[vector_dim]
        data_all[phase][ds_idx[dataset], i, j] = acc
        if dataset == TARGET_DATASET:
            data[phase][i, j] = acc

    print(f"Loaded {len(rows)} rows from {CSV_NAME}")
    print(f"Dataset: {TARGET_DATASET}")
    print(f"Datasets (all): {datasets}")
    print(f"Num_levels: {levels}")
    print(f"Vector_dimension: {dims}")
    print(f"Accuracy field: {ACC_FIELD}")

    for i, lvl in enumerate(levels):
        plt.figure()
        plt.plot(dims, data["preopt-test"][i], marker="o", label="Pre-opt test")
        plt.plot(dims, data["postopt-test"][i], marker="o", label="Post-opt test")
        plt.title(f"Accuracy vs VECTOR_DIMENSION (dataset {TARGET_DATASET}, NUM_LEVELS={lvl})")
        plt.xlabel("VECTOR_DIMENSION")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

    # Aggregate across all datasets: mean +/- std
    for i, lvl in enumerate(levels):
        pre_vals = data_all["preopt-test"][:, i, :]
        post_vals = data_all["postopt-test"][:, i, :]

        pre_mean = np.nanmean(pre_vals, axis=0)
        pre_std = np.nanstd(pre_vals, axis=0)
        post_mean = np.nanmean(post_vals, axis=0)
        post_std = np.nanstd(post_vals, axis=0)

        plt.figure()
        plt.plot(dims, pre_mean, marker="o", label="Pre-opt test (mean)")
        plt.fill_between(dims, pre_mean - pre_std, pre_mean + pre_std, alpha=0.2)
        plt.plot(dims, post_mean, marker="o", label="Post-opt test (mean)")
        plt.fill_between(dims, post_mean - post_std, post_mean + post_std, alpha=0.2)
        plt.title(f"Accuracy vs VECTOR_DIMENSION (all datasets, NUM_LEVELS={lvl})")
        plt.xlabel("VECTOR_DIMENSION")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
