import csv
import os
import numpy as np
import matplotlib.pyplot as plt

CSV_NAME = "results.csv"
ACC_FIELD = "overall_accuracy"  # or "class_average_accuracy"

PHASES = {
    "preopt-test": "pre_test",
    "postopt-test": "post_test",
    "preopt-val": "pre_val",
    "postopt-val": "post_val",
}


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
    dims_set = set()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            info = parse_info(row.get("info", ""))
            phase = info.get("phase", None)
            if phase not in PHASES:
                continue
            try:
                vector_dim = int(row["vector_dimension"])
                acc = float(row[ACC_FIELD])
            except (KeyError, ValueError):
                continue
            dims_set.add(vector_dim)
            rows.append((vector_dim, phase, acc))

    dims = sorted(dims_set)
    if not rows:
        print("No usable rows found. Check CSV header and info field.")
        return

    dim_idx = {d: i for i, d in enumerate(dims)}
    pre_test = np.full(len(dims), np.nan)
    post_test = np.full(len(dims), np.nan)
    pre_val = np.full(len(dims), np.nan)
    post_val = np.full(len(dims), np.nan)

    for vector_dim, phase, acc in rows:
        i = dim_idx[vector_dim]
        if phase == "preopt-test":
            pre_test[i] = acc
        elif phase == "postopt-test":
            post_test[i] = acc
        elif phase == "preopt-val":
            pre_val[i] = acc
        elif phase == "postopt-val":
            post_val[i] = acc

    print(f"Loaded {len(rows)} rows from {CSV_NAME}")
    print(f"Vector dimensions: {dims}")
    print(f"Accuracy field: {ACC_FIELD}")

    plt.figure()
    plt.plot(dims, pre_test, marker="o", label="Pre-opt test")
    plt.plot(dims, post_test, marker="o", label="Post-opt test")
    plt.title("Test accuracy vs VECTOR_DIMENSION")
    plt.xlabel("VECTOR_DIMENSION")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(dims, pre_val, marker="o", label="Pre-opt val")
    plt.plot(dims, post_val, marker="o", label="Post-opt val")
    plt.title("Validation accuracy vs VECTOR_DIMENSION")
    plt.xlabel("VECTOR_DIMENSION")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
