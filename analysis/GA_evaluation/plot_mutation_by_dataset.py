import csv
import os
from collections import defaultdict
from statistics import mean

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install with 'python -m pip install matplotlib'."
    ) from exc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "results", "results_all.csv")
OUT_PATH = os.path.join(BASE_DIR, "mutation_rate_by_dataset_postopt_test.png")


def parse_info(info):
    parts = {}
    for token in info.split(","):
        if "=" in token:
            key, value = token.split("=", 1)
            parts[key.strip()] = value.strip()
    return parts


def load_dataset_postopt_test(path):
    values = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            info = row.get("info", "")
            if "scope=dataset" not in info or "phase=postopt-test" not in info:
                continue

            parts = parse_info(info)
            if "dataset" not in parts:
                continue

            dataset = int(parts["dataset"])
            mutation = float(row["ga_mutation_rate"])
            accuracy = float(row["overall_accuracy"])
            values[(dataset, mutation)].append(accuracy)

    return values


def main():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}")

    values = load_dataset_postopt_test(RESULTS_PATH)
    if not values:
        raise RuntimeError("No dataset postopt-test rows found in results_all.csv")

    datasets = sorted({d for d, _ in values.keys()})
    mutations = sorted({m for _, m in values.keys()})

    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset in datasets:
        xs = []
        ys = []
        for mutation in mutations:
            arr = values.get((dataset, mutation), [])
            if not arr:
                continue
            xs.append(mutation)
            ys.append(mean(arr) * 100.0)

        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"Dataset {dataset}")

    ax.set_title("Post-opt Test Accuracy vs Mutation Rate (per dataset)")
    ax.set_xlabel("Mutation rate")
    ax.set_ylabel("Accuracy [%]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=180)
    plt.close(fig)

    print(f"Saved plot to: {OUT_PATH}")


if __name__ == "__main__":
    main()
