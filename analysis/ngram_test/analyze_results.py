import csv
import os
import numpy as np
import matplotlib.pyplot as plt

CSV_NAME = "results.csv"
ACC_FIELD = "overall_accuracy"  # or "class_average_accuracy"

PHASES = {
    "preopt-test": "pre",
    "postopt-test": "post",
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
    ngram_set = set()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            info = parse_info(row.get("info", ""))
            phase = info.get("phase", None)
            if phase not in PHASES:
                continue
            try:
                n_gram = int(row["n_gram_size"])
                acc = float(row[ACC_FIELD])
            except (KeyError, ValueError):
                continue
            ngram_set.add(n_gram)
            rows.append((n_gram, phase, acc))

    ngrams = sorted(ngram_set)
    if not rows:
        print("No usable rows found. Check CSV header and info field.")
        return

    idx = {n: i for i, n in enumerate(ngrams)}
    pre = np.full(len(ngrams), np.nan)
    post = np.full(len(ngrams), np.nan)

    for n_gram, phase, acc in rows:
        i = idx[n_gram]
        if phase == "preopt-test":
            pre[i] = acc
        elif phase == "postopt-test":
            post[i] = acc

    print(f"Loaded {len(rows)} rows from {CSV_NAME}")
    print(f"N_GRAM_SIZE: {ngrams}")
    print(f"Accuracy field: {ACC_FIELD}")

    plt.figure()
    plt.plot(ngrams, pre, marker="o", label="Pre-opt test")
    plt.plot(ngrams, post, marker="o", label="Post-opt test")
    plt.title("Test accuracy vs N_GRAM_SIZE")
    plt.xlabel("N_GRAM_SIZE")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
