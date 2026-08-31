#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_rows(rows, min_ngram, max_ngram, phase):
    return [
        row
        for row in rows
        if min_ngram <= int(row["n_gram_size"]) <= max_ngram
        and row["phase"] == phase
    ]


def plot_overall(summary_rows, output_path, min_ngram, max_ngram):
    fig, ax = plt.subplots(figsize=(8, 5))
    for phase, label in [
        ("preopt-validation", "Validation accuracy"),
        ("preopt-test", "Test accuracy"),
    ]:
        rows = filter_rows(summary_rows, min_ngram, max_ngram, phase)
        rows.sort(key=lambda row: int(row["n_gram_size"]))
        ax.errorbar(
            [int(row["n_gram_size"]) for row in rows],
            [float(row["mean_accuracy"]) * 100.0 for row in rows],
            yerr=[float(row["std_accuracy"]) * 100.0 for row in rows],
            marker="o",
            capsize=3,
            label=label,
        )

    ax.set_xlabel("N_GRAM_SIZE")
    ax.set_ylabel("Accuracy [%]")
    ax.set_xticks(list(range(min_ngram, max_ngram + 1)))
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_dataset(per_dataset_rows, summary_rows, output_path, min_ngram, max_ngram):
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = filter_rows(per_dataset_rows, min_ngram, max_ngram, "preopt-test")

    for dataset in sorted({int(row["dataset"]) for row in rows}):
        dataset_rows = [row for row in rows if int(row["dataset"]) == dataset]
        dataset_rows.sort(key=lambda row: int(row["n_gram_size"]))
        ax.plot(
            [int(row["n_gram_size"]) for row in dataset_rows],
            [float(row["mean_accuracy"]) * 100.0 for row in dataset_rows],
            marker="o",
            linewidth=1.8,
            label=f"Dataset {dataset}",
        )

    average_rows = filter_rows(summary_rows, min_ngram, max_ngram, "preopt-test")
    average_rows.sort(key=lambda row: int(row["n_gram_size"]))
    print("Average line values (preopt-test):")
    print("n_gram_size,mean_accuracy_percent")
    for row in average_rows:
        print(f"{int(row['n_gram_size'])},{float(row['mean_accuracy']) * 100.0:.4f}")

    ax.plot(
        [int(row["n_gram_size"]) for row in average_rows],
        [float(row["mean_accuracy"]) * 100.0 for row in average_rows],
        color="black",
        linewidth=2.8,
        marker="o",
        label="Average",
    )

    ax.set_xlabel("N-gram size N")
    ax.set_ylabel("Test accuracy in %")
    ax.set_xticks(list(range(min_ngram, max_ngram + 1)))
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot n-gram sweep accuracy from summary CSV files."
    )
    parser.add_argument("--min-ngram", type=int, default=1)
    parser.add_argument("--max-ngram", type=int, default=5)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing summary.csv and per_dataset_summary.csv.",
    )
    parser.add_argument("--no-overall", action="store_true")
    parser.add_argument("--no-per-dataset", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    summary_rows = read_rows(args.base_dir / "summary.csv")
    per_dataset_rows = read_rows(args.base_dir / "per_dataset_summary.csv")

    outputs = []
    if not args.no_overall:
        output = (
            args.base_dir
            / f"ngram_accuracy_{args.min_ngram}_to_{args.max_ngram}_overall.png"
        )
        plot_overall(summary_rows, output, args.min_ngram, args.max_ngram)
        outputs.append(output)

    if not args.no_per_dataset:
        output = (
            args.base_dir
            / f"ngram_accuracy_{args.min_ngram}_to_{args.max_ngram}_per_dataset.png"
        )
        plot_per_dataset(
            per_dataset_rows, summary_rows, output, args.min_ngram, args.max_ngram
        )
        outputs.append(output)

    for output in outputs:
        print(f"saved: {output}")

    if args.show:
        # Re-open through matplotlib only when explicitly requested.
        for output in outputs:
            image = plt.imread(output)
            plt.figure(figsize=(8, 5))
            plt.imshow(image)
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
