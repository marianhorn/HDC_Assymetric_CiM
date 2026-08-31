#!/usr/bin/env python3
"""Reproduce the final-generation Pareto table and correlations used in the thesis."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "dataset3_ga_cim_exports"
    / "final_generation_candidate_validation_test.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the accuracy/similarity Pareto front of the final GA generation."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float | int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "candidate": int(raw["candidate"]),
                    "validation": float(raw["header_validation_accuracy"]),
                    "recomputed_validation": float(raw["recomputed_validation_accuracy"]),
                    "similarity": float(raw["header_similarity"]),
                    "test": float(raw["test_accuracy"]),
                }
            )
    return rows


def first_pareto_front(rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    front = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and float(other["validation"]) >= float(candidate["validation"])
            and float(other["similarity"]) <= float(candidate["similarity"])
            and (
                float(other["validation"]) > float(candidate["validation"])
                or float(other["similarity"]) < float(candidate["similarity"])
            )
            for other in rows
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda row: (float(row["similarity"]), -float(row["validation"])))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            result[order[position]] = average_rank
        start = end
    return result


def main() -> None:
    args = parse_args()
    front = first_pareto_front(load_rows(args.input))

    print("candidate,similarity,validation_accuracy,test_accuracy,test_minus_validation")
    for row in front:
        validation = float(row["validation"])
        test = float(row["test"])
        print(
            f'{int(row["candidate"])},{float(row["similarity"]):.6f},'
            f"{validation:.6f},{test:.6f},{test - validation:+.6f}"
        )

    similarity = [float(row["similarity"]) for row in front]
    test = [float(row["test"]) for row in front]
    test_minus_validation = [
        float(row["test"]) - float(row["recomputed_validation"]) for row in front
    ]
    print(f"\nfront_size={len(front)}")
    print(
        "pearson_similarity_vs_test_minus_validation="
        f"{pearson(similarity, test_minus_validation):.6f}"
    )
    print(
        "spearman_similarity_vs_test_minus_validation="
        f"{pearson(ranks(similarity), ranks(test_minus_validation)):.6f}"
    )
    print(f"pearson_similarity_vs_test_accuracy={pearson(similarity, test):.6f}")


if __name__ == "__main__":
    main()
