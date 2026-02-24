import argparse
import csv
import os
from collections import defaultdict


def parse_info_field(info_value):
    result = {}
    if not info_value:
        return result
    for token in info_value.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def load_results(csv_path):
    grouped = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            info = parse_info_field(row.get("info", ""))
            scope = info.get("scope", "overall")
            dataset = info.get("dataset")
            dataset_id = int(dataset) if dataset is not None else -1

            key = (
                int(row["num_levels"]),
                int(row["vector_dimension"]),
                scope,
                dataset_id,
            )
            grouped[key].append(float(row["overall_accuracy"]))
    return grouped


def mean(values):
    return sum(values) / len(values) if values else 0.0


def format_key(key):
    num_levels, vector_dimension, scope, dataset_id = key
    if scope == "dataset":
        return f"L={num_levels} D={vector_dimension} dataset={dataset_id}"
    return f"L={num_levels} D={vector_dimension} scope={scope}"


def compare(mine, krischan, scope_filter, dataset_filter, eps):
    rows = []
    common_keys = sorted(set(mine.keys()) & set(krischan.keys()))
    for key in common_keys:
        num_levels, vector_dimension, scope, dataset_id = key
        if scope_filter != "all" and scope != scope_filter:
            continue
        if dataset_filter is not None and dataset_id != dataset_filter:
            continue

        mine_mean = mean(mine[key])
        kr_mean = mean(krischan[key])
        delta = mine_mean - kr_mean
        if delta > eps:
            winner = "mine"
        elif delta < -eps:
            winner = "krischan"
        else:
            winner = "tie"

        rows.append(
            {
                "num_levels": num_levels,
                "vector_dimension": vector_dimension,
                "scope": scope,
                "dataset_id": dataset_id,
                "mine_mean_accuracy": mine_mean,
                "krischan_mean_accuracy": kr_mean,
                "delta_mine_minus_krischan": delta,
                "winner": winner,
            }
        )
    return rows


def write_comparison_csv(output_path, rows):
    fieldnames = [
        "num_levels",
        "vector_dimension",
        "scope",
        "dataset_id",
        "mine_mean_accuracy",
        "krischan_mean_accuracy",
        "delta_mine_minus_krischan",
        "winner",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    default_mine = os.path.join(script_dir, "repeats_results_model.csv")
    default_krischan = os.path.join(repo_root, "krischans_model", "results", "repeats_results.csv")
    default_output = os.path.join(script_dir, "comparison_model_vs_krischan.csv")

    parser = argparse.ArgumentParser(
        description="Compare model accuracies between your run output and Krischan's run output."
    )
    parser.add_argument("--mine", default=default_mine, help="CSV from your model run.")
    parser.add_argument("--krischan", default=default_krischan, help="CSV from Krischan's run.")
    parser.add_argument("--scope", choices=["all", "overall", "dataset"], default="all",
                        help="Limit comparison scope.")
    parser.add_argument("--dataset", type=int, default=None,
                        help="If set, compare only one dataset id (scope=dataset).")
    parser.add_argument("--eps", type=float, default=1e-9,
                        help="Tie tolerance for accuracy difference.")
    parser.add_argument("--top", type=int, default=10,
                        help="How many strongest win-cases per model to print.")
    parser.add_argument("--output", default=default_output,
                        help="Path for detailed comparison CSV.")
    args = parser.parse_args()

    mine = load_results(args.mine)
    krischan = load_results(args.krischan)
    rows = compare(mine, krischan, args.scope, args.dataset, args.eps)

    if not rows:
        print("No overlapping cases found for the selected filters.")
        return

    wins_mine = [r for r in rows if r["winner"] == "mine"]
    wins_kr = [r for r in rows if r["winner"] == "krischan"]
    ties = [r for r in rows if r["winner"] == "tie"]

    print(f"Compared cases: {len(rows)}")
    print(f"Mine better: {len(wins_mine)}")
    print(f"Krischan better: {len(wins_kr)}")
    print(f"Tie: {len(ties)}")

    overall_mine = mean([r["mine_mean_accuracy"] for r in rows])
    overall_kr = mean([r["krischan_mean_accuracy"] for r in rows])
    print(f"Mean accuracy across compared cases -> mine: {overall_mine:.4f}, krischan: {overall_kr:.4f}")

    wins_mine_sorted = sorted(wins_mine, key=lambda r: r["delta_mine_minus_krischan"], reverse=True)
    wins_kr_sorted = sorted(wins_kr, key=lambda r: r["delta_mine_minus_krischan"])

    top_n = max(0, args.top)
    if top_n > 0 and wins_mine_sorted:
        print(f"\nTop {min(top_n, len(wins_mine_sorted))} cases where mine is better:")
        for row in wins_mine_sorted[:top_n]:
            key = (row["num_levels"], row["vector_dimension"], row["scope"], row["dataset_id"])
            print(
                f"  {format_key(key)} | mine={row['mine_mean_accuracy']:.4f} "
                f"krischan={row['krischan_mean_accuracy']:.4f} delta={row['delta_mine_minus_krischan']:+.4f}"
            )

    if top_n > 0 and wins_kr_sorted:
        print(f"\nTop {min(top_n, len(wins_kr_sorted))} cases where Krischan is better:")
        for row in wins_kr_sorted[:top_n]:
            key = (row["num_levels"], row["vector_dimension"], row["scope"], row["dataset_id"])
            print(
                f"  {format_key(key)} | mine={row['mine_mean_accuracy']:.4f} "
                f"krischan={row['krischan_mean_accuracy']:.4f} delta={row['delta_mine_minus_krischan']:+.4f}"
            )

    write_comparison_csv(args.output, rows)
    print(f"\nDetailed comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
