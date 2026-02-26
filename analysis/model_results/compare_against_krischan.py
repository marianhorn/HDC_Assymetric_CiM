#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install with 'python -m pip install matplotlib'."
    ) from exc

KRISCHAN_BASELINE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "krischan",
    "repeats_results.csv",
)


def parse_info_field(info_value):
    parsed = {}
    if not info_value:
        return parsed
    for token in info_value.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def load_grouped_results(csv_path, phase_filter=None):
    grouped = defaultdict(list)
    phases_found = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            info = parse_info_field(row.get("info", ""))
            phase = info.get("phase")
            if phase:
                phases_found.add(phase)
            # Only enforce phase filter on rows that actually contain a phase tag.
            if phase_filter is not None and phase is not None and phase != phase_filter:
                continue

            scope = info.get("scope", "overall")
            dataset_raw = info.get("dataset")
            dataset_id = int(dataset_raw) if dataset_raw is not None else -1
            key = (
                int(row["num_levels"]),
                int(row["vector_dimension"]),
                scope,
                dataset_id,
            )
            grouped[key].append(float(row["overall_accuracy"]))
    return grouped, sorted(phases_found)


def means_by_key(grouped):
    result = {}
    for key, values in grouped.items():
        if values:
            result[key] = sum(values) / len(values)
    return result


def compare(baseline, candidate, eps):
    rows = []
    keys = sorted(set(baseline.keys()) & set(candidate.keys()))
    for key in keys:
        b = baseline[key]
        c = candidate[key]
        delta = c - b
        if delta > eps:
            winner = "candidate"
        elif delta < -eps:
            winner = "krischan"
        else:
            winner = "tie"
        rows.append(
            {
                "num_levels": key[0],
                "vector_dimension": key[1],
                "scope": key[2],
                "dataset_id": key[3],
                "krischan_accuracy": b,
                "candidate_accuracy": c,
                "delta_candidate_minus_krischan": delta,
                "winner": winner,
            }
        )
    return rows


def save_comparison_csv(path, rows):
    fields = [
        "num_levels",
        "vector_dimension",
        "scope",
        "dataset_id",
        "krischan_accuracy",
        "candidate_accuracy",
        "delta_candidate_minus_krischan",
        "winner",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scope_name(row):
    if row["scope"] == "overall":
        return "overall"
    return f"dataset_{row['dataset_id']}"


def summarize(rows):
    lines = []

    def emit(line=""):
        print(line)
        lines.append(line)

    wins_candidate = [r for r in rows if r["winner"] == "candidate"]
    wins_krischan = [r for r in rows if r["winner"] == "krischan"]
    ties = [r for r in rows if r["winner"] == "tie"]

    emit(f"Compared cases: {len(rows)}")
    emit(f"Candidate better: {len(wins_candidate)}")
    emit(f"Krischan better: {len(wins_krischan)}")
    emit(f"Tie: {len(ties)}")

    mean_delta = sum(r["delta_candidate_minus_krischan"] for r in rows) / len(rows)
    emit(f"Mean delta (candidate - krischan): {mean_delta:+.4f}")

    scope_groups = defaultdict(list)
    for row in rows:
        scope_groups[scope_name(row)].append(row["delta_candidate_minus_krischan"])

    emit("\nPer-scope delta summary:")
    for scope in sorted(scope_groups):
        vals = scope_groups[scope]
        local_mean = sum(vals) / len(vals)
        pos = sum(1 for v in vals if v > 0)
        neg = sum(1 for v in vals if v < 0)
        emit(f"  {scope}: mean={local_mean:+.4f}, better={pos}, worse={neg}, total={len(vals)}")

    by_dim = defaultdict(list)
    by_lvl = defaultdict(list)
    for row in rows:
        by_dim[row["vector_dimension"]].append(row["delta_candidate_minus_krischan"])
        by_lvl[row["num_levels"]].append(row["delta_candidate_minus_krischan"])

    emit("\nBy vector dimension (mean delta):")
    for dim in sorted(by_dim):
        vals = by_dim[dim]
        emit(f"  D={dim}: {sum(vals)/len(vals):+.4f}")

    emit("\nBy num levels (mean delta):")
    for lvl in sorted(by_lvl):
        vals = by_lvl[lvl]
        emit(f"  L={lvl}: {sum(vals)/len(vals):+.4f}")

    if by_dim:
        best_dim = max(by_dim, key=lambda d: sum(by_dim[d]) / len(by_dim[d]))
        worst_dim = min(by_dim, key=lambda d: sum(by_dim[d]) / len(by_dim[d]))
        emit(
            "\nObserved pattern hint:"
            f" strongest average gains at D={best_dim}, strongest average losses at D={worst_dim}."
        )

    if by_lvl:
        best_lvl = max(by_lvl, key=lambda l: sum(by_lvl[l]) / len(by_lvl[l]))
        worst_lvl = min(by_lvl, key=lambda l: sum(by_lvl[l]) / len(by_lvl[l]))
        emit(
            "Observed pattern hint:"
            f" strongest average gains at L={best_lvl}, strongest average losses at L={worst_lvl}."
        )

    return "\n".join(lines) + "\n"


def build_matrix(rows, scope_key, levels, dims):
    lookup = {}
    for row in rows:
        if scope_name(row) != scope_key:
            continue
        lookup[(row["num_levels"], row["vector_dimension"])] = row["delta_candidate_minus_krischan"]

    matrix = []
    for lvl in levels:
        line = []
        for dim in dims:
            line.append(lookup.get((lvl, dim), float("nan")))
        matrix.append(line)
    return matrix


def plot_heatmaps(rows, out_path):
    levels = sorted({r["num_levels"] for r in rows})
    dims = sorted({r["vector_dimension"] for r in rows})
    scopes = ["overall"] + sorted({scope_name(r) for r in rows if scope_name(r) != "overall"})
    if not scopes:
        raise RuntimeError("No scopes available for plotting.")

    v_abs = max(abs(r["delta_candidate_minus_krischan"]) for r in rows)
    if v_abs == 0:
        v_abs = 1.0

    cols = 3
    rows_count = math.ceil(len(scopes) / cols)
    fig, axes = plt.subplots(
        rows_count,
        cols,
        figsize=(5.8 * cols, 4.4 * rows_count),
        constrained_layout=True,
    )
    if rows_count == 1 and cols == 1:
        axes = [axes]
    elif rows_count == 1:
        axes = list(axes)
    else:
        axes = [ax for line in axes for ax in line]

    image = None
    for idx, scope in enumerate(scopes):
        ax = axes[idx]
        matrix = build_matrix(rows, scope, levels, dims)
        image = ax.imshow(matrix, cmap="RdYlGn", vmin=-v_abs, vmax=v_abs, aspect="auto", origin="lower")
        ax.set_title(f"{scope} (delta: candidate - krischan)")
        ax.set_xlabel("Vector dimension")
        ax.set_ylabel("Num levels")
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels([str(d) for d in dims], rotation=45, ha="right")
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([str(l) for l in levels])

    for idx in range(len(scopes), len(axes)):
        axes[idx].axis("off")

    cbar = fig.colorbar(image, ax=axes[: len(scopes)], shrink=0.9)
    cbar.set_label("Accuracy delta (green=candidate better, red=krischan better)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description=(
            "Compare one run folder against Krischan baseline. "
            "The folder must contain big_run_results.csv."
        )
    )
    parser.add_argument(
        "run_dir",
        help="Path to run directory containing big_run_results.csv",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    candidate_csv = os.path.join(run_dir, "big_run_results.csv")
    if not os.path.isdir(run_dir):
        raise RuntimeError(f"Run directory does not exist: {run_dir}")
    if not os.path.isfile(candidate_csv):
        raise RuntimeError(f"Expected candidate CSV not found: {candidate_csv}")

    baseline_grouped, _ = load_grouped_results(KRISCHAN_BASELINE, phase_filter=None)
    candidate_all, candidate_phases = load_grouped_results(candidate_csv, phase_filter=None)

    if not baseline_grouped:
        raise RuntimeError("No baseline rows loaded after filtering.")
    if not candidate_all:
        raise RuntimeError("No candidate rows loaded after filtering.")

    candidate_phase = None
    if candidate_phases:
        if "test" in candidate_phases:
            candidate_phase = "test"
        elif len(candidate_phases) == 1:
            candidate_phase = candidate_phases[0]
        else:
            raise RuntimeError(
                "Candidate CSV contains multiple phases and no unique default phase.\n"
                f"Found phases: {candidate_phases}\n"
                "Please keep only one phase in the input CSV."
            )

    if candidate_phase is not None:
        print(f"Using candidate phase='{candidate_phase}' for comparison.")
    candidate_grouped, _ = load_grouped_results(candidate_csv, phase_filter=candidate_phase)

    baseline_means = means_by_key(baseline_grouped)
    candidate_means = means_by_key(candidate_grouped)
    rows = compare(baseline_means, candidate_means, eps=1e-9)
    if not rows:
        raise RuntimeError("No overlapping config/scope cases between baseline and candidate.")

    out_csv = os.path.join(run_dir, "comparison_against_krischan.csv")
    out_heatmap = os.path.join(run_dir, "comparison_against_krischan_heatmaps.png")
    out_report = os.path.join(run_dir, "comparison_to_baseline.txt")

    save_comparison_csv(out_csv, rows)
    summary_text = summarize(rows)
    plot_heatmaps(rows, out_heatmap)

    with open(out_report, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"Baseline CSV: {KRISCHAN_BASELINE}\n")
        handle.write(f"Candidate CSV: {candidate_csv}\n")
        if candidate_phase is not None:
            handle.write(f"Candidate phase used: {candidate_phase}\n")
        handle.write("\n")
        handle.write(summary_text)
        handle.write(f"\nSaved detailed comparison CSV: {out_csv}\n")
        handle.write(f"Saved heatmap figure: {out_heatmap}\n")

    print(f"\nSaved detailed comparison CSV: {out_csv}")
    print(f"Saved heatmap figure: {out_heatmap}")
    print(f"Saved text report: {out_report}")


if __name__ == "__main__":
    main()
