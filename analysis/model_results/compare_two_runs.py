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


def scope_name(row):
    if row["scope"] == "overall":
        return "overall"
    return f"dataset_{row['dataset_id']}"


def compare(run_a, run_b, eps):
    rows = []
    keys = sorted(set(run_a.keys()) & set(run_b.keys()))
    for key in keys:
        a = run_a[key]
        b = run_b[key]
        delta = b - a
        if delta > eps:
            winner = "run_b"
        elif delta < -eps:
            winner = "run_a"
        else:
            winner = "tie"
        rows.append(
            {
                "num_levels": key[0],
                "vector_dimension": key[1],
                "scope": key[2],
                "dataset_id": key[3],
                "run_a_accuracy": a,
                "run_b_accuracy": b,
                "delta_run_b_minus_run_a": delta,
                "winner": winner,
            }
        )
    return rows


def summarize(rows, label_a, label_b):
    lines = []

    def emit(line=""):
        print(line)
        lines.append(line)

    wins_a = [r for r in rows if r["winner"] == "run_a"]
    wins_b = [r for r in rows if r["winner"] == "run_b"]
    ties = [r for r in rows if r["winner"] == "tie"]

    emit(f"Compared cases: {len(rows)}")
    emit(f"{label_b} better: {len(wins_b)}")
    emit(f"{label_a} better: {len(wins_a)}")
    emit(f"Tie: {len(ties)}")

    mean_delta = sum(r["delta_run_b_minus_run_a"] for r in rows) / len(rows)
    emit(f"Mean delta ({label_b} - {label_a}): {mean_delta:+.4f}")

    scope_groups = defaultdict(list)
    for row in rows:
        scope_groups[scope_name(row)].append(row["delta_run_b_minus_run_a"])

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
        by_dim[row["vector_dimension"]].append(row["delta_run_b_minus_run_a"])
        by_lvl[row["num_levels"]].append(row["delta_run_b_minus_run_a"])

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


def save_comparison_csv(path, rows):
    fields = [
        "num_levels",
        "vector_dimension",
        "scope",
        "dataset_id",
        "run_a_accuracy",
        "run_b_accuracy",
        "delta_run_b_minus_run_a",
        "winner",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_matrix(rows, scope_key, levels, dims):
    lookup = {}
    for row in rows:
        if scope_name(row) != scope_key:
            continue
        lookup[(row["num_levels"], row["vector_dimension"])] = row["delta_run_b_minus_run_a"]

    matrix = []
    for lvl in levels:
        line = []
        for dim in dims:
            line.append(lookup.get((lvl, dim), float("nan")))
        matrix.append(line)
    return matrix


def plot_heatmaps(rows, out_path, label_a, label_b):
    levels = sorted({r["num_levels"] for r in rows})
    dims = sorted({r["vector_dimension"] for r in rows})
    scopes = ["overall"] + sorted({scope_name(r) for r in rows if scope_name(r) != "overall"})
    if not scopes:
        raise RuntimeError("No scopes available for plotting.")

    v_abs = max(abs(r["delta_run_b_minus_run_a"]) for r in rows)
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
        ax.set_title(f"{scope} (delta: {label_b} - {label_a})")
        ax.set_xlabel("Vector dimension")
        ax.set_ylabel("Num levels")
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels([str(d) for d in dims], rotation=45, ha="right")
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([str(l) for l in levels])

    for idx in range(len(scopes), len(axes)):
        axes[idx].axis("off")

    cbar = fig.colorbar(image, ax=axes[: len(scopes)], shrink=0.9)
    cbar.set_label(f"Accuracy delta (green={label_b} better, red={label_a} better)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def choose_phase(run_a_phases, run_b_phases, explicit_phase):
    if explicit_phase is not None:
        return explicit_phase

    if not run_a_phases and not run_b_phases:
        return None

    common = sorted(set(run_a_phases) & set(run_b_phases))
    if not common:
        return None
    if "test" in common:
        return "test"
    if len(common) == 1:
        return common[0]

    raise RuntimeError(
        "Multiple common phases found and no --phase set.\n"
        f"Run A phases: {run_a_phases}\n"
        f"Run B phases: {run_b_phases}\n"
        f"Common phases: {common}\n"
        "Pass --phase explicitly."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two of your run folders. "
            "Each folder must contain big_run_results.csv."
        )
    )
    parser.add_argument("run_a_dir", help="First run directory containing big_run_results.csv")
    parser.add_argument("run_b_dir", help="Second run directory containing big_run_results.csv")
    parser.add_argument("--phase", default=None, help="Optional phase filter (example: test)")
    parser.add_argument("--label-a", default=None, help="Legend/report name for run A")
    parser.add_argument("--label-b", default=None, help="Legend/report name for run B")
    args = parser.parse_args()

    run_a_dir = os.path.abspath(args.run_a_dir)
    run_b_dir = os.path.abspath(args.run_b_dir)
    run_a_csv = os.path.join(run_a_dir, "big_run_results.csv")
    run_b_csv = os.path.join(run_b_dir, "big_run_results.csv")

    if not os.path.isdir(run_a_dir):
        raise RuntimeError(f"Run A directory does not exist: {run_a_dir}")
    if not os.path.isdir(run_b_dir):
        raise RuntimeError(f"Run B directory does not exist: {run_b_dir}")
    if not os.path.isfile(run_a_csv):
        raise RuntimeError(f"Expected CSV missing in run A directory: {run_a_csv}")
    if not os.path.isfile(run_b_csv):
        raise RuntimeError(f"Expected CSV missing in run B directory: {run_b_csv}")

    grouped_a_all, phases_a = load_grouped_results(run_a_csv, phase_filter=None)
    grouped_b_all, phases_b = load_grouped_results(run_b_csv, phase_filter=None)
    if not grouped_a_all:
        raise RuntimeError("No rows loaded from run A CSV.")
    if not grouped_b_all:
        raise RuntimeError("No rows loaded from run B CSV.")

    phase = choose_phase(phases_a, phases_b, args.phase)
    if phase is not None:
        print(f"Using phase='{phase}' for both runs.")
    grouped_a, _ = load_grouped_results(run_a_csv, phase_filter=phase)
    grouped_b, _ = load_grouped_results(run_b_csv, phase_filter=phase)

    means_a = means_by_key(grouped_a)
    means_b = means_by_key(grouped_b)
    rows = compare(means_a, means_b, eps=1e-9)
    if not rows:
        raise RuntimeError("No overlapping config/scope cases between run A and run B.")

    label_a = args.label_a if args.label_a else os.path.basename(os.path.normpath(run_a_dir))
    label_b = args.label_b if args.label_b else os.path.basename(os.path.normpath(run_b_dir))

    out_dir = os.getcwd()
    out_csv = os.path.join(out_dir, "comparison_between_runs.csv")
    out_png = os.path.join(out_dir, "comparison_between_runs_heatmaps.png")
    out_txt = os.path.join(out_dir, "comparison_between_runs.txt")

    save_comparison_csv(out_csv, rows)
    summary_text = summarize(rows, label_a, label_b)
    plot_heatmaps(rows, out_png, label_a, label_b)

    with open(out_txt, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"Run A folder: {run_a_dir}\n")
        handle.write(f"Run B folder: {run_b_dir}\n")
        handle.write(f"Run A CSV: {run_a_csv}\n")
        handle.write(f"Run B CSV: {run_b_csv}\n")
        if phase is not None:
            handle.write(f"Phase used: {phase}\n")
        handle.write(f"Label A: {label_a}\n")
        handle.write(f"Label B: {label_b}\n\n")
        handle.write(summary_text)
        handle.write(f"\nSaved comparison CSV: {out_csv}\n")
        handle.write(f"Saved heatmap PNG: {out_png}\n")

    print(f"\nSaved comparison CSV: {out_csv}")
    print(f"Saved heatmap PNG: {out_png}")
    print(f"Saved text report: {out_txt}")


if __name__ == "__main__":
    main()
