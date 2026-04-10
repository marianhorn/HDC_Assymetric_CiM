import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from analyze_ga_cim import OUTPUT_DIR, list_run_dirs, load_cim, resolve_cim_path, resolve_run_dir


def compute_adjacent_metrics(V):
    # V shape: [num_levels, num_features, dimension]
    adjacent_diff = V[:-1] != V[1:]
    adjacent_hamming = adjacent_diff.mean(axis=2).T
    adjacent_flips = adjacent_diff.sum(axis=2).T
    adjacent_mean = adjacent_hamming.mean(axis=0)
    adjacent_std = adjacent_hamming.std(axis=0)
    return adjacent_mean, adjacent_std, adjacent_flips


def load_individual_stats(run_dir: Path, generation: int, individual: int):
    cim_path = resolve_cim_path(run_dir, generation, individual)
    header, mode, V = load_cim(cim_path)
    if mode != "precomputed":
        raise ValueError(f"Only precomputed CiMs are supported, got mode={mode} for {cim_path}")

    adjacent_mean, adjacent_std, adjacent_flips = compute_adjacent_metrics(V)
    return {
        "path": cim_path,
        "header": header,
        "adjacent_mean": adjacent_mean,
        "adjacent_std": adjacent_std,
        "adjacent_flips": adjacent_flips,
        "num_levels": V.shape[0],
        "num_features": V.shape[1],
        "dimension": V.shape[2],
    }


def build_stats_for_generation(run_dir: Path, generation: int, individual_count: int):
    stats = []
    for individual in range(individual_count):
        stats.append(load_individual_stats(run_dir, generation, individual))
    return stats


def collect_global_ranges(stats_a, stats_b):
    all_stats = list(stats_a) + list(stats_b)
    max_adjacent = 0.0
    max_flips = 0.0
    for stats in all_stats:
        max_adjacent = max(max_adjacent, float(np.max(stats["adjacent_mean"] + stats["adjacent_std"])))
        max_flips = max(max_flips, float(np.max(stats["adjacent_flips"])))
    return max_adjacent, max_flips


def plot_adjacent_panel(run_name: str, generation: int, stats_list, ymax: float, out_path: Path, show: bool):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, stats in zip(axes, stats_list):
        x = np.arange(stats["num_levels"] - 1)
        ax.plot(x, stats["adjacent_mean"], color="#d62728", linewidth=2.2)
        ax.fill_between(
            x,
            stats["adjacent_mean"] - stats["adjacent_std"],
            stats["adjacent_mean"] + stats["adjacent_std"],
            color="#d62728",
            alpha=0.2,
        )

        header = stats["header"]
        title = f'ind {int(header.get("candidate", -1)):02d}'
        if "accuracy" in header and "similarity" in header:
            title += f'\nacc {float(header["accuracy"]) * 100.0:.2f}% | sim {float(header["similarity"]):.3f}'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Adjacent level transition")
        ax.set_ylabel("Mean Hamming distance")
        ax.set_ylim(0.0, ymax * 1.05 if ymax > 0.0 else 1.0)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(stats_list):]:
        ax.axis("off")

    fig.suptitle(
        f"{run_name} | generation {generation} | adjacent level distance across first {len(stats_list)} individuals",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_flip_heatmap_panel(run_name: str, generation: int, stats_list, vmax: float, out_path: Path, show: bool):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), sharex=True, sharey=True)
    axes = axes.flatten()
    image = None

    for ax, stats in zip(axes, stats_list):
        image = ax.imshow(
            stats["adjacent_flips"],
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=vmax if vmax > 0.0 else None,
            cmap="viridis",
        )

        header = stats["header"]
        title = f'ind {int(header.get("candidate", -1)):02d}'
        if "accuracy" in header:
            title += f'\nacc {float(header["accuracy"]) * 100.0:.2f}%'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Adjacent level transition")
        ax.set_ylabel("Feature")

    for ax in axes[len(stats_list):]:
        ax.axis("off")

    fig.suptitle(
        f"{run_name} | generation {generation} | per-feature adjacent flip counts across first {len(stats_list)} individuals",
        fontsize=13,
    )
    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), shrink=0.88, label="Flips between adjacent levels")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def process_run(run_dir: Path, generation_a: int, generation_b: int, individual_count: int, show: bool):
    run_name = run_dir.name

    stats_a = build_stats_for_generation(run_dir, generation_a, individual_count)
    stats_b = build_stats_for_generation(run_dir, generation_b, individual_count)
    max_adjacent, max_flips = collect_global_ranges(stats_a, stats_b)

    out_dir = OUTPUT_DIR / f"{run_name}_gen{generation_a:04d}_vs_gen{generation_b:04d}_first{individual_count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_adjacent_panel(
        run_name,
        generation_a,
        stats_a,
        max_adjacent,
        out_dir / f"adjacent_generation_{generation_a:04d}.png",
        show,
    )
    plot_adjacent_panel(
        run_name,
        generation_b,
        stats_b,
        max_adjacent,
        out_dir / f"adjacent_generation_{generation_b:04d}.png",
        show,
    )
    plot_flip_heatmap_panel(
        run_name,
        generation_a,
        stats_a,
        max_flips,
        out_dir / f"flip_heatmap_generation_{generation_a:04d}.png",
        show,
    )
    plot_flip_heatmap_panel(
        run_name,
        generation_b,
        stats_b,
        max_flips,
        out_dir / f"flip_heatmap_generation_{generation_b:04d}.png",
        show,
    )

    print(f"Run: {run_name}")
    print(f"Saved plots to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot adjacent-level and per-feature flip-heatmap panels for the first 4 individuals "
            "of generation 0 and generation 31 from exported GA CiM runs."
        )
    )
    parser.add_argument("--run", default="all", help='GA run folder name under CiMs, "latest", or "all" (default).')
    parser.add_argument("--generation-a", type=int, default=0, help="First generation to visualize (default: 0).")
    parser.add_argument("--generation-b", type=int, default=31, help="Second generation to visualize (default: 31).")
    parser.add_argument("--individual-count", type=int, default=4, help="How many first individuals to include (default: 4).")
    parser.add_argument("--show", action="store_true", help="Show plots interactively in addition to saving them.")
    args = parser.parse_args()

    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install with: pip install matplotlib")
    if args.individual_count <= 0:
        raise ValueError("--individual-count must be > 0")
    if args.individual_count > 4:
        raise ValueError("--individual-count must be <= 4 for the fixed 2x2 subplot layout")

    if args.run == "all":
        run_dirs = list_run_dirs()
    else:
        run_dirs = [resolve_run_dir(args.run)]

    processed = 0
    skipped = 0
    for run_dir in run_dirs:
        try:
            process_run(run_dir, args.generation_a, args.generation_b, args.individual_count, args.show)
            processed += 1
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {run_dir.name}: {exc}")
            skipped += 1

    print(f"Processed runs: {processed}")
    if skipped > 0:
        print(f"Skipped runs: {skipped}")


if __name__ == "__main__":
    main()
