import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, 'runs')


def aggregate_mean_std_by_dataset_generation(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['dataset'], row['generation'])].append(row['new_selected'])

    dataset_to_xy = defaultdict(lambda: {'x': [], 'mean': [], 'std': []})
    for (dataset, generation), values in sorted(grouped.items()):
        arr = np.array(values, dtype=float)
        dataset_to_xy[dataset]['x'].append(generation)
        dataset_to_xy[dataset]['mean'].append(float(np.mean(arr)))
        dataset_to_xy[dataset]['std'].append(float(np.std(arr)))
    return dataset_to_xy


def load_generation_rows(metrics_csv):
    rows = []
    with open(metrics_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    'seed': int(row['seed']),
                    'dataset': int(row['dataset']),
                    'generation': int(row['generation']),
                    'new_selected': float(row['new_selected']),
                }
            )
    return rows


def plot_one_config(config_dir, show):
    metrics_csv = os.path.join(config_dir, 'generation_metrics.csv')
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f'Missing metrics file: {metrics_csv}')

    rows = load_generation_rows(metrics_csv)
    series = aggregate_mean_std_by_dataset_generation(rows)
    if not series:
        raise RuntimeError(f'No new-selected data found in {metrics_csv}')

    plots_dir = os.path.join(config_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    label = os.path.relpath(config_dir, RUNS_DIR).replace('\\', '/')

    plt.figure(figsize=(9, 5))
    for dataset in sorted(series.keys()):
        x = np.array(series[dataset]['x'], dtype=float)
        mean = np.array(series[dataset]['mean'], dtype=float)
        std = np.array(series[dataset]['std'], dtype=float)
        plt.plot(x, mean, marker='o', linewidth=1.8, markersize=3, label=f'Dataset {dataset}')
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel('Generation')
    plt.ylabel('Newly selected individuals')
    plt.title(f'{label}: newly selected individuals vs generation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(plots_dir, 'new_selected_vs_generation_mean_std.png')
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
    return out_path


def find_all_config_dirs():
    config_dirs = []
    for root, dirs, files in os.walk(RUNS_DIR):
        if 'generation_metrics.csv' in files:
            config_dirs.append(root)
    config_dirs.sort()
    return config_dirs


def main():
    parser = argparse.ArgumentParser(description='Plot convergence-fix results from generation_metrics.csv files.')
    parser.add_argument('--config', help='Relative config path under runs/, e.g. alpha_beta/a0_b0_w0.2_cx0.7_mut0.2')
    parser.add_argument('--show', action='store_true', help='Show plots interactively.')
    args = parser.parse_args()

    if args.config:
        config_dirs = [os.path.join(RUNS_DIR, args.config)]
    else:
        config_dirs = find_all_config_dirs()

    if not config_dirs:
        raise FileNotFoundError(f'No generation_metrics.csv files found under {RUNS_DIR}')

    for config_dir in config_dirs:
        out_path = plot_one_config(config_dir, args.show)
        print(f'Saved plot: {out_path}')


if __name__ == '__main__':
    main()
