import argparse
import os
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
CIMS_ROOT = REPO_ROOT / 'analysis' / 'generated_data' / 'ccim_exports'
OUTPUT_DIR = BASE_DIR / 'plots'

ALL_FEATURE_ALPHA = 0.18
ALL_FEATURE_LINEWIDTH = 0.9


def parse_csv_header(path: Path):
    header = {}
    with path.open('r', encoding='utf-8') as f:
        first = f.readline().strip()
    if not first.startswith('#'):
        return header
    for part in first[1:].split(','):
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()
        if value.replace('.', '', 1).replace('-', '', 1).isdigit():
            if any(ch in value for ch in '.eE'):
                try:
                    header[key] = float(value)
                    continue
                except ValueError:
                    pass
            try:
                header[key] = int(value)
                continue
            except ValueError:
                pass
        header[key] = value
    return header


def list_run_dirs():
    if not CIMS_ROOT.exists():
        raise FileNotFoundError(f'Missing CiMs root: {CIMS_ROOT}')
    runs = sorted(
        [
            p
            for p in CIMS_ROOT.iterdir()
            if p.is_dir()
            and any(
                child.is_dir() and child.name.startswith('generation_')
                for child in p.iterdir()
            )
        ]
    )
    if not runs:
        raise RuntimeError(f'No GA CiM runs found in {CIMS_ROOT}')
    return runs


def resolve_run_dir(run_name: str | None):
    runs = list_run_dirs()
    if run_name is None or run_name == 'latest':
        return runs[-1]
    candidate = CIMS_ROOT / run_name
    if not candidate.is_dir():
        raise FileNotFoundError(f'Run folder not found: {candidate}')
    return candidate


def resolve_cim_path(run_dir: Path, generation: int, individual: int):
    generation_dir = run_dir / f'generation_{generation:04d}'
    if not generation_dir.is_dir():
        raise FileNotFoundError(f'Generation folder not found: {generation_dir}')
    cim_path = generation_dir / f'cim_{individual:04d}.csv'
    if not cim_path.is_file():
        raise FileNotFoundError(f'CiM file not found: {cim_path}')
    return cim_path


def is_binary_vectors(V):
    return np.all((V == 0) | (V == 1))


def load_cim(path: Path):
    header = parse_csv_header(path)
    if not path.exists():
        raise FileNotFoundError(f"CiM file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"CiM file is empty: {path}")
    data = np.loadtxt(path, delimiter=',', comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)

    mode = header.get('mode', 'precomputed')
    dim = int(header.get('dimension', data.shape[1]))
    if data.shape[1] != dim:
        dim = data.shape[1]

    if mode == 'precomputed':
        num_levels = int(header['num_levels'])
        num_features = int(header['num_features'])
        expected_rows = num_levels * num_features
        if data.shape[0] != expected_rows:
            raise ValueError(
                f'Unexpected number of rows for precomputed CiM: got {data.shape[0]}, expected {expected_rows}'
            )
        V = data.reshape((num_levels, num_features, dim))
        return header, mode, V

    num_levels = int(header['num_levels'])
    expected_rows = num_levels
    if data.shape[0] != expected_rows:
        raise ValueError(
            f'Unexpected number of rows for continuous CiM: got {data.shape[0]}, expected {expected_rows}'
        )
    V = data.reshape((num_levels, 1, dim))
    return header, mode, V


def consecutive_hamming_distances(V):
    return (V[:-1] != V[1:]).mean(axis=1)


def adjacent_bitflips_by_feature(V):
    num_levels, num_features, dimension = V.shape
    adjacent_all = np.zeros((num_features, num_levels - 1), dtype=float)
    for feature in range(num_features):
        adjacent_all[feature] = consecutive_hamming_distances(V[:, feature, :])
    return adjacent_all * float(dimension)


def print_bitflip_summary(adjacent_bitflips_all, reference_bitflips=None):
    transition_mean = adjacent_bitflips_all.mean(axis=0)
    transition_min = adjacent_bitflips_all.min(axis=0)
    transition_max = adjacent_bitflips_all.max(axis=0)
    transition_std = adjacent_bitflips_all.std(axis=0)
    feature_mean = adjacent_bitflips_all.mean(axis=1)
    feature_min = adjacent_bitflips_all.min(axis=1)
    feature_max = adjacent_bitflips_all.max(axis=1)
    all_values = adjacent_bitflips_all.reshape(-1)

    print()
    print('Characteristic bitflip metrics:')
    print(f'  all_feature_transition_mean={all_values.mean():.3f}')
    print(f'  all_feature_transition_std={all_values.std():.3f}')
    print(f'  all_feature_transition_min={all_values.min():.3f}')
    print(f'  all_feature_transition_max={all_values.max():.3f}')
    print(f'  transition_mean_min={transition_mean.min():.3f}')
    print(f'  transition_mean_max={transition_mean.max():.3f}')
    print(f'  transition_mean_avg={transition_mean.mean():.3f}')
    print(f'  feature_mean_min={feature_mean.min():.3f}')
    print(f'  feature_mean_max={feature_mean.max():.3f}')
    print(f'  feature_mean_avg={feature_mean.mean():.3f}')
    print(f'  feature_range_max={(feature_max - feature_min).max():.3f}')
    if reference_bitflips is not None:
        print(f'  reference_mean={reference_bitflips.mean():.3f}')
        print(f'  reference_std={reference_bitflips.std():.3f}')
        print(f'  reference_min={reference_bitflips.min():.3f}')
        print(f'  reference_max={reference_bitflips.max():.3f}')
        print(f'  optimized_minus_reference_mean={(transition_mean - reference_bitflips).mean():.3f}')
        print(f'  optimized_minus_reference_min={(transition_mean - reference_bitflips).min():.3f}')
        print(f'  optimized_minus_reference_max={(transition_mean - reference_bitflips).max():.3f}')

    print()
    if reference_bitflips is None:
        print('Per-transition bitflips across features:')
        print('transition,mean,min,max,std')
        for transition in range(adjacent_bitflips_all.shape[1]):
            print(
                f'{transition},'
                f'{transition_mean[transition]:.3f},'
                f'{transition_min[transition]:.3f},'
                f'{transition_max[transition]:.3f},'
                f'{transition_std[transition]:.3f}'
            )
    else:
        print('Per-transition bitflips across features:')
        print('transition,mean,min,max,std,reference,mean_minus_reference')
        for transition in range(adjacent_bitflips_all.shape[1]):
            print(
                f'{transition},'
                f'{transition_mean[transition]:.3f},'
                f'{transition_min[transition]:.3f},'
                f'{transition_max[transition]:.3f},'
                f'{transition_std[transition]:.3f},'
                f'{reference_bitflips[transition]:.3f},'
                f'{transition_mean[transition] - reference_bitflips[transition]:.3f}'
            )


def plot_cim_analysis(run_dir: Path, cim_path: Path, header, mode, V, show: bool, reference_cim: Path | None):
    if plt is None:
        raise RuntimeError('matplotlib is not installed. Install with: pip install matplotlib')

    if not is_binary_vectors(V):
        raise ValueError('This analysis expects binary CCIM vectors containing only 0/1 values.')
    num_levels, num_features, dimension = V.shape

    line_distance_label = 'Bitflips'
    heatmap_distance_label = 'Hamming distance'
    metric_name = 'Hamming'

    adjacent_bitflips_all = adjacent_bitflips_by_feature(V)
    adjacent_all = adjacent_bitflips_all / float(dimension)
    adjacent_bitflips_mean = adjacent_bitflips_all.mean(axis=0)
    adjacent_bitflips_std = adjacent_bitflips_all.std(axis=0)
    reference_bitflips = None
    if reference_cim is not None:
        _, _, reference_vectors = load_cim(reference_cim)
        if not is_binary_vectors(reference_vectors):
            raise ValueError(f'Reference CIM is not binary: {reference_cim}')
        if reference_vectors.shape != V.shape:
            raise ValueError(f'Reference CIM shape {reference_vectors.shape} does not match GA CIM shape {V.shape}')
        reference_bitflips = adjacent_bitflips_by_feature(reference_vectors)[0]

    print_bitflip_summary(adjacent_bitflips_all, reference_bitflips)

    run_name = run_dir.name
    generation = int(header.get('generation', -1))
    candidate = int(header.get('candidate', -1))
    accuracy = header.get('accuracy', None)
    similarity = header.get('similarity', None)

    stem = f'{run_name}_gen{generation:04d}_cim{candidate:04d}'
    out_dir = OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    x = np.arange(num_levels - 1)
    for feature in range(num_features):
        ax.plot(
            x,
            adjacent_bitflips_all[feature],
            color='#0b3d91',
            alpha=0.45,
            linewidth=ALL_FEATURE_LINEWIDTH,
            label='GA-optimized flip counts per feature' if feature == 0 else None,
        )
    ax.plot(
        x,
        adjacent_bitflips_mean,
        color='#d62728',
        linewidth=2.4,
        label='GA-optimized flip counts - mean across features',
    )
    ax.fill_between(
        x,
        adjacent_bitflips_mean - adjacent_bitflips_std,
        adjacent_bitflips_mean + adjacent_bitflips_std,
        color='#d62728',
        alpha=0.18,
        label=r'GA-optimized flip counts - standard deviation',
    )
    if reference_bitflips is not None:
        ax.plot(
            x,
            reference_bitflips,
            color='black',
            linestyle='--',
            linewidth=2.0,
            label='Equally distributed flip counts (baseline)',
        )
    ax.set_xlabel(r'Level transition index $i$')
    ax.set_ylabel(r'Number of bit flips $b_i^f$')
    ax.set_xlim(0, 38)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'adjacent_distances.png', dpi=180)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    im = ax.imshow(adjacent_all, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(
        f'{run_name} | generation {generation} | individual {candidate}\n'
        f'Adjacent level {metric_name} distance per feature'
    )
    ax.set_xlabel('Level transition l -> l+1')
    ax.set_ylabel('Feature')
    fig.colorbar(im, ax=ax, label=heatmap_distance_label)
    fig.tight_layout()
    fig.savefig(out_dir / 'adjacent_distance_heatmap.png', dpi=180)
    if show:
        plt.show()
    plt.close(fig)

    summary_path = out_dir / 'summary.txt'
    with summary_path.open('w', encoding='utf-8') as f:
        f.write(f'run={run_name}\n')
        f.write(f'cim_path={cim_path}\n')
        f.write(f'mode={mode}\n')
        f.write(f'num_levels={num_levels}\n')
        f.write(f'num_features={num_features}\n')
        f.write(f'dimension={V.shape[2]}\n')
        if accuracy is not None:
            f.write(f'accuracy={accuracy}\n')
        if similarity is not None:
            f.write(f'similarity={similarity}\n')
        f.write('binary_mode=True\n')
        if reference_cim is not None:
            f.write(f'reference_cim={reference_cim}\n')
        f.write(f'adjacent_bitflips_mean_min={adjacent_bitflips_mean.min():.10f}\n')
        f.write(f'adjacent_bitflips_mean_max={adjacent_bitflips_mean.max():.10f}\n')
        f.write(f'adjacent_bitflips_mean_avg={adjacent_bitflips_mean.mean():.10f}\n')
        if reference_bitflips is not None:
            f.write(f'reference_adjacent_bitflips_min={reference_bitflips.min():.10f}\n')
            f.write(f'reference_adjacent_bitflips_max={reference_bitflips.max():.10f}\n')
            f.write(f'reference_adjacent_bitflips_avg={reference_bitflips.mean():.10f}\n')

    print(f'Loaded: {cim_path}')
    if accuracy is not None and similarity is not None:
        print(f'accuracy={float(accuracy) * 100.0:.3f}% similarity={float(similarity):.6f}')
    print(f'mode={mode} num_levels={num_levels} num_features={num_features} dimension={V.shape[2]}')
    print(f'Output directory: {out_dir}')



def main():
    parser = argparse.ArgumentParser(
        description='Visualize one exported GA CiM for a specific run, generation, and individual.'
    )
    parser.add_argument('--run', default='latest', help='GA run folder name under analysis/generated_data/ccim_exports, or "latest" (default).')
    parser.add_argument('--generation', type=int, default=0, help='Generation index to inspect (default: 0).')
    parser.add_argument('--individual', type=int, default=0, help='Individual index to inspect (default: 0).')
    parser.add_argument('--reference-cim', type=Path, help='Optional real naive/reference CIM CSV to overlay.')
    parser.add_argument('--show', action='store_true', help='Show plots interactively in addition to saving them.')
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    cim_path = resolve_cim_path(run_dir, args.generation, args.individual)
    header, mode, V = load_cim(cim_path)
    plot_cim_analysis(run_dir, cim_path, header, mode, V, args.show, args.reference_cim)


if __name__ == '__main__':
    main()
