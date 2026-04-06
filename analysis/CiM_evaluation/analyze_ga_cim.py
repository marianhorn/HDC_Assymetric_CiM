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
CIMS_ROOT = REPO_ROOT / 'CiMs'
OUTPUT_DIR = BASE_DIR / 'plots'

FORCE_BINARY_MODE = None
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
    runs = sorted([p for p in CIMS_ROOT.iterdir() if p.is_dir()])
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


def cosine_similarity_matrix(V):
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    Vn = V / norms
    return Vn @ Vn.T


def hamming_similarity_matrix(V):
    diff = V[:, None, :] != V[None, :, :]
    dist = diff.mean(axis=2)
    return 1.0 - dist


def consecutive_cosine_distances(V):
    norms = np.linalg.norm(V, axis=1) + 1e-12
    dots = np.sum(V[:-1] * V[1:], axis=1)
    cos = dots / (norms[:-1] * norms[1:] + 1e-12)
    return 1.0 - cos


def consecutive_hamming_distances(V):
    return (V[:-1] != V[1:]).mean(axis=1)


def classical_mds_from_distance(D, out_dim=2):
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals = np.clip(eigvals[:out_dim], a_min=0.0, a_max=None)
    return eigvecs[:, :out_dim] * np.sqrt(eigvals + 1e-12)


def similarity_matrix(V, binary_mode):
    return hamming_similarity_matrix(V) if binary_mode else cosine_similarity_matrix(V)


def consecutive_distances(V, binary_mode):
    return consecutive_hamming_distances(V) if binary_mode else consecutive_cosine_distances(V)


def plot_cim_analysis(run_dir: Path, cim_path: Path, header, mode, V, show: bool):
    if plt is None:
        raise RuntimeError('matplotlib is not installed. Install with: pip install matplotlib')

    binary_mode = bool(FORCE_BINARY_MODE) if FORCE_BINARY_MODE is not None else is_binary_vectors(V)
    num_levels, num_features, _ = V.shape

    distance_label = 'Hamming distance' if binary_mode else '1 - cosine similarity'
    similarity_label = 'Hamming similarity' if binary_mode else 'cosine similarity'
    metric_name = 'Hamming' if binary_mode else 'cosine'

    adjacent_all = np.zeros((num_features, num_levels - 1), dtype=float)
    similarity_sum = np.zeros((num_levels, num_levels), dtype=float)
    density = np.zeros((num_features, num_levels), dtype=float)

    for feature in range(num_features):
        Vf = V[:, feature, :]
        adjacent_all[feature] = consecutive_distances(Vf, binary_mode)
        similarity_sum += similarity_matrix(Vf, binary_mode)
        density[feature] = Vf.mean(axis=1)

    adjacent_mean = adjacent_all.mean(axis=0)
    adjacent_std = adjacent_all.std(axis=0)
    similarity_mean = similarity_sum / float(num_features)
    distance_mean = 1.0 - similarity_mean
    embedding = classical_mds_from_distance(distance_mean, out_dim=2)

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
        ax.plot(x, adjacent_all[feature], color='#1f77b4', alpha=ALL_FEATURE_ALPHA, linewidth=ALL_FEATURE_LINEWIDTH)
    ax.plot(x, adjacent_mean, color='#d62728', linewidth=2.4, label='Mean')
    ax.fill_between(x, adjacent_mean - adjacent_std, adjacent_mean + adjacent_std, color='#d62728', alpha=0.18, label='Mean +/- std')
    ax.set_title(f'{run_name} | generation {generation} | individual {candidate}\nAdjacent level {metric_name} distance across features')
    ax.set_xlabel('Level l (distance between l and l+1)')
    ax.set_ylabel(distance_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'adjacent_distances.png', dpi=180)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(similarity_mean, aspect='auto', origin='lower')
    ax.set_title(f'{run_name} | generation {generation} | individual {candidate}\nMean level {similarity_label} matrix')
    ax.set_xlabel('Level')
    ax.set_ylabel('Level')
    fig.colorbar(im, ax=ax, label=similarity_label)
    fig.tight_layout()
    fig.savefig(out_dir / 'similarity_heatmap.png', dpi=180)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.plot(embedding[:, 0], embedding[:, 1], marker='o')
    for level in range(num_levels):
        if level % max(1, num_levels // 10) == 0 or level == num_levels - 1:
            ax.text(embedding[level, 0], embedding[level, 1], str(level), fontsize=8)
    ax.set_title(f'{run_name} | generation {generation} | individual {candidate}\nClassical MDS of mean level distance')
    ax.set_xlabel('MDS dimension 1')
    ax.set_ylabel('MDS dimension 2')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'mds_levels.png', dpi=180)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    im = ax.imshow(density, aspect='auto', origin='lower')
    ax.set_title(f'{run_name} | generation {generation} | individual {candidate}\nMean vector density per feature and level')
    ax.set_xlabel('Level')
    ax.set_ylabel('Feature')
    fig.colorbar(im, ax=ax, label='Mean bit value')
    fig.tight_layout()
    fig.savefig(out_dir / 'feature_level_density.png', dpi=180)
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
        f.write(f'binary_mode={binary_mode}\n')
        f.write(f'adjacent_mean_min={adjacent_mean.min():.10f}\n')
        f.write(f'adjacent_mean_max={adjacent_mean.max():.10f}\n')
        f.write(f'adjacent_mean_avg={adjacent_mean.mean():.10f}\n')

    print(f'Loaded: {cim_path}')
    if accuracy is not None and similarity is not None:
        print(f'accuracy={float(accuracy) * 100.0:.3f}% similarity={float(similarity):.6f}')
    print(f'mode={mode} num_levels={num_levels} num_features={num_features} dimension={V.shape[2]}')
    print(f'Output directory: {out_dir}')



def main():
    parser = argparse.ArgumentParser(
        description='Visualize one exported GA CiM for a specific run, generation, and individual.'
    )
    parser.add_argument('--run', default='latest', help='GA run folder name under CiMs, or "latest" (default).')
    parser.add_argument('--generation', type=int, default=0, help='Generation index to inspect (default: 0).')
    parser.add_argument('--individual', type=int, default=0, help='Individual index to inspect (default: 0).')
    parser.add_argument('--show', action='store_true', help='Show plots interactively in addition to saving them.')
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    cim_path = resolve_cim_path(run_dir, args.generation, args.individual)
    header, mode, V = load_cim(cim_path)
    plot_cim_analysis(run_dir, cim_path, header, mode, V, args.show)


if __name__ == '__main__':
    main()
