import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
RUNS_DIR = os.path.join(BASE_DIR, 'runs')
SEEDS = list(range(1, 11))

MODEL_CANDIDATES = [
    os.path.join(REPO_ROOT, 'modelFoot'),
    os.path.join(REPO_ROOT, 'modelFoot.exe'),
]

BINNING_MODES = {
    'quantile': 1,
    'kmeans_1d': 2,
    'decision_tree_1d': 3,
    'chimerge': 4,
}


def choose_make_command():
    if shutil.which('make'):
        return 'make'
    if shutil.which('mingw32-make'):
        return 'mingw32-make'
    raise RuntimeError('No make command found (tried make and mingw32-make).')


def run_cmd(cmd, cwd, stdout=None, stderr=None):
    result = subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={result.returncode}, cwd={cwd}): {' '.join(cmd)}"
        )


def find_model_binary():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def build_vector_dimensions():
    dims = []
    dims.extend(range(201, 1000, 50))
    dims.append(1000)
    dims.extend(range(1500, 10001, 500))
    dims.extend(range(11000, 20001, 1000))
    return sorted(set(dims))


def build_num_levels():
    levels = list(range(5, 101))
    levels.extend(range(105, 201, 5))
    return levels


def ensure_clean_seed_dir(seed_dir, skip_clean):
    os.makedirs(seed_dir, exist_ok=True)
    logs_dir = os.path.join(seed_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    if skip_clean:
        return

    for name in os.listdir(logs_dir):
        path = os.path.join(logs_dir, name)
        if os.path.isfile(path):
            os.remove(path)

    for name in ['output_all.txt', 'run_manifest.csv', 'results.csv']:
        path = os.path.join(seed_dir, name)
        if os.path.exists(path):
            os.remove(path)


def init_manifest(manifest_path):
    with open(manifest_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'run_index',
                'total_runs',
                'timestamp',
                'item_mem_seed',
                'binning_mode_name',
                'binning_mode_value',
                'num_levels',
                'vector_dimension',
                'use_genetic_item_memory',
                'output_mode',
                'log_file',
                'duration_sec',
            ]
        )


def append_manifest(manifest_path, run_index, total_runs, item_mem_seed, mode_name, mode_value, num_levels, vector_dimension, log_file, duration_sec):
    with open(manifest_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                run_index,
                total_runs,
                datetime.now().isoformat(),
                item_mem_seed,
                mode_name,
                mode_value,
                num_levels,
                vector_dimension,
                0,
                2,
                log_file,
                f'{duration_sec:.3f}',
            ]
        )


def append_combined(combined_output_path, log_path, run_index, total_runs, item_mem_seed, mode_name, mode_value, num_levels, vector_dimension):
    with open(combined_output_path, 'a', encoding='utf-8') as out_f:
        out_f.write(
            '\n===== '
            f'run={run_index}/{total_runs}, '
            f'ITEM_MEM_SEED={item_mem_seed}, '
            f'BINNING_MODE={mode_value}({mode_name}), '
            f'NUM_LEVELS={num_levels}, VECTOR_DIMENSION={vector_dimension}, '
            'USE_GENETIC_ITEM_MEMORY=0, OUTPUT_MODE=2'
            ' =====\n'
        )
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as in_f:
            shutil.copyfileobj(in_f, out_f)


def parse_binning_modes(text):
    requested = [part.strip().lower() for part in text.split(',') if part.strip()]
    if not requested:
        raise ValueError('At least one binning mode must be provided.')

    selected = []
    for entry in requested:
        if entry in BINNING_MODES:
            selected.append((entry, BINNING_MODES[entry]))
            continue
        matched = None
        for name, value in BINNING_MODES.items():
            if entry == str(value):
                matched = (name, value)
                break
        if matched is None:
            valid = ', '.join(list(BINNING_MODES.keys()) + [str(v) for v in BINNING_MODES.values()])
            raise ValueError(f'Unknown binning mode: {entry}. Valid values: {valid}')
        selected.append(matched)

    deduped = []
    seen = set()
    for mode in selected:
        if mode[0] not in seen:
            deduped.append(mode)
            seen.add(mode[0])
    return deduped


def run_seed_sweep(mode_name, mode_value, item_mem_seed, make_cmd_name, num_levels_values, vector_dimensions, skip_clean):
    seed_dir = os.path.join(RUNS_DIR, f'binning_mode_{mode_name}', f'seed_{item_mem_seed:02d}')
    logs_dir = os.path.join(seed_dir, 'logs')
    combined_output_path = os.path.join(seed_dir, 'output_all.txt')
    manifest_path = os.path.join(seed_dir, 'run_manifest.csv')
    results_path = os.path.join(seed_dir, 'results.csv')
    results_rel = os.path.relpath(results_path, REPO_ROOT).replace(os.sep, '/')

    ensure_clean_seed_dir(seed_dir, skip_clean)
    init_manifest(manifest_path)

    model_path = None
    runs = [(num_levels, vector_dimension) for num_levels in num_levels_values for vector_dimension in vector_dimensions]
    total_runs = len(runs)

    print(f'\nMode {mode_name} (BINNING_MODE={mode_value})')
    print(f'Seed {item_mem_seed}')
    print(f'Output folder: {seed_dir}')

    for run_index, (num_levels, vector_dimension) in enumerate(runs, start=1):
        log_name = f'run_levels_{num_levels:03d}_dim_{vector_dimension:05d}.txt'
        log_path = os.path.join(logs_dir, log_name)

        print(
            f'[mode {mode_name}] '
            f'[seed {item_mem_seed:02d}] '
            f'[{run_index}/{total_runs}] '
            f'NUM_LEVELS={num_levels} VECTOR_DIMENSION={vector_dimension}'
        )

        start = time.perf_counter()
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(
                '=== quantizer-only resource-saving run ===\n'
                f'timestamp={datetime.now().isoformat()}\n'
                f'ITEM_MEM_SEED={item_mem_seed}\n'
                f'BINNING_MODE={mode_value} ({mode_name})\n'
                f'NUM_LEVELS={num_levels}\n'
                f'VECTOR_DIMENSION={vector_dimension}\n'
                'USE_GENETIC_ITEM_MEMORY=0\n'
                'OUTPUT_MODE=2\n\n'
            )
            log_file.flush()

            make_cmd = [
                make_cmd_name,
                'foot',
                'USE_OPENMP=1',
                'USE_GENETIC_ITEM_MEMORY=0',
                'OUTPUT_MODE=2',
                f'BINNING_MODE={mode_value}',
                f'ITEM_MEM_SEED={item_mem_seed}',
                f'NUM_LEVELS={num_levels}',
                f'VECTOR_DIMENSION={vector_dimension}',
                f'RESULT_CSV_PATH={results_rel}',
            ]
            run_cmd(make_cmd, REPO_ROOT, stdout=log_file, stderr=log_file)

            if model_path is None:
                model_path = find_model_binary()
            if not model_path:
                raise FileNotFoundError('modelFoot binary not found after build.')

            run_cmd([model_path], REPO_ROOT, stdout=log_file, stderr=log_file)

        duration_sec = time.perf_counter() - start
        append_manifest(
            manifest_path,
            run_index,
            total_runs,
            item_mem_seed,
            mode_name,
            mode_value,
            num_levels,
            vector_dimension,
            log_name,
            duration_sec,
        )
        append_combined(
            combined_output_path,
            log_path,
            run_index,
            total_runs,
            item_mem_seed,
            mode_name,
            mode_value,
            num_levels,
            vector_dimension,
        )


def main():
    parser = argparse.ArgumentParser(
        description='Run quantizer-only resource-saving sweeps for selected learned binning modes.'
    )
    parser.add_argument(
        '--binning-modes',
        default='quantile,kmeans_1d,decision_tree_1d,chimerge',
        help='Comma-separated binning modes to run. Names: quantile,kmeans_1d,decision_tree_1d,chimerge or numeric ids 1,2,3,4.',
    )
    parser.add_argument(
        '--skip-clean',
        action='store_true',
        help='Do not delete old logs/results in existing seed folders.',
    )
    args = parser.parse_args()

    selected_modes = parse_binning_modes(args.binning_modes)
    num_levels_values = build_num_levels()
    vector_dimensions = build_vector_dimensions()
    make_cmd_name = choose_make_command()

    os.makedirs(RUNS_DIR, exist_ok=True)

    print(f'Repo root: {REPO_ROOT}')
    print(f'Runs folder: {RUNS_DIR}')
    print(f'Binning modes: {selected_modes}')
    print(f'Seeds: {SEEDS}')
    print(f'Configurations per seed: {len(num_levels_values) * len(vector_dimensions)}')

    for mode_name, mode_value in selected_modes:
        for item_mem_seed in SEEDS:
            run_seed_sweep(mode_name, mode_value, item_mem_seed, make_cmd_name, num_levels_values, vector_dimensions, args.skip_clean)

    print('\nFinished all quantizer-only resource-saving sweeps.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Error: {exc}')
        sys.exit(1)
