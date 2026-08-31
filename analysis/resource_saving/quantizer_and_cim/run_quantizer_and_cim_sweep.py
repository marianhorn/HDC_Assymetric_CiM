import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
SEEDS = list(range(1, 6))

MODEL_CANDIDATES = [
    os.path.join(REPO_ROOT, "c_model", "build", "hdc_model"),
    os.path.join(REPO_ROOT, "c_model", "build", "hdc_model.exe"),
]

BINNING_MODES = {
    "quantile": 1,
    "kmeans_1d": 2,
    "decision_tree_1d": 3,
    "chimerge": 4,
}


def choose_make_command():
    if shutil.which("make"):
        return "make"
    if shutil.which("mingw32-make"):
        return "mingw32-make"
    raise RuntimeError("No make command found (tried make and mingw32-make).")


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


def build_runs():
    """Grid points present in cim_uniform but missing in quantizer_and_cim.

    Restricted to NUM_LEVELS <= 50 and VECTOR_DIMENSION <= 2000.
    These are grouped by repeated dimension patterns to keep the list explicit
    and reviewable.
    """
    patterns = [
        (
            [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            [251, 351, 451, 551, 651, 751, 851, 951],
        ),
        (
            [6, 8, 12, 16, 24, 32, 48],
            [251, 351, 401, 451, 551, 601, 651, 701, 801, 851, 901, 951],
        ),
        (
            [
                7, 9, 11, 13, 14, 17, 18, 19, 21, 22, 23, 26, 27, 28, 29,
                31, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 46, 47, 49,
            ],
            [201, 251, 301, 351, 401, 451, 501, 551, 601, 651, 701, 751, 801,
             851, 901, 951, 1000, 1500, 2000],
        ),
    ]

    runs = []
    for levels, dimensions in patterns:
        runs.extend((num_levels, vector_dimension)
                    for num_levels in levels
                    for vector_dimension in dimensions)
    return sorted(set(runs))


def build_level40_dense_dimension_runs():
    """Dense NUM_LEVELS=40 refinement over large dimensions."""
    return [(40, vector_dimension) for vector_dimension in range(1000, 10001, 100)]


def read_existing_configs(mode_name, seed):
    results_path = os.path.join(
        RUNS_DIR,
        f"binning_mode_{mode_name}",
        f"seed_{seed:02d}",
        "results.csv",
    )
    existing = set()

    if not os.path.exists(results_path):
        return existing

    with open(results_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                existing.add((int(row["num_levels"]), int(row["vector_dimension"])))
            except (KeyError, TypeError, ValueError):
                continue

    return existing


def filter_missing_runs(mode_name, seed, runs):
    existing = read_existing_configs(mode_name, seed)
    return [run for run in runs if run not in existing]


def parse_binning_modes(text):
    requested = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not requested:
        raise ValueError("At least one binning mode must be provided.")

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
            valid = ", ".join(list(BINNING_MODES.keys()) + [str(v) for v in BINNING_MODES.values()])
            raise ValueError(f"Unknown binning mode: {entry}. Valid values: {valid}")
        selected.append(matched)

    deduped = []
    seen = set()
    for mode in selected:
        if mode[0] not in seen:
            deduped.append(mode)
            seen.add(mode[0])
    return deduped


def parse_seeds(text):
    requested = [part.strip() for part in text.split(",") if part.strip()]
    if not requested:
        raise ValueError("At least one seed must be provided.")

    seeds = []
    for entry in requested:
        try:
            seed = int(entry)
        except ValueError as exc:
            raise ValueError(f"Invalid seed value: {entry}") from exc
        if seed <= 0:
            raise ValueError(f"Seed must be positive: {seed}")
        seeds.append(seed)

    deduped = []
    seen = set()
    for seed in seeds:
        if seed not in seen:
            deduped.append(seed)
            seen.add(seed)
    return deduped


def ensure_clean_seed_dir(seed_dir, skip_clean):
    os.makedirs(seed_dir, exist_ok=True)

    if skip_clean:
        return

    for name in ["output_all.txt", "run_manifest.csv", "results.csv"]:
        path = os.path.join(seed_dir, name)
        if os.path.exists(path):
            os.remove(path)


def init_manifest(manifest_path):
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_index",
                "total_runs",
                "timestamp",
                "seed",
                "item_mem_seed",
                "ga_seed",
                "binning_mode_name",
                "binning_mode_value",
                "num_levels",
                "vector_dimension",
                "use_genetic_item_memory",
                "output_mode",
                "log_file",
                "duration_sec",
            ]
        )


def ensure_manifest(manifest_path, append):
    if append and os.path.exists(manifest_path):
        return
    init_manifest(manifest_path)


def append_manifest(manifest_path, run_index, total_runs, seed, mode_name, mode_value, num_levels, vector_dimension, log_file, duration_sec):
    with open(manifest_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                run_index,
                total_runs,
                datetime.now().isoformat(),
                seed,
                seed,
                seed,
                mode_name,
                mode_value,
                num_levels,
                vector_dimension,
                1,
                2,
                log_file,
                f"{duration_sec:.3f}",
            ]
        )


def write_run_header(output_file, run_index, total_runs, seed, mode_name, mode_value, num_levels, vector_dimension):
    output_file.write(
        "\n===== "
        f"run={run_index}/{total_runs}, "
        f"ITEM_MEM_SEED={seed}, "
        f"GA_DEFAULT_SEED={seed}, "
        f"BINNING_MODE={mode_value}({mode_name}), "
        f"NUM_LEVELS={num_levels}, VECTOR_DIMENSION={vector_dimension}, "
        "USE_GENETIC_ITEM_MEMORY=1, OUTPUT_MODE=2"
        " =====\n"
    )


def write_log_header(output_file, seed, mode_name, mode_value, num_levels, vector_dimension):
    output_file.write(
        "=== quantizer-and-cim resource-saving run ===\n"
        f"timestamp={datetime.now().isoformat()}\n"
        f"ITEM_MEM_SEED={seed}\n"
        f"GA_DEFAULT_SEED={seed}\n"
        f"BINNING_MODE={mode_value} ({mode_name})\n"
        f"NUM_LEVELS={num_levels}\n"
        f"VECTOR_DIMENSION={vector_dimension}\n"
        "USE_GENETIC_ITEM_MEMORY=1\n"
        "OUTPUT_MODE=2\n\n"
    )


def run_seed_sweep(mode_name, mode_value, seed, make_cmd_name, runs, skip_clean):
    seed_dir = os.path.join(RUNS_DIR, f"binning_mode_{mode_name}", f"seed_{seed:02d}")
    combined_output_path = os.path.join(seed_dir, "output_all.txt")
    manifest_path = os.path.join(seed_dir, "run_manifest.csv")
    results_path = os.path.join(seed_dir, "results.csv")
    results_rel = os.path.relpath(results_path, REPO_ROOT).replace(os.sep, "/")

    ensure_clean_seed_dir(seed_dir, skip_clean)
    ensure_manifest(manifest_path, append=skip_clean)

    model_path = None
    total_runs = len(runs)

    print(f"\nMode {mode_name} (BINNING_MODE={mode_value})")
    print(f"Seed {seed}")
    print(f"Output folder: {seed_dir}")

    for run_index, (num_levels, vector_dimension) in enumerate(runs, start=1):
        log_name = "output_all.txt"

        print(
            f"[mode {mode_name}] "
            f"[seed {seed:02d}] "
            f"[{run_index}/{total_runs}] "
            f"NUM_LEVELS={num_levels} VECTOR_DIMENSION={vector_dimension}"
        )

        start = time.perf_counter()
        with open(combined_output_path, "a", encoding="utf-8") as output_file:
            write_run_header(output_file, run_index, total_runs, seed, mode_name, mode_value, num_levels, vector_dimension)
            write_log_header(output_file, seed, mode_name, mode_value, num_levels, vector_dimension)
            output_file.flush()
            make_cmd = [
                make_cmd_name,
                "foot",
                "USE_OPENMP=1",
                "USE_GENETIC_ITEM_MEMORY=1",
                "OUTPUT_MODE=2",
                f"BINNING_MODE={mode_value}",
                f"ITEM_MEM_SEED={seed}",
                f"GA_DEFAULT_SEED={seed}",
                f"NUM_LEVELS={num_levels}",
                f"VECTOR_DIMENSION={vector_dimension}",
                f"RESULT_CSV_PATH={results_rel}",
            ]
            run_cmd(make_cmd, REPO_ROOT, stdout=output_file, stderr=output_file)

            if model_path is None:
                model_path = find_model_binary()
            if not model_path:
                raise FileNotFoundError("C model binary not found after build.")

            run_cmd([model_path], REPO_ROOT, stdout=output_file, stderr=output_file)

        duration_sec = time.perf_counter() - start
        append_manifest(
            manifest_path,
            run_index,
            total_runs,
            seed,
            mode_name,
            mode_value,
            num_levels,
            vector_dimension,
            log_name,
            duration_sec,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run resource-saving sweeps with learned quantizers and GA item-memory optimization."
    )
    parser.add_argument(
        "--binning-modes",
        default="quantile,kmeans_1d,decision_tree_1d,chimerge",
        help="Comma-separated binning modes to run. Names: quantile,kmeans_1d,decision_tree_1d,chimerge or numeric ids 1,2,3,4.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Do not delete old output/results in existing seed folders.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in SEEDS),
        help="Comma-separated seeds to run. Each seed is used as both ITEM_MEM_SEED and GA_DEFAULT_SEED.",
    )
    parser.add_argument(
        "--level40-dense-dimensions-missing",
        action="store_true",
        help=(
            "Append only missing quantizer+GA runs for NUM_LEVELS=40 and "
            "VECTOR_DIMENSION=1000..10000 in steps of 100. Existing results "
            "are preserved automatically."
        ),
    )
    args = parser.parse_args()

    selected_modes = parse_binning_modes(args.binning_modes)
    selected_seeds = parse_seeds(args.seeds)
    if args.level40_dense_dimensions_missing:
        runs = build_level40_dense_dimension_runs()
        grid_description = "missing dense NUM_LEVELS=40 points, VECTOR_DIMENSION=1000..10000 step 100"
        append_results = True
    else:
        runs = build_runs()
        grid_description = "missing cim_uniform low-grid points (NUM_LEVELS <= 50, VECTOR_DIMENSION <= 2000)"
        append_results = args.skip_clean
    make_cmd_name = choose_make_command()

    os.makedirs(RUNS_DIR, exist_ok=True)

    print(f"Repo root: {REPO_ROOT}")
    print(f"Runs folder: {RUNS_DIR}")
    print(f"Binning modes: {selected_modes}")
    print(f"Seeds: {selected_seeds}")
    print(f"Grid: {grid_description}")
    print(f"Candidate configurations per seed/mode: {len(runs)}")

    for mode_name, mode_value in selected_modes:
        for seed in selected_seeds:
            runs_for_seed = runs
            if args.level40_dense_dimensions_missing:
                runs_for_seed = filter_missing_runs(mode_name, seed, runs)
                skipped = len(runs) - len(runs_for_seed)
                print(
                    f"\nMode {mode_name}, seed {seed:02d}: "
                    f"{len(runs_for_seed)} missing, {skipped} already present."
                )
                if not runs_for_seed:
                    continue

            run_seed_sweep(mode_name, mode_value, seed, make_cmd_name, runs_for_seed, append_results)

    print("\nFinished all quantizer-and-cim resource-saving sweeps.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
