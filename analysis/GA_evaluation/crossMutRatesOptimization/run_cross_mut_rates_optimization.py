import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


SEEDS_DEFAULT = [1, 2, 3]
RATES = [i / 10.0 for i in range(11)]  # 0.0 .. 1.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

LOG_DIR = os.path.join(BASE_DIR, "logs")
COMBINED_OUTPUT_PATH = os.path.join(BASE_DIR, "output_all.txt")
RESULTS_PATH = os.path.join(BASE_DIR, "results.csv")
MANIFEST_PATH = os.path.join(BASE_DIR, "run_manifest.csv")

MODEL_CANDIDATES = [
    os.path.join(REPO_ROOT, "modelFoot"),
    os.path.join(REPO_ROOT, "modelFoot.exe"),
]


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


def parse_int_list(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("At least one seed is required.")
    return [int(x) for x in parts]


def rate_tag(rate):
    return f"{rate:.1f}".replace(".", "p")


def reset_outputs(skip_clean):
    os.makedirs(LOG_DIR, exist_ok=True)
    if skip_clean:
        return

    for name in os.listdir(LOG_DIR):
        path = os.path.join(LOG_DIR, name)
        if os.path.isfile(path):
            os.remove(path)

    for path in [COMBINED_OUTPUT_PATH, RESULTS_PATH, MANIFEST_PATH]:
        if os.path.exists(path):
            os.remove(path)


def append_combined(log_path, seed, crossover_rate, mutation_rate):
    with open(COMBINED_OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        out_f.write(
            "\n===== "
            f"seed={seed}, cx={crossover_rate:.1f}, mut={mutation_rate:.1f}, "
            f"log={os.path.basename(log_path)}"
            " =====\n"
        )
        with open(log_path, "r", encoding="utf-8", errors="ignore") as in_f:
            shutil.copyfileobj(in_f, out_f)


def init_manifest():
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_index",
                "total_runs",
                "timestamp",
                "seed",
                "crossover_rate",
                "mutation_rate",
                "log_file",
                "duration_sec",
            ]
        )


def append_manifest(run_index, total_runs, seed, cx, mut, log_file, duration_sec):
    with open(MANIFEST_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                run_index,
                total_runs,
                datetime.now().isoformat(),
                seed,
                f"{cx:.1f}",
                f"{mut:.1f}",
                log_file,
                f"{duration_sec:.3f}",
            ]
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run GA evaluation over crossover/mutation grid (0.0..1.0 step 0.1) "
            "for multiple seeds."
        )
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(x) for x in SEEDS_DEFAULT),
        help="Comma-separated seeds (default: 1,2,3).",
    )
    parser.add_argument(
        "--use-openmp",
        type=int,
        default=1,
        choices=[0, 1],
        help="USE_OPENMP for make (default: 1).",
    )
    parser.add_argument(
        "--output-mode",
        type=int,
        default=2,
        help="OUTPUT_MODE for make (default: 2 = detailed).",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Do not remove existing logs/results before starting.",
    )
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    make_cmd_name = choose_make_command()
    reset_outputs(args.skip_clean)
    init_manifest()

    runs = []
    for seed in seeds:
        for cx in RATES:
            for mut in RATES:
                runs.append((seed, cx, mut))

    total_runs = len(runs)
    model_path = None
    results_rel = os.path.relpath(RESULTS_PATH, REPO_ROOT).replace(os.sep, "/")

    print(f"Repo root: {REPO_ROOT}")
    print(f"Total runs: {total_runs} (seeds={len(seeds)}, cx={len(RATES)}, mut={len(RATES)})")
    print(f"Logs dir: {LOG_DIR}")
    print(f"Results CSV: {RESULTS_PATH}")
    print("")

    for idx, (seed, cx, mut) in enumerate(runs, start=1):
        run_name = f"seed{seed}_cx{rate_tag(cx)}_mut{rate_tag(mut)}"
        log_filename = f"run_{run_name}.txt"
        log_path = os.path.join(LOG_DIR, log_filename)
        print(
            f"[{idx}/{total_runs}] "
            f"seed={seed}, crossover={cx:.1f}, mutation={mut:.1f}"
        )

        start = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                "=== crossMutRatesOptimization run ===\n"
                f"timestamp={datetime.now().isoformat()}\n"
                f"seed={seed}\n"
                f"crossover_rate={cx:.1f}\n"
                f"mutation_rate={mut:.1f}\n"
                f"use_openmp={args.use_openmp}\n"
                f"output_mode={args.output_mode}\n\n"
            )
            log_file.flush()

            make_cmd = [
                make_cmd_name,
                "foot",
                f"USE_OPENMP={args.use_openmp}",
                f"OUTPUT_MODE={args.output_mode}",
                f"GA_DEFAULT_SEED={seed}",
                f"GA_DEFAULT_CROSSOVER_RATE={cx:.1f}",
                f"GA_DEFAULT_MUTATION_RATE={mut:.1f}",
                f"RESULT_CSV_PATH={results_rel}",
            ]
            run_cmd(make_cmd, REPO_ROOT, stdout=log_file, stderr=log_file)

            if model_path is None:
                model_path = find_model_binary()
                if not model_path:
                    raise FileNotFoundError("modelFoot binary not found after build.")

            run_cmd([model_path], REPO_ROOT, stdout=log_file, stderr=log_file)

        elapsed = time.perf_counter() - start
        append_manifest(idx, total_runs, seed, cx, mut, log_filename, elapsed)
        append_combined(log_path, seed, cx, mut)

    print("\nFinished all runs.")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Per-run logs: {LOG_DIR}")
    print(f"Combined output: {COMBINED_OUTPUT_PATH}")
    print(f"Results CSV: {RESULTS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
