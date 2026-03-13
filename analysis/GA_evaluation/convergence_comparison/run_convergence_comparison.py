import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime


DEFAULT_SEEDS = [1, 2, 3, 4, 5]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
LOG_DIR = os.path.join(BASE_DIR, "logs")
COMBINED_OUTPUT_PATH = os.path.join(BASE_DIR, "output_all.txt")
RESULTS_PATH = os.path.join(BASE_DIR, "results.csv")

MODEL_CANDIDATES = [
    os.path.join(REPO_ROOT, "modelFoot"),
    os.path.join(REPO_ROOT, "modelFoot.exe"),
]


def parse_seeds(seed_text):
    parts = [s.strip() for s in seed_text.split(",") if s.strip()]
    if not parts:
        raise ValueError("At least one seed must be provided.")
    return [int(x) for x in parts]


def choose_make_command():
    if shutil.which("make"):
        return "make"
    if shutil.which("mingw32-make"):
        return "mingw32-make"
    raise RuntimeError("No make command found (tried make and mingw32-make).")


def run_cmd(cmd, cwd, stdout=None, stderr=None):
    result = subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit={result.returncode}): {' '.join(cmd)}")


def find_model_binary():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def reset_outputs(skip_clean):
    os.makedirs(LOG_DIR, exist_ok=True)
    if not skip_clean:
        for name in os.listdir(LOG_DIR):
            path = os.path.join(LOG_DIR, name)
            if os.path.isfile(path):
                os.remove(path)
        if os.path.exists(COMBINED_OUTPUT_PATH):
            os.remove(COMBINED_OUTPUT_PATH)
        if os.path.exists(RESULTS_PATH):
            os.remove(RESULTS_PATH)


def append_to_combined(log_path, seed):
    with open(COMBINED_OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        out_f.write(f"\n===== RUN seed={seed} file={os.path.basename(log_path)} =====\n")
        with open(log_path, "r", encoding="utf-8", errors="ignore") as in_f:
            shutil.copyfileobj(in_f, out_f)


def main():
    parser = argparse.ArgumentParser(
        description="Compile and run GA convergence comparison for multiple RNG seeds."
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated RNG seeds (default: 1,2,3,4,5).",
    )
    parser.add_argument(
        "--use-openmp",
        type=int,
        default=1,
        choices=[0, 1],
        help="Set USE_OPENMP for make (default: 1).",
    )
    parser.add_argument(
        "--output-mode",
        type=int,
        default=2,
        help="Set OUTPUT_MODE for make (default: 2 = detailed).",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Do not delete old logs/results before running.",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    make_cmd_name = choose_make_command()
    reset_outputs(args.skip_clean)

    results_rel = os.path.relpath(RESULTS_PATH, REPO_ROOT).replace(os.sep, "/")
    model_path = None

    print(f"Repo root: {REPO_ROOT}")
    print(f"Seeds: {seeds}")
    print(f"Logs: {LOG_DIR}")
    print("")

    for idx, seed in enumerate(seeds, start=1):
        log_path = os.path.join(LOG_DIR, f"run_seed_{seed}.txt")
        print(f"[{idx}/{len(seeds)}] Running seed={seed}")

        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                "=== GA convergence comparison run ===\n"
                f"timestamp={datetime.now().isoformat()}\n"
                f"seed={seed}\n"
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
                f"RESULT_CSV_PATH={results_rel}",
            ]
            run_cmd(make_cmd, REPO_ROOT, stdout=log_file, stderr=log_file)

            if model_path is None:
                model_path = find_model_binary()
            if not model_path:
                raise FileNotFoundError("modelFoot binary not found after build.")

            run_cmd([model_path], REPO_ROOT, stdout=log_file, stderr=log_file)

        append_to_combined(log_path, seed)

    print("\nFinished.")
    print(f"Per-run logs: {LOG_DIR}")
    print(f"Combined output: {COMBINED_OUTPUT_PATH}")
    print(f"Results CSV (from model): {RESULTS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
