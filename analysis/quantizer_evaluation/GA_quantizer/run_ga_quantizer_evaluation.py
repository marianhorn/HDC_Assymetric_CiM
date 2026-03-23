import os
import shutil
import subprocess
import sys
from datetime import datetime


LEVELS_FOR_BASE_DIMS = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 140]
BASE_VECTOR_DIMENSIONS = [512, 1024, 2048]
EXTRA_NUM_LEVELS = 20
EXTRA_VECTOR_DIMENSIONS = list(range(1024, 8193, 1024))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
OUTPUT_PATH = os.path.join(BASE_DIR, "output_all.txt")
RESULTS_PATH = os.path.join(BASE_DIR, "results.csv")

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


def run_cmd(cmd, cwd, log_file):
    result = subprocess.run(cmd, cwd=cwd, stdout=log_file, stderr=log_file)
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
    runs = []
    seen = set()

    for num_levels in LEVELS_FOR_BASE_DIMS:
        for vector_dimension in BASE_VECTOR_DIMENSIONS:
            key = (num_levels, vector_dimension)
            if key not in seen:
                seen.add(key)
                runs.append(key)

    for vector_dimension in EXTRA_VECTOR_DIMENSIONS:
        key = (EXTRA_NUM_LEVELS, vector_dimension)
        if key not in seen:
            seen.add(key)
            runs.append(key)

    return runs


def reset_outputs():
    for path in [OUTPUT_PATH, RESULTS_PATH]:
        if os.path.exists(path):
            os.remove(path)


def main():
    make_cmd_name = choose_make_command()
    runs = build_runs()
    results_rel = os.path.relpath(RESULTS_PATH, REPO_ROOT).replace(os.sep, "/")
    model_path = None

    reset_outputs()

    print(f"Repo root: {REPO_ROOT}")
    print(f"Total runs: {len(runs)}")
    print(f"Output log: {OUTPUT_PATH}")
    print(f"Results CSV: {RESULTS_PATH}")
    print("")

    for idx, (num_levels, vector_dimension) in enumerate(runs, start=1):
        print(
            f"[{idx}/{len(runs)}] "
            f"NUM_LEVELS={num_levels}, VECTOR_DIMENSION={vector_dimension}"
        )

        with open(OUTPUT_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(
                "\n===== "
                f"[{idx}/{len(runs)}] "
                f"timestamp={datetime.now().isoformat()} "
                f"NUM_LEVELS={num_levels} "
                f"VECTOR_DIMENSION={vector_dimension} "
                "BINNING_MODE=GA_REFINED_BINNING "
                "USE_GENETIC_ITEM_MEMORY=1 "
                "OUTPUT_MODE=OUTPUT_DETAILED"
                " =====\n"
            )
            log_file.flush()

            make_cmd = [
                make_cmd_name,
                "foot",
                "USE_OPENMP=1",
                "OUTPUT_MODE=2",
                "USE_GENETIC_ITEM_MEMORY=1",
                "BINNING_MODE=5",
                f"NUM_LEVELS={num_levels}",
                f"VECTOR_DIMENSION={vector_dimension}",
                f"RESULT_CSV_PATH={results_rel}",
            ]
            run_cmd(make_cmd, REPO_ROOT, log_file)

            if model_path is None:
                model_path = find_model_binary()
                if not model_path:
                    raise FileNotFoundError("modelFoot binary not found after build.")

            run_cmd([model_path], REPO_ROOT, log_file)

    print("\nFinished all runs.")
    print(f"Output log: {OUTPUT_PATH}")
    print(f"Results CSV: {RESULTS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
