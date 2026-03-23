import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


NUM_LEVELS_VALUES = list(range(10, 101, 5))
VECTOR_DIMENSIONS = [512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
BINNING_MODES = [
    ("uniform", 0),
    ("quantile", 1),
    ("kmeans_1d", 2),
    ("decision_tree_1d", 3),
    ("chimerge", 4),
    ("ga_refined", 5),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
OUTPUT_PATH = os.path.join(BASE_DIR, "output_all.txt")
RESULTS_PATH = os.path.join(BASE_DIR, "results.csv")
MANIFEST_PATH = os.path.join(BASE_DIR, "run_manifest.csv")


def windows_to_wsl_path(path):
    normalized = os.path.abspath(path).replace("\\", "/")
    drive, rest = os.path.splitdrive(normalized)
    if not drive:
        return normalized
    return f"/mnt/{drive[0].lower()}{rest}"


def choose_backend():
    if shutil.which("make"):
        return ("native", "make")
    if shutil.which("wsl"):
        return ("wsl", "make")
    raise RuntimeError("Neither native make nor wsl is available.")


def run_cmd(cmd, cwd, stdout=None, stderr=None):
    result = subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={result.returncode}, cwd={cwd}): {' '.join(cmd)}"
        )


def run_make(backend, make_args, log_file):
    mode, make_cmd = backend
    if mode == "native":
        run_cmd([make_cmd] + make_args, REPO_ROOT, stdout=log_file, stderr=log_file)
        return

    repo_root_wsl = windows_to_wsl_path(REPO_ROOT)
    command = " ".join([make_cmd] + make_args)
    run_cmd(
        ["wsl", "bash", "-lc", f"cd '{repo_root_wsl}' && {command}"],
        REPO_ROOT,
        stdout=log_file,
        stderr=log_file,
    )


def run_model(backend, log_file):
    mode, _ = backend
    if mode == "native":
        model_path = os.path.join(REPO_ROOT, "modelFoot.exe" if os.name == "nt" else "modelFoot")
        if not os.path.exists(model_path):
            alt_path = os.path.join(REPO_ROOT, "modelFoot")
            if os.path.exists(alt_path):
                model_path = alt_path
        if not os.path.exists(model_path):
            raise FileNotFoundError("modelFoot binary not found after build.")
        run_cmd([model_path], REPO_ROOT, stdout=log_file, stderr=log_file)
        return

    repo_root_wsl = windows_to_wsl_path(REPO_ROOT)
    run_cmd(
        ["wsl", "bash", "-lc", f"cd '{repo_root_wsl}' && ./modelFoot"],
        REPO_ROOT,
        stdout=log_file,
        stderr=log_file,
    )


def reset_outputs():
    for path in [OUTPUT_PATH, RESULTS_PATH, MANIFEST_PATH]:
        if os.path.exists(path):
            os.remove(path)


def write_manifest_header():
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "run_index",
                "total_runs",
                "timestamp",
                "mode_name",
                "binning_mode",
                "num_levels",
                "vector_dimension",
                "duration_sec",
            ]
        )


def append_manifest(run_index, total_runs, mode_name, mode_value, num_levels, vector_dimension, duration_sec):
    with open(MANIFEST_PATH, "a", encoding="utf-8", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                run_index,
                total_runs,
                datetime.now().isoformat(),
                mode_name,
                mode_value,
                num_levels,
                vector_dimension,
                f"{duration_sec:.3f}",
            ]
        )


def main():
    backend = choose_backend()
    results_rel = os.path.relpath(RESULTS_PATH, REPO_ROOT).replace(os.sep, "/")
    runs = [
        (mode_name, mode_value, num_levels, vector_dimension)
        for mode_name, mode_value in BINNING_MODES
        for num_levels in NUM_LEVELS_VALUES
        for vector_dimension in VECTOR_DIMENSIONS
    ]

    reset_outputs()
    write_manifest_header()

    print(f"Repo root: {REPO_ROOT}")
    print(f"Backend: {backend[0]}")
    print(f"Total runs: {len(runs)}")
    print(f"Output log: {OUTPUT_PATH}")
    print(f"Results CSV: {RESULTS_PATH}")
    print("")

    for run_index, (mode_name, mode_value, num_levels, vector_dimension) in enumerate(runs, start=1):
        print(
            f"[{run_index}/{len(runs)}] "
            f"mode={mode_name}, BINNING_MODE={mode_value}, "
            f"NUM_LEVELS={num_levels}, VECTOR_DIMENSION={vector_dimension}"
        )

        start = time.perf_counter()
        with open(OUTPUT_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(
                "\n===== "
                f"[{run_index}/{len(runs)}] "
                f"timestamp={datetime.now().isoformat()} "
                f"mode={mode_name} "
                f"BINNING_MODE={mode_value} "
                f"NUM_LEVELS={num_levels} "
                f"VECTOR_DIMENSION={vector_dimension} "
                "USE_GENETIC_ITEM_MEMORY=0 "
                "OUTPUT_MODE=OUTPUT_DETAILED"
                " =====\n"
            )
            log_file.flush()

            make_args = [
                "foot",
                "USE_OPENMP=1",
                "USE_GENETIC_ITEM_MEMORY=0",
                "OUTPUT_MODE=2",
                f"BINNING_MODE={mode_value}",
                f"NUM_LEVELS={num_levels}",
                f"VECTOR_DIMENSION={vector_dimension}",
                f"RESULT_CSV_PATH={results_rel}",
            ]
            run_make(backend, make_args, log_file)
            run_model(backend, log_file)

        duration_sec = time.perf_counter() - start
        append_manifest(
            run_index,
            len(runs),
            mode_name,
            mode_value,
            num_levels,
            vector_dimension,
            duration_sec,
        )

    print("\nFinished all runs.")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Output log: {OUTPUT_PATH}")
    print(f"Results CSV: {RESULTS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
