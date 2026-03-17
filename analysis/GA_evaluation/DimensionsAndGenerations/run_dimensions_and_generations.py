import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


NUM_LEVELS_FIXED = 100
VECTOR_DIMENSIONS_DEFAULT = list(range(1024, 8193, 1024))  # 1024 .. 8192
GENERATIONS_DEFAULT = [64, 128, 256, 512]

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


def parse_generations(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("At least one generation count must be provided.")
    gens = [int(x) for x in parts]
    if any(g <= 0 for g in gens):
        raise ValueError("All generation counts must be positive integers.")
    return gens


def build_vector_dimensions(start, stop, step):
    if step <= 0:
        raise ValueError("--vec-step must be > 0")
    if stop < start:
        raise ValueError("--vec-stop must be >= --vec-start")
    return list(range(start, stop + 1, step))


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


def init_manifest():
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_index",
                "total_runs",
                "timestamp",
                "num_levels",
                "vector_dimension",
                "ga_generations",
                "use_openmp",
                "output_mode",
                "log_file",
                "duration_sec",
            ]
        )


def append_manifest(
    run_index,
    total_runs,
    num_levels,
    vector_dimension,
    ga_generations,
    use_openmp,
    output_mode,
    log_file,
    duration_sec,
):
    with open(MANIFEST_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                run_index,
                total_runs,
                datetime.now().isoformat(),
                num_levels,
                vector_dimension,
                ga_generations,
                use_openmp,
                output_mode,
                log_file,
                f"{duration_sec:.3f}",
            ]
        )


def append_combined(log_path, run_index, total_runs, num_levels, vector_dimension, ga_generations):
    with open(COMBINED_OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        out_f.write(
            "\n===== "
            f"run={run_index}/{total_runs}, "
            f"nl={num_levels}, vd={vector_dimension}, gen={ga_generations}, "
            f"log={os.path.basename(log_path)}"
            " =====\n"
        )
        with open(log_path, "r", encoding="utf-8", errors="ignore") as in_f:
            shutil.copyfileobj(in_f, out_f)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run GA evaluation over vector dimensions and generation counts. "
            "Keeps all other model settings unchanged."
        )
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=NUM_LEVELS_FIXED,
        help=f"NUM_LEVELS value (default: {NUM_LEVELS_FIXED}).",
    )
    parser.add_argument(
        "--vec-start",
        type=int,
        default=VECTOR_DIMENSIONS_DEFAULT[0],
        help=f"Start VECTOR_DIMENSION (default: {VECTOR_DIMENSIONS_DEFAULT[0]}).",
    )
    parser.add_argument(
        "--vec-stop",
        type=int,
        default=VECTOR_DIMENSIONS_DEFAULT[-1],
        help=f"Stop VECTOR_DIMENSION inclusive (default: {VECTOR_DIMENSIONS_DEFAULT[-1]}).",
    )
    parser.add_argument(
        "--vec-step",
        type=int,
        default=1024,
        help="Step for VECTOR_DIMENSION sweep (default: 1024).",
    )
    parser.add_argument(
        "--generations",
        default=",".join(str(x) for x in GENERATIONS_DEFAULT),
        help="Comma-separated GA generations list (default: 64,128,256,512).",
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
        help="Do not remove old logs/results before starting.",
    )
    args = parser.parse_args()

    if args.num_levels <= 1:
        raise ValueError("--num-levels must be > 1")

    vector_dimensions = build_vector_dimensions(args.vec_start, args.vec_stop, args.vec_step)
    generations_list = parse_generations(args.generations)
    make_cmd_name = choose_make_command()
    reset_outputs(args.skip_clean)
    init_manifest()

    results_rel = os.path.relpath(RESULTS_PATH, REPO_ROOT).replace(os.sep, "/")
    model_path = None

    runs = []
    for vector_dim in vector_dimensions:
        for ga_generations in generations_list:
            runs.append((vector_dim, ga_generations))

    total_runs = len(runs)

    print(f"Repo root: {REPO_ROOT}")
    print(f"Total runs: {total_runs}")
    print(f"NUM_LEVELS: {args.num_levels}")
    print(f"VECTOR_DIMENSION sweep: {vector_dimensions}")
    print(f"GA generations sweep: {generations_list}")
    print(f"Logs dir: {LOG_DIR}")
    print(f"Results CSV: {RESULTS_PATH}")
    print("")

    for idx, (vector_dim, ga_generations) in enumerate(runs, start=1):
        run_name = f"nl{args.num_levels}_vd{vector_dim}_gen{ga_generations}"
        log_file_name = f"run_{run_name}.txt"
        log_path = os.path.join(LOG_DIR, log_file_name)
        print(
            f"[{idx}/{total_runs}] "
            f"NUM_LEVELS={args.num_levels}, VECTOR_DIMENSION={vector_dim}, GA_GENERATIONS={ga_generations}"
        )

        start = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                "=== DimensionsAndGenerations run ===\n"
                f"timestamp={datetime.now().isoformat()}\n"
                f"num_levels={args.num_levels}\n"
                f"vector_dimension={vector_dim}\n"
                f"ga_generations={ga_generations}\n"
                f"use_openmp={args.use_openmp}\n"
                f"output_mode={args.output_mode}\n\n"
            )
            log_file.flush()

            make_cmd = [
                make_cmd_name,
                "foot",
                f"USE_OPENMP={args.use_openmp}",
                f"OUTPUT_MODE={args.output_mode}",
                f"NUM_LEVELS={args.num_levels}",
                f"VECTOR_DIMENSION={vector_dim}",
                f"GA_DEFAULT_GENERATIONS={ga_generations}",
                f"RESULT_CSV_PATH={results_rel}",
            ]
            run_cmd(make_cmd, REPO_ROOT, stdout=log_file, stderr=log_file)

            if model_path is None:
                model_path = find_model_binary()
                if not model_path:
                    raise FileNotFoundError("modelFoot binary not found after build.")

            run_cmd([model_path], REPO_ROOT, stdout=log_file, stderr=log_file)

        elapsed = time.perf_counter() - start
        append_manifest(
            run_index=idx,
            total_runs=total_runs,
            num_levels=args.num_levels,
            vector_dimension=vector_dim,
            ga_generations=ga_generations,
            use_openmp=args.use_openmp,
            output_mode=args.output_mode,
            log_file=log_file_name,
            duration_sec=elapsed,
        )
        append_combined(
            log_path=log_path,
            run_index=idx,
            total_runs=total_runs,
            num_levels=args.num_levels,
            vector_dimension=vector_dim,
            ga_generations=ga_generations,
        )

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
