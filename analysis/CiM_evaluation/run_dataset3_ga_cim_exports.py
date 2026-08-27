import argparse
import os
import shutil
import subprocess
import sys
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "ga_cim_exports")
MODEL_CANDIDATES = [
    os.path.join(REPO_ROOT, "modelFoot"),
    os.path.join(REPO_ROOT, "modelFoot.exe"),
]


def choose_make_command():
    if shutil.which("make"):
        return "make"
    if shutil.which("mingw32-make"):
        return "mingw32-make"
    raise RuntimeError("No make command found.")


def run_cmd(cmd, cwd, stdout):
    result = subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit {result.returncode}: {' '.join(cmd)}")


def find_model_binary():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def parse_int_list(text, name):
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError(f"At least one {name} is required.")
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Run GA-CIM exports for selected datasets."
    )
    parser.add_argument("--datasets", default="3", help="Comma-separated dataset indices.")
    parser.add_argument("--seeds", default="1", help="Comma-separated seeds.")
    parser.add_argument("--num-levels", type=int, default=40)
    parser.add_argument("--vector-dimension", type=int, default=10000)
    parser.add_argument("--population-size", type=int, default=128)
    parser.add_argument("--generations", type=int, default=64)
    args = parser.parse_args()

    datasets = parse_int_list(args.datasets, "dataset")
    seeds = parse_int_list(args.seeds, "seed")
    make_cmd_name = choose_make_command()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset in datasets:
        for seed in seeds:
            label = (
                f"dataset{dataset:02d}_seed{seed:02d}_"
                f"l{args.num_levels}_d{args.vector_dimension}_"
                f"pop{args.population_size}_gen{args.generations}"
            )
            log_path = os.path.join(OUTPUT_DIR, f"output_{label}.txt")
            results_path = os.path.join(OUTPUT_DIR, f"results_{label}.csv")
            results_rel = os.path.relpath(results_path, REPO_ROOT).replace(os.sep, "/")

            print(f"dataset {dataset}, seed {seed}: writing {log_path}")
            start = time.perf_counter()
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write("=== GA-CIM export run ===\n")
                log_file.write(f"dataset={dataset}\n")
                log_file.write(f"seed={seed}\n")
                log_file.write(f"NUM_LEVELS={args.num_levels}\n")
                log_file.write(f"VECTOR_DIMENSION={args.vector_dimension}\n")
                log_file.write(f"GA_DEFAULT_POPULATION_SIZE={args.population_size}\n")
                log_file.write(f"GA_DEFAULT_GENERATIONS={args.generations}\n\n")
                log_file.flush()

                make_cmd = [
                    make_cmd_name,
                    "foot",
                    "USE_OPENMP=1",
                    "USE_GENETIC_ITEM_MEMORY=1",
                    "GA_CIM_EXPORT_ENABLED=1",
                    "BINNING_MODE=0",
                    "OUTPUT_MODE=2",
                    f"ITEM_MEM_SEED={seed}",
                    f"GA_DEFAULT_SEED={seed}",
                    f"NUM_LEVELS={args.num_levels}",
                    f"VECTOR_DIMENSION={args.vector_dimension}",
                    f"GA_DEFAULT_POPULATION_SIZE={args.population_size}",
                    f"GA_DEFAULT_GENERATIONS={args.generations}",
                    f"DATASET_START={dataset}",
                    f"DATASET_END={dataset}",
                    f"GA_CIM_EXPORT_LABEL={label}",
                    f"RESULT_CSV_PATH={results_rel}",
                ]
                run_cmd(make_cmd, REPO_ROOT, stdout=log_file)

                model_path = find_model_binary()
                if not model_path:
                    raise FileNotFoundError("modelFoot binary not found after build.")
                run_cmd([model_path], REPO_ROOT, stdout=log_file)

            elapsed = time.perf_counter() - start
            print(f"dataset {dataset}, seed {seed}: done in {elapsed / 60.0:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
