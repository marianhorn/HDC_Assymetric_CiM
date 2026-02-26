import os
import shutil
import subprocess
import sys

NUM_LEVELS_LIST = list(range(21, 152, 10))
VECTOR_DIMENSIONS = [512, 1024, 2048, 4096, 8192]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_PATH = os.path.join(BASE_DIR, "big_test", "big_run_output_model.txt")
RESULTS_PATH = os.path.join(BASE_DIR, "big_test", "big_run_results.csv")
RESULTS_PATH_REL = os.path.relpath(RESULTS_PATH, REPO_ROOT).replace(os.sep, "/")

MODEL_CANDIDATES = [
    os.path.join(REPO_ROOT, "modelFoot"),
    os.path.join(REPO_ROOT, "modelFoot.exe"),
]


def run_cmd(cmd, cwd, ok_codes=(0,), stdout=None, stderr=None, echo=False):
    if echo:
        print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=stderr)
    if result.returncode not in ok_codes:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.returncode


def find_model_binary():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def choose_make_command():
    if shutil.which("make"):
        return "make"
    if shutil.which("mingw32-make"):
        return "mingw32-make"
    raise RuntimeError("No make command found (tried make and mingw32-make).")


def main():
    model_path = None
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    make_cmd_name = choose_make_command()

    for vector_dim in VECTOR_DIMENSIONS:
        for num_levels in NUM_LEVELS_LIST:
            print(
                f"NUM_LEVELS={num_levels} "
                f"VECTOR_DIMENSION={vector_dim} "
                "MODEL_VARIANT=2"
            )

            make_cmd = [
                make_cmd_name,
                "foot",
                "USE_OPENMP=1",
                "MODEL_VARIANT=2",
                "PRECOMPUTED_ITEM_MEMORY=0",
                "USE_GENETIC_ITEM_MEMORY=0",
                "VALIDATION_RATIO=0",
                "N_GRAM_SIZE=5",
                f"NUM_LEVELS={num_levels}",
                f"VECTOR_DIMENSION={vector_dim}",
                f"RESULT_CSV_PATH={RESULTS_PATH_REL}",
            ]

            with open(OUTPUT_PATH, "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\nNUM_LEVELS={num_levels} VECTOR_DIMENSION={vector_dim} MODEL_VARIANT=2\n"
                )
                log_file.flush()

                run_cmd(make_cmd, REPO_ROOT, stdout=log_file, stderr=log_file, echo=False)

                if model_path is None:
                    model_path = find_model_binary()
                if not model_path:
                    raise FileNotFoundError("modelFoot binary not found after build")

                rc = run_cmd([model_path], REPO_ROOT, ok_codes=(0,), stdout=log_file, stderr=log_file, echo=False)
                if rc != 0:
                    log_file.write(f"Model exited with code {rc}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
