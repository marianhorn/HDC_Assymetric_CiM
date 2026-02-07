import os
import subprocess
import sys

VECTOR_DIMENSIONS = [256, 512, 1024, 3072, 4096, 5120, 6144, 7168, 8192]
NUM_LEVELS = 61

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
OUTPUT_PATH = os.path.join(BASE_DIR, "output.txt")
RESULTS_PATH = os.path.join(BASE_DIR, "results.csv")
RESULTS_PATH_REL = os.path.relpath(RESULTS_PATH, REPO_ROOT)

MODEL_CANDIDATES = [
    os.path.join(REPO_ROOT, "modelFoot"),
    os.path.join(REPO_ROOT, "modelFoot.exe"),
]


def run_cmd(cmd, cwd, ok_codes=(0,), stdout=None, stderr=None):
    result = subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=stderr)
    if result.returncode not in ok_codes:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.returncode


def find_model_binary():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def main():
    model_path = None

    for vector_dim in VECTOR_DIMENSIONS:
        print(f"NUM_LEVELS={NUM_LEVELS} VECTOR_DIMENSION={vector_dim}")

        make_cmd = [
            "make",
            "foot",
            "USE_OPENMP=1",
            f"NUM_LEVELS={NUM_LEVELS}",
            f"VECTOR_DIMENSION={vector_dim}",
            f"RESULT_CSV_PATH={RESULTS_PATH_REL}",
        ]

        with open(OUTPUT_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"\nNUM_LEVELS={NUM_LEVELS} VECTOR_DIMENSION={vector_dim}\n"
            )
            log_file.flush()

            run_cmd(make_cmd, REPO_ROOT, stdout=log_file, stderr=log_file)

            if model_path is None:
                model_path = find_model_binary()
            if not model_path:
                raise FileNotFoundError("modelFoot binary not found after build")

            rc = run_cmd([model_path], REPO_ROOT, ok_codes=(0,), stdout=log_file, stderr=log_file)
            if rc != 0:
                log_file.write(f"Model exited with code {rc}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
