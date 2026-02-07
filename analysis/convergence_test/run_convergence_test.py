import os
import subprocess
import sys

VECTOR_DIMENSION = 1024
NUM_LEVELS = 61
GA_DEFAULT_GENERATIONS = 128
GA_SELECTION_MODES = [0, 1, 2]
GA_INIT_UNIFORMS = [0, 1]

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

    for selection_mode in GA_SELECTION_MODES:
        for init_uniform in GA_INIT_UNIFORMS:
            print(
                f"NUM_LEVELS={NUM_LEVELS} VECTOR_DIMENSION={VECTOR_DIMENSION} "
                f"GA_SELECTION_MODE={selection_mode} GA_INIT_UNIFORM={init_uniform} "
                f"GA_DEFAULT_GENERATIONS={GA_DEFAULT_GENERATIONS}"
            )

            make_cmd = [
                "make",
                "foot",
                "USE_OPENMP=1",
                f"NUM_LEVELS={NUM_LEVELS}",
                f"VECTOR_DIMENSION={VECTOR_DIMENSION}",
                f"GA_SELECTION_MODE={selection_mode}",
                f"GA_INIT_UNIFORM={init_uniform}",
                f"GA_DEFAULT_GENERATIONS={GA_DEFAULT_GENERATIONS}",
                f"RESULT_CSV_PATH={RESULTS_PATH_REL}",
            ]

            with open(OUTPUT_PATH, "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\nNUM_LEVELS={NUM_LEVELS} VECTOR_DIMENSION={VECTOR_DIMENSION} "
                    f"GA_SELECTION_MODE={selection_mode} GA_INIT_UNIFORM={init_uniform} "
                    f"GA_DEFAULT_GENERATIONS={GA_DEFAULT_GENERATIONS}\n"
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
