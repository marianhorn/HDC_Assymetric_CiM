import os
import subprocess
import sys

NUM_LEVELS_LIST = [21, 41, 61, 81, 101, 121]
VECTOR_DIMENSIONS = [512, 1024, 2048, 3072, 4096]
N_GRAM_SIZES = [None]
VALIDATION_RATIOS = [None]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

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


def main():
    output_path = os.path.join(BASE_DIR, "output.txt")
    model_path = None

    for num_levels in NUM_LEVELS_LIST:
        for vector_dim in VECTOR_DIMENSIONS:
            for n_gram in N_GRAM_SIZES:
                for val_ratio in VALIDATION_RATIOS:
                    print(
                        f"NUM_LEVELS={num_levels} "
                        f"VECTOR_DIMENSION={vector_dim}"
                    )

                    make_cmd = [
                        "make",
                        "foot",
                        "USE_OPENMP=1",
                        f"NUM_LEVELS={num_levels}",
                        f"VECTOR_DIMENSION={vector_dim}",
                    ]
                    if n_gram is not None:
                        make_cmd.append(f"N_GRAM_SIZE={n_gram}")
                    if val_ratio is not None:
                        make_cmd.append(f"VALIDATION_RATIO={val_ratio}")

                    with open(output_path, "a", encoding="utf-8") as log_file:
                        log_file.write(
                            f"\nNUM_LEVELS={num_levels} VECTOR_DIMENSION={vector_dim}\n"
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
