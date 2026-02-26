import os
import shutil
import subprocess
import sys


VECTOR_DIMENSIONS = [512, 1024, 2048, 4096, 8192]
NUM_LEVELS_LIST = list(range(21, 152, 10))


def choose_make_command():
    if shutil.which("make"):
        return "make"
    if shutil.which("mingw32-make"):
        return "mingw32-make"
    raise RuntimeError("No make command found (tried make and mingw32-make).")


def find_binary(repo_root):
    candidates = [
        os.path.join(repo_root, "modelFoot"),
        os.path.join(repo_root, "modelFoot.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("modelFoot binary not found after build.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    output_log = os.path.join(script_dir, "output_mine_krischan_mem.txt")
    results_csv = os.path.join(script_dir, "results_mine_krischan_mem.csv")
    results_csv_rel = os.path.relpath(results_csv, repo_root).replace(os.sep, "/")

    prepare_script = os.path.join(script_dir, "prepare_krischan_item_mem_csv.py")
    make_cmd = choose_make_command()

    os.makedirs(script_dir, exist_ok=True)
    with open(output_log, "a", encoding="utf-8") as log_file:
        for vector_dimension in VECTOR_DIMENSIONS:
            for num_levels in NUM_LEVELS_LIST:
                print(
                    f"NUM_LEVELS={num_levels} "
                    f"VECTOR_DIMENSION={vector_dimension} "
                    "MODEL_VARIANT=1 (Krischan IM/CM)"
                )

                subprocess.run(
                    [
                        sys.executable,
                        prepare_script,
                        "--dimension",
                        str(vector_dimension),
                        "--num-levels",
                        str(num_levels),
                    ],
                    cwd=repo_root,
                    stdout=log_file,
                    stderr=log_file,
                    check=True,
                )

                build_cmd = [
                    make_cmd,
                    "foot",
                    "USE_OPENMP=1",
                    "PRECOMPUTED_ITEM_MEMORY=0",
                    "USE_GENETIC_ITEM_MEMORY=0",
                    "VALIDATION_RATIO=0",
                    "N_GRAM_SIZE=5",
                    "MODEL_VARIANT=1",
                    f"VECTOR_DIMENSION={vector_dimension}",
                    f"NUM_LEVELS={num_levels}",
                    f"RESULT_CSV_PATH={results_csv_rel}",
                ]

                log_file.write(
                    f"\nRUN: VECTOR_DIMENSION={vector_dimension} "
                    f"NUM_LEVELS={num_levels} "
                    "(temporary Krischan IM/CM in modelFoot.c)\n"
                )
                log_file.flush()

                subprocess.run(build_cmd, cwd=repo_root, stdout=log_file, stderr=log_file, check=True)
                binary = find_binary(repo_root)
                subprocess.run([binary], cwd=repo_root, stdout=log_file, stderr=log_file, check=True)

    print(f"Done. Results in: {results_csv}")
    print(f"Log in: {output_log}")


if __name__ == "__main__":
    main()
