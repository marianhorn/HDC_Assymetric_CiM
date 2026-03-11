import os
import shutil
import subprocess
import sys

# Same grid as run_big_test.py, but with GA enabled.
NUM_LEVELS_LIST = list(range(21, 152, 10))
VECTOR_DIMENSIONS = [512, 1024, 2048, 4096, 8192]
VALIDATION_RATIOS = [0.3]
NGRAM_SIZES = [3]
MODEL_VARIANTS = [2]
USE_OPENMP_LIST = [1]
PRECOMPUTED_ITEM_MEMORY_LIST = [1]
USE_GENETIC_ITEM_MEMORY_LIST = [1]
DOWNSAMPLE_LIST = [1]

GA_SELECTION_MODES = [0]        # Pareto only
GA_INIT_UNIFORM_LIST = [1]      # equal initial vectors
GA_POPULATION_SIZES = [32]
GA_GENERATIONS = [100]
GA_CROSSOVER_RATES = [0.0]
GA_MUTATION_RATES = [0.7]
GA_TOURNAMENT_SIZES = [3]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
OUTPUT_PATH = os.path.join(BASE_DIR, "output.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "results_all.csv")

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


def run_cmd(cmd, cwd, ok_codes=(0,), stdout=None, stderr=None, echo=False):
    if echo:
        print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=stderr)
    if result.returncode not in ok_codes:
        raise RuntimeError(
            f"Command failed (exit={result.returncode}, cwd={cwd}): {' '.join(cmd)}"
        )
    return result.returncode


def find_model_binary():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _build_run_name(selection, init_uniform, mutation_rate, population_size, generations, crossover_rate):
    return (
        f"ga_sel{selection}"
        f"_init{init_uniform}"
        f"_mut{str(mutation_rate).replace('.', 'p')}"
        f"_pop{population_size}"
        f"_gen{generations}"
        f"_cx{str(crossover_rate).replace('.', 'p')}"
    )


def build_ga_configs():
    """Create the small cartesian product of GA settings used for this study."""
    configs = []
    for selection_mode in GA_SELECTION_MODES:
        for init_uniform in GA_INIT_UNIFORM_LIST:
            for mutation_rate in GA_MUTATION_RATES:
                for population_size in GA_POPULATION_SIZES:
                    for generations in GA_GENERATIONS:
                        for crossover_rate in GA_CROSSOVER_RATES:
                            for tournament_size in GA_TOURNAMENT_SIZES:
                                configs.append({
                                    "name": _build_run_name(
                                        selection_mode,
                                        init_uniform,
                                        mutation_rate,
                                        population_size,
                                        generations,
                                        crossover_rate,
                                    ),
                                    "GA_SELECTION_MODE": selection_mode,
                                    "GA_INIT_UNIFORM": init_uniform,
                                    "GA_DEFAULT_POPULATION_SIZE": population_size,
                                    "GA_DEFAULT_GENERATIONS": generations,
                                    "GA_DEFAULT_CROSSOVER_RATE": crossover_rate,
                                    "GA_DEFAULT_MUTATION_RATE": mutation_rate,
                                    "GA_DEFAULT_TOURNAMENT_SIZE": tournament_size,
                                })
    return configs


GA_TEST_CONFIGS = build_ga_configs()


STATIC_MAKE_DEFINES = [
    ("USE_OPENMP", USE_OPENMP_LIST),
    ("MODEL_VARIANT", MODEL_VARIANTS),
    ("PRECOMPUTED_ITEM_MEMORY", PRECOMPUTED_ITEM_MEMORY_LIST),
    ("USE_GENETIC_ITEM_MEMORY", USE_GENETIC_ITEM_MEMORY_LIST),
]


def _varying_ga_keys(configs):
    keys = [
        "GA_SELECTION_MODE",
        "GA_INIT_UNIFORM",
        "GA_DEFAULT_POPULATION_SIZE",
        "GA_DEFAULT_GENERATIONS",
        "GA_DEFAULT_CROSSOVER_RATE",
        "GA_DEFAULT_MUTATION_RATE",
        "GA_DEFAULT_TOURNAMENT_SIZE",
    ]
    varying = []
    for key in keys:
        values = {cfg[key] for cfg in configs}
        if len(values) > 1:
            varying.append(key)
    return varying


def _single_define(name, values):
    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one value for {name}, got {len(values)}."
            " Use nested loops if multiple values should be swept."
        )
    return values[0]


def run_evaluations():
    makefile_path = os.path.join(REPO_ROOT, "Makefile")
    if not os.path.exists(makefile_path):
        raise FileNotFoundError(f"Makefile not found at expected repo root: {makefile_path}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_rel = os.path.relpath(RESULTS_PATH, REPO_ROOT).replace(os.sep, "/")
    make_cmd_name = choose_make_command()
    model_path = None

    # Clear output log for this script run.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("=== GA evaluation log (auto-generated) ===\n")

    varying_ga_keys = _varying_ga_keys(GA_TEST_CONFIGS)
    print_num_levels = len(NUM_LEVELS_LIST) > 1
    print_vector_dimension = len(VECTOR_DIMENSIONS) > 1

    with open(OUTPUT_PATH, "a", encoding="utf-8") as log_file:
        for cfg in GA_TEST_CONFIGS:
            cfg_name = cfg["name"]

            for num_levels in NUM_LEVELS_LIST:
                for vector_dim in VECTOR_DIMENSIONS:
                    run_tag = f"cfg={cfg_name}_nl={num_levels}_vd={vector_dim}"
                    progress_parts = []
                    for key in varying_ga_keys:
                        progress_parts.append(f"{key}={cfg[key]}")
                    if print_num_levels:
                        progress_parts.append(f"NUM_LEVELS={num_levels}")
                    if print_vector_dimension:
                        progress_parts.append(f"VECTOR_DIMENSION={vector_dim}")
                    if progress_parts:
                        print(" ".join(progress_parts))

                    log_file.write(f"\n{run_tag}\n")
                    log_file.flush()

                    make_cmd = [
                        make_cmd_name,
                        "foot",
                        f"RESULT_CSV_PATH={results_rel}",
                    ]

                    for key, values in STATIC_MAKE_DEFINES:
                        make_cmd.append(f"{key}={_single_define(key, values)}")

                    make_cmd.extend([
                        f"VALIDATION_RATIO={_single_define('VALIDATION_RATIO', VALIDATION_RATIOS)}",
                        f"N_GRAM_SIZE={_single_define('N_GRAM_SIZE', NGRAM_SIZES)}",
                        f"DOWNSAMPLE={_single_define('DOWNSAMPLE', DOWNSAMPLE_LIST)}",
                        f"NUM_LEVELS={num_levels}",
                        f"VECTOR_DIMENSION={vector_dim}",
                        f"GA_SELECTION_MODE={cfg['GA_SELECTION_MODE']}",
                        f"GA_INIT_UNIFORM={cfg['GA_INIT_UNIFORM']}",
                        f"GA_DEFAULT_POPULATION_SIZE={cfg['GA_DEFAULT_POPULATION_SIZE']}",
                        f"GA_DEFAULT_GENERATIONS={cfg['GA_DEFAULT_GENERATIONS']}",
                        f"GA_DEFAULT_CROSSOVER_RATE={cfg['GA_DEFAULT_CROSSOVER_RATE']}",
                        f"GA_DEFAULT_MUTATION_RATE={cfg['GA_DEFAULT_MUTATION_RATE']}",
                        f"GA_DEFAULT_TOURNAMENT_SIZE={cfg['GA_DEFAULT_TOURNAMENT_SIZE']}",
                    ])

                    run_cmd(make_cmd, REPO_ROOT, stdout=log_file, stderr=log_file)

                    if model_path is None:
                        model_path = find_model_binary()
                        if not model_path:
                            raise FileNotFoundError("modelFoot binary not found after build")

                    run_cmd([model_path], REPO_ROOT, ok_codes=(0,), stdout=log_file, stderr=log_file)


if __name__ == "__main__":
    try:
        run_evaluations()
        print(f"\nDetailed results appended to: {RESULTS_PATH}")
        print(f"Execution log written to: {OUTPUT_PATH}")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
