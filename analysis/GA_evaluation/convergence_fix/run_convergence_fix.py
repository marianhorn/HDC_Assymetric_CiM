import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime


DEFAULT_SEEDS = [1, 2, 3]
DEFAULT_GA_CROSSOVER_ALPHAS = [x * 0.25 for x in range(0, 9)]
DEFAULT_GA_MUTATION_BETAS = [x * 0.25 for x in range(0, 9)]
DEFAULT_GA_CROSSOVER_CHUNK_WIDTH = 0.2
DEFAULT_GA_CROSSOVER_RATE = 0.7
DEFAULT_GA_MUTATION_RATE = 0.2
DEFAULT_GA_CROSSOVER_ALPHA = 0.8
DEFAULT_GA_MUTATION_BETA = 0.8
DEFAULT_GA_CROSSOVER_RATES = [0.50 + x * 0.1 for x in range(0, 6)]
DEFAULT_GA_MUTATION_RATES = [0.00 + x * 0.1 for x in range(0, 6)]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
RUNS_DIR = os.path.join(BASE_DIR, "runs")

MODEL_CANDIDATES = [
    os.path.join(REPO_ROOT, "modelFoot"),
    os.path.join(REPO_ROOT, "modelFoot.exe"),
]

SEED_FROM_NAME_RE = re.compile(r"run_seed_(\d+)\.txt$")
DATASET_RE = re.compile(r"Model for dataset #(\d+)")
GEN_RE = re.compile(r"GA generation (\d+)/(\d+)")
NEW_SEL_RE = re.compile(r"new selected individuals:\s*(\d+)/(\d+)")

def parse_seeds(seed_text):
    parts = [s.strip() for s in seed_text.split(",") if s.strip()]
    if not parts:
        raise ValueError("At least one seed must be provided.")
    return [int(x) for x in parts]


def parse_float_list(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("At least one value must be provided.")
    return [float(x) for x in parts]


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


def ensure_clean_dir(path, skip_clean):
    os.makedirs(path, exist_ok=True)
    if skip_clean:
        return
    for name in os.listdir(path):
        child = os.path.join(path, name)
        if os.path.isdir(child):
            shutil.rmtree(child)
        else:
            os.remove(child)


def parse_seed_from_filename(path):
    name = os.path.basename(path)
    match = SEED_FROM_NAME_RE.search(name)
    if not match:
        raise ValueError(f"Unexpected log filename format: {name}")
    return int(match.group(1))


def parse_single_log(path):
    seed = parse_seed_from_filename(path)
    run = {
        "seed": seed,
        "datasets": defaultdict(
            lambda: {
                "generations": defaultdict(lambda: {"new_selected": None, "population": None}),
            }
        ),
    }

    current_dataset = None
    current_generation = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m = DATASET_RE.search(line)
            if m:
                current_dataset = int(m.group(1))
                current_generation = None
                continue

            m = GEN_RE.search(line)
            if m and current_dataset is not None:
                current_generation = int(m.group(1))
                _ = run["datasets"][current_dataset]["generations"][current_generation]
                continue

            m = NEW_SEL_RE.search(line)
            if m and current_dataset is not None and current_generation is not None:
                gen = run["datasets"][current_dataset]["generations"][current_generation]
                gen["new_selected"] = int(m.group(1))
                gen["population"] = int(m.group(2))
                continue

    return run


def collect_runs(log_dir):
    paths = []
    for name in os.listdir(log_dir):
        if SEED_FROM_NAME_RE.search(name):
            paths.append(os.path.join(log_dir, name))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No run_seed_<n>.txt logs found in {log_dir}")
    return [parse_single_log(path) for path in paths]


def build_generation_rows(runs):
    rows = []
    for run in runs:
        seed = run["seed"]
        for dataset, ds_data in run["datasets"].items():
            for generation, gen_data in sorted(ds_data["generations"].items()):
                if gen_data["new_selected"] is None:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "generation": generation,
                        "new_selected": float(gen_data["new_selected"]),
                        "population": float(gen_data["population"]) if gen_data["population"] is not None else float("nan"),
                    }
                )
    return rows


def write_generation_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "dataset", "generation", "new_selected", "population"],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["seed"], r["dataset"], r["generation"])):
            writer.writerow(row)


def format_value(value):
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def build_alpha_beta_configurations(crossover_alphas, mutation_betas, chunk_width, crossover_rate, mutation_rate):
    configs = []
    for ga_crossover_alpha in crossover_alphas:
        for ga_mutation_beta in mutation_betas:
            label = os.path.join(
                "alpha_beta",
                (
                    f"a{format_value(ga_crossover_alpha)}_"
                    f"b{format_value(ga_mutation_beta)}_"
                    f"w{format_value(chunk_width)}_"
                    f"cx{format_value(crossover_rate)}_"
                    f"mut{format_value(mutation_rate)}"
                ),
            )
            configs.append(
                {
                    "label": label,
                    "overrides": [
                        ("GA_CROSSOVER_ALPHA", format_value(ga_crossover_alpha)),
                        ("GA_MUTATION_BETA", format_value(ga_mutation_beta)),
                        ("GA_CROSSOVER_CHUNK_WIDTH", format_value(chunk_width)),
                        ("GA_DEFAULT_CROSSOVER_RATE", format_value(crossover_rate)),
                        ("GA_DEFAULT_MUTATION_RATE", format_value(mutation_rate)),
                    ],
                }
            )
    return configs


def build_rate_configurations(crossover_rates, mutation_rates, crossover_alpha, mutation_beta, chunk_width):
    configs = []
    for crossover_rate in crossover_rates:
        for mutation_rate in mutation_rates:
            label = os.path.join(
                "rates",
                (
                    f"a{format_value(crossover_alpha)}_"
                    f"b{format_value(mutation_beta)}_"
                    f"w{format_value(chunk_width)}_"
                    f"cx{format_value(crossover_rate)}_"
                    f"mut{format_value(mutation_rate)}"
                ),
            )
            configs.append(
                {
                    "label": label,
                    "overrides": [
                        ("GA_CROSSOVER_ALPHA", format_value(crossover_alpha)),
                        ("GA_MUTATION_BETA", format_value(mutation_beta)),
                        ("GA_CROSSOVER_CHUNK_WIDTH", format_value(chunk_width)),
                        ("GA_DEFAULT_CROSSOVER_RATE", format_value(crossover_rate)),
                        ("GA_DEFAULT_MUTATION_RATE", format_value(mutation_rate)),
                    ],
                }
            )
    return configs


def run_configuration(config, seeds, make_cmd_name, output_mode, skip_clean):
    label = config["label"]
    overrides = config["overrides"]

    config_dir = os.path.join(RUNS_DIR, label)
    logs_dir = os.path.join(config_dir, "logs")
    combined_output_path = os.path.join(config_dir, "output_all.txt")
    results_path = os.path.join(config_dir, "results.csv")
    metrics_csv = os.path.join(config_dir, "generation_metrics.csv")
    manifest_path = os.path.join(config_dir, "run_manifest.csv")

    os.makedirs(config_dir, exist_ok=True)
    ensure_clean_dir(logs_dir, skip_clean)
    if not skip_clean:
        plots_dir = os.path.join(config_dir, "plots")
        if os.path.isdir(plots_dir):
            shutil.rmtree(plots_dir)
        for path in [combined_output_path, results_path, metrics_csv, manifest_path]:
            if os.path.exists(path):
                os.remove(path)

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "label", "overrides"])
        for seed in seeds:
            writer.writerow([seed, label, ",".join(f"{k}={v}" for k, v in overrides)])

    print(f"\nConfiguration: {label}")

    results_rel = os.path.relpath(results_path, REPO_ROOT).replace(os.sep, "/")
    model_path = None

    for idx, seed in enumerate(seeds, start=1):
        log_path = os.path.join(logs_dir, f"run_seed_{seed}.txt")
        print(f"[{idx}/{len(seeds)}] seed={seed}")

        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                "=== convergence fix run ===\n"
                f"timestamp={datetime.now().isoformat()}\n"
                f"label={label}\n"
                f"seed={seed}\n"
                f"output_mode={output_mode}\n"
                f"overrides={','.join(f'{k}={v}' for k, v in overrides)}\n\n"
            )
            log_file.flush()

            make_cmd = [
                make_cmd_name,
                "foot",
                "USE_OPENMP=1",
                f"OUTPUT_MODE={output_mode}",
                f"GA_DEFAULT_SEED={seed}",
                f"RESULT_CSV_PATH={results_rel}",
            ]
            for key, value in overrides:
                make_cmd.append(f"{key}={value}")

            run_cmd(make_cmd, REPO_ROOT, stdout=log_file, stderr=log_file)

            if model_path is None:
                model_path = find_model_binary()
            if not model_path:
                raise FileNotFoundError("modelFoot binary not found after build.")

            run_cmd([model_path], REPO_ROOT, stdout=log_file, stderr=log_file)

        with open(combined_output_path, "a", encoding="utf-8") as out_f:
            out_f.write(f"\n===== RUN label={label} seed={seed} file={os.path.basename(log_path)} =====\n")
            with open(log_path, "r", encoding="utf-8", errors="ignore") as in_f:
                shutil.copyfileobj(in_f, out_f)

    runs = collect_runs(logs_dir)
    rows = build_generation_rows(runs)
    write_generation_csv(rows, metrics_csv)
    return metrics_csv


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed GA convergence experiments for 3 seeds and save logs/results/metrics."
        )
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated RNG seeds (default: 1,2,3).",
    )
    parser.add_argument(
        "--ga-crossover-alphas",
        default=",".join(format_value(x) for x in DEFAULT_GA_CROSSOVER_ALPHAS),
        help="Comma-separated GA_CROSSOVER_ALPHA values.",
    )
    parser.add_argument(
        "--ga-mutation-betas",
        default=",".join(format_value(x) for x in DEFAULT_GA_MUTATION_BETAS),
        help="Comma-separated GA_MUTATION_BETA values.",
    )
    parser.add_argument(
        "--fixed-ga-crossover-chunk-width",
        type=float,
        default=DEFAULT_GA_CROSSOVER_CHUNK_WIDTH,
        help="Fixed GA_CROSSOVER_CHUNK_WIDTH used in both grids.",
    )
    parser.add_argument(
        "--fixed-ga-crossover-rate",
        type=float,
        default=DEFAULT_GA_CROSSOVER_RATE,
        help="Fixed GA_DEFAULT_CROSSOVER_RATE used in the alpha/beta grid.",
    )
    parser.add_argument(
        "--fixed-ga-mutation-rate",
        type=float,
        default=DEFAULT_GA_MUTATION_RATE,
        help="Fixed GA_DEFAULT_MUTATION_RATE used in the alpha/beta grid.",
    )
    parser.add_argument(
        "--fixed-ga-crossover-alpha",
        type=float,
        default=DEFAULT_GA_CROSSOVER_ALPHA,
        help="Fixed GA_CROSSOVER_ALPHA used in the rates grid.",
    )
    parser.add_argument(
        "--fixed-ga-mutation-beta",
        type=float,
        default=DEFAULT_GA_MUTATION_BETA,
        help="Fixed GA_MUTATION_BETA used in the rates grid.",
    )
    parser.add_argument(
        "--ga-crossover-rates",
        default=",".join(format_value(x) for x in DEFAULT_GA_CROSSOVER_RATES),
        help="Comma-separated GA_DEFAULT_CROSSOVER_RATE values for the rates grid.",
    )
    parser.add_argument(
        "--ga-mutation-rates",
        default=",".join(format_value(x) for x in DEFAULT_GA_MUTATION_RATES),
        help="Comma-separated GA_DEFAULT_MUTATION_RATE values for the rates grid.",
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
        help="Do not delete old logs/results for the selected configuration folders.",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if len(seeds) != 3:
        raise ValueError("This script is intended to run exactly 3 seeds.")

    alpha_beta_configs = build_alpha_beta_configurations(
        parse_float_list(args.ga_crossover_alphas),
        parse_float_list(args.ga_mutation_betas),
        args.fixed_ga_crossover_chunk_width,
        args.fixed_ga_crossover_rate,
        args.fixed_ga_mutation_rate,
    )
    rate_configs = build_rate_configurations(
        parse_float_list(args.ga_crossover_rates),
        parse_float_list(args.ga_mutation_rates),
        args.fixed_ga_crossover_alpha,
        args.fixed_ga_mutation_beta,
        args.fixed_ga_crossover_chunk_width,
    )
    configs = alpha_beta_configs + rate_configs
    make_cmd_name = choose_make_command()
    os.makedirs(RUNS_DIR, exist_ok=True)

    print(f"Repo root: {REPO_ROOT}")
    print(f"Runs folder: {RUNS_DIR}")
    print(f"Seeds: {seeds}")
    print(f"Alpha/Beta configurations: {len(alpha_beta_configs)}")
    print(f"Rate configurations: {len(rate_configs)}")
    print(f"Total configurations: {len(configs)}")

    for config in configs:
        metrics_csv = run_configuration(
            config,
            seeds,
            make_cmd_name,
            args.output_mode,
            args.skip_clean,
        )
        print(f"Saved metrics: {metrics_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
