import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


NUM_FEATURES = 32
WINDOW = 5
DOWNSAMPLE = 1
VALIDATION_RATIO = 0.0
DEFAULT_LEVELS = " ".join(str(level) for level in range(21, 152, 10))

CSV_HEADER = [
    "num_levels",
    "num_features",
    "vector_dimension",
    "bipolar_mode",
    "precomputed_item_memory",
    "use_genetic_item_memory",
    "ga_selection_mode",
    "n_gram_size",
    "window",
    "downsample",
    "validation_ratio",
    "overall_accuracy",
    "class_average_accuracy",
    "class_vector_similarity",
    "correct",
    "not_correct",
    "transition_error",
    "total",
    "info",
]


def parse_int_list(raw: str):
    tokens = re.split(r"[,\s]+", raw.strip())
    return [int(token) for token in tokens if token]


def choose_make_command():
    if shutil.which("mingw32-make"):
        return "mingw32-make"
    if shutil.which("make"):
        return "make"
    raise RuntimeError("No make command found (tried mingw32-make and make).")


def find_binary(root: Path):
    for name in ("hdc_model.exe", "hdc_model"):
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError("Could not find hdc_model binary after build.")


def ensure_csv_header(path: Path):
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(CSV_HEADER)


def append_result(path: Path, row):
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(row)

def regenerate_memory(script_dir: Path, project_root: Path, dimension: int, num_levels: int):
    randomvector_script = script_dir / "randomvector.py"
    bitflip_script = script_dir / "bitflipvector.py"
    python_bin = sys.executable

    subprocess.run(
        [python_bin, str(randomvector_script), str(dimension), str(NUM_FEATURES)],
        cwd=project_root,
        check=True,
    )
    subprocess.run(
        [python_bin, str(bitflip_script), str(dimension), str(num_levels)],
        cwd=project_root,
        check=True,
    )


def parse_accuracies(stdout_text: str):
    dataset_re = re.compile(r"Dataset\s+(\d+)\s+accuracy:\s*([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)
    overall_re = re.compile(r"^Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE | re.MULTILINE)

    dataset_to_percent = {}
    for match in dataset_re.finditer(stdout_text):
        dataset_to_percent[int(match.group(1))] = float(match.group(2))

    overall_percent = None
    match = overall_re.search(stdout_text)
    if match:
        overall_percent = float(match.group(1))

    return dataset_to_percent, overall_percent


def make_row(num_levels: int, dimension: int, accuracy_percent: float, info: str):
    accuracy = accuracy_percent / 100.0
    return [
        num_levels,
        NUM_FEATURES,
        dimension,
        0,
        0,
        0,
        -1,
        1,
        WINDOW,
        DOWNSAMPLE,
        VALIDATION_RATIO,
        accuracy,
        accuracy,
        -1.0,
        -1,
        -1,
        -1,
        -1,
        info,
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run Krischans model repeatedly with regenerated IM/CM and log ResultManager-like CSV rows."
    )
    parser.add_argument(
        "--d-values",
        default="512 1024 2048 4096 8192",
        help="Comma or space separated vector dimensions.",
    )
    parser.add_argument(
        "--m-values",
        default=DEFAULT_LEVELS,
        help="Comma or space separated num-level values.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Repetitions per (D, M) with mode fixed to 1.")
    parser.add_argument("--output", default="results/repeats_results.csv", help="CSV path relative to krischans_model.")
    parser.add_argument("--skip-build", action="store_true", help="Skip build step.")
    args = parser.parse_args()

    d_values = parse_int_list(args.d_values)
    m_values = parse_int_list(args.m_values)
    modes = [1]
    if not d_values or not m_values:
        raise ValueError("d-values and m-values must be non-empty.")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_path = project_root / args.output

    if not args.skip_build:
        make_cmd = choose_make_command()
        print(f"[build] {make_cmd} build")
        subprocess.run([make_cmd, "build"], cwd=project_root, check=True)

    binary = find_binary(project_root)
    ensure_csv_header(output_path)

    for dimension in d_values:
        for num_levels in m_values:
            for repeat in range(args.repeats):
                regenerate_memory(script_dir, project_root, dimension, num_levels)

                for mode in modes:
                    print(f"run: D={dimension} M={num_levels} mode={mode} repeat={repeat}")
                    completed = subprocess.run(
                        [str(binary), str(dimension), str(num_levels), str(mode)],
                        cwd=project_root,
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    dataset_acc, overall_acc = parse_accuracies(completed.stdout)
                    if not dataset_acc and overall_acc is None:
                        raise RuntimeError("Could not parse accuracy from model output.")

                    for dataset_id in sorted(dataset_acc):
                        info = (
                            f"model=krischan,scope=dataset,mode={mode},dataset={dataset_id},"
                            f"repeat={repeat}"
                        )
                        row = make_row(num_levels, dimension, dataset_acc[dataset_id], info)
                        append_result(output_path, row)

                    if overall_acc is not None:
                        info = f"model=krischan,scope=overall,mode={mode},repeat={repeat}"
                        row = make_row(num_levels, dimension, overall_acc, info)
                        append_result(output_path, row)

    print(f"Done. Results appended to {output_path}")


if __name__ == "__main__":
    main()
