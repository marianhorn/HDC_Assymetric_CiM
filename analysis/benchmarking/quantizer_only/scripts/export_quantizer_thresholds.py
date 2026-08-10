#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "threshold_analysis" / "exports"

BINNING_MODES = {
    "quantile": 1,
    "kmeans_1d": 2,
    "decision_tree_1d": 3,
    "chimerge": 4,
}
UNIFORM_MODE = {"uniform": 0}


def choose_make_command():
    if shutil.which("make"):
        return "make"
    if shutil.which("mingw32-make"):
        return "mingw32-make"
    raise RuntimeError("No make command found (tried make and mingw32-make).")


def find_model_binary():
    for name in ("modelFoot", "modelFoot.exe"):
        path = REPO_ROOT / name
        if path.exists():
            return path
    raise FileNotFoundError("modelFoot binary not found after build.")


def parse_modes(text, include_uniform):
    available = dict(BINNING_MODES)
    if include_uniform:
        available = {**UNIFORM_MODE, **available}

    selected = []
    for raw in text.split(","):
        entry = raw.strip().lower()
        if not entry:
            continue
        if entry in available:
            selected.append((entry, available[entry]))
            continue
        matched = None
        for name, value in available.items():
            if entry == str(value):
                matched = (name, value)
                break
        if matched is None:
            valid = ", ".join(list(available.keys()) + [str(v) for v in available.values()])
            raise ValueError(f"Unknown binning mode: {entry}. Valid values: {valid}")
        selected.append(matched)

    if not selected:
        raise ValueError("At least one binning mode must be selected.")

    deduped = []
    seen = set()
    for mode in selected:
        if mode[0] not in seen:
            deduped.append(mode)
            seen.add(mode[0])
    return deduped


def run_cmd(cmd, cwd, log_file):
    log_file.write("$ " + " ".join(str(part) for part in cmd) + "\n")
    log_file.flush()
    result = subprocess.run(cmd, cwd=cwd, stdout=log_file, stderr=log_file)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(str(part) for part in cmd)}")


def macro_path(path):
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def export_mode(make_cmd, mode_name, mode_value, seed, num_levels, vector_dimension, output_root):
    mode_dir = output_root / f"levels_{num_levels:03d}_dim_{vector_dimension:05d}" / f"seed_{seed:02d}" / mode_name
    mode_dir.mkdir(parents=True, exist_ok=True)

    cuts_template = macro_path(mode_dir / "cuts_dataset%02d.csv")
    centers_template = macro_path(mode_dir / "centers_dataset%02d.csv")
    results_path = macro_path(mode_dir / "results.csv")
    log_path = mode_dir / "export_log.txt"

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("=== quantizer threshold export ===\n")
        log_file.write(f"mode={mode_name}\n")
        log_file.write(f"BINNING_MODE={mode_value}\n")
        log_file.write(f"ITEM_MEM_SEED={seed}\n")
        log_file.write(f"NUM_LEVELS={num_levels}\n")
        log_file.write(f"VECTOR_DIMENSION={vector_dimension}\n")
        log_file.write(f"cuts_template={cuts_template}\n")
        log_file.write(f"centers_template={centers_template}\n\n")

        build_cmd = [
            make_cmd,
            "foot",
            "USE_OPENMP=1",
            "USE_GENETIC_ITEM_MEMORY=0",
            "OUTPUT_MODE=0",
            f"BINNING_MODE={mode_value}",
            f"ITEM_MEM_SEED={seed}",
            f"NUM_LEVELS={num_levels}",
            f"VECTOR_DIMENSION={vector_dimension}",
            "QUANTIZER_EXPORT_ENABLED=1",
            f"QUANTIZER_EXPORT_CUTS_PATH_TEMPLATE={cuts_template}",
            f"QUANTIZER_EXPORT_CENTERS_PATH_TEMPLATE={centers_template}",
            f"RESULT_CSV_PATH={results_path}",
        ]
        run_cmd(build_cmd, REPO_ROOT, log_file)

        model_path = find_model_binary()
        run_cmd([model_path], REPO_ROOT, log_file)

    return mode_dir


def main():
    parser = argparse.ArgumentParser(
        description="Export learned quantizer thresholds for all selected Foot EMG quantization methods."
    )
    parser.add_argument("--num-levels", type=int, default=40)
    parser.add_argument("--vector-dimension", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1, help="ITEM_MEM_SEED used by the model build.")
    parser.add_argument(
        "--binning-modes",
        default="quantile,kmeans_1d,decision_tree_1d,chimerge",
        help="Comma-separated modes. Defaults to learned quantizers only.",
    )
    parser.add_argument(
        "--include-uniform",
        action="store_true",
        help="Also export uniform-bin cuts as a reference.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output root for exported CSV files.",
    )
    args = parser.parse_args()

    selected_modes = parse_modes(args.binning_modes, args.include_uniform)
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    make_cmd = choose_make_command()

    print(f"Repo root: {REPO_ROOT}")
    print(f"Output root: {output_root}")
    print(f"Modes: {selected_modes}")
    print(f"NUM_LEVELS={args.num_levels}, VECTOR_DIMENSION={args.vector_dimension}, seed={args.seed}")

    for mode_name, mode_value in selected_modes:
        mode_dir = export_mode(
            make_cmd,
            mode_name,
            mode_value,
            args.seed,
            args.num_levels,
            args.vector_dimension,
            output_root,
        )
        print(f"Exported {mode_name}: {mode_dir}")


if __name__ == "__main__":
    main()
