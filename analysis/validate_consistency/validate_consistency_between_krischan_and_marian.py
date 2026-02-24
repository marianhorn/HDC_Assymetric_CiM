#!/usr/bin/env python3
import datetime
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
#
# IMPORTANT!!!!!!
#
#
#TO RUN THIS: Replace modelFoot.c by modelFoot.c from this directory for loading Cim and reporting results.


VECTOR_DIMENSION = 2048
NUM_LEVELS = 51
NUM_FEATURES = 32
N_GRAM_SIZE = 5
KRISCHAN_MODE = 1


def write_comment(log_file, text):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"\n# [{timestamp}] {text}\n")
    log_file.flush()


def run_command(log_file, title, cmd, cwd):
    write_comment(log_file, title)
    log_file.write(f"# cwd: {cwd}\n")
    log_file.write("# cmd: " + " ".join(shlex.quote(str(part)) for part in cmd) + "\n\n")
    log_file.flush()

    completed = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd),
        stdout=log_file,
        stderr=log_file,
        text=True,
        check=False,
    )
    log_file.write(f"\n# exit_code: {completed.returncode}\n")
    log_file.flush()
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed ({title}) with exit code {completed.returncode}")


def get_make_command(cwd, make_args):
    if shutil.which("make"):
        return ["make", *make_args], cwd

    if shutil.which("bash"):
        bash_cwd = to_bash_path(cwd)
        joined_args = " ".join(shlex.quote(arg) for arg in make_args)
        script = f"cd {shlex.quote(bash_cwd)} && make {joined_args}"
        return ["bash", "-lc", script], cwd

    if shutil.which("mingw32-make"):
        return ["mingw32-make", *make_args], cwd

    raise RuntimeError("No usable make command found (make, bash+make, mingw32-make).")


def to_bash_path(path):
    raw = str(path)
    if len(raw) >= 3 and raw[1] == ":" and (raw[2] == "\\" or raw[2] == "/"):
        drive = raw[0].lower()
        rest = raw[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return raw.replace("\\", "/")


def run_binary(log_file, title, binary_path, args, cwd):
    # On Windows hosts, binaries produced via bash/make are often Linux ELF files.
    # Run them via bash when no .exe is present.
    if os.name == "nt" and binary_path.suffix.lower() != ".exe" and shutil.which("bash"):
        bash_cwd = to_bash_path(cwd)
        bash_bin = to_bash_path(binary_path)
        joined_args = " ".join(shlex.quote(str(arg)) for arg in args)
        bash_cmd = f"cd {shlex.quote(bash_cwd)} && {shlex.quote(bash_bin)} {joined_args}"
        run_command(log_file, title, ["bash", "-lc", bash_cmd], cwd)
        return

    run_command(log_file, title, [str(binary_path), *[str(arg) for arg in args]], cwd)


def convert_bitstrings_to_item_mem_csv(input_path, output_path, expected_vectors, expected_dimension):
    lines = []
    with open(input_path, "r", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if line:
                lines.append(line)

    if len(lines) < expected_vectors:
        raise RuntimeError(
            f"{input_path} contains {len(lines)} vectors, expected at least {expected_vectors}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="ascii", newline="\n") as handle:
        handle.write(f"#item_mem,num_vectors={expected_vectors},dimension={expected_dimension}\n")
        for idx in range(expected_vectors):
            bits = lines[idx]
            if len(bits) != expected_dimension:
                raise RuntimeError(
                    f"{input_path} row {idx} has length {len(bits)}, expected {expected_dimension}."
                )
            if set(bits) - {"0", "1"}:
                raise RuntimeError(f"{input_path} row {idx} contains non-binary characters.")
            handle.write(",".join(bits) + "\n")


def find_binary(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No binary found. Tried: {[str(path) for path in candidates]}")


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    krischan_root = repo_root / "krischans_model"
    big_test_dir = repo_root / "analysis" / "big_test"

    output_file = script_dir / "validate_consistency_output.txt"
    marian_results_csv = script_dir / "results_marian_consistency.csv"
    marian_results_rel = marian_results_csv.relative_to(repo_root).as_posix()

    local_im_csv = script_dir / "krischan_position_vectors.csv"
    local_cm_csv = script_dir / "krischan_value_vectors.csv"

    # modelFoot currently imports from analysis/big_test hardcoded paths.
    imported_im_csv = big_test_dir / "krischan_position_vectors.csv"
    imported_cm_csv = big_test_dir / "krischan_value_vectors.csv"

    memoryfiles_dir = krischan_root / "memoryfiles"
    position_vectors_txt = memoryfiles_dir / "position-vectors.txt"
    value_vectors_txt = memoryfiles_dir / "value_vectors.txt"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8", newline="\n") as log_file:
        write_comment(log_file, "Consistency validation run started")
        log_file.write(
            f"# Settings: VECTOR_DIMENSION={VECTOR_DIMENSION}, "
            f"NUM_LEVELS={NUM_LEVELS}, N_GRAM_SIZE={N_GRAM_SIZE}, KRISCHAN_MODE={KRISCHAN_MODE}\n"
        )
        log_file.write("# This file contains complete logs from both pipelines.\n")

        # Step 1: Regenerate Krischan IM/CM text vectors.
        run_command(
            log_file,
            "Step 1/7: Regenerate Krischan IM via randomvector.py",
            [sys.executable, "scripts/randomvector.py", str(VECTOR_DIMENSION), str(NUM_FEATURES)],
            krischan_root,
        )
        run_command(
            log_file,
            "Step 2/7: Regenerate Krischan CM via bitflipvector.py",
            [sys.executable, "scripts/bitflipvector.py", str(VECTOR_DIMENSION), str(NUM_LEVELS)],
            krischan_root,
        )

        # Step 2: Convert to CSV for Marian import path.
        write_comment(log_file, "Step 3/7: Convert Krischan vectors to Marian CSV format")
        convert_bitstrings_to_item_mem_csv(
            input_path=position_vectors_txt,
            output_path=local_im_csv,
            expected_vectors=NUM_FEATURES,
            expected_dimension=VECTOR_DIMENSION,
        )
        convert_bitstrings_to_item_mem_csv(
            input_path=value_vectors_txt,
            output_path=local_cm_csv,
            expected_vectors=NUM_LEVELS,
            expected_dimension=VECTOR_DIMENSION,
        )
        imported_im_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_im_csv, imported_im_csv)
        shutil.copyfile(local_cm_csv, imported_cm_csv)
        log_file.write(f"# Wrote local IM CSV: {local_im_csv}\n")
        log_file.write(f"# Wrote local CM CSV: {local_cm_csv}\n")
        log_file.write(f"# Synced imported IM CSV: {imported_im_csv}\n")
        log_file.write(f"# Synced imported CM CSV: {imported_cm_csv}\n")
        log_file.flush()

        # Step 3: Build and run Marian model.
        marian_make_args = [
            "foot",
            "USE_OPENMP=1",
            "PRECOMPUTED_ITEM_MEMORY=0",
            "USE_GENETIC_ITEM_MEMORY=0",
            "VALIDATION_RATIO=0",
            f"N_GRAM_SIZE={N_GRAM_SIZE}",
            "ENCODER_ROLLING=1",
            f"VECTOR_DIMENSION={VECTOR_DIMENSION}",
            f"NUM_LEVELS={NUM_LEVELS}",
            f"RESULT_CSV_PATH={marian_results_rel}",
        ]
        marian_make_cmd, marian_make_cwd = get_make_command(repo_root, marian_make_args)
        run_command(log_file, "Step 4/7: Build Marian model", marian_make_cmd, marian_make_cwd)

        marian_binary = find_binary([repo_root / "modelFoot", repo_root / "modelFoot.exe"])
        run_binary(log_file, "Step 5/7: Run Marian model", marian_binary, [], repo_root)

        # Step 4: Build and run Krischan model.
        krischan_make_cmd, krischan_make_cwd = get_make_command(krischan_root, ["build"])
        run_command(log_file, "Step 6/7: Build Krischan model", krischan_make_cmd, krischan_make_cwd)

        krischan_binary = find_binary(
            [krischan_root / "hdc_model", krischan_root / "hdc_model.exe"]
        )
        run_binary(
            log_file,
            "Step 7/7: Run Krischan model (mode=1)",
            krischan_binary,
            [VECTOR_DIMENSION, NUM_LEVELS, KRISCHAN_MODE],
            krischan_root,
        )

        write_comment(log_file, "Consistency validation run finished")
        log_file.write(
            "# Hint: compare lines containing 'Dataset XX accuracy' and final 'Accuracy:'.\n"
        )

    print(f"Done. Consolidated output written to: {output_file}")


if __name__ == "__main__":
    main()
