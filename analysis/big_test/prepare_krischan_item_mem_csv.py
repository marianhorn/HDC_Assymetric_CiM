import argparse
import os
import subprocess
import sys


def convert_bitstrings_to_item_mem_csv(input_path, output_path, expected_vectors, expected_dimension):
    with open(input_path, "r", encoding="ascii") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if len(lines) < expected_vectors:
        raise RuntimeError(
            f"{input_path} has {len(lines)} vectors, expected at least {expected_vectors}."
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="ascii", newline="\n") as handle:
        handle.write(f"#item_mem,num_vectors={expected_vectors},dimension={expected_dimension}\n")
        for row_idx in range(expected_vectors):
            bits = lines[row_idx]
            if len(bits) != expected_dimension:
                raise RuntimeError(
                    f"{input_path} row {row_idx} has dimension {len(bits)}, expected {expected_dimension}."
                )
            if set(bits) - {"0", "1"}:
                raise RuntimeError(f"{input_path} row {row_idx} has non-binary characters.")
            handle.write(",".join(bits) + "\n")


def regenerate_krischan_vectors(krischan_root, dimension, num_levels, num_features):
    py = sys.executable
    scripts_dir = os.path.join(krischan_root, "scripts")
    subprocess.run(
        [py, os.path.join(scripts_dir, "randomvector.py"), str(dimension), str(num_features)],
        cwd=krischan_root,
        check=True,
    )
    subprocess.run(
        [py, os.path.join(scripts_dir, "bitflipvector.py"), str(dimension), str(num_levels)],
        cwd=krischan_root,
        check=True,
    )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    krischan_root = os.path.join(repo_root, "krischans_model")

    parser = argparse.ArgumentParser(
        description="Regenerate Krischan IM/CM and convert them to item_mem CSV files loadable by this framework."
    )
    parser.add_argument("--dimension", type=int, default=2048, help="VECTOR_DIMENSION to prepare.")
    parser.add_argument("--num-levels", type=int, default=51, help="NUM_LEVELS to prepare.")
    parser.add_argument("--num-features", type=int, default=32, help="Number of IM vectors (features).")
    parser.add_argument(
        "--no-regen",
        action="store_true",
        help="Skip regeneration and only convert existing position/value vectors.",
    )
    parser.add_argument(
        "--im-out",
        default=os.path.join(script_dir, "krischan_position_vectors.csv"),
        help="Output CSV path for channel/item memory.",
    )
    parser.add_argument(
        "--cm-out",
        default=os.path.join(script_dir, "krischan_value_vectors.csv"),
        help="Output CSV path for continuous item memory.",
    )
    args = parser.parse_args()

    if not args.no_regen:
        regenerate_krischan_vectors(
            krischan_root=krischan_root,
            dimension=args.dimension,
            num_levels=args.num_levels,
            num_features=args.num_features,
        )

    im_txt = os.path.join(krischan_root, "memoryfiles", "position-vectors.txt")
    cm_txt = os.path.join(krischan_root, "memoryfiles", "value_vectors.txt")

    convert_bitstrings_to_item_mem_csv(
        input_path=im_txt,
        output_path=args.im_out,
        expected_vectors=args.num_features,
        expected_dimension=args.dimension,
    )
    convert_bitstrings_to_item_mem_csv(
        input_path=cm_txt,
        output_path=args.cm_out,
        expected_vectors=args.num_levels,
        expected_dimension=args.dimension,
    )

    print(f"Wrote IM CSV: {args.im_out}")
    print(f"Wrote CM CSV: {args.cm_out}")
    print("Temporary modelFoot.c path expects these files for non-precomputed mode.")


if __name__ == "__main__":
    main()
