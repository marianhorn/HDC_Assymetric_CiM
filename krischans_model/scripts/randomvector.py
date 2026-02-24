import random
import sys


def generate_random_vector(size):
    return "".join(random.choice("01") for _ in range(size))


def main():
    # Backward compatible defaults used by the old script.
    vector_size = 2000
    num_vectors = 32

    if len(sys.argv) >= 2:
        vector_size = int(sys.argv[1])
    if len(sys.argv) >= 3:
        num_vectors = int(sys.argv[2])
    if len(sys.argv) > 3:
        print("Usage: randomvector.py [D] [NUM_VECTORS]")
        sys.exit(1)

    with open("memoryfiles/position-vectors.txt", "w", encoding="ascii", newline="\n") as handle:
        for _ in range(num_vectors):
            handle.write(generate_random_vector(vector_size) + "\n")

    print(f"Done! {num_vectors} random vectors saved to position-vectors.txt.")


if __name__ == "__main__":
    main()
