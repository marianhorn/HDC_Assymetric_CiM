import sys
import random

if len(sys.argv) != 3:
    print("Usage: bitflipvektor.py <D> <M>")
    exit(1)

vector_size = int(sys.argv[1])
num_vectors = int(sys.argv[2])

bit_flips = vector_size // 40   # proportional zur Vektorgröße

def generate_random_vector(size):
    return ''.join(random.choice('01') for _ in range(size))

def flip_bits(vector, flips):
    vector_list = list(vector)
    for _ in range(flips):
        pos = random.randint(0, len(vector_list) - 1)
        vector_list[pos] = '1' if vector_list[pos] == '0' else '0'
    return ''.join(vector_list)

with open('memoryfiles/value_vectors.txt', 'w') as f:
    vector = generate_random_vector(vector_size)
    f.write(vector + '\n')

    for _ in range(1, num_vectors):
        vector = flip_bits(vector, bit_flips)
        f.write(vector + '\n')
