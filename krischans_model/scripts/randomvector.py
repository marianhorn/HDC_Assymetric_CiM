import random

# Parameter für die Vektoren
vector_size = 2000
num_vectors = 32

# Generiere einen zufälligen Vektor
def generate_random_vector(size):
    return ''.join(random.choice('01') for _ in range(size))

# Speichere die Vektoren in eine Datei
with open('memoryfiles/position-vectors.txt', 'w') as f:
    for _ in range(num_vectors):
        vector = generate_random_vector(vector_size)
        f.write(vector + '\n')  # Vektor in die Datei schreiben

print("Done! 32 random vectors saved to position-vectors.txt.")