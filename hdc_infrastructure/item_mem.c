/**
 * @file item_mem.c
 * @brief Implements functions for generating and managing item memory used in hyperdimensional computing.
 * 
 * @details
 * This file provides functionality to initialize, manage, and manipulate item memory vectors. 
 * Item memory is a key component in HDC and stores base vectors for encoding input data. The implementation 
 * supports both bipolar and binary data representations.
 * 
 * Functions in this file include initialization of item memory for discrete and continuous items, 
 * vector interpolation, storing/loading item memory to/from files, and generating orthogonal vectors.
 * 
 * @author Marian Horn
 */
#include "item_mem.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include "vector.h"

void generate_random_hv(vector_element *data, int dimension);

static uint32_t item_mem_xorshift32(uint32_t *state) {
    uint32_t x = *state;
    if (x == 0u) {
        x = 0x6d2b79f5u;
    }
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static int item_mem_rand_range(uint32_t *state, int max) {
    if (max <= 0) {
        return 0;
    }
    return (int)(item_mem_xorshift32(state) % (uint32_t)max);
}

static uint32_t item_mem_seed_from_permutation(const int *perm, int length) {
    uint32_t hash = 2166136261u;
    if (!perm || length <= 0) {
        return 1u;
    }
    for (int i = 0; i < length; i++) {
        hash ^= (uint32_t)perm[i];
        hash *= 16777619u;
    }
    if (hash == 0u) {
        hash = 1u;
    }
    return hash;
}

static void generate_random_hv_with_rng(vector_element *data, int dimension, uint32_t *state) {
    for (int i = 0; i < dimension; i++) {
#if BIPOLAR_MODE
        data[i] = (item_mem_rand_range(state, 2) * 2) - 1;
#else
        data[i] = item_mem_rand_range(state, 2);
#endif
    }
}
/**
 * @brief Initializes item memory for discrete items, eg. features.
 * 
 * @details
 * This function generates a set of random base vectors, either bipolar (-1, 1) or binary (0, 1),
 * for encoding discrete items. The vectors are stored in the `item_memory` structure.
 * 
 * @param item_mem A pointer to the item memory structure to be initialized.
 * @param num_items The number of discrete items to encode.
 */
void init_item_memory(struct item_memory *item_mem, int num_items) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Initializing item memory for %d features.\n",num_items);
    }
    item_mem->num_vectors = num_items;
    item_mem->base_vectors = (Vector **)malloc(num_items * sizeof(Vector*));
    for (int i = 0; i < num_items; i++) {
        item_mem->base_vectors[i] = create_uninitialized_vector();
        for (int j = 0; j < VECTOR_DIMENSION; j++) {
#if BIPOLAR_MODE
            item_mem->base_vectors[i]->data[j] = (rand() % 2) * 2 - 1; //-1 or 1 for bipolar
#else
            item_mem->base_vectors[i]->data[j] = rand() % 2; //0 or 1 for binary
#endif
        }
    }
    if (output_mode >= OUTPUT_DEBUG) {
        print_item_memory(item_mem);
        printf("\n");
    }
}

/**
 * @brief Generates two orthogonal vectors.
 * 
 * @details
 * This function creates two vectors that are orthogonal to each other, 
 * which can be used for continuous item memory or other operations.
 * 
 * @param vector1 A pointer to the first vector to be generated.
 * @param vector2 A pointer to the second vector to be generated.
 * @param dimension The dimensionality of the vectors.
 */
void generate_orthogonal_vectors(Vector *vector1, Vector *vector2, int dimension) {
    for (int i = 0; i < dimension; i++) {
#if BIPOLAR_MODE
        vector1->data[i] = (rand() % 2) * 2 - 1; // -1 or 1 for bipolar
        vector2->data[i] = -vector1->data[i]; // Orthogonal for bipolar
#else
        vector1->data[i] = rand() % 2;
        vector2->data[i] = !vector1->data[i]; // Orthogonal for binary
#endif
    }
}

/**
 * @brief Interpolates between two vectors.
 * 
 * @details
 * This function creates a new vector by interpolating between two input vectors 
 * based on a specified ratio. The resulting vector contains elements randomly selected 
 * from the two input vectors according to the ratio.
 * 
 * @param vec1 A pointer to the first vector.
 * @param vec2 A pointer to the second vector.
 * @param result A pointer to the resulting interpolated vector.
 * @param dimension The dimensionality of the vectors.
 * @param ratio The ratio for interpolation (0.0 corresponds to `vec1`, 1.0 corresponds to `vec2`).
 * 
 * @note This is used to generate equidistant hypervectors for continuous item memory
 */
void interpolate_vectors(Vector *vec1, Vector *vec2, Vector *result, int dimension, double ratio) {
    int flip_count = (int)(dimension * ratio);
    memcpy(result->data, vec1->data, dimension * sizeof(vector_element));
    for (int i = 0; i < flip_count; i++) {
        int index = rand() % dimension;
        result->data[index] = vec2->data[index];
    }
}

/**
 * @brief Initializes item memory for continuous signal levels.
 * 
 * @details
 * This function generates a set of vectors representing continuous signal levels. 
 * It creates orthogonal vectors for the minimum and maximum levels and interpolates 
 * between them to generate intermediate levels.
 * 
 * @param item_mem A pointer to the item memory structure to be initialized.
 * @param num_levels The number of continuous signal levels.
 */
void init_continuous_item_memory(struct item_memory *item_mem, int num_levels) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Initializing continuous item memory with %d levels.\n",num_levels);
    }
    item_mem->num_vectors = num_levels;
    item_mem->base_vectors = (Vector **)malloc(num_levels * sizeof(Vector *));
    for (int i = 0; i < num_levels; i++) {
        item_mem->base_vectors[i] = create_uninitialized_vector();
    }

    Vector *min_vector = create_uninitialized_vector();

    // Generate min randomly.
    generate_random_hv(min_vector->data, VECTOR_DIMENSION);

    // Prepare a random permutation of indices [0..D-1].
    int *perm = (int *)malloc(VECTOR_DIMENSION * sizeof(int));
    for (int i = 0; i < VECTOR_DIMENSION; i++) {
        perm[i] = i;
    }
    for (int i = VECTOR_DIMENSION - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }

    // Total flip budget K (use D for exact complement, D/2 for common case).
    int total_flips = VECTOR_DIMENSION ;// / 2;
    if (total_flips < 0) {
        total_flips = 0;
    } else if (total_flips > VECTOR_DIMENSION) {
        total_flips = VECTOR_DIMENSION;
    }

    // Level 0 is the min vector.
    memcpy(item_mem->base_vectors[0]->data,
           min_vector->data,
           VECTOR_DIMENSION * sizeof(vector_element));

    if (num_levels > 1) {
        int steps = num_levels - 1;
        int prev_target = 0;
        for (int level = 1; level < num_levels; level++) {
            double exact = ((double)level * (double)total_flips) / (double)steps;
            int target = (int)(exact + 0.5); // balanced rounding
            if (target < 0) {
                target = 0;
            } else if (target > total_flips) {
                target = total_flips;
            }

            Vector *prev = item_mem->base_vectors[level - 1];
            Vector *curr = item_mem->base_vectors[level];
            memcpy(curr->data, prev->data, VECTOR_DIMENSION * sizeof(vector_element));

            for (int k = prev_target; k < target; k++) {
                int idx = perm[k];
#if BIPOLAR_MODE
                curr->data[idx] = -curr->data[idx];
#else
                curr->data[idx] = !curr->data[idx];
#endif
            }

            prev_target = target;
        }
    }

    free(perm);
    free_vector(min_vector);
    if (output_mode >= OUTPUT_DEBUG) {
        print_item_memory(item_mem);
        printf("\n");
    }
}

/**
 * @brief Initializes continuous item memory using per-level flip counts.
 *
 * @details
 * This function generates a set of vectors representing continuous signal levels.
 * It creates a deterministic minimum vector, applies the provided permutation of indices,
 * and flips bits cumulatively based on the provided flip counts B.
 *
 * @param item_mem A pointer to the item memory structure to be initialized.
 * @param num_levels The number of continuous signal levels.
 * @param B Array of size (num_levels-1) specifying flips from level i to i+1.
 * @param permutation Array of size VECTOR_DIMENSION specifying flip order.
 */
void init_continuous_item_memory_with_B(struct item_memory *item_mem,
                                        int num_levels,
                                        const int *B,
                                        const int *permutation) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Initializing continuous item memory (B-driven) with %d levels.\n", num_levels);
    }

    if (num_levels <= 0) {
        item_mem->num_vectors = 0;
        item_mem->base_vectors = NULL;
        return;
    }
    if (num_levels > 1 && (!B || !permutation)) {
        if (output_mode >= OUTPUT_BASIC) {
            fprintf(stderr, "init_continuous_item_memory_with_B: B or permutation is NULL.\n");
        }
        item_mem->num_vectors = 0;
        item_mem->base_vectors = NULL;
        return;
    }

    item_mem->num_vectors = num_levels;
    item_mem->base_vectors = (Vector **)malloc(num_levels * sizeof(Vector *));
    for (int i = 0; i < num_levels; i++) {
        item_mem->base_vectors[i] = create_uninitialized_vector();
    }

    Vector *min_vector = create_uninitialized_vector();
    uint32_t rng_state = item_mem_seed_from_permutation(permutation, VECTOR_DIMENSION);
    generate_random_hv_with_rng(min_vector->data, VECTOR_DIMENSION, &rng_state);

    memcpy(item_mem->base_vectors[0]->data,
           min_vector->data,
           VECTOR_DIMENSION * sizeof(vector_element));

    if (num_levels > 1) {
        int prev_target = 0;
        for (int level = 1; level < num_levels; level++) {
            int flips = B[level - 1];
            if (flips < 0) {
                flips = 0;
            }
            int target = prev_target + flips;
            if (target > VECTOR_DIMENSION) {
                target = VECTOR_DIMENSION;
            }

            Vector *prev = item_mem->base_vectors[level - 1];
            Vector *curr = item_mem->base_vectors[level];
            memcpy(curr->data, prev->data, VECTOR_DIMENSION * sizeof(vector_element));

            for (int k = prev_target; k < target; k++) {
                int idx = permutation[k];
#if BIPOLAR_MODE
                curr->data[idx] = -curr->data[idx];
#else
                curr->data[idx] = !curr->data[idx];
#endif
            }

            prev_target = target;
        }
    }

    free_vector(min_vector);

    if (output_mode >= OUTPUT_DEBUG) {
        print_item_memory(item_mem);
        printf("\n");
    }
}

void generate_random_hv(vector_element *data, int dimension) {
    for (int i = 0; i < dimension; i++) {
        #if BIPOLAR_MODE
        data[i] = (rand() % 2) * 2 - 1; // Randomly assign -1 or 1 for bipolar

        #else
        data[i] = rand() % 2; // Randomly assign 0 or 1 for binary
        #endif
    }
}
/**
 * @brief Initializes binary item memory for precomputed feature-level representations.
 * 
 * @details
 * This function generates a precomputed item memory for binary data, 
 * where each feature and level combination is assigned a unique vector.
 * 
 * @param item_mem A pointer to the item memory structure to be initialized.
 * @param num_levels The number of signal levels.
 * @param num_features The number of features to encode.
 * @note If used, activate PRECOMPUTED_ITEM_MEMORY in config.h
 */
void init_precomp_item_memory(struct item_memory *item_mem, int num_levels, int num_features) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Initializing precomputed item memory with %d levels for %d features.\n",num_levels,num_features);
    }
    int total_vectors = num_levels * num_features; // Total vectors required
    item_mem->num_vectors = total_vectors;
    item_mem->base_vectors = (Vector **)malloc(total_vectors * sizeof(Vector *));
    for (int i = 0; i < num_levels*num_features; i++) {
        item_mem->base_vectors[i] = create_uninitialized_vector();
    }
    uint32_t rng_state = 1u;
    // Total flip budget K (use D for exact complement, D/2 for common case).
    int total_flips = VECTOR_DIMENSION ;// / 2;
    if (total_flips < 0) {
        total_flips = 0;
    } else if (total_flips > VECTOR_DIMENSION) {
        total_flips = VECTOR_DIMENSION;
    }

    for (int feature = 0; feature < num_features; feature++) {
        Vector *min_vector = create_uninitialized_vector();

        // Generate min randomly.
        generate_random_hv_with_rng(min_vector->data, VECTOR_DIMENSION, &rng_state);

        // Prepare a random permutation of indices [0..D-1].
        int *perm = (int *)malloc(VECTOR_DIMENSION * sizeof(int));
        for (int i = 0; i < VECTOR_DIMENSION; i++) {
            perm[i] = i;
        }
        for (int i = VECTOR_DIMENSION - 1; i > 0; i--) {
            int j = item_mem_rand_range(&rng_state, i + 1);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }

        // Level 0 is the min vector.
        memcpy(item_mem->base_vectors[feature]->data,
               min_vector->data,
               VECTOR_DIMENSION * sizeof(vector_element));

        if (num_levels > 1) {
            int steps = num_levels - 1;
            int prev_target = 0;
            for (int level = 1; level < num_levels; level++) {
                double exact = ((double)level * (double)total_flips) / (double)steps;
                int target = (int)(exact + 0.5); // balanced rounding
                if (target < 0) {
                    target = 0;
                } else if (target > total_flips) {
                    target = total_flips;
                }

                Vector *prev = item_mem->base_vectors[(level - 1) * num_features + feature];
                Vector *curr = item_mem->base_vectors[level * num_features + feature];
                memcpy(curr->data, prev->data, VECTOR_DIMENSION * sizeof(vector_element));

                for (int k = prev_target; k < target; k++) {
                    int idx = perm[k];
    #if BIPOLAR_MODE
                    curr->data[idx] = -curr->data[idx];
    #else
                    curr->data[idx] = !curr->data[idx];
    #endif
                }

                prev_target = target;
            }
        }

        free(perm);
        free_vector(min_vector);
    }
    if (output_mode >= OUTPUT_DEBUG) {
        print_item_memory(item_mem);
        printf("\n");
    }
}

/**
 * @brief Initializes precomputed item memory using per-level flip counts.
 *
 * @details
 * This function generates precomputed item memory for binary data, where each
 * feature has its own continuous sequence of levels. The integer matrix B is
 * treated as row-major [num_features][num_levels-1] and specifies how many
 * bits to flip from level i to level i+1 for each feature. Flips are applied
 * along a provided per-feature permutation to ensure consistent ordering.
 *
 * @param item_mem A pointer to the item memory structure to be initialized.
 * @param num_levels The number of signal levels.
 * @param num_features The number of features to encode.
 * @param B Row-major matrix of size num_features x (num_levels-1) with flip counts.
 * @param permutations Row-major matrix of size num_features x VECTOR_DIMENSION with permutations.
 */
void init_precomp_item_memory_with_B(struct item_memory *item_mem,
                                     int num_levels,
                                     int num_features,
                                     const int *B,
                                     const int *permutations) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Initializing precomputed item memory (B-driven) with %d levels for %d features.\n",
               num_levels, num_features);
    }

    if (!B || !permutations) {
        if (output_mode >= OUTPUT_BASIC) {
            fprintf(stderr, "init_precomp_item_memory_with_B: B or permutations is NULL.\n");
        }
        return;
    }

    int total_vectors = num_levels * num_features;
    item_mem->num_vectors = total_vectors;
    item_mem->base_vectors = (Vector **)malloc(total_vectors * sizeof(Vector *));
    for (int i = 0; i < total_vectors; i++) {
        item_mem->base_vectors[i] = create_uninitialized_vector();
    }

    int max_flips = VECTOR_DIMENSION;

    for (int feature = 0; feature < num_features; feature++) {
        Vector *min_vector = create_uninitialized_vector();
        const int *perm = permutations + (size_t)feature * VECTOR_DIMENSION;

        // Generate min randomly.
        uint32_t rng_state = item_mem_seed_from_permutation(perm, VECTOR_DIMENSION);
        generate_random_hv_with_rng(min_vector->data, VECTOR_DIMENSION, &rng_state);

        // Level 0 is the min vector.
        memcpy(item_mem->base_vectors[feature]->data,
               min_vector->data,
               VECTOR_DIMENSION * sizeof(vector_element));

        if (num_levels > 1) {
            int prev_target = 0;
            for (int level = 1; level < num_levels; level++) {
                int flips = B[feature * (num_levels - 1) + (level - 1)];
                if (flips < 0) {
                    flips = 0;
                }
                int target = prev_target + flips;
                if (target > max_flips) {
                    target = max_flips;
                }

                Vector *prev = item_mem->base_vectors[(level - 1) * num_features + feature];
                Vector *curr = item_mem->base_vectors[level * num_features + feature];
                memcpy(curr->data, prev->data, VECTOR_DIMENSION * sizeof(vector_element));

                for (int k = prev_target; k < target; k++) {
                    int idx = perm[k];
    #if BIPOLAR_MODE
                    curr->data[idx] = -curr->data[idx];
    #else
                    curr->data[idx] = !curr->data[idx];
    #endif
                }

                prev_target = target;
            }
        }

        free_vector(min_vector);
    }

    if (output_mode >= OUTPUT_DEBUG) {
        print_item_memory(item_mem);
        printf("\n");
    }
}

/**
 * @brief Frees the memory allocated for item memory.
 * 
 * @details
 * This function releases all vectors stored in the item memory structure, 
 * as well as the structure itself.
 * 
 * @param item_mem A pointer to the item memory structure to be freed.
 */
void free_item_memory(struct item_memory *item_mem) {
    for (int i = 0; i < item_mem->num_vectors; i++) {
        free_vector(item_mem->base_vectors[i]);
    }
    free(item_mem->base_vectors);
}

/**
 * @brief Retrieves the vector for a specific item.
 * 
 * @details
 * This function fetches the base vector corresponding to a given item ID from the item memory.
 * 
 * @param item_mem A pointer to the item memory structure.
 * @param item_id The ID of the item whose vector is to be retrieved.
 * @return A pointer to the vector corresponding to the item ID, or NULL if the ID is invalid.
 */
Vector* get_item_vector(struct item_memory *item_mem, int item_id) {
    if (item_id >= 0 && item_id < item_mem->num_vectors) {
        return item_mem->base_vectors[item_id];
    }
    return NULL;
}
/**
 * @brief Prints the contents of the item memory.
 * 
 * @details
 * This function outputs the details of the item memory, including the number of vectors 
 * and their values, for debugging purposes.
 * 
 * @param item_mem A pointer to the item memory structure.
 */
void print_item_memory(struct item_memory *item_mem) {
    printf("Item memory contains %d vectors of dimension %d\n", item_mem->num_vectors, VECTOR_DIMENSION);
    for (int j = 0; j < VECTOR_DIMENSION; j += 1000) {
        for (int i = 0; i < item_mem->num_vectors; i++) {
            printf("%d ", item_mem->base_vectors[i]->data[j]);
        }
        printf("\n");
    }
}
/**
 * @brief Stores item memory vectors to a binary file.
 * 
 * @details
 * This function writes the data of all vectors stored in the item memory to a binary file.
 * 
 * The layout in the binary file is as follows:
 * - Each vector is stored sequentially.
 * - Each vector consists of `VECTOR_DIMENSION` elements.
 * - The data type of each element is `vector_element`, which is defined as:
 *   - `int` for bipolar mode (values: -1 or 1).
 *   - `bool` for binary mode (values: 0 or 1).
 * 
 * The binary file will contain `num_vectors * VECTOR_DIMENSION` elements, 
 * written as a contiguous array.
 * 
 * @param item_mem A pointer to the item memory structure.
 * @param filepath The path to the binary file where the vectors should be stored.
 * 
 * @note Ensure that the correct `VECTOR_DIMENSION` is used when reading this file.
 */
void store_item_mem_to_bin(struct item_memory *item_mem, const char *filepath) {
    FILE *file = fopen(filepath, "wb");
    if (!file) {
        perror("Failed to open file for writing item memory");
        exit(EXIT_FAILURE);
    }

    // Write each vector's data
    for (int i = 0; i < item_mem->num_vectors; i++) {
        fwrite(item_mem->base_vectors[i]->data, sizeof(vector_element), VECTOR_DIMENSION, file);
    }

    fclose(file);
    printf("Item memory successfully stored to %s\n", filepath);
}

/**
 * @brief Stores item memory vectors to a CSV file.
 *
 * @details
 * This function writes each vector as one row in a CSV file. Each row contains
 * `VECTOR_DIMENSION` elements separated by commas.
 *
 * @param item_mem A pointer to the item memory structure.
 * @param filepath The path to the CSV file where the vectors should be stored.
 */
void store_item_mem_to_csv(struct item_memory *item_mem, const char *filepath) {
    FILE *file = fopen(filepath, "w");
    if (!file) {
        perror("Failed to open file for writing item memory CSV");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "#item_mem,num_vectors=%d,dimension=%d\n",
            item_mem ? item_mem->num_vectors : 0,
            VECTOR_DIMENSION);
    for (int i = 0; i < item_mem->num_vectors; i++) {
        for (int j = 0; j < VECTOR_DIMENSION; j++) {
            fprintf(file, "%d", (int)item_mem->base_vectors[i]->data[j]);
            if (j < VECTOR_DIMENSION - 1) {
                fputc(',', file);
            }
        }
        fputc('\n', file);
    }

    fclose(file);
    printf("Item memory successfully stored to %s\n", filepath);
}

void store_precomp_item_mem_to_bin(struct item_memory *item_mem,
                                   const char *filepath,
                                   int num_levels,
                                   int num_features) {
    if (num_levels <= 0 || num_features <= 0) {
        fprintf(stderr, "store_precomp_item_mem_to_bin: invalid dimensions.\n");
        return;
    }
    int expected = num_levels * num_features;
    if (item_mem && item_mem->num_vectors != expected && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr, "store_precomp_item_mem_to_bin: expected %d vectors, got %d.\n",
                expected, item_mem ? item_mem->num_vectors : 0);
    }
    store_item_mem_to_bin(item_mem, filepath);
}

void store_precomp_item_mem_to_csv(struct item_memory *item_mem,
                                   const char *filepath,
                                   int num_levels,
                                   int num_features) {
    if (num_levels <= 0 || num_features <= 0) {
        fprintf(stderr, "store_precomp_item_mem_to_csv: invalid dimensions.\n");
        return;
    }
    int expected = num_levels * num_features;
    if (item_mem && item_mem->num_vectors != expected && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr, "store_precomp_item_mem_to_csv: expected %d vectors, got %d.\n",
                expected, item_mem ? item_mem->num_vectors : 0);
    }
    FILE *file = fopen(filepath, "w");
    if (!file) {
        perror("Failed to open file for writing precomp item memory CSV");
        exit(EXIT_FAILURE);
    }

    fprintf(file,
            "#precomp_item_mem,num_levels=%d,num_features=%d,num_vectors=%d,dimension=%d\n",
            num_levels,
            num_features,
            expected,
            VECTOR_DIMENSION);

    for (int i = 0; i < item_mem->num_vectors; i++) {
        for (int j = 0; j < VECTOR_DIMENSION; j++) {
            fprintf(file, "%d", (int)item_mem->base_vectors[i]->data[j]);
            if (j < VECTOR_DIMENSION - 1) {
                fputc(',', file);
            }
        }
        fputc('\n', file);
    }

    fclose(file);
    printf("Item memory successfully stored to %s\n", filepath);
}
/**
 * @brief Loads item memory vectors from a binary file.
 * 
 * @details
 * This function reads vectors from a binary file and initializes the item memory structure.
 * 
 * The binary file must have the following layout:
 * - Each vector is stored sequentially.
 * - Each vector consists of `VECTOR_DIMENSION` elements.
 * - The data type of each element must match the expected type:
 *   - `int` for bipolar mode (values: -1 or 1).
 *   - `bool` for binary mode (values: 0 or 1).
 * 
 * When reading the file, the function:
 * - Initializes the item memory structure with `num_items` vectors.
 * - Reads `num_items * VECTOR_DIMENSION` elements from the binary file and assigns them to the vectors.
 * 
 * @param item_mem A pointer to the item memory structure to be loaded.
 * @param filepath The path to the binary file containing the item memory vectors.
 * @param num_items The number of vectors to load into the item memory.
 * 
 * @note Ensure that the binary file corresponds to the correct `VECTOR_DIMENSION` and `num_items`.
 */
void load_item_mem_from_bin(struct item_memory *item_mem, const char *filepath, int num_items) {
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        perror("Failed to open file for reading item memory");
        exit(EXIT_FAILURE);
    }

    init_item_memory(item_mem, num_items);

    for (int i = 0; i < num_items; i++) {
        size_t items_read = fread(item_mem->base_vectors[i]->data, sizeof(vector_element), VECTOR_DIMENSION, file);
        if (items_read != VECTOR_DIMENSION) {
            fprintf(stderr, "Error: Incomplete vector data at row %d with only %ld elements\n", i,items_read);
            exit(EXIT_FAILURE);
        }
    }

   
    

    fclose(file);
    printf("Item memory successfully loaded from %s\n", filepath);
}

/**
 * @brief Loads item memory vectors from a CSV file.
 *
 * @details
 * This function reads vectors from a CSV file and initializes the item memory structure.
 * The CSV file must have one vector per row and `VECTOR_DIMENSION` comma-separated elements.
 *
 * @param item_mem A pointer to the item memory structure to be loaded.
 * @param filepath The path to the CSV file containing the item memory vectors.
 * @param num_items The number of vectors to load into the item memory.
 */
static char *trim_in_place(char *s) {
    while (*s && isspace((unsigned char)*s)) {
        s++;
    }
    if (*s == '\0') {
        return s;
    }
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) {
        *end-- = '\0';
    }
    return s;
}

static int parse_csv_header(FILE *file,
                            int *num_vectors,
                            int *num_levels,
                            int *num_features,
                            int *dimension) {
    long pos = ftell(file);
    char line[512];
    if (!fgets(line, sizeof(line), file)) {
        fseek(file, pos, SEEK_SET);
        return 0;
    }
    if (line[0] != '#') {
        fseek(file, pos, SEEK_SET);
        return 0;
    }

    char *cursor = line + 1;
    char *token = strtok(cursor, ",");
    while (token) {
        char *entry = trim_in_place(token);
        char *eq = strchr(entry, '=');
        if (eq) {
            *eq = '\0';
            char *key = trim_in_place(entry);
            char *value = trim_in_place(eq + 1);
            int parsed = atoi(value);
            if (strcmp(key, "num_vectors") == 0 && num_vectors) {
                *num_vectors = parsed;
            } else if (strcmp(key, "num_levels") == 0 && num_levels) {
                *num_levels = parsed;
            } else if (strcmp(key, "num_features") == 0 && num_features) {
                *num_features = parsed;
            } else if (strcmp(key, "dimension") == 0 && dimension) {
                *dimension = parsed;
            }
        }
        token = strtok(NULL, ",");
    }

    return 1;
}

static void load_item_mem_from_csv_stream(struct item_memory *item_mem, FILE *file, int num_items) {
    init_item_memory(item_mem, num_items);

    for (int i = 0; i < num_items; i++) {
        for (int j = 0; j < VECTOR_DIMENSION; j++) {
            int value = 0;
            if (fscanf(file, "%d", &value) != 1) {
                fprintf(stderr, "Error: Incomplete vector data at row %d, col %d\n", i, j);
                exit(EXIT_FAILURE);
            }
            item_mem->base_vectors[i]->data[j] = (vector_element)value;
            if (j < VECTOR_DIMENSION - 1) {
                int ch = fgetc(file);
                if (ch != ',') {
                    fprintf(stderr, "Error: Expected ',' at row %d, col %d\n", i, j);
                    exit(EXIT_FAILURE);
                }
            }
        }
        int ch = fgetc(file);
        if (ch != '\n' && ch != EOF) {
            fprintf(stderr, "Error: Expected end of line at row %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

void load_item_mem_from_csv(struct item_memory *item_mem, const char *filepath, int num_items) {
    FILE *file = fopen(filepath, "r");
    if (!file) {
        perror("Failed to open file for reading item memory CSV");
        exit(EXIT_FAILURE);
    }

    int header_vectors = 0;
    int header_levels = 0;
    int header_features = 0;
    int header_dim = 0;
    int has_header = parse_csv_header(file, &header_vectors, &header_levels, &header_features, &header_dim);
    if (has_header && header_vectors > 0) {
        if (num_items > 0 && num_items != header_vectors && output_mode >= OUTPUT_BASIC) {
            fprintf(stderr, "load_item_mem_from_csv: header vectors %d override requested %d.\n",
                    header_vectors, num_items);
        }
        num_items = header_vectors;
    }
    if (num_items <= 0) {
        fprintf(stderr, "load_item_mem_from_csv: invalid num_items.\n");
        fclose(file);
        return;
    }
    load_item_mem_from_csv_stream(item_mem, file, num_items);
    fclose(file);
    printf("Item memory successfully loaded from %s\n", filepath);
}

void load_precomp_item_mem_from_bin(struct item_memory *item_mem,
                                    const char *filepath,
                                    int num_levels,
                                    int num_features) {
    if (num_levels <= 0 || num_features <= 0) {
        fprintf(stderr, "load_precomp_item_mem_from_bin: invalid dimensions.\n");
        return;
    }
    int total = num_levels * num_features;
    load_item_mem_from_bin(item_mem, filepath, total);
}

void load_precomp_item_mem_from_csv(struct item_memory *item_mem,
                                    const char *filepath,
                                    int num_levels,
                                    int num_features) {
    FILE *file = fopen(filepath, "r");
    if (!file) {
        perror("Failed to open file for reading precomp item memory CSV");
        exit(EXIT_FAILURE);
    }

    int header_vectors = 0;
    int header_levels = 0;
    int header_features = 0;
    int header_dim = 0;
    int has_header = parse_csv_header(file, &header_vectors, &header_levels, &header_features, &header_dim);
    if (has_header) {
        if (header_levels > 0) {
            num_levels = header_levels;
        }
        if (header_features > 0) {
            num_features = header_features;
        }
    }

    if (num_levels <= 0 || num_features <= 0) {
        fprintf(stderr, "load_precomp_item_mem_from_csv: invalid dimensions.\n");
        fclose(file);
        return;
    }

    int total = num_levels * num_features;
    if (header_vectors > 0 && header_vectors != total && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr, "load_precomp_item_mem_from_csv: header vectors %d override derived %d.\n",
                header_vectors, total);
        total = header_vectors;
    }

    load_item_mem_from_csv_stream(item_mem, file, total);
    fclose(file);
    printf("Item memory successfully loaded from %s\n", filepath);
}
