/**
 * @file operations.c

 * @brief Implements vector operations for hyperdimensional computing.
 *
 * This file contains functions for common operations on binary hypervectors, including:
 * - **Binding:** Combines two vectors element-wise via XOR.
 * - **Bundling:** Aggregates multiple vectors using majority voting.
 * - **Permutation:** Performs cyclic shifts on vectors for encoding temporal information.
 * - **Similarity:** Computes normalized Hamming similarity.
 *
 * @details
 * The operations are optimized for binary vector mode. These functions form the core of data encoding, aggregation,
 * and classification pipelines.
 */
#include "operations.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "vector.h"

#if MODEL_VARIANT == MODEL_VARIANT_KRISCHAN
static void permute_like_krischan(Vector *vector, int offset, Vector *result) {
    int chunks = (VECTOR_DIMENSION + 31) / 32;
    uint32_t *in_words = (uint32_t *)calloc((size_t)chunks, sizeof(uint32_t));
    uint32_t *out_words = (uint32_t *)calloc((size_t)chunks, sizeof(uint32_t));
    if (!in_words || !out_words) {
        fprintf(stderr, "Memory allocation failed in permute_like_krischan\n");
        free(in_words);
        free(out_words);
        exit(EXIT_FAILURE);
    }

    // Match krischan loader bit order: index i maps to bit 31-(i%32) in chunk i/32.
    for (int i = 0; i < VECTOR_DIMENSION; i++) {
        if (vector_get_bit(vector, i)) {
            int chunk = i / 32;
            int bit_in_chunk = 31 - (i % 32);
            in_words[chunk] |= (1u << bit_in_chunk);
        }
    }

    int shift_bits = offset;
    int total_bits = chunks * 32;
    if (total_bits > 0) {
        shift_bits %= total_bits;
        if (shift_bits < 0) {
            shift_bits += total_bits;
        }
    }

    int word_shift = shift_bits / 32;
    int bit_shift = shift_bits % 32;
    // Intentional parity path with Krischan implementation, including bit_shift==0 behavior.
    for (int i = 0; i < chunks; i++) {
        uint32_t a = in_words[(i + word_shift) % chunks];
        uint32_t b = in_words[(i + word_shift + 1) % chunks];
        out_words[i] = (a >> bit_shift) | (b << (32 - bit_shift));
    }

    vector_zero(result);
    for (int i = 0; i < VECTOR_DIMENSION; i++) {
        int chunk = i / 32;
        int bit_in_chunk = 31 - (i % 32);
        vector_set_bit(result, i, (out_words[chunk] >> bit_in_chunk) & 1u);
    }

    free(out_words);
    free(in_words);
}
#endif

static void permute_binary_words(const Vector *vector, int right_shift, Vector *result) {
    int shift = right_shift % VECTOR_DIMENSION;
    if (shift < 0) {
        shift += VECTOR_DIMENSION;
    }

    if (shift == 0) {
        memcpy(result->data, vector->data, vector_storage_bytes());
        return;
    }

    if ((VECTOR_DIMENSION & 63) == 0) {
        int words = (int)vector_storage_count();
        int word_shift = shift >> 6;
        int bit_shift = shift & 63;
        for (int w = 0; w < words; w++) {
            int src = w - word_shift;
            while (src < 0) {
                src += words;
            }
            src %= words;

            if (bit_shift == 0) {
                result->data[w] = vector->data[src];
            } else {
                int src_prev = src - 1;
                if (src_prev < 0) {
                    src_prev += words;
                }
                result->data[w] =
                    (vector->data[src] << bit_shift) |
                    (vector->data[src_prev] >> (64 - bit_shift));
            }
        }
        return;
    }

    vector_zero(result);
    for (int i = 0; i < VECTOR_DIMENSION; i++) {
        int out_idx = i + shift;
        if (out_idx >= VECTOR_DIMENSION) {
            out_idx -= VECTOR_DIMENSION;
        }
        vector_set_bit(result, out_idx, vector_get_bit(vector, i));
    }
    vector_mask_tail(result);
}
/**
 * @brief Combines two hypervectors element-wise.
 *
 * Performs binary binding via XOR.
 *
 * @param vector1 The first input vector.
 * @param vector2 The second input vector.
 * @param result The resulting bound vector.
 *
 * @note All input vectors must be initialized. The `result` vector is modified in-place.
 * @warning The function exits the program if any of the input vectors are uninitialized.
 */
void bind(Vector* vector1, Vector* vector2, Vector* result) {
    if (vector1 == NULL || vector2 == NULL || result == NULL) {
        printf("Input vector for binding not initialized");
        exit(EXIT_FAILURE);
    }
    size_t words = vector_storage_count();
    for (size_t w = 0; w < words; w++) {
        result->data[w] = vector1->data[w] ^ vector2->data[w]; // XOR for binary
    }
}

/**
 * @brief Aggregates two hypervectors.
 *
 * Bundling combines two binary vectors using strict majority across two inputs.
 *
 * @param vector1 The first input vector.
 * @param vector2 The second input vector.
 * @param result The resulting bundled vector.
 *
 * @note The `result` vector is modified in-place.
 * @warning The function exits the program if any of the input vectors are uninitialized.
 */
void bundle(Vector* vector1, Vector* vector2, Vector* result) {
    if (vector1 == NULL || vector2 == NULL || result == NULL) {
        printf("Input vector for bundling not initialized");
        exit(EXIT_FAILURE);
    }
    size_t words = vector_storage_count();
    for (size_t w = 0; w < words; w++) {
        // For two vectors and strict majority (>1), binary bundle is bitwise AND.
        result->data[w] = vector1->data[w] & vector2->data[w];
    }
    vector_mask_tail(result);
}

/**
 * @brief Aggregates multiple hypervectors.
 *
 * Combines an array of binary vectors into a single bundled vector using majority voting.
 *
 * @param vectors An array of pointers to the vectors to bundle.
 * @param num_vectors The number of vectors to bundle.
 * @param result The resulting bundled vector.
 *
 * @note The `result` vector is modified in-place and should be initialized before calling.
 * @warning The function exits the program if `vectors` or `result` is uninitialized.
 */
void bundle_multi(Vector** vectors, int num_vectors, Vector* result) {
    if (vectors == NULL || result == NULL) {
        printf("Input vector for bundling not initialized");
        exit(EXIT_FAILURE);
    }
    vector_zero(result);
    int threshold = num_vectors / 2;
    int nbits = 0;
    int max_count = num_vectors;
    while ((1 << nbits) <= max_count && nbits < 31) {
        nbits++;
    }
    if (nbits < 1) {
        nbits = 1;
    }

    uint64_t *planes = (uint64_t *)calloc((size_t)nbits, sizeof(uint64_t));
    if (!planes) {
        fprintf(stderr, "Failed to allocate bit-sliced counters in bundle_multi\n");
        exit(EXIT_FAILURE);
    }

    size_t words = vector_storage_count();
    for (size_t w = 0; w < words; w++) {
        memset(planes, 0, (size_t)nbits * sizeof(uint64_t));

        for (int v = 0; v < num_vectors; v++) {
            uint64_t carry = vectors[v]->data[w];
            for (int b = 0; b < nbits; b++) {
                uint64_t t = planes[b];
                planes[b] = t ^ carry;
                carry = t & carry;
            }
        }

        uint64_t out_word = 0ull;
        for (int bit = 0; bit < 64; bit++) {
            int count = 0;
            for (int b = 0; b < nbits; b++) {
                count |= (int)(((planes[b] >> bit) & 1ull) << b);
            }
            if (count >= threshold) {
                out_word |= (1ull << bit);
            }
        }
        result->data[w] = out_word;
    }
    free(planes);
    vector_mask_tail(result);
}

/**
 * @brief Performs cyclic permutation (shift) on a vector.
 *
 * Shifts the elements of the input vector by the specified offset:
 * - **Positive offset:** Right shift.
 * - **Negative offset:** Left shift.
 *
 * @param vector The input vector to permute.
 * @param offset The number of positions to shift. Positive values shift right; negative values shift left.
 * @param result The resulting permuted vector.
 *
 * @note The `result` vector is modified in-place and should be initialized before calling.
 */
void permute(Vector* vector, int offset, Vector* result) {
#if MODEL_VARIANT == MODEL_VARIANT_KRISCHAN
    permute_like_krischan(vector, offset, result);
#else
    if (offset > 0) {
        permute_binary_words(vector, offset, result);
    } else {
        int left_shift = -offset;
        int right_shift = VECTOR_DIMENSION - (left_shift % VECTOR_DIMENSION);
        if (right_shift == VECTOR_DIMENSION) {
            right_shift = 0;
        }
        permute_binary_words(vector, right_shift, result);
    }
#endif
}

/**
 * @brief Computes the Hamming distance between two binary vectors.
 *
 * Calculates the fraction of differing elements between the two vectors and projects it onto the range [-1, 1].
 *
 * @param vec1 The first input vector.
 * @param vec2 The second input vector.
 * 
 * @return The normalized Hamming distance in the range [-1, 1].
 *
 * @note This function is applicable only to binary vectors.
 */
double hamming_distance(Vector *vec1, Vector *vec2) {
    int distance = 0;
    size_t words = vector_storage_count();
    for (size_t w = 0; w < words; w++) {
        uint64_t diff = vec1->data[w] ^ vec2->data[w];
        if (w + 1 == words && (VECTOR_DIMENSION & 63) != 0) {
            uint64_t mask = (1ull << (VECTOR_DIMENSION & 63)) - 1ull;
            diff &= mask;
        }
        distance += __builtin_popcountll(diff);
    }
    // Project Hamming distance onto the range -1 to 1
    // distance of 0 (identical) -> 1, max distance -> -1
    return 1.0 - 2.0 * ((double)distance / VECTOR_DIMENSION);
}

/**
 * @brief Computes the similarity between two binary vectors.
 *
 * @param vec1 The first input vector.
 * @param vec2 The second input vector.
 * 
 * @return The similarity value, or -2 if any input vector is `NULL`.
 */
double similarity_check(Vector *vec1, Vector *vec2) {
    if (vec1 == NULL || vec2 == NULL) {
        fprintf(stderr, "Error: NULL vector passed to similarityCheck\n");
        return -2;
    }

    return hamming_distance(vec1, vec2);
}
