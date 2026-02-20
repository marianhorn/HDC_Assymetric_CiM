/**
 * @file operations.c

 * @brief Implements vector operations for hyperdimensional computing.
 *
 * This file contains functions for common operations on hypervectors, including:
 * - **Binding:** Combines two vectors element-wise, either via multiplication (bipolar) or XOR (binary).
 * - **Bundling:** Aggregates multiple vectors, using summation (bipolar) or majority voting (binary).
 * - **Permutation:** Performs cyclic shifts on vectors for encoding temporal information.
 * - **Similarity:** Computes similarity metrics, such as cosine similarity (bipolar) or Hamming distance (binary).
 *
 * @details
 * The operations are optimized for both bipolar and binary vector modes, allowing flexibility in
 * hyperdimensional computing applications. These functions form the core of data encoding, aggregation,
 * and classification pipelines.
 */
#include "operations.h"
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "vector.h"
/**
 * @brief Combines two hypervectors element-wise.
 *
 * Performs binding by:
 * - **Bipolar mode:** Multiplying corresponding elements.
 * - **Binary mode:** Applying XOR on corresponding elements.
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
    for (int i = 0; i < VECTOR_DIMENSION; i++) {
#if BIPOLAR_MODE
        result->data[i] = vector1->data[i] * vector2->data[i]; //multiplication for bipolar
#else
        result->data[i] = vector1->data[i] ^ vector2->data[i]; // XOR for binary
#endif
    }
}

/**
 * @brief Aggregates two hypervectors.
 *
 * Bundling combines two vectors:
 * - **Bipolar mode:** Adds corresponding elements.
 * - **Binary mode:** Uses majority voting across corresponding elements.
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
    for (int i = 0; i < VECTOR_DIMENSION; i++) {
#if BIPOLAR_MODE
        result->data[i] = vector1->data[i] + vector2->data[i]; //Addition for bipolar
#else
        int count_true[VECTOR_DIMENSION] = {0};

        for (int i = 0; i < VECTOR_DIMENSION; i++) {
            count_true[i] += vector1->data[i];
            count_true[i] += vector2->data[i];
        }

        for (int i = 0; i < VECTOR_DIMENSION; i++) {
            result->data[i] = count_true[i] > 1; // Majority voting: true if count of true is more than half
        }
#endif
    }
}
/**
 * @brief Aggregates multiple hypervectors.
 *
 * Combines an array of vectors into a single bundled vector:
 * - **Bipolar mode:** Sums the elements of all input vectors.
 * - **Binary mode:** Applies majority voting across corresponding elements.
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
    for (int i = 0; i < VECTOR_DIMENSION; i++) {
        result->data[i] = 0;
    }
#if BIPOLAR_MODE
    for (int v = 0; v < num_vectors; v++) {
        for (int i = 0; i < VECTOR_DIMENSION; i++) {
            result->data[i] += vectors[v]->data[i];
        }
    }

#else
    int count_true[VECTOR_DIMENSION] = {0};

    for (int v = 0; v < num_vectors; v++) {
        for (int i = 0; i < VECTOR_DIMENSION; i++) {
            count_true[i] += vectors[v]->data[i];
        }
    }

    for (int i = 0; i < VECTOR_DIMENSION; i++) {
        result->data[i] = (count_true[i] >= num_vectors / 2) ? 1 : 0;
    }

#endif
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
    if(offset>0){
        for (int i = 0; i < VECTOR_DIMENSION; i++) {
            result->data[(i + offset) % VECTOR_DIMENSION] = vector->data[i];
        }
    }else {
        // Negative offset (left shift)
        offset = -offset;  // Make offset positive for easier calculation
        for (int i = 0; i < VECTOR_DIMENSION; i++) {
            result->data[i] = vector->data[(i + offset) % VECTOR_DIMENSION];
        }
    }
}

/**
 * @brief Computes the cosine similarity between two bipolar vectors.
 *
 * Measures the cosine of the angle between the two input vectors, returning a value in the range [-1, 1].
 *
 * @param vec1 The first input vector.
 * @param vec2 The second input vector.
 * 
 * @return The cosine similarity value, or -2 if any input vector is `NULL` or has zero norm.
 *
 * @note This function is applicable only to bipolar vectors.
 */
double cosine_similarity(Vector *vec1, Vector *vec2) {
    if (vec1 == NULL || vec2 == NULL) {
        fprintf(stderr, "Error: NULL vector passed to cosine_similarity\n");
        return -2;
    }

    double dot_product = 0;
    long int norm1 = 0;
    long int norm2 = 0;
    long int summand1 = 0;
    long int summand2 = 0;

    for (int i = 0; i < VECTOR_DIMENSION; i++) {
        dot_product += vec1->data[i] * vec2->data[i];
        summand1 = (long int) vec1->data[i] * vec1->data[i];
        summand2 = (long int) vec2->data[i] * vec2->data[i];
        norm1 += summand1;
        norm2 += summand2;
    }

    if (norm1 == 0 || norm2 == 0) {
        return -2; // Handle divide-by-zero case
    } else {
        return dot_product / (sqrt(norm1) * sqrt(norm2));
    }
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
    for (int i = 0; i < VECTOR_DIMENSION; i++) {
        if (vec1->data[i] != vec2->data[i]) {
            distance++;
        }
    }
    // Project Hamming distance onto the range -1 to 1
    // distance of 0 (identical) -> 1, max distance -> -1
    return 1.0 - 2.0 * ((double)distance / VECTOR_DIMENSION);
}

/**
 * @brief Computes the similarity between two vectors based on the vector mode.
 *
 * - **Bipolar mode:** Uses cosine similarity.
 * - **Binary mode:** Uses normalized Hamming distance.
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

#if BIPOLAR_MODE
    // Use cosine similarity for bipolar vectors
    return cosine_similarity(vec1, vec2);
#else
    // Use Hamming distance for binary vectors, projected onto -1 to 1
    return hamming_distance(vec1, vec2);
#endif
}