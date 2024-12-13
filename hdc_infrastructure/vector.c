/**
 * @file vector.c
 * @brief Implementation of functions for managing and debugging hyperdimensional vectors.
 *
 * This file contains functions to initialize, free, and debug vectors used in 
 * hyperdimensional computing (HDC). Vectors are represented as arrays of elements 
 * and can be either binary or bipolar, depending on the configuration.
 *
 * @details 
 * - **Vector Initialization:** Vectors are allocated dynamically, with values initialized
 *   to default values based on the chosen mode:
 *   - Binary Mode: Elements are initialized to `false` (0).
 *   - Bipolar Mode: Elements are initialized to `-1`.
 * - **Vector Debugging:** The file includes functions to print vector values, aiding in debugging.
 *
 * @note Ensure proper memory management to avoid memory leaks.
 */
#include "vector.h"

/**
 * @brief Allocates and initializes a new vector.
 *
 * This function creates a vector of dimension `VECTOR_DIMENSION` and initializes its elements 
 * based on the selected mode (binary or bipolar).
 *
 * @return A pointer to the newly created vector.
 *
 * @note 
 * - In **bipolar mode**, elements are initialized to `-1`.
 * - In **binary mode**, elements are initialized to `false` (0).
 *
 * @warning 
 * - The function exits with an error if memory allocation fails.
 * - Ensure the allocated vector is freed after use with `free_vector`.
 */

Vector* create_vector() {
    Vector* vec = (Vector*)malloc(sizeof(Vector));
    vec->data = (vector_element*)malloc(VECTOR_DIMENSION * sizeof(vector_element));
    if (!(vec&&vec->data)) {
        fprintf(stderr, "Memory allocation failed for vector\n");
        exit(EXIT_FAILURE);
    }
#if BIPOLAR_MODE
    for (size_t i = 0; i < VECTOR_DIMENSION; i++) {
        vec->data[i] = -1; // Initialize to -1 for bipolar
    }
#else
    for (size_t i = 0; i < VECTOR_DIMENSION; i++) {
        vec->data[i] = false; // Initialize to 0 for binary
    }
#endif
    return vec;
}
/**
 * @brief Allocates a new vector without initializing its elements.
 *
 * This function creates a vector of dimension `VECTOR_DIMENSION`. The vector's
 * data is allocated but not initialized, leaving its contents undefined.
 *
 * @return A pointer to the newly created uninitialized vector.
 *
 * @warning 
 * - The function exits with an error if memory allocation fails.
 * - Ensure the allocated vector is initialized before use.
 * - Free the vector after use with `free_vector` to prevent memory leaks.
 */
Vector* create_uninitialized_vector() {
    Vector* vec = (Vector*)malloc(sizeof(Vector));
    vec->data = (vector_element*)malloc(VECTOR_DIMENSION * sizeof(vector_element));
    return vec;
}

/**
 * @brief Frees the memory allocated for a vector.
 *
 * This function releases the memory used by the vector and its data.
 *
 * @param vec A pointer to the vector to be freed.
 *
 * @note The vector must have been created with `create_vector` or `create_uninitialized_vector`.
 */
void free_vector(Vector* vec) {
    free(vec->data);
    free(vec);
}

/**
 * @brief Prints the first 10,000 elements of a vector.
 *
 * This function prints the values of a vector for debugging purposes.
 *
 * @param vec A pointer to the vector to be printed.
 *
 * @note 
 * - Only the first 100 elements are printed to avoid excessive output.
 * - Ensure the vector is not NULL before calling this function.
 */
void print_vector(const Vector* vec) {
    for (size_t i = 0; i < 100; i+=1) {
        printf("%d ", vec->data[i]);
    }
    printf("\n");
}