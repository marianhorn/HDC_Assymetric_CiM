#ifndef VECTOR_H
#define VECTOR_H

#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#elif defined(CUSTOM)
#include "../customModel/configCustom.h"
#else
#error "No EMG type defined. Please define HAND_EMG or FOOT_EMG."
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#if BIPOLAR_MODE
typedef int vector_element;
#else
typedef uint64_t vector_element;
#define VECTOR_WORD_BITS 64
#define VECTOR_WORD_COUNT ((VECTOR_DIMENSION + VECTOR_WORD_BITS - 1) / VECTOR_WORD_BITS)
#endif
/**
 * @brief Represents the item memory used in Hyperdimensional Computing.
 *
 * The item memory stores base hypervectors for discrete or continuous input features.
 * It supports precomputed and dynamically generated item memories.
 * 
 * Members:
 * - **num_vectors**: The total number of base vectors in the item memory.
 * - **base_vectors**: Array of pointers to the base hypervectors.
 */
typedef struct {
    vector_element *data; /**< Array of elements representing the vector. */
} Vector;

static inline size_t vector_storage_count(void) {
#if BIPOLAR_MODE
    return (size_t)VECTOR_DIMENSION;
#else
    return (size_t)VECTOR_WORD_COUNT;
#endif
}

static inline size_t vector_storage_bytes(void) {
    return vector_storage_count() * sizeof(vector_element);
}

static inline void vector_zero(Vector *vec) {
    memset(vec->data, 0, vector_storage_bytes());
}

static inline void vector_copy(Vector *dst, const Vector *src) {
    memcpy(dst->data, src->data, vector_storage_bytes());
}

#if BIPOLAR_MODE
static inline int vector_get_bit(const Vector *vec, int bit_idx) {
    return vec->data[bit_idx];
}

static inline void vector_set_bit(Vector *vec, int bit_idx, int value) {
    vec->data[bit_idx] = value;
}

static inline void vector_flip_bit(Vector *vec, int bit_idx) {
    vec->data[bit_idx] = -vec->data[bit_idx];
}

static inline void vector_mask_tail(Vector *vec) {
    (void)vec;
}
#else
static inline int vector_get_bit(const Vector *vec, int bit_idx) {
    int word = bit_idx >> 6;
    int shift = bit_idx & 63;
    return (int)((vec->data[word] >> shift) & 1ull);
}

static inline void vector_set_bit(Vector *vec, int bit_idx, int value) {
    int word = bit_idx >> 6;
    int shift = bit_idx & 63;
    uint64_t mask = 1ull << shift;
    if (value) {
        vec->data[word] |= mask;
    } else {
        vec->data[word] &= ~mask;
    }
}

static inline void vector_flip_bit(Vector *vec, int bit_idx) {
    int word = bit_idx >> 6;
    int shift = bit_idx & 63;
    vec->data[word] ^= (1ull << shift);
}

static inline void vector_mask_tail(Vector *vec) {
    int rest = VECTOR_DIMENSION & 63;
    if (rest == 0) {
        return;
    }
    uint64_t mask = (1ull << rest) - 1ull;
    vec->data[VECTOR_WORD_COUNT - 1] &= mask;
}
#endif

// Function declarations
Vector* create_vector();
Vector* create_uninitialized_vector();
void free_vector(Vector* vec);
void print_vector(const Vector* vec);

#endif // VECTOR_H
